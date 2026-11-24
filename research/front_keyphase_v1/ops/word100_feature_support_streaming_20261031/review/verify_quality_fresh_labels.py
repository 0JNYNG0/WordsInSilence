"""Independent reconciliation/ROI reconstruction. No project or model imports."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

U=Path(__file__).resolve().parent.parent
OLD=U.parent/'word100_stage_feedback_cycle_20261031'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    fresh=U/'fresh';inputs=read(fresh/'INPUTS.json');visual={r['alias']:r for r in read(fresh/'VISUAL_INDEX.json')['rows']}
    rubric=read(OLD/'RGB_RUBRIC.json');all_targets=read(OLD/'LEARNING_INPUTS.json')['all_targets']
    primaries=read(fresh/'RGB_LABELS_A.json')['rows']+read(fresh/'RGB_LABELS_B.json')['rows']
    primary={r['alias']:r for r in primaries};secondary={r['alias']:r for r in read(fresh/'RGB_LABELS_ROOT.json')['rows']}
    audit=read(fresh/'LABEL_AUDIT.json');saved_audit={r['alias']:r for r in audit['rows']}
    checks=0;failures=[]
    def check(ok,name,detail=None):
        nonlocal checks
        checks+=1
        if not bool(ok):failures.append(dict(check=name,detail=detail))
    check(len(primaries)==len(primary)==32,'unique_primary32')
    check(len(secondary)==audit['cross_reviewed_videos']==8,'cross_reviewed8')
    check(set(primary)=={r['alias'] for r in inputs['rows']},'primary_exact_input_aliases')
    for path,digest in audit['pins'].items():check(sha(path)==digest,'label_pin/'+Path(path).name)
    total_known=0;statuses=Counter();cross=[];per_clip=[]
    for row in inputs['rows']:
        alias=row['alias'];p=primary[alias];q=secondary.get(alias);vf=visual[alias]['frames']
        with np.load(row['input'],allow_pickle=False) as z:frames=z['frame_id'];timestamps=z['timestamp_us']
        with np.load(row['observation'],allow_pickle=False) as z:
            xy=z['xy_px'];observed=z['observed'];obframes=z['frame_id']
        check(np.array_equal(obframes,frames),'alignment/'+alias)
        n=len(frames);box=np.asarray(row['box']);support={}
        # At least one direct finite hand point, and every such point in the fixed ROI.
        for side,start in [('L',33),('R',54)]:
            points=xy[:,start:start+21];valid=observed[:,start:start+21]&np.isfinite(points).all(-1)
            inside=(points[:,:,0]>=box[0])&(points[:,:,0]<box[2])&(points[:,:,1]>=box[1])&(points[:,:,1]<box[3])
            support[side]=valid.any(-1)&(~valid|inside).all(-1)
        expected_y=np.zeros((n,len(all_targets)),np.float32);expected_mask=np.zeros_like(expected_y,bool);entries=[]
        st={r['ordinal']:r for r in p['static']};qt={} if q is None else {r['ordinal']:r for r in q['static']}
        check(set(st)=={3,6,9},'static_ordinals/'+alias)
        for ordinal in [3,6,9]:
            ix=int(np.flatnonzero(frames==vf[ordinal]['frame_id'])[0])
            check(len(st[ordinal]['values'])==len(rubric['features'])==9,'static_features/'+alias)
            for j,name in enumerate(rubric['features']):
                entries.append((name,st[ordinal]['values'][j],None if q is None else qt[ordinal]['values'][j],ix,np.array([ix]),'static'))
        motion={(r['start_ordinal'],r['end_ordinal']):r for r in p['motion']}
        qm={} if q is None else {(r['start_ordinal'],r['end_ordinal']):r for r in q['motion']}
        spans=[(0,3),(3,6),(6,9),(9,12)]
        check(set(motion)==set(spans),'motion_spans/'+alias)
        for first,last in spans:
            left,right=vf[first]['frame_id'],vf[last]['frame_id'];indices=np.flatnonzero((frames>=left)&(frames<=right))
            check(np.array_equal(frames[indices],np.arange(left,right+1)),'all_interval_frames/'+alias)
            middle=int(np.argmin(np.abs(timestamps/1e6-(vf[first]['seconds']+vf[last]['seconds'])/2)))
            for side in ['R','L']:
                category=motion[(first,last)][side];other=None if q is None else qm[(first,last)][side]
                check(category in rubric['motion_values'],'motion_category/'+alias)
                for direction in ['up','down','left','right']:
                    value=None if category in [None,'mixed'] else int(direction in category)
                    second=None if other in [None,'mixed'] else int(direction in other)
                    entries.append((side+'_motion_'+direction,value,second,middle,indices,'interval'))
        check(len(entries)==59,'59_annotation_cells/'+alias)
        recorded=saved_audit[alias]['rows']
        check(len(recorded)==len(entries),'audit_row_count/'+alias)
        for name,value,second,index,indices,kind in entries:
            initial=value;status='single_Codex_RGB_draft'
            if q is not None:
                cross.append(dict(alias=alias,feature=name,frame=int(frames[index]),primary=value,secondary=second))
                if value is not None and value==second:status='same_Codex_cross_agreement'
                else:value=None;status='cross_unknown_or_disagreement'
            required=['R','L'] if name=='hands_overlap' else [name[0]]
            roi=all(bool(support[side][indices].all()) for side in required)
            if value is not None and not roi:value=None;status='outside_or_no_direct_hand_ROI_evidence'
            statuses[status]+=1
            if value is not None:expected_y[index,all_targets.index(name)]=value;expected_mask[index,all_targets.index(name)]=True
            matching=[r for r in recorded if (r['feature'],r['frame'],r['kind'])==(name,int(frames[index]),kind)]
            check(len(matching)==1,'audit_unique/'+alias+'/'+name)
            if len(matching)==1:
                expect=dict(feature=name,frame=int(frames[index]),kind=kind,initial=initial,secondary=second if q else 'not_reviewed',value=value,status=status,interval_frame_ids=[int(frames[indices[0]]),int(frames[indices[-1]])])
                check(matching[0]==expect,'audit_exact/'+alias+'/'+name)
        target=fresh/'targets'/f'{alias}.npz'
        check(str(target) in audit['pins'],'target_pinned/'+alias)
        with np.load(target,allow_pickle=False) as z:
            check(np.array_equal(z['y'],expected_y),'target_values/'+alias)
            check(z['target_mask'].dtype==np.bool_ and np.array_equal(z['target_mask'],expected_mask),'target_masks/'+alias)
        known=int(expected_mask.sum());total_known+=known
        check(known==saved_audit[alias]['known'],'known_count/'+alias)
        per_clip.append(dict(alias=alias,known=known))
    joint=[r for r in cross if r['primary'] is not None and r['secondary'] is not None]
    disagreement=[r for r in joint if r['primary']!=r['secondary']]
    check(audit['cross']==cross,'complete_cross_records')
    check(audit['joint_known']==len(joint),'joint_known_count')
    check(audit['known_disagreements']==len(disagreement),'cross_disagreement_count')
    check(audit['disagreements']==disagreement,'cross_disagreement_records')
    check(audit['known']==total_known,'total_known')
    check(audit['human_truth'] is False,'not_human_truth')
    result=dict(status='PASS' if not failures else 'FAIL',reviewer='Codex rgb_labels_b, separate from label assembler; not independent human truth',model_inference_calls=0,project_or_torch_imports=False,checks=checks,failures=failures,known=total_known,joint_known=len(joint),known_disagreements=len(disagreement),annotation_status_counts=dict(statuses),clips=per_clip,source_pins={str(fresh/n):sha(fresh/n) for n in ['INPUTS.json','VISUAL_INDEX.json','RGB_LABELS_A.json','RGB_LABELS_B.json','RGB_LABELS_ROOT.json','LABEL_AUDIT.json']},verifier_sha256=sha(__file__),scope='Checks faithful reconciliation, direct-hand fixed-ROI masks and target encoding, not whether visual draft labels are expert-correct.')
    out=U/'review/QUALITY_VERIFY_FRESH_LABELS.json'
    with out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],checks=checks,failures=len(failures),output=str(out)),ensure_ascii=False))
    return int(bool(failures))

if __name__=='__main__':raise SystemExit(main())
