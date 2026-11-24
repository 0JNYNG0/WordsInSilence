"""FIT-only sparse physical-label lineage/context audit. No model/media calls."""
from pathlib import Path
from collections import Counter
import hashlib, json, sys
import numpy as np

U=Path(__file__).resolve().parent; O=U.parent
P=O/'word100_learned_guide_feedback_20261102'
M=O/'word100_multipeople_live_feedback_20261101'
SEM=O/'word100_semantic_feature_cycle_20261101'
CORE=O/'word100_core_feature_split_20261101'
V=O/'word100_visible_temporal_targets_20261101'
Q=O/'word100_predicate_supervision_20261102'
S=O/'word100_source_component_connector_20261102'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={};checks=0

def load(p):
    pins[str(p)]=sha(p)
    return read(p)

def nullable(r,targets,order):
    return [int(r['y'][targets.index(t)]) if r['target_mask'][targets.index(t)] else None for t in order]

def direct_values(r,targets):
    result={}
    for h in 'RL':
        for name,val in zip(('index_only','open_four','closed_four'),r['center_shape'][h]):
            result[h+'_'+name]=val if r['roi_center'][h] is True else None
    for name in ('R_higher','L_higher'):
        rel=r['center_relation']
        result[name]=int(rel==name) if rel is not None and all(r['roi_center'][h] is True for h in 'RL') else None
    return [result[t] for t in targets]

def counts(rr,index):
    return {name:sum(r['values'][index]==value for r in rr) for name,value in [('positive',1),('negative',0),('unknown',None)]}

def compact_report(result):
    """Reduce repeated paths/keys, retaining exact pointers and every target value."""
    result=dict(result)
    artifacts={}
    def intern_paths(value):
        if isinstance(value,str) and (value.startswith(str(O)) or value.startswith('C:\\MyUnityProjects\\WordsInSilence\\research\\front_keyphase_v1\\ops\\')):
            key=next((k for k,v in artifacts.items() if v==value),None)
            if key is None:key='file'+str(len(artifacts)+1);artifacts[key]=value
            return '@'+key
        if isinstance(value,list):return [intern_paths(x) for x in value]
        if isinstance(value,dict):return {k:intern_paths(v) for k,v in value.items()}
        return value
    centers=result['center_bindings']
    # These keys are uniquely determined by the clip_id and already checked.
    columns=['clip_id','center_frame','timestamp_us','values','manifest_pointer','train_dataset_pointer',
             'label_origin','center_image_sha256','stage_overlaps']
    result['center_binding_columns']=columns
    result['center_bindings']=[[intern_paths(r[k]) for k in columns] for r in centers]
    result['center_binding_role']='fit; word/person encoded in clip_id and recorded in source_person_support'
    result['source_person_count_columns']=['clip_id','counts_in_target_order_positive_negative_unknown']
    result['source_person_support']=[[r['clip_id'],[[r['target_counts'][t][k] for k in ('positive','negative','unknown')]
                                    for t in result['targets']]] for r in result['source_person_support']]
    # Each center image digest remains in its exact manifest-bound row. Avoid a second image path/hash copy.
    image_paths={str(Path(read(P/'INPUT_MANIFEST.json')['rows'][int(r['manifest_pointer'].split('/')[2])]
                              ['labels'][int(r['manifest_pointer'].split('/')[4])]['center_image'])) for r in centers}
    result['pins']={p:h for p,h in result['pins'].items() if p not in image_paths}
    result['image_hash_policy']='All108 existing center files hashed; hashes stay in center_bindings, exact paths are resolved through pinned INPUT_MANIFEST pointers. No image was newly decoded or visually reannotated.'
    result['artifacts']=artifacts
    result['pins'][str(Path(__file__))]=sha(Path(__file__))
    return result

def main():
    global checks
    out=U/'PHYSICAL_GUIDE_BINDING_AUDIT.json';assert not out.exists()
    manifest=load(P/'INPUT_MANIFEST.json');guide=load(P/'GUIDE_SUPPORT.json')
    train=load(M/'TRAIN_DATASET.json');sem=load(SEM/'DATASET.json');core=load(CORE/'DATASET.json')
    prior=load(V/'DATASET.json');expanded=load(M/'data_audit/LABELS.json')
    manual=load(M/'data_audit/MANUAL_CENTER_REVIEW.json');rubric=load(CORE/'RUBRIC.json')
    load(V/'RUBRIC.json')
    targets=manifest['targets'];assert targets==train['targets']==sem['targets']==expanded['targets']==manual['targets']
    fit=[(i,r) for i,r in enumerate(manifest['rows']) if r['role']=='fit']
    assert len(fit)==36 and {r['person'] for _,r in fit}==set(range(1,10))
    fitids={r['clip_id'] for _,r in fit}
    by_train={(r['clip_id'],r['center_frame']):(i,r) for i,r in enumerate(train['rows']) if r['clip_id'] in fitids}
    by_sem={(r['clip_id'],r['center_frame']):(i,r) for i,r in enumerate(sem['rows']) if r['clip_id'] in fitids}
    by_core={(r['clip_id'],r['center_frame']):(i,r) for i,r in enumerate(core['rows']) if r['clip_id'] in fitids}
    by_prior={(r['clip_id'],r['center_frame']):(i,r) for i,r in enumerate(prior['rows']) if r['clip_id'] in fitids}
    by_new={(r['clip_id'],r['center_frame']):(i,r) for i,r in enumerate(expanded['rows']) if r['clip_id'] in fitids}
    desired={r['sample_id'] for _,r in by_prior.values()}
    direct={}
    for name in ('LABELS_A.json','LABELS_B.json'):
        f=V/'review'/name;j=load(f)
        for i,r in enumerate(j['rows']):
            if r['sample_id'] in desired:direct[r['sample_id']]=(f'/rows/{i}',str(f),r)
    f=V/'review/LABELS_SECONDARY.json';second=load(f)
    secondary={r['sample_id']:(f'/rows/{i}',str(f),r) for i,r in enumerate(second['rows']) if r['sample_id'] in desired}
    for p,h in manifest['source_pins'].items():
        if p in pins:assert pins[p]==h;checks+=1
    records=[];axes={};cohorts=Counter();stage_counts=Counter();max_dt=0
    for ri,r in fit:
        assert r['person']<=9 and len(r['labels'])==3
        f=Path(r['input']);assert sha(f)==r['input_sha256'];pins[str(f)]=r['input_sha256'];checks+=1
        with np.load(f,allow_pickle=False) as z:
            frames=z['frame_id'].astype(int).tolist();times=z['timestamp_us'].astype(int).tolist()
        assert len(frames)==len(set(frames)) and all(a<b for a,b in zip(frames,frames[1:]))
        axes[r['clip_id']]=dict(zip(frames,times));checks+=1
        stage_counts[r['stage_supervision']]+=len(r['stages'])
        assert (r['person']==1)==(r['stage_supervision']=='canonical_visual')
        for li,l in enumerate(r['labels']):
            key=(r['clip_id'],l['center_frame']);values=nullable(l,targets,targets)
            ti,t=by_train[key]
            assert t==l and l['role']=='fit' and l['person']==r['person'];checks+=1
            assert len(l['frame_ids'])==13 and l['frame_ids'][6]==l['center_frame']
            assert [axes[r['clip_id']][k] for k in l['frame_ids']]==l['timestamps_us']
            assert l['center_timestamp_us']==l['timestamps_us'][6];checks+=3
            origin={};cohort='prior_direct_RGB' if r['person']<=4 else 'additional_direct_RGB'
            if r['person']<=4:
                si,s=by_sem[key];ci,c=by_core[key];vi,v=by_prior[key]
                assert values==nullable(s,sem['targets'],targets)==nullable(c,core['targets'],targets)==nullable(v,prior['targets'],targets)
                assert c['prior_dataset_sha256']==pins[str(V/'DATASET.json')]
                pointer,path,a=direct[v['sample_id']];resolved=direct_values(a,targets)
                secondary_pointer=None
                if v['sample_id'] in secondary:
                    sp,sf,b=secondary[v['sample_id']];bv=direct_values(b,targets)
                    resolved=[x if x==y else None for x,y in zip(resolved,bv)]
                    secondary_pointer={'path':sf,'pointer':sp}
                assert resolved==values;checks+=3
                origin=dict(primary={'path':path,'pointer':pointer},secondary=secondary_pointer,
                            prior_dataset_pointer=f'/rows/{vi}',semantic_dataset_pointer=f'/rows/{si}',
                            core_dataset_pointer=f'/rows/{ci}',roi_and_disagreement_masks_preserved=True)
            else:
                ni,n=by_new[key];pro=n['source_provenance'];alias=pro['alias']
                k=int(pro['sample_id'].rsplit('W',1)[1]);m=manual['rows'][alias]
                assert values==nullable(n,expanded['targets'],targets)==m['y'][k]
                assert pro['type']=='NEW_EXACT_CENTER_RGB_PRIMARY_REVIEW'
                origin=dict(primary={'path':str(M/'data_audit/MANUAL_CENTER_REVIEW.json'),'pointer':f'/rows/{alias}/y/{k}'},
                            packaged_pointer=f'/rows/{ni}',same_codex_not_independent_human=True)
                checks+=2
            image=Path(l['center_image']);image_digest=sha(image);pins[str(image)]=image_digest
            if 'center_image_sha256' in l:assert image_digest==l['center_image_sha256'];checks+=1
            # Sparse center overlap only. The 13-frame context or a DTW span is never labelled.
            overlaps=[]
            for sti,stage in enumerate(r['stages']):
                lo,hi=stage['frames']
                if lo<=l['center_frame']<=hi:
                    overlaps.append(dict(stage_id=stage['canonical_stage_id'],stage_pointer=f'/rows/{ri}/stages/{sti}',
                                         relation='center_inside_context_only',stage_scope=r['stage_supervision'],
                                         clause_ids=sorted({x['clause_id'] for x in stage['source_cards']})))
            records.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],role='fit',
                                center_frame=l['center_frame'],timestamp_us=l['center_timestamp_us'],values=values,
                                manifest_pointer=f'/rows/{ri}/labels/{li}',train_dataset_pointer=f'/rows/{ti}',
                                label_origin=origin,center_image_sha256=image_digest,stage_overlaps=overlaps))
            cohorts[cohort]+=1
    assert len(records)==108 and len({(r['clip_id'],r['center_frame']) for r in records})==108
    summary={}
    for j,t in enumerate(targets):
        totals=counts(records,j);assert totals==guide['label_counts']['fit|'+t]
        summary[t]=dict(totals=totals,**{name:{'people':sorted({r['person'] for r in records if r['values'][j]==v}),
                'words':sorted({r['word_id'] for r in records if r['values'][j]==v}),
                'source_clips':len({r['clip_id'] for r in records if r['values'][j]==v})}
                for name,v in [('positive_support',1),('negative_support',0)]})
    person_support=[dict(clip_id=r['clip_id'],person=r['person'],word_id=r['word_id'],
                    target_counts={t:counts([q for q in records if q['clip_id']==r['clip_id']],j) for j,t in enumerate(targets)}) for _,r in fit]
    # Later typed source-component evidence already exists; preserve each native scope.
    typed=load(S/'DIRECT_TARGETS.json');oldtyped=load(Q/'TYPED_TARGETS_v2.json')
    links=load(Q/'CLAUSE_COMPONENT_LINKS.json');review=load(S/'INDEPENDENT_PREPARATION_REVIEW.json')
    assert review['status'].startswith('PASS')
    existing=[];typed_summary={};same_point=[]
    bypoint={(r['clip_id'],r['center_frame']):r for r in records}
    for i,t in enumerate(typed['rows']):
        if t['role']!='fit':continue
        assert t['person'] in range(1,10) and t['clip_id'] in fitids
        assert t['whole_clause_truth'] is None and t['learner_error'] is None
        for frame,time in zip(t['native_frames'],t['native_timestamps_us']):
            dt=abs(axes[t['clip_id']][frame]-time);max_dt=max(max_dt,dt);assert dt<=2;checks+=1
        val=int(t['y']) if t['mask'] else None
        assert val in (0,1,None)
        row=dict(pointer=f'/rows/{i}',target=t['target'],kind=t['kind'],clip_id=t['clip_id'],person=t['person'],
                 value=val,native_frames=t['native_frames'],source_clause_ids=t['source_clause_ids'])
        existing.append(row)
        if t['target']=='right_non_thumb_closed' and t['kind']=='point' and len(t['native_frames'])==1:
            p=bypoint.get((t['clip_id'],t['native_frames'][0]))
            if p:
                pv=p['values'][targets.index('R_closed_four')]
                assert val is None or pv is None or val==pv
                same_point.append(dict(clip_id=t['clip_id'],frame=t['native_frames'][0],P_value=pv,S_value=val))
    assert len(existing)==550
    for target in sorted({r['target'] for r in existing}):
        rr=[r for r in existing if r['target']==target]
        typed_summary[target]=dict(kinds={k:dict(Counter('unknown' if r['value'] is None else 'positive' if r['value'] else 'negative' for r in rr if r['kind']==k)) for k in sorted({r['kind'] for r in rr})},
                                   source_clips=len({r['clip_id'] for r in rr}),
                                   positive_people=sorted({r['person'] for r in rr if r['value']==1}),
                                   negative_people=sorted({r['person'] for r in rr if r['value']==0}),
                                   record_pointers=[r['pointer'] for r in rr])
    for f,h in typed['pins'].items():
        if f in pins:assert pins[f]==h;checks+=1
    overlap_counts=Counter()
    for r in records:
        if not r['stage_overlaps']:overlap_counts['no_stage_overlap']+=1
        for ov in r['stage_overlaps']:overlap_counts[ov['stage_scope']]+=1
    result=dict(status='PASS_FIT_SPARSE_PHYSICAL_PROVENANCE_AND_CONTEXT_AUDIT_NOT_SEMANTIC_GOLD',
        checks=checks,fit_clips=36,fit_people=list(range(1,10)),center_labels=108,targets=targets,
        definitions={'shape':rubric['shape_definitions'],'relation':rubric['relation_definition'],
                     'negative':'Directly visibly incompatible with this exact physical predicate, not an incorrect word or missing required step.',
                     'unknown':'ROI/visibility/identity/ambiguous shape or review disagreement; zero-filled numeric y is masked, not a negative.'},
        fit_support=summary,source_person_support=person_support,provenance_counts=dict(cohorts),
        center_bindings=records,stage_spans_by_scope=dict(stage_counts),center_stage_overlaps=dict(overlap_counts),
        overlap_policy='Center lies in a stored visual/DTW span only supplies context. No stage-wide label, no source-clause agreement/contradiction target, no exact event boundary.',
        later_evidence=dict(source=str(S/'DIRECT_TARGETS.json'),fit_records=550,typed_support=typed_summary,
             definitions_source=str(Q/'TYPED_TARGETS_v2.json')+'#/types',prior_independent_preparation_review=review['status'],
             same_definition_R_closed_four_intersections=same_point,native_time_max_error_us=max_dt,
             runtime_supported_types=['right_non_thumb_closed','right_hand_at_left_arm_projection','bilateral_upper_ends_converged','right_hand_traverses_left_arm_toward_wrist'],
             already_available_not_newly_missing=['Direct thumb-up with closed finger masses has9positive and9negative FIT points; not inferable from closed_four alone.',
                'Arm projection, convergence, oblique finger axes, directed traversal and alternation have separate typed observations beyond these7center outputs.',
                'Native point, interval and sparse-sequence scopes stay separate. Multiple records are correlated within36source clips.'],
             still_unsupported=['Variant-specific lowering and all-fingertip gathering remain unknown in this typed set.',
                'Physical source-component observations do not label whole source clauses, exact contact, arbitrary required repeats, or educational acceptance tolerance.']),
        learning_contract_recommendation=[
            'Reuse exact observed center supervision and later typed labels with their masks/scopes; no need to relabel all observations or add a detector.',
            'R_index_only has no positive FIT example in these4expressions: retain the limitation; do not train a newly usable positive detector from this subset alone.',
            'Link target-guide expected components separately from the independent observed branch. Source text cannot turn unknown observed fingers into expected fingers.',
            'Before a learned guide-difference claim, directly verify source-form and per-clause temporal binding plus visible contrary and acceptable-variation examples. Mere canonical/DTW overlap is insufficient.',
            'Do not equate open_four0 to closed_four1, closed_four1 to complete fist, palm wording to open_four1, higher to alternation, or image convergence to actual contact.'
        ],
        new_labels=0,new_media=0,new_model_inference=0,new_training=0,development_rows_used=0,
        historical_metadata_note='Shared manifests contain other roles; only explicit FIT01-09 records enter this audit. One initial key inventory displayed existing aggregate role-count metadata; no DEV records/predictions/media were analyzed or used for design.',
        first_inspection_issue='Initial read-only console inventory hit cp949 UnicodeEncodeError; rerun with PYTHONUTF8=1, no source/artifact mutation.',
        expert_truth=False,whole_goal_complete=False,pins=pins)
    pins[str(Path(__file__))]=sha(Path(__file__))
    with out.open('x',encoding='utf8') as f:json.dump(compact_report(result),f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    print(json.dumps({'status':result['status'],'checks':checks,'centers':108,'typed_fit_records':550,
                      'stage_counts':result['center_stage_overlaps'],'same_definition_intersections':len(same_point),
                      'output_bytes':out.stat().st_size},ensure_ascii=False))

if __name__=='__main__':
    if '--compact-saved' in sys.argv:
        p=U/'PHYSICAL_GUIDE_BINDING_AUDIT.json';old=U/'PHYSICAL_GUIDE_BINDING_AUDIT_expanded.json'
        assert p.exists() and not old.exists()
        data=p.read_bytes();j=json.loads(data);assert 'center_binding_columns' not in j
        old.write_bytes(data)
        j=compact_report(j);j['lossless_report_reformat_from_sha256']=hashlib.sha256(data).hexdigest()
        p.write_text(json.dumps(j,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')
        print('Saved report compacted only; no source re-audit/model execution',p.stat().st_size)
    else:main()
