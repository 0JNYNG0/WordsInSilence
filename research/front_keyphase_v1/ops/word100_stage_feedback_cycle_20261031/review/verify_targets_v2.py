"""Read-only comparison of conservative ROI target revision; no training."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import json
import hashlib
from collections import Counter
import numpy as np

REVIEW = Path(__file__).resolve().parent
U = REVIEW.parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def check(v,message):
    if not bool(v):
        raise AssertionError(message)


def npz(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def main():
    oldpath=U/'before/labels_v1/LEARNING_INPUTS.json'
    newpath=U/'LEARNING_INPUTS.json'
    old,new=read(oldpath),read(newpath)
    oldaudit=read(U/'before/labels_v1/LABEL_AUDIT.json')
    newaudit=read(U/'LABEL_AUDIT.json')
    fallback=read(REVIEW/'FALLBACK_CELLS.json')
    input_report=read(REVIEW/'INPUTS_INDEPENDENT.json')
    check(sha(oldpath)==fallback['source_pins'][str(U/'LEARNING_INPUTS.json')],'frozen original learning manifest hash')
    check(sha(U/'before/labels_v1/LABEL_AUDIT.json')==fallback['source_pins'][str(U/'LABEL_AUDIT.json')],'frozen original label audit hash')
    expected={(r['alias'],r['frame_id'],r['feature']) for r in fallback['rows']}
    check(len(expected)==fallback['count']==449,'449 distinct frozen fallback cells')
    aliases=[f'P{i:03}' for i in range(1,81)]
    check([r['alias'] for r in old['rows']]==[r['alias'] for r in new['rows']]==aliases,'all80 same ordered aliases')
    check(old['all_targets']==new['all_targets'],'unchanged17 target order')
    targets=new['all_targets']
    check(len(targets)==17,'17 full targets')
    oldrows={r['alias']:r for r in old['rows']}
    oldar={r['alias']:r for r in oldaudit['rows']}
    newar={r['alias']:r for r in newaudit['rows']}
    verified_inputs={r['alias']:r for r in input_report['rows']}
    removed=set()
    row_reports=[]
    all_data={}
    roi_known_checks=0
    roi_frame_hand_checks=0
    no_hand_fallback=0
    fit_support={f:{'0':[],'1':[]} for f in targets}
    for row in new['rows']:
        a=row['alias'];prior=oldrows[a]
        before=npz(prior['targets_file']);after=npz(row['targets_file'])
        check(sha(prior['targets_file'])==prior['targets_sha256']==fallback['target_pins'][a],'old target pin '+a)
        check(sha(row['targets_file'])==row['targets_sha256'],'new target pin '+a)
        check(Path(row['targets_file']).parent.resolve()==(U/'targets_v2').resolve(),'new target version path')
        ignored={'targets_file','targets_sha256','known_cells'}
        check({k:v for k,v in prior.items() if k not in ignored}=={k:v for k,v in row.items() if k not in ignored},'unchanged input metadata '+a)
        check(sha(row['input'])==verified_inputs[a]['input_sha256']==row['input_sha256'],'unchanged verified input '+a)
        data=npz(row['input']);raw=npz(row['observation'])
        ids=data['frame_id'];lookup={int(f):i for i,f in enumerate(ids)}
        bm,nm=before['target_mask'],after['target_mask'];by,ny=before['y'],after['y']
        check(bm.shape==nm.shape==by.shape==ny.shape==(len(ids),len(targets)),'target full shape')
        check(bm.dtype==nm.dtype==bool and by.dtype==ny.dtype==np.float32,'target dtypes')
        check(np.isfinite(ny).all() and np.isin(ny[nm],[0,1]).all(),'new finite binary target')
        check(not (nm & ~bm).any(),'no new known target cell')
        check(np.array_equal(by[nm],ny[nm]),'unchanged retained target values')
        check((ny[~nm]==0).all(),'unknown values zero with false mask')
        removed_here={(a,int(ids[i]),targets[j]) for i,j in np.argwhere(bm & ~nm)}
        check(removed_here=={c for c in expected if c[0]==a},'exact expected fallback removed '+a)
        removed |= removed_here
        check(int(nm.sum())==row['known_cells'],'declared known count')
        ar,br=newar[a],oldar[a]
        check(ar['observations']==br['observations'],'unchanged RGB evidence '+a)
        check(ar['conflicts']==br['conflicts'] and ar['secondary_review']==br['secondary_review'],'unchanged conflict/review evidence '+a)
        newdrops={(e['frame_id'],e['feature']) for e in ar['roi_exclusions']}
        olddrops={(e['frame_id'],e['feature']) for e in br['roi_exclusions']}
        check(newdrops-olddrops=={(f,t) for aa,f,t in removed_here},'only449 added ROI exclusions '+a)
        check(olddrops<=newdrops,'old ROI exclusions preserved '+a)
        spans={(e['frame_id'],e['feature']):e['interval_frames'] for e in ar['observations'] if e['kind']=='interval_direction'}
        x0,y0,x1,y1=row['box']
        for i,j in np.argwhere(nm):
            fid,f=int(ids[i]),targets[j]
            sides=('R','L') if f=='hands_overlap' else (f[0],)
            if '_motion_' in f:
                check((fid,f) in spans,'motion span available')
                lo,hi=spans[(fid,f)]
                frame_ids=list(range(lo,hi+1))
            else:
                frame_ids=[fid]
            for source_id in frame_ids:
                t=lookup[source_id]
                for side in sides:
                    off=54 if side=='R' else 33
                    points=raw['xy_px'][t,off:off+21]
                    good=raw['observed'][t,off:off+21] & np.isfinite(points).all(axis=1)
                    if not good.any():
                        no_hand_fallback+=1
                    check(good.any(),'new target uses absent-hand wrist fallback')
                    check(all(x0<=float(p[0])<x1 and y0<=float(p[1])<y1 for p in points[good]),'known target hand outside ROI at original frame '+str(source_id))
                    roi_frame_hand_checks+=1
            roi_known_checks+=1
        for drop in ar['roi_exclusions']:
            check(not nm[lookup[drop['frame_id']],targets.index(drop['feature'])],'ROI excluded target reinstated')
        for j,f in enumerate(targets):
            if row['role']=='fit':
                for val in (0,1):
                    cells=int(((ny[:,j]==val)&nm[:,j]).sum())
                    if cells:
                        fit_support[f][str(val)].append(dict(alias=a,word=row['word_id'],person=row['person'],cells=cells))
        row_reports.append(dict(alias=a,old_known=int(bm.sum()),new_known=int(nm.sum()),removed=len(removed_here),
                                old_sha256=sha(prior['targets_file']),new_sha256=sha(row['targets_file']),
                                role=row['role'],frames=len(ids),status='PASS'))
    check(removed==expected,'global exactly449 removed')
    check(sum(r['old_known'] for r in row_reports)==3833,'original3833')
    check(sum(r['new_known'] for r in row_reports)==3384,'new3384')
    support_report={}
    supported=[]
    for f,vs in fit_support.items():
        calculated={}
        for val,entries in vs.items():
            calculated[val]=dict(words=sorted({e['word'] for e in entries}),people=sorted({e['person'] for e in entries}),cells=sum(e['cells'] for e in entries),clips=len(entries))
        passed=all(len(calculated[v]['words'])>=4 and len(calculated[v]['people'])>=3 for v in ('0','1'))
        expected_support=dict(passed=passed,by_value=calculated)
        check(newaudit['support'][f]==expected_support,'independent fit-only support '+f)
        support_report[f]=expected_support
        if passed:
            supported.append(f)
    check(supported==new['targets']==newaudit['selected_targets'],'selected targets from independent support')
    check(new['target_indices']==[targets.index(f) for f in supported],'selected target indices')
    check(len(supported)==10,'ten supported targets')
    removed_targets=[f for f in old['targets'] if f not in supported]
    check(removed_targets==['L_motion_up','L_motion_left'],'only support-lost left directions additionally withheld')
    check(not set(supported)-set(old['targets']),'no added target from outcomes')
    roles=dict(Counter(r['role'] for r in new['rows']))
    check(roles=={'fit':36,'development':12,'held_person':24,'held_expression_person':8},'all80 person split unchanged')
    for key in ('disagreements','reviewed_cells','joint_known','known_disagreements','known_disagreement_rate','bad'):
        check(oldaudit[key]==newaudit[key],'unchanged review disagreement '+key)
    paths=[Path(__file__),oldpath,newpath,U/'before/labels_v1/LABEL_AUDIT.json',U/'LABEL_AUDIT.json',REVIEW/'FALLBACK_CELLS.json',U/'build_labels.py']
    report=dict(status='PASS_TARGET_MASKS',reviewer='Same Codex separate subagent rgb_labels_a; arithmetic/provenance audit, not independent human truth',
                no_new_model_inference=True,no_training=True,source_data_written=False,
                pins={str(p):sha(p) for p in paths},clips=80,roles=roles,total_original_frames=sum(r['frames'] for r in row_reports),
                old_known_cells=3833,new_known_cells=3384,exact_removed_frozen_fallback_cells=449,
                removed_by_role=dict(Counter(r['role'] for r in fallback['rows'])),
                new_known_target_additions=0,retained_values_unchanged=True,all_original80_sources_retained=True,
                roi=dict(known_target_cells_checked=roi_known_checks,original_frame_hand_checks=roi_frame_hand_checks,
                         zero_hand_observation_fallbacks_retained=no_hand_fallback,interval_policy='Every original native frame from start through end inclusive',
                         scope='Observed finite hand points within fixed source ROI; not a proof of pixel-level visibility or anatomical identity'),
                selected_targets=supported,additionally_withheld_due_fit_support=removed_targets,
                support_rule='For each binary value, at least4 distinct fit words and all3 fit people; no held scores consulted',
                support=support_report,visual_semantic_preservation='NOT_ESTABLISHED',rows=row_reports)
    # Keep the final known-cell total separate from number of newly added cells.
    report['new_known_cells']=3384
    output=REVIEW/'TARGETS_V2_INDEPENDENT.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pins','rows','support')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
