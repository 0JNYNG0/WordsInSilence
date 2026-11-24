"""Independent count/change audit on cached output only; no candidate execution."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np

U=Path(__file__).resolve().parent
P=U.with_name('word100_hand_assignment_20261016')
V2=U.with_name('word100_hand_assignment_repair_20261016')

def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def category(a,b):return 'unchanged' if a==b else 'newly_assigned' if a<0 else 'dropped' if b<0 else 'side_changed'
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    return x

def run():
    assert not (U/'CHANGE_AUDIT.json').exists(),'Preserve completed audit'
    result=read(U/'REPLAY_RESULT.json');protected=read(U/'PROTECTED_FILES.json')
    pins=dict(protected)
    for r in result['rows']:pins.update(r['outputs'])
    for p in [Path(__file__),U/'REPLAY_RESULT.json',U/'BROADENED_CAPTURE_RESULT.json',U/'BROADENED_TRAIN_PLAN.json']:
        pins[str(p)]=sha(p)
    for p,h in pins.items():assert sha(p)==h,p
    plans={r['clip_id']:r for r in read(P/'RAW_CAPTURE_CONTEXT.json')['items']}
    plans.update({r['clip_id']:r for r in read(U/'BROADENED_TRAIN_PLAN.json')['items']})
    new_codes={r['clip_id'] for r in read(U/'BROADENED_TRAIN_PLAN.json')['items']}
    groups={name:dict(conditions=0,frames=0,raw_detections=0,versus_baseline=Counter(),versus_v2=Counter())
        for name in ['known_REAL01_24','additional_REAL02_REAL03_16']}
    changes=[];stores={};row_reports=[]
    for row in result['rows']:
        cid,mode=row['clip_id'],row['mode'];group='additional_REAL02_REAL03_16' if cid in new_codes else 'known_REAL01_24'
        rd=Path(row['raw_dir']);dest=U/'replay'/cid/mode
        raw=load(rd/'raw.npz');base=load(rd/'baseline.npz')
        prev=load(dest/'previous_assignment.npz')['sides'];new=load(dest/'assignment.npz')['sides']
        candidate=load(dest/'candidate.npz');trace=read(dest/'trace.json')
        stores[cid,mode]=dict(raw=raw,prev=prev,new=new,candidate=candidate,baseline=base)
        gs=groups[group];gs['conditions']+=1;gs['frames']+=len(raw['raw_count']);gs['raw_detections']+=int(raw['raw_count'].sum())
        c0,c1=Counter(),Counter();actual0=[];actual1=[];details=[]
        a,b=plans[cid]['core_seconds']
        for f in range(len(raw['raw_count'])):
            n=int(raw['raw_count'][f]);now=float(raw['timestamp_us'][f]/1e6)
            for i in range(n):
                bs,ps,ns=int(raw['baseline_side'][f,i]),int(prev[f,i]),int(new[f,i])
                q0,q1=category(bs,ns),category(ps,ns);c0[q0]+=1;c1[q1]+=1
                if q0!='unchanged':actual0.append((f,i,bs,ns,q0))
                if q1!='unchanged':actual1.append((f,i,ps,ns,q1))
                if q0=='unchanged' and q1=='unchanged':continue
                pts=raw['hand_xy_px'][f,i];w,h=int(raw['width'][f]),int(raw['height'][f]);fin=np.isfinite(pts).all(1)
                inside=fin&(pts[:,0]>=0)&(pts[:,0]<w)&(pts[:,1]>=0)&(pts[:,1]<h)
                old_target=[j for j in range(n) if raw['baseline_side'][f,j]==ns] if ns>=0 else []
                prior_target=[j for j in range(n) if prev[f,j]==ns] if ns>=0 else []
                addition0=None if q0!='newly_assigned' else ('raw_replacement_of_occupied_side' if old_target else 'additional_recorded_side_needs_visual_truth')
                addition1=None if q1!='newly_assigned' else ('raw_replacement_of_occupied_side' if prior_target else 'additional_recorded_side_needs_visual_truth')
                d=dict(group=group,clip_id=cid,label=row['label'],mode=mode,frame=f,raw_index=i,
                    timestamp_seconds=now,phase='before' if now<a else 'after' if now>b else 'core',
                    baseline_side=bs,v2_side=ps,joint_side=ns,versus_baseline=q0,versus_v2=q1,
                    raw_label=int(raw['raw_label'][f,i]),raw_score=float(raw['raw_score'][f,i]),
                    strict_inside_points=int(inside.sum()),strict_palm5_inside_points=int(inside[[0,5,9,13,17]].sum()),
                    pose_wrists_xy=raw['pose_xy_px'][f,[15,16]],pose_wrists_usable=raw['pose_usable'][f,[15,16]],
                    reason=trace[f]['reasons'][i],baseline_raws_for_target_side=old_target,v2_raws_for_target_side=prior_target,
                    added_type_vs_baseline=addition0,added_type_vs_v2=addition1,
                    candidate_any_hand_observed=bool(candidate['observed'][f,33:].any()),
                    candidate_any_hand_usable=bool(candidate['usable'][f,33:].any()))
                if bs>=0:
                    sl=slice(33+21*bs,54+21*bs)
                    d.update(baseline_usable_points=int(base['usable'][f,sl].sum()),
                        baseline_side_still_observed=bool(candidate['observed'][f,sl].any()))
                changes.append(d);details.append(d)
        for key in ['unchanged','newly_assigned','side_changed','dropped']:
            assert c0[key]==row['counts'].get(key,0),(cid,mode,'baseline',key)
            assert c1[key]==row['counts_vs_previous'].get(key,0),(cid,mode,'v2',key)
        declared0=read(dest/'changes.json');declared1=read(dest/'changes_vs_previous.json')
        assert actual0==[(r['frame'],r['raw_index'],r['old_side'],r['new_side'],r['category']) for r in declared0]
        assert actual1==[(r['frame'],r['raw_index'],r['previous_side'],r['new_side'],r['category']) for r in declared1]
        gs['versus_baseline'].update(c0);gs['versus_v2'].update(c1)
        row_reports.append(dict(clip_id=cid,label=row['label'],mode=mode,group=group,
            versus_baseline=c0,versus_v2=c1,changed_frames_vs_baseline=sorted({r[0] for r in actual0}),
            changed_frames_vs_v2=sorted({r[0] for r in actual1})))
    expected={('NIA_SL_WORD1139_REAL01_F','native',71,1,0),
        ('NIA_SL_WORD1139_REAL01_F','crop',56,1,0),('NIA_SL_WORD0106_REAL01_F','crop',52,0,1)}
    target_checks=[]
    for cid,mode,f,i,side in sorted(expected):
        st=stores[cid,mode]
        target_checks.append(dict(clip_id=cid,mode=mode,frame=f,raw_index=i,expected_side=side,
            v2_side=int(st['prev'][f,i]),joint_side=int(st['new'][f,i]),pass_check=int(st['new'][f,i])==side))
    actual_known={(c['clip_id'],c['mode'],c['frame'],c['raw_index'],c['joint_side']) for c in changes
        if c['group']=='known_REAL01_24' and c['versus_v2']!='unchanged'}
    additional_new=[c for c in changes if c['group']=='additional_REAL02_REAL03_16' and c['versus_v2']!='unchanged']
    visual_frames=defaultdict(list)
    for c in changes:
        if c['group']=='additional_REAL02_REAL03_16':
            visual_frames[c['clip_id'],c['mode'],c['frame']].append(c)
    visual=[]
    for (cid,mode,f),ds in sorted(visual_frames.items()):
        visual.append(dict(clip_id=cid,label=ds[0]['label'],mode=mode,frame=f,
            priority='joint_new_difference' if any(d['versus_v2']!='unchanged' for d in ds) else 'existing_v2_difference_vs_production',
            raw_changes=[dict(raw_index=d['raw_index'],baseline_side=d['baseline_side'],v2_side=d['v2_side'],joint_side=d['joint_side'],
                versus_baseline=d['versus_baseline'],versus_v2=d['versus_v2'],raw_score=d['raw_score']) for d in ds]))
    prior_review=read(V2/'CHANGE_AUDIT.json')
    preserved=[]
    for c in prior_review['all_changes']:
        if c['versus_baseline'] not in ['side_changed','newly_assigned']:continue
        st=stores[c['clip_id'],c['mode']]
        actual=int(st['new'][c['frame'],c['raw_index']])
        preserved.append(dict(clip_id=c['clip_id'],mode=c['mode'],frame=c['frame'],raw_index=c['raw_index'],
            previous_side=c['v2_side'],joint_side=actual,preserved=actual==c['v2_side']))
    for p,h in pins.items():assert sha(p)==h,p
    report=dict(status='SAVED_ARRAY_NUMERIC_AUDIT_COMPLETE_VISUAL_REVIEW_SEPARATE',
        method='Independent standard library/NumPy saved assignment counts; no HandAssigner or production module imports',
        groups=groups,conditions=len(result['rows']),rows=row_reports,target_checks=target_checks,
        all_target_checks_pass=all(x['pass_check'] for x in target_checks),
        known24_only_three_target_detections_changed=actual_known==expected,
        previous_ten_side_and_two_added_observations=preserved,
        previous_repairs_all_preserved=all(x['preserved'] for x in preserved),
        added16_new_differences_vs_v2=additional_new,
        added16_no_new_drops_vs_v2=not any(c['versus_v2']=='dropped' for c in additional_new),
        all_changes=changes,visual_targets=visual,
        protected_files_unchanged=len(protected),total_pinned_files_unchanged=len(pins),
        limitation='Changed detection counts and missing/usable rates are not accuracy. Added TRAIN words/signers02/03 are engineering regression checks, not untouched recognition evaluation. Existing v2 differences need visual review even if joint introduces none.',
        candidate_execution=0,classifier_inference=0,training=0,
        input_hashes={str(p):sha(p) for p in [U/'REPLAY_RESULT.json',U/'BROADENED_TRAIN_PLAN.json',Path(__file__)]})
    with (U/'CHANGE_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(clean(report),f,ensure_ascii=False,indent=2,allow_nan=False)
    with (U/'NUMERIC_VISUAL_TARGETS.json').open('x',encoding='utf-8') as f:json.dump(clean(visual),f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(clean(dict(groups=groups,target_checks=target_checks,
        known24_only_three_target_detections_changed=actual_known==expected,previous_repairs_all_preserved=all(x['preserved'] for x in preserved),
        added16_new_differences=len(additional_new),visual_frames=len(visual),protected_files_unchanged=len(protected))),ensure_ascii=False))

if __name__=='__main__':run()
