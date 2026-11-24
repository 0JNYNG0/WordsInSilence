"""Independent v2 saved-array audit. No imports from either candidate unit."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
sys.dont_write_bytecode = True
import numpy as np

U = Path(__file__).resolve().parent
P = U.with_name('word100_hand_assignment_20261016')
PALM = [0, 5, 9, 13, 17]


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def clean(v):
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, (np.integer, np.bool_)): return v.item()
    if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
    if isinstance(v, list): return [clean(x) for x in v]
    if isinstance(v, tuple): return [clean(x) for x in v]
    if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
    return v


def kind(a,b):
    return 'unchanged' if a==b else 'newly_assigned' if a<0 else 'dropped' if b<0 else 'side_changed'


def runs(mask, timestamps):
    ids=np.flatnonzero(mask)
    if not len(ids): return []
    groups=np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1)
    cadence=int(np.median(np.diff(timestamps)))
    return [dict(first_frame=int(g[0]),last_frame=int(g[-1]),frames=len(g),
        start_seconds=float(timestamps[g[0]]/1e6),
        duration_seconds=float(((timestamps[g[-1]+1] if g[-1]+1<len(timestamps)
        else timestamps[g[-1]]+cadence)-timestamps[g[0]])/1e6)) for g in groups]


def run():
    assert not (U/'CHANGE_AUDIT.json').exists(),'Completed audit is immutable'
    results=read(U/'REPLAY_RESULT.json'); prior_result=read(P/'REPLAY_RESULT.json')
    protected=read(U/'PROTECTED_FILES.json')
    assert len(protected)==362, len(protected)
    pins=dict(protected)
    for row in results['rows']: pins.update(row['outputs'])
    for p in [U/'REPLAY_RESULT.json',U/'REPLAY_CONTEXT.json',Path(__file__)]: pins[str(p.resolve())]=sha(p)
    for p,h in pins.items(): assert sha(p)==h,p
    manifest={r['clip_id']:r for r in read(P/'RAW_CAPTURE_CONTEXT.json')['items']}
    previous_corrections=read(P/'CHANGE_AUDIT.json')['side_changes_and_new_assignments']
    prod_counts,v1_counts=Counter(),Counter()
    all_changes,missing_rows,conservation=[],[],[]
    stores={}
    for row in results['rows']:
        cid,mode=row['clip_id'],row['mode']; key=(cid,mode)
        raw=load(P/'raw_capture'/cid/mode/'raw.npz')
        old=load(P/'raw_capture'/cid/mode/'baseline.npz')
        v1=load(P/'replay'/cid/mode/'candidate.npz')
        v2=load(U/'replay'/cid/mode/'candidate.npz')
        s1=load(P/'replay'/cid/mode/'assignment.npz')['sides']
        s2=load(U/'replay'/cid/mode/'assignment.npz')['sides']
        trace=read(U/'replay'/cid/mode/'trace.json')
        stores[key]=dict(raw=raw,baseline=old,v1=v1,v2=v2,s1=s1,s2=s2,label=row['label'])
        t=raw['timestamp_us'];a,b=manifest[cid]['core_seconds']
        checks=[]
        for column in ['timestamp_us','frame_id']:
            assert np.array_equal(old[column],v2[column]);checks.append(column)
        for column in ['xy_px','conf']:
            assert np.array_equal(old[column][:,:33],v2[column][:,:33],equal_nan=True); checks.append('pose_'+column)
        for column in ['observed','usable']:
            assert np.array_equal(old[column][:,:33],v2[column][:,:33]); checks.append('pose_'+column)
        for f in range(len(t)):
            n=int(raw['raw_count'][f]); assigned=[int(s) for s in s2[f,:n] if s>=0]
            assert len(assigned)==len(set(assigned))
            for i in range(n):
                pside,qside,rside=int(raw['baseline_side'][f,i]),int(s1[f,i]),int(s2[f,i])
                kp,k1=kind(pside,rside),kind(qside,rside)
                prod_counts[kp]+=1;v1_counts[k1]+=1
                if rside>=0:
                    sl=slice(33+21*rside,54+21*rside)
                    assert np.array_equal(v2['xy_px'][f,sl],raw['hand_xy_px'][f,i].astype(np.float32),equal_nan=True)
                if kp=='unchanged' and k1=='unchanged': continue
                points=raw['hand_xy_px'][f,i];w,h=int(raw['width'][f]),int(raw['height'][f])
                finite=np.isfinite(points).all(1)
                inside=finite&(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
                before_slot=[j for j in range(n) if int(raw['baseline_side'][f,j])==rside] if rside>=0 else []
                v1_slot=[j for j in range(n) if int(s1[f,j])==rside] if rside>=0 else []
                phase='before' if t[f]/1e6<a else 'after' if t[f]/1e6>b else 'core'
                addition_type=None
                if kp=='newly_assigned':
                    addition_type='raw_replacement_in_already_occupied_anatomical_slot' if before_slot else 'new_recorded_anatomical_slot_requires_visual_truth'
                rec=dict(clip_id=cid,label=row['label'],mode=mode,frame=f,raw_index=i,
                    timestamp_seconds=float(t[f]/1e6),phase=phase,baseline_side=pside,v1_side=qside,v2_side=rside,
                    versus_baseline=kp,versus_v1=k1,raw_label=int(raw['raw_label'][f,i]),raw_score=float(raw['raw_score'][f,i]),
                    strict_inside_points=int(inside.sum()),strict_palm5_inside_points=int(inside[PALM].sum()),
                    pose_wrists_xy=raw['pose_xy_px'][f,[15,16]],pose_wrists_usable=raw['pose_usable'][f,[15,16]],
                    baseline_raw_indices_for_v2_side=before_slot,v1_raw_indices_for_v2_side=v1_slot,
                    new_assignment_type=addition_type,candidate_reason=trace[f]['reasons'][i],
                    v2_any_hand_observed=bool(v2['observed'][f,33:].any()),v2_any_hand_usable=bool(v2['usable'][f,33:].any()))
                if pside>=0:
                    osl=slice(33+21*pside,54+21*pside)
                    rec.update(baseline_usable_points=int(old['usable'][f,osl].sum()),
                        baseline_slot_still_observed=bool(v2['observed'][f,osl].any()))
                all_changes.append(rec)
            if n==0:
                assert not v2['observed'][f,33:].any() and np.isnan(v2['xy_px'][f,33:]).all()
        conservation.append(dict(clip_id=cid,mode=mode,checks=checks,coordinates_are_raw_only=True,
            unique_side_assignment=True,zero_detection_frames_preserved=int((raw['raw_count']==0).sum())))
        gap=dict(clip_id=cid,label=row['label'],mode=mode,versus_baseline=[],versus_v1=[])
        for name,arr in [('versus_baseline',old),('versus_v1',v1)]:
            for side in [0,1]:
                sl=slice(33+21*side,54+21*side)
                missing=arr['usable'][:,sl].any(1)&~v2['usable'][:,sl].any(1)
                gap[name].append(dict(side=side,new_missing_frames=int(missing.sum()),runs=runs(missing,t)))
        missing_rows.append(gap)
    for k in ['unchanged','dropped','side_changed','newly_assigned']:
        assert prod_counts[k]==results['counts'].get(k,0),(k,prod_counts,results['counts'])
    preserved=[]
    for c in previous_corrections:
        if c['category']!='side_changed':continue
        st=stores[c['clip_id'],c['mode']]
        actual=int(st['s2'][c['frame'],c['raw_index']])
        preserved.append(dict(clip_id=c['clip_id'],label=c['label'],mode=c['mode'],frame=c['frame'],
            raw_index=c['raw_index'],previous_corrected_side=c['new_side'],v2_side=actual,preserved=actual==c['new_side']))
    assert len(preserved)==9
    fixtures=[]
    def fixture(cid,mode,frame,side,raw_index=None):
        st=stores[cid,mode]; n=int(st['raw']['raw_count'][frame])
        selected=[i for i in range(n) if st['s2'][frame,i]==side]
        before=[i for i in range(n) if st['s1'][frame,i]==side]
        check=selected==[raw_index] if raw_index is not None else len(selected)==1
        fixtures.append(dict(clip_id=cid,label=st['label'],mode=mode,frame=frame,side=side,
            expected_raw_index=raw_index,v1_raw_indices=before,v2_raw_indices=selected,pass_check=check))
    fixture('NIA_SL_WORD1139_REAL01_F','crop',49,0)
    for f in [63,64,65]: fixture('NIA_SL_WORD2177_REAL01_F','native',f,1,1)
    fixture('NIA_SL_WORD2177_REAL01_F','crop',78,1)
    fixture('NIA_SL_WORD0500_REAL01_F','crop',47,1,1)
    added=[r for r in all_changes if r['versus_baseline']=='newly_assigned']
    dropped=[r for r in all_changes if r['versus_baseline']=='dropped']
    new_losses=[r for r in all_changes if r['versus_v1']=='dropped']
    changed_after_v1=[r for r in all_changes if r['versus_v1']!='unchanged']
    for p,h in pins.items():assert sha(p)==h,p
    report=dict(status='NUMERIC_AUDIT_COMPLETE_VISUAL_INTERPRETATION_SEPARATE',
        method='No candidate import or execution: exact saved raw/baseline/v1/v2 array comparison',
        protected362_files_unchanged=True,total_pinned_files_unchanged=len(pins),conditions=len(stores),
        counts_vs_baseline=prod_counts,counts_vs_v1=v1_counts,
        previous_nine_side_corrections=preserved,previous_nine_all_preserved=all(r['preserved'] for r in preserved),
        named_recovery_fixtures=fixtures,named_recovery_fixtures_all_pass=all(r['pass_check'] for r in fixtures),
        raw_new_assignments=added,
        newly_recorded_slot_count=sum(r['new_assignment_type']=='new_recorded_anatomical_slot_requires_visual_truth' for r in added),
        raw_replacement_count=sum(r['new_assignment_type']=='raw_replacement_in_already_occupied_anatomical_slot' for r in added),
        confirmed_previous_true_additional_hand_fixture='believe crop49 left; previously visually verified. Other added slots are not automatically true extra observations.',
        all_changes=all_changes,changes_since_v1=changed_after_v1,new_raw_drops_since_v1=new_losses,
        drops_vs_baseline=dropped,drop_phases=Counter(r['phase'] for r in dropped),
        drops_all21_inside_score_ge_0_8=[r for r in dropped if r['strict_inside_points']==21 and r['raw_score']>=.8],
        conservation=conservation,missing_intervals=missing_rows,
        limitation='Counts are assignment changes, not accuracy. Replacing a raw detection in one anatomical slot does not add a hand. Semantic correctness requires visual review; all24 are previously inspected TRAIN fixtures, not heldout recognition.',
        inputs={str(p):sha(p) for p in [U/'REPLAY_RESULT.json',P/'REPLAY_RESULT.json',U/'PROTECTED_FILES.json',Path(__file__)]})
    with (U/'CHANGE_AUDIT.json').open('x',encoding='utf-8') as f:
        json.dump(clean(report),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(clean({k:report[k] for k in ['conditions','counts_vs_baseline','counts_vs_v1',
        'previous_nine_all_preserved','named_recovery_fixtures_all_pass','newly_recorded_slot_count','raw_replacement_count',
        'protected362_files_unchanged','total_pinned_files_unchanged']}),ensure_ascii=False))


if __name__=='__main__':run()
