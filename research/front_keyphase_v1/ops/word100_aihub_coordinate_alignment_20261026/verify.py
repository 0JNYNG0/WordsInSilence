"""Verify saved diagnostics from frozen sources without calling study.compute."""
import json, hashlib
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent
P=U.parent/'word100_aihub_coordinate_quality_20261026'
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def eq(a,b): np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-8,equal_nan=True)
def stat(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(n=int(x.size),median=float(np.median(x)) if x.size else None,p90=float(np.percentile(x,90)) if x.size else None)
def checkstat(actual,saved):
    assert actual['n']==saved['n']
    for k in ['median','p90']:
        if actual[k] is None: assert saved[k] is None
        else: eq(actual[k],saved[k])
def run():
    f=load(U/'FROZEN.json');done=load(U/'DONE.json')
    assert len(f['rows'])==len(done['rows'])==100
    pins={**f['pins'],**f['protected']};derived={};both=0;flips=0;pair=[];bmax=0.;temporalmax=0.;missing=[];allsteps=[];originalsteps=[];availability=[0,0]
    for src,saved in zip(f['rows'],done['rows']):
        assert src['clip_id']==saved['clip_id']
        pins[src['array']]=src['array_sha'];pins[src['mp']]=src['mp_sha'];pins[saved['array']]=saved['array_sha']
        z=np.load(src['array']);m=np.load(src['mp']);q=np.load(saved['array'])
        xyz=z['official_pose'];xy=m['xy_px'];v=z['official_valid'];n=src['frames']
        eq(q['A'],z['official'][...,:2]);assert np.array_equal(q['source_valid'],v)
        assert np.array_equal(q['frame_id'],z['frame_id']) and np.array_equal(q['timestamp_us'],z['timestamp_us'])
        # Calibration only uses shoulders/elbows in first 15 frames, not hand targets.
        shifts=[]
        for t in range(min(n,15)):
            for a,b in zip([5,2,6,3],[11,12,13,14]):
                if np.isfinite(xyz[t,a]).all() and xyz[t,a,2]>0 and m['observed'][t,b] and m['conf'][t,b]>=.5 and np.isfinite(xy[t,b]).all():shifts.append(xy[t,b]-xyz[t,a,:2])
        shift=np.median(shifts,axis=0);assert len(shifts)==saved['fit_points'] and len(shifts)>=12
        eq(shift,saved['global_translation']);eq(q['B'],q['A']+shift)
        cv=np.zeros((n,2),bool);c=np.full_like(q['C'],np.nan)
        for h,(aw,ae,mw,me) in enumerate([(7,6,15,13),(4,3,16,14)]):
            for t in range(n):
                good=all(np.isfinite(xyz[t,j]).all() and xyz[t,j,2]>0 for j in [aw,ae]) and all(m['observed'][t,j] and m['conf'][t,j]>=.5 and np.isfinite(xy[t,j]).all() for j in [mw,me])
                cv[t,h]=good
                if good:c[t,h]=q['A'][t,h]+xy[t,mw]-xyz[t,aw,:2]
        eq(c,q['C']);assert np.array_equal(cv,q['c_available'])
        assert not np.any(np.isfinite(q['B'])&~np.isfinite(q['A']))
        assert not np.any(np.isfinite(q['C'])&~np.isfinite(q['A']))
        availability[0]+=int((cv&v[:,:,0]).sum());availability[1]+=int(v[:,:,0].sum())
        for k in ['B','C']:
            mask=v&v[:,:,0,None]
            if k=='C': mask&=cv[:,:,None]
            lhs=q[k]-q[k][:,:,0,None,:];rhs=q['A']-q['A'][:,:,0,None,:]
            eq(lhs[mask],rhs[mask])
        ok=v[:,:,0].all(1)&cv.all(1)
        pa=q['A'][:,0,0]-q['A'][:,1,0];pb=q['B'][:,0,0]-q['B'][:,1,0];pc=q['C'][:,0,0]-q['C'][:,1,0]
        eq(pa[ok],pb[ok]);bmax=max(bmax,float(np.max(np.abs(pa[ok]-pb[ok]))))
        both+=int(ok.sum());flips+=int(((pa[:,0]*pc[:,0]<0)&ok).sum());pair.extend(np.linalg.norm(pc[ok]-pa[ok],axis=1).tolist())
        vs=v[1:]&v[:-1];stepa=np.diff(q['A'],axis=0);stepb=np.diff(q['B'],axis=0);stepc=np.diff(q['C'],axis=0)
        eq(stepa[vs],stepb[vs]);temporalmax=max(temporalmax,float(np.max(np.abs(stepa[vs]-stepb[vs]))))
        vcs=vs&cv[1:,:,None]&cv[:-1,:,None]
        allsteps.extend(np.linalg.norm(stepc[vcs],axis=-1).tolist());originalsteps.extend(np.linalg.norm(stepa[vcs],axis=-1).tolist())
        active=np.zeros(n,bool);t=z['timestamp_us']/1e6
        for lo,hi in src['intervals']:active|=(t>=lo)&(t<=hi)
        for scope,mask in [('post_calibration',np.arange(n)>=15),('annotated_action',active),('all',np.ones(n,bool))]:
            common=v&z['mp_valid']&mask[:,None,None];paired=common&cv[:,:,None]
            for k in 'ABC':
                dist=np.sqrt(((q[k]-z['mp_xy'])**2).sum(-1));used=paired if k=='C' else common
                vals={'all_points':dist[used],'wrist':dist[:,:,0][used[:,:,0]],'paired_wrist':dist[:,:,0][paired[:,:,0]],'paired_fingertips':dist[:,:,[4,8,12,16,20]][paired[:,:,[4,8,12,16,20]]]}
                for key,arr in vals.items():checkstat(stat(arr),saved['metrics'][scope][k][key])
        if saved['metrics']['annotated_action']['A']['paired_wrist']['n']==0:missing.append(src['clip_id'])
    for p,h in pins.items(): assert sha(p)==h,p
    previous=load(P/'DIAGNOSTICS.json')['rows'];high={r['clip_id'] for r in previous if r['median_wrist_disagreement']>40};assert len(high)==10
    for scope in ['post_calibration','annotated_action']:
        derived[scope]={}
        for group in ['previous_large10','other90']:
            rows=[r for r in done['rows'] if (r['clip_id'] in high)==(group=='previous_large10')]
            available=[r for r in rows if r['metrics'][scope]['A']['paired_wrist']['median'] is not None]
            metrics={k:np.array([r['metrics'][scope][k]['paired_wrist']['median'] for r in available]) for k in 'ABC'}
            derived[scope][group]=dict(total=len(rows),evaluable=len(available),not_evaluable=len(rows)-len(available),clip_median_distribution={k:stat(x) for k,x in metrics.items()},B_closer=int((metrics['B']<metrics['A']).sum()),B_farther=int((metrics['B']>metrics['A']).sum()),C_closer=int((metrics['C']<metrics['A']).sum()),C_farther=int((metrics['C']>metrics['A']).sum()))
    out=dict(status='PASS_SAVED_ARRAY_AND_ARITHMETIC_VERIFICATION',videos=100,recording_ids=[2],protected_files=len(f['protected']),pins=pins,summary=derived,missing_action_comparison=missing,global_between_hand_max_delta=bmax,global_temporal_max_delta=temporalmax,per_hand_translation=dict(existing_hand_frames=availability,horizontal_order_changed_frames=flips,both_available_frames=both,relative_wrist_vector_change=stat(pair),pooled_point_step_before=stat(originalsteps),pooled_point_step_after=stat(allsteps),warning='Horizontal order changes are not verified identity swaps. Original labels may be wrong; no side slot reassignment was performed.'),independent_human_groundtruth=False,verification_role='Separate source reconstruction and arithmetic pass by same Codex; no fresh inference or source archive re-audit.',training=0,operating_change=False)
    with (U/'VERIFIED.json').open('x',encoding='utf-8') as w:json.dump(out,w,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:out[k] for k in ['status','videos','summary','per_hand_translation']},ensure_ascii=False))
if __name__=='__main__':run()
