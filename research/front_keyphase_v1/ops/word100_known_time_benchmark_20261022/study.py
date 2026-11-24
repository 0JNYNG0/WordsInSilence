import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import sys,json,shutil,traceback,importlib.util
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];P=O/'word100_motion_alignment_evidence_20261021'
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
prev=module('known_time_previous',P/'study.py');C=module('known_time_candidate',U/'candidate.py')
read,load,write,sha=prev.read,prev.load,prev.write,prev.sha
def now():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()

NORMAL=('identity','fast075','slow15','pace','pause')
KINDS=NORMAL+tuple(f'skip{s}_{p}' for s in ('3','20pct') for p in (25,50,75))
FIELDS=('centre','radius','palm_valid','angle_low','angle_high','angle_valid','observed','source_rest','missing','unconfirmed','timestamp_us','frame_id','core')

def mapping(n,core,kind):
    deleted=np.array([],int);active=np.flatnonzero(core)
    assert len(active)>=5 and np.all(np.diff(active)==1)
    if kind=='identity':ix=np.arange(n,dtype=float)
    elif kind in ('fast075','slow15'):
        factor=.75 if kind=='fast075' else 1.5;ix=np.linspace(0,n-1,round((n-1)*factor)+1)
    elif kind=='pace':ix=np.interp(np.linspace(0,1,n),[0,.3,.7,1],np.array([0,.2,.8,1])*(n-1))
    elif kind=='pause':
        mid=active[len(active)//2];ix=np.r_[np.arange(mid+1),np.full(15,mid),np.arange(mid+1,n)].astype(float)
    else:
        size,portion=kind[4:].split('_');length=3 if size=='3' else max(3,round(.2*len(active)))
        centre=int(active[round((len(active)-1)*int(portion)/100)])
        start=int(np.clip(centre-length//2,active[0],active[-1]-length+1));deleted=np.arange(start,start+length)
        ix=np.setdiff1d(np.arange(n),deleted).astype(float)
    assert ix[0]==0 and ix[-1]==n-1 and np.all(np.diff(ix)>=0)
    return ix,deleted

def resample(q,e,ix):
    a=np.floor(ix).astype(int);b=np.ceil(ix).astype(int);w=ix-a
    def transform(v):
        if v.dtype.kind=='b':return v[a]&v[b]
        if v.dtype.kind in 'iu':return v[a].copy()
        t=w.reshape((-1,)+(1,)*(v.ndim-1));out=v[a]*(1-t)+v[b]*t
        return np.where(np.isfinite(v[a])&np.isfinite(v[b]),out,np.nan)
    zz={k:transform(e[k]) for k in FIELDS};qq={k:transform(q[k]) for k in ('features','present','core')}
    # A runtime observer does not get original indices or omitted-interval labels.
    zz['frame_id']=np.arange(len(ix));zz['timestamp_us']=np.rint(np.arange(len(ix))*1e6/30).astype(np.int64)
    zz['unconfirmed']=~zz['angle_valid'];zz['missing']=e['missing'][a]|e['missing'][b]
    return qq,zz

def freeze():
    f=read(P/'FROZEN.json');prev.prev.prev.pipe.check(f['pins']);prev.prev.prev.pipe.check(f['protected'])
    assert read(P/'SYNC_CHECK.json')['status']=='PASS' and not list(P.glob('*RUNNING.json'))
    pins=dict(f['pins'])
    for p in [U/'candidate.py',U/'study.py',P/'FINAL_AUDIT.json',P/'VERIFY.json',P/'ANALYSIS.json']+list((P/'evidence').glob('*.npz')):pins[str(p)]=sha(p)
    (U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
    specs=[]
    for r in f['rows']:
        z=load(P/'evidence'/(r['clip_id']+'.npz'))
        for kind in KINDS:
            ix,deleted=mapping(len(z['core']),z['core'],kind)
            specs.append(dict(clip=r['clip_id'],kind=kind,source_index=ix.tolist(),deleted=deleted.tolist()))
    write(U/'FROZEN.json',dict(at=now(),authorization='2026-10-22 user: same public video known time transforms; normal speed vs skipped movement; preserve C/T3-F/user exclusion; no learning.',
      rows=f['rows'],roles=f['roles'],pins=pins,protected=f['protected'],cases=specs,
      conditions=list(KINDS),scope='Same27 public TRAIN clips3words9people; self-source benchmark, not unseen-person accuracy. 135 normal cases incl27identity and162 core deletion cases. No newRGB extraction.',
      transforms='Features and existing rechecked geometry uniformly resampled with finite endpoints, booleanAND. Normal duration.75/1.5, nonuniform pace,15frame pause. Delete3 or20%coreframes at25/50/75%core; deterministic before scores, all cases retained.',
      forbidden='No user/mentor, DEV/heldout, training, classifier, new detector, camera, external, downloads, Git, runtime or model edits.',
      scientific_limits=['Deleted core window not automatically linguistically essential. Semantic correctness is unknown.', 'Resampled coordinates/features approximate timing variation, not new video or MediaPipe processing.', 'Same-source is an intentionally easy identifiable benchmark; correlated noise and identical body are limitations.', 'Linear omitted movement may be observationally indistinguishable from faster sampling. Missing source intervals cannot be used as prediction inputs.'],
      diagnostic='Frozen current baseline DTW weights and band.2 unchanged. Compare previous plateau warning with new reference-angle local support envelope: matched query valid samples plus immediate valid neighbor, never across invalid sample. Convex interval allows normal interpolation; no threshold fitted. Persistent mismatch3referenceframes for sameangle is diagnostic only. Unknown reference/query evidence separate; expected reference core/required/restfixed.',
      evaluation='Source indices/deleted labels accessible ONLY evaluator: median DTW reference time error vs knownsource; report core/knowngeometricchange separately. Deletion overlap and nondeleted collateral warnings; normal false alarms. No classifier score, optimal threshold or semantic success claim.',
      verification='Independent DP path optimal accumulated cost, mappings/masks, reference denominator, local envelopes/runlength, known timing metrics and count summaries; protected pins; graph and RGB sample review.'))
    print('FROZEN',len(specs),flush=True)

def metrics(e,path,ix,deleted,result,old):
    pred=np.array([np.median(path[path[:,0]==i,1]) for i in range(len(ix))]);err=abs(pred-ix)
    a=np.floor(ix).astype(int);b=np.ceil(ix).astype(int);core=e['core'][a]&e['core'][b]
    # Evaluation-only original motion flag, never detector input.
    change=(e['measured_change'][a].any(1)|e['measured_change'][b].any(1))&core
    ps=result['persistent'].any((1,2));raw=result['mismatch'].any((1,2));mask=np.zeros(len(e['core']),bool);mask[deleted]=True
    def timing(s):
        x=err[s];return dict(n=int(s.sum()),mae_frames=float(x.mean()) if len(x) else None,p95_frames=float(np.percentile(x,95)) if len(x) else None,within2_frames=float((x<=2).mean()) if len(x) else None)
    scored=result['expected']&~result['unknown'];dvals=result['distance'][scored]
    return dict(core_timing=timing(core),change_timing=timing(change),previous_warning=bool(old['angle_warning'].any()),local_warning=bool(ps.any()),
      expected_angle_cells=int(result['expected'].sum()),unknown_angle_cells=int(result['unknown'].sum()),mismatch_angle_cells=int(result['mismatch'].sum()),persistent_angle_cells=int(result['persistent'].sum()),
      max_gap_deg=float(np.nanmax(dvals)) if len(dvals) else None,mean_gap_deg=float(np.nanmean(dvals)) if len(dvals) else None,
      deleted_frame_count=len(deleted),deleted_frames_with_warning=int((ps&mask).sum()),deleted_frames_with_any_mismatch=int((raw&mask).sum()),
      preserved_frames_with_warning=int((ps&~mask).sum()),hit_deleted=bool((ps&mask).any()),
      deleted_confirmed_angle_cells=int((result['expected']&e['angle_valid']&mask[:,None,None]).sum()),
      normal_or_missing_label='temporal_deletion' if len(deleted) else 'timing_only',wrong_sign=None)

def run():
    f=read(U/'FROZEN.json');prev.prev.prev.pipe.check(f['pins']);prev.prev.prev.pipe.check(f['protected'])
    write(U/'RUNNING.json',dict(at=now(),pid=os.getpid()));(U/'cases').mkdir();allrows=[]
    for row in f['rows']:
        cid=row['clip_id'];q=prev.prev.sample(row);e=load(P/'evidence'/(cid+'.npz'));req=f['roles'].get(row['word_code'],{}).get('required_sides',['L','R'])
        weights=prev.prev.C.weights_for(req,(False,False),prev.prev.F.GROUPS)
        for spec in [s for s in f['cases'] if s['clip']==cid]:
            kind=spec['kind'];ix=np.asarray(spec['source_index']);deleted=np.asarray(spec['deleted'],int);qq,z=resample(q,e,ix)
            cost,ev=prev.prev.legacy(qq,q,group_weights=weights,hand_participation=False,one_sided_mode='flat');path=np.asarray(prev.prev.dp(cost,.2),int)
            assert len(path);out=C.diagnose(z,e,path,req)
            old=prev.C.inspect(z,e,path,out['expected'],req)
            m=metrics(e,path,ix,deleted,out,old);m.update(clip=cid,label=row['label'],kind=kind,path_length=len(path),dtw_accumulated=float(cost[tuple(path.T)].sum()),dtw_mean=float(cost[tuple(path.T)].mean()),evidence_mean=float(ev[tuple(path.T)].mean()))
            name=kind+'__'+cid
            np.savez_compressed(U/'cases'/(name+'.npz'),**out,path=path,source_index=ix,deleted=deleted,cost=cost,
               previous_angle_warning=old['angle_warning'],query_angle_valid=z['angle_valid'],query_angle_low=z['angle_low'],query_angle_high=z['angle_high'],query_core=z['core'])
            write(U/'cases'/(name+'.json'),m);allrows.append(m)
        print('DONE',cid,flush=True)
    write(U/'RESULT.json',dict(at=now(),rows=allrows,no_training=True))
    prev.prev.prev.pipe.check(f['pins']);prev.prev.prev.pipe.check(f['protected']);(U/'RUNNING.json').unlink()

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        p=U/'FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
