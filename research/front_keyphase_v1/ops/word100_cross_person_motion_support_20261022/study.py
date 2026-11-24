import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import sys, json, shutil, traceback, importlib.util
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
K=O/'word100_known_time_benchmark_20261022'
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
prior=module('cross_person_known_time',K/'study.py');C=module('cross_person_candidate',U/'candidate.py')
read,load,write,sha,now=prior.read,prior.load,prior.write,prior.sha,prior.now
A=prior.prev.prev
check=A.prev.pipe.check
KINDS=prior.NORMAL

def data(row):
    q=A.sample(row);e=load(prior.P/'evidence'/(row['clip_id']+'.npz'))
    # Preserve existing arm feature/group validity. Split elbow/wrist reporting,
    # without pretending an unavailable old arm group was newly observed.
    e['arm_xy']=q['features'][:,:8].reshape(-1,2,2,2).astype(float)
    e['arm_valid']=np.broadcast_to(q['present'][:,:2,None],e['arm_xy'].shape[:-1]).copy()&np.isfinite(e['arm_xy']).all(-1)
    return q,e

def transform(q,e,kind):
    ix,_=prior.mapping(len(e['core']),e['core'],kind);qq,z=prior.resample(q,e,ix)
    lo=np.floor(ix).astype(int);hi=np.ceil(ix).astype(int);w=(ix-lo)[:,None,None,None]
    z['arm_xy']=e['arm_xy'][lo]*(1-w)+e['arm_xy'][hi]*w
    z['arm_valid']=e['arm_valid'][lo]&e['arm_valid'][hi]&np.isfinite(z['arm_xy']).all(-1)
    return qq,z,ix

def freeze():
    assert not (U/'FROZEN.json').exists()
    f=read(K/'FROZEN.json');check(f['pins']);check(f['protected'])
    assert read(K/'SYNC_CHECK.json')['status']=='PASS' and not list(K.glob('*RUNNING*'))
    pins=dict(f['pins'])
    for p in [U/'candidate.py',U/'study.py',K/'FINAL_AUDIT.json',K/'VERIFY.json',K/'FROZEN.json']:
        pins[str(p)]=sha(p)
    (U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
    write(U/'FROZEN.json',dict(at=now(),rows=f['rows'],roles=f['roles'],pins=pins,protected=f['protected'],
      authorization='2026-10-22 user: other public people normal speed warnings; handposition and arm trajectories; preserve C/T3-F, no user data or training before review.',
      scope='27 public TRAIN clips,3words9people,each to8differentpeople. 216pairs x5timings=1080. No heldout or user footage. Reuse known27 evidence, not broad100word validation.',
      kinds=KINDS,alignment='Unchanged existing baseline features,weights,band.2; no new path weights. Same both-required/onehand roles; fixed reference source core/rest. Exact correspondence across people unknown.',
      angle='Reuse frozen known-time local envelope and run3. Add strict all-matched-observed fraction to expose partially observed plateaux. Absence of warning never whole-action pass.',
      geometry='Separate palm/elbow/wrist body-normalized time-series distances. Palm two-estimate discs plus1px from existing evidence are not inter-person tolerance. Arm raw distances from original observed arm groups, no fabricated confidence. Nearest actual supported point at matched/adjacent query, not virtual hull location; no new binary error threshold.',
      evaluation='Compare old plateau/new angle warnings for originals and temporal variants; paired change versus same query identity with each other reference. Record warning+coverage together. Different-person no reference-time truth, so not false-positive rate/recognition accuracy.',
      controls='For each27 query,lowest-other-signer reference,each required hand: wholehand missing,3coreframe hand gap,wholearm missing=135 fixed-path counterfactuals. Expected unchanged; unknown where no current evidence; arms may never fill missing handshape. These test evidence layer, not missing-input realignment or RGB detector.',
      pass_criteria='All cases retained; no input/identity/model change; independent arithmetic, coverage/missing preservation; normal lowering excluded only by frozen reference evidence; no semantic success verdict. Warning improvement may fail without changing thresholds.',
      forbidden=['training','classifier inference','user/mentor','DEV/heldout','runtime/web/model edit','camera','downloads','external','Git']))
    print('FROZEN1080+135',flush=True)

def evaluate(q,r,path,req):
    out=prior.C.diagnose(q,r,path,req)
    out['angle_strict'],out['angle_fraction']=C.angle_strict(q,r,path,out['expected'])
    out.update(C.geometry(q,r,path,req))
    old=prior.prev.C.inspect(q,r,path,out['expected'],req)
    out['old_warning']=old['angle_warning']
    return out

def stats(o):
    def mean(x):return float(np.nanmean(x)) if np.isfinite(x).any() else None
    return dict(old_warning=bool(o['old_warning'].any()),new_warning=bool(o['persistent'].any()),
      expected=int(o['expected'].sum()),unknown=int(o['unknown'].sum()),strict=int(o['angle_strict'].sum()),
      mismatch=int(o['mismatch'].sum()),persistent=int(o['persistent'].sum()),gap_deg=mean(o['distance']),
      geometry=[dict(expected=int(o['geom_expected'][:,:,k].sum()),unknown=int(o['geom_unknown'][:,:,k].sum()),
                     strict=int(o['geom_strict'][:,:,k].sum()),distance_sw=mean(o['geom_distance'][:,:,k]),gap_sw=mean(o['geom_gap'][:,:,k])) for k in range(3)],
      whole_action_correct=None)

def run():
    f=read(U/'FROZEN.json');check(f['pins']);check(f['protected']);assert not (U/'RUNNING.json').exists()
    write(U/'RUNNING.json',dict(at=now(),pid=os.getpid()));(U/'cases').mkdir();(U/'inputs').mkdir();(U/'controls').mkdir()
    samples={r['clip_id']:data(r) for r in f['rows']};allrows=[];controls=[]
    for cid,(q,e) in samples.items():
        np.savez_compressed(U/'inputs'/(cid+'.npz'),**e)
    for qr in f['rows']:
        cid=qr['clip_id'];q,e=samples[cid];req=f['roles'].get(qr['word_code'],{}).get('required_sides',['L','R'])
        refs=[r for r in f['rows'] if r['word_code']==qr['word_code'] and r['signer_id']!=qr['signer_id']]
        weights=A.C.weights_for(req,(False,False),A.F.GROUPS)
        for rr in refs:
            rid=rr['clip_id'];r,re=samples[rid]
            for kind in KINDS:
                qq,qe,ix=transform(q,e,kind)
                cost,ev=A.legacy(qq,r,group_weights=weights,hand_participation=False,one_sided_mode='flat')
                path=np.asarray(A.dp(cost,.2),int);out=evaluate(qe,re,path,req)
                name=kind+'__'+cid+'__'+rid
                np.savez_compressed(U/'cases'/(name+'.npz'),**out,path=path,source_index=ix,cost=cost)
                row=dict(name=name,query=cid,reference=rid,label=qr['label'],kind=kind,**stats(out),alignment_evidence=float(ev[tuple(path.T)].mean()))
                allrows.append(row)
                if kind=='identity' and rr==refs[0]:
                    for h,side in enumerate(('L','R')):
                        if side not in req:continue
                        for mode in ('wholehand','shortgap','wholearm'):
                            zz={k:v.copy() for k,v in qe.items()};ids=np.arange(len(qe['core']))
                            if mode=='shortgap':
                                act=np.flatnonzero(qe['core']);mid=act[len(act)//2];ids=np.arange(mid-1,mid+2)
                            if mode=='wholearm':zz['arm_valid'][ids,h]=False;zz['arm_xy'][ids,h]=np.nan
                            else:
                                zz['angle_valid'][ids,h]=False;zz['angle_low'][ids,h]=np.nan;zz['angle_high'][ids,h]=np.nan
                                zz['palm_valid'][ids,h]=False;zz['centre'][ids,h]=np.nan;zz['radius'][ids,h]=np.nan
                            co=evaluate(zz,re,path,req);cn=mode+'_'+side+'__'+cid
                            np.savez_compressed(U/'controls'/(cn+'.npz'),**co,path=path,removed=ids)
                            controls.append(dict(name=cn,parent=name,mode=mode,side=side,**stats(co)))
        write(U/'PROGRESS.json',dict(at=now(),query=cid,completed=len(allrows),total=1080))
        print('DONE',cid,len(allrows),flush=True)
    assert len(allrows)==1080 and len(controls)==135
    write(U/'RESULT.json',dict(at=now(),rows=allrows,controls=controls,training=0,user_data=0))
    check(f['pins']);check(f['protected']);(U/'RUNNING.json').unlink()

if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException:
        p=U/'FIRST_ERROR.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
