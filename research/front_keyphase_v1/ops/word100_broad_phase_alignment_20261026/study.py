import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
from pathlib import Path
import sys,json,hashlib,datetime,traceback,importlib.util,time
import numpy as np
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;O=U.parent;BP=O/'word100_broad_phase_preflight_20261026'
sys.path.insert(0,str(O.parent/'src'));sys.path.insert(0,str(BP))
import prepare as prep
import time_transform as T
from wis_lab.core import features as F,dtw as D
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
P=module('broad_fractional_phase',O/'word100_fractional_phase_alignment_20261026/candidate.py')
A=module('broad_local_angle',O/'word100_angle_evidence_20261021/candidate.py')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def write(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def check(pins):
 for p,h in pins.items():assert sha(p)==h,p
NORMAL=['fast075','slow15','pace','pause']
KINDS=['identity']+NORMAL+[f'skip{s}_{p}' for s in ('3','20pct') for p in (25,50,75)]
MISSING=['wholeL','wholeR','gapL','gapR']
GROUPS={'angle':(0,15),'palm':(15,17),'elbow':(17,19),'pose_wrist':(19,21),'hand_wrist':(21,23)}

def freeze():
 r=read(BP/'PREPARED.json');v=read(BP/'VERIFIED_PREPARE.json');s=read(BP/'SCOPE.json');assert v['counts']['clips']==300
 check(v['pins']);check(r['input_pins']);check(r['output_pins']);check(s['protected'])
 pins={str(p):sha(p) for p in [Path(__file__),Path(F.__file__),Path(D.__file__),Path(P.__file__),Path(A.__file__),BP/'prepare.py',BP/'time_transform.py',BP/'PREPARED.json',BP/'VERIFIED_PREPARE.json']}
 write(U/'FROZEN.json',dict(at=now(),rows=r['rows'],input_pins=r['output_pins'],pins=pins,protected=s['protected'],kinds=KINDS,missing=MISSING,
  scope='100words public developed DEV300 IDs11/12/13,3300known-time self comparisons+1200missing-input realignments. Not fresh test people, no other199/200, no semantic evaluation or peer calibration.',
  method='OriginalC fullframe→ROI observations, transformed coordinates with observed-bracket AND, original per-frame shoulder midpoint resampled and original scalar scale frozen; recompute all features/motion. Fixed production DTW weights, flat1.5 missing,band.2; no oracle input. Median path correspondence vs fixed binomial bounded1frame fractional correspondence. Source interval union never shrinks. Entire5component per-channel descriptor fallback to original when refined unavailable; retain original instantaneous result separately.',
  domain='Fixed reference core x2hands x23channels audit grid; not a claim that both hands are semantically mandatory. No resting-hand correctness verdict without positive rest schedule. Native missing explicitly unknown. Reference domain never determined by query visibility. Shape quality is conservative existing corroboration only, not independent truth.',
  metrics='Known temporal absolute error in original reference frames, normal per-group point and sequence residual separately, existing-observation/quality support. Deleted-window residuals and supported-before/mid/after distinguish missing evidence from numerical discrepancy. No threshold trained here; not binary sign-error detection. Original and candidate residuals retained, never minimum-score fusion.',
  acceptance='Require unchanged domain/input/point evidence; old complete sequence support preserved; lower mean normal timing and per-group sequence discrepancy on identical support, without reducing any deleted-window positive discrepancy to numerical zero (1e-6 same previous arithmetic guard). Report perclip regressions regardless average. This is diagnostic gate only; cannot authorize operating feedback or classifier replacement.',
  limitations='Synthetic time/delete/missing on stored coordinates, not new RGB occlusion test. No anatomical or semantic labels. Motion computed correctly but default motion group weights0, so not claimed cause of previous path errors. ThreeIDs do not support a five-person normal range.',
  no_training=True,no_new_detector=True,operating_change=False))
 print('FROZEN300_4500',flush=True)

def channels(obs,origin,scale,quality):
 n=obs.n_frames;x=obs.xy_px[:,33:].reshape(n,2,21,2).astype(float);ok=obs.usable[:,33:].reshape(n,2,21)
 d=A.angle_evidence(x,ok,np.zeros((n,2),bool),np.zeros((n,2,15),bool),np.zeros((n,2,15),bool))
 values=np.full((n,2,23),np.nan);valid=np.zeros(values.shape,bool);values[:,:,:15]=d['angle_deg'];valid[:,:,:15]=d['computable']
 shoulders=obs.usable[:,11]&obs.usable[:,12]
 for h in (0,1):
  for a,pts in ((15,[0,5,9,13,17]),(21,[0])):
   values[:,h,a:a+2]=(np.mean(x[:,h,pts],axis=1)-origin)/scale
   valid[:,h,a:a+2]=(ok[:,h,pts].all(-1)&shoulders)[:,None]
  for a,j in ((17,13+h),(19,15+h)):
   values[:,h,a:a+2]=(obs.xy_px[:,j]-origin)/scale;valid[:,h,a:a+2]=(obs.usable[:,j]&shoulders)[:,None]
 valid &= np.isfinite(values);return dict(values=np.where(valid,values,np.nan),valid=valid,quality=valid&quality)

def transformed(z,kind):
 ix=T.mapping(len(z['core']),z['core'],'identity' if kind in MISSING else kind);obs=T.observation(z,ix);a=np.floor(ix).astype(int);b=np.ceil(ix).astype(int)
 if kind=='identity' or kind in MISSING:
  obs.timestamp_us=z['timestamp_us'].copy();obs.frame_id=z['frame_id'].copy()
 quality=z['quality_eligible'][a]&z['quality_eligible'][b]
 if kind in MISSING:
  side=0 if kind.endswith('L') else 1;ids=np.arange(len(ix))
  if kind.startswith('gap'):
   active=np.flatnonzero(z['core']);mid=active[len(active)//2];ids=np.arange(mid-1,mid+2)
  sl=slice(33+21*side,54+21*side)
  obs.xy_px[ids,sl]=np.nan;obs.observed[ids,sl]=False;obs.usable[ids,sl]=False;quality[ids,side]=False
 origin=T.body_origin(z,ix).astype(z['origin'].dtype);features,present=prep.own_features(obs,origin,float(z['scale']));ch=channels(obs,origin,float(z['scale']),quality)
 return dict(features=features,present=present),ch,ix

def mapped(c,index,first,last):
 a=np.floor(index).astype(int);b=np.ceil(index).astype(int);w=(index-a)[:,None,None]
 valid=c['valid'][a]&c['valid'][b];quality=c['quality'][a]&c['quality'][b]
 values=np.where(valid,c['values'][a]*(1-w)+c['values'][b]*w,np.nan)
 return dict(values=values,valid=valid,quality=quality,index=index,first=first,last=last,source_valid=c['valid'],source_quality=c['quality'])

def descriptor(m,domain):
 n=len(domain);valid=np.zeros(domain.shape,bool);quality=valid.copy();values=np.full(domain.shape+(5,),np.nan)
 for j in range(2,n-2):
  ex=domain[j-2:j+3].all(0);start=int(m['first'][j-2]);end=int(m['last'][j+2]);idx=[j-2,j,j+2]
  good=ex&m['source_valid'][start:end+1].all(0)&m['valid'][idx].all(0)
  quality[j]=good&m['source_quality'][start:end+1].all(0)&m['quality'][idx].all(0)
  p=np.moveaxis(m['values'][idx],0,-1);values[j]=np.concatenate([p,np.diff(p,axis=-1)],-1);valid[j]=good
 return np.where(valid[...,None],values,np.nan),valid,quality

def evaluate(q,r,ch,ref,ix,core):
 cost,evidence=D.frame_cost_matrix(q,r,one_sided_mode='flat',hand_participation=False);path=np.asarray(D._optimal_path(cost,.2),int)
 n=len(core);assert len(path) and (path[0]==[0,0]).all() and (path[-1]==[len(ix)-1,n-1]).all()
 ids=[path[path[:,1]==j,0] for j in range(n)];base=np.array([np.median(x) for x in ids]);first=np.array([x.min() for x in ids]);last=np.array([x.max() for x in ids]);new=P.phase(base)
 domain=np.broadcast_to(core[:,None,None],(n,2,23)).copy()
 old=mapped(ch,base,first,last);newm=mapped(ch,new,np.minimum(first,np.floor(new).astype(int)),np.maximum(last,np.ceil(new).astype(int)))
 anchor=mapped(ref,np.arange(n,dtype=float),np.arange(n),np.arange(n));rd,rv,rq=descriptor(anchor,domain);bd,bv,bq=descriptor(old,domain);nd,nv,nq=descriptor(newm,domain)
 source=np.where(nv,1,np.where(bv,2,0)).astype(np.uint8);candidate=np.where((source==1)[...,None],nd,bd);cv=source>0;cq=np.where(source==1,nq,bq)&cv
 raw_point_ok=domain&old['valid']&ref['valid'];raw_point=np.where(raw_point_ok,abs(old['values']-ref['values']),np.nan)
 old_ok=bv&rv;new_ok=cv&rv;old_delta=np.where(old_ok[...,None],abs(bd-rd),np.nan);new_delta=np.where(new_ok[...,None],abs(candidate-rd),np.nan)
 old_time=np.interp(base,np.arange(len(ix)),ix);new_time=np.interp(new,np.arange(len(ix)),ix)
 channel_time=np.where(source==1,new_time[:,None,None],np.where(source==2,old_time[:,None,None],np.nan))
 return dict(path=path,phase_old=base,phase_new=new,span_first=first,span_last=last,phase_source=source,domain=domain,point_ok=raw_point_ok,point_delta=raw_point.astype(np.float32),
  sequence_old_ok=old_ok,sequence_new_ok=new_ok,sequence_old_delta=old_delta.astype(np.float32),sequence_new_delta=new_delta.astype(np.float32),sequence_quality=bq&rq,sequence_new_quality=cq&rq,
  old_error=np.abs(old_time-np.arange(n)),new_error=np.abs(new_time-np.arange(n)),effective_error=np.abs(channel_time-np.arange(n)[:,None,None]),source_index=ix,
  mean_path_evidence=np.array(evidence[tuple(path.T)].mean()),missing_original=domain&~ref['valid'],core=core)

def stats(out):
 core=out['core'];common=out['sequence_old_ok']&out['sequence_new_ok'];d={}
 for g,(a,b) in GROUPS.items():
  mask=common[:,:,a:b];old=out['sequence_old_delta'][:,:,a:b];new=out['sequence_new_delta'][:,:,a:b]
  d[g]=dict(common_cells=int(mask.sum()),old_mean=float(old[mask].mean()) if mask.any() else None,new_mean=float(new[mask].mean()) if mask.any() else None)
 return dict(domain=int(out['domain'].sum()),old_supported=int(out['sequence_old_ok'].sum()),new_supported=int(out['sequence_new_ok'].sum()),lost_support=int((out['sequence_old_ok']&~out['sequence_new_ok']).sum()),fallback=int((out['phase_source']==2).sum()),old_time=float(out['old_error'][core].mean()),new_time=float(out['new_error'][core].mean()),groups=d)

def run():
 f=read(U/'FROZEN.json');check(f['pins']);check(f['input_pins']);check(f['protected']);write(U/'RUNNING.json',dict(at=now(),pid=os.getpid()));(U/'cases').mkdir();rows=[];pins={};start=time.perf_counter()
 for nr,row in enumerate(f['rows']):
  z=load(row['output']);r,ref,identity=transformed(z,'identity');assert np.array_equal(r['features'],z['features'],equal_nan=True) and np.array_equal(r['present'],z['present'])
  np.testing.assert_allclose(ref['values'],z['values'],atol=2e-5,rtol=1e-6,equal_nan=True);assert np.array_equal(ref['valid'],z['available'])
  for kind in KINDS+MISSING:
   q,ch,ix=(r,ref,identity) if kind=='identity' else transformed(z,kind);out=evaluate(q,r,ch,ref,ix,z['core']);name=row['clip_id']+'__'+kind;p=U/'cases'/(name+'.npz');np.savez_compressed(p,**out);pins[str(p)]=sha(p)
   rows.append(dict(name=name,clip_id=row['clip_id'],word_code=row['word_code'],signer_id=row['signer_id'],kind=kind,**stats(out)))
  print('CLIP',nr+1,300,'CASES',len(rows),'SECONDS',round(time.perf_counter()-start,1),flush=True)
  (U/'PROGRESS.json').write_text(json.dumps(dict(at=now(),clips=nr+1,total_clips=300,cases=len(rows),seconds=time.perf_counter()-start)),encoding='utf-8')
 assert len(rows)==4500;check(f['pins']);check(f['input_pins']);check(f['protected']);write(U/'RESULT.json',dict(at=now(),status='COMPUTED_PENDING_INDEPENDENT_ARITHMETIC',rows=rows,pins=pins,seconds=time.perf_counter()-start,training=0,detector=0,production_changed=False));(U/'RUNNING.json').unlink();print('COMPLETE4500',flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  p=U/'FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
