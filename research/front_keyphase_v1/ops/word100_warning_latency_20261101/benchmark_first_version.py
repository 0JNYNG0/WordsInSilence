"""Root-only fixed matched latency/equality experiment, public inputs only."""
from pathlib import Path
import json,hashlib,time,importlib.util,sys,traceback
import numpy as np
U=Path(__file__).resolve().parent;B=U.parent/'word100_multipeople_live_feedback_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def compare(p,q):
 out={}
 with np.load(p,allow_pickle=False) as a,np.load(q,allow_pickle=False) as b:
  assert a.files==b.files
  for k in a.files:
   x,y=a[k],b[k];same=x.shape==y.shape
   eq=same and np.array_equal(x,y,equal_nan=True)
   d=None
   if same and x.dtype.kind in 'fiu':
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.any():d=float(np.max(np.abs(x[ok].astype(float)-y[ok].astype(float))))
   out[k]=dict(shape=list(x.shape),equal=eq,max_abs=d,finite_mask_equal=same and np.array_equal(np.isfinite(x),np.isfinite(y)))
 return out
def main():
 f=read(U/'FROZEN.json')
 for p,h in f['source_pins'].items():assert sha(p)==h,p
 save(U/'BENCHMARK_STARTED.json',dict(time=time.time(),candidate_sha256=sha(U/'candidate_adapter.py'),frozen_sha256=sha(U/'FROZEN.json')))
 base=module('latency_serial',B/'adapter.py');cand=module('latency_candidate',U/'candidate_adapter.py')
 # Scope redirection only: frozen serial code and all resources remain original.
 base.UNIT=U
 # Shared resident frozen model removes model-loading/device-temperature confounds.
 cand._compile_signrep=base._compile_signrep;cand._head_runtime=base._head_runtime
 t=time.perf_counter();warm=base.warmup();save(U/'WARMUP.json',dict(seconds=time.perf_counter()-t,result=warm))
 runs=[];eq=[]
 for ci,row in enumerate(f['public_clips']):
  source=Path(row['source']);assert source.is_file()
  paths=[]
  for i,condition in enumerate(f['order']):
   rid=f'clip{ci+1}_{i+1}_{condition}';dest=U/'bench'/rid;paths.append(dest/'adapter')
   print('START',rid,flush=True);t=time.perf_counter()
   m=base if condition=='serial' else cand
   a=m.analyze_video(source,dest)
   r=dict(id=rid,clip=row['clip_id'],condition=condition,wall_seconds=time.perf_counter()-t,metadata=a['metadata'],output=str(dest/'adapter'))
   runs.append(r);save(U/'bench'/f'{rid}.json',r);print('DONE',rid,round(r['wall_seconds'],3),flush=True)
  for i in [1,2,3]:
   for fn in ['analysis.npz','rgb_windows.npz','observations.npz','face.npz','raw.npz']:
    eq.append(dict(clip=row['clip_id'],comparison=f'1_vs_{i+1}',file=fn,arrays=compare(paths[0]/fn,paths[i]/fn)))
 save(U/'BENCHMARK.json',dict(runs=runs,equality=eq,source_pins=f['source_pins'],new_training=False,old_results_rerun=False,experiment='New fixed matched optimization experiment'))
 print('BENCHMARK complete',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'BENCHMARK_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
