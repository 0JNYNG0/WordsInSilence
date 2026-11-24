from pathlib import Path
import sys,json,gzip,hashlib,importlib.util
sys.dont_write_bytecode=True
import numpy as np
U=Path(__file__).resolve().parent; OPS=U.parent; ROOT=U.parents[3]
REL=OPS/'word100_relational_sequence_compare_20261031'
STAGE=OPS/'word100_stage_feedback_cycle_20261031'
PREV=OPS/'word100_alignment_context_review_20261101'
TARGET=OPS/'word100_target_known_feedback_20261101'
PHASE=OPS/'word100_core_stage_evidence_20261031/PHASE_DATA.json'
COMPONENTS={'L_wrist':('L_location',[0,1]),'R_wrist':('R_location',[0,1]),
 'hands_delta':('hands_relation',[0,1]),'L_to_R_forearm':('L_to_R_forearm',[0,1]),'R_to_L_forearm':('R_to_L_forearm',[0,1])}
def read(p):
 p=Path(p)
 return json.loads(gzip.decompress(p.read_bytes()).decode('utf8')) if p.suffix=='.gz' else json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.gz':
  with p.open('xb') as f:f.write(gzip.compress(json.dumps(d,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf8'),mtime=0))
 else:
  with p.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def update(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
legacy=module(REL/'relational.py','fixed_relational_geometry_and_control_baseline')
def array(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
def components(a):
 g=legacy.features(a)
 return {name:{'values':g[key][0][:,ix].copy(),'mask':g[key][1][:,ix].copy(),
  'frame_id':a['frame_id'].copy(),'timestamp_us':a['timestamp_us'].copy()} for name,(key,ix) in COMPONENTS.items()}
def subset(a,lo,hi):return {k:v[lo:hi+1].copy() for k,v in a.items()}
def crop(path,core):
 p=np.asarray(path,int);a,b=map(int,core);ix=np.flatnonzero((p[:,1]>=a)&(p[:,1]<=b));assert len(ix)
 assert np.array_equal(ix,np.arange(ix[0],ix[-1]+1))
 old=p[ix];qlo,qhi=int(old[:,0].min()),int(old[:,0].max())
 assert set(old[:,0])==set(range(qlo,qhi+1)) and set(old[:,1])==set(range(a,b+1))
 return old-np.array([qlo,a]),[qlo,qhi],[a,b],old
def checkpins():
 d=read(U/'FROZEN.json')
 for k in ['pins','protected']:
  for p,h in d[k].items():assert sha(p)==h,p
 return d
