"""Fake decoder/detector/compiled model tests. No real video/model/GPU calls."""
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
import ast,hashlib,importlib.util,json,sys,threading,time
import numpy as np

U=Path(__file__).resolve().parent
def load(path):
 s=importlib.util.spec_from_file_location('_latency_candidate_synthetic',path)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
 m=load(U/'candidate_adapter.py');checks=[]
 def check(name,value):
  if not value:raise AssertionError(name)
  checks.append(name)
 def rejects(name,fn,kind=Exception):
  try:fn()
  except kind:check(name,True)
  else:check(name,False)
 def no_worker():return not any(t.name=='candidate-signrep-single-consumer' for t in threading.enumerate())
 old=ast.parse((m.BASELINE_UNIT/'adapter.py').read_text(encoding='utf8'))
 new=ast.parse((U/'candidate_adapter.py').read_text(encoding='utf8'))
 functions=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 oldf,newf=functions(old),functions(new)
 for name in ['normalize','input_support','window_centers','roi_arrays','_compile_signrep_uncached','_compile_signrep','_load_head_models','_head_runtime','_frozen_heads','_heads','warmup']:
  check('unchanged frozen calculation '+name,oldf[name]==newf[name])
 check('baseline source preserved',sha(m.BASELINE_UNIT/'adapter.py')==m.BASELINE_ADAPTER_SHA256)
 check('candidate output isolation',m.UNIT==U and m.BASELINE_UNIT!=U and m.SEMANTIC.parent==m.OPS)

 decoded=[];converted=[];closed=[];frames=[]
 class Frame:
  def __init__(self,pts,number):self.pts=pts;self.number=number
  def to_ndarray(self,format):
   assert format=='rgb24';converted.append(self.number)
   return np.full((4,5,3),self.number,np.uint8)
 class Packet:
  def decode(self):
   for item in frames:decoded.append(item.number);yield item
 class Container:
  def __init__(self):
   stream=SimpleNamespace(time_base=Fraction(1,30),average_rate=Fraction(30),codec_context=SimpleNamespace(width=5,height=4,name='fake',pix_fmt='rgb24'))
   self.streams=SimpleNamespace(video=[stream])
  def __enter__(self):return self
  def __exit__(self,*args):closed.append(True)
  def demux(self,stream):return [Packet()]
 fakeav=ModuleType('av');fakeav.open=lambda p:Container()
 core=ModuleType('wis_lab.core.observation')
 def empty(n):return dict(frame_id=np.zeros(n,np.int64),timestamp_us=np.zeros(n,np.int64),marker=np.zeros(n,int))
 class Observation:
  def __init__(self,**kwargs):self.__dict__.update(kwargs);self.sidecar={}
  def quality_summary(self):return dict(synthetic=True,frames=len(self.frame_id))
 core.empty_frames=empty;core.Observation=Observation;core.sha256_file=lambda p:'synthetic-source-hash'
 class Detector:
  contract={'hand_assignment':{'policy':'same-frozen-policy'}}
  handedness_disagreements=0;handedness_comparisons=3;unassigned_hands=0
  def __init__(self):self.calls=[]
  def process(self,rgb,ts,out,row):self.calls.append((row,ts));out['marker'][row]=rgb[0,0,0]
  def extractor_id(self):return 'synthetic-extractor'
 with patch.dict(sys.modules,{'av':fakeav,'wis_lab.core.observation':core}):
  frames[:]=[Frame(p,i) for i,p in enumerate([None,100,100,99,101,104])]
  pre=m._bounded_video_preflight('never-opened.video')
  check('preflight distinguishes all decoded from timestamped count',pre['decoded_frames']==6 and pre['probe_metadata']['frame_count']==5)
  check('preflight preserves first/max PTS and metadata',pre['first_source_pts']==100 and pre['maximum_elapsed_us']==133333 and pre['probe_metadata']['duration_ms']==133.333 and pre['probe_metadata']['time_base']=='1/30')
  check('preflight does not convert pixels or call detector',not converted)
  decoded.clear();d=Detector();ob=m._extract_with_metadata('never-opened.video',d,'test',pre['probe_metadata'])
  check('metadata extraction consumes exactly one decoder traversal',len(decoded)==6)
  check('local extraction preserves original drop/frame-ID semantics',ob.frame_id.tolist()==[0,3,4] and ob.timestamp_us.tolist()==[0,33333,133333] and ob.marker.tolist()==[1,4,5])
  check('nonmonotonic frames never enter detector',d.calls==[(0,0),(1,33333),(2,133333)])
  check('sidecar preserves original sources and counts',ob.sidecar['decoded_frames']==3 and ob.sidecar['dropped_non_monotonic_pts']==2 and ob.sidecar['width']==5 and ob.sidecar['height']==4)
  frames[:]=[Frame(0,0),Frame(600,1)];check('exact20seconds accepted',m._bounded_video_preflight('fake')['maximum_elapsed_us']==20_000_000)
  frames[:]=[Frame(0,0),Frame(601,1)];rejects('beyond20seconds rejected',lambda:m._bounded_video_preflight('fake'),ValueError)
  frames[:]=[Frame(i,i) for i in range(1201)];rejects('all decoded frame cap enforced before any detector',lambda:m._bounded_video_preflight('fake'),ValueError)
  frames[:]=[Frame(None,i) for i in range(1201)];rejects('noPTS frames also consume decoded cap',lambda:m._bounded_video_preflight('fake'),ValueError)
  frames[:]=[Frame(None,0)];rejects('no actual PTS rejected',lambda:m._bounded_video_preflight('fake'),ValueError)
  frames[:]=[];rejects('empty video rejected',lambda:m._bounded_video_preflight('fake'),ValueError)
  rejects('unbounded metadata rejected',lambda:m._extract_with_metadata('fake',Detector(),'t',dict(frame_count=1201)),ValueError)

 shape=(1,3,16,224,224);thread_ids=[];active=[0,0];seen=[]
 def compiled(args):
  active[0]+=1;active[1]=max(active);thread_ids.append(threading.get_ident())
  try:
   x=args[0];value=float(x.flat[0]);seen.append(value);time.sleep(.002)
   return [np.full((1,768),value,np.float32)]
  finally:active[0]-=1
 reused=np.zeros(shape,np.float32)
 def items():
  for i in range(10):
   reused.fill(i);yield i*2,reused
 profile={};values,starts=m._window_pipeline(items(),compiled,total_windows=10,profile=profile)
 check('window ordering and immutable queued snapshot',starts.tolist()==list(range(0,20,2)) and np.array_equal(values[:,0],np.arange(10)) and seen==list(range(10)))
 check('single sequential non-main compiled owner',len(set(thread_ids))==1 and thread_ids[0]!=threading.get_ident() and active[1]==1)
 check('queue capacity4 and nonnegative timings',0<profile['maximum_queue_depth']<=4 and all(v>=0 for v in profile.values()))
 check('normal completion joins worker',no_worker() and active[0]==0)
 rejects('empty windows do not become predictions',lambda:m._window_pipeline([],compiled),ValueError)
 check('empty-window worker joined',no_worker())
 rejects('invalid iterator leaves no thread',lambda:m._window_pipeline(None,compiled),TypeError)
 check('invalid-iterator worker absent',no_worker())
 rejects('wrong input shape rejected',lambda:m._window_pipeline([(0,np.zeros((1,2),np.float32))],compiled),ValueError)
 check('bad tensor worker joined',no_worker())
 def broken(args):raise RuntimeError('synthetic device failure')
 rejects('consumer failure propagates while producer can be blocked',lambda:m._window_pipeline(((i,np.zeros(shape,np.float32)) for i in range(20)),broken),RuntimeError)
 check('failed consumer is joined',no_worker())
 def source_failure():
  yield 0,np.zeros(shape,np.float32)
  raise ValueError('synthetic decode failure')
 rejects('producer failure propagates and drains ownership',lambda:m._window_pipeline(source_failure(),compiled),ValueError)
 check('producer failure joins any in-flight call',no_worker() and active[0]==0)
 def nonfinite(args):return [np.full((1,768),np.nan,np.float32)]
 rejects('nonfinite inference cannot yield partial features',lambda:m._window_pipeline([(0,np.zeros(shape,np.float32))],nonfinite),ValueError)
 check('nonfinite failure joined',no_worker())
 cancel=threading.Event();cancel.set();rejects('pre-cancelled run refuses all output',lambda:m._window_pipeline([(0,np.zeros(shape,np.float32))],compiled,cancel_event=cancel),m.EncodingCancelled)
 check('pre-cancelled worker joined',no_worker())
 cancel=threading.Event()
 def cancellable(args):
  cancel.set();time.sleep(.01);return [np.zeros((1,768),np.float32)]
 rejects('in-flight cancellation returns no partial predictions',lambda:m._window_pipeline(((i,np.zeros(shape,np.float32)) for i in range(12)),cancellable,cancel_event=cancel),m.EncodingCancelled)
 check('cancel waits until in-flight worker exits',no_worker())
 rejects('expected window count enforced',lambda:m._window_pipeline([(0,np.zeros(shape,np.float32))],compiled,total_windows=2),ValueError)
 check('count mismatch joined',no_worker())

 # End-to-end encoder on fake pixels/decoder with a scalar fake model. This
 # checks exact original preprocessing/window maths without any real runtime.
 def norm(rgb):
  mean=np.array([.485,.456,.406],np.float32)*255
  denom=1/(np.array([.229,.224,.225],np.float32)*255)
  return ((rgb.astype(np.float32)-mean)*denom).transpose(0,3,1,2).copy()
 def starts_for(n):
  seq=list(range(0,n-15,2))
  if seq[-1]!=n-16:seq.append(n-16)
  return seq
 common=SimpleNamespace(norm=norm,starts=starts_for)
 fakecv=ModuleType('cv2');fakecv.INTER_LINEAR=1
 fakecv.resize=lambda im,size,interpolation:np.repeat(np.repeat(im[:1,:1],224,0),224,1)
 compiled_inputs=[]
 def capture_compiled(args):
  compiled_inputs.append(args[0].copy());return [np.full((1,768),float(args[0].mean()),np.float32)]
 with patch.dict(sys.modules,{'av':fakeav,'cv2':fakecv}),patch.object(m,'_module',return_value=common):
  frames[:]=[Frame(100+i,i) for i in range(19)];ids=np.arange(19);ts=np.round(np.arange(19)/30*1e6).astype(np.int64)
  rgb,features,starts,centers=m._encode('not-opened',(0,0,5,4),ids,ts,capture_compiled,None)
  normalized=[norm(np.full((1,224,224,3),i,np.uint8))[0] for i in range(19)]
  check('encoder keeps stride2 plus final whole window',starts.tolist()==[0,2,3] and len(compiled_inputs)==3)
  for j,s in enumerate(starts):check('exact preprocessing/window tensor '+str(s),np.array_equal(compiled_inputs[j],np.ascontiguousarray(np.stack(normalized[s:s+16]).transpose(1,0,2,3)[None])))
  check('encoder keeps source midpoint PTS',np.array_equal(centers,(ts[starts]+ts[starts+15])/2))
  expected=np.stack([np.interp(ts,centers,features[:,i]) for i in range(768)],1).astype(np.float32)
  check('encoder original768 interpolation exact',np.array_equal(rgb,expected))
  wrong=ts.copy();wrong[5]+=1
  rejects('RGB/source PTS mismatch rejects instead of silently joining',lambda:m._encode('fake',(0,0,5,4),ids,wrong,capture_compiled,None),ValueError)
  check('PTS error closes decoder and joins worker',no_worker())
 check('source file still unchanged',sha(m.BASELINE_UNIT/'adapter.py')==m.BASELINE_ADAPTER_SHA256)
 return dict(status='PASS_SYNTHETIC_ONLY_PUBLIC_NOT_RUN',checks=len(checks),results=checks,
  candidate_sha256=sha(U/'candidate_adapter.py'),selftest_sha256=sha(__file__),baseline_sha256=m.BASELINE_ADAPTER_SHA256,
  actual_video_decodes=0,actual_model_inferences=0,GPU_calls=0,server_starts=0,camera_calls=0,
  limitations=['Fake compiled output does not demonstrate actual GPU speed or numeric parity.','Cooperative cancellation joins a synchronous in-flight call; no forced device-call cancellation.','Actual public tensor/feature/69/mask/posterior parity and matched timing remain root-owned and not run here.'])

if __name__=='__main__':
 output=U/'CANDIDATE_ADAPTER_SELFTEST.json'
 if output.exists():raise FileExistsError(output)
 value=run()
 with output.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
 print(json.dumps({'status':value['status'],'checks':value['checks'],'candidate_sha256':value['candidate_sha256']}))
