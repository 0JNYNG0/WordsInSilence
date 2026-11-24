"""Synthetic wrapper/socket compatibility. No model, DP, socket or video calls."""
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
from tempfile import TemporaryDirectory
from copy import deepcopy
import argparse, hashlib, importlib.util, io, json, sys, threading, time
import numpy as np

U=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
 spec.loader.exec_module(module);return module

def run(output):
 out=U/'review'/output
 if out.exists():raise FileExistsError(out)
 pins={str(U/n):sha(U/n) for n in ['server.py','live_engine.py','evidence_guard.py','engine.py','adapter.py','app.js']}
 pins[str(Path(__file__))]=sha(__file__)
 server=load('audited_synthetic_server',U/'server.py')
 checks=[];failures=[];observations=[]
 def check(name,good,detail=None):
  checks.append(name)
  if not good:failures.append(dict(name=name,detail=detail))
 with TemporaryDirectory(prefix='wrapper_test_',dir=U/'review') as folder:
  root=Path(folder).resolve()
  if not root.is_relative_to((U/'review').resolve()):raise ValueError('Wrong synthetic output scope')
  n=20; arrays=dict(x=np.zeros((n,69)),mask=np.ones((n,69),bool),rgb=np.zeros((n,768)),
    probabilities=np.tile(np.arange(7,dtype=float)/10,(n,1)),frame_id=np.arange(n),timestamp_us=np.arange(n)*33333)
  names=['feature_'+str(i) for i in range(69)]; events=[];guard_calls=[]
  base_result=dict(word='WORD0001',references=[dict(id='R1'),dict(id='R2')],selected_whole_references=['R1','R2'],
    stages=[dict(id='1',status='SIMILAR_REFERENCE_RANGE')],latency=dict(total_adapter_seconds=.25,total_analysis_seconds=.5),training_used_submission=False)
  engine=ModuleType('engine');guard=ModuleType('evidence_guard')
  engine.catalog=lambda:dict(words=[],model='protected_baseline')
  engine.read=read
  engine.save=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,allow_nan=False),encoding='utf8')
  bank=dict(observer='protected_baseline',rows=[dict(clip_id='R1',word_id='WORD0001'),dict(clip_id='R2',word_id='WORD0001'),dict(clip_id='R3',word_id='WORD0002')])
  engine.bank=lambda:(bank,{'R1':{},'R2':{},'R3':{}})
  def base_analyze(path,jobdir,progress):
   events.append('base_called')
   progress(dict(stage='predicting_attributes'))
   progress('old complete',100,'old detail')
   engine.save(Path(jobdir)/'FEEDBACK.json',base_result)
   return deepcopy(base_result)
  engine.analyze_submission=base_analyze
  def fake_guard(a,rows,reference_arrays,base):
   guard_calls.append(dict(ids=[r['clip_id'] for r in rows],keys=list(a)))
   check('wrapper passes saved posterior exactly',np.array_equal(a['probabilities'],arrays['probabilities']))
   check('wrapper passes all six source arrays',all(np.array_equal(a[k],v) for k,v in arrays.items()))
   check('wrapper passes declared feature names',a['feature_names']==names)
   check('wrapper filters same-target bank only',[r['clip_id'] for r in rows]==['R1','R2'])
   result=deepcopy(base);result['stages'][0]['status']='INSUFFICIENT_EVIDENCE'
   result['guard_audit']=dict(status='SYNTHETIC_GUARD',new_inference=0,aggregate_changes=[dict(changed=True)])
   return result
  guard.guard_result=fake_guard
  with patch.dict(sys.modules,{'engine':engine,'evidence_guard':guard}):
   live=load('audited_live_wrapper',U/'live_engine.py')
   job=root/'wrapper';(job/'adapter').mkdir(parents=True)
   np.savez(job/'adapter/analysis.npz',**arrays)
   engine.save(job/'adapter/COMPLETE.json',dict(feature_names=names))
   progress=[];result=live.analyze_submission('synthetic-not-opened.video',job,lambda *a,**k:progress.append((a,k)))
   check('single base analysis and single guard invocation',len(events)==len(guard_calls)==1)
   check('guard returns dict rather than tuple',isinstance(result,dict) and result['evidence_guard']['applied'])
   check('original saved base remains distinct',read(job/'FEEDBACK.json')==base_result)
   check('final guarded output and full audit persisted',read(job/'FINAL_FEEDBACK.json')==result and read(job/'EVIDENCE_GUARD.json')['status']=='SYNTHETIC_GUARD')
   check('response keeps change summary without duplicate full audit','guard_audit' not in result and result['evidence_guard']['aggregate_changes']==[dict(changed=True)])
   check('final schema serializes without NaN',isinstance(json.dumps(result,allow_nan=False),str))
   check('adapter dict progress retained',progress[0][0]==(dict(stage='predicting_attributes'),))
   check('completion follows guard and only once',sum(len(a)>1 and a[1]==100 for a,k in progress)==1 and progress[-1][0][1]==100)
   check('guard processing progress appears',any(len(a)>1 and a[1]==95 for a,k in progress))
   check('latency retains adapter and separates guard',result['latency']['total_adapter_seconds']==.25 and result['latency']['evidence_guard_seconds']>=0 and result['latency']['total_analysis_seconds']>=0)
   check('catalog exposes additive guard metadata',live.catalog()['evidence_guard']=='arm_alignment_union_support_v1')
   unsupported=dict(stages=[dict(status='INSUFFICIENT_EVIDENCE')],references=[],latency={})
   engine.analyze_submission=lambda *a,**k:deepcopy(unsupported)
   check('unsupported input unchanged without adapter NPZ access',live.analyze_submission('not-opened',root/'absent')==unsupported)
   check('unsupported never calls guard',len(guard_calls)==1)
   engine.analyze_submission=base_analyze;bank['observer']='expanded_nine_people'
   other=root/'wrong_observer';(other/'adapter').mkdir(parents=True)
   np.savez(other/'adapter/analysis.npz',**arrays);engine.save(other/'adapter/COMPLETE.json',dict(feature_names=names))
   try:live.analyze_submission('not-opened',other)
   except ValueError as exc:check('different observer rejected explicitly','protected baseline' in str(exc))
   else:check('different observer rejected explicitly',False)
   check('different observer cannot reach guard',len(guard_calls)==1)

  # Invoke Handler methods without constructing a TCP server/real connection.
  for exc in [ConnectionAbortedError,ConnectionResetError,BrokenPipeError]:
   h=object.__new__(server.Handler)
   with patch.object(server.BaseHTTPRequestHandler,'handle',side_effect=exc('synthetic socket close')):
    check('request loop accepts '+exc.__name__,h.handle() is None)
  for exc in [ValueError,RuntimeError,OSError]:
   h=object.__new__(server.Handler)
   with patch.object(server.BaseHTTPRequestHandler,'handle',side_effect=exc('not a named socket disconnect')):
    try:h.handle()
    except exc:check('request loop preserves '+exc.__name__,True)
    else:check('request loop preserves '+exc.__name__,False)
  with patch.object(server.BaseHTTPRequestHandler,'handle',return_value='normal'):
   check('request loop retains normal return',object.__new__(server.Handler).handle()=='normal')

  def post_fixture(label,ack_error=None,submit_error=False,body=b'1234'):
   directory=root/label;directory.mkdir();submits=[];rejects=[]
   s=SimpleNamespace(ready=True,lock=threading.RLock(),words={'WORD0001':{}},jobs={},session_dir=directory,
      origins=lambda:{'http://127.0.0.1:8802'},process=lambda job:None)
   def submit(fn,key):
    if submit_error:raise RuntimeError('synthetic enqueue failure')
    submits.append(key);s.jobs[key]['status']='RUNNING'
   s.executor=SimpleNamespace(submit=submit)
   h=object.__new__(server.Handler);h.server=s;h.base_guard=h.token_guard=lambda:True
   h.path='/api/analyze?word=WORD0001';h.headers={'Origin':'http://127.0.0.1:8802','Content-Length':'4','Content-Type':'video/mp4'}
   h.connection=SimpleNamespace(settimeout=lambda t:None);h.rfile=io.BytesIO(body)
   h.reject=lambda *a:rejects.append(a)
   def ack(*args):
    if ack_error:raise ack_error('synthetic response disconnected')
   h.json=ack
   try:h.do_POST()
   except (ConnectionAbortedError,ConnectionResetError,BrokenPipeError):pass # enclosing handle absorbs only transport close
   job=list(s.jobs.values())[0]
   return job,submits,rejects
  for exc in [ConnectionAbortedError,ConnectionResetError,BrokenPipeError]:
   job,submits,rejects=post_fixture('ack_'+exc.__name__,ack_error=exc)
   observations.append(dict(case='accepted_upload_response_'+exc.__name__,status=job['status'],submits=len(submits),rejects=rejects))
   check('accepted job survives '+exc.__name__,job['status']=='RUNNING',job['status'])
   check('accepted job enqueued exactly once '+exc.__name__,len(submits)==1)
   check('accepted disconnect cannot advise re-upload '+exc.__name__,not rejects,rejects)
   check('accepted upload retains exact byte hash '+exc.__name__,job['submission_sha256']==hashlib.sha256(b'1234').hexdigest())
  job,submits,rejects=post_fixture('upload_truncated',body=b'12')
  check('truncated upload remains failed with no analysis',job['status']=='FAILED' and not submits and rejects[0][0]==400)
  job,submits,rejects=post_fixture('submit_failure',submit_error=True)
  check('enqueue failure remains failed and recorded',job['status']=='FAILED' and not submits and job['error']['type']=='RuntimeError' and rejects[0][0]==400)
  job,submits,rejects=post_fixture('normal_post')
  check('normal upload starts one job',job['status']=='RUNNING' and len(submits)==1 and not rejects)

  # The analysis boundary must retain genuine pipeline failures even for a
  # similarly named exception; transport suppression is NOT a job-failure catch.
  for error in [ValueError,ConnectionAbortedError]:
   directory=root/('process_'+error.__name__);directory.mkdir()
   s=object.__new__(server.DemoServer);s.ready=True;s.lock=threading.RLock();s.started_job_count=0
   s.startup=dict(total_startup_seconds=3,warmup_seconds=2);s.session_dir=directory
   def fail(*a,**k):raise error('synthetic pipeline failure')
   s.engine=SimpleNamespace(analyze_submission=fail)
   jobdir=directory/'job';jobdir.mkdir()
   s.jobs={'job':dict(_path=jobdir/'not-opened.video',_dir=jobdir,status='QUEUED')}
   s.process('job')
   check('pipeline failure persists '+error.__name__,s.jobs['job']['status']=='FAILED' and (jobdir/'SERVER_FIRST_FAILURE.txt').exists())
   check('pipeline failed timing persists '+error.__name__,read(jobdir/'REQUEST_TIMING.json')['status']=='FAILED')
 for p,h in pins.items():check('source unchanged during synthetic checks '+Path(p).name,sha(p)==h)
 value=dict(status='PASS_SYNTHETIC_COMPATIBILITY' if not failures else 'FAIL_COUNTEREXAMPLE_PRESERVED',checks=len(checks),failure_count=len(failures),failures=failures,
  results=checks,observations=observations,source_pins=pins,actual_server_starts=0,actual_sockets=0,model_forwards=0,DTW_calls=0,real_video_reads=0,camera_calls=0,
  scope='Wrapper uses a fake engine/guard and synthetic array fixtures; HTTP methods use fake streams/executors. This checks API and exception boundaries, not real upload or guard arithmetic.',
  limitations=['Expanded observer is rejected; final adopted protected observer only.','Candidate guard efficacy/adoption is verified separately.','Old live process and completed raw uploads are untouched.'])
 with out.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps(dict(status=value['status'],checks=len(checks),failures=len(failures),output=str(out),sha256=sha(out))))
 return not failures

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 if Path(a.output).name!=a.output:raise ValueError('Output must be a basename')
 if not run(a.output):raise SystemExit(1)
