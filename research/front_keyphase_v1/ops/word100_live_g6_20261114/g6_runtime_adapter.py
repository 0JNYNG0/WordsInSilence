"""Latest qualified public-verified evidence, existing TCN/DTW/capture unchanged."""
from pathlib import Path
import sys,subprocess,threading,queue,time,json,atexit,copy,importlib.util
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
from runtime_adapter import FeedbackConsumer as PreviousConsumer, CaptureService as PreviousCapture, bind_dependency
from capture_pacing import paced_loop
_paced_camera_loop=paced_loop(PreviousCapture.camera_loop)

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
project=load('g6_display_contract',O/'word100_visual_feedback_g5_20261114/adapter.py').project
_base_project=project
sys.path.insert(0,str(O/'word100_finger_repair_20261115'))
from projection_extension import project_fingers
def project(row,guides,frames):return project_fingers(_base_project,row,guides,frames)
GUIDES=json.loads((O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json').read_text('utf-8-sig'))['words']
CARDS=json.loads((O/'word100_action_feedback_demo_20261108/CARDS.json').read_text('utf-8-sig'))['words']

class Worker:
 def __init__(self):
  self.lock=threading.Lock();self.messages=queue.Queue();self.log=(U/'worker_stderr.log').open('a',encoding='utf-8')
  self.process=subprocess.Popen([sys.executable,'-X','utf8','-B',str(U/'worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW)
  threading.Thread(target=self.reader,daemon=True).start();atexit.register(self.close)
  ready=self.receive(90)
  if not ready.get('ready'):raise RuntimeError('최신 판정부 준비에 실패했어요.')
  self.prepare_seconds=ready['seconds']
 def reader(self):
  for line in self.process.stdout:
   try:self.messages.put(json.loads(line))
   except json.JSONDecodeError:self.messages.put(dict(ok=False,error='Invalid worker response'))
  self.messages.put(dict(ok=False,error='Worker stopped'))
 def receive(self,timeout):
  try:return self.messages.get(timeout=timeout)
  except queue.Empty:
   self.close();raise RuntimeError('동작 분석 시간이 너무 길어졌어요. 결과를 확정하지 않았어요.')
 def infer(self,raw,word):
  with self.lock:
   if self.process.poll() is not None:raise RuntimeError('분석기를 다시 준비해야 해요. 서버를 다시 시작해 주세요.')
   payload=dict(word=word,raw={k:np.asarray(raw[k]).tolist() for k in ['xy_px','score','timestamp_us','frame_id','source_width','source_height']})
   started=time.perf_counter();self.process.stdin.write(json.dumps(payload,allow_nan=True)+'\n');self.process.stdin.flush();reply=self.receive(90)
   if not reply.get('ok'):raise RuntimeError('최신 판정을 계산하지 못했어요. '+reply.get('error',''))
   reply['result']['runtime']['worker_roundtrip_seconds']=time.perf_counter()-started
   return reply['result']
 def close(self):
  if self.process.poll() is None:
   try:self.process.stdin.write('{"close":true}\n');self.process.stdin.flush();self.process.wait(timeout=2)
   except (OSError,ValueError,subprocess.TimeoutExpired):self.process.terminate()
  if not self.log.closed:self.log.close()

def presentation(row,word,ts):
 projected=project(row,[s for s in GUIDES[word] if s['required']],len(ts));cards=[];cues=[];notes=[]
 for p in projected:
  guide=next(c for c in CARDS[word] if c['id']==p['id']);spans=p['spans'];lo=min((s[0] for s in spans),default=None);hi=max((s[1] for s in spans),default=None)
  differences=[]
  for cue in p['cues']:
   differences.extend(cue['spans']);a=min(s[0] for s in cue['spans']);b=max(s[1] for s in cue['spans'])
   cues.append(dict(kind=cue['kind'],card_ids=cue.get('card_ids',[p['id']]),start_seconds=float(ts[a])/1e6,end_seconds=float(ts[b])/1e6,observation=cue['message'],action=cue['action'],hands=cue['hands'],fingers=cue.get('fingers',[]),spans=cue['spans'],evidence=cue['evidence']))
  cards.append(dict(id=p['id'],title=guide['title'],instruction=p['instruction'],hands=p['hands'],reference=guide['reference'],display_status=p['status'],status=p['raw_state'],message=p['message'],query_spans=spans,query_indices=[lo,hi] if lo is not None else None,query_seconds=[float(ts[lo])/1e6,float(ts[hi])/1e6] if lo is not None else None,difference_spans=differences,whole_action_complete=None,educational_wrong=False))
  notes.append(dict(card_id=p['id'],status=p['status'],message=p['message']))
 # Dedupe the same order cue appearing on both involved cards.
 unique=[];seen=set()
 for c in cues:
  key=(c['kind'],tuple(c['card_ids']),str(c['spans']),c['action'])
  if key not in seen:seen.add(key);unique.append(c)
 summary=unique[0]['observation']+' '+unique[0]['action'] if unique else '확인한 특징은 설명과 비슷해요. 회색 구간은 아직 판단하지 못했어요.' if any(p['status']=='similar' for p in projected) else '이 촬영에서는 교정할 차이를 확정하지 못했어요. 회색 구간은 오답이 아니에요.'
 return cards,dict(summary=summary,notes=notes,cues=unique,whole_sign_correct=None),projected

class FeedbackConsumer(PreviousConsumer):
 def __init__(self):
  super().__init__();self.latest=Worker()
 def raw(self,raw,word):
  started=time.perf_counter();out=super().raw(raw,word);previous=time.perf_counter()-started
  # Identical quarantined coordinates drive TCN, latest evidence and skeletons.
  observed=dict(raw);xy=np.asarray(out['query_xy'],np.float32);xy[xy<0]=np.nan;observed['xy_px']=xy
  qualified=self.latest.infer(observed,word);cards,coaching,projected=presentation(qualified,word,out['timestamp_us'])
  out.update(cards=cards,coaching=coaching,qualified_evidence=qualified,feedback_version='G6-finger-20261115',feedback_projection=projected,whole_sign_correct=None,overall_pass=False,correction_validated=False)
  out['timing'].update(previous_tcn_dtw_seconds=previous,latest_evidence_seconds=qualified['runtime']['seconds'],worker_roundtrip_seconds=qualified['runtime']['worker_roundtrip_seconds'])
  out['compute_seconds']=time.perf_counter()-started
  return out

class CaptureService(PreviousCapture):
 prepare=bind_dependency(PreviousCapture.prepare,'FeedbackConsumer',FeedbackConsumer)
 def camera_loop(self):
  _paced_camera_loop(self)
  if self.state=='done' and self.result is not None:
   self.result['_ready_clock']=time.perf_counter()
   self.result['_stop_clock']=self.result['_ready_clock']-self.result['stop_to_result_seconds']
   self.result['capture_policy']='RTMW acquisition backpressure, max2 pending; all received frames retained'
   # The loop's last preview snapshot precedes queue draining; expose final counts.
   self.diagnostics.update(recording_written=self.result['recorded_frames'],rtmw_completed=self.result['observed_frames'],recording_queue=0,rtmw_queue=0)
 def status(self):
  result=super().status();result['feedback_version']='G6-finger-20261115';result['monotonic_ms']=time.perf_counter()*1000
  if self.state=='done' and self.result is not None:
   result['stop_to_result_seconds']=self.result.get('stop_to_result_seconds')
  return result

def attach(result):
 # Legacy examples retain their original labeled provenance, never masquerade as G6.
 if result.get('feedback_version')not in ['G6-frozen-G4','G6-finger-20261115']:
  from runtime_adapter import attach as legacy_attach
  return legacy_attach(result)
 return copy.deepcopy(result)
