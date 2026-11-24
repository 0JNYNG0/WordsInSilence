from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sys,json,time,traceback
from contextlib import redirect_stdout

import torch,hashlib
_loaded=[];_load=torch.load
def _trace(path,*args,**kwargs):
 p=Path(path).resolve()
 forbidden=['public_view_','public_temporal_visibility','public_cycle_observability','public_motion_decoupling','projection_head','inspection_relation']
 if any(x in str(p) for x in forbidden):raise RuntimeError('Non-frontal feedback weight rejected: '+str(p))
 _loaded.append(str(p));return _load(path,*args,**kwargs)
torch.load=_trace

with redirect_stdout(sys.stderr):
 from short_arrival import ShortArrivalLesson as ThanksCandidate
 from frontal_candidate import *
 from frontal_presentation import present
 from display_alignment import align_observed
 sys.path.insert(0,str(O/'word100_learned_guide_feedback_20261102'))
 from alignment import dtw
 torch.set_num_threads(2);torch.set_num_interop_threads(1);started=time.perf_counter();net=ThanksCandidate()
 ref=read(V/'web/REFERENCE.json');ref_features=np.load(V/'web/reference_features.npz')['features']
 pattern=read(V/'web/DISPLAY_PATTERN.json')
Path(__file__).with_name('LOADED_WEIGHTS.json').write_text(json.dumps([dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())for p in dict.fromkeys(_loaded)],indent=2),encoding='utf-8')
print(json.dumps(dict(ready=True,seconds=time.perf_counter()-started)),flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line)
  if req.get('close'):break
  if req['word']!='WORD1290':raise ValueError('Unsupported lesson')
  raw={k:np.asarray(v)for k,v in req['raw'].items()}
  with redirect_stdout(sys.stderr):
   q=net.analyze(raw);cards,coaching=present(q,ref['action_span']);t=time.perf_counter();path,display=align_observed(net.last_features,ref_features,q['events'],pattern['events'],dtw);align_seconds=time.perf_counter()-t
  result=dict(lesson_acceptance=q,cards=cards,coaching=coaching,path=path,display_alignment=display,runtime=dict(seconds=q['seconds'],alignment_seconds=align_seconds))
  print(json.dumps(dict(ok=True,result=result),ensure_ascii=False,allow_nan=False),flush=True)
 except Exception as exc:
  traceback.print_exc(file=sys.stderr);print(json.dumps(dict(ok=False,error=type(exc).__name__+': '+str(exc))),flush=True)
