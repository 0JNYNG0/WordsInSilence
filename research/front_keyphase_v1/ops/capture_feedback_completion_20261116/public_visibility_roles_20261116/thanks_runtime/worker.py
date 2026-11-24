from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sys,json,time,traceback
from contextlib import redirect_stdout
with redirect_stdout(sys.stderr):
 from thanks_candidate import ThanksCandidate
 from projection_guard import *
 from repair_presentation import present
 from display_alignment import align_observed
 sys.path.insert(0,str(O/'word100_learned_guide_feedback_20261102'))
 from alignment import dtw
 torch.set_num_threads(2);torch.set_num_interop_threads(1);started=time.perf_counter();net=ThanksCandidate()
 ref=read(V/'web/REFERENCE.json');ref_features=np.load(V/'web/reference_features.npz')['features']
 pattern=read(V/'web/DISPLAY_PATTERN.json')
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
