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
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'public_visibility_roles_20261116'))
 from apple_candidate import RoleEvidenceCandidate
 from required_role_candidate import *
 from apple_presentation import present
 from clause_alignment import align_clauses
 sys.path.insert(0,str(O/'word100_learned_guide_feedback_20261102'));from alignment import dtw
 torch.set_num_threads(2);torch.set_num_interop_threads(1);started=time.perf_counter();net=RoleEvidenceCandidate();ref=read(U/'web/REFERENCE.json');rf=np.load(U/'web/reference_features.npz')['features']
Path(__file__).with_name('LOADED_WEIGHTS.json').write_text(json.dumps([dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())for p in dict.fromkeys(_loaded)],indent=2),encoding='utf-8')
print(json.dumps(dict(ready=True,seconds=time.perf_counter()-started)),flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line)
  if req.get('close'):break
  if req['word']!='WORD0099':raise ValueError('Unsupported lesson')
  raw={k:np.asarray(v)for k,v in req['raw'].items()}
  with redirect_stdout(sys.stderr):
   start=time.perf_counter();q=net.analyze(raw);q['schema']='observed_lesson_requirements_v1';cards,coaching=present(q,ref['clauses']);analysis=time.perf_counter()-start;t=time.perf_counter()
   events=sorted(q['events'],key=lambda e:e['start']);path,display=align_clauses(net.last_features,rf,events,ref['events'],dtw);alignment=time.perf_counter()-t
   # High-volume debug arrays stay local, not in every game result.
   q.pop('observations',None);q.pop('terminal_states',None)
  print(json.dumps(dict(ok=True,result=dict(lesson_acceptance=q,cards=cards,coaching=coaching,path=path,display_alignment=display,runtime=dict(seconds=analysis,alignment_seconds=alignment))),ensure_ascii=False,allow_nan=False),flush=True)
 except Exception as e:
  traceback.print_exc(file=sys.stderr);print(json.dumps(dict(ok=False,error=type(e).__name__+': '+str(e))),flush=True)
