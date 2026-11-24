import sys,json,time,traceback
from contextlib import redirect_stdout
with redirect_stdout(sys.stderr):
 from required_role_candidate import *
 from apple_presentation import present
 from clause_alignment import align_clauses
 sys.path.insert(0,str(O/'word100_learned_guide_feedback_20261102'));from alignment import dtw
 torch.set_num_threads(2);torch.set_num_interop_threads(1);started=time.perf_counter();net=RequiredRoleCandidate();ref=read(U/'web/REFERENCE.json');rf=np.load(U/'web/reference_features.npz')['features']
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
