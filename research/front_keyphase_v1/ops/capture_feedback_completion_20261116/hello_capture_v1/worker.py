import sys,json,time,traceback
from contextlib import redirect_stdout
started=time.perf_counter()
with redirect_stdout(sys.stderr):
 from feedback_lesson import *
 from presentation import present
 sys.path.insert(0,str(U.parent/'apple_v1'));from clause_alignment import align_clauses
 sys.path.insert(0,str(O/'word100_learned_guide_feedback_20261102'));from alignment import dtw
 torch.set_num_threads(2);torch.set_num_interop_threads(1);net=FeedbackLesson();ref=read(U/'web/REFERENCE.json');rf=np.load(U/'web/reference_features.npz')['features']
print(json.dumps(dict(ready=True,seconds=time.perf_counter()-started)),flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line)
  if req.get('close'):break
  if req['word']!='SEN0354':raise ValueError('Unsupported lesson')
  raw={k:np.asarray(v)for k,v in req['raw'].items()}
  with redirect_stdout(sys.stderr):
   st=time.perf_counter();q=net.analyze(raw);q['schema']='observed_lesson_requirements_v1';cards,coaching=present(q,ref['clauses']);analysis=time.perf_counter()-st;st=time.perf_counter()
   ch=net.action.adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
   path,display=align_clauses(ch['tcn280'],rf,q.get('display_events',[]),ref['events'],dtw);align=time.perf_counter()-st
   q.pop('trace',None)
  print(json.dumps(dict(ok=True,result=dict(lesson_acceptance=q,cards=cards,coaching=coaching,path=path,display_alignment=display,runtime=dict(seconds=analysis,alignment_seconds=align))),ensure_ascii=False,allow_nan=False),flush=True)
 except Exception as e:
  traceback.print_exc(file=sys.stderr);print(json.dumps(dict(ok=False,error=type(e).__name__+': '+str(e))),flush=True)
