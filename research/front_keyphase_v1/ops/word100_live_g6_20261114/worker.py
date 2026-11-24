"""Persistent local worker. Private model imports never collide with live TCN imports."""
import sys,json,time,traceback
from contextlib import redirect_stdout
import numpy as np
with redirect_stdout(sys.stderr):
 import engine
 engine.torch.set_num_threads(4);engine.torch.set_num_interop_threads(1)
 started=time.perf_counter();models=engine.initialize()
print(json.dumps(dict(ready=True,seconds=time.perf_counter()-started)),flush=True)
for line in sys.stdin:
 try:
  q=json.loads(line)
  if q.get('close'):break
  raw={k:np.asarray(v) for k,v in q['raw'].items()}
  A=engine.load('g6_shared_adapter',engine.O/'word100_rtmw_shared_input_20261108/adapter.py')
  with redirect_stdout(sys.stderr):
   ch=A.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
   result=engine.infer(models,ch,raw,q['word'])
  print(json.dumps(dict(ok=True,result=result),ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()),flush=True)
 except Exception as exc:
  traceback.print_exc(file=sys.stderr)
  print(json.dumps(dict(ok=False,error=type(exc).__name__+': '+str(exc))),flush=True)
