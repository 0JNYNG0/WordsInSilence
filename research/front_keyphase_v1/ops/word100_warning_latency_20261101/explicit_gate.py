from benchmark import *
import threading,os
def main():
 from explicit_request import ExplicitRequest
 assert not (U/'EXPLICIT_GATE_STARTED.json').exists()
 save(U/'EXPLICIT_GATE_STARTED.json',dict(time=time.time(),pid=os.getpid(),code_sha256=sha(U/'explicit_request.py')))
 def stop():
  save(U/'EXPLICIT_GATE_TIMEOUT.json',dict(status='TIMEOUT_NO_PARTIAL_ACCEPTED',limit_seconds=45,pid=os.getpid()))
  os._exit(124)
 timer=threading.Timer(45,stop);timer.daemon=True;timer.start()
 try:
  m=module('explicit_gate_adapter',U/'candidate_adapter.py');compiled,device=m._compile_signrep()
  compiled([np.zeros((1,3,16,224,224),np.float32)])
  row=read(U/'FROZEN.json')['public_clips'][0];d=U/'bench/clip1_1_serial/adapter'
  meta=read(d/'COMPLETE.json')
  with np.load(d/'analysis.npz') as z:fid=z['frame_id'].copy();ts=z['timestamp_us'].copy()
  profile=[];runner=ExplicitRequest(compiled,profile=profile,jsonl_path=U/'EXPLICIT_WINDOW_TRACE.jsonl')
  t=time.perf_counter();a=m._encode(Path(row['source']),meta['calibration']['box_source_pixels'],fid,ts,runner,None);elapsed=time.perf_counter()-t
  out=U/'explicit_gate_arrays.npz'
  with out.open('xb') as f:np.savez_compressed(f,**dict(zip(['rgb','features','starts','centers'],a)))
  eq=compare(U/'gate_1_serial.npz',out);passed=all(x['equal'] for x in eq.values())
  save(U/'EXPLICIT_GATE.json',dict(status='PASS' if passed else 'FAIL',seconds=elapsed,equality=eq,profile=profile,device=device,windows=len(a[1]),training=False))
  print('PASS' if passed else 'FAIL',elapsed,flush=True)
 finally:timer.cancel()
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'EXPLICIT_GATE_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
