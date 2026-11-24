from pathlib import Path
import sys,time,json,threading,numpy as np,av
U=Path(__file__).resolve().parent;D=U.parent/'word100_action_feedback_demo_20261108';sys.path.insert(0,str(D))
from capture import Recorder
def main():
 checks=[];cases={
 'irregular_30':[0,50000,70000,105000,155000,175000,210000,240000,290000,320000],
 'regular_30':[round(i*1e6/30) for i in range(40)],
 'regular_60':[round(i*1e6/60) for i in range(60)],
 'near_timestamps':[0,1,10001,10002,30000,49999,50000,83000]}
 for name,ts in cases.items():
  p=U/(name+'.mp4');r=Recorder(p,320,240);accepted=[]
  for i,t in enumerate(ts):accepted.append(r.submit(i,t,np.full((240,320,3),i%200,np.uint8)))
  before=time.perf_counter();out=r.finish();elapsed=time.perf_counter()-before
  with av.open(str(p)) as v:
   frames=list(v.decode(video=0));got=[round(float(f.pts*f.time_base)*1e6) for f in frames]
  checks.append(dict(name=name,pass_=all(accepted) and not out['errors'] and not out['dropped'] and got==ts and not out['worker_alive'],submitted=len(ts),encoded=out['count'],decoded=len(got),max_pts_error_us=max([abs(a-b) for a,b in zip(got,ts)],default=0),finish_seconds=elapsed,errors=out['errors']))
 # An encoder that has already died must not leave cleanup blocked on a full queue.
 release=threading.Event()
 class FailedRecorder(Recorder):
  def work(self):
   release.wait(1);self.errors.append('injected_encoder_failure');self.worker_done.set()
 r=FailedRecorder(U/'unused.mp4',320,240)
 for i in range(64):assert r.submit(i,i*33333,np.zeros((240,320,3),np.uint8))
 release.set();r.thread.join(1);t=time.perf_counter();out=r.finish(timeout=.3);elapsed=time.perf_counter()-t
 checks.append(dict(name='full_queue_dead_encoder_cleanup',pass_=elapsed<.2 and not out['worker_alive'] and not r.submit(65,2200000,np.zeros((240,320,3),np.uint8)),finish_seconds=elapsed))
 # Genuine library failure, not just a simulated worker exception.
 r=Recorder(U/'directory_does_not_exist'/'bad.mp4',320,240);r.submit(0,0,np.zeros((240,320,3),np.uint8));r.worker_done.wait(1);t=time.perf_counter();out=r.finish(timeout=.3)
 checks.append(dict(name='actual_encoder_open_failure',pass_=bool(out['errors']) and not out['worker_alive'] and time.perf_counter()-t<.2))
 report=dict(status='PASS' if all(x['pass_'] for x in checks) else 'FAIL',checks=checks,source='Synthetic pixels and independent timing schedules only; no user pixels/keypoints used',learning=False)
 (U/'RECORDER_VERIFIED.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False));assert report['status']=='PASS'
if __name__=='__main__':main()
