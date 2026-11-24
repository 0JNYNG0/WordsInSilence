"""Exercise camera service lifecycle with a public-input fake device; NO webcam.

Cached RTMW is explicitly substituted for this plumbing test. This is not an
end-to-end hardware latency or new-person recognition benchmark.
"""
import json,time,threading,hashlib
from pathlib import Path
import numpy as np
import cv2
import server
from coach import Coach

U=Path(__file__).resolve().parent

def main():
    original_capture=cv2.VideoCapture
    cap=original_capture(str(server.REFERENCE));frames=[]
    while True:
        ok,im=cap.read()
        if not ok:break
        frames.append(cv2.resize(im,(640,360)))
    cap.release()
    with np.load(U/'source/reference.npz')as z:raw={k:z[k].copy()for k in z.files}
    assert len(frames)==len(raw['timestamp_us'])==131
    fake_calls=[]
    class PublicCamera:
        def __init__(self,index,*args):
            assert index==0;fake_calls.append(index);self.opened=True;self.n=0;self.t0=time.perf_counter()
        def isOpened(self):return self.opened
        def set(self,*args):return True
        def release(self):self.opened=False
        def read(self):
            wait=self.t0+self.n/30-time.perf_counter()
            if wait>0:time.sleep(wait)
            # Three-second preparation, then play the whole source at native rate.
            i=max(0,min(130,self.n-90));self.n+=1;im=frames[i].copy();im[0,0,0]=i
            return True,im
    class CachedEngine:
        def infer(self,im):
            i=int(im[0,0,0]);return dict(xy=raw['xy_px'][i]/3,score=raw['score'][i],compute_ms=0)
    service=server.LiveService();service.coach=Coach();service.engine=CachedEngine();service.camera_available=True;service.state='ready'
    cv2.VideoCapture=PublicCamera
    observed=[]
    try:
        service.start('camera');deadline=time.perf_counter()+12
        while time.perf_counter()<deadline:
            s=service.status();observed.append(dict(state=s['state'],countdown=s['countdown'],completed=(s['observation']or{}).get('completed')))
            if s['state']in ['done','error']:break
            time.sleep(.04)
        assert service.state=='done',service.status()
        end=service.status();assert end['observation']['completed']==[True,True],end
        assert end['observation']['whole_word_correct']is None
        assert any(r['state']=='countdown'for r in observed)
        assert any(r['state']=='live'for r in observed)
        # Late old-generation output and stopped output must never revive progress.
        gen=service.generation;before=service.seq;service.generation+=1
        service.publish(frames[0],raw['xy_px'][0]/3,{'completed':[True,True]}, {},gen)
        assert service.seq==before
        service.stop_event.set();service.publish(frames[0],raw['xy_px'][0]/3,{'completed':[True,True]}, {},service.generation)
        assert service.seq==before
        result=dict(passed=True,physical_camera_opened=False,fake_camera_calls=len(fake_calls),cached_rtmw=True,
            training=False,private_data=False,countdown_to_live_to_done=True,old_generation_and_stop_publish_blocked=True,
            final=end,states=observed,scope=__doc__,server_sha256=hashlib.sha256((U/'server.py').read_bytes()).hexdigest())
        (U/'action_integration/SERVICE_FIXTURE_ACTION_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items()if k not in ['states','final','scope']},ensure_ascii=False))
    finally:
        service.stop();
        if service.worker:service.worker.join(4)
        cv2.VideoCapture=original_capture

if __name__=='__main__':main()
