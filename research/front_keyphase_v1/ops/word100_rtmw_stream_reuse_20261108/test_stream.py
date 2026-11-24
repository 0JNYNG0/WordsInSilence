import threading,tempfile,time,json
from pathlib import Path
import numpy as np
from stream_runtime import StreamSession,reuse_observations,U

class Fake:
    def __init__(self,block=False,fail=False):
        self.calls=0;self.session_lock=threading.Lock();self.enter=threading.Event();self.release=threading.Event();self.fail=fail
        if not block:self.release.set()
    def infer(self,im):
        self.enter.set();self.release.wait(3);self.calls+=1
        if self.fail:raise RuntimeError('controlled detector failure')
        xy=np.full((133,2),float(im[0,0,0]),np.float32)
        xy[5]=[2,2];xy[6]=[8,2]
        for s in [91,112]:xy[s]=[3,5];xy[s+5]=[4,5];xy[s+17]=[7,5]
        return dict(xy=xy,score=np.ones(133,np.float32)*9,bbox=np.array([0,0,10,10]),n_people=1,compute_ms=0.)

def main():
    checks=[]
    with tempfile.TemporaryDirectory(dir=U,prefix='synthetic_') as tmp:
        base=Path(tmp);im=np.ones((10,10,3),np.uint8)*4;e=Fake(block=True)
        s=StreamSession(e,10,10,1);assert s.submit(0,0,im)['accepted'];assert e.enter.wait(2)
        im[:]=8
        try:StreamSession(e,10,10);raise AssertionError('Concurrent session allowed')
        except RuntimeError:pass
        assert s.submit(1,33333,im)['accepted'];assert not s.submit(2,66667,im)['accepted']
        for fid,ts in [(2,100000),(3,66667)]:
            try:s.submit(fid,ts,im);raise AssertionError('Non-monotone accepted')
            except ValueError:pass
        e.release.set();m=s.finish(base/'overflow')
        assert m['status']=='INCOMPLETE' and m['dropped']==1 and m['completed']==2
        with np.load(base/'overflow/observations.npz') as z:assert z['xy_px'][0,0,0]==4 and z['xy_px'][1,0,0]==8
        calls=e.calls;r=reuse_observations(base/'overflow');assert e.calls==calls
        assert r['body_xy'].shape==(2,48,2) and not r['visibility_validated']
        assert np.array_equal(r['timestamp_us'],[0,33333]);assert np.allclose(r['body_xy'][0,2],[(-1)/6,2/6])
        try:s.submit(3,100000,im);raise AssertionError('Closed session accepted')
        except RuntimeError:pass
        checks+=['caller_buffer_ownership','exclusive_engine_session','queue_overflow_explicit','nonmonotone_rejected','closed_rejected','no_redetection_on_reuse','relative_coordinates','timestamps_preserved','visibility_not_invented']
        s=StreamSession(e,10,10);s.submit(0,0,im);s.submit(2,66667,im);m=s.finish(base/'gap')
        assert m['missing_input_id_ranges']==[[1,1]] and m['status']=='INCOMPLETE';checks+=['missing_submission_audited']
        e=Fake(fail=True);s=StreamSession(e,10,10);s.submit(0,0,im);m=s.finish(base/'failure')
        assert len(m['errors'])==1 and m['status']=='INCOMPLETE' and not e.session_lock.locked();checks+=['detector_failure_not_success','lock_released_after_failure']
        e=Fake();s=StreamSession(e,10,10);m=s.finish(base/'empty');assert m['status']=='INCOMPLETE';checks+=['empty_not_success']
    result=dict(status='PASS',checks=checks,camera_used=False,training=False)
    (U/'UNIT_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
