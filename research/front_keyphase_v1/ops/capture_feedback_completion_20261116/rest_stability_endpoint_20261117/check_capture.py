from check_runtime import *
import torch,cv2
from stream_runtime import Engine
import av,threading,copy,importlib.util,shutil
from stable_runtime import CaptureService,ConsumerProxy
import capture

def wait_for(test,seconds=20):
    end=time.monotonic()+seconds
    while not test()and time.monotonic()<end:time.sleep(.02)
    assert test(),'Fixture state timeout'

def main():
    torch.set_num_threads(2);cv2.setNumThreads(2);service=CaptureService();frames=[];fallback=[];rows=[]
    class PublicCamera:
        def __init__(self,*args):self.i=0
        def isOpened(self):return True
        def set(self,*args):return True
        def get(self,key):return {cv2.CAP_PROP_FRAME_WIDTH:640,cv2.CAP_PROP_FRAME_HEIGHT:480,cv2.CAP_PROP_FPS:30}.get(key,0)
        def read(self):
            time.sleep(1/30)
            if service.state=='recording':
                im=frames[min(self.i,len(frames)-1)];self.i+=1
                if self.i>len(frames)+90:fallback.append(True);service.finish_requested=True
                return True,im.copy()
            return True,frames[0].copy()
        def release(self):pass
    class PublicEngine:
        def __init__(self,inner):self.inner=inner;self.i=0;self.drop=False
        def __getattr__(self,k):return getattr(self.inner,k)
        def infer(self,im):
            q=self.inner.infer(im)
            if service.state=='recording':
                i=self.i;self.i+=1
                if self.drop and 45<=i<=49:
                    q['xy'][125:129]=np.nan;q['score'][125:129]=0
            return q
    previous=capture.cv2.VideoCapture;capture.cv2.VideoCapture=PublicCamera
    net=FeedbackConsumer();engine=PublicEngine(Engine());service.engine=engine;service.consumer=ConsumerProxy(net,service);service.retry_admitted=True;service.state='ready'
    try:
        for word,folder in [('WORD0099',G/'apple_v1'),('WORD1290',G/'completion_v1')]:
            source=read(folder/'web/SOURCES.json')['normal']['video']
            with av.open(source)as f:
                frames=[]
                for fr in f.decode(video=0):
                    im=fr.to_ndarray(format='bgr24');h,w=im.shape[:2];frames.append(cv2.resize(im[int(h*.08):int(h*.75),int(w*.25):int(w*.75)],(640,480)))
            for kind in ['normal','short_public_observation_dropout']:
                engine.i=0;engine.drop=kind!='normal';service.open_camera();wait_for(lambda:service.state=='preview'and service.status()['auto_end_ready'],25)
                service.start(word);wait_for(lambda:service.state=='recording',8);wait_for(lambda:service.state in ['done','error'],40)
                assert service.state=='done',service.status();service.camera_thread.join(5);q=service.result
                meta=q.get('selective_retry',{});passed=q['stop_reason']=='arms_returned'and q['auto_end_calibrated']and q['recorded_frames']==q['observed_frames']and not fallback
                if kind=='normal':passed=passed and q['overall_pass']and q['second_pass_detector_calls']==0 and q['stop_to_result_seconds']<1.5
                else:passed=passed and q['stop_to_result_seconds']<2 and q['second_pass_detector_calls']<=16
                row=dict(word=word,kind=kind,passed=bool(passed),auto_stop=q['stop_reason'],overall_pass=q['overall_pass'],frames=q['recorded_frames'],stop_to_result_seconds=q['stop_to_result_seconds'],retry=meta,source=source,directory=str(service.directory),physical_camera=False)
                assert read(service.directory/'auto_end_diagnostic.json')['version']=='bounded_weak_rest_pause_v2'
                out=D/'public_capture_artifacts'/f'{word}_{kind}';(out/'rtmw').mkdir(parents=True,exist_ok=True)
                for name in ['recorded.mp4','rtmw/observations.npz','auto_end_diagnostic.json']:
                    shutil.copy2(service.directory/name,out/name)
                row['directory']=str(out)
                q['source']='public_capture_fixture';write(D/'web'/f'capture_{word}_{kind}.json',q);rows.append(row);print('CAPTURE',json.dumps(row,ensure_ascii=False),flush=True)
        # Finalizer readiness/error handling is shared and read-only. Its existing
        # auto-stop controls remain directly applicable; actual loop verified above.
        write(D/'web/CAPTURE_CHECK.json',dict(rows=rows,passed=all(r['passed']for r in rows),physical_camera=False,private_data=False,measured_scope='Public camera fixture: end trigger to final calculation, includes encoding drain and any recovery; browser paint excluded'))
    finally:
        service.stop_event.set()
        if service.camera_thread:service.camera_thread.join(5)
        capture.cv2.VideoCapture=previous;close(net)
if __name__=='__main__':main()
