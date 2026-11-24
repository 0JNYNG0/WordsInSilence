from pathlib import Path
import sys,json,queue,threading,time,hashlib,uuid,importlib.util
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
U=Path(__file__).resolve().parent;O=U.parent;LAB=U.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'));sys.path.insert(0,str(LAB/'src'))
from stream_runtime import Engine,StreamSession,reuse_observations
from wis_lab.service import word100_candidate as C,word100_t3f as T
from wis_lab.core.observation import Observation,empty_frames

class MPWorker:
    def __init__(self):
        self.det=C.CandidateLandmarker(json.loads((LAB/'config/contract.json').read_text(encoding='utf-8')))
        self.q=queue.Queue(8);self.out=empty_frames(900);self.count=0;self.calls=0;self.records=[];self.errors=[];self.dropped=[]
        self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()
    def submit(self,fid,ts,rgb):
        try:self.q.put_nowait((fid,ts,time.perf_counter(),rgb));return True
        except queue.Full:self.dropped.append(fid);return False
    def work(self):
        while True:
            x=self.q.get()
            try:
                if x is None:return
                fid,ts,at,rgb=x;n=self.count
                if n>=900:raise ValueError('Capture too long')
                start=time.perf_counter();self.out['frame_id'][n]=fid
                self.det.process(rgb,ts,self.out,n);self.calls+=1;self.count+=1
                self.records.append(dict(frame_id=fid,timestamp_us=ts,lag_ms=(time.perf_counter()-at)*1000,compute_ms=(time.perf_counter()-start)*1000))
            except Exception as e:self.errors.append(dict(frame_id=x[0],error=repr(e)))
            finally:self.q.task_done()
    def finish(self,d):
        d.mkdir(parents=True,exist_ok=True)
        self.q.put(None);self.thread.join()
        try:
            ob=Observation(**{k:v[:self.count].copy() for k,v in self.out.items()})
            if self.count:ob=self.det.finalize_observation(ob)
            ob.save(str(d/'mp_observations.npz'))
            t=time.perf_counter();pair=T.recognise_pair(ob) if self.count else ({'model_status':'INSUFFICIENT_OBSERVATION'},)*2
            result=dict(count=self.count,detector_calls=self.calls,dropped=self.dropped,errors=self.errors,records=self.records,
                        tcn_seconds=time.perf_counter()-t,T3F=pair[0],C=pair[1],target_word_used=False)
            (d/'MP_TCN.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            return result
        finally:self.det.close()

class CaptureApp:
    def __init__(self):
        self.engine=Engine();self.model_status=T.identity()['model_status']
        if self.model_status!='READY':raise RuntimeError('Frozen TCN unavailable')
        self.lock=threading.Lock();self.session=None;self.mp=None;self.sid=None
        self.completed={};self.receipts=[];self.pools=ThreadPoolExecutor(2)
    def start(self,w,h):
        with self.lock:
            if self.session is not None:raise ValueError('Capture already active')
            # Per-take VIDEO state is fresh; construct before the first frame.
            mp=MPWorker()
            try:s=StreamSession(self.engine,w,h)
            except BaseException:mp.finish(U/'failed_start');raise
            self.mp=mp;self.session=s;self.sid=uuid.uuid4().hex;self.receipts=[]
            return dict(id=self.sid,width=w,height=h,ready=True)
    def frame(self,sid,fid,ts,body):
        with self.lock:
            s=self.session
            if s is None or sid!=self.sid:raise ValueError('Wrong session')
            if fid>=900 or ts>30000000:raise ValueError('Capture bounds exceeded')
            if len(body)!=s.width*s.height*4:raise ValueError('Wrong RGBA size')
            rgba=np.frombuffer(body,np.uint8).reshape(s.height,s.width,4)
            rgb=np.ascontiguousarray(rgba[:,:,:3]);bgr=np.ascontiguousarray(rgb[:,:,::-1])
            received=time.perf_counter();ack=s.submit(fid,ts,bgr);mp_ok=self.mp.submit(fid,ts,rgb)
            self.receipts.append(dict(frame_id=fid,timestamp_us=ts,received_monotonic=received,
                rgba_sha256=hashlib.sha256(body).hexdigest(),rgb_thumbnail=cv2.resize(rgb,(32,18),interpolation=cv2.INTER_AREA).tolist(),
                rtmw_accepted=ack['accepted'],mp_accepted=mp_ok))
            return dict(frame_id=fid,rtmw_accepted=ack['accepted'],mp_accepted=mp_ok)
    def finish(self,sid,expected,cancel=False):
        with self.lock:
            if sid!=self.sid or self.session is None:raise ValueError('Wrong session')
            d=U/'runs'/sid;d.mkdir(parents=True,exist_ok=False);t=time.perf_counter()
            s=self.session;mp=self.mp
            try:
                sf=self.pools.submit(s.finish,d/'rtmw');mf=self.pools.submit(mp.finish,d)
                sm=sf.result();mm=mf.result();features=reuse_observations(d/'rtmw') if sm['completed'] else {}
                np.savez_compressed(d/'pose_analysis_input.npz',**{k:v for k,v in features.items() if isinstance(v,np.ndarray)})
                result=dict(id=sid,expected=expected,rtmw_frames=sm['completed'],mp_frames=mm['count'],
                    rtmw_dropped=sm['dropped'],mp_dropped=mm['dropped'],mp_errors=mm['errors'],
                    input_complete=sm['status']=='COMPLETE' and mm['count']==expected==sm['completed'] and not mm['errors'],
                    finish_to_tcn_and_pose_ms=(time.perf_counter()-t)*1000,cancelled=cancel,
                    T3F=mm['T3F'],C=mm['C'],second_pass_detector_calls=0,
                    feedback_model_ready=False,recording_verified=False)
                if cancel:result['input_complete']=False
                (d/'CAPTURE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
                (d/'RECEIPTS.json').write_text(json.dumps(self.receipts,indent=2),encoding='utf-8')
                self.completed[sid]=result
                return result
            finally:self.session=None;self.mp=None;self.sid=None
    def save_video(self,sid,body):
        if sid not in self.completed:raise ValueError('Unknown completed capture')
        d=U/'runs'/sid
        with (d/'recorded.webm').open('xb') as f:f.write(body)
        return dict(saved=True,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
    def close(self):
        if self.session is not None:self.finish(self.sid,len(self.receipts),cancel=True)
        self.pools.shutdown()
