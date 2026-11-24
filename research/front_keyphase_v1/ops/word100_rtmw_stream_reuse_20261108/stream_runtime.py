"""Timestamped single-consumer RTMW stream; no camera, training or file decoding.

Raw scores are preserved, NEVER interpreted as visibility probabilities.
One session per engine. Explicit overflow is audited; no hidden frame dropping.
"""
from pathlib import Path
import os,sys,time,queue,threading,hashlib,json
import numpy as np
U=Path(__file__).resolve().parent;B=U.parent/'word100_rtmw_mediapipe_compare_20261108'
os.environ['OPENVINO_TELEMETRY_DISABLED']='1'
os.environ['INTEL_OPENVINO_TELEMETRY_CONSENT']='0'
sys.dont_write_bytecode=True

class Engine:
    def __init__(self,device='gpu'):
        sys.path.insert(0,str(B/'vendor'))
        import openvino as ov
        from rtmlib import RTMPose,YOLOX
        core=ov.Core();pose=next((B/'models/rtmw_l_384').rglob('*.onnx'))
        det=next((B/'models/person_yolox_m').rglob('*.onnx'))
        sh=core.read_model(pose).input(0).partial_shape
        if str(sh[2])!='384' or str(sh[3])!='288':raise ValueError('Unexpected model shape')
        t=time.perf_counter()
        self.det=YOLOX(str(det),model_input_size=(640,640),backend='openvino',device=device)
        self.pose=RTMPose(str(pose),model_input_size=(288,384),backend='openvino',device=device)
        self.compile_seconds=time.perf_counter()-t
        self.devices={k:list(map(str,v.compiled_model.get_property('EXECUTION_DEVICES'))) for k,v in [('det',self.det),('pose',self.pose)]}
        if not all(device.upper() in s for a in self.devices.values() for s in a):raise RuntimeError(self.devices)
        t=time.perf_counter()
        for _ in range(3):
            self.det(np.zeros((640,640,3),np.uint8))
            self.pose(np.zeros((384,288,3),np.uint8),bboxes=[[0,0,288,384]])
        self.warm_seconds=time.perf_counter()-t
        self.calls=0;self.session_lock=threading.Lock()
    def infer(self,bgr):
        import cv2
        t=time.perf_counter();bb=self.det(bgr)
        xy=np.full((133,2),np.nan,np.float32);score=np.full(133,np.nan,np.float32);box=np.full(4,np.nan,np.float32)
        if len(bb):
            box=np.asarray(bb[np.argmax((bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1]))],np.float32)
            pts,sc=self.pose(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),bboxes=[box])
            if pts.shape!=(1,133,2):raise ValueError('Unexpected RTMW layout')
            xy=pts[0].astype(np.float32);score=sc[0].astype(np.float32)
        self.calls+=1
        return dict(xy=xy,score=score,bbox=box,n_people=len(bb),compute_ms=(time.perf_counter()-t)*1000)

class StreamSession:
    def __init__(self,engine,width,height,capacity=8):
        if width<1 or height<1 or width*height>4096*2160:raise ValueError('Unsupported frame size')
        if not engine.session_lock.acquire(False):raise RuntimeError('Engine already owns a session')
        self.engine=engine;self.width=int(width);self.height=int(height)
        self.queue=queue.Queue(capacity);self.results=[];self.receipts=[];self.errors=[]
        self.last_id=-1;self.last_ts=-1;self.closed=False;self.calls_start=engine.calls
        self.t0=time.perf_counter();self.submit_lock=threading.Lock();self.max_queue=0
        self.thread=threading.Thread(target=self._work,name='rtmw-stream',daemon=True);self.thread.start()
    def submit(self,frame_id,timestamp_us,bgr):
        with self.submit_lock:
            if self.closed:raise RuntimeError('Session closed')
            if type(frame_id) is not int or type(timestamp_us) is not int or frame_id<=self.last_id or timestamp_us<=self.last_ts:raise ValueError('IDs and timestamps must increase strictly')
            if bgr.dtype!=np.uint8 or bgr.shape!=(self.height,self.width,3):raise ValueError('Full BGR uint8 frame shape changed')
            received=time.perf_counter()
            # Own the pixels: capture caller may overwrite its next frame buffer.
            frame=np.array(bgr,copy=True,order='C')
            r=dict(frame_id=frame_id,timestamp_us=timestamp_us,received_s=received-self.t0,accepted=False)
            self.last_id=frame_id;self.last_ts=timestamp_us
            try:
                self.queue.put_nowait((frame_id,timestamp_us,received,frame));r['accepted']=True
                self.max_queue=max(self.max_queue,self.queue.qsize())
            except queue.Full:r['reason']='queue_full'
            self.receipts.append(r)
            return dict(r)
    def _work(self):
        while True:
            item=self.queue.get()
            try:
                if item is None:return
                fid,ts,received,im=item;t=time.perf_counter()
                result=self.engine.infer(im)
                result.update(frame_id=fid,timestamp_us=ts,received_s=received-self.t0,
                              queue_ms=(t-received)*1000,frame_sha256=hashlib.sha256(im.tobytes()).hexdigest())
                result['completed_s']=time.perf_counter()-self.t0
                self.results.append(result)
            except Exception as e:self.errors.append(dict(frame_id=item[0],error=repr(e)))
            finally:self.queue.task_done()
    def finish(self,output):
        with self.submit_lock:
            if self.closed:raise RuntimeError('Already finished')
            self.closed=True
        t=time.perf_counter()
        try:
            self.queue.put(None);self.thread.join()
            drain=time.perf_counter()-t
            p=Path(output)
            if p.exists():raise FileExistsError(p)
            p.mkdir(parents=True)
            r=self.results;ids=np.array([a['frame_id'] for a in r],np.int64)
            xy=np.stack([a['xy'] for a in r]) if r else np.empty((0,133,2),np.float32)
            scores=np.stack([a['score'] for a in r]) if r else np.empty((0,133),np.float32)
            np.savez_compressed(p/'observations.npz',xy_px=xy,score=scores,frame_id=ids,
                timestamp_us=np.array([a['timestamp_us'] for a in r],np.int64),
                frame_sha256=np.array([a['frame_sha256'] for a in r]),
                source_width=self.width,source_height=self.height,
                bbox=np.array([a['bbox'] for a in r],np.float32).reshape(-1,4),
                n_people=np.array([a['n_people'] for a in r],np.int64))
            gaps=[];previous=-1
            for a in self.receipts:
                if a['frame_id']>previous+1:gaps.append([previous+1,a['frame_id']-1])
                previous=a['frame_id']
            complete=bool(r) and not gaps and not self.errors and len(r)==len(self.receipts)
            metadata=dict(status='COMPLETE' if complete else 'INCOMPLETE',missing_input_id_ranges=gaps,
                schema='rtmw133_stream_v1',source='streamed_frames',detector_calls=self.engine.calls-self.calls_start,
                received=len(self.receipts),accepted=sum(x['accepted'] for x in self.receipts),completed=len(r),
                dropped=sum(not x['accepted'] for x in self.receipts),errors=self.errors,receipts=self.receipts,
                queue_capacity=self.queue.maxsize,max_queue=self.max_queue,drain_seconds=drain,
                finish_save_seconds=time.perf_counter()-t,
                records=[{k:v for k,v in a.items() if k not in ['xy','score','bbox']} for a in r],
                raw_score_semantics='SimCC response; NOT visibility or correctness probability',
                whole_video_redetection=False,training=False)
            with (p/'SESSION.json').open('x',encoding='utf-8') as f:json.dump(metadata,f,ensure_ascii=False,indent=2,allow_nan=False)
            return metadata
        finally:self.engine.session_lock.release()

def reuse_observations(path):
    """No detector or source video argument: only the saved stream is consumed.

    A training-ready spatial representation, not an educational error verdict.
    In-bounds finite points can still be hallucinations/occlusions.
    """
    with np.load(Path(path)/'observations.npz',allow_pickle=False) as z:
        xy=z['xy_px'].astype(np.float32);ts=z['timestamp_us'];fid=z['frame_id']
        w,h=int(z['source_width']),int(z['source_height'])
    if not len(ts) or not np.all(np.diff(ts)>0) or not np.all(np.diff(fid)>0):raise ValueError('Invalid stream time axis')
    finite=np.isfinite(xy).all(2)
    bounds=finite&(xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<h)
    shoulders=(xy[:,5]+xy[:,6])/2;scale=np.linalg.norm(xy[:,5]-xy[:,6],axis=1)
    body_ok=bounds[:,5]&bounds[:,6]&np.isfinite(scale)&(scale>1)
    norm=(xy-shoulders[:,None,:])/np.where(body_ok,scale,np.nan)[:,None,None]
    mask=bounds&body_ok[:,None]
    # 48 points: shoulders/elbows/body wrists + 21 joints per hand. No fake 3D z.
    used=np.array([5,6,7,8,9,10]+list(range(91,133)),np.int64)
    body=np.where(mask[:,used,None],norm[:,used],0).astype(np.float32)
    local=[];local_mask=[]
    for start in [91,112]:
        hand=xy[:,start:start+21];hm=bounds[:,start:start+21]
        palm=np.linalg.norm(hand[:,5]-hand[:,17],axis=1)
        ok=hm[:,0]&hm[:,5]&hm[:,17]&np.isfinite(palm)&(palm>1)
        v=(hand-hand[:,0:1])/np.where(ok,palm,np.nan)[:,None,None]
        m=hm&ok[:,None];local.append(np.where(m[:,:,None],v,0));local_mask.append(m)
    return dict(body_xy=body,body_mask=mask[:,used],hand_local_xy=np.stack(local,1).astype(np.float32),
                hand_local_mask=np.stack(local_mask,1),frame_id=fid,timestamp_us=ts,
                schema='rtmw2d_body48_hands21_masks_v1',visibility_validated=False)
