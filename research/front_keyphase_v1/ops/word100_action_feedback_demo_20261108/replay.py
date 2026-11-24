"""Public integration replay; does not open a camera or re-run completed scores."""
from task import *
import numpy as np,cv2,av,traceback
from capture import Recorder,Engine
from consumer import FeedbackConsumer
from auto_end import ReturnToRest
def main():
 assert not (U/'INTEGRATION_COMPLETE.json').exists();prep=read(U/'PREPARED.json');rows=[r for r in prep['rows'] if r['person']==11];f=read(U/'FROZEN.json');write(U/'INTEGRATION_FROZEN.json',dict(rows=[r['clip_id'] for r in rows],purpose='New recording+new evidence head+UI API integration timing, not another model accuracy scoring run',once=True,real_camera=False,edited_cases=['WORD0739','WORD2523'],expected_input_loss=0,second_detector_calls=0,protected_sha=sha(U/'FROZEN.json')))
 t=time.perf_counter();engine=Engine();consumer=FeedbackConsumer();prepare=time.perf_counter()-t;results=[];examples=[]
 for row in rows:
  word=prep['words'][row['word']];eid=word+'_normal';d=U/'examples'/eid;d.mkdir(parents=True,exist_ok=False);raw=[];rest=ReturnToRest();stop_at=None;frames=[];t0=None
  with av.open(row['source']) as source:
   stream=source.streams.video[0];stream.thread_count=2;w,h=stream.width,stream.height;rec=Recorder(d/'video.mp4',w,h)
   for fid,frame in enumerate(source.decode(stream)):
    bgr=frame.to_ndarray(format='bgr24');ts=round(float(frame.pts*frame.time_base)*1e6)
    if t0 is None:t0=time.perf_counter()-ts/1e6
    delay=t0+ts/1e6-time.perf_counter()
    if delay>0:time.sleep(delay)
    assert rec.submit(fid,ts,bgr);o=engine.infer(bgr);raw.append(o);frames.append(dict(frame_id=fid,timestamp_us=ts,hash=hashlib.sha256(bgr.tobytes()).hexdigest()))
    if ts<400000:rest.calibrate(o['xy'],w,h)
    else:
     if rest.started is None:rest.begin(ts)
     if rest.update(o['xy'],w,h,ts) and stop_at is None:stop_at=ts/1e6
  stopped=time.perf_counter();rr=rec.finish();arrays=dict(xy_px=np.stack([r['xy'] for r in raw]),score=np.stack([r['score'] for r in raw]),timestamp_us=np.array([r['timestamp_us'] for r in frames]),frame_id=np.arange(len(raw)),source_width=w,source_height=h);np.savez_compressed(d/'observations.npz',**arrays)
  result=consumer.raw(arrays,word);result.update(video=f'/examples/{eid}/video.mp4',source='public_normal_ID11',synthetic=False,stop_to_result_seconds=time.perf_counter()-stopped);write(d/'result.json',result)
  timing=dict(word=word,frames=len(raw),recorded=rr['count'],errors=rr['errors'],drops=rr['dropped'],stop_to_result_seconds=result['stop_to_result_seconds'],last_planned_frame_to_result_seconds=time.perf_counter()-(t0+arrays['timestamp_us'][-1]/1e6),compute_seconds=result['compute_seconds'],auto_end_calibrated=rest.rest is not None,auto_end_seconds=stop_at,main_interval=row['intervals'],detector_calls=len(raw),second_pass_detector_calls=0,source_sha256=sha(row['source']))
  write(d/'REPLAY.json',dict(**timing,records=frames,recorder=rr));results.append(timing);examples.append(dict(id=eid,word=word,label=f['labels'][word]+' · 공개 정상 동작',url=f'/examples/{eid}/result.json'));event('integration_replay',**timing)
  if word not in ['WORD0739','WORD2523']:continue
  control=next(r for r in read(U/'EVALUATION.json')['rows'] if r['seed']==42 and r['clip_id']==row['clip_id'] and r['kind']=='omit_last_main_then_rest');mapping=np.asarray(control['mapping']);eid=word+'_edited';dd=U/'examples'/eid;dd.mkdir();rec=Recorder(dd/'video.mp4',w,h);selected=set(mapping.tolist());k=0
  with av.open(row['source']) as source:
   for i,frame in enumerate(source.decode(video=0)):
    if i in selected:assert rec.submit(k,round(k*1e6/30),frame.to_ndarray(format='bgr24'));k+=1
  rr=rec.finish();assert rr['count']==len(mapping) and not rr['errors'] and not rr['dropped'];a={key:(v[mapping] if key in ['xy_px','score'] else v) for key,v in arrays.items()};a['timestamp_us']=np.rint(np.arange(len(mapping))*1e6/30).astype(np.int64);a['frame_id']=np.arange(len(mapping));result=consumer.raw(a,word);result.update(video=f'/examples/{eid}/video.mp4',source='public_edited_not_real_learner',synthetic=True,edit_note='공개 정상 영상의 주요 구간 뒤쪽을 일부러 지운 시험입니다. AI가 모든 누락을 찾아내지는 못했습니다.',original_removed_frames=control['removed']);write(dd/'result.json',result);examples.append(dict(id=eid,word=word,label=f['labels'][word]+' · 일부러 뒤 동작을 지운 시험',url=f'/examples/{eid}/result.json'))
 write(U/'EXAMPLES.json',examples);write(U/'INTEGRATION_COMPLETE.json',dict(prepare_seconds=prepare,devices=engine.devices,runs=results,actual_webcam=False,training_use=False,accuracy_rescored=False));event('integration_complete',runs=len(results))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'INTEGRATION_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
