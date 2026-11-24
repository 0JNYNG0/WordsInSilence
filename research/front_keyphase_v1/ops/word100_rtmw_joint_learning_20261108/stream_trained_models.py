"""New condition: one live-paced RTMW stream feeds both newly trained networks."""
from common import *
import io,zipfile,traceback
from concurrent.futures import ThreadPoolExecutor
import av,cv2,numpy as np,torch
from runtime_candidate import Candidate
sys.path.insert(0,str(O/'word100_rtmw_browser_capture_20261108'))
from native_replay import Recorder,prefetched_frames
from stream_runtime import Engine,StreamSession

def main():
 assert not (U/'TRAINED_STREAM_COMPLETE.json').exists();assert (U/'TCN_TRAINING_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(2)
 rows=[r for r in read(U/'SOURCES.json') if r['feedback'] and r['role']=='development'][:3]
 write(U/'TRAINED_STREAM_SCOPE.json',dict(conditions='RTMW-only observation, both fresh and transfer TCN/feedback consumers, all fixed first seeds; all outputs retained',
  clips=[r['clip_id'] for r in rows],camera=False,frame_loss_allowed=0,post_capture_budget_ms=1000,new_condition_not_previous_stream_rerun=True,
  actual_image_bounds=True,no_rest_wrist_crop=True,score_as_probability=False,educational_error_accuracy=False))
 engine=Engine();t=time.perf_counter();models={}
 for tc in ['finetune_C','fresh']:
  for fc in ['pose_transfer','pose_fresh']:models[tc+'+'+fc]=Candidate(tc,fc)
 load_seconds=time.perf_counter()-t;runs=[]
 # Warm all shapes before starting capture; separately report startup cost.
 for r in rows:
  rec=read(U/'records'/f"{r['clip_id']}.json");z=np.load(rec['channel_path'])
  for model in models.values():model.from_channels(z['full'],z['timestamp_us'],z['frame_id'],r['word_id'])
 for r in rows:
  d=U/'trained_stream'/r['clip_id'];d.mkdir(parents=True,exist_ok=False)
  with zipfile.ZipFile(r['archive']) as z:data=z.read(r['member'])
  assert hashlib.sha256(data).hexdigest()==r['source_sha256'];receipts=[]
  with av.open(io.BytesIO(data)) as con:
   vst=con.streams.video[0];vst.thread_count=2;ss=StreamSession(engine,vst.width,vst.height,capacity=64);recorder=Recorder(d/'recorded.mp4',vst.width,vst.height)
   t0=None;base=None
   for fid,(pts,tb,bgr) in enumerate(prefetched_frames(vst,con)):
    if base is None:base=pts
    ts=round(float((pts-base)*tb)*1e6);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    if t0 is None:t0=time.perf_counter()
    due=t0+ts/1e6;delay=due-time.perf_counter()
    if delay>0:time.sleep(delay)
    at=time.perf_counter();accepted=ss.submit(fid,ts,bgr)['accepted'];recorded=recorder.submit(fid,ts,rgb)
    receipts.append(dict(frame_id=fid,timestamp_us=ts,accepted=accepted,recorded=recorded,lateness_ms=(at-due)*1000))
  last_due=t0+receipts[-1]['timestamp_us']/1e6;stop=time.perf_counter()
  with ThreadPoolExecutor(2) as pool:
   sf=pool.submit(ss.finish,d/'rtmw');rf=pool.submit(recorder.finish);sm=sf.result();coordinate_ready=time.perf_counter();rm=rf.result();recording_ready=time.perf_counter()
  raw=np.load(d/'rtmw/observations.npz');condition_results=[]
  # Measure each consumer independently from the same saved observations. No repeated detection.
  for condition_index,(name,model) in enumerate(models.items()):
   start=time.perf_counter();out=model.from_raw(raw,r['word_id']);finished=time.perf_counter();seconds=finished-start
   path=d/(name.replace('+','_')+'.json');write(path,out)
   condition_results.append(dict(condition=name,models_seconds=seconds,
      timing_kind='direct_measured_first_consumer' if condition_index==0 else 'common_drain_plus_own_compute_estimate',
      post_last_capture_ms=((finished-last_due) if condition_index==0 else (max(coordinate_ready,recording_ready)-last_due+seconds))*1000,
      post_actual_stop_ms=((finished-stop) if condition_index==0 else (max(coordinate_ready,recording_ready)-stop+seconds))*1000,output=str(path),sha256=sha(path),timing=out['timing']))
  assert sm['completed']==r['frames_header']==rm['count'] and sm['status']=='COMPLETE' and not rm['errors'] and not rm['dropped']
  out=dict(clip_id=r['clip_id'],frames=r['frames_header'],rtmw_frames=sm['completed'],recorded_frames=rm['count'],receipts=receipts,
   raw_path=str(d/'rtmw/observations.npz'),raw_sha256=sha(d/'rtmw/observations.npz'),recording=str(d/'recorded.mp4'),
   last_capture_to_coordinates_ms=(coordinate_ready-last_due)*1000,last_capture_to_recording_ms=(recording_ready-last_due)*1000,
   capture_lateness_max_ms=max(x['lateness_ms'] for x in receipts),conditions=condition_results,second_pass_detection=0,RGB_model_calls=0,
   note='Fixed first pair finetune_C+pose_transfer is actual wall time, selected before TCN scores. Other pairs are common drain plus their own measured compute, explicitly estimates. Not a selected deployment.')
  write(d/'RESULT.json',out);write(d/'RECORDING.json',rm);runs.append(out);event('trained_stream_result',clip=r['clip_id'],conditions=[{k:x[k] for k in ['condition','post_last_capture_ms']} for x in condition_results])
 write(U/'TRAINED_STREAM_COMPLETE.json',dict(status='COMPLETE_PENDING_SAVED_AUDIT',runs=runs,engine_compile_seconds=engine.compile_seconds,engine_warm_seconds=engine.warm_seconds,
   neural_load_seconds=load_seconds,runtime=dict(python=sys.version,torch=torch.__version__,device=engine.devices),
   camera=False,web_or_unity_rendering_included=False,production_changed=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'STREAM_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
