"""One new clip -> learned network -> learned prototype; never nine references."""
from pathlib import Path
from functools import lru_cache
import os,sys,time,traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from engine import LearnedGuideEngine,WORDS
from prepare import U,read,sha
from prepare_projection import load_module
O=U.parent;S=O/'word100_signrep_feasibility_20261029'
base=load_module('_learned_newclip_capture',O/'word100_seated_web_framing_20261102/adapter.py')
def write(p,d):base._write(p,d)
@lru_cache(maxsize=1)
def compile_rgb():
 os.environ['OPENVINO_TELEMETRY_DISABLED']='1';os.environ['INTEL_OPENVINO_TELEMETRY_CONSENT']='0';sys.modules.setdefault('openvino_telemetry',None)
 import openvino as ov
 exported=read(S/'EXPORT.json')
 for name,key in [('xml','xml_sha256'),('bin','bin_sha256')]:
  assert sha(S/f'converted/signrep_features_fp32.{name}')==exported[key]
 core=ov.Core();attempts=[]
 for device in (['GPU','CPU'] if any(x.startswith('GPU') for x in core.available_devices) else ['CPU']):
  try:
   ir=core.read_model(S/'converted/signrep_features_fp32.xml');last=ir.output(0).get_node().input_value(0).get_node()
   assert last.get_type_name()=='MatMul' and 'fc.1' in last.get_friendly_name()
   assert np.array_equal(np.asarray(last.input_value(1).get_node().get_data()),np.load(U/'PRETRAINED_PROJECTION.npy'))
   ir.add_outputs([last.input_value(0)]);ir.reshape([1,3,16,224,224])
   compiled=core.compile_model(ir,device,{'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':'f16' if device=='GPU' else 'f32','NUM_STREAMS':'1'})
   actual=list(compiled.get_property('EXECUTION_DEVICES'));assert all(device in x for x in actual)
   return compiled,dict(requested=device,actual=actual,failed_attempts=attempts)
  except Exception as e:attempts.append(dict(device=device,error=repr(e)))
 raise RuntimeError(str(attempts))
@lru_cache(maxsize=2)
def engine(condition):
 assert condition in ('projection_finetune','frozen_projection')
 e=LearnedGuideEngine(condition,42);e.model.requires_grad_(False);return e
def warmup():
 t=time.perf_counter();m,dev=compile_rgb();m([np.zeros((1,3,16,224,224),np.float32)])
 for c in ('projection_finetune','frozen_projection'):engine(c)
 return dict(device=dev,seconds=time.perf_counter()-t,source='synthetic zeros only')
def source_axis(source):
 import av
 from fractions import Fraction
 ids=[];times=[];origin=None;last=-1;dropped=0;decoded=0
 with av.open(str(source)) as container:
  stream=container.streams.video[0];tb=stream.time_base or Fraction(1,1000)
  width,height=int(stream.codec_context.width),int(stream.codec_context.height)
  for packet in container.demux(stream):
   for frame in packet.decode():
    decoded+=1
    if decoded>base.MAX_DECODED_FRAMES:raise ValueError('영상 프레임 제한을 초과했습니다.')
    if frame.pts is None:continue
    if origin is None:origin=frame.pts
    ts=int(round(float((frame.pts-origin)*tb)*1e6))
    if ts>base.MAX_ELAPSED_US:raise ValueError('영상은 20초 이하로 선택해 주세요.')
    if ts<=last:dropped+=1;continue
    last=ts;ids.append(len(ids)+dropped);times.append(ts)
 return np.asarray(ids,np.int64),np.asarray(times,np.int64),width,height
def extract_rgb(source,ids,ts,width,height,compiled,progress):
 t=time.perf_counter()
 class Tap:
  def __init__(self):self.pre=[]
  def __call__(self,inputs):
   out=compiled(inputs);self.pre.append(np.array(out[1],np.float32,copy=True).reshape(1,768));return {0:np.array(out[0],np.float32,copy=True)}
 tap=Tap();rgb,windows,starts,centers=base._encode(source,[0,0,width,height],ids,ts,tap,progress)
 pre=np.concatenate(tap.pre);prelinear=np.stack([np.interp(ts,centers,pre[:,j]) for j in range(768)],1).astype(np.float32)
 return rgb,prelinear,time.perf_counter()-t
def analyze(source,directory,word,condition='projection_finetune',progress=None):
 assert word in WORDS;source=Path(source).resolve();directory=Path(directory).resolve()
 assert directory.is_relative_to(U.resolve());directory.mkdir(parents=True,exist_ok=False)
 t0=time.perf_counter();timings={};source_hash=sha(source)
 write(directory/'STARTED.json',dict(source_sha256=source_hash,training=False,target=word,condition=condition))
 def notify(s):
  if progress:progress(s)
 try:
  t=time.perf_counter();ids,ts,width,height=source_axis(source);timings['source_pts_scan_seconds']=time.perf_counter()-t
  if len(ids)<16:raise ValueError('비교할 실제 영상이 너무 짧습니다.')
  t=time.perf_counter();compiled,device=compile_rgb();timings['compile_seconds']=time.perf_counter()-t
  with ThreadPoolExecutor(max_workers=1) as pool:
   rgb_future=pool.submit(extract_rgb,source,ids,ts,width,height,compiled,progress)
   t=time.perf_counter();ob,face=base._capture(source,directory,progress);timings['decode_mediapipe_seconds']=time.perf_counter()-t
   rgb,prelinear,rgb_seconds=rgb_future.result()
  assert np.array_equal(ids,ob.frame_id) and np.array_equal(ts,ob.timestamp_us),'Parallel branches source PTS mismatch'
  if ob.n_frames<16:raise ValueError('비교할 실제 영상이 너무 짧습니다. 동작을 마친 후 촬영을 종료해 주세요.')
  width,height=int(ob.sidecar['width']),int(ob.sidecar['height'])
  # Actual sensor extent: no artificial horizontal cutoff or rest-wrist requirement.
  full={k:getattr(ob,k).copy() for k in ('xy_px','usable','observed','frame_id','timestamp_us')}
  observer=base._module('_learned_newclip_geometry',O/'word100_guide_video_feedback_20261028/motion_observer.py')
  feat=observer.extract_features(full)
  if not feat['mask'].any():raise ValueError('화면 안에서 손·팔 정보를 확인하지 못했습니다. 준비할 때 손은 화면 밖에 있어도 됩니다.')
  a=dict(x=np.nan_to_num(feat['x']).astype(np.float32),mask=feat['mask'],frame_id=ob.frame_id,timestamp_us=ob.timestamp_us,image_available=np.ones(ob.n_frames,bool))
  a['rgb']=rgb;a['prelinear']=prelinear;timings['rgb_signrep_seconds']=rgb_seconds
  e=engine(condition);prediction=e.infer(a,word);result=e.compare(prediction,a,word)
  timings['neural_head_seconds']=prediction['head_seconds'];timings['learned_dtw_seconds']=result['alignment_seconds']
  base._save(directory/'analysis.npz',**a,**{k:v for k,v in prediction.items() if isinstance(v,np.ndarray)})
  assert sha(source)==source_hash
  timings['total_analysis_seconds']=time.perf_counter()-t0
  result.update(condition=condition,seed=42,latency=timings,device=device,source_sha256=source_hash,training=False,
   actual_sensor_extent=[0,0,width,height],rest_hands_required=False,frames=ob.n_frames,parallel_observers=True,
   expression_face_analysis=False,face_geometry_saved_but_not_learned_feature=True,
   observation_notes='Full frame detector and features. Image availability means decoded frames, not verified visibility of every finger.')
  xy=ob.xy_px.astype(float);xy[~ob.usable]=np.nan
  points=[[[float(v) if np.isfinite(v) else None for v in pt] for pt in frame] for frame in xy]
  result['overlay']=dict(width=width,height=height,times=(ob.timestamp_us/1e6).tolist(),points=points)
  write(directory/'RESULT.json',result);notify(dict(stage='complete',seconds=timings['total_analysis_seconds']))
  return result
 except BaseException:
  (directory/'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
