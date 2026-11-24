"""Shared extraction, separate neural shape/sequence branches, one learned DTW."""
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys,time,traceback
import numpy as np
from relation_runtime import V as U,O,P,read,RelationsEngine
sys.path.insert(0,str(P))
import live_adapter as shared
from engine import WORDS
write=shared.write
@lru_cache(maxsize=2)
def engine(condition):return RelationsEngine(condition)
def image_information(source,ids):
 import av
 valid=[];last=-1;origin=None;dropped=0;kept=0
 with av.open(str(source)) as c:
  stream=c.streams.video[0]
  for packet in c.demux(stream):
   for frame in packet.decode():
    if frame.pts is None:continue
    if origin is None:origin=frame.pts
    ts=int(round(float((frame.pts-origin)*stream.time_base)*1e6))
    if ts<=last:dropped+=1;continue
    last=ts;assert kept+dropped==ids[kept];kept+=1
    # Entirely uniform/blank pixels are not an observation of a hand.
    # This is image availability, NOT per-finger visibility confidence.
    im=frame.to_ndarray(format='rgb24')[::16,::16].astype(float)
    valid.append(bool(im.std()>1 or im.max()-im.min()>4))
 assert kept==len(ids);return np.asarray(valid,bool)
def analyze(source,directory,word,condition="rgb_geometry",progress=None):
 source=Path(source);d=Path(directory).resolve();assert d.is_relative_to(U.resolve());d.mkdir(parents=True,exist_ok=False)
 t0=time.perf_counter();timing={}
 try:
  ids,ts,w,h=shared.source_axis(source);assert len(ids)>=16
  compiled,device=shared.compile_rgb()
  with ThreadPoolExecutor(max_workers=2) as pool:
   rgb_job=pool.submit(shared.extract_rgb,source,ids,ts,w,h,compiled,None);quality=pool.submit(image_information,source,ids)
   t=time.perf_counter();ob,face=shared.base._capture(source,d,None);timing['mediapipe_seconds']=time.perf_counter()-t
   rgb,pre,rgb_seconds=rgb_job.result();available=quality.result()
  assert np.array_equal(ids,ob.frame_id) and np.array_equal(ts,ob.timestamp_us)
  full={k:getattr(ob,k).copy() for k in ['xy_px','usable','observed','frame_id','timestamp_us']}
  observer=shared.base._module('_retained_motion_geometry',O/'word100_guide_video_feedback_20261028/motion_observer.py');feat=observer.extract_features(full)
  a=dict(x=np.nan_to_num(feat['x']).astype(np.float32),mask=feat['mask'],rgb=rgb,prelinear=pre,frame_id=ids,timestamp_us=ts,image_available=available)
  e=engine(condition);p=e.infer(a,word);result=e.compare(p,a,word)
  timing.update(rgb_seconds=rgb_seconds,sequence_seconds=p['head_seconds'],shape_seconds=p['retained_head_seconds'],dtw_seconds=result['alignment_seconds'],total_seconds=time.perf_counter()-t0)
  timing['total_analysis_seconds']=timing['total_seconds'];timing['relation_head_seconds']=p['relation_head_seconds']
  result.update(latency=timing,device=device,frames=len(ids),source_sha256=shared.sha(source),new_training=0,training=False,
    actual_sensor_extent=[0,0,w,h],rest_hands_required=False)
  xy=ob.xy_px.astype(float);xy[~ob.usable]=np.nan
  result['overlay']=dict(width=w,height=h,times=(ts/1e6).tolist(),points=[[[float(v) if np.isfinite(v) else None for v in pt] for pt in frame] for frame in xy])
  shared.base._save(d/'analysis.npz',**a,**{k:v for k,v in p.items() if isinstance(v,np.ndarray)})
  shared.write(d/'RESULT.json',result);return result
 except BaseException:
  (d/'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8');raise
def warmup():
 result=shared.warmup()
 for c in ["rgb_geometry","geometry_only"]:engine(c)
 return result
