"""One fixed public full-frame extraction; no training or evaluation scores."""
from common import *
import zipfile,io,traceback,importlib.util
import numpy as np
import av,cv2
cv2.setNumThreads(2)
sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'))
from stream_runtime import Engine
spec=importlib.util.spec_from_file_location('shared_rtmw',O/'word100_rtmw_shared_input_20261108/adapter.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)

def convert_views(raw,row):
 full=A.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
 old=read(row['baseline_record']);bottom=int(old['calibration']['box_source_pixels'][3])
 masked=A.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),bottom)
 result=dict(full=full['tcn280'],upper_mask=masked['tcn280'],frame_id=raw['frame_id'],timestamp_us=raw['timestamp_us'],
  point_available=full['point_available'],upper_available=masked['point_available'],native67=full['native67'],native_available=full['native_available'],
  body_relative_xy=full['body_relative_xy'],body_relative_available=full['body_relative_available'],source_width=raw['source_width'],source_height=raw['source_height'],upper_bottom=np.int64(bottom))
 if row['feedback']:
  oldfb=next(r for r in read(O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json')['rows'] if r['clip_id']==row['clip_id'])
  fb=A.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(oldfb['box'][3]))
  result['feedback']=fb['tcn280'];result['feedback_available']=fb['point_available']
 return result

def main():
 f=read(U/'FROZEN.json');assert sha(U/'SOURCES.json')==f['source_manifest_sha256'];assert not (U/'EXTRACTION_COMPLETE.json').exists()
 rows=read(U/'SOURCES.json');engine=None;archives={};reports=[];t0=time.perf_counter()
 (U/'records').mkdir(exist_ok=True);(U/'raw').mkdir(exist_ok=True);(U/'channels').mkdir(exist_ok=True)
 for i,row in enumerate(rows):
  cid=row['clip_id'];rp=U/'records'/f'{cid}.json'
  if rp.exists():
   rr=read(rp);assert rr['source_sha256']==row['source_sha256'];assert sha(rr['raw_path'])==rr['raw_sha256']
   if rr.get('channel_path'):assert sha(rr['channel_path'])==rr['channel_sha256']
   reports.append(rr);continue
  start=time.perf_counter();cache=row['rtmw_cache'];reuse=bool(cache);cost=[]
  if reuse:
   assert sha(cache)==row['rtmw_cache_sha256']
   with np.load(cache,allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
   rawpath=Path(cache)
  else:
   if engine is None:
    engine=Engine();write(U/'ENGINE.json',dict(compile_seconds=engine.compile_seconds,warm_seconds=engine.warm_seconds,devices=engine.devices));event('engine_ready')
   if row['archive'] not in archives:archives[row['archive']]=zipfile.ZipFile(row['archive'])
   data=archives[row['archive']].read(row['member']);assert hashlib.sha256(data).hexdigest()==row['source_sha256']
   xs=[];ss=[];ids=[];ts=[];bb=[];nperson=[];framehash=[]
   with av.open(io.BytesIO(data)) as con:
    stream=con.streams.video[0];stream.thread_type='AUTO';stream.codec_context.thread_count=2;base=None
    for fid,frame in enumerate(con.decode(stream)):
     if base is None:base=frame.pts
     tus=round(float((frame.pts-base)*stream.time_base)*1e6);bgr=frame.to_ndarray(format='bgr24')
     q=engine.infer(bgr);xs.append(q['xy']);ss.append(q['score']);bb.append(q['bbox']);nperson.append(q['n_people']);cost.append(q['compute_ms']);ids.append(fid);ts.append(tus)
     framehash.append(hashlib.sha256(bgr.tobytes()).hexdigest())
   raw=dict(xy_px=np.asarray(xs),score=np.asarray(ss),frame_id=np.asarray(ids,np.int64),timestamp_us=np.asarray(ts,np.int64),
    bbox=np.asarray(bb),n_people=np.asarray(nperson),frame_sha256=np.asarray(framehash),source_width=np.int64(row['width']),source_height=np.int64(row['height']))
   rawpath=U/'raw'/f'{cid}.npz';assert not rawpath.exists(),'Uncommitted raw: review before resume';np.savez_compressed(rawpath,**raw)
  n=len(raw['frame_id']);assert n==row['frames_header'];assert np.array_equal(raw['frame_id'],np.arange(n));assert np.all(np.diff(raw['timestamp_us'])>0)
  if row['feedback']:
   fid=np.asarray(row['feedback_frame_ids']);assert np.array_equal(raw['timestamp_us'][fid],row['feedback_timestamp_us']),cid+' source/label PTS mismatch'
  status='OK';reason=None;channelpath=None
  try:
   channels=convert_views(raw,row);assert np.isfinite(channels['full']).all() and np.isfinite(channels['upper_mask']).all()
   channelpath=U/'channels'/f'{cid}.npz';np.savez_compressed(channelpath,**channels)
  except A.PilotDataError as e:status='INPUT_UNAVAILABLE';reason=str(e)
  rr=dict(clip_id=cid,role=row['role'],person=row['person'],word_id=row['word_id'],label_index=row['label_index'],source_sha256=row['source_sha256'],
   raw_path=str(rawpath),raw_sha256=sha(rawpath),channel_path=str(channelpath) if channelpath else None,channel_sha256=sha(channelpath) if channelpath else None,
   frames=n,reused=reuse,seconds=time.perf_counter()-start,compute_ms_sum=float(sum(cost)),compute_ms_p50=float(np.median(cost)) if cost else None,
   compute_ms_p95=float(np.percentile(cost,95)) if cost else None,status=status,reason=reason,feedback=row['feedback'],frozen_sha256=sha(U/'FROZEN.json'))
  write(rp,rr);reports.append(rr)
  write(U/'EXTRACTION_PROGRESS.json',dict(status='RUNNING',pid=os.getpid(),completed=len(reports),total=len(rows),frames=sum(r['frames'] for r in reports),elapsed_seconds=time.perf_counter()-t0,last=cid))
  if i<50 or i%20==0:event('extracted',done=i+1,clip=cid,seconds=round(rr['seconds'],3),reused=reuse)
 for z in archives.values():z.close()
 write(U/'EXTRACTION_COMPLETE.json',dict(status='COMPLETE_PENDING_INPUT_VERIFICATION',clips=len(reports),frames=sum(r['frames'] for r in reports),
  failures=[r['clip_id'] for r in reports if r['status']!='OK'],reused=sum(r['reused'] for r in reports),wall_seconds=time.perf_counter()-t0,
  record_hashes={r['clip_id']:sha(U/'records'/f"{r['clip_id']}.json") for r in reports}))
 event('extraction_complete',clips=len(reports))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'EXTRACTION_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
