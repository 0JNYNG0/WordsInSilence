"""Two public source clips only. Resumable extraction, exact-front reuse, no training."""
from pathlib import Path
import os,sys,json,hashlib,zipfile,time
os.environ['OMP_NUM_THREADS']='2';sys.dont_write_bytecode=True
import numpy as np,cv2,av
U=Path(__file__).resolve().parent;O=U.parent.parent
sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'))
from stream_runtime import Engine
cv2.setNumThreads(2)
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
engine=None
inputs=read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']
for entry in read(U/'SOURCE_AUDIT.json')['rows']:
 clip=entry['clip'];D=U/clip;V=D/'video_only';V.mkdir(parents=True,exist_ok=True)
 original=next(r for r in inputs if r['clip_id']==clip+'_F');rows=[]
 archive=Path(entry['archive']);stat=archive.stat();start=time.perf_counter()
 with zipfile.ZipFile(archive)as z:
  for view,infos in entry['views'].items():
   target=V/(clip+'_'+view);done=target.with_suffix('.json')
   if done.exists():
    row=read(done);assert sha(row['video'])==row['video_sha256'] and sha(row['raw'])==row['raw_sha256'];rows.append(row);continue
   info=infos[0];video=target.with_suffix('.mp4');payload=z.read(info['member'])
   assert len(payload)==info['bytes'];video.write_bytes(payload);del payload
   reuse=view=='F';rp=Path(original['raw'])if reuse else target.with_suffix('.npz')
   if reuse:assert sha(video)==original['source_sha256'];assert sha(rp)==original['raw_sha256'];old=np.load(rp,allow_pickle=False)
   xy=[];score=[];ts=[];hashes=[];boxes=[];cost=[];people=[]
   with av.open(str(video))as c:
    stream=c.streams.video[0];stream.codec_context.thread_count=2;base=None
    fps=float(stream.average_rate)
    for fid,f in enumerate(c.decode(stream)):
     if base is None:base=f.pts
     timestamp=round(float((f.pts-base)*stream.time_base)*1e6);im=f.to_ndarray(format='bgr24');h=hashlib.sha256(im.tobytes()).hexdigest()
     ts.append(timestamp);hashes.append(h)
     if reuse:
      assert int(old['frame_id'][fid])==fid and int(old['timestamp_us'][fid])==timestamp
      if 'frame_sha256'in old:assert old['frame_sha256'][fid]==h
     else:
      if engine is None:engine=Engine();write(U/'ENGINE.json',dict(devices=engine.devices,compile_seconds=engine.compile_seconds))
      q=engine.infer(im);xy.append(q['xy']);score.append(q['score']);boxes.append(q['bbox']);cost.append(q['compute_ms']);people.append(q['n_people'])
   if reuse:assert len(ts)==len(old['frame_id'])
   else:np.savez_compressed(rp,xy_px=xy,score=score,timestamp_us=np.array(ts,np.int64),frame_id=np.arange(len(ts)),frame_sha256=hashes,bbox=boxes,n_people=people,compute_ms=cost,source_width=im.shape[1],source_height=im.shape[0])
   row=dict(clip_id=clip+'_'+view,view=view,video=str(video),video_sha256=sha(video),raw=str(rp),raw_sha256=sha(rp),frames=len(ts),fps=fps,width=im.shape[1],height=im.shape[0],timestamps_us=ts,reused_existing=reuse,source_member=info['member'])
   write(done,row);rows.append(row);print(clip,view,len(ts),'reused',reuse,flush=True)
 assert archive.stat().st_size==stat.st_size and archive.stat().st_mtime_ns==stat.st_mtime_ns
 assert len(rows)==5 and all(r['timestamps_us']==rows[0]['timestamps_us']for r in rows)
 write(V/'EXTRACTION_COMPLETE.json',dict(status='FIVE_VIEW_OBSERVATIONS_COMPLETE',rows=rows,same_timestamp_arrays=True,source_original_unchanged=True,hardware_sync_proven=False,elapsed_seconds=time.perf_counter()-start))
 print('CLIP_DONE',clip,flush=True)
