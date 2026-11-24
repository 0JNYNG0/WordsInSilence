"""Read exactly one public REAL02 clip from five views; reuse front RTMW by source/frame hashes."""
from pathlib import Path
import os,sys,json,hashlib,zipfile,time,traceback
os.environ['OMP_NUM_THREADS']='2';sys.dont_write_bytecode=True
import numpy as np,av,cv2
U=Path(__file__).resolve().parent;O=U.parent;V=U/'video_only';V.mkdir(exist_ok=True)
sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'))
from stream_runtime import Engine
cv2.setNumThreads(2)
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
contract=json.loads((U/'VIDEO_ONLY_CONTRACT.json').read_text('utf-8'))
archive=Path(contract['source_archive']);stat=archive.stat()
pin=dict(archive=str(archive),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,contract_sha256=sha(U/'VIDEO_ONLY_CONTRACT.json'))
if (V/'SOURCE_PIN.json').exists():assert json.loads((V/'SOURCE_PIN.json').read_text())==pin
else:write(V/'SOURCE_PIN.json',pin)
engine=None;rows=[]
try:
    with zipfile.ZipFile(archive)as z:
        for view in contract['views']:
            cid=contract['clip']+'_'+view;member='02/'+cid+'.mp4';info=z.getinfo(member);video=V/(cid+'.mp4')
            done=V/(cid+'.json')
            if done.exists():
                row=json.loads(done.read_text());assert sha(video)==row['video_sha256'];assert sha(Path(row['raw']))==row['raw_sha256'];rows.append(row);continue
            payload=z.read(info);assert len(payload)==info.file_size
            if video.exists():assert sha(video)==hashlib.sha256(payload).hexdigest()
            else:video.write_bytes(payload)
            del payload
            source_hash=sha(video)
            cache=O/'word100_rtmw_mediapipe_compare_20261108/runs'/cid/'rtmw_gpu'
            reuse=view=='F' and(cache/'RUN.json').exists()
            if reuse:
                old=json.loads((cache/'RUN.json').read_text());assert old['source_sha256']==source_hash
                rawpath=cache/'rtmw.npz';oldraw=np.load(rawpath,allow_pickle=False)
            else:rawpath=V/(cid+'.npz')
            xy=[];scores=[];ts=[];hashes=[];boxes=[];counts=[];cost=[];native_pts=[]
            start=time.perf_counter()
            with av.open(str(video))as c:
                s=c.streams.video[0];s.codec_context.thread_count=2;base=None
                stream_meta=dict(time_base=str(s.time_base),average_rate=str(s.average_rate),start_time=s.start_time)
                for fid,f in enumerate(c.decode(s)):
                    if base is None:base=f.pts
                    tus=round(float((f.pts-base)*s.time_base)*1e6);im=f.to_ndarray(format='bgr24');hs=hashlib.sha256(im.tobytes()).hexdigest()
                    native_pts.append(f.pts);ts.append(tus);hashes.append(hs)
                    if reuse:
                        assert oldraw['frame_id'][fid]==fid and oldraw['timestamp_us'][fid]==tus and oldraw['frame_sha256'][fid]==hs
                    else:
                        if engine is None:
                            engine=Engine();write(V/'ENGINE.json',dict(devices=engine.devices,compile_seconds=engine.compile_seconds,warm_seconds=engine.warm_seconds))
                        r=engine.infer(im);xy.append(r['xy']);scores.append(r['score']);boxes.append(r['bbox']);counts.append(r['n_people']);cost.append(r['compute_ms'])
                    if fid in [0,32,64,96,126]:cv2.imwrite(str(V/(cid+'_frame%03d.jpg'%fid)),im)
            if reuse:assert len(ts)==len(oldraw['frame_id'])
            else:np.savez_compressed(rawpath,xy_px=np.asarray(xy),score=np.asarray(scores),timestamp_us=np.asarray(ts,np.int64),frame_id=np.arange(len(ts),dtype=np.int64),frame_sha256=np.array(hashes),bbox=np.array(boxes),n_people=np.array(counts),compute_ms=np.array(cost),source_width=np.int64(im.shape[1]),source_height=np.int64(im.shape[0]))
            row=dict(clip_id=cid,view=view,source_member=member,source_crc32=info.CRC,source_member_bytes=info.file_size,
                video=str(video),video_sha256=source_hash,frames=len(ts),timestamps_us=ts,native_pts=native_pts,stream_metadata=stream_meta,
                width=im.shape[1],height=im.shape[0],raw=str(rawpath),raw_sha256=sha(rawpath),reused_existing=reuse,seconds=time.perf_counter()-start)
            write(done,row);rows.append(row)
            print('VIEW_DONE',view,len(ts),round(row['seconds'],3),'reused',reuse,flush=True)
    assert archive.stat().st_size==stat.st_size and archive.stat().st_mtime_ns==stat.st_mtime_ns
    write(V/'EXTRACTION_COMPLETE.json',dict(status='FIVE_VIEW_OBSERVATIONS_COMPLETE',rows=rows,
        same_timestamp_arrays=all(r['timestamps_us']==rows[0]['timestamps_us']for r in rows),
        source_original_unchanged=True,hardware_sync_proven=False,
        raw_scores='RTMW response, not visibility/correctness probability',new_model_download=False))
except Exception as e:
    p=V/'FIRST_EXTRACTION_ERROR.json'
    if not p.exists():write(p,dict(error=repr(e),traceback=traceback.format_exc()))
    raise
