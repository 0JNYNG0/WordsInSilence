from common import *
import av,os,traceback
sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'))
from stream_runtime import Engine
adapter=load('_exp_adapter',O/'word100_rtmw_shared_input_20261108/adapter.py')

def main():
 sources=read(U/'SOURCES_NEW.json');engine=None;rows=[];fail=[]
 for d in ['raw','channels','records']:(U/d).mkdir(exist_ok=True)
 write(U/'EXTRACT_STATUS.json',dict(status='RUNNING',pid=os.getpid(),planned=len(sources['rows'])))
 for r in sources['rows']:
  cid=r['clip_id'];rp=U/'records'/f'{cid}.json'
  if rp.exists():
   old=read(rp);assert sha(old['raw'])==old['raw_sha256'] and sha(old['channels'])==old['channels_sha256'];rows.append(old);continue
  assert sha(r['video'])==r['source_sha256']
  rawp=U/'raw'/f'{cid}.npz';cp=U/'channels'/f'{cid}.npz';started=time.perf_counter()
  try:
   if rawp.exists():
    # A committed raw input may be reused after interruption; never overwrite it.
    z=np.load(rawp,allow_pickle=False);raw={k:z[k].copy() for k in z.files}
   else:
    if engine is None:
     engine=Engine();write(U/'ENGINE.json',dict(devices=engine.devices,compile_seconds=engine.compile_seconds,warm_seconds=engine.warm_seconds))
    xy=[];sc=[];ts=[];cost=[]
    with av.open(r['video']) as c:
     st=c.streams.video[0];st.codec_context.thread_count=2;tbase=None
     for frame in c.decode(st):
      if tbase is None:tbase=frame.pts
      tus=round(float((frame.pts-tbase)*st.time_base)*1e6);im=frame.to_ndarray(format='bgr24');q=engine.infer(im)
      xy.append(q['xy']);sc.append(q['score']);ts.append(tus);cost.append(q['compute_ms'])
    assert len(ts)>0
    raw=dict(xy_px=np.asarray(xy),score=np.asarray(sc),timestamp_us=np.asarray(ts,np.int64),frame_id=np.arange(len(ts),dtype=np.int64),source_width=np.int64(im.shape[1]),source_height=np.int64(im.shape[0]),compute_ms=np.asarray(cost))
    np.savez_compressed(rawp,**raw)
   a=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));np.savez_compressed(cp,**a)
   out=dict(r,raw=str(rawp),raw_sha256=sha(rawp),channels=str(cp),channels_sha256=sha(cp),frames=len(raw['frame_id']),elapsed_seconds=time.perf_counter()-started)
   write(rp,out);rows.append(out);event('extracted',clip_id=cid,done=len(rows),seconds=out['elapsed_seconds'])
  except Exception as exc:
   er=dict(clip_id=cid,error=repr(exc),traceback=traceback.format_exc());fail.append(er);write(U/f'EXTRACT_FAILURE_{time.time_ns()}.json',er);event('extract_failed',clip_id=cid,error=repr(exc))
  write(U/'EXTRACT_STATUS.json',dict(status='RUNNING',pid=os.getpid(),done=len(rows),failed=len(fail),planned=len(sources['rows'])))
 write(U/'INPUTS_NEW.json',dict(rows=rows,failures=fail,planned=len(sources['rows']),complete=True))
 write(U/'EXTRACT_STATUS.json',dict(status='COMPLETE_WITH_FAILURES' if fail else 'COMPLETE',pid=None,done=len(rows),failed=len(fail),frames=sum(r['frames'] for r in rows)))
 event('extraction_complete',clips=len(rows),failures=len(fail))
if __name__=='__main__':main()
