from pathlib import Path
import sys,json,time,hashlib,importlib.util,traceback
import numpy as np,av,cv2
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
def main():
 assert not (U/'EXTRACTION_COMPLETE.json').exists();rows=[r for r in read(U/'REVIEW_INPUTS.json')['rows'] if r['person']!=3];cv2.setNumThreads(2);engine=None;out=[]
 for r in rows:
  raw=r['existing_rtmw']
  if raw and Path(raw).exists():out.append(dict(r,raw=raw,raw_sha256=sha(raw),reused=True));continue
  if engine is None:
   p=O/'word100_rtmw_stream_reuse_20261108/stream_runtime.py';s=importlib.util.spec_from_file_location('_joint_rtm',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);engine=m.Engine();write(U/'EXTRACTION_ENGINE.json',dict(devices=engine.devices,compile_seconds=engine.compile_seconds,warm_seconds=engine.warm_seconds))
  dest=U/'raw'/f"{r['clip_id']}.npz";dest.parent.mkdir(exist_ok=True);assert not dest.exists();assert sha(r['source'])==r['source_sha256'];start=time.perf_counter();xx=[];sc=[];ts=[];cost=[]
  with av.open(r['source']) as con:
   st=con.streams.video[0];st.thread_type='AUTO';st.codec_context.thread_count=2;origin=None
   for i,f in enumerate(con.decode(st)):
    if origin is None:origin=f.pts
    ts.append(round(float((f.pts-origin)*st.time_base)*1e6));im=f.to_ndarray(format='bgr24');v=engine.infer(im);xx.append(v['xy']);sc.append(v['score']);cost.append(v['compute_ms'])
   width,height=im.shape[1],im.shape[0]
  n=len(xx);np.savez_compressed(dest,xy_px=np.asarray(xx),score=np.asarray(sc),timestamp_us=np.asarray(ts,np.int64),frame_id=np.arange(n),source_width=np.int64(width),source_height=np.int64(height));v=dict(r,raw=str(dest),raw_sha256=sha(dest),frames=n,seconds=time.perf_counter()-start,compute_ms=sum(cost),reused=False);out.append(v);write(dest.with_suffix('.json'),v);print(json.dumps(dict(clip=r['clip_id'],frames=n,seconds=v['seconds'])),flush=True)
 write(U/'EXTRACTION_COMPLETE.json',dict(rows=out,new_detector_calls=engine.calls if engine else 0,training=False,camera=False))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EXTRACTION_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
