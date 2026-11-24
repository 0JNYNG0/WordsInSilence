from pathlib import Path
import json,hashlib,importlib.util,time,sys
import numpy as np, av,cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 text=json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item()if isinstance(v,np.generic)else v.tolist())
 with Path(p).open('x',encoding='utf-8')as f:f.write(text)
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 cfg=read(U/'FROZEN.json');assert not(U/'EXTRACTED.json').exists()
 for p,h in cfg['model_pins'].items():assert sha(p)==h
 S=load('g4stream',O/'word100_rtmw_stream_reuse_20261108/stream_runtime.py');A=load('g4adapter',O/'word100_rtmw_shared_input_20261108/adapter.py')
 (U/'raw').mkdir(exist_ok=True);(U/'review').mkdir(exist_ok=True)
 engine=S.Engine();rows=[];write(U/'ENGINE.json',dict(devices=engine.devices,compile_seconds=engine.compile_seconds,warm_seconds=engine.warm_seconds))
 for r in cfg['sources']:
  p=U/'raw'/(r['clip_id']+'.npz');rec=p.with_suffix('.json')
  if rec.exists():
   old=read(rec);assert sha(p)==old['raw_sha256'];rows.append(old);continue
  assert not p.exists();assert sha(r['path'])==r['sha256']
  xy=[];score=[];ts=[];hashes=[];ims=[];cost=[];t=time.perf_counter()
  with av.open(r['path'])as con:
   st=con.streams.video[0];st.thread_type='AUTO';st.codec_context.thread_count=2;base=None
   for frame in con.decode(st):
    if base is None:base=frame.pts
    im=frame.to_ndarray(format='bgr24');v=engine.infer(im)
    xy.append(v['xy']);score.append(v['score']);ts.append(round(float((frame.pts-base)*st.time_base)*1e6));hashes.append(hashlib.sha256(im.tobytes()).hexdigest());cost.append(v['compute_ms']);ims.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
  fid=np.arange(len(xy),dtype=np.int64);ts=np.array(ts,np.int64);h,w=ims[0].shape[:2]
  np.savez_compressed(p,xy_px=np.array(xy),score=np.array(score),timestamp_us=ts,frame_id=fid,source_width=w,source_height=h,frame_sha256=np.array(hashes))
  ch=A.load(p);cp=U/'raw'/(r['clip_id']+'_channels.npz');np.savez_compressed(cp,**ch)
  # Even source-time samples, before any feedback prediction. No model-selected frames.
  ids=np.unique(np.linspace(0,len(ims)-1,24).round().astype(int));sheet=Image.new('RGB',(1500,4*270),'white');dr=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
  for k,i in enumerate(ids):
   image=Image.fromarray(ims[i]);image.thumbnail((246,240));x=(k%6)*250;y=(k//6)*270;sheet.paste(image,(x+(250-image.width)//2,y));dr.text((x+4,y+242),f'f{i}  {ts[i]/1e6:.2f}s',font=font,fill='#18283f')
  sheet.save(U/'review'/(r['clip_id']+'.jpg'),quality=93)
  result=dict(**r,frames=len(fid),seconds=time.perf_counter()-t,infer_ms_total=sum(cost),raw_path=str(p),raw_sha256=sha(p),channel_path=str(cp),channel_sha256=sha(cp),review_frames=ids.tolist())
  write(rec,result);rows.append(result);print(r['clip_id'],len(fid),'frames',round(result['seconds'],2),'seconds',flush=True)
 write(U/'EXTRACTED.json',dict(rows=rows,total_frames=sum(r['frames']for r in rows),camera=False,training=False))
if __name__=='__main__':main()
