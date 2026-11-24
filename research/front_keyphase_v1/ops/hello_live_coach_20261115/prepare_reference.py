"""Decode approved public SEN0354 once; no camera, private media or training."""
from pathlib import Path
import json,sys,time,hashlib
import numpy as np
import cv2
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent;D=U/'source'
sys.dont_write_bytecode=True
sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'))
from stream_runtime import Engine
REFERENCE=O.parent/'runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4'

def main():
 D.mkdir(exist_ok=True)
 p=D/'reference.npz'
 if p.exists():print('Existing public observation retained');return
 engine=Engine();c=cv2.VideoCapture(str(REFERENCE));fps=c.get(cv2.CAP_PROP_FPS)
 xy=[];scores=[];ts=[];cost=[];i=0;t=time.perf_counter();selected={38:'sweep_start',44:'sweep_end',53:'fists_start',69:'fists_repeat'}
 while True:
  ok,bgr=c.read()
  if not ok:break
  h,w=bgr.shape[:2];r=engine.infer(bgr)
  xy.append(r['xy']);scores.append(r['score']);ts.append(round(i/fps*1e6));cost.append(r['compute_ms'])
  if i in selected:
   crop=bgr[int(h*.07):int(h*.81),int(w*.27):int(w*.73)]
   cv2.imwrite(str(D/(selected[i]+'.jpg')),crop)
  i+=1
 c.release();np.savez_compressed(p,xy_px=np.array(xy),score=np.array(scores),timestamp_us=np.array(ts,np.int64),frame_id=np.arange(i),source_width=w,source_height=h)
 m=dict(source=str(REFERENCE),sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),fps=fps,frames=i,width=w,height=h,raw=str(p),raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),extraction_seconds=time.perf_counter()-t,infer_ms=cost,devices=engine.devices,compile_seconds=engine.compile_seconds,warm_seconds=engine.warm_seconds,public_only=True,training=False,review='Native RGB source inspected. Single reference feasibility, not independent generalization.',stages=[dict(id='sweep',label='편 손으로 팔을 따라 손목 쪽으로',rgb_interval=[36,45],image='sweep_start.jpg'),dict(id='fists',label='두 주먹을 나란히 위아래로',rgb_interval=[51,78],image='fists_start.jpg')])
 (D/'REFERENCE.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in m.items()if k not in['infer_ms']},ensure_ascii=False))
if __name__=='__main__':main()
