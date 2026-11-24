from pathlib import Path
import json,hashlib
import numpy as np
import cv2
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;S=U.parent/'word100_goal_state_learning_20261114';T=U/'transition_shape';T.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text('utf-8-sig'))
src=read(S/'REVIEW_INPUTS.json')['rows'];arr=np.load(S/'source_arrays.npz');rows=[];tiles=[]
for rec in range(2,10):
 rr=sorted([r for r in src if r['person']==rec and r['family']==2],key=lambda r:r['review_slot']);r=rr[0]
 fid=arr[r['clip_id']+'_frame_id'];a=int(np.flatnonzero(fid==rr[0]['frame'])[0]);b=int(np.flatnonzero(fid==rr[-1]['frame'])[0]);cap=cv2.VideoCapture(r['source'])
 for fraction in [.125,.25,.375]:
  i=a+int(round((b-a)*fraction));fr=int(fid[i]);cap.set(cv2.CAP_PROP_POS_FRAMES,fr);ok,im=cap.read();assert ok
  h,w=im.shape[:2];im=im[int(h*.07):int(h*.64),int(w*.27):int(w*.67)]
  rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));rgb.thumbnail((420,350))
  tile=Image.new('RGB',(424,385),'white');tile.paste(rgb,((424-rgb.width)//2,30));ImageDraw.Draw(tile).text((8,6),f'REAL{rec:02d} f{fr} {fraction:.1%}',fill='#15283d');tiles.append(tile)
  rows.append(dict(clip_id=r['clip_id'],record_id=rec,split='fit'if rec<=7 else'exposed_probe',frame=fr,
   array_index=i,fraction=fraction,source=r['source'],source_sha256=r['source_sha256'],label=None))
 cap.release()
for start in [0,12]:
 canvas=Image.new('RGB',(424*3,385*4),'white')
 for j,tile in enumerate(tiles[start:start+12]):canvas.paste(tile,((j%3)*424,(j//3)*385))
 canvas.save(T/f'review_{start//12+1}.jpg')
with(T/'REVIEW_INPUTS.json').open('x',encoding='utf-8')as f:json.dump(dict(rows=rows,selection='Fixed 12.5/25/37.5 percent source interval cuts, no model output input'),f,indent=2)
