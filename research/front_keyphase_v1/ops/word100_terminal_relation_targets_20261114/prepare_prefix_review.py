"""Fixed fractional source cuts, selected before looking at model predictions."""
from pathlib import Path
import json,hashlib
import numpy as np
import cv2
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;S=U.parent/'word100_goal_state_learning_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
src=read(S/'REVIEW_INPUTS.json')['rows'];arr=np.load(S/'source_arrays.npz');rows=[];frames=[]
for rec in [8,9]:
 rr=sorted([r for r in src if r['person']==rec and r['family']==2],key=lambda r:r['review_slot'])
 r=rr[0];fid=arr[r['clip_id']+'_frame_id'];a=int(np.flatnonzero(fid==rr[0]['frame'])[0]);b=int(np.flatnonzero(fid==rr[-1]['frame'])[0])
 video=Path(r['source']);assert sha(video)==r['source_sha256'];cap=cv2.VideoCapture(str(video))
 for fraction in [0.,.25,.5,.75,1.]:
  i=a+int(round((b-a)*fraction));fr=int(fid[i]);cap.set(cv2.CAP_PROP_POS_FRAMES,fr);ok,im=cap.read();assert ok
  rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));rgb.thumbnail((380,310))
  tile=Image.new('RGB',(384,350),'white');tile.paste(rgb,((384-rgb.width)//2,35));ImageDraw.Draw(tile).text((5,5),f'REAL{rec:02d} frame {fr} / {fraction:.0%}',fill='#17283f')
  frames.append(tile);rows.append(dict(record_id=rec,clip_id=r['clip_id'],fraction=fraction,frame=fr,array_index=i,
    source_start_index=a,source=str(video),source_sha256=r['source_sha256'],label=None))
 cap.release()
canvas=Image.new('RGB',(384*5,350*2),'white')
for i,img in enumerate(frames):canvas.paste(img,((i%5)*384,(i//5)*350))
canvas.save(U/'PREFIX_REVIEW.jpg')
with(U/'PREFIX_REVIEW_FROZEN.json').open('x',encoding='utf-8')as f:json.dump(dict(rows=rows,selection='fixed source fractions, no model scores',shape_only=True,whole_semantic_error_truth=False),f,indent=2)
print(U/'PREFIX_REVIEW.jpg')
