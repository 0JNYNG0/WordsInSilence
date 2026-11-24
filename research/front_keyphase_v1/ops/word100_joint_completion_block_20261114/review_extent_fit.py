from pathlib import Path
import json,hashlib
import numpy as np,torch,cv2
from PIL import Image,ImageDraw,ImageFont
from region_extent import RegionExtent,features
from body_core import face_geometry
U=Path(__file__).resolve().parent;S=U.parent/'word100_goal_state_learning_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not(U/'EXTENT_FIT_REVIEW.json').exists()
rev=read(S/'REVIEW_INPUTS.json')['rows'];arr=np.load(S/'source_arrays.npz');face=np.load(U/'face_arrays.npz')
known={(r['clip_id'],r['frame'],r['hand'])for r in read(U/'EXTENT_LABELS.json')['rows']if r['label']}
models=[]
for rec in read(U/'EXTENT_TRAINED.json')['rows']:
 m=RegionExtent();m.load_state_dict(torch.load(rec['path'],weights_only=True));m.eval();models.append(m)
rows=[]
for cid in sorted({r['clip_id']for r in rev if r['split']=='fit'}):
 rr=[r for r in rev if r['clip_id']==cid];lo=min(r['frame']for r in rr);hi=max(r['frame']for r in rr)
 if cid+'_face' in face:fg,fo=face[cid+'_face'],face[cid+'_face_ok']
 else:
  record=read(U.parent/'word100_rtmw_joint_learning_20261108'/'records'/f'{cid}.json')
  assert sha(Path(record['raw_path']))==record['raw_sha256'] and sha(Path(record['channel_path']))==record['channel_sha256']
  raw=np.load(record['raw_path']);ch=np.load(record['channel_path']);fg,fo=face_geometry(raw['xy_px'],ch['point_available'])
 f,ok=features(arr[cid+'_x'],fg,fo,'R')
 with torch.no_grad():p=np.stack([m(torch.tensor(f)).softmax(-1).numpy()for m in models])
 uncertainty=p.max(-1).mean(0);indices=[i for i in range(lo,hi+1)if ok[i]and(cid,i,'R')not in known]
 if not indices:continue
 i=min(indices,key=lambda i:uncertainty[i]);r=rr[0]
 rows.append(dict(clip_id=cid,frame=i,hand='R',source=r['source'],source_sha256=r['source_sha256'],split='fit',reason='lowest certainty within previously reviewed source action extent',model_prediction_withheld_from_sheet=True))
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
for page in range((len(rows)+11)//12):
 im=Image.new('RGB',(1440,1140),'white');d=ImageDraw.Draw(im)
 for k,r in enumerate(rows[page*12:(page+1)*12]):
  assert sha(Path(r['source']))==r['source_sha256'];cap=cv2.VideoCapture(r['source']);cap.set(cv2.CAP_PROP_POS_FRAMES,r['frame']);ok,frame=cap.read();cap.release();assert ok
  h,w=frame.shape[:2];crop=frame[int(h*.03):int(h*.65),int(w*.2):int(w*.8)];pic=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));pic.thumbnail((350,330));x=k%4*360;y=k//4*380
  im.paste(pic,(x+(360-pic.width)//2,y+45));d.text((x+5,y+3),f'{page*12+k}: '+r['clip_id'][7:23]+f' f{r["frame"]}',font=font,fill='#172d49')
 im.save(U/f'FIT_REGION_REVIEW_{page}.jpg',quality=94)
(U/'EXTENT_FIT_REVIEW.json').write_text(json.dumps(dict(rows=rows,fit_only=True,selection_before_annotation=True),ensure_ascii=False,indent=2),'utf-8')
print(len(rows))
