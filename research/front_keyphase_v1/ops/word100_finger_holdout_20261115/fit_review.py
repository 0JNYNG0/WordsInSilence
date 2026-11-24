from holdout_common import *
import cv2
from PIL import Image,ImageDraw,ImageFont
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
out=U/'fit_crops';out.mkdir(exist_ok=True);rows=[]
for word in ['WORD2700','WORD2718']:
 for person in [3,6,9]:
  cid=f'NIA_SL_{word}_REAL{person:02d}_F';rec=read(J/'records'/f'{cid}.json');path=O/'word100_wrist_only_inputs_20261026/sources'/f'{cid}.mp4'
  assert rec['role']=='fit' and sha(path)==rec['source_sha256'];raw=np.load(rec['raw_path']);n=len(raw['frame_id']);ids=np.unique(np.linspace(round(n*.27),min(n-1,round(n*.77)),12).round().astype(int))
  sheet=Image.new('RGB',(1280,900),'white');dr=ImageDraw.Draw(sheet);cap=cv2.VideoCapture(str(path))
  for k,i in enumerate(ids):
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read();assert ok
   pts=raw['xy_px'][max(0,i-2):min(n,i+3),112:133].reshape(-1,2);pts=pts[np.isfinite(pts).all(1)];lo=pts.min(0);hi=pts.max(0);c=(lo+hi)/2;s=max(max(hi-lo)+65,130);h,w=im.shape[:2];box=[max(0,int(c[0]-s/2)),max(0,int(c[1]-s/2)),min(w,int(c[0]+s/2)),min(h,int(c[1]+s/2))]
   rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));crop=rgb.crop(box);crop.thumbnail((310,240));x=(k%4)*320;y=(k//4)*300;sheet.paste(crop,(x+(320-crop.width)//2,y+50));dr.text((x+5,y+3),f'{k}: R f{i}',font=font,fill='#18314a')
   rows.append(dict(clip_id=cid,word=word,person=person,role='fit',side='R',index=k,frame=int(i),box=box,source_video=str(path),source_sha256=sha(path),channels=rec['channel_path'],channels_sha256=rec['channel_sha256']))
  cap.release();sheet.save(out/f'{cid}.jpg',quality=96)
write(U/'FIT_CROP_INPUTS.json',dict(rows=rows,selection='Fixed12times27%-77%; FIT people3,6,9 only; no model predictions used to pick frames'))
