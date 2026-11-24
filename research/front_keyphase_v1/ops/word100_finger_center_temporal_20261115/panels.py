from common import *
import cv2
from PIL import Image,ImageDraw,ImageFont
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17);out=U/'review';out.mkdir(exist_ok=True)
rows=read(U/'READY_SOURCES.json')['rows'];records=[]
for r in rows:
 raw=np.load(r['raw_path']);n=len(raw['frame_id']);ids=np.linspace(round(n*.25),min(n-1,round(n*.78)),16).round().astype(int);cap=cv2.VideoCapture(r['path'])
 sheet=Image.new('RGB',(1400,1200),'white');dr=ImageDraw.Draw(sheet);base=112
 for k,f in enumerate(ids):
  cap.set(1,int(f));ok,im=cap.read();assert ok;p=raw['xy_px'][f,base:base+21];lo=p.min(0);hi=p.max(0);c=(lo+hi)/2;size=max(max(hi-lo)+75,170)
  box=[int(c[0]-size/2),int(c[1]-size/2),int(c[0]+size/2),int(c[1]+size/2)];pic=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).crop(box);pic.thumbnail((340,265));x=k%4*350;y=k//4*300;sheet.paste(pic,(x+(350-pic.width)//2,y+28));dr.text((x+4,y+2),f'{r["person"]:02} {r["word"]} f{f}',font=font,fill='#162b43')
  records.append(dict(clip_id=r['clip_id'],frame=int(f),box=box))
 cap.release();sheet.save(out/(r['clip_id']+'_overview.jpg'),quality=97)
write(U/'REVIEW_FRAMES.json',dict(rows=records,selection='16fixed equally spaced source times, no posture-model outputs'))
