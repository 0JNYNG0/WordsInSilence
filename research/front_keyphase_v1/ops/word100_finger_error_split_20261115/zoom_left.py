from common_split import *
import cv2
from PIL import Image,ImageDraw
s=next(r for r in read(U/'EXTRACTED.json')['rows']if r['clip_id']=='NIA_SL_WORD0071_REAL16_F');raw=np.load(s['raw_path']);cap=cv2.VideoCapture(s['path']);sheet=Image.new('RGB',(1500,1000),'white')
for k,f in enumerate([72,75,78]):
 cap.set(1,f);ok,im=cap.read();im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));pts=raw['xy_px'][f,91:112];lo=pts.min(0)-70;hi=pts.max(0)+70;box=tuple(int(a)for a in [lo[0],lo[1],hi[0],hi[1]])
 for j in [0,1]:
  pic=im.copy()
  if j:
   d=ImageDraw.Draw(pic)
   for b,color in [(5,'#2563eb'),(9,'#009999'),(13,'#ed8623'),(17,'#b334be')]:d.line([tuple(p)for p in raw['xy_px'][f,[91]+list(range(91+b,91+b+4))]],fill=color,width=2)
  pic=pic.crop(box);pic=pic.resize((480,int(pic.height*480/pic.width)));sheet.paste(pic,(k*500,j*500+25));ImageDraw.Draw(sheet).text((k*500,j*500),str(f),fill='black')
sheet.save(U/'review/controls/reserve_left_transition.jpg')
