from common_split import *
import cv2
from PIL import Image,ImageDraw,ImageFont
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20);catalog=read(U/'SOURCE_CATALOG.json')['found'];records=[]
out=U/'review'/'controls';out.mkdir(exist_ok=True)
for person in [2,5,7]:
 cid=f'NIA_SL_WORD1290_REAL{person:02d}_F';s=catalog[cid];r=read(s['record']);raw=np.load(r['raw_path']);n=len(raw['frame_id']);cap=cv2.VideoCapture(s['path']);sheet=Image.new('RGB',(1200,1500),'white');d=ImageDraw.Draw(sheet)
 for k,(hand,frac)in enumerate([(h,f)for h in ['R','L']for f in [.4,.5,.6]]):
  frame=round(n*frac);base=112 if hand=='R'else 91;pts=raw['xy_px'][max(0,frame-3):frame+4,base:base+21].reshape(-1,2);lo=pts.min(0);hi=pts.max(0);c=(lo+hi)/2;sz=max(max(hi-lo)+65,145);box=(int(c[0]-sz/2),int(c[1]-sz/2),int(c[0]+sz/2),int(c[1]+sz/2))
  for col,off in enumerate([-3,0,3,0]):
   f=frame+off;cap.set(1,f);ok,im=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
   if col==3:
    dr=ImageDraw.Draw(im)
    for b,color in [(5,'#2563eb'),(9,'#009999'),(13,'#ed8623'),(17,'#b334be')]:dr.line([tuple(p)for p in raw['xy_px'][f,[base]+list(range(base+b,base+b+4))]],fill=color,width=3)
   im=im.crop(box);im.thumbnail((290,210));sheet.paste(im,(col*300+(300-im.width)//2,k*250+32));d.text((col*300,k*250),f'{k}:{hand} f{f}',font=font,fill='black')
  records.append(dict(clip_id=cid,person=person,word='WORD1290',role='fit',index=k,frame=frame,hand=hand,path=s['path'],raw_path=r['raw_path'],channel_path=r['channel_path'],source_sha256=sha(s['path']),raw_sha256=sha(r['raw_path']),channel_sha256=sha(r['channel_path']),image=str(out/f'{person}.jpg')))
 sheet.save(out/f'{person}.jpg',quality=97);cap.release()
write(U/'controls_REVIEW_INPUTS.json',dict(rows=records))
# Larger paired views for the reserved little-finger quality decision; no model predictions.
cid='NIA_SL_WORD2700_REAL16_F';s=next(r for r in read(U/'EXTRACTED.json')['rows']if r['clip_id']==cid);raw=np.load(s['raw_path']);cap=cv2.VideoCapture(s['path']);sheet=Image.new('RGB',(1600,1500),'white');d=ImageDraw.Draw(sheet)
for k,f in enumerate([60,65,70]):
 cap.set(1,f);ok,im=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));pts=raw['xy_px'][f,112:133];lo=pts.min(0)-50;hi=pts.max(0)+50;box=tuple(int(a)for a in [lo[0],lo[1],hi[0],hi[1]])
 for col in [0,1]:
  pic=im.copy()
  if col:
   dr=ImageDraw.Draw(pic)
   for b,color in [(5,'#2563eb'),(9,'#009999'),(13,'#ed8623'),(17,'#b334be')]:dr.line([tuple(p)for p in raw['xy_px'][f,[112]+list(range(112+b,112+b+4))]],fill=color,width=2)
  pic=pic.crop(box);pic.thumbnail((770,465));sheet.paste(pic,(col*800,k*500+30));d.text((col*800,k*500),f'f{f} '+('overlay'if col else'RGB'),font=font,fill='black')
sheet.save(out/'reserve_quality_zoom.jpg',quality=98)
