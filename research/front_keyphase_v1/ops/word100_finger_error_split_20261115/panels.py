from common_split import *
import cv2
from PIL import Image,ImageDraw,ImageFont
role=sys.argv[1];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17);out=U/'review'/role;out.mkdir(parents=True,exist_ok=True)
if role=='fit':
 sources=[];catalog=read(U/'SOURCE_CATALOG.json')['found']
 for w in ['WORD2700','WORD2718']:
  for p in [2,5,7]:
   cid=f'NIA_SL_{w}_REAL{p:02d}_F';s=catalog[cid];r=read(s['record']);sources.append(dict(clip_id=cid,word=w,person=p,hand='R',role='fit',path=s['path'],raw_path=r['raw_path'],channel_path=r['channel_path']))
else:sources=[r for r in read(U/'EXTRACTED.json')['rows']if r['role']==role]
records=[]
for s in sources:
 raw=np.load(s['raw_path']);n=len(raw['frame_id']);ids=np.linspace(round(n*.27),min(n-1,round(n*.77)),12).round().astype(int);cap=cv2.VideoCapture(s['path']);base=91 if s['hand']=='L'else 112
 for page in range(2):
  sheet=Image.new('RGB',(1200,1500),'white');dr=ImageDraw.Draw(sheet)
  for local,k in enumerate(range(page*6,page*6+6)):
   f=ids[k];pts=raw['xy_px'][max(0,f-3):min(n,f+4),base:base+21].reshape(-1,2);pts=pts[np.isfinite(pts).all(1)];lo=pts.min(0);hi=pts.max(0);c=(lo+hi)/2;size=max(max(hi-lo)+65,145);box=[int(c[0]-size/2),int(c[1]-size/2),int(c[0]+size/2),int(c[1]+size/2)]
   for col,offset in enumerate([-3,0,3,0]):
    frame=int(np.clip(f+offset,0,n-1));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();assert ok;rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    if col==3:
     d=ImageDraw.Draw(rgb)
     for b,color in [(5,'#2563eb'),(9,'#009999'),(13,'#ed8623'),(17,'#b334be')]:d.line([tuple(p)for p in raw['xy_px'][frame,[base]+list(range(base+b,base+b+4))]],fill=color,width=3)
    crop=rgb.crop(box);crop.thumbnail((290,210));sheet.paste(crop,(col*300+(300-crop.width)//2,local*250+32));dr.text((col*300+4,local*250+3),f'{k}: f{frame} '+('overlay'if col==3 else'RGB'),font=font,fill='#18314a')
   records.append(s|dict(index=k,frame=int(f),neighbor_frames=[max(0,int(f)-3),int(f),min(n-1,int(f)+3)],image=str(out/f'{s["clip_id"]}_{page}.jpg'),source_sha256=sha(s['path']),raw_sha256=sha(s['raw_path']),channel_sha256=sha(s['channel_path'])))
  sheet.save(out/f'{s["clip_id"]}_{page}.jpg',quality=97)
 cap.release()
write(U/f'{role}_REVIEW_INPUTS.json',dict(rows=records,selection='12fixedtimes27%-77%, source +/-3; no candidate model scores'))
