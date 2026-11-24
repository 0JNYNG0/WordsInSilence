from quality_common import *
import cv2
from PIL import Image,ImageDraw,ImageFont
role=sys.argv[1];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17);out=U/'review'/role;out.mkdir(parents=True,exist_ok=True);sources=[]
if role in ['fit','fit_left','fit_more','fit_front']:
 for word in (['WORD2700','WORD2718']if role in ['fit','fit_more']else ['WORD0949']if role=='fit_front'else['WORD0071']):
  for person in ([1,4,8]if role=='fit_more'else[3,6,9]):
   cid=f'NIA_SL_{word}_REAL{person:02d}_F';r=read(J/'records'/f'{cid}.json');path=O/'word100_wrist_only_inputs_20261026/sources'/f'{cid}.mp4'
   if not path.exists():path=O/'word100_label_only_audit_20261018/videos'/f'{cid}.mp4'
   if not path.exists():path=O/'word100_role_dtw_20261017/public_videos'/f'{cid}.mp4'
   assert sha(path)==r['source_sha256'];sources.append(dict(clip_id=cid,path=str(path),raw_path=r['raw_path'],channel_path=r['channel_path'],role='fit',hand='L'if role=='fit_left'else'R',person=person,word=word))
else:
 for r in read(U/'EXTRACTED.json')['rows']:
  if r['role']==role:sources.append(dict(**r,hand='L'if r['word']=='WORD0071'else'R'))
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
     for b,color in [(5,'#2563eb'),(9,'#009999'),(13,'#ed8623'),(17,'#b334be')]:
      q=raw['xy_px'][frame,[base]+list(range(base+b,base+b+4))];d.line([tuple(p)for p in q],fill=color,width=3)
      for p in q:d.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill=color)
    crop=rgb.crop(box);crop.thumbnail((290,210));sheet.paste(crop,(col*300+(300-crop.width)//2,local*250+32));dr.text((col*300+4,local*250+3),f'{k}: f{frame} '+('overlay'if col==3 else'RGB'),font=font,fill='#18314a')
   records.append(s|dict(index=k,frame=int(f),neighbor_frames=[max(0,int(f)-3),int(f),min(n-1,int(f)+3)],box=box,image=str(out/f'{s["clip_id"]}_{page}.jpg'),source_sha256=sha(s['path']),raw_sha256=sha(s['raw_path']),channel_sha256=sha(s['channel_path'])))
  sheet.save(out/f'{s["clip_id"]}_{page}.jpg',quality=97)
 cap.release()
write(U/f'{role}_REVIEW_INPUTS.json',dict(rows=records,selection='fixed12times27%-77%, neighbor +/-3frames; RGB+center RTMW; no quality/phase model predictions'))
