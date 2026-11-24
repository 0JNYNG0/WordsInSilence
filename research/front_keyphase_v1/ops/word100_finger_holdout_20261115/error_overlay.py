from holdout_common import *
import cv2
from PIL import Image,ImageDraw,ImageFont
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18);errors=[r for r in read(U/'diagnostic_old/RESULTS.json')['rows']if r['wrong']]
sheet=Image.new('RGB',(1200,320*len(errors)),'white');dr=ImageDraw.Draw(sheet)
for k,r in enumerate(errors):
 s=next(v for v in read(U/'EXTRACTED.json')['rows']if v['clip_id']==r['clip_id']);raw=np.load(s['raw_path']);cap=cv2.VideoCapture(s['path']);cap.set(cv2.CAP_PROP_POS_FRAMES,r['frame']);ok,im=cap.read();cap.release();assert ok
 box=r['box'];rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));crop=rgb.crop(box);crop.thumbnail((480,270));sheet.paste(crop,(10,k*320+40));over=rgb.copy();d=ImageDraw.Draw(over);start=112
 for f,col in [(5,'#555555'),(9,'#777777'),(13,'#306bdb'),(17,'#f08b25')]:
  pts=raw['xy_px'][r['frame'],[start,start+f,start+f+1,start+f+2,start+f+3]]
  if np.isfinite(pts).all():
   d.line([tuple(p)for p in pts],fill=col,width=3)
   for p in pts:d.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill=col)
 cut=over.crop(box);cut.thumbnail((480,270));sheet.paste(cut,(550,k*320+40));dr.text((5,k*320+5),r['clip_id']+f" f{r['frame']} / blue ring orange little",font=font,fill='#18314a')
sheet.save(U/'ERROR_OVERLAY.jpg',quality=97)
