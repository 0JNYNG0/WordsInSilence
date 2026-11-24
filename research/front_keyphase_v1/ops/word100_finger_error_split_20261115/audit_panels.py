from common_split import *
from PIL import Image,ImageDraw,ImageFont
import cv2
catalog=read(U/'SOURCE_CATALOG.json')['found'];rows=read(U/'PRIOR_LOSS_AUDIT.json')['rows'];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17);out=U/'review_loss';out.mkdir(exist_ok=True);records=[];cache={}
for i,r in enumerate(rows):
 cid=r['clip_id'];s=catalog.get(cid)
 if not s:continue
 rec=read(s['record']);raw=np.load(rec['raw_path']);base=91 if r['side']=='L'else 112;f=r['frame'];q=raw['xy_px'][f,base:base+21];lo=q.min(0);hi=q.max(0);c=(lo+hi)/2;size=max(max(hi-lo)+65,145);box=[int(c[0]-size/2),int(c[1]-size/2),int(c[0]+size/2),int(c[1]+size/2)]
 cap=cv2.VideoCapture(s['path']);cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();cap.release();assert ok;rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));over=rgb.copy();d=ImageDraw.Draw(over)
 for b,col in [(5,'#2563eb'),(9,'#009999'),(13,'#ed8623'),(17,'#b334be')]:d.line([tuple(v)for v in raw['xy_px'][f,[base]+list(range(base+b,base+b+4))]],fill=col,width=3)
 cache[i]=(rgb.crop(box),over.crop(box));records.append(dict(**r,audit_index=i,source=s,raw_path=rec['raw_path']))
for off in range(0,len(records),8):
 rr=records[off:off+8];sheet=Image.new('RGB',(1200,1200),'white');d=ImageDraw.Draw(sheet)
 for k,r in enumerate(rr):
  col=k%2;row=k//2;x=col*600;y=row*300;d.text((x+5,y+2),f"#{r['audit_index']} {r['clip_id'].replace('NIA_SL_','')} {r['side']} f{r['frame']} finger{r['finger']}",font=font,fill='#18314a')
  for j,im in enumerate(cache[r['audit_index']]):im.thumbnail((290,257));sheet.paste(im,(x+j*300+(300-im.width)//2,y+35))
 sheet.save(out/f'{off//8}.jpg',quality=96)
write(U/'LOSS_REVIEW_INPUTS.json',dict(rows=records))
