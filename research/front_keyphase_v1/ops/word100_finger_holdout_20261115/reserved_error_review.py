from holdout_common import *
import cv2
from PIL import Image,ImageDraw,ImageFont
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
cid='NIA_SL_WORD2718_REAL18_F';s=next(r for r in read(U/'EXTRACTED.json')['rows']if r['clip_id']==cid);raw=np.load(s['raw_path']);cap=cv2.VideoCapture(s['path']);sheet=Image.new('RGB',(1500,650),'white');dr=ImageDraw.Draw(sheet)
for k,f in enumerate([31,33,35,37,39]):
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok;rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));pts=raw['xy_px'][f,112:133];lo=pts.min(0)-55;hi=pts.max(0)+55;box=[int(v)for v in [lo[0],lo[1],hi[0],hi[1]]];cut=rgb.crop(box);cut.thumbnail((290,270));sheet.paste(cut,(k*300,40));d=ImageDraw.Draw(rgb)
 for base,col in [(5,'#f08b25'),(9,'#306bdb'),(13,'#444444'),(17,'#777777')]:d.line([tuple(v)for v in raw['xy_px'][f,[112]+list(range(112+base,116+base))]],fill=col,width=3)
 cut=rgb.crop(box);cut.thumbnail((290,270));sheet.paste(cut,(k*300,360));dr.text((k*300+5,5),f'f{f} / index orange',font=font,fill='#18314a')
cap.release();sheet.save(U/'RESERVED_ERROR.jpg',quality=97)
