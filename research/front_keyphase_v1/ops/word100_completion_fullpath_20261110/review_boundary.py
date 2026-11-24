from common import *
import cv2
from PIL import Image,ImageDraw
for r in read(F/'FRESH_REVIEW.json')['rows']:
 if r['word'] not in ['WORD0739','WORD2523']:continue
 frames=list(range(60,76)) if r['word']=='WORD0739' else list(range(58,74))
 cap=cv2.VideoCapture(r['source']);sheet=Image.new('RGB',(1600,1120),'white');d=ImageDraw.Draw(sheet)
 for k,f in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
  h,w=im.shape[:2];im=im[int(.03*h):int(.80*h),int(.22*w):int(.78*w)];tile=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));tile.thumbnail((398,254));x=k%4*400;y=k//4*280;sheet.paste(tile,(x+(400-tile.width)//2,y+24));d.text((x+5,y+5),str(f),fill='black')
 cap.release();sheet.save(F/'fresh_review'/(r['clip_id']+'_boundary.jpg'),quality=92)
