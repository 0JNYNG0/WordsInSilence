"""Small source/ROI integrity view, not identity matching or pose ground truth."""
from common import *
import cv2
from PIL import Image,ImageDraw,ImageFont

def main():
 f=U/'fresh199';rows=read(f/'EVAL_CONTEXT.json')['rows'];selected=[rows[i] for i in [0,30,60]]
 canvas=Image.new('RGB',(1440,3*330),(244,247,252));draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
 details=[]
 for y,row in enumerate(selected):
  r=read(f/'data/records'/f"{row['clip_id']}.json");a=np.load(r['native_capture']['candidate']);n=len(a['frame_id']);cap=cv2.VideoCapture(r['source_file']);box=r['calibration']['box_source_pixels'];side=read(Path(r['native_capture']['candidate']).with_suffix('.sidecar.json')) if False else None
  frame_checks=[]
  for x,frac in enumerate([.25,.5,.75]):
   ix=int((n-1)*frac);frame=int(a['frame_id'][ix]);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=cap.read();assert ok
   h,w=img.shape[:2];assert box[0]==0 and box[1]==0 and box[2]==w
   cv2.rectangle(img,(box[0],box[1]),(box[2]-1,box[3]-1),(0,220,255),4)
   for j,xy in enumerate(a['xy_px'][ix]):
    if not a['observed'][ix,j] or not np.isfinite(xy).all():continue
    color=(240,190,0) if j<33 else ((60,230,90) if j<54 else (255,100,190))
    cv2.circle(img,tuple(int(round(v)) for v in xy),4,color,-1)
   rgb=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));rgb.thumbnail((475,270));canvas.paste(rgb,(x*480,y*330+54))
   draw.text((x*480+6,y*330+4),row['canonical_label']+' / '+row['clip_id'],font=font,fill='#162b43');draw.text((x*480+6,y*330+27),f'frame {frame}, native image + fixed ROI',font=font,fill='#304960')
   frame_checks.append({'frame':frame,'bbox':box,'image_width':w,'image_height':h})
  cap.release();details.append({'clip_id':row['clip_id'],'checks':frame_checks})
 canvas.save(U/'INPUT_VISUAL_CONTACT.png')
 write(U/'INPUT_VISUAL_CONTACT.json',{'status':'CONTACT_CREATED_PENDING_VISUAL_INSPECTION','selection':'Fixed manifest indices0/30/60, quarter/middle/threequarter frames; no score-based choice','clips':details,'purpose':'Upper frame/crop integrity only. No person identity comparison, no model tuning or finger accuracy assertion.'})
 print(str(U/'INPUT_VISUAL_CONTACT.png'))
if __name__=='__main__':main()
