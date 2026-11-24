"""Prediction-blind contact sheets for fixed existing public DEV videos."""
import sys,json
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
from prepare import U,read,sha,save
def main():
 assert not (U/'REVIEW_CONTACTS.json').exists()
 records=[]
 for r in read(U/'INPUT_MANIFEST.json')['rows']:
  if r['role']!='development':continue
  assert sha(Path(r['source']))==r['source_sha256']
  ob=np.load(r['observation'],allow_pickle=False)
  xy=ob['xy_px'] if 'xy_px' in ob.files else None
  cap=cv2.VideoCapture(r['source']);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=cap.get(cv2.CAP_PROP_FPS)
  if xy is not None:
   pts=xy[:,[11,12,13,14,15,16],:2].reshape(-1,2);pts=pts[np.isfinite(pts).all(1)]
   x0=max(0,int(pts[:,0].min()-100));x1=min(w,int(pts[:,0].max()+100))
  else:x0,x1=int(w*.25),int(w*.75)
  y0,y1=0,min(h,int(r['box'][3])+80)
  desired=list(range(0,r['frames'],3));chosen={};i=0
  while True:
   ok,bgr=cap.read()
   if not ok:break
   if i in desired:chosen[i]=cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)
   i+=1
  cap.release();assert len(chosen)==len(desired)
  tw,th,cols=180,190,8
  sheet=Image.new('RGB',(cols*tw,35+((len(desired)+cols-1)//cols)*th),'white');draw=ImageDraw.Draw(sheet)
  draw.text((8,8),r['clip_id']+'  each 3 original frames; no model predictions',fill='black')
  for j,fid in enumerate(desired):
   im=Image.fromarray(chosen[fid]);im.thumbnail((tw-4,th-23))
   x=(j%cols)*tw;y=35+(j//cols)*th;sheet.paste(im,(x+(tw-im.width)//2,y))
   draw.text((x+3,y+th-20),f'f{fid:03d} {fid/fps:.2f}s',fill='black')
  p=U/'review_contacts'/f"{r['clip_id']}.jpg";p.parent.mkdir(exist_ok=True);sheet.save(p,quality=94)
  records.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],path=str(p),frame_ids=desired,source_sha256=r['source_sha256'],crop=[x0,y0,x1,y1]))
 save('REVIEW_CONTACTS.json',dict(rows=records,model_predictions_used=False,role='existing public development',human_expert=False))
 print('Prepared',len(records),'public-only blind sheets')
if __name__=='__main__':main()
