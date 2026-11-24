from pathlib import Path
import zipfile,hashlib,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from prepare_real_fit import N,read,sha,exclusive,capacity
def main():
 rows=read(N/'HELDOUT_LOCK.json')['rows'];review=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
 for r in rows:
  d=N/'locked_sources';d.mkdir(exist_ok=True);p=d/(r['clip_id']+'.mp4');capacity()
  if not p.exists():
   with zipfile.ZipFile(r['source']['archive']) as z:data=z.read(r['source']['member'])
   assert hashlib.sha256(data).hexdigest()==r['source_sha256']
   with p.open('xb') as f:f.write(data)
  else:assert sha(p)==r['source_sha256']
  c=cv2.VideoCapture(str(p));assert c.isOpened();ids=list(range(20,max(21,r['frames']-16),4));images={};i=0
  while True:
   ok,frame=c.read()
   if not ok:break
   if i in ids:images[i]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
   i+=1
  c.release();assert i==r['frames']
  cols=8;canvas=Image.new('RGB',(cols*220,int(np.ceil(len(ids)/cols))*230),'white');draw=ImageDraw.Draw(canvas)
  for j,fid in enumerate(ids):
   im=Image.fromarray(images[fid]);w,h=im.size;im=im.crop((int(w*.25),0,int(w*.75),int(h*.88)));im.thumbnail((216,200));x=(j%cols)*220;y=(j//cols)*230;canvas.paste(im,(x+(220-im.width)//2,y+25));draw.text((x+3,y+2),f'ID{r["person"]} frame{fid}',font=font,fill='black')
  image=N/'locked_review'/f'{r["clip_id"]}.jpg';image.parent.mkdir(exist_ok=True);assert not image.exists();canvas.save(image,quality=94)
  review.append(dict(**r,local_source=str(p),contact=str(image),contact_sha256=sha(image),frames_shown=ids))
 exclusive(N/'LOCKED_REVIEW_MANIFEST.json',dict(rows=review,predictions_loaded=False,source_description='raw public RGB before candidate inference',not_independent_human_gold=True))
 print('Eight raw public review sheets prepared; no predictions/training')
if __name__=='__main__':main()
