from common import *
import zipfile,cv2,numpy as np
from PIL import Image,ImageDraw
assert not (F/'FRESH_REVIEW.json').exists();rows=[]
for r in read(F/'EXPOSURE_AUDIT.json')['rows']:
 if not r['eligible_untouched_candidate']:continue
 with zipfile.ZipFile(r['archive']) as z:data=z.read(r['member'])
 assert hashlib.sha256(data).hexdigest()==r['source_sha256'];p=F/'fresh_sources'/(r['clip_id']+'.mp4');p.parent.mkdir(exist_ok=True);p.write_bytes(data)
 cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=np.rint(np.linspace(0,n-1,20)).astype(int).tolist();sheet=Image.new('RGB',(1600,1400),'white');draw=ImageDraw.Draw(sheet)
 for k,f in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok;h,w=im.shape[:2];im=im[int(h*.03):int(h*.85),int(w*.15):int(w*.85)];tile=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));tile.thumbnail((398,254));x=k%4*400;y=k//4*280;sheet.paste(tile,(x+(400-tile.width)//2,y+24));draw.text((x+5,y+5),f"{r['clip_id']} | {f}",fill='black')
 cap.release();dest=F/'fresh_review'/(r['clip_id']+'.jpg');dest.parent.mkdir(exist_ok=True);sheet.save(dest,quality=90);rows.append(dict(r,source=str(p),frames=n,review_frames=frames,sheet=str(dest),sheet_sha256=sha(dest)))
write(F/'FRESH_REVIEW.json',dict(rows=rows,scope='Recorded ID10; pre-output RGB review. Eligibility pending lineage search.',training=False));print('FRESH_REVIEW',len(rows))
