from pathlib import Path
import json,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw
N=Path(__file__).resolve().parent;O=N.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (N/'REVIEW_INPUTS.json').exists()
sources=read(O/'word100_action_feedback_demo_20261108/PREPARED.json')['rows']
old={r['clip_id']:r for r in read(O/'word100_attempt_event_supervision_20261109/INPUTS.json')['rows'] if r['kind']=='full_normal'}
rows=[]
for r in sources:
 if r['role']!='fit':continue
 cid=r['clip_id'];src=Path(r['source']);assert src.exists() and sha(src)==r['source_sha256'];a,b=old[cid]['main'];frames=np.unique(np.rint(np.linspace(max(0,a-6),b+9,16)).astype(int)).tolist()
 cap=cv2.VideoCapture(str(src));sheet=Image.new('RGB',(1600,1120),'white');draw=ImageDraw.Draw(sheet)
 for k,f in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok,(cid,f);h,w=im.shape[:2]
  # A fixed broad torso crop for visibility review only; no coordinate extraction.
  im=im[int(h*.03):int(h*.85),int(w*.15):int(w*.85)];tile=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));tile.thumbnail((398,254));x=(k%4)*400;y=(k//4)*280;sheet.paste(tile,(x+(400-tile.width)//2,y+24));draw.text((x+5,y+5),f'{cid} | frame {f}',fill='black')
 cap.release();p=N/'review'/f'{cid}.jpg';p.parent.mkdir(exist_ok=True);sheet.save(p,quality=90)
 rows.append(dict(clip_id=cid,person=r['person'],word=r['word'],source=str(src),source_sha256=r['source_sha256'],frames=frames,sheet=str(p),sheet_sha256=sha(p),prior_main=[a,b]))
write(N/'REVIEW_INPUTS.json',dict(rows=rows,scope='Public FIT only; contact sheets are coarse samples, not every original frame. No model inference or training.',private=False))
print('PUBLIC_FIT_REVIEW_SHEETS',len(rows))
