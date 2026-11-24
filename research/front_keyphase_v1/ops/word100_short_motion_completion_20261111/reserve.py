from pathlib import Path
import json,hashlib,zipfile
import cv2
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent
def rd(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (U/'RESERVED_PUBLIC.json').exists();sources={r['clip_id']:r for r in rd(O/'word100_rtmw_joint_learning_20261108/SOURCES.json')};out=[]
for word in ['WORD0003','WORD0500','WORD0678','WORD1139','WORD1290','WORD2551']:
 clip=f'NIA_SL_{word}_REAL13_F';r=sources[clip];p=U/'public_rgb'/f'{clip}.mp4';p.parent.mkdir(exist_ok=True)
 with zipfile.ZipFile(r['archive']) as z:data=z.read(r['member'])
 assert hashlib.sha256(data).hexdigest()==r['source_sha256'];p.write_bytes(data);cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=list(range(0,n,3)) if word!='WORD1290' else list(range(30,min(n,110)));canvas=Image.new('RGB',(1500,190*((len(frames)+5)//6)),'white');draw=ImageDraw.Draw(canvas)
 for j,f in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok;h,w=bgr.shape[:2];im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop((int(.26*w),int(.03*h),int(.74*w),int(.88*h)));im.thumbnail((246,166));x,y=j%6*250,j//6*190;canvas.paste(im,(x+(250-im.width)//2,y+22));draw.text((x+3,y+3),f'{word} ID13 f{f}',fill='black')
 cap.release();sheet=p.with_suffix('.jpg');canvas.save(sheet,quality=92);out.append(dict(clip_id=clip,word_id=word,person=13,source=str(p),source_sha256=r['source_sha256'],sheet=str(sheet),sheet_sha256=sha(sheet),original_frames=n,exposure='Common TCN/development previously exposed; new fine labels reserved before current motion-model scores.'))
(U/'RESERVED_PUBLIC.json').write_text(json.dumps(dict(rows=out),ensure_ascii=False,indent=2),'utf-8');print('Reserved six public ID13 originals')
