from pathlib import Path
import json,hashlib,zipfile
import cv2
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent
rd=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (U/'REVIEW_INPUTS.json').exists()
src=rd(O/'word100_rtmw_joint_learning_20261108/SOURCES.json');sources={r['clip_id']:r for r in src};ex=rd(O/'word100_completion_fullpath_20261110/EXPOSURE_AUDIT.json')['rows']
archives={a:zipfile.ZipFile(a) for a in sorted({r['archive'] for r in src+ex})}
names={a:{Path(n).stem:n for n in z.namelist() if n.endswith('.mp4')} for a,z in archives.items()}
fitsha={r['source_sha256'] for r in src if r['role']=='fit'};out=[]
words=['WORD0003','WORD0500','WORD0678','WORD1139','WORD1290','WORD2551']
for person in [5,14]:
 for word in words:
  clip=f'NIA_SL_{word}_REAL{person:02d}_F';s=sources.get(clip)
  if s:a,n=s['archive'],s['member']
  else:
   matches=[(a,ns[clip]) for a,ns in names.items() if clip in ns];assert len(matches)==1,(clip,matches);a,n=matches[0]
  data=archives[a].read(n);h=hashlib.sha256(data).hexdigest()
  if s:assert h==s['source_sha256']
  if person==14:assert h not in fitsha
  p=U/'public_rgb'/f'{clip}.mp4';p.parent.mkdir(exist_ok=True);p.write_bytes(data)
  cap=cv2.VideoCapture(str(p));num=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=list(range(0,num,3));canvas=Image.new('RGB',(1500,190*((len(frames)+5)//6)),'white');draw=ImageDraw.Draw(canvas)
  for j,f in enumerate(frames):
   cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok;hh,w=bgr.shape[:2];im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop((int(.26*w),int(.03*hh),int(.74*w),int(.88*hh)));im.thumbnail((246,166));x,y=j%6*250,j//6*190;canvas.paste(im,(x+(250-im.width)//2,y+22));draw.text((x+3,y+3),f'{word} ID{person} f{f}',fill='black')
  cap.release();sheet=p.with_suffix('.jpg');canvas.save(sheet,quality=92)
  out.append(dict(clip_id=clip,word_id=word,person=person,role='fit_additional' if person==5 else 'reserved_fine_evaluation',source=str(p),source_sha256=h,sheet=str(sheet),sheet_sha256=sha(sheet),original_frames=num,existing_rtmw=s.get('rtmw_cache') if s else None,archive=a,member=n,exposure='Existing 100word FIT; new manual fine ranges' if person==5 else 'Earlier public TCN evaluation possible; fine motion evaluation reserved before current scores. Record ID only, not independently verified identity.'))
for z in archives.values():z.close()
(U/'REVIEW_INPUTS.json').write_text(json.dumps(dict(rows=out,selection='Six previously included nonfocus expressions, additional FIT05 and score-unseen fine14, fixed before training/scoring',fit_eval_sha_disjoint=True),ensure_ascii=False,indent=2),'utf-8')
print('Prepared12 public originals,6FIT/6fine-evaluation; no model inference')
