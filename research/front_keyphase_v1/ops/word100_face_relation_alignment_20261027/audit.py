from pathlib import Path
from datetime import datetime
import sys,json,hashlib,shutil
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[4];LAB=ROOT/'research/front_keyphase_v1';U=Path(__file__).resolve().parent
OLD=LAB/'ops/word100_face_body_stage_20261027'
sys.path.insert(0,str(LAB/'src'))
from wis_lab.core.observation import sha256_file
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item()),encoding='utf-8')
F=read(OLD/'FROZEN.json');allbad=[];groups=[]
if not (U/'AUTHORIZATION.json').exists():
 (U/'before').mkdir(parents=True,exist_ok=True)
 for p in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',LAB/'ops/STATE.json'):shutil.copy2(p,U/'before'/p.name)
 write(U/'AUTHORIZATION.json',dict(at=datetime.now().astimezone().isoformat(),
    request='Exclude expressions; compare hands/wrists/arms with facial regions and proceed next public validation',
    scope=['reuse public18','audit45 annotation-discordant frames','separate body-relation diagnostic and alignment candidate','verify and visualize'],
    no_expressions=True,no_training=True,no_user_or_mentor_access=True,no_operating_replacement=True,no_new_downloads=True))
for q in F['rows']:
 if q['signer']==1:continue
 r=next(x for x in F['rows'] if x['word']==q['word'] and x['signer']==1)
 with np.load(q['observation']) as qo,np.load(r['observation']) as ro,np.load(OLD/'data'/q['clip_id']/'cross_paths.npz') as z:
  a,b=z['B_reference'],z['B_query'];bad=[]
  for i,t in enumerate(qo['timestamp_us']):
   if not any(x*1e6<=t<=y*1e6 for x,y in q['intervals']):continue
   near=np.abs(b-i).min();j=int(np.median(a[np.abs(b-i)==near]));rt=ro['timestamp_us'][j]
   if any(x*1e6<=rt<=y*1e6 for x,y in r['intervals']):continue
   d=dict(query=q['clip_id'],reference=r['clip_id'],label=q['label'],query_frame=i,reference_frame=j,
          query_us=int(t),reference_us=int(rt),outside_ref_boundary_ms=min(abs(rt-k*1e6)/1000 for pair in r['intervals'] for k in pair),
          query_any_hand=bool(qo['usable'][i,33:].any()),reference_any_hand=bool(ro['usable'][j,33:].any()),
          query_hand_points=int(qo['usable'][i,33:].sum()),reference_hand_points=int(ro['usable'][j,33:].sum()))
   bad.append(d);allbad.append(d)
  runs=[]
  for row in bad:
   if not runs or row['query_frame']>runs[-1][-1]['query_frame']+1:runs.append([])
   runs[-1].append(row)
  for run in runs:groups.append(dict(id=f'G{len(groups)+1:02}',query=q['clip_id'],reference=r['clip_id'],label=q['label'],rows=run))
assert len(allbad)==45
(U/'images').mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
for g in groups:
 g['image']='images/'+g['id']+'.jpg'
 if (U/g['image']).exists():continue
 picks=sorted(set((0,len(g['rows'])//2,len(g['rows'])-1)))
 canvas=Image.new('RGB',(1260,650),'#132231');draw=ImageDraw.Draw(canvas)
 draw.text((15,10),f"{g['id']} {g['label']} · 주석 경계 불일치 {len(g['rows'])}프레임",font=font,fill='white')
 for col,idx in enumerate(picks):
  pair=g['rows'][idx]
  for side in (0,1):
   key='query' if side==0 else 'reference';cid=g[key];r=next(x for x in F['rows'] if x['clip_id']==cid)
   video=read(OLD/'data'/cid/'DONE.json')['video'];t=pair[key+'_frame'];cap=cv2.VideoCapture(video);cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,bgr=cap.read();assert ok;ts=int(round(cap.get(cv2.CAP_PROP_POS_MSEC)*1000));cap.release();assert abs(ts-pair[key+'_us'])<=1000
   # Display-only central crop, entire head and both arms still included. Input never changes.
   crop=bgr[:,480:1440];im=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));im.thumbnail((400,250));x=col*420+15;y=75+side*285
   canvas.paste(im,(x+(400-im.width)//2,y));draw.text((x,y-28),f"{'비교' if side==0 else '참고'} ID{r['signer']:02} f{t} / {ts/1e6:.3f}초",font=small,fill='#b6d6e9')
 g['image']='images/'+g['id']+'.jpg';canvas.save(U/g['image'],quality=95)
write(U/'BOUNDARY_AUDIT.json',dict(prior_label='45 annotation-discordant frames, not yet 45 visually proven wrong alignments',
    rows=allbad,groups=groups,
    near_boundary_counts={str(ms):sum(x['outside_ref_boundary_ms']<=ms for x in allbad) for ms in (33.334,100,200)},
    missing_hand_count=sum(not x['query_any_hand'] or not x['reference_any_hand'] for x in allbad)))
print(json.dumps(dict(frames=len(allbad),groups=len(groups),distance_counts={str(ms):int(sum(x['outside_ref_boundary_ms']<=ms for x in allbad)) for ms in (33.334,100,200)},missing=sum(not x['query_any_hand'] or not x['reference_any_hand'] for x in allbad)),ensure_ascii=False))
