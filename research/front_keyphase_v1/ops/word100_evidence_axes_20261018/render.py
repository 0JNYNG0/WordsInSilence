"""Public RGB review of visible-palm vs inferred-arm boundary combinations."""
import json,sys,zipfile,hashlib
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(O/'word100_hand_evidence_training_20261018'))
from adapter import frame_inputs
from axes import EvidenceAxes
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
c=read(U/'CONTEXT.json');rows={r['clip_id']:r for r in c['rows']};selection=read(U/'ARM_OBSERVATION_REVIEW_SELECTION.json')
p=load(U/'PAIRED_DIAGNOSTICS.npz');extra=[]
for ri,r in enumerate(c['rows']):
 q=load(U/'diagnostics'/f"{r['clip_id']}_sensor.npz");ind=np.flatnonzero(p['clip_index']==ri)[::2];lookup={int(f):i for i,f in enumerate(q['frame_id'])};ids=[lookup[int(f)] for f in p['frame_id'][ind]]
 flag=q['palm5'][ids]&(q['arm_state'][ids]==2)&q['arm_outside_estimate'][ids]
 for ii,side in np.argwhere(flag):extra.append({'view':'sensor','kind':'outside_all_four','clip_id':r['clip_id'],'label':r['label'],'frame_id':int(q['frame_id'][ids[ii]]),'side':int(side),'source':r['source'],'box':r['box']})
assert len(extra)==4
selection=selection[:2]+extra
evidence=U/'evidence';evidence.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
archives={};videos={};results=[];images=[]
try:
 for n,e in enumerate(selection,1):
  r=rows[e['clip_id']];s=e['source'];cid=e['clip_id'];w,h=e['box'][2:]
  if cid not in videos:
   if s['archive'] not in archives:archives[s['archive']]=zipfile.ZipFile(s['archive'])
   z=archives[s['archive']];zi=z.getinfo(s['member']);assert zi.CRC==s['crc32'] and zi.file_size==s['bytes'];b=z.read(s['member']);assert hashlib.sha256(b).hexdigest()==s['sha256']
   dest=evidence/f'{cid}.mp4';assert not dest.exists();dest.write_bytes(b);videos[cid]=dest
  cap=cv2.VideoCapture(str(videos[cid]));cap.set(cv2.CAP_PROP_POS_FRAMES,e['frame_id']);ok,im=cap.read();cap.release();assert ok
  obs,raw,ass=[load(r['views'][e['view']][k]) for k in ('observation','raw','assignment')];idx=int(np.flatnonzero(obs['frame_id']==e['frame_id'])[0]);a=frame_inputs(obs,raw,ass,idx,e['view'],w,h);out=EvidenceAxes().step(*a[:3],int(obs['timestamp_us'][idx]),w,h,assigned=a[3],unassigned_raw=a[4],assignment_evidence_available=e['view']=='sensor')[e['side']]
  im[h:]=(im[h:]*.2).astype(np.uint8);cv2.line(im,(0,h),(w-1,h),(0,220,255),3)
  side=e['side'];color=(100,255,100);xy=obs['xy_px'][idx];mask=obs['usable'][idx]
  for j in range(33+side*21,54+side*21):
   if mask[j] and np.isfinite(xy[j]).all():cv2.circle(im,tuple(xy[j].astype(int)),5,color,-1)
  wr=15+side;pp=raw['pose_xy_px'][idx,wr];cv2.circle(im,tuple(pp.astype(int)),12,(0,200,255),3)
  page=Image.new('RGB',(1120,870),'#152437');d=ImageDraw.Draw(page);d.text((20,15),f"{n:02} {e['label']} / {cid} / 프레임 {e['frame_id']} / {e['view']}",font=font,fill='white')
  page.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((1080,608)),(20,55))
  d.text((20,680),'초록 점: 실제 사용 중인 손 관절 / 노란 원: Pose 손목 추정 / 노란 선: 화면 경계',font=small,fill='white')
  d.text((20,720),f"실제 관측: {out['observation']['usable_point_count']}/21점 (기존 그대로)",font=font,fill='#81e5ad')
  d.text((20,755),f"팔 추정: 경계/밖 · Pose손목 y={pp[1]:.2f}, 경계={h}",font=font,fill='#ffe68c')
  handw=xy[33+side*21];d.text((20,790),f"Hand21 손목 y={handw[1]:.2f} · 팔 추정으로 실제 손 관측을 지우거나 바꾸지 않음",font=small,fill='white')
  target=evidence/f'{n:02}.png';page.save(target);images.append(page)
  results.append({**e,'output':out,'hand21_wrist_xy':handw.tolist(),'image':str(target)})
 for start in range(0,len(images),3):
  canvas=Image.new('RGB',(1120,870*len(images[start:start+3])),'white')
  for j,im in enumerate(images[start:start+3]):canvas.paste(im,(0,870*j))
  canvas.save(evidence/f'contact_{start//3+1}.jpg',quality=90)
 (U/'BOUNDARY_REVIEW.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8');print('Rendered',len(results),'public boundary evidence panels.')
finally:
 for z in archives.values():z.close()
