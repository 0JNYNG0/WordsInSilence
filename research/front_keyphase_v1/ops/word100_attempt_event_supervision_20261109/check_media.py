"""Check public RGB overlays/video frames and render tail-review contact sheet."""
from pathlib import Path
import json,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=read(U/'VIEW_DATA.json');m=read(U/'VIEW_MANIFEST.json');checks=[]
def ck(b,s):assert b,s;checks.append(s)
ck(sha(U/'VIEW_DATA.json')==m['data_sha256'],'data frozen')
for k,p in m['media'].items():
 ck(sha(p['path'])==p['sha256'],'video hash '+k);cap=cv2.VideoCapture(p['path']);n=0
 while True:
  ok,im=cap.read()
  if not ok:break
  ck(im.shape[:2]==(p['height'],p['width']),'dimensions');n+=1
 cap.release();ck(n==p['frames'],'every video frame decodes '+k)
reviews={r['clip_id']:r for r in read(O/'word100_gru_semantic_finetune_20261109/REVIEW_INPUTS.json')['rows']};errors=[]
for row in v['rows']:
 ck(len(row['phase'])==len(row['source_indices'])==len(row['points']),'aligned rendered rows')
 if row['kind']!='full_normal':continue
 cid=row['clip'];meta=read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json');z=np.load(meta['raw_path']);xy=z['xy_px'];ids=[0,5,6,7,8,9,10,11,12]+list(range(91,133));l,t,r,b=reviews[cid]['crop'];expect=(xy[:,ids]-[l,t])/[r-l,b-t];points=np.array(row['points']);valid=points[:,:,0]>=0
 err=np.max(np.abs((points-expect)*[r-l,b-t])[valid]);errors.append(float(err));ck(err<.03,'actual joint projection subpixel '+cid)
sources={r['clip_id']:r for r in read(O/'word100_action_feedback_demo_20261108/PREPARED.json')['rows']};review=read(U/'EVENT_REVIEW.json')['rows']
sheet=Image.new('RGB',(900,len(review)*210),'#fff');draw=ImageDraw.Draw(sheet)
for ri,row in enumerate(review):
 cid=row['clip_id'];frames=[row['kept_pre_event_frame'],row['removed_reviewed_anchors'][0],row['tail_source_frames'][-1]];cap=cv2.VideoCapture(sources[cid]['source']);l,t,r,b=reviews[cid]['crop']
 for j,f in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();ck(ok,'review public frame');tile=Image.fromarray(cv2.cvtColor(im[t:b,l:r],cv2.COLOR_BGR2RGB));tile.thumbnail((296,178));sheet.paste(tile,(j*300+(300-tile.width)//2,ri*210+27));draw.text((j*300+4,ri*210+5),f'{cid[7:]} f{f} '+['KEEP','REMOVE','TAIL'][j],fill='#172535')
 cap.release()
sheet.save(U/'TAIL_REVIEW.jpg',quality=90)
(U/'MEDIA_CHECK.json').write_text(json.dumps(dict(status='PASS',checks=len(checks),max_overlay_pixel_error=max(errors),videos=len(m['media']),scope='Actual public RGB and original joints; video frame count; no model rerun',tail_visual_review='Pending same Codex visual inspection of TAIL_REVIEW.jpg'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('PASS',len(checks),'max overlay pixel',max(errors))
