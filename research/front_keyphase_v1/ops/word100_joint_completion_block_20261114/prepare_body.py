from pathlib import Path
import json,hashlib,collections
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from body_core import features,face_geometry,CLASSES
U=Path(__file__).resolve().parent;O=U.parent;T=O/'word100_terminal_relation_targets_20261114';S=O/'word100_goal_state_learning_20261114';RT=O/'word100_rtmw_joint_learning_20261108'
read=lambda p:json.loads(p.read_text('utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):
 with(U/n).open('x',encoding='utf-8')as f:json.dump(d,f,ensure_ascii=False,indent=2)
old=read(T/'LABELS.json')['rows'];rows=[dict(r,origin='original_direct_annotation')for r in old if r['domain']!='shape'];review=read(S/'REVIEW_INPUTS.json')['rows'];arr=np.load(S/'source_arrays.npz');added=[]
# New labels use only the FIT RGB sheets actually inspected. Word IDs below
# address annotation records; no word or stage ID enters features/model/runtime.
for r in review:
 if r['split']!='fit':continue
 if r['word_id']=='WORD1133':
  added.append(dict(clip_id=r['clip_id'],record_id=r['person'],split='fit',frame=r['frame'],domain='forearm',hand='R',label='across',target=0,origin='new_direct_RGB_across_arm'))
 if r['word_id']=='WORD2523'and r['review_slot']==0:
  for hand in ['R','L']:added.append(dict(clip_id=r['clip_id'],record_id=r['person'],split='fit',frame=r['frame'],domain='forearm',hand=hand,label='upright',target=1,origin='new_direct_RGB_raised_forearm'))
 # Two previously unlabelled, clearly collar/upper-chest FIT observations.
 if(r['person'],r['word_id'],r['family'],r['review_slot'])in [(3,'WORD0739',2,1),(5,'WORD0739',2,1)]:
  added.append(dict(clip_id=r['clip_id'],record_id=r['person'],split='fit',frame=r['frame'],domain='region',hand='R',label='upper_body',target=1,origin='new_direct_RGB_upper_torso'))
for a in added:
 match=next((r for r in rows if(r['clip_id'],r['frame'],r['domain'],r['hand'])==(a['clip_id'],a['frame'],a['domain'],a['hand'])),None)
 if match:assert match['label']is None;rows.remove(match)
 rows.append(a)
cache={};pins={str(T/'LABELS.json'):sha(T/'LABELS.json')};sourceout={};dims={}
for cid in sorted({r['clip_id']for r in rows}):
 rec=read(RT/'records'/f'{cid}.json');rawpath=Path(rec['raw_path']);assert sha(rawpath)==rec['raw_sha256'];raw=np.load(rawpath)
 chpath=Path(rec['channel_path']);assert sha(chpath)==rec['channel_sha256'];ch=np.load(chpath)
 assert np.array_equal(raw['frame_id'],ch['frame_id'])and np.array_equal(arr[cid+'_frame_id'],raw['frame_id'])
 fg,ok=face_geometry(raw['xy_px'],ch['point_available']);cache[cid]=dict(x=arr[cid+'_x'],face=fg,face_ok=ok,frame_id=raw['frame_id'],ts=raw['timestamp_us'])
 sourceout[cid+'_face']=fg;sourceout[cid+'_face_ok']=ok;pins[str(rawpath)]=rec['raw_sha256'];pins[str(chpath)]=rec['channel_sha256']
for i,r in enumerate(rows):
 c=cache[r['clip_id']];index=int(np.flatnonzero(c['frame_id']==r['frame'])[0]);f,ok=features(c['x'],r['domain'],r['hand'],c['face'],c['face_ok']);dims[r['domain']]=f.shape[1]
 r.update(index=i,array_index=index,feature=f[index].tolist(),available=bool(ok[index]));r['target']=CLASSES[r['domain']].index(r['label'])if r['label']else-1
np.savez_compressed(U/'face_arrays.npz',**sourceout)
write('BODY_LABELS.json',dict(rows=rows,added=added,dimensions=dims,pins=pins,unknown_labels_excluded_from_loss=True,probe_labels_unchanged=True,
 review='FIT sheets02-07 inspected, 90 RGB panels. Source08/09 reviewed for diagnosis only, no new labels.',
 limits='Broad face-front vs upper torso/shoulder, across-arm vs raised forearm. Not exact face contact, fine facial parts, depth or sign correctness.'))
write('BODY_FROZEN.json',dict(seeds=[42,43,44],steps=1000,lr=.005,weight_decay=.01,confidence=.8,
 architecture='Body-normalized region including cached face jaw/bounds; upper-arm plus forearm geometry. Two small MLP24 heads. No word/time/expected-order input.',
 fit_records=list(range(2,8)),probe_records=[8,9],no_checkpoint_selection=True,
 gate='All original known probe body states correct without abstention; FIT label audit and normal/short/partial controls required before use. Original shape models immutable.',
 dims=dims,pins={str(p):sha(p)for p in [U/'body_core.py',U/'BODY_LABELS.json',U/'face_arrays.npz',Path(__file__)]}))
# Show that saved facial observations are real source-frame coordinates.
chosen=[(2,62),(2,72),(3,74),(8,66)];sheet=Image.new('RGB',(1000,660),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
for j,(record,fid)in enumerate(chosen):
 cid=f'NIA_SL_WORD0739_REAL{record:02}_F';src=next(r for r in review if r['clip_id']==cid);assert sha(Path(src['source']))==src['source_sha256']
 cap=cv2.VideoCapture(src['source']);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();cap.release();assert ok
 rr=read(RT/'records'/f'{cid}.json');xy=np.load(rr['raw_path'])['xy_px'][fid];h,w=im.shape[:2]
 # Draw only saved jaw and coarse anchors, no fabricated corrected points.
 for a,b in zip(range(23,39),range(24,40)):cv2.line(im,tuple(xy[a].round().astype(int)),tuple(xy[b].round().astype(int)),(130,100,30),2)
 for k in [0,5,6,31,112,121]:cv2.circle(im,tuple(xy[k].round().astype(int)),5,(20,100,230),-1)
 crop=im[int(h*.04):int(h*.63),int(w*.22):int(w*.78)];pic=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));pic.thumbnail((490,290));xx=j%2*500;yy=j//2*330
 sheet.paste(pic,(xx+(500-pic.width)//2,yy+35));draw.text((xx+10,yy+6),f'기록 {record:02} / 원본 프레임 {fid} / 저장 RTMW',font=font,fill='#172d49')
sheet.save(U/'FACE_SOURCE_CHECK.jpg',quality=92)
print(json.dumps(dict(labels=len(rows),added=len(added),dims=dims,known={s:{d:sum(r['split']==s and r['domain']==d and r['target']>=0 for r in rows)for d in CLASSES}for s in ['fit','exposed_probe']})))
