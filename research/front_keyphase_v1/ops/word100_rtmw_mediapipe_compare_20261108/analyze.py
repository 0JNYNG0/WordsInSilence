from pathlib import Path
import json,sys,hashlib,math
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
F=read(U/'FROZEN.json')
def prepare_review():
 old=read(O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json');lookup={r['clip_id']:r for r in old['rows']};samples=[]
 for row in F['rows']:
  r=lookup.get(row['clip_id'])
  if r:
   assert r['source_sha256']==row['source_sha256']
   for t in r['labels']:
    samples.append(dict(clip_id=row['clip_id'],frame=t['center_frame'],reason='all existing exact-center RGB labels',note=t['primary_note'],targets=t['y'],target_mask=t['target_mask']))
  elif row['word_id']=='WORD0003':
   # Fixed evenly spaced fractions of saved active span; not selected by detector outcome.
   spans=row['intervals']
   if spans:
    a,b=spans[0][0],spans[-1][1]
    for fraction in [.25,.5,.75]:samples.append(dict(clip_id=row['clip_id'],time=a+(b-a)*fraction,reason='fixed fraction of saved main span',note='이번 RGB 관찰; 독립 사람 정답 아님'))
   else:
    for fraction in [.4,.5,.6]:samples.append(dict(clip_id=row['clip_id'],fraction=fraction,reason='fixed 40/50/60 percent of crossing clip; no candidate-dependent selection',note='교차·회전: 전체 영상의 고정 시점'))
 for word,person,frame in [('0185',1,66),('0185',6,20),('0004',2,35),('0949',6,35)]:
  samples.append(dict(clip_id=f'NIA_SL_WORD{word}_REAL{person:02d}_F',frame=frame,reason='historical difficult frame, fixed before this comparison',note='과거 손 누락 사례'))
 save(U/'REVIEW_SELECTION.json',dict(samples=samples,source=str(O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json'),purpose='Qualitative RGB comparison, no threshold training or landmark ground truth; hidden fingertips remain unknown'))
def mapped(r):
 mp=np.load(U/'runs'/r['clip_id']/'mediapipe/observations.npz'); rt=np.load(U/'runs'/r['clip_id']/'rtmw_gpu/rtmw.npz')
 assert np.array_equal(mp['frame_id'],rt['frame_id'])
 assert np.max(np.abs(mp['timestamp_us']-rt['timestamp_us']))<=1
 # Common display: left shoulder,elbow,wrist,right shoulder,elbow,wrist + 21L +21R.
 mi=[11,13,15,12,14,16]+list(range(33,75));ri=[5,7,9,6,8,10]+list(range(91,133))
 a=mp['xy_px'][:,mi]; am=mp['usable'][:,mi];b=rt['xy_px'][:,ri];bs=rt['score'][:,ri]
 return mp,rt,a,am,b,bs
def rough(x,m,scale,main):
 # Shoulder-normalized second position difference. Actual rapid movements also increase it.
 center=(x[:,0]+x[:,3])/2
 rel=(x-center[:,None,:])/scale[:,None,None]
 v=np.linalg.norm(rel[2:]-2*rel[1:-1]+rel[:-2],axis=-1)
 valid=m[2:]&m[1:-1]&m[:-2]&main[2:,None]&main[1:-1,None]&main[:-2,None]
 out=[]
 for j in [6,27,14,35]:
  z=v[:,j][valid[:,j]&np.isfinite(v[:,j])]
  out.append(dict(joint=j,n=len(z),median=float(np.median(z)) if len(z) else None,p95=float(np.quantile(z,.95)) if len(z) else None))
 return out
def basic():
 rows=[];data=[]
 for r in F['rows']:
  mp,rt,a,am,b,bs=mapped(r);T=len(a);secs=mp['timestamp_us']/1e6;main=np.zeros(T,bool)
  for lo,hi in r['intervals'] or []:main|=(secs>=lo)&(secs<=hi)
  scale=np.linalg.norm(a[:,0]-a[:,3],axis=-1);scale[scale<20]=np.nan
  ma=am[:,6:].reshape(T,2,21);mb=(bs[:,6:]>=.3).reshape(T,2,21)
  row={k:r[k] for k in ['clip_id','word_id','label','person']}
  row.update(frames=T,main_frames=int(main.sum()),mp_main_hands_at_least_16=int((ma.sum(2)[main]>=16).sum()),rt_main_hands_at_least_16=int((mb.sum(2)[main]>=16).sum()),main_hand_slots=int(main.sum()*2),mp_main_keypoints=int(ma[main].sum()),rt_main_keypoints=int(mb[main].sum()),main_keypoint_slots=int(main.sum()*42),mp_roughness=rough(a,am,scale,main),rt_roughness=rough(b,bs>=.3,scale,main))
  row.update(mp_all_hand_slots=int((ma.sum(2)>=16).sum()),rt_all_hand_slots=int((mb.sum(2)>=16).sum()),all_hand_slots=T*2)
  row['threshold_sensitivity']={str(t):int(((bs[:,6:]>=t).reshape(T,2,21).sum(2)[main]>=16).sum()) for t in [.2,.3,.5]}
  for method in ['mediapipe','rtmw_gpu']:row[method]=read(U/'runs'/r['clip_id']/method/'RUN.json')
  row['mp_timing']=read(U/'runs'/r['clip_id']/'mediapipe/CAPTURE_TIMING.json')['totals'];rows.append(row)
  def clean(x):return np.where(np.isfinite(x),np.round(x,1),-9999).tolist()
  data.append(dict(**{k:r[k] for k in ['clip_id','label','person','intervals']},width=int(rt['source_width']),height=int(rt['source_height']),ts=secs.tolist(),mp=clean(a),mp_ok=am.tolist(),rt=clean(b),rt_score=np.round(np.nan_to_num(bs,nan=0),3).tolist(),stats=row))
 totals={k:sum(r[k] for r in rows) for k in ['frames','main_frames','mp_main_hands_at_least_16','rt_main_hands_at_least_16','main_hand_slots','mp_main_keypoints','rt_main_keypoints','main_keypoint_slots','mp_all_hand_slots','rt_all_hand_slots','all_hand_slots']}
 totals['sensitivity']={str(t):sum(r['threshold_sensitivity'][str(t)] for r in rows) for t in [.2,.3,.5]}
 totals['zero_person_frames']=sum(r['rtmw_gpu']['zero_person_frames'] for r in rows)
 totals['timing_wall_median']={m:float(np.median([r[m]['wall_seconds'] for r in rows])) for m in ['mediapipe','rtmw_gpu']}
 save(U/'METRICS.json',dict(rows=rows,totals=totals,caveats=['Coverage is not accuracy or human visibility. Both hands counted even when not linguistically active.','Confidence and output schemas differ; .3 is a frozen display threshold, not calibrated equivalence.','Second differences contain real motion and are not measured jitter error.','RTMW GPU timing excludes persistent engine startup; current MP capture includes per-clip model construction and repeated decode.']))
 save(U/'viewer_data.json',dict(clips=data));print(json.dumps(totals,ensure_ascii=False,indent=2))
def sheets():
 import cv2
 from PIL import Image,ImageDraw,ImageFont
 sel=read(U/'REVIEW_SELECTION.json')['samples'];out=U/'visuals';out.mkdir(exist_ok=True)
 font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14)
 edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
 cells=[];entries=[]
 for row in F['rows']:
  samples=[s for s in sel if s['clip_id']==row['clip_id']]
  if not samples:continue
  mp,rt,a,am,b,bs=mapped(row);cap=cv2.VideoCapture(row['source'])
  for s in samples:
   f=s.get('frame',round((len(a)-1)*s['fraction']) if 'fraction' in s else int(np.argmin(abs(mp['timestamp_us']/1e6-s.get('time',0)))))
   cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok
   # Fixed upper-body original view plus union of hand points for detailed common crop. Crop has no inference effect.
   hands=np.concatenate([a[f,6:][am[f,6:]],b[f,6:][bs[f,6:]>=.3]])
   hands=hands[np.isfinite(hands).all(1)]
   if len(hands):
    lo=hands.min(0)-35;hi=hands.max(0)+35;c=(lo+hi)/2;size=max(hi-lo);size=max(size,220);lo=c-size/2;hi=c+size/2
   else:lo=np.array([500,100]);hi=np.array([1400,1000])
   lo=np.maximum(lo,0).astype(int);hi=np.minimum(hi,[bgr.shape[1],bgr.shape[0]]).astype(int)
   panels=[]
   for mode in range(3):
    img=bgr.copy();xy=a[f] if mode==1 else b[f];valid=am[f] if mode==1 else bs[f]>=.3
    if mode:
     for side,offset in enumerate([6,27]):
      color=(235,165,0) if side==0 else (0,95,245)
      for p,q in edges:
       if valid[offset+p] and valid[offset+q]:cv2.line(img,tuple(np.round(xy[offset+p]).astype(int)),tuple(np.round(xy[offset+q]).astype(int)),color,2,cv2.LINE_AA)
      for j in range(offset,offset+21):
       if valid[j] and np.isfinite(xy[j]).all():cv2.circle(img,tuple(np.round(xy[j]).astype(int)),3,color,-1,cv2.LINE_AA)
    img=cv2.cvtColor(img[lo[1]:hi[1],lo[0]:hi[0]],cv2.COLOR_BGR2RGB);panels.append(Image.fromarray(img).resize((320,320)))
   tile=Image.new('RGB',(960,365),'white');dr=ImageDraw.Draw(tile)
   for i,img in enumerate(panels):tile.paste(img,(320*i,45));dr.text((320*i+8,24),['원본','MediaPipe 현재','RTMW-l GPU'][i],font=small,fill='#253043')
   name=f'{row["clip_id"]}_f{f:04d}';dr.text((8,2),f'{row["label"]} · 기록ID {row["person"]:02d} · f{f} / {f/30:.2f}초',font=font,fill='#162134')
   tile.save(out/(name+'.jpg'),quality=94);cells.append(tile)
   entries.append({**s,'frame':f,'image':'visuals/'+name+'.jpg','crop':[int(x) for x in np.r_[lo,hi]]})
  cap.release()
 for i in range(0,len(cells),3):
  sheet=Image.new('RGB',(960,365*len(cells[i:i+3])),'#eef1f5')
  for j,c in enumerate(cells[i:i+3]):sheet.paste(c,(0,j*365))
  sheet.save(out/f'sheet_{i//3:02d}.jpg',quality=92)
 save(U/'VISUAL_INDEX.json',entries);print('samples',len(entries))
if __name__=='__main__':
 if '--prepare' in sys.argv:prepare_review()
 elif '--sheets' in sys.argv:sheets()
 else:basic()
