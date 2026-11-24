from common import *
import cv2, traceback
from PIL import Image, ImageDraw, ImageFont
FINGERS=[(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]
FONT='C:/Windows/Fonts/malgun.ttf'
def runs(mask,t):
 out=[];start=0
 for i in range(1,len(mask)+1):
  if i==len(mask) or mask[i]!=mask[start]:
   out.append(dict(start=float(t[start]),end=float(t[i]) if i<len(t) else float(t[-1]),value=bool(mask[start]),samples=i-start))
   start=i
 return out
def signals(t,xy,ok):
 center=(xy[:,11]+xy[:,12])/2; width=np.linalg.norm(xy[:,11]-xy[:,12],axis=1)
 shoulders=ok[:,11]&ok[:,12]&(width>1)
 result={}; traces={}
 for hand,offset in [('L',33),('R',54)]:
  points=xy[:,offset:offset+21];full=ok[:,offset:offset+21].all(1)&shoulders
  palm=(np.mean(points[:,[0,5,9,13,17]],axis=1)-center)/np.maximum(width[:,None],1)
  bends=[]
  for finger in FINGERS:
   for j in range(1,4):
    a=points[:,finger[j-1]]-points[:,finger[j]]; b=points[:,finger[j+1]]-points[:,finger[j]]
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    full &= den>1e-5
    bends.append(np.sum(a*b,axis=1)/np.maximum(den,1e-5))
  bend=np.stack(bends,1); pos=np.full(len(t),np.nan); shape=pos.copy()
  for i in range(len(t)):
   before=np.flatnonzero((t>=t[i]-.133333)&(t<t[i])); after=np.flatnonzero((t>t[i])&(t<=t[i]+.133333))
   allwin=np.r_[before,i,after]
   if len(before)<2 or len(after)<2 or not full[allwin].all():continue
   pos[i]=np.linalg.norm(np.median(palm[after],0)-np.median(palm[before],0))
   shape[i]=np.mean(np.abs(np.median(bend[after],0)-np.median(bend[before],0)))
  result[hand]=(full,pos,shape)
  traces[hand]=dict(complete=full.tolist(),position_change=[None if not np.isfinite(v) else float(v) for v in pos],bend_change=[None if not np.isfinite(v) else float(v) for v in shape])
 return result,traces
def candidates(t,features,lo,hi):
 queues=[[],[]]
 for hand,(_,pos,shape) in features.items():
  for j,v in enumerate([pos,shape]):
   valid=v[(t>=lo)&(t<=hi)&np.isfinite(v)&(v>1e-8)]
   if len(valid)==0:continue
   threshold=float(np.median(valid))
   for i in range(len(t)):
    if not lo+.30<=t[i]<=hi-.30 or not np.isfinite(v[i]) or v[i]<=threshold:continue
    ids=np.flatnonzero((abs(t-t[i])<=.10)&np.isfinite(v))
    if len(ids) and i==ids[int(np.argmax(v[ids]))]:
     queues[j].append(dict(index=i,time=float(t[i]),hand=hand,kind=['position_change','bend_change'][j],value=float(v[i]),within_clip_median=threshold))
 for q in queues:q.sort(key=lambda x:(-x['value'],x['index'],x['hand']))
 chosen=[]; k=0
 while any(queues) and len(chosen)<5:
  q=queues[k%2];k+=1
  if not q:continue
  c=q.pop(0)
  if all(abs(c['time']-v['time'])>=.30 for v in chosen): chosen.append(c)
 return sorted(chosen,key=lambda c:c['time'])
def make_item(row):
 data=np.load(row['observation']);t=data['timestamp_us']/1e6;xy=data['xy_px'];ok=data['usable'];fid=data['frame_id']
 lo,hi=map(float,row['word_interval']); assert t[0]<=lo<hi<=t[-1]+.04
 features,traces=signals(t,xy,ok); marks=candidates(t,features,lo,hi)
 bounds=[lo]+[m['time'] for m in marks]+[hi]; spans=[]
 for j,(a,b) in enumerate(zip(bounds,bounds[1:])):
  idx=np.flatnonzero((t>=a)&(t<(b if j<len(bounds)-2 else b+.0000001)))
  images=[]
  for tag,sec in [('start',a),('middle',(a+b)/2),('end',max(a,b-.033333))]:
   i=int(np.argmin(abs(t-sec)));images.append(dict(tag=tag,frame_id=int(fid[i]),seconds=float(t[i]),observation_index=i))
  handstats={}
  for hand,off in [('L',33),('R',54)]:
   cnt=ok[idx,off:off+21].sum(1)
   handstats[hand]=dict(full21_frames=int((cnt==21).sum()),partial_frames=int(((cnt>0)&(cnt<21)).sum()),zero_frames=int((cnt==0).sum()),total_frames=len(idx))
  spans.append(dict(id=f'P{j+1}',start=a,end=b,source_frame_indices=idx.tolist(),images=images,observation=handstats,
   status='AUTOMATIC_CANDIDATE_NOT_SEMANTIC_PHASE',required_optional=None,required_hands=None,precedence=None,repeat_policy=None,repeat_range=None,expert_reviewed=False))
 cap=cv2.VideoCapture(row['video_path']);assert cap.isOpened(),row['video_path']
 fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));assert fps>0
 sheet=Image.new('RGB',(960,len(spans)*212+58),'#112133');d=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,17)
 d.text((12,8),f"{row['label']} · {row['word_code']} · 자동 구간 후보 {len(spans)}개 / 의미 미확정",font=font,fill='white')
 d.text((12,32),'원본 전체 화면 · 노란 선: 기존 입력 하단 · 좌우 관절 색상은 저장된 배정',font=ImageFont.truetype(FONT,13),fill='#c7d4e1')
 (U/'frames'/row['word_code']).mkdir(parents=True,exist_ok=True)
 for j,s in enumerate(spans):
  for k,im in enumerate(s['images']):
   cap.set(cv2.CAP_PROP_POS_FRAMES,im['frame_id']);success,bgr=cap.read();assert success,(row['clip_id'],im)
   rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);p=Image.fromarray(rgb);p.thumbnail((640,360))
   draw=ImageDraw.Draw(p);scale=p.width/w;yy=row['roi'][3]*scale
   draw.line([(0,yy),(p.width,yy)],fill='#ffdc6b',width=2)
   i=im['observation_index']
   for off,color in [(33,'#20e4cf'),(54,'#fb7099')]:
    for n in range(off,off+21):
     if ok[i,n]:
      x,y=xy[i,n]*scale;draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
   rel=f"frames/{row['word_code']}/{s['id']}_{im['tag']}.jpg";p.save(U/rel,quality=84);im['image']=rel;im['image_sha256']=sha(U/rel)
   thumb=p.resize((320,180));sheet.paste(thumb,(k*320,j*212+88));d.text((k*320+5,j*212+62),f"{s['id']} {im['seconds']:.2f}s · {['시작','중간','끝'][k]}",font=font,fill='white')
 cap.release();(U/'sheets').mkdir(exist_ok=True);rel=f"sheets/{row['word_code']}.jpg";sheet.save(U/rel,quality=88)
 return dict(**row,video_uri=Path(row['video_path']).as_uri(),video_metadata=dict(fps=fps,frames=frames,width=w,height=h),
  marks=marks,spans=spans,timestamps=t.tolist(),frame_ids=fid.tolist(),traces=traces,
  hand_observation_runs={hand:runs(ok[:,off:off+21].all(1),t) for hand,off in [('L',33),('R',54)]},
  source_note_status='PRIOR_SAMPLED_VISUAL_OBSERVATION_NOT_KSL_SEMANTICS',
  sheet=rel,sheet_sha256=sha(U/rel),new_visual_review=None,expert_reviewed=False,
  repeat_review='UNRESOLVED_ONE_REFERENCE_NOT_VARIANT_POLICY',order_review='OBSERVED_ORDER_ONLY_NOT_REQUIRED_ORDER')
if __name__=='__main__':
 try:
  s=read(U/'SCOPE.json');assert not (U/'CATALOG.json').exists();write(U/'RUNNING.json',dict(at=now(),stage='candidate_bookmarks_and_public_frames'))
  items=[]
  for i,row in enumerate(s['items']):
   path=U/'records'/f"{row['word_code']}.json"
   if path.exists():item=read(path);assert item['source_sha256']==row['source_sha256']
   else:item=make_item(row);write(path,item)
   items.append(item)
   if (i+1)%10==0:print(f'{i+1}/100 candidate clips complete',flush=True)
  write(U/'CATALOG.json',dict(schema='wis_phase_candidate_review_v1',at=now(),scope_sha256=sha(U/'SCOPE.json'),items=items))
  write(U/'BUILD_COMPLETE.json',dict(at=now(),clips=len(items),spans=sum(len(i['spans']) for i in items),marks=sum(len(i['marks']) for i in items),training_runs=0))
  (U/'RUNNING.json').unlink()
 except Exception:
  if not (U/'FIRST_BUILD_ERROR.json').exists():write(U/'FIRST_BUILD_ERROR.json',dict(at=now(),error=traceback.format_exc()))
  raise
