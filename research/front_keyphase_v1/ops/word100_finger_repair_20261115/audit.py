from common import *
import collections,cv2
from PIL import Image,ImageDraw,ImageFont
rows=read(P/'FINGER_INPUTS.json')['rows'];z=np.load(P/'finger_arrays.npz');truth=z['targets'];mask=(truth>=0)&z['availability'];errors=collections.defaultdict(list)
for seed in [42,43,44]:
 pred=np.load(P/'finger_models_v2'/f'{seed}_predictions.npz')['predictions']
 for i,r in enumerate(rows):
  if r['role']!='development':continue
  for f in np.flatnonzero(mask[i]&(pred[i]>=0)&(pred[i]!=truth[i])):errors[i].append(dict(seed=seed,finger=int(f),truth=int(truth[i,f]),predicted=int(pred[i,f]),baseline=int(z['baseline'][i,f])))
images={}
for name in ['ORDINARY_LABELS.json','EXTERNAL_LABELS.json']:
 for r in read(E/'annotation_review'/name)['rows']:images[r['clip_id'],r['frame']]=r
for r in read(P/'FINGER_RGB_LABELS.json')['rows']:
 if(r['clip_id'],r['frame'])not in images:images[r['clip_id'],r['frame']]=dict(r,source_video=str(E/'annotation_review/media'/f"{r['clip_id']}.mp4"))
out=[];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
for i,err in errors.items():
 r=rows[i];cid=r['clip_id'];rec=read(J/'records'/f'{cid}.json');raw=np.load(rec['raw_path']);frames=raw['frame_id'];ix=int(np.flatnonzero(frames==r['frame'])[0]);src=images.get((cid,r['frame']),{});vp=Path(src.get('source_video',str(E/'annotation_review/media'/f'{cid}.mp4')))
 if not vp.exists():vp=Path(rec.get('video_path',''))
 board=Image.new('RGB',(1350,540),'white');d=ImageDraw.Draw(board);d.text((12,5),f"{cid} {r['side']} f{r['frame']} target:{r['targets']} errors:{err}",font=font,fill='#132b40');cap=cv2.VideoCapture(str(vp));observed=[]
 for k,ti in enumerate([max(0,ix-4),max(0,ix-2),ix,min(len(frames)-1,ix+2),min(len(frames)-1,ix+4)]):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames[ti]));ok,im=cap.read()
  if not ok:continue
  sl=slice(112,133)if r['side']=='R'else slice(91,112);xy=raw['xy_px'][ti,sl];lo=np.maximum(xy.min(0)-55,0).astype(int);hi=np.minimum(xy.max(0)+55,[im.shape[1],im.shape[0]]).astype(int);crop=im[lo[1]:hi[1],lo[0]:hi[0]].copy();pic=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));pic.thumbnail((260,230));board.paste(pic,(k*270,50));overlay=crop.copy();pts=np.round(xy-lo).astype(int)
  for j in range(21):cv2.circle(overlay,tuple(pts[j]),2,(0,210,230),-1)
  for a,b in [(0,1),(0,5),(0,9),(0,13),(0,17)]+[(i,i+1)for s in [1,5,9,13,17]for i in range(s,s+3)]:cv2.line(overlay,tuple(pts[a]),tuple(pts[b]),(10,220,220),1)
  q=Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB));q.thumbnail((260,230));board.paste(q,(k*270,292));d.text((k*270+5,270),f'f{int(frames[ti])}',font=font,fill='#132b40');observed.append(int(frames[ti]))
 cap.release();board.save(U/f'error_{i}.jpg',quality=96);out.append(dict(index=i,**r,errors=err,video=str(vp),video_exists=vp.is_file(),context_frames=observed,image=str(U/f'error_{i}.jpg')))
patterns={}
for name in ['ORDINARY_LABELS.json','EXTERNAL_LABELS.json']:
 rr=read(E/'annotation_review'/name)['rows'];different=[r for r in rr if r['y'][3]>=0 and r['y'][4]>=0 and r['y'][3]!=r['y'][4]]
 patterns[name]=dict(rows=len(rr),different_ring_little=[dict(clip_id=r['clip_id'],frame=r['frame'],y=r['y'],role=r['role'],image=r['image'])for r in different],patterns=[dict(y=list(k),n=v)for k,v in collections.Counter(tuple(r['y'])for r in rr).items()])
write(U/'ERROR_AUDIT_INPUTS.json',dict(rows=out,combination_inventory=patterns))
print(json.dumps(dict(errors=out,combination_inventory=patterns),ensure_ascii=False))
