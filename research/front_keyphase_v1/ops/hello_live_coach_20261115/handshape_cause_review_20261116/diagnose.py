"""Read-only model diagnosis on explicitly public sources. No camera/user inputs."""
from pathlib import Path
import sys,json,hashlib,collections
import numpy as np
import torch,cv2
V=Path(__file__).resolve().parent;U=V.parent;O=U.parent
sys.path[:0]=[str(O/'basic_demo_feedback_20261115'),str(O/'word100_rtmw_shared_input_20261108')]
from shape_consensus import ShapeConsensus,combine
import adapter
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def load(p):
 with np.load(p)as z:return {k:z[k].copy()for k in z.files}
def channel(z):return adapter.convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']))['tcn280']
def inspect(net,x,h):
 f=net.finger.infer(x,h);a,ap=net.before.sequence(x,h);b,bp=net.expanded.sequence(x,h);s,base=combine(f['states'],a,b)
 return dict(states=s,baseline=base,fingers=f['states'],available=f['available'],before=a,expanded=b,old_probs=ap,new_probs=bp,**{'finger_'+k:v for k,v in f['probabilities'].items()})
def main():
 torch.set_num_threads(2);net=ShapeConsensus();out=[]
 paths={'native':U/'source/reference.npz','letterbox':U/'seated_diagnosis_20261116/public_video_640_letterbox.npz','upper':U/'seated_diagnosis_20261116/public_video_upper_body_640.npz','heldout_view':U/'seated_diagnosis_20261116/heldout_crop.npz'}
 plan=dict(task='Split observation vs classification vs actual transition; do not infer truth from target word.',public_only=True,private_training=False,new_training=False,threshold_search=False,paths={k:str(v)for k,v in paths.items()},
  design='First inspect predictions and source RGB; annotate observation quality separately. Any repair is generic and must preserve known errors, normal classes and missing-data abstention. Existing reference variants are diagnostic, not unseen-user validation.')
 if not(V/'PLAN.json').exists():write(V/'PLAN.json',plan)
 for name,path in paths.items():
  raw=load(path);x=channel(raw);arr={}
  for h in ['L','R']:
   q=inspect(net,x,h);arr.update({h+'_'+k:v for k,v in q.items()})
   ids=np.r_[np.arange(36,44)]if h=='R'else np.array([],int)
   for label,ids,expected in [('sweep',ids,0),('fists',np.arange(54,76),1)]:
    if not len(ids):continue
    cc=collections.Counter(q['states'][ids].tolist())
    bad=ids[q['states'][ids]!=expected]
    out.append(dict(case=name,hand=h,segment=label,expected_from_prior_rgb_review=expected,counts=dict(cc),nonmatching_frames=bad.tolist(),
     details=[dict(frame=int(i),state=int(q['states'][i]),fingers=q['fingers'][i].tolist(),available=q['available'][i].tolist(),before=int(q['before'][i]),expanded=int(q['expanded'][i]))for i in bad]))
  np.savez_compressed(V/(name+'_predictions.npz'),**arr)
  # Center-of-received-window contract vs full-sequence local computation.
  mismatch=0
  for i in [36,40,42,51,54,60,66,75]:
   z={k:(v[i-3:i+4]if k in ['xy_px','score','timestamp_us','frame_id']else v)for k,v in raw.items()};xx=channel(z)
   for h in ['L','R']:
    s=net.sequence(xx,h)
    mismatch+=int(s['states'][3]!=arr[h+'_states'][i])
  out.append(dict(case=name,stream_batch_difference=mismatch,compared=16))
 write(V/'DIAGNOSIS.json',out);print(json.dumps(out,ensure_ascii=False),flush=True)
 # Original pixels and extracted skeleton for visual review; no predicted text.
 cap=cv2.VideoCapture(str(O.parent/'runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4'))
 for name in ['native','letterbox','upper']:
  raw=load(paths[name]);tiles=[]
  for fid in [36,42,48,51,54,60,66,75]:
   cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,img=cap.read();assert ok
   if name=='letterbox':im=np.zeros((480,640,3),np.uint8);im[60:420]=cv2.resize(img,(640,360));img=im
   if name=='upper':img=cv2.resize(img[60:780,480:1440],(640,480))
   for hand,start in [('R',112),('L',91)]:
    p=raw['xy_px'][fid,start:start+21];good=np.isfinite(p).all(1);pp=p[good];lo=np.floor(pp.min(0)-.25*np.ptp(pp,axis=0)-8).astype(int);hi=np.ceil(pp.max(0)+.25*np.ptp(pp,axis=0)+8).astype(int);lo=np.maximum(lo,0);hi=np.minimum(hi,np.array([img.shape[1],img.shape[0]]))
    crop=img[lo[1]:hi[1],lo[0]:hi[0]].copy();over=crop.copy();scale=240/max(crop.shape[:2]);crop=cv2.resize(crop,None,fx=scale,fy=scale);over=cv2.resize(over,None,fx=scale,fy=scale)
    for k in [1,5,9,13,17]:
     chain=[0]+list(range(k,k+4))
     for a,b in zip(chain,chain[1:]):
      if good[a]and good[b]:cv2.line(over,tuple(np.rint((p[a]-lo)*scale).astype(int)),tuple(np.rint((p[b]-lo)*scale).astype(int)),(0,210,240),1,cv2.LINE_AA)
    tile=np.full((280,500,3),255,np.uint8);tile[30:30+crop.shape[0],:crop.shape[1]]=crop;tile[30:30+over.shape[0],250:250+over.shape[1]]=over;cv2.putText(tile,f'{name} frame {fid} {hand}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,30,45),1);tiles.append(tile)
  sheet=np.vstack([np.hstack(tiles[i:i+4])for i in range(0,len(tiles),4)]);cv2.imwrite(str(V/(name+'_review.jpg')),sheet)
 cap.release()
if __name__=='__main__':main()
