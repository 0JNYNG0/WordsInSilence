from pathlib import Path
import sys,os,json,hashlib,time,traceback,queue,threading
import numpy as np
U=Path(__file__).resolve().parent; OPS=U.parent; OLD=OPS/'word100_signrep_feasibility_20261029'
sys.path.insert(0,str(OLD));from common import norm,starts,sha,read
sys.path.insert(0,str(OPS/'word100_guide_video_feedback_20261028'))
from motion_observer import read_observation,extract_features
def save(n,x):
 p=U/n
 if p.exists(): raise FileExistsError(p)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def compile_gpu():
 import openvino as ov
 core=ov.Core();ir=core.read_model(OLD/'converted/signrep_features_fp32.xml');ir.reshape([1,3,16,224,224])
 c=core.compile_model(ir,'GPU',{'INFERENCE_PRECISION_HINT':'f16','PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':'1'})
 assert all('GPU' in v for v in c.get_property('EXECUTION_DEVICES'))
 c([np.zeros((1,3,16,224,224),np.float32)])
 return c
def encode(source,box,n,comp):
 import av,cv2
 from collections import deque
 seq=starts(n);ends={s+15:s for s in seq};work=queue.Queue(4);ys=[];errors=[];stopped=threading.Event()
 def consume():
  try:
   while True:
    item=work.get()
    if item is None:break
    i,a=item;ys.append((i,np.array(comp([a])[0],copy=True)))
  except BaseException:errors.append(traceback.format_exc())
  finally:stopped.set()
 def put(x):
  while True:
   if stopped.is_set():raise RuntimeError(errors or 'Unexpected worker exit')
   try:work.put(x,timeout=.5);return
   except queue.Full:pass
 t=time.perf_counter();worker=threading.Thread(target=consume,daemon=True);worker.start();buf=deque(maxlen=16);ts=[]
 with av.open(str(source)) as c:
  for i,f in enumerate(c.decode(video=0)):
   im=f.to_ndarray(format='rgb24');x0,y0,x1,y1=box
   # Fixed source coordinates; no changing crop, no detector on cropped pixels.
   im[:y0]=[124,116,104];im[y1:]=[124,116,104];im[:,:x0]=[124,116,104];im[:,x1:]=[124,116,104]
   im=cv2.resize(im,(224,224),interpolation=cv2.INTER_LINEAR)
   buf.append(norm(im[None])[0]);ts.append(round(float(f.pts*f.time_base)*1e6))
   if i in ends:put((ends[i],np.ascontiguousarray(np.stack(buf).transpose(1,0,2,3)[None])))
 put(None);worker.join(120)
 if worker.is_alive() or errors:raise RuntimeError(errors or 'timeout')
 assert len(ts)==n and [s for s,_ in ys]==seq
 features=np.concatenate([y for _,y in ys]).astype(np.float32)
 return features,np.asarray(ts,np.int64),seq,time.perf_counter()-t
def build_targets(alias,frame_ids,labels,targets):
 lookup={int(f):i for i,f in enumerate(frame_ids)};values={};conflicts=[];hard_unknown=set()
 for r in labels['static']:
  if r['alias']!=alias:continue
  for k,v in r['labels'].items():
   if v is not None:values.setdefault((r['frame_id'],k),[]).append(v)
 # Recheck downgrades cannot be silently reinstated by a duplicate source label.
 for r in read(U/'LABEL_REVIEW.json')['uncertain_downgrades']:
  if r['alias']==alias:hard_unknown.add((r['frame_id'],r['feature']))
 for s in labels['sequences']:
  if s['source_alias']!=alias:continue
  for r in s['frames']:
   for k,v in r['direct'].items():
    k=s['side']+'_'+k
    if k in targets and v is not None:values.setdefault((r['frame_id'],k),[]).append(v)
 y=np.zeros((len(frame_ids),len(targets)),np.float32);m=np.zeros_like(y,bool)
 for (f,k),vs in values.items():
  assert f in lookup
  if len(set(vs))>1:conflicts.append(dict(frame=f,feature=k,values=vs));continue
  if (f,k) in hard_unknown:continue
  y[lookup[f],targets.index(k)]=vs[0];m[lookup[f],targets.index(k)]=True
 return y,m,conflicts
def main():
 assert read(U/'LABEL_REVIEW.json')['status']=='PASS_PROVISIONAL'
 out=U/'inputs';out.mkdir(exist_ok=False)
 frozen=read(OPS/'word100_shared_feature_bundle_20261027/FROZEN.json')['rows']
 idx={r['alias']:r for r in read(OPS/'word100_shared_feature_bundle_20261027/visual_inputs/INDEX.json')['rows']}
 labels=read(U/'LABELS.json');targets=read(U/'FROZEN.json')['features']
 save('INPUT_FREEZE.json',dict(pins={str(p):sha(p) for p in [U/'FROZEN.json',U/'LABELS.json',U/'LABEL_REVIEW.json',Path(__file__),OLD/'assets/ckpt.pt',OLD/'converted/signrep_features_fp32.xml',OLD/'converted/signrep_features_fp32.bin']},expected_clips=36))
 comp=compile_gpu();rows=[];hashes=[]
 for rr in frozen:
  a=rr['alias'];r=idx[a];assert sha(rr['record'])==rr['record_sha256']
  source=r['source'];h=sha(source);assert h==r['source_sha256'];assert h not in hashes;hashes.append(h)
  record=read(rr['record']);box=record['calibration']['box_source_pixels'];obspath=record['views']['native']['observation']
  obs=read_observation(obspath);xy=obs['xy_px'];x0,y0,x1,y1=box
  inside=(xy[:,:,0]>=x0)&(xy[:,:,0]<x1)&(xy[:,:,1]>=y0)&(xy[:,:,1]<y1)
  obs['usable'] &= inside
  feat=extract_features(obs);x=np.nan_to_num(feat['x']).astype(np.float32);m=feat['mask']
  rgb,ts,seq,secs=encode(source,box,len(x),comp)
  assert np.max(np.abs(ts-obs['timestamp_us']))<=1
  mid=(ts[np.array(seq)]+ts[np.array(seq)+15])/2
  z=np.stack([np.interp(ts,mid,rgb[:,j]) for j in range(768)],axis=1).astype(np.float32)
  y,ym,conflicts=build_targets(a,obs['frame_id'],labels,targets)
  dest=out/f'{a}.npz';np.savez_compressed(dest,x=x,mask=m,rgb=z,window_features=rgb,window_starts=seq,window_center_us=mid,frame_id=obs['frame_id'],timestamp_us=ts,y=y,target_mask=ym)
  row=dict(alias=a,clip_id=r['clip_id'],word_id=r['word_id'],label=r['label'],person=r['recorded_person_id'],role=r['role'],source=source,source_sha256=h,record=rr['record'],record_sha256=rr['record_sha256'],observation=obspath,observation_sha256=sha(obspath),box=box,frames=len(x),windows=len(seq),known_target_cells=int(ym.sum()),conflicts=conflicts,input=str(dest),input_sha256=sha(dest),rgb_seconds=secs)
  rows.append(row);print(json.dumps(dict(alias=a,frames=len(x),cells=int(ym.sum()),seconds=round(secs,3)),ensure_ascii=False),flush=True)
 save('INPUTS.json',dict(status='COMPLETE',rows=rows,feature_names=feat['feature_names'],targets=targets,pins=read(U/'INPUT_FREEZE.json')['pins'],note='Linear interpolation of 16-frame representation, not reconstruction of labels; same full-frame timestamps; post-capture future context'))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'PREP_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
