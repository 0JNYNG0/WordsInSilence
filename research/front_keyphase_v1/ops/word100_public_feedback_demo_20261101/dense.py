"""One inference on new native centers only; old predictions are copied exactly."""
import sys,time,traceback
sys.dont_write_bytecode=True
from common import *

def main():
 assert not (U/'DENSE_STARTED.json').exists(),'Inspect prior execution; do not rerun'
 pins();save('DENSE_STARTED.json',dict(at=time.time(),status='RUNNING',seed=42))
 import torch
 torch.set_num_threads(4)
 sys.path.insert(0,str(S))
 from comparison import WindowHead
 head=WindowHead(4)
 head.load_state_dict(torch.load(S/'runs/protected_shape_split_features_42/head.pt',map_location='cpu',weights_only=True))
 head.eval();head.requires_grad_(False)
 case=read(O/'word100_action_feedback_casebook_20261101/INPUTS.json')
 data=read(S/'DATASET.json'); names=data['selected_targets'];stats=read(S/'NORMALIZATION.json')
 centers={(r['clip_id'],r['center_frame']):i for i,r in enumerate(data['rows'])}
 split=np.load(S/'runs/protected_shape_split_features_42/probabilities.npy',allow_pickle=False)
 old=read(B/'LEARNING_INPUTS.json');oldindex={r['clip_id']:i for i,r in enumerate(old['rows'])}
 native=np.load(B/'runs/pose_plus_signrep_42/probabilities.npy',allow_pickle=False)
 outdir=U/'predictions';outdir.mkdir(exist_ok=False);manifest=[]
 for alias,row in sorted(case['rows'].items()):
  start=time.perf_counter()
  with np.load(row['input'],allow_pickle=False) as z:a={k:z[k].copy() for k in ['x','mask','rgb','frame_id','timestamp_us']}
  x,z=normalize(a,stats);n=len(x);prob=np.full((n,7),np.nan,np.float32);sources=np.zeros(n,np.int8)
  oi=oldindex[row['clip_id']]
  assert old['rows'][oi]['input_sha256']==row['input_sha256']
  for j in range(3):prob[:,j]=native[oi,:n,old['targets'].index(names[j])]
  todo=[]; reused=[]
  for i in range(6,n-6):
   frame=int(a['frame_id'][i]);ci=centers.get((row['clip_id'],frame))
   if ci is not None:
    assert np.array_equal(prob[i,:3],split[ci,:3]);prob[i,3:]=split[ci,3:];sources[i]=1;reused.append(i)
   else:todo.append(i)
  t=time.perf_counter()
  if todo:
   xx=torch.from_numpy(np.stack([x[i-6:i+7] for i in todo]));zz=torch.from_numpy(np.stack([z[i-6:i+7] for i in todo]))
   with torch.inference_mode():pp=torch.sigmoid(head(xx,zz)).numpy()
   prob[todo,3:]=pp;sources[todo]=2
  infer=time.perf_counter()-t
  dest=outdir/f'{alias}.npz';np.savez_compressed(dest,probabilities=prob,new_four_source=sources,frame_id=a['frame_id'],timestamp_us=a['timestamp_us'])
  manifest.append(dict(alias=alias,frames=n,new_centers=len(todo),reused_centers=len(reused),boundary_centers_without_new_four=12,
    inference_seconds=infer,cached_input_total_seconds=time.perf_counter()-start,path=str(dest),sha256=sha(dest),
    new_frame_ids=[int(a['frame_id'][i]) for i in todo],reused_frame_ids=[int(a['frame_id'][i]) for i in reused]))
 save('DENSE_MANIFEST.json',dict(status='COMPLETE',rows=manifest,seed=42,targets=names,
   new_center_predictions=sum(r['new_centers'] for r in manifest),reused_center_predictions=sum(r['reused_centers'] for r in manifest),
   new_training=0,old_three_unchanged=True,completed_center_reruns=0,new_rgb_extractions=0,new_detector_calls=0,
   latency_scope='cached inputs/new four head only; raw video extraction, cold initialization and DTW excluded'))
 print(json.dumps({'new_centers':sum(r['new_centers'] for r in manifest),'reused':sum(r['reused_centers'] for r in manifest),'inference_s':sum(r['inference_seconds'] for r in manifest)}))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'DENSE_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
