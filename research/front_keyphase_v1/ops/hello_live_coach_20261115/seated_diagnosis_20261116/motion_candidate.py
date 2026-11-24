"""One fixed public-only motion-head experiment. Never opens personal records."""
from pathlib import Path
import sys,copy,json,importlib.util,hashlib
import numpy as np
import torch
V=Path(__file__).resolve().parent;U=V.parent;F=U/'action_integration/finetune'
sys.path.insert(0,str(U))
from action_runtime import ActionObserver,CHECKPOINT_SHA256
spec=importlib.util.spec_from_file_location('public_motion_old_experiment',F/'experiment.py'); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def main():
 assert not (V/'MOTION_CANDIDATE.pt').exists()
 plan=dict(seed=42,steps=400,lr=.001,batch={'old_public_fit':32,'original_hello_fit':16,'public_view_wait_variants':32},
     training='copy of already admitted motion_head only',frozen='all encoder, handshape, normalization, criteria and thresholds',
     labels=str(F/'SOURCE_LABELS.json'),views=['native','640_letterbox','upper_body_640'],waits=['none','16sec_real_start_pose_before_repeat'],
     provenance='Same nine previously RGB-reviewed full direction/hold intervals. Resizing/cropping and waiting change observation conditions, not action labels. No automatically labeled subspans.',
     evaluation='Historical public DEV32 normal intervals, prior24 positive/negative conditions, held-out crop view; all same-person variants remain development evidence, not independent user accuracy.',
     forbidden_inputs='No DIAGNOSTIC_ONLY/AGGREGATE/private video/coordinates or webcam. No private tuning or selection.',threshold=.65)
 old.write(V/'MOTION_FROZEN_PLAN.json',plan)
 torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);torch.use_deterministic_algorithms(True)
 observer=ActionObserver();net=observer.net;rc=observer.rc;adapter=observer.adapter
 before_hash=hashlib.sha256((F/'CANDIDATE.pt').read_bytes()).hexdigest()
 labels=old.read(F/'SOURCE_LABELS.json'); features=np.load(F/'FEATURES.npz');index=old.read(F/'FEATURE_INDEX.json')
 new=[]
 for name,path in [('native',U/'source/reference.npz'),('letterbox',V/'public_video_640_letterbox.npz'),('upper_body',V/'public_video_upper_body_640.npz')]:
  with np.load(path) as z:raw={k:z[k].copy()for k in z.files}
  for wait in [False,True]:
   a={k:v.copy()for k,v in raw.items()}
   if wait:
    ix=np.r_[np.arange(52),np.repeat(51,480),np.arange(52,len(raw['timestamp_us']))]
    a['xy_px']=a['xy_px'][ix];a['score']=a['score'][ix];a['frame_id']=np.arange(len(ix));a['timestamp_us']=np.rint(a['frame_id']*1e6/30).astype(np.int64)
   for item in labels['intervals']:
    if wait and item['axis']!=0:continue
    for stride in [1,2]:
     start=item['start_frame']+(480 if wait else 0);end=item['end_frame']+(480 if wait else 0)
     keep=np.arange(0,len(a['frame_id']),stride);inside=np.flatnonzero((keep>=start)&(keep<=end))
     ai,bi=int(inside[0]),int(inside[-1]);selected=keep[:bi+4]
     # Exactly the same bounded received history as deployed ActionObserver.
     selected=selected[-320:];selected=selected[a['timestamp_us'][selected]>=a['timestamp_us'][selected[-1]]-12_000_000]
     aa=int(np.flatnonzero(selected==keep[ai])[0]);bb=int(np.flatnonzero(selected==keep[bi])[0])
     ch,x,h=old.prefix(rc,net,adapter,a,selected)
     for reverse in [False,True]:
      fn=old.reverse_geom if reverse else old.geometry3
      g,eligible,e=fn(rc,x,ch['timestamp_us'],aa,bb,item['axis'],item['moving'],item['anchor'])
      if not eligible:continue
      target=2-item['target'] if reverse else item['target']
      meta=dict(id=item['id'],view=name,wait=wait,stride=stride,reverse=reverse,target=target,eligible=True,evidence=e,
                source_start_frame=item['start_frame'],source_end_frame=item['end_frame'],independent=False)
      new.append((old.head_features(net,h,g,bb if reverse else aa,aa if reverse else bb),meta))
 oldfit=[(f,m)for f,m in zip(features['old_fit'],index['old_fit'])if m['eligible']]
 hello=list(zip(features['supplemental_fit'],index['supplemental_fit'])); dev=list(zip(features['old_dev'],index['old_dev']))
 sets=[oldfit,hello,new]; x=[old.tensor(s)for s in sets];ys=[torch.tensor([m['target']for _,m in s],dtype=torch.long)for s in sets]
 candidate=copy.deepcopy(net.motion_head);candidate.requires_grad_(True);candidate.train();opt=torch.optim.Adam(candidate.parameters(),lr=.001);rng=np.random.default_rng(42)
 def choose(y,n):
  cls=np.resize(np.arange(3),n);rng.shuffle(cls);return np.array([rng.choice(np.flatnonzero(y.numpy()==k))for k in cls])
 for step in range(400):
  ids=[choose(y,n)for y,n in zip(ys,[32,16,32])];xx=torch.cat([v[i]for v,i in zip(x,ids)]); yy=torch.cat([y[i]for y,i in zip(ys,ids)])
  loss=torch.nn.functional.cross_entropy(candidate(xx),yy);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(candidate.parameters(),2.);opt.step()
 candidate.eval(); results={}
 with torch.inference_mode():
  for name,rows in [('old_fit',oldfit),('old_dev',dev),('original_hello',hello),('variants_fit',new)]:
   f=old.tensor(rows);results[name]=old.summarize(rows,net.motion_head(f).softmax(-1).numpy(),candidate(f).softmax(-1).numpy())
 torch.save(dict(schema='public_motion_head_only_v1',state=candidate.state_dict(),base_sha256=CHECKPOINT_SHA256,seed=42,steps=400,training_private_data=False),V/'MOTION_CANDIDATE.pt')
 old.write(V/'MOTION_RESULTS.json',dict(results=results,source_index=[m for _,m in new],training_private_data=False,production_applied=False,old_admitted_hash_unchanged=hashlib.sha256((F/'CANDIDATE.pt').read_bytes()).hexdigest()==before_hash))
 print(json.dumps({k:{'normal':v['native'],'all':{q:v['all'][q]for q in ['total','before','after']}}for k,v in results.items()},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
