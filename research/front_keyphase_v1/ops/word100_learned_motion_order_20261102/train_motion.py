from pathlib import Path
from collections import Counter
import time,hashlib,json,traceback
import numpy as np,torch
from motion_model import M,P,read,public_arrays,normalize,MotionNet,batch,predict
from label_motion import save,sha

def scores(y,m,p):
 result=[]
 for j in range(6):
  k=m[:,j];t=y[k,j]>.5;q=p[k,j]>=.5;tp=int((t&q).sum());tn=int((~t&~q).sum());fp=int((~t&q).sum());fn=int((t&~q).sum())
  result.append(dict(TP=tp,TN=tn,FP=fp,FN=fn,positive=tp+fn,negative=tn+fp,
   balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)) if (tp+fn)*(tn+fp) else None))
 return result

def main():
 torch.set_num_threads(4);c=read(M/'TRAINING_CONTRACT.json');labels=read(M/'MOTION_TARGETS.json')['rows'];manifest=read(P/'INPUT_MANIFEST.json')['rows']
 arrays={r['clip_id']:public_arrays(r) for r in manifest}
 xx=np.concatenate([np.where(arrays[r['clip_id']][2],arrays[r['clip_id']][1],np.nan) for r in manifest if r['role']=='fit'])
 mean=np.nan_to_num(np.nanmean(xx,0));std=np.nanstd(xx,0);std=np.where(np.isfinite(std)&(std>.01),std,1)
 stats=dict(mean=mean.tolist(),std=std.tolist(),fit_people=c['fit_people'])
 if (M/'NORMALIZATION.json').exists():assert read(M/'NORMALIZATION.json')==stats
 else:save('NORMALIZATION.json',stats)
 seq=[];ys=[];ms=[];records=[]
 def add(r,ix,y,mask,kind):
  a,z,m=arrays[r['clip_id']];seq.append(normalize(z[ix],m[ix],stats));ys.append(y.copy());ms.append(mask.copy())
  records.append(dict(episode=r['id'],role=r['role'],person=r['person'],word_id=r['word_id'],kind=kind,clip_id=r['clip_id'],frame_ids=a['frame_id'][ix].tolist(),timestamps_us=a['timestamp_us'][ix].tolist()))
 for r in labels:
  a,_,_=arrays[r['clip_id']];ix=np.flatnonzero((a['frame_id']>=r['frames'][0])&(a['frame_id']<=r['frames'][1]));y=np.array(r['y'],np.float32);mask=np.array(r['mask'],bool)
  add(r,ix,y,mask,'native_'+r['kind'])
  if r['kind']=='directed':
   rev=y.copy();rev[:5]=0
   if y[2]:rev[4]=1
   if y[4]:rev[2]=1
   add(r,ix[::-1],rev,mask,'feature_reversed_direction')
   q=max(1,int(np.floor(len(ix)*.25)));cut=np.r_[ix[:q],ix[-q:]];cy=np.zeros(6,np.float32);cm=np.zeros(6,bool);cm[5]=True
   add(r,cut,cy,cm,'feature_middle_omitted')
  if r['role']=='fit':
   for speed in [.8,1.25]:
    pick=np.rint(np.linspace(0,len(ix)-1,max(3,round(len(ix)/speed)))).astype(int);add(r,ix[pick],y,mask,f'fit_uniform_speed_{speed}')
 y=np.array(ys);mask=np.array(ms);fit=np.array([i for i,r in enumerate(records) if r['role']=='fit']);dev=np.array([i for i,r in enumerate(records) if r['role']=='development'])
 if not (M/'DATASET.json').exists():save('DATASET.json',dict(rows=records,y=y.tolist(),mask=mask.tolist(),scope='Native episodes plus explicitly marked feature-level controls; not extra people/videos'))
 # Model receives only features, not metadata or raw timestamps. Padding is excluded.
 if not (M/'PREFLIGHT.json').exists():
  torch.manual_seed(313);pool=MotionNet('orderless_pool');ordered=MotionNet('ordered_gru');x,l=batch([seq[0],seq[0][::-1].copy()]);a=pool(x,l).detach().numpy();assert np.allclose(a[0],a[1],atol=1e-6)
  pad=torch.nn.functional.pad(x,(0,0,0,9));assert torch.allclose(ordered(x,l),ordered(pad,l),atol=1e-6)
  assert set(records[i]['person'] for i in fit).isdisjoint(records[i]['person'] for i in dev)
  save('PREFLIGHT.json',dict(status='PASS',native_episodes=len(labels),fit_windows=len(fit),development_windows=len(dev),fit_kinds=dict(Counter(records[i]['kind'] for i in fit)),
   dev_kinds=dict(Counter(records[i]['kind'] for i in dev)),orderless_reversal_invariance=True,padding_ignored=True,target_word_not_input=True,
   labels_sha256=sha(M/'MOTION_TARGETS.json'),contract_sha256=sha(M/'TRAINING_CONTRACT.json'),public_inputs_only=True,old_models_optimizer_parameters=False))
 result=[]
 for cond in c['conditions']:
  for seed in c['seeds']:
   d=M/'runs'/f'{cond}_{seed}'
   if (d/'COMPLETE.json').exists():result.append(read(d/'COMPLETE.json'));continue
   assert not d.exists(),'Do not overwrite partial run';d.mkdir(parents=True)
   torch.manual_seed(seed);rng=np.random.default_rng(seed);model=MotionNet(cond);optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=.01)
   yy=torch.tensor(y);mm=torch.tensor(mask);pos=(yy[fit]*mm[fit]).sum(0);neg=((1-yy[fit])*mm[fit]).sum(0);pw=(neg/pos.clamp_min(1)).clamp(.25,8)
   started=time.perf_counter();history=[]
   for epoch in range(c['epochs']):
    model.train();losses=[]
    for ix in np.array_split(rng.permutation(fit),8):
     x,l=batch([seq[i] for i in ix]);z=model(x,l);loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,yy[ix],pos_weight=pw,reduction='none')*mm[ix]).sum()/mm[ix].sum()
     assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step();losses.append(float(loss.detach()))
    if epoch%30==0 or epoch==119:history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
   model.eval();p=predict(model,seq);torch.save(model.state_dict(),d/'model.pt');np.savez_compressed(d/'predictions.npz',p=p,y=y,mask=mask)
   groups={k:scores(y[ix],mask[ix],p[ix]) for k in sorted({r['kind'] for r in records if r['role']=='development'}) if len(ix:=np.array([i for i in dev if records[i]['kind']==k]))}
   latency=[]
   for _ in range(5):
    t=time.perf_counter();predict(model,[seq[i] for i in dev]);latency.append(time.perf_counter()-t)
   rec=dict(condition=cond,seed=seed,epochs=c['epochs'],training_seconds=time.perf_counter()-started,history=history,
    development=scores(y[dev],mask[dev],p[dev]),development_by_kind=groups,fit=scores(y[fit],mask[fit],p[fit]),
    parameters=sum(v.numel() for v in model.parameters()),median_head_all_dev_windows_seconds=float(np.median(latency)),checkpoint_sha256=sha(d/'model.pt'),
    independent_unseen_final_test=False,new_backbone_finetuning=False)
   with (d/'COMPLETE.json').open('x',encoding='utf8') as f:json.dump(rec,f,ensure_ascii=False,indent=2,allow_nan=False)
   result.append(rec);print(cond,seed,'DEV',rec['development'],flush=True)
 save('TRAINING_RESULT.json',dict(status='SIX_MOTION_HEAD_RUNS_COMPLETE',runs=result,frozen_previous_networks=True,counts_do_not_mean_independent_clips=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  if not (M/'FIRST_FAILURE.txt').exists():(M/'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
  raise
