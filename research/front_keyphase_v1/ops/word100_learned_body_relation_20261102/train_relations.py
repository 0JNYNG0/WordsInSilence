from pathlib import Path
import json,hashlib,time,traceback
import numpy as np,torch
from relation_model import V,RelationNet,read,normalized,dense
P=V.parent/'word100_learned_guide_feedback_20261102'
def save(p,x):
 p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(y,p,m):
 out=[]
 for j in range(3):
  k=m[:,j];g=y[k,j]>.5;q=p[k,j]>=.5;pos=int(g.sum());neg=int((~g).sum())
  out.append(dict(positive=pos,negative=neg,TP=int((q&g).sum()),TN=int((~q&~g).sum()),FP=int((q&~g).sum()),FN=int((~q&g).sum()),
   balanced_accuracy=float(.5*(q[g].mean()+(~q[~g]).mean()))))
 return dict(targets=out,macro_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in out])))
def main():
 torch.set_num_threads(4);contract=read(V/'TRAINING_CONTRACT.json');labels=read(V/'RELATION_TARGETS.json')['rows']
 manifest=read(P/'INPUT_MANIFEST.json')['rows'];sources={r['clip_id']:r for r in manifest};arrays={k:dict(np.load(r['input'])) for k,r in sources.items()}
 fit=[r for r in manifest if r['role']=='fit'];X=np.concatenate([np.where(arrays[r['clip_id']]['mask'],arrays[r['clip_id']]['x'],np.nan) for r in fit]);Z=np.concatenate([arrays[r['clip_id']]['rgb'] for r in fit])
 pm=np.nan_to_num(np.nanmean(X,0));ps=np.nanstd(X,0);ps=np.where(np.isfinite(ps)&(ps>1e-4),ps,1)
 stats=dict(pm=pm.tolist(),ps=ps.tolist(),zm=Z.mean(0).tolist(),zs=np.maximum(Z.std(0),1e-4).tolist(),fit_people=list(range(1,10)))
 if (V/'NORMALIZATION.json').exists():assert stats==read(V/'NORMALIZATION.json')
 else:save(V/'NORMALIZATION.json',stats)
 cache={k:normalized(a,stats) for k,a in arrays.items()};inputs=[];rgbs=[];ys=[];ms=[]
 for r in labels:
  a=arrays[r['clip_id']];i=int(np.flatnonzero(a['frame_id']==r['center_frame'])[0]);assert a['timestamp_us'][i]==r['timestamp_us']
  assert np.array_equal(a['frame_id'][i-6:i+7],np.arange(r['center_frame']-6,r['center_frame']+7))
  p,z=cache[r['clip_id']];inputs.append(p[i-6:i+7]);rgbs.append(z[i-6:i+7]);ys.append(r['y']);ms.append(r['mask'])
 X=torch.tensor(np.array(inputs));Z=torch.tensor(np.array(rgbs));Y=torch.tensor(np.array(ys),dtype=torch.float32);M=torch.tensor(np.array(ms),dtype=torch.bool)
 fitix=np.array([i for i,r in enumerate(labels) if r['role']=='fit']);devix=np.array([i for i,r in enumerate(labels) if r['role']=='development'])
 assert set(labels[i]['person'] for i in fitix)==set(range(1,10));assert set(labels[i]['person'] for i in devix)=={11,12,13}
 pos=(Y[fitix]*M[fitix]).sum(0);neg=((1-Y[fitix])*M[fitix]).sum(0);pw=(neg/pos.clamp_min(1)).clamp(.25,8)
 if not (V/'PREFLIGHT.json').exists():
  torch.manual_seed(123);check=RelationNet();check.eval();assert torch.isfinite(check(X[:2],Z[:2])).all()
  # No target ID, guide, person ID or time index is accepted by the neural forward.
  save(V/'PREFLIGHT.json',dict(status='PASS',fit_samples=len(fitix),development_samples=len(devix),
   unknown_excluded=int((~M).sum()),label_sha256=sha(V/'RELATION_TARGETS.json'),contract_sha256=sha(V/'TRAINING_CONTRACT.json'),
   goal_blind_forward=True,same_word_positive_negative_checked=True,old_models_not_optimizer_parameters=True))
 reports=[]
 for cond in contract['conditions']:
  for seed in contract['seeds']:
   d=V/'runs'/f'{cond}_{seed}'
   if (d/'COMPLETE.json').exists():reports.append(read(d/'COMPLETE.json'));continue
   assert not d.exists(),'Preserve partial run, inspect before any restart';d.mkdir(parents=True)
   torch.manual_seed(seed);rng=np.random.default_rng(seed);model=RelationNet(cond=='rgb_geometry')
   optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);start=time.perf_counter();history=[]
   for epoch in range(contract['epochs']):
    model.train();losses=[]
    for ix in np.array_split(rng.permutation(fitix),6):
     p=X[ix].clone();z=Z[ix];selected=rng.random(len(ix))<.15
     # Short joint-only gaps: current RGB and original labels retained, not zero-as-truth.
     p[selected,5:8,:]=0
     pred=model(p,z);loss=(torch.nn.functional.binary_cross_entropy_with_logits(pred,Y[ix],pos_weight=pw,reduction='none')*M[ix]).sum()/M[ix].sum()
     assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step();losses.append(float(loss.detach()))
    if epoch%30==0 or epoch==119:history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
   model.eval()
   with torch.inference_mode():pred=model(X,Z).sigmoid().numpy()
   torch.save(model.state_dict(),d/'model.pt');np.savez_compressed(d/'center_predictions.npz',p=pred,y=Y.numpy(),mask=M.numpy())
   dd=d/'predictions';dd.mkdir();lat=[]
   for cid,a in arrays.items():
    t=time.perf_counter();prob=dense(model,a,stats);lat.append(time.perf_counter()-t)
    np.savez_compressed(dd/f'{cid}.npz',probability=prob,frame_id=a['frame_id'],timestamp_us=a['timestamp_us'])
   rec=dict(condition=cond,seed=seed,epochs=120,training_seconds=time.perf_counter()-start,history=history,
    development=score(Y.numpy()[devix],pred[devix],M.numpy()[devix]),fit=score(Y.numpy()[fitix],pred[fitix],M.numpy()[fitix]),
    median_head_seconds=float(np.median(lat)),checkpoint_sha256=sha(d/'model.pt'),parameters=sum(p.numel() for p in model.parameters()),
    new_backbone_finetuning=False,old_shape_phase_models_preserved=True,full_independent_test=False)
   save(d/'COMPLETE.json',rec);reports.append(rec);print(cond,seed,rec['development'],flush=True)
 save(V/'TRAINING_RESULT.json',dict(status='SIX_RELATION_HEAD_RUNS_COMPLETE',runs=reports,labels_are_codex_visual_not_human_gold=True,model_selection='No best seed; all fixed42/43/44 reported'))
if __name__=='__main__':
 try:main()
 except BaseException:
  if not (V/'FIRST_FAILURE.txt').exists():(V/'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
  raise
