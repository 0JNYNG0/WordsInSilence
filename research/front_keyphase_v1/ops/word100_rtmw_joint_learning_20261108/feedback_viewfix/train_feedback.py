from common import *
import numpy as np,torch,traceback
from torch.nn import functional as F
from feedback_model import PoseGuideNet,collate

def metrics(y,p,m,targets):
 out=[]
 for j,name in enumerate(targets):
  keep=m[:,j]>0;g=y[keep,j]>.5;pr=p[keep,j]>=.5;pos=int(g.sum());neg=int((~g).sum())
  tp=int((pr&g).sum());tn=int((~pr&~g).sum())
  out.append(dict(target=name,n=len(g),positive=pos,negative=neg,tp=tp,tn=tn,accuracy=(tp+tn)/len(g) if len(g) else None,
    balanced_accuracy=(tp/pos+tn/neg)/2 if pos and neg else None))
 eligible=[r['balanced_accuracy'] for r in out if r['balanced_accuracy'] is not None]
 return dict(targets=out,macro_balanced_accuracy=float(np.mean(eligible)),known=sum(r['n'] for r in out),correct=sum(r['tp']+r['tn'] for r in out))

def main():
 checks=read(U/'LEARNING_CODE_CHECK.json');assert checks['status']=='PASS';assert all(sha(p)==h for p,h in checks['pins'].items())
 ready=read(U/'FEEDBACK_READY.json');f=read(U/'FROZEN.json')['feedback'];stats=read(U/'FEEDBACK_NORMALIZATION.json')
 assert ready['frozen_sha256']==sha(U/'FROZEN.json');assert not (U/'FEEDBACK_TRAINING_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 rows=[]
 for r in ready['rows']:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path'],allow_pickle=False);rows.append(dict(r,**{k:z[k] for k in z.files}))
 fit=[r for r in rows if r['role']=='fit'];targets=read(O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json')['targets'];reports=[]
 pw=torch.tensor(np.clip(np.asarray(ready['fit_negative'])/np.maximum(ready['fit_positive'],1),.25,4),dtype=torch.float32)
 for condition in f['conditions']:
  for seed in f['seeds']:
   d=U/'feedback_runs'/f'{condition}_{seed}';d.mkdir(parents=True,exist_ok=True)
   if (d/'COMPLETE.json').exists():reports.append(read(d/'COMPLETE.json'));continue
   assert not (d/'last.pt').exists(),'Review interrupted run before recovery'
   torch.manual_seed(seed);np.random.seed(seed);model=PoseGuideNet(stats['guide_vectors'],stats['guide_words'],stats['prototype_lengths']);copied=[]
   if condition=='pose_transfer':
    p=O/f'word100_learned_guide_feedback_20261102/runs/projection_finetune_{seed}/model.pt';assert sha(p)==read(U/'FROZEN.json')['pins'][str(p)]
    old=torch.load(p,map_location='cpu',weights_only=True);sd=model.state_dict()
    for k in sd:
     if k.startswith(('temporal.','context.','attributes.','stage_query.','guide_encoder.')) or k in ['stage_embedding','prototype']:
      assert k in old and old[k].shape==sd[k].shape;sd[k]=old[k].clone();copied.append(k)
    model.load_state_dict(sd)
   initial={k:v.clone() for k,v in model.state_dict().items()};opt=torch.optim.AdamW(model.parameters(),lr=f['lr'],weight_decay=f['weight_decay']);history=[];t=time.perf_counter()
   for epoch in range(1,f['epochs']+1):
    model.train();order=np.random.default_rng(seed+epoch).permutation(len(fit));losses=[]
    for rr in fit:rr['x']=rr['x_upper'] if (epoch+int(hashlib.sha256(rr['clip_id'].encode()).hexdigest(),16))%2 else rr['x_full']
    for lo in range(0,len(order),f['batch_size']):
     b=collate([fit[i] for i in order[lo:lo+f['batch_size']]]);o=model(b['x'],b['lengths'],b['word'])
     attr=(F.binary_cross_entropy_with_logits(o['attributes'],b['y'],reduction='none',pos_weight=pw)*b['ym']).sum()/b['ym'].sum().clamp_min(1)
     ce=F.cross_entropy(o['stage'].transpose(1,2),b['sy'],ignore_index=-1,reduction='none');stage=(ce*b['sw']).sum()/b['sw'].sum().clamp_min(1)
     valid=(b['ci']>=0)&o['valid'];ci=b['ci'].clamp_min(0);proto=model.prototype[b['word'][:,None],ci]
     align=((o['latent']-proto).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
     loss=attr+.35*stage+.1*align;assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),f['gradient_clip']);opt.step();losses.append(float(loss.detach()))
    history.append(dict(epoch=epoch,loss=float(np.mean(losses)),elapsed=time.perf_counter()-t))
    torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),epoch=epoch,history=history,torch_rng=torch.get_rng_state()),d/'last.pt')
    if epoch%20==0:event('feedback_training',condition=condition,seed=seed,**history[-1])
   train_seconds=time.perf_counter()-t;model.eval();ys=[];ps=[];ms=[];predrows=[]
   with torch.inference_mode():
    for r in rows:
     r['x']=r['x_full']
     b=collate([r]);start=time.perf_counter();o=model(b['x'],b['lengths'],b['word']);duration=time.perf_counter()-start
     p=o['attributes'][0].sigmoid().numpy();sp=o['stage'][0].softmax(-1).numpy();latent=o['latent'][0].numpy()
     path=d/'predictions'/f"{r['clip_id']}.npz";path.parent.mkdir(exist_ok=True);np.savez_compressed(path,probability=p,stage_probability=sp,latent=latent,frame_id=r['frame_id'],timestamp_us=r['timestamp_us'])
     predrows.append(dict(clip_id=r['clip_id'],role=r['role'],path=str(path),sha256=sha(path),head_seconds=duration))
     if r['role']=='development':ys.append(r['y']);ps.append(p);ms.append(r['ym'])
   torch.save(model.state_dict(),d/'model.pt')
   delta=sum(float((model.state_dict()[k]-v).float().square().sum()) for k,v in initial.items())**.5
   out=dict(condition=condition,seed=seed,epochs=f['epochs'],parameters=sum(p.numel() for p in model.parameters()),transferred_keys=copied,weight_delta_l2=delta,
    training_seconds=train_seconds,metrics=metrics(np.concatenate(ys),np.concatenate(ps),np.concatenate(ms),targets),predictions=predrows,history=history,
    checkpoint_sha256=sha(d/'model.pt'),frozen_sha256=sha(U/'FROZEN.json'),rgb_used=False,expert_correctness_validated=False)
   write(d/'COMPLETE.json',out);reports.append(out);event('feedback_trained',condition=condition,seed=seed,balanced_accuracy=out['metrics']['macro_balanced_accuracy'])
 write(U/'FEEDBACK_TRAINING_COMPLETE.json',dict(status='SIX_FIXED_RUNS_COMPLETE_PENDING_COMPARISON',runs=reports,production_replaced=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'FEEDBACK_TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
