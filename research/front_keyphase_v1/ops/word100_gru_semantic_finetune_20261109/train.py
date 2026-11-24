from semantic_model import *
import traceback

def main():
 cfg=read(U/'FROZEN.json');check=read(U/'PRETRAIN_CHECK.json');assert check['status']=='PASS'
 assert check['frozen_sha256']==sha(U/'FROZEN.json')
 for p,h in cfg['source_pins'].items():assert sha(p)==h
 for p,h in check['code_pins'].items():assert sha(p)==h
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 fit=[r for r in load_rows() if r['role']=='fit'];assert {r['person'] for r in fit}==set(range(1,10))
 pos=sum((r['y']*r['ym']).sum(0) for r in fit);neg=sum(((1-r['y'])*r['ym']).sum(0) for r in fit)
 pw=torch.tensor(np.clip(neg/np.maximum(pos,1),.25,4),dtype=torch.float32)
 for condition in cfg['conditions']:
  for seed in cfg['seeds']:
   d=U/'runs'/f'{condition}_{seed}';d.mkdir(parents=True,exist_ok=True)
   if (d/'COMPLETE.json').exists():continue
   assert not (d/'last.pt').exists(),'Inspect interrupted training before resume'
   torch.manual_seed(seed);np.random.seed(seed);model=SemanticNet();model.base.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=True))
   initial={k:v.clone() for k,v in model.state_dict().items()}
   if condition=='frozen_heads':
    for p in model.base.parameters():p.requires_grad_(False)
   groups=[dict(params=list(model.direction.parameters())+list(model.order.parameters()),lr=cfg['lr_heads'])]
   if condition=='finetune':groups.append(dict(params=model.base.parameters(),lr=cfg['lr_base']))
   opt=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay']);history=[];start=time.perf_counter()
   for epoch in range(1,cfg['epochs']+1):
    model.train();order=np.random.default_rng(seed+epoch).permutation(len(fit));losses=[]
    batchrows=[transform(r,reverse=bool(r['pairs'] and (epoch%2==0)),view='upper' if (epoch+int(hashlib.sha256(r['clip_id'].encode()).hexdigest(),16))%2 else 'full') for r in fit]
    for lo in range(0,len(order),cfg['batch_size']):
     lossdict=batch_losses(model,[batchrows[i] for i in order[lo:lo+cfg['batch_size']]],pw)
     loss=sum(cfg['losses'][k]*v for k,v in lossdict.items());assert torch.isfinite(loss)
     # A frozen-base batch with no reviewed direction/order has no trainable
     # supervision. Skip it; never turn unknown examples into fake negatives.
     if loss.requires_grad:
      opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);opt.step()
     losses.append({k:float(v.detach()) for k,v in lossdict.items()})
    row=dict(epoch=epoch,elapsed=time.perf_counter()-start,**{k:float(np.mean([a[k] for a in losses])) for k in losses[0]});history.append(row)
    if epoch%20==0:
     torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),epoch=epoch,history=history,torch_rng=torch.get_rng_state()),d/'last.pt')
     event('training',condition=condition,seed=seed,**row)
   torch.save(model.state_dict(),d/'model.pt')
   deltas={prefix:float(sum((model.state_dict()[k]-v).float().square().sum() for k,v in initial.items() if k.startswith(prefix)).sqrt()) for prefix in ['base.input','base.temporal','base.context','base.attributes','base.stage','base.guide','direction','order']}
   if condition=='frozen_heads':assert all(v==0 for k,v in deltas.items() if k.startswith('base'))
   else:assert all(deltas[k]>0 for k in ['base.temporal','base.context','base.attributes'])
   result=dict(condition=condition,seed=seed,epochs=cfg['epochs'],seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),deltas=deltas,history=history,checkpoint_sha256=sha(d/'model.pt'),frozen_sha256=sha(U/'FROZEN.json'),fit_persons=sorted({r['person'] for r in fit}),development_seen=False)
   write(d/'COMPLETE.json',result);event('training_complete',condition=condition,seed=seed,seconds=result['seconds'],deltas=deltas)
 write(U/'TRAINING_COMPLETE.json',dict(status='SIX_RUNS_COMPLETE_PENDING_EVALUATION',runs=[str(p) for p in (U/'runs').glob('*/COMPLETE.json')]))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
