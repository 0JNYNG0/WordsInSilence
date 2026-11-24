from core import *
import os,traceback
def main():
 cfg=read(U/'FROZEN.json');assert read(U/'PRETRAIN_CHECK.json')['frozen_sha256']==sha(U/'FROZEN.json')
 for p,h in {**cfg['source_pins'],**cfg['code_pins']}.items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 allrows=load_rows();fit=[r for r in allrows if r['role']=='fit'];lab=[r for r in fit if r['has_labels']]
 pos=sum((r['y']*r['ym']).sum(0) for r in lab);neg=sum(((1-r['y'])*r['ym']).sum(0) for r in lab);pw=torch.tensor(np.clip(neg/np.maximum(pos,1),.25,4),dtype=torch.float32)
 for condition in cfg['conditions']:
  for seed in cfg['seeds']:
   d=U/'runs'/f'{condition}_{seed}';d.mkdir(parents=True,exist_ok=True)
   if (d/'COMPLETE.json').exists():continue
   assert not (d/'last.pt').exists(),'Interrupted run requires resume review'
   torch.manual_seed(seed);np.random.seed(seed);net=SharedNet();net.base.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=True));initial={k:v.clone() for k,v in net.base.state_dict().items()}
   opt=torch.optim.AdamW([dict(params=net.base.parameters(),lr=cfg['lr_base']),dict(params=net.word_head.parameters(),lr=cfg['lr_head'])],weight_decay=cfg['weight_decay'])
   history=[];seen=set();start=time.perf_counter();step=0
   write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid(),condition=condition,seed=seed))
   for epoch in range(1,cfg['epochs']+1):
    net.train();rng=np.random.default_rng(seed*1000+epoch);auxorder=rng.permutation(len(fit));vals=[]
    for k in range(cfg['steps_per_epoch']):
     chosen=[lab[i] for i in rng.choice(len(lab),cfg['supervised_batch'],replace=False)]
     lo=k*cfg['aux_batch'];ids=auxorder[lo:lo+cfg['aux_batch']];aux=[fit[int(i)] for i in ids]
     if not aux:continue
     view='upper' if (epoch+k)%2 else 'full';masked=(step%cfg['mask_every_steps']==0)
     b=batch(chosen,view,masked,rng);loss,detail=supervised_loss(net,b,pw)
     ab=batch(aux,view,False);ao=net(ab['x'],ab['lengths'],ab['word']);wl=F.cross_entropy(ao['word_logits'],ab['class'])
     if condition=='shared100':loss=loss+cfg['word_loss_weight']*wl;seen.update(r['clip_id'] for r in aux)
     opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();step+=1
     assert torch.isfinite(loss);vals.append(dict(loss=float(loss.detach()),word=float(wl.detach()),**detail))
    history.append(dict(epoch=epoch,seconds=time.perf_counter()-start,**{k:float(np.mean([v[k] for v in vals])) for k in vals[0]}))
    if epoch%4==0:
     torch.save(dict(model=net.state_dict(),opt=opt.state_dict(),epoch=epoch,history=history,torch_rng=torch.get_rng_state()),d/'last.pt');event('epoch',condition=condition,seed=seed,**history[-1])
   torch.save(net.state_dict(),d/'model.pt');delta={prefix:float(sum((net.base.state_dict()[k]-v).float().square().sum() for k,v in initial.items() if k.startswith(prefix)).sqrt()) for prefix in ['input','temporal','context','attributes','stage_query','guide_encoder','prototype']}
   assert delta['temporal']>0 and delta['context']>0
   if condition=='shared100':assert len(seen)==899
   write(d/'COMPLETE.json',dict(condition=condition,seed=seed,steps=step,seconds=time.perf_counter()-start,deltas=delta,aux_supervised_clips=len(seen),history=history,checkpoint_sha256=sha(d/'model.pt'),development_training=False))
   event('run_complete',condition=condition,seed=seed,seconds=time.perf_counter()-start)
 write(U/'TRAINING_COMPLETE.json',dict(status='SIX_FIXED_RUNS_COMPLETE',runs=[str(p) for p in (U/'runs').glob('*/COMPLETE.json')]))
 write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED_PENDING_EVALUATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
