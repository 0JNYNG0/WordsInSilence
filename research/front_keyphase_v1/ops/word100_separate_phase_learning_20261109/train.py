from lib import *
import os,traceback
def main():
 cfg=read(P/'FROZEN.json');assert not (P/'TRAINING_COMPLETE.json').exists()
 for p,h in {**cfg['source_pins'],**cfg['code_pins']}.items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);rows=load_rows();fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development'];assert len(fit)==36 and len(dev)==12
 assert not({r['person'] for r in fit}&{r['person'] for r in dev}) and not({r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev})
 assert all(np.all(r['stage_y']==-1) and np.all(r['stage_w']==0) for r in dev)
 counts=sum(np.bincount(r['stage_y'][r['stage_y']>=0],weights=r['stage_w'][r['stage_y']>=0],minlength=14) for r in fit);cw=np.sqrt(counts.sum()/np.maximum(counts,1));cw/=cw.mean();cw=torch.tensor(cw,dtype=torch.float32)
 write(P/'PRETRAIN_CHECK.json',dict(status='PASS',fit=36,dev=12,weighted_phase_counts=counts.tolist(),frozen_sha256=sha(P/'FROZEN.json'),DEV_losses_zero=True,base_readonly=True))
 initial=torch.load(D/'runs/42/model.pt',map_location='cpu',weights_only=True)
 for cond in cfg['conditions']:
  for seed in cfg['seeds']:
   d=P/'runs'/f'{cond}_{seed}';d.mkdir(parents=True,exist_ok=True)
   if (d/'COMPLETE.json').exists():continue
   assert not (d/'last.pt').exists(),'Inspect interrupted run before resume'
   torch.manual_seed(seed);net=Heads(cond=='matched_relations');state=net.state_dict()
   for k,v in initial.items():
    if k=='temporal.0.weight' and cond=='matched_relations':state[k].zero_();state[k][:,:71]=v
    else:state[k]=v.clone()
   net.load_state_dict(state);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);history=[];t=time.perf_counter()
   write(P/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid(),condition=cond,seed=seed))
   for epoch in range(cfg['epochs']):
    net.train();order=np.random.default_rng(seed+epoch).permutation(len(fit));losses=[]
    for k in range(0,len(fit),cfg['batch_size']):
     b=batch([fit[int(i)] for i in order[k:k+cfg['batch_size']]],'full' if epoch%2==0 else 'upper');expected=torch.randint(0,2,b['y'].shape).float();truth=(b['y']!=expected).float();ph,dif=net(b['h'],b['p'],b['g'],b['word'],b['valid'],expected)
     ce=nn.functional.cross_entropy(ph.transpose(1,2),b['sy'],ignore_index=-1,weight=cw,reduction='none');pl=(ce*b['sw']).sum()/b['sw'].sum().clamp_min(1);dl=(nn.functional.binary_cross_entropy_with_logits(dif,truth,reduction='none')*b['ym']).sum()/b['ym'].sum().clamp_min(1);loss=pl+cfg['difference_loss_weight']*dl
     assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();losses.append(float(loss.detach()))
    history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
    if (epoch+1)%25==0:
     torch.save(dict(model=net.state_dict(),opt=opt.state_dict(),epoch=epoch+1,history=history,rng=torch.get_rng_state()),d/'last.pt');event('epoch',condition=cond,seed=seed,epoch=epoch+1,seconds=time.perf_counter()-t)
   torch.save(net.state_dict(),d/'model.pt');write(d/'COMPLETE.json',dict(condition=cond,seed=seed,epochs=cfg['epochs'],seconds=time.perf_counter()-t,history=history,checkpoint_sha256=sha(d/'model.pt'),base_sha256=sha(CHECKPOINT),base_changed=False,DEV_used=False));event('run_complete',condition=cond,seed=seed,seconds=time.perf_counter()-t)
 assert sha(CHECKPOINT)==cfg['base_sha256'];write(P/'TRAINING_COMPLETE.json',dict(status='SIX_MATCHED_HEAD_RUNS_COMPLETE',runs=[str(p) for p in (P/'runs').glob('*/COMPLETE.json')],common_handshape_preserved=True));write(P/'CURRENT_EXECUTION.json',dict(status='TRAINED_PENDING_EVALUATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:
  (P/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
