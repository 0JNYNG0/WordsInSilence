from phase_core import *
import os,traceback
def main():
 cfg=read(U/'FROZEN.json');torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for p,h in {**cfg['pins'],**cfg['code_pins'],**cfg['protections']}.items():assert sha(p)==h,p
 fit=[r for r in rows() if r['role']=='fit']
 for r in fit:
  z=np.load(U/'cache'/f"{r['clip_id']}.npz");r.update({k:z[k] for k in z.files})
 fine=[r for r in fit if r['pairs']];cw=torch.tensor(cfg['class_weights'])
 for seed in cfg['seeds']:
  dst=U/'runs'/str(seed);dst.mkdir(parents=True,exist_ok=True)
  if (dst/'COMPLETE.json').exists():continue
  assert not (dst/'last.pt').exists();torch.manual_seed(seed);net=PhaseNet();initial={k:v.clone() for k,v in net.encoder.state_dict().items()};opt=torch.optim.AdamW([dict(params=net.encoder.parameters(),lr=cfg['lr']),dict(params=list(net.phase.parameters())+list(net.word.parameters()),lr=cfg['head_lr'])],weight_decay=.01);hist=[];start=time.perf_counter();seen=set();write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',seed=seed,pid=os.getpid()))
  for epoch in range(1,cfg['epochs']+1):
   rng=np.random.default_rng(seed*1000+epoch);perm=rng.permutation(899);losses=[];epoch_seen=set();net.train()
   for step in range(38):
    common=[fit[int(j)] for j in perm[step*24:(step+1)*24]];extra=[fine[int(j)] for j in rng.choice(len(fine),6,replace=False)];view='upper' if (epoch+step)%2 else 'full';g,q,fh,l,y,w=batch(common+extra,view);ph,word,_=net(g,q,fh,l);n=len(common);wl=TF.cross_entropy(word[:n],torch.tensor([r['class_index'] for r in common]));pl=(-(ph.log_softmax(-1)*y*cw).sum(-1)*w).sum()/w.sum().clamp_min(1)
    if step%2==0:
     g,q,fh,l,y,w=batch(extra,view,True);ph,_,_=net(g,q,fh,l);pl=(pl+(-(ph.log_softmax(-1)*y*cw).sum(-1)*w).sum()/w.sum().clamp_min(1))/2
    loss=.2*wl+pl;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();assert torch.isfinite(loss);losses.append([float(loss.detach()),float(wl.detach()),float(pl.detach())]);epoch_seen.update(r['clip_id'] for r in common)
   assert len(epoch_seen)==899;seen.update(epoch_seen);hist.append(dict(epoch=epoch,all100_seen=len(epoch_seen),loss=np.mean(losses,0).tolist(),seconds=time.perf_counter()-start));event('epoch',seed=seed,**hist[-1]);torch.save(dict(epoch=epoch,model=net.state_dict()),dst/'last.pt')
  torch.save(net.state_dict(),dst/'model.pt');delta=float(sum((net.encoder.state_dict()[k]-v).square().sum() for k,v in initial.items()).sqrt());assert delta>0;write(dst/'COMPLETE.json',dict(seed=seed,history=hist,seen=sorted(seen),encoder_delta=delta,sha256=sha(dst/'model.pt'),seconds=time.perf_counter()-start,shape_direction_checkpoint=cfg['pins'][str(P/'runs/coupled_42/model.pt')]))
 write(U/'TRAINING_COMPLETE.json',dict(seeds=cfg['seeds']));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
