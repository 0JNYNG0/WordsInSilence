from link_core import *
import os,traceback
def main():
 cfg=read(U/'FROZEN.json');torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for p,h in {**cfg['protections'],**cfg['pins'],**read(U/'CODE_PINS.json')}.items():assert sha(p)==h,p
 rr=[r for r in old.rows() if r['role']=='fit']
 for r in rr:
  z=np.load(B/'cache'/f"{r['clip_id']}.npz");r.update({k:z[k] for k in z.files});p=U/'labels'/f"{r['clip_id']}.npz"
  if p.exists():a=np.load(p);r['link_y']=a['y'];r['link_w']=a['w']
  else:r['link_y']=np.full(len(r['frame_id']),-1,np.int64);r['link_w']=np.zeros(len(r['frame_id']),np.float32)
 fine=[r for r in rr if r['link_w'].sum()];assert len(fine)==36
 for seed in cfg['seeds']:
  dest=U/'runs'/str(seed);dest.mkdir(parents=True,exist_ok=True)
  if (dest/'COMPLETE.json').exists():continue
  assert not (dest/'last.pt').exists();torch.manual_seed(seed);net=LinkNet();initial={k:v.clone() for k,v in net.encoder.state_dict().items()};opt=torch.optim.AdamW([dict(params=net.encoder.parameters(),lr=cfg['lr_encoder']),dict(params=list(net.link.parameters())+list(net.word.parameters()),lr=cfg['lr_head'])],weight_decay=.01);hist=[];start=time.perf_counter();seen=set();write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid(),seed=seed))
  for ep in range(1,cfg['epochs']+1):
   rng=np.random.default_rng(seed*1000+ep);perm=rng.permutation(len(rr));losses=[];epoch_seen=set();net.train()
   for step in range(cfg['steps']):
    common=[rr[int(j)] for j in perm[step*24:(step+1)*24]];extra=[fine[int(j)] for j in rng.choice(len(fine),6,replace=False)];view='upper' if (ep+step)%2 else 'full';g,q,fh,l,y,w=batch(common+extra,view);pr,word,_=net(g,q,fh,l);n=len(common);wl=F.cross_entropy(word[:n],torch.tensor([r['class_index'] for r in common]));sl=(F.cross_entropy(pr.transpose(1,2),y,ignore_index=-1,reduction='none')*w).sum()/w.sum().clamp_min(1)
    if step%2==0:
     g,q,fh,l,y,w=batch(extra,view,True);pr,_,_=net(g,q,fh,l);sl=(sl+(F.cross_entropy(pr.transpose(1,2),y,ignore_index=-1,reduction='none')*w).sum()/w.sum().clamp_min(1))/2
    loss=.2*wl+sl;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();assert torch.isfinite(loss);losses.append([float(loss.detach()),float(wl.detach()),float(sl.detach())]);epoch_seen.update(r['clip_id'] for r in common)
   assert len(epoch_seen)==899;seen.update(epoch_seen);hist.append(dict(epoch=ep,common_seen=899,loss=np.mean(losses,0).tolist()));event('epoch',seed=seed,**hist[-1]);torch.save(dict(epoch=ep,model=net.state_dict()),dest/'last.pt')
  torch.save(net.state_dict(),dest/'model.pt');delta=float(sum((net.encoder.state_dict()[k]-v).square().sum() for k,v in initial.items()).sqrt());write(dest/'COMPLETE.json',dict(seed=seed,history=hist,seen=sorted(seen),encoder_delta=delta,sha256=sha(dest/'model.pt'),seconds=time.perf_counter()-start))
 write(U/'TRAINING_COMPLETE.json',dict(seeds=cfg['seeds']));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
