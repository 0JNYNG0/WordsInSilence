from episode_core import *
import os,traceback
def main():
 cfg=read(U/'FROZEN.json');torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for p,h in {**cfg['pins'],**cfg['protections'],**read(U/'CODE_PINS.json')}.items():assert sha(p)==h,p
 rr=[r for r in old.rows() if r['role']=='fit']
 for r in rr:
  z=np.load(B/'cache'/f"{r['clip_id']}.npz");r.update({k:z[k] for k in z.files});p=U/'labels'/f"{r['clip_id']}.npz"
  if p.exists():z=np.load(p);r['role_y']=z['y'];r['role_w']=z['w']
  else:r['role_y']=np.full(len(r['frame_id']),-1,np.int64);r['role_w']=np.zeros(len(r['frame_id']),np.float32)
 fine=[r for r in rr if r['role_w'].sum()];bounds=[r for r in rr if r['boundary_w'].sum()];assert len(fine)==36 and len(bounds)==18;pw=torch.tensor(cfg['pos_weight'])
 for seed in cfg['seeds']:
  d=U/'runs'/str(seed);d.mkdir(parents=True,exist_ok=True)
  if (d/'COMPLETE.json').exists():continue
  assert not (d/'last.pt').exists();torch.manual_seed(seed);net=EpisodeNet();initial={k:v.clone() for k,v in net.encoder.state_dict().items()};opt=torch.optim.AdamW([dict(params=net.encoder.parameters(),lr=cfg['lr_encoder']),dict(params=list(net.role.parameters())+list(net.boundary.parameters())+list(net.word.parameters()),lr=cfg['lr_head'])],weight_decay=.01);hist=[];seen=set();start=time.perf_counter();write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid(),seed=seed))
  for epoch in range(1,cfg['epochs']+1):
   rng=np.random.default_rng(seed*1000+epoch);perm=rng.permutation(len(rr));losses=[];epochseen=set();net.train()
   for step in range(cfg['steps']):
    common=[rr[int(i)] for i in perm[step*24:(step+1)*24]];extra=[fine[int(i)] for i in rng.choice(len(fine),4,False)]+[bounds[int(i)] for i in rng.choice(len(bounds),4,False)];view='upper' if (epoch+step)%2 else 'full'
    def supervise(rows,reverse=False):
     g,q,fh,ll,by,bw,y,w=batch(rows,view,reverse);rp,bp,word,_=net(g,q,fh,ll,drop_appearance=step%4==0)
     rl=(F.cross_entropy(rp.transpose(1,2),y,ignore_index=-1,reduction='none')*w).sum()/w.sum().clamp_min(1)
     bl=(F.binary_cross_entropy_with_logits(bp,by,pos_weight=pw,reduction='none')*bw).sum()/bw.sum().clamp_min(1)
     return rl+bl,word,rl,bl
    su,word,rl,bl=supervise(common+extra);wl=F.cross_entropy(word[:len(common)],torch.tensor([r['class_index'] for r in common]))
    if step%2==0:rv,_,_,_=supervise(extra,True);su=(su+rv)/2
    loss=.2*wl+su;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();assert torch.isfinite(loss);losses.append([float(z.detach()) for z in [loss,wl,rl,bl]]);epochseen.update(r['clip_id'] for r in common)
   assert len(epochseen)==899;seen.update(epochseen);hist.append(dict(epoch=epoch,common_seen=899,loss=np.mean(losses,0).tolist()));event('epoch',seed=seed,**hist[-1]);torch.save(dict(epoch=epoch,model=net.state_dict()),d/'last.pt')
  torch.save(net.state_dict(),d/'model.pt');delta=float(sum((net.encoder.state_dict()[k]-v).square().sum() for k,v in initial.items()).sqrt());write(d/'COMPLETE.json',dict(seed=seed,history=hist,seen=sorted(seen),encoder_delta=delta,seconds=time.perf_counter()-start,sha256=sha(d/'model.pt')))
 write(U/'TRAINING_COMPLETE.json',dict(seeds=cfg['seeds']));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
