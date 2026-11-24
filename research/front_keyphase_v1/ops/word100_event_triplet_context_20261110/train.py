from triplet_core import *
import os,traceback
def main():
 cfg=read(U/'FROZEN.json');torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for p,h in {**cfg['protections'],**cfg['pins'],**read(U/'CODE_PINS.json')}.items():assert sha(p)==h,p
 am={r['clip_id']:r for r in read(U/'SEGMENT_LABELS.json')['rows']};rr=[r for r in old.rows() if r['role']=='fit']
 for r in rr:
  z=np.load(B/'cache'/f"{r['clip_id']}.npz");r.update({k:z[k] for k in z.files});p=L/'labels'/f"{r['clip_id']}.npz"
  if p.exists():a=np.load(p);r['link_y']=a['y'];r['link_w']=a['w']
  else:r['link_y']=np.full(len(r['frame_id']),-1,np.int64);r['link_w']=np.zeros(len(r['frame_id']),np.float32)
  r['segments']=am.get(r['clip_id'],{}).get('segments',[])
 fine=[r for r in rr if r['segments']];assert len(fine)==36
 for seed in cfg['seeds']:
  d=U/'runs'/str(seed);d.mkdir(parents=True,exist_ok=True)
  if (d/'COMPLETE.json').exists():continue
  assert not (d/'last.pt').exists();torch.manual_seed(seed);net=TripletNet();initial={k:v.clone() for k,v in net.encoder.state_dict().items()}
  opt=torch.optim.AdamW([dict(params=net.encoder.parameters(),lr=cfg['lr_encoder']),dict(params=list(net.link.parameters())+list(net.word.parameters())+list(net.event_head.parameters())+list(net.core_head.parameters()),lr=cfg['lr_head'])],weight_decay=.01)
  hist=[];seen=set();start=time.perf_counter();write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid(),seed=seed))
  for ep in range(1,cfg['epochs']+1):
   rng=np.random.default_rng(seed*1000+ep);perm=rng.permutation(len(rr));losses=[];epoch_seen=set();net.train()
   for step in range(cfg['steps']):
    common=[rr[int(j)] for j in perm[step*24:(step+1)*24]];extra=[fine[int(j)] for j in rng.choice(len(fine),6,replace=False)];view='upper' if (ep+step)%2 else 'full'
    def supervised(rows,reverse=False):
     g,q,fh,ll,y,w,ii,ey,cy,ts=batch(rows,view,reverse);p,word,h=net(g,q,fh,ll);ev,co=net.segment(h,q,ii,ts);_,masked=net.segment(h,q,ii,ts,True)
     fl=(F.cross_entropy(p.transpose(1,2),y,ignore_index=-1,reduction='none')*w).sum()/w.sum().clamp_min(1)
     sl=F.cross_entropy(ev,ey);cl=(F.cross_entropy(co,cy)+F.cross_entropy(masked,torch.zeros_like(cy)))/2
     return .4*fl+sl+.3*cl,word,torch.stack([fl,sl,cl])
    sup,word,parts=supervised(common+extra);n=len(common);wl=F.cross_entropy(word[:n],torch.tensor([r['class_index'] for r in common]))
    if step%2==0:rev,_,_=supervised(extra,True);sup=(sup+rev)/2
    loss=.2*wl+sup;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();assert torch.isfinite(loss);losses.append([float(loss.detach()),float(wl.detach()),*parts.detach().tolist()]);epoch_seen.update(r['clip_id'] for r in common)
   assert len(epoch_seen)==899;seen.update(epoch_seen);hist.append(dict(epoch=ep,common_seen=899,loss=np.mean(losses,0).tolist()));event('epoch',seed=seed,**hist[-1]);torch.save(dict(epoch=ep,model=net.state_dict()),d/'last.pt')
  torch.save(net.state_dict(),d/'model.pt');delta=float(sum((net.encoder.state_dict()[k]-v).square().sum() for k,v in initial.items()).sqrt());write(d/'COMPLETE.json',dict(seed=seed,history=hist,seen=sorted(seen),encoder_delta=delta,sha256=sha(d/'model.pt'),seconds=time.perf_counter()-start))
 write(U/'TRAINING_COMPLETE.json',dict(seeds=cfg['seeds']));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
