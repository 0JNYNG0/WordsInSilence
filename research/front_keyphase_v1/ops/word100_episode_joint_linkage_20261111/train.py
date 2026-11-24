from joint_core import *
import os,traceback
def load_rows():
 rr=[r for r in old.rows() if r['role']=='fit'];ann=read(U/'ALL_FIT_BOUNDARIES.json')['rows']
 for r in rr:
  z=np.load(B/'cache'/f"{r['clip_id']}.npz");r.update({k:z[k] for k in z.files});r['annotations']=[a for a in ann if a['clip_id']==r['clip_id']];r['boundary_y'],r['boundary_w']=bc.targets(r,r['annotations']);p=P/'labels'/f"{r['clip_id']}.npz"
  if p.exists():z=np.load(p);r['role_y'],r['role_w']=z['y'],z['w']
  else:r['role_y']=np.full(len(r['frame_id']),-1,np.int64);r['role_w']=np.zeros(len(r['frame_id']),np.float32)
 return rr
def losses(net,rows,view,reverse,pw):
 g,q,fh,ll,by,bw,y,w=ep.batch(rows,view,reverse);rp,bp,word,h=net(g,q,fh,ll)
 rl=(F.cross_entropy(rp.transpose(1,2),y,ignore_index=-1,reduction='none')*w).sum()/w.sum().clamp_min(1);bl=(F.binary_cross_entropy_with_logits(bp,by,pos_weight=pw,reduction='none')*bw).sum()/bw.sum().clamp_min(1)
 ss,yy,ps,py=example_specs(rows,reverse);sf=net.segment(h,q,fh,ss);valid=[bool((q[i,a:b+1,k,-2:].max(-1).values>.5).float().mean()>=.8) for i,a,b,k in ss]
 ql=h.sum()*0;pl=h.sum()*0
 if any(valid):ql=F.cross_entropy(net.quality(sf)[valid],torch.tensor(yy)[valid])
 pp=[(a,b,gap,c) for (a,b,gap),c in zip(ps,py) if valid[a] and valid[b]]
 if pp:
  pf=torch.stack([torch.cat([sf[a],sf[b],torch.tensor([gap])]) for a,b,gap,c in pp]);pl=F.cross_entropy(net.pair(pf),torch.tensor([c for a,b,gap,c in pp]))
 return rl+bl+ql+pl,word,[rl,bl,ql,pl]
def check_probe(net,rr):
 rows=[];net.eval()
 with torch.inference_mode():
  for r in rr:
   if r['person'] not in [3,7]:continue
   if not r['annotations'] and not r['role_w'].sum():continue
   g,q,fh,ll,by,bw,y,w=ep.batch([r],'full');rp,bp,_,h=net(g,q,fh,ll);ss,yy,ps,py=example_specs([r]);sf=net.segment(h,q,fh,ss)
   qp=net.quality(sf).argmax(-1).tolist() if ss else [];pred=[]
   if ps:pf=torch.stack([torch.cat([sf[a],sf[b],torch.tensor([gap])]) for a,b,gap in ps]);pred=net.pair(pf).argmax(-1).tolist()
   rows.append(dict(clip_id=r['clip_id'],quality_truth=yy,quality_pred=qp,pair_truth=py,pair_pred=pred,role_correct=int(((rp.argmax(-1)==y)&(w==1)).sum()),role_n=int((w==1).sum())))
 return rows
def main():
 cfg=read(U/'FROZEN.json');torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for p,h in {**cfg['pins'],**cfg['protections'],**read(U/'CODE_PINS.json')}.items():assert sha(p)==h,p
 allrows=load_rows();pw=torch.tensor(cfg['pos_weight'])
 for name,seed,epochs in [('annotation_probe',42,cfg['probe_epochs'])]+[(str(s),s,cfg['epochs']) for s in cfg['seeds']]:
  d=U/'runs'/name;d.mkdir(parents=True,exist_ok=True)
  if (d/'COMPLETE.json').exists():continue
  assert not (d/'last.pt').exists();rr=copy.deepcopy(allrows)
  if name=='annotation_probe':
   for r in rr:
    if r['person'] in cfg['probe_heldout_annotation_ids']:r['annotations']=[];r['boundary_w']*=0;r['role_w']*=0
  fine=[r for r in rr if r['role_w'].sum()];bounds=[r for r in rr if r['annotations']];torch.manual_seed(seed);net=JointNet();initial={k:v.clone() for k,v in net.encoder.state_dict().items()};opt=torch.optim.AdamW([dict(params=net.encoder.parameters(),lr=cfg['lr_encoder']),dict(params=list(net.role.parameters())+list(net.boundary.parameters())+list(net.word.parameters())+list(net.quality.parameters())+list(net.pair.parameters()),lr=cfg['lr_head'])],weight_decay=.01);hist=[];start=time.perf_counter();write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid(),run=name))
  for epoch in range(1,epochs+1):
   rng=np.random.default_rng(seed*1000+epoch);perm=rng.permutation(len(rr));ls=[];seen=set();net.train()
   for step in range(38):
    common=[rr[int(i)] for i in perm[step*24:(step+1)*24]];extra=[fine[int(i)] for i in rng.choice(len(fine),4,False)]+[bounds[int(i)] for i in rng.choice(len(bounds),4,False)];view='upper' if (step+epoch)%2 else 'full';su,word,parts=losses(net,common+extra,view,False,pw);wl=F.cross_entropy(word[:len(common)],torch.tensor([r['class_index'] for r in common]))
    if step%2==0:rv,_,_=losses(net,extra,view,True,pw);su=(su+rv)/2
    loss=.2*wl+su;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();assert torch.isfinite(loss);ls.append([float(z.detach()) for z in [loss,wl]+parts]);seen.update(r['clip_id'] for r in common)
   assert len(seen)==899;hist.append(dict(epoch=epoch,common_seen=899,loss=np.mean(ls,0).tolist()));event('epoch',run=name,**hist[-1]);torch.save(dict(epoch=epoch,model=net.state_dict()),d/'last.pt')
  torch.save(net.state_dict(),d/'model.pt');delta=float(sum((net.encoder.state_dict()[k]-v).square().sum() for k,v in initial.items()).sqrt());write(d/'COMPLETE.json',dict(seed=seed,history=hist,encoder_delta=delta,seconds=time.perf_counter()-start,sha256=sha(d/'model.pt')))
  if name=='annotation_probe':write(U/'ANNOTATION_HOLDOUT.json',dict(rows=check_probe(net,allrows),scope=cfg['probe_scope'],not_used_for_selection=True))
 write(U/'TRAINING_COMPLETE.json',dict(seeds=cfg['seeds'],common_videos=899));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
