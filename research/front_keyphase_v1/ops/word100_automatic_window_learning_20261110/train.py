from window_core import *
import os,traceback
def main():
 cfg=read(U/'FROZEN.json');norm=read(U/'NORMALIZATION.json')
 for p,h in {**cfg['pins'],**cfg['code_pins'],**cfg['protections']}.items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 rr=load_rows();fit=[r for r in rr if r['role']=='fit'];fine=[r for r in fit if r['pairs'] or r['orders']];autos={r['clip_id']:r['auto_span'] for r in read(U/'FIT_WINDOW_AUDIT.json')['rows']}
 for mode in cfg['conditions']:
  for seed in cfg['seeds']:
   dst=U/'runs'/f'{mode}_{seed}';dst.mkdir(parents=True,exist_ok=True)
   if (dst/'COMPLETE.json').exists():continue
   assert not (dst/'last.pt').exists(),'Interrupted checkpoint needs review'
   torch.manual_seed(seed);net=WindowNet(mode,norm);initial={k:v.clone() for k,v in net.encoder.state_dict().items()};opt=torch.optim.AdamW([dict(params=net.encoder.parameters(),lr=cfg['lr']),dict(params=[p for n,p in net.named_parameters() if not n.startswith('encoder.')],lr=cfg['head_lr'])],weight_decay=cfg['weight_decay']);start=time.perf_counter();history=[];counts=Counter();seen=set()
   write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',mode=mode,seed=seed,pid=os.getpid()))
   for ep in range(1,cfg['epochs']+1):
    rng=np.random.default_rng(seed*1000+ep);perm=rng.permutation(len(fit));losses=[];epseen=set();net.train()
    for k in range(cfg['steps_per_epoch']):
     sample=[fit[int(i)] for i in perm[k*24:(k+1)*24]];view='upper' if (ep+k)%2 else 'full';gg,fh,l=batch_g(sample,'g_'+view);hh,v=net.encode(gg,l);wl=TF.cross_entropy(net.classify_words(hh,fh,l,v),torch.tensor([r['class_index'] for r in sample]));ml=[];cl=[];ol=[];al=[]
     for ri in rng.choice(len(fine),cfg['supervised_clips'],replace=False):
      r=fine[int(ri)];rev=bool(rng.integers(0,2));x,ids=transformed(r,view,rev);g=geometry(x,r['timestamp_us']);h,_=net.encode(torch.tensor(g[None]),torch.tensor([len(g)]));h=h[0]
      choices=list(r['pairs']);rng.shuffle(choices)
      for p in choices[:3]:
       a,b,_=map_interval(ids,p['a'],p['b']);left=int(rng.choice(cfg['padding_frames']));right=int(rng.choice(cfg['padding_frames']));a=max(0,a-left);b=min(len(x)-1,b+right)
       if not rev and p['axis']==2 and r['clip_id'] in autos and k%3==0:a,b=autos[r['clip_id']];counts['automatic_fit_window']+=1
       y,core,intervals=window_truth(r,ids,a,b,p['axis'],p.get('moving','R'));q=relations(x,p['axis'],p.get('moving','R'),p.get('anchor','L'));pred,cp,att,valid=net.movement(h,torch.tensor(q),a,b,p['axis'])
       if float(valid.float().mean())<.8 or int(valid.sum())<4:counts['unobserved_excluded_from_supervised_loss']+=1;continue
       ml.append(TF.cross_entropy(pred[None],torch.tensor([y])));bg=frame_background(r,ids,a,b);weight=np.maximum(core*.2,bg);weight[core>0]=.2;mask=torch.tensor(weight)*valid
       if mask.sum()>0:cl.append((TF.binary_cross_entropy_with_logits(cp,torch.tensor(core),reduction='none')*mask).sum()/mask.sum())
       counts['motion_class_'+str(y)]+=1;counts['reversed_window' if rev else 'normal_window']+=1
      oo=list(r['orders']);rng.shuffle(oo)
      for p in oo[:2]:
       aa=int(np.flatnonzero(ids==p['a'])[0]);bb=int(np.flatnonzero(ids==p['b'])[0]);a=max(0,min(aa,bb)-int(rng.choice(cfg['padding_frames'])));b=min(len(x)-1,max(aa,bb)+int(rng.choice(cfg['padding_frames'])));pred,att=net.ordering(h,a,b,p['guide_a'],p['guide_b']);ol.append(TF.binary_cross_entropy_with_logits(pred,torch.tensor(float(aa<bb))));al.append(-.5*(att[aa-a,0].clamp_min(1e-8).log()+att[bb-a,1].clamp_min(1e-8).log()));counts['reversed_order' if aa>bb else 'normal_order']+=1
     zero=wl*0;mean=lambda vals:torch.stack(vals).mean() if vals else zero
     loss=.2*wl+mean(ml)+.2*mean(cl)+.3*mean(ol)+.1*mean(al);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss)
     epseen.update(r['clip_id'] for r in sample);seen.update(epseen);losses.append([float(t.detach()) for t in [loss,wl,mean(ml),mean(ol)]])
    assert len(epseen)==899;history.append(dict(epoch=ep,all100_clips_seen=len(epseen),loss=np.mean(losses,axis=0).tolist(),seconds=time.perf_counter()-start))
    if ep%3==0:torch.save(dict(model=net.state_dict(),epoch=ep,opt=opt.state_dict(),history=history),dst/'last.pt');event('epoch',mode=mode,seed=seed,**history[-1])
   torch.save(net.state_dict(),dst/'model.pt');delta=float(sum((net.encoder.state_dict()[k]-v).square().sum() for k,v in initial.items()).sqrt());assert delta>0 and len(seen)==899
   write(dst/'COMPLETE.json',dict(mode=mode,seed=seed,seen=sorted(seen),history=history,counts=dict(counts),encoder_delta=delta,seconds=time.perf_counter()-start,sha256=sha(dst/'model.pt'),shape_base_unchanged=sha(P/'runs/coupled_42/model.pt')==cfg['pins'][str(P/'runs/coupled_42/model.pt')],development_training=False));event('trained',mode=mode,seed=seed,seconds=time.perf_counter()-start)
 write(U/'TRAINING_COMPLETE.json',dict(runs=[str(p) for p in (U/'runs').glob('*/COMPLETE.json')]));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED',pid=None))
if __name__=='__main__':
 from collections import Counter
 try:main()
 except BaseException:
  (U/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
