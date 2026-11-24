from motion_core import *
import traceback,os
def main():
 cfg=read(U/'FROZEN.json');torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 for p,h in {**cfg['pins'],**cfg['protections'],**read(U/'CODE_PINS.json')}.items():assert sha(p)==h,p
 samples=read(U/'TRAIN_SAMPLES.json')['rows'];start=time.perf_counter()
 for name,seed,probe in [('probe',42,True)]+[(str(s),s,False) for s in cfg['seeds']]:
  dest=U/'runs'/name;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'COMPLETE.json').exists():continue
  rows=[r for r in samples if not probe or r['person'] not in cfg['probe_ids']];X=torch.tensor([r['g'] for r in rows]);Y=torch.tensor([r['label'] for r in rows]);torch.manual_seed(seed);net=MotionNet();initial={k:v.clone() for k,v in net.heads().items()};opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=.01);weights=(1/torch.bincount(Y,minlength=3).float().clamp_min(1)).sqrt();hist=[];write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid(),run=name))
  for epoch in range(cfg['epochs']):
   rng=np.random.default_rng(seed*1000+epoch);ii=rng.permutation(len(rows));losses=[];net.train()
   for j in range(0,len(ii),cfg['batch']):
    idx=ii[j:j+cfg['batch']];loss=F.cross_entropy(net.predict(X[idx]),Y[idx],weight=weights);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],2);opt.step();assert torch.isfinite(loss);losses.append(float(loss.detach()))
   hist.append(float(np.mean(losses)))
  net.eval();torch.save(net.heads(),dest/'model.pt');delta=float(sum((net.heads()[k]-v).square().sum() for k,v in initial.items()).sqrt());write(dest/'COMPLETE.json',dict(seed=seed,samples=len(rows),epochs=cfg['epochs'],history=hist,delta=delta,sha256=sha(dest/'model.pt')))
  if probe:
   held=[r for r in samples if r['person'] in cfg['probe_ids'] and not r['reverse'] and r['view']=='full' and r['a']==r['onset'][0] and r['b']==r['offset'][1]]
   with torch.inference_mode():p=net.predict(torch.tensor([r['g'] for r in held])).softmax(-1).numpy()
   write(U/'PROBE.json',dict(rows=[dict(**{k:v for k,v in r.items() if k!='g'},pred=int(q.argmax()),confidence=float(q.max())) for r,q in zip(held,p)],scope='New fine-label loss holds out03/07; old pretrained model had earlier FIT exposure. Not globally unseen-person validation.'))
  event('trained',run=name,delta=delta)
 write(U/'TRAINING_COMPLETE.json',dict(seeds=cfg['seeds'],seconds=time.perf_counter()-start,common100_encoder_changed=False));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
