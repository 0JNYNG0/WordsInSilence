from role_core import *
import os,traceback

def main():
 assert not (U/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 cfg=read(U/'FROZEN.json');records=read(U/'TRAIN_INPUTS_COMPLETE.json')['rows']
 for p,h in cfg['pins'].items():assert sha(p)==h,p
 arrays=[]
 for r in records:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);arrays.append((r,{k:z[k] for k in z.files}))
 X=torch.tensor(np.concatenate([z['z'][z['w']>0] for r,z in arrays]));Y=torch.tensor(np.concatenate([z['y'][z['w']>0] for r,z in arrays]),dtype=torch.long);W=torch.tensor(np.concatenate([z['w'][z['w']>0] for r,z in arrays]));Q=torch.tensor(np.concatenate([z['quality'] for r,z in arrays]));QY=torch.tensor(np.concatenate([z['quality_y'] for r,z in arrays]),dtype=torch.long);PZ=torch.tensor(np.concatenate([z['pair'] for r,z in arrays]));PY=torch.tensor(np.concatenate([z['pair_y'] for r,z in arrays]),dtype=torch.long)
 assert set(Y.tolist())<=set(range(10));assert torch.isfinite(X).all();assert len(Q)>0 and len(PZ)>0
 cw=(torch.bincount(Y,minlength=11).float().clamp_min(1).reciprocal().sqrt());qw=torch.bincount(QY,minlength=4).float().clamp_min(1).reciprocal().sqrt();pw=torch.bincount(PY,minlength=3).float().clamp_min(1).reciprocal().sqrt()
 write(U/'PRETRAIN_CHECK.json',dict(passed=True,role_samples=len(X),quality_samples=len(Q),pair_samples=len(PZ),role_classes=torch.bincount(Y,minlength=11).tolist(),quality_classes=torch.bincount(QY,minlength=4).tolist(),pair_classes=torch.bincount(PY,minlength=3).tolist(),source='FIT only, augmented variants not independent clips',old_completed_evaluation_in_training=False,common100_frozen=True))
 for seed in cfg['seeds']:
  d=U/'runs'/str(seed);d.mkdir(parents=True,exist_ok=True);assert not (d/'model.pt').exists();net=RoleAdapter(seed);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01);hist=[];start=time.perf_counter();write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid(),seed=seed))
  for epoch in range(cfg['epochs']):
   rng=np.random.default_rng(seed*1000+epoch);perm=rng.permutation(len(X));losses=[]
   for step,i in enumerate(range(0,len(X),cfg['batch'])):
    ix=perm[i:i+cfg['batch']];xx=X[ix].clone()
    if step%2==0:
     for j in range(3):xx[:,j*228+164:(j+1)*228]=0
    logits=net(xx);logits[:,10]=-1e4
    rl=(F.cross_entropy(logits,Y[ix],weight=cw,reduction='none')*W[ix]).sum()/W[ix].sum().clamp_min(1)
    iq=rng.choice(len(Q),min(128,len(Q)),False);ip=rng.choice(len(PZ),min(128,len(PZ)),False);ql=F.cross_entropy(net.quality(Q[iq]),QY[iq],weight=qw);pl=F.cross_entropy(net.pair(PZ[ip]),PY[ip],weight=pw);loss=rl+ql+pl
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();assert torch.isfinite(loss);losses.append(float(loss.detach()))
   hist.append(float(np.mean(losses)))
  torch.save(net.state_dict(),d/'model.pt');write(d/'COMPLETE.json',dict(seed=seed,epochs=cfg['epochs'],history=hist,seconds=time.perf_counter()-start,sha256=sha(d/'model.pt')));event('trained',seed=seed,seconds=time.perf_counter()-start,loss=hist[-1])
 write(U/'TRAINING_COMPLETE.json',dict(seeds=cfg['seeds'],frozen_encoder=True,trainable='role context adaptor, existing quality/pair heads; no common encoder/shape/direction weights updated'));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
