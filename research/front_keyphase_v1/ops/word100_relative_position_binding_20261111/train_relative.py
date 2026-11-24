from relative_core import *
import os,traceback

def main():
 assert (U/'PREPARED.json').exists() and not (U/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);cfg=read(U/'FROZEN.json')
 for p,h in cfg['pins'].items():assert sha(p)==h,p
 write(U/'TRAIN_CODE_PIN.json',{str(U/'train_relative.py'):sha(U/'train_relative.py')})
 z=dict(np.load(U/'completion_train.npz'));t={k:torch.tensor(v) for k,v in z.items()};N=len(z['quality']);cnt=np.bincount(z['quality'][z['quality_mask']>0],minlength=4);qw=torch.tensor(sum(cnt)/np.maximum(cnt,1)/4,dtype=torch.float32);runs=[]
 for mode in cfg['completion_modes']:
  for seed in cfg['seeds']:
   dest=U/'runs'/f'{mode}_{seed}';assert not (dest/'model.pt').exists();dest.mkdir(parents=True,exist_ok=True);n=RelativeNet(seed,cfg['norm'],cfg['position_norm'],mode);opt=torch.optim.AdamW(n.parameters(),lr=cfg['completion_lr'],weight_decay=.0001);g=torch.Generator().manual_seed(seed);hist=[];start=time.perf_counter();write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING_COMPLETION',mode=mode,seed=seed,pid=os.getpid()))
   for epoch in range(cfg['completion_epochs']):
    losses=[]
    for ix in torch.randperm(N,generator=g).split(128):
     d,b,q=n(t['seq'][ix],t['mask'][ix],t['meta'][ix],t['position'][ix]);qm=t['quality_mask'][ix];bm=t['boundary_mask'][ix];dm=t['direction'][ix]!=-100
     loss=(F.cross_entropy(q,t['quality'][ix],weight=qw,reduction='none')*qm).sum()/qm.sum().clamp_min(1)+(F.binary_cross_entropy_with_logits(b,t['boundaries'][ix],reduction='none')*bm).sum()/bm.sum().clamp_min(1)+.5*F.cross_entropy(d,t['direction'][ix],ignore_index=-100,reduction='none').sum()/dm.sum().clamp_min(1)
     assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(n.parameters(),2);opt.step();losses.append(float(loss.detach()))
    hist.append(float(np.mean(losses)))
   torch.save(n.state_dict(),dest/'model.pt');r=dict(mode=mode,seed=seed,path=str(dest/'model.pt'),sha256=sha(dest/'model.pt'),seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in n.parameters()));write(dest/'TRAINING.json',dict(**r,loss=hist));runs.append(r);event('trained_completion',**r)
 # Train only a small sequence role residual on cached frozen public features.
 seqs=[]
 for r in read(U/'ROLE_INPUTS.json'):
  z=dict(np.load(r['path']));seqs.append(z)
 counts=np.bincount(np.concatenate([z['y'][z['w']>0] for z in seqs]),minlength=11);cw=torch.tensor(1/np.sqrt(np.maximum(counts,1)),dtype=torch.float32)
 for seed in cfg['seeds']:
  dest=U/'runs'/f'binding_{seed}';dest.mkdir(parents=True,exist_ok=True);assert not (dest/'model.pt').exists();start=time.perf_counter();oldnet=rc.RoleAdapter(seed);oldnet.load_state_dict(torch.load(R/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));oldnet.eval();oldnet.requires_grad_(False)
  bases=[]
  with torch.inference_mode():
   for z in seqs:bases.append(oldnet(torch.tensor(z['z'])).detach().clone())
  n=SequenceBinding(seed);opt=torch.optim.AdamW(n.parameters(),lr=cfg['role_lr'],weight_decay=.001);hist=[];write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING_BINDING',seed=seed,pid=os.getpid()))
  for epoch in range(cfg['role_epochs']):
   rng=np.random.default_rng(seed*1000+epoch);losses=[]
   for chunk in np.array_split(rng.permutation(len(seqs)),max(1,(len(seqs)+7)//8)):
    lengths=torch.tensor([len(seqs[i]['z']) for i in chunk]);maxlen=int(lengths.max());xx=torch.zeros(len(chunk),maxlen,684);yy=torch.full((len(chunk),maxlen),-100,dtype=torch.long);ww=torch.zeros(len(chunk),maxlen);base=torch.zeros(len(chunk),maxlen,11)
    for j,i in enumerate(chunk):
     z=seqs[i];L=len(z['z']);xx[j,:L]=torch.tensor(z['z']);yy[j,:L]=torch.tensor(np.where(z['w']>0,z['y'],-100));ww[j,:L]=torch.tensor(z['w']);base[j,:L]=bases[i]
    logits=n(xx,lengths)+base;logits[:,:,10]=-1e4;loss=(F.cross_entropy(logits.transpose(1,2),yy,weight=cw,ignore_index=-100,reduction='none')*ww).sum()/ww.sum().clamp_min(1)
    assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(n.parameters(),2);opt.step();losses.append(float(loss.detach()))
   hist.append(float(np.mean(losses)))
  torch.save(n.state_dict(),dest/'model.pt');r=dict(mode='binding',seed=seed,path=str(dest/'model.pt'),sha256=sha(dest/'model.pt'),seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in n.parameters()));write(dest/'TRAINING.json',dict(**r,loss=hist));runs.append(r);event('trained_binding',**r)
 write(U/'TRAINING_COMPLETE.json',dict(runs=runs,completion_samples=N,role_sequences=len(seqs),common100_frozen=True));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED_PENDING_EVALUATION',pid=None))

if __name__=='__main__':
 try:main()
 except BaseException:(U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
