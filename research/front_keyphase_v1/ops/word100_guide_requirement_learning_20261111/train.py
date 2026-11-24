from requirement_core import *
import time,os,traceback
def main():
 assert not (U/'TRAINING_COMPLETE.json').exists()
 cfg=read(U/'FROZEN.json');assert read(U/'PRETRAIN_CHECK.json')['pass_']
 assert all(sha(p)==s for p,s in cfg['pins'].items())
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 z=np.load(U/'train.npz');meta=read(U/'TRAIN_MANIFEST.json');idx=np.flatnonzero(z['mask']>0)
 data={k:torch.tensor(z[k]) for k in ['ob','req','base','y']}
 # Equal task contribution prevents many shape cells overwhelming direction/order.
 counts={t:sum(meta[i]['task']==t for i in idx) for t in ['shape','direction','order']}
 weight=torch.tensor([len(idx)/(3*counts[m['task']]) for m in meta],dtype=torch.float32)
 records=[]
 for seed in cfg['seeds']:
  torch.manual_seed(seed);rng=np.random.default_rng(seed);n=RequirementAdapter(np.array(cfg['mean'],np.float32),np.array(cfg['std'],np.float32));opt=torch.optim.AdamW(n.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);history=[];start=time.perf_counter()
  write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',seed=seed,pid=os.getpid()))
  for epoch in range(cfg['epochs']):
   shuffled=rng.permutation(idx);losses=[]
   for j in range(0,len(idx),cfg['batch']):
    ii=torch.tensor(shuffled[j:j+cfg['batch']]);pred=n(data['ob'][ii],data['req'][ii],data['base'][ii]);loss=(nn.functional.binary_cross_entropy_with_logits(pred,data['y'][ii],reduction='none')*weight[ii]).mean()
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(n.parameters(),2);opt.step();losses.append(float(loss.detach()))
   history.append(float(np.mean(losses)))
  p=U/'runs'/str(seed)/'model.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(n.state_dict(),p)
  r=dict(seed=seed,path=str(p),sha256=sha(p),seconds=time.perf_counter()-start,loss=history,parameters=sum(p.numel() for p in n.parameters()));records.append(r);write(U/'TRAINING_PROGRESS.json',records);print(seed,history[0],history[-1],r['seconds'],flush=True)
 write(U/'TRAINING_COMPLETE.json',dict(rows=records,base_frozen=True));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED_PENDING_EVALUATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
