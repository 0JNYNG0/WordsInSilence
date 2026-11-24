from transfer_local_core import *
import time,traceback

def main():
 assert not (TQ/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 cfg=read(TQ/'FROZEN.json');meta=read(TQ/'INPUTS_COMPLETE.json');assert sha(meta['arrays'])==meta['arrays_sha256'];d=np.load(meta['arrays']);old=np.load(read(N/'INPUTS_COMPLETE.json')['arrays']);assert all(np.array_equal(d[k],old[k]) for k in ['y','mask','train'])
 for p,h in meta['pins'].items():assert sha(p)==h
 tr=d['train'];weights=np.load(Q/'TRAIN_WEIGHTS.npz')['weights'];assert weights.shape==d['y'][tr].shape
 xx=np.concatenate([d['x'][tr],d['x_reverse'][tr]]);yy=np.concatenate([d['y'][tr]]*2);ww=np.concatenate([weights]*2)*.5
 mean=xx.mean(0);std=np.maximum(xx.std(0),.05);x=torch.tensor(xx);y=torch.tensor(yy);w=torch.tensor(ww)
 write(TQ/'PRETRAIN_VERIFICATION.json',dict(at=stamp(),same_labels=True,normal_and_reversed_windows=int(len(x)),trained_only_on_people=[2,4,5,6],probe_people=[8,9],mean_std_fit_only=True,weights_sum=float(ww.sum()),encoder_frozen=True,private_access=False,final8_access=False))
 runs=[]
 for seed in cfg['seeds']:
  dst=TQ/'runs'/str(seed);dst.mkdir(parents=True,exist_ok=True);assert not(dst/'model.pt').exists();torch.manual_seed(seed);net=TransferHead(mean,std);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);t=time.perf_counter();log=[]
  for step in range(cfg['steps']):
   opt.zero_grad();loss=(torch.nn.functional.binary_cross_entropy_with_logits(net(x),y,reduction='none')*w).sum();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
   if step%300==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())));print('transfer',seed,step+1,float(loss.detach()),flush=True)
  net.eval();torch.save(net.state_dict(),dst/'model.pt')
  with torch.inference_mode():p=net(torch.tensor(d['x'])).sigmoid().numpy();pr=net(torch.tensor(d['x_reverse'])).sigmoid().numpy()
  np.savez_compressed(dst/'predictions.npz',p=p,p_reverse=pr);runs.append(dict(seed=seed,model=str(dst/'model.pt'),model_sha256=sha(dst/'model.pt'),predictions=str(dst/'predictions.npz'),predictions_sha256=sha(dst/'predictions.npz'),seconds=time.perf_counter()-t,log=log));write(TQ/'TRAINING_PROGRESS.json',dict(at=stamp(),completed=runs))
 write(TQ/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,parameters=sum(p.numel() for p in net.parameters()),frozen_common100=True,web_changed=False))
if __name__=='__main__':
 try:main()
 except BaseException:(TQ/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
