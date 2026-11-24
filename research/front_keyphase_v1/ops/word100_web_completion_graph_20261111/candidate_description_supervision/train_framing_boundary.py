from framing_boundary_core import *
import time,traceback

def main():
 assert not (R/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);cfg=read(R/'FROZEN.json');meta=read(R/'INPUTS_COMPLETE.json');assert sha(meta['arrays'])==meta['arrays_sha256'];d=np.load(meta['arrays']);rows=meta['rows']
 for p,h in {**cfg['pins'],**meta['pins']}.items():assert sha(p)==h,p
 tr=d['train'];fullw=balanced_weights(d['y'],d['mask'],rows,tr);oldw=np.load(Q/'TRAIN_WEIGHTS.npz')['weights'];oldn=int(meta['old_windows']);assert not d['crop'][:oldn].any();assert np.array_equal(d['y'][:oldn],np.load(read(N/'INPUTS_COMPLETE.json')['arrays'])['y']);runs=[];allweights={}
 for condition in CONDITIONS:
  crop=condition!='boundary_only';boundary=condition!='crop_local';take=tr&(d['crop']|~d['crop']) if crop else tr&~d['crop'];w0=fullw if crop else oldw;feature='x_boundary' if boundary else 'x_local';xx=np.concatenate([d[feature][take],d[feature+'_reverse'][take]]);yy=np.concatenate([d['y'][take]]*2);ww=np.concatenate([w0]*2)*.5;assert len(xx)==len(ww) and np.isclose(ww.sum(),1);allweights[condition]=ww
  mean=xx.mean(0);std=np.maximum(xx.std(0),.05);x=torch.tensor(xx);y=torch.tensor(yy);w=torch.tensor(ww)
  for seed in cfg['seeds']:
   dst=R/'runs'/condition/str(seed);dst.mkdir(parents=True,exist_ok=True);assert not (dst/'model.pt').exists();torch.manual_seed(seed);net=TransferHead(mean,std);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);t=time.perf_counter();log=[]
   for step in range(cfg['steps']):
    opt.zero_grad();loss=(torch.nn.functional.binary_cross_entropy_with_logits(net(x),y,reduction='none')*w).sum();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
    if step%300==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())));print(condition,seed,step+1,float(loss.detach()),flush=True)
   net.eval();torch.save(net.state_dict(),dst/'model.pt')
   with torch.inference_mode():p=net(torch.tensor(d[feature])).sigmoid().numpy();pr=net(torch.tensor(d[feature+'_reverse'])).sigmoid().numpy()
   np.savez_compressed(dst/'predictions.npz',p=p,p_reverse=pr);runs.append(dict(condition=condition,seed=seed,model=str(dst/'model.pt'),model_sha256=sha(dst/'model.pt'),predictions=str(dst/'predictions.npz'),predictions_sha256=sha(dst/'predictions.npz'),seconds=time.perf_counter()-t,log=log));write(R/'TRAINING_PROGRESS.json',dict(at=stamp(),completed=runs))
 np.savez_compressed(R/'TRAIN_WEIGHTS.npz',**allweights);write(R/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,frozen_common100=True,web_changed=False,fit_persons=[2,4,5,6],probe_persons=[8,9],training_is_normal_and_attempt_axis_reversal=True))
if __name__=='__main__':
 try:main()
 except BaseException:(R/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
