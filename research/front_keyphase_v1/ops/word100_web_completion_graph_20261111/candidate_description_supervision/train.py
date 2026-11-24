from core import *
import time,traceback
def main():
 assert not (W/'TRAINING_COMPLETE.json').exists();cfg=read(W/'FROZEN.json');assert all(sha(p)==h for p,h in cfg['pins'].items());torch.set_num_threads(4);torch.set_num_interop_threads(1)
 meta=read(W/'INPUTS_COMPLETE.json');a=np.load(meta['arrays']);assert sha(meta['arrays'])==cfg['data_sha256'];idx=a['train'];x=a['x'][idx];y=a['y'][idx];mask=a['mask'][idx];rows=[r for r,t in zip(meta['rows'],idx) if t];mu=x.mean(0);sd=np.maximum(x.std(0),cfg['std_floor']);weights=np.zeros_like(mask)
 for j in range(5):
  for h in range(3):
   for v in [0,1]:
    take=(mask[:,j,h]>0)&(y[:,j,h]==v);clips=sorted({r['clip_id'] for r,t in zip(rows,take) if t})
    for cid in clips:
     k=take & np.array([r['clip_id']==cid for r in rows]);weights[k,j,h]=1/(30*len(clips)*k.sum())
 assert np.isclose(weights.sum(),1);tx=torch.tensor(x);ty=torch.tensor(y);tw=torch.tensor(weights);out=[]
 np.savez_compressed(W/'TRAIN_WEIGHTS.npz',weights=weights,mean=mu,std=sd)
 for seed in cfg['seeds']:
  p=W/'runs'/str(seed);p.mkdir(parents=True,exist_ok=True);assert not (p/'model.pt').exists();torch.manual_seed(seed);net=CandidateHead(mu,sd);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);started=time.perf_counter();log=[]
  for step in range(cfg['steps']):
   opt.zero_grad();loss=(torch.nn.functional.binary_cross_entropy_with_logits(net(tx),ty,reduction='none')*tw).sum();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']);opt.step()
   if step%300==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())));print('B train',seed,step+1,float(loss.detach()),flush=True)
  net.eval();torch.save(net.state_dict(),p/'model.pt')
  with torch.inference_mode():pred=net(torch.tensor(a['x'])).sigmoid().numpy()
  np.savez_compressed(p/'predictions.npz',p=pred);out.append(dict(seed=seed,model=str(p/'model.pt'),model_sha256=sha(p/'model.pt'),predictions=str(p/'predictions.npz'),predictions_sha256=sha(p/'predictions.npz'),seconds=time.perf_counter()-started,steps=cfg['steps'],log=log))
 write(W/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=out,code_sha256=sha(__file__),parameters=sum(v.numel() for v in net.parameters()),feature_encoder_updated=False,web_changed=False))
if __name__=='__main__':
 try:main()
 except BaseException:(W/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
