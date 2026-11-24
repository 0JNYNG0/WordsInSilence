from local_sequence_core import *
import time,traceback

def main():
 assert not (Q/'TRAINING_COMPLETE.json').exists();Q.mkdir(exist_ok=True);torch.set_num_threads(4);torch.set_num_interop_threads(1);meta=read(N/'INPUTS_COMPLETE.json');data=np.load(meta['arrays']);rows=meta['rows'];train=data['train'];idx=np.flatnonzero(train);sources={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['role']=='fit' and r['clip_id'] in {r['clip_id'] for r in rows}};seq={};pins={}
 for r in rows:
  key=r['clip_id'],r['view']
  if key in seq:continue
  assert sha(r['cache'])==r['cache_sha256'];raw=sources[r['clip_id']];assert sha(raw['path'])==raw['sha256'];z=np.load(r['cache'])['z'];x=np.load(raw['path'])['x_'+r['view']];seq[key]=local_input(z,x);pins[r['cache']]=r['cache_sha256'];pins[raw['path']]=raw['sha256']
 assert len(seq)==48;cfg=dict(at=stamp(),hypothesis='Full-clip context and pooled moments can hide within-candidate start/end and movement. Read only actual local relation/hand samples, preserving all frames.',conditions=['local_sequence'],seeds=[42,43,44],steps=2400,batch_size=128,lr=.0003,weight_decay=.01,reverse_augmentation=.5,reverse_meaning='Same comparison-axis attempt, never a correct-sign target; output direction still judged by frozen semantic head',inputs='100 current normalized wrist/body relations + 84 local hand coordinates + 42 visibility masks; no outside latent',frame_subsampling=False,padding_masked=True,source_split=meta['source_person_split'],upstream_probe_exposed=True,semantic_completion=None,pins={**pins,str(N/'INPUTS_COMPLETE.json'):sha(N/'INPUTS_COMPLETE.json'),str(Path(__file__)):sha(__file__),str(W/'local_sequence_core.py'):sha(W/'local_sequence_core.py')});write(Q/'FROZEN.json',cfg)
 y=data['y'][train];mask=data['mask'][train];rr=[rows[i] for i in idx];weights=np.zeros_like(mask)
 for j in range(5):
  for h in range(3):
   for v in [0,1]:
    take=(mask[:,j,h]>0)&(y[:,j,h]==v);clips=sorted({r['clip_id'] for r,t in zip(rr,take) if t})
    for cid in clips:
     k=take&np.array([r['clip_id']==cid for r in rr]);weights[k,j,h]=1/(30*len(clips)*k.sum())
 assert np.isclose(weights.sum(),1);np.savez_compressed(Q/'TRAIN_WEIGHTS.npz',weights=weights);write(Q/'PRETRAIN_VERIFICATION.json',dict(at=stamp(),train_people=sorted({r['person'] for r in rr}),probe_people=sorted({r['person'] for r,t in zip(rows,train) if not t}),clips=len(sources),samples=len(rows),missing_visibility_preserved=True,all_real_frames_in_window_retained=True,private=False,final8=False))
 def get(i,rev=False):
  r=rows[int(i)];a,b=r['span'];v=seq[r['clip_id'],r['view']][a:b+1];return v[::-1].copy() if rev else v
 runs=[]
 for seed in cfg['seeds']:
  dst=Q/'runs'/str(seed);dst.mkdir(parents=True,exist_ok=True);assert not (dst/'model.pt').exists();torch.manual_seed(seed);rng=np.random.default_rng(seed);net=LocalCandidate();opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);t=time.perf_counter();log=[]
  for step in range(cfg['steps']):
   ix=rng.integers(0,len(idx),cfg['batch_size']);xb,lens=batch([get(idx[i],rng.random()<.5) for i in ix]);target=torch.tensor(y[ix]);wt=torch.tensor(weights[ix]*len(idx)/len(ix));opt.zero_grad();loss=(torch.nn.functional.binary_cross_entropy_with_logits(net(xb,lens),target,reduction='none')*wt).sum();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
   if step%400==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())));print('local',seed,step+1,float(loss.detach()),flush=True)
  net.eval();torch.save(net.state_dict(),dst/'model.pt');pred=[]
  for off in range(0,len(rows),128):
   xb,lens=batch([get(i) for i in range(off,min(len(rows),off+128))])
   with torch.inference_mode():pred.extend(net(xb,lens).sigmoid().numpy())
  np.savez_compressed(dst/'predictions.npz',p=np.array(pred));runs.append(dict(seed=seed,model=str(dst/'model.pt'),model_sha256=sha(dst/'model.pt'),predictions=str(dst/'predictions.npz'),predictions_sha256=sha(dst/'predictions.npz'),seconds=time.perf_counter()-t,log=log));write(Q/'TRAINING_PROGRESS.json',dict(at=stamp(),completed=runs))
 write(Q/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,upstream_changed=False,web_changed=False,parameters=sum(p.numel() for p in net.parameters())))
if __name__=='__main__':
 try:main()
 except BaseException:Q.mkdir(exist_ok=True);(Q/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
