from core import *
import traceback,os
def run(name,seed,excluded):
 dest=E/'runs'/name;assert not (dest/'COMPLETE.json').exists();dest.mkdir(parents=True,exist_ok=True)
 m,ck=encoder(name);records=[];by={};torch.manual_seed(seed);model=ProgressHead();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 with torch.inference_mode():
  for r in read(E/'INPUTS.json')['rows']:
   assert sha(r['path'])==r['sha256'];z=np.load(r['path']);f=features(z['raw'],z['timestamp_us'],m,ck);p=E/'features'/name/f"{r['clip_id']}_{r['kind']}.npz";p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,features=f)
   rr=dict(r,x=f,y=z['target'],w=z['weight']);records.append(rr)
   if r['person'] not in excluded:by.setdefault(r['clip_id'],[]).append(rr)
 clips=sorted(by);normal=[r for rr in by.values() for r in rr if r['kind']=='full_normal'];cnt=sum(np.bincount(r['y'][r['y']>=0],weights=r['w'][r['y']>=0],minlength=19) for r in normal);cw=np.sqrt(cnt.sum()/np.maximum(cnt,1));cw=torch.tensor(cw/cw.mean(),dtype=torch.float32)
 seen={};history=[];start=time.perf_counter()
 for ep in range(100):
  rng=np.random.default_rng(seed*1000+ep);order=rng.permutation(len(clips));losses=[]
  for k in range(0,len(order),6):
   rr=[]
   for i in order[k:k+6]:
    options=by[clips[int(i)]];options=[r for r in options if r['kind'] in ['full_normal','actual_crop','upper_normal']] if ep%2==0 else options;r=options[int(rng.integers(len(options)))];rr.append(r);seen[r['kind']]=seen.get(r['kind'],0)+1
   x=torch.tensor(np.concatenate([r['x'] for r in rr]));y=torch.tensor(np.concatenate([r['y'] for r in rr]));w=torch.tensor(np.concatenate([r['w'] for r in rr]));pred=model(x);ce=nn.functional.cross_entropy(pred,y,weight=cw,ignore_index=-1,reduction='none');loss=(ce*w).sum()/w.sum().clamp_min(1);assert torch.isfinite(loss)
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();losses.append(float(loss.detach()))
  history.append(float(np.mean(losses)))
  if (ep+1)%25==0:event('epoch',run=name,epoch=ep+1)
 torch.save(model.state_dict(),dest/'model.pt');write(dest/'COMPLETE.json',dict(seed=seed,excluded_people=excluded,fit_clips=clips,encoder_sha256=sha(N/'runs'/name/'model.pt'),head_sha256=sha(dest/'model.pt'),epochs=100,loss=history,views=seen,seconds=time.perf_counter()-start))
 event('trained',run=name)
def main():
 assert not (E/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);cfg=read(E/'FROZEN.json')
 write(E/'TRAIN_CODE_PINS.json',{str(p):sha(p) for p in [E/'core.py',E/'train.py',E/'prepare.py',E/'labels.py',E/'FROZEN.json']});write(E/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid()))
 for fold in cfg['folds']:run('heldout_'+'_'.join(map(str,fold)),42,fold)
 for seed in cfg['seeds']:run(f'final_{seed}',seed,[])
 write(E/'TRAINING_COMPLETE.json',dict(status='SIX_FIXED_HEAD_RUNS_COMPLETE',dev_used=False));write(E/'CURRENT_EXECUTION.json',dict(status='TRAINED_PENDING_FIXED_COMPARISON'))
if __name__=='__main__':
 try:main()
 except BaseException:
  (E/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
