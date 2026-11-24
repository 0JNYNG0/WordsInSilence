from unit import *
import os,traceback
NORMAL=['full_normal','upper_normal','speed075','speed125','gap2']
ATTEMPT=['close_R','close_L','close_both','reverse_main','close_reverse','omit_final_event']
def run(rows,condition,seed,split):
 dest=U/'runs'/f'{split}_{condition}_{seed}';assert not (dest/'COMPLETE.json').exists();dest.mkdir(parents=True,exist_ok=True)
 torch.manual_seed(seed);net=init_model();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01);clips=sorted({r['clip_id'] for r in rows});by={(r['clip_id'],r['kind']):r for r in rows};normal=[r for r in rows if r['kind']=='full_normal']
 count=sum(np.bincount(r['sy'][r['sy']>=0],weights=r['sw'][r['sy']>=0],minlength=14) for r in normal);cw=np.sqrt(count.sum()/np.maximum(count,1));cw=torch.tensor(cw/cw.mean(),dtype=torch.float32)
 history=[];seen={};start=time.perf_counter()
 for epoch in range(100):
  rng=np.random.default_rng(seed+epoch);ix=rng.permutation(len(clips));losses=[]
  for offset in range(0,len(ix),6):
   selected=[]
   for j in ix[offset:offset+6]:
    cid=clips[int(j)];kinds=ATTEMPT if condition=='attempt_supervised' and epoch%2 else NORMAL;kinds=[k for k in kinds if (cid,k) in by];kind=kinds[int(rng.integers(len(kinds)))];selected.append(by[(cid,kind)]);seen[kind]=seen.get(kind,0)+1
   b=batch(selected);expected=torch.randint(0,2,b['y'].shape).float();target=(expected!=b['y']).float();ph,df,pr=net(b['h'],b['p'],b['g'],b['word'],b['valid'],expected)
   ce=nn.functional.cross_entropy(ph.transpose(1,2),b['sy'],ignore_index=-1,weight=cw,reduction='none');phase=(ce*b['sw']).sum()/b['sw'].sum().clamp_min(1)
   shape=(nn.functional.binary_cross_entropy_with_logits(df,target,reduction='none')*b['ym']).sum()/b['ym'].sum().clamp_min(1)
   presence=(nn.functional.binary_cross_entropy_with_logits(pr,b['present'],reduction='none')*b['pmask']).sum()/b['pmask'].sum().clamp_min(1)
   loss=phase+.5*shape+.2*presence;assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();losses.append(float(loss.detach()))
  history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
  if (epoch+1)%25==0:event('epoch',run=dest.name,epoch=epoch+1)
 torch.save(net.state_dict(),dest/'model.pt');write(dest/'COMPLETE.json',dict(condition=condition,seed=seed,split=split,fit_clips=clips,epochs=100,history=history,seen=seen,seconds=time.perf_counter()-start,sha256=sha(dest/'model.pt')));event('trained',run=dest.name,seconds=time.perf_counter()-start)
def main():
 assert not (U/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 rows=[r for r in load_rows() if r['role']=='fit'];cfg=read(U/'FROZEN.json')
 write(U/'TRAIN_FROZEN.json',dict(conditions=cfg['conditions'],schedule='100 epochs,6clips,both conditions same steps; alternating normal vs counterfactual only in attempt condition',initial_model=str(D/'runs/42/model.pt'),primary='final_attempt_supervised_42',internal='FIT01-06 train/FIT07-09 diagnostic once; no hyperparameter or architecture selection',seeds=cfg['seeds'],code_sha256=sha(U/'train.py')))
 write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid()))
 for condition in cfg['conditions']:run([r for r in rows if r['person']<=6],condition,42,'inner')
 for condition in cfg['conditions']:
  for seed in cfg['seeds']:run(rows,condition,seed,'final')
 write(U/'TRAINING_COMPLETE.json',dict(status='EIGHT_FIXED_RUNS_COMPLETE',no_DEV_used=True,base_unchanged=sha(prior.CHECKPOINT)==cfg['base_sha256']));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED_PENDING_EVAL',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
