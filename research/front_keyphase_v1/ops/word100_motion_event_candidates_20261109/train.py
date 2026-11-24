from core import *
import os,traceback
def run(rows,name,seed):
 dest=N/'runs'/name;assert not (dest/'COMPLETE.json').exists();dest.mkdir(parents=True,exist_ok=True)
 torch.manual_seed(seed);model=MotionProposal();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);by={}
 for r in rows:by.setdefault(r['clip_id'],[]).append(r)
 clips=sorted(by);stats_rows=[r for r in rows if r['kind'] in ['full_normal','actual_crop']];mean,std=norm_stats(stats_rows)
 counts=sum(np.bincount(r['target'][r['target']>=0],weights=r['weight'][r['target']>=0],minlength=11) for r in stats_rows);cw=np.sqrt(counts.sum()/np.maximum(counts,1));cw=torch.tensor(cw/cw.mean(),dtype=torch.float32)
 history=[];seen={};started=time.perf_counter()
 for epoch in range(100):
  rng=np.random.default_rng(seed*1000+epoch);order=rng.permutation(len(clips));losses=[]
  for i in range(0,len(order),6):
   chosen=[]
   for k in order[i:i+6]:
    rr=by[clips[int(k)]];normal=[r for r in rr if r['kind'] in ['full_normal','actual_crop','upper_normal']];available=normal if epoch%2==0 else rr;r=available[int(rng.integers(len(available)))];chosen.append(r);seen[r['kind']]=seen.get(r['kind'],0)+1
   g,lengths,y,w=batch(chosen,mean,std);logits=model(g,lengths);ce=nn.functional.cross_entropy(logits.transpose(1,2),y,ignore_index=-1,weight=cw,reduction='none');loss=(ce*w).sum()/w.sum().clamp_min(1);assert torch.isfinite(loss)
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();losses.append(float(loss.detach()))
  history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
  if (epoch+1)%25==0:event('epoch',run=name,epoch=epoch+1)
 torch.save(dict(model=model.state_dict(),mean=torch.tensor(mean),std=torch.tensor(std)),dest/'model.pt');write(dest/'COMPLETE.json',dict(seed=seed,fit_clips=clips,fit_people=sorted({r['person'] for r in rows}),initialization='random geometry only; no pretrained project weights',epochs=100,history=history,views_seen=seen,seconds=time.perf_counter()-started,sha256=sha(dest/'model.pt')));event('trained',run=name,seconds=time.perf_counter()-started)
def main():
 assert read(N/'INPUT_CHECK.json')['status']=='PASS';assert not (N/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 cfg=read(N/'FROZEN.json');rows=[r for r in load_rows() if r['role']=='fit' and r['kind'] in cfg['training_views']]
 write(N/'TRAIN_FROZEN.json',dict(code_sha256=sha(N/'train.py'),core_sha256=sha(N/'core.py'),input_sha256=sha(N/'INPUTS.json'),schedule=cfg,all_final_runs_before_any_eval=True));write(N/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid()))
 for fold in cfg['folds']:run([r for r in rows if r['person'] not in fold],'heldout_'+'_'.join(map(str,fold)),42)
 for seed in cfg['seeds']:run(rows,f'final_{seed}',seed)
 write(N/'TRAINING_COMPLETE.json',dict(status='SIX_FIXED_GEOMETRY_RUNS_COMPLETE',no_dev_used=True));write(N/'CURRENT_EXECUTION.json',dict(status='TRAINED_PENDING_FIXED_EVALUATION',training=None))
if __name__=='__main__':
 try:main()
 except BaseException:
  (N/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
