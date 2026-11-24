from core import *
from collections import Counter,defaultdict
import copy,time,traceback
R=W/'boundary_direction_repair'
def main():
 assert not (R/'TRAINING_COMPLETE.json').exists();meta=read(R/'INPUTS_COMPLETE.json');assert all(sha(p)==h for p,h in meta['pins'].items());a=np.load(meta['arrays']);teacher=np.load(meta['teacher']);assert sha(meta['arrays'])==meta['arrays_sha256'] and sha(meta['teacher'])==meta['teacher_sha256'];rows=meta['rows'];assert np.isfinite(a['x']).all();assert {r['source_video_sha256'] for r in rows if r['split']=='train'}.isdisjoint(r['source_video_sha256'] for r in rows if r['split']=='probe')
 role=load('_boundary_finetune_role',O/'word100_candidate_role_learning_20261111/role_core.py');base=role.frozen_model().boundary;state=copy.deepcopy(base.state_dict());torch.set_num_threads(4);torch.set_num_interop_threads(1)
 cells=[(i,h) for i,r in enumerate(rows) if r['split']=='train' for h in range(3) if a['mask'][i,h]];counts=Counter((rows[i]['family'],h,int(a['y'][i,h]),rows[i]['clip_id']) for i,h in cells);clips=Counter((f,h,c) for f,h,c,cid in counts);classes=defaultdict(set)
 for f,h,c,cid in counts:classes[f,h].add(c)
 weights=np.array([1/(len(classes)*len(classes[rows[i]['family'],h])*clips[rows[i]['family'],h,int(a['y'][i,h])]*counts[rows[i]['family'],h,int(a['y'][i,h]),rows[i]['clip_id']]) for i,h in cells]);weights/=weights.sum();ii=torch.tensor([i for i,h in cells]);hh=torch.tensor([h for i,h in cells]);targets=torch.tensor([a['y'][i,h] for i,h in cells]);x=torch.tensor(a['x']);tx=torch.tensor(teacher['x']);tp=torch.tensor(teacher['p']);tc=read(R/'PRESERVATION_META.json')['clip_ids'];cnt=Counter(tc);tw=torch.tensor([1/(len(cnt)*cnt[c]) for c in tc],dtype=torch.float64)
 cfg=dict(at=stamp(),seeds=[42,43,44],steps=1200,batch=256,teacher_batch=512,lr=.0003,weight_decay=.01,gradient_clip=2.,preservation_weight=.2,initialization='Same frozen boundary head; random minibatch order differs by seed',supervision='Reviewed progression/nonmotion and uncertain boundary-band membership; no exact physical-boundary labels invented',loss='Class/family/head/source-balanced sampled BCE plus .2 normal FIT teacher BCE',proposal_rule='Unchanged candidate_spans thresholds. Union old and new proposals; no forced desired-order/direction filter. Rank with frozen normal and reverse-augmented candidate heads, presence>=.5.',runtime=False,pins={**meta['pins'],str(R/'INPUTS_COMPLETE.json'):sha(R/'INPUTS_COMPLETE.json'),str(Path(__file__)):sha(__file__),str(role.J/'runs/42/model.pt'):sha(role.J/'runs/42/model.pt')})
 assert not (R/'FROZEN.json').exists();write(R/'FROZEN.json',cfg);write(R/'PRETRAIN_VERIFICATION.json',dict(at=stamp(),status='PASS_LIMITED_BAND_SUPERVISION',source_split=True,train_cells=len(cells),preservation_clips=len(cnt),no_dense_motion_truth=True,private=False,model_only_boundary=True))
 results=[]
 for seed in cfg['seeds']:
  torch.manual_seed(seed);net=copy.deepcopy(base);net.requires_grad_(True);net.train();opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);start=time.perf_counter();history=[]
  for step in range(cfg['steps']):
   sel=torch.multinomial(torch.tensor(weights),cfg['batch'],replacement=True);q=net(x[ii[sel]]);loss=torch.nn.functional.binary_cross_entropy_with_logits(q[torch.arange(len(sel)),hh[sel]],targets[sel]);ti=torch.multinomial(tw,cfg['teacher_batch'],replacement=True);keep=torch.nn.functional.binary_cross_entropy_with_logits(net(tx[ti]),tp[ti]);total=loss+cfg['preservation_weight']*keep;opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']);opt.step()
   if step%300==0 or step==cfg['steps']-1:history.append(dict(step=step+1,supervised=float(loss.detach()),preservation=float(keep.detach())));print('boundary train',seed,step+1,history[-1],flush=True)
  net.eval();dst=R/'runs'/str(seed);dst.mkdir(parents=True,exist_ok=True);assert not(dst/'model.pt').exists();torch.save(net.state_dict(),dst/'model.pt')
  with torch.inference_mode():p=net(x).sigmoid().numpy()
  np.savez_compressed(dst/'predictions.npz',p=p);results.append(dict(seed=seed,model=str(dst/'model.pt'),model_sha256=sha(dst/'model.pt'),predictions=str(dst/'predictions.npz'),predictions_sha256=sha(dst/'predictions.npz'),seconds=time.perf_counter()-start,history=history))
 write(R/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=results,upstream_changed=False,old_model_changed=False,web_changed=False))
if __name__=='__main__':
 try:main()
 except BaseException:(R/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
