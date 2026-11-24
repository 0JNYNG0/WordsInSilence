from ranking_core import *
import time,traceback
def main():
 assert not(K/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);cfg=read(K/'FROZEN.json');meta=read(K/'INPUTS_COMPLETE.json')
 for p,h in {**cfg['pins'],**meta['pins']}.items():assert sha(p)==h,p
 seq=meta['sequences'];bags=[b for b in meta['bags']if not b['skipped']];data=[]
 for b in bags:
  assert sha(b['path'])==b['sha256'];a=np.load(b['path']);assert np.isfinite(a['x']).all() and a['y'].any() and (~a['y']).any();data.append({k:a[k].copy()for k in a.files})
 assert {r['person']for r in seq if r['split']=='train'}=={2,3,4,5,6,7};assert {r['person']for r in seq if r['split']=='probe'}=={8,9};assert not({r['source_sha256']for r in seq if r['split']=='train'}&{r['source_sha256']for r in seq if r['split']=='probe'})
 # New sparse labels and ranges must refer only to inspected RGB frames.
 labels=read(D/'SOURCE_EPISODES_ADDED.json');old_sources=read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows'];assert not({r['source_sha256']for r in labels['rows']}&{r['source_sha256']for r in old_sources})
 for r in labels['rows']:
  seen=set()
  for s in r['sheets']:assert s['viewed'] and sha(s['path'])==s['sha256'];seen.update(s['frames'])
  for e in r['events']:assert set(e['onset']+e['offset']+e['progression_witness_frames'])<=seen and e['linguistic_completion']is None
 write(K/'PRETRAIN_VERIFICATION.json',dict(at=stamp(),finite=True,new_source_rgb_viewed=True,new_subjects_only_fit=True,probe_unchanged=True,fit_people=[2,3,4,5,6,7],probe_people=[8,9],skipped_bags=[b for b in meta['bags']if b['skipped']],train_code_sha256=sha(__file__),expert_gold=False));runs=[]
 for mode in ['rank_original','rank_extended']:
  ids=[i for i,b in enumerate(bags)if b['split']=='train' and(mode=='rank_extended'or not b['extended_only'])];groups={f:[i for i in ids if bags[i]['family']==f]for f in F};assert all(groups.values());xx=np.concatenate([data[i]['x']for i in ids]);mean=xx.mean(0);std=np.maximum(xx.std(0),.05)
  for seed in cfg['seeds']:
   dst=K/'runs'/mode/str(seed);dst.mkdir(parents=True,exist_ok=True);assert not(dst/'model.pt').exists();torch.manual_seed(seed);rng=np.random.default_rng(seed);net=RankHead(mean,std);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);t=time.perf_counter();log=[]
   for step in range(cfg['steps']):
    chosen=[int(i)for f in F for i in rng.choice(groups[f],2,replace=False)];xs=[];info=[];start=0
    for i in chosen:
     d=data[i];pos=np.flatnonzero(d['y']);neg=np.flatnonzero(~d['y']);neg=rng.choice(neg,min(56,len(neg)),replace=False);ix=np.r_[pos,neg];xs.append(d['x'][ix]);info.append((start,start+len(ix),len(pos),F.index(bags[i]['family'])));start+=len(ix)
    z=net(torch.tensor(np.concatenate(xs)));loss=torch.stack([torch.logsumexp(z[a:b,j],0)-torch.logsumexp(z[a:a+np_,j],0)for a,b,np_,j in info]).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
    if step%300==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())));print('rank',mode,seed,step+1,float(loss.detach()),flush=True)
   net.eval();torch.save(net.state_dict(),dst/'model.pt');pred={}
   with torch.inference_mode():
    for i,d in enumerate(data):pred[f'logits_{i}']=net(torch.tensor(d['x'])).numpy()
   np.savez_compressed(dst/'predictions.npz',**pred);runs.append(dict(condition=mode,seed=seed,model=str(dst/'model.pt'),model_sha256=sha(dst/'model.pt'),predictions=str(dst/'predictions.npz'),predictions_sha256=sha(dst/'predictions.npz'),seconds=time.perf_counter()-t,log=log));write(K/'TRAINING_PROGRESS.json',dict(at=stamp(),completed=runs))
 # Same direct-boundary model/loss as the previous unit, with the eight new FIT sources.
 zs=[];trainseq=[]
 for r in seq:
  assert sha(r['path'])==r['sha256'];a=np.load(r['path']);zs.append({k:a[k].copy()for k in a.files})
 ids=[i for i,r in enumerate(seq)if r['split']=='train'];trainseq=[seq[i]for i in ids];zz=[zs[i]for i in ids];T=max(len(z['x'])for z in zz);n=len(zz);sx=np.zeros((n,T,150),np.float32);sy=np.zeros((n,T,5),np.float32);sm=np.zeros_like(sy);band=np.zeros((n,T,5,2),bool);valid=np.zeros_like(band)
 for i,z in enumerate(zz):
  l=len(z['x']);sx[i,:l]=z['x'];sy[i,:l]=z['y'];sm[i,:l]=z['mask'];band[i,:l]=z['bands'];valid[i,:l]=z['valid']
 pw=np.zeros_like(sy);bw=np.zeros((n,5,2),np.float32)
 for j in range(5):
  for v in [0,1]:
   take=(sm[:,:,j]>0)&(sy[:,:,j]==v);clips=sorted({r['clip_id']for r,t in zip(trainseq,take.any(1))if t})
   for cid in clips:
    ii=[i for i,r in enumerate(trainseq)if r['clip_id']==cid and take[i].any()]
    for i in ii:pw[i,take[i],j]=1/(10*len(clips)*len(ii)*take[i].sum())
  for h in range(2):
   take=band[:,:,j,h].any(1);clips=sorted({r['clip_id']for r,t in zip(trainseq,take)if t})
   for cid in clips:
    ii=[i for i,r in enumerate(trainseq)if r['clip_id']==cid and take[i]]
    for i in ii:bw[i,j,h]=1/(10*len(clips)*len(ii))
 assert np.isclose(pw.sum(),1)and np.isclose(bw.sum(),1)and(band<=valid).all();np.savez_compressed(K/'BOUNDARY_WEIGHTS.npz',presence=pw,boundary=bw);empty=~band.any(1);band[:,0]|=empty;valid[:,0]|=empty;bx=torch.tensor(sx);by=torch.tensor(sy);bpw=torch.tensor(pw);bbw=torch.tensor(bw);bb=torch.tensor(band);bv=torch.tensor(valid);xx=np.concatenate([z['x']for z in zz]);mean=xx.mean(0);std=np.maximum(xx.std(0),.05)
 for seed in cfg['seeds']:
  dst=K/'runs/boundary_extended'/str(seed);dst.mkdir(parents=True,exist_ok=True);assert not(dst/'model.pt').exists();torch.manual_seed(seed);net=BoundaryHead(mean,std);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);t=time.perf_counter();log=[]
  for step in range(cfg['steps']):
   z=net(bx);lp=(torch.nn.functional.binary_cross_entropy_with_logits(z[:,:,:,2],by,reduction='none')*bpw).sum();den=torch.logsumexp(z[:,:,:,:2].masked_fill(~bv,-torch.inf),dim=1);num=torch.logsumexp(z[:,:,:,:2].masked_fill(~bb,-torch.inf),dim=1);lb=((den-num)*bbw).sum();loss=(lp+2*lb)/3;assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
   if step%300==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())));print('boundary_extended',seed,step+1,float(loss.detach()),flush=True)
  net.eval();torch.save(net.state_dict(),dst/'model.pt')
  with torch.inference_mode():pred={f'logits_{i}':net(torch.tensor(z['x'])).numpy()for i,z in enumerate(zs)}
  np.savez_compressed(dst/'predictions.npz',**pred);runs.append(dict(condition='boundary_extended',seed=seed,model=str(dst/'model.pt'),model_sha256=sha(dst/'model.pt'),predictions=str(dst/'predictions.npz'),predictions_sha256=sha(dst/'predictions.npz'),seconds=time.perf_counter()-t,log=log));write(K/'TRAINING_PROGRESS.json',dict(at=stamp(),completed=runs))
 write(K/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,shared100_frozen=True,web_changed=False,private=False,final8=False));print('Ranking and source diversity comparison trained')
if __name__=='__main__':
 try:main()
 except BaseException:(K/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
