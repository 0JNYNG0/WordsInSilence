from local_rank_core import *
import time,traceback
def main():
 assert not(L/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);cfg=read(L/'FROZEN.json');meta=read(L/'INPUTS_COMPLETE.json');bags=meta['bags'];data=[]
 for p,h in {**cfg['pins'],**meta['pins']}.items():assert sha(p)==h,p
 for r in bags:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path']);old=np.load(r['original_path']);assert np.array_equal(z['y'],old['y'])and np.array_equal(z['spans'],old['spans']);assert all(np.isfinite(z[k]).all()for k in z.files);data.append({k:z[k].copy()for k in z.files})
 tr=[i for i,r in enumerate(bags)if r['split']=='train'];assert {bags[i]['person']for i in tr}=={2,3,4,5,6,7};assert {r['person']for r in bags if r['split']=='probe'}=={8,9};assert not({bags[i]['source_sha256']for i in tr}&{r['source_sha256']for r in bags if r['split']=='probe'});groups={f:[i for i in tr if bags[i]['family']==f]for f in F};assert all(groups.values());runs=[]
 # The ablation preserves body inputs and real frame interval, never substitutes progress.
 test=np.arange(3*280,dtype=np.float32).reshape(3,280);q=np.arange(300,dtype=np.float32).reshape(3,100);xx,qq=drop_hand_information(test,q);assert np.array_equal(xx[:,:18],test[:,:18])and np.array_equal(xx[:,186:195],test[:,186:195])and np.array_equal(xx[:,279],test[:,279]);assert not xx[:,18:186].any()and not xx[:,195:279].any();assert np.array_equal(qq.reshape(3,10,10)[...,:4],q.reshape(3,10,10)[...,:4]);assert np.array_equal(qq.reshape(3,10,10)[...,8],q.reshape(3,10,10)[...,8]);assert not qq.reshape(3,10,10)[...,4:8].any()
 write(L/'PRETRAIN_VERIFICATION.json',dict(at=stamp(),finite=True,same_candidate_labels_and_sampling=True,split_checked=True,ablation_retains_body_and_dt=True,head_train_people=[2,3,4,5,6,7],probe_people=[8,9],upstream_probe_exposed=True,code_sha256=sha(__file__),no_expert_gold=True))
 for mode in ['local_full','local_no_hands']:
  xx=np.concatenate([data[i][mode]for i in tr]);mean=xx.mean(0);std=np.maximum(xx.std(0),.05)
  for seed in cfg['seeds']:
   dst=L/'runs'/mode/str(seed);dst.mkdir(parents=True,exist_ok=True);assert not(dst/'model.pt').exists();torch.manual_seed(seed);rng=np.random.default_rng(seed);net=LocalRankHead(mean,std);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);t=time.perf_counter();log=[]
   for step in range(cfg['steps']):
    chosen=[int(i)for f in F for i in rng.choice(groups[f],2,replace=False)];xs=[];info=[];start=0
    for i in chosen:
     d=data[i];pos=np.flatnonzero(d['y']);neg=np.flatnonzero(~d['y']);neg=rng.choice(neg,min(56,len(neg)),replace=False);ix=np.r_[pos,neg];xs.append(d[mode][ix]);info.append((start,start+len(ix),len(pos),F.index(bags[i]['family'])));start+=len(ix)
    z=net(torch.tensor(np.concatenate(xs)));loss=torch.stack([torch.logsumexp(z[a:b,j],0)-torch.logsumexp(z[a:a+np_,j],0)for a,b,np_,j in info]).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
    if step%300==0 or step==cfg['steps']-1:log.append(dict(step=step+1,loss=float(loss.detach())));print(mode,seed,step+1,float(loss.detach()),flush=True)
   net.eval();torch.save(net.state_dict(),dst/'model.pt')
   with torch.inference_mode():pred={f'logits_{i}':net(torch.tensor(d[mode])).numpy()for i,d in enumerate(data)}
   np.savez_compressed(dst/'predictions.npz',**pred);runs.append(dict(condition=mode,seed=seed,model=str(dst/'model.pt'),model_sha256=sha(dst/'model.pt'),predictions=str(dst/'predictions.npz'),predictions_sha256=sha(dst/'predictions.npz'),seconds=time.perf_counter()-t,log=log));write(L/'TRAINING_PROGRESS.json',dict(at=stamp(),completed=runs))
 write(L/'TRAINING_COMPLETE.json',dict(at=stamp(),rows=runs,encoder_frozen=True,web_changed=False,private=False,final8=False));print('Local hand-information comparison trained')
if __name__=='__main__':
 try:main()
 except BaseException:(L/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
