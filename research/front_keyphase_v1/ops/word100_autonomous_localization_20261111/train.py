from localization_core import *
import os,traceback
def main():
 assert not (U/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 ann=read(U/'FIT_LABELS.json')['rows'];rows=fitrows();rows=[r for r in rows if any(a['clip_id']==r['clip_id'] for a in ann)]
 durations=[]
 for r in rows:
  r['annotations']=[a for a in ann if a['clip_id']==r['clip_id']];r['cached']=dict(np.load(B/'cache'/f"{r['clip_id']}.npz"))
  for ar in r['annotations']:
   for e in ar['events']:durations.append(int(r['timestamp_us'][e['offset'][1]]-r['timestamp_us'][e['onset'][0]]))
 med=int(np.median(durations));lags=sorted(set([33333,max(33333,med//4),max(66666,med//2)]))
 cfg=dict(seeds=[42,43,44],primary_seed=42,epochs=60,lr=.001,batch=8,lags_us=lags,fit_duration_median_us=med,state_threshold=.65,boundary_radius=1,min_span_frames=3,old_encoder_frozen=True,common100_videos=899,train_fine_videos=len(rows),fine_events=sum(len(a['events']) for a in ann),fine_holds=sum(len(a['holds']) for a in ann),runtime_expected_word_input=False,comparison=['baseline','motion_only','localization_only','combined'],evaluation_no_tuning=True,protections=read(M/'FROZEN.json')['protections'],pins={str(p):sha(p) for p in [U/'FIT_LABELS.json',U/'EVALUATION_LABELS.json',M/'runs/42/model.pt',M/'RESULT.json',V/'runs/42/model.pt']})
 assert not (U/'FROZEN.json').exists();write(U/'FROZEN.json',cfg)
 records=[];audit=[]
 for r in rows:
  for view in ['full','upper']:
   for rev in [False,True]:
    x=prior.physical(r['x_'+view]);x=x[::-1].copy() if rev else x;ts=r['timestamp_us'];fh=r['cached']['frozen_h_'+view];fh=fh[::-1].copy() if rev else fh;feat,valid=relation_input(x,ts,lags);y,by,bw=frame_targets(len(x),r['annotations'],valid,rev)
    records.append(dict(clip=r['clip_id'],person=r['person'],features=feat,fh=fh,y=y,by=by,bw=bw,n=len(x)))
  audit.append(dict(clip_id=r['clip_id'],person=r['person'],source_sha256=r['source_sha256'],events=sum(len(a['events']) for a in r['annotations']),holds=sum(len(a['holds']) for a in r['annotations'])))
 allx=np.concatenate([r['features'][r['y']>=0] for r in records]);norm=dict(mean=allx.mean(0).tolist(),std=np.maximum(allx.std(0),.05).tolist());write(U/'NORMALIZATION.json',norm);write(U/'FIT_AUDIT.json',dict(rows=audit,augmented_sequences=len(records),all_source_sha_pinned=True,unknown_not_hold=True,review='Same Codex RGB; not independent human'))
 def batch(rr):
  n=len(rr);t=max(r['n'] for r in rr);d=len(norm['mean']);ft=np.zeros((n,t,10,d),np.float32);fh=np.zeros((n,t,64),np.float32);y=np.full((n,t,10),-1,np.int64);by=np.zeros((n,t,10,2),np.float32);bw=by.copy()
  for i,r in enumerate(rr):z=r['n'];ft[i,:z]=r['features'];fh[i,:z]=r['fh'];y[i,:z]=r['y'];by[i,:z]=r['by'];bw[i,:z]=r['bw']
  return [torch.tensor(v) for v in [ft,fh,np.array([r['n'] for r in rr]),y,by,bw]]
 counts=np.bincount(np.concatenate([r['y'][r['y']>=0] for r in records]),minlength=4);cw=torch.tensor(1/np.sqrt(np.maximum(counts,1)),dtype=torch.float32);cw/=cw.mean()
 for seed in cfg['seeds']:
  torch.manual_seed(seed);net=Localizer(norm);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.01);history=[];start=time.perf_counter();write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING_LOCALIZER',seed=seed,pid=os.getpid()))
  for epoch in range(cfg['epochs']):
   rng=np.random.default_rng(seed*1000+epoch);ids=rng.permutation(len(records));losses=[];net.train()
   for starti in range(0,len(ids),cfg['batch']):
    ft,fh,ll,y,by,bw=batch([records[i] for i in ids[starti:starti+cfg['batch']]]);sp,bp=net(ft,fh,ll);ce=F.cross_entropy(sp.reshape(-1,4),y.reshape(-1),weight=cw,ignore_index=-1);bl=F.binary_cross_entropy_with_logits(bp,by,reduction='none',pos_weight=torch.tensor([3.,3.]));loss=ce+.4*(bl*bw).sum()/bw.sum().clamp_min(1);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();losses.append(float(loss.detach()))
   history.append(float(np.mean(losses)))
   if epoch%15==0:event('localizer_training',seed=seed,epoch=epoch,loss=history[-1])
  dest=U/'runs'/str(seed);dest.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),dest/'model.pt');write(dest/'COMPLETE.json',dict(seed=seed,epochs=cfg['epochs'],history=history,seconds=time.perf_counter()-start,sha256=sha(dest/'model.pt'),parameters=sum(p.numel() for p in net.parameters())))
  event('localizer_trained',seed=seed,seconds=time.perf_counter()-start)
 write(U/'TRAINING_COMPLETE.json',dict(seeds=cfg['seeds'],train_fine_videos=len(rows),records=len(records),common100_checkpoint_unchanged=True));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED_PENDING_EVALUATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
