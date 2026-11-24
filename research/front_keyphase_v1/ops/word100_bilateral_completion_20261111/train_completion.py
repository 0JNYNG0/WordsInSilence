from completion_model import *
import os,traceback

def intervals(ann):
 out=[]
 for e in ann['events']:
  a,b=e['onset'][0],e['offset'][1];mid=(e['onset'][1]+e['offset'][0])//2
  out.append((a,b,2,[1,1],e['direction'],'whole'))
  # Cuts are strictly outside the uncertain start/end bands.
  if e['onset'][1]<mid<e['offset'][0]:
   out.extend([(a,mid,1,[1,0],e['direction'],'prefix'),(mid,b,1,[0,1],e['direction'],'suffix')])
  c,d=e['onset'][1]+1,e['offset'][0]-1
  if d-c>=3:out.append((c,d,1,[0,0],e['direction'],'middle'))
 for a,b in ann['holds']:out.append((a,b,0,[0,0],1,'hold'))
 for a,b in ann['background']:out.append((a,b,0,[0,0],-100,'background'))
 for e,f in zip(ann['events'],ann['events'][1:]):out.append((e['onset'][0],f['offset'][1],3,[0,0],-100,'multiple'))
 return [r for r in out if r[1]-r[0]>=3]

def main():
 assert not (U/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 labels=read(U/'LABELS.json');sources={r['clip_id']:r for r in read(U/'PUBLIC_CACHE.json')['rows']};cache={};samples=[];features=[];masks=[];metas=[];ys=[];bs=[];ds=[]
 for ann in labels['rows']:
  if ann['role']!='fit':continue
  clip=ann['clip_id'];src=sources[clip];assert src['source_sha256']==ann['source_sha256'] and sha(src['path'])==src['sha256']
  if clip not in cache:cache[clip]=dict(np.load(src['path']))
  z=cache[clip]
  for view in ['x','upper']:
   for speed in [1.,.8,1.25]:
    for a,b,y,by,dy,kind in intervals(ann):
     seq,mask,meta,info=observed_features(z[view],z['ts']/speed,a,b,ann['axis'],ann['moving'])
     # Missing coordinates never become correctness labels; keep excluded rows.
     if not info['eligible']:samples.append(dict(clip_id=clip,side=ann['moving'],kind=kind,view=view,speed=speed,excluded='observation',span=[a,b]));continue
     samples.append(dict(clip_id=clip,side=ann['moving'],axis=ann['axis'],kind=kind,view=view,speed=speed,span=[a,b],sample=len(ys),quality=y,boundaries=by,direction=dy))
     features.append(seq);masks.append(mask);metas.append(meta);ys.append(y);bs.append(by);ds.append(dy)
 arr=np.stack(features);norm=np.maximum(arr.reshape(-1,4).std(0),.05)
 cfg=dict(seeds=[42,43,44],epochs=60,lr=.001,batch=128,norm=norm.tolist(),decision_threshold=.8,context_us=160000,modes=['baseline','head_only','binding_only','combined'],no_development_selection=True,common100_encoder='Preserved frozen 899-source model; no expansion claimed',shape_direction='Previous deployed and research weights unchanged; new separate candidate',gate=dict(normal_added_warning=0,old_errors_lost=0,unknown_promoted=0,partial_false_complete=0,full_recall_no_regression=True,reverse_false_similar=0,all_seeds_required=True),pins={str(p):sha(p) for p in [U/'LABELS.json',U/'completion_model.py',U/'train_completion.py',U/'bilateral_core.py']})
 write(U/'FROZEN.json',cfg);write(U/'TRAIN_SAMPLES.json',samples)
 p=U/'train_arrays.npz';np.savez_compressed(p,seq=arr,mask=np.stack(masks),meta=np.stack(metas),quality=ys,boundaries=bs,direction=ds)
 tx,tm,tt=torch.tensor(arr),torch.tensor(np.stack(masks)),torch.tensor(np.stack(metas));ty=torch.tensor(ys);tb=torch.tensor(bs,dtype=torch.float32);td=torch.tensor(ds)
 qw=torch.tensor(len(ys)/np.maximum(1,np.bincount(ys,minlength=4))/4,dtype=torch.float32);runs=[]
 for seed in cfg['seeds']:
  write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid(),seed=seed));t=time.perf_counter();net=CompletionNet(seed,norm);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.0001);history=[];gen=torch.Generator().manual_seed(seed)
  for epoch in range(cfg['epochs']):
   total=0.
   for ix in torch.randperm(len(ty),generator=gen).split(cfg['batch']):
    d,b,q=net(tx[ix],tm[ix],tt[ix]);loss=F.cross_entropy(q,ty[ix],weight=qw)+F.binary_cross_entropy_with_logits(b,tb[ix])+.5*F.cross_entropy(d,td[ix],ignore_index=-100)
    assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();total+=float(loss.detach())*len(ix)
   history.append(total/len(ty))
  dest=U/'runs'/str(seed);dest.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),dest/'model.pt');write(dest/'TRAINING.json',dict(seed=seed,loss=history,seconds=time.perf_counter()-t,parameters=sum(p.numel() for p in net.parameters()),sample_count=len(ys),unique_clips=len(cache)))
  runs.append(dict(seed=seed,model=str(dest/'model.pt'),sha256=sha(dest/'model.pt'),seconds=time.perf_counter()-t));event('training_finished',**runs[-1])
 write(U/'TRAINING_COMPLETE.json',dict(runs=runs,samples=len(ys),unique_clips=len(cache),fit_left_clips=len({r['clip_id'] for r in samples if r['side']=='L' and 'sample' in r}),arrays_sha256=sha(p)));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED_NOT_EVALUATED',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
