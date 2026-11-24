from targets import *
import os,traceback

def main():
 assert not (U/'TRAINING_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 labels=read(U/'LABELS.json')['rows'];sources={r['clip_id']:r for r in read(P/'PUBLIC_CACHE.json')['rows']};cached={};samples=[];xx=[];mm=[];gg=[];yy=[];qm=[];by=[];bm=[];dy=[]
 for ann in labels:
  if ann['role']!='fit':continue
  clip=ann['clip_id'];src=sources[clip];assert sha(src['path'])==src['sha256'];assert src['source_sha256']==ann['source_sha256']
  if clip not in cached:cached[clip]=dict(np.load(src['path']))
  z=cached[clip]
  for view in ['x','upper']:
   for speed in [1.,.8,1.25]:
    for row in supervised_spans(ann,len(z['x'])):
     seq,mask,meta,info=prev.observed_features(z[view],z['ts']/speed,row['a'],row['b'],ann['axis'],ann['moving']);rec=dict(row,clip_id=clip,axis=ann['axis'],side=ann['moving'],view=view,speed=speed)
     if not info['eligible']:samples.append(dict(rec,excluded='observation'));continue
     samples.append(dict(rec,sample=len(xx)));xx.append(seq);mm.append(mask);gg.append(meta);yy.append(row['quality']);qm.append(row['quality_mask']);by.append(row['boundaries']);bm.append(row['boundary_mask']);dy.append(row['direction'])
 norm=read(P/'FROZEN.json')['norm'];cfg=dict(seeds=[42,43,44],models=['data_repaired','endpoints'],epochs=40,lr=.0005,batch=128,norm=norm,threshold=.8,primary='endpoints_joint',ablations=['data_repaired_strict','data_repaired_joint','endpoints_strict'],no_dev_selection=True,pins={str(p):sha(p) for p in [U/'targets.py',U/'train.py',U/'boundary_core.py',U/'LABELS.json']},gates=dict(partial_false_complete_not_increased=True,normal_whole_not_decreased=True,normal_warnings_not_increased=True,shape_direction_retained=True,unknown_not_promoted=True,all_three_seeds=True))
 write(U/'FROZEN.json',cfg);write(U/'TRAIN_SAMPLES.json',samples)
 z=dict(seq=np.stack(xx),mask=np.stack(mm),meta=np.stack(gg),quality=np.array(yy),quality_mask=np.array(qm,np.float32),boundaries=np.array(by,np.float32),boundary_mask=np.array(bm,np.float32),direction=np.array(dy))
 np.savez_compressed(U/'train_arrays.npz',**z);t={k:torch.tensor(v) for k,v in z.items()};runs=[]
 counts=np.bincount(np.array(yy)[np.array(qm)>0],minlength=4);weights=torch.tensor(sum(counts)/np.maximum(1,counts)/4,dtype=torch.float32)
 for mode in cfg['models']:
  for seed in cfg['seeds']:
   write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid(),mode=mode,seed=seed));start=time.perf_counter();net=EndpointNet(seed,norm,mode);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=.0001);gen=torch.Generator().manual_seed(seed);hist=[]
   for epoch in range(cfg['epochs']):
    total=0.
    for ix in torch.randperm(len(xx),generator=gen).split(cfg['batch']):
     d,b,q=net(t['seq'][ix],t['mask'][ix],t['meta'][ix]);qmask=t['quality_mask'][ix];bmask=t['boundary_mask'][ix];dl=F.cross_entropy(d,t['direction'][ix],ignore_index=-100,reduction='none');dmask=t['direction'][ix]!=-100
     loss=(F.cross_entropy(q,t['quality'][ix],weight=weights,reduction='none')*qmask).sum()/qmask.sum().clamp_min(1)+(F.binary_cross_entropy_with_logits(b,t['boundaries'][ix],reduction='none')*bmask).sum()/bmask.sum().clamp_min(1)+.5*dl.sum()/dmask.sum().clamp_min(1)
     assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step();total+=float(loss.detach())*len(ix)
    hist.append(total/len(xx))
   dest=U/'runs'/f'{mode}_{seed}';dest.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),dest/'model.pt');write(dest/'TRAINING.json',dict(loss=hist,seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in net.parameters()),samples=len(xx)))
   runs.append(dict(mode=mode,seed=seed,path=str(dest/'model.pt'),sha256=sha(dest/'model.pt'),seconds=time.perf_counter()-start));event('trained',**runs[-1])
 write(U/'TRAINING_COMPLETE.json',dict(runs=runs,samples=len(xx),fit_videos=len(cached),fit_events=sum(len(a['events']) for a in labels if a['role']=='fit'),arrays_sha256=sha(U/'train_arrays.npz')));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED_PENDING_EVALUATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
