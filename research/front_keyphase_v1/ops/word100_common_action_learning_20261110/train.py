from common_action import *
import os,traceback
from torch.nn import functional as TF
def main():
 cfg=read(U/'FROZEN.json')
 for p,h in {**cfg['source_pins'],**cfg['code_pins'],**cfg['protected']}.items():assert sha(p)==h,p
 assert not (U/'TRAINING_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 rr=[r for r in rows() if r['role']=='fit'];norm=read(U/'NORMALIZATION.json');teacher=base_model().eval();cached={}
 with torch.inference_mode():
  for r in rr:
   for view in ['full','upper']:
    b=build_batch([r],view);o=teacher(b['x'],b['lengths'],b['word']);cached[(r['clip_id'],view)]={k:v[0].clone() for k,v in o.items() if k in ['latent','attributes']}
 pos=sum((r['y']*r['ym']).sum(0) for r in rr);neg=sum(((1-r['y'])*r['ym']).sum(0) for r in rr);pw=torch.tensor(np.clip(neg/np.maximum(pos,1),.25,4),dtype=torch.float32)
 counts=np.asarray(norm['motion_class_counts']);cw=torch.tensor(counts.sum()/(3*counts),dtype=torch.float32)
 for condition in cfg['conditions']:
  for seed in cfg['seeds']:
   folder=U/'runs'/f'{condition}_{seed}';folder.mkdir(parents=True,exist_ok=True)
   if (folder/'COMPLETE.json').exists():continue
   assert not (folder/'STARTED.json').exists(),'Interrupted run: review checkpoint before resuming'
   torch.manual_seed(seed);net=CommonActionNet(norm['mean'],norm['std'])
   for name,p in net.base.named_parameters():p.requires_grad_(condition=='joint_finetune' and name.split('.')[0] in cfg['base_finetune_parts'])
   initial={k:v.clone() for k,v in net.base.state_dict().items()}
   groups=[dict(params=net.motion_head.parameters(),lr=cfg['lr_head'])]
   if condition=='joint_finetune':groups.append(dict(params=[p for p in net.base.parameters() if p.requires_grad],lr=cfg['lr_base']))
   opt=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay']);history=[];start=time.perf_counter()
   write(folder/'STARTED.json',dict(pid=os.getpid(),frozen=sha(U/'FROZEN.json'),code=sha(__file__)))
   write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',pid=os.getpid(),condition=condition,seed=seed))
   for epoch in range(1,cfg['epochs']+1):
    rng=np.random.default_rng(seed*1000+epoch);order=rng.permutation(len(rr));values=[]
    for starti in range(0,len(rr),cfg['batch']):
     selected=[rr[int(i)] for i in order[starti:starti+cfg['batch']]];view='full' if (epoch+starti//cfg['batch'])%2 else 'upper';b=build_batch(selected,view)
     out=net.base(b['x'],b['lengths'],b['word']);h=out['latent'];mask=b['ym'];valid=out['valid'];obs=(TF.binary_cross_entropy_with_logits(out['attributes'],b['y'],reduction='none',pos_weight=pw)*mask).sum()/mask.sum().clamp_min(1)
     ce=TF.cross_entropy(out['stage'].transpose(1,2),b['sy'],ignore_index=-1,reduction='none');stage=(ce*b['sw']).sum()/b['sw'].sum().clamp_min(1)
     oldh=torch.zeros_like(h);olda=torch.zeros_like(out['attributes'])
     for i,r in enumerate(selected):
      n=len(r['x_full']);c=cached[(r['clip_id'],view)];oldh[i,:n]=c['latent'];olda[i,:n]=c['attributes']
     preserve=((h-oldh).square().mean(-1)*valid).sum()/valid.sum()
     preserve_attr=((out['attributes'].sigmoid()-olda.sigmoid()).square().mean(-1)*valid).sum()/valid.sum()
     if len(b['motion_y']):move=TF.cross_entropy(net.movement(h,b['geometry'],b['indices']),b['motion_y'],weight=cw)
     else:move=sum(p.sum()*0 for p in net.motion_head.parameters())
     loss=move+.5*obs+.1*stage+preserve+.25*preserve_attr
     assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],2);opt.step()
     values.append([float(v.detach()) for v in [loss,move,obs,stage,preserve]])
    history.append(dict(epoch=epoch,losses=np.mean(values,0).tolist()))
    if epoch%10==0:event('epoch',condition=condition,seed=seed,epoch=epoch,loss=history[-1]['losses'][0],seconds=time.perf_counter()-start)
   delta={key:float(sum((net.base.state_dict()[k]-v).float().square().sum() for k,v in initial.items() if k.startswith(key)).sqrt()) for key in ['input','temporal','context','attributes','stage_query','guide_encoder','prototype']}
   assert all(delta[k]==0 for k in ['attributes','stage_query','guide_encoder','prototype'])
   assert (delta['temporal']>0 and delta['context']>0) if condition=='joint_finetune' else all(v==0 for v in delta.values())
   torch.save(net.state_dict(),folder/'model.pt');write(folder/'COMPLETE.json',dict(condition=condition,seed=seed,seconds=time.perf_counter()-start,epochs=cfg['epochs'],history=history,deltas=delta,model_sha256=sha(folder/'model.pt'),fit_clips=len(rr),dev_used=False));event('complete',condition=condition,seed=seed,seconds=time.perf_counter()-start)
 write(U/'TRAINING_COMPLETE.json',dict(runs=[str(p) for p in sorted((U/'runs').glob('*/COMPLETE.json'))],frozen=sha(U/'FROZEN.json'),code=sha(__file__)))
 write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED_PENDING_EVALUATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
