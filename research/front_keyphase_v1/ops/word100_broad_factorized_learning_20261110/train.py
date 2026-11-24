from core import *
import os,traceback
def main():
 cfg=read(U/'FROZEN.json')
 for p,h in {**cfg['source_pins'],**cfg['code_pins']}.items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 rr=rows();fit=[r for r in rr if r['role']=='fit'];lab=[r for r in fit if r['has_labels']];norm=read(U/'NORMALIZATION.json')
 pos=sum((r['y']*r['ym']).sum(0) for r in lab);neg=sum(((1-r['y'])*r['ym']).sum(0) for r in lab);pw=torch.tensor(np.clip(neg/np.maximum(pos,1),.25,4),dtype=torch.float32)
 for cond in cfg['conditions']:
  for seed in cfg['seeds']:
   dst=U/'runs'/f'{cond}_{seed}';dst.mkdir(parents=True,exist_ok=True)
   if (dst/'COMPLETE.json').exists():continue
   assert not (dst/'last.pt').exists(),'Review interrupted checkpoint before resuming'
   torch.manual_seed(seed);np.random.seed(seed);net=Network(cond,norm);net.shared.load_state_dict(torch.load(B/'runs'/f'shared100_{seed}'/'model.pt',weights_only=True,map_location='cpu'))
   initial={k:v.clone() for k,v in net.shared.state_dict().items()};opt=torch.optim.AdamW([dict(params=net.shared.parameters(),lr=cfg['lr_base']),dict(params=[p for n,p in net.named_parameters() if not n.startswith('shared.')],lr=cfg['lr_head'])],weight_decay=cfg['weight_decay']);history=[];seen=set();start=time.perf_counter()
   write(U/'CURRENT_EXECUTION.json',dict(status='TRAINING',condition=cond,seed=seed,pid=os.getpid()))
   for ep in range(1,cfg['epochs']+1):
    net.train();rng=np.random.default_rng(seed*1000+ep);order=rng.permutation(len(fit));losses=[];epoch_seen=set()
    for k in range(cfg['steps_per_epoch']):
     aux=[fit[int(i)] for i in order[k*24:(k+1)*24]];chosen=[lab[int(i)] for i in rng.choice(len(lab),cfg['supervised_batch'],replace=False)];view='upper' if (ep+k)%2 else 'full'
     b=component_batch(chosen,view);o=net.shared(b['x'],b['lengths'],b['word']);w=cfg['weights'];mask=b['ym']
     attr=(TF.binary_cross_entropy_with_logits(o['attributes'],b['y'],reduction='none',pos_weight=pw)*mask).sum()/mask.sum().clamp_min(1)
     st=(TF.cross_entropy(o['stage'].transpose(1,2),b['sy'],ignore_index=-1,reduction='none')*b['sw']).sum()/b['sw'].sum().clamp_min(1)
     valid=(b['ci']>=0)&o['valid'];proto=net.shared.base.prototype[b['word'][:,None],b['ci'].clamp_min(0)];al=((o['latent']-proto).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
     ml=TF.cross_entropy(net.movement(o['latent'],b['g'],b['ii']),b['my']) if len(b['my']) else attr*0
     ol=TF.binary_cross_entropy_with_logits(net.ordering(o['latent'],b['oi'],b['og']),b['oy']) if len(b['oy']) else attr*0
     ab=shared.batch(aux,view);ao=net.shared(ab['x'],ab['lengths'],ab['word']);wl=TF.cross_entropy(ao['word_logits'],ab['class'])
     loss=w['shape']*attr+w['weak_stage']*st+w['prototype']*al+w['movement']*ml+w['order']*ol+w['word']*wl
     opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();assert torch.isfinite(loss)
     seen.update(r['clip_id'] for r in aux);epoch_seen.update(r['clip_id'] for r in aux);losses.append([float(v.detach()) for v in (loss,attr,ml,ol,wl)])
    assert len(epoch_seen)==899
    hist=dict(epoch=ep,all100_clips_seen=len(epoch_seen),losses=np.mean(losses,axis=0).tolist(),seconds=time.perf_counter()-start);history.append(hist)
    if ep%3==0:torch.save(dict(model=net.state_dict(),opt=opt.state_dict(),epoch=ep,history=history),dst/'last.pt');event('epoch',condition=cond,seed=seed,**hist)
   torch.save(net.state_dict(),dst/'model.pt');delta={prefix:float(sum((net.shared.state_dict()[k]-v).float().square().sum() for k,v in initial.items() if k.startswith(prefix)).sqrt()) for prefix in ['base.input','base.temporal','base.context','base.attributes','word_head']}
   assert len(seen)==899 and delta['base.context']>0
   write(dst/'COMPLETE.json',dict(condition=cond,seed=seed,epochs=cfg['epochs'],history=history,fit_seen=sorted(seen),fit_words=100,parameter_deltas=delta,seconds=time.perf_counter()-start,sha256=sha(dst/'model.pt'),development_training=False));event('run_complete',condition=cond,seed=seed,seconds=time.perf_counter()-start)
 write(U/'TRAINING_COMPLETE.json',dict(runs=[str(p) for p in (U/'runs').glob('*/COMPLETE.json')],status='SIX_FIXED_CONTINUATION_RUNS_COMPLETE'));write(U/'CURRENT_EXECUTION.json',dict(status='TRAINED',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
