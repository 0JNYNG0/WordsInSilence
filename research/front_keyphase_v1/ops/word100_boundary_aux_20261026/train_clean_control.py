"""Matched clean-only continuation to distinguish augmentation from more head training."""
from pathlib import Path
import sys,json,time,hashlib,copy
import numpy as np
import torch
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
U=Path('C:/MyUnityProjects/WordsInSilence/research/front_keyphase_v1/ops/word100_boundary_aux_20261026')
sys.path.insert(0,str(U));import run_aux as a

def main():
 a.checks();assert (U/'DONE.json').exists() and not (U/'RUNNING.json').exists()
 a.write('CONTROL_FROZEN.json',dict(at=time.time(),script_sha256=a.b.sha(__file__),
  reason='Additional head training is a confound. Same initialization, schedule, optimizer,40epochs, clean DEV selection incl0; replace augmented paired training example with duplicate clean example.',
  declaration='Added after augmented training started; explanatory development control, not pre-registered initial arm.',
  no_tuning=True,cache_only=True,training_data='same899 public TRAIN',seeds=a.SEEDS))
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 ctx,tr,dv=a.load_source_rows();ty=np.array([r['label_index'] for r in tr],np.int64);dy=np.array([r['label_index'] for r in dv],np.int64)
 for seed in a.SEEDS:
  z=a.b.arrays(U/f'features_{seed}.npz');ck=torch.load(a.b.T/f'T3F_{seed}_best.pt',map_location='cpu',weights_only=True)
  initial=ck['state_dict']['phase_weight'].numpy().copy();w=torch.nn.Parameter(torch.from_numpy(initial.copy()))
  opt=torch.optim.AdamW([w],lr=a.LR,weight_decay=.001);initial_ce,_=a.eval_clean(z['dev'],initial,dy)
  best=dict(epoch=0,ce=initial_ce,weight=initial.copy());hist=[];trace=U/f'CONTROL_{seed}_TRACE.jsonl';assert not trace.exists()
  for epoch in range(1,a.EPOCHS+1):
   order=np.random.default_rng(seed+epoch).permutation(899)
   for step,j in enumerate(range(0,899,32),1):
    ii=order[j:j+32];ff=[];yy=[];ids=[]
    for i in ii:
     row=tr[i];v,t=a.schedule(row['clip_id'],epoch,'C');feature=z['train'][i,a.VIEWS.index(v),a.TEMP.index(t),0]
     ff.extend([feature,feature]);yy.extend([int(ty[i])]*2);ids.append([row['clip_id'],v,t,'clean_twice'])
    x=torch.from_numpy(np.stack(ff));y=torch.tensor(yy,dtype=torch.long);opt.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(x[:,:100]+torch.nn.functional.linear(x[:,100:],w),y)
    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([w],1.);opt.step()
    with trace.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(epoch=epoch,batch=step,clips=ids),ensure_ascii=False)+'\n')
   ww=w.detach().numpy().copy();ce,hits=a.eval_clean(z['dev'],ww,dy)
   if ce<best['ce']-1e-6:best=dict(epoch=epoch,ce=ce,weight=ww.copy())
   hist.append(dict(epoch=epoch,clean_dev_ce=ce,clean_hits=hits,selected_epoch=best['epoch']))
  selected=copy.deepcopy(ck);selected['state_dict']['phase_weight']=torch.from_numpy(best['weight'].copy())
  selected.update(arm='CLEAN_CONTINUATION_CONTROL',epoch=best['epoch'],control_protocol_sha256=a.b.sha(U/'CONTROL_FROZEN.json'))
  assert all(torch.equal(v,selected['state_dict'][k]) for k,v in ck['state_dict'].items() if k!='phase_weight')
  cp=U/f'CONTROL_{seed}_best.pt';assert not cp.exists();torch.save(selected,cp)
  a.write(f'CONTROL_{seed}_RESULT.json',dict(seed=seed,selected_epoch=best['epoch'],selected_ce=best['ce'],initial_ce=initial_ce,
    history=hist,predictions=a.predict_features(z['dev'],best['weight'],dv),checkpoint_sha256=a.b.sha(cp),trace_sha256=a.b.sha(trace),
    frozen_C_tensors_equal=True,original_clip_exposures=35960,training_examples=71920,
    protocol_sha256=a.b.sha(U/'CONTROL_FROZEN.json'),role='explanatory clean continuation control',user_recordings=0))
  print('CLEAN CONTROL',seed,'selected',best['epoch'],flush=True)
 a.checks()
if __name__=='__main__':main()
