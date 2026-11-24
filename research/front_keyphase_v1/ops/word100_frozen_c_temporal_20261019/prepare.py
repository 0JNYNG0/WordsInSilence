"""Validate unchanged inputs and actual pretrained-C freeze before training."""
from common import *
import copy,traceback,ast
from collections import Counter
import train as candidate
from training_core import MODEL,TRAIN,SEEDS,validate_channels,schedule
from pilot.model_masked_v1 import MaskedSignTCN
from temporal import phase_weights
torch=candidate.torch

def main():
 assert not (U/'READY.json').exists() and not (U/'TRAIN_RUNNING.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 original=read(BASE/'training_CONTEXT.json');prior=O/'word100_temporal_phase_training_20261019';oldready=read(prior/'READY.json')
 assert oldready['status']=='PASS' and oldready['hash_identical_to_original_C']
 assert sha(U/'temporal.py')==sha(prior/'temporal.py')
 check(oldready['pins'])
 rows=original['ordered_rows'];tr=[r for r in rows if r['role']=='train_candidate'];dev=[r for r in rows if r['role']=='development_only']
 assert len(rows)==len({r['clip_id'] for r in rows})==1199 and len(tr)==899 and len(dev)==300
 assert set(r['signer_id'] for r in tr)==set(range(1,10)) and Counter(r['signer_id'] for r in dev)=={11:100,12:100,13:100}
 assert not {r['clip_id'] for r in tr}&{r['clip_id'] for r in dev}
 assert original['model']==MODEL and original['training']==TRAIN
 pins={str(BASE/n):sha(BASE/n) for n in ['training_CONTEXT.json','READY.json','training_core.py']}
 pins.update({str(prior/n):sha(prior/n) for n in ['READY.json','temporal.py','COMPARISON_INDEPENDENT.json','FINAL_AUDIT.json']})
 protected=read(prior/'FINAL_AUDIT.json')['original_checkpoint_pins'];check(protected);pins.update(protected)
 for seed in SEEDS:
  for folder,stem in [(BASE,f'training_C_{seed}'),(prior,f'T3_{seed}')]:
   rr=read(folder/f'{stem}_RESULT.json');assert sha(folder/f'{stem}_best.pt')==rr['checkpoint_sha256']
   for suffix in ['RESULT.json','history.json','best.pt']:p=folder/f'{stem}_{suffix}';pins[str(p)]=sha(p)
 for r in rows:
  for v in VIEWS:
   i=r['views'][v];p=i['channel_file'];assert p and Path(p).resolve().is_relative_to((BASE/'data/clips').resolve());assert sha(p)==i['sha256']==original['pins'][p];pins[p]=i['sha256']
 p=ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py';pins[str(p)]=sha(p);assert pins[str(p)]==original['pins'][str(p)]
 c={'at':now(),'rows':rows,'pins':pins,'semantic_readiness_reused':str(BASE/'READY.json'),'previous_identical_input_audit':str(prior/'READY.json'),'no_new_extraction':True}
 train,development=candidate.loadrows(c);tensors=values=0
 for r in train+development:
  for a in r['arrays']['T3F'].values():
   for x in a.values():validate_channels(x);tensors+=1;values+=x.size
 assert tensors==4196
 def funcs(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(Path(p).read_text(encoding='utf8')).body if isinstance(x,ast.FunctionDef)}
 of=funcs(prior/'train.py');nf=funcs(U/'train.py')
 for name in ['statehash','savept','batch','scores']:assert of[name]==nf[name],name
 tests={};initial={};gradient=[];initial_error=0.;padding_error=0.;probe_checks=0
 # Same predeclared TRAIN samples for all seeds, no DEV used for tuning.
 groups=[train[:8],train[300:308],train[-3:]]
 for seed in SEEDS:
  cp=torch.load(BASE/f'training_C_{seed}_best.pt',map_location='cpu',weights_only=True)
  base=MaskedSignTCN(**MODEL);base.load_state_dict(cp['state_dict']);base.eval()
  model=candidate.initialize(seed);frozen=candidate.statehash(model.frozen_state());full=candidate.statehash(model.state_dict())
  assert frozen==candidate.statehash(base.state_dict())
  assert [n for n,p in model.named_parameters() if p.requires_grad]==['phase_weight']
  assert torch.count_nonzero(model.phase_weight)==0
  model.train();model.assert_frozen()
  for group in groups:
   for epoch in range(1,5):
    x,p,y,ch=candidate.batch(group,'T3F',epoch=epoch);xx=x.clone();pp=p.clone();rng=torch.get_rng_state().clone()
    with torch.no_grad():a=base(x,p);b=model(x,p);again=model(x,p)
    assert torch.equal(a,b) and torch.equal(b,again);assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(x,xx) and torch.equal(p,pp);probe_checks+=1
  x,p,y,_=candidate.batch(train[:8],'T3F',epoch=1)
  with torch.no_grad():fixed=model.base_and_contrast(x,p)[0].clone()
  opt=torch.optim.AdamW([model.phase_weight],lr=TRAIN['lr'],weight_decay=TRAIN['weight_decay'])
  assert len(opt.param_groups)==1 and opt.param_groups[0]['params']==[model.phase_weight]
  for step in range(3):
   model.train();opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(x,p),y);loss.backward()
   assert torch.isfinite(loss) and model.phase_weight.grad is not None and torch.isfinite(model.phase_weight.grad).all() and model.phase_weight.grad.norm()>0
   gradient.append(float(model.phase_weight.grad.norm()))
   assert all(q.grad is None for name,q in model.named_parameters() if name!='phase_weight')
   torch.nn.utils.clip_grad_norm_([model.phase_weight],1.);opt.step();model.assert_frozen()
   assert candidate.statehash(model.frozen_state())==frozen
   with torch.no_grad():assert torch.equal(model.base_and_contrast(x,p)[0],fixed)
  assert torch.count_nonzero(model.phase_weight)>0
  assert candidate.statehash(model.state_dict())!=full
  assert len(opt.state)==1 and list(opt.state)[0] is model.phase_weight
  # Padding with nonzero irrelevant sentinels must not affect real output.
  with torch.no_grad():
   a=model(x,p);b=model(torch.cat([x,torch.full((len(x),17,280),13.)],1),torch.cat([p,torch.zeros(len(p),17)],1))
   err=float((a-b).abs().max());padding_error=max(padding_error,err);assert torch.allclose(a,b,atol=2e-5,rtol=2e-5)
  # train/eval switching must never turn on a base module, even after loading saved state.
  model.eval();model.train();model.assert_frozen();assert candidate.statehash(model.frozen_state())==frozen
  initial[str(seed)]={'full_state_sha256':full,'frozen_state_sha256':frozen,'C_checkpoint_sha256':sha(BASE/f'training_C_{seed}_best.pt'),'C_selected_epoch':cp['epoch'],'scratch_updates':3,'base_probe_max_error':0.}
 tests.update(three_seed_initial_C_output_bitwise_equal=True,all_base_parameters_and_BN_buffers_frozen_after_updates=True,base_dropout_eval_and_RNG_unchanged=True,optimizer_only_9600_phase_parameters=True,base_gradients_absent_and_head_gradients_nonzero=True,public_TRAIN_probe_unchanged_after_update=True,train_eval_switch_keeps_base_eval=True,input_and_masks_preserved=True,nonzero_head_padding_invariant=True)
 model=candidate.initialize(SEEDS[0]);model.train()
 for n in [1,2,9,31]:
  x=torch.zeros(2,n,280);x[:,:,279]=.033;x[:,0,279]=0;p=torch.ones(2,n);p[1]=0
  out=model(x,p);assert torch.isfinite(out).all();out.square().mean().backward();assert torch.isfinite(model.phase_weight.grad).all();model.zero_grad(set_to_none=True)
  before=phase_weights(x,p);x[:,1:4,:279]=0;assert torch.equal(before,phase_weights(x,p))
 model.assert_frozen();tests['short_missing_all_padding_finite_and_timeline_kept']=True
 # No re-run of previous scalar phase audit is needed: implementation byte-identical.
 tests['prior3596TRAIN_scalar_weight_and_schedule_audit_reused']=sha(U/'temporal.py')==sha(prior/'temporal.py')
 check(pins);write(U/'PREP_CONTEXT.json',c,new=True)
 own={str(U/n):sha(U/n) for n in ['AUTHORIZATION.json','FROZEN_PROTOCOL.json','common.py','temporal.py','frozen_temporal.py','train.py','prepare.py','verify_results.py','PREP_CONTEXT.json']}
 write(U/'READY.json',{'at':now(),'status':'PASS','training_allowed':True,'blockers':[],'pins':own,'initial_states':initial,'train':899,'dev':300,'input_npz_files':2398,'input_tensors':tensors,'channel_values_checked':values,'input_hashes_identical_C_T3':True,'parameters':{'total':47236,'frozen':37636,'trainable':9600},'tests':tests,'initial_output_max_error':initial_error,'initial_public_TRAIN_batches':probe_checks,'scratch_optimizer_steps':9,'phase_gradient_norms':gradient,'nonzero_head_padding_max_error':padding_error,'previous_ready_reused':str(prior/'READY.json'),'no_new_extraction':True,'user_data_used':False,'operator':'same Codex implementation and explicit freeze tests; no independent agent/expert'},new=True)
 event(stage='READY_FOR_THREE_T3F_RUNS',tests=len(tests),base_output_error=0,trainable=9600,train=899,dev=300)

if __name__=='__main__':
 try:main()
 except BaseException:
  if not (U/'FIRST_PREP_ERROR.json').exists():write(U/'FIRST_PREP_ERROR.json',{'at':now(),'traceback':traceback.format_exc()},new=True)
  raise
