"""Pre-training integrity, numerical reference, initialization and backward gates."""
from common import *
from collections import Counter
import ast, copy, importlib.util, traceback
import train as candidate
from training_core import MODEL,TRAIN,SEEDS,schedule,validate_channels
from temporal import phase_weights,summarize,TemporalPhaseTCN
from pilot.model_masked_v1 import MaskedSignTCN
torch=candidate.torch

def main():
 assert not (U/'READY.json').exists() and not (U/'TRAIN_RUNNING.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 original=read(BASE/'training_CONTEXT.json');oldready=read(BASE/'READY.json')
 assert all(oldready[k] for k in ['independent_input_validation_passed','fresh_sensor_detector_state_verified','roi_hidden_coordinate_invariance_passed'])
 rows=original['ordered_rows'];tr=[r for r in rows if r['role']=='train_candidate'];dev=[r for r in rows if r['role']=='development_only']
 assert len(rows)==len({r['clip_id'] for r in rows})==1199 and len(tr)==899 and len(dev)==300
 assert set(r['signer_id'] for r in tr)==set(range(1,10)) and Counter(r['signer_id'] for r in dev)=={11:100,12:100,13:100}
 assert not {r['clip_id'] for r in tr}&{r['clip_id'] for r in dev}
 assert original['model']==MODEL and original['training']==TRAIN
 prior=O/'word100_temporal_phase_plan_20261019'
 pins={str(prior/n):sha(prior/n) for n in ['PLAN.json','ANALYSIS.json','VERIFICATION.json']}
 for n in ['training_CONTEXT.json','READY.json','FULL_INPUTS_INDEPENDENT.json','training_core.py','training_driver.py']:
  pins[str(BASE/n)]=sha(BASE/n)
 model=ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py';assert sha(model)==original['pins'][str(model)];pins[str(model)]=sha(model)
 protected=read(O/'word100_balanced_batch_20261019/FINAL_AUDIT.json')['original_checkpoint_pins'];check(protected);pins.update(protected)
 for seed in SEEDS:
  for suffix in ['RESULT.json','history.json','last.pt']:
   p=BASE/f'training_C_{seed}_{suffix}';pins[str(p)]=sha(p)
 for r in rows:
  for info in r['views'].values():
   p=info['channel_file'];assert p and Path(p).resolve().is_relative_to((BASE/'data/clips').resolve())
   assert sha(p)==info['sha256']==original['pins'][p];pins[p]=info['sha256']
 c={'at':now(),'rows':rows,'pins':pins,'semantic_readiness_reused':str(BASE/'READY.json'),'no_new_extraction':True}
 train,development=candidate.loadrows(c);tensors=0;values=0
 for r in train+development:
  for a in r['arrays']['T3'].values():
   for x in a.values():validate_channels(x);tensors+=1;values+=x.size
 assert tensors==4196
 # Exact same original32x28+3, view schedule and tensor assembly.
 schedule_checks=0
 for seed in SEEDS:
  for epoch in range(1,81):
   order=np.random.default_rng(seed+epoch).permutation(899);parts=[order[i:i+32] for i in range(0,899,32)]
   assert list(map(len,parts))==[32]*28+[3] and len(set(int(i) for p in parts for i in p))==899
   for i in order:
    cid=train[i]['clip_id'];parity=int(hashlib.sha256(cid.encode()).hexdigest(),16)%2;sp=int(hashlib.sha256(('sensor_view:'+cid).encode()).hexdigest(),16)%2
    assert schedule(cid,epoch,'C')==('sensor' if ((epoch-1)//2+sp)%2 else 'masked','x_context' if (epoch+parity)%2 else 'x_whole');schedule_checks+=1
 spec=importlib.util.spec_from_file_location('original_c_driver',BASE/'training_driver.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 assemblies=0
 for subset in [train[:32],train[-3:],development[:32],development[-12:]]:
  legacy=[{**r,'arrays':r['arrays']['T3']} for r in subset]
  for view in ['masked','sensor']:
   a=candidate.batch(subset,'T3',view=view);b=old.torch_batch(legacy,view=view);assert all(torch.equal(x,y) for x,y in zip(a[:3],b));assemblies+=1
  if subset[0]['role']=='train_candidate':
   for epoch in range(1,5):
    a=candidate.batch(subset,'T3',epoch=epoch);b=old.torch_batch(legacy,epoch=epoch,arm='C');assert all(torch.equal(x,y) for x,y in zip(a[:3],b));assemblies+=1
 # Literal existing trainer methods except isolated architecture/initialization.
 def funcs(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(Path(p).read_text(encoding='utf8')).body if isinstance(x,ast.FunctionDef)}
 of=funcs(O/'word100_balanced_batch_20261019/train.py');nf=funcs(U/'train.py')
 unchanged=['statehash','savept','batch','scores']
 for name in unchanged:assert of[name]==nf[name]
 tests={};initial={};max_initial=0.;shared_grad=0.;phase_grad=[]
 x,p,y,_=candidate.batch(train[:8],'T3',epoch=1);x0=x.clone();p0=p.clone()
 for seed in SEEDS:
  torch.manual_seed(seed);base=MaskedSignTCN(**MODEL);rng=torch.get_rng_state().clone()
  new=candidate.initialize(seed)
  assert torch.equal(rng,torch.get_rng_state()) and torch.count_nonzero(new.phase_weight)==0
  shared={k:v for k,v in new.state_dict().items() if k!='phase_weight'}
  assert candidate.statehash(shared)==candidate.statehash(base.state_dict())==read(BASE/f'training_C_{seed}_RESULT.json')['initial_state_sha256']
  assert base.parameter_count()==37636 and new.parameter_count()==47236
  initial[str(seed)]={'shared_state_sha256':candidate.statehash(shared),'full_state_sha256':candidate.statehash(new.state_dict()),'rng_equal':True}
  base.eval();new.eval()
  with torch.no_grad():a=base(x,p);b=new(x,p)
  assert torch.equal(a,b);max_initial=max(max_initial,float((a-b).abs().max()))
  # Same dropout random draws -> same initial train forward and shared gradients.
  base.train();new.train();rr=torch.get_rng_state().clone();a=base(x,p);torch.set_rng_state(rr);b=new(x,p);assert torch.equal(a,b)
  torch.nn.functional.cross_entropy(a,y).backward();torch.nn.functional.cross_entropy(b,y).backward()
  for name,param in base.named_parameters():
   q=dict(new.named_parameters())[name];assert torch.allclose(param.grad,q.grad,atol=2e-6,rtol=2e-6);shared_grad=max(shared_grad,float((param.grad-q.grad).abs().max()))
  assert torch.isfinite(new.phase_weight.grad).all() and new.phase_weight.grad.norm()>0
  phase_grad.append(float(new.phase_weight.grad.norm()))
 tests['three_seed_initial_RNG_weights_eval_and_train_output_shared_gradients']=True
 tests['zero_head_has_finite_nonzero_gradient']=True
 assert torch.equal(x,x0) and torch.equal(p,p0);tests['input_and_masks_not_modified']=True
 # Pooling reference on every TRAIN whole/context view; existing source timestamps audit reused.
 maxerr=0.;seqs=0
 with torch.no_grad():
  for r in train:
   for arrays in r['arrays']['T3'].values():
    for a in arrays.values():
     xx=torch.from_numpy(a)[None];pp=torch.ones(1,len(a));w=phase_weights(xx,pp)[0].numpy()
     t=np.cumsum(a[:,279].astype(float));u=(t-t[0])/(t[-1]-t[0])
     independent=np.array([[max(1-2*z,0),1-abs(2*z-1),max(2*z-1,0)] for z in u])
     err=float(abs(w-independent).max());maxerr=max(maxerr,err);assert err<2e-6
     assert np.allclose(w.sum(1),1,atol=2e-7) and (w>=0).all();seqs+=1
 tests['all3596TRAIN_phase_weights_match_scalar_reference']=seqs==3596
 # Short/zero mass/all missing/all padding; padding has arbitrary finite sentinels.
 new=candidate.initialize(SEEDS[0]);new.eval()
 with torch.no_grad():new.phase_weight.fill_(.03)
 for n in [1,2,9,31]:
  xx=torch.randn(2,n,280);xx[:,:,279]=.033;xx[:,0,279]=0;pp=torch.ones(2,n);pp[1]=0
  w=phase_weights(xx,pp);h=torch.randn(2,32,n);g,d=summarize(h,pp,w)
  assert torch.isfinite(g).all() and torch.isfinite(d).all() and torch.count_nonzero(d[1])==0
  for i in range(2):
   valid=pp[i].bool();hh=h[i,:,valid].T.numpy();ww=w[i,valid].numpy();gg=hh.mean(0) if len(hh) else np.zeros(32)
   parts=[(hh*ww[:,k,None]).sum(0)/ww[:,k].sum() if len(hh) and ww[:,k].sum()>0 else gg for k in range(3)]
   assert np.allclose(g[i].numpy(),gg,atol=2e-6) and np.allclose(d[i].numpy(),(np.array(parts)-gg).reshape(-1),atol=2e-6)
  with torch.no_grad():
   aa=new(xx,pp);bb=new(torch.cat([xx,torch.full((2,17,280),13.)],1),torch.cat([pp,torch.zeros(2,17)],1))
   assert torch.isfinite(aa).all() and torch.allclose(aa,bb,atol=2e-5,rtol=2e-5)
 tests['short_empty_mass_all_padding_and_scalar_pooling']=True;tests['eval_padding_invariance_nonzero_head']=True
 # Missing-hand mask has no effect on time placement; no collapsing missing frames.
 xx=x[:1].clone();pp=p[:1].clone();ww=phase_weights(xx,pp);xx[:,1:4,:279]=0
 assert torch.equal(ww,phase_weights(xx,pp));tests['missing_real_frames_keep_timeline']=True
 # Dropout-free training padding invariance, including BN buffers.
 config={**MODEL,'dropout':0.};torch.manual_seed(7);a=TemporalPhaseTCN(**config);b=copy.deepcopy(a)
 with torch.no_grad():a.phase_weight.fill_(.01);b.phase_weight.fill_(.01)
 aa=a(x,p);bb=b(torch.cat([x,torch.zeros(len(x),17,280)],1),torch.cat([p,torch.zeros(len(p),17)],1))
 assert torch.allclose(aa,bb,atol=2e-5,rtol=2e-5)
 for k,v in a.state_dict().items():assert torch.allclose(v,b.state_dict()[k],atol=2e-6,rtol=2e-6)
 tests['train_padding_invariance_dropout_disabled_including_BN']=True
 # Finite backward with degenerate actual batch; does not train a saved model.
 model=candidate.initialize(SEEDS[0]);xx=torch.zeros(2,2,280);pp=torch.tensor([[1.,0.],[0.,0.]])
 loss=model(xx,pp).square().mean();loss.backward();assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
 tests['degenerate_backward_finite']=True
 assert all(tests.values());check(pins)
 write(U/'PREP_CONTEXT.json',c,new=True)
 own={str(U/n):sha(U/n) for n in ['AUTHORIZATION.json','FROZEN_PROTOCOL.json','prepare.py','train.py','common.py','temporal.py','PREP_CONTEXT.json','verify_results.py','matched_window_audit.py']}
 write(U/'READY.json',{'at':now(),'status':'PASS','training_allowed':True,'blockers':[],'pins':own,'train':899,'dev':300,'input_npz_files':2398,'input_tensors':tensors,'channel_values_checked':values,'hash_identical_to_original_C':True,'original_semantic_readiness_reused':str(BASE/'READY.json'),'batch_sizes':[32]*28+[3],'epochs_checked':240,'schedule_checks':schedule_checks,'tensor_assembly_cases':assemblies,'unchanged_function_AST':unchanged,'initial_states':initial,'parameters':{'C':37636,'T3':47236},'tests':tests,'initial_output_max_error':max_initial,'initial_shared_gradient_max_error':shared_grad,'phase_gradient_norms':phase_grad,'phase_scalar_max_error':maxerr,'phase_sequences_checked':seqs,'user_data_used':False,'new_training_runs':0,'verification':'Scalar reference and original C function/initialization comparisons by same Codex; no independent person/agent.','no_missing_imputation_or_new_extraction':True},new=True)
 event(stage='READY_FOR_THREE_T3_RUNS',tests=len(tests),train=899,dev=300,tensors=tensors,initial_output_max_error=max_initial)

if __name__=='__main__':
 try:main()
 except BaseException:
  if not (U/'FIRST_PREP_ERROR.json').exists():write(U/'FIRST_PREP_ERROR.json',{'at':now(),'traceback':traceback.format_exc()},new=True)
  raise
