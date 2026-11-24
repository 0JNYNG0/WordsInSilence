from common import *
import copy,shutil,torch,traceback
from objective import objective,LAMBDA_AUX
from multi_position_aux import phase_readout
from frozen_order_aux import FrozenOrderAux
from temporal import phase_weights

def numpy_gru(seq,state):
 h=np.zeros(16,dtype=np.float64);out=[]
 w={k:v.detach().numpy().astype(np.float64) for k,v in state.items()}
 for x in seq:
  gi=w['weight_ih_l0']@x+w['bias_ih_l0'];gh=w['weight_hh_l0']@h+w['bias_hh_l0']
  r=1/(1+np.exp(-(gi[:16]+gh[:16])));z=1/(1+np.exp(-(gi[16:32]+gh[16:32])))
  n=np.tanh(gi[32:]+r*gh[32:]);h=(1-z)*n+z*h;out.append(h.copy())
 return np.array(out)

def phase_reference(sequence,dt):
 elapsed=np.cumsum(dt.astype(np.float64));span=elapsed[-1]-elapsed[0]
 u=(elapsed-elapsed[0])/span if span>0 else np.full(len(dt),.5)
 weights=np.array([[max(0,1-2*t),1-abs(2*t-1),max(0,2*t-1)] for t in u])
 result=[]
 for i in range(3):
  mass=weights[:,i].sum()
  result.append((sequence*weights[:,i,None]).sum(0)/mass if mass>0 else sequence.mean(0))
 return np.concatenate(result),weights

def prepare_experiment(initialize,loadrows,batch,MODEL,TRAIN,NEW_NAMES):
 try:_prepare(initialize,loadrows,batch,MODEL,TRAIN,NEW_NAMES)
 except BaseException:
  if not (U/'FIRST_PREP_ERROR.json').exists():write(U/'FIRST_PREP_ERROR.json',{'at':now(),'traceback':traceback.format_exc()},new=True)
  raise

def _prepare(initialize,loadrows,batch,MODEL,TRAIN,NEW_NAMES):
 assert not (U/'TRAIN_CONTEXT.json').exists() and not (U/'AUTHORIZATION.json').exists()
 plan=read(D/'NEXT_PLAN.json');assert plan['proposed_candidate']['parameter_budget']['new_trainable_total']==7200
 previous=read(A/'TRAIN_CONTEXT.json');check(previous['pins'])
 for folder in [A,D]:
  for name,h in read(folder/'FINAL_AUDIT.json')['files'].items():assert sha(folder/name)==h,(folder,name)
 assert not read(A/'CURRENT_EXECUTION.json')['running'] and not (A/'TRAIN_RUNNING.json').exists()
 write(U/'AUTHORIZATION.json',{'at':now(),'request':'그 보완방법대로 진행해줘','approved_plan':str(D/'NEXT_PLAN.json'),
  'scope':'Isolated multi-position readout implementation, preflight, if passed three-seed training, saved comparison and report. No production replacement.',
  'overrides_prior_plan_wait':True,'user_mentor_data_forbidden':True,'existing_C_T3F_unchanged':True,'implementer':'Codex local',
  'verification':'Separate NumPy reference and saved trace/score verifier by same Codex; not an independent agent.'},new=True)
 protocol={'at':now(),'plan_sha256':sha(D/'NEXT_PLAN.json'),'plan':plan,'seeds':SEEDS,'model':MODEL,'training':TRAIN,
  'arm':ARM,'lambda_aux':LAMBDA_AUX,'frozen_parameters':47236,'trainable_parameters':7200,'total_parameters':54436,
  'readout':'Existing elapsed triangular3 weighted means over all real GRU hidden states, concat48 then zero bias-free100head. Zero mass bin fallback to whole real hidden mean.',
  'loss':'CE(b+a,y)+0.1CE(a,y)','initialization':'Fresh matching-seed T3F and exactly prior fresh GRU; new head zero, no resume.',
  'selection':'Mean final masked/sensor DEV CE, strict1e-6, max80, patience12; not aux accuracy or perturbations.',
  'success_gate':read(A/'FROZEN_PROTOCOL.json')['plan']['comparison']['success_gate'],
  'budget_confound':'Output head has3200 extra parameters; any gain belongs to complete candidate, not solely order causation.',
  'no_order_loss':True,'no_per_word_rules':True,'no_automatic_adoption':True,'no_retuning':True,'new_runs':3,'personal_data_read':0}
 write(U/'FROZEN_PROTOCOL.json',protocol,new=True)
 train,dev=loadrows(previous)
 assert previous['rows']==read(BASE/'training_CONTEXT.json')['ordered_rows']
 assert sha(U/'objective.py')==sha(A/'objective.py')
 # Reuse the unchanged objective's prior independent numerical/label-permutation proof.
 prior_math=read(A/'PREFLIGHT.json')['loss_math']
 assert len(prior_math)==3 and max(q['max_math_error'] for q in prior_math)<1e-12
 initial=[];tests=[];zero_checks=0;maxerr=0.;padding_error=0.;GRU_initial=[]
 for seed in SEEDS:
  m=initialize(seed);init=statehash(m.state_dict());frozen=statehash(m.base.state_dict())
  torch.manual_seed(seed);old=FrozenOrderAux(MODEL,m.base.state_dict());old.eval()
  assert all(torch.equal(v,old.memory.state_dict()[k]) for k,v in m.memory.state_dict().items())
  assert preflight_hash(old)==read(P/'RESULT.json')['initial_state_hashes'][str(seed)]
  memory_hash=statehash(m.memory.state_dict());GRU_initial.append({'seed':seed,'memory_sha256':memory_hash,'old_fresh_memory_exact':True})
  # Whole and context inputs for every TRAIN video in both views: no fitting or DEV selection.
  m.eval()
  with torch.no_grad():
   for view in VIEWS:
    for key in ['x_whole','x_context']:
     for j in range(0,899,32):
      rr=train[j:j+32];xx,pp=padded_arrays([r['arrays'][view][key] for r in rr])
      x=torch.from_numpy(xx);pad=torch.from_numpy(pp);prior=x.clone();pm=pad.clone()
      b,a,readout,h,lengths=m.components(x,pad);ob,oa,_,_,_=old.components(x,pad)
      assert torch.equal(b,ob) and torch.count_nonzero(a)==0 and torch.equal(b+a,ob+oa)
      assert torch.equal(x,prior) and torch.equal(pad,pm)
      assert readout.shape==(len(rr),48) and torch.isfinite(readout).all()
      weights=phase_weights(x,pad);assert torch.allclose(weights.sum(-1),pad,rtol=0,atol=1e-6)
      zero_checks+=len(rr)
  assert statehash(m.state_dict())==init
  # Independent NumPy recurrence and temporal pooling on synthetic variable-length batch.
  gen=torch.Generator().manual_seed(771);features=torch.randn(4,17,32,generator=gen);lengths=torch.tensor([7,17,1,11])
  pad=(torch.arange(17)[None,:]<lengths[:,None]).float();x=torch.zeros(4,17,280)
  for i,n in enumerate(lengths):x[i,1:n,279]=torch.linspace(.025,.08,int(n)-1)
  features[1,5:8]=0 # Real missing-like hidden values remain in recurrence, never padding.
  before=features.clone()
  with torch.no_grad():
   hs=m.memory_sequence(features,lengths);weights=phase_weights(x,pad);out=phase_readout(hs,pad,weights)
   assert torch.equal(features,before) and hs.shape==(4,17,16)
   for i,n in enumerate(lengths):
    seq=numpy_gru(features[i,:n].numpy(),m.memory.state_dict());pool,ww=phase_reference(seq,x[i,:n,279].numpy())
    err=max(float(np.max(np.abs(seq-hs[i,:n].numpy()))),float(np.max(np.abs(pool-out[i].numpy()))),float(np.max(np.abs(ww-weights[i,:n].numpy()))))
    assert err<2e-6,err;maxerr=max(maxerr,err)
    assert torch.count_nonzero(hs[i,n:])==0
   assert torch.allclose(out[2,:16],out[2,16:32],atol=0,rtol=0) and torch.equal(out[2,:16],out[2,32:])
   # Model predictions independent of batch composition and extra trailing padding.
   xx,pp,_,_=batch(train[:3],epoch=1);b1,a1,r1,_,_=m.components(xx,pp)
   xp=torch.nn.functional.pad(xx,(0,0,0,15));ppad=torch.nn.functional.pad(pp,(0,15))
   b2,a2,r2,_,_=m.components(xp,ppad)
   error=max(float((b1-b2).abs().max()),float((r1-r2).abs().max()));padding_error=max(padding_error,error)
   assert error<1e-5
   singlex=xx[1:2,:int(pp[1].sum())];singlep=torch.ones(1,singlex.shape[1])
   sb,sa,sr,_,_=m.components(singlex,singlep)
   error=max(float((sb[0]-b1[1]).abs().max()),float((sr[0]-r1[1]).abs().max()));padding_error=max(padding_error,error)
   assert error<1e-5
  # Fresh model gets no optimizer step; first zero head and subsequent fixed-head gradient path.
  m.train();xx,pp,y,_=batch(train[:8],epoch=1);b,a,_,_,_=m.components(xx,pp);objective(b,a,y)[0].backward()
  assert all(q.grad is None for q in m.base.parameters())
  assert m.residual.weight.grad.norm()>0 and all(torch.count_nonzero(q.grad)==0 for q in m.memory.parameters())
  probe=copy.deepcopy(m);probe.zero_grad(set_to_none=True)
  with torch.no_grad():probe.residual.weight.copy_(torch.arange(4800).reshape(100,48).sin()*.001)
  b,a,_,_,_=probe.components(xx,pp);objective(b,a,y)[0].backward()
  assert all(q.grad is None for q in probe.base.parameters())
  assert all(q.grad is not None and torch.isfinite(q.grad).all() and q.grad.norm()>0 for q in probe.memory.parameters())
  assert statehash(m.state_dict())==init and statehash(probe.base.state_dict())==frozen
  initial.append({'seed':seed,'full_state_sha256':init,'frozen_state_sha256':frozen,'memory_sha256':memory_hash})
  tests.append({'seed':seed,'packed_order_numpy_equal':True,'no_input_mutation':True,'zero_mass_fallback':True,'padding_batch_invariance':True,
   'initial_output_exact':True,'zero_head_memory_first_grad_zero':True,'nonzero_head_memory_grad_positive':True,'base_no_gradient':True,'optimizer_steps':0})
  print(json.dumps({'stage':'PREFLIGHT_SEED_PASS','seed':seed,'exact_initial_train_examples_cumulative':zero_checks}),flush=True)
 assert zero_checks==10788
 # General output/readout rule, no word branches; inherited loss checks already cover relabeling.
 with torch.no_grad():
  features=torch.randn(5,48,generator=torch.Generator().manual_seed(19))
  for k in [7,100,137]:
   head=torch.randn(k,48,generator=torch.Generator().manual_seed(k));perm=torch.randperm(k,generator=torch.Generator().manual_seed(k+1))
   assert torch.equal(torch.nn.functional.linear(features,head)[...,perm],torch.nn.functional.linear(features,head[perm]))
 pf={'status':'PASS','at':now(),'exact_initial_T3F_cases':zero_checks,'new_trainable_parameters':7200,'total_parameters':54436,'GRU_initial':GRU_initial,
  'tests':tests,'numpy_recurrence_and_pool_max_error':maxerr,'max_padding_or_batch_error':padding_error,'no_optimizer_steps':True,
  'loss_math_reused':{'source':str(A/'PREFLIGHT.json'),'sha256':sha(A/'PREFLIGHT.json'),'unchanged_objective_sha256':sha(U/'objective.py')},
  'general_readout_class_permutation':[7,100,137],'input_validation_reused':str(A/'TRAIN_CONTEXT.json'),'user_recordings_read':0}
 write(U/'PREFLIGHT.json',pf,new=True)
 # Evidence for evaluating sequence claims is reused; no unsupported semantic labels are invented.
 audit=O/'word100_sequence_input_audit_20261019'
 evidence={'status':'PUBLIC_OBSERVATION_EVIDENCE_REUSED_SEMANTIC_LABELS_NOT_ESTABLISHED','at':now(),
  'sources':{str(audit/n):sha(audit/n) for n in ['SUMMARY.json','VISUAL_REVIEW.json','VISUAL_MANIFEST.json','LOCAL_ANCHOR_AUDIT.json','NEXT_PLAN.json']},
  'protocol':'Re-use all100 public reference visual/input audit. Do not label arbitrary reversed/cropped signs wrong. New classifier comparison uses unchanged word labels only.',
  'distinctions':['normal repeated movement','actual sensor gap','out-of-frame hand','observable sequence change versus semantic error'],
  'required_for_future_completion_claims':'Competent sign-language verification of mandatory stages and valid omission/order contrasts; not supplied by current word labels.',
  'semantic_stage_accuracy_reportable':False,'new_private_or_heldout_read':0}
 write(U/'EVALUATION_EVIDENCE.json',evidence,new=True)
 pins=dict(previous['pins'])
 for p,h in evidence['sources'].items():pins[p]=h
 files=[U/n for n in ['AUTHORIZATION.json','FROZEN_PROTOCOL.json','common.py','train.py','objective.py','multi_position_aux.py','preparation.py','verify.py','verify_aux.py','PREFLIGHT.json','EVALUATION_EVIDENCE.json']]
 files += [A/'FINAL_AUDIT.json',A/'TRAIN_CONTEXT.json',A/'COMPARISON_INDEPENDENT.json',D/'NEXT_PLAN.json',D/'FINAL_AUDIT.json']
 for seed in SEEDS:
  files += [A/f'FG16AUX01_{seed}_{suffix}' for suffix in ['best.pt','RESULT.json','history.json']]
 for p in files:pins[str(p)]=sha(p)
 context={'at':now(),'arms':[ARM],'seeds':SEEDS,'model':MODEL,'training':TRAIN,'lambda_aux':.1,'pins':pins,'rows':previous['rows'],
  'initial_states':initial,'labels':previous['labels'],'word_codes':previous['word_codes'],'missingness_groups':previous['missingness_groups'],
  'ordered_train_ids':[r['clip_id'] for r in train],'trainable_parameter_names':NEW_NAMES,'runtime':{'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__},
  'general_word_rule':True,'evaluation_scope_words':100,'all_sign_language_validated':False,'user_recordings_read':0}
 check(pins);write(U/'TRAIN_CONTEXT.json',context,new=True)
 write(U/'READY.json',{'status':'PASS','training_allowed':True,'context_sha256':sha(U/'TRAIN_CONTEXT.json'),'preflight_sha256':sha(U/'PREFLIGHT.json'),'new_runs':3},new=True)
 backup=U/'before';backup.mkdir(exist_ok=False)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,backup/p.name)
 event(stage='READY_APPROVED_M3_THREE_SEEDS',train=899,dev=300,trainable=7200,preflight_optimizer_steps=0)
