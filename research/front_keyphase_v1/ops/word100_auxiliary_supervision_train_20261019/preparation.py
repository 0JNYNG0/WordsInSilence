from common import *
import copy,shutil,torch
from objective import objective,LAMBDA_AUX

def prepare_experiment(initialize,loadrows,batch,MODEL,TRAIN,NEW_NAMES):
 assert not (U/'TRAIN_CONTEXT.json').exists()
 plan=read(R/'PROPOSED_PROTOCOL.json');assert plan['objective']['lambda_aux']==LAMBDA_AUX==0.1
 for folder in [R,F,P]:
  for name,h in read(folder/'FINAL_AUDIT.json')['files'].items():assert sha(folder/name)==h,(folder,name)
 assert read(F/'CURRENT_EXECUTION.json')['running'] is False and read(F/'EXIT.json')['exit_code']==0
 assert not (F/'TRAIN_RUNNING.json').exists()
 check(read(R/'SCOPE.json')['source_pins'])
 auth={'at':now(),'request':'기존 모델을 멋대로 수정 및 바꾸거나 하지 않는 선에서 계획 진행 순서에 맞게 모두 작업 진행. 모든 수어 단어에 범용 적용 목표. 사용자는 실사용 테스터, 학습·튜닝 자료 아님.',
  'approved_plan':str(R/'PROPOSED_PROTOCOL.json'),'scope':'Isolated auxiliary classification objective implementation, preflight, three runs and saved-score comparison',
  'overrides_prior_plan_wait':True,'existing_models_preserved':True,'no_user_mentor_data':True,
  'generality':'Word-independent learning objective; current100classes are the experiment scope, not proof for all signs. Future vocabulary expansion needs matching data/class head/retraining/validation.',
  'implementer':'Codex local','verification':'Separate verifier and numerical reference by same Codex; no independent agent'}
 write(U/'AUTHORIZATION.json',auth,new=True)
 protocol={'at':now(),'approved_plan_sha256':sha(R/'PROPOSED_PROTOCOL.json'),'plan':plan,'training':TRAIN,'model':MODEL,'seeds':SEEDS,'lambda_aux':LAMBDA_AUX,
  'frozen':47236,'trainable':4000,'no_per_word_rules':True,'selection':'mean FINAL DEV masked/sensor CE; strict1e-6,patience12,max80','runtime_wall_limit_per_seed_seconds':3600,
  'no_automatic_adoption':True,'do_not_retune':True,'public_input_unchanged':True,'no_user_data':True,'new_runs':3}
 write(U/'FROZEN_PROTOCOL.json',protocol,new=True)
 previous=read(F/'TRAIN_CONTEXT.json');check(previous['pins']);original=read(BASE/'training_CONTEXT.json')
 assert previous['rows']==original['ordered_rows']
 train,dev=loadrows(previous)
 # Independent double-precision loss and gradient math; the function is class-count agnostic.
 numerical=[]
 for classes in [7,100,137]:
  gen=torch.Generator().manual_seed(47+classes)
  b=torch.randn(5,classes,generator=gen,dtype=torch.float64,requires_grad=True)
  a=torch.randn(5,classes,generator=gen,dtype=torch.float64,requires_grad=True)
  y=torch.arange(5,dtype=torch.long)%classes
  total,fc,ac=objective(b,a,y);gg=torch.autograd.grad(total,a)[0].numpy()
  bn=b.detach().numpy();an=a.detach().numpy();yn=y.numpy()
  def ref(z):
   z=z-z.max(1,keepdims=True);e=np.exp(z);p=e/e.sum(1,keepdims=True)
   return np.mean(np.log(e.sum(1))-z[np.arange(5),yn]),p
  f,pf=ref(bn+an);v,pa=ref(an);gold=(pf+0.1*pa-1.1*np.eye(classes)[yn])/5
  err=max(abs(float(fc.detach())-f),abs(float(ac.detach())-v),float(np.max(abs(gg-gold))))
  assert err<1e-12 and b.grad is None
  # Relabeling every class consistently must not change the learning rule.
  perm=torch.randperm(classes,generator=gen);inv=torch.argsort(perm)
  relabeled=objective(b[:,perm],a[:,perm],inv[y])[0]
  assert torch.allclose(total,relabeled,rtol=0,atol=1e-12)
  control=objective(b,a,y,weight=0)[0]
  assert torch.equal(control,torch.nn.functional.cross_entropy(b.detach()+a,y))
  numerical.append({'classes':classes,'max_math_error':err,'consistent_label_permutation_invariant':True,'lambda0_equal':True})
 initial=[];gradient_checks=[]
 for seed in SEEDS:
  m=initialize(seed);h=statehash(m.state_dict());fh=statehash(m.base.state_dict())
  expected=next(z for z in previous['initial_states'] if z['seed']==seed)
  assert h==expected['full_state_sha256'] and fh==expected['frozen_state_sha256']
  x,p,y,_=batch(train[:8],epoch=1);m.train();b,a,_,_,_=m.components(x,p)
  assert torch.count_nonzero(a)==0 and torch.equal(b+a,m(x,p))
  loss,fc,ac=objective(b,a,y);loss.backward()
  assert all(q.grad is None for q in m.base.parameters())
  assert m.residual.weight.grad.norm()>0 and all(torch.count_nonzero(q.grad)==0 for q in m.memory.parameters())
  assert statehash(m.state_dict())==h
  # Fixed artificial head on a disposable copy, no optimizer step or tuning.
  probe=copy.deepcopy(m);probe.zero_grad(set_to_none=True)
  with torch.no_grad():probe.residual.weight.copy_(torch.arange(1600).reshape(100,16).sin()*0.001)
  pb,pa,_,_,_=probe.components(x,p);objective(pb,pa,y)[0].backward()
  assert all(q.grad is None for q in probe.base.parameters())
  assert all(q.grad is not None and torch.isfinite(q.grad).all() and q.grad.norm()>0 for q in probe.memory.parameters())
  assert statehash(probe.base.state_dict())==fh and statehash(m.state_dict())==h
  initial.append(expected)
  gradient_checks.append({'seed':seed,'initial_output_equal':True,'zero_head_first_memory_grad_zero':True,'fixed_nonzero_head_memory_gradient':True,'frozen_base_no_gradient':True,'optimizer_steps':0})
 preflight={'status':'PASS','at':now(),'loss_math':numerical,'gradients':gradient_checks,'original_initial_hashes_identical':True,'public_train':899,'public_dev':300,
  'unchanged_input_preflight_reused':str(P/'RESULT.json'),'optimizer_steps':0,'synthetic_head_probe_only':True,'user_recordings_read':0,'no_word_specific_branches':True}
 write(U/'PREFLIGHT.json',preflight,new=True)
 pins=dict(previous['pins']);pins.update(read(R/'SCOPE.json')['source_pins'])
 for p in [U/n for n in ['AUTHORIZATION.json','FROZEN_PROTOCOL.json','common.py','train.py','objective.py','preparation.py','verify.py','verify_aux.py','PREFLIGHT.json']]+[R/'PROPOSED_PROTOCOL.json',R/'FINAL_AUDIT.json',F/'TRAIN_CONTEXT.json',F/'COMPARISON_INDEPENDENT.json',F/'FINAL_AUDIT.json']:
  pins[str(p)]=sha(p)
 for seed in SEEDS:
  for suffix in ['_best.pt','_RESULT.json','_history.json']:
   path=F/f'FG16_{seed}{suffix}';pins[str(path)]=sha(path)
 ctx={'at':now(),'arms':[ARM],'seeds':SEEDS,'model':MODEL,'training':TRAIN,'lambda_aux':LAMBDA_AUX,'pins':pins,'rows':previous['rows'],
  'initial_states':initial,'labels':previous['labels'],'word_codes':previous['word_codes'],'missingness_groups':previous['missingness_groups'],
  'ordered_train_ids':[r['clip_id'] for r in train],'trainable_parameter_names':NEW_NAMES,'runtime':{'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__},
  'general_word_rule':True,'evaluation_scope_words':100,'all_sign_language_validated':False,'user_recordings_read':0}
 check(pins);write(U/'TRAIN_CONTEXT.json',ctx,new=True)
 write(U/'READY.json',{'status':'PASS','training_allowed':True,'context_sha256':sha(U/'TRAIN_CONTEXT.json'),'preflight_sha256':sha(U/'PREFLIGHT.json'),'new_runs':3},new=True)
 backup=U/'before';backup.mkdir(exist_ok=False)
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,backup/p.name)
 event(stage='READY_APPROVED_AUX01_THREE_SEEDS',train=899,dev=300,lambda_aux=LAMBDA_AUX,optimizer_preflight_steps=0)
