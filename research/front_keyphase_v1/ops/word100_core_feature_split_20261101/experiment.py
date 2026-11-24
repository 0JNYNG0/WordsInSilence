"""One prespecified six-run system comparison; no automatic tuning or deployment."""
from pathlib import Path
import sys,json,time,hashlib,os,traceback
sys.dont_write_bytecode=True
import numpy as np
import torch
import comparison as c
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 Path(p).parent.mkdir(exist_ok=True,parents=True)
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def validate_frozen(f):
 for p,h in {**f['pins'],**f['source_pins'],**f['protected']}.items():assert sha(p)==h,p
def freeze():
 audit=read(U/'LABEL_AUDIT.json');assert audit['status']=='READY_PROVISIONAL_PHYSICAL_FEATURE_STUDY'
 assert read(U/'review/INPUT_NUMERIC_VERIFICATION.json')['status'].startswith('PASS')
 assert read(U/'review/INPUT_LABEL_VERIFICATION.json')['status'].startswith('PASS')
 contract=read(U/'review/MODEL_CONTRACT.json')
 assert contract['selftest']['status']=='PASS_SYNTHETIC_INTERFACE_AND_FREEZE_ONLY'
 assert sha(U/'comparison.py')==contract['implementation_sha256']
 assert read(U/'review/TRAIN_CODE_REVIEW.json')['status'].startswith('PASS')
 info=read(U/'DATASET.json');sel=read(U/'SELECTION.json')
 assert info['selected_targets']==read(U/'RUBRIC.json')['fixed_comparison_targets']==contract['mapping']['joint_targets']
 for p,h in {**info['pins'],**sel['protected']}.items():assert sha(p)==h,p
 paths=[U/n for n in ['AUTHORIZATION.json','PLAN_KO.txt','RUBRIC.json','SELECTION.json','DATASET.json','LABEL_AUDIT.json','comparison.py','experiment.py','build_targets.py','review/MODEL_CONTRACT.json','review/TRAIN_CODE_REVIEW.json','review/INPUT_LABEL_VERIFICATION.json','review/INPUT_NUMERIC_VERIFICATION.json']]
 write(U/'TRAIN_FROZEN.json',dict(status='FROZEN_BEFORE_NEW_SCORES',conditions=['joint','protected_shape_split_motion'],seeds=[42,43,44],epochs=160,lr=.001,weight_decay=.01,threshold=.5,
  selection='Fixed final epoch, no score-based target/threshold/epoch/model changes',targets=info['selected_targets'],fit_samples=[r['sample_id'] for r in info['rows'] if r['role']=='fit'],
  comparison_scope='System comparison with same raw inputs/new fit motion exposure and same initial motion trunk. Split additionally retains prior-trained shape branch; unequal total/trainable parameter counts and prior shape exposure disclosed. Not a pure architecture-causality claim.',
  protected=sel['protected'],pins={str(p):sha(p) for p in paths},source_pins={**info['pins'],**contract['source_pins']},
  interpretation='Per target and family held-expression results take priority over total average. A family improvement cannot erase shape/motion regressions. Unsupported denominators remain descriptive, unknown labels never correct/incorrect.',
  no_production_adoption=True,no_linguistic_correctness_claim=True))
 print('FROZEN six-run system comparison')
def train():
 assert not (U/'TRAINING.json').exists() and not (U/'TRAIN_RUNNING.json').exists()
 f=read(U/'TRAIN_FROZEN.json');info=read(U/'DATASET.json')
 validate_frozen(f)
 write(U/'TRAIN_RUNNING.json',dict(pid=os.getpid(),started=time.time()))
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 batch,stats=c.assemble(info);write(U/'NORMALIZATION.json',stats)
 fit=[i for i,r in enumerate(batch.rows) if r['role']=='fit']; fb=batch.subset(fit)
 allresults=[]
 for seed in f['seeds']:
  joint,split=c.build_pair(seed,tuple(info['selected_targets']))
  # Existing48 use their stored outputs; only new48 run the protected head once.
  frozen_start=time.perf_counter();protected,pa=c.complete_protected_probabilities(batch,seed,allow_new_inference=True,frozen=split.frozen)
  pa['new_frozen_inference_seconds']=time.perf_counter()-frozen_start
  cache=U/'protected'/f'probabilities_{seed}.npy';cache.parent.mkdir(exist_ok=True)
  with cache.open('xb') as out:np.save(out,protected.numpy(),allow_pickle=False)
  pa['probabilities_sha256']=sha(cache);write(U/'protected'/f'AUDIT_{seed}.json',pa)
  stored,missing=c.stored_old_shape_probabilities(info,seed)
  old=[i for i in range(len(batch.rows)) if i not in set(missing)]
  assert np.array_equal(stored[old],protected.numpy()[old])
  frozen_before={k:v.clone() for k,v in split.frozen.state_dict().items()}
  for name,model in [('joint',joint),('protected_shape_split_motion',split)]:
   dest=U/'runs'/f'{name}_{seed}';dest.mkdir(parents=True,exist_ok=False)
   ids=list(model.trainable_target_indices);cw,ew=c.training_weights(batch,fit,ids)
   optimizer=torch.optim.AdamW(model.optimizer_parameters(),lr=f['lr'],weight_decay=f['weight_decay'])
   total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
   start=time.perf_counter();losses=[]
   with (dest/'EXPOSURES.jsonl').open('x',encoding='utf8') as log:
    for epoch in range(f['epochs']):
     model.train();optimizer.zero_grad();logits=model(fb)
     loss=c.masked_loss(logits,fb.y[:,ids],fb.mask[:,ids],cw,ew[fit]);assert torch.isfinite(loss)
     loss.backward();torch.nn.utils.clip_grad_norm_(list(model.optimizer_parameters()),1);optimizer.step();losses.append(float(loss.detach()))
     log.write(json.dumps(dict(epoch=epoch+1,sample_ids=f['fit_samples'],learned_targets=[batch.targets[i] for i in ids],loss=losses[-1]))+'\n')
   seconds=time.perf_counter()-start;model.eval()
   with torch.no_grad():
    p=model(batch).sigmoid()
    if name!='joint':p=split.combine_probabilities(p,protected)
    prob=p.numpy()
   if name!='joint':
    assert all(torch.equal(v,split.frozen.state_dict()[k]) for k,v in frozen_before.items())
    assert np.array_equal(prob[:,list(split.shape_indices)],protected.numpy())
   # Only newly learned branch is saved; original shape checkpoint remains read-only.
   torch.save(model.head.state_dict(),dest/'head.pt')
   with (dest/'probabilities.npy').open('xb') as out:np.save(out,prob,allow_pickle=False)
   result=dict(condition=name,seed=seed,epochs=f['epochs'],selected_epoch=f['epochs'],training_seconds=seconds,trainable_parameters=trainable,total_parameters=total,learned_targets=[batch.targets[i] for i in ids],losses=losses,
    checkpoint_sha256=sha(dest/'head.pt'),probabilities_sha256=sha(dest/'probabilities.npy'),exposures_sha256=sha(dest/'EXPOSURES.jsonl'),old_clip_reinference=0,protected_probability_sha256=sha(cache),old_shape_weights_unchanged=True)
   write(dest/'COMPLETE.json',result);allresults.append({k:result[k] for k in ['condition','seed','training_seconds','trainable_parameters','total_parameters']});print(json.dumps(allresults[-1]),flush=True)
 assert all(sha(p)==h for p,h in f['protected'].items())
 write(U/'TRAINING.json',dict(status='COMPLETE_SIX_FIXED_RUNS',runs=allresults,backbone_updates=0,existing_shape_updates=0))
 (U/'TRAIN_RUNNING.json').unlink()
def metrics(p,y,m,names):
 items=[]
 for j,name in enumerate(names):
  pred=p[:,j]>=.5;gt=y[:,j];known=m[:,j]
  tp=int((known&(gt==1)&pred).sum());tn=int((known&(gt==0)&~pred).sum());fp=int((known&(gt==0)&pred).sum());fn=int((known&(gt==1)&~pred).sum());n=tp+tn+fp+fn
  items.append(dict(target=name,known=n,unknown=int((~known).sum()),tp=tp,tn=tn,fp=fp,fn=fn,balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None))
 good=[r['balanced_accuracy'] for r in items if r['balanced_accuracy'] is not None]
 return dict(targets=items,macro_balanced_accuracy=float(np.mean(good)) if good else None,balanced_supported_targets=len(good))
def score():
 info=read(U/'DATASET.json');f=read(U/'TRAIN_FROZEN.json');ids=info['selected_indices'];names=info['selected_targets']
 validate_frozen(f)
 y=np.asarray([r['y'] for r in info['rows']])[:,ids];m=np.asarray([r['target_mask'] for r in info['rows']])[:,ids];out=[]
 families={'all':list(range(len(names))),'shape':[i for i,n in enumerate(names) if n in c.SHAPE_TARGETS],'relation':[i for i,n in enumerate(names) if n.endswith('higher')],'motion':[i for i,n in enumerate(names) if '_motion_' in n]}
 runs=read(U/'TRAINING.json')['runs'];assert len(runs)==6 and {(r['condition'],r['seed']) for r in runs}=={(a,s) for a in f['conditions'] for s in f['seeds']}
 for rr in runs:
  d=U/'runs'/f"{rr['condition']}_{rr['seed']}";complete=read(d/'COMPLETE.json')
  for file,key in [('probabilities.npy','probabilities_sha256'),('head.pt','checkpoint_sha256'),('EXPOSURES.jsonl','exposures_sha256')]:assert sha(d/file)==complete[key]
  p=np.load(d/'probabilities.npy',allow_pickle=False)
  assert p.shape==y.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
  for role in sorted({r['role'] for r in info['rows']}):
   rows=[i for i,r in enumerate(info['rows']) if r['role']==role]
   for fam,cols in families.items():out.append(dict(condition=rr['condition'],seed=rr['seed'],role=role,family=fam,windows=len(rows),**metrics(p[np.ix_(rows,cols)],y[np.ix_(rows,cols)],m[np.ix_(rows,cols)],[names[i] for i in cols])))
 summary=[]
 for condition in f['conditions']:
  for role in sorted({r['role'] for r in info['rows']}):
   for family in families:
    rr=[r for r in out if r['condition']==condition and r['role']==role and r['family']==family];v=[r['macro_balanced_accuracy'] for r in rr if r['macro_balanced_accuracy'] is not None]
    summary.append(dict(condition=condition,role=role,family=family,mean=float(np.mean(v)) if v else None,range=[min(v),max(v)] if v else None,balanced_supported_targets=rr[0]['balanced_supported_targets']))
 write(U/'SCORES.json',dict(status='FIXED_STORED_SCORE_COMPARISON',rows=out,summary=summary,targets=names,not_word_or_educational_accuracy=True,broader_target_claim_support=read(U/'LABEL_AUDIT.json')['evaluation_supported_for_broader_target_claim']))
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
 try:{'freeze':freeze,'train':train,'score':score}[sys.argv[1]]()
 except BaseException:
  p=U/f'{sys.argv[1].upper()}_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
