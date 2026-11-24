"""Validate frozen C input reuse and batch partition before any training."""
from common import *
from collections import Counter
import ast, math, importlib.util, traceback
from balanced import balanced_batches
import train as candidate
from training_core import validate_channels, schedule, MODEL, TRAIN, SEEDS

def main():
 assert not (U/'READY.json').exists() and not (U/'TRAIN_RUNNING.json').exists()
 original=read(BASE/'training_CONTEXT.json'); oldready=read(BASE/'READY.json')
 assert oldready['independent_input_validation_passed'] and oldready['fresh_sensor_detector_state_verified'] and oldready['roi_hidden_coordinate_invariance_passed']
 rows=original['ordered_rows'];assert len(rows)==1199 and len({r['clip_id'] for r in rows})==1199
 tr=[r for r in rows if r['role']=='train_candidate'];dev=[r for r in rows if r['role']=='development_only']
 assert len(tr)==899 and len(dev)==300
 assert set(r['signer_id'] for r in tr)==set(range(1,10))
 assert Counter(r['signer_id'] for r in dev)=={11:100,12:100,13:100}
 assert not {r['clip_id'] for r in tr}&{r['clip_id'] for r in dev}
 assert original['model']==MODEL and original['training']==TRAIN
 pins={};oldpins=original['pins']
 for name in ['training_CONTEXT.json','READY.json','FULL_INPUTS_INDEPENDENT.json','training_core.py','training_driver.py']:
  pins[str(BASE/name)]=sha(BASE/name)
 model=ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py'
 pins[str(model)]=sha(model);assert pins[str(model)]==oldpins[str(model)]
 protected=read(O/'word100_confusion_late_training_diagnosis_20261019/FINAL_AUDIT.json')['original_checkpoint_pins'];check(protected);pins.update(protected)
 for seed in SEEDS:
  for suffix in ['RESULT.json','history.json','last.pt']:
   p=BASE/f'training_C_{seed}_{suffix}';pins[str(p)]=sha(p)
 for r in rows:
  for view in ['masked','sensor']:
   info=r['views'][view];p=info['channel_file'];assert p is not None
   assert Path(p).resolve().is_relative_to((BASE/'data/clips').resolve())
   assert sha(p)==info['sha256']==oldpins[p];pins[p]=info['sha256']
 c={'at':now(),'rows':rows,'pins':pins,'original_readiness_reused':True,'no_new_extraction':True}
 train,development=candidate.loadrows(c)
 tensors=0;values=0
 for r in train+development:
  for z in r['arrays']['BAL31'].values():
   for x in z.values():validate_channels(x);tensors+=1;values+=x.size
 edgecases=0
 for size in [1,2,7,31,32,64]:
  for n in list(range(150))+[899,900,1023,1024,1025]:
   order=list(range(n));parts=balanced_batches(order,size)
   assert [i for p in parts for i in p]==order
   assert len(parts)==math.ceil(n/size)
   if n: assert max(map(len,parts))<=size and max(map(len,parts))-min(map(len,parts))<=1 and min(map(len,parts))>=1
   edgecases+=1
 for bad in [0,-1,1.5,True]:
  try:balanced_batches([1],bad)
  except ValueError:pass
  else:raise AssertionError('invalid max size accepted')
 schedule_checks=0
 for seed in SEEDS:
  for epoch in range(1,81):
   order=np.random.default_rng(seed+epoch).permutation(899)
   new=balanced_batches(order,32);old=[order[i:i+32] for i in range(0,899,32)]
   assert len(new)==len(old)==29 and all(len(p)==31 for p in new)
   flat=[int(i) for p in new for i in p];assert flat==[int(i) for p in old for i in p] and len(set(flat))==899
   for i in flat:
    cid=train[i]['clip_id'];h=int(hashlib.sha256(cid.encode()).hexdigest(),16)%2
    sp=int(hashlib.sha256(('sensor_view:'+cid).encode()).hexdigest(),16)%2
    independent=('sensor' if ((epoch-1)//2+sp)%2 else 'masked','x_context' if (epoch+h)%2 else 'x_whole')
    assert schedule(cid,epoch,'C')==independent;schedule_checks+=1
 # Preserve math in the known trainer; only load path, train partition, and logging differ.
 def funcs(path):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(x,ast.FunctionDef)}
 oldf=funcs(O/'word100_assignment_scale_factorial_20261019/train.py');newf=funcs(U/'train.py')
 same=['statehash','savept','initialize','batch','scores']
 for name in same:assert oldf[name]==newf[name],name
 spec=importlib.util.spec_from_file_location('original_c_driver',BASE/'training_driver.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 # Compare the tensor assembly with original C without running the model.
 compared=0
 for subset in [train[:31],train[-31:],development[:32],development[-12:]]:
  legacy=[{**r,'arrays':r['arrays']['BAL31']} for r in subset]
  for view in ['masked','sensor']:
   a=candidate.batch(subset,'BAL31',view=view);b=old.torch_batch(legacy,view=view)
   assert all(candidate.torch.equal(x,y) for x,y in zip(a[:3],b));compared+=1
  if subset[0]['role']=='train_candidate':
   for epoch in range(1,5):
    a=candidate.batch(subset,'BAL31',epoch=epoch);b=old.torch_batch(legacy,epoch=epoch,arm='C')
    assert all(candidate.torch.equal(x,y) for x,y in zip(a[:3],b));compared+=1
 initial={str(s):candidate.statehash(candidate.initialize(s).state_dict()) for s in SEEDS}
 check(pins)
 write(U/'PREP_CONTEXT.json',c,new=True)
 own={str(U/n):sha(U/n) for n in ['AUTHORIZATION.json','FROZEN_PROTOCOL.json','prepare.py','train.py','common.py','balanced.py','PREP_CONTEXT.json']}
 write(U/'READY.json',{'at':now(),'status':'PASS','training_allowed':True,'blockers':[],'pins':own,'train':899,'dev':300,'input_npz_files':2398,'input_tensors':tensors,'channel_values_checked':values,'hash_identical_to_original_C':True,'generic_partition_cases':edgecases,'invalid_partition_rejections':4,'epochs_checked':240,'individual_schedule_checks':schedule_checks,'batches_per_epoch':29,'batch_sizes':[31]*29,'same_shuffled_order':True,'same_optimizer_steps':True,'evaluation_batch_size':32,'tensor_assembly_equivalence_cases':compared,'unchanged_function_AST':same,'initial_states':initial,'semantic_readiness_reused_from_unchanged_C':str(BASE/'READY.json'),'independent_method':'Separate arithmetic and original assembly comparison by same Codex operator; no independent agent/expert','user_data_used':False,'model_forward_passes':0,'new_training_runs':0},new=True)
 event(stage='READY_FOR_THREE_PAIRED_RUNS',train=899,dev=300,batches=29,batch_size=31,tensors=tensors,schedule_checks=schedule_checks)

if __name__=='__main__':
 try:main()
 except BaseException:
  if not (U/'FIRST_PREP_ERROR.json').exists():write(U/'FIRST_PREP_ERROR.json',{'at':now(),'traceback':traceback.format_exc()},new=True)
  raise
