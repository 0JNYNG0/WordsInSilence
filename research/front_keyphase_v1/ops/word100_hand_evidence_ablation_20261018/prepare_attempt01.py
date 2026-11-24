"""Freeze authorization and validate runtime channel removal before training."""
from common import *
import copy,shutil,traceback
import numpy as np
import torch
import train as impl

def main():
 assert not (SRC/'TRAIN_RUNNING.json').exists()
 assert read(SRC/'FINAL_AUDIT.json')['status']=='PASS'
 assert read(DIAG/'FINAL_AUDIT.json')['status']=='PASS'
 assert not (U/'TRAIN_RUNNING.json').exists()
 write(U/'AUTHORIZATION.json',{'at':now(),'user':'그래 그럼 네 묶음을 하나씩 제외하는 12회 비교를 계획대로 진행해줘. 기존 C는 유지하고 내 촬영은 학습이나 튜닝에 쓰지 말고, 어떤 정보가 성능에 영향을 줬는지와 다음 보완 방향을 결과로 정리해줘. 중요한 문제 생기면 바로 알려줘.','scope':'Four leave-one-group-out fresh training arms x3 seeds and stored-score independent comparison. No deployment/user data/new detector.'},new=True)
 before=U/'before';before.mkdir()
 for p in (ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json'):shutil.copy2(p,before/p.name)
 c=read(SRC/'PREP_CONTEXT.json');checkpins(c['pins']);checkpins(read(SRC/'TRAIN_CONTEXT.json')['pins'])
 tr,dv=impl.loadrows(c);assert len(tr)==899 and len(dv)==300
 assert all(r['signer_id'] in range(1,10) for r in tr)
 assert all(r['signer_id'] in (11,12,13) for r in dv)
 assert not ({r['clip_id'] for r in tr}&{r['clip_id'] for r in dv})
 # Literal independent index oracle, rather than GROUPS calculation.
 oracle={'minus_position':list(range(280,286))+list(range(295,301)),
         'minus_extent':[286,287,288,301,302,303],
         'minus_intersection':[289,290,291,304,305,306],
         'minus_proximity':[292,293,294,307,308,309]}
 assert set(oracle)==set(ARMS)
 checked=0;preserved=0
 for rows in (tr,dv):
  for start in range(0,len(rows),32):
   rr=rows[start:start+32]
   # 4 epochs cover both temporal and spatial schedules for every clip.
   for epoch in range(1,5):
    for arm,drop in oracle.items():
     x,p,y,choices=impl.batch(rr,arm,epoch=epoch);x=x.numpy();p=p.numpy()
     for i,(r,(v,t)) in enumerate(zip(rr,choices)):
      original=r['arrays'][v][t];n=len(original)
      assert x.shape[2]==310 and np.isfinite(x).all()
      assert np.all(x[i,:n,drop]==0)
      keep=[k for k in range(310) if k not in drop]
      assert np.array_equal(x[i,:n,keep].T,original[:,keep])
      assert np.array_equal(x[i,:n,:280],original[:,:280])
      assert np.all(x[i,n:]==0) and np.all(p[i,:n]==1) and np.all(p[i,n:]==0)
      assert y[i].item()==r['label_index'];checked+=1;preserved+=n*280
    # DEV whole-view batching explicitly checked too (mask uses same code).
   for view in ('masked','sensor'):
    for arm,drop in oracle.items():
     x,p,y,_=impl.batch(rr,arm,view=view)
     for i,r in enumerate(rr):
      a=r['arrays'][view]['x_whole'];expected=a.copy();expected[:,drop]=0
      assert np.array_equal(x[i,:len(a)].numpy(),expected)
 # Verify no shared-array mutation by comparing to immutable on-disk inputs.
 pins={}
 for r in tr+dv:
  mf=SRC/'inputs_v2'/f"{r['clip_id']}.json";m=read(mf);pins[str(mf)]=sha(mf);pins.update(m['outputs'])
  for v in ('masked','sensor'):
   a=loadnp(m['views'][v]);assert set(a)==set(r['arrays'][v])
   for k in a:assert np.array_equal(a[k],r['arrays'][v][k],equal_nan=True)
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 impl.initial_check(tr)
 # Zeroed channels must have zero gradient in actual training forward path.
 gradients=[]
 for arm,drop in oracle.items():
  _,m=impl.initialize(20261013);m.train();x,p,y,_=impl.batch(tr[:32],arm,epoch=1)
  loss=torch.nn.functional.cross_entropy(m(x,p),y);loss.backward()
  grad=m.proj.weight.grad
  assert torch.count_nonzero(grad[:,drop]).item()==0 and torch.isfinite(grad).all()
  kept=[k for k in range(280,310) if k not in drop];assert torch.count_nonzero(grad[:,kept]).item()>0
  gradients.append({'arm':arm,'excluded_gradient_zero':True,'remaining_extra_gradient_nonzero':True})
 write(U/'INPUTS_INDEPENDENT.json',{'status':'PASS','at':now(),'records':1199,'runtime_train_clip_conditions_checked':checked,'preserved_original280_values_in_repeated_checks':preserved,'original_arrays_unmodified':True,'excluded_channel_oracle':oracle,'gradient_checks':gradients,'public_only':True,'model_training_updates':0},new=True)
 write(U/'READY.json',{'status':'PASS','at':now(),'training_allowed':True,'blockers':[],'visual_review':'Reuse prior READY/VISUAL_REVIEW/INPUTS_INDEPENDENT; no detector/ROI/coordinates/frames modified. Only designated added channels are zeroed.','input_independent_sha256':sha(U/'INPUTS_INDEPENDENT.json')},new=True)
 for p in [U/n for n in ('AUTHORIZATION.json','READY.json','INPUTS_INDEPENDENT.json','INITIAL_EQUIVALENCE.json','common.py','train.py','verify.py','prepare.py','bootstrap.py')]+[SRC/n for n in ('PREP_CONTEXT.json','TRAIN_CONTEXT.json','COMPARISON.json','FINAL_AUDIT.json','train.py','common.py','READY.json','INPUTS_INDEPENDENT.json','VISUAL_REVIEW.json')]+[DIAG/'NEXT_PLAN.json',BASE/'training_core.py',ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py']:
  pins[str(p)]=sha(p)
 for seed in impl.SEEDS:
  for arm in ('C0','C1'):
   for suffix in ('RESULT.json','best.pt','history.json'):
    p=SRC/f'{arm}_{seed}_{suffix}';pins[str(p)]=sha(p)
  for suffix in ('RESULT.json','best.pt'):
   p=BASE/f'training_C_{seed}_{suffix}';pins[str(p)]=sha(p)
 write(U/'TRAIN_CONTEXT.json',{'at':now(),'pins':pins,'arms':ARMS,'seeds':impl.SEEDS,'model':{**impl.MODEL,'in_channels':310},'training':impl.TRAIN,'groups':GROUPS,'runtime':{'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__},'protocol':read(DIAG/'NEXT_PLAN.json'),'gate':'Same prior gates versus C and C0; >=2seed sensor improvement versus C1 as proposed. All outcomes reported, no deployment.','user_mentor_heldout_used':False},new=True)
 write(U/'STATUS.json',{'stage':'READY_FOR_12_RUNS','at':now(),'training_runs':0,'production_changed':False})
 print('READY: 1199 public inputs; 4 arms; initial equivalence and gradients verified; no optimizer step.')

if __name__=='__main__':
 try:main()
 except BaseException:
  if not (U/'FIRST_PREP_ERROR.json').exists():write(U/'FIRST_PREP_ERROR.json',{'at':now(),'traceback':traceback.format_exc()},new=True)
  raise
