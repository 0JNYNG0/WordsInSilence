"""Targeted implementation checks for the new loss, schedules and numerical path."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

H=Path(__file__).resolve().parent;OLD=H.parent/'context_training_pair_20260921'
sys.path[:0]=[str(H),str(OLD)]
import generated_v2 as old
import generated_candidate as new
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(2)
args=[np.array([[1,0],[-1,1],[0,2],[2,-1]],np.float32),np.array([0,1,2,0],np.int64),
      np.array([[1.2,-1],[.5,1],[-1,2]],np.float32),np.array([0,1,2],np.int64),
      np.array([[.1,-.2],[.3,.1],[-.1,.2]],np.float32),np.array([.1,0,-.1],np.float32)]
settings=dict(steps=4,batch_size=4,learning_rate=.01,weight_decay=.02,anchor_coefficient=.4)
# The synthetic zero-new-loss control proves unchanged optimizer integration, not a repeated project run.
reference=old.run_pair(args[0],args[1],args[2],args[2],args[3],args[4],args[5],seed=41,**settings)
base_ids=reference['schedule']['base_indices'];pair_ids=reference['schedule']['pair_indices']
f=np.column_stack((args[0].astype(float),np.ones(4)));weights=1/(3*np.bincount(args[1])[args[1]])
gram=f.T@(weights[:,None]*f)
full=args+[base_ids,pair_ids,gram];copies=[v.copy() for v in full]
checks=[]
def check(name,ok,**details):
    checks.append(dict(name=name,passed=bool(ok),**details))
    if not ok:raise AssertionError(name)
nr=np.random.get_state();tr=torch.random.get_rng_state().clone();threads=torch.get_num_threads()
zero=new.train_head(*full,output_coefficient=0.,**settings)
check('zero_new_loss_matches_verified_old_path',np.array_equal(zero['weight'],reference['treatment']['weight']) and np.array_equal(zero['bias'],reference['treatment']['bias']) and np.array_equal(zero['pre_update_losses'][:,[0,1,3]],reference['treatment']['pre_update_losses']))
got=new.train_head(*full,output_coefficient=1.,**settings)
check('inputs_and_global_state_unchanged',all(np.array_equal(v,c) for v,c in zip(full,copies)) and all(np.array_equal(v,c) for v,c in zip(nr,np.random.get_state())) and torch.equal(tr,torch.random.get_rng_state()) and threads==torch.get_num_threads())
check('provided_schedule_exact',np.array_equal(got['schedule']['base_indices'],base_ids) and np.array_equal(got['schedule']['pair_indices'],pair_ids))
# Independent dense-row gradient, numpy softmax and AdamW; do not reuse Gram gradient implementation.
wp=args[4].astype(float);bp=args[5].astype(float);w=wp.copy();b=bp.copy()
mw=np.zeros_like(w);mb=np.zeros_like(b);vw=np.zeros_like(w);vb=np.zeros_like(b);losses=[]
for t in range(4):
    bx=np.concatenate((args[0][base_ids[t]],args[2][pair_ids[t]])).astype(float);by=np.concatenate((args[1][base_ids[t]],args[3][pair_ids[t]]))
    logits=bx@w.T+b;shift=logits-logits.max(axis=1,keepdims=True);prob=np.exp(shift);prob/=prob.sum(axis=1,keepdims=True)
    ce=-np.log(prob[np.arange(4),by]).mean();anchor=.2*(np.square(w-wp).sum()+np.square(b-bp).sum())
    delta_scores=args[0].astype(float)@(w-wp).T+(b-bp);delta_scores-=delta_scores.mean(axis=1,keepdims=True)
    outloss=.5/3*float(np.sum(weights[:,None]*delta_scores**2));losses.append([ce,anchor,outloss,ce+anchor+outloss])
    scoregrad=weights[:,None]*delta_scores/3
    gradw_extra=scoregrad.T@args[0].astype(float);gradb_extra=scoregrad.sum(axis=0)
    prob[np.arange(4),by]-=1;prob/=4
    gw=prob.T@bx+.4*(w-wp)+gradw_extra;gb=prob.sum(axis=0)+.4*(b-bp)+gradb_extra
    mw=.9*mw+.1*gw;mb=.9*mb+.1*gb;vw=.999*vw+.001*gw**2;vb=.999*vb+.001*gb**2
    w*=1-.01*.02;b*=1-.01*.02
    w-=.01*(mw/(1-.9**(t+1)))/(np.sqrt(vw/(1-.999**(t+1)))+1e-8)
    b-=.01*(mb/(1-.9**(t+1)))/(np.sqrt(vb/(1-.999**(t+1)))+1e-8)
error=max(float(np.max(np.abs(w-got['weight']))),float(np.max(np.abs(b-got['bias']))),float(np.max(np.abs(np.array(losses)-got['pre_update_losses']))))
check('four_updates_match_independent_direct_row_loss_gradient_adamw',error<2e-6,max_abs_error=error)
check('new_penalty_has_gradient_effect',not np.array_equal(got['weight'],zero['weight']) and got['pre_update_losses'][1:,2].min()>0)
d=np.column_stack((got['weight'].astype(float)-wp,got['bias'].astype(float)-bp));center=d-d.mean(axis=0,keepdims=True)
check('final_distance_and_output_loss_after_last_update',abs(float(np.square(d).sum())-got['final_anchor_distance_squared'])<1e-12 and abs(.5/3*float(np.sum(weights[:,None]*(f@center.T)**2))-got['final_output_loss_raw'])<1e-12)
check('loss_columns_and_dtypes',got['weight'].dtype==np.float32 and got['bias'].dtype==np.float32 and got['pre_update_losses'].dtype==np.float64 and got['pre_update_losses'].shape==(4,4) and got['pre_update_losses'][0,1]==0 and got['pre_update_losses'][0,2]==0)
bad=[('bool_schedule',6,base_ids.astype(bool)),('float_schedule',7,pair_ids.astype(float)),('overflow_schedule',6,np.full((4,2),2**63,np.uint64)),('negative_schedule',6,np.full((4,2),-1,np.int64)),('out_of_range_schedule',7,np.full((4,2),3,np.int64)),('schedule_shape',6,np.zeros((4,1),np.int64)),('gram_shape',8,np.eye(2)),('asymmetric_gram',8,gram+np.array([[0,.1,0],[0,0,0],[0,0,0]])),('negative_gram',8,-np.eye(3)),('nonfinite_gram',8,np.full((3,3),np.nan)),('boolean_gram',8,np.eye(3,dtype=bool)),('complex_gram',8,gram.astype(complex))]
for label,index,value in bad:
    aa=[v.copy() for v in full];aa[index]=value
    try:new.train_head(*aa,output_coefficient=1.,**settings);rejected=False
    except (ValueError,TypeError,OverflowError):rejected=True
    check('reject_'+label,rejected)
for val in [-1,True,float('inf')]:
    try:new.train_head(*full,output_coefficient=val,**settings);rejected=False
    except (ValueError,TypeError):rejected=True
    check('reject_output_coefficient_'+str(val),rejected)
audit=dict(status='PASS',candidate_module_sha256=sha(H/'generated_candidate.py'),reused_validation_module_sha256=sha(OLD/'generated_v2.py'),
           old_checks_reused=str(OLD/'generated_v2_AUDIT.json'),objective_identity_checks_reused=str(H.parent/'public_prediction_retention_20260922/INDEPENDENT_AUDIT.json'),
           checks=checks,synthetic_only=True,project_training_runs=0,max_new_numeric_error=error)
with (H/'IMPLEMENTATION_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(audit,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in audit.items() if k!='checks'}))
