"""Independent tiny analytic optimizer and API checks, never project training."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
name=sys.argv[1]
spec=importlib.util.spec_from_file_location('candidate',HERE/(name+'.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(2)
args=[np.array([[1,0],[-1,1],[0,2],[2,-1]],np.float32),np.array([0,1,2,0],np.int64),
      np.array([[.2,1],[-.5,2],[1,-2]],np.float32),np.array([[1.2,-1],[.5,1],[-1,2]],np.float32),
      np.array([0,1,2],np.int64),np.array([[.1,-.2],[.3,.1],[-.1,.2]],np.float32),np.array([.1,0,-.1],np.float32)]
kwargs=dict(steps=4,batch_size=4,seed=41,learning_rate=.01,weight_decay=.02,anchor_coefficient=.4)
checks=[]
def check(name,ok,**details):checks.append(dict(name=name,pass_=bool(ok),**details))
before=[a.copy() for a in args];nr=np.random.get_state();tr=torch.random.get_rng_state().clone();threads=torch.get_num_threads()
result=module.run_pair(*args,**kwargs)
nr2=np.random.get_state()
check('inputs_unchanged',all(np.array_equal(a,b) for a,b in zip(args,before)))
check('global_rng_and_threads_unchanged',all(np.array_equal(a,b) for a,b in zip(nr,nr2)) and torch.equal(tr,torch.random.get_rng_state()) and threads==torch.get_num_threads())
# Independently reconstruct declared draws, in the exact documented sequence.
rng=np.random.default_rng(41);schedule=[np.empty((4,2),np.int64),np.empty((4,2),np.int64)]
for step in range(4):
    for ys,dst in [(args[1],schedule[0]),(args[4],schedule[1])]:
        cs=sorted(set(ys.tolist()))
        for slot in range(2):
            c=cs[int(rng.integers(len(cs)))];choices=[i for i,y in enumerate(ys) if y==c]
            dst[step,slot]=choices[int(rng.integers(len(choices)))]
check('shared_schedule_contract',np.array_equal(schedule[0],result['schedule']['base_indices']) and np.array_equal(schedule[1],result['schedule']['pair_indices']))
# Analytical softmax/gradient and scalar AdamW in float64, no autograd/optimizer reuse.
for arm,x2 in [('control',args[2]),('treatment',args[3])]:
    wp=args[5].astype(np.float64);bp=args[6].astype(np.float64);w=wp.copy();b=bp.copy()
    m=[np.zeros_like(w),np.zeros_like(b)];v=[np.zeros_like(w),np.zeros_like(b)];loss=[]
    for t in range(4):
        x=np.concatenate((args[0][schedule[0][t]],x2[schedule[1][t]])).astype(np.float64)
        y=np.concatenate((args[1][schedule[0][t]],args[4][schedule[1][t]]))
        z=x@w.T+b;z-=z.max(axis=1,keepdims=True);prob=np.exp(z);prob/=prob.sum(axis=1,keepdims=True)
        ce=-np.log(prob[np.arange(4),y]).mean();anchor=.2*(np.square(w-wp).sum()+np.square(b-bp).sum())
        loss.append([ce,anchor,ce+anchor]);prob[np.arange(4),y]-=1;prob/=4
        gradients=[prob.T@x+.4*(w-wp),prob.sum(axis=0)+.4*(b-bp)]
        for j,param in enumerate([w,b]):
            m[j]=.9*m[j]+.1*gradients[j];v[j]=.999*v[j]+.001*np.square(gradients[j])
            param*=1-.01*.02
            param-=.01*(m[j]/(1-.9**(t+1)))/(np.sqrt(v[j]/(1-.999**(t+1)))+1e-8)
    got=result[arm]
    error=max(float(np.max(np.abs(w-got['weight']))),float(np.max(np.abs(b-got['bias']))),float(np.max(np.abs(np.array(loss)-got['pre_update_losses']))))
    check(arm+'_analytic_four_updates',error<2e-6,max_abs_error=error)
    dist=float(np.square(got['weight'].astype(np.float64)-wp).sum()+np.square(got['bias'].astype(np.float64)-bp).sum())
    check(arm+'_post_final_distance',dist==got['final_anchor_distance_squared'])
    check(arm+'_output_shape_dtype',got['weight'].dtype==np.float32 and got['bias'].dtype==np.float32 and got['pre_update_losses'].shape==(4,3) and got['pre_update_losses'].dtype==np.float64)
ident=[a.copy() for a in args];ident[3]=ident[2].copy();equal=module.run_pair(*ident,**kwargs)
check('identical_inputs_produce_identical_arms',all(np.array_equal(equal['control'][k],equal['treatment'][k]) for k in ['weight','bias','pre_update_losses']))
check('real_context_substitution_has_effect',not np.array_equal(result['control']['weight'],result['treatment']['weight']))
cases=[('floating_labels',1,args[1].astype(float)),('bool_labels',1,args[1].astype(bool)),('out_of_range',4,np.array([0,1,3])),('nan_features',0,np.full((4,2),np.nan,np.float32)),('complex_features',0,args[0].astype(complex)),('bool_features',0,args[0].astype(bool)),('different_pair_shape',3,np.zeros((2,2),np.float32)),('missing_base_class',1,np.array([0,1,1,0]))]
for label,idx,val in cases:
    aa=[a.copy() for a in args];aa[idx]=val
    try:module.run_pair(*aa,**kwargs);rejected=False
    except (ValueError,TypeError):rejected=True
    check('reject_'+label,rejected)
for key,val in [('steps',True),('batch_size',3),('seed',-1),('learning_rate',0),('anchor_coefficient',-1),('weight_decay',float('inf'))]:
    kw=dict(kwargs);kw[key]=val
    try:module.run_pair(*args,**kw);rejected=False
    except (ValueError,TypeError):rejected=True
    check('reject_'+key,rejected)
out=dict(status='PASS' if all(c['pass_'] for c in checks) else 'FAIL',module=name,module_sha256=hashlib.sha256((HERE/(name+'.py')).read_bytes()).hexdigest(),checks=checks,failed=[c['name'] for c in checks if not c['pass_']],project_training_runs=0)
with (HERE/(name+'_AUDIT.json')).open('x',encoding='utf-8',newline='\n') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
sys.exit(0 if out['status']=='PASS' else 1)
