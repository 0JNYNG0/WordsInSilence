"""Independent tests of new coefficient plumbing only, with synthetic arrays."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import traceback
import numpy as np
import torch
import torch.nn.functional as F

OPS=Path(__file__).resolve().parent;WEB=OPS.parent.parent
sys.path[:0]=[str(WEB),str(WEB/'src'),str(WEB.parent/'sign_model_pilot_v1/src')]
import tools_public_anchor_strength_pair as subject
import tools_head_only_native_pair as h
import tools_native_data_pair as base
from pilot.model_masked_v1 import MaskedSignTCN
checks=[]
def check(name,v):
    checks.append(dict(name=name,passed=bool(v)))
    if not v:raise AssertionError(name)
def rejects(name,fn):
    try:fn()
    except (ValueError,TypeError,RuntimeError):check(name,True)
    else:check(name,False)
def config():
    return dict(optimizer_steps=3,batch=dict(total=8,source=8,nikl=0),lr=dict(encoder=0.,head=.001),weight_decay=.001,seed=331,batch_size=8,cpu_threads=2,augmentation=False)
def main():
    oldF=h.F;oldCE=F.cross_entropy;oldtrainer=h.run_head_only_training;oldhost=h._ORIGINAL_BASE_RUN;oldsample=h._sample_batch;oldslot=h.u.run_training
    plans=[json.loads((OPS/(k+'_plan.json')).read_text(encoding='utf-8')) for k in ['lambda_0p1','lambda_0p01']]
    # Both closures are created first; executing either must not borrow the other's coefficient.
    wrappers=[]
    for plan in plans:
        coefficient=plan['parent_anchor']['coefficient']
        def probe(model,source,nikl,classes,setting,rng,log=print,steps=None,c=coefficient):
            logits=torch.tensor([[.2,.1,.3]],requires_grad=True);target=torch.tensor([2]);pure=F.cross_entropy(logits,target)
            zero=h.F.cross_entropy(logits,target)
            check(str(c)+' starts at zero penalty',torch.equal(pure,zero))
            with torch.no_grad():model.head.weight.add_(.2);model.head.bias.sub_(.3)
            total=h.F.cross_entropy(logits,target);total.backward()
            expected=.5*c*(model.head.weight.numel()*.2**2+model.head.bias.numel()*.3**2)
            check(str(c)+' differentiated coefficient matches loss and both gradients',abs(float((total-pure).detach())-expected)<1e-6 and torch.allclose(model.head.weight.grad,torch.full_like(model.head.weight,.2*c),atol=1e-7) and torch.allclose(model.head.bias.grad,torch.full_like(model.head.bias,-.3*c),atol=1e-7))
            return dict(steps=2,losses=[round(float(zero.detach()),6),round(float(total.detach()),6)])
        wrappers.append(subject.make_trainer_wrapper(probe,plan))
    for plan,wrapper in zip(plans,wrappers):
        model=torch.nn.Module();model.head=torch.nn.Linear(4,3);cfg=config();result=wrapper(model,[],[],['a','b','c'],cfg,None)
        check(str(plan['parent_anchor']['coefficient'])+' same coefficient in stats/settings',result['parent_anchor']['coefficient']==cfg['parent_anchor']['coefficient']==plan['parent_anchor']['coefficient'])
        check('no global functional change',h.F is oldF and F.cross_entropy is oldCE)
    for value in (1.,0.,-1.,True,'0.1',float('nan'),float('inf'),.2):
        p=copy.deepcopy(plans[0]);p['parent_anchor']['coefficient']=value
        rejects('unsupported coefficient rejected '+repr(value),lambda:subject.make_trainer_wrapper(lambda *a:None,p))
    torch.manual_seed(129);torch.set_num_threads(2)
    parent=MaskedSignTCN(width=8,dilations=(1,),dropout=.5,n_classes=3)
    gen=np.random.default_rng(129);pool=[dict(sample_id='synthetic%d'%i,label=['a','b','c'][i],class_index=i,x=gen.normal(size=(5,280)).astype(np.float32)) for i in range(3)]
    details={}
    try:
        for plan in plans:
            coefficient=plan['parent_anchor']['coefficient'];runs=[];shared=config();before=copy.deepcopy(plan)
            def host(p,prepared,out):
                for _ in range(2):
                    model=copy.deepcopy(parent);rng=base.UniformSchedule(np.random.default_rng(111).random(48).tolist())
                    stat=h.u.run_training(model,pool,[],['a','b','c'],shared,rng,log=lambda _:None)
                    runs.append((model,stat,rng.consumed))
                return 'OK'
            h._ORIGINAL_BASE_RUN=host
            check(str(coefficient)+' actual shared-setting two-arm integration',subject.run(plan,{'settings':{'encoder_lr':.0001}},'unused')=='OK' and len(runs)==2)
            check(str(coefficient)+' own arm starts at zero and preserves same RNG',all(s['parent_anchor']['squared_distance'][0]==0 and n==48 for m,s,n in runs))
            check(str(coefficient)+' shared metadata/current coefficient valid',shared['parent_anchor']['coefficient']==coefficient and all(s['parent_anchor']['coefficient']==coefficient for m,s,n in runs))
            check(str(coefficient)+' encoder/buffers and original plan unchanged',plan==before and all(torch.equal(t,m.state_dict()[k]) for m,s,n in runs for k,t in parent.state_dict().items() if not k.startswith('head.')))
            check(str(coefficient)+' initial anchors independent',all(torch.equal(t,runs[1][0].state_dict()[k]) for k,t in runs[0][0].state_dict().items()))
            wrong=copy.deepcopy(shared);wrong['parent_anchor']['coefficient']=.01 if coefficient==.1 else .1;entered=[]
            def sentinel(*a,**kw):entered.append(True);raise RuntimeError('must reject before entering')
            rejects(str(coefficient)+' mismatched prior-arm coefficient rejected',lambda:subject.make_trainer_wrapper(sentinel,plan)(copy.deepcopy(parent),pool,[],['a','b','c'],wrong,None))
            check(str(coefficient)+' mismatch did not train',not entered)
            details[str(coefficient)]={'steps_each':3,'final_distance':runs[-1][1]['parent_anchor']['final_squared_distance']}
        check('all hooks restored',h.F is oldF and h.run_head_only_training is oldtrainer and h._sample_batch is oldsample and h.u.run_training is oldslot)
    finally:h._ORIGINAL_BASE_RUN=oldhost
    return details

if __name__=='__main__':
    error=None;details={}
    try:details=main()
    except Exception:error=traceback.format_exc()
    report=dict(status='PASS_COEFFICIENT_PLUMBING' if error is None else 'FAIL',passed=sum(c['passed'] for c in checks),total=len(checks),checks=checks,error=error,details=details,helper_sha256=hashlib.sha256((WEB/'tools_public_anchor_strength_pair.py').read_bytes()).hexdigest(),real_arrays_used=False,reused_previous_unchanged_checks='28 loss/freeze/restoration +10 shared-setting historical; affected coefficient cases above')
    with (OPS/'verification_v1.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:report[k] for k in ['status','passed','total','error']}));raise SystemExit(0 if error is None else 1)
