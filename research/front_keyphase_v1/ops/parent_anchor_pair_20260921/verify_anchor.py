"""Independent synthetic checks for the newly added anchored loss, not quality."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import traceback

OPS = Path(__file__).resolve().parent
WEB = OPS.parent.parent
sys.path[:0] = [str(WEB), str(WEB/'src'), str(WEB.parent/'sign_model_pilot_v1/src')]
import numpy as np
import torch
import torch.nn.functional as global_F
import tools_native_data_pair as base
import tools_head_only_native_pair as h
import tools_parent_anchor_pair as subject
from pilot.model_masked_v1 import MaskedSignTCN

checks = []
def check(name, value):
    checks.append({'name': name, 'passed': bool(value)})
    if not value:
        raise AssertionError(name)

def rejects(name, fn):
    try:
        fn()
    except (ValueError, RuntimeError, TypeError, AssertionError):
        check(name, True)
    else:
        check(name, False)

def setting():
    return dict(optimizer_steps=3, batch=dict(total=8,source=8,nikl=0),
                lr=dict(encoder=0.,head=.001),weight_decay=.001,seed=331,
                batch_size=8,cpu_threads=2,augmentation=False)

def main():
    plan=json.loads((OPS/'plan.json').read_text(encoding='utf-8'))
    pristine=copy.deepcopy(plan)
    torch.set_num_threads(2)
    old_F, old_sample, old_trainer = h.F, h._sample_batch, h.run_head_only_training
    old_slot, old_base, old_host = h.u.run_training, base.run, h._ORIGINAL_BASE_RUN
    ce_identity=global_F.cross_entropy
    torch.manual_seed(331)
    toy=torch.nn.Module(); toy.head=torch.nn.Linear(4,3)
    detail={}
    def loss_probe(model, source, nikl, classes, config, rng, log=print, steps=None):
        logits=torch.tensor([[.1,.2,.4]],requires_grad=True)
        y=torch.tensor([2]); pure=global_F.cross_entropy(logits,y)
        zero=h.F.cross_entropy(logits,y)
        check('anchor starts exactly at arm initial values', torch.equal(zero,pure))
        check('global torch functional CE never patched', global_F.cross_entropy is ce_identity)
        with torch.no_grad():
            model.head.weight.add_(.2); model.head.bias.sub_(.3)
        loss=h.F.cross_entropy(logits,y)
        loss.backward()
        expected=.5*(model.head.weight.numel()*.2**2+model.head.bias.numel()*.3**2)
        check('penalty equals half sum distance without normalization', abs(float((loss-pure).detach())-expected)<1e-6)
        check('penalty gradient flows to weight and bias', torch.allclose(model.head.weight.grad,torch.full_like(model.head.weight,.2),atol=1e-7) and torch.allclose(model.head.bias.grad,torch.full_like(model.head.bias,-.3),atol=1e-7))
        return dict(steps=2,losses=[round(float(zero.detach()),6),round(float(loss.detach()),6)],loss_first=round(float(zero.detach()),6),loss_last=round(float(loss.detach()),6))
    detail['analytic_probe']=subject.make_trainer_wrapper(loss_probe,plan)(toy,[],[],['a','b','c'],setting(),None)
    check('CE namespace restored after analytic probe',h.F is old_F)
    def error_probe(*args,**kwargs):
        raise RuntimeError('intentional synthetic failure')
    rejects('trainer error propagated',lambda:subject.make_trainer_wrapper(error_probe,plan)(toy,[],[],['a','b','c'],setting(),None))
    check('CE namespace restored on trainer failure',h.F is old_F)
    def sum_probe(model,*args,**kwargs):
        return h.F.cross_entropy(torch.zeros(1,3),torch.tensor([0]),reduction='sum')
    rejects('nonmean CE rejected',lambda:subject.make_trainer_wrapper(sum_probe,plan)(toy,[],[],['a','b','c'],setting(),None))
    check('CE namespace restored on reduction error',h.F is old_F)
    for key,value in [('coefficient',True),('coefficient',float('nan')),('coefficient',2.),('reduction','mean'),('parameters',['head.bias'])]:
        bad=copy.deepcopy(plan);bad['parent_anchor'][key]=value
        rejects('invalid '+key+' '+repr(value),lambda:subject.make_trainer_wrapper(error_probe,bad))
    rejects('plan-owned settings alias rejected before training',lambda:subject.make_trainer_wrapper(error_probe,plan)(toy,[],[],['a','b','c'],plan['settings'],None))
    parent=MaskedSignTCN(width=8,dilations=(1,),dropout=.5,n_classes=3)
    classes=['a','b','c']; gen=np.random.default_rng(331)
    pool=[dict(sample_id='synthetic_%d'%i,label=classes[i%3],class_index=i%3,x=gen.normal(size=(4+i%2,280)).astype(np.float32)) for i in range(9)]
    raw=gen.random(3*8*2).tolist(); prepared={'settings':{'encoder_lr':.0001}}
    before_prepared=copy.deepcopy(prepared); runs=[]
    def fake_host(p,prep,out):
        for _ in range(2):
            model=copy.deepcopy(parent); config=setting(); rng=base.UniformSchedule(raw)
            stats=h.u.run_training(model,pool,[],classes,config,rng,log=lambda _:None)
            runs.append((model,stats,config,rng.consumed))
        return 'OK'
    try:
        h._ORIGINAL_BASE_RUN=fake_host
        check('two arms use existing host and trainer',subject.run(plan,prepared,'unused')=='OK' and len(runs)==2)
        check('unchanged RNG budget and no augmentation',all(r[3]==48 and r[1]['augmentation_examples']==[] for r in runs))
        check('original sampler identity preserved',h._sample_batch is old_sample)
        check('encoder and buffers exact after new loss',all(torch.equal(v,m.state_dict()[k]) for m,_,_,_ in runs for k,v in parent.state_dict().items() if not k.startswith('head.')))
        check('same per-arm initial anchors give same learned parameters',all(torch.equal(v,runs[1][0].state_dict()[k]) for k,v in runs[0][0].state_dict().items()))
        check('head changes through anchored existing optimizer',not torch.equal(parent.head.weight,runs[0][0].head.weight))
        check('final distance is recomputed after last optimizer step',all(abs(stats['parent_anchor']['final_squared_distance']-sum(float(((m.state_dict()[k]-parent.state_dict()[k])**2).sum()) for k in ('head.weight','head.bias')))<1e-7 for m,stats,_,_ in runs))
        check('loss components agree and legacy loss is labeled total',all(all(abs(t-ce-pen)<1e-6 and abs(pen-.5*d)<1e-7 for t,ce,pen,d in zip(r[1]['parent_anchor']['total_loss'],r[1]['parent_anchor']['cross_entropy'],r[1]['parent_anchor']['penalty'],r[1]['parent_anchor']['squared_distance'])) and r[1]['parent_anchor']['legacy_losses_are_total_including_anchor'] for r in runs))
        check('configuration truthfully records added penalty',all(r[2]['parent_anchor']['coefficient']==1. and r[2]['lr']=={'encoder':0.,'head':.001} for r in runs))
        check('plan and prepared metadata unchanged',plan==pristine and prepared==before_prepared)
        check('all hooks restored after success',h.run_head_only_training is old_trainer and h.u.run_training is old_slot and base.run is old_base and h.F is old_F)
        h._ORIGINAL_BASE_RUN=error_probe
        rejects('host failure propagated',lambda:subject.run(plan,prepared,'unused'))
        check('all hooks restored after host failure',h.run_head_only_training is old_trainer and h.u.run_training is old_slot and base.run is old_base and h.F is old_F)
    finally:
        h._ORIGINAL_BASE_RUN=old_host
    detail['real_synthetic_stats']=runs[0][1]
    return detail

if __name__=='__main__':
    detail={}; error=None
    try: detail=main()
    except Exception: error=traceback.format_exc()
    report=dict(status='PASS_NEW_ANCHOR' if error is None else 'FAIL',passed=sum(c['passed'] for c in checks),total=len(checks),checks=checks,error=error,details=detail,helper_sha256=hashlib.sha256((WEB/'tools_parent_anchor_pair.py').read_bytes()).hexdigest(),real_data_arrays_used=False,real_models_loaded=False)
    with (OPS/'verification_v1.json').open('x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('status','passed','total','error')}))
    raise SystemExit(0 if error is None else 1)
