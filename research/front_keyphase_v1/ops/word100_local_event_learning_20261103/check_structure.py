"""Random-weight synthetic checks; no real-video inference or training."""
from pathlib import Path
import hashlib,json,torch
from local_events import LocalEventNet,bag_logit,RECEPTIVE_FIELD
U=Path(__file__).resolve().parent
torch.manual_seed(42);torch.set_num_threads(4)
model=LocalEventNet().eval()
checks=[]
def check(name,test):
    assert bool(test),name;checks.append(name)
def reject(name,fn):
    try:fn()
    except ValueError:checks.append(name);return
    raise AssertionError(name)
def sample(n):
    x=torch.randn(1,n,915);x[:,:,73:146]=(torch.rand(1,n,73)>.1).float();x[:,:,914]=1
    return x
with torch.inference_mode():
    x=sample(50);a=model(x,torch.tensor([50]))
    prefix=sample(20);suffix=sample(20)
    y=torch.cat([prefix,x,suffix],1);b=model(y,torch.tensor([90]))
    radius=(RECEPTIVE_FIELD-1)//2
    check('same_interior_scores_after_prefix_suffix',torch.allclose(a['logits'][:,radius:-radius],b['logits'][:,20+radius:70-radius],atol=1e-6,rtol=1e-6))
    check('same_event_bag_readout_after_time_shift',torch.allclose(bag_logit(a,10,39,0),bag_logit(b,30,59,0),atol=1e-6,rtol=1e-6))
    c=model(torch.cat([x,torch.zeros(1,40,915)],1),torch.tensor([50]))
    check('padding_does_not_change_real_rows',torch.allclose(a['logits'],c['logits'][:,:50],atol=1e-6,rtol=1e-6))
    check('padding_not_valid',not c['time_valid'][:,50:].any())
    check('all_native_outputs_kept',a['logits'].shape==(1,50,5))
    z=x.clone();z[:,20:25,914]=0;z[:,20:25,146:914]=999
    missing=model(z,torch.tensor([50]));z[:,20:25,146:914]=-999
    missing2=model(z,torch.tensor([50]))
    check('unavailable_RGB_values_cannot_change_result',torch.equal(missing['logits'],missing2['logits']))
    check('missing_rows_still_exist',missing['logits'].shape==(1,50,5))
    check('geometry_survives_RGB_only_gap',missing['motion_anchor'][0,20:25].all())
    blank=z.clone();blank[:,20:25,73:146]=0
    unavailable=model(blank,torch.tensor([50]))
    reject('all_evidence_missing_bag_rejected',lambda:bag_logit(unavailable,20,24,0))
    check('unknown_rows_keep_axis_and_are_not_anchors',unavailable['time_valid'][0,20:25].all() and not unavailable['motion_anchor'][0,20:25].any())
    changed=blank.clone();changed[:,:20,:73]+=7
    separated=model(changed,torch.tensor([50]))
    check('complete_gap_prevents_cross_fragment_leakage',torch.equal(unavailable['logits'][:,25:],separated['logits'][:,25:]))
    single=sample(1);one=model(single,torch.tensor([1]))
    reject('single_observation_is_not_motion',lambda:bag_logit(one,0,0,0))
    reject('padding_bag_rejected',lambda:bag_logit(c,49,51,0))
    reject('float_coordinates_rejected',lambda:bag_logit(a,1.5,5,0))
    reject('bad_target_rejected',lambda:bag_logit(a,2,5,5))
    reject('zero_length_rejected',lambda:model(x,torch.tensor([0])))
    reject('float_length_rejected',lambda:model(x,torch.tensor([50.])))
    q=x.clone();q[0,0,0]=float('nan')
    reject('unmasked_nan_rejected',lambda:model(q,torch.tensor([50])))
result=dict(status='PASS_SYNTHETIC_FINITE_CONTEXT_CANDIDATE_ONLY',checks=len(checks),names=checks,
            parameters=sum(p.numel() for p in model.parameters()),receptive_field=RECEPTIVE_FIELD,
            real_video_inferences=0,checkpoint_loads=0,optimizer_steps=0,
            semantics_or_accuracy_proven=False,whole_goal_complete=False,
            code_sha256=hashlib.sha256((U/'local_events.py').read_bytes()).hexdigest())
with (U/'STRUCTURAL_CHECKS_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(result)
