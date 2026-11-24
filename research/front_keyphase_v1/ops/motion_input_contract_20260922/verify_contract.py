"""Independent analytic feature and backward-path checks; synthetic data only."""
from pathlib import Path
import copy,json,hashlib,sys,math
import numpy as np
import torch
H=Path(__file__).resolve().parent;L=H.parents[1]
sys.path[:0]=[str(H),str(L.parent/'sign_model_pilot_v1/src')]
from pilot.model_masked_v1 import MaskedSignTCN
from generated_v1 import motion_features,fit_velocity_scale,MotionAugmentedTCN
checks={}
def ck(n,v):
    checks[n]=bool(v)
    if not v:raise AssertionError(n)
def independent(x,pad):
    b,t,c=x.shape;p=(c-1)//3;out=np.zeros((b,t,p*3),dtype=np.float32)
    for bi in range(b):
        for ti in range(1,t):
            for j in range(p):
                valid=pad[bi,ti]==pad[bi,ti-1]==1 and x[bi,ti,2*p+j]==x[bi,ti-1,2*p+j]==1 and x[bi,ti,-1]>0
                if valid:
                    for axis in range(2):out[bi,ti,2*j+axis]=(float(x[bi,ti,2*j+axis])-float(x[bi,ti-1,2*j+axis]))/float(x[bi,ti,-1])
                    out[bi,ti,2*p+j]=1
    return out
def line(speed,valid=True):
    x=np.array([[0,0,int(valid),0],[speed,0,int(valid),1],[2*speed,0,int(valid),1]],dtype=np.float32)
    return x
def main():
    torch.set_num_threads(2);torch.manual_seed(940);rng=np.random.default_rng(940)
    x=np.array([[[0,0,10,10,1,1,0],[2,4,900,800,1,0,.5],[8,10,30,40,1,1,1],[12,14,80,70,1,1,0],[14,16,90,80,1,1,2]]],dtype=np.float32)
    pad=np.ones((1,5),dtype=np.float32);orig=x.copy()
    actual=motion_features(torch.from_numpy(x),torch.from_numpy(pad)).numpy()
    ck('analytic_masked_rates',np.array_equal(actual,independent(x,pad)))
    ck('first_row_invalid',not actual[:,0].any())
    ck('missing_not_zero_motion',np.array_equal(actual[0,1:3,[2,3,5]],np.zeros((3,2),dtype=np.float32)))
    ck('nonpositive_dt_invalid',not actual[0,3].any())
    xnegative=x.copy();xnegative[0,3,-1]=-1
    ck('negative_dt_invalid',not motion_features(torch.from_numpy(xnegative),torch.from_numpy(pad))[0,3].any())
    gapped=pad.copy();gapped[0,2]=0
    ck('no_bridge_across_padding',np.array_equal(motion_features(torch.from_numpy(x),torch.from_numpy(gapped)).numpy(),independent(x,gapped)))
    strange=x.copy();strange[0,1,2:4]=[-1e20,1e20]
    ck('absent_coordinates_ignored',np.array_equal(motion_features(torch.from_numpy(strange),torch.from_numpy(pad)).numpy(),actual))
    time=np.array([0,.25,1.25,3.25,4.25],dtype=np.float32);vary=np.column_stack([2*time,4*time,np.ones(5),np.r_[0,np.diff(time)]]).astype(np.float32)
    got=motion_features(torch.from_numpy(vary[None]),torch.ones(1,5)).numpy()[0]
    ck('variable_dt_constant_rate',np.allclose(got[1:,:2],[2,4],atol=0,rtol=0))
    stationary=line(0);stat=motion_features(torch.from_numpy(stationary[None]),torch.ones(1,3)).numpy()[0]
    ck('observed_stationary_distinct_from_absent',not stat[:,:2].any() and np.array_equal(stat[:,2],[0,1,1]))
    scales=fit_velocity_scale([line(2),line(4),line(10),line(30,False)],np.array([0,0,1,2],dtype=np.int64))
    ck('class_balanced_rms',np.allclose(scales['scale'],[math.sqrt(55),1],rtol=1e-7,atol=1e-7))
    ck('unobserved_clip_not_in_denominator',np.allclose(scales['valid_clip_weight'],[2/3,2/3],atol=1e-12,rtol=0))
    ck('zero_rate_channel_scale_one',scales['zero_or_unobserved_channels']==[1])
    model=MaskedSignTCN(in_channels=7,width=4,dilations=(1,2),dropout=.3,n_classes=3).eval()
    snapshot=copy.deepcopy(model.state_dict());state=torch.random.get_rng_state().clone()
    wrapper=MotionAugmentedTCN(model,np.ones(4,dtype=np.float32),enabled=True)
    ck('constructor_rng_unchanged',torch.equal(state,torch.random.get_rng_state()))
    tx=torch.from_numpy(x);pm=torch.from_numpy(pad)
    with torch.no_grad():before=model(tx,pm);after=wrapper(tx,pm)
    ck('zero_residual_exact_parent',torch.equal(before,after))
    padded=torch.cat([tx,torch.zeros(1,3,7)],dim=1);padded_m=torch.cat([pm,torch.zeros(1,3)],dim=1)
    with torch.no_grad():ck('padded_parent_preserved',torch.equal(model(padded,padded_m),wrapper(padded,padded_m)))
    ck('source_model_unmodified',all(torch.equal(v,snapshot[n]) for n,v in model.state_dict().items()))
    ck('only_adapter_and_head_trainable',{n for n,p in wrapper.named_parameters() if p.requires_grad}=={'velocity_weight','base.head.weight','base.head.bias'})
    buffers={n:v.clone() for n,v in wrapper.named_buffers()};wrapper.train(True)
    loss=wrapper(tx,pm).square().sum();loss.backward()
    ck('gradient_reaches_adapter_through_frozen_blocks',wrapper.velocity_weight.grad is not None and torch.isfinite(wrapper.velocity_weight.grad).all() and wrapper.velocity_weight.grad.abs().sum()>0)
    ck('base_blocks_have_no_grad',all(p.grad is None for n,p in wrapper.base.named_parameters() if not n.startswith('head.')))
    ck('all_base_modules_remain_eval',all(not q.training for q in wrapper.base.modules()))
    ck('buffers_fixed',all(torch.equal(v,buffers[n]) for n,v in wrapper.named_buffers()))
    with torch.no_grad():wrapper.velocity_weight.fill_(.01)
    wrapper.enabled=False
    with torch.no_grad():ck('disabled_branch_exact_parent',torch.equal(model(tx,pm),wrapper(tx,pm)))
    ck('input_unmodified',np.array_equal(x,orig))
    random=rng.normal(size=(3,8,7)).astype(np.float32);random[:,:,4:6]=rng.integers(0,2,size=(3,8,2));random[:,:,-1]=rng.uniform(.01,.3,size=(3,8));rp=np.ones((3,8),dtype=np.float32);rp[1,5:]=0
    ck('random_scalar_oracle',np.allclose(motion_features(torch.from_numpy(random),torch.from_numpy(rp)).numpy(),independent(random,rp),atol=3e-5,rtol=2e-7))
    for name,bad in [('nonbinary_observation',x.copy()),('nonfinite',x.copy())]:
        if name=='nonbinary_observation':bad[0,1,4]=.2
        else:bad[0,1,0]=np.nan
        try:motion_features(torch.from_numpy(bad),pm)
        except (TypeError,ValueError,RuntimeError):ck('reject_'+name,True)
        else:ck('reject_'+name,False)
    try:MotionAugmentedTCN(model,np.zeros(4,dtype=np.float32))
    except (TypeError,ValueError,RuntimeError):ck('reject_nonpositive_scale',True)
    else:ck('reject_nonpositive_scale',False)
    record={'status':'PASS','checks':checks,'count':len(checks),'module_sha256':hashlib.sha256((H/'generated_v1.py').read_bytes()).hexdigest(),'real_training':0,'real_observation_arrays':0}
    with (H/'IMPLEMENTATION_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    print(json.dumps({'status':'PASS','checks':len(checks)}),flush=True)
if __name__=='__main__':main()
