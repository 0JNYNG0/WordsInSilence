"""Check only new position branch and new full-graph learner; prior motion tests reused."""
import copy,math
from pair_io import *
from pilot.model_masked_v1 import MaskedSignTCN
import torch.nn.functional as F
checks={}
def ck(n,v):
    checks[n]=bool(v)
    if not v:raise AssertionError(n)
def batch(xs):
    x=torch.nn.utils.rnn.pad_sequence([torch.from_numpy(q) for q in xs],batch_first=True)
    mask=(torch.arange(x.shape[1])[None,:]<torch.tensor([len(q) for q in xs])[:,None]).float()
    return x,mask
def oracle_features(x,mask,mode):
    a=x.numpy();m=mask.numpy();p=(a.shape[2]-1)//3;out=np.zeros((*a.shape[:2],3*p),dtype=np.float32)
    for b in range(len(a)):
        for t in range(a.shape[1]):
            for j in range(p):
                valid=m[b,t]==1 and a[b,t,2*p+j]==1
                if mode=='motion':valid=valid and t>0 and m[b,t-1]==1 and a[b,t-1,2*p+j]==1 and a[b,t,-1]>0
                if valid:
                    for k in range(2):out[b,t,2*j+k]=a[b,t,2*j+k] if mode=='position' else (float(a[b,t,2*j+k])-float(a[b,t-1,2*j+k]))/float(a[b,t,-1])
                    out[b,t,2*p+j]=1
    return torch.from_numpy(out)
def ref_forward(model,x,mask,mode):
    m=mask[:,None,:];phi=oracle_features(x,mask,mode);p=model.num_pairs
    phi=torch.cat([phi[:,:,:2*p]/model.velocity_scale,phi[:,:,2*p:]],dim=2)
    h=model.base.proj(x.transpose(1,2)*m)*m+F.conv1d(phi.transpose(1,2),model.velocity_weight)*m
    for block in model.base.blocks:h=block(h,m)
    pool=(h*m).sum(2)/m.sum(2).clamp_min(1)
    return model.base.head(pool)
def line(a,valid=True):return np.array([[0,0,valid,0],[a,0,valid,1],[2*a,0,valid,1]],dtype=np.float32)
def main():
    impl=implementation();torch.set_num_threads(2);torch.manual_seed(931);rng=np.random.default_rng(931)
    fit=impl.fit_position_scale([line(2),line(4),line(10),line(30,False)],np.array([0,0,1,2],dtype=np.int64))
    ck('position_class_balanced_rms',np.allclose(fit['scale'],[math.sqrt(55*5/3),1],rtol=1e-7,atol=1e-7))
    ck('missing_not_in_scale_denominator',np.allclose(fit['valid_clip_weight'],2/3,atol=1e-12,rtol=0))
    ck('zero_position_scale_fallback',fit['zero_or_unobserved_channels']==[1])
    seq=[]
    for n in [4,7,11,9,12,5]:
        x=rng.normal(size=(n,7)).astype(np.float32);x[:,4:6]=1;x[:,-1]=.05;x[0,-1]=0
        if n>5:x[2:4,5]=0
        seq.append(x)
    base=seq[:4];ctx=seq[4:];by=np.array([0,1,2,0],np.int64);cy=np.array([1,2],np.int64)
    pos=impl.fit_position_scale(base,by)['scale'];vel=motion.fit_velocity_scale(base,by)['scale']
    parent_model=MaskedSignTCN(in_channels=7,width=4,dilations=(1,2),dropout=.3,n_classes=3).eval()
    snapshot=copy.deepcopy(parent_model.state_dict());rngbefore=torch.random.get_rng_state().clone();b,mask=batch(base)
    ck('position_features_scalar_oracle',torch.equal(impl.position_features(b,mask),oracle_features(b,mask,'position')))
    copies=[x.copy() for x in seq]
    models={}
    for arm,mode,scale in [('control','position',pos),('treatment','motion',vel)]:
        w=impl.PairedInputTCN(parent_model,scale,mode);models[arm]=w
        with torch.no_grad():ck(arm+'_initial_parent_exact',torch.equal(w(b,mask),parent_model(b,mask)))
        with torch.no_grad():w.velocity_weight.fill_(.002)
        ck(arm+'_nonzero_branch_oracle',torch.allclose(w(b,mask),ref_forward(w,b,mask,mode),atol=3e-6,rtol=1e-6))
        with torch.no_grad():w.velocity_weight.zero_()
    ck('matched_trainable_count',sum(p.numel() for p in models['control'].parameters() if p.requires_grad)==sum(p.numel() for p in models['treatment'].parameters() if p.requires_grad))
    bi=np.array([[0,1],[2,3],[1,0]],np.int64);ci=np.array([[0,1],[1,0],[0,0]],np.int64)
    with torch.no_grad():teacher=parent_model(b,mask).numpy().astype(float)+np.array([.3,-.2,.1])
    kwargs=dict(learning_rate=.002,weight_decay=.001,anchor_coefficient=.03,output_coefficient=.7)
    got=impl.run_pair(parent_model,base,by,ctx,cy,teacher,bi,ci,pos,vel,**kwargs)
    worst=0.;lossworst=0.
    for arm,mode in [('control','position'),('treatment','motion')]:
        ref=copy.deepcopy(models[arm]);initial=copy.deepcopy(ref.state_dict());params={n:p for n,p in ref.named_parameters() if p.requires_grad}
        mm={n:np.zeros(tuple(p.shape),dtype=float) for n,p in params.items()};vv=copy.deepcopy(mm);losses=[]
        for step in range(3):
            ref.zero_grad(set_to_none=True);raw=[base[i] for i in bi[step]]+[ctx[i] for i in ci[step]];x,m=batch(raw)
            logits=ref_forward(ref,x,m,mode);labels=torch.from_numpy(np.r_[by[bi[step]],cy[ci[step]]])
            ce=(torch.logsumexp(logits,1)-logits[torch.arange(4),labels]).mean()
            anchor=.5*.03*sum(((v-initial[n])**2).sum() for n,v in params.items())
            delta=logits[:2].double()-torch.from_numpy(teacher[bi[step]]);delta-=delta.sum(1,keepdim=True)/3
            output=.5*.7*(delta**2).sum()/6;total=ce.double()+anchor.double()+output;total.backward()
            losses.append([ce.detach().item(),anchor.detach().item(),output.detach().item(),total.detach().item()])
            with torch.no_grad():
                for n,v in params.items():
                    grad=v.grad.numpy().astype(float);mm[n]=.9*mm[n]+.1*grad;vv[n]=.999*vv[n]+.001*grad**2
                    update=(mm[n]/(1-.9**(step+1)))/(np.sqrt(vv[n]/(1-.999**(step+1)))+1e-8)
                    new=v.numpy().astype(float)*(1-.002*.001)-.002*update;v.copy_(torch.from_numpy(new.astype(np.float32)))
        for n,v in ref.state_dict().items():
            if v.is_floating_point():worst=max(worst,float((v-got[arm]['state_dict'][n]).abs().max()))
        lossworst=max(lossworst,float(np.max(np.abs(np.array(losses)-got[arm]['losses']))))
        ck(arm+'_scalar_features_numpy_adam',all(torch.allclose(v,got[arm]['state_dict'][n],atol=5e-6,rtol=2e-5) for n,v in ref.state_dict().items()))
        ck(arm+'_independent_losses',np.allclose(losses,got[arm]['losses'],atol=5e-6,rtol=2e-5))
        ck(arm+'_exact_frozen_and_buffers',all(torch.equal(v,got[arm]['state_dict'][n]) for n,v in initial.items() if n not in params))
        ck(arm+'_only_three_trainable_tensors',set(got[arm]['trainable_names'])=={'velocity_weight','base.head.weight','base.head.bias'})
        ck(arm+'_adapter_actually_changed',not torch.equal(initial['velocity_weight'],got[arm]['state_dict']['velocity_weight']))
    ck('input_arrays_unchanged',all(np.array_equal(a,b) for a,b in zip(seq,copies)))
    ck('caller_parent_unchanged',all(torch.equal(v,snapshot[n]) for n,v in parent_model.state_dict().items()))
    ck('rng_unchanged',torch.equal(torch_rng:=rngbefore,torch.random.get_rng_state()))
    ck('schedule_unchanged',np.array_equal(got['schedule']['base_indices'],bi) and np.array_equal(got['schedule']['context_indices'],ci))
    args=[parent_model,base,by,ctx,cy,teacher,bi,ci,pos,vel]
    for j,(idx,value) in enumerate([(2,by.astype(float)),(5,np.full_like(teacher,np.nan)),(6,np.full_like(bi,99)),(7,ci[:1]),(8,np.zeros_like(pos))]):
        bad=list(args);bad[idx]=value
        try:impl.run_pair(*bad,**kwargs)
        except (ValueError,TypeError,RuntimeError):ck('reject_bad_input_'+str(j),True)
        else:ck('reject_bad_input_'+str(j),False)
    result={'status':'PASS','checks':checks,'count':len(checks),'module_sha256':sha(H/'generated_v1.py'),'base_motion_module_sha256':sha(MOTION/'generated_v1.py'),
      'max_state_difference':worst,'max_loss_difference':lossworst,'new_real_training':0,'prior_motion_checks_reused':27}
    write(H/'IMPLEMENTATION_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
if __name__=='__main__':main()
