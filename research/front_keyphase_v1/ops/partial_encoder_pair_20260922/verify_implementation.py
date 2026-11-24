"""Independent synthetic checks: full temporal graph vs cached-prefix training."""
import copy,traceback
from local_io import *
import generated_v1 as g
from pilot.model_masked_v1 import MaskedSignTCN
from pilot.data import collate

checks={}
def check(name,cond):
    checks[name]=bool(cond)
    if not cond:raise AssertionError(name)
def full_batch(model,xs):
    x=torch.nn.utils.rnn.pad_sequence([torch.from_numpy(a) for a in xs],batch_first=True)
    m=(torch.arange(x.shape[1])[None,:]<torch.tensor([len(a) for a in xs])[:,None]).float()
    return model(x,m)

def main():
    torch.set_num_threads(2);torch.manual_seed(147);rng=np.random.default_rng(147)
    model=MaskedSignTCN(in_channels=5,width=4,dilations=(1,2),dropout=.3,n_classes=3).eval()
    original=copy.deepcopy(model.state_dict());torch_rng=torch.random.get_rng_state().clone()
    bx=[rng.normal(size=(n,5)).astype(np.float32) for n in [3,7,9,12]]
    cx=[rng.normal(size=(n,5)).astype(np.float32) for n in [10,5]]
    copies=[x.copy() for x in bx+cx]
    bp=g.extract_prefix(model,bx);cp=g.extract_prefix(model,cx)
    with torch.no_grad():expected=full_batch(model,bx).numpy()
    cached=g.predict_prefix(model,bp,batch_size=4)['logits']
    check('prefix_equals_full_graph',np.allclose(cached,expected,atol=2e-6,rtol=1e-6))
    one=g.predict_prefix(model,bp,batch_size=1)['logits']
    check('padding_does_not_change_scores',np.allclose(cached,one,atol=2e-6,rtol=1e-6))
    by=np.array([0,1,2,0],dtype=np.int64);cy=np.array([1,2],dtype=np.int64)
    teacher=expected.astype(float)+np.array([.7,-.3,.2])[None,:]
    bi=np.array([[0,1],[2,3],[1,0]],dtype=np.int64);ci=np.array([[0,1],[1,0],[0,0]],dtype=np.int64)
    settings=dict(learning_rate=.002,weight_decay=.001,anchor_coefficient=.03,output_coefficient=.7)
    result=g.run_pair(model,bp,by,cp,cy,teacher,bi,ci,**settings)
    worst=0.;lossworst=0.
    for arm in ['control','treatment']:
        # Reference uses full raw temporal graph, explicit logsumexp CE, and NumPy AdamW.
        ref=copy.deepcopy(model).eval();names=[]
        for n,p in ref.named_parameters():
            yes=n.startswith('head.') or (arm=='treatment' and n.startswith('blocks.1.'))
            p.requires_grad_(yes)
            if yes:names.append(n)
        params=dict(ref.named_parameters());mom={n:np.zeros(tuple(params[n].shape),dtype=np.float64) for n in names};vel=copy.deepcopy(mom)
        ls=[]
        for step in range(3):
            ref.zero_grad(set_to_none=True)
            raw=[bx[i] for i in bi[step]]+[cx[i] for i in ci[step]]
            logits=full_batch(ref,raw);ys=torch.from_numpy(np.r_[by[bi[step]],cy[ci[step]]])
            ce=(torch.logsumexp(logits,1)-logits[torch.arange(4),ys]).mean()
            anchor=.5*.03*sum(((p-original[n])**2).sum() for n,p in ref.named_parameters())
            delta=logits[:2].double()-torch.from_numpy(teacher[bi[step]])
            centered=delta-delta.sum(1,keepdim=True)/3
            output=.5*.7*(centered**2).sum()/6
            total=ce.double()+anchor.double()+output;total.backward()
            ls.append([ce.detach().item(),anchor.detach().item(),output.detach().item(),total.detach().item()])
            with torch.no_grad():
                for n in names:
                    grad=params[n].grad.numpy().astype(float);mom[n]=.9*mom[n]+.1*grad;vel[n]=.999*vel[n]+.001*grad**2
                    update=(mom[n]/(1-.9**(step+1)))/(np.sqrt(vel[n]/(1-.999**(step+1)))+1e-8)
                    new=params[n].numpy().astype(float)*(1-.002*.001)-.002*update
                    params[n].copy_(torch.from_numpy(new.astype(np.float32)))
        for name,v in ref.state_dict().items():
            difference=float((v-result[arm]['state_dict'][name]).abs().max()) if v.is_floating_point() else 0.
            worst=max(worst,difference)
        lossworst=max(lossworst,float(np.max(np.abs(np.array(ls)-result[arm]['losses']))))
        check(arm+'_full_graph_numpy_adam_match',all(torch.allclose(v,result[arm]['state_dict'][n],atol=4e-6,rtol=2e-5) for n,v in ref.state_dict().items()))
        check(arm+'_loss_matches_independent_formula',np.allclose(ls,result[arm]['losses'],atol=4e-6,rtol=2e-5))
        check(arm+'_trainable_names',result[arm]['trainable_names']==names)
        check(arm+'_buffers_unchanged',all(torch.equal(v,result[arm]['state_dict'][n]) for n,v in model.named_buffers()))
        check(arm+'_frozen_weights_unchanged',all(torch.equal(v,result[arm]['state_dict'][n]) for n,v in model.named_parameters() if n not in names))
    check('last_block_actually_learns',not torch.equal(original['blocks.1.conv1.weight'],result['treatment']['state_dict']['blocks.1.conv1.weight']))
    check('control_last_block_fixed',torch.equal(original['blocks.1.conv1.weight'],result['control']['state_dict']['blocks.1.conv1.weight']))
    check('source_model_unchanged',all(torch.equal(v,original[n]) for n,v in model.state_dict().items()))
    check('inputs_unmodified',all(np.array_equal(a,b) for a,b in zip(bx+cx,copies)))
    check('torch_rng_unchanged',torch.equal(torch_rng,torch.random.get_rng_state()))
    check('schedule_return_unchanged',np.array_equal(result['schedule']['base_indices'],bi) and np.array_equal(result['schedule']['context_indices'],ci))
    shifted=g.run_pair(model,bp,by,cp,cy,teacher+np.array([1.,2.,3.,4.])[:,None],bi,ci,**settings)
    check('teacher_row_offset_invariance',all(torch.allclose(v,shifted[a]['state_dict'][n],atol=1e-6,rtol=1e-6) for a in ['control','treatment'] for n,v in result[a]['state_dict'].items()))
    args=[model,bp,by,cp,cy,teacher,bi,ci]
    bads=[(2,by.astype(np.float32)),(2,np.array([9,1,2,0],dtype=np.int64)),(5,np.full_like(teacher,np.nan)),(6,bi.astype(float)),(6,np.full_like(bi,999)),(7,ci[:1]),(1,[])]
    for j,(ix,val) in enumerate(bads):
        wrong=list(args);wrong[ix]=val
        try:g.run_pair(*wrong,**settings)
        except (ValueError,TypeError):check('reject_invalid_'+str(j),True)
        else:check('reject_invalid_'+str(j),False)
    write(H/'IMPLEMENTATION_AUDIT.json',{'status':'PASS','checks':checks,'count':len(checks),'module_sha256':sha(H/'generated_v1.py'),'max_state_difference':worst,'max_loss_difference':lossworst,'real_training':0,'real_observations':0})
    print(json.dumps({'status':'PASS','checks':len(checks),'max_state_difference':worst,'max_loss_difference':lossworst}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        name='IMPLEMENTATION_FAILURE.json' if not (H/'IMPLEMENTATION_FAILURE.json').exists() else 'IMPLEMENTATION_FAILURE_FOLLOWUP.json'
        write(H/name,{'status':'FAIL','checks':checks,'traceback':traceback.format_exc()});raise
