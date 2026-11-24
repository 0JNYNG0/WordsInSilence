import time,os,traceback
from flow_model import *
from evidence_contract import behavior

def main():
    assert not (FLOW_UNIT/'FROZEN.json').exists();torch.set_num_threads(2)
    prepared=read(FLOW_UNIT/'FLOW_INPUTS.json');rr=get_rows('fit');stats=read(D/'NORMALIZATION.json')
    torch.manual_seed(42);m=FlowRoleNet();control=BridgeNet(False);m.eval();control.eval()
    x,im,l,y,roles=batch_j(rr[:2],stats);flow=flow_batch(rr[:2],prepared)
    with torch.inference_mode():
        ac,ro=m(x,im,l,flow);ca,cr=control(x,im,l)
        torch.testing.assert_close(ac,ca,rtol=0,atol=0);torch.testing.assert_close(ro,cr,rtol=0,atol=0)
        changed=m(x,im,l,-flow)[0];torch.testing.assert_close(ac,changed,rtol=0,atol=0)
        aa,ar=m(x[:1,:l[0]],im[:1,:l[0]],l[:1],flow[:1,:l[0]])
        torch.testing.assert_close(ac[0,:l[0]],aa[0],rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(ro[0,:l[0]],ar[0],rtol=1e-5,atol=1e-5)
    m(x,im,l,flow)[1].square().sum().backward();assert m.flow_bridge.weight.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in m.actor.parameters())
    pins=dict(prepared['pins'])
    for p in [FLOW_UNIT/'FLOW_INPUTS.json',FLOW_UNIT/'train_flow.py',FLOW_UNIT/'evaluate_flow.py',FLOW_UNIT/'verify_flow.py',START,V/'VERIFIED.json',V/'TRAINING_RESULT.json',V/'evidence_contract.py']:
        pins[str(p)]=sha(p)
    for item in prepared['items'].values():pins[item['path']]=item['sha256']
    plan=dict(at=clock(),primary='flow_connected',fit_rows=252,epochs=80,seed=42,batch_size=12,lr=.0005,weight_decay=.01,
        control='Reuse V role_only SAME L initial state,252FIT inputs, labels,weights,optimizer,80epochs,seed and batch order. No completed control rerun.',
        scope='Supervised process-role fine-tuning with target learned-prototype DTW hints. Actor and rest observer immutable. No physical target conditioning. Same guides as original learned prototype.',
        labels='Existing source-reviewed sparse process labels only. DTW span membership is not a target. Unlabelled remains -1.',
        no_forced_order=True,old_thresholds_unchanged=True,new_runtime_reference_videos=0,
        gate=dict(raw_exact=True,main_exclusions_below_control9=True,normal_order_above_control6=True,normal_clean_order_at_least1=True,
            short_reverse_min2=True,reverse_false_supported0=True,normal_false_order0=True,normal_rest_min96=True,image_gap_held=True,whole_event_missing0=True),
        evaluation='88 public exposed DEV conditions once; IDs11-13 never in training. Old saved control untouched. Not independent final validation.',
        preflight=dict(zero_init_equal_control=True,physical_content_independent_of_flow=True,padding_invariant=True,actor_frozen=True,flow_gradients=True,behavior=behavior()),
        pins=pins,whole_goal_complete=False)
    exclusive(FLOW_UNIT/'FROZEN.json',plan)
    torch.manual_seed(42);m=FlowRoleNet();before={k:v.clone() for k,v in m.state_dict().items()};rng=np.random.default_rng(42)
    ys=np.concatenate([np.where(r['role']<0,-1,(r['role']==1).astype('int64')) for r in rr]);counts=np.bincount(ys[ys>=0],minlength=2)
    weights=torch.tensor(np.clip(counts.mean()/counts,.25,4),dtype=torch.float32)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0005,weight_decay=.01)
    dest=FLOW_OUT/'run';dest.mkdir(parents=True,exist_ok=False);history=[];start=time.perf_counter()
    for epoch in range(80):
        m.train();order=rng.permutation(len(rr));losses=[]
        for i in range(0,len(rr),12):
            br=[rr[k] for k in order[i:i+12]];x,im,l,y,act=batch_j(br,stats);flow=flow_batch(br,prepared)
            ac,ro=m(x,im,l,flow);loss=loss_fn(ro,act,weights);assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();losses.append(float(loss.detach()))
        if epoch%20==0 or epoch==79:
            h=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.perf_counter()-start);history.append(h);print(h,flush=True)
            write(FLOW_UNIT/'TRAIN_PROGRESS.json',dict(at=clock(),status='TRAINING',pid=os.getpid(),epoch=epoch+1))
    delta={k:float((v-before[k]).abs().max()) for k,v in m.state_dict().items() if v.dtype.is_floating_point}
    assert all(d==0 for k,d in delta.items() if k.startswith('actor.')) and delta['flow_bridge.weight']>0
    assert delta['content_bridge.weight']==0
    m.eval();x,im,l,_,_=batch_j(rr[:2],stats);flow=flow_batch(rr[:2],prepared)
    with torch.inference_mode():
        a,b=m(x,im,l,flow);c,d=m(x,im,l,torch.flip(flow,[1]))
        assert torch.equal(a,c) and not torch.equal(b,d)
    torch.save(m.state_dict(),dest/'model.pt')
    for p,h in pins.items():assert sha(p)==h
    exclusive(FLOW_UNIT/'TRAINING_RESULT.json',dict(at=clock(),sha256=sha(dest/'model.pt'),history=history,changes=delta,seconds=time.perf_counter()-start,
        posttrain_flow_changes_role_not_physical_content=True,whole_goal_complete=False))
    write(FLOW_UNIT/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (FLOW_UNIT/'TRAIN_FIRST_FAILURE.txt').exists():(FLOW_UNIT/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
