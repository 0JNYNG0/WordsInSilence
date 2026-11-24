import time,os,traceback,shutil
from bridge_model import *
from evidence_contract import behavior

def main():
    assert not (V/'FROZEN.json').exists();torch.set_num_threads(2);rr=rows();stats=read(D/'NORMALIZATION.json')
    assert len(rr)==252 and {r['person'] for r in rr}==set(range(1,10))
    OUT.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(OUT).free>1024**3
    torch.manual_seed(42);a=BridgeNet(False);torch.manual_seed(42);b=BridgeNet(True);a.eval();b.eval()
    x,im,l,y,act=batch_j(rr[:2],stats)
    with torch.inference_mode():
        aa,ab=a(x,im,l);ba,bb=b(x,im,l);torch.testing.assert_close(aa,ba,rtol=0,atol=0);torch.testing.assert_close(ab,bb,rtol=0,atol=0)
        ca,cb=b(x[:1,:l[0]],im[:1,:l[0]],l[:1]);torch.testing.assert_close(ba[0,:l[0]],ca[0],rtol=1e-5,atol=1e-5);torch.testing.assert_close(bb[0,:l[0]],cb[0],rtol=1e-5,atol=1e-5)
    ba,bb=b(x,im,l);bb.square().sum().backward();assert b.content_bridge.weight.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in b.actor.parameters())
    files=[V/'bridge_model.py',V/'evidence_contract.py',V/'train_bridge.py',V/'evaluate_bridge.py',START,L/'PAIR_MANIFEST.json',D/'NORMALIZATION.json',O/'word100_action_entry_contract_20261102/REVIEW.json',J/'evaluate_witness.py',F/'evaluate_split.py']
    pins={str(p):sha(p) for p in files}
    for r in rr:pins[r['cache']]=r['cache_sha256']
    plan=dict(at=clock(),previous_turn='PROGRESS: L36new inputs,2matched fine-tunes176comparisons; raw587to589 but role suppression3to14. U sparse entry/exit audit no label-overwrite, no true dense boundary annotations.',
        primary='content_connected',conditions=['role_only','content_connected'],seed=42,epochs=80,batch_size=12,lr=.0005,weight_decay=.01,fit_rows=252,
        scope='Same initial L primary checkpoint and inputs. Actor including readout frozen. Two-way role neural layers fine-tuned. Candidate adds zero-init9probability-to122feature projection. Control feeds zeros to same projection. No target word or expected order added.',
        evidence_contract='Raw content immutable; process eligibility separate. Return configuration never relabelled as resting hand. Point errors, eligibility losses, whole event/order and important omission remain separate metrics.',
        evaluation='Fixed88Linput cases12coherent native plus76controls, 2new models once. Same previously exposed development IDs11-13, not final evaluation. Existing L saved raw content/observer arrays must match exactly.',
        training_labels='Same sparse directly reviewed role labels; unknown=-1. No guessed onset boundary or source text weaklabels promoted to dense gold.',
        comparison_gate=dict(raw_predictions_exact=True,main_eligibility_losses_below_parent14=True,normal_order_at_least_parent6=True,normal_clean_order_at_least_parent1=True,short_reverse_at_least2of3=True,false_supported_reverse0=True,image_gap_never_certified=True,normal_rest_at_least96of107=True),
        broader_acceptance_gate=read(D/'FROZEN.json')['gate'],preflight=dict(zero_init_equivalent=True,padding_invariant=True,actor_frozen=True,bridge_gradient=True,behavior=behavior()),pins=pins,whole_goal_complete=False)
    exclusive(V/'FROZEN.json',plan);reports=[]
    ys=np.concatenate([np.where(r['role']<0,-1,(r['role']==1).astype('int64')) for r in rr]);counts=np.bincount(ys[ys>=0],minlength=2);weights=torch.tensor(np.clip(counts.mean()/counts,.25,4),dtype=torch.float32)
    for tag in plan['conditions']:
        torch.manual_seed(42);m=BridgeNet(tag=='content_connected');before={k:v.clone() for k,v in m.state_dict().items()};rng=np.random.default_rng(42)
        dest=OUT/'runs'/tag;dest.mkdir(parents=True,exist_ok=False);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=plan['weight_decay']);history=[];start=time.perf_counter()
        for epoch in range(plan['epochs']):
            m.train();order=rng.permutation(len(rr));losses=[]
            for i in range(0,len(rr),12):
                batchrows=[rr[k] for k in order[i:i+12]];x,im,l,y,act=batch_j(batchrows,stats);ac,ro=m(x,im,l);loss=loss_fn(ro,act,weights);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();losses.append(float(loss.detach()))
            if epoch%20==0 or epoch==79:
                q=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.perf_counter()-start);history.append(q);print(tag,q,flush=True)
                write(V/'TRAIN_PROGRESS.json',dict(at=clock(),status='TRAINING',pid=os.getpid(),condition=tag,epoch=epoch+1))
        delta={k:float((v-before[k]).abs().max()) for k,v in m.state_dict().items() if v.dtype.is_floating_point};assert all(d==0 for k,d in delta.items() if k.startswith('actor.'))
        assert delta['role_output.weight']>0
        if tag=='content_connected':assert delta['content_bridge.weight']>0
        else:assert delta['content_bridge.weight']==0
        torch.save(m.state_dict(),dest/'model.pt');result=dict(condition=tag,sha256=sha(dest/'model.pt'),seconds=time.perf_counter()-start,history=history,changes=delta);exclusive(dest/'COMPLETE.json',result);reports.append(result)
    for p,h in pins.items():assert sha(p)==h
    exclusive(V/'TRAINING_RESULT.json',dict(at=clock(),runs=reports,actor_frozen=True,protected_inputs_unchanged=True,whole_goal_complete=False));write(V/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None))
if __name__=='__main__':
    try:main()
    except BaseException:
        if not (V/'TRAIN_FIRST_FAILURE.txt').exists():(V/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
