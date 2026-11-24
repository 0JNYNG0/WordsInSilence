import time,os,traceback
from order_model import *
from order_contract import behavior

def preflight(rr,stats,flow):
    torch.manual_seed(42);m=OccurrenceOrderNet();m.eval()
    c=FlowRoleNet();c.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=True));c.eval()
    x,im,l,_,act=batch_j(rr[:2],stats);fl=flow_batch(rr[:2],flow)
    with torch.inference_mode():
        ac,ro,oc=m(x,im,l,fl);ca,cr=c(x,im,l,fl)
        torch.testing.assert_close(ac,ca,rtol=0,atol=0);torch.testing.assert_close(ro,cr,rtol=0,atol=0)
        torch.testing.assert_close(oc.sigmoid(),ro.softmax(-1)[:,:,1,None].expand_as(oc),rtol=1e-5,atol=1e-6)
        aa,ar,ao=m(x[:1,:l[0]],im[:1,:l[0]],l[:1],fl[:1,:l[0]])
        for a,b in [(ac[0,:l[0]],aa[0]),(ro[0,:l[0]],ar[0]),(oc[0,:l[0]],ao[0])]:torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(ac,m(x,im,l,-fl)[0],rtol=0,atol=0)
        p,_=directed_probability(ac[0].softmax(-1),oc[0].sigmoid(),fl[0,:,5],0,1)
        q,_=directed_probability(ac[0].softmax(-1),oc[0].sigmoid(),fl[0,:,5],1,0)
        torch.testing.assert_close(p+q,p.new_tensor(1.),rtol=1e-6,atol=1e-6)
    ac,ro,oc=m(x,im,l,fl);targets=occurrence_targets(rr[:2],x.shape[1]);loss=occurrence_loss(oc,targets,torch.ones(8))
    loss.backward();assert m.occurrence_query.weight.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in m.actor.parameters())
    unknown=[dict(r,order_queries=[dict(q,target=-1) for q in r['order_queries']]) for r in rr[:2]]
    missing,n=order_loss(ac.softmax(-1),oc.sigmoid(),unknown);assert n==0 and float(missing.detach())==0
    return dict(actor_frozen=True,initial_equal_flow_role=True,query_residual_zero=True,physical_target_and_flow_invariance=True,
        swapped_direction_complement=True,padding_invariance=True,unknown_order_loss_ignored=True,occurrence_gradient=True,behavior=behavior())

def main():
    assert not (ORDER_UNIT/'FROZEN.json').exists();torch.set_num_threads(2)
    assert read(ORDER_UNIT/'INPUTS_VERIFIED.json')['known_queries']==576
    rr=attach_queries(get_rows('fit'));stats=read(D/'NORMALIZATION.json');flow=read(FLOW_UNIT/'FLOW_INPUTS.json')
    pf=preflight(rr,stats,flow);exclusive(ORDER_UNIT/'PREFLIGHT.json',dict(at=clock(),**pf))
    pins=dict(read(FLOW_UNIT/'FROZEN.json')['pins'])
    for r in rr:pins[r['cache']]=r['cache_sha256']
    for p in [INITIAL,ORDER_UNIT/'PLAN.json',ORDER_UNIT/'ORDER_SUPERVISION.json',ORDER_UNIT/'INPUTS_VERIFIED.json',ORDER_UNIT/'PREFLIGHT.json',
        ORDER_UNIT/'order_model.py',ORDER_UNIT/'order_contract.py',ORDER_UNIT/'train_order.py',ORDER_UNIT/'evaluate_order.py',ORDER_UNIT/'verify_order.py',
        D/'NORMALIZATION.json',D/'evaluate_episode.py',F/'evaluate_split.py',T/'sequence_full.py',FLOW_UNIT/'VERIFIED.json']:
        pins[str(p)]=sha(p)
    prior=read(FLOW_OUT/'evaluation.json');normal={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
    family_baseline={str(g):sum(p['kind'] in normal and p['y']==g and p['pred']==g and p['eligible'] for p in prior['points']) for g in range(8)}
    plan=dict(at=clock(),previous_goal_turn='PROGRESS: V2fine-tunes176comparisons,Flow1fine-tune88comparisons,source contract audit,630query preparation and independent576known/54held verification.',
        conditions=['occurrence_only','directed_order'],primary='directed_order',seed=42,epochs=80,batch_size=12,lr=.0005,weight_decay=.01,
        initial=str(INITIAL),initial_is_experimental_not_adopted=True,fit_rows=252,fit_ids=list(range(1,10)),queries_known=576,queries_held=54,
        losses=dict(role_ce=1.,occurrence_bce=1.,order_by_condition=dict(occurrence_only=0.,directed_order=.25)),
        loss_averaging='Query order losses averaged per video, then per batch; no implicit three-edge word multiplier.',
        occurrence_labels='Reviewed matching main family positive; reviewed nonmain negative for relevance only; other and unlabelled remain unknown.',
        input_prohibition='No transform name, recordID, source index, time mapping or order target enters model.forward. Only observation tensors and learned flow hints; guide keys are fixed source-linked vectors.',
        interpretation='Source-conditioned relevance learned for all8families; physical readout frozen. Soft directed readout cannot override missing or short contradictory physical evidence.',
        baseline_known_main_eligible=582,baseline_family_eligible=family_baseline,
        gates=dict(raw_observation_exact=True,major_eligible_at_least582=True,family_eligible_not_worse=True,
            normal_order_above6_and_matched_control=True,normal_clean_at_least1_and_control=True,short_reverse_at_least2_and_control=True,
            reversed_false_supported0=True,normal_false_order0=True,image_gap_never_certified=True,normal_nonmain_false_main0=True,missing_major_families0=True),
        exact_rest_vs_context_is_diagnostic=True,thresholds=dict(observed_confidence=.8,main_confidence=.8,nonmain_confidence=.8,soft_order_confirmation=.8),
        major_gate_detail='Primary must retain at least582known main points AND not lose more than the same-architecture occurrence-only control. Each family must preserve the saved Flow primary baseline count.',
        evaluation='Two new models88same exposed DEV cases each once. Source IDs11-13 excluded from this training but previously exposed development, not final validation.',
        scope='Action occurrence/order component; full existing shape/body/motion feedback must still be integrated and validated. Not a whole-sign linguistic certificate.',
        preflight=pf,pins=pins,whole_goal_complete=False)
    exclusive(ORDER_UNIT/'FROZEN.json',plan)
    ys=np.concatenate([np.where(r['role']<0,-1,(r['role']==1).astype('int64')) for r in rr]);counts=np.bincount(ys[ys>=0],minlength=2)
    rw=torch.tensor(np.clip(counts.mean()/counts,.25,4),dtype=torch.float32)
    pos=np.zeros(8);neg=np.zeros(8)
    for r in rr:
        a=occurrence_targets([r],len(r['x']))[0].numpy();pos+=(a==1).sum(0);neg+=(a==0).sum(0)
    ow=torch.tensor(np.clip(neg/np.maximum(pos,1),.25,4),dtype=torch.float32);reports=[]
    for tag in plan['conditions']:
        torch.manual_seed(42);m=OccurrenceOrderNet();before={k:v.clone() for k,v in m.state_dict().items()};rng=np.random.default_rng(42)
        dest=ORDER_OUT/'runs'/tag;dest.mkdir(parents=True,exist_ok=False)
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0005,weight_decay=.01);history=[];start=time.perf_counter()
        for epoch in range(80):
            m.train();order=rng.permutation(len(rr));losses=[]
            for i in range(0,len(rr),12):
                br=[rr[k] for k in order[i:i+12]];x,im,l,_,act=batch_j(br,stats);fl=flow_batch(br,flow)
                ac,ro,oc=m(x,im,l,fl);ty=occurrence_targets(br,x.shape[1]);rl=loss_fn(ro,act,rw);cl=occurrence_loss(oc,ty,ow)
                ol,used=order_loss(ac.softmax(-1),oc.sigmoid(),br);loss=rl+cl+plan['losses']['order_by_condition'][tag]*ol
                assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
                losses.append([float(loss.detach()),float(rl.detach()),float(cl.detach()),float(ol.detach())])
            if epoch%20==0 or epoch==79:
                value=np.mean(losses,axis=0);h=dict(epoch=epoch+1,total=float(value[0]),role=float(value[1]),occurrence=float(value[2]),order=float(value[3]),seconds=time.perf_counter()-start)
                history.append(h);print(tag,h,flush=True);write(ORDER_UNIT/'TRAIN_PROGRESS.json',dict(at=clock(),status='TRAINING',pid=os.getpid(),condition=tag,epoch=epoch+1))
        delta={k:float((v-before[k]).abs().max()) for k,v in m.state_dict().items() if v.dtype.is_floating_point}
        assert all(d==0 for k,d in delta.items() if k.startswith('actor.') or k=='family_guide_keys') and delta['occurrence_query.weight']>0
        torch.save(m.state_dict(),dest/'model.pt');report=dict(condition=tag,sha256=sha(dest/'model.pt'),seconds=time.perf_counter()-start,history=history,changes=delta)
        exclusive(dest/'COMPLETE.json',report);reports.append(report)
    for p,h in pins.items():assert sha(p)==h
    exclusive(ORDER_UNIT/'TRAINING_RESULT.json',dict(at=clock(),runs=reports,occurrence_positive_counts=pos.tolist(),occurrence_negative_counts=neg.tolist(),weights=ow.tolist(),whole_goal_complete=False))
    write(ORDER_UNIT/'TRAIN_PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (ORDER_UNIT/'TRAIN_FIRST_FAILURE.txt').exists():(ORDER_UNIT/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
