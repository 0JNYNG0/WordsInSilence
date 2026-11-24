import os,time,traceback
from torch.nn import functional as F
from process_model import *

def load_rows():
    rows=[]
    for meta in read(B/'PREPARED.json')['rows']:
        assert sha(meta['cache'])==meta['cache_sha256'];r=dict(meta);r.update(dict(np.load(meta['cache'])));rows.append(r)
    return rows

def protocol():
    assert not (B/'FROZEN.json').exists()
    pins={str(p):sha(p) for p in [B/'PLAN.json',B/'process_model.py',B/'prepare.py',B/'train.py',B/'SUPERVISION_AUDIT.json',B/'PREPARED.json',A/'runs/temporal_42/model.pt',A/'runs/pilot_temporal_42/model.pt',T/'runs/current_42/model.pt',A/'role_sequence.py',S/'LEARNED_STATE_GRAPH.json',P/'GUIDE_SUPPORT.json']}
    obj=dict(at=clock(),primary='joint_42',conditions=['rgb_role','joint'],seeds=[42,43,44],epochs=80,batch=12,lr=.0005,weight_decay=.01,
      architecture='Fine-tune a copied A BiGRU role branch plus zero-initialized RGB residual. Joint condition also learns sparse physical states and adjacent observed from-to endpoint contrasts. Existing observations never replaced.',
      losses=dict(role=1,physical=.25,pair=.25),
      split='01-09 FIT; pilot01-06 and07-09 development. Initial pilot role weights saw01-06 only, but frozen visual/physical pretraining previously saw01-09; pilot is not independent validation. DEV11-13 exposed development. No H14/15 reuse.',
      supervision='No dense interpolation. No assumed action completion from endpoints. Contrasts describe measured physical from-to orientation; they are not expert linguistic correctness. Cycles have no invented order/count.',
      pilot_gate=dict(role_macro_min=.8,core_recall_min=.9,pair_balanced_min=.75),
      gate=dict(normal_role_macro_min=.85,normal_core_recall_min=.95,normal_usable_order_min=21,normal_previous_order_loss=0,normal_false_order=0,strict_local_reverse_min=2,strict_local_reverse_false_supported=0,cut_motion_false_completion=0,image_gap_certified=0,observations_preserved=True),
      interpretation='Frozen A interpretation with0.8confidence, full native axis and unknown/short events. Joint auxiliary pair output cannot independently certify sequence. No DEV-driven seed/threshold selection.',
      integration='Reuse completed R motion observations to reject completion where movement continuity is absent. Only gate-passing scope enters separate runtime; full latency measured then. Head timing is not end-to-end.',pins=pins)
    exclusive(B/'FROZEN.json',obj);return obj

def ce(out,y,weights=None):
    v=F.cross_entropy(out.transpose(1,2),y,weight=weights,ignore_index=-1,reduction='none')
    return (v.sum(1)/(y>=0).sum(1).clamp_min(1)).mean()

def evaluate_points(m,rows,s):
    pp=[];phys=[];pair=[];m.eval()
    with torch.inference_mode():
        for r in rows:
            x,im,l,w,y,p=batch([r],s);a,b,h=m(x,im,l,w);pp.append(a.softmax(-1)[0].numpy());phys.append(b.softmax(-1)[0].numpy())
            logits,yy=m.pairs(h,[r])
            if len(yy):
                pair.extend([dict(y=int(v),probability=float(q),pred=int(q>=.5)) for v,q in zip(yy,logits.sigmoid())])
    return dict(role=state.metrics(point_records(rows,pp)),physical=state.metrics(point_records(rows,phys,'physical')),pair=state.metrics(pair),role_points=point_records(rows,pp),pair_points=pair)

def run(kind,seed,rows,s,tag,plan,pilot=False):
    d=B/'runs'/tag;assert not d.exists();d.mkdir(parents=True)
    torch.manual_seed(seed);rng=np.random.default_rng(seed);m=ProcessNet(pilot)
    # A matching architecture is used in both conditions; unused auxiliary
    # weights are frozen in the RGB-role control.
    if kind=='rgb_role':
        for part in [m.physical,m.query_delta,m.pair]:part.requires_grad_(False)
    init={k:v.clone() for k,v in m.state_dict().items()};opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=plan['weight_decay'])
    counts=np.bincount(np.concatenate([r['y'][r['y']>=0] for r in rows]),minlength=3)
    weights=torch.tensor(np.clip(counts.mean()/counts,.25,4),dtype=torch.float32)
    start=time.perf_counter();log=[]
    for ep in range(plan['epochs']):
        m.train();order=rng.permutation(len(rows));losses=[]
        for j in range(0,len(rows),plan['batch']):
            rr=[rows[k] for k in order[j:j+plan['batch']]];x,im,l,w,y,p=batch(rr,s);role,physical,h=m(x,im,l,w)
            loss=ce(role,y,weights)
            if kind=='joint':
                pair,py=m.pairs(h,rr);pl=F.binary_cross_entropy_with_logits(pair,py) if len(py) else pair
                loss=loss+.25*ce(physical,p)+.25*pl
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();losses.append(float(loss.detach()))
        if ep%20==0 or ep==plan['epochs']-1:
            v=dict(epoch=ep+1,loss=float(np.mean(losses)),seconds=time.perf_counter()-start);log.append(v);print(tag,v,flush=True)
            write(B/'PROGRESS.json',dict(at=clock(),status='TRAINING',pid=os.getpid(),run=tag,epoch=ep+1))
    changes={k:float((v-init[k]).abs().max()) for k,v in m.state_dict().items() if v.dtype.is_floating_point};assert changes['rgb.2.weight']>0
    torch.save(m.state_dict(),d/'model.pt');exclusive(d/'COMPLETE.json',dict(condition=kind,seed=seed,people=sorted({r['person'] for r in rows}),inputs=len(rows),seconds=time.perf_counter()-start,sha256=sha(d/'model.pt'),changes=changes,log=log))
    return m

def main():
    assert not (B/'TRAINING_RESULT.json').exists();torch.set_num_threads(2);rows=load_rows();plan=protocol();s=stats(rows)
    exclusive(B/'NORMALIZATION.json',s)
    # Identity at initialization, finite inputs, padding independence, image
    # absence gate, disjoint candidate/observer storage before any optimization.
    m=ProcessNet().eval();x,im,l,w,y,p=batch(rows[:2],s)
    with torch.inference_mode():
        a,b,h=m(x,im,l,w);torch.testing.assert_close(a,m.role(x,l),rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(a[0,:l[0]],m(x[:1,:l[0]],im[:1,:l[0]],l[:1],w[:1])[0][0],rtol=1e-5,atol=1e-5)
        empty=im.clone();empty[:,:,-1]=0;v=m(x,empty,l,w)[0];empty[:,:,:768]=13;torch.testing.assert_close(v,m(x,empty,l,w)[0])
    protected=old.observer();assert not ({p.data_ptr() for p in protected.parameters()}&{p.data_ptr() for p in m.parameters()})
    assert all(np.isfinite(r['rgb']).all() for r in rows)
    exclusive(B/'PREFLIGHT.json',dict(status='PASS',initial_role_parity=True,padding_independent=True,missing_rgb_does_not_contribute=True,no_shared_protected_weights=True,pairs=sum(len(r['pairs']) for r in rows),transformed_inputs=len(rows)))
    pilot=[r for r in rows if r['person']<=6];val=[r for r in rows if r['person']>=7];ps=stats(pilot);results={}
    exclusive(B/'PILOT_NORMALIZATION.json',ps)
    for kind in plan['conditions']:
        mm=run(kind,42,pilot,ps,'pilot_'+kind,plan,True);results[kind]=evaluate_points(mm,val,ps)
    mm=results['joint'];core=mm['role']['groups']['1'];passed=mm['role']['macro_recall']>=.8 and core['correct']/core['known']>=.9 and mm['pair']['macro_recall']>=.75
    exclusive(B/'PILOT.json',dict(passed=passed,results=results,scope=plan['split']))
    if not passed:
        write(B/'PROGRESS.json',dict(at=clock(),status='PILOT_GATE_FAILED',active=None));print('PILOT_GATE_FAILED',flush=True);return
    reports=[]
    for kind in plan['conditions']:
        for seed in plan['seeds']:
            tag=f'{kind}_{seed}';run(kind,seed,rows,s,tag,plan);reports.append(read(B/'runs'/tag/'COMPLETE.json'))
    for p,h in plan['pins'].items():assert sha(p)==h
    for p,h in read(B/'PREPARED.json')['pins'].items():assert sha(p)==h
    exclusive(B/'TRAINING_RESULT.json',dict(runs=reports,old_weights_unchanged=True,backbone_updates=0,supervised_copied_role_and_guide_finetuning=True))
    write(B/'PROGRESS.json',dict(at=clock(),status='COMPLETE',active=None,runs=6))

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (B/'TRAIN_FIRST_FAILURE.txt').exists():(B/'TRAIN_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
