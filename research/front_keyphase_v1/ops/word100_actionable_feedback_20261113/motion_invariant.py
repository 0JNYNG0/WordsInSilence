"""Bounded FIT-only test: reversal-invariant translation-present head.
Not a direction, completion, or sign correctness head. Original models unchanged.
"""
from pathlib import Path
import json, hashlib, time, sys
import numpy as np
U=Path(__file__).resolve().parent
OLD=U.parent/'word100_feedback_recovery_20261113/motion_recovery'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

def invariant_features(q,valid,exists):
    g=np.asarray(valid,bool)&np.asarray(exists,bool)&np.isfinite(q).all(1)
    p=np.asarray(q,dtype=np.float64)[g]
    pair=g[1:]&g[:-1]
    d=np.linalg.norm(np.diff(np.asarray(q,dtype=np.float64),axis=0)[pair],axis=1)
    # Sort before reduction to avoid order-dependent rounding differences.
    d=np.sort(d)
    if len(p)<2:return np.zeros(12,np.float32)
    distances=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=-1)
    center=np.array([np.sort(p[:,i]).mean() for i in range(2)])
    radii=np.sort(np.linalg.norm(p-center,axis=1))
    path=float(d.sum())
    return np.array([distances.max(),path,np.linalg.norm(p[-1]-p[0]),*np.ptp(p,axis=0),
        d.mean() if len(d) else 0,np.median(d) if len(d) else 0,d.max() if len(d) else 0,
        d.std() if len(d) else 0,radii.max(),radii.mean(),d[-3:].sum()],np.float32)

def prepare():
    assert not (U/'MOTION_FROZEN.json').exists()
    z=np.load(OLD/'PHYSICAL_INPUTS.npz');meta=read(OLD/'PHYSICAL_INDEX.json')['rows']
    x=np.stack([invariant_features(q,v,e) for q,v,e in zip(z['q'],z['valid'],z['exists'])])
    xr=np.stack([invariant_features(q[::-1],v[::-1],e[::-1]) for q,v,e in zip(z['q'],z['valid'],z['exists'])])
    assert np.array_equal(x,xr),float(np.max(abs(x-xr)))
    # Large camera/body translation cannot change these features, within float tolerance.
    xt=np.stack([invariant_features(q.astype(float)+[3.0,-2.0],v,e) for q,v,e in zip(z['q'],z['valid'],z['exists'])])
    assert np.allclose(x,xt,rtol=1e-5,atol=1e-6)
    assert len(meta)==len(x)==1380
    folds=read(OLD/'PHYSICAL_FROZEN.json')['folds']
    for f in folds:
        w=np.load(f['weights'])['weights'];tr=w>0;held=z['person']==f['person']
        assert not (tr&held).any()
        assert set(r['source_sha256'] for r,b in zip(meta,tr) if b).isdisjoint(set(r['source_sha256'] for r,b in zip(meta,held) if b))
        assert not (tr&(z['y']<0)).any()
    np.savez_compressed(U/'MOTION_INPUTS.npz',x=x,eligible=z['eligible'],y=z['y'],person=z['person'])
    pins={str(p):sha(p) for p in [Path(__file__),OLD/'PHYSICAL_INPUTS.npz',OLD/'PHYSICAL_INDEX.json',OLD/'PHYSICAL_SOURCE_LABELS.json',OLD/'PHYSICAL_ADOPTION_CONTRACT.json',U/'MOTION_INPUTS.npz']}
    for f in folds:
        pins[f['weights']]=sha(f['weights']);pins[f['normalization_rule']]=sha(f['normalization_rule'])
    put(U/'MOTION_FROZEN.json',dict(condition='invariant_translation_statistics_12',seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,folds=folds,pins=pins,
        hidden=16,features='pairwise_max,path,endpoint_norm,axis_spans,step_mean/median/max/std,centred_radii_max/mean,largest3_steps_sum',
        output='M when logit>=0 else H; same eligibility U; no magnitude threshold tuning',
        scope='FIT-only held-record diagnostic; previously exposed records; not fresh evaluation',
        adoption='Original PHYSICAL_ADOPTION_CONTRACT all preservation gates plus no held known cell lost against old same-seed MLP; direction/semantic/link gates remain separate',
        old_model_or_direction_changed=False,full_common100_retraining=False))
    put(U/'MOTION_PRETRAIN_VERIFIED.json',dict(passed=True,rows=len(x),unique_source_hands=len(set(r['source_key'] for r in meta)),exact_reverse_invariance=True,translation_invariance=True,held_source_separation=True,unknown_not_trained=True,frozen_sha256=sha(U/'MOTION_FROZEN.json')))

def train():
    import torch
    from torch import nn
    cfg=read(U/'MOTION_FROZEN.json');verify=read(U/'MOTION_PRETRAIN_VERIFIED.json')
    assert verify['passed'] and verify['frozen_sha256']==sha(U/'MOTION_FROZEN.json')
    for p,h in cfg['pins'].items():assert sha(p)==h,p
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    z=np.load(U/'MOTION_INPUTS.npz');x=z['x'];y=torch.tensor(z['y'],dtype=torch.float32);done=[]
    for f in cfg['folds']:
        pid=f['person'];w=np.load(f['weights'])['weights'];tr=w>0
        mean=x[tr].astype(float).mean(0);std=np.maximum(x[tr].astype(float).std(0),1e-4)
        xt=torch.tensor((x-mean)/std,dtype=torch.float32);wt=torch.tensor(w[tr]);mask=torch.tensor(tr)
        for seed in cfg['seeds']:
            d=U/'motion_runs'/f'{pid}_{seed}';d.mkdir(parents=True,exist_ok=True)
            if (d/'COMPLETE.json').exists():done.append(read(d/'COMPLETE.json'));continue
            assert not (d/'STARTED.json').exists(),'Audit interrupted run first'
            put(d/'STARTED.json',dict(pid=pid,seed=seed,frozen_sha256=sha(U/'MOTION_FROZEN.json')))
            t=time.perf_counter();torch.manual_seed(seed)
            net=nn.Sequential(nn.Linear(12,16),nn.GELU(),nn.Linear(16,1))
            opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);lossfn=nn.BCEWithLogitsLoss(reduction='none')
            for step in range(cfg['steps']):
                opt.zero_grad(set_to_none=True);loss=(lossfn(net(xt[mask]).squeeze(1),y[mask])*wt).sum();assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),2);opt.step()
            net.eval()
            with torch.inference_mode():logits=net(xt).squeeze(1).numpy()
            np.savez_compressed(d/'predictions.npz',logits=logits,held=z['person']==pid,eligible=z['eligible'])
            torch.save(dict(state_dict=net.state_dict(),mean=mean,std=std,task='window_contains_translation',features=cfg['features']),d/'model.pt')
            row=dict(person=pid,seed=seed,seconds=time.perf_counter()-t,loss=float(loss),parameters=sum(p.numel() for p in net.parameters()),prediction_sha256=sha(d/'predictions.npz'),model_sha256=sha(d/'model.pt'))
            put(d/'COMPLETE.json',row);done.append(row);print(f'New fixed motion candidate {len(done)}/18',flush=True)
    if not (U/'MOTION_TRAINED.json').exists():put(U/'MOTION_TRAINED.json',dict(runs=done,frozen_sha256=sha(U/'MOTION_FROZEN.json')))

def compare():
    z=np.load(U/'MOTION_INPUTS.npz');oldz=np.load(OLD/'PHYSICAL_INPUTS.npz');rows=read(OLD/'PHYSICAL_INDEX.json')['rows'];cfg=read(U/'MOTION_FROZEN.json');gates=read(OLD/'PHYSICAL_ADOPTION_CONTRACT.json')
    ys=z['y'];eligible=z['eligible'];stats=[];details=[];allpass=True
    for seed in cfg['seeds']:
        new=np.full(len(ys),-1,int);old=new.copy();rule=new.copy()
        for f in cfg['folds']:
            pid=f['person'];idx=z['person']==pid;n=np.load(U/'motion_runs'/f'{pid}_{seed}'/'predictions.npz');p=np.load(OLD/'physical_runs'/f'{pid}_{seed}'/'predictions.npz')
            assert np.array_equal(n['held'],idx) and np.array_equal(p['held'],idx)
            new[idx&eligible]=(n['logits'][idx&eligible]>=0);old[idx&eligible]=(p['logits'][idx&eligible]>=0)
            rule[idx&eligible]=p['rule_prediction'][idx&eligible]
        for view in ['full','upper','actual_crop']:
            for reverse in [False,True]:
                base=np.array([r['view']==view and r['reverse']==reverse for r in rows])&(ys>=0)
                subsets={'all':base,'M':base&(ys==1),'H':base&(ys==0)}
                for hand in ['L','R']:
                    h=np.array([r['side']==hand for r in rows]);subsets[hand]=base&h
                    for label,value in [('M',1),('H',0)]:subsets[hand+label]=base&h&(ys==value)
                cyc=base&np.array([r['word_id']=='WORD1567' and r['kind'].startswith('main_') for r in rows])
                subsets['cycle']=cyc
                for hand in ['L','R']:subsets['cycle'+hand]=cyc&np.array([r['side']==hand for r in rows])
                subsets['short']=base&np.array([r['source_key'] in gates['short_change']['source_hand_keys'] for r in rows])
                for name,m in subsets.items():
                    a=int((new[m]==ys[m]).sum());b=int((rule[m]==ys[m]).sum());c=int((old[m]==ys[m]).sum());passed=a>=b and a>=c
                    stats.append(dict(seed=seed,view=view,reverse=reverse,group=name,total=int(m.sum()),candidate=a,rule=b,previous=c,passed=passed));allpass&=passed
        losses=(ys>=0)&(old==ys)&(new!=ys);allpass&=not losses.any()
        for i,r in enumerate(rows):
            details.append(dict(**r,seed=seed,prediction=int(new[i]),previous=int(old[i]),rule=int(rule[i]),eligible=bool(eligible[i]),truth=int(ys[i])))
    put(U/'MOTION_PREDICTIONS.json',details)
    put(U/'MOTION_COMPARISON.json',dict(adoption_gates_passed=bool(allpass),statistics=stats,scope=cfg['scope'],whole_G2_pass=False,new_direction_model=False,web_applied=False))
    print(json.dumps({'adoption_gates_passed':bool(allpass),'primary':[x for x in stats if x['view']=='full' and not x['reverse'] and x['group']=='all']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    globals()[sys.argv[1]]()
