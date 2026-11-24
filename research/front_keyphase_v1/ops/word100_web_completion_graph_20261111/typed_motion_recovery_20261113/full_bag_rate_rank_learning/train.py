"""One fixed full-bag/rate comparison to K; root review before freeze/run.

Retain six public FIT persons, all labels/candidates, original normalization.
Selected sources contribute all available rates, and each bag contributes every
cached positive/negative. New optimizer/fixed final checkpoints; no DEV tuning.
"""
from pathlib import Path
import collections, json, sys, time, traceback
import numpy as np
U=Path(__file__).resolve().parent;H=U.parent;G=H.parent
W=G/'candidate_description_supervision';K=H/'temporal_rate_rank_learning';J=H/'splice_rank_learning'
sys.path.insert(0,str(H))
from common import read,write,sha,stamp
sys.path.insert(0,str(W))
from local_rank_core import LocalRankHead
import torch
FAMILIES=[2,3,4,5,7]
MODES=['full_bag_rate_rank','old_local_full_plus_fullbag_half']
def sourcekey(r):return r['clip_id'],r['view'],r['reverse'],r['family']
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def stop():
    for p in[U,H,G]:
        for n in['STOP','STOP.txt','STOP_REQUESTED','STOP_REQUESTED.json']:assert not(p/n).exists(),str(p/n)

def load_bags(meta):
    assert meta['status']=='CANONICAL_RATE_FEATURES_READY_NOT_TRAINED'
    assert meta['no_dev_inputs']and meta['all_source_x0_279_exact']and meta['candidate_local_frozen_encoder']
    assert meta['source_contract_sha256']==sha(K/'FROZEN.json')
    assert meta['preparation_contract_sha256']==sha(K/'PREPARATION_FROZEN.json')
    assert len(meta['replay_original'])==178 and len(meta['replay_splice'])==72 and len(meta['bags'])==288
    assert sum(bool(r.get('skipped',False))for r in meta['bags'])==78
    assert meta['new_feature_bags']==210 and meta['held_cases']==78 and meta['source_clips']==6
    original=[];variants=collections.defaultdict(lambda:collections.defaultdict(list));bags=[];identities=set()
    def item(r,branch,rate):
        assert r['split']=='train'and r['person']in range(2,8)
        assert r['clip_id'].startswith('NIA_SL_WORD')and r['clip_id'].split('_')[3]==f"REAL{r['person']:02d}"
        assert sha(r['path'])==r['sha256']
        with np.load(r['path'],allow_pickle=False)as z:
            x=z['local_full'].copy();y=z['y'].copy()
            if 'rank_mask'in z.files:assert z['rank_mask'].all()
        assert x.dtype==np.float32 and x.shape==(len(y),1160)and np.isfinite(x).all()
        assert y.dtype==np.bool_ and y.ndim==1 and y.any()and(~y).any()
        pos=np.flatnonzero(y);neg=np.flatnonzero(~y);ids=np.r_[pos,neg]
        assert len(neg)<=512 and np.array_equal(np.sort(ids),np.arange(len(y)))
        identity=(str(r['path']),branch,float(rate));assert identity not in identities;identities.add(identity)
        d=dict(row=r,x=x,y=y,positive=pos,negative=neg,all_indices=ids,branch=branch,rate=float(rate),index=len(bags))
        bags.append(d);return d
    for r in meta['replay_original']:
        d=item(r,'normal',1.);original.append(d)
        if r['family']in[2,3]:variants[sourcekey(r)]['normal'].append(d)
    for r in meta['replay_splice']:
        assert r['family']in[2,3]and not r.get('skipped',False)
        variants[sourcekey(r)]['splice'].append(item(r,'splice',1.))
    for r in meta['bags']:
        assert r['split']=='train'and r['person']in range(2,8)
        assert r['family']in[2,3]and r['base_kind']in['normal','splice']and r['rate']in[.8,1.2]
        if r.get('skipped',False):
            assert r['positive']==0 or r['negative']==0
            continue
        variants[sourcekey(r)][r['base_kind']].append(item(r,r['base_kind'],r['rate']))
    assert len(original)==178 and len(bags)==460 and len(variants)==72
    assert len({sourcekey(d['row'])for d in original})==178
    assert set(variants)=={sourcekey(d['row'])for d in original if d['row']['family']in[2,3]}
    for v in variants.values():
        assert set(v)=={'normal','splice'}
        for branch,entries in v.items():
            rates=[d['rate']for d in entries]
            assert 1. in rates and set(rates)<={.8,1.,1.2}and len(rates)==len(set(rates))
            entries.sort(key=lambda d:d['rate'])
    groups={f:[i for i,d in enumerate(original)if d['row']['family']==f]for f in FAMILIES}
    assert {f:len(v)for f,v in groups.items()}=={2:36,3:36,4:34,5:36,7:36}
    assert {d['row']['person']for d in bags}==set(range(2,8))
    assert len({d['row']['clip_id']for d in bags})==24
    assert len({d['row']['clip_id']for d in bags if d['row']['family']in[2,3]})==6
    return original,variants,groups,bags

def load_schedule(original):
    schedule=read(U/'SOURCE_SCHEDULE.json')
    verification=read(U/'SOURCE_SCHEDULE_VERIFICATION.json')
    assert schedule['status']=='FROZEN_SOURCE_SCHEDULE_MATCHING_K_RNG'
    assert verification['status']=='SOURCE_SCHEDULE_SAVED_RNG_COUNTS_VERIFIED_NO_TRAINING'
    assert verification['source_schedule_sha256']==sha(U/'SOURCE_SCHEDULE.json')and verification['all_pins_unchanged']
    assert verification['replayed_bag_count_checks']==1380
    assert schedule['family_order']==FAMILIES and schedule['seeds']==[42,43,44]and schedule['steps_per_seed']==1200
    assert len(schedule['original_bags'])==178
    for i,(record,d)in enumerate(zip(schedule['original_bags'],original)):
        assert record==dict(index=i,**{k:d['row'][k]for k in['clip_id','person','view','reverse','family','path','sha256']})
    expected={}
    for seed in[42,43,44]:
        rows=schedule['schedules'][str(seed)];assert len(rows)==1200
        counts=np.zeros(178,np.int64)
        for row in rows:
            assert len(row)==5
            for family,indices in zip(FAMILIES,row):
                assert len(indices)==2 and len(set(indices))==2
                for i in indices:
                    assert type(i)is int and 0<=i<178 and original[i]['row']['family']==family
                    counts[i]+=1
        expected[seed]=counts
    check(schedule['pins'])
    return schedule,expected

def full_batch(original,variants,step_sources,bag_counts,exposure):
    """Exact saved K source choices; all branch/rate candidates are used."""
    xs=[];pieces=[];draws=[];offset=0;step_bags=set()
    for family,source_indices in zip(FAMILIES,step_sources):
        assert len(source_indices)==2 and len(set(source_indices))==2
        for original_index in source_indices:
            d=original[int(original_index)];selected=[]
            assert d['row']['family']==family
            if family in[2,3]:
                vv=variants[sourcekey(d['row'])]
                for branch in['normal','splice']:
                    entries=vv[branch]
                    selected.extend((entry,.5/len(entries))for entry in entries)
            else:selected=[(d,1.)]
            assert np.isclose(sum(w for _,w in selected),1)
            chosen=[]
            for data,source_weight in selected:
                bi=data['index'];assert bi not in step_bags;step_bags.add(bi)
                ids=data['all_indices'];n=len(ids);npositive=len(data['positive'])
                # Every existing candidate exactly once, with positives first
                # only to express the unchanged listwise-positive-mass loss.
                assert n==len(data['y']) and npositive+len(data['negative'])==n
                xs.append(data['x'][ids])
                pieces.append((offset,offset+n,npositive,FAMILIES.index(family),source_weight/10))
                offset+=n;bag_counts[bi]+=1;exposure[bi][ids]+=1
                chosen.append(dict(bag_index=bi,branch=data['branch'],rate=data['rate'],
                    loss_weight=source_weight/10,candidates=n,positive=npositive,negative=len(data['negative'])))
            draws.append(dict(original_index=int(original_index),clip_id=d['row']['clip_id'],view=d['row']['view'],
                reverse=d['row']['reverse'],family=family,bags=chosen))
    assert len(draws)==10 and np.isclose(sum(p[-1]for p in pieces),1)
    for family in FAMILIES:assert np.isclose(sum(p[-1]for p in pieces if p[3]==FAMILIES.index(family)),.2)
    return np.concatenate(xs),pieces,draws

def freeze():
    stop();assert not(U/'TRAINING_FROZEN.json').exists()and not(U/'runs').exists()
    meta=read(K/'PREPARATION_INPUTS_COMPLETE.json');comparison=read(K/'TRAINING_COMPLETE.json')
    prior=read(J/'TRAINING_COMPLETE.json');parents=meta['parent_models']
    assert comparison['status']=='FIXED_RATE_HEAD_TRAINING_COMPLETE_NOT_ADOPTED'
    assert len(parents)==len(prior['rows'])==len(comparison['rows'])==3
    assert {r['seed']for r in parents}=={42,43,44}
    assert {r['seed']:(r['model'],r['model_sha256'])for r in parents}=={r['seed']:(r['model'],r['model_sha256'])for r in prior['rows']}
    assert {r['seed']:(r['model'],r['model_sha256'])for r in parents}=={r['seed']:(r['initial_model'],r['initial_model_sha256'])for r in comparison['rows']}
    original,variants,groups,bags=load_bags(meta)
    schedule,expected=load_schedule(original)
    pins=read(G/'PROTECTIONS.json').copy();assert len(pins)==477;check(pins)
    files=[Path(__file__),U/'README_CONTRACT.txt',K/'PREPARATION_INPUTS_COMPLETE.json',K/'PREPARATION_FROZEN.json',
        K/'FROZEN.json',K/'TRAINING_COMPLETE.json',K/'TRAINING_FROZEN.json',K/'FIT_FAILURE_CAUSES.json',
        J/'TRAINING_COMPLETE.json',U/'SOURCE_SCHEDULE.json',U/'SOURCE_SCHEDULE_VERIFICATION.json',W/'local_rank_core.py',H/'common.py']
    pins.update({str(p):sha(p)for p in files})
    for p,h in schedule['pins'].items():
        assert p not in pins or pins[p]==h
        pins[p]=h
    for d in bags:pins[d['row']['path']]=d['row']['sha256']
    for r in prior['rows']:pins[r['model']]=r['model_sha256']
    check(pins)
    write(U/'TRAINING_FROZEN.json',dict(at=stamp(),status='FIXED_FULL_BAG_RATE_RECIPE_BEFORE_TRAINING',
        seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,gradient_clip_norm=2,cpu_threads=4,interop_threads=1,
        parent='Same matching J seed parent that initialized K; NOT the trained K checkpoint. Every initial weight/normalization is identical to K. New AdamW optimizer state. No shared encoder change.',
        inputs=dict(original=178,splice=72,new_rate=210,held_rate=78,all_rate_audit=288,training_bags=460,total_source_clips=24,rate_expansion_source_clips=6,persons=list(range(2,8))),
        loss='Unchanged listwise positive mass: logsumexp(all cached candidates)-logsumexp(all positive candidates). All existing negative candidates (<=512) included every selected bag draw.',
        weighting='Five families each1/5; same two original sources per family/step as K each1/10. Family2/3 each source normal/splice each1/2; all available rates simultaneously receive branchweight/nrate. Other families use original full bag.',
        source_rng='No new source RNG: SOURCE_SCHEDULE reproduces all K source choices after replaying its original rate and negative RNG consumption. Every source/step schedule is identical to K.',
        exposure='Every step source/bag draw saved. Every positive and negative has exactly its bag draw count. Unselected bag is reported, never forces seed retry.',
        held='All78 no-full-positive variants remain held audit entries; no new label, candidate, source, oversampling, probe or DEV selection.',
        architecture='Original LocalRankHead1160->96 GELU->5 unchanged; old mean/std buffers frozen.',
        checkpoints='Fixed final1200step for all3seeds; no epoch/seed choice.',
        evaluation_conditions=MODES,runtime_scope='Families2/3 only; other outputs must be copied original local_full by evaluator.',
        gates='OriginalG2/source-level normal/speed/short/cross/shape/direction/absence, then changed-span semantic verification. No gate weakening.',
        no_private=True,no_final8=True,no_web=True,no_training_label_change=True,no_threshold_change=True,no_encoder_inference=True,
        lineage='Addresses full cached negative and rate exposure, not proof that underexposure is the only cause or that optimization will solve boundary errors.',
        pins=pins))
    print('Full-bag rate contract frozen; no optimizer run.',flush=True)

def run():
    stop();assert not(U/'runs').exists()and not(U/'TRAINING_COMPLETE.json').exists()
    cfg=read(U/'TRAINING_FROZEN.json');check(cfg['pins'])
    assert cfg['seeds']==[42,43,44]and cfg['steps']==1200 and cfg['lr']==.001 and cfg['weight_decay']==.001
    metadata=read(K/'PREPARATION_INPUTS_COMPLETE.json')
    original,variants,groups,bags=load_bags(metadata);schedule,expected_source_counts=load_schedule(original)
    priors={r['seed']:r for r in metadata['parent_models']};assert len(metadata['parent_models'])==3 and set(priors)=={42,43,44}
    original_index={sourcekey(d['row']):i for i,d in enumerate(original)}
    torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(cfg['interop_threads'])
    (U/'runs').mkdir();runs=[]
    for seed in cfg['seeds']:
        stop();torch.manual_seed(seed);p=priors[seed]
        assert sha(p['model'])==p['model_sha256']
        state=torch.load(p['model'],map_location='cpu',weights_only=True)
        model=LocalRankHead(state['mean'].numpy(),state['std'].numpy());model.load_state_dict(state)
        assert all(torch.equal(v,state[k])and v.data_ptr()!=state[k].data_ptr()for k,v in model.state_dict().items())
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        counts=np.zeros(len(bags),np.int64);exposure=[np.zeros(len(d['y']),np.int32)for d in bags]
        dst=U/'runs'/str(seed);dst.mkdir();drawpath=dst/'SOURCE_STEP_DRAWS.jsonl';history=[];tic=time.perf_counter();model.train()
        with drawpath.open('x',encoding='utf-8')as df:
            for step in range(cfg['steps']):
                stop();step_sources=schedule['schedules'][str(seed)][step]
                x,pieces,draws=full_batch(original,variants,step_sources,counts,exposure)
                logits=model(torch.tensor(x))
                loss=sum(w*(torch.logsumexp(logits[a:b,j],0)-torch.logsumexp(logits[a:a+n,j],0))for a,b,n,j,w in pieces)
                assert torch.isfinite(loss);optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm']);optimizer.step()
                df.write(json.dumps(dict(step=step+1,sources=draws),ensure_ascii=False,separators=(',',':'))+'\n')
                if step%300==0 or step==cfg['steps']-1:
                    df.flush();history.append(dict(step=step+1,loss=float(loss.detach()),batch_candidates=len(x),batch_bags=len(pieces)))
                    print('full-bag rate rank',seed,step+1,history[-1],flush=True)
        training_seconds=time.perf_counter()-tic;model.eval()
        assert torch.equal(model.mean,state['mean'])and torch.equal(model.std,state['std'])
        for i,d in enumerate(bags):
            # No sampled negative can escape its bag's exposure count.
            assert counts[i]==expected_source_counts[seed][original_index[sourcekey(d['row'])]]
            assert np.all(exposure[i]==counts[i])
            assert int(exposure[i][d['negative']].min())>=int(counts[i])
            assert int(exposure[i][d['positive']].min())==int(counts[i])
        modelpath=dst/'model.pt';torch.save(model.state_dict(),modelpath)
        ep=dst/'candidate_exposures.npz';np.savez_compressed(ep,**{f'bag_{i}':v for i,v in enumerate(exposure)})
        pred={};index=[];prediction_tic=time.perf_counter()
        with torch.inference_mode():
            for i,d in enumerate(bags):
                name=f'bag_{i}';pred[name]=model(torch.tensor(d['x'])).numpy()
                index.append(dict(key=name,path=d['row']['path'],sha256=d['row']['sha256'],family=d['row']['family'],
                    clip_id=d['row']['clip_id'],person=d['row']['person'],view=d['row']['view'],reverse=d['row']['reverse'],
                    base_kind=d['branch'],rate=d['rate'],sampled_steps=int(counts[i]),candidates=len(d['y']),
                    positive=len(d['positive']),negative=len(d['negative']),
                    positive_exposure_min=int(exposure[i][d['positive']].min()),positive_exposure_max=int(exposure[i][d['positive']].max()),
                    negative_exposure_min=int(exposure[i][d['negative']].min()),negative_exposure_max=int(exposure[i][d['negative']].max())))
        pp=dst/'fit_predictions.npz';np.savez_compressed(pp,**pred);ip=dst/'FIT_PREDICTION_INDEX.json';write(ip,index)
        rec=dict(seed=seed,condition=MODES[0],model=str(modelpath),model_sha256=sha(modelpath),
            initial_model=p['model'],initial_model_sha256=p['model_sha256'],normalization_unchanged=True,
            fit_predictions=str(pp),fit_predictions_sha256=sha(pp),fit_prediction_index=str(ip),fit_prediction_index_sha256=sha(ip),
            source_step_draws=str(drawpath),source_step_draws_sha256=sha(drawpath),
            candidate_exposures=str(ep),candidate_exposures_sha256=sha(ep),
            training_seconds=training_seconds,fit_prediction_seconds=time.perf_counter()-prediction_tic,log=history,steps=cfg['steps'],
            all_training_bags_sampled=bool((counts>0).all()),minimum_bag_draws=int(counts.min()),maximum_bag_draws=int(counts.max()),
            every_candidate_exposure_equals_bag_draws=True,no_retry_for_coverage=True)
        runs.append(rec);write(U/'TRAINING_PROGRESS.json',dict(at=stamp(),completed=len(runs),rows=runs))
    check(cfg['pins']);stop()
    write(U/'TRAINING_COMPLETE.json',dict(at=stamp(),status='FULL_BAG_RATE_TRAINING_COMPLETE_NOT_ADOPTED',rows=runs,
        encoder_unchanged=True,labels_candidates_unchanged=True,private_access=False,final8=False,web_changed=False,
        no_epoch_or_seed_selection=True,training_frozen_sha256=sha(U/'TRAINING_FROZEN.json')))

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1]in['freeze','run-reviewed']
        freeze()if sys.argv[1]=='freeze'else run()
    except BaseException:
        U.mkdir(parents=True,exist_ok=True)
        (U/f'TRAIN_FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise

