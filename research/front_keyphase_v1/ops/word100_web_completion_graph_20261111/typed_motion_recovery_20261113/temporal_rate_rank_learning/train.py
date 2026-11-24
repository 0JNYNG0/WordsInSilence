"""Fixed FIT time-rate continuation of J, after input and parent review.

The encoder and original models remain immutable. Select fixed final checkpoints,
never a DEV-selected seed or epoch. Different rate variants share one source's
sampling weight; more variants do not make a person more important.
"""
from pathlib import Path
import collections,sys,time,traceback
import numpy as np
U=Path(__file__).resolve().parent;H=U.parent;W=H.parent/'candidate_description_supervision';J=H/'splice_rank_learning'
sys.path.insert(0,str(H))
from common import read,write,sha,stamp
sys.path.insert(0,str(W))
from local_rank_core import LocalRankHead
import torch
FAMILIES=[2,3,4,5,7]

def key(r):return r['clip_id'],r['view'],r['reverse'],r['family']
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def stop():
    for d in [U,H,H.parent]:
        for n in ['STOP_REQUESTED','STOP_REQUESTED.json','STOP','STOP.txt']:assert not(d/n).exists(),str(d/n)

def load_bags(meta):
    assert meta['status']=='CANONICAL_RATE_FEATURES_READY_NOT_TRAINED'
    assert len(meta['replay_original'])==178 and len(meta['replay_splice'])==72 and len(meta['bags'])==288
    assert sum(bool(r.get('skipped',False))for r in meta['bags'])==78
    assert meta['new_feature_bags']==210 and meta['held_cases']==78 and meta['source_clips']==6
    assert {r['seed']for r in meta['parent_models']}=={42,43,44}
    original=[];variants=collections.defaultdict(lambda:collections.defaultdict(list));allbags=[]
    def item(r,kind,rate):
        assert r['split']=='train'and r['person']in range(2,8)
        assert sha(r['path'])==r['sha256'];z=np.load(r['path']);x=z['local_full'];y=z['y']
        assert x.shape==(len(y),1160)and np.isfinite(x).all()and y.dtype==np.bool_ and y.any()and(~y).any()
        if 'rank_mask'in z.files:assert z['rank_mask'].all()
        out=dict(row=r,x=x,y=y,kind=kind,rate=rate,identity=(str(r['path']),kind,rate));allbags.append(out);return out
    for r in meta['replay_original']:
        d=item(r,'normal',1.);original.append(d)
        if r['family']in[2,3]:variants[key(r)]['normal'].append(d)
    for r in meta['replay_splice']:
        if not r.get('skipped',False):variants[key(r)]['splice'].append(item(r,'splice',1.))
    for r in meta['bags']:
        assert r['family']in[2,3]and r['base_kind']in['normal','splice']and r['rate']in[.8,1.2]
        if not r.get('skipped',False):variants[key(r)][r['base_kind']].append(item(r,r['base_kind'],r['rate']))
    assert len(original)==178
    assert len(allbags)==460 and len(variants)==72
    for k,v in variants.items():
        assert set(v)=={'normal','splice'}
        for mode in v:
            rates=[a['rate']for a in v[mode]];assert 1.0 in rates and len(rates)==len(set(rates))
    groups={f:[i for i,r in enumerate(original)if r['row']['family']==f]for f in FAMILIES}
    assert {f:len(v)for f,v in groups.items()}=={2:36,3:36,4:34,5:36,7:36}
    return original,variants,groups,allbags

def batch(original,variants,groups,rng,counts):
    xs=[];parts=[];offset=0
    for family in FAMILIES:
        for ix in rng.choice(groups[family],2,replace=False):
            d=original[int(ix)]
            if family in[2,3]:
                v=variants[key(d['row'])];selected=[(v[k][int(rng.integers(len(v[k])))],.5)for k in['normal','splice']]
            else:selected=[(d,1.)]
            for data,weight in selected:
                pi=np.flatnonzero(data['y']);ni=np.flatnonzero(~data['y']);ni=rng.choice(ni,min(len(ni),56),replace=False)
                ids=np.r_[pi,ni];xs.append(data['x'][ids]);parts.append((offset,offset+len(ids),len(pi),FAMILIES.index(family),weight/10))
                offset+=len(ids);counts[data['identity']]+=1
    assert np.isclose(sum(p[-1]for p in parts),1)
    return np.concatenate(xs),parts

def freeze():
    assert not(U/'TRAINING_FROZEN.json').exists();stop()
    meta=read(U/'PREPARATION_INPUTS_COMPLETE.json');old=read(J/'TRAINING_COMPLETE.json')
    assert {r['seed']:r['model_sha256']for r in meta['parent_models']}=={r['seed']:r['model_sha256']for r in old['rows']}
    _,variants,_,allbags=load_bags(meta)
    pins={str(p):sha(p)for p in[Path(__file__),U/'PREPARATION_INPUTS_COMPLETE.json',U/'PREPARATION_FROZEN.json',U/'FROZEN.json',J/'TRAINING_COMPLETE.json',W/'local_rank_core.py']}
    for d in allbags:pins[d['row']['path']]=d['row']['sha256']
    for r in old['rows']:pins[r['model']]=r['model_sha256']
    write(U/'TRAINING_FROZEN.json',dict(at=stamp(),status='FIXED_RECIPE_BEFORE_NEW_TRAINING',seeds=[42,43,44],steps=1200,lr=.001,weight_decay=.001,
      parent='Matching J seed, all weights/normalization copied exactly. New optimizer state; common100 encoder unchanged.',
      loss='Same listwise positive mass. 2sources/family perstep, allpositive+56randomnegative each. Clipgrad2. Fivefamilies equalweight.',
      sampling='Family2/3: each source normal/splice each1/2; choose uniformly among available rate1/.8/1.2 within each branch. Otherfamilies unchanged original replay. Morevariants never increase sourceweight.',
      skipped='A time variant without preserved original witnesses/full-positive is not a rank training bag and is retained in all audit denominators.',
      evaluation_conditions=['rate_head','old_local_full_plus_rate_half'],runtime_scope='Family2/3 only; otherfamily cards/scores copied unchanged. Equal blend coefficient exactly.5, no sweep.',
      gates='OriginalG2/source-level normal/speed/short/cross/shape/direction/absence regression, prior exposedDEV diagnostic only. No promotion from intermediate weaker baseline.',
      no_selection=True,no_private=True,no_final8=True,no_web=True,pins=pins,variant_groups=len(variants),training_bags=len(allbags)))
    print('Rate training frozen; no optimizer execution')

def run():
    assert not(U/'runs').exists()and not(U/'TRAINING_COMPLETE.json').exists();stop()
    cfg=read(U/'TRAINING_FROZEN.json');check(cfg['pins']);meta=read(U/'PREPARATION_INPUTS_COMPLETE.json')
    original,variants,groups,allbags=load_bags(meta);priors={r['seed']:r for r in read(J/'TRAINING_COMPLETE.json')['rows']}
    torch.set_num_threads(4);torch.set_num_interop_threads(1);(U/'runs').mkdir();runs=[]
    for seed in cfg['seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);r=priors[seed]
        state=torch.load(r['model'],map_location='cpu',weights_only=True);m=LocalRankHead(state['mean'].numpy(),state['std'].numpy());m.load_state_dict(state)
        assert all(torch.equal(v,state[k])for k,v in m.state_dict().items())
        opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);counts=collections.Counter();log=[];tic=time.perf_counter();m.train()
        for step in range(cfg['steps']):
            stop();x,parts=batch(original,variants,groups,rng,counts);z=m(torch.tensor(x))
            loss=sum(w*(torch.logsumexp(z[a:b,j],0)-torch.logsumexp(z[a:a+n,j],0))for a,b,n,j,w in parts)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
            if step%300==0 or step==cfg['steps']-1:
                log.append(dict(step=step+1,loss=float(loss.detach())));print('rate rank',seed,step+1,log[-1]['loss'],flush=True)
        m.eval();assert torch.equal(m.mean,state['mean'])and torch.equal(m.std,state['std'])
        dst=U/'runs'/str(seed);dst.mkdir();mp=dst/'model.pt';torch.save(m.state_dict(),mp);pred={};index=[]
        with torch.inference_mode():
            for i,d in enumerate(allbags):
                name=f'bag_{i}';pred[name]=m(torch.tensor(d['x'])).numpy();index.append(dict(key=name,path=d['row']['path'],sha256=d['row']['sha256'],family=d['row']['family'],base_kind=d['kind'],rate=d['rate'],sampled_steps=counts[d['identity']]))
        pp=dst/'fit_predictions.npz';np.savez_compressed(pp,**pred);write(dst/'FIT_PREDICTION_INDEX.json',index)
        record=dict(seed=seed,condition='rate_head',model=str(mp),model_sha256=sha(mp),initial_model=r['model'],initial_model_sha256=r['model_sha256'],normalization_unchanged=True,
          fit_predictions=str(pp),fit_predictions_sha256=sha(pp),seconds=time.perf_counter()-tic,log=log,steps=cfg['steps'],all_training_variants_sampled=all(counts[d['identity']]>0 for d in allbags))
        runs.append(record);write(U/'TRAINING_PROGRESS.json',dict(at=stamp(),completed=len(runs),rows=runs))
    check(cfg['pins']);write(U/'TRAINING_COMPLETE.json',dict(at=stamp(),status='FIXED_RATE_HEAD_TRAINING_COMPLETE_NOT_ADOPTED',rows=runs,encoder_unchanged=True,private_access=False,final8=False,web_changed=False))

if __name__=='__main__':
    try:
        assert len(sys.argv)==2 and sys.argv[1]in['freeze','run-reviewed'];freeze()if sys.argv[1]=='freeze'else run()
    except BaseException:
        (U/f'TRAIN_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
