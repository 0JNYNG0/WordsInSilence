"""New head once on the frozen82 scenarios; reuse old features and gates."""
from pathlib import Path
import json,hashlib,copy,collections,time,itertools
import numpy as np
import torch
from extent_model import ExtentChoice
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111';H=G/'typed_motion_recovery_20261113';N=H/'null_candidate_learning'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not(U/'FULLPATH_COMPARISON.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    parents=read(N/'evaluation/COMPARISON.json');fixed=read(H/'full_bag_rate_rank_learning/evaluation/COMPARISON.json')
    prior={(r['seed'],r['clip_id'],r['kind']):r for r in parents['rows']};latest={(r['seed'],r['clip_id'],r['kind']):r for r in fixed['rows']}
    pm={(r['seed'],r['clip_id'],r['kind']):r for r in parents['predictions']};models={};runs=read(U/'TRAINING_COMPLETE.json')['rows']
    for r in runs:
        assert sha(Path(r['model']))==r['model_sha256'];net=ExtentChoice(torch.load(r['model'],map_location='cpu',weights_only=True));net.eval();models[r['seed']]=net
    out=[];pout=[];timings=[]
    for cid,kind in dict.fromkeys((r['clip_id'],r['kind'])for r in parents['rows']):
        t=time.perf_counter();pr=pm[42,cid,kind]
        assert sha(Path(pr['feature_path']))==pr['feature_sha256'];assert sha(Path(pr['source']))==pr['source_sha256']
        features=np.load(pr['feature_path']);sp=features['spans'];lu={tuple(s):i for i,s in enumerate(sp)};ff=features['x']
        z=np.load(pr['source']);qr=z['z'][:,292:392].reshape(-1,10,10);ts=z['ts'];predfile=U/'fullpath_predictions'/f'{cid}_{kind}.npz';predfile.parent.mkdir(exist_ok=True)
        saved={}
        for seed,net in models.items():
            source=prior[seed,cid,kind];current=copy.deepcopy(latest[seed,cid,kind]['conditions']['full_bag_rate_rank'])
            current['order']=copy.deepcopy(latest[seed,cid,kind]['pts_order']['full_bag_rate_rank'])
            p=pm[seed,cid,kind];assert sha(Path(p['path']))==p['sha256'];pa=np.load(p['path'])
            with torch.inference_mode():scores=net(torch.tensor(ff)).numpy();null=net.null_logits.detach().numpy()
            saved[f'logits_{seed}']=scores;saved[f'null_{seed}']=null;cards=[]
            for i,c in enumerate(source['conditions']['local_null']['cards']):
                if c['family']==2:
                    cards.append(copy.deepcopy(next(q for q in current['cards']if q['card']==c['card'])));continue
                spans=pa[f'spans_{i}'];presence=pa[f'presence_{i}'];ids=np.array([lu[tuple(s)]for s in spans],int);j=[2,3,4,5,7].index(c['family']);axis={3:1,4:2,5:4,7:3}[c['family']]*2
                vals=np.where(presence>=0,scores[ids,j],-np.inf);chosen=None
                if len(vals) and np.isfinite(vals).any() and vals.max()>null[j]:
                    at=int(vals.argmax());a,b=map(int,spans[at]);obs=np.arange(a,b+1);obs=obs[qr[obs,axis,8]>.5]
                    chosen=dict(span=[a,b],source='typed_extent_choice',body_observed=obs.tolist(),hand_qualified=obs[qr[obs,axis,9]>.5].tolist(),score=float(vals[at]),null_logit=float(null[j]),semantic_completion=None)
                card=dict(card=c['card'],family=c['family'],chosen=chosen,semantic_completion=None)
                if 'truth'in c:
                    obs=set(chosen['body_observed'])if chosen else set();card['truth']=dict(c['truth'],all_witness=set(c['truth']['witness'])<=obs,wrong_points=len(set(c['truth']['wrong_witness'])&obs))
                cards.append(card)
            order=[]
            for a,b in itertools.combinations(cards,2):
                if sorted([a['family'],b['family']])!=[2,3]:continue
                first,second=a['chosen'],b['chosen'];state='unobserved';overlap=None
                if first and second:
                    af,al=first['span'];bf,bl=second['span'];overlap=max(0,int(min(ts[al],ts[bl])-max(ts[af],ts[bf])))
                    state='ambiguous_overlap'if overlap>0 else'forward'if ts[al]<=ts[bf]else'reverse'if ts[bl]<=ts[af]else'ambiguous_overlap'
                order.append(dict(cards=[a['card'],b['card']],state=state,overlap_duration_us=overlap))
            out.append(dict(seed=seed,clip_id=cid,kind=kind,conditions=dict(parent_null=source['conditions']['local_null'],fixed_rank=current,typed_extent=dict(cards=cards,order=order))))
        np.savez_compressed(predfile,**saved);pout.append(dict(clip_id=cid,kind=kind,path=str(predfile),sha256=sha(predfile),feature_path=pr['feature_path'],feature_sha256=pr['feature_sha256']))
        timings.append(dict(clip_id=cid,kind=kind,seconds=time.perf_counter()-t));print('new extent fullpath',len(pout),'/82',flush=True)
    assert len(pout)==82 and len(out)==246
    summary=[]
    for seed,kind in itertools.product([42,43,44],sorted({r['kind']for r in out})):
        rr=[r for r in out if r['seed']==seed and r['kind']==kind]
        for condition in ['fixed_rank','typed_extent']:
            cc=[c for r in rr for c in r['conditions'][condition]['cards']];tt=[c['truth']for c in cc if c.get('truth',{}).get('witness')]
            summary.append(dict(seed=seed,kind=kind,condition=condition,cases=len(rr),cards=len(cc),selected=sum(c['chosen']is not None for c in cc),
                directly_labeled_cards=len(tt),all_witnesses=sum(t['all_witness']for t in tt),wrong_points=sum(t['wrong_points']for t in tt),
                order=dict(collections.Counter(o['state']for r in rr for o in r['conditions'][condition]['order']))))
    result=dict(status='NEW_FIXED_HEAD_FULLPATH_EVALUATED_ONCE',rows=out,predictions=pout,summary=summary,timing=timings,
        model_adopted=False,no_rtmw_reextraction=True,no_encoder_reinference=True,no_private=True,no_final8=True,
        timing_scope='offline exhaustive candidate-head comparison, not webcam latency',
        limitation='Candidate extent choice is not whole-sign completion. Saved empty truths excluded, never treated as success/absence.',
        training_complete_sha256=sha(U/'TRAINING_COMPLETE.json'))
    with (U/'FULLPATH_COMPARISON.json').open('x',encoding='utf-8')as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('New fullpath comparison saved. No model adopted.',flush=True)
if __name__=='__main__':main()
