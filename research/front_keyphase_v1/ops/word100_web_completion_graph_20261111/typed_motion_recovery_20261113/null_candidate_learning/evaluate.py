from null_net import *
from transfer_local_timefix_core import encode_windows
from collections import Counter
import itertools,copy,time,traceback
def main():
    E=N/'evaluation';E.mkdir(exist_ok=True);assert not(E/'COMPARISON.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    prior=read(old.L/'evaluation/COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in prior['predictions']};rm={(r['seed'],r['clip_id'],r['kind']):r for r in prior['rows']};models={};base=old.encoder()
    for run in read(N/'TRAINING_COMPLETE.json')['rows']:
        assert sha(run['model'])==run['model_sha256'];st=torch.load(run['model'],map_location='cpu',weights_only=True);m=NullRank(old.LocalRankHead(st['rank.mean'].numpy(),st['rank.std'].numpy()));m.load_state_dict(st);m.eval();models[run['seed']]=m
    cfg=dict(at=stamp(),conditions=['baseline','local_full','local_null'],same82source_scenarios=True,seeds=[42,43,44],family2='Exactly copy frozen local_full card; no null supervision available',selection='same original body-window presence gate; candidate raw score vs learned null logit, no added threshold',scope='absence of local progression evidence, not missing sign or completed meaning',pins={str(p):sha(p)for p in[Path(__file__),N/'null_net.py',N/'FROZEN.json',N/'TRAINING_COMPLETE.json',old.L/'evaluation/COMPARISON.json']},private_access=False,final8=False,web_changed=False)
    write(E/'FROZEN.json',cfg);out=[];preds=[];timing=[]
    for cid,kind in dict.fromkeys((r['clip_id'],r['kind'])for r in prior['rows']):
        start=time.perf_counter();pr=pm[42,cid,kind];assert sha(pr['source'])==pr['source_sha256'];z=np.load(pr['source']);qr=z['z'][:,292:392].reshape(-1,10,10);pose=old.normalized_pose(z['x']);old_arrays=np.load(pr['path']);cc=rm[42,cid,kind]['conditions']['local_full']['cards'];ss=[old_arrays[f'spans_{i}']for i in range(len(cc))]
        union=sorted(set(tuple(s)for spans in ss for s in spans));sp=np.asarray(union,np.int32).reshape(-1,2);lu={v:i for i,v in enumerate(union)};ids=[np.array([lu[tuple(s)]for s in a],int)for a in ss]
        ff=np.concatenate([encode_windows(base,pose,qr.reshape(-1,100),sp[a:a+128])for a in range(0,len(sp),128)])if len(sp)else np.empty((0,1160),np.float32)
        assert np.isfinite(ff).all();fp=E/'features'/f'{cid}_{kind}.npz';fp.parent.mkdir(exist_ok=True);np.savez_compressed(fp,x=ff,spans=sp)
        for seed in cfg['seeds']:
            rr=rm[seed,cid,kind];old_arrays=np.load(pm[seed,cid,kind]['path']);conditions={m:rr['conditions'][m]for m in ['baseline','local_full']};cards=[];arrays={}
            with torch.inference_mode():scores=models[seed](torch.tensor(ff)).numpy();null=models[seed].null_logits.detach().numpy()
            for i,c in enumerate(cc):
                if c['family']==2:cards.append(copy.deepcopy(rr['conditions']['local_full']['cards'][i]));continue
                j=old.F.index(c['family']);axis=old.AX[c['family']]*2;spans=ss[i];presence=old_arrays[f'presence_{i}'];logits=scores[ids[i],j];vals=np.where(presence>=0,logits,-np.inf);chosen=None
                if len(vals)and np.isfinite(vals).any()and vals.max()>null[j]:
                    at=int(vals.argmax());a,b=map(int,spans[at]);obs=np.arange(a,b+1);obs=obs[qr[obs,axis,8]>.5];chosen=dict(span=[a,b],source='local_null',body_observed=obs.tolist(),hand_qualified=obs[qr[obs,axis,9]>.5].tolist(),score=float(vals[at]),null_logit=float(null[j]),semantic_completion=None)
                card=dict(card=c['card'],family=c['family'],chosen=chosen,semantic_completion=None)
                if'truth'in c:
                    observed=set(chosen['body_observed'])if chosen else set();card['truth']=dict(c['truth'],all_witness=set(c['truth']['witness'])<=observed,wrong_points=len(set(c['truth']['wrong_witness'])&observed))
                cards.append(card);arrays[f'spans_{i}']=spans;arrays[f'presence_{i}']=presence;arrays[f'logits_{i}']=logits;arrays[f'null_{i}']=np.array(null[j])
            order=[]
            for i,j in itertools.combinations(range(len(cards)),2):
                if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
                a,b=cards[i]['chosen'],cards[j]['chosen'];state='unobserved'if a is None or b is None else'ambiguous_overlap'if max(a['span'][0],b['span'][0])<=min(a['span'][1],b['span'][1])else'forward'if a['span'][1]<b['span'][0]else'reverse';order.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state))
            conditions['local_null']=dict(cards=cards,order=order);out.append(dict(seed=seed,clip_id=cid,kind=kind,conditions=conditions))
            path=E/'predictions'/str(seed)/f'{cid}_{kind}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays);preds.append(dict(pm[seed,cid,kind],path=str(path),sha256=sha(path),feature_path=str(fp),feature_sha256=sha(fp)))
        timing.append(dict(clip_id=cid,kind=kind,seconds=time.perf_counter()-start,scope='research exhaustive candidate feature refresh, not deployed/capture latency'));print('null selection',len(out),'/',len(prior['rows']),flush=True)
    summary=[]
    for seed,condition,kind in itertools.product([42,43,44],cfg['conditions'],sorted({r['kind']for r in out})):
        rr=[r for r in out if r['seed']==seed and r['kind']==kind];cards=[c for r in rr for c in r['conditions'][condition]['cards']];truth=[c['truth']for c in cards if'truth'in c];order=[o['state']for r in rr for o in r['conditions'][condition]['order']]
        summary.append(dict(seed=seed,condition=condition,kind=kind,cases=len(rr),cards=len(cards),missing=sum(c['chosen']is None for c in cards),known_groups=len(truth),all_witness=sum(t['all_witness']for t in truth),wrong_points=sum(t['wrong_points']for t in truth),order=dict(Counter(order))))
    write(E/'COMPARISON.json',dict(at=stamp(),rows=out,predictions=preds,timing=timing,summary=summary,web_changed=False))
    for r in summary:
        if r['condition']=='local_null'and r['kind']in ['full_normal','dynamic_complete_blocks','omit_final_event','reversed_core_auto_order']:print(r,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:(N/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
