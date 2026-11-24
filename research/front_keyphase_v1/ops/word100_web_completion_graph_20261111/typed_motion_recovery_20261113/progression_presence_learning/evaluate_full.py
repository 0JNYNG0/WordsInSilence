from presence_core import *
from collections import Counter
import copy,itertools,time,traceback
sys.path.insert(0,str(N))
from shared_boundary_audit import temporal_relation

def orders(cards,ts):
    out=[]
    for i,j in itertools.combinations(range(len(cards)),2):
        if sorted([cards[i]['family'],cards[j]['family']])!=[2,3]:continue
        a,b=[c['chosen']['span']if c['chosen']else None for c in [cards[i],cards[j]]]
        at=[int(ts[k])for k in a]if a else None;bt=[int(ts[k])for k in b]if b else None
        state,detail,overlap=temporal_relation(at,bt)
        out.append(dict(cards=[cards[i]['card'],cards[j]['card']],state=state,detail=detail,overlap_duration_us=overlap))
    return out

def main():
    E=P/'evaluation';E.mkdir(exist_ok=True);assert not(E/'FROZEN.json').exists()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    prior=read(N/'evaluation/COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in prior['predictions']};models={}
    for r in read(P/'TRAINING_COMPLETE.json')['rows']:
        assert sha(r['model'])==r['model_sha256'];st=torch.load(r['model'],map_location='cpu',weights_only=True)
        m=ProgressionHead(st['mean'].numpy(),st['std'].numpy());m.load_state_dict(st);m.eval();models[r['seed']]=m
    cfg=dict(at=stamp(),conditions=['baseline','local_full','progression_separate'],seeds=[42,43,44],scenarios=82,
        selection='Old body gate retained; new binary logits>=0. Original rank and chosen span untouched. Supported original extent may contribute order; unsupported original extent is preserved as raw evidence but not used as a reliable order link.',
        states=['selected_extent_supported','progression_observed_extent_uncertain','no_progression_support','no_body_gate_admission','original_extent_unavailable'],
        interpretation='Support is observed progression only, never educational correctness/completion. No support is not proof of omission. Another supported candidate with unsupported original selection means progression exists but extent is uncertain.',
        family2='copy original card exactly',order='all conditions same positive-duration-overlap sample-PTS arithmetic; preserve raw original order separately',
        source='cached frozen common100 features of exact same82scenarios; no new encoder inference',
        pins={str(p):sha(p)for p in[Path(__file__),P/'presence_core.py',P/'TRAINING_COMPLETE.json',N/'evaluation/COMPARISON.json',N/'shared_boundary_audit.py']},web_changed=False,private_access=False,final8=False)
    write(E/'FROZEN.json',cfg);rows=[];preds=[];cache={};times=[]
    for r in prior['rows']:
        key=(r['seed'],r['clip_id'],r['kind']);pr=pm[key];sid=(r['clip_id'],r['kind'])
        if sid not in cache:
            assert sha(pr['feature_path'])==pr['feature_sha256'] and sha(pr['source'])==pr['source_sha256']
            f=np.load(pr['feature_path']);s=np.load(pr['source']);cache[sid]=(f['x'],{tuple(v):i for i,v in enumerate(f['spans'])},s['ts'])
        ff,lu,ts=cache[sid];assert sha(pr['path'])==pr['sha256'];aa=np.load(pr['path'])
        tick=time.perf_counter()
        with torch.inference_mode():scores=models[r['seed']](torch.tensor(ff)).numpy()
        times.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],seconds=time.perf_counter()-tick,candidates=len(ff),scope='new small head only on cached exhaustive features, not total web latency'))
        conditions={}
        for mode in ['baseline','local_full']:
            c=copy.deepcopy(r['conditions'][mode]);c['original_order']=copy.deepcopy(c['order']);c['order']=orders(c['cards'],ts);conditions[mode]=c
        cards=[];arrays={};diagnostics=[]
        for i,c in enumerate(r['conditions']['local_full']['cards']):
            if c['family']==2:cards.append(copy.deepcopy(c));continue
            spans=aa[f'spans_{i}'];presence=aa[f'presence_{i}'];j=old.F.index(c['family']);ix=np.array([lu[tuple(v)]for v in spans],int);logits=scores[ix,j]
            eligible=presence>=0;support=eligible&(logits>=0);chosen=c['chosen'];ci=None
            if chosen:
                zz=np.flatnonzero((spans==np.array(chosen['span'])).all(1));assert len(zz)==1;ci=int(zz[0]);assert eligible[ci]
            cs=bool(ci is not None and support[ci])
            state='no_body_gate_admission'if not eligible.any()else'no_progression_support'if not support.any()else'original_extent_unavailable'if ci is None else'selected_extent_supported'if cs else'progression_observed_extent_uncertain'
            new=copy.deepcopy(c);new['original_extent']=copy.deepcopy(chosen);new['chosen']=copy.deepcopy(chosen)if cs else None
            new['progression_state']=state;new['progression_observed']=bool(support.any());new['semantic_completion']=None
            if 'truth'in new:
                obs=set(chosen['body_observed'])if cs else set();new['truth']['all_witness']=set(new['truth']['witness'])<=obs;new['truth']['wrong_points']=len(set(new['truth']['wrong_witness'])&obs)
            cards.append(new);arrays[f'spans_{i}']=spans;arrays[f'presence_{i}']=presence;arrays[f'progression_logits_{i}']=logits
            diagnostics.append(dict(card=c['card'],family=c['family'],state=state,eligible_candidates=int(eligible.sum()),progression_candidates=int(support.sum()),original_index=ci,original_extent_supported=cs,original_logit=float(logits[ci])if ci is not None else None))
        conditions['progression_separate']=dict(cards=cards,order=orders(cards,ts),original_order=copy.deepcopy(conditions['local_full']['order']),diagnostics=diagnostics)
        rows.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],conditions=conditions))
        dest=E/'predictions'/str(r['seed'])/f"{r['clip_id']}_{r['kind']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arrays)
        preds.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],path=str(dest),sha256=sha(dest),source=pr['source'],source_sha256=pr['source_sha256'],feature_path=pr['feature_path'],feature_sha256=pr['feature_sha256']))
    summary=[]
    for seed,mode,kind in itertools.product(cfg['seeds'],cfg['conditions'],sorted({r['kind']for r in rows})):
        rr=[r for r in rows if r['seed']==seed and r['kind']==kind];cc=[c for r in rr for c in r['conditions'][mode]['cards']];truth=[c['truth']for c in cc if'truth'in c]
        summary.append(dict(seed=seed,condition=mode,kind=kind,cases=len(rr),cards=len(cc),unsupported_extent=sum(c['chosen']is None for c in cc),known_groups=len(truth),all_witness=sum(t['all_witness']for t in truth),wrong_points=sum(t['wrong_points']for t in truth),order=dict(Counter(o['state']for r in rr for o in r['conditions'][mode]['order'])),progression_states=dict(Counter(c['progression_state']for c in cc if'progression_state'in c))))
    write(E/'COMPARISON.json',dict(at=stamp(),rows=rows,predictions=preds,summary=summary,timing=times,web_changed=False))
    for r in summary:
        if r['condition']=='progression_separate'and r['kind']in ['full_normal','speed075','speed125','dynamic_complete_blocks','omit_final_event','reversed_core_auto_order']:print(r,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:(P/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
