from common import *
import copy,time,traceback
from collections import Counter
def main():
    E=HERE/'evaluation';E.mkdir(exist_ok=True);assert not(E/'RESULTS.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rc=old.load('_tm_eval_req',OPS/'word100_guide_requirement_learning_20261111/requirement_core.py');observer=rc.frozen_observer();models={}
    for r in read(HERE/'TRAINING_COMPLETE.json')['rows']:
        assert sha(r['model'])==r['model_sha256'];m=old.DirectionHead(observer);m.load_state_dict(torch.load(r['model'],weights_only=True,map_location='cpu'));m.eval();models[r['condition'],r['seed']]=m
    shadow=read(old.L/'evaluation/shadow_components/RESULTS.json');comp=read(old.L/'evaluation/COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in comp['predictions']}
    saved={name:{(r['seed'],r['clip_id'],r['kind']):r for r in read(PRIOR/name/'evaluation/RESULTS.json')['rows']}for name in ['boundary_direction_learning','projected_direction_learning']}
    write(E/'FROZEN.json',dict(at=stamp(),conditions=['baseline','local_old','saved_old_context_original','saved_old_context_projected']+read(HERE/'FROZEN.json')['conditions'],runtime_scope='same saved local_full spans and original runtime features. Replace only axis2 direction. No TCN, DTW, proposal or common100 encoder rerun.',pins={str(p):sha(p)for p in[Path(__file__),HERE/'common.py',HERE/'TRAINING_COMPLETE.json',HERE/'TRAINING_CONTRACT.json',old.L/'evaluation/shadow_components/RESULTS.json',old.L/'evaluation/COMPARISON.json',PRIOR/'boundary_direction_learning/evaluation/RESULTS.json',PRIOR/'projected_direction_learning/evaluation/RESULTS.json']},private_access=False,final8=False,web_changed=False))
    cache={};out=[];pred=[]
    for r in shadow['rows']:
        k=(r['seed'],r['clip_id'],r['kind']);p=pm[k];key=k[1:]
        if key not in cache:
            assert sha(p['source'])==p['source_sha256'];z=np.load(p['source']);cache[key]={n:z[n]for n in ['x','fh','ts']}
        d=cache[key];conditions=dict(baseline=r['conditions']['baseline'],local_old=r['conditions']['local_full'])
        for condition in ['saved_old_context_original','saved_old_context_projected']+read(HERE/'FROZEN.json')['conditions']:
            cards=copy.deepcopy(r['conditions']['local_full'])
            for card in cards:
                for event in card['evidence']:
                    if event['task']!='direction'or event.get('axis')!=2:continue
                    if condition.startswith('saved'):
                        folder='projected_direction_learning'if condition.endswith('projected')else'boundary_direction_learning'
                        source=saved[folder][k]['conditions']['local_tuned'];sc=next(c for c in source if c['card']==card['card'])
                        se=next(e for e in sc['evidence']if e['task']=='direction'and e.get('axis')==2)
                        event.clear();event.update(copy.deepcopy(se));continue
                    a,b=event['span'];spec=next(s for s in rc.FAMILIES[r['clip_id'].split('_')[2]]if s['id']==card['card']);side=spec['hands'][0]
                    fn=old.geometry if 'projected'in condition else old.prior.geometry
                    g,ok,info=fn(d['x'],d['ts'],a,b,2,side,'L'if side=='R'else'R');ha,hb=d['fh'][[a,b]];ctx=np.r_[ha,hb,hb-ha]
                    with torch.inference_mode():logits=models[condition,r['seed']](torch.tensor(ctx[None]),torch.tensor(g[None]))[0];prob=logits.softmax(-1).numpy()
                    event.update(probabilities=prob.tolist(),state=rc.direction_verdict(prob,event['desired'],ok),model_scope=condition)
                    pred.append(dict(condition=condition,seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=card['card'],span=[a,b],axis=2,desired=event['desired'],eligible=bool(ok),geometry=info,logits=logits.tolist(),probabilities=prob.tolist(),state=event['state'],source=p['source'],source_sha256=p['source_sha256']))
                states=[e['state']for e in card['evidence']]
                card['status']='association_unknown'if card['span']is None else'difference_evidence'if'difference'in states or'opposite_direction'in states else'supported_features_only'if states and all(v=='supported'for v in states)else'insufficient_evidence'
            conditions[condition]=cards
        out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],conditions=conditions))
    summary=[]
    for seed in [42,43,44]:
        for condition in out[0]['conditions']:
            for kind in sorted({r['kind']for r in out}):
                cards=[c for r in out if r['seed']==seed and r['kind']==kind for c in r['conditions'][condition]];events=[e for c in cards for e in c['evidence']if e['task']=='direction'and e.get('axis')==2]
                summary.append(dict(seed=seed,condition=condition,kind=kind,n=len(events),states=dict(Counter(e['state']for e in events))))
    write(E/'RESULTS.json',dict(at=stamp(),rows=out,direction_predictions=pred,summary=summary,web_changed=False))
    for r in summary:
        if r['condition'].startswith('reencoded')and r['kind']in['full_normal','main_reversed','closed_hand_and_reversed']:print(r,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:(HERE/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
