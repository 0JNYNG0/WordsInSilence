from isolation_net import *
from collections import Counter
import copy,time,traceback
def main():
    E=S/'evaluation';E.mkdir(exist_ok=True);assert not(E/'RESULTS.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    cfg=read(S/'FROZEN.json');rc=old.load('_di_eval_req',OPS/'word100_guide_requirement_learning_20261111/requirement_core.py');observer=rc.frozen_observer();models={}
    for run in read(S/'TRAINING_COMPLETE.json')['rows']:
        assert sha(run['model'])==run['model_sha256'];m=BodyDirection(observer,run['condition'].endswith('symmetric'));m.load_state_dict(torch.load(run['model'],map_location='cpu',weights_only=True));m.eval();models[run['condition'],run['seed']]=m
    parent=read(HERE/'evaluation/RESULTS.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in read(old.L/'evaluation/COMPARISON.json')['predictions']}
    write(E/'FROZEN.json',dict(at=stamp(),same_saved_spans=True,replace_axis2_only=True,conditions=['baseline','local_old','reencoded_projected_geometry']+cfg['conditions'],pins={str(p):sha(p)for p in[Path(__file__),S/'isolation_net.py',S/'FROZEN.json',S/'TRAINING_COMPLETE.json',HERE/'evaluation/RESULTS.json',old.L/'evaluation/COMPARISON.json']}))
    out=[];pred=[];cache={}
    for r in parent['rows']:
        k=r['seed'],r['clip_id'],r['kind'];p=pm[k];key=k[1:]
        if key not in cache:
            assert sha(p['source'])==p['source_sha256'];z=np.load(p['source']);cache[key]={n:z[n]for n in ['x','ts']}
        d=cache[key];conditions={c:r['conditions'][c]for c in ['baseline','local_old','reencoded_projected_geometry']}
        for name in cfg['conditions']:
            cards=copy.deepcopy(r['conditions']['local_old'])
            for card in cards:
                for e in card['evidence']:
                    if e['task']!='direction'or e.get('axis')!=2:continue
                    a,b=e['span'];spec=next(s for s in rc.FAMILIES[r['clip_id'].split('_')[2]]if s['id']==card['card']);side=spec['hands'][0]
                    g,ok,info=old.geometry(d['x'],d['ts'],a,b,2,side,'L'if side=='R'else'R')
                    with torch.inference_mode():lo=models[name,r['seed']](torch.zeros((1,192)),torch.tensor(g[None]))[0];prob=lo.softmax(-1).numpy()
                    e.update(probabilities=prob.tolist(),state=rc.direction_verdict(prob,e['desired'],ok),model_scope=name)
                    pred.append(dict(condition=name,seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=card['card'],span=[a,b],axis=2,desired=e['desired'],eligible=bool(ok),geometry=info,logits=lo.tolist(),probabilities=prob.tolist(),state=e['state'],source=p['source'],source_sha256=p['source_sha256']))
                states=[e['state']for e in card['evidence']]
                card['status']='association_unknown'if card['span']is None else'difference_evidence'if'difference'in states or'opposite_direction'in states else'supported_features_only'if states and all(v=='supported'for v in states)else'insufficient_evidence'
            conditions[name]=cards
        out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],conditions=conditions))
    summary=[]
    for seed in [42,43,44]:
        for c in out[0]['conditions']:
            for kind in sorted({r['kind']for r in out}):
                events=[e for r in out if r['seed']==seed and r['kind']==kind for card in r['conditions'][c]for e in card['evidence']if e['task']=='direction'and e.get('axis')==2]
                summary.append(dict(seed=seed,condition=c,kind=kind,n=len(events),states=dict(Counter(e['state']for e in events))))
    write(E/'RESULTS.json',dict(at=stamp(),rows=out,direction_predictions=pred,summary=summary,web_changed=False))
    for r in summary:
        if r['condition'].startswith('body')and r['kind']in['full_normal','main_reversed','closed_hand_and_reversed','right_closed_hand_main','all_hand_missing']:print(r,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:(S/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
