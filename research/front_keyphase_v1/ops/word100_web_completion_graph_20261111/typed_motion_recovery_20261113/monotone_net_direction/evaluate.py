from core_direction import *
from collections import Counter
import copy

def main():
    E=M/'evaluation';E.mkdir(exist_ok=True);assert not(E/'FROZEN.json').exists();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    source=HERE/'evaluation/RESULTS.json';data=read(source);saved={(r['seed'],r['clip_id'],r['kind'],r['card']):r for r in data['direction_predictions']if r['condition']=='reencoded_projected_geometry'}
    rc=old.load('_monotone_requirement',OPS/'word100_guide_requirement_learning_20261111/requirement_core.py');models={}
    for r in read(M/'TRAINING_COMPLETE.json')['runs']:
        assert sha(r['model'])==r['model_sha256'];m=NetDirection();m.load_state_dict(torch.load(r['model'],weights_only=True,map_location='cpu'));m.eval();models[r['seed']]=m
    write(E/'FROZEN.json',dict(at=stamp(),conditions=['baseline','local_old','reencoded_projected_geometry','monotone_net'],scope='Same recorded body projection geometry and eligibility; axis2 only, original card/span and every other event preserved.',pins={str(p):sha(p)for p in[Path(__file__),M/'core_direction.py',M/'TRAINING_COMPLETE.json',source,OPS/'word100_guide_requirement_learning_20261111/requirement_core.py']},web_changed=False))
    out=[];pred=[]
    for r in data['rows']:
        conditions={k:copy.deepcopy(r['conditions'][k])for k in ['baseline','local_old','reencoded_projected_geometry']};cards=copy.deepcopy(r['conditions']['local_old'])
        for c in cards:
            for e in c['evidence']:
                if e['task']!='direction'or e.get('axis')!=2:continue
                p=saved[r['seed'],r['clip_id'],r['kind'],c['card']];assert p['span']==e['span'];d=p['geometry'].get('signed_change',0.)
                with torch.inference_mode():v=models[r['seed']](torch.tensor([d]))[0];prob=v.softmax(-1).numpy()
                e.update(probabilities=prob.tolist(),state=rc.direction_verdict(prob,e['desired'],p['eligible']),model_scope='monotone_net_displacement_only');pred.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],span=e['span'],delta=d,eligible=p['eligible'],desired=e['desired'],logits=v.tolist(),probabilities=prob.tolist(),state=e['state']))
            states=[e['state']for e in c['evidence']];c['status']='association_unknown'if c['span']is None else'difference_evidence'if'difference'in states or'opposite_direction'in states else'supported_features_only'if states and all(v=='supported'for v in states)else'insufficient_evidence'
        conditions['monotone_net']=cards;out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],conditions=conditions))
    summary=[]
    for seed in [42,43,44]:
        for kind in sorted({r['kind']for r in out}):
            rr=[r for r in pred if r['seed']==seed and r['kind']==kind];summary.append(dict(seed=seed,kind=kind,n=len(rr),states=dict(Counter(r['state']for r in rr))))
    write(E/'RESULTS.json',dict(at=stamp(),rows=out,predictions=pred,summary=summary,web_changed=False))
    for r in summary:
        if r['kind']in['full_normal','speed075','speed125','right_closed_hand_main','main_reversed','closed_hand_and_reversed']:print(r,flush=True)

if __name__=='__main__':main()
