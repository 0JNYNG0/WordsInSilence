from core_palm import *
from collections import Counter

def main():
    E=UNIT/'evaluation';E.mkdir(exist_ok=True)
    assert not(E/'FROZEN.json').exists();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    meta,z=load_inputs();completed=read(UNIT/'TRAINING_COMPLETE.json')
    rcpath=OPS/'word100_guide_requirement_learning_20261111/requirement_core.py'
    rc=old.load('_palm_runtime_requirement',rcpath);observer=rc.frozen_observer()
    helper=D/'palm_projection_core.py';assert sha(helper)==MEASUREMENT_SHA256
    measure=old.load('_palm_runtime_measurement',helper)
    source=HERE/'evaluation/RESULTS.json';previous=read(source)
    saved={(r['seed'],r['clip_id'],r['kind'],r['card']):r for r in previous['direction_predictions'] if r['condition']=='reencoded_projected_geometry'}
    pins={str(p):sha(p) for p in [Path(__file__),UNIT/'core_palm.py',UNIT/'TRAINING_COMPLETE.json',UNIT/'TRAINING_CONTRACT.json',source,helper,rcpath]}
    write(E/'FROZEN.json',dict(at=stamp(),conditions=['baseline','local_old','reencoded_projected_geometry','palm_progression'],
        input='Same saved82scenario sources and selected local_old intervals. New palm58 measurement and trained head only; no TCN/DTW/proposal/common100 reexecution.',
        immutable='All other axes, handshape, spans, guide association and thresholds copied unchanged.',
        primary_gate='Normal/speed axis2 support no decrease in every seed; reversed/compound needs explicit opposite evidence, not just withheld output. Full G2 still required.',
        fixture_limitation='Legacy aligned_fist also changed MCP/palm location; report unchanged but cannot call it isolated handshape perturbation. New MCP-preserving controls are numerical invariance only, not closed-hand truth.',
        unavailable='Preserved in denominator, state observation_unknown, no neural prediction interpreted.',
        pins=pins,private_access=False,final8=False,web_changed=False))
    models={}
    for r in completed['runs']:
        assert sha(r['model'])==r['model_sha256']
        net=PalmDirection(observer,z['mean'],z['std'],float(z['delta_scale']))
        net.load_state_dict(torch.load(r['model'],map_location='cpu',weights_only=True));net.eval();models[r['seed']]=net
    out=[];pred=[];cache={};source_pins={}
    for r in previous['rows']:
        stop_if_requested();conditions={k:copy.deepcopy(r['conditions'][k]) for k in ['baseline','local_old','reencoded_projected_geometry']}
        cards=copy.deepcopy(r['conditions']['local_old'])
        for c in cards:
            for e in c['evidence']:
                if e['task']!='direction' or e.get('axis')!=2:continue
                oldpred=saved[r['seed'],r['clip_id'],r['kind'],c['card']]
                assert e['span']==oldpred['span']
                path=oldpred['source']
                if path not in cache:
                    assert sha(path)==oldpred['source_sha256'];source_pins[path]=oldpred['source_sha256']
                    with np.load(path,allow_pickle=False) as loaded:cache[path]=(loaded['x'].copy(),loaded['ts'].copy())
                x,ts=cache[path];a,b=e['span']
                spec=next(s for s in rc.FAMILIES[r['clip_id'].split('_')[2]] if s['id']==c['card']);side=spec['hands'][0]
                g,ok,info=measure.geometry(x,ts,a,b,2,side,'L' if side=='R' else 'R')
                if ok:
                    with torch.inference_mode():logits=models[r['seed']](torch.tensor(g[None]))[0];prob=logits.softmax(-1).numpy();lv=logits.tolist()
                else:prob=np.zeros(3);lv=None
                state=rc.direction_verdict(prob,e['desired'],ok)
                e.update(probabilities=prob.tolist(),state=state,model_scope='palm_progression_sign_and_hold')
                pred.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],span=e['span'],desired=e['desired'],eligible=bool(ok),geometry=info,g=g.tolist(),logits=lv,probabilities=prob.tolist(),state=state,source=path,source_sha256=oldpred['source_sha256']))
            states=[e['state'] for e in c['evidence']]
            c['status']='association_unknown' if c['span'] is None else 'difference_evidence' if 'difference' in states or 'opposite_direction' in states else 'supported_features_only' if states and all(v=='supported' for v in states) else 'insufficient_evidence'
        conditions['palm_progression']=cards
        out.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],conditions=conditions))
    summary=[]
    for seed in (42,43,44):
        for condition in out[0]['conditions']:
            for kind in sorted({r['kind'] for r in out}):
                events=[e for r in out if r['seed']==seed and r['kind']==kind for c in r['conditions'][condition] for e in c['evidence'] if e['task']=='direction' and e.get('axis')==2]
                summary.append(dict(seed=seed,condition=condition,kind=kind,n=len(events),states=dict(Counter(e['state'] for e in events))))
    for p,h in {**pins,**source_pins}.items():assert sha(p)==h,p
    write(E/'RESULTS.json',dict(at=stamp(),rows=out,predictions=pred,summary=summary,source_pins=source_pins,web_changed=False))
    for s in summary:
        if s['condition']=='palm_progression' and s['kind'] in ['full_normal','speed075','speed125','main_reversed','right_closed_hand_main','closed_hand_and_reversed','all_hand_missing']:print(s,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:preserve_failure('EVALUATION');raise
