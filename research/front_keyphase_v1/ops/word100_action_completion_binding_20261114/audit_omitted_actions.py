"""Explain fixed saved omission outputs with exact source-frame provenance."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
E=O/'word100_web_completion_graph_20261111/typed_motion_recovery_20261113/full_bag_rate_rank_learning/evaluation'
J=read(E/'COMPARISON.json');cfg=read(E/'FROZEN.json');condition=read(U/'PLAN.json')['fixed_candidate']
M=O/'word100_completion_fullpath_20261110/REPLAY_INPUTS.json';raw={(r['clip_id'],r['kind']):r for r in read(M)['rows']}
normal={r['clip_id']:r for r in J['rows'] if r['seed']==42 and r['kind']=='full_normal'}
scenarios={(r['clip_id'],r['kind']):r for r in cfg['scenarios']};rows=[];checked=set();pins={str(p):sha(p) for p in [M,E/'COMPARISON.json',E/'FROZEN.json']}
for r in J['rows']:
    if r['kind']!='omit_final_event':continue
    m=raw[r['clip_id'],r['kind']];base=raw[r['clip_id'],'full_normal'];f=scenarios[r['clip_id'],r['kind']]
    if r['clip_id'] not in checked:
        for p,h in [(m['path'],m['sha256']),(base['path'],base['sha256']),(f['source'],f['source_sha256'])]:
            assert sha(Path(p))==h;pins[p]=h
        with np.load(m['path']) as z,np.load(base['path']) as bz,np.load(f['source']) as feat:
            lookup={int(s):i for i,s in enumerate(base['source_indices'])};idx=[lookup[int(s)] for s in m['source_indices']]
            assert np.array_equal(z['xy_px'],bz['xy_px'][idx],equal_nan=True)
            assert np.array_equal(z['score'],bz['score'][idx],equal_nan=True)
            assert np.array_equal(z['timestamp_us'],feat['ts'])
        assert m['raw_source_sha256']==base['raw_source_sha256'];checked.add(r['clip_id'])
    source_ids=m['source_indices'];prior=normal[r['clip_id']]['conditions'][condition]['cards'];cards=[]
    for c in r['conditions'][condition]['cards']:
        origin=next(q for q in prior if q['card']==c['card']);w=origin.get('truth',{}).get('witness',[])
        own_source=[base['source_indices'][i] for i in w]
        other_source=[base['source_indices'][i] for q in prior if q['card']!=c['card'] for i in q.get('truth',{}).get('witness',[])]
        span=c.get('chosen',{}).get('span') if c.get('chosen') else None
        selected=set(source_ids[span[0]:span[1]+1]) if span else set();retained=set(source_ids)&set(own_source)
        other_hits=sorted(selected&set(other_source))
        if not own_source:state='no_direct_truth_for_target'
        elif not retained and other_hits:state='removed_target_witnesses_but_previous_action_selected'
        elif not retained and span:state='removed_target_witnesses_but_other_interval_selected'
        elif not retained:state='removed_target_witnesses_and_no_selection'
        elif other_hits:state='mixed_action_witnesses_in_selection'
        elif set(own_source)<=selected:state='retained_source_identity_preserved'
        else:state='retained_source_identity_incomplete'
        cards.append(dict(card=c['card'],span=span,state=state,original_witnesses=own_source,
            retained_source_witnesses=sorted(retained),selected_other_action_witnesses=other_hits,
            removed_witnesses=sorted(set(own_source)-set(source_ids)),
            inference_scope='source-edited test; not evidence of natural learner-error accuracy',linguistic_completion=None))
    rows.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],cards=cards))
counts=collections.Counter(c['state'] for r in rows for c in r['cards'])
out=dict(status='OMISSION_SOURCE_IDENTITY_AUDIT',rows=rows,summary=dict(counts),unique_public_inputs=len(checked),
    pins=pins,new_inference=False,new_accuracy=False,model_selection=False,web_changed=False)
with (U/'OMISSION_DIAGNOSTIC.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({'summary':dict(counts),'inputs':len(checked)},ensure_ascii=False))
