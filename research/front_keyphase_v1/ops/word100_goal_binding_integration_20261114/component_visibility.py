"""Keep body measurements separate from a claim about the observed hands.

The feature and motion models are unchanged. A body-wrist trajectory is not
hand-position evidence when all required hand landmarks are unavailable.
No word name or expected motion direction is used here.
"""
import copy
import numpy as np

def hand_visibility(x,span,hands,minimum_fraction=.8):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=280:raise ValueError('Native feedback280 required')
    if span is None:return dict(qualified=False,reason='no_observation_span',hands=[])
    a,b=map(int,span)
    if not 0<=a<=b<len(x):raise ValueError('Invalid observed span')
    rows=[]
    for hand in hands:
        if hand not in ('L','R'):raise ValueError('Known anatomical side required')
        root=9 if hand=='L'else 30
        ids=[root,root+5,root+9,root+13,root+17]
        # Visibility channels, not zero-filled coordinates or body-wrist proxy.
        ok=(x[a:b+1,186+np.asarray(ids)]>.5).all(1)
        rows.append(dict(hand=hand,valid_frames=int(ok.sum()),total_frames=len(ok),
                         fraction=float(ok.mean()),qualified=bool(ok.mean()>=minimum_fraction)))
    return dict(qualified=bool(rows)and all(r['qualified']for r in rows),hands=rows,
                minimum_fraction=minimum_fraction,
                reason='observed_hand_position'if rows and all(r['qualified']for r in rows)else'required_hand_not_sufficiently_observed')

def guard_component(component,x,hands):
    out=copy.deepcopy(component);visibility=hand_visibility(x,out['span'],hands)
    out['hand_position_observation']=visibility
    out['measurement_evidence']=copy.deepcopy(component['evidence'])
    for e in out['evidence']:
        if e['task']=='direction'and e.get('state')not in ('sequence_required','association_unknown')and not visibility['qualified']:
            e['measurement_state']=e['state'];e['state']='observation_unknown'
            e['display_reason']='required_hand_not_sufficiently_observed'
    states=[e['state']for e in out['evidence']]
    out['status']='difference_evidence'if any(s in ('difference','opposite_direction')for s in states)else'supported_features_only'if states and all(s=='supported'for s in states)else'insufficient_evidence'
    out['completion']=None
    return out
