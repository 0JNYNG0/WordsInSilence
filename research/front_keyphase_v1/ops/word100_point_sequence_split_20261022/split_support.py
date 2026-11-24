"""Research overlay: preserve pointwise result; separately report temporal evidence."""
import numpy as np
EPS=1e-6
MIN_PEERS=5
COMPONENTS=('before','middle','after','before_to_middle','middle_to_after')

def gaps(a,b):
    return np.maximum(np.maximum(a['lo']-b['hi'],b['lo']-a['hi']),0.)

def calibration(anchor,peers):
    valid=np.stack([p['valid'] & anchor['valid'] for p in peers])
    count=valid.sum(0);support=count>=MIN_PEERS
    dist=np.stack([gaps(p,anchor) for p in peers])
    bound=np.max(np.where(valid[...,None],dist,-np.inf),axis=0)
    return {'component_bound':np.where(support[...,None],bound,np.nan),
            'count':count,'support':support}

def evaluate(old,query,anchor,cal):
    expected=old['expected']
    sequence_ok=expected & query['valid'] & anchor['valid'] & cal['support']
    distance=gaps(query,anchor)
    bad=sequence_ok[...,None] & (distance-cal['component_bound']>EPS)
    point_ok=old['point_comparable'];point_bad=old['point_bad']
    # Four exclusive evidence states inside the fixed domain; not correct/wrong verdicts.
    state=np.zeros(expected.shape,np.uint8)
    state[expected & ~point_ok & ~sequence_ok]=1  # neither supported
    state[point_ok & ~sequence_ok]=2            # pointwise only, retain its original result
    state[sequence_ok & ~point_ok]=3            # sequence only; never upgrades old evidence
    state[point_ok & sequence_ok]=4             # both supported
    copykeys=('expected','point_comparable','point_bad','native_expected','native_unknown',
              'path','source_index','query_phase_index','query_span','query_collapsed',
              'context_unavailable','anchor_incomplete','query_incomplete','peer_insufficient',
              'query_span_bad','query_rest_bad')
    out={k:old[k].copy() for k in copykeys}
    out.update(point_unknown=expected & ~point_ok,point_within=point_ok & ~point_bad,
               sequence_comparable=sequence_ok,sequence_unknown=expected & ~sequence_ok,
               evidence_state=state,component_distance=np.where(sequence_ok[...,None],distance,np.nan),
               component_excess=np.where(sequence_ok[...,None],distance-cal['component_bound'],np.nan),
               component_bad=bad,component_within=sequence_ok[...,None] & ~bad,
               level_bad=bad[...,:3].any(-1),change_bad=bad[...,3:].any(-1),
               split_any_bad=bad.any(-1),prior_combined_bad=old['sequence_bad'].copy())
    return out

def selftest():
    shape=(3,2,21);lo=np.zeros(shape+(5,));a={'lo':lo,'hi':lo,'valid':np.ones(shape,bool)}
    peers=[dict(lo=lo.copy(),hi=lo.copy(),valid=a['valid'].copy()) for _ in range(5)]
    for p in peers:p['lo'][...,0]=10;p['hi'][...,0]=10
    cal=calibration(a,peers)
    q=dict(lo=lo.copy(),hi=lo.copy(),valid=a['valid'].copy());q['lo'][...,3]=2;q['hi'][...,3]=2
    o={k:np.zeros(shape,bool) for k in ('point_bad','native_unknown','query_collapsed','context_unavailable','anchor_incomplete','query_incomplete','peer_insufficient','query_span_bad','query_rest_bad','sequence_bad')}
    o.update(expected=np.ones(shape,bool),point_comparable=np.ones(shape,bool),native_expected=np.ones(shape,bool),path=np.array([[0,0],[1,1],[2,2]]),source_index=np.arange(3),query_phase_index=np.zeros((3,3)),query_span=np.zeros((3,2)))
    z=evaluate(o,q,a,cal);assert z['change_bad'].all() and not z['level_bad'].any()
    q['valid'][1,0,0]=False;o['point_bad'][1,0,0]=True
    z=evaluate(o,q,a,cal);assert z['evidence_state'][1,0,0]==2 and z['point_bad'][1,0,0] and not z['component_within'][1,0,0].any()
    assert z['sequence_comparable'][1,0,1]  # neighboring angle unaffected
    z=evaluate(o,q,a,calibration(a,peers[:4]));assert not z['sequence_comparable'].any() and z['point_comparable'].all()
    o['expected'][0]=False;o['point_comparable'][0]=False
    z=evaluate(o,q,a,cal);assert not z['component_bad'][0].any() and not z['evidence_state'][0].any()
    return {'component_cross_relaxation_removed':True,'point_error_retained_when_sequence_unknown':True,
            'neighbor_angle_retained':True,'few_peers_unknown_not_zero':True,'rest_domain_not_expanded':True}
