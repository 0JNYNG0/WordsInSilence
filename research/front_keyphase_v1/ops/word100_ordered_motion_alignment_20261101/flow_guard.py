"""Keep full sequence alignment; local retrieval is corroboration, never truth.

No word, performer, RGB labels, thresholds or model outputs enter this policy.
"""
from collections import defaultdict

def integer(x):
    return type(x) is int and x >= 0

def validate_path(path, n, m):
    assert integer(n) and n > 0 and integer(m) and m > 0
    assert path and len(path) == len({tuple(p) for p in path})
    for p in path:
        assert len(p) == 2 and all(integer(v) for v in p)
        assert p[0] < n and p[1] < m
    assert path[0] == [0, 0] and path[-1] == [n-1, m-1]
    for a,b in zip(path,path[1:]):
        assert (b[0]-a[0], b[1]-a[1]) in {(1,0),(0,1),(1,1)}
    assert {p[0] for p in path} == set(range(n))
    assert {p[1] for p in path} == set(range(m))

def guard(path, n, m, anchors):
    validate_path(path,n,m)
    by=defaultdict(list)
    for q,r in path:by[r].append(q)
    mapping=[dict(reference_frame=r,query_frames=by[r],
                  query_range=[min(by[r]),max(by[r])],
                  display_representative=by[r][(len(by[r])-1)//2],
                  representative_is_display_only=True) for r in range(m)]
    out=[];ids=set()
    for a in anchors:
        sid=a['segment_id'];assert sid not in ids;ids.add(sid)
        lo,hi=a['reference_core'];assert integer(lo) and integer(hi) and lo<=hi<m
        links=[p for p in path if lo<=p[1]<=hi]
        full=[min(p[0] for p in links),max(p[0] for p in links)]
        local=a.get('selected');span=local['query_core'] if local else None
        flags=[];observed=False
        if span is None:relation='local_unavailable';flags.append(relation)
        else:
            assert len(span)==2 and all(integer(v) for v in span) and 0<=span[0]<=span[1]<n
            for g in local['groups'].values():
                assert integer(g['expected']) and type(g['sufficient']) is bool
            expected=[g for g in local['groups'].values() if g['expected']>0]
            observed=bool(expected) and all(g['sufficient'] for g in expected)
            relation=('disjoint' if span[1]<full[0] or full[1]<span[0]
                      else 'contained' if full[0]<=span[0] and span[1]<=full[1]
                      else 'overlap_extends')
            if relation=='disjoint':flags.append('local_global_location_conflict')
            elif relation=='overlap_extends':flags.append('local_global_boundary_disagreement')
            if not observed:flags.append('local_observation_insufficient')
            if a.get('state')=='ambiguous_occurrence':flags.append('local_occurrence_ambiguous')
            elif a.get('state')!='supported_core_candidate':flags.append('local_source_state_not_supported')
        out.append(dict(segment_id=sid,reference_core=[lo,hi],global_query_range=full,
            local_query_range=span,range_relation=relation,local_observation_supported=observed,
            original_local_state=a.get('state'),flags=flags,selected_path=links,
            core_reference_opportunities=hi-lo+1,local_not_used_to_move_global=True))
    ordered=sorted(out,key=lambda a:(a['reference_core'][0],a['reference_core'][1]))
    relationships=[]
    for a,b in zip(ordered,ordered[1:]):
        x,y=a['local_query_range'],b['local_query_range']
        relation=('reference_intervals_overlap' if a['reference_core'][1]>=b['reference_core'][0]
                  else 'local_unavailable' if x is None or y is None
                  else 'forward_separated' if x[1]<y[0]
                  else 'reverse_separated' if y[1]<x[0] else 'overlap')
        relationships.append(dict(first=a['segment_id'],second=b['segment_id'],state=relation))
        if relation in ['reverse_separated','overlap']:
            for z in [a,b]:z['flags'].append('local_order_unresolved')
    for a in out:
        a['flags']=sorted(set(a['flags']))
        a['status']='local_corroboration_unresolved' if a['flags'] else 'window_compatible_not_exact_phase_proof'
        a.update(sign_correct=None,sign_error=None,correction_instruction=None)
    return dict(path=path,query_scope_retained=n,reference_scope_retained=m,
        mapping=mapping,anchors=out,local_relationships=relationships,
        scope='Full path unchanged; compatible intervals do not prove exact stages or KSL correctness.',
        numeric_difference_or_support_recomputed=False,new_alignment=False)
