"""A candidate boundary and independently observed action support stay separate.

This does not trim, delete or reorder candidates. It reports all intersecting
role observations, then checks their chronology. Expected direction/shape/order
never chooses a candidate or a role witness.
"""
import copy,itertools

def attach_role_core(candidates,occurrences):
    out=copy.deepcopy(candidates)
    for c in out:
        chosen=c.get('chosen');support=[];other=[]
        if chosen:
            a,b=chosen['span']
            for e in occurrences:
                lo,hi=max(a,e['a']),min(b,e['b'])
                if lo>hi:continue
                item=dict(family=e['family'],span=[lo,hi],original_role_span=[e['a'],e['b']])
                (support if e['family']==c['family']else other).append(item)
        c['role_confirmation']=dict(matching=support,other_observed_roles=other,
            state='matching_role_observed'if support else'other_role_observed'if other else'no_role_confirmation',
            candidates_retained=True,educational_error=None,semantic_completion=None)
    return out

def core_order(cards):
    out=[]
    for a,b in itertools.combinations(cards,2):
        if a['family']==b['family']:continue
        aa=a['role_confirmation']['matching'];bb=b['role_confirmation']['matching']
        state='unobserved'
        if aa and bb:
            if max(x['span'][1]for x in aa)<min(x['span'][0]for x in bb):state='forward'
            elif max(x['span'][1]for x in bb)<min(x['span'][0]for x in aa):state='reverse'
            else:state='ambiguous_overlap'
        out.append(dict(cards=[a['card'],b['card']],families=[a['family'],b['family']],state=state,
            witness_spans=[copy.deepcopy(aa),copy.deepcopy(bb)],whole_action_complete=None,
            expected_order_used_for_selection=False))
    return out
