"""Keep independently observed events; never choose a desired chronology.

This is an observation ledger, not a claim that a repeated/reversed pair is an
incorrect sign. A pose witness is not a complete movement or physical contact.
"""
import copy


def ordered_event_trace(occurrences, requirements):
    events=sorted(copy.deepcopy(occurrences),key=lambda e:(e['a'],e['b'],e['family']))
    for i,e in enumerate(events):e['event_id']=i
    relations=[]
    for req in requirements:
        fa,fb=req['families']
        aa=[e for e in events if e['family']==fa]
        bb=[e for e in events if e['family']==fb]
        pairs=[]
        for a in aa:
            for b in bb:
                order='forward' if a['b']<b['a'] else 'reverse' if b['b']<a['a'] else 'overlapping'
                pairs.append(dict(events=[a['event_id'],b['event_id']],order=order,
                                  witness_indices=[(a['a']+a['b'])//2,(b['a']+b['b'])//2]))
        kinds={p['order']for p in pairs}
        state='unobserved' if not pairs else 'forward_only' if kinds=={'forward'} else 'reverse_only' if kinds=={'reverse'} else 'mixed_or_overlap'
        relations.append(dict(cards=req['cards'],families=[fa,fb],state=state,pairs=pairs,
            reverse_witnesses=[p for p in pairs if p['order']=='reverse'],
            semantic_order_error=None,whole_action_completion=None))
    return dict(events=events,relations=relations,expected_order_used_for_selection=False,
                discarded_events=0,educational_error=None)


def combine_components(goal_event,direction_events,shape_evidence):
    """Conjoin evidence ONLY for the same native interval, preserving conflicts."""
    a,b=goal_event['span'];own=[m for m in direction_events if m.get('span') and not(m['span'][1]<a or b<m['span'][0])]
    bad_motion=[m for m in own if m['state']=='contradicted']
    bad_shape=[s for s in shape_evidence if any(not(y<a or b<x) for x,y in s.get('contradicted_spans',[]))]
    # A compatible direction elsewhere in the video never validates this event.
    motion_ok=bool(own) and all(m['state']=='supported'for m in own)
    goal=goal_event['evidence']
    if bad_motion or bad_shape:state='component_difference_observed'
    elif goal['status']=='ordered_state_change_observed':state='state_transition_observed'
    elif goal['status']=='observed_before_goal':state='goal_not_observed_in_this_interval'
    else:state='needs_more_evidence'
    return dict(span=[a,b],state=state,goal_state=goal['status'],direction_supported=motion_ok,
        direction_contradictions=[m['span']for m in bad_motion],
        shape_contradictions=[s['feature']for s in bad_shape],
        semantic_completion=None,educational_error=None,display_enabled=False)
