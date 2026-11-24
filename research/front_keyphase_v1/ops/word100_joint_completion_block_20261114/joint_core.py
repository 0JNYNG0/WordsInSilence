"""Combine evidence about the SAME independently located action.

No word-specific numeric thresholds, desired-order search, or candidate removal.
An observed feature sequence is deliberately distinct from whole-sign correctness.
"""
import copy

def joint_sequence(shape, region, available, times, source_frames, association=True):
    n=len(region)
    if not(n==len(available)==len(times)==len(source_frames)):raise ValueError('source correspondence')
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('native increasing time required')
    seq=shape['compatible_states']
    if len(seq)!=n:raise ValueError('shape/region interval differs')
    start=[i for i in range(n)if available[i]and region[i]=='face_front'and seq[i]=='four_open']
    goal=[i for i in range(n)if available[i]and region[i]=='upper_body'and seq[i]=='four_closed']
    pairs=[(a,b)for a in start for b in goal if a<b]
    witnesses=[];positive=False;max_gap=None
    if pairs:
        a,b=min(pairs,key=lambda p:p[1]);visible=[i for i in range(a,b+1)if available[i]]
        max_gap=max((times[j]-times[i]for i,j in zip(visible,visible[1:])),default=None)
        continuous=max_gap is not None and max_gap<=120000 and len(visible)/(b-a+1)>=.8
        witnesses=[a,b]
        positive=continuous and shape['observed_shape_change_to_folded'] and n-1 in goal
    status='joint_goal_unconfirmed'
    observed=[i for i,v in enumerate(available)if v]
    largest_gap=max((times[j]-times[i]for i,j in zip(observed,observed[1:])),default=0)
    if not available[-1]or sum(available)/n<.8 or largest_gap>120000:status='observation_insufficient';positive=False
    elif positive:status='shape_and_body_goal_observed'
    elif goal and not start:status='goal_pose_without_observed_start'
    elif shape['content_status']=='final_configuration_unconfirmed':status='final_configuration_unconfirmed'
    elif start and goal and all(a>=b for a in start for b in goal):status='reverse_feature_sequence'
    if not association:status='action_link_uncertain';positive=False
    return dict(status=status,observed_feature_sequence=positive,
        witness_indices=witnesses,witness_source_frames=[source_frames[i]for i in witnesses],
        witness_times_us=[times[i]for i in witnesses],maximum_gap_us=max_gap,
        endpoint_observed=bool(n-1 in goal),region_states=region,available=available,
        source_frames=source_frames,timestamps_us=times,
        whole_action_complete=None,educational_error=None,display_enabled=False,
        two_dimensional_relation_not_contact=True)

def integrate(saved, extra):
    out=copy.deepcopy(saved);extras={e['card']:e for e in extra};cards=[]
    association={c['card']:c['state']for c in saved['original_association']['cards']}
    for c in saved['original_candidates']:
        name=c['card'];known=association[name]=='role_observations_available'
        linked=next((r for r in saved['linked_observations']if r['card']==name),None)
        obs=copy.deepcopy(linked['observations'])if linked else[]
        differences=copy.deepcopy(linked['attributable_differences'])if linked else[]
        orders=[copy.deepcopy(r)for r in saved['original_orders']if name in r['cards']]
        # Reversed order remains independent from handshape/movement differences.
        reverse=[r for r in orders if r['state']=='reverse']
        unobserved=[e for e in obs if e['state']not in['supported','difference','opposite_direction']]
        state='observed_features_only'
        if not known:state='action_link_uncertain'
        elif differences or reverse:state='attributable_feature_difference'
        elif any(e['state']=='association_unknown'for e in unobserved):state='component_link_uncertain'
        elif any(e['state']=='sequence_required'for e in unobserved):state='unimplemented_sequence_requirement'
        elif unobserved:state='observation_insufficient'
        elif obs and all(e['state']=='supported'for e in obs):state='measured_features_agree'
        ex=extras.get(name)
        if ex and ex.get('joint'):
            # Positive individual features must not conceal unresolved final state.
            j=ex['joint']
            if state not in ['attributable_feature_difference','action_link_uncertain']:
                state='measured_features_and_goal_agree'if j['observed_feature_sequence']else j['status']
        cards.append(dict(card=name,state=state,observations=obs,differences=differences,
            order_evidence=orders,body_relation=ex,whole_action_complete=None,
            show_as_correct_sign=False,web_enabled=False))
    out['joint_cards']=cards;out['all_required_actions_complete']=None
    return out
