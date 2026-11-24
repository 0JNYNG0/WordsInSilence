"""Provenance labels for observed source-action coverage, NOT runtime inference.

No word IDs, duration threshold, expected class order or coordinate-distance
threshold appears in this evaluator. Missing measurements cannot imply omission.
"""
def evaluate_trace(group, events, source_indices, masked_source_frames=(), transform='original'):
    ids=list(map(int,source_indices)); masked=set(masked_source_frames)
    selected=[e for e in events if e['event_id'] in group['event_ids']]
    assert len(selected)==len(group['event_ids']) and ids
    start=set(range(group['onset'][0],group['onset'][1]+1))
    finish=set(range(group['offset'][0],group['offset'][1]+1))
    progress=set(k for e in selected for k in e['progression_witness_frames'])
    foreign=set(k for e in events if e['event_id'] not in group['event_ids'] for k in e['progression_witness_frames'])
    present=set(ids); visible=present-masked
    unique=[v for i,v in enumerate(ids) if i==0 or v!=ids[i-1]]
    order=('forward' if all(a<b for a,b in zip(unique,unique[1:])) else
           'reversed' if all(a>b for a,b in zip(unique,unique[1:])) else 'mixed') if len(unique)>1 else 'stationary_source_frame'
    coverage={k:dict(required=sorted(s),present=sorted(s&present),visible=sorted(s&visible),
        masked=sorted(s&present&masked),not_in_input=sorted(s-present))
        for k,s in [('start',start),('progress',progress),('finish',finish)]}
    missing=(start|progress|finish)-present
    hidden=(start|progress|finish)&present&masked
    extent=(min(ids)<=group['onset'][0] and max(ids)>=group['offset'][1])
    if hidden:state='observation_unknown'
    elif order!='forward':state='source_order_changed'
    elif transform=='endpoint_only':state='progression_removed_between_endpoints'
    elif not missing:state='full_reviewed_source_trace'
    elif transform in ('subsample_every2','duplicate_each2'):state='sampled_witnesses_unavailable'
    elif progress&present and not extent:state='partial_reviewed_source_trace'
    elif not progress&present:state='no_target_progression_witness_in_input'
    else:state='incomplete_source_evidence'
    return dict(source_trace_state=state,anchors=coverage,source_order=order,
        foreign_progression_points=sorted(foreign&visible),source_group_id=group['group_id'],
        linguistic_completion=None,educational_error=None,
        source_extent_label_only=True,model_prediction=False)
