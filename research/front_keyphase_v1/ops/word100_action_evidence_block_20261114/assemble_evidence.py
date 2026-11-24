"""Separate observed content, observation limits, and clause ownership.

No target word or expected temporal order participates in this assembly.
Neither endpoint agreement nor an isolated goal posture means action completion.
"""
import copy

def qualify_path(evidence):
    e=copy.deepcopy(evidence)
    old=e['original_states'];new=e['new_configuration_labels'];compatible=e['compatible_states']
    conflicts=[i for i,(a,b)in enumerate(zip(old,new))if a is not None and b is not None and a!=b]
    sequence=[]
    for i,s in enumerate(compatible):
        if s is None:continue
        if not sequence or sequence[-1]['configuration']!=s:
            sequence.append(dict(configuration=s,indices=[i]))
        else:sequence[-1]['indices'].append(i)
    content=e['content_status']
    if content=='shape_maintenance_observed':
        # A close-open excursion may finish at its starting configuration.
        content='same_endpoint_configuration' if len(sequence)<=1 else 'internal_configuration_changes_observed'
    elif content=='closing_seen_final_shape_unconfirmed':
        # A cropped prefix can confuse the learned temporal state. Do not turn
        # that disagreement into an instruction to correct a user's fingers.
        content='final_configuration_unconfirmed'
    elif content=='observed_opening_direction':
        folded=[i for i,s in enumerate(compatible)if s=='four_closed']
        opened=[i for i,s in enumerate(compatible)if s=='four_open']
        if not any(a<b for a in folded for b in opened):content='change_direction_unconfirmed'
    return dict(raw=e,content_status=content,endpoint_disagreement_indices=conflicts,
        observed_configuration_sequence=sequence,sequence_is_not_dense_ground_truth=True,
        association_confirmed=e['association_confirmed'],source_span=e.get('automatic_span'),
        complete_action=None,educational_error=None,show_as_correction=False,
        net_change_is_not_whole_interval_maintenance=True)

def assemble(row):
    out=copy.deepcopy(row)
    bycard={c['card']:c for c in row['original_association']['cards']}
    signals=[]
    for c in row['original_qualified_components']:
        association=bycard[c['card']]['state']
        # Preserve every old observation; attribute a difference only when the
        # independent role linkage exists, never because the target expects it.
        signals.append(dict(card=c['card'],span=c['span'],association=association,
            observations=copy.deepcopy(c['evidence']),
            attributable_differences=[copy.deepcopy(e)for e in c['evidence']
                if e['state']in ['difference','opposite_direction']and association=='role_observations_available'],
            action_complete=None,educational_error=None))
    paths=[]
    for p in row['shape_path']:
        paths.append(dict(card=p['card'],status=p['status'],hands=[qualify_path(e)for e in p.get('hands',[])]))
    out['linked_observations']=signals;out['qualified_shape_paths']=paths
    out['user_feedback_enabled']=False
    return out
