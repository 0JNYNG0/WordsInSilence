"""Keep change direction, endpoint observation and clause ownership separate."""
import copy
import numpy as np

def summarize(change, positions, original_states, available, timestamps_us, association=True):
    pp=np.asarray(positions);v=np.asarray(available,bool);ts=np.asarray(timestamps_us)
    if pp.ndim!=2 or pp.shape[1]!=len(v)or len(ts)!=len(v):raise ValueError('Native sequence correspondence required')
    if len(ts)>1 and not np.all(np.diff(ts)>0):raise ValueError('Native times must increase')
    # Fixed learned label prototypes: open=0, folded=1, visibly curved=.5.
    # Nearest-prototype labels are model outputs, not educational angle limits.
    votes=np.abs(pp[:,:,None]-np.array([0.,1.,.5])).argmin(-1)
    agreed=(votes==votes[:1]).all(0)&v
    observed=[['four_open','four_closed','curved'][int(votes[0,i])]if agreed[i]else None for i in range(len(v))]
    compatible=[observed[i]if observed[i]==original_states[i]else None for i in range(len(v))]
    start=[i for i,s in enumerate(compatible)if s=='four_open'];goal=[i for i,s in enumerate(compatible)if s=='four_closed']
    pairs=[(a,b)for a in start for b in goal if a<b]
    linked=False;witness=[];maximum_gap=None
    if pairs:
        a,b=min(pairs,key=lambda p:p[1]);good=np.flatnonzero(v[a:b+1])+a
        maximum_gap=int(np.diff(ts[good]).max())if len(good)>1 else None
        linked=maximum_gap is not None and maximum_gap<=120000 and v[a:b+1].mean()>=.8
        witness=[a,b]
    if not v.any()or not v[-1]:status='observation_insufficient'
    elif change=='closing'and linked:status='observed_open_to_folded'
    elif change=='opening':status='observed_opening_direction'
    elif change=='closing':status='closing_seen_final_shape_unconfirmed'
    elif change=='shape_maintained':status='shape_maintenance_observed'
    else:status='change_uncertain'
    if pairs and not linked:status='endpoints_seen_intervening_observation_insufficient'
    content_status=status
    if not association:status='action_association_unconfirmed'
    return dict(status=status,content_status=content_status,raw_change=change,
        new_configuration_labels=observed,original_states=copy.deepcopy(original_states),
        compatible_states=compatible,witness_indices=witness,maximum_gap_us=maximum_gap,
        observed_shape_change_to_folded=bool(linked and change=='closing'),
        source_interval_retained=True,minimum_duration=None,whole_action_complete=None,
        educational_error=None,display_enabled=False,association_confirmed=association)

def attach(original,evidence):
    out=copy.deepcopy(original);out['shape_state_path']=copy.deepcopy(evidence);return out
