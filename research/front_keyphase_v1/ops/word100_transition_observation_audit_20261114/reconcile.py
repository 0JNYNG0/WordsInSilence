"""Additive diagnostic only; no unvalidated model output becomes a correction."""
import copy

def reconcile(endpoint_states,change,available,goal=None,association_ok=False):
    known=[(i,s)for i,s in enumerate(endpoint_states)if s is not None and available[i]]
    runs=[]
    for i,s in known:
        if not runs or runs[-1][1]!=s:runs.append((i,s))
    transitions=[]
    for (i,a),(j,b)in zip(runs,runs[1:]):
        if a=='four_open'and b=='four_closed':transitions.append(dict(indices=[i,j],direction='closing'))
        if a=='four_closed'and b=='four_open':transitions.append(dict(indices=[i,j],direction='opening'))
    result=dict(raw_change_prediction=change,endpoint_states=copy.deepcopy(endpoint_states),
        observed_category_transitions=transitions,goal_posture_observed=any(s==goal for _,s in known)if goal else None,
        whole_action_complete=None,educational_error=None,comparison_retained=True,display_enabled=False,
        action_association_confirmed=association_ok)
    if not known or sum(available)<.8*len(available):status='observation_uncertain'
    elif change is None:status='change_uncertain'
    elif len(transitions)>1:status='multiple_category_changes_retained'
    elif transitions and transitions[0]['direction']==change:status='category_and_change_agree'
    elif transitions:status='component_disagreement'
    elif change=='shape_maintained':status='no_category_change_observed'
    else:status='change_without_confirmed_category_change'
    result['component_status']=status
    # A partial bend and an apparent change due to projection can both land in
    # the last state. Do not invent which one happened or label either wrong.
    result['status']=status if association_ok else'action_association_unconfirmed'
    return result

def attach(original,observation):
    out=copy.deepcopy(original);out['research_handshape_change']=copy.deepcopy(observation);return out
