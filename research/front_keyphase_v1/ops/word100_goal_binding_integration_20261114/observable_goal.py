"""A terminal posture, a visible state change and completion are different facts."""
import copy
import numpy as np

def assess_observed_goal(event,timestamps_us,max_gap_us=120000):
    out=copy.deepcopy(event);a,b=out['span'];ts=np.asarray(timestamps_us)[a:b+1]
    if len(ts)!=len(out['states']) or len(ts)!=len(out['available']):raise ValueError('State/time correspondence')
    if len(ts)>1 and not np.all(np.diff(ts)>0):raise ValueError('Native time must increase')
    known=np.asarray(out['available'],bool)&np.array([s is not None for s in out['states']])
    witnesses=np.asarray(out['witness_indices'],dtype=int)-a
    if len(witnesses) and (min(witnesses)<0 or max(witnesses)>=len(ts)):raise ValueError('Witness outside interval')
    maximum_gap=None;observed_link=False
    if out['evidence']['ordered_transition_observed'] and len(witnesses)>=2:
        lo,hi=int(witnesses[0]),int(witnesses[-1]);good=np.flatnonzero(known & (np.arange(len(ts))>=lo)&(np.arange(len(ts))<=hi))
        maximum_gap=int(np.max(np.diff(ts[good]))) if len(good)>=2 else None
        observed_link=maximum_gap is not None and maximum_gap<=max_gap_us
    coverage=dict(known_fraction=float(known.mean())if len(known)else 0.,maximum_known_gap_us=maximum_gap,
                  max_gap_contract_us=max_gap_us,visible_state_link=observed_link,
                  original_endpoint_evidence=copy.deepcopy(out['evidence']))
    # Preserve the endpoint evidence, but do not silently bridge a long missing
    # interval. A short uncertain gap also never reconstructs finger coordinates.
    if out['evidence']['ordered_transition_observed'] and not observed_link:
        out['evidence'].update(status='endpoint_pair_only',ordered_transition_observed=False,
                               intervening_motion_observed=False)
        out['hint']=dict(status='uncertain',message='앞뒤 손 위치는 보이지만, 그 사이 동작은 충분히 확인되지 않았어요.',
                         semantic_completion=None,educational_error=None,display_enabled=False)
    out['coverage']=coverage
    out['semantic_completion']=None;out['educational_error']=None
    return out
