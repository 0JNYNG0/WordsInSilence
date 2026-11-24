"""Describe what an action assignment establishes; never discard the action."""
import copy
from ownership_evidence import inspect_ownership

def scope_associations(cards,requires_temporal):
    ownership=inspect_ownership(cards)
    conflicts={s for p in ownership['conflicts']for s in p['cards']}
    out=[]
    for c in cards:
        observations=c['role_confirmation']['matching']
        if c['card']in conflicts:status='competing_action_assignment'
        elif not observations:status='action_identity_unconfirmed'
        elif requires_temporal.get(c['card'],False)and not any(e['span'][1]>e['span'][0]for e in observations):status='posture_snapshot_only'
        else:status='role_observations_available'
        out.append(dict(card=c['card'],state=status,action_candidate_retained=True,
            comparison_span=None if not c.get('chosen')else copy.deepcopy(c['chosen']['span']),
            role_witnesses=copy.deepcopy(observations),
            permits_whole_completion=False,permits_educational_error=False,
            note='Temporal role samples identify an observed interval only; actual movement, required goal and correctness remain separate.'))
    return dict(cards=out,ownership=ownership)
