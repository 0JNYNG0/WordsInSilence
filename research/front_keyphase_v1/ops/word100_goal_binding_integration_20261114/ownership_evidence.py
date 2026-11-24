"""Expose competing assignment evidence without deleting a found action.

Two independent action cores may have a chronological order while their wider
comparison windows borrow each other's observed action. Those are two facts.
No expected order, word ID, desired handshape or duration threshold is used.
"""
import copy,itertools

def overlaps_interior(a,b):
    # A shared boundary sample alone is not evidence of conflicting ownership.
    return max(a[0],b[0])<min(a[1],b[1])

def inspect_ownership(cards):
    pairs=[]
    for a,b in itertools.combinations(cards,2):
        if a['family']==b['family']or not a.get('chosen')or not b.get('chosen'):continue
        aw=a['chosen']['span'];bw=b['chosen']['span']
        aa=a['role_confirmation']['matching'];bb=b['role_confirmation']['matching']
        a_inside_b=[copy.deepcopy(e)for e in aa if overlaps_interior(e['span'],bw)]
        b_inside_a=[copy.deepcopy(e)for e in bb if overlaps_interior(e['span'],aw)]
        if a_inside_b and b_inside_a:
            pairs.append(dict(cards=[a['card'],b['card']],candidate_spans=[aw,bw],
              cross_owned_witnesses=[a_inside_b,b_inside_a],state='mutual_assignment_conflict',
              semantic_error=None,source_order_not_rewritten=True))
    ambiguous={s for p in pairs for s in p['cards']}
    return dict(conflicts=pairs,cards=[dict(card=c['card'],
      state='assignment_ambiguous'if c['card']in ambiguous else'no_mutual_conflict_observed',
      action_retained=True,semantic_completion=None)for c in cards],
      policy='Keep full-span shape/direction measurements; do not attribute a correction or completion to a particular clause when ownership conflicts.')
