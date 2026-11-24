"""Offline reference agreement. Counts denote whole-reference traversals, not linguistic repetitions."""
import itertools
import numpy as np

SIDES = ('L', 'R')
AXES = ('handshape', 'location', 'movement')
MIN_PEOPLE = 5


def cost_table(q, r, required):
    ids = [0, 1] if required == ['L'] else [2, 3] if required == ['R'] else [0, 1, 2, 3]
    joint = q['valid'][:, None, ids] & r['valid'][None, :, ids]
    dist = np.linalg.norm(np.nan_to_num(q['xy'][:, None, ids] - r['xy'][None, :, ids]), axis=-1)
    return np.where(joint, dist, 1.5).mean(-1), joint.mean(-1)


def reference_evidence(q, r, entries, required, person):
    opts = [(e['score'], e['cycles']) for e in entries if e['available']]
    best = min(opts)[0] if opts else None
    winners = [k for score, k in opts if score == best]
    preferred = winners[0] if len(winners) == 1 else None
    n = len(q['ts'])
    core_ids = np.flatnonzero(r['reference_core'])
    evidence = dict(person=person, preferred=preferred, boundaries={})
    if preferred not in (2, 3) or not len(core_ids):
        return evidence
    before = np.arange(core_ids[0])
    after = np.arange(core_ids[-1] + 1, len(r['ts']))
    # A real return witness must resemble both reference endpoints more than
    # its closest core pose. Absence, a DTW wrap, or a low cost alone is insufficient.
    anchor = np.zeros(n, bool)
    if len(before) and len(after):
        cost, support = cost_table(q, r, required)
        nearest = [ids[np.argmin(cost[:, ids], axis=1)] for ids in (before, after, core_ids)]
        ii = np.arange(n)
        anchor = (support[ii, nearest[0]] >= .5) & (support[ii, nearest[1]] >= .5)
        anchor &= (cost[ii, nearest[0]] < cost[ii, nearest[2]]) & (cost[ii, nearest[1]] < cost[ii, nearest[2]])
    e = entries[preferred - 1]
    path = np.asarray(e['path'], int)
    good = np.asarray(e['good'], bool)
    m = len(r['ts'])
    qi, j = path.T
    for c in range(preferred - 1):
        left = qi[good & (j // m == c) & r['reference_core'][j % m]]
        right = qi[good & (j // m == c + 1) & r['reference_core'][j % m]]
        boundary = np.zeros(n, bool)
        if len(left) and len(right):
            boundary = anchor & (np.arange(n) > left.max()) & (np.arange(n) < right.min())
        evidence['boundaries'][str(c + 1)] = np.flatnonzero(boundary).tolist()
    return evidence


def decide(evidence, n):
    assert len({e['person'] for e in evidence}) == len(evidence), 'duplicate person'
    votes = {k: [e for e in evidence if e['preferred'] == k] for k in (1, 2, 3)}
    present = [k for k, v in votes.items() if v]
    result = dict(count=None, status='COUNT_UNCERTAIN', reason='insufficient_people',
                  preferences={str(k): len(v) for k, v in votes.items()},
                  abstaining_people=sum(e['preferred'] is None for e in evidence),
                  supporting_people=[], boundaries=[])
    if len(present) > 1:
        result['reason'] = 'reference_count_disagreement'
        return result
    if not present or len(votes[present[0]]) < MIN_PEOPLE:
        return result
    k = present[0]
    if k == 1:
        result.update(count=1, status='REFERENCE_SUPPORTED', reason='unopposed_single_traversal',
                      supporting_people=[e['person'] for e in votes[1]])
        return result
    # The SAME five people must support every boundary; do not switch voters.
    for group in itertools.combinations(votes[k], MIN_PEOPLE):
        intersections = []
        for c in range(1, k):
            common = set(range(n))
            for e in group:
                common.intersection_update(e['boundaries'].get(str(c), []))
            intersections.append(sorted(common))
        if all(intersections) and all(max(a) < min(b) for a, b in zip(intersections, intersections[1:])):
            result.update(count=k, status='REFERENCE_SUPPORTED', reason='same_people_observed_return_boundaries',
                          supporting_people=[e['person'] for e in group], boundaries=intersections)
            return result
    result['reason'] = 'return_boundary_not_jointly_supported'
    return result


def combine(projections, decision, n):
    viable = [k for k, p in projections.items() if p['available'] >= MIN_PEOPLE]
    z = np.zeros(n, bool)
    strong_k = [decision['count']] if decision['count'] is not None else viable
    def and_masks(masks):
        return np.logical_and.reduce(masks) if masks else z.copy()
    def or_masks(masks):
        return np.logical_or.reduce(masks) if masks else z.copy()
    # Always retain the envelope of all supported hypotheses. Abstention must
    # not erase a short interval merely because a different count is easier.
    result = dict(core=and_masks([projections[k]['core'] for k in strong_k]),
                  possible_core=or_masks([projections[k]['possible_core'] for k in viable]),
                  expected={}, possible_expected={}, rest={}, semantic_verdict=None, whole_action_pass=None)
    result['outside'] = and_masks([projections[k]['outside'] for k in viable]) & ~result['possible_core']
    result['unknown'] = ~(result['core'] | result['outside'])
    for s in SIDES:
        result['rest'][s] = and_masks([projections[k]['rest'][s] for k in viable])
        result['expected'][s] = {a: and_masks([projections[k]['expected'][s][a] for k in strong_k]) for a in AXES}
        result['possible_expected'][s] = {a: or_masks([projections[k]['possible_expected'][s][a] for k in viable]) for a in AXES}
    return result


def candidate(q, refs, bank, people, required, projections):
    evidence = [reference_evidence(q, r, b, required, p) for r, b, p in zip(refs, bank, people)]
    decision = decide(evidence, len(q['ts']))
    return dict(decision=decision, reference_evidence=evidence,
                projection=combine(projections, decision, len(q['ts'])))
