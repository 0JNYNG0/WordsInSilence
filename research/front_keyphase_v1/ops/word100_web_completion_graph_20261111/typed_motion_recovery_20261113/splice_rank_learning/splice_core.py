"""Pure provenance/extent contracts. No model, word, expected-order or RGB input."""
import numpy as np


def enumerate_spans(n):
    return np.asarray([(a, b) for a in range(n - 3) for b in range(a + 3, n)], np.int32)


def splice_timeline(x, source_pts, blocks, reverse=False):
    """Preserve within-block real timing; joins use source median positive interval.

    Original PTS and frame indices remain provenance only. Query PTS is explicitly
    synthetic and monotone; channel 279 is seconds, never normalized progression.
    """
    source_pts = np.asarray(source_pts, np.int64)
    order = np.concatenate([np.arange(b['source_start'], b['source_end'] + 1) for b in blocks])
    n = len(source_pts)
    assert np.array_equal(np.sort(order), np.arange(n))
    assert np.all(np.diff(source_pts) > 0)
    cadence = int(np.median(np.diff(source_pts)))
    intervals = np.where(np.diff(order) == 1, np.diff(source_pts[order]), cadence)
    query_pts = np.r_[0, np.cumsum(intervals)].astype(np.int64)
    if reverse:
        order = order[::-1].copy()
        query_pts = query_pts[-1] - query_pts[::-1]
    baseline_index = n - 1 - order if reverse else order.copy()
    out = np.asarray(x)[order].copy()
    out[:, 279] = np.r_[0, np.diff(query_pts).astype(np.float32) / 1e6]
    assert np.array_equal(out[:, :279], x[order, :279])
    assert np.all(np.diff(query_pts) > 0)
    return dict(x=out, timestamp_us=query_pts, source_pts=source_pts[order],
                source_index=order.astype(np.int32), baseline_index=baseline_index.astype(np.int32),
                join_right=np.flatnonzero(np.diff(baseline_index) != 1).astype(np.int32) + 1,
                join_interval_us=cadence)


def extent_targets(source, baseline_index, family_index):
    """Only inherited full extents and explicit join crossings are rank-labelled.

    No absence/shape/direction labels are generated. Inherited witness/unknown
    masks are returned for auditing only and cannot be used as rank negatives.
    """
    m = np.asarray(baseline_index)
    n = len(m)
    spans = enumerate_spans(n)
    a, b = spans.T
    length = b - a + 1
    va = source['valid'][:, family_index].any(1)
    bs, be = source['bands'][:, family_index].T
    pos = (source['mask'][:, family_index] > 0) & (source['y'][:, family_index] > 0)
    neg = (source['mask'][:, family_index] > 0) & (source['y'][:, family_index] == 0)
    def counts(v, aa=a, bb=b):
        cs = np.r_[0, np.cumsum(v)]
        return cs[bb + 1] - cs[aa]
    joins = np.r_[False, np.diff(m) != 1]
    # A join immediately before a is outside this candidate.
    crossing = (counts(joins) - joins[a]) > 0
    eligible = va[m[a]] & va[m[b]] & (counts(va[m]) >= .8 * length)
    original_a, original_b = m[a], m[b]
    source_eligible = va[original_a] & va[original_b]
    positive = (eligible & ~crossing & source_eligible & bs[original_a] & be[original_b]
                & (counts(pos[m]) == pos.sum()) & (counts(neg[m]) == 0))
    negative = eligible & crossing
    rank_mask = positive | negative
    # Partial is an audit classification of retained positive witness coverage;
    # it is not a new progression target and never enters the extent loss.
    partial = eligible & ~crossing & ~positive & (counts(pos[m]) > 0)
    assert not np.any(positive & negative)
    assert not np.any(partial & rank_mask)
    return dict(spans=spans, eligible=eligible, crossing=crossing, positive=positive,
                negative=negative, rank_mask=rank_mask, partial_audit=partial,
                observed_witness_count=counts(pos[m]).astype(np.int32),
                inherited_y=source['y'][m, family_index], inherited_mask=source['mask'][m, family_index],
                inherited_bands=source['bands'][m, family_index], inherited_valid=source['valid'][m, family_index])


def select_bag(targets, rng, max_negatives=512):
    """All positives plus duration-stratified crossings; unknowns remain excluded.

    Every sampled crossing gets a same-length eligible contained audit partner.
    The partner has NO new label and is not placed in the training bag.
    """
    sp = targets['spans']; length = sp[:, 1] - sp[:, 0] + 1
    pi = np.flatnonzero(targets['positive'])
    ni = np.flatnonzero(targets['negative'])
    contained = np.flatnonzero(targets['eligible'] & ~targets['crossing'])
    strata = {int(k): list(rng.permutation(ni[length[ni] == k]))
              for k in np.unique(length[ni]) if np.any(length[contained] == k)}
    picked = []
    while len(picked) < max_negatives and any(strata.values()):
        for k in sorted(strata):
            if strata[k] and len(picked) < max_negatives:
                picked.append(int(strata[k].pop()))
    ni = np.sort(np.asarray(picked, np.int64))
    partners = np.asarray([rng.choice(contained[length[contained] == length[i]]) for i in ni], np.int64)
    return dict(selected=np.r_[pi, ni], positive_count=len(pi), negative_count=len(ni),
                negative_partners=partners, skipped=not len(pi) or not len(ni),
                unmatched_duration_negative_count=int(sum(not np.any(length[contained] == length[i])
                                                          for i in np.flatnonzero(targets['negative']))))
