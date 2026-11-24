"""Observed body-progression order; never extent repair or completion.

The caller supplies frozen model logits and observation eligibility. Labels,
word identity, expected order and annotation witnesses are not runtime inputs.
All supported points inside each selected candidate are retained. We do not
pick a convenient peak or discard a later repeated motion to force an order.
"""
import numpy as np


def _timestamps(values):
    ts = np.asarray(values)
    if ts.ndim != 1 or not np.issubdtype(ts.dtype, np.integer):
        raise ValueError('Original integer microsecond timestamps required')
    if len(ts) < 2 or not (np.diff(ts) > 0).all():
        raise ValueError('Timestamps must be strictly increasing')
    return ts


def support(timestamp_us, candidate_span, progression_logits, body_observed):
    ts = _timestamps(timestamp_us)
    logits = np.asarray(progression_logits)
    observed = np.asarray(body_observed)
    if logits.shape != ts.shape or observed.shape != ts.shape:
        raise ValueError('Scores and masks must match original frames')
    if not np.isfinite(logits).all() or observed.dtype != np.bool_:
        raise ValueError('Finite scores and an explicit Boolean observation mask required')
    if candidate_span is None:
        return dict(candidate_span=None, candidate_extent_us=None,
                    support_indices=[], support_timestamp_us=[], support_extent_us=None,
                    state='association_unknown', reason='no_candidate', completion=None)
    s = np.asarray(candidate_span)
    if s.shape != (2,) or not np.issubdtype(s.dtype, np.integer):
        raise ValueError('Original integer frame indices required')
    a, b = map(int, s)
    if not 0 <= a <= b < len(ts):
        raise ValueError('Selected candidate is outside the observation')
    indices = np.arange(a, b + 1)
    indices = indices[observed[a:b + 1] & (logits[a:b + 1] >= 0)]
    enough = len(indices) >= 2
    observed_count = int(observed[a:b + 1].sum())
    times = ts[indices].astype(np.int64).tolist()
    return dict(candidate_span=[a, b], candidate_extent_us=[int(ts[a]), int(ts[b])],
                support_indices=indices.tolist(), support_timestamp_us=times,
                support_extent_us=[times[0], times[-1]] if enough else None,
                observed_body_count=observed_count,
                state='observed_progression' if enough else 'body_unknown' if observed_count < 2 else 'insufficient_progression_support',
                reason=None if enough else 'fewer_than_two_body_observations' if observed_count < 2 else 'fewer_than_two_observed_progression_points',
                completion=None)


def extent_relation(first, second):
    """The same strict timestamp ordering applies to either kind of extent."""
    if first is None or second is None:
        return dict(state='unobserved', overlap_duration_us=None)
    a, b = first
    c, d = second
    if b <= c:
        return dict(state='forward', overlap_duration_us=0)
    if d <= a:
        return dict(state='reverse', overlap_duration_us=0)
    return dict(state='ambiguous_overlap', overlap_duration_us=int(min(b, d) - max(a, c)))


def compare(timestamp_us, first_span, second_span, first_logits, second_logits,
            first_body_observed, second_body_observed):
    a = support(timestamp_us, first_span, first_logits, first_body_observed)
    b = support(timestamp_us, second_span, second_logits, second_body_observed)
    return dict(first=a, second=b,
                original_candidate_extent_order=extent_relation(a['candidate_extent_us'], b['candidate_extent_us']),
                observed_body_progression_order=extent_relation(a['support_extent_us'], b['support_extent_us']),
                scope='Observed body-motion evidence only; not full action, handshape, or educational correctness',
                unresolved_reasons=[r['reason'] for r in [a, b] if r['reason'] is not None],
                candidate_spans_unchanged=True, expected_order_used=False, completion=None)
