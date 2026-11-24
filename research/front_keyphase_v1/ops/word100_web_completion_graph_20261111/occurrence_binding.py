"""Choose an observed occurrence without shrinking its action interval.

Correspondence evidence locates an action; it does not prove its shape,
direction, endpoint, number of repetitions, or semantic completion.
"""
import numpy as np


def contiguous_runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return [[int(a), int(b)] for a, b in zip(
        np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1)]


def select_occurrence(phase, role, valid, timestamps, threshold, min_ms,
                      endpoint_child=False):
    phase, role = np.asarray(phase), np.asarray(role)
    valid, timestamps = np.asarray(valid, bool), np.asarray(timestamps)
    assert phase.shape == role.shape == valid.shape == timestamps.shape
    assert np.all(np.diff(timestamps) > 0)
    dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.
    joint = np.sqrt(np.clip(phase, 0, 1) * np.clip(role, 0, 1))
    candidates = []
    for a, b in contiguous_runs((phase >= threshold) & valid):
        duration = (timestamps[b] - timestamps[a] + dt) / 1000
        # A sampled arrival is a child observation, not a separate motion.
        # Do not erase two observed endpoint samples when playback is faster.
        eligible = b > a if endpoint_child else duration >= min_ms
        score = float(joint[a:b + 1].max())
        support = np.flatnonzero(joint[a:b + 1] >= threshold) + a
        candidates.append(dict(span=[a, b], score=score, duration_ms=duration,
                               eligible=bool(eligible),
                               anchor_indices=support.tolist()))
    eligible = [r for r in candidates if r['eligible'] and r['score'] >= threshold]
    selected = max(eligible, key=lambda r: r['score']) if eligible else None
    return dict(span=selected['span'] if selected else None,
                anchors=selected['anchor_indices'] if selected else [],
                candidates=candidates, semantic_completion=None,
                correspondence_only=True)


def union_indices(spans):
    """Return only observed intervals. The gap between them is not evidence."""
    return sorted({i for a, b in spans for i in range(a, b + 1)})
