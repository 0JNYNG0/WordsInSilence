"""Measure sampled body-relative motion without making correctness decisions.

Inputs must be observed relation values, not interpolated replacements. No word,
expected direction/order, reference index, synthetic edit or model score enters
this module. Masks are explicit booleans supplied by the existing observation
contract. A sampled step is not proof that the path between samples was seen.
Exact nonzero counts are descriptive, including tiny numeric noise; there is no
learned or invented biological motion threshold here.
"""
from copy import deepcopy
import numpy as np


def _runs(mask, offset=0):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [[int(a + offset), int(b + offset)] for a, b in zip(
        np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1)]


def candidate_motion_evidence(values, observed, timestamps_us, a=0, b=None):
    """Return per-relation measurements on inclusive interval ``a..b``.

    values: [T] or [T,D]; each dimension keeps its own units (no mixed-unit norm).
    observed: boolean [T] or [T,D], never a confidence threshold invented here.
    timestamps_us: strictly increasing real capture/sample times [T].

    Adjacent differences never cross a masked/nonfinite sample. Uneven sampling
    intervals are reported verbatim, not declared continuous by a fixed FPS.
    Nothing is interpolated, smoothed, rejected or labelled educationally wrong.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not len(x) or not x.shape[1]:
        raise ValueError('values must have nonempty [T] or [T,D] shape')
    mask = np.asarray(observed)
    if mask.dtype.kind != 'b':
        raise ValueError('observed must be boolean; apply the observation contract upstream')
    if mask.shape == (len(x),):
        mask = np.broadcast_to(mask[:, None], x.shape)
    if mask.shape != x.shape:
        raise ValueError('observed must have shape [T] or [T,D]')
    ts = np.asarray(timestamps_us)
    if ts.shape != (len(x),) or ts.dtype.kind not in 'iuf' or not np.isfinite(ts).all():
        raise ValueError('timestamps_us must be finite numeric [T]')
    if np.any(ts[1:] <= ts[:-1]):
        raise ValueError('timestamps_us must be strictly increasing')
    b = len(x) - 1 if b is None else b
    if not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)) or not 0 <= a <= b < len(x):
        raise ValueError('invalid inclusive candidate interval')
    a, b = int(a), int(b)
    xx, mm, tt = x[a:b + 1], mask[a:b + 1], ts[a:b + 1]
    finite = np.isfinite(xx)
    effective = mm & finite
    dt = np.diff(tt).astype(np.float64)
    dims = []
    for k in range(x.shape[1]):
        good = effective[:, k]
        adjacent = good[:-1] & good[1:]
        starts = np.flatnonzero(adjacent)
        delta = xx[starts + 1, k] - xx[starts, k]
        abs_delta = np.abs(delta)
        variation = float(abs_delta.sum())
        square_sum = float(np.square(delta).sum())
        nonzero = delta != 0
        count = int(good.sum())
        state = ('none' if count == 0 else 'one_sample' if count == 1 else
                 'gapped_samples' if not good.all() else 'all_samples_observed')
        changes = int(nonzero.sum())
        change_pattern = ('no_observed_neighbor_pair' if not len(starts) else
                          'no_change_in_observed_neighbor_pairs' if changes == 0 else
                          'one_observed_change' if changes == 1 else 'multiple_observed_changes')
        biggest = int(np.argmax(abs_delta)) if len(starts) else None
        dims.append(dict(
            dimension=k, observation_state=state, sampled_change_pattern=change_pattern,
            observed_samples=count, observed_indices=(np.flatnonzero(good) + a).tolist(),
            unobserved_indices=(np.flatnonzero(~good) + a).tolist(),
            claimed_observed_nonfinite=int((mm[:, k] & ~finite[:, k]).sum()),
            observed_sample_runs=_runs(good, a),
            observed_neighbor_pairs=[[int(i + a), int(i + a + 1)] for i in starts],
            observed_neighbor_dt_us=dt[starts].tolist(),
            observed_neighbor_delta=delta.tolist(),
            observed_neighbor_velocity_per_second=(delta / (dt[starts] / 1e6)).tolist(),
            neighbor_pair_count=len(starts), exact_nonzero_neighbor_changes=changes,
            endpoint_difference=float(xx[-1, k] - xx[0, k]) if good[0] and good[-1] else None,
            endpoint_difference_is_path_evidence=False,
            connected_signed_change=float(delta.sum()),
            connected_absolute_variation=variation,
            positive_variation=float(np.maximum(delta, 0).sum()),
            negative_variation=float(-np.minimum(delta, 0).sum()),
            largest_neighbor_step=float(delta[biggest]) if biggest is not None else None,
            largest_neighbor_pair=[int(starts[biggest] + a), int(starts[biggest] + a + 1)] if biggest is not None else None,
            largest_step_fraction_of_variation=float(abs_delta[biggest] / variation) if variation else None,
            effective_change_steps=variation * variation / square_sum if square_sum else None,
            sampled_range=float(np.ptp(xx[good, k])) if count else None,
        ))
    return dict(
        version=1, interval=[a, b], samples=b-a+1,
        duration_us=float(tt[-1]-tt[0]), sample_dt_us=dt.tolist(),
        dimensions=dims, interpolation_used=False, inputs_modified=False,
        intersample_path_observed=False, observation_discontinuity=None,
        physical_motion_complete=None, guide_requirement_satisfied=None,
        interpretation='Measurements only. A single sampled change may be real fast motion or a jump; multiple changes do not prove semantic completion.',
    )


def attach_candidate_motion_evidence(candidates, values, observed, timestamps_us):
    """Attach measurements to every candidate, including rejected/short ones.

    Caller chooses a single declared relation/side for the supplied values. This
    wrapper never ranks candidates or modifies interval bounds/eligibility.
    """
    out = []
    for original in candidates:
        record = deepcopy(original)
        span = record.get('span')
        if span is None:
            span = [record['a'], record['b']]
        record['motion_observation_evidence'] = candidate_motion_evidence(
            values, observed, timestamps_us, span[0], span[1])
        out.append(record)
    return out
