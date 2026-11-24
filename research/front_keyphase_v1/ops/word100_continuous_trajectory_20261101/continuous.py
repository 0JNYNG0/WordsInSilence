"""Continuous polygonal Frechet distance in fixed two-dimensional L-infinity.

Free-space cells are convex intersections of linear coordinate constraints.
Reachability propagates boundary intervals; no correspondence path is selected.
The general free-space method is due to Alt and Godau (1995):
https://webspace.science.uu.nl/~kreve101/asci/ag-cfdbt-95.pdf

Floating reachability proposes a bracket. Final endpoints and epsilon=0 are
checked with exact rational arithmetic on the supplied IEEE float coordinates.
This certifies the stored polygonal curves, not physical motion between samples.
An unobserved coordinate makes the full comparison unavailable: no gap stitching.
Position agreement does not evaluate fist orientation, wrist bend, or handshape.
"""
from fractions import Fraction
import math
import numpy as np

DEFAULT_ATOL = 1e-7
DEFAULT_MAX_ITERATIONS = 60


def _curve(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("Curve coordinates must have shape N x 2")
    return a


def _fraction_curve(a):
    return [tuple(Fraction.from_float(float(v)) for v in row) for row in a]


def _linf(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _interval(start, end, point, epsilon):
    """Closed t interval where ||start+t(end-start)-point||_inf <= eps.

The same algebra accepts float or Fraction operands. Fractions make every
boundary, degenerate segment, and interval intersection exact.
"""
    lo, hi = 0, 1
    for k in range(2):
        slope = end[k] - start[k]
        offset = start[k] - point[k]
        if slope == 0:
            if abs(offset) > epsilon:
                return None
            continue
        a, b = (-epsilon - offset) / slope, (epsilon - offset) / slope
        if a > b:
            a, b = b, a
        lo, hi = max(lo, a), min(hi, b)
        if lo > hi:
            return None
    return lo, hi


def _suffix(interval, minimum):
    if interval is None:
        return None
    lo, hi = max(interval[0], minimum), interval[1]
    return None if lo > hi else (lo, hi)


def _outer_boundary(curve, fixed_point, epsilon):
    intervals, reached_start = [], True
    for i in range(len(curve) - 1):
        free = _interval(curve[i], curve[i + 1], fixed_point, epsilon) if reached_start else None
        reachable = free if free is not None and free[0] <= 0 <= free[1] else None
        intervals.append(reachable)
        reached_start = reachable is not None and reachable[0] <= 1 <= reachable[1]
    return intervals


def _reachable(p, q, epsilon):
    n, m = len(p), len(q)
    if not n or not m:
        return False
    if _linf(p[0], q[0]) > epsilon or _linf(p[-1], q[-1]) > epsilon:
        return False
    if n == 1:
        return all(_linf(p[0], vertex) <= epsilon for vertex in q)
    if m == 1:
        return all(_linf(vertex, q[0]) <= epsilon for vertex in p)
    bottom_outer = _outer_boundary(p, q[0], epsilon)
    left_outer = _outer_boundary(q, p[0], epsilon)
    previous_right = [None] * (m - 1)
    last_top = None
    for i in range(n - 1):
        current_right = [None] * (m - 1)
        previous_top = None
        for j in range(m - 1):
            left = previous_right[j] if i else left_outer[j]
            bottom = previous_top if j else bottom_outer[i]
            right, top = None, None
            if left is not None or bottom is not None:
                free_right = _interval(q[j], q[j + 1], p[i + 1], epsilon)
                free_top = _interval(p[i], p[i + 1], q[j + 1], epsilon)
                # A reachable bottom admits all free right points; otherwise
                # t cannot decrease from the lowest reachable left point.
                right = free_right if bottom is not None else _suffix(free_right, left[0])
                # Symmetrically, a reachable left admits all free top points.
                top = free_top if left is not None else _suffix(free_top, bottom[0])
            current_right[j], previous_top = right, top
        previous_right, last_top = current_right, previous_top
    right = previous_right[-1]
    return bool((right is not None and right[0] <= 1 <= right[1]) or
                (last_top is not None and last_top[0] <= 1 <= last_top[1]))


def decision(query_values, reference_values, epsilon, exact=True):
    """Decide full continuous Frechet <= epsilon; exact binary-input math default.

No masks are accepted by this lower-level method: every coordinate must be
finite. compare() handles observation eligibility without deleting vertices.
"""
    p, q = _curve(query_values), _curve(reference_values)
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("decision requires fully finite observed curves")
    if not isinstance(epsilon, (int, float, np.integer, np.floating, Fraction)) or not math.isfinite(float(epsilon)) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if exact:
        e = epsilon if isinstance(epsilon, Fraction) else Fraction.from_float(float(epsilon))
        return _reachable(_fraction_curve(p), _fraction_curve(q), e)
    return _reachable(p.tolist(), q.tolist(), float(epsilon))


def _float_bound(value, upper):
    """Outward-rounded float bound for an exact nonnegative Fraction."""
    try:
        result = float(value)
    except OverflowError:
        return math.inf if upper else np.finfo(np.float64).max
    if not math.isfinite(result):
        return result
    represented = Fraction.from_float(result)
    if upper and represented < value:
        result = math.nextafter(result, math.inf)
    elif not upper and represented > value:
        result = math.nextafter(result, -math.inf)
    return result


def _common(atol, max_iterations, p, q, pm, qm):
    return dict(numerical_tolerance=atol, max_iterations=max_iterations,
                query_frames=len(p), reference_frames=len(q), dimensions=2,
                query_possible_cells=int(pm.size), reference_possible_cells=int(qm.size),
                query_observed_cells=int(pm.sum()), reference_observed_cells=int(qm.sum()),
                query_missing_cells=int((~pm).sum()), reference_missing_cells=int((~qm).sum()),
                path_claim=False, timestamp_alignment_claim=False,
                original_global_path_replaced=False, input_values_and_masks_unchanged=True,
                missing_values_imputed=False, gap_segments_created=False,
                coordinate_identity="FIXED_COLUMNS_NO_HAND_REASSIGNMENT",
                orientation_or_shape_evaluated=False, handshape_outputs_replaced=False,
                semantic_verdict=None, whole_sign_verdict=None, correction_advice=None,
                position_zero_is_whole_sign_agreement=False,
                certificate_domain="Exact rational interpretation of supplied IEEE float vertices; not unseen physical motion",
                continuous_segment_assumption="Straight interpolation only between consecutive fully observed vertices")


def compare(query_values, reference_values, query_mask, reference_mask,
            atol=DEFAULT_ATOL, max_iterations=DEFAULT_MAX_ITERATIONS):
    """Return certified [lower_bound, upper_bound], not a chosen distance/path.

complete means observation completeness only. certified describes exact endpoint
certificates; converged additionally requires interval width <= atol. A floating
proposal that cannot be certified is reported as a numerical failure, never a
small distance. No semantic threshold is used. Numerical guard = atol/8 per
endpoint after float search to width atol/2, fixed before data evaluation.
"""
    if not isinstance(atol, (int, float)) or isinstance(atol, bool) or not math.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or not 1 <= max_iterations <= 60:
        raise ValueError("max_iterations must be an integer from 1 to 60")
    atol = float(atol)
    p, q = _curve(query_values), _curve(reference_values)
    pm, qm = np.asarray(query_mask), np.asarray(reference_mask)
    if pm.shape != p.shape or qm.shape != q.shape or pm.dtype.kind != "b" or qm.dtype.kind != "b":
        raise ValueError("Masks must be boolean arrays matching their N x 2 coordinates")
    if not np.isfinite(p[pm]).all() or not np.isfinite(q[qm]).all():
        raise ValueError("Observed coordinates must be finite")
    common = _common(atol, max_iterations, p, q, pm, qm)
    complete = bool(len(p) and len(q) and pm.all() and qm.all())
    if not complete:
        return dict(complete=False, status="EMPTY_INPUT" if not len(p) or not len(q) else "INCOMPLETE_OBSERVATION",
                    certified=False, converged=False, lower_bound=None, upper_bound=None,
                    zero_distance_certified=None, zero_distance_decision_exact=False,
                    iterations=0, exact_decision_calls=0, **common)
    ep, eq = _fraction_curve(p), _fraction_curve(q)
    zero = _reachable(ep, eq, Fraction(0))
    exact_calls = 1
    if zero:
        return dict(complete=True, status="ZERO_EXACT_CERTIFIED", certified=True, converged=True,
                    lower_bound=0.0, upper_bound=0.0, interval_width=0.0,
                    zero_distance_certified=True, zero_distance_decision_exact=True,
                    lower_certificate="NONNEGATIVITY", upper_certificate="EXACT_ZERO_REACHABILITY",
                    iterations=0, exact_decision_calls=exact_calls, **common)
    if len(p) == 1 or len(q) == 1:
        distance = max(_linf(a, b) for a in ep for b in eq)
        lo, hi = _float_bound(distance, False), _float_bound(distance, True)
        finite = math.isfinite(lo) and math.isfinite(hi)
        converged = finite and hi - lo <= atol
        return dict(complete=True, status="POINT_CURVE_CERTIFIED" if converged else "FLOAT_PRECISION_LIMIT",
                    certified=finite, converged=converged, lower_bound=lo if finite else None,
                    upper_bound=hi if finite else None, interval_width=hi - lo if finite else None,
                    zero_distance_certified=False, zero_distance_decision_exact=True,
                    lower_certificate="EXACT_POINT_TO_POLYLINE_VERTEX_MAX",
                    upper_certificate="EXACT_POINT_TO_POLYLINE_VERTEX_MAX",
                    position_only_case=True, iterations=0, exact_decision_calls=exact_calls, **common)
    endpoint_lower = max(_linf(ep[0], eq[0]), _linf(ep[-1], eq[-1]))
    # Every point in a segment lies in its vertex bounding box. This all-pairs
    # coordinate bound therefore supplies a valid complete free-space upper bound.
    geometric_upper = max(max(v[k] for v in ep) - min(v[k] for v in eq) for k in range(2))
    geometric_upper = max(geometric_upper, max(max(v[k] for v in eq) - min(v[k] for v in ep) for k in range(2)))
    initial_lo, initial_hi = _float_bound(endpoint_lower, False), _float_bound(geometric_upper, True)
    if not math.isfinite(initial_hi):
        return dict(complete=True, status="FLOAT_RANGE_LIMIT", certified=False, converged=False,
                    lower_bound=None, upper_bound=None, zero_distance_certified=False,
                    zero_distance_decision_exact=True, iterations=0, exact_decision_calls=exact_calls, **common)
    lo, hi, iterations, precision_limited = initial_lo, initial_hi, 0, False
    fp, fq = p.tolist(), q.tolist()
    while hi - lo > atol / 2 and iterations < max_iterations:
        midpoint = lo + (hi - lo) / 2
        if midpoint == lo or midpoint == hi:
            precision_limited = True
            break
        feasible = _reachable(fp, fq, midpoint)
        if feasible:
            hi = midpoint
        else:
            lo = midpoint
        iterations += 1
    # A fixed numerical guard makes equality rounding visible inside the bracket;
    # no geometric feature or label is changed. Exact tests below remain decisive.
    lower = max(initial_lo, lo - atol / 8)
    upper = min(initial_hi, hi + atol / 8)
    elower, eupper = Fraction.from_float(lower), Fraction.from_float(upper)
    if elower <= endpoint_lower:
        lower_ok, lower_kind = True, "EXACT_ENDPOINT_LOWER_BOUND"
    else:
        lower_ok = not _reachable(ep, eq, elower)
        exact_calls += 1
        lower_kind = "EXACT_INFEASIBLE_LOWER_ENDPOINT"
    if eupper >= geometric_upper:
        upper_ok, upper_kind = True, "EXACT_ALL_PAIRS_GEOMETRIC_UPPER_BOUND"
    else:
        upper_ok = _reachable(ep, eq, eupper)
        exact_calls += 1
        upper_kind = "EXACT_FEASIBLE_UPPER_ENDPOINT"
    certified = bool(lower_ok and upper_ok and lower <= upper)
    width = upper - lower
    converged = certified and width <= atol
    status = ("NUMERICAL_CERTIFICATION_FAILED" if not certified else "BRACKET_CERTIFIED" if converged
              else "FLOAT_PRECISION_LIMIT" if precision_limited else "ITERATION_LIMIT")
    return dict(complete=True, status=status, certified=certified, converged=converged,
                lower_bound=lower if certified else None, upper_bound=upper if certified else None,
                interval_width=width if certified else None,
                proposed_bounds=[lower, upper] if not certified else None,
                lower_certificate=lower_kind, upper_certificate=upper_kind,
                lower_certificate_pass=bool(lower_ok), upper_certificate_pass=bool(upper_ok),
                zero_distance_certified=False, zero_distance_decision_exact=True,
                numerical_guard=atol / 8, floating_search_tolerance=atol / 2,
                iterations=iterations, exact_decision_calls=exact_calls, **common)
