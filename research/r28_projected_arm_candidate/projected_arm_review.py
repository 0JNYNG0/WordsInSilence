"""R28: projected arm geometry as a COMPLEMENTARY location cue.

Reads an existing ``selected_interval_feedback_v1`` report (see
reference_code/phase_feedback.py) plus an explicit start/end reference-sample index
range, and summarises two unsigned 2D projected angles per anatomical side:

  shoulder_angle : angle at own shoulder between (opposite shoulder - own shoulder)
                   and (elbow - own shoulder)
  elbow_angle    : angle at own elbow between (own shoulder - elbow)
                   and (hand wrist - elbow)

The "hand wrist" is the existing ``hands[side].shoulder_position`` coordinate, i.e. a
hand-detector wrist anchor expressed in the same shoulder-origin / clip-median
shoulder-width frame as the body points. It is NOT a pose-model wrist; no detector is
switched and no missing joint is invented.

These are 2D projected interior angles, not anatomical 3D joint flexion. Nothing here
is a verdict: ``accepted`` and ``correction`` stay null, no threshold is fitted, no DTW,
neural inference or interpolation is performed, and the input report is never mutated.
"""
from __future__ import annotations

import math

SCHEMA = "projected_arm_review_v1"
INPUT_SCHEMA = "selected_interval_feedback_v1"
SIDES = ("L", "R")
MEASURES = ("shoulder_angle", "elbow_angle")
EPS = 1e-9

MEASURE_DOC = {
    "shoulder_angle": {
        "definition": "unsigned 2D angle at own shoulder between "
                      "(opposite shoulder - own shoulder) and (elbow - own shoulder)",
        "required_points": ["own_shoulder", "opposite_shoulder", "elbow"],
        "independent_of": ["hand_wrist"],
    },
    "elbow_angle": {
        "definition": "unsigned 2D angle at own elbow between (own shoulder - elbow) "
                      "and (hand-detector wrist - elbow)",
        "required_points": ["own_shoulder", "elbow", "hand_wrist"],
        "independent_of": ["opposite_shoulder"],
    },
}

INTERPRETATION = (
    "2D projected interior angles in the existing shoulder frame. Invariant to image "
    "translation, uniform scale and whole-scene IN-PLANE rotation/reflection, therefore also "
    "blind to those differences: equal angles do NOT imply equal hand location. They are also "
    "unchanged when projected segment lengths change while their projected directions stay "
    "fixed - a conditional invariance, not proportion-freedom and not a certificate of "
    "linguistic equivalence. Out-of-plane yaw foreshortens segments and generally does change "
    "these angles. Complementary evidence only; never a sole location rule and never a "
    "sign-correctness signal."
)


# --------------------------------------------------------------------------- geometry

def _as_point(value):
    """Return ((x, y), 'ok') or (None, 'missing'|'nonfinite').

    Only the first two components are used; they are the projected x, y of the shared
    shoulder frame (x video-right, y down). A depth-like third component, if present, is
    deliberately ignored rather than reinterpreted.
    """
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None, "missing"
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None, "missing"
    if not (math.isfinite(x) and math.isfinite(y)):
        return None, "nonfinite"
    return (x, y), "ok"


def _body_point(points, name, index):
    seq = points.get(name) if isinstance(points, dict) else None
    if not isinstance(seq, (list, tuple)) or index < 0 or index >= len(seq):
        return None, "missing"
    return _as_point(seq[index])


def angle_deg(u, v):
    """Unsigned interior angle in 0..180 deg, or None for a zero-length/nonfinite vector."""
    if not (all(math.isfinite(c) for c in u) and all(math.isfinite(c) for c in v)):
        return None
    if math.hypot(u[0], u[1]) < EPS or math.hypot(v[0], v[1]) < EPS:
        return None
    cross = u[0] * v[1] - u[1] * v[0]
    dot = u[0] * v[0] + u[1] * v[1]
    value = math.degrees(math.atan2(abs(cross), dot))
    return value if math.isfinite(value) else None


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _measure_angle(side, measure, points, index, wrist_value):
    """Angle for one endpoint (one report row), or (None, reason)."""
    other = "R" if side == "L" else "L"
    needed = [("own_shoulder", "shoulder_" + side), ("elbow", "elbow_" + side)]
    if measure == "shoulder_angle":
        needed.insert(1, ("opposite_shoulder", "shoulder_" + other))
    got = {}
    for role, name in needed:
        point, state = _body_point(points, name, index)
        if point is None:
            return None, ("missing_" if state == "missing" else "nonfinite_") + role
        got[role] = point
    if measure == "elbow_angle":
        point, state = _as_point(wrist_value)
        if point is None:
            return None, ("missing_" if state == "missing" else "nonfinite_") + "hand_wrist"
        got["hand_wrist"] = point

    if measure == "shoulder_angle":
        u = _sub(got["opposite_shoulder"], got["own_shoulder"])
        v = _sub(got["elbow"], got["own_shoulder"])
    else:
        u = _sub(got["own_shoulder"], got["elbow"])
        v = _sub(got["hand_wrist"], got["elbow"])
    value = angle_deg(u, v)
    if value is None:
        return None, "degenerate_vector"
    return value, None


# ------------------------------------------------------------------------ aggregation

def _mean(values):
    kept = [v for v in values if v is not None and math.isfinite(v)]
    if not kept:
        return None
    total = math.fsum(kept) / len(kept)
    return total if math.isfinite(total) else None


def _equal_reference_mean(per_sample, key):
    """Mean within each reference sample j, then equal mean across observed j."""
    return _mean([_mean(rec[key]) for rec in per_sample.values()])


def _observed_samples(per_sample, key):
    return sum(1 for rec in per_sample.values() if _mean(rec[key]) is not None)


def _wrist_value(step, side, who):
    hands = step.get("hands")
    hand = hands.get(side) if isinstance(hands, dict) else None
    pair = hand.get("shoulder_position") if isinstance(hand, dict) else None
    return pair.get(who) if isinstance(pair, dict) else None


def _unavailable(reason, reference_range=None):
    return {"schema": SCHEMA, "status": "UNAVAILABLE", "reason": reason,
            "source_schema": INPUT_SCHEMA, "reference_sample_range": reference_range,
            "sides": None, "accepted": None, "correction": None,
            "interpretation": INTERPRETATION}


def _validate_range(reference_range):
    if not isinstance(reference_range, (list, tuple)) or len(reference_range) != 2:
        return None, "invalid_reference_range"
    start, end = reference_range
    if any(type(v) is not int for v in (start, end)):
        return None, "invalid_reference_range"
    if start < 0 or end < start:
        return None, "invalid_reference_range"
    return (start, end), None


def review_projected_arm(report, reference_range):
    """Summarise projected arm angles over the selected path pairs. Pure read-only."""
    if not isinstance(report, dict):
        return _unavailable("invalid_report")
    if report.get("schema") != INPUT_SCHEMA:
        return _unavailable("unsupported_schema")
    if report.get("status") != "AVAILABLE":
        return _unavailable("report_not_available")
    steps = report.get("steps")
    if not isinstance(steps, list):
        return _unavailable("invalid_steps")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("query"), dict) \
            or not isinstance(inputs.get("reference"), dict):
        return _unavailable("missing_inputs")

    bounds, problem = _validate_range(reference_range)
    if problem:
        return _unavailable(problem, None)
    start, end = bounds
    rows = inputs["reference"].get("rows")
    if isinstance(rows, int) and not isinstance(rows, bool) and end >= rows:
        return _unavailable("range_outside_reference", [start, end])

    query_points = inputs["query"].get("body_points") or {}
    reference_points = inputs["reference"].get("body_points") or {}

    selected, malformed = [], 0
    for step in steps:
        if not isinstance(step, dict) or type(step.get("i")) is not int \
                or type(step.get("j")) is not int:
            malformed += 1
            continue
        if start <= step["j"] <= end:
            selected.append(step)
    if not selected:
        return _unavailable("no_selected_steps_in_range", [start, end])

    sides = {}
    for side in SIDES:
        sides[side] = {}
        for measure in MEASURES:
            per_sample = {}
            reasons = {"query": {}, "reference": {}}
            counts = {"query": 0, "reference": 0, "paired": 0}
            for step in selected:
                j = step["j"]
                record = per_sample.setdefault(
                    j, {"query": [], "reference": [], "signed": [], "absolute": []})
                q, q_reason = _measure_angle(side, measure, query_points, step["i"],
                                             _wrist_value(step, side, "query"))
                r, r_reason = _measure_angle(side, measure, reference_points, j,
                                             _wrist_value(step, side, "reference"))
                for who, value, reason in (("query", q, q_reason), ("reference", r, r_reason)):
                    if value is None:
                        reasons[who][reason] = reasons[who].get(reason, 0) + 1
                    else:
                        counts[who] += 1
                        record[who].append(value)
                if q is not None and r is not None:
                    counts["paired"] += 1
                    record["signed"].append(q - r)
                    record["absolute"].append(abs(q - r))

            per_sample_out = [{
                "j": j,
                "query_steps": len(rec["query"]), "reference_steps": len(rec["reference"]),
                "paired_steps": len(rec["signed"]),
                "query_angle_deg": _mean(rec["query"]),
                "reference_angle_deg": _mean(rec["reference"]),
                "signed_difference_deg": _mean(rec["signed"]),
                "absolute_difference_deg": _mean(rec["absolute"]),
            } for j, rec in sorted(per_sample.items())]

            sides[side][measure] = {
                "measure": measure,
                "definition": MEASURE_DOC[measure]["definition"],
                "required_points": list(MEASURE_DOC[measure]["required_points"]),
                "independent_of": list(MEASURE_DOC[measure]["independent_of"]),
                "query_angle_deg_mean": _equal_reference_mean(per_sample, "query"),
                "reference_angle_deg_mean": _equal_reference_mean(per_sample, "reference"),
                "signed_difference_deg_mean": _equal_reference_mean(per_sample, "signed"),
                "absolute_difference_deg_mean": _equal_reference_mean(per_sample, "absolute"),
                "counts": {
                    "steps_in_range": len(selected),
                    "reference_samples_in_range": len(per_sample),
                    "query_observed_steps": counts["query"],
                    "reference_observed_steps": counts["reference"],
                    "paired_observed_steps": counts["paired"],
                    "query_observed_samples": _observed_samples(per_sample, "query"),
                    "reference_observed_samples": _observed_samples(per_sample, "reference"),
                    "paired_observed_samples": _observed_samples(per_sample, "signed"),
                },
                "unavailable_reasons": reasons,
                "per_reference_sample": per_sample_out,
            }

    observed = any(sides[s][m]["counts"]["query_observed_steps"]
                   or sides[s][m]["counts"]["reference_observed_steps"]
                   for s in SIDES for m in MEASURES)
    return {
        "schema": SCHEMA, "status": "AVAILABLE", "reason": None,
        "source_schema": INPUT_SCHEMA,
        "reference_sample_range": [start, end],
        "reference_clip_id": report.get("reference_clip_id"),
        "reference_word_id": report.get("reference_word_id"),
        "sides": sides,
        "any_angle_observed": observed,
        "malformed_steps_skipped": malformed,
        "angle_unit": "degrees of 2D projected interior angles, not anatomical 3D flexion",
        "wrist_source": "hands[side].shoulder_position (hand-detector wrist anchor, "
                        "not a pose-model wrist)",
        "aggregation": "Mean within each reference sample j over its matched query steps, "
                       "then equal mean across observed j. Differences use the paired "
                       "support only; missing values are excluded, never filled as zero. "
                       "Signed differences are plain query-minus-reference, not wrapped.",
        "accepted": None, "correction": None,
        "is_a_verdict": False,
        "interpretation": INTERPRETATION,
    }


# ------------------------------------------------------- synthetic fixture + controls
# Everything below builds small synthetic geometry for the controlled experiments.
# No participant data, no raw records, and none of these numbers is an accuracy.

BONE = {"L": {"upper": (0.5, 0.8660254037844386), "fore": (0.8660254037844386, -0.5)},
        "R": {"upper": (-0.5, 0.8660254037844386), "fore": (-0.8660254037844386, -0.5)}}


def base_scene(upper_len=60.0, fore_len=50.0, half_span=40.0):
    """One raw-pixel frame: shoulders on a horizontal line, both arms nondegenerate."""
    scene = {"nose": (0.0, -60.0), "mouth": (0.0, -40.0)}
    for side, sign in (("L", 1.0), ("R", -1.0)):
        shoulder = (sign * half_span, 0.0)
        u1, u2 = BONE[side]["upper"], BONE[side]["fore"]
        elbow = (shoulder[0] + upper_len * u1[0], shoulder[1] + upper_len * u1[1])
        wrist = (elbow[0] + fore_len * u2[0], elbow[1] + fore_len * u2[1])
        scene["shoulder_" + side] = shoulder
        scene["elbow_" + side] = elbow
        scene["wrist_" + side] = wrist
    return scene


def rebuild_arm(scene, side, upper_len=None, fore_len=None, forearm_rotation_deg=0.0):
    """Coherently recompute elbow and wrist: directions fixed, lengths/forearm angle set."""
    out = dict(scene)
    shoulder = scene["shoulder_" + side]
    elbow0, wrist0 = scene["elbow_" + side], scene["wrist_" + side]
    u1 = _sub(elbow0, shoulder)
    n1 = math.hypot(*u1) or 1.0
    l1 = upper_len if upper_len is not None else n1
    elbow = (shoulder[0] + l1 * u1[0] / n1, shoulder[1] + l1 * u1[1] / n1)
    u2 = _sub(wrist0, elbow0)
    n2 = math.hypot(*u2) or 1.0
    l2 = fore_len if fore_len is not None else n2
    theta = math.radians(forearm_rotation_deg)
    dx, dy = u2[0] / n2, u2[1] / n2
    rx = dx * math.cos(theta) - dy * math.sin(theta)
    ry = dx * math.sin(theta) + dy * math.cos(theta)
    out["elbow_" + side] = elbow
    out["wrist_" + side] = (elbow[0] + l2 * rx, elbow[1] + l2 * ry)
    return out


def transform_scene(scene, scale=1.0, rotation_deg=0.0, reflect=False, translate=(0.0, 0.0)):
    """Whole-scene similarity transform (optionally with reflection) in raw pixels."""
    theta = math.radians(rotation_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    out = {}
    for name, (x, y) in scene.items():
        if reflect:
            x = -x
        rx = scale * (x * cos - y * sin) + translate[0]
        ry = scale * (x * sin + y * cos) + translate[1]
        out[name] = (rx, ry)
    return out


def reflect_arms_across_shoulder_line(scene, sides=SIDES):
    """Mirror elbows/wrists across the line through both shoulders (shoulders are fixed)."""
    sl, sr = scene["shoulder_L"], scene["shoulder_R"]
    axis = _sub(sl, sr)
    norm = math.hypot(*axis) or 1.0
    ax, ay = axis[0] / norm, axis[1] / norm
    origin = ((sl[0] + sr[0]) / 2.0, (sl[1] + sr[1]) / 2.0)
    out = dict(scene)
    for side in sides:
        for part in ("elbow_", "wrist_"):
            dx, dy = _sub(scene[part + side], origin)
            proj = dx * ax + dy * ay
            out[part + side] = (origin[0] + 2 * proj * ax - dx,
                                origin[1] + 2 * proj * ay - dy)
    return out


def normalize_rows(scenes):
    """Shoulder-midpoint origin + clip-median shoulder-width scale, as in the pipeline."""
    widths = sorted(math.hypot(*_sub(s["shoulder_L"], s["shoulder_R"])) for s in scenes)
    mid = len(widths) // 2
    scale = widths[mid] if len(widths) % 2 else (widths[mid - 1] + widths[mid]) / 2.0
    rows = []
    for scene in scenes:
        sl, sr = scene["shoulder_L"], scene["shoulder_R"]
        ox, oy = (sl[0] + sr[0]) / 2.0, (sl[1] + sr[1]) / 2.0
        rows.append({name: [(p[0] - ox) / scale, (p[1] - oy) / scale]
                     for name, p in scene.items()})
    return rows, scale


def build_report(query_scenes, reference_scenes, path, drop=None):
    """A minimal, correctly shaped selected_interval_feedback_v1 report for tests/controls.

    ``drop`` maps ('query'|'reference', row_index, point_name) -> replacement value
    (usually None or a nan/inf pair) so missingness can be exercised.
    """
    drop = drop or {}
    rows = {"query": normalize_rows(query_scenes)[0],
            "reference": normalize_rows(reference_scenes)[0]}
    inputs = {}
    for who, series in rows.items():
        points = {name: [list(row[name]) for row in series]
                  for name in ("nose", "mouth", "shoulder_L", "shoulder_R", "elbow_L", "elbow_R")}
        for (target, index, name), value in drop.items():
            if target == who and name in points:
                points[name][index] = value
        inputs[who] = {"body_points": points, "rows": len(series),
                       "normalization": {"method": "shoulder_midpoint_and_clip_median_width"},
                       "frame_ids": list(range(len(series)))}
    steps = []
    for i, j in path:
        hands = {}
        for side in SIDES:
            q = rows["query"][i]["wrist_" + side]
            r = rows["reference"][j]["wrist_" + side]
            for (target, index, name), value in drop.items():
                if name != "wrist_" + side:
                    continue
                if target == "query" and index == i:
                    q = value
                if target == "reference" and index == j:
                    r = value
            delta = [q[0] - r[0], q[1] - r[1]] if q is not None and r is not None else None
            hands[side] = {
                "shoulder_position": {"query": list(q) if q is not None else None,
                                      "reference": list(r) if r is not None else None,
                                      "delta": delta},
                "nose_position": {"query": None, "reference": None, "delta": None},
            }
        steps.append({"i": int(i), "j": int(j), "hands": hands})
    return {"schema": INPUT_SCHEMA, "status": "AVAILABLE", "steps": steps, "inputs": inputs,
            "reference_clip_id": "synthetic", "reference_word_id": "synthetic",
            "accepted": None, "execution_assessment": "UNCONFIGURED"}


def _angles(report, index=0, side="L"):
    """Convenience: the two query/reference angles of a single step of a built report."""
    step = report["steps"][index]
    out = {}
    for measure in MEASURES:
        for who, key in (("query", "i"), ("reference", "j")):
            value, _ = _measure_angle(side, measure,
                                      report["inputs"][who]["body_points"], step[key],
                                      _wrist_value(step, side, who))
            out[who + "_" + measure] = value
    return out


def _location_delta(report, index=0, side="L"):
    delta = report["steps"][index]["hands"][side]["shoulder_position"]["delta"]
    return None if delta is None else math.hypot(delta[0], delta[1])


def synthetic_controls():
    """Run the four controlled geometry experiments on the synthetic fixture.

    Every number below is synthetic geometry in shoulder-width / degree units. None of
    them is a recognition accuracy, a participant measurement or an acceptance decision.
    """
    reference = [base_scene()]
    query = [rebuild_arm(base_scene(), "L", forearm_rotation_deg=-30.0)]
    path = [(0, 0)]
    plain = build_report(query, reference, path)
    plain_angles, plain_delta = _angles(plain), _location_delta(plain)

    # 1. global similarity invariance, with the baseline renormalised the same way.
    transforms = {
        "translation": dict(translate=(137.0, -52.0)),
        "uniform_scale": dict(scale=2.75),
        "rotation": dict(rotation_deg=37.0),
        "reflection": dict(reflect=True),
        "combined": dict(scale=0.4, rotation_deg=-115.0, reflect=True, translate=(9.0, 3.0)),
    }
    invariance = []
    for name, kwargs in transforms.items():
        moved = build_report([transform_scene(s, **kwargs) for s in query],
                            [transform_scene(s, **kwargs) for s in reference], path)
        angles = _angles(moved)
        invariance.append({
            "transform": name,
            "max_abs_angle_change_deg": max(abs(angles[k] - plain_angles[k])
                                            for k in plain_angles),
            "baseline_location_delta_norm": _location_delta(moved),
            "baseline_location_delta_change": abs(_location_delta(moved) - plain_delta),
        })
    query_only = build_report([transform_scene(s, rotation_deg=25.0) for s in query],
                              reference, path)
    query_only_angles = _angles(query_only)

    # 2. projected bone lengths changed, directions fixed (kinematic sensitivity only).
    longer = [rebuild_arm(rebuild_arm(base_scene(), "L", upper_len=96.0, fore_len=30.0),
                          "L", forearm_rotation_deg=-30.0)]
    length = build_report(longer, reference, path)
    length_angles = _angles(length)

    # 3. forearm rotated about the elbow, bone length fixed.
    rotated = [rebuild_arm(query[0], "L", forearm_rotation_deg=-25.0)]
    forearm = build_report(rotated, reference, path)
    forearm_angles = _angles(forearm)

    # 4. counterexample: equal unsigned angles, different hand location.
    mirrored_ref = [base_scene()]
    mirrored_qry = [reflect_arms_across_shoulder_line(base_scene())]
    mirrored = build_report(mirrored_qry, mirrored_ref, path)
    mirrored_angles = _angles(mirrored)

    return {
        "schema": "projected_arm_synthetic_controls_v1",
        "value_type": "synthetic geometric fixture; NOT recognition accuracy, NOT participant data",
        "fixture": {"reference": "base_scene()", "query": "left forearm rotated -30 deg",
                    "path": [list(p) for p in path], "side": "L",
                    "baseline_reference_angles_deg": {
                        "shoulder": plain_angles["reference_shoulder_angle"],
                        "elbow": plain_angles["reference_elbow_angle"]}},
        "experiments": [
            {"name": "global_similarity_invariance",
             "claim": "translation, uniform scale and whole-scene IN-PLANE rotation/reflection "
                      "leave both projected angles unchanged",
             "caveat": "A mathematical invariance only. It is not evidence that these "
                       "transformations preserve linguistic meaning, and the existing "
                       "shoulder normalisation already absorbs translation and scale: the "
                       "renormalised baseline location is reported next to it so the "
                       "candidate gets no fake advantage.",
             "per_transform": invariance,
             "query_only_rotation_25deg": {
                 "max_abs_angle_change_deg": max(abs(query_only_angles[k] - plain_angles[k])
                                                 for k in plain_angles if k.startswith("query")),
                 "baseline_location_delta_norm": _location_delta(query_only),
                 "baseline_location_delta_change": abs(_location_delta(query_only) - plain_delta),
                 "rotation_kind": "2D in-plane rotation: it models image or torso ROLL. It does "
                                  "not model yaw; out-of-plane yaw foreshortens segments and "
                                  "generally does change these projected angles.",
                 "reading": "Angles ignore a query-only in-plane rotation that the normalised "
                            "location feature does see. That is ambiguous, not automatically "
                            "better: the roll could be camera/torso roll OR a real location "
                            "difference."}},
            {"name": "projected_bone_length_sensitivity",
             "claim": "changing upper-arm and forearm PROJECTED lengths while their PROJECTED "
                      "directions stay fixed leaves both angles unchanged while the hand moves",
             "label": "kinematic sensitivity experiment, NOT a certified acceptable "
                      "sign/body variant: a semantic hand target could have moved",
             "scope_of_invariance": "Conditional on fixed projected directions. It is not a "
                                    "general body-proportion invariance - real proportion or "
                                    "depth differences usually rotate the projected segments "
                                    "too - and it certifies no linguistic equivalence.",
             "upper_len_px": [60.0, 96.0], "fore_len_px": [50.0, 30.0],
             "max_abs_angle_change_deg": max(abs(length_angles[k] - plain_angles[k])
                                             for k in plain_angles),
             "baseline_location_delta_norm": _location_delta(length),
             "baseline_location_delta_change": abs(_location_delta(length) - plain_delta)},
            {"name": "forearm_rotation_about_elbow",
             "claim": "rotating the forearm by 25 deg about the elbow with bone length fixed "
                      "moves the elbow angle by a known amount and leaves the shoulder angle fixed",
             "caveat": "Nondegenerate fixture (elbow angle away from the 0/180 deg folds). No "
                       "universal monotonic angle-vs-rotation relation is claimed.",
             "applied_rotation_deg": 25.0,
             "elbow_angle_change_deg": forearm_angles["query_elbow_angle"]
                                       - plain_angles["query_elbow_angle"],
             "shoulder_angle_change_deg": forearm_angles["query_shoulder_angle"]
                                          - plain_angles["query_shoulder_angle"]},
            {"name": "reflection_counterexample",
             "claim": "unsigned angles coincide while the hand is in a different place",
             "construction": "query arm mirrored across the line through both shoulders",
             "shoulder_angle_difference_deg": mirrored_angles["query_shoulder_angle"]
                                              - mirrored_angles["reference_shoulder_angle"],
             "elbow_angle_difference_deg": mirrored_angles["query_elbow_angle"]
                                           - mirrored_angles["reference_elbow_angle"],
             "baseline_location_delta_norm": _location_delta(mirrored),
             "reading": "Angle-only comparison cannot replace spatial location; the original "
                        "shoulder-frame location difference stays as independent evidence."},
        ],
        "conclusion": "Projected arm angles are a complementary cue. They are rejected as a "
                      "sole location rule: reflection and bone-length families collapse to "
                      "identical angles with different hand targets.",
    }
