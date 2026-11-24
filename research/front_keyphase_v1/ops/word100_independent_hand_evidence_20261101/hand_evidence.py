"""Independent projected hand measurements on an unchanged supplied path.

No model, alignment, position score, word, phase, or guide is used. All original
axis/feature opportunities remain. Finger internal angles are stored half-turns
(angle/pi); degree conversion is exactly *180, not circular wrapping. Hand-axis
and hand-to-forearm differences use a fixed [-pi, pi) wrap. Only absolute
differences are averaged: signed circular differences can cancel at +/-pi.

Source definition: word100_guide_video_feedback_20261028/motion_observer.py.
Hand axis is projected hand wrist0 -> middle MCP9; forearm uses a separate pose
estimator's elbow -> wrist. Neither is a palm normal or validated 3D wrist angle.
"""
from collections import Counter
import numpy as np

NORM_MIN = 1e-8
FINGERS = ["thumb_mcp", "thumb_ip"] + [f + "_" + joint
    for f in ("index", "middle", "ring", "little") for joint in ("mcp", "pip", "dip")]


def feature_names():
    names = []
    for hand in ("L", "R"):
        names += [hand + "_" + item for item in ("hand_wrist_x", "hand_wrist_y", "pose_wrist_x", "pose_wrist_y",
                  "elbow_x", "elbow_y", "index_tip_x", "index_tip_y", "palm_center_x", "palm_center_y", "elbow_angle")]
        names += [hand + "_" + item + "_angle" for item in FINGERS]
        names += [hand + "_palm_axis_x", hand + "_palm_axis_y", hand + "_palm_width"]
    names += ["hands_wrist_delta_x", "hands_wrist_delta_y", "hands_wrist_distance", "L_wrist_estimator_distance", "R_wrist_estimator_distance"]
    for hand in ("L", "R"):
        names += [hand + "_wrist_" + item for item in ("velocity_x", "velocity_y", "motion_direction_x", "motion_direction_y")]
    return names


def wrap_radians(angle):
    """Canonical half-open range; exact antipodes use -pi, with sign ambiguity."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _input(a):
    x, mask = np.asarray(a["x"], dtype=np.float64), np.asarray(a["mask"])
    if x.ndim != 2 or x.shape[1] != 69 or mask.shape != x.shape or mask.dtype.kind != "b":
        raise ValueError("Expected N x 69 values and equally shaped boolean masks")
    if not np.isfinite(x[mask]).all():
        raise ValueError("Observed source features must be finite")
    for key in ("names", "feature_names"):
        if key in a and list(a[key]) != feature_names():
            raise ValueError("Source feature identity/order differs from the fixed 69-column schema")
    frames, pts = np.asarray(a["frame_id"]), np.asarray(a["timestamp_us"])
    for key, values in (("frame_id", frames), ("timestamp_us", pts)):
        if values.shape != (len(x),) or values.dtype.kind not in "iu":
            raise ValueError(key + " must match the original frame axis as integers")
        if len(values) > 1 and not np.all(values[1:] > values[:-1]):
            raise ValueError(key + " must increase; no source frame reordering is permitted")
    for offset in (0, 28):
        values = x[:, offset + 11:offset + 25]
        known = mask[:, offset + 11:offset + 25]
        if np.any((values[known] < 0) | (values[known] > 1)):
            raise ValueError("Observed finger internal angles must be half-turn values in [0,1]; never clip invalid input")
    return dict(x=x, mask=mask, frame_id=frames, timestamp_us=pts)


def _nullable(values, mask):
    return [[float(value) if known else None for value, known in zip(row, valid)]
            for row, valid in zip(values, mask)]


def _measurements(a):
    x, m = a["x"], a["mask"]
    groups = {}
    for hand, off in (("L", 0), ("R", 28)):
        angle_columns = list(range(off + 11, off + 25))
        known = m[:, angle_columns].copy()
        # Unknown placeholders are never used in derived arithmetic.
        angle_values = np.where(known, x[:, angle_columns], 0) * 180
        groups[hand + "_finger_angles14"] = dict(values=angle_values, observed=known,
            labels=[hand + "_" + name + "_angle" for name in FINGERS], circular=False,
            columns=angle_columns, input_mask=m[:, angle_columns],
            own_reason=np.where(known, "OBSERVED", "SOURCE_ANGLE_UNOBSERVED"),
            definition="Independent projected triplet internal angles, original half_turn * 180 degrees")
        axis_cols, forearm_cols = [off + 25, off + 26], [off + 2, off + 3, off + 4, off + 5]
        axis_source = m[:, axis_cols].all(axis=1)
        axis = np.where(m[:, axis_cols], x[:, axis_cols], 0)
        axis_norm = np.linalg.norm(axis, axis=1)
        axis_valid = axis_source & np.isfinite(axis_norm) & (axis_norm > NORM_MIN)
        heading = wrap_radians(np.arctan2(axis[:, 1], axis[:, 0]))
        axis_reason = np.where(~axis_source, "HAND_AXIS_SOURCE_UNOBSERVED",
                               np.where(~axis_valid, "DEGENERATE_HAND_AXIS", "OBSERVED"))
        groups[hand + "_hand_axis2d"] = dict(values=np.rad2deg(heading)[:, None], observed=axis_valid[:, None],
            labels=[hand + "_hand_axis_heading_2d"], circular=True, columns=axis_cols,
            input_mask=m[:, axis_cols], own_reason=axis_reason[:, None],
            definition="atan2(y,x) of projected hand wrist0 -> middle MCP9, not a palm normal",
            diagnostics=dict(hand_axis_norm=_nullable(axis_norm[:, None], axis_source[:, None])))
        forearm_source = m[:, forearm_cols].all(axis=1)
        source_positions = np.where(m[:, forearm_cols], x[:, forearm_cols], 0)
        forearm = source_positions[:, :2] - source_positions[:, 2:]
        forearm_norm = np.linalg.norm(forearm, axis=1)
        forearm_valid = forearm_source & np.isfinite(forearm_norm) & (forearm_norm > NORM_MIN)
        forearm_heading = wrap_radians(np.arctan2(forearm[:, 1], forearm[:, 0]))
        relative = wrap_radians(heading - forearm_heading)
        relative_valid = axis_valid & forearm_valid
        relative_reason = np.full(len(x), "OBSERVED", dtype=object)
        for i in range(len(x)):
            reasons = []
            if not axis_valid[i]:
                reasons.append(str(axis_reason[i]))
            if not forearm_source[i]:
                reasons.append("FOREARM_SOURCE_UNOBSERVED")
            elif not forearm_valid[i]:
                reasons.append("DEGENERATE_FOREARM")
            if reasons:
                relative_reason[i] = "+".join(reasons)
        groups[hand + "_hand_forearm_relative2d"] = dict(values=np.rad2deg(relative)[:, None],
            observed=relative_valid[:, None], labels=[hand + "_hand_axis_minus_forearm_heading_2d"], circular=True,
            columns=axis_cols + forearm_cols, input_mask=m[:, axis_cols + forearm_cols], own_reason=relative_reason[:, None],
            definition="wrap(projected hand-axis heading minus pose elbow-to-wrist heading); separate hand/pose estimators",
            diagnostics=dict(hand_axis_norm=_nullable(axis_norm[:, None], axis_source[:, None]),
                forearm_norm=_nullable(forearm_norm[:, None], forearm_source[:, None]),
                forearm_heading_degrees=_nullable(np.rad2deg(forearm_heading)[:, None], forearm_valid[:, None])))
    return groups


def _path(path, n, m):
    out, previous = [], None
    for pair in path:
        if len(pair) != 2 or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in pair):
            raise ValueError("Fixed path must contain integer pairs")
        i, j = map(int, pair)
        if not (0 <= i < n and 0 <= j < m):
            raise ValueError("Fixed path index outside source axes")
        if previous is not None and (i < previous[0] or j < previous[1] or [i, j] == previous):
            raise ValueError("Fixed path must be monotone, without duplicate pairs")
        out.append([i, j]); previous = [i, j]
    return out


def _axis(a, own, other, path, differences, shared, side):
    n, d = own["values"].shape
    sums = np.zeros((n, d)); peaks = np.zeros((n, d))
    matched = np.zeros((n, d), dtype=np.int64); missing = np.zeros((n, d), dtype=np.int64)
    counterpart_observed = np.zeros((n, d), dtype=np.int64); aligned = np.zeros(n, dtype=np.int64)
    for k, pair in enumerate(path):
        i, j = pair[side], pair[1 - side]
        aligned[i] += 1
        matched[i] += shared[k]
        counterpart_observed[i] += other["observed"][j]
        missing[i] += own["observed"][i] & ~other["observed"][j]
        sums[i, shared[k]] += differences[k, shared[k]]
        peaks[i, shared[k]] = np.maximum(peaks[i, shared[k]], differences[k, shared[k]])
    supported = matched > 0
    mixed = supported & (missing > 0)
    means = np.divide(sums, matched, out=np.zeros_like(sums), where=supported)
    statuses = np.full((n, d), "NOT_COMPARABLE", dtype=object)
    statuses[supported] = "COMPARED"; statuses[mixed] = "PARTIAL_SHARED_SUPPORT"
    reasons = np.full((n, d), "OWN_UNOBSERVED", dtype=object)
    reasons[own["observed"]] = "COUNTERPART_UNOBSERVED"
    reasons[own["observed"] & (aligned[:, None] == 0)] = "NO_ALIGNED_COUNTERPART"
    reasons[supported] = "SHARED_OBSERVATION"; reasons[mixed] = "SHARED_AND_MISSING_COUNTERPART_LINKS"
    return dict(possible_shape=[n, d], dimension_labels=own["labels"],
                frame_id=a["frame_id"].tolist(), timestamp_us=a["timestamp_us"].tolist(),
                values_degrees=_nullable(own["values"], own["observed"]), own_observed=own["observed"].tolist(),
                own_reason=own["own_reason"].tolist(), original_source_mask=own["input_mask"].tolist(),
                matched_support=supported.tolist(), mixed_counterpart_missing=mixed.tolist(),
                mean_abs_deg=_nullable(means, supported), max_abs_deg=_nullable(peaks, supported),
                aligned_link_count=aligned.tolist(), matched_link_count=matched.tolist(),
                missing_counterpart_link_count=missing.tolist(), counterpart_observed_link_count=counterpart_observed.tolist(),
                cell_status=statuses.tolist(), cell_reason=reasons.tolist(),
                counts=dict(possible=n * d, own_observed=int(own["observed"].sum()),
                    own_unobserved=int((~own["observed"]).sum()), matched_support=int(supported.sum()),
                    direct_complete=int((supported & ~mixed).sum()), partial_shared=int(mixed.sum()),
                    not_comparable=int((~supported).sum()),
                    observed_without_shared_counterpart=int((own["observed"] & ~supported).sum())),
                own_reason_counts=dict(Counter(own["own_reason"].ravel().tolist())),
                unaligned_frame_indices=np.flatnonzero(aligned == 0).tolist(),
                diagnostics=own.get("diagnostics", {}), all_opportunities_retained=True,
                signed_differences_averaged=False, semantic_verdict=None)


def compare(query, reference, path):
    """Preserve every independent hand-feature cell on both original axes."""
    q, r = _input(query), _input(reference)
    path = _path(path, len(q["x"]), len(r["x"]))
    qgroups, rgroups = _measurements(q), _measurements(r)
    groups = {}
    for name, qg in qgroups.items():
        rg = rgroups[name]
        d = qg["values"].shape[1]
        shared = np.zeros((len(path), d), dtype=bool)
        signed = np.zeros((len(path), d)); absolute = np.zeros_like(signed)
        antipodal = np.zeros_like(shared)
        for k, (i, j) in enumerate(path):
            shared[k] = qg["observed"][i] & rg["observed"][j]
            delta = qg["values"][i] - rg["values"][j]
            if qg["circular"]:
                # Work in degrees here to preserve the declared half-open range
                # exactly at +/-180 after derived angles have been converted.
                delta = (delta + 180.) % 360. - 180.
                antipodal[k] = shared[k] & (np.abs(delta) == 180.)
            signed[k, shared[k]] = delta[shared[k]]
            absolute[k, shared[k]] = np.abs(delta[shared[k]])
        by_dimension = []
        for dim in range(d):
            indices = np.flatnonzero(shared[:, dim])
            values = absolute[indices, dim]
            by_dimension.append(dict(dimension=dim, label=qg["labels"][dim], possible_links=len(path),
                shared_links=len(indices), mean_abs_deg=float(values.mean()) if len(indices) else None,
                max_abs_deg=float(values.max()) if len(indices) else None,
                peak_link_indices=indices[values == values.max()].tolist() if len(indices) else []))
        groups[name] = dict(definition=qg["definition"], source_columns=qg["columns"],
            source_feature_names=[feature_names()[i] for i in qg["columns"]], units="degrees",
            circular=qg["circular"], query=_axis(q, qg, rg, path, absolute, shared, 0),
            reference=_axis(r, rg, qg, path, absolute, shared, 1),
            links=dict(shared_support=shared.tolist(), signed_difference_deg=_nullable(signed, shared),
                absolute_difference_deg=_nullable(absolute, shared),
                antipodal_signed_direction_ambiguous=antipodal.tolist(), signed_difference_convention="query_minus_reference"),
            summary_by_dimension=by_dimension, scalar_agreement=None, semantic_verdict=None)
    full = bool(len(q["x"]) and len(r["x"]) and {i for i, j in path} == set(range(len(q["x"])))
                and {j for i, j in path} == set(range(len(r["x"]))))
    return dict(schema="wis_independent_hand_evidence_v1", groups=groups, path=path,
                path_full_scope=full, path_unchanged=True, new_alignment_computed=False,
                query_frame_ids=q["frame_id"].tolist(), reference_frame_ids=r["frame_id"].tolist(),
                all_feature_and_frame_opportunities_retained=True,
                position_used_for_gating=False, learned_shape_used_for_gating=False,
                position_curve_recomputed=False, finger_internal_angles_are_non_circular=True,
                direction_wrap="[-180,180) degrees; exact antipodes use -180 with signed ambiguity flag",
                norm_degeneracy_threshold=NORM_MIN, zero_difference_is_correct_sign=False,
                scalar_agreement=None, semantic_verdict=None, whole_sign_verdict=None,
                correction_advice=None, missing_values_imputed=False,
                three_dimensional_rotation_validated=False, direct_wrist_flexion_validated=False,
                input_values_and_masks_unchanged=True,
                limits=["Projected hand axis is wrist0 to middle MCP9, not the palm surface normal or full 3D fist rotation.",
                    "Relative heading combines separate hand and pose estimators; it is not a validated anatomical wrist-bend angle.",
                    "Projected finger internal angles mix articulation and view; raw differences are not educational errors.",
                    "The supplied path is retained, not certified as same-event correspondence."])
