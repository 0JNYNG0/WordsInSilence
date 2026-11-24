"""Synthetic independent-hand evidence checks. No public data or model access."""
import copy
import json
import numpy as np
import hand_evidence as H


def sample(n=7):
    x = np.zeros((n, 69), dtype=np.float64)
    mask = np.zeros_like(x, dtype=bool)
    for off, finger in ((0, .8), (28, .4)):
        x[:, off:off + 6] = [.2, .1, .2, .1, -.8, .1]
        mask[:, off:off + 6] = True
        x[:, off + 11:off + 25] = finger
        mask[:, off + 11:off + 25] = True
        x[:, off + 25:off + 27] = [1, 0]
        mask[:, off + 25:off + 27] = True
    return dict(x=x, mask=mask, names=H.feature_names(), frame_id=np.arange(40, 40 + n, dtype=np.int64),
                timestamp_us=np.arange(40, 40 + n, dtype=np.int64) * 33333)


def rotate_hand(a, degrees, off=28, rows=None):
    rows = np.arange(len(a["x"])) if rows is None else np.asarray(rows)
    angle = np.deg2rad(degrees)
    a["x"][rows, off + 25] = np.cos(angle)
    a["x"][rows, off + 26] = np.sin(angle)


def rotate_forearm(a, degrees, off=28, rows=None):
    rows = np.arange(len(a["x"])) if rows is None else np.asarray(rows)
    direction = np.array([np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))])
    a["x"][rows, off + 4:off + 6] = a["x"][rows, off + 2:off + 4] - direction


def run():
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)
    def rejects(name, fn):
        try:
            fn()
        except ValueError:
            check(name, True)
        else:
            check(name, False)
    def compare(q, r, path=None):
        path = [[i, i] for i in range(len(q["x"]))] if path is None else path
        saved = copy.deepcopy((q, r, path))
        out = H.compare(q, r, path)
        check("source_input_unchanged_" + str(len(checks)), all(np.array_equal(a[k], old[k], equal_nan=True)
              for a, old in ((q, saved[0]), (r, saved[1])) for k in ("x", "mask", "frame_id", "timestamp_us")) and saved[2] == path)
        check("path_preserved_no_semantic_grade_" + str(len(checks)), out["path"] == path and not out["new_alignment_computed"]
              and out["whole_sign_verdict"] is None and not out["position_used_for_gating"] and not out["learned_shape_used_for_gating"])
        for name, group in out["groups"].items():
            for axis in ("query", "reference"):
                a = group[axis]; counts = a["counts"]
                check("opportunity_partition_" + name + "_" + axis + "_" + str(len(checks)),
                      counts["possible"] == np.prod(a["possible_shape"]) == counts["own_observed"] + counts["own_unobserved"] and
                      counts["possible"] == counts["direct_complete"] + counts["partial_shared"] + counts["not_comparable"])
        return out
    def peak(out, group, dimension=0):
        return out["groups"][group]["summary_by_dimension"][dimension]["max_abs_deg"]
    def near(a, b):
        return np.isclose(a, b, rtol=0, atol=1e-10)

    base = sample()
    identical = compare(base, copy.deepcopy(base))
    check("schema_exact69", len(H.feature_names()) == 69 and H.feature_names()[39] == "R_thumb_mcp_angle")
    check("all_identity_features_zero", all(row["max_abs_deg"] == 0 for g in identical["groups"].values() for row in g["summary_by_dimension"]))
    check("raw_halfturn_degree_conversion", identical["groups"]["R_finger_angles14"]["query"]["values_degrees"][0][0] == 72)
    q = copy.deepcopy(base); rotate_hand(q, 90)
    axis = compare(q, base)
    check("stationary_wrist_axis_change_preserved", np.array_equal(q["x"][:, 28:30], base["x"][:, 28:30])
          and near(peak(axis, "R_hand_axis2d"), 90) and near(peak(axis, "R_hand_forearm_relative2d"), 90))
    check("hand_axis_change_does_not_change_finger_evidence", peak(axis, "R_finger_angles14", 2) == 0)
    q = copy.deepcopy(base); q["x"][:, 41] = .9
    finger = compare(q, base)
    check("stationary_wrist_finger_change_preserved", near(peak(finger, "R_finger_angles14", 2), 90)
          and peak(finger, "R_hand_axis2d") == 0 and np.array_equal(q["x"][:, 28:30], base["x"][:, 28:30]))
    q = copy.deepcopy(base); r = copy.deepcopy(base)
    q["x"][:, 41] = 1; r["x"][:, 41] = 0
    noncyclic = compare(q, r)
    check("finger_internal_angle_not_wrapped", peak(noncyclic, "R_finger_angles14", 2) == 180
          and noncyclic["groups"]["R_finger_angles14"]["links"]["signed_difference_deg"][0][2] == 180)
    q = copy.deepcopy(base); rotate_hand(q, 70); rotate_forearm(q, 70)
    rigid = compare(q, base)
    check("joint_arm_hand_rotation_relative_invariant", near(peak(rigid, "R_hand_axis2d"), 70)
          and near(peak(rigid, "R_hand_forearm_relative2d"), 0))
    q = copy.deepcopy(base); rotate_forearm(q, 50)
    forearm = compare(q, base)
    check("forearm_change_independent_of_absolute_hand_axis", peak(forearm, "R_hand_axis2d") == 0
          and near(peak(forearm, "R_hand_forearm_relative2d"), 50))

    q = copy.deepcopy(base); q["mask"][2, 53] = False
    axis_unknown = compare(q, base)
    check("missing_axis_not_normal", axis_unknown["groups"]["R_hand_axis2d"]["query"]["mean_abs_deg"][2][0] is None
          and axis_unknown["groups"]["R_hand_forearm_relative2d"]["query"]["mean_abs_deg"][2][0] is None)
    check("axis_missing_keeps_fingers_and_other_hand", axis_unknown["groups"]["R_finger_angles14"]["query"]["counts"]["matched_support"] == 98
          and axis_unknown["groups"]["L_hand_axis2d"]["query"]["counts"]["matched_support"] == 7)
    q = copy.deepcopy(base); q["mask"][3, 41] = False
    one = compare(q, base)
    g = one["groups"]["R_finger_angles14"]["query"]
    check("one_missing_finger_does_not_gate14", g["counts"]["possible"] == 98 and g["counts"]["matched_support"] == 97
          and sum(g["matched_support"][3]) == 13 and g["max_abs_deg"][3][2] is None)
    q = copy.deepcopy(base); q["mask"][:, 28:30] = False
    no_position = compare(q, base)
    check("missing_body_normalized_wrist_cannot_gate_observed_hand_features", no_position["groups"]["R_hand_axis2d"]["query"]["counts"]["matched_support"] == 7
          and no_position["groups"]["R_finger_angles14"]["query"]["counts"]["matched_support"] == 98)
    q = copy.deepcopy(base); q["mask"][1, 32] = False
    forearm_missing = compare(q, base)
    check("missing_forearm_keeps_hand_axis", forearm_missing["groups"]["R_hand_axis2d"]["query"]["max_abs_deg"][1][0] == 0
          and forearm_missing["groups"]["R_hand_forearm_relative2d"]["query"]["max_abs_deg"][1][0] is None)

    q = sample(21); r = sample(21); rotate_hand(q, 45, rows=np.arange(8, 13)); q["x"][8:13, 41] += .25
    short = compare(q, r)
    changed = short["groups"]["R_hand_axis2d"]["query"]["max_abs_deg"]
    check("short_five_frame_direction_change_retained", sum(near(row[0], 45) for row in changed) == 5
          and short["groups"]["R_hand_axis2d"]["summary_by_dimension"][0]["peak_link_indices"] == list(range(8, 13)))
    check("short_finger_change_retained", near(peak(short, "R_finger_angles14", 2), 45)
          and near(short["groups"]["R_hand_axis2d"]["summary_by_dimension"][0]["mean_abs_deg"], 45 * 5 / 21))
    q = copy.deepcopy(base)
    for off in (0, 28):
        q["x"][:, off:off + 6] += .002
    jitter = compare(q, base)
    check("common_translation_does_not_become_hand_rotation", all(near(peak(jitter, name), 0)
          for name in ("L_hand_axis2d", "R_hand_axis2d", "L_hand_forearm_relative2d", "R_hand_forearm_relative2d")))
    q = copy.deepcopy(base); rotate_hand(q, 1e-5)
    tiny = compare(q, base)
    check("small_measured_rotation_not_suppressed_by_norm_rule", 0 < peak(tiny, "R_hand_axis2d") < .00002)
    q = copy.deepcopy(base); r = copy.deepcopy(base); rotate_hand(q, 179); rotate_hand(r, -179)
    wrapped = compare(q, r)
    check("plus_minus_pi_boundary_small_residual", near(peak(wrapped, "R_hand_axis2d"), 2)
          and near(wrapped["groups"]["R_hand_axis2d"]["links"]["signed_difference_deg"][0][0], -2))
    q = copy.deepcopy(base); q["x"][:, 53:55] = [-1, 0]
    anti = compare(q, base)
    check("exact_antipode_magnitude_and_signed_ambiguity", peak(anti, "R_hand_axis2d") == 180
          and anti["groups"]["R_hand_axis2d"]["links"]["signed_difference_deg"][0][0] == -180
          and anti["groups"]["R_hand_axis2d"]["links"]["antipodal_signed_direction_ambiguous"][0][0])
    q = copy.deepcopy(base); q["x"][0, 53:55] = [H.NORM_MIN, 0]
    degenerate = compare(q, base)
    check("hand_axis_norm_boundary_invalid", degenerate["groups"]["R_hand_axis2d"]["query"]["own_reason"][0][0] == "DEGENERATE_HAND_AXIS")
    q = copy.deepcopy(base); q["x"][0, 30:34] = [H.NORM_MIN, 0, 0, 0]
    degenerate_arm = compare(q, base)
    check("forearm_norm_boundary_invalid_only_relative", degenerate_arm["groups"]["R_hand_forearm_relative2d"]["query"]["mean_abs_deg"][0][0] is None
          and degenerate_arm["groups"]["R_hand_axis2d"]["query"]["mean_abs_deg"][0][0] == 0)

    crossing = sample(9)
    crossing["x"][:, 0] = np.linspace(-1, 1, 9); crossing["x"][:, 28] = np.linspace(1, -1, 9)
    crossed = compare(crossing, copy.deepcopy(crossing))
    check("crossing_preserves_existing_hand_identity", peak(crossed, "L_finger_angles14") == 0 and peak(crossed, "R_finger_angles14") == 0
          and crossed["groups"]["L_finger_angles14"]["query"]["values_degrees"][4][0] == 144
          and crossed["groups"]["R_finger_angles14"]["query"]["values_degrees"][4][0] == 72)
    q = copy.deepcopy(crossing); q["x"][:, :28], q["x"][:, 28:56] = q["x"][:, 28:56].copy(), q["x"][:, :28].copy()
    swapped = compare(q, crossing)
    check("no_crossing_based_reassignment_of_swapped_features", near(peak(swapped, "L_finger_angles14"), 72)
          and near(peak(swapped, "R_finger_angles14"), 72))
    q = sample(2); r = sample(3); r["mask"][1, 41] = False; q["x"][0, 41] += .25
    partial = compare(q, r, [[0, 0], [0, 1], [1, 2]])
    a = partial["groups"]["R_finger_angles14"]["query"]
    check("mixed_counterpart_support_not_promoted_complete", a["mixed_counterpart_missing"][0][2]
          and a["cell_status"][0][2] == "PARTIAL_SHARED_SUPPORT" and near(a["mean_abs_deg"][0][2], 45)
          and a["missing_counterpart_link_count"][0][2] == 1 and a["matched_link_count"][0][2] == 1)
    incomplete_path = compare(base, base, [[0, 0], [6, 6]])
    check("unaligned_original_frames_kept", not incomplete_path["path_full_scope"] and
          incomplete_path["groups"]["R_hand_axis2d"]["query"]["unaligned_frame_indices"] == [1, 2, 3, 4, 5] and
          incomplete_path["groups"]["R_hand_axis2d"]["query"]["cell_reason"][3][0] == "NO_ALIGNED_COUNTERPART")
    q = copy.deepcopy(base); q.update(word="hint", phase=[99], position_curve_distance=0, shape_probability=1.)
    unrelated = compare(q, base)
    check("word_position_and_probability_metadata_cannot_gate", unrelated == identical)

    q = copy.deepcopy(base); q["x"][0, 41] = 1.01
    rejects("invalid_finger_range_not_clipped", lambda: H.compare(q, base, [[i, i] for i in range(7)]))
    q = copy.deepcopy(base); q["x"][0, 53] = np.nan
    rejects("observed_nonfinite_rejected", lambda: H.compare(q, base, [[i, i] for i in range(7)]))
    q["mask"][0, 53] = False
    hidden_nan = compare(q, base)
    check("masked_nonfinite_remains_unknown", hidden_nan["groups"]["R_hand_axis2d"]["query"]["values_degrees"][0][0] is None)
    q = copy.deepcopy(base); q["mask"] = q["mask"].astype(int)
    rejects("numeric_mask_rejected", lambda: H.compare(q, base, []))
    q = copy.deepcopy(base); q["names"][41] = "wrong_feature"
    rejects("source_feature_identity_mismatch_rejected", lambda: H.compare(q, base, []))
    q = copy.deepcopy(base); q["timestamp_us"][2] = q["timestamp_us"][1]
    rejects("source_pts_order_rejected", lambda: H.compare(q, base, []))
    rejects("invalid_path_rejected", lambda: H.compare(base, base, [[0, 1], [1, 0]]))
    return dict(status="PASS_SYNTHETIC_INDEPENDENT_HAND_EVIDENCE_ONLY", checks=len(checks), results=checks,
                public_data_executed=False, model_inference_executed=False, alignment_executed=False,
                three_dimensional_rotation_accuracy=None, anatomical_wrist_flexion_accuracy=None,
                educational_accuracy=None)


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
