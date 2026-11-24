"""Synthetic-only contract checks. Root executes; this module writes no files."""
import copy
import json
import numpy as np
import trajectory as T


def sample(values, mask=None, pts=None):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n = len(values)
    observed = np.ones_like(values, dtype=bool) if mask is None else np.asarray(mask)
    if observed.ndim == 1:
        observed = np.repeat(observed[:, None], values.shape[1], axis=1)
    return dict(values=values, mask=observed, frame_id=np.arange(n, dtype=np.int64),
                timestamp_us=np.arange(n, dtype=np.int64) * 10000 if pts is None else np.asarray(pts, dtype=np.int64))


def brute(costs):
    """Enumerate every monotone path for tiny independent DP fixtures."""
    n, m = costs.shape
    answers = []
    def visit(i, j, bottleneck):
        if i == n - 1 and j == m - 1:
            answers.append(bottleneck)
            return
        for di, dj in ((1, 1), (1, 0), (0, 1)):
            ni, nj = i + di, j + dj
            if ni < n and nj < m:
                visit(ni, nj, max(bottleneck, costs[ni, nj]))
    visit(0, 0, costs[0, 0])
    best = min(answers)
    return best, min(2, sum(value == best for value in answers))


def run():
    checks = []
    def check(name, value):
        checks.append(dict(name=name, passed=bool(value)))
        if not value:
            raise AssertionError(name)
    def rejects(name, fn):
        try:
            fn()
        except ValueError:
            check(name, True)
        else:
            check(name, False)
    def compared(q, r, path=None):
        old_q, old_r, old_path = copy.deepcopy(q), copy.deepcopy(r), copy.deepcopy(path)
        result = T.compare(q, r, path)
        check("original_inputs_unchanged_" + str(len(checks)), all(
            np.array_equal(source[key], old[key], equal_nan=True)
            for source, old in ((q, old_q), (r, old_r))
            for key in ("values", "mask", "frame_id", "timestamp_us")) and path == old_path)
        check("no_semantic_verdict_" + str(len(checks)), result["semantic_verdict"] is None and
              result["correction_advice"] is None and not result["alignment_verified"] and
              not result["missing_values_imputed"] and result["candidate"]["educational_error"] is None)
        for axis, source in (("query", q), ("reference", r)):
            a = result[axis]
            check("all_source_opportunities_" + axis + "_" + str(len(checks)),
                  a["possible_cells"] == source["values"].size == a["observed_cells"] + a["missing_cells"]
                  and a["possible_frames"] == sum(x["frame_count"] for x in a["observed_blocks"] + a["missing_intervals"]))
        if result["candidate"]["complete"]:
            cp = result["candidate"]["path"]
            check("candidate_endpoints_and_full_axes_" + str(len(checks)), cp[0] == [0, 0] and
                  cp[-1] == [len(q["values"]) - 1, len(r["values"]) - 1] and
                  {i for i, j in cp} == set(range(len(q["values"]))) and
                  {j for i, j in cp} == set(range(len(r["values"]))))
        return result

    q = sample([[0, 0], [1, .5], [2, 0]])
    identity_path = [[0, 0], [1, 1], [2, 2]]
    out = compared(q, copy.deepcopy(q), identity_path)
    check("identity_distance_and_baseline_zero", out["candidate"]["distance"] == 0 and out["baseline"]["mean"] == 0
          and out["baseline"]["complete_link_max"] == 0)
    check("no_replacement_of_original_global_path", out["baseline"]["path"] == identity_path
          and not out["candidate"]["original_global_path_replaced"])
    check("all_bottleneck_witnesses_address_original_frames", all(
          w["query_frame_id"] == q["frame_id"][w["query_index"]] and
          w["reference_frame_id"] == q["frame_id"][w["reference_index"]]
          for w in out["candidate"]["bottleneck_witnesses"]))

    # Exact repeated points, including optional holds, leave discrete distance unchanged.
    held = sample(np.repeat(q["values"], [4, 1, 3], axis=0))
    hold = compared(held, q)
    check("hold_duplication_invariant", hold["candidate"]["distance"] == out["candidate"]["distance"])
    check("fresh_control_without_saved_baseline_is_explicit", hold["baseline"]["status"] == "NOT_SUPPLIED"
          and hold["baseline"]["mean"] is None and hold["baseline"]["path"] == [])
    shifted = sample([[0, .2], [1, .7], [2, .2]])
    base_shift = compared(q, shifted)
    held_shift = compared(held, shifted)
    check("duplication_invariant_for_nonzero_distance", base_shift["candidate"]["distance"] == held_shift["candidate"]["distance"])
    constant = compared(sample([[1, 2]] * 4), sample([[1, 2]] * 3))
    check("constant_hold_ambiguity_retained", constant["candidate"]["distance"] == 0
          and constant["candidate"]["optimal_path_ambiguous"] and constant["candidate"]["optimal_path_count_capped"] == 2)
    check("hold_is_not_measured_movement", not constant["candidate"]["query_has_two_distinct_positions"])
    speed = sample(q["values"], pts=[0, 100000, 100001])
    timed = compared(speed, q, identity_path)
    check("normal_pts_timewarp_no_geometry_change", timed["candidate"]["distance"] == 0
          and timed["query"]["timestamp_us"] == [0, 100000, 100001])

    # Paired L/R columns remain fixed even when hands cross.
    crossed = compared(sample([[-1, 1], [0, 0], [1, -1]]), sample([[1, -1], [0, 0], [-1, 1]]))
    check("crossing_no_left_right_reassignment", crossed["candidate"]["distance"] == 2)

    # A single missing coordinate makes the whole-vector candidate incomplete.
    partial_q = sample([[0, 0], [10, 10], [0, 0]], [[True, True], [True, False], [True, True]])
    partial = compared(partial_q, sample([[0, 0]] * 3), identity_path)
    check("one_coordinate_gap_blocks_whole_distance", partial["candidate"]["distance"] is None
          and not partial["candidate"]["complete"] and partial["candidate"]["path"] == [])
    check("continuous_observed_blocks_do_not_stitch", [(b["start"], b["end"]) for b in partial["query"]["observed_blocks"]] == [(0, 0), (2, 2)]
          and partial["query"]["missing_intervals"][0]["frame_range"] == [1, 1])
    check("partial_coordinate_peak_retained_separately", partial["baseline"]["point_costs"] == [0, 10, 0]
          and partial["baseline"]["max"] == 10 and partial["baseline"]["complete_link_max"] == 0
          and partial["baseline"]["partial_link_count"] == 1
          and partial["baseline"]["query"]["matched_support"][1] == [True, False])
    check("one_point_blocks_are_position_only", all(not b["has_at_least_two_frames"] and
          not b["has_two_distinct_observed_positions"] for b in partial["query"]["observed_blocks"]))
    hidden = compared(sample([0, 1, 0]), sample([0, 0, 0], [True, False, True]), identity_path)
    check("hidden_peak_is_unknown_despite_zero_partial_mean", hidden["candidate"]["distance"] is None
          and hidden["baseline"]["mean"] == 0 and hidden["baseline"]["point_costs"] == [0, None, 0]
          and hidden["baseline"]["query"]["mean"][1] is None)
    none = compared(sample([[0, 0], [0, 0]], [[False, False], [False, False]]), sample([[0, 0]] * 2), [[0, 0], [1, 1]])
    check("no_support_keeps_denominator_without_distance", none["query"]["possible_cells"] == 4
          and none["query"]["observed_blocks"] == [] and none["candidate"]["distance"] is None
          and none["baseline"]["mean"] is None and none["baseline"]["unsupported_link_count"] == 2)
    single = compared(sample([[1, 2]]), sample([[2, 2]]), [[0, 0]])
    check("single_point_position_distance_not_trajectory", single["candidate"]["distance"] == 1
          and not single["candidate"]["has_at_least_two_frames_each_axis"])
    empty = compared(sample(np.zeros((0, 2))), sample([[0, 0]]))
    check("empty_scope_retained", empty["candidate"]["status"] == "EMPTY_INPUT" and empty["query"]["possible_frames"] == 0
          and empty["reference"]["possible_frames"] == 1)

    # A short observed excursion is not diluted by the candidate's max objective.
    peak_q = sample([0] * 10 + [1] + [0] * 10)
    peak = compared(peak_q, sample([0] * 21), [[i, i] for i in range(21)])
    check("known_short_peak_average_dilution_visible", np.isclose(peak["baseline"]["mean"], 1 / 21)
          and peak["baseline"]["max"] == 1 and peak["candidate"]["distance"] == 1)
    check("short_peak_has_original_witness", any(w["query_index"] == 10 and w["point_cost"] == 1
          for w in peak["candidate"]["bottleneck_witnesses"]))
    reverse = compared(sample([0, 1, 2]), sample([2, 1, 0]))
    check("reverse_geometry_can_differ", reverse["candidate"]["distance"] == 2)
    order = compared(sample([0, 1, 2, 0]), sample([0, 2, 1, 0]))
    check("same_endpoints_point_set_different_order", order["candidate"]["distance"] == 1)
    symmetric = compared(sample([0, 1, 0]), sample([0, 1, 0]))
    check("symmetric_reversal_cannot_be_identified", symmetric["candidate"]["distance"] == 0
          and symmetric["semantic_verdict"] is None)
    # Deleting straight-path samples is not automatically a linguistic omission.
    straight = compared(sample([0, .5, 1]), sample([0, 1]))
    check("straight_sampling_difference_not_educational_error", straight["candidate"]["distance"] == .5
          and straight["candidate"]["educational_error"] is None)
    duplicate_deletion = compared(sample([0, 0, 1, 1]), sample([0, 1]))
    check("straight_duplicate_deletion_is_indistinguishable", duplicate_deletion["candidate"]["distance"] == 0)

    epsilon = .002
    noisy = copy.deepcopy(q); noisy["values"] += np.array([[epsilon, -epsilon], [-epsilon, epsilon], [epsilon, epsilon]])
    noise = compared(noisy, q)
    check("jitter_not_unit_direction_amplified", noise["candidate"]["distance"] <= epsilon + 1e-12)
    noisy_shift = copy.deepcopy(shifted); noisy_shift["values"] -= epsilon
    two_noise = compared(noisy, noisy_shift)
    check("both_curves_linf_perturbation_bound", abs(two_noise["candidate"]["distance"] - base_shift["candidate"]["distance"]) <= 2 * epsilon + 1e-12)
    meta_q = copy.deepcopy(q); meta_q.update(word="arbitrary", phase=[2, 1, 0], original_frame_ids=[99, 2, 8])
    meta = compared(meta_q, q)
    check("word_phase_source_index_metadata_unused", meta["candidate"] == out["candidate"])
    for index, matrix in enumerate(([[0]], [[0, 1], [0, 2]], [[0, 0, 0], [0, 0, 0]],
                                   [[0, 1, 0], [1, .2, 1], [0, 1, 0]], [[.2, 1], [0, .3], [.4, 0]])):
        matrix = np.asarray(matrix)
        expected, count = brute(matrix)
        solved = T.solve(matrix)
        check("independent_minmax_enumeration_" + str(index), solved["distance"] == expected
              and solved["optimal_path_count_capped"] == count
              and max(solved["path_point_costs"]) == expected)
    check("global_ties_include_nonoptimal_prefixes", T.solve([[0, 1], [0, 2]])["optimal_path_count_capped"] == 2)

    bad = sample([0, float("nan")])
    rejects("observed_nan_rejected", lambda: T.compare(bad, sample([0, 0])))
    bad = sample([0, float("inf")])
    rejects("observed_inf_rejected", lambda: T.compare(bad, sample([0, 0])))
    masked = sample([0, float("nan")], [True, False])
    allowed = compared(masked, sample([0, 0]), [[0, 0], [1, 1]])
    check("masked_nonfinite_is_unknown_not_imputed", allowed["candidate"]["distance"] is None
          and allowed["query"]["masked_nonfinite_cells"] == 1 and allowed["baseline"]["point_costs"][1] is None)
    bad = sample([0, 1]); bad["mask"] = np.ones((2, 1), dtype=int)
    rejects("numeric_mask_rejected", lambda: T.compare(bad, sample([0, 1])))
    bad = sample([0, 1]); bad["mask"] = np.ones((1, 1), dtype=bool)
    rejects("mask_shape_mismatch_rejected", lambda: T.compare(bad, sample([0, 1])))
    rejects("coordinate_dimension_mismatch_rejected", lambda: T.compare(sample([0, 1]), sample([[0, 0], [1, 1]])))
    bad = sample([0, 1]); bad["frame_id"] = np.array([2, 1])
    rejects("nonincreasing_frame_id_rejected", lambda: T.compare(bad, sample([0, 1])))
    bad = sample([0, 1]); bad["timestamp_us"] = np.array([0, 0])
    rejects("nonincreasing_pts_rejected", lambda: T.compare(bad, sample([0, 1])))
    rejects("nonmonotone_baseline_rejected", lambda: T.compare(sample([0, 1]), sample([0, 1]), [[1, 0], [0, 1]]))
    rejects("out_of_bounds_baseline_rejected", lambda: T.compare(sample([0]), sample([0]), [[0, 1]]))
    return dict(status="PASS_SYNTHETIC_TRAJECTORY_ONLY", checks=len(checks), results=checks,
                public_data_executed=False, model_inference_executed=False, educational_accuracy_claim=None,
                unresolved_limits=["Sampling deletion and normal speed are not generally identifiable from stored points.",
                                   "Missing motion is unknown, and symmetric/repeated trajectories can remain ambiguous."])


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
