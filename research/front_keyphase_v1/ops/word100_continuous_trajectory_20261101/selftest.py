"""Synthetic geometry checks only; no public arrays, model, or old-result access."""
import copy
import json
from fractions import Fraction
import numpy as np
import continuous as C


def curve(points):
    return np.asarray(points, dtype=np.float64).reshape((-1, 2))


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
    def compare(p, q, pm=None, qm=None, **kwargs):
        p, q = curve(p), curve(q)
        pm = np.ones_like(p, dtype=bool) if pm is None else np.asarray(pm)
        qm = np.ones_like(q, dtype=bool) if qm is None else np.asarray(qm)
        before = [x.copy() for x in (p, q, pm, qm)]
        answer = C.compare(p, q, pm, qm, **kwargs)
        check("input_arrays_unchanged_" + str(len(checks)), all(np.array_equal(a, b, equal_nan=True)
              for a, b in zip((p, q, pm, qm), before)))
        check("no_path_or_semantic_claim_" + str(len(checks)), answer["path_claim"] is False and
              answer["whole_sign_verdict"] is None and answer["semantic_verdict"] is None and
              answer["orientation_or_shape_evaluated"] is False and not answer["handshape_outputs_replaced"])
        check("all_opportunities_preserved_" + str(len(checks)), answer["query_possible_cells"] == p.size and
              answer["reference_possible_cells"] == q.size and answer["query_observed_cells"] + answer["query_missing_cells"] == p.size and
              answer["reference_observed_cells"] + answer["reference_missing_cells"] == q.size)
        if answer["converged"]:
            check("certified_width_" + str(len(checks)), answer["certified"] and
                  0 <= answer["lower_bound"] <= answer["upper_bound"] and
                  answer["upper_bound"] - answer["lower_bound"] <= answer["numerical_tolerance"])
            check("independent_final_exact_high_" + str(len(checks)), C.decision(p, q, answer["upper_bound"]))
            lo = answer["lower_bound"]
            endpoints = max(np.max(np.abs(p[0] - q[0])), np.max(np.abs(p[-1] - q[-1])))
            # Point cases and endpoint lower bounds need not be infeasible at equality.
            check("lower_bound_interpretation_" + str(len(checks)), lo <= endpoints or
                  answer.get("position_only_case", False) or not C.decision(p, q, lo))
        return answer
    def contains(answer, expected):
        return answer["converged"] and answer["lower_bound"] <= expected <= answer["upper_bound"]

    p = [(0, 0), (1, 1), (2, 0)]
    identity = compare(p, p)
    check("identical_exact_zero", identity["zero_distance_certified"] and identity["upper_bound"] == 0)
    duplicate = compare([(0, 0)] * 3 + [(1, 1)] * 5 + [(2, 0)] * 2, p)
    check("stationary_duplicates_exact_zero", duplicate["zero_distance_certified"])
    straight = [(0, 0), (2, 2)]
    subdivided = [(0, 0), (.25, .25), (.5, .5), (1, 1), (2, 2)]
    same_line = compare(straight, subdivided)
    check("collinear_subdivision_exact_zero", same_line["zero_distance_certified"])
    changed_sampling = compare([(0, 0), (1, 0), (2, 0)], [(0, 0), (2, 0)])
    check("straight_sample_deletion_exact_zero", changed_sampling["zero_distance_certified"])
    hold_edges = compare([(0, 0), (0, 0), (2, 0)], [(0, 0), (2, 0), (2, 0)])
    check("degenerate_segments_are_holds", hold_edges["zero_distance_certified"])
    parallel = compare(straight, [(0, .5), (2, 2.5)])
    parallel_sub = compare(subdivided, [(0, .5), (.5, 1), (2, 2.5)])
    check("nonzero_subdivision_invariance", contains(parallel, .5) and contains(parallel_sub, .5))

    deleted_turn = compare(p, [(0, 0), (2, 0)])
    check("sharp_turn_deletion_preserved", contains(deleted_turn, 1) and not deleted_turn["zero_distance_certified"])
    square = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
    opposite_square = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
    order = compare(square, opposite_square)
    check("noncollinear_order_with_same_endpoints_and_image", contains(order, 2))
    check("order_blocks_below_two", not C.decision(curve(square), curve(opposite_square), Fraction(1999, 1000)))
    backtrack = compare([(0, 0), (1, 0), (0, 0), (1, 0)], [(0, 0), (1, 0)])
    check("collinear_reverse_traversal_not_removed", contains(backtrack, .5))
    reversed_both = compare(list(reversed(p)), [(2, 0), (0, 0)])
    check("reversing_both_curves_preserves_distance", contains(reversed_both, 1))
    swapped = compare([(0, 0), (2, 0)], p)
    check("curve_exchange_symmetry", contains(swapped, 1))
    crossing = compare([(-1, 1), (0, 0), (1, -1)], [(1, -1), (0, 0), (-1, 1)])
    check("fixed_column_cross_identity_no_reassignment", contains(crossing, 2))
    shared_position = compare([(0, 0), (1, 1)], [(0, 0), (1, 1)])
    check("position_zero_does_not_discard_fist_rotation_or_wrist_bend", shared_position["zero_distance_certified"] and
          not shared_position["position_zero_is_whole_sign_agreement"] and not shared_position["orientation_or_shape_evaluated"])

    gap = compare(p, p, [[True, True], [True, False], [True, True]])
    check("single_missing_coordinate_nulls_whole_parent", not gap["complete"] and gap["lower_bound"] is None and
          gap["upper_bound"] is None and gap["zero_distance_certified"] is None and not gap["gap_segments_created"])
    hidden_turn = compare(p, [(0, 0), (1, 0), (2, 0)], [[True, True], [False, False], [True, True]])
    check("hidden_peak_not_replaced_by_endpoint_line", hidden_turn["status"] == "INCOMPLETE_OBSERVATION")
    point = compare([(0, 0)], [(1, 0), (0, 2), (-1, 0)])
    check("point_to_polyline_maximum_is_at_vertex", contains(point, 2) and point["position_only_case"])
    both_points = compare([(1, 2)], [(1.5, 2)])
    check("two_points_linf", contains(both_points, .5))
    point_hold = compare([(1, 2)], [(1, 2)] * 4)
    check("point_and_constant_curve_zero", point_hold["zero_distance_certified"])
    empty = compare([], p)
    check("empty_not_certified", empty["status"] == "EMPTY_INPUT" and not empty["certified"] and empty["query_possible_cells"] == 0)
    missing_all = compare(p, p, np.zeros((3, 2), dtype=bool))
    check("unobserved_keeps_source_denominator", missing_all["query_missing_cells"] == 6 and missing_all["upper_bound"] is None)

    tiny_turn = compare([(0, 0), (1, 1e-12), (2, 0)], [(0, 0), (2, 0)])
    check("numeric_tolerance_never_changes_exact_zero_test", contains(tiny_turn, 1e-12) and
          not tiny_turn["zero_distance_certified"] and tiny_turn["zero_distance_decision_exact"])
    tangent_p, tangent_q = curve([(0, 0), (2, 0)]), curve([(0, 1), (2, 1)])
    check("closed_free_space_boundary_feasible", C.decision(tangent_p, tangent_q, 1))
    check("below_closed_boundary_infeasible", not C.decision(tangent_p, tangent_q, float(np.nextafter(1., 0.))))
    check("boundary_interval_exact_fraction", C._interval((Fraction(0), Fraction(0)), (Fraction(2), Fraction(0)),
          (Fraction(1), Fraction(0)), Fraction(1, 2)) == (Fraction(1, 4), Fraction(3, 4)))
    check("constant_segment_outside_free_space", C._interval((0, 0), (0, 0), (1, 0), .5) is None)
    check("constant_segment_full_interval", C._interval((0, 0), (0, 0), (1, 0), 1) == (0, 1))

    original = curve(p)
    perturbation = np.array([[.002, -.002], [-.002, .002], [.002, .002]])
    jitter = compare(original + perturbation, original)
    check("jitter_linf_bound_no_direction_normalization", jitter["upper_bound"] <= .002 + C.DEFAULT_ATOL)
    changed_turn = curve([(0, 0), (1, .5), (2, 0)])
    before = compare(original, changed_turn)
    after = compare(original + perturbation, changed_turn - perturbation)
    check("two_curve_perturbation_bound", after["lower_bound"] - before["upper_bound"] <= .004 + C.DEFAULT_ATOL and
          before["lower_bound"] - after["upper_bound"] <= .004 + C.DEFAULT_ATOL)
    limited = compare(p, [(0, 0), (2, 0)], max_iterations=1)
    check("iteration_limit_not_observation_failure", limited["complete"] and limited["certified"] and
          not limited["converged"] and limited["status"] == "ITERATION_LIMIT" and limited["iterations"] == 1)

    # Several asymmetric cases compare the numerical proposal with exact decisions.
    cases = [([(0, 0), (1, 2), (3, 1)], [(0, 1), (2, 2), (3, 0)]),
             ([(0, 0), (1, 0), (1, 1), (2, 1)], [(0, 0), (2, 1)]),
             ([(0, 0), (1, 1), (0, 2), (1, 1)], [(0, 0), (0, 2), (1, 1)])]
    for k, (a, b) in enumerate(cases):
        answer = compare(a, b)
        check("nontrivial_certified_case_" + str(k), answer["complete"] and answer["certified"] and answer["converged"])
        for e in (0, .25, .5, 1, 2):
            check("float_exact_reachability_crosscheck_" + str(k) + "_" + str(e), C.decision(curve(a), curve(b), e) ==
                  C.decision(curve(a), curve(b), e, exact=False))
    bad_nan = curve([(0, 0), (1, float("nan"))])
    rejects("observed_nonfinite_rejected", lambda: C.compare(bad_nan, curve(straight), np.ones((2, 2), bool), np.ones((2, 2), bool)))
    masked_nonfinite = compare(bad_nan, straight, [[True, True], [True, False]])
    check("masked_nonfinite_not_imputed", masked_nonfinite["status"] == "INCOMPLETE_OBSERVATION")
    rejects("numeric_mask_rejected", lambda: C.compare(curve(straight), curve(straight), np.ones((2, 2), int), np.ones((2, 2), bool)))
    rejects("wrong_dimension_rejected", lambda: C.compare(np.zeros((2, 3)), curve(straight), np.ones((2, 3), bool), np.ones((2, 2), bool)))
    rejects("negative_epsilon_rejected", lambda: C.decision(curve(straight), curve(straight), -1))
    rejects("zero_numerical_tolerance_rejected", lambda: C.compare(curve(straight), curve(straight), np.ones((2, 2), bool), np.ones((2, 2), bool), atol=0))
    rejects("unbounded_iteration_request_rejected", lambda: C.compare(curve(straight), curve(straight), np.ones((2, 2), bool), np.ones((2, 2), bool), max_iterations=61))
    return dict(status="PASS_SYNTHETIC_CONTINUOUS_GEOMETRY_ONLY", checks=len(checks), results=checks,
                public_data_executed=False, new_model_inference=False,
                orientation_shape_or_educational_accuracy_claim=None,
                notes=["Exact certificates concern binary-float polygonal vertices, not real motion between samples.",
                       "Independent method review must cross-check reachability; selftest is implementation verification."])


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
