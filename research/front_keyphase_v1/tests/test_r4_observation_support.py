"""R4: evidence of comparison is not the same thing as cost, and a missing side must say so.

The first check REPRODUCES the defect inline before requiring the fix. Under the previous runtime,
frame_cost_matrix used `both | one` as the denominator for BOTH the cost and the evidence, so a
group present on one side only counted as proof that a comparison had happened. Independent review
fed eight frames with every group present on one side and every group absent on the other and got
status OK, mean_evidence_fraction 1.0 and distance 1.5, with zero groups observed on both sides.

Cases here are the ones that change directly with that fix: full match, all missing, complementary
availability, low partial support, NaN/Inf while flagged present, and custom weights. Two
separations are asserted unchanged: ranking is word identity only, and execution_assessment stays
UNCONFIGURED. No threshold is tuned and no pass rate is adjusted.

Run: py -3.13 tests/test_r4_observation_support.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.core import dtw as T  # noqa: E402
from wis_lab.core import recognize as R  # noqa: E402
from wis_lab.core.features import FEATURE_WIDTH, GROUP_ORDER, GROUPS  # noqa: E402

CHECKS = []
MEASURED = {}
G = len(GROUP_ORDER)
NL = chr(10)
IDX = {g: i for i, g in enumerate(GROUP_ORDER)}


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def seq(n=8, present_groups=None, value=0.0, nan_groups=()):
    """A feature sequence with an explicit presence mask, so every case is stated not implied."""
    f = np.full((n, FEATURE_WIDTH), float(value), dtype=np.float32)
    p = np.zeros((n, G), dtype=bool)
    names = GROUP_ORDER if present_groups is None else present_groups
    for g in names:
        p[:, IDX[g]] = True
    cursor = 0
    span = {}
    for g in GROUP_ORDER:
        w = GROUPS[g][0]
        span[g] = slice(cursor, cursor + w)
        cursor += w
    for g, fill in nan_groups:
        f[:, span[g]] = fill
    return {"features": f, "present": p}


@check("the previous runtime counted a one-sided group as evidence; the current one does not")
def _reproduce_defect():
    a = seq(present_groups=GROUP_ORDER)
    b = seq(present_groups=[])
    pa, pb = a["present"], b["present"]
    weights = T.weight_vector(None)

    # the old denominator, written out rather than trusted from memory
    both = pa[:, None, :] & pb[None, :, :]
    one = pa[:, None, :] ^ pb[None, :, :]
    old_evidence = np.where(both | one, weights, 0.0).sum(axis=2) / weights.sum()
    assert abs(float(old_evidence[0, 0]) - 1.0) < 1e-9, float(old_evidence[0, 0])
    assert int(both[0, 0].sum()) == 0, "nothing was observed on both sides"

    r = T.dtw(a, b)
    MEASURED["defect"] = {"old_evidence_fraction": round(float(old_evidence[0, 0]), 4),
                          "groups_observed_on_both_sides": int(both[0, 0].sum()),
                          "new_status": r["status"],
                          "new_evidence_fraction": r.get("mean_evidence_fraction")}
    assert r["status"] == "INSUFFICIENT", r
    assert r["distance"] is None
    assert r["mean_evidence_fraction"] == 0.0, r["mean_evidence_fraction"]


@check("full match: every group observed on both sides gives distance 0 and full evidence")
def _full_match():
    a = seq()
    r = T.dtw(a, a)
    assert r["status"] == "OK", r
    assert r["distance"] == 0.0, r["distance"]
    assert r["mean_evidence_fraction"] == 1.0, r["mean_evidence_fraction"]


@check("all missing: nothing observed anywhere is INSUFFICIENT, not a cheap distance")
def _all_missing():
    r = T.dtw(seq(present_groups=[]), seq(present_groups=[]))
    assert r["status"] == "INSUFFICIENT", r
    assert r["distance"] is None


@check("complementary availability: every group present somewhere, none on both sides")
def _complementary():
    left, right = GROUP_ORDER[:8], GROUP_ORDER[8:]
    a, b = seq(present_groups=left), seq(present_groups=right)
    assert set(left) | set(right) == set(GROUP_ORDER), "the split must cover every group"
    cost, ev = T.frame_cost_matrix(a, b)
    r = T.dtw(a, b)
    MEASURED["complementary"] = {"cell_cost": round(float(cost[0, 0]), 4),
                                 "cell_evidence": round(float(ev[0, 0]), 4),
                                 "status": r["status"]}
    assert float(ev[0, 0]) == 0.0, "a one-sided group was counted as evidence"
    assert float(cost[0, 0]) > 0.0, "the one-sided penalty must still be charged"
    assert r["status"] == "INSUFFICIENT", r


@check("low partial support falls under the guard; enough support passes it")
def _partial_support():
    thin = ["arm_L"]                                             # weight 1.0 of 14.0
    thick = ["shape_L", "shape_R", "handpos_L", "handpos_R"]      # weight 7.0 of 14.0
    for names, want in ((thin, "INSUFFICIENT"), (thick, "OK")):
        a = seq(present_groups=names)
        r = T.dtw(a, a)
        share = sum(GROUPS[g][1] for g in names) / T.TOTAL_WEIGHT
        MEASURED.setdefault("partial_support", {})[",".join(names)] = {
            "weight_share": round(share, 4), "status": r["status"],
            "evidence": r.get("mean_evidence_fraction")}
        assert r["status"] == want, (names, r)
        # the reported fraction is rounded to four places, so compare at that resolution
        assert abs(r["mean_evidence_fraction"] - share) < 1e-4, (names, share, r)


@check("present but NaN or Inf is unobserved, never a distance of zero")
def _nan_inf_with_present():
    for fill in (np.nan, np.inf, -np.inf):
        a = seq(present_groups=GROUP_ORDER, nan_groups=[(g, fill) for g in GROUP_ORDER])
        b = seq(present_groups=GROUP_ORDER)
        cost, ev = T.frame_cost_matrix(a, b)
        assert np.isfinite(cost).all(), f"{fill} leaked into the cost"
        assert float(ev[0, 0]) == 0.0, f"{fill} was counted as evidence"
        r = T.dtw(a, b)
        assert r["status"] == "INSUFFICIENT", (fill, r)
        assert r["distance"] is None, (fill, r)

    # GAP 1, reproduced by review: only the FIRST component of each group is NaN. _nanmean used
    # to skip it and average the remaining zeros, returning distance 0 with full evidence.
    for fill in (np.nan, np.inf, -np.inf):
        a = seq(present_groups=GROUP_ORDER)
        b = seq(present_groups=GROUP_ORDER)
        cursor = 0
        for g in GROUP_ORDER:
            b["features"][:, cursor] = fill          # one component of this group only
            cursor += GROUPS[g][0]
        old_style = float(np.sqrt(np.nanmean(np.square(
            a["features"][0, :GROUPS[GROUP_ORDER[0]][0]]
            - b["features"][0, :GROUPS[GROUP_ORDER[0]][0]]))))
        r = T.dtw(a, b)
        MEASURED.setdefault("one_component_nan", {})[str(fill)] = {
            "nanmean_over_the_rest_would_be": (round(old_style, 4)
                                               if np.isfinite(old_style) else None),
            "status": r["status"], "evidence": r.get("mean_evidence_fraction")}
        assert r["status"] == "INSUFFICIENT", (fill, r)
        assert r["mean_evidence_fraction"] == 0.0, (fill, r)

    # only ONE group unmeasurable: the rest still counts, and that group is reported as such
    one_bad = seq(present_groups=GROUP_ORDER, nan_groups=[("shape_L", np.nan)])
    clean = seq(present_groups=GROUP_ORDER)
    res = T.group_residuals(one_bad, clean)
    g = res["groups"]["shape_L"]
    assert g["mean_distance"] is None, g
    assert g["unmeasurable_steps"] == res["path_length"], g
    assert g["compared_steps"] == 0, g
    assert res["groups"]["shape_R"]["mean_distance"] == 0.0
    MEASURED["single_unmeasurable_group"] = {
        "shape_L": {k: g[k] for k in ("compared_steps", "unmeasurable_steps", "mean_distance")},
        "measured_weight_fraction": res["support"]["measured_weight_fraction"]}


@check("custom weights are the ones reported and the ones evidence is measured against")
def _custom_weights():
    names = ["shape_L", "shape_R"]
    a = seq(present_groups=names)
    custom = {g: 0.0 for g in GROUP_ORDER}
    custom["shape_L"] = 3.0
    custom["shape_R"] = 1.0
    r = T.dtw(a, a, group_weights=custom)
    assert r["status"] == "OK", r
    assert abs(r["mean_evidence_fraction"] - 1.0) < 1e-6, r      # 4.0 of 4.0 applied weight
    res = T.group_residuals(a, a, group_weights=custom)
    assert res["applied_weights"]["shape_L"] == 3.0
    assert res["applied_weights"]["shape_R"] == 1.0
    assert res["applied_weights"]["arm_L"] == 0.0
    assert res["total_weight_applied"] == 4.0, res["total_weight_applied"]
    assert res["groups"]["shape_L"]["weight_applied"] == 3.0
    assert res["groups"]["shape_L"]["weight_default"] == GROUPS["shape_L"][1]
    MEASURED["custom_weights"] = {"total_applied": res["total_weight_applied"],
                                  "evidence": r["mean_evidence_fraction"]}


@check("a half-observed axis is reported as partial and cannot be shown as a clean verdict")
def _axis_partial_support():
    groups = {g: {"mean_distance": 0.1, "compared_steps": 8, "one_sided_steps": 0,
                  "unmeasurable_steps": 0, "unobserved_steps": 0} for g in GROUP_ORDER}
    full = R._axis_support(groups, R.AXIS_GROUPS["handshape"])
    assert full["partial"] is False and full["fraction"] == 1.0

    groups["bend_L"] = {"mean_distance": None, "compared_steps": 0, "one_sided_steps": 8,
                        "unmeasurable_steps": 0, "unobserved_steps": 0}
    half = R._axis_support(groups, R.AXIS_GROUPS["handshape"])
    assert half["partial"] is True, half
    assert half["measured_groups"] == ["bend_R"] and half["missing_groups"] == ["bend_L"]
    assert half["fraction"] == 0.5
    assert half["per_group"]["bend_L"]["one_sided_steps"] == 8

    # GAP 2, reproduced by review: every group produced a distance, but from one frame of eight.
    # Group coverage alone called that fully measured.
    thin = {g: {"mean_distance": 0.1, "compared_steps": 1, "one_sided_steps": 7,
                "unmeasurable_steps": 0, "unobserved_steps": 0} for g in GROUP_ORDER}
    t = R._axis_support(thin, R.AXIS_GROUPS["handshape"])
    assert t["group_fraction"] == 1.0, "every group did produce a distance"
    assert t["time_fraction_min"] == 0.125, t["time_fraction_min"]
    assert t["partial"] is True, t
    assert "partial_time_coverage" in t["partial_reason"], t["partial_reason"]
    MEASURED["time_coverage"] = {"group_fraction": t["group_fraction"],
                                 "time_fraction_min": t["time_fraction_min"],
                                 "partial": t["partial"], "reason": t["partial_reason"]}

    # a fully compared axis stays non-partial, so the flag still distinguishes something
    whole = {g: {"mean_distance": 0.1, "compared_steps": 8, "one_sided_steps": 0,
                 "unmeasurable_steps": 0, "unobserved_steps": 0} for g in GROUP_ORDER}
    w = R._axis_support(whole, R.AXIS_GROUPS["handshape"])
    assert w["partial"] is False and w["time_fraction_min"] == 1.0, w

    # This used to assert the colour rule 'v.status === "IN_BAND" ? "ok"', so that a partial axis
    # could not be painted as a pass. Colour was removed from the axis line entirely in a later
    # stage, which satisfies the same concern more strongly: nothing about the status reaches a
    # class name. The guard now checks that, rather than a code path that no longer exists.
    js = open(os.path.join(ROOT, "web", "app.js"), encoding="utf-8").read()
    axis_block = js.split("const hsLine = bars +")[1].split("const nnLine")[0]
    assert 'const cls = "";' in axis_block, "the axis line grew a class again; re-check colouring"
    assert '"ok"' not in axis_block and '"err"' not in axis_block,         "a verdict colour reappeared on the axis line"
    assert "v.status" in axis_block, "the axis line still reads the status it must not colour"
    assert "IN_BAND_PARTIAL_SUPPORT" in js, "the partial status is not handled by the display"
    assert "IN_BAND_PARTIAL_SUPPORT" != "IN_BAND"
    MEASURED["axis_support"] = {"full": full["fraction"], "half": half["fraction"],
                                "half_status_would_be": "IN_BAND_PARTIAL_SUPPORT"}


@check("ranking stays word identity and execution_assessment stays UNCONFIGURED")
def _separation_unchanged():
    src = open(os.path.join(ROOT, "src", "wis_lab", "core", "recognize.py"),
               encoding="utf-8").read()
    assert "UNCONFIGURED" in src, "the execution assessment status disappeared"
    state = json.load(open(os.path.join(ROOT, "ops", "STATE.json"), encoding="utf-8"))
    assert state["execution_assessment_status"].startswith("UNCONFIGURED"), \
        state["execution_assessment_status"]
    # no threshold was touched by this change
    assert T.MIN_PATH_EVIDENCE_FRACTION == 0.35, T.MIN_PATH_EVIDENCE_FRACTION
    assert T.ONE_SIDED_PENALTY == 1.5, T.ONE_SIDED_PENALTY
    MEASURED["unchanged_constants"] = {"MIN_PATH_EVIDENCE_FRACTION": T.MIN_PATH_EVIDENCE_FRACTION,
                                       "ONE_SIDED_PENALTY": T.ONE_SIDED_PENALTY,
                                       "runtime_version": T.RUNTIME_VERSION}


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"{NL}{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    print(json.dumps(MEASURED, ensure_ascii=False, indent=1, default=float))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
