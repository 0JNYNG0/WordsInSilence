"""functional_24: what each hand contributed to an axis mean, and what that does NOT mean.

Synthetic fixtures only. No training, no participant replay, no recording is opened, no camera,
no server. Nothing here changes a score, a band or a threshold.

Run: py -3.13 tests/test_r21_group_evidence.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from wis_lab.core.recognize import AXIS_GROUPS, _axis_support  # noqa: E402

FAILS = []


def ok(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def group(mean=None, compared=0, unobserved=0, one_sided=0, unmeasurable=0):
    """Shaped as dtw_group_residuals writes it: mean_distance is always present, None when the
    group produced nothing."""
    return {"compared_steps": compared, "unobserved_steps": unobserved,
            "one_sided_steps": one_sided, "unmeasurable_steps": unmeasurable,
            "mean_distance": mean}


def axis_mean(groups, support):
    """The aggregate exactly as recognize.py computes it, so the test cannot drift from it."""
    return float(np.mean([groups[g]["mean_distance"] for g in support["measured_groups"]]))


# ------------------------------------------------------- 1. the reported case, reconstructed
def test_uneven_support_reconstructs_the_existing_mean():
    """Root's abstract reproducer: one hand seen once, the other twenty-seven times."""
    groups = {"bend_L": group(0.36627, compared=1, unobserved=32),
              "bend_R": group(0.15999, compared=27, unobserved=6)}
    sup = _axis_support(groups, AXIS_GROUPS["handshape"])
    per = sup["per_group"]

    ok(round(axis_mean(groups, sup), 5) == 0.26313,
       "the axis mean is unchanged: 0.36627 and 0.15999 still average to 0.26313")
    ok(per["bend_L"]["mean_distance"] == 0.36627
       and per["bend_R"]["mean_distance"] == 0.15999,
       "each hand's own distance is now visible")
    ok(per["bend_L"]["aggregate_weight"] == per["bend_R"]["aggregate_weight"] == 0.5,
       "and both carry the same 1/2 weight, however much evidence each rests on")
    ok(per["bend_L"]["compared_steps"] == 1 and per["bend_L"]["path_steps"] == 33
       and per["bend_R"]["compared_steps"] == 27 and per["bend_R"]["path_steps"] == 33,
       "the counts say the evidence is not equal: 1 of 33 against 27 of 33")

    # the reconstruction is exact, from the published numbers alone
    rebuilt = sum(per[g]["mean_distance"] * per[g]["aggregate_weight"]
                  for g in sup["measured_groups"])
    ok(round(rebuilt, 5) == 0.26313,
       "a reader can rebuild the published mean from the per-group numbers")

    ok(sup["aggregation"]["rule"].startswith("equal mean"),
       "the aggregation names itself as an equal mean of measured group means")
    ok("DTW path or ranking group weights" in sup["aggregation"]["is_not"],
       "and says it is not the DTW path or ranking weights")
    ok(sup["partial"] is True and sup["no_pass_line"],
       "PARTIAL_SUPPORT and the absence of a pass line are unchanged")


# ------------------------------------------------------- 2. a missing group is not a zero
def test_missing_groups_are_excluded_not_zero():
    groups = {"bend_R": group(0.2, compared=10)}          # the left hand produced nothing
    sup = _axis_support(groups, AXIS_GROUPS["handshape"])
    per = sup["per_group"]

    ok(per["bend_L"]["measured"] is False and per["bend_L"]["mean_distance"] is None,
       "an unmeasured hand has no distance at all, not a zero")
    ok(per["bend_L"]["aggregate_weight"] == 0.0,
       "and no weight, so it cannot pull the mean toward zero")
    ok(per["bend_R"]["aggregate_weight"] == 1.0,
       "the one measured hand carries the whole mean")
    ok(round(axis_mean(groups, sup), 5) == 0.2,
       "so the axis mean is the measured hand's own value, exactly as before")
    ok(sup["aggregation"]["missing_groups_contribute"].startswith("nothing"),
       "the payload says a missing group contributes nothing")
    ok("not a zero difference" in sup["aggregation"]["missing_groups_contribute"],
       "and that it is not a zero difference")
    ok(sup["partial"] is True and "missing_groups" in sup["partial_reason"],
       "the axis stays partial, for the stated reason")


def test_nothing_measured_stays_unmeasurable():
    groups = {"bend_L": group(compared=0, unobserved=20), "bend_R": group(compared=0,
                                                                          unobserved=20)}
    sup = _axis_support(groups, AXIS_GROUPS["handshape"])
    ok(sup["measured_groups"] == [], "with nothing measured there is no measured group")
    ok(all(sup["per_group"][g]["mean_distance"] is None for g in ("bend_L", "bend_R")),
       "no distance is invented for either hand")
    ok(sup["aggregation"]["weight_per_measured_group"] == 0.0
       and sup["aggregation"]["measured_group_count"] == 0,
       "and no weight is defined, because there is nothing to weight")
    ok(sup["partial"] is False,
       "unmeasurable is not 'partial': the axis produced nothing to be partial about")


def test_a_four_group_axis_splits_evenly():
    """Three measured groups: the coefficient must be the exact 1/3, not a rounded 0.333333.

    round(1/3, 6) three times sums to 0.999999, and rebuilding 0.1/0.2/0.3 from it gives
    0.1999998 instead of 0.2 - a published number that does not add up to the published mean.
    """
    import math
    groups = {"handpos_L": group(0.1, compared=5), "handpos_R": group(0.2, compared=5),
              "facerel_R": group(0.3, compared=5)}       # facerel_L absent
    sup = _axis_support(groups, AXIS_GROUPS["location"])
    per = sup["per_group"]
    ok(len(sup["measured_groups"]) == 3
       and all(per[g]["aggregate_weight"] == 1 / 3 for g in sup["measured_groups"]),
       "three measured groups each carry the exact one third")
    ok(sum(per[g]["aggregate_weight"] for g in sup["measured_groups"]) == 1.0,
       "so the weights sum to exactly one, not to 0.999999")
    ok(per["facerel_L"]["aggregate_weight"] == 0.0, "the absent one carries none")

    mean = axis_mean(groups, sup)
    rebuilt = sum(per[g]["mean_distance"] * per[g]["aggregate_weight"]
                  for g in sup["measured_groups"])
    ok(math.isclose(rebuilt, mean, rel_tol=1e-12),
       "and the reconstruction matches the unrounded mean to floating precision")
    ok(abs(rebuilt - 0.2) < 1e-12 and round(rebuilt, 7) != 0.1999998,
       "0.1/0.2/0.3 rebuild to 0.2, not to the truncated 0.1999998")
    ok(sup["aggregation"]["weight_per_measured_group"] == 1 / 3,
       "the aggregation block carries the exact coefficient too")
    ok("rounded for display" in sup["aggregation"]["weight_precision"],
       "and says the published distance field is the rounded one")
    ok(sup["aggregation"]["validated"].startswith("no"),
       "the aggregation states that its suitability for judging an execution is not established")


# ------------------------------------------------------- 3. old records still read
def test_old_payloads_remain_readable():
    """A record written before this stage has no mean_distance or weight in per_group."""
    old = {"expected_groups": ["bend_L", "bend_R"], "measured_groups": ["bend_R"],
           "missing_groups": ["bend_L"], "partial": True,
           "per_group": {"bend_L": {"measured": False, "compared_steps": 0, "path_steps": 12,
                                    "time_fraction": 0.0},
                         "bend_R": {"measured": True, "compared_steps": 12, "path_steps": 12,
                                    "time_fraction": 1.0}}}
    js = open(os.path.join(ROOT, "web", "app.js"), encoding="utf-8").read()
    ok("d.mean_distance === null || d.mean_distance === undefined" in js,
       "the screen treats an absent per-group distance as unknown")
    ok("d.aggregate_weight === null || d.aggregate_weight === undefined" in js,
       "and an absent weight the same way")
    ok("알 수 없음" in js, "shown as unknown, never as zero")
    ok(old["per_group"]["bend_R"].get("mean_distance") is None,
       "the old record genuinely carries no distance, which is what makes that path necessary")


# ------------------------------------------------------- 4. the screen
def affirmative_uses(text, word):
    """sentences of text that contain word WITHOUT negating it (않 / 아니)"""
    import re
    parts = re.split(r"[.`<>\n]", text)
    return [p for p in parts if word in p and not ("않" in p or "아니" in p)]


def test_ui_renders_the_breakdown():
    js = open(os.path.join(ROOT, "web", "app.js"), encoding="utf-8").read()
    block = js.split("function groupBreakdown(")[1].split("const hsLine")[0]

    ok("groupBreakdown(sup)" in js, "the breakdown is attached to the axis line")
    for g, name in (("bend_L", "왼손 손가락 굽힘"), ("bend_R", "오른손 손가락 굽힘"),
                    ("handpos_L", "왼손 위치"), ("arm_R", "오른팔"),
                    ("motion_L", "왼손 이동"), ("facerel_R", "오른손 얼굴기준 위치")):
        ok(f"{g}: \"{name}\"" in js, f"{g} is named in Korean by side and feature type")
    ok("d.compared_steps}/${d.path_steps" in block,
       "the actual compared and path counts are drawn, not a fraction alone")
    ok("aggregate_weight * 100" in block, "and each group's share of the mean")
    ok("한 구간만 관측된 항목도 평균에 그대로 들어갑니다" in block,
       "the screen says a single observed interval still enters the mean")
    ok("틀렸다는 증거도" in block and "맞다는 증거도 아닙니다" in block,
       "and that a partial comparison proves neither an error nor correctness")
    ok("영상의 서로 다른 프레임 수가" in block,
       "matched pairs are not described as distinct video frames")
    ok("차이 0도, 잘한 것도, 안 쓰는 손도 아닙니다" in block,
       "a missing hand is not a zero, not correct, and not an unused hand")
    ok("판정이 아닙니다" in block, "and the table says it is an explanation, not a verdict")

    # The screen must not CLAIM a pass or a correct answer. A sentence that says it does NOT judge
    # correctness ("정답·오답을 판정하지 않습니다") is the disclaimer this test wants, so a word is
    # allowed only inside a sentence that negates it. The badge class stays strictly forbidden.
    for wrong in ("통과", "합격", "정답"):
        ok(not affirmative_uses(block, wrong),
           f"the breakdown makes no affirmative '{wrong}' claim (negated disclaimers allowed)")
    ok("class=\"hs ok\"" not in block, "the breakdown introduces no pass badge class")
    # the narrowed check must still catch what it exists to catch
    ok(affirmative_uses("`<b>정답입니다</b>`", "정답") and affirmative_uses("`합격 표시`", "합격")
       and not affirmative_uses("`이 숫자로 정답·오답을 판정하지 않습니다.`", "정답"),
       "self-check: an affirmative claim is flagged, a negated disclaimer is not")
    ok("bar-fill" not in block, "and no new coloured bar or threshold")


def test_no_threshold_or_scoring_change():
    src = open(os.path.join(ROOT, "src", "wis_lab", "core", "recognize.py"),
               encoding="utf-8").read()
    ok('d = float(np.mean(vals))' in src,
       "the axis distance is still the plain mean of the measured group means")
    ok("no threshold decides whether" in src, "and no pass line was introduced")
    ok("has NOT been validated" in src and "That is not wrong" not in src,
       "the docstring no longer calls the existing aggregation right; it says it is unvalidated")
    ok("coverage threshold has been validated" in src or "no_pass_line" in src,
       "the payload still states that no coverage threshold is validated")

    # 2026-09-26 observed_difference_only_v1: this used to require the personal band and its
    # headroom to remain. Removing them is now the intended change, so the condition is inverted;
    # the mean, support and no-pass-line checks around it are unchanged.
    ok("band_max" not in src and "headroom" not in src,
       "no personal band limit or headroom is computed by the axis check")

    # the payload keeps every key it had before this stage: an older reader must still work
    sup = _axis_support({"bend_L": group(0.4, compared=2), "bend_R": group(0.2, compared=8)},
                        AXIS_GROUPS["handshape"])
    for key in ("expected_groups", "measured_groups", "missing_groups", "group_fraction",
                "fraction", "time_fraction_min", "time_fraction_mean", "time_fraction_per_group",
                "partial", "partial_reason", "no_pass_line", "per_group"):
        ok(key in sup, f"the support payload still carries {key}")
    for key in ("measured", "compared_steps", "path_steps", "time_fraction", "one_sided_steps",
                "unmeasurable_steps", "unobserved_steps"):
        ok(key in sup["per_group"]["bend_L"], f"and each group still carries {key}")


def test_architecture_doc_is_current():
    raw = open(os.path.join(ROOT, "architecture.md"), encoding="utf-8").read()
    doc = " ".join(raw.split())          # the prose wraps; match the sentence, not the layout
    ok("union93_candidate_v1" in raw and "union93_visibility_v1" in raw,
       "the doc names both opt-in candidates")
    ok("80 AIHub words" in doc and "93 words covering 18 of the 19" in doc,
       "and the actual coverage: base 80 with five shared, candidates 93 covering 18")
    ok("Quality validation is still pending" in doc,
       "it says quality validation is still pending")
    ok("686/696" in doc and "18/18" in doc and "17/18" in doc,
       "the three figures are all present")
    ok("held-out-for-regression SOURCE validation" in doc
       and "TRAINING REFERENCE checks" in doc
       and "ONE-PERSON DEVELOPMENT trials" in doc,
       "and each is labelled as its own kind of evidence, not grouped together")
    ok("not added together" in doc, "with an explicit statement that they are not combined")


if __name__ == "__main__":
    test_uneven_support_reconstructs_the_existing_mean()
    test_missing_groups_are_excluded_not_zero()
    test_nothing_measured_stays_unmeasurable()
    test_a_four_group_axis_splits_evenly()
    test_old_payloads_remain_readable()
    test_ui_renders_the_breakdown()
    test_no_threshold_or_scoring_change()
    test_architecture_doc_is_current()
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
