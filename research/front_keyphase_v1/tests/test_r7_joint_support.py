"""R7: per-joint observation support in joint_deviation.

A joint compared on ONE pair of the warping path used to arrive with a mean and a status of OK,
indistinguishable from a joint compared on every pair. The first check reproduces that against the
old counting rule before requiring the new one.

Synthetic sequences only, plus the frozen reference set for a realistic shape. No training, no
threshold change, no new display.

Run: py -3.13 tests/test_r7_joint_support.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from test_r2_handshape import CONTRACT, REFS, obs_for  # noqa: E402

from wis_lab.core import features as F  # noqa: E402
from wis_lab.core import joints as J  # noqa: E402
from wis_lab.core.dtw import _SLICES  # noqa: E402
from wis_lab.core.features import FEATURE_WIDTH, GROUP_ORDER  # noqa: E402

CHECKS = []
MEASURED = {"fixture": "synthetic sequences plus the frozen reference set; not a measurement"}
NL = chr(10)
G = len(GROUP_ORDER)
IDX = {g: i for i, g in enumerate(GROUP_ORDER)}


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def seq(n=8, value=0.0):
    return {"features": np.full((n, FEATURE_WIDTH), float(value), dtype=np.float32),
            "present": np.ones((n, G), dtype=bool)}


def q_ref(word="ALDA"):
    """A real query/reference pair in the shape joint_deviation expects."""
    s = F.sequence_features(obs_for(word))
    q = F.uniform_time_sample(s, int(CONTRACT["sampling_A"]["budget"]))
    ref = next(r for r in REFS if r["word_id"] == word)
    return q, ref


@check("the old rule counted a joint seen once exactly like one seen throughout")
def _reproduce():
    a, b = seq(), seq()
    group = J.LANDMARK_MAP[33][0]                    # the left wrist's group
    b["present"][1:, IDX[group]] = False             # comparable on one pair only
    res = J.joint_distances(a, b, {})
    compared = res["compared_pairs"][33]
    assert compared == 1, compared
    # the old report exposed only the mean, with nothing to say it rested on a single pair
    assert 33 in res["per_joint"] and res["per_joint"][33] == 0.0
    dev = J.joint_deviation(a, b, CONTRACT)
    assert dev["status"] == "OK"
    assert dev["deviation"][33] == 0.0, "the distance itself must not change"
    sup = dev["support"]["per_joint"][33]
    assert sup["compared_pairs"] == 1 and sup["path_pairs"] == res["path_pairs"]
    assert sup["time_fraction"] < 0.2, sup
    MEASURED["single_pair_joint"] = {"deviation": dev["deviation"][33], **sup}


@check("a joint that was never comparable is listed unmeasured, never filled with zero")
def _unmeasured_not_zero():
    a, b = seq(), seq()
    group = J.LANDMARK_MAP[54][0]                    # the right wrist's group
    b["present"][:, IDX[group]] = False
    dev = J.joint_deviation(a, b, CONTRACT)
    assert 54 not in dev["deviation"], "an unmeasured joint got a number"
    sup = dev["support"]["per_joint"][54]
    assert sup["measured"] is False and sup["reason"] == "NO_COMPARABLE_PAIR"
    assert sup["compared_pairs"] == 0
    assert 54 in dev["support"]["joints_unmeasured"]
    assert dev["support"]["partial"] is True
    assert "unmeasured_joints" in dev["support"]["partial_reason"]


@check("comparability follows the whole-group finiteness rule, as the DTW path does")
def _group_finiteness():
    a, b = seq(), seq()
    group, off = J.LANDMARK_MAP[34]                  # a finger point, group shape_L
    sl = _SLICES[group]
    # one OTHER component of the same group is NaN: the group is not wholly finite, so no joint
    # inside it counts, even though this joint's own two dimensions are fine
    other = sl.start + (off + 4) % (sl.stop - sl.start)
    b["features"][:, other] = np.nan
    res = J.joint_distances(a, b, {})
    for lm, (g, _) in J.LANDMARK_MAP.items():
        if g == group:
            assert res["compared_pairs"][lm] == 0, (lm, res["compared_pairs"][lm])
    dev = J.joint_deviation(a, b, CONTRACT)
    assert 34 not in dev["deviation"]
    assert dev["support"]["per_joint"][34]["reason"] == "NO_COMPARABLE_PAIR"
    MEASURED["group_finiteness"] = {"group": group,
                                    "joints_unmeasured": len(dev["support"]["joints_unmeasured"])}


@check("the drawn frame says whether each joint was actually compared on that pair")
def _frame_comparability():
    a, b = seq(), seq()
    group = J.LANDMARK_MAP[33][0]
    b["present"][:, IDX[group]] = False              # the reference never carries this group
    dev = J.joint_deviation(a, b, CONTRACT)
    assert 33 in dev["frame_features"], "the query's own joint should still be drawable"
    # the original shape is preserved for existing consumers
    assert isinstance(dev["frame_features"][33], list) and len(dev["frame_features"][33]) == 2
    cmp_map = dev["frame_comparison"]["per_joint"]
    assert cmp_map[33] is False, "an uncompared joint was marked compared"
    other = next(lm for lm in dev["frame_features"] if lm != 33)
    assert cmp_map[other] is True
    assert dev["frame_comparison"]["compared"] < dev["frame_comparison"]["drawn"]
    assert dev["frame_pair"][0] == dev["frame_index"]
    MEASURED["frame"] = {"pair": dev["frame_pair"],
                         "drawn": dev["frame_comparison"]["drawn"],
                         "compared": dev["frame_comparison"]["compared"]}


@check("a fully observed pair reports no partial support")
def _full_support():
    a = seq()
    dev = J.joint_deviation(a, a, CONTRACT)
    sup = dev["support"]
    assert sup["joints_measured"] == sup["joints_expected"], sup
    assert sup["time_fraction_min"] == 1.0, sup
    assert sup["partial"] is False and sup["partial_reason"] == []
    assert all(v == 0.0 for v in dev["deviation"].values())


@check("a real clip against its own reference reports its support honestly")
def _real_pair():
    q, ref = q_ref("ALDA")
    dev = J.joint_deviation(q, ref, CONTRACT)
    assert dev["status"] == "OK", dev
    sup = dev["support"]
    assert sup["path_pairs"] > 0
    assert 0 < sup["joints_measured"] <= sup["joints_expected"]
    assert set(dev["deviation"]) == {lm for lm in sup["per_joint"]
                                     if sup["per_joint"][lm]["measured"]}
    for lm, e in sup["per_joint"].items():
        assert e["compared_pairs"] <= e["path_pairs"]
        assert e["measured"] == (lm in dev["deviation"])
    MEASURED["real_pair"] = {"path_pairs": sup["path_pairs"],
                             "measured": sup["joints_measured"],
                             "expected": sup["joints_expected"],
                             "time_fraction_min": sup["time_fraction_min"],
                             "partial": sup["partial"],
                             "partial_reason": sup["partial_reason"]}


@check("the unit text matches what the code divides by, and no threshold moved")
def _unit_and_no_threshold():
    q, ref = q_ref("ALDA")
    dev = J.joint_deviation(q, ref, CONTRACT)
    assert "GROUP" in dev["unit"], dev["unit"]
    assert "Per-joint denominators were not used" in dev["unit"]
    scales = CONTRACT["handshape_check"]["joint_scales"]
    # every scale key is a GROUP name, never a landmark id
    assert set(scales) <= set(GROUP_ORDER), sorted(set(scales) - set(GROUP_ORDER))
    lm = next(iter(dev["deviation"]))
    group = J.LANDMARK_MAP[lm][0]
    raw = J.joint_distances(q, ref, {k: v for k, v in CONTRACT.get("cost", {}).items()
                                     if k in ("one_sided_penalty", "one_sided_mode",
                                              "group_weights", "hand_participation")})
    assert abs(dev["deviation"][lm] - round(raw["per_joint"][lm] / scales[group], 3)) < 1e-9
    assert "verdict" not in dev["status"].lower()
    assert dev["support"]["no_pass_line"]
    MEASURED["unit"] = {"scale_keys_are_groups": True, "example_group": group}


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
    print(json.dumps(MEASURED, ensure_ascii=False, indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
