"""R10: the hand-position trajectory along the shared warping path.

Checks the contract in design/functional_06_location_path.md: body-normalised coordinates kept
apart from the hand-local ones, each input's own relative time, and above all that a line is never
drawn across a gap - including a gap that fell BETWEEN two sampled rows and so left no trace in
the sample itself.

Synthetic sequences plus the frozen reference set. No training, no threshold, no camera.

Run: py -3.13 tests/test_r10_location_path.py
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
from test_r7_joint_support import IDX, seq  # noqa: E402

from wis_lab.core import features as F  # noqa: E402
from wis_lab.core import trajectory as T  # noqa: E402
from wis_lab.core.dtw import _SLICES  # noqa: E402
from wis_lab.core.features import GROUP_ORDER  # noqa: E402

CHECKS = []
MEASURED = {"fixture": "synthetic sequences plus the frozen reference set; not a measurement"}
NL = chr(10)
BUDGET = int(CONTRACT["sampling_A"]["budget"])


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def real_pair(word="ALDA", ref_word=None):
    s = F.sequence_features(obs_for(word))
    q = F.uniform_time_sample(s, BUDGET)
    ref = next(r for r in REFS if r["word_id"] == (ref_word or word))
    return s, q, ref


def timed(n=8, step_us=33000):
    """A synthetic sample with its own clock and its own picked rows."""
    d = seq(n)
    d["timestamp_us"] = np.arange(n, dtype=np.int64) * step_us + 5_000_000
    d["picked_rows"] = np.arange(n)
    return d


@check("positions are body units and are never the hand-local ones")
def _space():
    s, q, ref = real_pair()
    lp = T.location_path(q, ref, CONTRACT, query_source=T.source_observation(s))
    assert lp["status"] == "OK", lp
    assert lp["space"] == "body_normalised_shoulder_width"
    assert "shoulder width" in lp["unit"]
    assert "hand-local" in lp["not_mixed_with"]
    assert "FRONT-VIEW ORIGINAL VIDEO" in lp["axes"]
    # the coordinate frame and the hand labels are two different claims
    assert "anatomical" in lp["side_labels"] and "_assign_sides" in lp["side_labels"]
    assert "not the left and right of the video frame" in lp["side_labels"]
    # every plotted point comes from the handpos slices, not from shape_*
    for side in ("L", "R"):
        sl = _SLICES[f"handpos_{side}"]
        gi = GROUP_ORDER.index(f"handpos_{side}")
        for t, p in enumerate(lp["points"]["query"][side]):
            if p is None:
                continue
            xy = q["features"][t, sl]
            assert abs(p[0] - float(xy[0])) < 1e-4 and abs(p[1] - float(xy[1])) < 1e-4
            assert bool(q["present"][t, gi])
    MEASURED["space"] = {"query_frames": len(lp["points"]["query"]["L"]),
                         "steps": lp["path_pairs"], "compared": lp["steps_compared"]}


@check("each input's time is its own elapsed time, never a difference between the two clocks")
def _time():
    a, b = timed(), timed()
    b["timestamp_us"] = b["timestamp_us"] + 900_000_000        # a completely different clock
    lp = T.location_path(a, b, CONTRACT, query_source=T.source_observation(a))
    assert lp["relative_time_s"]["query"][0] == 0.0
    assert lp["relative_time_s"]["reference"][0] == 0.0, "the reference kept an absolute offset"
    assert lp["relative_time_s"]["query"] == lp["relative_time_s"]["reference"]
    for st in lp["steps"]:
        for k in ("query", "reference"):
            assert st["relative_time_s"][k] >= 0
    assert "never subtracted from each other" in lp["time_rule"]
    assert "not the real" in lp["time_rule"]
    MEASURED["time"] = {"query_last_s": lp["relative_time_s"]["query"][-1],
                        "reference_last_s": lp["relative_time_s"]["reference"][-1]}


@check("a line is never drawn to a point that was not observed")
def _endpoint_gap():
    a, b = timed(), timed()
    a["present"][3, IDX["handpos_L"]] = False
    lp = T.location_path(a, b, CONTRACT, query_source=T.source_observation(a))
    segs = lp["segments"]["query"]["L"]
    assert lp["points"]["query"]["L"][3] is None, "an unobserved point was plotted"
    assert segs["connect"][2] is False and segs["connect"][3] is False
    assert segs["reason"][2] == "ENDPOINT_NOT_OBSERVED"
    assert segs["connect"][0] is True
    for st in lp["steps"]:
        if st["i"] == 3:
            assert st["observed"]["L"]["query"] is False
            assert st["difference"]["L"] is None and st["distance"]["L"] is None


@check("a gap that fell BETWEEN two sampled rows is not bridged either")
def _dropped_gap():
    # 9 original rows, every other one sampled: the dropout at row 3 never reaches the sample
    full = seq(9)
    full["timestamp_us"] = np.arange(9, dtype=np.int64) * 33000
    full["present"][3, IDX["handpos_L"]] = False
    picks = np.array([0, 2, 4, 6, 8])
    sample = {"features": full["features"][picks], "present": full["present"][picks],
              "timestamp_us": full["timestamp_us"][picks], "picked_rows": picks}
    # the sample itself shows nothing wrong
    assert all(sample["present"][:, IDX["handpos_L"]]), "the fixture leaked the gap into the sample"
    src = T.source_observation(full)
    lp = T.location_path(sample, timed(5), CONTRACT, query_source=src)
    segs = lp["segments"]["query"]["L"]
    assert segs["checked"] is True
    assert segs["connect"][1] is False, "a line was drawn over a dropout between two picks"
    assert segs["reason"][1] == "GAP_IN_ORIGINAL"
    assert segs["connect"][0] is True and segs["connect"][2] is True
    MEASURED["dropped_gap"] = {"picked_rows": picks.tolist(), "connect": segs["connect"],
                               "drawn": segs["drawn_segments"], "possible": segs["possible_segments"]}


@check("a present row with a non-finite hand position counts as a break, not as observation")
def _nonfinite_is_a_break():
    # present stays True and only the coordinate is spoiled: the row is still unplottable, so the
    # continuity flag must treat it exactly like an absent one
    full = seq(9)
    full["timestamp_us"] = np.arange(9, dtype=np.int64) * 33000
    sl = _SLICES["handpos_L"]
    full["features"][3, sl.start] = np.nan
    full["features"][5, sl.start + 1] = np.inf
    assert bool(full["present"][3, IDX["handpos_L"]]) and bool(full["present"][5, IDX["handpos_L"]])
    src = T.source_observation(full)
    assert src["handpos_present"]["L"][3] is False, "a NaN row was counted as observed"
    assert src["handpos_present"]["L"][5] is False, "an Inf row was counted as observed"
    assert src["handpos_present"]["L"][4] is True
    picks = np.array([0, 2, 4, 6, 8])
    sample = {"features": full["features"][picks], "present": full["present"][picks],
              "timestamp_us": full["timestamp_us"][picks], "picked_rows": picks}
    lp = T.location_path(sample, timed(5), CONTRACT, query_source=src)
    segs = lp["segments"]["query"]["L"]
    assert segs["connect"][1] is False and segs["reason"][1] == "GAP_IN_ORIGINAL"
    assert segs["connect"][2] is False and segs["reason"][2] == "GAP_IN_ORIGINAL"
    assert segs["connect"][0] is True and segs["connect"][3] is True
    MEASURED["nonfinite_break"] = {"connect": segs["connect"],
                                   "drawn": segs["drawn_segments"],
                                   "possible": segs["possible_segments"]}


@check("a side whose original is unavailable is not drawn as continuous")
def _unchecked_side():
    a, b = timed(), timed()
    lp = T.location_path(a, b, CONTRACT)          # no query_source, and b carries no source either
    for side in ("L", "R"):
        for who in ("query", "reference"):
            segs = lp["segments"][who][side]
            assert segs["checked"] is False
            assert not any(segs["connect"]), "an unchecked side was joined up anyway"
            assert set(segs["reason"]) == {"ORIGINAL_NOT_AVAILABLE"}
    assert "not drawn as continuous" in lp["continuity_rule"]


@check("a built reference carries its source timeline without changing what it already had")
def _reference_carries_source():
    from wis_lab.core.recognize import build_reference  # noqa: PLC0415

    obs = obs_for("ALDA")
    s = F.sequence_features(obs)
    plain = F.uniform_time_sample(s, BUDGET)
    built = build_reference(obs, "clip_test", "ALDA", BUDGET)
    # unobserved slots are NaN, and NaN != NaN, so the comparison has to allow it
    assert np.array_equal(built["features"], plain["features"], equal_nan=True)
    assert np.array_equal(built["present"], plain["present"])
    assert np.array_equal(built["timestamp_us"], plain["timestamp_us"])
    src = built["source_observation"]
    assert src["rows"] == len(s["present"])
    for side in ("L", "R"):
        gi = GROUP_ORDER.index(f"handpos_{side}")
        assert src["handpos_present"][side] == s["present"][:, gi].tolist()
    lp = T.location_path(plain, built, CONTRACT, query_source=T.source_observation(s))
    assert lp["segments"]["reference"]["R"]["checked"] is True
    MEASURED["reference_source"] = {
        "rows": src["rows"],
        "observed_R": sum(src["handpos_present"]["R"]),
        "observed_L": sum(src["handpos_present"]["L"]),
        "reference_drawn_segments_R": lp["segments"]["reference"]["R"]["drawn_segments"]}


@check("the difference vector is the two positions subtracted, and carries no verdict")
def _difference():
    s, q, ref = real_pair("ALDA", "NA")
    lp = T.location_path(q, ref, CONTRACT, query_source=T.source_observation(s))
    hits = 0
    for st in lp["steps"]:
        for side in ("L", "R"):
            if not st["observed"][side]["both"]:
                assert st["difference"][side] is None
                continue
            a = lp["points"]["query"][side][st["i"]]
            b = lp["points"]["reference"][side][st["j"]]
            assert abs(st["difference"][side][0] - round(a[0] - b[0], 4)) < 1e-9
            assert abs(st["difference"][side][1] - round(a[1] - b[1], 4)) < 1e-9
            assert abs(st["distance"][side] - round(float(np.hypot(a[0] - b[0],
                                                                   a[1] - b[1])), 4)) < 1e-9
            hits += 1
    assert hits > 0, "no step compared both sides"
    assert lp["is_a_verdict"] is False and "never whether the movement was correct" in lp["note"]
    MEASURED["difference"] = {"compared_side_steps": hits,
                              "steps_compared": lp["steps_compared"]}


@check("each step carries the hand shape of that same pair, in hand-local units")
def _step_shape():
    s, q, ref = real_pair("ALDA", "NA")
    lp = T.location_path(q, ref, CONTRACT, query_source=T.source_observation(s))
    assert lp["step_handshape"] is True
    assert lp["shape_space"] == "hand_local_wrist_normalised"
    seen = 0
    for st in lp["steps"]:
        for side in ("L", "R"):
            blk = st["shape"][side]
            if not blk["observed"]["both"]:
                assert blk["query"] is None or blk["reference"] is None
                assert blk["difference"] is None
                continue
            sl = _SLICES[f"shape_{side}"]
            for who, src, t in (("query", q, st["i"]), ("reference", ref, st["j"])):
                v = src["features"][t, sl]
                for k, pt in enumerate(blk[who]):
                    assert abs(pt[0] - float(v[2 * k])) < 1e-4, (who, side, k)
                    assert abs(pt[1] - float(v[2 * k + 1])) < 1e-4, (who, side, k)
            assert len(blk["query"]) == T.SHAPE_JOINTS and len(blk["difference"]) == 20
            seen += 1
    assert seen > 0, "no step compared a hand shape"
    MEASURED["step_shape"] = {"compared_side_steps": seen,
                              "steps": lp["path_pairs"],
                              "observed": {s2: lp["shape_view"][s2]["steps_observed"]
                                           for s2 in ("L", "R")}}


@check("the hand-shape scale is one fixed box per hand, and the wrist body coordinate is not in it")
def _fixed_scale():
    s, q, ref = real_pair("ALDA", "NA")
    lp = T.location_path(q, ref, CONTRACT, query_source=T.source_observation(s))
    for side in ("L", "R"):
        view = lp["shape_view"][side]
        assert view["steps_total"] == lp["path_pairs"]
        assert 0 < view["steps_observed"] <= view["steps_total"]
        lo, hi = view["min"], view["max"]
        # every observed point of BOTH sides lies inside the one box the slider will use
        for st in lp["steps"]:
            blk = st["shape"][side]
            if not blk["observed"]["both"]:
                continue
            for who in ("query", "reference"):
                for pt in blk[who]:
                    assert lo[0] - 1e-6 <= pt[0] <= hi[0] + 1e-6, (side, who, pt)
                    assert lo[1] - 1e-6 <= pt[1] <= hi[1] + 1e-6, (side, who, pt)
        # the body-normalised wrist is a different space and never appears among these points
        wrist = lp["points"]["query"][side]
        for st in lp["steps"]:
            blk = st["shape"][side]
            w = wrist[st["i"]]
            if blk["observed"]["both"] and w is not None:
                assert all(pt != w for pt in blk["query"]), "a body coordinate leaked into shape"
    assert "ONE scale per hand" in lp["shape_scale_rule"]
    assert "never mixed in" in lp["shape_unit"]
    MEASURED["fixed_scale"] = {s2: lp["shape_view"][s2] for s2 in ("L", "R")}


@check("the step shape rides on the path already walked, and joint_deviation is unchanged")
def _no_recompute():
    from wis_lab.core.joints import joint_deviation, joint_distances  # noqa: PLC0415

    s, q, ref = real_pair("ALDA", "NA")
    lp = T.location_path(q, ref, CONTRACT, query_source=T.source_observation(s))
    cost_cfg = {k: v for k, v in CONTRACT.get("cost", {}).items()
                if k in ("one_sided_penalty", "one_sided_mode", "group_weights",
                         "hand_participation")}
    raw = joint_distances(q, ref, cost_cfg)
    assert lp["path_pairs"] == raw["path_pairs"], "a different path was walked"
    dev = joint_deviation(q, ref, CONTRACT)
    # every field the representative-frame drawing and the support report rely on is still there
    for key in ("deviation", "support", "frame_features", "reference_frame_features",
                "frame_joint_deviation", "frame_comparison", "frame_spaces", "frame_pair", "unit"):
        assert key in dev, key
    assert dev["support"]["joints_expected"] == 46
    MEASURED["no_recompute"] = {"path_pairs": lp["path_pairs"],
                                "joint_deviation_frame_pair": dev["frame_pair"],
                                "joints_measured": dev["support"]["joints_measured"]}


@check("recognition is untouched by the added diagnostic")
def _recognition_untouched():
    from wis_lab.core.recognize import predict  # noqa: PLC0415

    obs = obs_for("ALDA")
    res = predict(obs, REFS, CONTRACT)
    assert res["observation_status"] == "OK"
    ranking = [(r["word_id"], r["distance"]) for r in res["ranking"]]
    # the references now carry source_observation; the ranking must not know about it
    stripped = [{k: v for k, v in r.items() if k != "source_observation"} for r in REFS]
    again = predict(obs, stripped, CONTRACT)
    assert [(r["word_id"], r["distance"]) for r in again["ranking"]] == ranking
    MEASURED["recognition_untouched"] = {"top1": res["top1_word_id"],
                                         "identical_without_the_added_key": True}


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
