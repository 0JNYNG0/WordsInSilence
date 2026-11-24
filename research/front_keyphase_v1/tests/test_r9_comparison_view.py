"""R9: what the comparison view is allowed to draw.

The screen overlays the person's hand on the reference hand. For that to be honest the payload
has to carry BOTH sides of the drawn pair, the difference on that one pair, and which coordinate
space every landmark lives in - shape_* is hand-local, handpos_*/arm_* are body-normalised, and
mixing them would put a distance on screen that nothing measured.

Synthetic sequences plus the frozen reference set. No training, no threshold, no camera.

Run: py -3.13 tests/test_r9_comparison_view.py
"""
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from test_r2_handshape import CONTRACT, REFS, obs_for  # noqa: E402
from test_r7_joint_support import G, IDX, q_ref, seq  # noqa: E402

from wis_lab.core import joints as J  # noqa: E402
from wis_lab.core.dtw import _SLICES  # noqa: E402

CHECKS = []
MEASURED = {"fixture": "synthetic sequences plus the frozen reference set; not a measurement"}
NL = chr(10)
SCALES = CONTRACT["handshape_check"]["joint_scales"]


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("the drawn pair carries the reference side too, in the same [x, y] shape")
def _reference_side():
    q, ref = q_ref("ALDA")
    dev = J.joint_deviation(q, ref, CONTRACT)
    assert dev["status"] == "OK"
    rf = dev["reference_frame_features"]
    assert rf, "no reference frame to draw against"
    for lm, xy in rf.items():
        assert isinstance(xy, list) and len(xy) == 2, (lm, xy)
    # the existing query-side field keeps its shape
    for lm, xy in dev["frame_features"].items():
        assert isinstance(xy, list) and len(xy) == 2, (lm, xy)
    MEASURED["reference_side"] = {"query_drawn": len(dev["frame_features"]),
                                  "reference_drawn": len(rf)}


@check("the per-frame difference is the distance on THAT pair over the same group scale")
def _frame_difference():
    # a different word on the reference side, so the differences are not all zero
    q, _ = q_ref("ALDA")
    ref = next(r for r in REFS if r["word_id"] == "NA")
    dev = J.joint_deviation(q, ref, CONTRACT)
    i, j = dev["frame_pair"]
    fd = dev["frame_joint_deviation"]
    assert fd, "no per-frame difference"
    for lm, v in fd.items():
        group, off = J.LANDMARK_MAP[lm]
        s = _SLICES[group].start + off
        raw = float(np.linalg.norm(q["features"][i, s:s + 2] - ref["features"][j, s:s + 2]))
        assert abs(v - round(raw / SCALES[group], 3)) < 1e-9, (lm, v, raw)
    # a per-frame number is not the attempt's deviation, and the whole-path one is unchanged
    assert set(fd) <= set(dev["deviation"]), "a joint got a frame number without being measured"
    assert max(fd.values()) > 0, "a cross-word pair produced no difference at all"
    MEASURED["frame_difference"] = {"pair": [i, j], "joints": len(fd),
                                    "max": max(fd.values()), "min": min(fd.values())}


@check("a joint the reference did not carry is not drawn and gets no difference")
def _missing_reference_hand():
    a, b = seq(), seq()
    group = J.LANDMARK_MAP[55][0]                    # the right hand's shape group
    b["present"][:, IDX[group]] = False
    dev = J.joint_deviation(a, b, CONTRACT)
    for lm, (g, _) in J.LANDMARK_MAP.items():
        if g == group:
            assert lm not in dev["reference_frame_features"], lm
            assert lm not in dev["frame_joint_deviation"], lm
            assert dev["frame_comparison"]["per_joint"].get(lm) is False
            # the person's own hand is still drawable; it simply has nothing to compare with
            assert lm in dev["frame_features"], lm
    MEASURED["missing_reference_hand"] = {
        "group": group, "reference_drawn": len(dev["reference_frame_features"]),
        "query_drawn": len(dev["frame_features"])}


@check("hand-local and body-normalised coordinates are declared apart")
def _spaces():
    q, ref = q_ref("ALDA")
    sp = J.joint_deviation(q, ref, CONTRACT)["frame_spaces"]
    assert sp["per_group"]["shape_L"] == "hand_local_wrist_normalised"
    assert sp["per_group"]["shape_R"] == "hand_local_wrist_normalised"
    for g in ("handpos_L", "handpos_R", "arm_L", "arm_R"):
        assert sp["per_group"][g] == "body_normalised", g
    assert sp["panel_groups"] == {"L": "shape_L", "R": "shape_R"}
    # the wrists carry body units, so they are never points inside a hand panel
    assert sp["wrist_landmarks_are_not_panel_points"] == [33, 54]
    for lm, (g, _) in J.LANDMARK_MAP.items():
        assert sp["per_landmark"][lm] == sp["per_group"][g], lm
    panel = [lm for lm in sp["per_landmark"]
             if sp["per_landmark"][lm] == "hand_local_wrist_normalised"]
    assert len(panel) == 40, len(panel)       # 20 finger points per hand, no wrist
    MEASURED["spaces"] = {"panel_landmarks": len(panel)}


@check("a stored clip can be compared with a named reference without the recogniser seeing it")
def _explicit_comparison():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from wis_lab.service.app import _explicit_comparison  # noqa: PLC0415

    st = types.SimpleNamespace(refs=REFS, contract=CONTRACT)
    out = _explicit_comparison(obs_for("ALDA"), "NA", st)
    assert out["status"] == "OK", out
    assert out["reference_word_id"] == "NA"
    assert out["chosen_by"] == "explicit_ui_choice_not_the_recogniser"
    assert out["counts_as_human_attempt"] is False and out["is_a_verdict"] is False
    assert out["joint_deviation"]["status"] == "OK"
    assert out["axis_deviation"]["status"] in ("OK", "UNAVAILABLE")
    bad = _explicit_comparison(obs_for("ALDA"), "NOT_A_WORD", st)
    assert bad["status"] == "UNAVAILABLE" and bad["reason"] == "NO_SUCH_REFERENCE_WORD"
    MEASURED["explicit_comparison"] = {"reference_word_id": out["reference_word_id"],
                                       "joints": len(out["joint_deviation"]["deviation"])}


@check("the payload states it has no pass line and carries no correct/incorrect field")
def _no_verdict():
    q, ref = q_ref("ALDA")
    dev = J.joint_deviation(q, ref, CONTRACT)
    # the only mention of a pass line is the statement that there is none
    assert dev["support"]["no_pass_line"]
    assert "not a verdict" in dev["note"]
    # no field anywhere says whether the performance was right
    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                key = str(k).lower()
                for banned in ("is_correct", "correct", "passed", "verdict", "score", "grade"):
                    assert banned not in key, f"{path}/{k}"
                walk(v, path + "/" + str(k))
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
    walk(dev)
    MEASURED["no_verdict"] = {"no_pass_line_stated": True, "no_verdict_field": True}


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
