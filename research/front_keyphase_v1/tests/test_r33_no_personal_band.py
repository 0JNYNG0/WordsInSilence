"""observed_difference_only_v1: no personal range reaches a new result, on any path.

Public reference observations and synthetic edits of them only. No camera, no session, no
server, no network model, no participant file. The camera path is reproduced by calling the
same predict -> task_match -> _split_result sequence app._finish uses.

Run: py -3.13 tests/test_r33_no_personal_band.py
"""
import copy
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.cli import load_reference_set  # noqa: E402
from wis_lab.core import recognize as R  # noqa: E402
from wis_lab.core.observation import (  # noqa: E402
    LEFT_HAND_SLICE, RIGHT_HAND_SLICE, Observation,
)
from wis_lab.service.app import _replay_following, _split_result  # noqa: E402

CONTRACT_PATH = os.path.join(ROOT, "config", "contract.json")
CONTRACT = json.load(open(CONTRACT_PATH, encoding="utf-8"))
MANIFEST = json.load(open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8"))
BANDS = json.load(open(os.path.join(ROOT, "config", "handshape_bands.json"), encoding="utf-8"))
REFS, _ = load_reference_set()
BAND_KEYS = {"band_max", "band_n", "band_observed", "headroom", "band_verdict"}
BAND_STATUSES = {"IN_BAND", "OUT_OF_BAND", "IN_BAND_PARTIAL_SUPPORT",
                 "OUT_OF_BAND_PARTIAL_SUPPORT", "NO_BAND"}
COVERED = BANDS["covered_words"][0]       # had a personal range before
UNCOVERED = BANDS["uncovered_words"][0]   # never had one
FAILS = []


def ok(cond, name):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        FAILS.append(name)


def clip(word_id):
    return next(c for c in MANIFEST["clips"] if c["word_id"] == word_id)


def obs_for(word_id):
    return Observation.load(os.path.join(ROOT, clip(word_id)["observation_npz"]))


def band_hits(node, path="$"):
    """Every personal-range key or status anywhere in a payload."""
    hits = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k in BAND_KEYS:
                hits.append(f"{path}.{k}")
            if k == "status" and v in BAND_STATUSES:
                hits.append(f"{path}.status={v}")
            hits += band_hits(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits += band_hits(v, f"{path}[{i}]")
    return hits


def binding(word_id):
    return {"status": "BOUND", "demonstration_clip_id": clip(word_id)["clip_id"],
            "demonstration_word_id": word_id, "bound_at": "trial_creation"}


def camera_result(obs, requested, contract=CONTRACT):
    """app._finish's scoring order, without the camera, the neural model or the store."""
    result = R.predict(obs, REFS, contract)
    result["origin"] = "camera"
    result["task"] = R.task_match(result, requested, observation=obs,
                                  reference_set=REFS, frozen_config=contract)
    _split_result(result, {"reference_binding": binding(requested)})
    return result


# ------------------------------------------------------------------ 1. no band on any path
for obs_word, requested in ((COVERED, COVERED), (UNCOVERED, UNCOVERED), (UNCOVERED, COVERED)):
    res = camera_result(obs_for(obs_word), requested)
    hits = band_hits(res)
    ok(not hits, f"camera assembly {obs_word}->{requested}: no personal range anywhere {hits[:3]}")
    df = res["demonstration_following"]
    ok(df["status"] == "AVAILABLE" and df["reference_word_id"] == requested,
       f"  and the demonstration binding stays on the requested word {requested}")
    ok(res["top1_word_id"] == R.predict(obs_for(obs_word), REFS, CONTRACT)["top1_word_id"],
       "  and the ranking does not depend on the requested word")
    checks = df["axis_checks"]
    ok(set(checks) == set(R.AXIS_GROUPS)
       and all(c["policy"] == R.AXIS_CHECK_POLICY and c["is_a_verdict"] is False
               for c in checks.values()),
       "  every axis is reported, under the named policy, never as a verdict")
    dev = df["axis_deviation"]["axes"]
    ok(all(checks[a]["distance"] == dev[a]["distance"] for a in dev),
       "  and the check's distance is the same observed number as the deviation block")

# replay: same predict output at the top level, demonstration_following from the explicit path
rep = R.predict(obs_for(COVERED), REFS, CONTRACT)
rep["origin"] = "replay"
erc = {"status": "OK", "reference_word_id": COVERED, "axis_deviation": {"status": "OK"},
       "joint_deviation": {"status": "OK"}, "location_path": {"status": "OK"}}
_split_result(rep, {"reference_binding": binding(COVERED)})
_replay_following(rep, binding(COVERED), erc)
ok(not band_hits(rep), "replay top-level axis_checks / handshape_check carry no personal range")

# ------------------------------------------------------------------ 2. contract keys are inert
stripped = copy.deepcopy(CONTRACT)
stripped["handshape_check"].pop("axes", None)
flag_off = copy.deepcopy(CONTRACT)
flag_off["handshape_check"]["enabled"] = False
def dump(x):   # NaN-safe equality
    return json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)


base = camera_result(obs_for(COVERED), COVERED)
for name, c in (("personal axes removed", stripped), ("enabled=False", flag_off)):
    ok(dump(camera_result(obs_for(COVERED), COVERED, c)) == dump(base),
       f"the whole camera result is identical with {name}: the ranges are not read")
ok(stripped["handshape_check"].get("axis_scales") == CONTRACT["handshape_check"].get("axis_scales"),
   "the public-reference axis_scales are untouched by the fixture")
same = open(CONTRACT_PATH, "rb").read() == open(os.path.join(
    ROOT, "ops", "web_goal_audit_20260926", "before", "config", "contract.json"), "rb").read()
ok(same, "config/contract.json is byte-identical to the pre-change copy")

# ------------------------------------------------------------------ 3. none vs partial vs zero
self_hs = base["handshape_check"]
ok(self_hs["distance"] == 0.0 and self_hs["status"].startswith("OBSERVED"),
   "a reference against itself is an observed 0.0")

masked = obs_for(COVERED)
masked = Observation(xy_px=masked.xy_px.copy(), timestamp_us=masked.timestamp_us.copy(),
                     frame_id=masked.frame_id.copy(), observed=masked.observed.copy(),
                     usable=masked.usable.copy(), conf=masked.conf.copy(),
                     hand_score=masked.hand_score.copy(), sidecar=copy.deepcopy(masked.sidecar))
for sl in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
    masked.observed[:, sl] = False
    masked.usable[:, sl] = False
    masked.xy_px[:, sl] = np.nan
masked.hand_score[:] = np.nan
mres = camera_result(masked, COVERED)
ok(mres["observation_status"] != "OK" and mres.get("axis_checks") is None
   and mres["demonstration_following"]["status"] == "UNDETERMINED_NO_TASK_ANALYSIS"
   and mres["demonstration_following"]["axis_checks"] is None,
   "both hands hidden: no axis check at all, never a zero difference")

# a measured-nothing axis: groups present with no distance. Stubbed residuals, same function.
real = R.dtw_group_residuals
R.dtw_group_residuals = lambda *a, **k: {"groups": {g: {"mean_distance": None, "compared_steps": 0,
                                                        "unobserved_steps": 5}
                                                    for g in R.AXIS_GROUPS["handshape"]}}
try:
    q = {"word_id": "_"}   # not read: residuals are stubbed
    un = R.axis_check(q, COVERED, REFS, CONTRACT, "handshape")
finally:
    R.dtw_group_residuals = real
ok(un["status"] == "UNMEASURABLE" and "distance" not in un and un["is_a_verdict"] is False,
   "an axis with nothing compared is UNMEASURABLE with no distance, not 0")
nr = R.axis_check(q, "__NOT_A_WORD__", REFS, CONTRACT, "handshape")
ok(nr["status"] == "NO_REFERENCE" and "distance" not in nr, "an unknown word has no reference")

digest = hashlib.sha256(dump(base).encode()).hexdigest()
print(f"\ncamera-assembly result sha256 ({COVERED}->{COVERED}): {digest}")
print(f"{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAIL'}")
sys.exit(1 if FAILS else 0)
