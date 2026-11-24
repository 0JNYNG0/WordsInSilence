"""R1a self-check: the core must not agree with itself for free.

Synthetic corruptions of a real observation are scored against the same candidate set. A
corrupted attempt that still lands on its own word at a near-zero distance would mean the
distance is not measuring what it claims to measure.

Run: py -3.13 tests/test_r1a.py
"""
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.cli import load_reference_set  # noqa: E402
from wis_lab.core.observation import (  # noqa: E402
    LEFT_HAND_SLICE, RIGHT_HAND_SLICE, Observation,
)
from wis_lab.core.recognize import predict, task_match  # noqa: E402

CONTRACT = json.load(open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8"))
MANIFEST = json.load(open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8"))
REFS, LINEAGE = load_reference_set()
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def obs_for(word_id: str) -> Observation:
    clip = next(c for c in MANIFEST["clips"] if c["word_id"] == word_id)
    return Observation.load(os.path.join(ROOT, clip["observation_npz"]))


def clone(obs: Observation) -> Observation:
    return Observation(
        xy_px=obs.xy_px.copy(), timestamp_us=obs.timestamp_us.copy(),
        frame_id=obs.frame_id.copy(), observed=obs.observed.copy(),
        usable=obs.usable.copy(), conf=obs.conf.copy(),
        hand_score=obs.hand_score.copy(), sidecar=copy.deepcopy(obs.sidecar))


def distance_to(result: dict, word_id: str):
    for r in result["ranking"]:
        if r["word_id"] == word_id:
            return r["distance"], r.get("rank")
    return None, None


PROBE_WORDS = ["INSA", "GAMSAHAMNIDA", "SARANG", "JADA", "ALDA"]


@check("clean replay: every candidate is top-1 on its own clip at distance 0")
def _identity():
    for clip in MANIFEST["clips"]:
        if clip["readiness"] == "UNSUPPORTED_CUE":
            continue
        res = predict(obs_for(clip["word_id"]), REFS, CONTRACT)
        assert res["observation_status"] == "OK", clip["word_id"]
        assert res["top1_word_id"] == clip["word_id"], (clip["word_id"], res["top1_word_id"])
        assert res["top1_distance"] == 0.0, (clip["word_id"], res["top1_distance"])


@check("clean replay is deterministic: the same input twice gives the same ranking")
def _deterministic():
    a = predict(obs_for("INSA"), REFS, CONTRACT)
    b = predict(obs_for("INSA"), list(reversed(REFS)), CONTRACT)
    assert [r["word_id"] for r in a["ranking"]] == [r["word_id"] for r in b["ranking"]]
    assert a["top1_distance"] == b["top1_distance"]


@check("time reversal costs distance on its own word")
def _reversed():
    worse = 0
    for w in PROBE_WORDS:
        o = clone(obs_for(w))
        for name in ("xy_px", "observed", "usable", "conf", "hand_score"):
            setattr(o, name, getattr(o, name)[::-1].copy())
        res = predict(o, REFS, CONTRACT)
        d, _ = distance_to(res, w)
        assert d is not None and d > 0.05, (w, d)
        worse += 1
    assert worse == len(PROBE_WORDS)


@check("left/right swap costs distance on its own word")
def _swap_hands():
    for w in PROBE_WORDS:
        o = clone(obs_for(w))
        for name in ("xy_px", "observed", "usable", "conf"):
            arr = getattr(o, name)
            left = arr[:, LEFT_HAND_SLICE].copy()
            arr[:, LEFT_HAND_SLICE] = arr[:, RIGHT_HAND_SLICE]
            arr[:, RIGHT_HAND_SLICE] = left
        o.hand_score = o.hand_score[:, ::-1].copy()
        res = predict(o, REFS, CONTRACT)
        d, _ = distance_to(res, w)
        assert d is not None and d > 0.05, (w, d)


@check("wrong hand shape costs distance even when the trajectory is untouched")
def _wrong_shape():
    donor = obs_for("SARANG")
    for w in ("INSA", "GAMSAHAMNIDA", "JADA"):
        o = clone(obs_for(w))
        n = min(o.n_frames, donor.n_frames)
        for sl in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
            base = sl.start
            # keep each frame's wrist, replace the 20 finger points with the donor's local shape
            for t in range(n):
                if not (o.observed[t, base] and donor.observed[t, base]):
                    continue
                local = donor.xy_px[t, base + 1:base + 21] - donor.xy_px[t, base]
                o.xy_px[t, base + 1:base + 21] = o.xy_px[t, base] + local
                o.observed[t, base + 1:base + 21] = donor.observed[t, base + 1:base + 21]
                o.usable[t, base + 1:base + 21] = donor.usable[t, base + 1:base + 21]
        o.xy_px[~o.observed] = np.nan
        res = predict(o, REFS, CONTRACT)
        d, _ = distance_to(res, w)
        assert d is not None and d > 0.05, (w, d)


@check("a frozen still frame does not pass as a performed sign")
def _frozen():
    o = clone(obs_for("INSA"))
    mid = o.n_frames // 2
    for name in ("xy_px", "observed", "usable", "conf", "hand_score"):
        arr = getattr(o, name)
        arr[:] = arr[mid]
    res = predict(o, REFS, CONTRACT)
    if res["observation_status"] == "OK":
        d, _ = distance_to(res, "INSA")
        assert d is not None and d > 0.05, d
        assert res["execution_assessment"] == "UNCONFIGURED"
        assert res["accepted"] is None


@check("no hands at all is INSUFFICIENT, never a successful match")
def _no_hands():
    o = clone(obs_for("INSA"))
    for sl in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
        o.observed[:, sl] = False
        o.usable[:, sl] = False
        o.xy_px[:, sl] = np.nan
    o.hand_score[:] = np.nan
    res = predict(o, REFS, CONTRACT)
    assert res["observation_status"] == "INSUFFICIENT", res["observation_status"]
    assert res["accepted"] is None
    assert res["execution_assessment"] == "UNCONFIGURED"
    assert res["insufficient_reason"]


@check("no pose at all is INSUFFICIENT, not a large-distance match")
def _no_pose():
    o = clone(obs_for("INSA"))
    o.observed[:, :33] = False
    o.usable[:, :33] = False
    o.xy_px[:, :33] = np.nan
    o.conf[:, :33] = np.nan
    res = predict(o, REFS, CONTRACT)
    assert res["observation_status"] == "INSUFFICIENT"
    assert res["accepted"] is None


@check("predict() is blind to the answer: the requested word is scored separately")
def _blind():
    import inspect
    params = list(inspect.signature(predict).parameters)
    assert params == ["observation", "reference_set", "frozen_config"], params
    res = predict(obs_for("INSA"), REFS, CONTRACT)
    t = task_match(res, "SARANG")
    assert t["task_match"] is False
    assert t["requested_rank"] is not None
    # scoring the task must not change the ranking
    assert res["top1_word_id"] == "INSA"


@check("R1 states stay honest: no pass line, no calibrated probability")
def _states():
    res = predict(obs_for("INSA"), REFS, CONTRACT)
    assert res["recognition"] == "UNCONFIGURED"
    assert res["execution_assessment"] == "UNCONFIGURED"
    assert res["accepted"] is None
    assert "not a sign-language accuracy score" in res["execution_assessment_note"]


@check("candidate set excludes nothing silently")
def _candidates():
    assert LINEAGE["candidate_count"] == 18
    assert len(LINEAGE["excluded"]) == 1
    assert LINEAGE["excluded"][0]["word_id"] == "KKEUDEOGIDA"
    assert LINEAGE["excluded"][0]["reason"]


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
