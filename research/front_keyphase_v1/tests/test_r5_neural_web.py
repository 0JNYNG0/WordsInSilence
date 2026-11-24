"""R5: the neural candidate runs on the same observation, and cannot take the DTW path with it.

Only the wiring introduced by this change is checked. No training, no re-evaluation on the source
data, and no accuracy claim: a name matching is not evidence that a sign was performed correctly.

Run: py -3.13 tests/test_r5_neural_web.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from fastapi.testclient import TestClient  # noqa: E402

from test_r2_handshape import clone, obs_for  # noqa: E402
from wis_lab.service import neural as N  # noqa: E402
from wis_lab.service.app import API, app  # noqa: E402

client = TestClient(app)
CHECKS = []
MEASURED = {}
NL = chr(10)


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("the pinned checkpoint is the reviewed one and loads to a cached model")
def _identity():
    cap = N.capability()
    assert cap["model_status"] == "READY", cap
    assert cap["checkpoint_sha256"] == N.CHECKPOINT_SHA256
    assert cap["trained_class_count"] == 80, cap["trained_class_count"]
    assert cap["is_a_verdict"] is False
    first = N._load()["model"]
    assert N._load()["model"] is first, "the model was loaded twice"
    MEASURED["identity"] = {"status": cap["model_status"],
                            "classes": cap["trained_class_count"],
                            "project_words_in_model": cap["project_words_in_model"]}


@check("reloading the checkpoint reproduces the same ranking on the same observation")
def _reload_reproducible():
    obs = obs_for("ALDA")
    a = N.recognise(obs)
    saved = dict(N._CACHE)
    N._CACHE.update({"model": None, "classes": None, "status": None, "detail": None})
    b = N.recognise(obs)
    assert N._CACHE["status"] == "READY", "the reload failed"
    assert a["ranking"] == b["ranking"], "a reload changed the ranking"
    N._CACHE.update(saved)


@check("the ranking is produced blind: the requested word never reaches the model")
def _blind():
    src = open(os.path.join(ROOT, "src", "wis_lab", "service", "neural.py"),
               encoding="utf-8").read()
    body = src.split("def recognise(")[1].split(NL + "def ")[0]
    for forbidden in ("requested_word", "attempt_label"):
        assert forbidden not in body, f"{forbidden} appears inside recognise()"
    obs = obs_for("ALDA")
    assert N.recognise(obs)["ranking"] == N.recognise(obs)["ranking"]


@check("missing, too short and non-monotonic observations are reported, not guessed")
def _degenerate_inputs():
    obs = clone(obs_for("ALDA"))
    obs.usable[:, 33:75] = False                       # no usable hand anywhere
    r = N.recognise(obs)
    assert r["model_status"] == "INSUFFICIENT_OBSERVATION", r
    assert r["ranking"] is None and r["detail"] == "NO_USABLE_HAND_FRAME"

    short = clone(obs_for("ALDA"))
    keep = np.zeros(short.n_frames, dtype=bool)
    keep[:2] = True
    short.usable[~keep] = False
    r2 = N.recognise(short)
    assert r2["model_status"] == "INSUFFICIENT_OBSERVATION", r2
    assert r2["detail"] in ("TOO_FEW_USABLE_FRAMES", "NO_USABLE_HAND_FRAME"), r2

    bad = clone(obs_for("ALDA"))
    bad.timestamp_us[:] = 0                            # every stamp identical
    r3 = N.recognise(bad)
    assert r3["model_status"] == "INSUFFICIENT_OBSERVATION", r3
    assert r3["detail"] == "NON_MONOTONIC_TIMESTAMPS", r3
    MEASURED["degenerate"] = {"no_hand": r["detail"], "short": r2["detail"],
                              "clock": r3["detail"]}


@check("a model failure is isolated: the DTW result still comes back")
def _isolated_failure():
    saved = dict(N._CACHE)
    N._CACHE.update({"model": None, "classes": None,
                     "status": "CHECKPOINT_HASH_MISMATCH", "detail": "simulated"})
    try:
        r = client.post(API + "/replays", json={"observation_id": "INSA"}).json()
        assert r["top1_word_id"], "the DTW ranking disappeared"
        assert r["learned_recognition"]["model_status"] == "CHECKPOINT_HASH_MISMATCH"
        assert r["learned_recognition"]["ranking"] is None
        h = client.get(API + "/health").json()
        assert h["status"] == "OK", "health went down with the model"
        assert h["learned_recognition"]["model_status"] == "CHECKPOINT_HASH_MISMATCH"
    finally:
        N._CACHE.update(saved)


@check("replay carries the adapter result and it matches a direct call on the same input")
def _replay_matches_direct():
    r = client.post(API + "/replays", json={"observation_id": "ALDA"}).json()
    lr = r["learned_recognition"]
    assert lr["model_status"] == "READY", lr
    direct = N.recognise(obs_for("ALDA"))
    assert lr["ranking"] == direct["ranking"], "the API and a direct call disagree"
    assert lr["requested_word_coverage"]["status"] == "IN_MODEL"
    MEASURED["replay"] = {
        "all_80_top1": lr["ranking"]["over_all_trained_classes"]["top1"],
        "restricted_top1": lr["ranking"]["restricted_to_project_words"]["top1"],
        "elapsed_ms": lr["elapsed_ms"]}


@check("a project word the checkpoint never saw is marked NOT_IN_MODEL, never scored as known")
def _untrained_word():
    r = client.post(API + "/replays", json={"observation_id": "INSA"}).json()
    lr = r["learned_recognition"]
    assert lr["requested_word_coverage"]["status"] == "NOT_IN_MODEL", lr
    assert "INSA" not in N.PROJECT_OVERLAP
    assert len(N.PROJECT_OVERLAP) == 5, N.PROJECT_OVERLAP
    MEASURED["coverage"] = {"in_model": sorted(N.PROJECT_OVERLAP),
                            "example_not_in_model": "INSA"}


@check("the neural candidate never becomes a verdict or a score")
def _never_a_verdict():
    r = client.post(API + "/replays", json={"observation_id": "ALDA"}).json()
    assert r["accepted"] in (None, False), r["accepted"]
    assert r["execution_assessment"], "the execution assessment field vanished"
    lr = r["learned_recognition"]
    assert lr["is_a_verdict"] is False
    assert lr["accepted"] is None
    assert lr["execution_assessment"] == "UNCONFIGURED"
    js = open(os.path.join(ROOT, "web", "app.js"), encoding="utf-8").read()
    block = js.split("const lr = r.learned_recognition;")[1].split("$(\"result\").innerHTML")[0]
    assert '"ok"' not in block and "class=\"hs ok\"" not in block, \
        "the neural line can be painted as a pass"
    state = json.load(open(os.path.join(ROOT, "ops", "STATE.json"), encoding="utf-8"))
    assert state["execution_assessment_status"].startswith("UNCONFIGURED")


@check("an unreadable checkpoint is a status, not an exception that reaches the caller")
def _hash_read_failure():
    saved = dict(N._CACHE)
    old = N.CHECKPOINT
    N._CACHE.update({"model": None, "classes": None, "status": None, "detail": None})
    # a directory passes os.path.exists and then fails to open: the hash read used to raise
    N.CHECKPOINT = os.path.dirname(old) + os.sep
    try:
        cap = N.capability()          # must not raise
        assert cap["model_status"] in ("CHECKPOINT_UNREADABLE", "CHECKPOINT_MISSING"), cap
        r = client.post(API + "/replays", json={"observation_id": "ALDA"}).json()
        assert r["top1_word_id"], "the DTW ranking disappeared with an unreadable checkpoint"
        assert r["learned_recognition"]["ranking"] is None
        h = client.get(API + "/health").json()
        assert h["status"] == "OK"
        MEASURED["unreadable_checkpoint"] = {"status": cap["model_status"],
                                             "dtw_still_answers": bool(r["top1_word_id"])}
    finally:
        N.CHECKPOINT = old
        N._CACHE.update(saved)


@check("two visible frames at the ends of a long clip are not a sufficient observation")
def _span_is_not_observation():
    obs = clone(obs_for("ALDA"))
    idx = np.flatnonzero(obs.usable[:, 33:75].any(axis=1))
    keep = np.zeros(obs.n_frames, dtype=bool)
    keep[idx[0]] = keep[idx[-1]] = True
    obs.usable[~keep, 33:75] = False
    r = N.recognise(obs)
    assert r["model_status"] == "INSUFFICIENT_OBSERVATION", r
    assert r["detail"] == "TOO_FEW_OBSERVED_HAND_FRAMES", r
    assert r["ranking"] is None, "a word was named from two frames"
    sup = r["support"]
    assert sup["span_frames"] > 50, sup          # the span really is long
    assert sup["observed_hand_frames"] == 2, sup
    assert sup["observed_hand_fraction"] < 0.05, sup

    good = N.recognise(obs_for("ALDA"))
    assert good["model_status"] == "READY"
    assert good["support"]["observed_hand_frames"] > N.MIN_OBSERVED_HAND_FRAMES
    MEASURED["span_vs_observation"] = {
        "span_frames": sup["span_frames"], "observed_hand_frames": sup["observed_hand_frames"],
        "detail": r["detail"],
        "healthy_clip": {"span": good["support"]["span_frames"],
                         "observed": good["support"]["observed_hand_frames"]}}


@check("capability is answerable before any session exists")
def _capability_before_session():
    h = client.get(API + "/health").json()
    cap = h["learned_recognition"]
    for k in ("model_status", "trained_class_count", "project_words_in_model",
              "known_weakness", "is_a_verdict"):
        assert k in cap, k
    assert "Traceback" not in json.dumps(cap)
    assert ROOT.replace(os.sep, "/") not in json.dumps(cap), "an internal path leaked to the UI"


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
    print(json.dumps(MEASURED, ensure_ascii=False, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
