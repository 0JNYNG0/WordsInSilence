"""Request-local query preparation: sequence_features at most once per live result / replay.

Public reference observations and synthetic edits of them only. The live path runs the real
app._finish with a stand-in camera object that only hands over a stored Observation (no device
is touched); the neural adapter is stubbed. The replay path runs the real replay handler. Every
"expected" value comes from the OLD standalone calls on a raw Observation (sequence_features per
call), so equality here means the reuse changed no output.

Run: py -3.13 tests/test_r35_query_reuse.py
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.cli import load_reference_set  # noqa: E402
from wis_lab.core import features as F  # noqa: E402
from wis_lab.core import recognize as R  # noqa: E402
from wis_lab.core.observation import (  # noqa: E402
    LEFT_HAND_SLICE, RIGHT_HAND_SLICE, Observation,
)
import wis_lab.service.app as A  # noqa: E402

CONTRACT = json.load(open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8"))
MANIFEST = json.load(open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8"))
CANDIDATE = copy.deepcopy(CONTRACT)
CANDIDATE["feedback"] = json.load(open(os.path.join(
    ROOT, "config", "observed_feedback_candidate_v1.json"), encoding="utf-8"))["feedback"]
REFS, LINEAGE = load_reference_set()
W, OTHER = REFS[0]["word_id"], REFS[1]["word_id"]
ANALYSIS = ("observation_status", "insufficient_reason", "quality_summary", "ranking",
            "top1_word_id", "top3_word_ids", "top1_distance", "margin_to_second", "axis_checks",
            "axis_deviation", "handshape_check", "task", "explicit_reference_comparison",
            "demonstration_following", "reference_binding")
TMP = tempfile.mkdtemp(prefix="wis_r35_")
FAILS = []


def ok(cond, name):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        FAILS.append(name)


def dump(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)


def same(a, b):
    """Exact structural equality, arrays compared element-wise (NaN equal to NaN)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
            a, b, equal_nan=a.dtype.kind in "fc")
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return type(a) is type(b) and a == b


def analysis(res):
    return dump({k: res.get(k) for k in ANALYSIS})


def clip(word_id):
    return next(c for c in MANIFEST["clips"] if c["word_id"] == word_id)


def obs_for(word_id):
    return Observation.load(os.path.join(ROOT, clip(word_id)["observation_npz"]))


def rebuild(o, rows=slice(None)):
    return Observation(xy_px=o.xy_px[rows].copy(), timestamp_us=o.timestamp_us[rows].copy(),
                       frame_id=o.frame_id[rows].copy(), observed=o.observed[rows].copy(),
                       usable=o.usable[rows].copy(), conf=o.conf[rows].copy(),
                       hand_score=o.hand_score[rows].copy(), sidecar={})


def no_hands(word_id):
    o = rebuild(obs_for(word_id))
    for sl in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
        o.observed[:, sl] = False
        o.usable[:, sl] = False
        o.xy_px[:, sl] = np.nan
    o.hand_score[:] = np.nan
    return o


def short(word_id, n=5):
    """n frames from the first frame with a usable hand: prepares, but too few to rank."""
    o = obs_for(word_id)
    any_hand = o.usable[:, LEFT_HAND_SLICE].any(axis=1) | o.usable[:, RIGHT_HAND_SLICE].any(axis=1)
    start = int(np.flatnonzero(any_hand)[0])
    return rebuild(o, slice(start, start + n))


class Counter:
    def __init__(self, real, fail=None):
        self.real, self.fail, self.calls = real, fail, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.fail is not None:          # fail the first call only; a retry would succeed
            exc, self.fail = self.fail, None
            raise exc
        return self.real(*args, **kwargs)


def counting(fail=None):
    c = Counter(F.sequence_features, fail)
    return c, patch.object(F, "sequence_features", c)


def binding(st, word_id):
    """Exactly how the handlers bind: the prepared reference's clip, or nothing when unnamed."""
    ref_clip = next((r["clip_id"] for r in st.refs if r["word_id"] == word_id), None) \
        if word_id else None
    return A._reference_binding(st, ref_clip, word_id)


def stub_neural():
    return (patch.object(A.neural, "recognise",
                         side_effect=lambda *a, **k: {"schema": "stub", "model_status": "NOT_RUN",
                                                      "ranking": None, "is_a_verdict": False}),
            patch.object(A.neural, "word_coverage", side_effect=lambda *a, **k: {}))


def store(contract):
    return types.SimpleNamespace(refs=REFS, contract=contract, lineage=LINEAGE, manifest=MANIFEST)


def run_finish(obs, requested, contract=CONTRACT):
    """The real _finish; the 'camera' only returns the stored observation."""
    run_dir = tempfile.mkdtemp(dir=TMP)
    os.makedirs(os.path.join(run_dir, "trials"))
    st = store(contract)
    st.camera = types.SimpleNamespace(
        take=lambda: (obs, {"end_candidate_ns": 0, "end_reason": "hand_down"}), clock_id="clk")
    s = {"session_id": "ses_r35", "run_dir": run_dir, "trial_ids": [], "request_ids": {},
         "state": "OPEN", "purpose": "diagnostic", "participant_id": "synthetic"}
    t = {"trial_id": "tri_r35", "session_id": "ses_r35", "state": "RECORDING",
         "requested_word_id": requested, "sample_label": None, "attempt_label": None,
         "clock_id": "clk", "result": None, "error": None, "timing": {}, "origin": "camera",
         "reference_binding": binding(st, requested)}
    p1, p2 = stub_neural()
    with p1, p2:
        return A._finish(t, st, s)["result"]


def expected_finish(obs, requested, contract=CONTRACT):
    """The old live assembly: raw Observation into predict and task_match separately."""
    st = store(contract)
    res = R.predict(obs, REFS, contract)
    res["task"] = R.task_match(res, requested, observation=obs, reference_set=REFS,
                               frozen_config=contract)
    A._split_result(res, {"reference_binding": binding(st, requested)})
    return res


def run_replay(observation_id, chosen, contract=CONTRACT):
    st = store(contract)
    p1, p2 = stub_neural()
    with patch.object(A, "store", st), p1, p2:
        return A.replay(A.ReplayCreate(observation_id=observation_id, compare_to_word_id=chosen))


def expected_replay(obs, chosen, contract=CONTRACT):
    """The old replay assembly: predict and _explicit_comparison each on the raw Observation."""
    st = store(contract)
    res = R.predict(obs, REFS, contract)
    if chosen:
        res["explicit_reference_comparison"] = A._explicit_comparison(obs, chosen, st)
    b = binding(st, chosen)
    b["bound_at"] = "replay_request"
    b["chosen_by"] = "explicit_ui_choice_not_the_recogniser"
    A._split_result(res, {"reference_binding": b})
    A._replay_following(res, b, res.get("explicit_reference_comparison") or {})
    return res


try:
    # ------------------------------------------------ 1. live: once, and identical to old
    for requested in (W, OTHER):
        want = expected_finish(obs_for(W), requested)
        c, p = counting()
        with p:
            got = run_finish(obs_for(W), requested)
        ok(c.calls == 1, f"live {W} requested {requested}: sequence_features ran {c.calls}x")
        ok(analysis(got) == analysis(want),
           "  and ranking, axes, task analysis and demonstration block equal the old path")
    ok(dump(run_finish(obs_for(W), W)["ranking"]) == dump(run_finish(obs_for(W), OTHER)["ranking"]),
       "the requested word does not change the ranking")

    # ------------------------------------------------ 2. replay: once, same and other reference
    for chosen in (W, OTHER, None):
        want = expected_replay(obs_for(W), chosen)
        c, p = counting()
        with p:
            got = run_replay(W, chosen)
        ok(c.calls == 1, f"replay {W} vs {chosen}: sequence_features ran {c.calls}x")
        ok(analysis(got) == analysis(want), "  and the replay analysis equals the old path")

    # ------------------------------------------------ 3. insufficient inputs, same as before
    for label, make in (("no hands", lambda: no_hands(W)), ("5 frames", lambda: short(W))):
        want = expected_replay(make(), W)
        c, p = counting()
        with p:
            prep = R.QueryPreparation(make())
            got = R.predict(prep, REFS, CONTRACT)
            got["explicit_reference_comparison"] = A._explicit_comparison(prep, W, store(CONTRACT))
        b = binding(store(CONTRACT), W)
        b.update(bound_at="replay_request", chosen_by="explicit_ui_choice_not_the_recogniser")
        A._split_result(got, {"reference_binding": b})
        A._replay_following(got, b, got["explicit_reference_comparison"])
        ok(c.calls == 1 and got["observation_status"] == "INSUFFICIENT",
           f"{label}: INSUFFICIENT with one preparation ({c.calls}x)")
        ok(analysis(got) == analysis(want),
           f"  {label}: explicit comparison / following status identical to the old path")
        want_live = expected_finish(make(), W)
        c, p = counting()
        with p:
            got_live = run_finish(make(), W)
        ok(c.calls == 1 and analysis(got_live) == analysis(want_live),
           f"  {label}: live assembly identical, one preparation ({c.calls}x)")

    # ------------------------------------------------ 4. a FeatureError is kept, not recomputed
    err = F.FeatureError("synthetic preparation failure")
    c, p = counting(fail=err)
    with p:
        prep = R.QueryPreparation(obs_for(W))
        res = R.predict(prep, REFS, CONTRACT)
        erc = A._explicit_comparison(prep, W, store(CONTRACT))
    ok(c.calls == 1 and res["insufficient_reason"] == "synthetic preparation failure"
       and erc == {"status": "UNAVAILABLE", "reason": "FeatureError", "reference_word_id": W},
       "a FeatureError is reported by both consumers exactly as before, from one attempt")
    c, p = counting(fail=RuntimeError("not a feature error"))
    with p:
        prep = R.QueryPreparation(obs_for(W))
        try:
            R.predict(prep, REFS, CONTRACT)
            raised = False
        except RuntimeError:
            raised = True
        again = R.predict(prep, REFS, CONTRACT)
    ok(raised and c.calls == 2 and again["observation_status"] == "OK",
       "any other exception still escapes predict as before and is not cached")

    # ------------------------------------------------ 5. isolation between consumers / requests
    prep = R.QueryPreparation(obs_for(W))
    seq1, q1 = prep.sample(int(CONTRACT["sampling_A"]["budget"]))
    q1["features"][:] = 0
    seq1["features"][:] = 0
    seq1["movement_observation"]["hands"]["L"]["wrist_xy"].clear()
    q1["feedback_body_context"] = None
    fresh_seq = F.sequence_features(obs_for(W))
    fresh_q = F.uniform_time_sample(fresh_seq, int(CONTRACT["sampling_A"]["budget"]))
    seq2, q2 = prep.sample(int(CONTRACT["sampling_A"]["budget"]))
    ok(same(q2, fresh_q) and same(seq2, fresh_seq),
       "a consumer's in-place writes never reach the next consumer")
    seq3, q3 = prep.sample(8)
    ok(same(q3, F.uniform_time_sample(F.sequence_features(obs_for(W)), 8)),
       "a different sampling budget is sampled for that budget, never a stale sample")
    a_prep, b_prep = R.QueryPreparation(obs_for(W)), R.QueryPreparation(obs_for(OTHER))
    ra, rb = R.predict(a_prep, REFS, CONTRACT), R.predict(b_prep, REFS, CONTRACT)
    ok(dump(ra) == dump(R.predict(obs_for(W), REFS, CONTRACT))
       and dump(rb) == dump(R.predict(obs_for(OTHER), REFS, CONTRACT)),
       "two interleaved preparations keep their own observations")
    try:
        R.QueryPreparation(a_prep)
        wrapped = False
    except TypeError:
        wrapped = True
    ok(wrapped, "a preparation cannot wrap another preparation")
    ok(not any(isinstance(v, R.QueryPreparation) for v in vars(R).values())
       and not any(isinstance(v, R.QueryPreparation) for v in vars(A).values()),
       "no preparation is held at module level")
    obs = obs_for(W)
    fields = ("xy_px", "observed", "usable", "timestamp_us", "frame_id", "conf", "hand_score")
    before = {k: getattr(obs, k).copy() for k in fields}
    prep = R.QueryPreparation(obs)
    R.task_match(R.predict(prep, REFS, CONTRACT), W, observation=prep, reference_set=REFS,
                 frozen_config=CONTRACT)
    ok(same({k: getattr(obs, k) for k in fields}, before), "the observation itself is not modified")

    # ------------------------------------------------ 6. candidate feedback config unchanged
    want = expected_finish(obs_for(W), W, CANDIDATE)
    c, p = counting()
    with p:
        got = run_finish(obs_for(W), W, CANDIDATE)
    ok(c.calls == 1 and analysis(got) == analysis(want),
       "candidate feedback config: live assembly identical with one preparation")
    want = expected_replay(obs_for(W), W, CANDIDATE)
    ok(analysis(run_replay(W, W, CANDIDATE)) == analysis(want),
       "candidate feedback config: replay identical")

    # ------------------------------------------------ 7. standalone callers unchanged
    c, p = counting()
    with p:
        res = R.predict(obs_for(W), REFS, CONTRACT)
        R.task_match(res, W, observation=obs_for(W), reference_set=REFS, frozen_config=CONTRACT)
        R.task_match(res, W)
    ok(c.calls == 2, "a raw Observation still takes the old standalone path (2 preparations)")
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAIL'}")
sys.exit(1 if FAILS else 0)
