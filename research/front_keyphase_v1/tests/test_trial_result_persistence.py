"""The result JSON is committed whole before a trial is published as FEEDBACK.

Synthetic only: no device, media, reference set or model is read. The real app._finish and the
real stop / get / display-ack / review handlers run against a stand-in store whose 'camera' hands
over a fake observation; predict, task_match and the neural adapter are stubbed, so the numbers
are canned and only the persistence ordering is under test. Faults are injected at the temp write
(open), at os.replace, and at serialization.

Run: py -3.13 tests/test_trial_result_persistence.py
"""
import builtins
import json
import os
import shutil
import sys
import tempfile
import threading
import types
from unittest.mock import patch

from fastapi import HTTPException

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import wis_lab.service.app as A  # noqa: E402

TMP = tempfile.mkdtemp(prefix="wis_persist_")
FAILS = []
RAW = b"synthetic raw observation bytes"


def ok(cond, name):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        FAILS.append(name)


class FakeObs:
    n_frames = 3
    sidecar = {"observation_sha256": "sha_synthetic"}

    def save(self, path):
        with builtins.open(path, "wb") as fh:
            fh.write(RAW)


class FakeCamera:
    clock_id = "clk"

    def __init__(self):
        self.takes = 0

    def request_stop(self, reason):
        pass

    def end_candidate(self):
        return 123

    def take(self):
        self.takes += 1
        return FakeObs(), {"end_candidate_ns": 0, "end_reason": "hand_down"}


def canned_predict(prep, refs, contract, extra=None):
    res = {"observation_status": "OK", "insufficient_reason": None,
           "ranking": [{"word_id": "w1", "distance": 1.5}], "top1_word_id": "w1",
           "top1_distance": 1.5, "margin_to_second": 0.25,
           "execution_assessment": "SYNTHETIC", "accepted": True}
    res.update(extra or {})
    return res


def make(tid="tri_p", requested="w1"):
    run_dir = tempfile.mkdtemp(dir=TMP)
    os.makedirs(os.path.join(run_dir, "trials"))
    s = {"session_id": "ses_p", "run_dir": run_dir, "trial_ids": [tid], "request_ids": {},
         "state": "OPEN", "purpose": "diagnostic", "participant_id": "synthetic",
         "client_id": None}
    t = {"trial_id": tid, "session_id": "ses_p", "state": "RECORDING",
         "requested_word_id": requested, "sample_label": None, "attempt_label": None,
         "clock_id": "clk", "result": None, "error": None, "timing": {}, "origin": "camera"}
    st = types.SimpleNamespace(
        refs=[], contract={"contract_id": "c_syn"},
        lineage={"set_id": "set_syn", "extractor_id": "ext_syn"},
        camera=FakeCamera(), camera_session="ses_p", lock=threading.RLock(),
        trials={tid: t}, sessions={"ses_p": s}, telemetry_write_failures=0)
    return st, s, t


def paths(s, t):
    d = os.path.join(s["run_dir"], "trials")
    return (os.path.join(d, t["trial_id"] + ".result.json"),
            os.path.join(d, t["trial_id"] + ".npz"), d)


def decision_rows(s):
    p = os.path.join(s["run_dir"], "decisions.jsonl")
    if not os.path.exists(p):
        return []
    with builtins.open(p, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def temps(d):
    return [n for n in os.listdir(d) if ".tmp-" in n]


def stubs(st, extra=None):
    neural_out = {"schema": "stub", "model_status": "NOT_RUN", "ranking": None,
                  "is_a_verdict": False}
    return [patch.object(A, "store", st),
            patch.object(A, "predict", side_effect=lambda *a, **k: canned_predict(*a, extra=extra)),
            patch.object(A, "QueryPreparation", side_effect=lambda o: o),
            patch.object(A, "task_match", side_effect=lambda *a, **k: {"task_match": "MATCH"}),
            patch.object(A.neural, "recognise", side_effect=lambda *a, **k: dict(neural_out)),
            patch.object(A.neural, "word_coverage", side_effect=lambda *a, **k: {})]


def run(st, fn, *patches):
    ps = stubs(st) + list(patches)
    for p in ps:
        p.start()
    try:
        return fn()
    finally:
        for p in reversed(ps):
            p.stop()


def failing_open(marker):
    def fake(path, *a, **k):
        if marker in str(path):
            raise OSError(28, "No space left on device (synthetic)")
        return builtins.open(path, *a, **k)
    return patch.object(A, "open", fake, create=True)


def failing_replace():
    def fake(src, dst):
        raise OSError(13, "Access is denied (synthetic)")
    return patch.object(A.os, "replace", fake)


def review_rejected(st, t):
    calls = []
    with patch.object(A, "store", st), \
            patch.object(A.review_log, "submit", side_effect=lambda *a, **k: calls.append(a)):
        try:
            A.trial_review(t["trial_id"], A.TrialReview(session_id=t["session_id"],
                                                        actual_word_id="w1",
                                                        followed_guidance="yes"))
            return False
        except HTTPException as exc:
            return exc.status_code == 409 and not calls


def ack_rejected(st, t):
    with patch.object(A, "store", st):
        try:
            A.display_ack(t["trial_id"], A.Ack(result_id="res_anything"))
            return False
        except HTTPException as exc:
            return exc.status_code == 404


def check_failure(label, st, s, t, resp, final_should_exist=False, final_before=None):
    final, npz, d = paths(s, t)
    ok(resp["state"] == "ERROR" and resp["result"] is None
       and resp["error"] == "RESULT_PERSIST_FAILED",
       f"{label}: response is an explicit ERROR / RESULT_PERSIST_FAILED with no result")
    ok(t["state"] == "ERROR" and t["result"] is None and t.get("result_persist_error"),
       f"{label}: trial is terminal ERROR, result None, persist error recorded")
    if final_should_exist:
        with builtins.open(final, "rb") as fh:
            ok(fh.read() == final_before, f"{label}: the pre-existing final file is untouched")
    else:
        ok(not os.path.exists(final), f"{label}: no result file was produced")
    ok(not temps(d), f"{label}: no temp artifact left behind")
    with builtins.open(npz, "rb") as fh:
        ok(fh.read() == RAW, f"{label}: the raw observation is preserved")
    ok(not decision_rows(s), f"{label}: no successful decision line")
    ok(t.get("closed_event", {}).get("detail") == "RESULT_PERSIST_FAILED",
       f"{label}: the closed event names RESULT_PERSIST_FAILED")
    again = run(st, lambda: A.stop_trial(t["trial_id"]))
    ok(again.get("already_final") and again["state"] == "ERROR" and "result" not in again,
       f"{label}: a repeated stop cannot manufacture a success")
    got = run(st, lambda: A.get_trial(t["trial_id"]))
    ok(got["state"] == "ERROR" and got["result"] is None
       and got["error"] == "RESULT_PERSIST_FAILED",
       f"{label}: GET reports the explicit error, not FEEDBACK")
    ok(st.camera.takes == 1, f"{label}: inference was not retried ({st.camera.takes} takes)")
    ok(review_rejected(st, t), f"{label}: review is rejected and nothing is submitted")
    ok(ack_rejected(st, t), f"{label}: display-ack is rejected")


try:
    # ------------------------------------------------ 1. normal: atomic write, then FEEDBACK
    st, s, t = make()
    seen = {}
    real_replace = os.replace

    def spy_replace(src, dst):
        seen["state_at_replace"] = t["state"]
        seen["result_at_replace"] = t["result"]
        seen["src_dir"] = os.path.dirname(src)
        return real_replace(src, dst)

    resp = run(st, lambda: A.stop_trial(t["trial_id"]), patch.object(A.os, "replace", spy_replace))
    final, npz, d = paths(s, t)
    ok(resp["state"] == "FEEDBACK" and resp["result"] is not None and resp["error"] is None,
       "manual stop: FEEDBACK with a result")
    ok(seen.get("state_at_replace") != "FEEDBACK" and seen.get("result_at_replace") is None,
       "the file is committed before the result is published")
    ok(seen.get("src_dir") == d, "the temp file is in the same directory as the final file")
    with builtins.open(final, encoding="utf-8") as fh:
        text = fh.read()
    ok(text == json.dumps(t["result"], ensure_ascii=False, indent=2),
       "the final file holds exactly the published result, same formatting as before")
    ok(json.loads(text)["result_id"] == t["result"]["result_id"], "result ids agree")
    ok(not temps(d), "no temp artifact left after success")
    rows = decision_rows(s)
    ok(len(rows) == 1 and rows[0]["trial_id"] == t["trial_id"], "one decision line on success")
    with patch.object(A, "store", st), \
            patch.object(A.review_log, "submit",
                         side_effect=lambda *a, **k: {"status": "STORED"}):
        out = A.trial_review(t["trial_id"], A.TrialReview(session_id="ses_p",
                                                          followed_guidance="yes"))
    ok(out["status"] == "STORED", "review is still accepted for a persisted result")

    # ------------------------------------------------ 2. auto end via GET, normal
    st, s, t = make(tid="tri_auto", requested="w2")
    got = run(st, lambda: A.get_trial(t["trial_id"]))
    final, _, d = paths(s, t)
    ok(got["state"] == "FEEDBACK" and os.path.exists(final) and not temps(d),
       "auto end via GET: FEEDBACK with a committed file")

    # ------------------------------------------------ 3. temp write fails (manual stop)
    st, s, t = make()
    resp = run(st, lambda: A.stop_trial(t["trial_id"]), failing_open(".result.json.tmp-"))
    check_failure("write OSError (manual)", st, s, t, resp)

    # ------------------------------------------------ 4. os.replace fails (manual stop)
    st, s, t = make()
    resp = run(st, lambda: A.stop_trial(t["trial_id"]), failing_replace())
    check_failure("os.replace OSError (manual)", st, s, t, resp)

    # ------------------------------------------------ 5. os.replace fails over an existing final
    st, s, t = make()
    final, _, _ = paths(s, t)
    before = b'{"pre_existing": true}'
    with builtins.open(final, "wb") as fh:
        fh.write(before)
    resp = run(st, lambda: A.stop_trial(t["trial_id"]), failing_replace())
    check_failure("os.replace OSError over existing final", st, s, t, resp,
                  final_should_exist=True, final_before=before)

    # ------------------------------------------------ 6. auto end via GET, write fails
    st, s, t = make(tid="tri_auto_fail")
    got = run(st, lambda: A.get_trial(t["trial_id"]), failing_open(".result.json.tmp-"))
    check_failure("write OSError (auto GET)", st, s, t, got)

    # ------------------------------------------------ 7. serialization fails
    st, s, t = make()
    ps = [p for p in stubs(st, extra={"unserialisable": object()})]
    for p in ps:
        p.start()
    try:
        resp = A.stop_trial(t["trial_id"])
    finally:
        for p in reversed(ps):
            p.stop()
    check_failure("serialization TypeError", st, s, t, resp)
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAIL'}")
sys.exit(1 if FAILS else 0)
