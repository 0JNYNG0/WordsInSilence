"""R13: the demonstration binding, the split blocks, the new review field and the timing scope.

Only the new boundaries are checked. No camera, no training, no model inference, no repeat of an
earlier suite, and no whole-diagnostic re-run.

Run: py -3.13 tests/test_r13_reference_binding.py
"""
import json
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.service import review_log  # noqa: E402
from wis_lab.service.app import _reference_binding, _split_result  # noqa: E402

CHECKS = []
MEASURED = {"fixture": "synthetic store and payloads; no camera and no model"}
NL = chr(10)
TMP = ""


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def store():
    """A stand-in with just the fields the binding reads."""
    return types.SimpleNamespace(
        lineage={"set_id": "nikl_front_19_v1"},
        manifest={"clips": [
            {"clip_id": "NIKL_ALDA_01", "word_id": "ALDA", "observation_npz": "cache/x/a.npz",
             "observation_sha256": "aaa", "sha256": "src_a", "readiness": "READY"},
            {"clip_id": "NIKL_NA_01", "word_id": "NA", "observation_npz": "cache/x/n.npz",
             "observation_sha256": "nnn", "sha256": "src_n", "readiness": "NEEDS_REVIEW"},
            {"clip_id": "NIKL_KKEUDEOGIDA_01", "word_id": "KKEUDEOGIDA",
             "observation_npz": "cache/x/k.npz", "observation_sha256": "kkk", "sha256": "src_k",
             "readiness": "UNSUPPORTED_CUE"}]},
        refs=[{"clip_id": "NIKL_ALDA_01", "word_id": "ALDA"},
              {"clip_id": "NIKL_NA_01", "word_id": "NA"}])


def result_with_task(top1="NA"):
    return {"observation_status": "OK", "top1_word_id": top1,
            "ranking": [{"word_id": top1, "distance": 0.1, "status": "OK"},
                        {"word_id": "ALDA", "distance": 0.4, "status": "OK"}],
            "learned_recognition": {"model_status": "READY", "ranking": {
                "over_all_trained_classes": {"top1": "나", "candidate_count": 80},
                "restricted_to_project_words": {"top1": "NA", "candidate_count": 5}}},
            "task": {"requested_word_id": "ALDA", "task_match": False,
                     "axis_deviation_vs_requested": {"status": "OK"},
                     "axis_checks_vs_requested": {"handshape": {"status": "IN_BAND"}},
                     "joint_deviation_vs_requested": {"status": "OK"},
                     "location_path_vs_requested": {"status": "OK"}}}


@check("the binding names the clip actually shown, with its hashes and review state")
def _bound():
    b = _reference_binding(store(), "NIKL_ALDA_01", "ALDA")
    assert b["status"] == "BOUND", b
    assert b["demonstration_clip_id"] == "NIKL_ALDA_01"
    assert b["demonstration_word_id"] == "ALDA"
    assert b["observation_sha256"] == "aaa" and b["source_sha256"] == "src_a"
    assert b["reference_set_id"] == "nikl_front_19_v1"
    assert b["bound_at"] == "trial_creation"
    assert "form_review_state" in b
    MEASURED["bound"] = {k: b[k] for k in ("status", "demonstration_clip_id", "readiness")}


@check("a prediction different from the demonstration leaves the binding untouched")
def _prediction_does_not_move_the_binding():
    st = store()
    b = _reference_binding(st, "NIKL_ALDA_01", "ALDA")
    res = result_with_task(top1="NA")            # the recogniser says NA, the demo was ALDA
    t = {"reference_binding": b}
    _split_result(res, t)
    assert res["reference_binding"]["demonstration_clip_id"] == "NIKL_ALDA_01"
    assert res["reference_binding"]["demonstration_word_id"] == "ALDA"
    assert res["demonstration_following"]["reference_clip_id"] == "NIKL_ALDA_01"
    assert res["demonstration_following"]["reference_word_id"] == "ALDA"
    assert res["lexical_identification"]["dtw"]["top1"] == "NA"
    assert res["lexical_identification"]["requested_word_given_to_recognisers"] is False
    MEASURED["independence"] = {"identification_top1": "NA", "bound_demonstration": "ALDA"}


@check("no demonstration, an unknown one or an unprepared one is undetermined, never substituted")
def _undetermined():
    st = store()
    cases = {
        None: "UNDETERMINED_NO_DEMONSTRATION_SPECIFIED",
        "NIKL_NOT_A_CLIP": "UNDETERMINED_UNKNOWN_DEMONSTRATION",
        "NIKL_KKEUDEOGIDA_01": "UNDETERMINED_DEMONSTRATION_NOT_A_PREPARED_CANDIDATE",
    }
    for clip_id, expect in cases.items():
        b = _reference_binding(st, clip_id, "ALDA")
        assert b["status"] == expect, (clip_id, b["status"])
        res = result_with_task(top1="NA")
        _split_result(res, {"reference_binding": b})
        d = res["demonstration_following"]
        assert d["status"] == expect
        assert d["axis_deviation"] is None and d["joint_deviation"] is None
        assert d["location_path"] is None and d["axis_checks"] is None
        assert d["reference_clip_id"] == clip_id
        # the identification is still reported; it is simply not used as a stand-in
        assert res["lexical_identification"]["dtw"]["top1"] == "NA"
        assert "NOT used as a stand-in" in d["why"]
    MEASURED["undetermined"] = list(cases.values())


@check("a demonstration for a different word than requested is flagged, not silently accepted")
def _changed_demonstration():
    b = _reference_binding(store(), "NIKL_NA_01", "ALDA")
    assert b["status"] == "UNDETERMINED_DEMONSTRATION_WORD_DIFFERS_FROM_REQUEST", b
    res = result_with_task()
    _split_result(res, {"reference_binding": b})
    assert res["demonstration_following"]["axis_deviation"] is None


@check("the new review field is separate, validated, and never back-fills an older answer")
def _review_field():
    run = os.path.join(TMP, "run1")
    os.makedirs(os.path.join(run, "trials"), exist_ok=True)
    known = {"ALDA", "NA"}
    # an older-style confirmation, without the new field
    a = review_log.submit(run, "ses", "t1", "ALDA", "yes", None, known_words=known)
    assert a["stored"] and a["revision"] == 1
    rows, err = review_log.read_all(run)
    assert err is None and rows[0]["followed_demonstration_form"] is None, rows[0]
    # an invalid value is refused
    bad = review_log.submit(run, "ses", "t2", "ALDA", "yes", None, known_words=known,
                            followed_demonstration_form="sort of")
    assert bad["status"] == "REJECTED", bad
    # the same content twice is one revision
    b = review_log.submit(run, "ses", "t2", "ALDA", "yes", None, known_words=known,
                          followed_demonstration_form="other_form")
    c = review_log.submit(run, "ses", "t2", "ALDA", "yes", None, known_words=known,
                          followed_demonstration_form="other_form")
    assert b["revision"] == 1 and c["status"] == "DUPLICATE_IGNORED", (b, c)
    # changing ONLY the new field is a correction, not a duplicate
    d = review_log.submit(run, "ses", "t2", "ALDA", "yes", None, known_words=known,
                          followed_demonstration_form="no")
    assert d["revision"] == 2, d
    rows, _ = review_log.read_all(run)
    t2 = sorted([r for r in rows if r["trial_id"] == "t2"], key=lambda r: r["revision"])
    assert [r["followed_demonstration_form"] for r in t2] == ["other_form", "no"]
    assert all(r["followed_guidance"] == "yes" for r in t2), "the old axis was altered"
    MEASURED["review_field"] = {"revisions": [r["revision"] for r in t2]}


@check("the report tallies the new axis under its own denominator")
def _report():
    run = os.path.join(TMP, "run2")
    os.makedirs(os.path.join(run, "trials"), exist_ok=True)
    known = {"ALDA", "NA"}
    review_log.submit(run, "ses", "t1", "ALDA", "yes", None, known_words=known)
    review_log.submit(run, "ses", "t2", "NA", "yes", None, known_words=known,
                      followed_demonstration_form="other_form")
    review_log.submit(run, "ses", "t3", "NA", "no", None, known_words=known,
                      followed_demonstration_form="yes")
    rep = review_log.report(run, trials=["t1", "t2", "t3"])
    f = rep["demonstration_form_followed"]
    assert f["answered"] == 2 and f["not_answered"] == 1, f
    assert f["other_form"] == 1 and f["yes"] == 1 and f["no"] == 0
    assert "NEVER read as a yes" in f["denominator"]
    assert "never merged" in f["separate_from_followed_guidance"].lower()
    MEASURED["report"] = {k: f[k] for k in ("answered", "not_answered", "yes", "other_form")}


@check("a slow neural adapter lands inside the reported time, success and failure alike")
def _timing_scope():
    """Run _finish itself on a fake clock. No camera, no model, no real waiting."""
    from wis_lab.service import app as A

    class FakeClock:
        """A stand-in for the module's time, advancing only when something says it should."""

        def __init__(self):
            self.ns = 1_000_000_000

        def perf_counter_ns(self):
            return self.ns

        def advance_ms(self, ms):
            self.ns += int(ms * 1_000_000)

        def __getattr__(self, name):              # strftime, time, sleep ... stay real
            import time as real
            return getattr(real, name)

    class FakeObs:
        n_frames = 12
        sidecar = {"observation_sha256": "obs_sha_fake"}

        def save(self, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(b"not a real npz")

    def run_once(nn_ms, fail):
        clock = FakeClock()
        run_dir = tempfile.mkdtemp(prefix="wis_r13_fin_", dir=TMP)
        os.makedirs(os.path.join(run_dir, "trials"), exist_ok=True)
        end_candidate = clock.ns
        camera = types.SimpleNamespace(
            take=lambda: (FakeObs(), {"end_candidate_ns": end_candidate,
                                      "end_reason": "hand_down"}),
            clock_id="clk_fake")
        st = types.SimpleNamespace(
            camera=camera, refs=[{"clip_id": "NIKL_ALDA_01", "word_id": "ALDA"}],
            contract={"contract_id": "wis_front_contract_v1_5"},
            lineage={"set_id": "set", "extractor_id": "ext"},
            manifest={"clips": [{"clip_id": "NIKL_ALDA_01", "word_id": "ALDA",
                                 "observation_npz": "x", "observation_sha256": "a",
                                 "sha256": "b", "readiness": "READY"}]})
        s_rec = {"session_id": "ses_fake", "run_dir": run_dir, "trial_ids": [],
                 "request_ids": {}, "state": "OPEN", "purpose": "diagnostic",
                 "participant_id": "fake"}
        t = {"trial_id": "tri_fake", "session_id": "ses_fake", "state": "RECORDING",
             "requested_word_id": "ALDA", "sample_label": None, "attempt_label": None,
             "clock_id": "clk_fake", "result": None, "error": None, "timing": {},
             "origin": "camera",
             "reference_binding": A._reference_binding(st, "NIKL_ALDA_01", "ALDA")}

        def fake_predict(obs, refs, contract):
            clock.advance_ms(10)                  # the DTW path costs something too
            return {"observation_status": "OK", "top1_word_id": "ALDA", "top1_distance": 0.1,
                    "ranking": [{"word_id": "ALDA", "distance": 0.1, "status": "OK"}],
                    "margin_to_second": 0.2, "insufficient_reason": None,
                    # the fields the decision row reads; predict() supplies them in production
                    "execution_assessment": "UNCONFIGURED", "accepted": None,
                    "recognition": "UNCONFIGURED", "quality_summary": None}

        def fake_task_match(result, word, observation=None, reference_set=None,
                            frozen_config=None):
            return {"requested_word_id": word, "task_match": True,
                    "axis_deviation_vs_requested": {"status": "OK"},
                    "axis_checks_vs_requested": {}, "joint_deviation_vs_requested": {"status": "OK"},
                    "location_path_vs_requested": {"status": "OK"}}

        def fake_recognise(obs):
            clock.advance_ms(nn_ms)               # the adapter takes this long
            if fail:
                raise RuntimeError("adapter exploded")
            return {"schema": "x", "model_status": "READY", "ranking": None}

        saved = (A.time, A.predict, A.task_match, A.neural)
        A.time = clock
        A.predict = fake_predict
        A.task_match = fake_task_match
        A.neural = types.SimpleNamespace(recognise=fake_recognise, ADAPTER_SCHEMA="x",
                                         word_coverage=lambda w: {"status": "IN_MODEL"})
        try:
            out = A._finish(t, st, s_rec)
        finally:
            A.time, A.predict, A.task_match, A.neural = saved
        return out["result"], clock

    # success: the adapter's 250 ms must be inside the reported figure
    res, clock = run_once(250, fail=False)
    tm = res["timing"]
    assert tm["timing_scope_id"] == "server_end_candidate_to_assembled_result_v2"
    assert tm["server_end_candidate_to_result_ms"] == 260.0, tm
    assert res["learned_recognition"]["model_status"] == "READY"

    # failure: the same 250 ms is still inside, and a total is still reported
    res_f, _ = run_once(250, fail=True)
    tm_f = res_f["timing"]
    assert res_f["learned_recognition"]["model_status"] == "ADAPTER_ERROR", res_f
    assert tm_f["server_end_candidate_to_result_ms"] == 260.0, tm_f
    assert tm_f["timing_scope_id"] == tm["timing_scope_id"]

    # and the old scope would have reported 10 ms: the difference is the adapter
    assert tm["server_end_candidate_to_result_ms"] - 10.0 == 250.0
    MEASURED["timing"] = {"neural_ms": 250, "reported_total_ms_success": tm[
        "server_end_candidate_to_result_ms"],
        "reported_total_ms_adapter_error": tm_f["server_end_candidate_to_result_ms"],
        "old_scope_would_have_reported_ms": 10.0}


@check("AVAILABLE is refused when the analysis is for another word or the observation failed")
def _available_is_earned():
    b = _reference_binding(store(), "NIKL_ALDA_01", "ALDA")

    other = result_with_task()                       # task is for ALDA, binding is ALDA: fine
    _split_result(other, {"reference_binding": b})
    assert other["demonstration_following"]["status"] == "AVAILABLE"

    wrong = result_with_task()
    wrong["task"]["requested_word_id"] = "NA"        # the analysis is about a different word
    _split_result(wrong, {"reference_binding": b})
    d = wrong["demonstration_following"]
    assert d["status"] == "UNDETERMINED_ANALYSIS_IS_FOR_A_DIFFERENT_WORD", d
    assert d["axis_deviation"] is None and d["joint_deviation"] is None
    assert d["analysis_word_id"] == "NA"

    thin = result_with_task()
    thin["observation_status"] = "INSUFFICIENT_HAND_FRAMES"
    _split_result(thin, {"reference_binding": b})
    d = thin["demonstration_following"]
    assert d["status"] == "UNDETERMINED_OBSERVATION_INSUFFICIENT", d
    assert d["axis_deviation"] is None and d["location_path"] is None

    empty = result_with_task()
    empty["task"] = {"requested_word_id": "ALDA", "task_match": True}   # no analysis inside
    _split_result(empty, {"reference_binding": b})
    assert empty["demonstration_following"]["status"] == "UNDETERMINED_NO_TASK_ANALYSIS"
    MEASURED["available_earned"] = ["AVAILABLE", "different word", "insufficient observation",
                                    "no analysis"]


def main() -> int:
    global TMP
    TMP = tempfile.mkdtemp(prefix="wis_r13_")
    failed = 0
    try:
        for name, fn in CHECKS:
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:                 # noqa: BLE001 - any raise is a failure
                failed += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"{NL}{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    print(json.dumps(MEASURED, ensure_ascii=False, indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
