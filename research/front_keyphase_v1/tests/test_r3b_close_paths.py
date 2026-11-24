"""R3b: the termination paths that previously left no trace, and never leave two.

Only the close paths are checked here. tests/test_r3_ack_persistence.py already covers the
acknowledgement itself and the aggregate, and is not repeated.

The gap this closes: a trial cancelled by lease expiry or by session close produced no
trial_closed row, so an aggregate built from timing_events.jsonl silently shrank its own
denominator instead of counting the trial. The first check reproduces that absence against the
pre-fix behaviour before requiring the row.

Synthetic clock, temporary directory, hand-built store. No camera, no server, no user session.

Run: py -3.13 tests/test_r3b_close_paths.py
"""
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.service import app as A  # noqa: E402
from wis_lab.service import latency_report as L  # noqa: E402

CHECKS = []
MEASURED = {}
NL = chr(10)


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def fixture(tmp, states=("RECORDING",), expired=False):
    """A session with one trial per entry in `states`. `expired` ages the lease past its TTL."""
    run_dir = os.path.join(tmp, "ses_close")
    os.makedirs(os.path.join(run_dir, "trials"), exist_ok=True)
    store = A.Store()
    store.telemetry_write_failures = 0
    s = {"session_id": "ses_close", "run_dir": run_dir, "purpose": "close_path_check",
         "participant_id": "synthetic", "client_id": None, "state": "OPEN",
         "camera": False, "trial_ids": [],
         "lease_touched_monotonic": time.monotonic() - (A.LEASE_TTL_S + 5 if expired else 0)}
    store.sessions[s["session_id"]] = s
    for i, state in enumerate(states):
        tid = f"tri_close{i:02d}"
        t = {"trial_id": tid, "session_id": s["session_id"], "state": state,
             "clock_id": "clk_synthetic", "error": None, "requested_word_id": "INSA",
             "end_reason": None, "timing": {},
             "result": ({"result_id": f"res_{i:02d}", "observation_status": "OK",
                         "end_reason": "manual", "origin": "synthetic", "timing": {}}
                        if state == "FEEDBACK" else None)}
        store.trials[tid] = t
        s["trial_ids"].append(tid)
    A.store = store
    return store, s


def rows(run_dir, event="trial_closed"):
    p = os.path.join(run_dir, A.TIMING_EVENTS)
    if not os.path.exists(p):
        return []
    return [r for r in (json.loads(line) for line in open(p, encoding="utf-8") if line.strip())
            if r.get("event") == event]


@check("the reported gap: the pre-fix lease-expiry and close paths wrote nothing")
def _reproduce_gap():
    tmp = tempfile.mkdtemp()
    try:
        store, s = fixture(tmp, ("RECORDING", "RECORDING"), expired=True)

        # the pre-fix bodies of both branches: state is changed, nothing is recorded
        for tid in s["trial_ids"]:
            t = store.trials[tid]
            if t["state"] not in ("FEEDBACK",) + A.NON_SUCCESS_STATES:
                t["state"] = "CANCELLED"
                t["error"] = "SESSION_LEASE_EXPIRED"
        assert rows(s["run_dir"]) == [], "the old path was not supposed to write anything"
        rep = L.report(s["run_dir"])
        MEASURED["gap"] = {"trials_cancelled": 2,
                           "rows_written_by_old_path": 0,
                           "counted_by_the_aggregate": rep["accounting"][
                               "trials_closed_without_a_result"]}
        assert rep["accounting"]["trials_closed_without_a_result"] == 0, \
            "the gap is that they were invisible"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("lease expiry now leaves one row per cancelled trial, and the aggregate counts them")
def _lease_expiry():
    tmp = tempfile.mkdtemp()
    try:
        store, s = fixture(tmp, ("RECORDING", "END_PENDING"), expired=True)
        dead = store.reap_expired()
        assert dead == [s["session_id"]], dead
        written = rows(s["run_dir"])
        assert len(written) == 2, written
        assert {w["detail"] for w in written} == {"SESSION_LEASE_EXPIRED"}
        assert all(w["outcome"] == "CANCELLED" and w["has_result"] is False for w in written)
        rep = L.report(s["run_dir"])
        assert rep["accounting"]["trials_closed_without_a_result"] == 2
        assert rep["latency_headline"]["n"] == 0
        MEASURED["lease_expiry"] = {"rows": len(written),
                                    "breakdown": rep["accounting"]["closed_breakdown"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("session close leaves one row per cancelled trial and reports persistence per trial")
def _session_close():
    tmp = tempfile.mkdtemp()
    try:
        store, s = fixture(tmp, ("RECORDING", "FEEDBACK"))
        r = A.close_session(s["session_id"])
        written = rows(s["run_dir"])
        assert len(written) == 1, "only the unfinished trial should be closed"
        assert written[0]["detail"] == "SESSION_CLOSED"
        assert len(r["cancelled_trials"]) == 1 and r["telemetry_unpersisted"] == 0, r
        assert r["cancelled_trials"][0]["telemetry_persisted"] is True
        MEASURED["session_close"] = {"rows": len(written), "response": r["cancelled_trials"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("a trial closed twice writes one row, and the second call reports the first outcome")
def _no_duplicate_rows():
    tmp = tempfile.mkdtemp()
    try:
        store, s = fixture(tmp, ("RECORDING",), expired=True)
        store.reap_expired()
        first = rows(s["run_dir"])
        assert len(first) == 1, first
        s["state"] = "OPEN"                       # force the close path over an already-closed trial
        r = A.close_session(s["session_id"])
        again = A.close_session(s["session_id"])
        t = store.trials[s["trial_ids"][0]]
        ok, err = A._closed_event(s, t, "CANCELLED", "DIRECT_REPEAT")
        assert len(rows(s["run_dir"])) == 1, "a duplicate row was written"
        assert ok is True and err is None, (ok, err)
        assert t["closed_event"]["detail"] == "SESSION_LEASE_EXPIRED", \
            "the first outcome was overwritten"
        assert r["cancelled_trials"] == [] and again["cancelled_trials"] == []
        MEASURED["idempotent"] = {"rows_after_three_closes": len(rows(s["run_dir"])),
                                  "kept_detail": t["closed_event"]["detail"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("a failed write shows up in the trial record, the response and the health counter")
def _write_failure_visible():
    tmp = tempfile.mkdtemp()
    try:
        store, s = fixture(tmp, ("RECORDING",))
        # make the telemetry path unwritable by turning the target into a directory
        os.makedirs(os.path.join(s["run_dir"], A.TIMING_EVENTS), exist_ok=True)
        r = A.close_session(s["session_id"])
        t = store.trials[s["trial_ids"][0]]
        assert r["telemetry_unpersisted"] == 1, r
        assert r["cancelled_trials"][0]["telemetry_persisted"] is False
        assert r["cancelled_trials"][0]["telemetry_error"], "no reason was given"
        assert t["closed_event"]["persisted"] is False
        assert t["closed_event"]["error"], "the trial record lost the reason"
        assert store.telemetry_write_failures == 1, store.telemetry_write_failures
        assert t["state"] == "CANCELLED", "the cancellation itself must still have happened"
        MEASURED["write_failure"] = {
            "response_unpersisted": r["telemetry_unpersisted"],
            "health_counter": store.telemetry_write_failures,
            "error_class": t["closed_event"]["error"].split(":")[0]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("a finished trial is never given a closed row by either path")
def _finished_untouched():
    tmp = tempfile.mkdtemp()
    try:
        store, s = fixture(tmp, ("FEEDBACK",), expired=True)
        store.reap_expired()
        assert rows(s["run_dir"]) == [], "a finished trial was closed"
        s["state"] = "OPEN"
        A.close_session(s["session_id"])
        assert rows(s["run_dir"]) == [], "a finished trial was closed on session close"
        assert store.trials[s["trial_ids"][0]]["state"] == "FEEDBACK"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
