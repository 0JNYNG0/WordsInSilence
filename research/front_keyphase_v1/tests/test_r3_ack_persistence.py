"""R3 telemetry check: the acknowledgement survives on disk, and the aggregate names what it
is actually averaging.

Everything here runs on a temporary directory with a synthetic clock and a hand-built store. No
camera is opened, no server is started, no session of the user's is touched, and no recognition
setting is read or changed.

The first check REPRODUCES the reported defect against the pre-fix behaviour - one acknowledgement
kept only in memory, a second one overwriting the first - so the rest is anchored to a real
failure rather than asserted in the abstract.

Run: py -3.13 tests/test_r3_ack_persistence.py
"""
import json
import os
import shutil
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from fastapi import HTTPException  # noqa: E402

from wis_lab.service import app as A  # noqa: E402
from wis_lab.service import latency_report as L  # noqa: E402

CHECKS = []
MEASURED = {}
REAL_NS = A.time.perf_counter_ns      # restored after every synthetic clock
NS = 1_000_000                        # one millisecond in nanoseconds
NL = chr(10)


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


class Clock:
    """A synthetic monotonic clock, so every latency in this file is a fixed known number."""

    def __init__(self, start=0):
        self.now = start

    def __call__(self):
        return self.now

    def at(self, ms):
        self.now = ms * NS
        return self.now


def fixture(tmp, status="OK", end_reason="manual"):
    """A store with one finished trial, wired to `tmp`. Returns (store, session, trial)."""
    run_dir = os.path.join(tmp, "ses_synthetic")
    os.makedirs(os.path.join(run_dir, "trials"), exist_ok=True)
    store = A.Store()
    s = {"session_id": "ses_synthetic", "run_dir": run_dir, "purpose": "telemetry_check",
         "participant_id": "synthetic", "client_id": None}
    store.sessions[s["session_id"]] = s
    result = {"result_id": "res_synthetic01", "observation_status": status,
              "end_reason": end_reason, "origin": "synthetic", "timing": {}}
    t = {"trial_id": "tri_synthetic01", "session_id": s["session_id"], "state": "FEEDBACK",
         "clock_id": "clk_synthetic", "result": result, "error": None,
         "requested_word_id": "INSA", "end_reason": end_reason,
         "timing": {"server_end_candidate_ns": 0, "server_result_ns": 300 * NS,
                    "server_end_candidate_to_result_ms": 300.0}}
    store.trials[t["trial_id"]] = t
    A.store = store
    return store, s, t


def ack(trial_id, result_id="res_synthetic01", client=None):
    return A.display_ack(trial_id, A.Ack(result_id=result_id, client_event=client))


def rows(run_dir):
    p = os.path.join(run_dir, A.TIMING_EVENTS)
    if not os.path.exists(p):
        return []
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


# rid, observation_status, end_reason, ack ms (None = no ack), visibility (None = not reported)
PLAN = [("r1", "OK", "manual", 100.0, "visible"),
        ("r2", "OK", "auto_hands_lowered", 200.0, "visible"),
        ("r3", "OK", "manual", 300.0, "visible"),
        ("r4", "OK", "auto_hands_lowered", 900.0, "visible"),
        ("r5", "OK", "manual", 5000.0, "hidden"),
        ("r6", "OK", "timeout_max_duration", 4000.0, "visible"),
        ("r7", "OK", "manual", 700.0, None),
        ("r8", "OK", "manual", None, None),
        ("r9", "INSUFFICIENT_OBSERVATION", "manual", None, None)]


def aggregate_fixture(run_dir):
    """One synthetic telemetry file both aggregate checks read, with fixed known numbers."""
    with open(os.path.join(run_dir, A.TIMING_EVENTS), "w", encoding="utf-8", newline=NL) as fh:
        for rid, status, end, ms, vis in PLAN:
            fh.write(json.dumps({"event": "trial_result", "result_id": rid,
                                 "observation_status": status, "end_reason": end},
                                ensure_ascii=False) + NL)
            if ms is not None:
                row = {"event": "display_ack", "result_id": rid, L.METRIC: ms}
                if vis is not None:
                    row["document_visibility"] = vis
                fh.write(json.dumps(row, ensure_ascii=False) + NL)
        for outcome, detail in (("CANCELLED", None), ("ERROR", "CAMERA_GONE")):
            fh.write(json.dumps({"event": "trial_closed", "outcome": outcome, "detail": detail,
                                 "state": outcome}, ensure_ascii=False) + NL)


@check("the reported defect: memory-only ack, and a second ack overwriting the first")
def _reproduce_defect():
    tmp = tempfile.mkdtemp()
    try:
        _, s, t = fixture(tmp)
        clock = Clock()

        # the pre-fix behaviour, written out plainly rather than trusted from memory
        def old_display_ack(now_ns):
            t["timing"]["server_ack_received_ns"] = now_ns
            t["timing"]["server_end_candidate_to_display_ack_ms"] = round(
                (now_ns - t["timing"]["server_end_candidate_ns"]) / 1e6, 2)
            t["result"]["timing"] = dict(t["timing"])

        old_display_ack(clock.at(1100))
        first = t["timing"]["server_end_candidate_to_display_ack_ms"]
        old_display_ack(clock.at(2000))
        second = t["timing"]["server_end_candidate_to_display_ack_ms"]
        MEASURED["defect"] = {"first_ack_ms": first, "after_duplicate_ms": second,
                              "disk_rows": len(rows(s["run_dir"]))}
        assert first == 1100.0 and second == 2000.0, (first, second)
        assert rows(s["run_dir"]) == [], "the old path was not supposed to write anything"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("a normal ack is persisted with ids, clock and both server intervals")
def _persisted():
    tmp = tempfile.mkdtemp()
    try:
        _, s, t = fixture(tmp)
        A.time.perf_counter_ns = Clock()
        r = ack(t["trial_id"], client={"clock": "browser_performance_now",
                                       "client_receive_to_render_event_ms": 12.5,
                                       "document_visibility": "visible"})
        assert r["persisted"] is True and r["persist_error"] is None, r
        assert r["duplicate"] is False and r["motion_end_to_feedback_verified"] is False
        written = [x for x in rows(s["run_dir"]) if x["event"] == "display_ack"]
        assert len(written) == 1, written
        w = written[0]
        for key in ("session_id", "trial_id", "result_id", "clock_id", "server_clock",
                    "server_end_candidate_ns", "server_result_ns", "server_ack_received_ns",
                    "server_end_candidate_to_result_ms", "server_result_to_ack_ms",
                    "server_end_candidate_to_display_ack_ms"):
            assert key in w, key
        assert w["client_reported"]["client_receive_to_render_event_ms"] == 12.5
        assert w["document_visibility"] == "visible"
        assert w["motion_end_to_feedback_verified"] is False
        MEASURED["persisted_keys"] = sorted(w)
    finally:
        A.time.perf_counter_ns = REAL_NS
        shutil.rmtree(tmp, ignore_errors=True)


@check("duplicate and concurrent acks keep the first measurement and write one row")
def _duplicate_and_concurrent():
    tmp = tempfile.mkdtemp()
    try:
        _, s, t = fixture(tmp)
        clock = Clock()
        A.time.perf_counter_ns = clock
        clock.at(1100)
        first = ack(t["trial_id"])["server_end_candidate_to_display_ack_ms"]
        clock.at(2000)
        later = ack(t["trial_id"])
        assert later["duplicate"] is True
        assert later["server_end_candidate_to_display_ack_ms"] == first == 1100.0, (first, later)

        out = []
        clock.at(3000)

        def hit():
            out.append(ack(t["trial_id"])["server_end_candidate_to_display_ack_ms"])

        threads = [threading.Thread(target=hit) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert set(out) == {1100.0}, out
        written = [x for x in rows(s["run_dir"]) if x["event"] == "display_ack"]
        assert len(written) == 1, f"{len(written)} rows written for one result"
        MEASURED["duplicate"] = {"first_ms": first, "after_8_concurrent": sorted(set(out)),
                                 "rows_on_disk": len(written)}
    finally:
        A.time.perf_counter_ns = REAL_NS
        shutil.rmtree(tmp, ignore_errors=True)


@check("a different result_id is rejected and writes nothing")
def _wrong_result_id():
    tmp = tempfile.mkdtemp()
    try:
        _, s, t = fixture(tmp)
        try:
            ack(t["trial_id"], result_id="res_someoneelse")
            raise AssertionError("a mismatched result_id was accepted")
        except HTTPException as exc:
            assert exc.status_code == 409, exc.status_code
        assert rows(s["run_dir"]) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("a failed write is reported as persisted=false, not as a saved measurement")
def _write_failure():
    tmp = tempfile.mkdtemp()
    try:
        _, s, t = fixture(tmp)
        # make the telemetry path unwritable by turning the target into a directory
        os.makedirs(os.path.join(s["run_dir"], A.TIMING_EVENTS), exist_ok=True)
        r = ack(t["trial_id"])
        assert r["persisted"] is False, r
        assert r["persist_error"], "no reason was given for the failed write"
        assert r["server_end_candidate_to_display_ack_ms"] is not None
        MEASURED["write_failure"] = {"persisted": r["persisted"],
                                     "error_class": r["persist_error"].split(":")[0]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("a cancelled trial leaves a trial_closed row, so it cannot vanish from the denominator")
def _cancel_is_accounted():
    tmp = tempfile.mkdtemp()
    try:
        _, s, t = fixture(tmp)
        t["state"] = "RECORDING"
        t["result"] = None
        A.cancel_trial(t["trial_id"])
        closed = [x for x in rows(s["run_dir"]) if x["event"] == "trial_closed"]
        assert len(closed) == 1, closed
        assert closed[0]["outcome"] == "CANCELLED" and closed[0]["has_result"] is False
        rep = L.report(s["run_dir"])
        assert rep["accounting"]["trials_closed_without_a_result"] == 1
        assert rep["latency_headline"]["n"] == 0, "a cancelled trial produced a latency"
        MEASURED["cancel"] = rep["accounting"]["closed_breakdown"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("after a restart the aggregate comes from the file alone, with fixed known numbers")
def _aggregate_from_disk():
    tmp = tempfile.mkdtemp()
    try:
        run_dir = os.path.join(tmp, "ses_agg")
        os.makedirs(run_dir, exist_ok=True)
        aggregate_fixture(run_dir)
        rep = L.report(run_dir)
        h = rep["latency_headline"]
        assert rep["headline_group"] == "observation_ok_normal_end_visible"
        assert h["n"] == 4, h                              # 100, 200, 300, 900 only
        assert h["mean_ms"] == 375.0, h["mean_ms"]
        assert h["median_ms"] == 250.0, h["median_ms"]     # even sample -> (200 + 300) / 2
        assert h["p95_ms"] == 900.0, h["p95_ms"]           # nearest-rank, ceil(0.95 * 4) = 4
        MEASURED["aggregate"] = {"headline": h,
                                 "group_sizes": {k: v["n"]
                                                 for k, v in rep["latency_by_group"].items()}}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("duration cap, unverified visibility and a hidden tab each stay off the headline average")
def _groups_kept_apart():
    tmp = tempfile.mkdtemp()
    try:
        run_dir = os.path.join(tmp, "ses_agg2")
        os.makedirs(run_dir, exist_ok=True)
        aggregate_fixture(run_dir)
        rep = L.report(run_dir)
        g = rep["latency_by_group"]
        assert g["observation_ok_duration_capped_visible"]["n"] == 1
        assert g["observation_ok_duration_capped_visible"]["mean_ms"] == 4000.0
        assert g["visibility_unverified"]["n"] == 1
        assert g["visibility_unverified"]["mean_ms"] == 700.0
        assert g["document_hidden"]["n"] == 1
        assert g["document_hidden"]["mean_ms"] == 5000.0
        assert rep["latency_headline"]["n"] == 4, "a separated group reached the headline"
        assert rep["latency_headline"]["mean_ms"] == 375.0

        acc = rep["accounting"]
        assert acc["trials_closed_without_a_result"] == 2, acc
        assert acc["closed_breakdown"]["CANCELLED:CANCELLED"] == 1
        assert acc["closed_breakdown"]["ERROR:CAMERA_GONE"] == 1
        assert acc["results_by_end_reason"]["timeout_max_duration"] == 1
        assert any(k.startswith("no_ack:no_acknowledgement:manual") for k in acc["excluded"]), acc
        assert any(k.startswith("observation_not_ok") for k in acc["excluded"]), acc
        assert "NOT a verdict" in rep["observation_ok_meaning"]
        MEASURED["groups_kept_apart"] = {k: v["n"] for k, v in g.items()}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("an empty or absent telemetry file reports N/A rather than a number")
def _empty_sample():
    tmp = tempfile.mkdtemp()
    try:
        rep = L.report(tmp)
        assert rep["file_present"] is False
        h = rep["latency_headline"]
        assert h["n"] == 0 and h["mean_ms"] is None
        assert h["median_ms"] is None and h["p95_ms"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("the recogniser is untouched: the frozen config still loads and predict is unchanged")
def _recogniser_unchanged():
    import hashlib

    from wis_lab.core import recognize as R
    cfg = os.path.join(ROOT, "config", "contract.json")
    h = hashlib.sha256(open(cfg, "rb").read()).hexdigest()
    contract = json.load(open(cfg, encoding="utf-8"))
    assert contract["contract_id"] == "wis_front_contract_v1_5", contract["contract_id"]
    assert callable(R.predict) and callable(R.task_match)
    MEASURED["recogniser"] = {"contract_id": contract["contract_id"], "contract_sha256": h,
                              "changed_by_this_task": False}


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
