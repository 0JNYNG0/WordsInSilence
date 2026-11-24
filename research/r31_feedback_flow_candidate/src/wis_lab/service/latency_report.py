"""Aggregate display-acknowledgement latency from the persisted telemetry, and nothing else.

Reads only <run_dir>/timing_events.jsonl, so a report can be produced after a restart with no
process state. What the numbers are, stated before they are used:

  * the metric is server end-candidate to display ack received, on the server's monotonic clock,
    return trip included. It is a PROXY for feedback latency, never a measurement of motion end to
    feedback seen. That quantity stays UNVERIFIED until a person confirms by hand which frame the
    hand actually stopped on, and is not reported here.
  * observation_status == OK means the observation met its conditions. It does NOT mean the sign
    was performed correctly, so the headline group is named for the observation condition, not for
    success. No group here is a performance figure.

Grouping, rather than one average that hides its own contents:
  * the headline group requires observation_status OK, a NORMAL end reason, and a document that
    reported itself visible. A trial stopped by the duration cap (timeout_max_duration) ended
    because the cap fired, not because the signer finished, so it is aggregated on its own line.
  * document_visibility missing is unverified, not visible. It gets its own line and never joins
    the headline average.
  * a hidden tab throttles requestAnimationFrame, so its acknowledgement is not a field latency.
    Its own line as well.
  * a trial that never produced a result - cancelled, or a technical error - is counted from its
    trial_closed row. Without that row it would vanish from the denominator instead of being
    reported, so its absence would be silent rather than visible.

No time is ever invented: a trial with no acknowledgement is counted as missing with the reason it
ended, and an empty sample reports nulls rather than a number.

Run: py -3.13 -m wis_lab.service.latency_report <run_dir>
"""
from __future__ import annotations

import collections
import json
import os
import sys

METRIC = "server_end_candidate_to_display_ack_ms"
SERVER_CLOCK = "time.perf_counter_ns, monotonic, one server process"
EVENTS = "timing_events.jsonl"
OK_STATUS = "OK"
NORMAL_END = ("manual", "auto_hands_lowered")
CAPPED_END = ("timeout_max_duration",)
HEADLINE = "observation_ok_normal_end_visible"


def _read(run_dir: str):
    path = os.path.join(run_dir, EVENTS)
    rows, bad = [], 0
    if not os.path.exists(path):
        return rows, bad, path
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad, path


def _stats(values):
    """mean, median and p95 with the rule for each stated, or nulls when there is no sample.

    median: the middle value, or the mean of the two middle values on an even sample.
    p95: nearest-rank, ceil(0.95 * n) on the sorted sample. Nearest-rank is used for the tail
    because interpolating a tail from a handful of points invents precision that is not there.
    """
    if not values:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None,
                "note": "no qualifying sample; N/A rather than a fabricated number"}
    v = sorted(values)
    n = len(v)
    mid = (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0)
    k = max(1, min(n, int(-(-0.95 * n // 1))))
    return {"n": n, "mean_ms": round(sum(v) / n, 2), "median_ms": round(mid, 2),
            "p95_ms": round(v[k - 1], 2),
            "median_rule": "middle value; mean of the two middle values on an even sample",
            "p95_rule": "nearest-rank, ceil(0.95*n) on the sorted sample"}


def _group_of(res, ack):
    """Which line this acknowledgement belongs on, or None with a reason for exclusion."""
    if res.get("observation_status") != OK_STATUS:
        return None, f"observation_not_ok:{res.get('observation_status')}"
    vis = ack.get("document_visibility")
    if vis is None:
        return "visibility_unverified", None
    if vis != "visible":
        return f"document_{vis}", None
    end = res.get("end_reason")
    if end in NORMAL_END:
        return HEADLINE, None
    if end in CAPPED_END:
        return "observation_ok_duration_capped_visible", None
    return f"observation_ok_other_end_{end}_visible", None


def report(run_dir: str) -> dict:
    rows, malformed, path = _read(run_dir)
    results = {r["result_id"]: r for r in rows if r.get("event") == "trial_result"}
    closed = [r for r in rows if r.get("event") == "trial_closed"]
    acks, duplicates = {}, 0
    for r in rows:
        if r.get("event") != "display_ack":
            continue
        if r["result_id"] in acks:
            duplicates += 1          # should not happen: the server keeps only the first ack
            continue
        acks[r["result_id"]] = r

    groups = collections.defaultdict(list)
    excluded = collections.Counter()
    for rid, res in results.items():
        ack = acks.get(rid)
        if ack is None:
            key = "observation_not_ok" if res.get("observation_status") != OK_STATUS else "no_ack"
            excluded[f"{key}:no_acknowledgement:{res.get('end_reason') or 'unknown'}"] += 1
            continue
        value = ack.get(METRIC)
        if value is None:
            excluded["acknowledgement_without_a_measurement"] += 1
            continue
        group, reason = _group_of(res, ack)
        if group is None:
            excluded[reason] += 1
            continue
        groups[group].append(value)

    by_end = collections.Counter(f"{r.get('end_reason')}" for r in results.values())
    by_status = collections.Counter(f"{r.get('observation_status')}" for r in results.values())
    by_closed = collections.Counter(f"{r.get('outcome')}:{r.get('detail') or r.get('state')}"
                                    for r in closed)
    return {
        "run_dir": run_dir, "source_file": path, "file_present": os.path.exists(path),
        "metric": METRIC, "clock": SERVER_CLOCK,
        "metric_definition": "server end-candidate to display ack received, return trip included",
        "motion_end_to_feedback_seen": "UNVERIFIED - not measured, and not derivable from an ack",
        "observation_ok_meaning": "the observation met its conditions; NOT a verdict that the sign "
                                  "was performed correctly. No group here is a performance figure.",
        "headline_group": HEADLINE,
        "headline_inclusion_rule": ("observation_status OK, end_reason in "
                                    f"{list(NORMAL_END)}, document_visibility == 'visible', and "
                                    "exactly one display_ack carrying the metric"),
        "latency_by_group": {k: _stats(v) for k, v in sorted(groups.items())},
        "latency_headline": _stats(groups.get(HEADLINE, [])),
        "accounting": {
            "trial_results": len(results),
            "trials_closed_without_a_result": len(closed),
            "closed_breakdown": dict(by_closed),
            "acknowledgements": len(acks),
            "results_by_end_reason": dict(by_end),
            "results_by_observation_status": dict(by_status),
            "excluded": dict(excluded),
            "excluded_total": sum(excluded.values()),
            "duplicate_ack_rows_in_file": duplicates,
            "acks_without_a_matching_result": sorted(set(acks) - set(results)),
            "malformed_lines": malformed,
        },
        "known_gaps": [
            "sessions recorded before this telemetry existed have no rows at all; they are left "
            "missing and no time is back-filled for them",
        ],
        "close_paths_covered": ["explicit cancel", "session close", "lease expiry",
                                "NO_FRAMES_RECORDED", "CAMERA_GONE"],
        "close_path_note": "each of those writes exactly one trial_closed row per trial, so a "
                           "trial that never produced a result is counted rather than dropped",
    }


def main(argv) -> int:
    if len(argv) != 2:
        print("usage: py -3.13 -m wis_lab.service.latency_report <run_dir>")
        return 2
    print(json.dumps(report(argv[1]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
