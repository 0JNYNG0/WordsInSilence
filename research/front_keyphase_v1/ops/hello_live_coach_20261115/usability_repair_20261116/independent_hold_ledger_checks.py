"""Independent RAM-reset and plateau-query contracts; no model or device access."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

V = Path(__file__).resolve().parent
U = V.parent
sys.path.insert(0, str(U))
from independent_checks import points, shapes
from diagnostic_ledger import DiagnosticLedger


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-hold", action="store_true")
    ap.add_argument("--output", type=Path, default=V / "INDEPENDENT_LEDGER_FIX.json")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous reviews; select a new output path")
    pins = {name: digest(U / name) for name in ["diagnostic_ledger.py", "coach.py"]}
    rows = []

    def check(name, condition, **details):
        rows.append(dict(name=name, passed=bool(condition), **details))

    ledger = DiagnosticLedger()
    ledger.begin("synthetic_public_status", 1)
    normal = dict(assessed_timestamp_us=1000000, stage_index=1,
                  diagnostic=dict(category="progress", code="moving"))
    ledger.observe(normal, {})
    lost = dict(assessed_timestamp_us=None, stage_index=1,
                diagnostic=dict(category="observation_delay", code="warming_up",
                                reset_reason="capture_gap_exceeded"),
                assessed_xy=[[123, 456]], image="DO_NOT_RETAIN_IMAGE")
    metrics = dict(elapsed_seconds=1.4, arbitrary="DO_NOT_RETAIN_METRIC")
    ledger.observe(lost, metrics)
    s = ledger.snapshot()["session"]
    check("missing_assessment_capture_reset_recorded", s["resets"] == 1 and
          s["last_reason"]["reset_reason"] == "capture_gap_exceeded" and
          s["last_reason"]["relative_ms"] == 400.0 and
          s["last_reason"]["time_basis"] == "received_metric")
    ledger.observe(lost, metrics)
    check("duplicate_fallback_reset_not_double_counted", ledger.snapshot()["session"]["observed_updates"] == 2)
    warm = copy.deepcopy(lost); warm["diagnostic"].pop("reset_reason")
    ledger.observe(warm, dict(elapsed_seconds=1.5))
    check("fallback_only_for_explicit_reset", ledger.snapshot()["session"]["observed_updates"] == 2)
    ledger.observe(lost, {})
    check("no_invented_timestamp_without_metric", ledger.snapshot()["session"]["observed_updates"] == 2)
    dimensional = copy.deepcopy(lost)
    dimensional["diagnostic"]["reset_reason"] = "source_dimensions_changed"
    ledger.observe(dimensional, dict(elapsed_seconds=1.6))
    s = ledger.snapshot()["session"]
    check("dimension_reset_reason_distinct", s["resets"] == 2 and s["last_reason"]["relative_ms"] == 600)
    normal["assessed_timestamp_us"] = 1600000
    ledger.observe(normal, {})
    s = ledger.snapshot()["session"]
    check("same_numeric_time_different_basis_not_deduplicated", s["observed_updates"] == 4 and
          s["last_reason"]["time_basis"] == "assessed" and s["counts"]["progress"] == 2)
    text = json.dumps(ledger.snapshot())
    check("fallback_retains_no_pixels_coordinates_or_arbitrary_fields",
          all(v not in text for v in ["assessed_xy", "DO_NOT_RETAIN", "123", "456"]))
    ledger.end("done"); frozen = ledger.snapshot()
    ledger.observe(dimensional, dict(elapsed_seconds=2.0))
    check("ended_session_rejects_fallback_updates", ledger.snapshot() == frozen)
    ledger.begin("synthetic_public_status", 2)
    s = ledger.snapshot()
    check("new_generation_isolated_previous_reset_summary_preserved",
          s["session"]["resets"] == 0 and s["recent_sessions"][-1]["resets"] == 2)

    if args.include_hold:
        from coach import Coach

        class WindowDirection:
            """Controller-only stand-in: direction needs a fresh 3-point window.

            This emulates the observed query sensitivity without executing or
            claiming neural direction accuracy. It does not authorize a model.
            """
            def reset(self):
                self.samples = {}; self.queries = []

            def movement(self, start, end, axis, moving, anchor):
                sample = [(t, y) for t, y in self.samples.items() if start <= t <= end]
                delta = sample[-1][1] - sample[0][1] if sample else 0
                eligible = len(sample) >= 3 and end - start <= 200000
                state = 0 if delta > .001 else 2 if delta < -.001 else 1
                self.queries.append(dict(start=start, end=end, samples=len(sample), moving=moving,
                                         eligible=eligible, state=state))
                return dict(eligible=eligible, state=state)

        action = WindowDirection()
        c = Coach(model=object(), action=action)
        c.stage_index = 1; c.completed[0] = True
        now = 1000000

        def feed(y, unknown=False):
            nonlocal now
            now += 33333
            action.samples[now] = y
            return c.consume_observation(points(y=y), now, 1000, 800,
                                         shapes(-1, -1) if unknown else shapes(1, 1))

        for _ in range(12): feed(0)
        high_end = now
        check("plateau_alone_cannot_count_first_down", c.repeats == 0 and not c.events and not c.completed[1])
        feed(.03); feed(.08)
        check("down_uses_latest_exact_equal_high", c.repeats == 1 and c.high_ts == high_end,
              expected_start=high_end, observed_start=c.high_ts)
        for _ in range(12): feed(.08)
        low_end = now; held_events = len(c.events)
        check("bottom_plateau_alone_cannot_rearm", c.repeats == 1 and c.turn == "up" and held_events == 1)
        feed(.06); feed(.02)
        up_queries = [q for q in action.queries if q["state"] == 2]
        check("real_up_after_plateau_uses_latest_equal_low", c.turn == "down" and bool(up_queries) and
              all(q["start"] == low_end and q["samples"] == 3 and q["eligible"] for q in up_queries),
              upward_queries=up_queries)
        feed(0)
        for _ in range(12): feed(0)
        check("second_high_hold_does_not_count", c.repeats == 1 and not c.completed[1])
        feed(.03); feed(.08)
        check("two_visible_strokes_and_visible_return_complete", c.completed == [True, True] and c.repeats == 2)

        # Recheck the dangerous boundary with long exact plateaus surrounding it.
        action = WindowDirection(); c = Coach(model=object(), action=action)
        c.stage_index = 1; c.completed[0] = True
        for y in [0, .03, .08]: feed(y)
        for _ in range(12): feed(.08)
        feed(.08, unknown=True); feed(0)
        for y in [.03, .08] + [.08] * 12: feed(y)
        check("plateau_refresh_does_not_accept_hidden_upward_return", c.repeats == 1 and not c.completed[1])

    report = dict(passed=all(r["passed"] for r in rows), passes=sum(r["passed"] for r in rows),
                  total=len(rows), checks=rows, code_sha256=pins,
                  unchanged_during_review=all(digest(U / name) == value for name, value in pins.items()),
                  camera=False, server_access=False, model_execution=False, training=False,
                  private_data=False, note=__doc__)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["passed", "passes", "total", "unchanged_during_review"]}))
    for row in rows:
        if not row["passed"]: print(json.dumps(row))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
