"""Offline runner for the R28 projected-arm candidate.

    python run_geometry_challenge.py --input PATH --output PATH
    python run_geometry_challenge.py --synthetic-controls --output PATH

Input is a JSON list of {"id": str, "report": selected_interval_feedback_v1,
"range": [start, end]} records; extra fields are ignored. Reports are read only, never
written back, and the output carries angle summaries plus availability counts only --
no landmark arrays, no thresholds, no retraining, no acceptance decision.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from projected_arm_review import MEASURES, SIDES, review_projected_arm, synthetic_controls

OUTPUT_SCHEMA = "projected_arm_challenge_v1"
NOTES = [
    "Angle summaries are 2D projected interior angles in shoulder-frame units; not 3D flexion.",
    "Availability is aggregated across records; angle values are NOT pooled across records, "
    "because pooling different bodies/clips would invent a population statistic.",
    "No threshold is selected, no model is trained, and accepted/correction stay null.",
]


def load_records(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("input must be a JSON list of records")
    return payload


def review_record(record, index):
    if not isinstance(record, dict):
        return {"id": None, "status": "UNAVAILABLE", "reason": "invalid_record", "review": None}
    ident = record.get("id")
    ident = str(ident) if ident is not None else "record_%d" % index
    review = review_projected_arm(record.get("report"), record.get("range"))
    return {"id": ident, "status": review["status"], "reason": review.get("reason"),
            "review": review if review["status"] == "AVAILABLE" else None}


def aggregate(records):
    by_measure = {}
    for side in SIDES:
        for measure in MEASURES:
            by_measure["%s.%s" % (side, measure)] = {
                "records_with_any_paired_step": 0, "steps_in_range": 0,
                "reference_samples_in_range": 0, "query_observed_steps": 0,
                "reference_observed_steps": 0, "paired_observed_steps": 0,
                "paired_observed_samples": 0, "unavailable_reasons": {},
            }
    reasons = {}
    available = 0
    for entry in records:
        if entry["status"] != "AVAILABLE":
            key = entry.get("reason") or "unknown"
            reasons[key] = reasons.get(key, 0) + 1
            continue
        available += 1
        for side in SIDES:
            for measure in MEASURES:
                summary = entry["review"]["sides"][side][measure]
                bucket = by_measure["%s.%s" % (side, measure)]
                counts = summary["counts"]
                for key in ("steps_in_range", "reference_samples_in_range",
                            "query_observed_steps", "reference_observed_steps",
                            "paired_observed_steps", "paired_observed_samples"):
                    bucket[key] += counts[key]
                if counts["paired_observed_steps"]:
                    bucket["records_with_any_paired_step"] += 1
                for who, table in summary["unavailable_reasons"].items():
                    for reason, count in table.items():
                        name = "%s:%s" % (who, reason)
                        bucket["unavailable_reasons"][name] = \
                            bucket["unavailable_reasons"].get(name, 0) + count
    for bucket in by_measure.values():
        total = bucket["steps_in_range"]
        bucket["paired_step_availability"] = \
            (bucket["paired_observed_steps"] / total) if total else None
    return {"records_total": len(records), "records_available": available,
            "records_unavailable_reasons": reasons, "by_side_measure": by_measure}


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, sort_keys=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="R28 projected arm geometry review")
    parser.add_argument("--input", help="JSON list of {id, report, range} records")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--synthetic-controls", action="store_true",
                        help="run the controlled synthetic geometry experiment instead")
    args = parser.parse_args(argv)

    if not args.synthetic_controls and not args.input:
        parser.error("--input is required unless --synthetic-controls is given")
    if args.input and os.path.abspath(args.input) == os.path.abspath(args.output):
        parser.error("--output must not overwrite --input")

    if args.synthetic_controls:
        controls = synthetic_controls()
        payload = {"schema": OUTPUT_SCHEMA, "mode": "synthetic_controls",
                   "synthetic_controls": controls, "records": [], "aggregate": None,
                   "notes": NOTES + ["Synthetic fixture values are geometry, never accuracy."]}
        write_json(args.output, payload)
        print("synthetic-controls: %d experiments -> %s"
              % (len(controls["experiments"]), args.output))
        return 0

    try:
        raw = load_records(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("input error: %s" % exc, file=sys.stderr)
        return 2

    records = [review_record(record, index) for index, record in enumerate(raw)]
    summary = aggregate(records)
    payload = {"schema": OUTPUT_SCHEMA, "mode": "reports", "records": records,
               "aggregate": summary, "notes": NOTES}
    write_json(args.output, payload)
    print("records: %d available, %d unavailable -> %s"
          % (summary["records_available"],
             summary["records_total"] - summary["records_available"], args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
