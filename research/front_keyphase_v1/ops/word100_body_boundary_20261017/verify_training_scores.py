"""Independently recount saved word100 A/B/C logits; never load a model or arrays.

Preparation: --self-test runs only in-memory synthetic cases.  After all nine
runs finish, --verify reads JSON scores/metadata and checks the driver's report.
No training_core/driver, NumPy, Torch, detector, checkpoint loader, or subprocess
is imported.  A failed input remains wrong in the fixed 300-case denominator.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SEEDS = (20261013, 20261014, 20261015)
ARMS = ("A", "B", "C")
VIEWS = ("masked", "sensor")
SIGNERS = (11, 12, 13)
GROUPS = {"one_hand_never_usable", "internal_reentry", "other"}
M0_SHA = "e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f"
MANIFEST_SHA = "3140045381a3316153f0f998979a63fda8609340409ce05f09e673715ebfa76c"
FAILURE_CE = math.log(100)


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def close_number(a, b):
    return (type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-9))


def same(saved, calculated, location="value"):
    """Strict structural agreement; only finite numeric roundoff is tolerated."""
    if isinstance(calculated, dict):
        ensure(isinstance(saved, dict) and set(saved) == set(calculated),
               location + ": keys differ")
        for key, value in calculated.items():
            same(saved[key], value, location + "." + str(key))
    elif isinstance(calculated, list):
        ensure(isinstance(saved, list) and len(saved) == len(calculated),
               location + ": list length differs")
        for index, value in enumerate(calculated):
            same(saved[index], value, location + "[" + str(index) + "]")
    elif type(calculated) in (int, float):
        ensure(close_number(saved, calculated), location + ": number differs")
    else:
        ensure(type(saved) is type(calculated) and saved == calculated,
               location + ": value differs")


def counts(rows):
    n = len(rows)
    top1 = sum(r["correct"] for r in rows)
    top5 = sum(r["top5_correct"] for r in rows)
    return {
        "total": n, "top1_correct": top1, "top5_correct": top5,
        "top1_percent": 100 * top1 / n if n else None,
        "top5_percent": 100 * top5 / n if n else None,
        "mean_ce": math.fsum(r["cross_entropy"] for r in rows) / n if n else None,
        "input_failures": sum(r["status"] != "OK" for r in rows),
    }


def recount(saved, expected, failures, *, fixed300=True):
    """Recompute ranks from raw logits, never from saved correct/top5 booleans."""
    ensure(isinstance(saved, list), "Predictions must be a list")
    ids = [r["clip_id"] for r in saved]
    ensure(len(set(ids)) == len(ids), "Duplicate prediction ID")
    ensure(set(ids) == set(expected), "Missing/extra prediction ID")
    if fixed300:
        ensure(len(ids) == 300, "Expected fixed denominator 300")
        ensure(Counter(r["signer_id"] for r in expected.values()) ==
               {11: 100, 12: 100, 13: 100}, "Expected 100 cases per DEV signer")
        ensure(Counter(r["label_index"] for r in expected.values()) ==
               {c: 3 for c in range(100)}, "Expected 3 cases per word")
    ensure(set(failures) <= set(expected), "Failure ID outside DEV")
    out = []
    for r in saved:
        cid = r["clip_id"]
        e = expected[cid]
        ensure(type(r["label_index"]) is int and type(r["signer_id"]) is int,
               cid + ": noninteger identity")
        ensure(r["label_index"] == e["label_index"] and r["signer_id"] == e["signer_id"],
               cid + ": label/person identity changed")
        ensure(0 <= r["label_index"] < 100, cid + ": label outside 100 classes")
        q = {k: r[k] for k in ("clip_id", "label_index", "signer_id")}
        if cid in failures:
            ensure(r["status"] == "INPUT_UNAVAILABLE" and r.get("logits") is None,
                   cid + ": failed input has model prediction")
            ensure(isinstance(r.get("reason"), str) and r["reason"],
                   cid + ": failed input reason absent")
            if failures[cid] is not None:
                ensure(r["reason"] == failures[cid], cid + ": frozen failure reason changed")
            q.update(status="INPUT_UNAVAILABLE", prediction=None, top5=[], correct=False,
                     top5_correct=False, cross_entropy=FAILURE_CE)
        else:
            ensure(r["status"] == "OK", cid + ": unexpected input failure")
            logits = r.get("logits")
            ensure(isinstance(logits, list) and len(logits) == 100,
                   cid + ": expected 100 saved logits")
            ensure(all(type(x) in (int, float) and math.isfinite(x) for x in logits),
                   cid + ": logits must be finite numeric values")
            # Tuple sort independently enforces the frozen lower-index tie rule.
            order = sorted(range(100), key=lambda index: (-logits[index], index))
            peak = max(logits)
            ce = peak + math.log(math.fsum(math.exp(x - peak) for x in logits)) - logits[r["label_index"]]
            q.update(status="OK", prediction=order[0], top5=order[:5],
                     correct=order[0] == r["label_index"],
                     top5_correct=r["label_index"] in order[:5], cross_entropy=ce)
        for field in ("prediction", "top5", "correct", "top5_correct", "cross_entropy"):
            same(r.get(field), q[field], cid + "." + field)
        out.append(q)
    return sorted(out, key=lambda row: row["clip_id"])


def detailed(rows, group_map):
    ensure(set(group_map) == {r["clip_id"] for r in rows}, "Missingness group IDs differ")
    ensure(set(group_map.values()) <= GROUPS, "Unknown missingness group")
    report = counts(rows)
    report["by_signer"] = {str(s): counts([r for r in rows if r["signer_id"] == s]) for s in SIGNERS}
    report["by_class"] = {str(c): counts([r for r in rows if r["label_index"] == c]) for c in range(100)}
    report["by_observation_pattern"] = {
        group: counts([r for r in rows if group_map[r["clip_id"]] == group])
        for group in sorted(set(group_map.values()))}
    return report


def paired(before, after, field="correct"):
    aa = {r["clip_id"]: r for r in before}
    bb = {r["clip_id"]: r for r in after}
    ensure(len(aa) == len(before) and len(bb) == len(after) and set(aa) == set(bb),
           "Paired comparison ID mismatch")
    for cid in aa:
        ensure(all(aa[cid][k] == bb[cid][k] for k in ("label_index", "signer_id")),
               "Paired comparison identity mismatch: " + cid)
    gains = sorted(cid for cid in aa if not aa[cid][field] and bb[cid][field])
    losses = sorted(cid for cid in aa if aa[cid][field] and not bb[cid][field])
    return {"improved": gains, "regressed": losses, "net_correct": len(gains) - len(losses),
            "percentage_point_difference": 100 * (len(gains) - len(losses)) / len(aa)}


def three_seed_summary(metrics):
    ensure(len(metrics) == 3, "Exactly three seeds required")
    result = {}
    for field in ("top1_correct", "top5_correct", "top1_percent", "top5_percent", "input_failures"):
        values = [float(m[field]) for m in metrics]
        result[field] = {"mean": math.fsum(values) / 3, "minimum": min(values),
                         "maximum": max(values), "values": values}
    result["by_signer"] = {str(s): {
        "top1_correct_mean": math.fsum(m["by_signer"][str(s)]["top1_correct"] for m in metrics) / 3,
        "top1_percent_mean": math.fsum(m["by_signer"][str(s)]["top1_percent"] for m in metrics) / 3,
    } for s in SIGNERS}
    return result


def progression(metrics, baseline):
    """All decisions use integer sums of per-seed counts, with no best seed."""
    c = [metrics[("C", seed)]["sensor"] for seed in SEEDS]
    b = [metrics[("B", seed)]["sensor"] for seed in SEEDS]
    cm = [metrics[("C", seed)]["masked"] for seed in SEEDS]
    checks = {
        "C_sensor_mean_exceeds_B": sum(x["top1_correct"] for x in c) > sum(x["top1_correct"] for x in b),
        "C_sensor_mean_exceeds_M0": sum(x["top1_correct"] for x in c) > 3 * baseline["sensor"]["top1_correct"],
        "C_sensor_better_than_B_at_least_two_seeds": sum(x["top1_correct"] > y["top1_correct"] for x, y in zip(c, b)) >= 2,
        "C_sensor_failure_no_increase": max(x["input_failures"] for x in c) <= min(
            baseline["sensor"]["input_failures"], min(x["input_failures"] for x in b)),
        "original_pipeline_mean_regression_at_most_three_cases": sum(x["top1_correct"] for x in cm) >=
            3 * (baseline["native"]["top1_correct"] - 3),
    }
    for signer in SIGNERS:
        for newview, oldview in (("masked", "native"), ("sensor", "sensor")):
            total = sum(metrics[("C", seed)][newview]["by_signer"][str(signer)]["top1_correct"] for seed in SEEDS)
            reference = baseline[oldview]["by_signer"][str(signer)]["top1_correct"]
            checks[f"{newview}_signer{signer}_mean_regression_at_most_three_cases"] = total >= 3 * (reference - 3)
    return {"pass_to_next_review": all(checks.values()), "checks": checks,
            "automatic_adoption": False, "generalization_proven": False}


def verify(unit):
    pins = {}

    def read(path):
        path = Path(path).resolve()
        content = path.read_bytes()
        pins[str(path)] = hashlib.sha256(content).hexdigest()
        return json.loads(content.decode("utf-8-sig"))

    context_path = unit / "training_CONTEXT.json"
    context = read(context_path)
    context_sha = pins[str(context_path)]
    ensure(context["schema"] == "body_boundary_ABC_training_v1", "Wrong context schema")
    ensure(context["arms"] == list(ARMS) and context["seeds"] == list(SEEDS), "Arm/seed plan differs")
    ensure(context["train_count"] == 899 and context["development_count"] == 300, "Count plan differs")
    manifest_path = unit.parent / "word100_training_20261013/MANIFEST.json"
    manifest = read(manifest_path)
    ensure(pins[str(manifest_path)] == MANIFEST_SHA, "Manifest no longer frozen original")
    ensure(context["pins"][str(manifest_path)] == MANIFEST_SHA, "Context manifest pin differs")
    original = {r["clip_id"]: r for r in manifest}
    ensure(len(original) == len(manifest) == 1199, "Manifest IDs invalid")
    identities = context["ordered_rows"]
    ensure([r["clip_id"] for r in identities] == [r["clip_id"] for r in manifest], "Context manifest order changed")
    for row in identities:
        ensure(all(row[k] == original[row["clip_id"]][k] for k in
                   ("role", "signer_id", "label_index", "word_code")), "Context identity mismatch")
    train = [r for r in identities if r["role"] == "train_candidate"]
    dev = {r["clip_id"]: r for r in identities if r["role"] == "development_only"}
    ensure(len(train) == 899 and set(r["signer_id"] for r in train) == set(range(1, 10)), "TRAIN split differs")
    ensure(len(dev) == 300 and set(r["signer_id"] for r in dev.values()) == set(SIGNERS), "DEV split differs")
    groups = context["missingness_groups"]
    failures = {}
    for view in ("masked", "sensor", "m0_sensor"):
        failures[view] = {cid: row["views"][view]["reason"] for cid, row in dev.items()
                          if row["views"][view]["channel_file"] is None}
        ensure(set(failures[view]) == set(context["fixed_input_failures"][view]), "Frozen failure set differs")
    for view in VIEWS:
        ensure(set(groups[view]) == set(dev) and set(groups[view].values()) <= GROUPS, "Missingness groups invalid")

    base_path = unit.parent / "word100_roi_training_20261017/baseline_PREDICTIONS.json"
    native_saved = read(base_path)
    ensure(context["pins"][str(base_path)] == pins[str(base_path)], "Baseline score pin differs")
    native = recount(native_saved, dev, {})
    ensure(counts(native)["top1_correct"] == 290 and counts(native)["top5_correct"] == 299,
           "Frozen M0 native must reproduce 290/300 and 299/300")
    sensor_path = unit / "training_m0_sensor.json"
    sensor_saved = read(sensor_path)
    ensure(sensor_saved["status"] == "SAVED_ONCE", "M0 sensor score not complete")
    ensure(sensor_saved["training_context_sha256"] == context_sha and sensor_saved["checkpoint_sha256"] == M0_SHA,
           "M0 sensor provenance mismatch")
    sensor = recount(sensor_saved["predictions"], dev, failures["m0_sensor"])
    baseline_rows = {"native": native, "sensor": sensor}
    baseline = {"native": detailed(native, groups["masked"]), "sensor": detailed(sensor, groups["sensor"])}
    same(sensor_saved["metrics"], baseline["sensor"], "M0.sensor.metrics")

    metrics, predictions, run_details, score_paths = {}, {}, {}, [sensor_path]
    initial_hashes = {seed: [] for seed in SEEDS}
    for arm in ARMS:
        for seed in SEEDS:
            path = unit / f"training_{arm}_{seed}_RESULT.json"
            saved = read(path)
            score_paths.append(path)
            ensure(saved["status"] == "COMPLETE_DEVELOPMENT_ONLY", "Incomplete run: " + path.name)
            ensure(saved["arm"] == arm and saved["seed"] == seed and saved["context_sha256"] == context_sha,
                   "Run provenance mismatch: " + path.name)
            ensure(saved["parameter_count"] == 37636, "Parameter count changed")
            ensure(all(saved[k] is False for k in ("selected_scores_reinferred", "final_blind_test", "user_data_used", "production_model_replaced")),
                   "Unexpected experiment/deployment scope")
            ensure(type(saved["selected_epoch"]) is int and 1 <= saved["selected_epoch"] <= saved["epochs_completed"] <= 80,
                   "Invalid selected/completed epoch")
            ensure(saved["total_steps"] == saved["epochs_completed"] * 29, "Expected 29 batches per epoch")
            ensure(saved["epochs_completed"] == 80 or saved["epochs_completed"] - saved["selected_epoch"] >= 12,
                   "Run not finished under max80/patience12")
            initial_hashes[seed].append(saved["initial_state_sha256"])
            rr, mm = {}, {}
            ensure(set(saved["predictions"]) == set(VIEWS), "Expected exactly two views")
            for view in VIEWS:
                rr[view] = recount(saved["predictions"][view], dev, failures[view])
                mm[view] = detailed(rr[view], groups[view])
                same(saved["metrics"][view], mm[view], f"{arm}/{seed}/{view}.metrics")
            selected_ce = (mm["masked"]["mean_ce"] + mm["sensor"]["mean_ce"]) / 2
            same(saved["selected_ce"], selected_ce, f"{arm}/{seed}.selected_ce")
            metrics[(arm, seed)] = mm
            predictions[(arm, seed)] = rr
            run_details[f"{arm}_{seed}"] = {
                "selected_epoch": saved["selected_epoch"], "epochs_completed": saved["epochs_completed"],
                "checkpoint_sha256_as_recorded": saved["checkpoint_sha256"],
                "selected_ce_independent": selected_ce, "metrics": mm,
            }
    ensure(all(len(set(values)) == 1 for values in initial_hashes.values()), "A/B/C did not share same-seed initialization")
    arms = {a: {v: three_seed_summary([metrics[(a, s)][v] for s in SEEDS]) for v in VIEWS} for a in ARMS}
    pairs, top5_pairs = {}, {}
    for seed in SEEDS:
        pairs[str(seed)], top5_pairs[str(seed)] = {}, {}
        for view in VIEWS:
            sources = {
                "A_to_B": (predictions[("A", seed)][view], predictions[("B", seed)][view]),
                "B_to_C": (predictions[("B", seed)][view], predictions[("C", seed)][view]),
                "M0_pipeline_to_C": (baseline_rows["native" if view == "masked" else "sensor"], predictions[("C", seed)][view]),
            }
            pairs[str(seed)][view] = {key: paired(*rows) for key, rows in sources.items()}
            top5_pairs[str(seed)][view] = {key: paired(*rows, field="top5_correct") for key, rows in sources.items()}
    gate = progression(metrics, baseline)
    comparison_path = unit / "training_COMPARISON.json"
    comparison = read(comparison_path)
    ensure(comparison["status"] == "COMPLETE_PENDING_INDEPENDENT_RECOUNT" and comparison["context_sha256"] == context_sha,
           "Comparison not complete or context differs")
    same(comparison["arms"], arms, "comparison.arms")
    same(comparison["M0"], baseline, "comparison.M0")
    same(comparison["paired"], pairs, "comparison.paired")
    for key in gate:
        same(comparison["progression"][key], gate[key], "comparison.progression." + key)
    ensure(comparison["predefined_future_deployment_seed"] == SEEDS[0], "Deployment seed changed")
    ensure(comparison["final_blind_test"] is False and comparison["production_replaced"] is False,
           "Comparison incorrectly claims blind test/deployment")
    same(comparison["result_pins"], {str(p): pins[str(p)] for p in score_paths}, "comparison.result_pins")
    # Ensure no JSON source changed while the recount was being performed.
    ensure(all(digest(p) == value for p, value in pins.items()), "A saved JSON source changed during verification")
    return {
        "status": "PASS_INDEPENDENT_SAVED_SCORE_RECOUNT", "training_context_sha256": context_sha,
        "implementation": "Python standard library; logits independently ranked and CE recomputed; no training code import",
        "input_json_sha256": pins, "expected_runs": 9, "recounted_run_views": 18,
        "fixed_cases_per_view": 300, "independent_people": 3,
        "note_on_repeated_cases": "Two views and three seeds reuse the same 300 development clips; they are not 1800 independent samples.",
        "M0": baseline, "per_run": run_details, "three_seed_summaries": arms,
        "top1_matched_pairs": pairs, "top5_matched_pairs": top5_pairs, "progression": gate,
        "model_inference": 0, "training": 0, "detector": 0, "array_or_checkpoint_loaded": 0,
        "automatic_adoption": False, "generalization_proven": False,
        "limitations": [
            "Recounts saved logits; does not independently reproduce model forward execution.",
            "Checkpoint files are not loaded or rehashed; their recorded digests are reported only.",
            "DEV selected checkpoints and was used in previous experiments; this is not a final blind test.",
            "Missingness groups are frozen observation patterns, not linguistic one-/two-hand labels.",
            "Actual cropped RGB is an offline condition, not a new-person webcam test.",
        ],
    }


def self_test():
    """Small synthetic score sets only; does not access real results or arrays."""
    import copy
    checks = []

    def ok(name, condition):
        ensure(condition, "Synthetic test failed: " + name)
        checks.append(name)

    def must_fail(name, callback):
        try:
            callback()
        except (ValueError, KeyError, TypeError):
            checks.append(name)
            return
        raise ValueError("Synthetic negative test did not fail: " + name)

    def raw(cid, label, logits=None):
        identity = {"clip_id": cid, "label_index": label, "signer_id": 11}
        if logits is None:
            return dict(identity, status="INPUT_UNAVAILABLE", reason="synthetic_unavailable", logits=None,
                        prediction=None, top5=[], correct=False, top5_correct=False, cross_entropy=FAILURE_CE)
        ranking = sorted(range(100), key=lambda i: (-logits[i], i))
        maximum = max(logits)
        ce = math.log(sum(math.exp(x - maximum) for x in logits)) + maximum - logits[label]
        return dict(identity, status="OK", logits=logits, prediction=ranking[0], top5=ranking[:5],
                    correct=ranking[0] == label, top5_correct=label in ranking[:5], cross_entropy=ce)

    logits = [-10000.] * 100
    logits[6] = 10000.
    rows = [raw("extreme", 6, logits), raw("tie", 2, [0.] * 100), raw("fail", 3)]
    expected = {r["clip_id"]: r for r in rows}
    out = recount(rows, expected, {"fail": "synthetic_unavailable"}, fixed300=False)
    mm = counts(out)
    ok("extreme_logits_no_overflow", out[0]["correct"] and out[0]["cross_entropy"] == 0)
    ok("tie_lower_index_top5", out[2]["prediction"] == 0 and out[2]["top5"] == [0, 1, 2, 3, 4])
    ok("failed_input_preserved_as_wrong", mm["total"] == 3 and mm["top1_correct"] == 1 and mm["top5_correct"] == 2 and mm["input_failures"] == 1)
    ok("failure_ln100_ce", close_number(mm["mean_ce"], 2 * FAILURE_CE / 3))
    must_fail("fixed300_rejects_small_set", lambda: recount(rows, expected, {"fail": None}))
    must_fail("duplicate_prediction_rejected", lambda: recount(rows + [rows[0]], expected, {"fail": None}, fixed300=False))
    must_fail("missing_prediction_rejected", lambda: recount(rows[:-1], expected, {"fail": None}, fixed300=False))
    bad = copy.deepcopy(rows); bad[2]["correct"] = True
    must_fail("failure_cannot_be_marked_correct", lambda: recount(bad, expected, {"fail": None}, fixed300=False))
    bad = copy.deepcopy(rows); bad[0]["logits"][0] = float("nan")
    must_fail("nonfinite_logit_rejected", lambda: recount(bad, expected, {"fail": None}, fixed300=False))
    bad = copy.deepcopy(rows); bad[1]["prediction"] = 2
    must_fail("stored_prediction_disagreement_rejected", lambda: recount(bad, expected, {"fail": None}, fixed300=False))
    bad = copy.deepcopy(rows); bad[0]["label_index"] = 7
    must_fail("changed_label_rejected", lambda: recount(bad, expected, {"fail": None}, fixed300=False))
    must_fail("unfrozen_failure_rejected", lambda: recount(rows, expected, {}, fixed300=False))
    alternate = copy.deepcopy(out)
    alternate[0]["correct"] = False
    alternate[1]["correct"] = True
    match = paired(out, alternate)
    ok("matched_gain_loss_not_just_net", match["improved"] == ["fail"] and match["regressed"] == ["extreme"] and match["net_correct"] == 0)

    def metric(correct, per_signer, failed=0):
        return {"top1_correct": correct, "input_failures": failed,
                "by_signer": {str(s): {"top1_correct": n} for s, n in zip(SIGNERS, per_signer)}}

    # Only tiny aggregate fixtures: no synthetic300 pseudo-dataset is generated.
    baseline = {"native": metric(290, [98, 92, 100]), "sensor": metric(270, [90, 90, 90])}
    metrics = {}
    for arm in ARMS:
        for seed in SEEDS:
            metrics[(arm, seed)] = {"masked": metric(287, [95, 92, 100]), "sensor": metric(272 if arm == "C" else 270, [90, 90, 92] if arm == "C" else [90, 90, 90])}
    gate = progression(metrics, baseline)
    ok("exact_three_case_margin_accepted", gate["pass_to_next_review"])
    bad = copy.deepcopy(metrics); bad[("C", SEEDS[0])]["masked"] = metric(286, [94, 92, 100])
    ok("one_case_over_mean_margin_rejected", not progression(bad, baseline)["checks"]["original_pipeline_mean_regression_at_most_three_cases"])
    bad = copy.deepcopy(metrics)
    for seed in SEEDS:
        bad[("C", seed)]["sensor"] = metric(270, [90, 90, 90])
    ok("strict_improvement_rejects_equal", not progression(bad, baseline)["checks"]["C_sensor_mean_exceeds_B"])
    bad = copy.deepcopy(metrics)
    for index, seed in enumerate(SEEDS):
        bad[("C", seed)]["sensor"] = metric(280 if index == 0 else 270, [90, 90, 100] if index == 0 else [90, 90, 90])
    ok("one_lucky_seed_is_insufficient", not progression(bad, baseline)["checks"]["C_sensor_better_than_B_at_least_two_seeds"])
    bad = copy.deepcopy(metrics); bad[("C", SEEDS[0])]["sensor"]["input_failures"] = 1
    ok("increased_failure_blocks_gate", not progression(bad, baseline)["checks"]["C_sensor_failure_no_increase"])
    bad = copy.deepcopy(metrics)
    for seed in SEEDS:
        bad[("C", seed)]["sensor"] = metric(272, [86, 93, 93])
    ok("signer_regression_blocks_despite_overall_gain", not progression(bad, baseline)["checks"]["sensor_signer11_mean_regression_at_most_three_cases"])
    groups = {r["clip_id"]: "other" for r in out}
    groups["fail"] = "one_hand_never_usable"
    grouped = detailed(out, groups)
    ok("subgroup_failure_denominator_preserved", grouped["by_observation_pattern"]["one_hand_never_usable"]["total"] == 1 and grouped["by_observation_pattern"]["one_hand_never_usable"]["top1_correct"] == 0)
    return {"status": "PASS_SYNTHETIC_ONLY", "checks_passed": len(checks), "checks": checks,
            "real_prediction_files_read": 0, "model_inference": 0, "training": 0, "detector": 0,
            "array_or_checkpoint_loaded": 0,
            "actual_nine_run_recount": "NOT_RUN_NO_SCORES_YET"}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="Only small in-memory synthetic cases; no real score access")
    mode.add_argument("--verify", action="store_true", help="After all nine runs finish: independently recount saved JSON logits")
    parser.add_argument("--unit-dir", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, help="New JSON report inside unit; defaults to training_scores_INDEPENDENT.json for --verify")
    args = parser.parse_args()
    unit = args.unit_dir.resolve()
    output = args.output.resolve() if args.output else (unit / "training_scores_INDEPENDENT.json" if args.verify else None)
    if output:
        ensure(output.parent == unit, "Output must remain in the experiment unit")
    if output and output.exists():
        ensure(args.verify, "Existing synthetic report preserved; select a new output or omit it")
        old = json.loads(output.read_text(encoding="utf-8-sig"))
        ensure(old["status"] == "PASS_INDEPENDENT_SAVED_SCORE_RECOUNT", "Existing report is not PASS")
        ensure(old["verifier_sha256"] == digest(__file__), "Verifier changed; preserve report and review")
        ensure(all(digest(p) == value for p, value in old["input_json_sha256"].items()), "Prior evidence changed")
        print(json.dumps({"status": "EXISTING_INDEPENDENT_RECOUNT_REUSED", "report": str(output)}, ensure_ascii=False))
        return
    started = time.perf_counter()
    result = self_test() if args.self_test else verify(unit)
    result.update(at_utc=datetime.now(timezone.utc).isoformat(), verifier=str(Path(__file__).resolve()),
                  verifier_sha256=digest(__file__), elapsed_seconds=time.perf_counter() - started,
                  reviewer="Codex roi_training_protocol independent standard-library recount")
    if output:
        write_new(output, result)
    summary = {k: result[k] for k in ("status", "checks_passed", "expected_runs", "recounted_run_views", "elapsed_seconds") if k in result}
    if output:
        summary["report"] = str(output)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
