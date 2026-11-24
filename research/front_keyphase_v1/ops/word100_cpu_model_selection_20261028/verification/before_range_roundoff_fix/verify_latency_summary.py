"""Audit saved timing, fused parity and report arithmetic; never run models.

Requires completed native/fused results, SAVED9_INDEPENDENT and SUMMARY. Only
stdlib and NumPy are imported. This is not a new speed or accuracy experiment.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math

import numpy as np

U = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, tol=1e-12):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol


def quantile(values, fraction):
    # Independent order-statistic interpolation, not a producer import.
    a = sorted(float(x) for x in values)
    pos = (len(a) - 1) * fraction
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)


def main():
    required = ["TRAINING_COMPLETE.json", "FROZEN_PROTOCOL.json", "TRAINING_MANIFEST.json",
                "CPU_LATENCY_PROTOCOL.json", "LATENCY_RESULTS.json", "REPVIT_FUSED_LATENCY_PROTOCOL.json",
                "REPVIT_FUSED_RESULTS.json", "REPVIT_FUSED_PARITY_LOGITS.npz", "SUMMARY.json", "SELECTION.json",
                "verification/SAVED9_INDEPENDENT.json", "RESULT_KO.html", "RESULT_KO.txt"]
    absent = [p for p in required if not (U / p).is_file()]
    if absent:
        raise SystemExit("PENDING; no model operations: " + ", ".join(absent))
    destination = U / "verification" / "FINAL_INDEPENDENT.json"
    assert not destination.exists(), "Preserve completed independent verification"
    checks = []

    def ck(name, ok, detail=None):
        assert ok, name
        checks.append(dict(name=name, passed=True, detail=detail))

    def time_check(label, value, repeats):
        raw = value["raw_seconds"]
        ck(label + "_samples", len(raw) == repeats == value["repeats"]
           and all(math.isfinite(float(t)) and t >= 0 for t in raw))
        ck(label + "_p50_p95", close(value["p50_seconds"], quantile(raw, .5))
           and close(value["p95_seconds"], quantile(raw, .95))
           and close(value["p50_ms"], 1000 * quantile(raw, .5), 1e-9)
           and close(value["p95_ms"], 1000 * quantile(raw, .95), 1e-9)
           and close(value["min_seconds"], min(raw)) and close(value["max_seconds"], max(raw)))

    def whole_timing(label, row, protocol):
        time_check(label + "_single", row["single_full_model"], protocol["single_repeats"])
        ck(label + "_two_workloads", len(row["clips"]) == 2 and
           [r["hand_crops"] for r in row["clips"]] == [120, 300])
        for r in row["clips"]:
            n = r["hand_crops"]
            expected = [min(16, n - i) for i in range(0, n, 16)]
            ck(label + f"_{n}_exact_batch_counts", r["batches"] == expected)
            time_check(label + f"_{n}_model", r["full_model_only"], protocol["clip_repeats"])
            time_check(label + f"_{n}_inclusive", r["rgb_decode_preprocess_full_model"], protocol["clip_repeats"])

    training = read(U / "FROZEN_PROTOCOL.json")
    manifest = read(U / "TRAINING_MANIFEST.json")
    native_p = read(U / "CPU_LATENCY_PROTOCOL.json")
    native = read(U / "LATENCY_RESULTS.json")
    fused_p = read(U / "REPVIT_FUSED_LATENCY_PROTOCOL.json")
    fused = read(U / "REPVIT_FUSED_RESULTS.json")
    score_audit = read(U / "verification/SAVED9_INDEPENDENT.json")
    summary = read(U / "SUMMARY.json")
    selection = read(U / "SELECTION.json")
    ck("saved9_independent_pass", score_audit["status"] == "PASS" and score_audit["fixed_run_count"] == 9
       and score_audit["protocol_sha256"] == sha(U / "FROZEN_PROTOCOL.json"))
    ck("training_code_and_manifest_pinned", training["code_sha256"] == sha(U / "compare_cpu.py")
       and training["manifest_sha256"] == sha(U / "TRAINING_MANIFEST.json"))
    for filename, key in [("measure_cpu.py", "measure_cpu_sha256"), ("compare_cpu.py", "compare_cpu_sha256"),
                          ("TRAINING_MANIFEST.json", "training_manifest_sha256"), ("PROTECTED.json", "protected_manifest_sha256")]:
        ck("native_pin_" + filename, native_p[key] == sha(U / filename))
    ck("native_completion", native["status"] == "COMPLETE"
       and native["protocol_sha256"] == sha(U / "CPU_LATENCY_PROTOCOL.json")
       and native["training_protocol_sha256"] == sha(U / "FROZEN_PROTOCOL.json")
       and native["training_complete_sha256"] == sha(U / "TRAINING_COMPLETE.json"))
    ck("native_cpu_configuration", native["environment"]["threads"] == 4
       and native["environment"]["interop_threads"] == 1 and native["environment"]["dtype"] == "float32"
       and native["environment"]["cuda_used"] is False)
    ck("native_component_not_web", native["component_only"] is True and native["prefix_cache_used"] is False
       and native["accuracy_scoring"] is False and native["final_test_images_accessed"] == 0
       and native["validation_images_accessed"] == 0 and native["user_mentor_access"] is False)
    fit = [r for r in manifest["rows"] if r["role"] == "fit"]
    ck("fit2489_and_no_spent_people", len(fit) == 2489
       and {r["signer"] for r in fit} == {"signer01", "signer02", "signer04", "signer05"}
       and all(r["signer"] not in {"signer08", "signer09"} for r in manifest["rows"]))
    salt = native_p["selection"]["salt"]
    order = sorted(fit, key=lambda r: (hashlib.sha256((salt + r["id"]).encode()).hexdigest(), r["id"]))[:300]
    expected = [{k: r[k] for k in ["id", "image", "image_sha256", "signer", "role"]} for r in order]
    ck("latency_exact_300_fit_inputs", native_p["selected"] == expected
       and len({r["id"] for r in expected}) == 300)
    ck("latency_fit_image_bytes", all(sha(r["image"]) == r["image_sha256"] for r in expected))
    names = ["v3", "repvit", "v4hybrid"]
    ck("three_native_unique_models", len(native["models"]) == 3
       and [r["name"] for r in native["models"]] == names)
    latencies = {r["name"]: r for r in native["models"]}
    for name in names:
        row = latencies[name]
        ck(name + "_checkpoint_and_native_path", row["seed"] == 28101 and row["epoch"] == 20
           and row["checkpoint_sha256"] == sha(U / "models" / f"{name}_28101.pt")
           and row["structural_fusion"] is False and row["input_size"] == training["specs"][name]["input_size"]
           and row["parameters"] == training["specs"][name]["total_parameters"])
        whole_timing(name, row, native_p)
    for filename, key in [("measure_repvit_fused.py", "fused_code_sha256"), ("measure_cpu.py", "native_code_sha256"),
                          ("CPU_LATENCY_PROTOCOL.json", "native_protocol_sha256")]:
        ck("fused_pin_" + filename, fused_p[key] == sha(U / filename))
    ck("fused_dependency_pin", fused_p["repvit_module_sha256"] == sha(fused_p["repvit_module"]))
    ck("fused_result_pins", fused["protocol_sha256"] == sha(U / "REPVIT_FUSED_LATENCY_PROTOCOL.json")
       and fused["native_result_sha256"] == sha(U / "LATENCY_RESULTS.json")
       and fused["checkpoint_sha256"] == sha(U / "models/repvit_28101.pt"))
    ck("fused_scope_separate", fused["native_selection_rule_unchanged"] is True
       and fused["training"] is False and fused["accuracy_scoring"] is False
       and fused["final_test_access"] is False and fused["user_mentor_access"] is False)
    parity = fused["parity"]
    archive = U / "REPVIT_FUSED_PARITY_LOGITS.npz"
    ck("parity_archive_pin", Path(parity["archive"]).resolve() == archive.resolve()
       and parity["archive_sha256"] == sha(archive))
    with np.load(archive, allow_pickle=False) as z:
        a = z["native_logits"].copy()
        b = z["fused_logits"].copy()
        ck("parity_shapes", a.shape == b.shape == (300, 45)
           and a.dtype == b.dtype == np.float32 and np.isfinite(a).all() and np.isfinite(b).all())
        ck("parity_row_and_model_identity", z["ids"].tolist() == [r["id"] for r in expected]
           and z["checkpoint_sha256"].item() == fused["checkpoint_sha256"]
           and z["protocol_sha256"].item() == sha(U / "REPVIT_FUSED_LATENCY_PROTOCOL.json"))
    atol = fused_p["parity"]["atol"]
    rtol = fused_p["parity"]["rtol"]
    ck("parity_fixed_tolerances", atol == rtol == 1e-4 and parity["atol"] == atol and parity["rtol"] == rtol)
    absolute = np.abs(b - a)
    tolerance = np.float32(atol) + np.float32(rtol) * np.abs(a)
    allclose = bool(np.all(absolute <= tolerance))
    max_abs = float(np.max(absolute))
    max_ratio = float(np.max(absolute / tolerance))
    equal = int(np.sum(np.argmax(a, axis=1) == np.argmax(b, axis=1)))
    ck("parity_saved_arithmetic", allclose == parity["allclose"] and equal == parity["argmax_equal"]
       and close(max_abs, parity["max_abs_logit_difference"], 1e-8)
       and close(max_ratio, parity["max_difference_over_allclose_tolerance"], 1e-6)
       and parity["fit_crops"] == 300 and parity["scalar_logits"] == 13500)
    parity_pass = allclose and equal == 300
    ck("parity_gate", parity["passed"] == parity_pass)
    if parity_pass:
        ck("fused_pass_then_timing", fused["status"] == "COMPLETE" and fused["timing_executed"] is True
           and fused["structural_fusion"] is True)
        whole_timing("repvit_fused", fused, native_p)
    else:
        ck("fused_failure_no_timing", fused["status"] == "PARITY_FAILED_NO_TIMING"
           and fused["timing_executed"] is False and "clips" not in fused and "single_full_model" not in fused)
    # Reuse the separately computed nine-run arithmetic only after binding its
    # result/checkpoint/prediction hashes to current saved files.
    for stem, pins in score_audit["artifacts"].items():
        ck(stem + "_unchanged_since_score_audit", sha(U / "models" / f"{stem}.pt") == pins["checkpoint"]
           and sha(U / "models" / f"{stem}.json") == pins["result"]
           and sha(U / "models" / f"{stem}_dev.npz") == pins["predictions"])
    sruns = score_audit["runs"]
    ck("summary_scope", summary["status"] == "DEVELOPMENT_COMPARISON_COMPLETE_NOT_DEPLOYED"
       and summary["runs"] == 9 and summary["fit_images"] == 2489 and summary["development_images"] == 488
       and summary["classes"] == 45 and summary["previous_test382_accessed"] is False
       and summary["user_mentor_data_used"] is False and summary["operating_changed"] is False
       and summary["end_to_end_web_latency_measured"] is False)
    ck("summary_models", len(summary["models"]) == 3 and [r["name"] for r in summary["models"]] == names)
    dev_classes = {r["target"] for r in manifest["rows"] if r["role"] == "validation"}
    fit_classes = {r["target"] for r in fit}
    ck("development_class_coverage", len(dev_classes) == summary["development_supported_classes"] == 30
       and summary["classes_missing_in_development"] == [manifest["classes"][i] for i in sorted(fit_classes - dev_classes)])
    smodels = {r["name"]: r for r in summary["models"]}
    for name in names:
        row = smodels[name]
        rr = [r for r in sruns if r["model"] == name]
        acc = [r["accuracy"] for r in rr]
        bal = [r["balanced_macro_f1"] for r in rr]
        ck(name + "_summary_mean_range", close(row["accuracy_mean"], np.mean(acc))
           and row["accuracy_range"] == [min(acc), max(acc)]
           and close(row["balanced_macro_f1_mean"], np.mean(bal))
           and row["balanced_macro_f1_range"] == [min(bal), max(bal)])
        for person in ["signer03", "signer06", "signer07"]:
            q = row["per_signer"][person]
            ck(name + person + "_mean", q["n"] == rr[0]["per_person"][person]["n"]
               and close(q["accuracy_mean"], np.mean([r["per_person"][person]["accuracy"] for r in rr]))
               and close(q["macro_f1_mean"], np.mean([r["per_person"][person]["macro_f1_supported"] for r in rr])))
        ck(name + "_summary_native_latency", row["latency"] == latencies[name])
        paired = []
        for seed in training["seeds"]:
            with np.load(U / "models" / f"{name}_{seed}_dev.npz", allow_pickle=False) as z, \
                 np.load(U / "models" / f"v3_{seed}_dev.npz", allow_pickle=False) as baseline:
                ck(name + str(seed) + "_paired_same_rows", np.array_equal(z["ids"], baseline["ids"])
                   and np.array_equal(z["targets"], baseline["targets"]))
                target = z["targets"]
                right = z["logits"].argmax(axis=1) == target
                baseline_right = baseline["logits"].argmax(axis=1) == target
                paired.append(dict(seed=seed, improved=int(np.sum(right & ~baseline_right)),
                                   regressed=int(np.sum(~right & baseline_right)),
                                   both_wrong=int(np.sum(~right & ~baseline_right))))
        ck(name + "_paired_changes", paired == row["paired_vs_v3"])
    maximum = max(r["balanced_macro_f1_mean"] for r in summary["models"])
    eligible = [r for r in summary["models"] if maximum - r["balanced_macro_f1_mean"] <= .01 + 1e-12]
    chosen = min(eligible, key=lambda r: latencies[r["name"]]["single_full_model"]["p95_ms"])
    ck("frozen_native_selection", selection["selected_model"] == chosen["name"]
       and selection["eligible_within_one_point"] == [r["name"] for r in eligible]
       and selection["rule"] == training["selection"] + " " + training["latency_tiebreak"]
       and selection["production_changed"] is False and summary["selection"] == selection)
    ck("fused_auxiliary_not_substituted", summary["repvit_fused_auxiliary"] == fused)
    change_pp = 100 * (chosen["accuracy_mean"] - smodels["v3"]["accuracy_mean"])
    text = (U / "RESULT_KO.txt").read_text(encoding="utf-8-sig")
    page = (U / "RESULT_KO.html").read_text(encoding="utf-8-sig")
    ck("reported_percentage_point_change", f"{change_pp:+.2f}%p" in text and f"{change_pp:+.2f}%p" in page)
    ck("report_scope_labels", "30종" in text and "30종" in page and "15종" in text and "15종" in page
       and "전체 시간이 아닙니다" in page and "별도의 배포 최적화" in page)
    protected = read(U / "PROTECTED.json")
    ck("protected32_unchanged", len(protected) == 32 and all(sha(f) == h for f, h in protected.items()))
    out = dict(status="PASS", at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               check_count=len(checks), checks=checks,
               scope="Independent saved arithmetic/pins only; not rerun timing, model inference, training, or a new final accuracy test.",
               inference_performed=False, training_performed=False, timing_rerun=False,
               final382_accessed=False, user_mentor_media_accessed=False,
               selected_development_model=chosen["name"], chosen_accuracy_delta_pp=change_pp,
               fused_parity=dict(passed=parity_pass, fit_images=300, logits=13500,
                                 argmax_identical=equal, max_abs_logit_difference=max_abs,
                                 max_difference_over_allclose_tolerance=max_ratio),
               native_120_inclusive_p95_seconds={n: latencies[n]["clips"][0]["rgb_decode_preprocess_full_model"]["p95_seconds"] for n in names},
               protected_files_verified=len(protected),
               files_sha256={f: sha(U / f) for f in required}, script_sha256=sha(__file__),
               limits=["Native timing remains the frozen architecture-selection tiebreak. Fused timing is a separate deployment diagnostic.",
                       "Five request repetitions give a sample p95, not a reliable population tail or complete webcam latency.",
                       "Model order and warm OS cache are fixed; background system load is not fully controlled.",
                       "Agreement of300 fit logits after fusion is finite-sample numerical equivalence, not independent correctness or universal mathematical proof.",
                       "Development selection uses488 reused public images covering30of45 labels; three seeds are not extra people.",
                       "No Korean Sign Language educational correctness, full-occlusion recovery or TCN improvement was evaluated."])
    destination.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="PASS", checks=len(checks), selected=chosen["name"], delta_pp=change_pp), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", required=True)
    parser.parse_args()
    main()
