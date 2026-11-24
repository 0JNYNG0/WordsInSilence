"""Check saved epoch history and checkpoint links without training or inference.

Use --self-test now (small synthetic histories/checkpoint dictionaries only).  After nine runs AND the
score recount finish, use --verify.  This imports no model/training/detector code;
--verify uses torch.load(weights_only=True) solely to inspect local checkpoints.
Frozen driver/core/model/context files and prior evidence are never modified.
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
BUFFER_SUFFIXES = (".running_mean", ".running_var", ".num_batches_tracked")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def near(a, b):
    return (type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-11))


def exposure_counts(ids, epoch, arm):
    # Independent implementation of the frozen two temporal x two spatial cycle.
    result = Counter()
    for cid in ids:
        temporal_bit = int.from_bytes(hashlib.sha256(cid.encode()).digest(), "big") & 1
        temporal = "x_context" if (epoch + temporal_bit) & 1 else "x_whole"
        if arm == "A":
            spatial = "native"
        elif arm == "B":
            spatial = "masked"
        else:
            spatial_bit = int.from_bytes(hashlib.sha256(("sensor_view:" + cid).encode()).digest(), "big") & 1
            spatial = "sensor" if (((epoch - 1) // 2) + spatial_bit) & 1 else "masked"
        result[spatial + "/" + temporal] += 1
    return dict(result)


def inspect_history(history, ids, arm, result, *, required_train_count=899):
    require(arm in ARMS and len(ids) == len(set(ids)) == required_train_count, "Invalid arm/TRAIN IDs")
    require(isinstance(history, list) and 1 <= len(history) <= 80, "Empty/too long epoch history")
    best_epoch, best_ce, prior_elapsed = None, None, -1.0
    batches = (required_train_count + 31) // 32
    for epoch, row in enumerate(history, 1):
        require(row["epoch"] == epoch, "Non-contiguous or repeated history epoch")
        if best_epoch is not None:
            require(epoch - 1 - best_epoch < 12, "History continued past frozen early stop")
        require(row["steps"] == epoch * batches, "History step count differs")
        require(row["exposure"] == exposure_counts(ids, epoch, arm), "Spatial/temporal exposure differs")
        require(sum(row["exposure"].values()) == required_train_count, "Wrong source exposure total")
        for key in ("train_ce", "masked_ce", "sensor_ce", "selected_rule_ce"):
            require(type(row[key]) in (int, float) and math.isfinite(row[key]) and row[key] >= -1e-9,
                    "Invalid finite CE: " + key)
        ce = 0.5 * row["masked_ce"] + 0.5 * row["sensor_ce"]
        require(near(row["selected_rule_ce"], ce), "Checkpoint selection is not 50:50 DEV CE")
        if best_ce is None or ce < best_ce - 1e-6:
            best_ce, best_epoch = ce, epoch
        require(row["best_epoch"] == best_epoch, "Recorded best epoch violates strict 1e-6 rule/tie policy")
        for key in ("masked_top1", "sensor_top1"):
            require(type(row[key]) is int and 0 <= row[key] <= 300, "Invalid DEV count")
        elapsed = row["elapsed_seconds"]
        require(type(elapsed) in (int, float) and math.isfinite(elapsed)
                and prior_elapsed <= elapsed <= 3600, "Elapsed time invalid/nonmonotone/budget exceeded")
        prior_elapsed = elapsed
    total = len(history)
    require(total == 80 or total - best_epoch >= 12, "History ended before max80/patience12")
    require(result["epochs_completed"] == total and result["selected_epoch"] == best_epoch,
            "Final selected/completed epoch does not match full history")
    require(result["total_steps"] == total * batches and near(result["selected_ce"], best_ce),
            "Final steps/selected CE do not match history")
    require(near(result["seconds"], prior_elapsed), "Final elapsed does not match last history")
    return {"epochs": total, "selected_epoch": best_epoch, "selected_ce": best_ce,
            "train_source_ids": required_train_count, "expected_source_exposures": total * required_train_count,
            "batches_per_epoch": batches, "steps": total * batches,
            "all_recorded_exposure_counts_match": True, "strict_best_and_early_stop_match": True,
            "elapsed_seconds_as_recorded": prior_elapsed,
            "elapsed_limit": "Committed-epoch cumulative measurement; killed/uncommitted work and all host downtime cannot be reconstructed."}


def state_digest(state):
    value = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value.update(key.encode())
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def inspect_checkpoint_dicts(torch, last, best, result, history, context, context_sha, arm, seed):
    """Inspect saved tensors/metadata only; no module construction or forward."""
    require(last["context_sha256"] == context_sha and last["arm"] == arm and last["seed"] == seed,
            "Last-checkpoint provenance differs")
    require(best["training_context_sha256"] == context_sha and best["arm"] == arm and best["seed"] == seed,
            "Selected-checkpoint provenance differs")
    require(last["epoch"] == result["epochs_completed"] and last["step"] == result["total_steps"],
            "Last checkpoint not linked to completed history")
    require(last["history"] == history, "JSON history differs from committed checkpoint history")
    require(last["initial_state_sha256"] == result["initial_state_sha256"], "Initial-state digest differs")
    require(near(last["elapsed_seconds"], result["seconds"]), "Committed elapsed differs")
    chosen = last["best"]
    require(chosen["epoch"] == best["epoch"] == result["selected_epoch"], "Selected epoch linkage differs")
    require(chosen["step"] == best["step"] == result["selected_epoch"] * 29, "Selected step linkage differs")
    require(near(chosen["ce"], best["development_ce"]) and near(chosen["ce"], result["selected_ce"]),
            "Selected checkpoint CE differs")
    require(chosen["predictions"] == result["predictions"] and chosen["metrics"] == result["metrics"],
            "Final score JSON differs from committed selected-epoch scores")
    selected_row = history[result["selected_epoch"] - 1]
    for view in ("masked", "sensor"):
        require(near(selected_row[view + "_ce"], result["metrics"][view]["mean_ce"]),
                "Selected history CE is disconnected from saved scores")
        require(selected_row[view + "_top1"] == result["metrics"][view]["top1_correct"],
                "Selected history count is disconnected from saved scores")
    require(best["classes"] == context["labels"] and best["word_codes"] == context["word_codes"],
            "Selected model output label order changed")
    require(best["in_channels"] == 280 and best["model_type"] == "wis_masked_tcn" and best["model_version"] == "masked_v1",
            "Selected model type differs")
    require(best["config"] == {k: v for k, v in context["model"].items() if k not in ("in_channels", "n_classes")},
            "Selected model config differs")
    require(best["initialization"] == "fresh random; AIHub899 only", "Initialization metadata differs")
    require(set(best["state_dict"]) == set(chosen["model"]) == set(last["model"]), "Checkpoint tensor keys differ")
    parameter_names = [name for name in last["model"] if not name.endswith(BUFFER_SUFFIXES)]
    require(sum(last["model"][name].numel() for name in parameter_names) == 37636, "Checkpoint parameter count differs")
    for name in best["state_dict"]:
        a, b, final = best["state_dict"][name], chosen["model"][name], last["model"][name]
        require(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and isinstance(final, torch.Tensor),
                "State contains a non-tensor")
        require(a.dtype == b.dtype == final.dtype and a.shape == b.shape == final.shape, "Tensor layout differs: " + name)
        require(torch.equal(a, b), "Selected weight/buffer differs from committed best: " + name)
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(final).all()), "Nonfinite model state")
        if name.endswith(".num_batches_tracked"):
            require(a.numel() == final.numel() == 1 and int(a.item()) == best["step"]
                    and int(final.item()) == last["step"], "BatchNorm update count differs: " + name)
    optimizer = last["optimizer"]
    require(len(optimizer["param_groups"]) == 1, "Expected one AdamW parameter group")
    group = optimizer["param_groups"][0]
    require(near(group["lr"], 0.001) and near(group["weight_decay"], 0.001), "Saved optimizer hyperparameters differ")
    parameter_ids = group["params"]
    require(len(parameter_ids) == len(set(parameter_ids)) == len(parameter_names)
            and set(parameter_ids) == set(optimizer["state"]), "Optimizer state parameter coverage differs")
    for parameter_id, name in zip(parameter_ids, parameter_names):
        moment = optimizer["state"][parameter_id]
        step = moment["step"].item() if isinstance(moment["step"], torch.Tensor) else moment["step"]
        require(step == last["step"], "Optimizer parameter step differs: " + name)
        for field in ("exp_avg", "exp_avg_sq"):
            tensor = moment[field]
            require(isinstance(tensor, torch.Tensor) and tensor.shape == last["model"][name].shape
                    and bool(torch.isfinite(tensor).all()), "Invalid optimizer moment: " + name)
    rng = last["torch_rng"]
    require(isinstance(rng, torch.Tensor) and rng.dtype == torch.uint8 and rng.ndim == 1 and rng.numel() > 0,
            "Resumable CPU RNG state absent")
    return {"selected_tensor_state_sha256": state_digest(best["state_dict"]),
            "last_tensor_state_sha256": state_digest(last["model"]),
            "selected_tensors_equal_committed_best": True, "saved_scores_equal_committed_best": True,
            "optimizer_step_per_parameter": last["step"], "optimizer_parameter_tensors": len(parameter_ids),
            "batchnorm_counters_match_steps": True, "resume_rng_state_present": True,
            "initial_state_sha256_as_recorded": result["initial_state_sha256"]}


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
    require(context["schema"] == "body_boundary_ABC_training_v1" and context["seeds"] == list(SEEDS)
            and context["arms"] == list(ARMS), "Frozen experiment identity differs")
    require(context["train_count"] == 899 and context["development_count"] == 300, "Data count differs")
    expected_training = {"batch_size": 32, "epochs_max": 80, "early_stopping_patience": 12,
                         "lr": 0.001, "weight_decay": 0.001, "gradient_clip_norm": 1.0,
                         "cpu_threads": 4, "wall_safety_seconds_per_arm": 3600, "strict_ce_improvement": 1e-6}
    require(context["training"] == expected_training, "Frozen training conditions differ")
    ids = [row["clip_id"] for row in context["ordered_rows"] if row["role"] == "train_candidate"]
    require(len(ids) == len(set(ids)) == 899, "TRAIN source IDs differ")
    score_audit = read(unit / "training_scores_INDEPENDENT.json")
    require(score_audit["status"] == "PASS_INDEPENDENT_SAVED_SCORE_RECOUNT"
            and score_audit["training_context_sha256"] == context_sha, "Run score recount first")
    score_verifier_path = str(unit / "verify_training_scores.py")
    pins[score_verifier_path] = sha(score_verifier_path)
    require(score_audit["verifier_sha256"] == pins[score_verifier_path], "Score verifier changed")
    for path, digest in score_audit["input_json_sha256"].items():
        require(sha(path) == digest, "Recounted JSON changed: " + path)
        pins[path] = digest
    # Only fixed code is rehashed here; this deliberately does not reread detector arrays.
    for name in ("training_driver.py", "training_core.py"):
        path = str(unit / name)
        pins[path] = sha(path)
        require(context["pins"][path] == pins[path], "Frozen training code changed: " + name)
    model_sources = [path for path in context["pins"] if path.replace("\\", "/").endswith("/pilot/model_masked_v1.py")]
    require(len(model_sources) == 1, "Expected one pinned model source")
    for path in model_sources:
        pins[path] = sha(path)
        require(pins[path] == context["pins"][path], "Frozen model source changed")
    static = read(unit / "EXECUTION_CONTRACT_INDEPENDENT.json")
    require(static["status"] == "PASS_PURE_AND_MOCKED_CONTRACTS", "Earlier execution-contract evidence is not PASS")
    for name in ("training_driver.py", "training_core.py"):
        require(static["unchanged_pins"][str(unit / name)] == pins[str(unit / name)], "Earlier static/mock evidence code differs")
    import torch  # Restricted checkpoint deserialization only; no model import/constructor.
    details, initial = {}, {seed: [] for seed in SEEDS}
    for arm in ARMS:
        for seed in SEEDS:
            prefix = unit / f"training_{arm}_{seed}"
            result = read(str(prefix) + "_RESULT.json")
            history = read(str(prefix) + "_history.json")
            require(result["status"] == "COMPLETE_DEVELOPMENT_ONLY" and result["context_sha256"] == context_sha
                    and result["arm"] == arm and result["seed"] == seed, "Incomplete or wrong run")
            history_report = inspect_history(history, ids, arm, result)
            last_path, best_path = Path(str(prefix) + "_last.pt"), Path(str(prefix) + "_best.pt")
            pins[str(last_path)], pins[str(best_path)] = sha(last_path), sha(best_path)
            require(pins[str(best_path)] == result["checkpoint_sha256"], "Selected checkpoint bytes differ from result pin")
            last = torch.load(last_path, map_location="cpu", weights_only=True)
            best = torch.load(best_path, map_location="cpu", weights_only=True)
            tensor_report = inspect_checkpoint_dicts(torch, last, best, result, history, context, context_sha, arm, seed)
            initial[seed].append(result["initial_state_sha256"])
            details[f"{arm}_{seed}"] = {"history": history_report, "checkpoint_link": tensor_report}
            del last, best
    require(all(len(set(values)) == 1 for values in initial.values()), "Same-seed A/B/C initial digests differ")
    require(len({values[0] for values in initial.values()}) == 3, "Different seed initial digests are unexpectedly equal")
    require(all(sha(path) == digest for path, digest in pins.items()), "Evidence changed during trace audit")
    return {"status": "PASS_INDEPENDENT_SAVED_TRAINING_TRACE", "training_context_sha256": context_sha,
            "input_file_sha256": pins, "runs_verified": 9, "per_run": details,
            "same_seed_initial_state_match": True, "distinct_seed_initial_state_digests": True,
            "checkpoint_load_policy": "torch.load(weights_only=True, map_location=cpu); no model construction",
            "model_constructions": 0, "model_forward": 0, "optimizer_steps_executed": 0,
            "new_training": 0, "new_detector": 0, "new_dev_inference": 0,
            "scope_limits": [
                "Actual per-batch clip ID lists were not logged. Recorded exposure totals are checked against all899 IDs and the frozen schedule; prior reviewed driver proves its permutation/uniqueness assertion. This is not an independent per-example execution transcript.",
                "Initial weights are not retained separately after epoch0. Equal per-seed A/B/C recorded hashes are checked; exact seed-to-initial-tensor replay is not performed.",
                "Only selected epoch predictions are retained. Weighted CE and best selection are checked over every history row; nonselected epoch logits cannot be independently recounted.",
                "Checkpoint tensors match stored best state and saved scores match its stored predictions. No new forward pass proves logits numerically came from those tensors; frozen execution-contract tests provide complementary evidence.",
                "RNG state and optimizer moments/counters are present and consistent; past randomness, interruptions and per-batch execution are not replayed.",
                "No data arrays are reopened here. Source/ROI/mask semantics rely on the separate input audit.",
                "This trace PASS is not proof of new-person generalization and never authorizes production replacement.",
            ]}


def self_test():
    import copy
    checks = []
    ids = ["synthetic_1", "synthetic_2", "synthetic_3"]

    def build(ces):
        history, best_ce, best_epoch = [], None, None
        for epoch, ce in enumerate(ces, 1):
            if best_ce is None or ce < best_ce - 1e-6:
                best_ce, best_epoch = ce, epoch
            history.append({"epoch": epoch, "steps": epoch, "train_ce": 0.5,
                            "masked_ce": ce + 0.2, "sensor_ce": ce - 0.2,
                            "selected_rule_ce": ce, "masked_top1": 290, "sensor_top1": 270,
                            "best_epoch": best_epoch, "elapsed_seconds": float(epoch),
                            "exposure": exposure_counts(ids, epoch, "C")})
        result = {"epochs_completed": len(ces), "selected_epoch": best_epoch, "selected_ce": best_ce,
                  "total_steps": len(ces), "seconds": float(len(ces))}
        return history, result

    def run(h, r):
        return inspect_history(h, ids, "C", r, required_train_count=3)

    def rejects(name, h, r):
        try:
            run(h, r)
        except ValueError:
            checks.append(name)
            return
        raise ValueError("Synthetic negative test did not reject: " + name)

    h, r = build([1.0, 0.9999995, 0.999998] + [0.999998] * 12)
    require(run(h, r)["selected_epoch"] == 3, "Strict minimum synthetic test failed")
    checks.append("strict_threshold_and_ties_choose_epoch3_not2")
    h, r = build([1.0] * 13)
    require(run(h, r)["selected_epoch"] == 1, "Earliest-tie test failed")
    checks.append("earliest_tie_and_patience12")
    bad = copy.deepcopy(h); bad[2]["epoch"] = 2
    rejects("duplicate_history_epoch_rejected", bad, r)
    bad = copy.deepcopy(h); bad[2]["steps"] = 99
    rejects("wrong_batch_step_total_rejected", bad, r)
    bad = copy.deepcopy(h); bad[2]["exposure"] = {"native/x_whole": 3}
    rejects("wrong_views_despite_same_count_rejected", bad, r)
    bad = copy.deepcopy(h); key = next(iter(bad[2]["exposure"])); bad[2]["exposure"][key] += 1
    rejects("extra_source_exposure_rejected", bad, r)
    bad = copy.deepcopy(h); bad[2]["selected_rule_ce"] = 0.8
    rejects("incorrect_dev_weighted_ce_rejected", bad, r)
    bad = copy.deepcopy(h); bad[2]["best_epoch"] = 3
    rejects("non_strict_best_epoch_rejected", bad, r)
    short_h, short_r = build([1.0] * 12)
    rejects("premature_early_stop_rejected", short_h, short_r)
    long_h, long_r = build([1.0] * 14)
    rejects("continued_after_early_stop_rejected", long_h, long_r)
    bad_r = dict(r, selected_epoch=2)
    rejects("disconnected_final_selected_epoch_rejected", h, bad_r)
    bad = copy.deepcopy(h); bad[5]["elapsed_seconds"] = 2.0
    rejects("nonmonotone_saved_elapsed_rejected", bad, r)
    # New trace-only linkage tests. These are plain synthetic tensors/dicts, not
    # a network, checkpoint file, real sample, optimizer object, or forward pass.
    import torch
    selected_state = {"synthetic.weight": torch.zeros(37636),
                      "synthetic_norm.running_mean": torch.zeros(1),
                      "synthetic_norm.running_var": torch.ones(1),
                      "synthetic_norm.num_batches_tracked": torch.tensor(29)}
    final_state = copy.deepcopy(selected_state)
    final_state["synthetic_norm.num_batches_tracked"] = torch.tensor(377)
    context = {"labels": ["synthetic"], "word_codes": ["SYNTHETIC"],
               "model": {"in_channels": 280, "width": 32, "dilations": [1, 2, 4, 8],
                         "kernel_size": 3, "dropout": 0.1, "n_classes": 100}}
    result = dict(r, total_steps=377, initial_state_sha256="0" * 64,
                  predictions={"masked": [{"synthetic": True}], "sensor": [{"synthetic": True}]},
                  metrics={"masked": {"mean_ce": 1.2, "top1_correct": 290},
                           "sensor": {"mean_ce": 0.8, "top1_correct": 270}})
    best = {"training_context_sha256": "synthetic", "arm": "C", "seed": SEEDS[0],
            "epoch": 1, "step": 29, "development_ce": 1.0, "state_dict": selected_state,
            "classes": context["labels"], "word_codes": context["word_codes"],
            "in_channels": 280, "model_type": "wis_masked_tcn", "model_version": "masked_v1",
            "config": {k: v for k, v in context["model"].items() if k not in ("in_channels", "n_classes")},
            "initialization": "fresh random; AIHub899 only"}
    last = {"context_sha256": "synthetic", "arm": "C", "seed": SEEDS[0], "epoch": 13, "step": 377,
            "history": h, "initial_state_sha256": "0" * 64, "elapsed_seconds": 13.0,
            "model": final_state,
            "best": {"epoch": 1, "step": 29, "ce": 1.0, "model": copy.deepcopy(selected_state),
                     "predictions": copy.deepcopy(result["predictions"]), "metrics": copy.deepcopy(result["metrics"])},
            "optimizer": {"param_groups": [{"params": [0], "lr": 0.001, "weight_decay": 0.001}],
                          "state": {0: {"step": torch.tensor(377.), "exp_avg": torch.zeros(37636), "exp_avg_sq": torch.zeros(37636)}}},
            "torch_rng": torch.tensor([1, 2, 3], dtype=torch.uint8)}

    def linked(l=last, b=best):
        return inspect_checkpoint_dicts(torch, l, b, result, h, context, "synthetic", "C", SEEDS[0])

    require(linked()["selected_tensors_equal_committed_best"], "Synthetic checkpoint linkage failed")
    checks.append("synthetic_checkpoint_tensor_score_optimizer_rng_linkage")

    def rejects_link(name, l, b):
        try:
            linked(l, b)
        except ValueError:
            checks.append(name)
            return
        raise ValueError("Synthetic checkpoint negative did not reject: " + name)

    bad_best = copy.deepcopy(best); bad_best["state_dict"]["synthetic.weight"][0] = 1.
    rejects_link("changed_selected_weight_rejected", last, bad_best)
    bad_last = copy.deepcopy(last); bad_last["best"]["predictions"]["sensor"] = []
    rejects_link("disconnected_selected_score_rejected", bad_last, best)
    bad_last = copy.deepcopy(last); bad_last["optimizer"]["state"][0]["step"] = torch.tensor(376.)
    rejects_link("optimizer_step_mismatch_rejected", bad_last, best)
    bad_last = copy.deepcopy(last); bad_last["model"]["synthetic_norm.num_batches_tracked"] = torch.tensor(376)
    rejects_link("batchnorm_step_mismatch_rejected", bad_last, best)
    bad_last = copy.deepcopy(last); bad_last["torch_rng"] = None
    rejects_link("missing_resume_rng_rejected", bad_last, best)
    return {"status": "PASS_SYNTHETIC_TRACE_ONLY", "checks": checks, "checks_passed": len(checks),
            "real_run_files_read": 0, "checkpoint_loads": 0, "torch_imported": True,
            "model_constructions": 0, "model_forward": 0, "optimizer_steps_executed": 0,
            "new_training": 0, "new_detector": 0,
            "actual_history_and_checkpoint_audit": "NOT_RUN; execute only after completed nine-run scores audit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true", help="Synthetic history/tensor dictionaries only; no model or real checkpoint")
    group.add_argument("--verify", action="store_true", help="Completed saved runs only; requires PASS score recount first")
    parser.add_argument("--unit-dir", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, help="New JSON inside unit; default training_trace_INDEPENDENT.json for --verify")
    args = parser.parse_args()
    unit = args.unit_dir.resolve()
    output = args.output.resolve() if args.output else (unit / "training_trace_INDEPENDENT.json" if args.verify else None)
    if output:
        require(output.parent == unit, "Output must be inside unit")
    if output and output.exists():
        require(args.verify, "Existing synthetic evidence preserved")
        old = json.loads(output.read_text(encoding="utf-8-sig"))
        require(old["status"] == "PASS_INDEPENDENT_SAVED_TRAINING_TRACE", "Existing trace report is not PASS")
        require(old["verifier_sha256"] == sha(__file__), "Verifier changed; preserve previous audit")
        require(all(sha(p) == digest for p, digest in old["input_file_sha256"].items()), "Previous trace evidence changed")
        print(json.dumps({"status": "EXISTING_TRAINING_TRACE_REUSED", "report": str(output)}, ensure_ascii=False))
        return
    start = time.perf_counter()
    result = self_test() if args.self_test else verify(unit)
    result.update(at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - start,
                  verifier=str(Path(__file__).resolve()), verifier_sha256=sha(__file__),
                  reviewer="Codex roi_training_protocol")
    if output:
        with output.open("x", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_passed", "runs_verified", "elapsed_seconds") if key in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
