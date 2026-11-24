"""Root-gated, immutable seven-target system comparison on existing public inputs.

CLI: experiment.py prepare|run|aggregate|benchmark [--frozen FROZEN.json] [--ready READY.json]
No downloading, input extraction, model selection, evaluation-time tuning, or
production writes. Each stage refuses to overwrite its outputs. A failed run
keeps its first traceback and all partial exposure records for investigation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
import torch
import comparison as c

U = Path(__file__).resolve().parent
CONDITIONS = ("joint", "protected_shape_split_features")
EXPECTED_ROLES = {"fit": 180, "development": 60,
                  "new_feature_expression_evaluation": 48}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(U):
        raise ValueError("Output outside this experiment is forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def save_array(path, value):
    with Path(path).open("xb") as f:
        np.save(f, value, allow_pickle=False)


def check_pins(pins):
    for path, expected in pins.items():
        if c.sha(path) != expected:
            raise ValueError(f"Frozen artifact changed: {path}")


def validate_contract(frozen_path, ready_path):
    frozen_path, ready_path = Path(frozen_path), Path(ready_path)
    f, ready, info = c.read(frozen_path), c.read(ready_path), c.read(U / "DATASET.json")
    expected = dict(status="FROZEN_BEFORE_NEW_SCORES", conditions=list(CONDITIONS),
                    seeds=list(c.SEEDS), epochs=160, lr=.001, weight_decay=.01,
                    threshold=.5, targets=list(c.DEFAULT_TARGETS))
    for name, value in expected.items():
        if f.get(name) != value:
            raise ValueError(f"Unexpected frozen {name}: {f.get(name)!r}")
    if (not str(ready.get("status", "")).startswith("PASS")
            or ready.get("allow_training") is not True
            or ready.get("frozen_sha256") != c.sha(frozen_path)
            or ready.get("dataset_sha256") != c.sha(U / "DATASET.json")):
        raise ValueError("Root READY must explicitly authorize this frozen dataset")
    pins = {**f["pins"], **f["source_pins"], **f["protected"]}
    normalized = {str(Path(p).resolve()): h for p, h in pins.items()}
    required = [U / n for n in ("DATASET.json", "comparison.py", "experiment.py")]
    required += [c.OLD / n for n in ("NORMALIZATION.json", "LEARNING_INPUTS.json", "learning.py")]
    required += [c.OLD / "runs" / f"pose_plus_signrep_{s}" / n
                 for s in c.SEEDS for n in ("RUN.json", "head.pt", "probabilities.npy")]
    if any(str(p.resolve()) not in normalized for p in required):
        raise ValueError("Dataset, implementation, or frozen baseline source is unpinned")
    check_pins(pins)
    check_pins(info.get("pins", {}))
    if tuple(info["selected_targets"]) != c.DEFAULT_TARGETS:
        raise ValueError("Dataset/frozen target mismatch")
    if info["selected_indices"] != [info["targets"].index(k) for k in c.DEFAULT_TARGETS]:
        raise ValueError("Selected name/index mismatch")
    rows = info["rows"]
    if len(rows) != 288 or len({r["clip_id"] for r in rows}) != 96:
        raise ValueError("Only the preserved 96 clips / 288 windows are in scope")
    if Counter(r["role"] for r in rows) != EXPECTED_ROLES:
        raise ValueError("Frozen split count mismatch")
    if [r["sample_id"] for r in rows if r["role"] == "fit"] != f["fit_samples"]:
        raise ValueError("Fit exposure list differs from frozen contract")
    roles_by_clip, roles_by_person = {}, {}
    for row in rows:
        roles_by_clip.setdefault(row["clip_id"], set()).add(row["role"])
        roles_by_person.setdefault(row["person"], set()).add(row["role"])
        if len(row["y"]) != len(info["targets"]) or len(row["target_mask"]) != len(info["targets"]):
            raise ValueError("Target vector width mismatch")
        if not all(isinstance(v, bool) for v in row["target_mask"]):
            raise ValueError("Unknown mask must be explicit booleans")
    if any(len(v) != 1 for v in roles_by_clip.values()):
        raise ValueError("Clip leakage across roles")
    if any("fit" in v and len(v) > 1 for v in roles_by_person.values()):
        raise ValueError("Fit-person leakage across roles")
    identity = dict(frozen_path=str(frozen_path.resolve()), frozen_sha256=c.sha(frozen_path),
                    ready_path=str(ready_path.resolve()), ready_sha256=c.sha(ready_path),
                    dataset_sha256=c.sha(U / "DATASET.json"))
    return f, info, identity


def prepare(frozen_path, ready_path):
    f, info, identity = validate_contract(frozen_path, ready_path)
    for name in ("PREPARED.json", "NORMALIZATION.json"):
        if (U / name).exists():
            raise FileExistsError(U / name)
    batch, stats = c.assemble(info)
    fit = [i for i, r in enumerate(batch.rows) if r["role"] == "fit"]
    weights, events = c.training_weights(batch, fit, range(len(batch.targets)))
    counts = torch.stack([((batch.y[fit] == k) & batch.mask[fit]).sum(0)
                          for k in (0, 1)]).tolist()
    # Loading original probabilities is read-only and never runs a model.
    old_audits = []
    for seed in c.SEEDS:
        stored, missing = c.stored_old_shape_probabilities(info, seed)
        old_audits.append(dict(seed=seed, original_windows_reused=len(rows := batch.rows)-len(missing),
                               new_windows_needing_inference=len(missing),
                               new_clips_needing_inference=len({rows[i]["clip_id"] for i in missing})))
        if len(missing) != 144 or not np.isfinite(stored[[i for i in range(288) if i not in set(missing)]]).all():
            raise ValueError("Expected exactly 48 original and 48 new clips")
    write(U / "NORMALIZATION.json", stats)
    write(U / "PREPARED.json", dict(
        status="PREPARED_NO_MODEL_INFERENCE_OR_TRAINING", at=utc(), **identity,
        normalization_sha256=c.sha(U / "NORMALIZATION.json"),
        rows=288, clips=96, role_windows=dict(EXPECTED_ROLES), targets=list(batch.targets),
        fit_class_counts_zero_one=counts, fit_class_weights_zero_one=weights.tolist(),
        fit_event_weight_sum=events[fit][batch.mask[fit]].sum().item(),
        input_shapes={"pose_mask": list(batch.x.shape), "signrep": list(batch.z.shape)},
        original_probability_reuse=old_audits,
        historical_shape_exposure="Original 36 fit clips, 160 epochs; retained only by protected system",
        comparison="System comparison; unequal historical exposure and parameter counts",
        new_normalization="Only fit windows; overlapping occurrences retained",
        original_normalization="Original whole-clip fit statistics unchanged"))
    print("PREPARED: 288 windows / 96 clips; zero model inference and zero training", flush=True)


def validate_prepared(identity):
    prepared = c.read(U / "PREPARED.json")
    if any(prepared[k] != value for k, value in identity.items()):
        raise ValueError("Prepared identity differs from current READY/FROZEN")
    if c.sha(U / "NORMALIZATION.json") != prepared["normalization_sha256"]:
        raise ValueError("Prepared normalization changed")
    return prepared


def run(frozen_path, ready_path):
    f, info, identity = validate_contract(frozen_path, ready_path)
    prepared = validate_prepared(identity)
    for name in ("TRAIN_STARTED.json", "TRAINING.json", "TRAIN_RUNNING.json", "runs", "protected"):
        if (U / name).exists():
            raise FileExistsError(f"Existing run artifact must be preserved: {U / name}")
    write(U / "TRAIN_RUNNING.json", dict(pid=os.getpid(), started=utc(), **identity))
    write(U / "TRAIN_STARTED.json", dict(pid=os.getpid(), started=utc(), **identity))
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    batch, _ = c.assemble(info, c.read(U / "NORMALIZATION.json"))
    fit = [i for i, r in enumerate(batch.rows) if r["role"] == "fit"]
    fit_batch = batch.subset(fit)
    all_results = []
    (U / "protected").mkdir()
    for seed in c.SEEDS:
        joint, split = c.build_pair(seed, batch.targets)
        start = time.perf_counter()
        protected, audit = c.complete_protected_probabilities(
            batch, seed, allow_new_inference=True, frozen=split.frozen)
        audit.update(inference_seconds=time.perf_counter()-start,
                     original_normalization_sha256=c.sha(c.OLD / "NORMALIZATION.json"),
                     targets=list(c.SHAPE_TARGETS), sample_ids=[r["sample_id"] for r in batch.rows],
                     **identity)
        if audit["new_clips_inferred"] != 48 or audit["old_rows_reused"] != 144:
            raise ValueError("Frozen inference scope differs from 48 new / 48 stored clips")
        cache = U / "protected" / f"probabilities_{seed}.npy"
        save_array(cache, protected.numpy())
        audit["probabilities_sha256"] = c.sha(cache)
        write(U / "protected" / f"AUDIT_{seed}.json", audit)
        stored, missing = c.stored_old_shape_probabilities(info, seed)
        old = [i for i in range(len(batch.rows)) if i not in set(missing)]
        if not np.array_equal(stored[old], protected.numpy()[old]):
            raise ValueError("Original saved probabilities were not preserved exactly")
        frozen_before = {k: v.clone() for k, v in split.frozen.state_dict().items()}
        for condition, model in zip(CONDITIONS, (joint, split)):
            dest = U / "runs" / f"{condition}_{seed}"
            dest.mkdir(parents=True, exist_ok=False)
            ids = list(model.trainable_target_indices)
            class_weights, event_weights = c.training_weights(batch, fit, ids)
            optimizer = torch.optim.AdamW(model.optimizer_parameters(), lr=.001, weight_decay=.01)
            parameters = list(model.optimizer_parameters())
            start, losses = time.perf_counter(), []
            with (dest / "EXPOSURES.jsonl").open("x", encoding="utf8") as exposure:
                for epoch in range(1, 161):
                    model.train()
                    optimizer.zero_grad()
                    logits = model(fit_batch)
                    loss = c.masked_loss(logits, fit_batch.y[:, ids], fit_batch.mask[:, ids],
                                         class_weights, event_weights[fit])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss: {condition} {seed} {epoch}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    exposure.write(json.dumps(dict(epoch=epoch, sample_ids=f["fit_samples"],
                        clip_ids=sorted({batch.rows[i]["clip_id"] for i in fit}),
                        learned_targets=[batch.targets[i] for i in ids],
                        known_target_cells=int(fit_batch.mask[:, ids].sum()), loss=losses[-1])) + "\n")
                    exposure.flush()
                    if epoch % 40 == 0:
                        print(json.dumps(dict(condition=condition, seed=seed, epoch=epoch,
                                              fit_loss=losses[-1])), flush=True)
            elapsed = time.perf_counter() - start
            model.eval()
            eval_start = time.perf_counter()
            with torch.no_grad():
                probabilities = model(batch).sigmoid()
                if condition != "joint":
                    probabilities = split.combine_probabilities(probabilities, protected)
                p = probabilities.numpy()
            eval_seconds = time.perf_counter() - eval_start
            frozen_unchanged = all(torch.equal(v, split.frozen.state_dict()[k])
                                   for k, v in frozen_before.items())
            if not frozen_unchanged:
                raise ValueError("Frozen original branch changed")
            if condition != "joint" and not np.array_equal(p[:, list(split.shape_indices)], protected.numpy()):
                raise ValueError("Protected shape routing changed values")
            if p.shape != (288, 7) or not np.isfinite(p).all():
                raise ValueError("Invalid final probabilities")
            with (dest / "head.pt").open("xb") as out:
                torch.save(model.head.state_dict(), out)
            save_array(dest / "probabilities.npy", p)
            result = dict(status="COMPLETE_FIXED_FINAL_EPOCH", condition=condition, seed=seed,
                epochs=160, selected_epoch=160, selection="Fixed final epoch, no score-based selection",
                training_seconds=elapsed, final_head_inference_seconds=eval_seconds,
                trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                total_parameters=sum(v.numel() for v in model.parameters()),
                learned_targets=[batch.targets[i] for i in ids], losses=losses,
                class_weights_zero_one=class_weights.tolist(), fit_samples=f["fit_samples"],
                checkpoint_sha256=c.sha(dest / "head.pt"),
                probabilities_sha256=c.sha(dest / "probabilities.npy"),
                exposures_sha256=c.sha(dest / "EXPOSURES.jsonl"),
                normalization_sha256=prepared["normalization_sha256"],
                protected_probability_sha256=c.sha(cache), original_shape_weights_unchanged=frozen_unchanged,
                old_clip_reinference=0, backbone_updates=0, **identity)
            write(dest / "COMPLETE.json", result)
            compact = {k: result[k] for k in ("condition", "seed", "training_seconds",
                                             "trainable_parameters", "total_parameters")}
            all_results.append(compact)
            print(json.dumps(compact), flush=True)
    validate_contract(frozen_path, ready_path)
    write(U / "TRAINING.json", dict(status="COMPLETE_SIX_FIXED_RUNS", at=utc(), runs=all_results,
        backbone_updates=0, existing_shape_updates=0, old_clip_reinference=0,
        new_protected_clip_inferences_per_seed=48, **identity))
    # Only this process's transient lock is removed; immutable start/exposure files remain.
    (U / "TRAIN_RUNNING.json").unlink()


def metrics(p, y, mask, names):
    per_target = []
    for j, name in enumerate(names):
        prediction, truth, known = p[:, j] >= .5, y[:, j], mask[:, j]
        tp = int((known & (truth == 1) & prediction).sum())
        tn = int((known & (truth == 0) & ~prediction).sum())
        fp = int((known & (truth == 0) & prediction).sum())
        fn = int((known & (truth == 1) & ~prediction).sum())
        pos, neg, n = tp + fn, tn + fp, int(known.sum())
        per_target.append(dict(target=name, known=n, unknown=int((~known).sum()),
            positives=pos, negatives=neg, tp=tp, tn=tn, fp=fp, fn=fn,
            sensitivity=tp/pos if pos else None, specificity=tn/neg if neg else None,
            balanced_accuracy=(tp/pos+tn/neg)/2 if pos and neg else None,
            accuracy=(tp+tn)/n if n else None, label_coverage=n/len(known) if len(known) else None))
    supported = [r["balanced_accuracy"] for r in per_target if r["balanced_accuracy"] is not None]
    return dict(targets=per_target, known_cells=int(mask.sum()), unknown_cells=int((~mask).sum()),
                label_coverage=float(mask.mean()) if mask.size else None,
                balanced_supported_targets=len(supported),
                macro_balanced_accuracy=float(np.mean(supported)) if supported else None)


def aggregate(frozen_path, ready_path):
    f, info, identity = validate_contract(frozen_path, ready_path)
    validate_prepared(identity)
    if (U / "SCORES.json").exists():
        raise FileExistsError(U / "SCORES.json")
    training = c.read(U / "TRAINING.json")
    if training["status"] != "COMPLETE_SIX_FIXED_RUNS" or any(training[k] != v for k, v in identity.items()):
        raise ValueError("Training manifest does not match frozen identities")
    expected = {(a, s) for a in CONDITIONS for s in c.SEEDS}
    if len(training["runs"]) != 6 or {(r["condition"], r["seed"]) for r in training["runs"]} != expected:
        raise ValueError("The six prespecified runs are incomplete")
    ids, names = info["selected_indices"], info["selected_targets"]
    y = np.asarray([r["y"] for r in info["rows"]], np.float32)[:, ids]
    mask = np.asarray([r["target_mask"] for r in info["rows"]], bool)[:, ids]
    families = dict(all=list(range(7)), commonshape=[0, 1, 2], new_closed_shape=[3, 4],
                    all_shape=[0, 1, 2, 3, 4], relation=[5, 6], new_features=[3, 4, 5, 6])
    out, manifests = [], []
    for rr in training["runs"]:
        condition, seed = rr["condition"], rr["seed"]
        dest = U / "runs" / f"{condition}_{seed}"
        complete = c.read(dest / "COMPLETE.json")
        for key, filename in (("checkpoint_sha256", "head.pt"),
                              ("probabilities_sha256", "probabilities.npy"),
                              ("exposures_sha256", "EXPOSURES.jsonl")):
            if c.sha(dest / filename) != complete[key]:
                raise ValueError("Run artifact changed")
        if complete["selected_epoch"] != 160 or any(complete[k] != v for k, v in identity.items()):
            raise ValueError("Run identity or selected epoch changed")
        exposures = [json.loads(line) for line in (dest / "EXPOSURES.jsonl").read_text().splitlines()]
        if len(exposures) != 160 or any(e["epoch"] != i+1 or e["sample_ids"] != f["fit_samples"]
                                      for i, e in enumerate(exposures)):
            raise ValueError("Incomplete or non-fit training exposures")
        p = np.load(dest / "probabilities.npy", allow_pickle=False)
        if p.shape != y.shape or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError("Invalid prediction schema")
        if condition != "joint":
            cache = U / "protected" / f"probabilities_{seed}.npy"
            audit = c.read(U / "protected" / f"AUDIT_{seed}.json")
            if c.sha(cache) != audit["probabilities_sha256"] or c.sha(cache) != complete["protected_probability_sha256"]:
                raise ValueError("Protected cache changed")
            baseline = np.load(cache, allow_pickle=False)
            original, missing = c.stored_old_shape_probabilities(info, seed)
            old = [i for i in range(288) if i not in set(missing)]
            if not np.array_equal(p[:, :3], baseline) or not np.array_equal(original[old], baseline[old]):
                raise ValueError("Exact baseline preservation failed")
        for role in EXPECTED_ROLES:
            rows = [i for i, r in enumerate(info["rows"]) if r["role"] == role]
            for family, cols in families.items():
                out.append(dict(condition=condition, seed=seed, role=role, family=family,
                    windows=len(rows), **metrics(p[np.ix_(rows, cols)], y[np.ix_(rows, cols)],
                                                 mask[np.ix_(rows, cols)], [names[i] for i in cols])))
            if condition != "joint":
                out.append(dict(condition="original_protected_baseline", seed=seed, role=role,
                    family="commonshape", windows=len(rows), **metrics(baseline[rows],
                    y[np.ix_(rows, [0, 1, 2])], mask[np.ix_(rows, [0, 1, 2])], names[:3])))
        manifests.append(dict(condition=condition, seed=seed, path=str(dest / "COMPLETE.json"),
                              sha256=c.sha(dest / "COMPLETE.json")))
    summary = []
    for condition, role, family in sorted({(r["condition"], r["role"], r["family"]) for r in out}):
        group = [r for r in out if (r["condition"], r["role"], r["family"]) == (condition, role, family)]
        values = [r["macro_balanced_accuracy"] for r in group if r["macro_balanced_accuracy"] is not None]
        summary.append(dict(condition=condition, role=role, family=family,
            mean=float(np.mean(values)) if values else None, range=[min(values), max(values)] if values else None,
            known_cells=group[0]["known_cells"], unknown_cells=group[0]["unknown_cells"],
            balanced_supported_targets=group[0]["balanced_supported_targets"]))
    deltas, baseline_deltas = [], []
    for role in EXPECTED_ROLES:
        for j, name in enumerate(names):
            groups = {}
            for condition in CONDITIONS:
                rows = [r for r in out if r["condition"] == condition and r["role"] == role and r["family"] == "all"]
                groups[condition] = {r["seed"]: r["targets"][j] for r in rows}
            differences = [groups[CONDITIONS[1]][s]["balanced_accuracy"] - groups[CONDITIONS[0]][s]["balanced_accuracy"]
                           for s in c.SEEDS if groups[CONDITIONS[0]][s]["balanced_accuracy"] is not None]
            deltas.append(dict(role=role, target=name, comparison="protected_split_minus_joint",
                mean_balanced_accuracy_delta=float(np.mean(differences)) if differences else None,
                paired_seed_deltas=differences,
                regression=bool(differences and np.mean(differences) < 0),
                known=groups[CONDITIONS[0]][42]["known"],
                positives=groups[CONDITIONS[0]][42]["positives"], negatives=groups[CONDITIONS[0]][42]["negatives"]))
            if j < 3:
                baseline = {r["seed"]: r["targets"][j] for r in out if r["condition"] == "original_protected_baseline"
                            and r["role"] == role and r["family"] == "commonshape"}
                for condition in CONDITIONS:
                    values = [groups[condition][s]["balanced_accuracy"] - baseline[s]["balanced_accuracy"]
                              for s in c.SEEDS if baseline[s]["balanced_accuracy"] is not None]
                    baseline_deltas.append(dict(role=role, target=name, condition=condition,
                        comparison="new_system_minus_original_protected_baseline",
                        mean_balanced_accuracy_delta=float(np.mean(values)) if values else None,
                        paired_seed_deltas=values, regression=bool(values and np.mean(values) < 0)))
    write(U / "SCORES.json", dict(status="FIXED_STORED_SCORE_COMPARISON", at=utc(), **identity,
        rows=out, summary=summary, per_target_deltas=deltas, original_baseline_deltas=baseline_deltas,
        targets=names, run_manifests=manifests,
        baseline="Three compatible original outputs: saved original48 plus once-per-seed new48 inference",
        new_four_rule_baseline="Unavailable: no same-definition frozen rule/threshold identified; not invented",
        comparison_scope="Unequal system capacity and historical exposure; not architecture-isolation evidence",
        unknowns="Excluded per-target, never counted correct/incorrect; label coverage is not model abstention",
        evaluation_history="Previously exposed public TCN lineage; not a project-wide unseen test",
        not_word_or_educational_accuracy=True, no_production_adoption=True))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def benchmark(frozen_path, ready_path):
    """Saved new-head checkpoints, synthetic tensors; never re-infer real clips."""
    _, _, identity = validate_contract(frozen_path, ready_path)
    validate_prepared(identity)
    training = c.read(U / "TRAINING.json")
    if training["status"] != "COMPLETE_SIX_FIXED_RUNS" or not (U / "SCORES.json").exists():
        raise ValueError("Benchmark only follows completed fixed training and aggregation")
    if (U / "HEAD_RUNTIME.json").exists():
        raise FileExistsError(U / "HEAD_RUNTIME.json")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261101)
    results = []
    with torch.inference_mode():
        for rr in training["runs"]:
            dest = U / "runs" / f"{rr['condition']}_{rr['seed']}"
            complete = c.read(dest / "COMPLETE.json")
            if c.sha(dest / "head.pt") != complete["checkpoint_sha256"]:
                raise ValueError("Benchmark checkpoint changed")
            head = c.WindowHead(7 if rr["condition"] == "joint" else 4)
            head.load_state_dict(torch.load(dest / "head.pt", map_location="cpu", weights_only=True), strict=True)
            head.eval()
            for batch_size in (1, 3, 48):
                x, z = torch.randn(batch_size, 13, 138), torch.randn(batch_size, 13, 768)
                for _ in range(5):
                    head(x, z)
                milliseconds = []
                for _ in range(50):
                    start = time.perf_counter()
                    head(x, z)
                    milliseconds.append((time.perf_counter() - start)*1000)
                results.append(dict(condition=rr["condition"], seed=rr["seed"], batch_windows=batch_size,
                    checkpoint_sha256=complete["checkpoint_sha256"], repeats=50, warmup=5,
                    median_ms=float(np.median(milliseconds)), p95_ms=float(np.percentile(milliseconds, 95)),
                    min_ms=min(milliseconds), max_ms=max(milliseconds)))
    write(U / "HEAD_RUNTIME.json", dict(status="SYNTHETIC_SAVED_NEW_HEAD_RUNTIME_ONLY", at=utc(), **identity,
        rows=results, device="cpu", torch_version=torch.__version__, threads=4,
        scope="New learned branch forward only; 3 windows matches current per-clip window count",
        excluded="Decode, feature extraction, frozen original full-clip head, DTW, feedback and rendering",
        actual_clip_reinference=0, end_to_end_latency_claim=False))
    print("HEAD_RUNTIME: synthetic saved new-head forwards only, no end-to-end claim", flush=True)


def selftest():
    # Verify metrics do not reward unknowns and unsupported one-class BA is null.
    p = np.array([[.9, .9], [.7, .1], [.1, .9]], np.float32)
    y = np.array([[1, 0], [0, 1], [np.nan, 0]], np.float32)
    m = np.array([[1, 0], [1, 1], [0, 0]], bool)
    result = metrics(p, y, m, ["both", "one_class"])
    assert result["targets"][0]["balanced_accuracy"] == .5
    assert result["targets"][0]["fp"] == 1 and result["targets"][0]["unknown"] == 1
    assert result["targets"][1]["balanced_accuracy"] is None
    assert result["known_cells"] == 3 and result["unknown_cells"] == 3
    assert result["balanced_supported_targets"] == 1
    return dict(status="PASS_SYNTHETIC_METRICS", unknown_exclusion=True,
                one_class_ba_null=True, no_real_training_or_inference=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "aggregate", "benchmark", "selftest"))
    parser.add_argument("--frozen", type=Path, default=U / "FROZEN.json")
    parser.add_argument("--ready", type=Path, default=U / "READY.json")
    args = parser.parse_args()
    try:
        if args.command == "selftest":
            print(json.dumps(selftest()))
        else:
            globals()[args.command](args.frozen, args.ready)
    except BaseException:
        failure = U / f"{args.command.upper()}_FIRST_FAILURE.txt"
        try:
            with failure.open("x", encoding="utf8") as out:
                out.write(traceback.format_exc())
        except FileExistsError:
            pass
        raise


if __name__ == "__main__":
    main()
