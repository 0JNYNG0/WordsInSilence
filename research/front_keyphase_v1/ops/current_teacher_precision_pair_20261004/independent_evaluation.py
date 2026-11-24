"""Independent current-teacher precision-unit evaluation; never executes a learner.

Written by Codex for verification. Uses torch only to read tensor checkpoints.
No model construction, encoder forward, training, raw video, service, or camera.
This new-unit verifier leaves the technically stopped original unit unchanged.
Run after six frozen checkpoints exist:
  python independent_evaluation.py --config PREPARED_CONFIG.json
The sole output is create-exclusive INDEPENDENT_EVALUATION.json beside this file.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

GROUPS = {"annotation_dev": 699, "whole_dev": 74, "public": 32, "mp": 14}
FAMILIES = ("control", "current_teacher")
TOL_METRIC = 1e-12


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode("ascii"))
    h.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    h.update(value.tobytes(order="C"))
    return h.hexdigest()


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def check_pins(config):
    for kind in ("pins", "protected_files"):
        for path, expected in config[kind].items():
            require(file_sha(path) == expected, f"{kind} changed: {path}")


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def project_names(path):
    """Read the protected constant as syntax, without importing the service."""
    module = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    values = []
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PROJECT_18"
            for target in node.targets
        ):
            constant = ast.literal_eval(node.value)
            values.append(list(constant.values()) if isinstance(constant, dict) else list(constant))
    require(len(values) == 1 and len(set(values[0])) == 18, "PROJECT_18 constant contract")
    return set(values[0])


def read_state(path, expected_classes):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(list(payload["classes"]) == expected_classes, f"class order: {path}")
    result = {}
    for name, tensor in payload["state_dict"].items():
        require(isinstance(tensor, torch.Tensor), f"non-tensor state: {path}:{name}")
        value = tensor.detach().cpu().numpy().copy()
        require(np.isfinite(value).all(), f"nonfinite state: {path}:{name}")
        result[name] = value
    require(result["head.weight"].shape == (93, 64), f"head weight: {path}")
    require(result["head.bias"].shape == (93,), f"head bias: {path}")
    require(result["head.weight"].dtype == result["head.bias"].dtype == np.float32,
            f"head dtype: {path}")
    return result


def identical_encoder(parent, other, name):
    require(set(parent) == set(other), f"state keys differ: {name}")
    for key in parent:
        if key in ("head.weight", "head.bias"):
            continue
        a, b = parent[key], other[key]
        require(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(),
                f"non-head state not bitwise identical: {name}:{key}")


def affine(pooled, state):
    require(pooled.ndim == 2 and pooled.shape[1] == 64 and np.isfinite(pooled).all(),
            "invalid frozen pooled cache")
    scores = (pooled.astype(np.float64) @ state["head.weight"].astype(np.float64).T
              + state["head.bias"].astype(np.float64))
    require(scores.shape == (len(pooled), 93) and np.isfinite(scores).all(), "affine score contract")
    return scores


def receipt_array_sha(value):
    """Independently implement the runner's receipt byte encoding, not its helper."""
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode()
                          + value.tobytes(order="C")).hexdigest()


def verify_teachers(cfg, run_dir, classes, meta, new, parent, reference, training, config_sha):
    """Reconstruct both teachers from pinned training arrays; never use evaluation scores."""
    tp = cfg["teacher_preparation"]
    features = read_arrays(tp["base_pooled_path"])
    historical = read_arrays(tp["historical_teacher_scores_path"])
    cached = read_arrays(tp["cached_teacher_path"])
    rows = meta["base"]
    ids = [str(row["id"]) for row in rows]
    require(len(ids) == len(set(ids)) == 2122, "base training IDs")
    require(all(row.get("role") == "train" for row in rows), "non-training base row")
    require(all(row.get("signer_id") is None or int(row["signer_id"]) in range(1, 10)
                for row in rows), "base row from a held-out signer")
    labels = np.asarray([classes.index(row["label"]) for row in rows], dtype=np.int64)
    require(features["base_x"].shape == (2122, 64)
            and features["base_x"].dtype == np.float32, "base pooled contract")
    for source in (features, historical, cached):
        require(source["classes"].tolist() == classes, "teacher source class order")
    require(np.array_equal(features["base_y"], labels)
            and np.array_equal(cached["base_y"], labels), "teacher labels/row order")
    old_ids = historical["ids"].tolist()
    require(len(old_ids) == len(set(old_ids)), "historical teacher IDs not unique")
    lookup = {str(row_id): i for i, row_id in enumerate(old_ids)}
    require(all(row_id in lookup for row_id in ids), "missing teacher training ID")
    positions = np.asarray([lookup[row_id] for row_id in ids], dtype=np.int64)
    require(bool(np.all(np.diff(positions) > 0)), "historical teacher relative row order")
    old_teacher = cached["teacher_base_logits"]
    require(old_teacher.shape == (2122, 93) and old_teacher.dtype == np.float64,
            "old teacher dtype/shape")
    source_error = float(np.max(np.abs(historical["parent"][positions] - old_teacher)))
    require(source_error <= 1e-5, "historical parent teacher does not reproduce cached teacher")
    base = set(ids)
    overlaps = {group: sorted(base & {str(row["id"]) for row in meta[group]}) for group in GROUPS}
    require(not overlaps["annotation_dev"] and not overlaps["whole_dev"], "held-out native rows in teacher")
    require(not base & set(new["ids"].tolist()), "new KSL evaluation rows in teacher")
    # public and mp may be training-exposed regression references; do not reject or hide those overlaps.
    preflight_path, teachers_path = run_dir / "PREFLIGHT.json", run_dir / "teachers.npz"
    receipts = {str(p): file_sha(p) for p in (preflight_path, teachers_path)}
    preflight = read_json(preflight_path)
    frozen = read_arrays(teachers_path)
    require(preflight["config_sha256"] == config_sha, "preflight config hash")
    require(training["preflight_sha256"] == receipts[str(preflight_path)], "training/preflight link")
    require(training["teachers_npz_sha256"] == preflight["files"]["teachers.npz"]
            == receipts[str(teachers_path)], "frozen teacher receipt link")
    require(frozen["classes"].tolist() == classes and frozen["base_ids"].tolist() == ids,
            "frozen teacher class/row order")
    require(np.array_equal(frozen["base_y"], labels), "frozen teacher labels")
    require(np.array_equal(frozen["control_teacher_b8_cached"], old_teacher), "control teacher changed")
    b8_affine = affine(features["base_x"], parent)
    teacher832 = affine(features["base_x"], reference)
    require(np.array_equal(frozen["b8_affine_teacher"], b8_affine), "saved b8 affine mismatch")
    require(np.array_equal(frozen["current832_teacher"], teacher832), "saved 832 teacher mismatch")
    error_b8 = float(np.max(np.abs(b8_affine - old_teacher)))
    require(error_b8 <= 1e-5, "b8 affine does not reproduce cached teacher")
    hashes = {"control": receipt_array_sha(old_teacher), "current_teacher": receipt_array_sha(teacher832)}
    for family, digest in hashes.items():
        require(training["teacher_sha256"][family] == preflight["teachers"][family]["sha256"] == digest,
                f"teacher content receipt: {family}")
    require(str(frozen["base_x_sha256"].item()) == receipt_array_sha(features["base_x"])
            == preflight["teachers"]["base_x_sha256"], "base pooled receipt")
    require(preflight["teachers"]["b8_affine_sha256"] == receipt_array_sha(b8_affine), "b8 affine receipt")
    require(not np.array_equal(old_teacher, teacher832), "no difference between teachers")
    schedule = cached["base_indices"]
    require(schedule.shape == (660, 16) and schedule.min() >= 0 and schedule.max() < 2122,
            "teacher training schedule bounds")
    require(preflight["teacher_rows_used_by_base_schedule"] == len(np.unique(schedule)), "used teacher row count")
    return {"receipts": receipts, "training_rows": 2122, "classes": 93,
            "base_ids_sha256": array_sha(np.asarray(ids)), "teacher_sha256_runner_encoding": hashes,
            "historical_parent_vs_cache_max_abs": source_error,
            "b8_affine_vs_cached_max_abs": error_b8, "saved_teachers_reconstructed_exactly": True,
            "native_dev_and_new36_overlap_ids": 0,
            "training_exposed_regression_overlap_ids": {g: overlaps[g] for g in ("public", "mp")},
            "note": "Teachers use the fixed training base rows only; public/mp overlaps are existing training-reference reuse."}


def metric(y, predictions, rows, classes):
    require(len(y) == len(predictions) == len(rows), "metric lengths")
    good = predictions == y
    labels = {}
    for index in sorted(set(y.tolist())):
        select = y == index
        n = int(select.sum())
        count = int(good[select].sum())
        labels[classes[index]] = {"n": n, "correct": count, "recall": count / n}
    people = {}
    for person in sorted({str(r["signer_id"]) for r in rows if r.get("signer_id") is not None}):
        select = np.asarray([str(r.get("signer_id")) == person for r in rows], dtype=bool)
        people[person] = {"n": int(select.sum()), "correct": int(good[select].sum())}
    return {"n": len(rows), "correct": int(good.sum()),
            "macro_recall": float(np.mean([v["recall"] for v in labels.values()])) if labels else None,
            "per_class": labels, "per_signer": people}


def summarize(scores, rows, y, classes, target_indices):
    full = metric(y, np.argmax(scores, axis=1), rows, classes)
    selected = np.isin(y, target_indices)
    limited_predictions = target_indices[np.argmax(scores[selected][:, target_indices], axis=1)]
    limited = metric(y[selected], limited_predictions,
                     [row for row, keep in zip(rows, selected) if keep], classes)
    return {"full93": full, "project18_separate_task": limited}


def row_prediction(scores, truth, classes, indices):
    if truth not in indices:
        return None
    candidates = np.asarray(indices, dtype=np.int64)
    ordering = candidates[np.argsort(-scores[candidates], kind="stable")]
    predicted = int(ordering[0])
    alternatives = candidates[candidates != truth]
    return {"prediction": classes[predicted], "prediction_index": predicted,
            "correct": predicted == truth,
            "truth_rank_stable_class_order": int(np.flatnonzero(ordering == truth)[0]) + 1,
            "truth_margin": float(scores[truth] - np.max(scores[alternatives])),
            "top1_top2_margin": float(scores[ordering[0]] - scores[ordering[1]]),
            "top_score_ties": int(np.sum(scores[candidates] == scores[predicted]))}


def compare_metrics(actual, expected, name):
    for key in ("n", "correct"):
        require(actual[key] == expected[key], f"reference {name}:{key}")
    if expected.get("macro_recall") is not None:
        require(abs(actual["macro_recall"] - expected["macro_recall"]) <= TOL_METRIC,
                f"reference {name}:macro_recall")
    require(actual["per_signer"] == expected.get("per_signer", {}), f"reference {name}:signers")
    require(set(actual["per_class"]) == set(expected["per_class"]), f"reference {name}:class set")
    for label, values in actual["per_class"].items():
        for key in ("n", "correct"):
            require(values[key] == expected["per_class"][label][key], f"reference {name}:{label}:{key}")


def retention(summary, candidate, reference):
    def get(group, model, condition="intact", scope="full93"):
        return summary[group][model][condition][scope]
    a, b = get("annotation_dev", candidate), get("annotation_dev", reference)
    gates = {"annotation_full93_correct": a["correct"] >= b["correct"],
             "annotation_macro_recall": a["macro_recall"] >= b["macro_recall"] - TOL_METRIC,
             "annotation_each_signer": all(a["per_signer"][p]["correct"] >= b["per_signer"][p]["correct"]
                                           for p in ("11", "12", "13")),
             "annotation_project18_separate": get("annotation_dev", candidate, scope="project18_separate_task")["correct"]
                  >= get("annotation_dev", reference, scope="project18_separate_task")["correct"]}
    for group in ("whole_dev", "public", "mp"):
        for condition in summary[group][reference]:
            for scope in ("full93", "project18_separate_task"):
                gates[f"{group}/{condition}/{scope}"] = get(group, candidate, condition, scope)["correct"] >= get(group, reference, condition, scope)["correct"]
    return gates


def changed_rows(reference_scores, candidate_scores, y, rows, classes):
    before = np.argmax(reference_scores, axis=1)
    after = np.argmax(candidate_scores, axis=1)
    records = []
    for i in np.flatnonzero(before != after):
        row = rows[int(i)]
        records.append({"row": int(i), "id": row["id"], "condition": row.get("condition", "intact"),
                        "true": classes[int(y[i])], "before": classes[int(before[i])],
                        "after": classes[int(after[i])], "gain": bool(before[i] != y[i] and after[i] == y[i]),
                        "regression": bool(before[i] == y[i] and after[i] != y[i]),
                        "signer_id": row.get("signer_id")})
    return {"prediction_changes": len(records), "gains": sum(r["gain"] for r in records),
            "regressions": sum(r["regression"] for r in records), "rows": records}


def run(config_path):
    config_path = Path(config_path).resolve()
    initial_config_sha = file_sha(config_path)
    cfg = read_json(config_path)
    root, run_dir = Path(cfg["root"]), Path(cfg["out_dir"])
    destination = Path(__file__).resolve().parent / "INDEPENDENT_EVALUATION.json"
    require(not destination.exists(), "independent output already exists")
    check_pins(cfg)
    paths = cfg["paths"]
    meta = read_json(paths["metadata"])
    cached = read_arrays(paths["reference_evaluation_arrays"])
    new = read_arrays(paths["ksl_prefixes"])
    known = read_json(root / "runs/partial_encoder_pair_v1/evaluation.json")
    classes = cached["classes"].tolist()
    require(len(set(classes)) == 93 and new["classes"].tolist() == classes, "class contract")
    names18 = project_names(root / "src/wis_lab/service/neural.py")
    require(names18.issubset(classes), "unknown project18 class")
    target_indices = np.asarray([i for i, name in enumerate(classes) if name in names18], dtype=np.int64)
    model_paths = {"ref832": Path(paths["current832"])}
    model_paths.update({f"{family}_fold{fold}": run_dir / f"{family}_fold{fold}_final.pt"
                        for family in FAMILIES for fold in range(3)})
    require(all(path.is_file() for path in model_paths.values()), "six checkpoints and reference must exist")
    initial_hashes = {str(path): file_sha(path) for path in model_paths.values()}
    training_path = run_dir / "TRAINING.json"
    training_sha = file_sha(training_path)
    training_record = read_json(training_path)
    require(training_record["config_sha256"] == initial_config_sha, "training configuration hash")
    for path in model_paths.values():
        if path.parent == run_dir:
            require(training_record["files"][path.name] == initial_hashes[str(path)],
                    f"trained checkpoint hash differs from receipt: {path.name}")
    parent = read_state(paths["parent"], classes)
    states = {name: read_state(path, classes) for name, path in model_paths.items()}
    for name, state in states.items():
        identical_encoder(parent, state, name)
    teachers = verify_teachers(cfg, run_dir, classes, meta, new, parent, states["ref832"],
                               training_record, initial_config_sha)
    scores, summary, rows_out, changes, reproduction = {}, {}, [], {}, {}
    known_rows = {(r["group"], r["id"], r.get("condition", "intact")): r for r in known["rows"]}
    for group, count in GROUPS.items():
        rows = meta[group]
        require(len(rows) == count, f"{group} metadata count")
        labels = [r.get("label", r.get("true", r.get("reference_label"))) for r in rows]
        y = np.asarray([classes.index(label) for label in labels], dtype=np.int64)
        pooled = cached[group + "__control__pooled"]
        require(pooled.shape == (count, 64), f"{group} pooled shape")
        scores[group] = {name: affine(pooled, state) for name, state in states.items()}
        maximum = float(np.max(np.abs(scores[group]["ref832"] - cached[group + "__control__scores"])))
        require(maximum <= 1e-6, f"{group} ref832 score mismatch {maximum}")
        reproduction[group] = {"max_score_abs_difference": maximum,
                               "pooled_sha256": array_sha(pooled)}
        conditions = sorted({r.get("condition", "intact") for r in rows})
        summary[group] = {}
        for model_name, values in scores[group].items():
            summary[group][model_name] = {}
            for condition in conditions:
                select = np.asarray([r.get("condition", "intact") == condition for r in rows], dtype=bool)
                selected_rows = [r for r, keep in zip(rows, select) if keep]
                result = summarize(values[select], selected_rows, y[select], classes, target_indices)
                summary[group][model_name][condition] = result
                if model_name == "ref832":
                    for scope, oldscope in (("full93", "full93"), ("project18_separate_task", "project18")):
                        compare_metrics(result[scope], known["summary"][group]["control"][condition][oldscope],
                                        f"{group}/{condition}/{scope}")
        for i, row in enumerate(rows):
            key = (group, row["id"], row.get("condition", "intact"))
            require(key in known_rows and known_rows[key]["true"] == labels[i], f"metadata alignment {key}")
            item = {"group": group, "row": i, "id": row["id"], "condition": key[2],
                    "true": labels[i], "signer_id": row.get("signer_id"), "models": {}}
            for name, values in scores[group].items():
                item["models"][name] = {"full93": row_prediction(values[i], int(y[i]), classes, range(93)),
                                        "project18_separate_task": row_prediction(values[i], int(y[i]), classes, target_indices)}
            require(item["models"]["ref832"]["full93"]["prediction"] == known_rows[key]["models"]["control"]["full93"]["pred"],
                    f"reference prediction mismatch {key}")
            rows_out.append(item)
        changes[group] = {}
        for fold in range(3):
            candidate, control = f"current_teacher_fold{fold}", f"control_fold{fold}"
            for reference in ("ref832", control):
                changes[group][f"{candidate}_vs_{reference}"] = changed_rows(scores[group][reference], scores[group][candidate], y, rows, classes)
    require(summary["annotation_dev"]["ref832"]["intact"]["full93"]["correct"] == 691,
            "current reference is not expected 691/699")

    y = new["y"].astype(np.int64)
    fold_ids, exposed = new["folds"], new["exposed"]
    require(y.shape == (36,) and exposed.dtype == np.bool_ and int(exposed.sum()) == 2, "KSL36/exposed contract")
    require(set(fold_ids.tolist()) == {0, 1, 2}, "KSL folds")
    new_rows = [{"id": str(name), "condition": "ksl_original_excluded_development"} for name in new["ids"]]
    new_scores = {name: affine(new["pooled_832"], state) for name, state in states.items()}
    error = float(np.max(np.abs(new_scores["ref832"] - new["logits_cached"])))
    require(error <= 1e-5, f"KSL ref832 cache mismatch: {error}")
    reproduction["ksl36"] = {"max_score_abs_difference": error, "pooled_sha256": array_sha(new["pooled_832"])}
    for family in FAMILIES:
        new_scores[family + "_oof"] = np.stack([new_scores[f"{family}_fold{int(f)}"][i] for i, f in enumerate(fold_ids)])
    selections = {"primary_unexposed34": ~exposed, "previously_exposed2": exposed,
                  "supplement_all36": np.ones(36, dtype=bool)}
    selections.update({f"heldout_fold{fold}": fold_ids == fold for fold in range(3)})
    new_summary = {part: {name: summarize(value[select], [r for r, keep in zip(new_rows, select) if keep], y[select], classes, target_indices)
                          for name, value in new_scores.items()} for part, select in selections.items()}
    primary = new_summary["primary_unexposed34"]
    a, b = primary["current_teacher_oof"]["full93"], primary["control_oof"]["full93"]
    study, eat = "\uacf5\ubd80", "\uba39\ub2e4"
    require(a["per_class"][study]["n"] == 18 and a["per_class"][eat]["n"] == 16, "primary class counts")
    primary_gates = {"total_at_least31": a["correct"] >= 31,
                     "total_ge_control": a["correct"] >= b["correct"],
                     "study_at_least17": a["per_class"][study]["correct"] >= 17,
                     "eat_at_least14": a["per_class"][eat]["correct"] >= 14,
                     "each_class_ge_control": all(a["per_class"][label]["correct"] >= b["per_class"][label]["correct"] for label in (study, eat))}
    retention_gates = {}
    for fold in range(3):
        for reference in ("ref832", f"control_fold{fold}"):
            retention_gates[f"current_teacher_fold{fold}_vs_{reference}"] = retention(summary, f"current_teacher_fold{fold}", reference)
    annotation_minimum = {f"current_teacher_fold{fold}_annotation_dev_full93_ge_691":
                          summary["annotation_dev"][f"current_teacher_fold{fold}"]["intact"]["full93"]["correct"] >= 691
                          for fold in range(3)}
    pass_gates = (all(primary_gates.values()) and all(all(g.values()) for g in retention_gates.values())
                  and all(annotation_minimum.values()))
    new_predictions = []
    for i, row in enumerate(new_rows):
        new_predictions.append({"id": row["id"], "fold": int(fold_ids[i]), "previously_exposed": bool(exposed[i]),
                                "true": classes[int(y[i])], "models": {
            name: {"full93": row_prediction(value[i], int(y[i]), classes, range(93)),
                   "project18_separate_task": row_prediction(value[i], int(y[i]), classes, target_indices)}
            for name, value in new_scores.items()}})
    new_changes = {}
    for reference in ("ref832", "control_oof"):
        take = ~exposed
        new_changes[f"current_teacher_oof_vs_{reference}"] = changed_rows(new_scores[reference][take], new_scores["current_teacher_oof"][take],
                                                                    y[take], [r for r, keep in zip(new_rows, take) if keep], classes)
    score_hashes = {group: {name: array_sha(value) for name, value in values.items()} for group, values in scores.items()}
    score_hashes["ksl36"] = {name: array_sha(value) for name, value in new_scores.items()}
    comparison = {"status": "CANDIDATE_EVALUATION_NOT_PRESENT"}
    candidate_json = run_dir / "EVALUATION.json"
    candidate_npz = run_dir / "evaluation_arrays.npz"
    if candidate_json.exists() or candidate_npz.exists():
        require(candidate_json.exists() and candidate_npz.exists(), "partial candidate evaluation outputs")
        authored = read_json(candidate_json)
        require(authored["arrays_sha256"] == file_sha(candidate_npz), "candidate score receipt mismatch")
        require(authored["training_record_sha256"] == training_sha, "candidate training receipt mismatch")
        require(authored["config_sha256"] == initial_config_sha, "candidate config mismatch")
        difference, exact, score_count = 0.0, True, 0
        with np.load(candidate_npz, allow_pickle=False) as saved:
            require(saved["classes"].tolist() == classes, "candidate score class order")
            require(np.array_equal(saved["new__ids"], new["ids"]), "candidate KSL row order")
            for field in ("y", "folds", "exposed"):
                require(np.array_equal(saved[f"new__{field}"], new[field]), f"candidate KSL {field}")
            require(np.array_equal(saved["new__pooled"], new["pooled_832"]), "candidate KSL pooled cache")
            for group in GROUPS:
                require(saved[f"{group}__ids"].tolist() == [str(r["id"]) for r in meta[group]],
                        f"candidate {group} row order")
                require(saved[f"{group}__conditions"].tolist() == [str(r.get("condition", "intact")) for r in meta[group]],
                        f"candidate {group} conditions")
                expected_y = [classes.index(r.get("label", r.get("true", r.get("reference_label")))) for r in meta[group]]
                require(saved[f"{group}__y"].tolist() == expected_y, f"candidate {group} labels")
            for group, values in list(scores.items()) + [("new", new_scores)]:
                for name, independently_computed in values.items():
                    observed = saved[f"{group}__{name}__scores"]
                    require(observed.shape == independently_computed.shape, f"candidate shape {group}/{name}")
                    require(np.isfinite(observed).all(), f"candidate nonfinite {group}/{name}")
                    d = float(np.max(np.abs(observed - independently_computed)))
                    require(d <= 1e-12, f"candidate score discrepancy {group}/{name}: {d}")
                    require(np.array_equal(np.argmax(observed, axis=1), np.argmax(independently_computed, axis=1)),
                            f"candidate argmax discrepancy {group}/{name}")
                    difference = max(difference, d)
                    exact = exact and np.array_equal(observed, independently_computed)
                    score_count += observed.size
        for group, by_model in summary.items():
            for name, by_condition in by_model.items():
                for condition, by_scope in by_condition.items():
                    for scope, oldscope in (("full93", "full93"), ("project18_separate_task", "project18")):
                        compare_metrics(by_scope[scope], authored["regression"]["summary"][group][name][condition][oldscope],
                                        f"candidate/{group}/{name}/{condition}/{scope}")
        for part, otherpart in (("primary_unexposed34", "primary34"), ("previously_exposed2", "previously_exposed2"),
                                ("supplement_all36", "all36_supplementary")):
            for name in ("ref832", "control_oof", "current_teacher_oof"):
                for scope, oldscope in (("full93", "full93"), ("project18_separate_task", "project18")):
                    compare_metrics(new_summary[part][name][scope], authored["new36"]["summary"][otherpart][name][oldscope],
                                    f"candidate/{part}/{name}/{scope}")
        author_primary = all(authored["gates"]["primary"].values())
        author_retention = {key: all(value.values()) for key, value in authored["gates"]["retention"].items()}
        require(author_primary == all(primary_gates.values()), "candidate primary gate disagreement")
        require(author_retention == {key: all(value.values()) for key, value in retention_gates.items()},
                "candidate retention gates disagree")
        require(authored["gates"]["annotation_minimum"] == annotation_minimum, "candidate native minimum gates disagree")
        comparison = {"status": "PASS", "json_sha256": file_sha(candidate_json),
                      "npz_sha256": file_sha(candidate_npz), "score_values_compared": int(score_count),
                      "all_scores_exact": bool(exact), "maximum_absolute_score_difference": difference,
                      "all_top1_equal": True, "all_metrics_and_gate_decisions_agree": True}
    check_pins(cfg)
    require(file_sha(config_path) == initial_config_sha, "configuration changed during audit")
    require(file_sha(training_path) == training_sha, "training receipt changed during audit")
    require(all(file_sha(path) == digest for path, digest in initial_hashes.items()), "checkpoints changed during audit")
    require(all(file_sha(path) == digest for path, digest in teachers["receipts"].items()), "teacher receipts changed during audit")
    result = {"status": "INDEPENDENT_CACHED_EVALUATION_COMPLETE", "verification_passed": True,
              "candidate_frozen_gates_passed": pass_gates,
              "decision": "FROZEN_GATES_PASSED_NOT_PRODUCTION_APPROVED" if pass_gates else "REJECTED",
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "sources": {"config": {"path": str(config_path), "sha256": file_sha(config_path)},
                          "verifier": {"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)},
                          "training_record": {"path": str(training_path), "sha256": training_sha},
                          "checkpoint_hashes": initial_hashes, "input_pins": cfg["pins"], "protected_files": cfg["protected_files"]},
              "classes": classes, "project18_separate_task_classes": [classes[i] for i in target_indices],
              "reference_reproduction": reproduction, "nonhead_bitwise_parent_equal": list(states),
              "independent_teacher_provenance": teachers,
              "regression": summary, "regression_rows": rows_out, "regression_changes": changes,
              "new36": {"summary": new_summary, "rows": new_predictions, "primary_changes": new_changes},
              "gates": {"primary": primary_gates, "retention": retention_gates, "annotation_minimum": annotation_minimum},
              "score_hashes": score_hashes,
              "candidate_evaluation_comparison": comparison,
              "hash_encoding": "SHA256(dtype.str ASCII + compact JSON shape ASCII + C-order raw array bytes)",
              "execution": {"training": 0, "encoder_forward": 0, "service_imports": 0, "learner_imports": 0,
                            "raw_media_reads": 0, "user_session_reads": 0, "cached_affine_model_groups": 7 * 5},
              "limits": ["All evaluation sets are development or training-exposed checks; no new-person claim.",
                         "KSL fold IDs are source buckets, not verified different signers.",
                         "Restricted18 ranks are a separate task, never full93 success.",
                         "No-new-wrong counts are diagnostics; frozen gates use specified aggregate retention.",
                         "No optimizer implementation verification is substituted by this score audit; teacher provenance is independently checked."]}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(destination), "verification_passed": True,
                      "candidate_frozen_gates_passed": pass_gates,
                      "primary34_current_teacher": a["correct"], "primary34_control": b["correct"],
                      "native699": {name: summary["annotation_dev"][name]["intact"]["full93"]["correct"] for name in states}},
                     ensure_ascii=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("PREPARED_CONFIG.json"))
    run(parser.parse_args().config)
