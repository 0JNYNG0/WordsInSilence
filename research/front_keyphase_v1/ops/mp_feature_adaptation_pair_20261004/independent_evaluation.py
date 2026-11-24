"""Independent final-block evaluation; never imports or executes a learner.

Written by Codex for verification. Recomputes candidate final blocks and verifies fixed full-input/prefix pairs.
No training, raw video, service, or camera.
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
import sys
from pathlib import Path

import numpy as np
import torch

GROUPS = {"annotation_dev": 699, "whole_dev": 74, "public": 32, "mp": 14}
FAMILIES = ("head", "block")
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
        mutable = key.startswith("head.") or (name.startswith("block_fold") and key.startswith("blocks.3."))
        is_buffer = key.endswith(("running_mean", "running_var", "num_batches_tracked"))
        if mutable and not is_buffer:
            continue
        a, b = parent[key], other[key]
        require(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(),
                f"non-head state not bitwise identical: {name}:{key}")
    if name.startswith("block_fold"):
        require(any(parent[k].tobytes() != other[k].tobytes() for k in parent if k.startswith("blocks.3.")),
                f"candidate final block did not change: {name}")


def affine(pooled, state):
    require(pooled.ndim == 2 and pooled.shape[1] == 64 and np.isfinite(pooled).all(),
            "invalid frozen pooled cache")
    scores = (pooled.astype(np.float64) @ state["head.weight"].astype(np.float64).T
              + state["head.bias"].astype(np.float64))
    require(scores.shape == (len(pooled), 93) and np.isfinite(scores).all(), "affine score contract")
    return scores


def construct_model(root, state, architecture):
    sys.path.insert(0, str(root.parent / "sign_model_pilot_v1/src"))
    from pilot.model_masked_v1 import MaskedSignTCN
    model = MaskedSignTCN(width=architecture["width"], dilations=tuple(architecture["dilations"]),
                          kernel_size=architecture["kernel_size"], dropout=architecture["dropout"],
                          n_classes=architecture["n_classes"])
    require(len(model.blocks) == 4, "four-block architecture")
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in state.items()}, strict=True)
    return model.eval()


def unpack_prefixes(values, offsets):
    require(values.dtype == np.float32 and values.ndim == 2 and values.shape[1] == 64, "prefix values")
    require(offsets.dtype == np.int64 and offsets[0] == 0 and offsets[-1] == len(values)
            and np.all(np.diff(offsets) > 0) and np.isfinite(values).all(), "prefix offsets")
    return [values[int(a):int(b)].copy() for a, b in zip(offsets[:-1], offsets[1:])]


def independently_predict_prefix(model, prefixes, state):
    # Independent batching, masking, pooling and head evaluation. No G.predict_prefix/runner imports.
    pooled_parts, logits_parts = [], []
    with torch.no_grad():
        for start in range(0, len(prefixes), 32):
            part = prefixes[start:start+32]
            longest = max(len(p) for p in part)
            tensor = torch.zeros((len(part), 64, longest), dtype=torch.float32)
            mask = torch.zeros((len(part), 1, longest), dtype=torch.float32)
            for index, p in enumerate(part):
                tensor[index, :, :len(p)] = torch.from_numpy(p.T.copy())
                mask[index, 0, :len(p)] = 1
            transformed = model.blocks[3](tensor, mask)
            pooled = (transformed * mask).sum(2) / mask.sum(2).clamp_min(1.0)
            pooled_parts.append(pooled.numpy().copy())
            logits_parts.append(model.head(pooled).numpy().copy())
    pooled = np.concatenate(pooled_parts)
    logits = np.concatenate(logits_parts)
    scores = affine(pooled, state)
    require(np.max(np.abs(scores-logits)) <= 1e-4 and np.array_equal(scores.argmax(1), logits.argmax(1)),
            "independent float32 head/float64 score disagreement")
    return {"pooled": pooled, "logits": logits, "scores": scores}


def legacy_input_hash(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def fixed_full_checks(root, cfg, meta, ready, prefixes, new, states, models):
    """Fixed before results: 2 native, 2 whole, 3 public conditions, 2 MP conditions, 2 KSL labels."""
    from pilot.data import build_channels
    from pilot.web_input import active_span, to_native_like
    dependencies = cfg["evaluation_dependencies"]["coordinate_sources"]
    references = {(r["group"], r["id"]): r for r in dependencies}
    samples = []
    for group in ("annotation_dev", "whole_dev"):
        samples += [(group, 0), (group, len(meta[group])-1)]
    for condition in ("intact", "reviewed_masked", "additional_train"):
        samples.append(("public", next(i for i,r in enumerate(meta["public"]) if r["condition"] == condition)))
    for condition in ("whole_clip", "provided_annotation"):
        samples.append(("mp", next(i for i,r in enumerate(meta["mp"]) if r["condition"] == condition)))
    manifest = read_json(root / "ops/target_input_adaptation_20260928/prepared/MANIFEST.json")
    ksl_specs = {r["id"]:r for r in manifest["rows"]}
    ksl_indices = [next(i for i in range(36) if new["y"][i] == label and not new["exposed"][i])
                   for label in sorted(set(new["y"].tolist()))]
    samples += [("ksl36", i) for i in ksl_indices]
    output = []
    for group, index in samples:
        if group == "ksl36":
            sample_id = str(new["ids"][index]); spec = ksl_specs[sample_id]
            require(file_sha(spec["input_path"]) == spec["input_file_sha256"], "KSL full input hash")
            x = read_arrays(spec["input_path"])["x"]
            prefix = prefixes["ksl36"][index]
        else:
            spec = meta[group][index]; sample_id = spec["id"]; condition = spec.get("condition", "intact")
            source_id = sample_id.split("::")[0]
            source = references[group, source_id]
            require(file_sha(source["path"]) == source["sha256"], "full input coordinate source hash")
            raw = read_arrays(source["path"])
            if group == "whole_dev":
                x = raw["x"]
            elif group == "annotation_dev" or (group == "public" and condition == "additional_train"):
                x = build_channels(raw["xy_native"], raw["coordinate_present"], raw["timestamp_nominal_us"])["x"]
            else:
                xy, usable, ts = raw["xy_px"], raw["usable"], raw["timestamp_us"]
                if group == "mp" and condition == "provided_annotation":
                    a,b = source["annotation_rows"]; xy,usable,ts = xy[a:b+1],usable[a:b+1],ts[a:b+1]
                native,present = to_native_like(np.asarray(xy,np.float32),np.asarray(usable,bool))
                span = active_span(present); require(span is not None,"full input empty active span")
                a,b = span; native,present,ts = native[a:b+1],present[a:b+1],ts[a:b+1]
                if group == "public" and condition == "reviewed_masked":
                    side = source["reviewed_non_sign_hand"]; require(side in ("left","right"),"reviewed hand side")
                    sl = slice(25,46) if side == "left" else slice(46,67)
                    native = native.copy(); present = present.copy(); native[:,sl,:] = 0; present[:,sl] = False
                x = build_channels(native,present,ts)["x"]
            require(legacy_input_hash(x) == ready["input_array_sha256"][group][index],
                    f"independent reconstructed input differs: {group}/{sample_id}")
            prefix = prefixes[group][index]
        require(x.dtype == np.float32 and x.shape == (len(prefix),280) and np.isfinite(x).all(),"full sample input contract")
        with torch.no_grad():
            raw_tensor = torch.from_numpy(x.copy()).unsqueeze(0)
            raw_mask = torch.ones((1,len(x)),dtype=torch.float32)
            one_mask = raw_mask.unsqueeze(1)
            shared = models["ref832"].proj(raw_tensor.transpose(1,2)*one_mask)*one_mask
            for block in models["ref832"].blocks[:3]: shared = block(shared,one_mask)
            fresh = shared.squeeze(0).T.contiguous().numpy()
            prefix_error = float(np.max(np.abs(fresh-prefix)))
            require(prefix_error <= 1e-5,"independent front-block prefix reconstruction differs")
            for name,model in models.items():
                full = model(raw_tensor,raw_mask).numpy()
                checked = independently_predict_prefix(model,[prefix],states[name])
                err = float(np.max(np.abs(full-checked["logits"])))
                score_error = float(np.max(np.abs(full-checked["scores"])))
                require(err <= 1e-4 and score_error <= 1e-4 and np.array_equal(full.argmax(1),checked["scores"].argmax(1)),
                        f"full vs own-prefix forward differs: {group}/{sample_id}/{name}")
                output.append({"group":group,"index":int(index),"id":sample_id,"model":name,
                               "input_hash":legacy_input_hash(x),"front_prefix_max_abs":prefix_error,
                               "full_vs_prefix_logit_max_abs":err,"full_vs_affine_max_abs":score_error,
                               "top1_equal":True})
    require(len(output) == 77,"expected fixed 11 inputs times 7 models")
    return output


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
    prefix_path = run_dir / "eval_prefixes.npz"
    ready_path = run_dir / "PREFIX_READY.json"
    preflight_path = run_dir / "PREFLIGHT.json"
    ready, preflight = read_json(ready_path), read_json(preflight_path)
    require(preflight["config_sha256"] == initial_config_sha, "preflight configuration")
    require(file_sha(prefix_path) == ready["eval_prefixes_sha256"] == preflight["files"]["eval_prefixes.npz"], "prefix cache hash")
    require(file_sha(ready_path) == preflight["files"]["PREFIX_READY.json"], "prefix receipt hash")
    prefix_cache = read_arrays(prefix_path)
    prefixes = {g: unpack_prefixes(prefix_cache[g+"__prefix"],prefix_cache[g+"__offsets"]) for g in GROUPS}
    prefixes["ksl36"] = unpack_prefixes(new["prefix"],new["offsets"])
    initial_prefix_hashes = {str(p): file_sha(p) for p in (prefix_path, ready_path, preflight_path)}
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
    torch.set_num_threads(2)
    architecture = torch.load(paths["parent"],map_location="cpu",weights_only=True)
    models = {name:construct_model(root,state,architecture) for name,state in states.items()}
    own_prefix_results = {g:{name:independently_predict_prefix(model,prefixes[g],states[name])
                             for name,model in models.items()} for g in list(GROUPS)+["ksl36"]}
    full_prefix_checks = fixed_full_checks(root,cfg,meta,ready,prefixes,new,states,models)
    scores, summary, rows_out, changes, reproduction = {}, {}, [], {}, {}
    known_rows = {(r["group"], r["id"], r.get("condition", "intact")): r for r in known["rows"]}
    for group, count in GROUPS.items():
        rows = meta[group]
        require(len(rows) == count, f"{group} metadata count")
        labels = [r.get("label", r.get("true", r.get("reference_label"))) for r in rows]
        y = np.asarray([classes.index(label) for label in labels], dtype=np.int64)
        require(len(prefixes[group]) == count and prefix_cache[group+"__ids"].tolist() == [r["id"] for r in rows], "prefix row order")
        require(prefix_cache[group+"__conditions"].tolist() == [r.get("condition","intact") for r in rows], "prefix condition order")
        pooled = own_prefix_results[group]["ref832"]["pooled"]
        scores[group] = {name:value["scores"] for name,value in own_prefix_results[group].items()}
        maximum = float(np.max(np.abs(scores[group]["ref832"] - cached[group + "__control__scores"])))
        require(maximum <= 1e-4, f"{group} ref832 score mismatch {maximum}")
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
            candidate, head = f"block_fold{fold}", f"head_fold{fold}"
            for reference in ("ref832", head):
                changes[group][f"{candidate}_vs_{reference}"] = changed_rows(scores[group][reference], scores[group][candidate], y, rows, classes)
    require(summary["annotation_dev"]["ref832"]["intact"]["full93"]["correct"] == 691,
            "current reference is not expected 691/699")

    y = new["y"].astype(np.int64)
    fold_ids, exposed = new["folds"], new["exposed"]
    require(y.shape == (36,) and exposed.dtype == np.bool_ and int(exposed.sum()) == 2, "KSL36/exposed contract")
    require(set(fold_ids.tolist()) == {0, 1, 2}, "KSL folds")
    new_rows = [{"id": str(name), "condition": "ksl_original_excluded_development"} for name in new["ids"]]
    new_scores = {name:value["scores"] for name,value in own_prefix_results["ksl36"].items()}
    error = float(np.max(np.abs(new_scores["ref832"] - new["logits_cached"])))
    require(error <= 1e-5, f"KSL ref832 cache mismatch: {error}")
    reproduction["ksl36"] = {"max_score_abs_difference": error, "pooled_sha256": array_sha(own_prefix_results["ksl36"]["ref832"]["pooled"])}
    for family in FAMILIES:
        new_scores[family + "_oof"] = np.stack([new_scores[f"{family}_fold{int(f)}"][i] for i, f in enumerate(fold_ids)])
    selections = {"primary_unexposed34": ~exposed, "previously_exposed2": exposed,
                  "supplement_all36": np.ones(36, dtype=bool)}
    selections.update({f"heldout_fold{fold}": fold_ids == fold for fold in range(3)})
    new_summary = {part: {name: summarize(value[select], [r for r, keep in zip(new_rows, select) if keep], y[select], classes, target_indices)
                          for name, value in new_scores.items()} for part, select in selections.items()}
    primary = new_summary["primary_unexposed34"]
    a, b = primary["block_oof"]["full93"], primary["head_oof"]["full93"]
    study, eat = "\uacf5\ubd80", "\uba39\ub2e4"
    require(a["per_class"][study]["n"] == 18 and a["per_class"][eat]["n"] == 16, "primary class counts")
    primary_gates = {"total_at_least31": a["correct"] >= 31,
                     "total_ge_head": a["correct"] >= b["correct"],
                     "study_at_least17": a["per_class"][study]["correct"] >= 17,
                     "eat_at_least14": a["per_class"][eat]["correct"] >= 14,
                     "each_class_ge_head": all(a["per_class"][label]["correct"] >= b["per_class"][label]["correct"] for label in (study, eat))}
    retention_gates = {}
    for fold in range(3):
        for reference in ("ref832", f"head_fold{fold}"):
            retention_gates[f"block_fold{fold}_vs_{reference}"] = retention(summary, f"block_fold{fold}", reference)
    pass_gates = all(primary_gates.values()) and all(all(g.values()) for g in retention_gates.values())
    new_predictions = []
    for i, row in enumerate(new_rows):
        new_predictions.append({"id": row["id"], "fold": int(fold_ids[i]), "previously_exposed": bool(exposed[i]),
                                "true": classes[int(y[i])], "models": {
            name: {"full93": row_prediction(value[i], int(y[i]), classes, range(93)),
                   "project18_separate_task": row_prediction(value[i], int(y[i]), classes, target_indices)}
            for name, value in new_scores.items()}})
    new_changes = {}
    for reference in ("ref832", "head_oof"):
        take = ~exposed
        new_changes[f"block_oof_vs_{reference}"] = changed_rows(new_scores[reference][take], new_scores["block_oof"][take],
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
            for group, values in list(scores.items()) + [("new", new_scores)]:
                for name, independently_computed in values.items():
                    observed = saved[f"{group}__{name}__scores"]
                    require(observed.shape == independently_computed.shape, f"candidate shape {group}/{name}")
                    require(np.isfinite(observed).all(), f"candidate nonfinite {group}/{name}")
                    d = float(np.max(np.abs(observed - independently_computed)))
                    require(d <= 1e-4, f"candidate score discrepancy {group}/{name}: {d}")
                    require(np.array_equal(np.argmax(observed, axis=1), np.argmax(independently_computed, axis=1)),
                            f"candidate argmax discrepancy {group}/{name}")
                    difference = max(difference, d)
                    exact = exact and np.array_equal(observed, independently_computed)
                    score_count += observed.size
            for group, by_model in own_prefix_results.items():
                keygroup = "new" if group == "ksl36" else group
                for name,value in by_model.items():
                    saved_pool = saved[f"{keygroup}__{name}__pooled"]
                    require(saved_pool.shape == value["pooled"].shape and np.max(np.abs(saved_pool-value["pooled"])) <= 1e-5,
                            f"candidate own-block pooled mismatch: {group}/{name}")
        for group, by_model in summary.items():
            for name, by_condition in by_model.items():
                for condition, by_scope in by_condition.items():
                    for scope, oldscope in (("full93", "full93"), ("project18_separate_task", "project18")):
                        compare_metrics(by_scope[scope], authored["regression"]["summary"][group][name][condition][oldscope],
                                        f"candidate/{group}/{name}/{condition}/{scope}")
        for part, otherpart in (("primary_unexposed34", "primary34"), ("previously_exposed2", "previously_exposed00"),
                                ("supplement_all36", "all36_supplementary")):
            for name in ("ref832", "head_oof", "block_oof"):
                for scope, oldscope in (("full93", "full93"), ("project18_separate_task", "project18")):
                    compare_metrics(new_summary[part][name][scope], authored["new36"]["summary"][otherpart][name][oldscope],
                                    f"candidate/{part}/{name}/{scope}")
        author_primary = all(authored["gates"]["primary"].values())
        author_retention = {key: all(value.values()) for key, value in authored["gates"]["retention"].items()}
        require(author_primary == all(primary_gates.values()), "candidate primary gate disagreement")
        require(author_retention == {key: all(value.values()) for key, value in retention_gates.items()},
                "candidate retention gates disagree")
        comparison = {"status": "PASS", "json_sha256": file_sha(candidate_json),
                      "npz_sha256": file_sha(candidate_npz), "score_values_compared": int(score_count),
                      "all_scores_exact": bool(exact), "maximum_absolute_score_difference": difference,
                      "all_top1_equal": True, "all_metrics_and_gate_decisions_agree": True}
    check_pins(cfg)
    require(file_sha(config_path) == initial_config_sha, "configuration changed during audit")
    require(file_sha(training_path) == training_sha, "training receipt changed during audit")
    require(all(file_sha(p) == h for p,h in initial_prefix_hashes.items()), "preflight prefix artifacts changed during audit")
    require(all(file_sha(path) == digest for path, digest in initial_hashes.items()), "checkpoints changed during audit")
    result = {"status": "INDEPENDENT_FINAL_BLOCK_EVALUATION_COMPLETE", "verification_passed": True,
              "candidate_frozen_gates_passed": pass_gates,
              "decision": "FROZEN_GATES_PASSED_NOT_PRODUCTION_APPROVED" if pass_gates else "REJECTED",
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "sources": {"config": {"path": str(config_path), "sha256": file_sha(config_path)},
                          "verifier": {"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)},
                          "training_record": {"path": str(training_path), "sha256": training_sha},
                          "checkpoint_hashes": initial_hashes, "input_pins": cfg["pins"], "protected_files": cfg["protected_files"]},
              "classes": classes, "project18_separate_task_classes": [classes[i] for i in target_indices],
              "reference_reproduction": reproduction, "frozen_front_and_all_buffers_bitwise_parent_equal": list(states),
              "full_prefix_checks": full_prefix_checks, "prefix_cache_sources": initial_prefix_hashes,
              "regression": summary, "regression_rows": rows_out, "regression_changes": changes,
              "new36": {"summary": new_summary, "rows": new_predictions, "primary_changes": new_changes},
              "gates": {"primary": primary_gates, "retention": retention_gates},
              "score_hashes": score_hashes,
              "candidate_evaluation_comparison": comparison,
              "hash_encoding": "SHA256(dtype.str ASCII + compact JSON shape ASCII + C-order raw array bytes)",
              "execution": {"training": 0, "full_model_forwards": 77, "shared_front_sample_reconstructions": 11,
                            "own_final_block_model_groups": 7 * 5, "service_imports": 0, "learner_imports": 0,
                            "raw_media_reads": 0, "user_session_reads": 0},
              "limits": ["All evaluation sets are development or training-exposed checks; no new-person claim.",
                         "KSL fold IDs are source buckets, not verified different signers.",
                         "Restricted18 ranks are a separate task, never full93 success.",
                         "No-new-wrong counts are diagnostics; frozen gates use specified aggregate retention.",
                         "No optimizer implementation verification is substituted by this score audit; frozen front and all buffers verified."]}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(destination), "verification_passed": True,
                      "candidate_frozen_gates_passed": pass_gates,
                      "primary34_block": a["correct"], "primary34_head": b["correct"],
                      "native699": {name: summary["annotation_dev"][name]["intact"]["full93"]["correct"] for name in states}},
                     ensure_ascii=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("PREPARED_CONFIG.json"))
    run(parser.parse_args().config)
