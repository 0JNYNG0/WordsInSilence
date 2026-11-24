"""Bounded, read-only arithmetic diagnosis; no learner call or changed experiment contract."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "DIAGNOSIS_PRECISION.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arrsha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def loadnp(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def delta_stats(d):
    a = np.abs(d)
    ij = np.unravel_index(np.argmax(a), a.shape)
    return {"max_abs": float(a[ij]), "mean_abs": float(a.mean()),
            "p99_abs": float(np.quantile(a, .99)), "above_frozen_1e_5": int((a > 1e-5).sum()),
            "maximum_index": [int(v) for v in ij]}


def run():
    assert not OUT.exists(), "exclusive diagnosis output already exists"
    cfg_path = HERE / "PREPARED_CONFIG.json"
    cfg = json.loads(cfg_path.read_text("utf-8-sig"))
    cfgsha = sha(cfg_path)
    pinned = dict(cfg["pins"])
    pinned.update(cfg["protected_files"])
    assert all(sha(p) == h for p, h in pinned.items()), "input/protected hash mismatch"
    module_path = Path(cfg["paths"]["experiment_module"])
    spec = importlib.util.spec_from_file_location("readonly_teacher_precision_experiment", module_path)
    E = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = E
    spec.loader.exec_module(E)
    _, parent_ck, parent, classes, ref_ck, ref = E.load_models()
    base, y, _, _, teacher, base_schedule, _ = E.load_prior_cache(classes)
    tp = cfg["teacher_preparation"]
    feature = loadnp(tp["base_pooled_path"])
    old = loadnp(tp["historical_teacher_scores_path"])
    meta = json.loads(Path(tp["metadata_path"]).read_text("utf-8-sig"))
    rows = meta["base"]
    ids = [r["id"] for r in rows]
    assert len(base) == len(ids) == 2122 and feature["base_x"].shape == (2122, 64)
    assert feature["classes"].tolist() == old["classes"].tolist() == classes
    assert old["ids"].tolist() == ids
    assert np.array_equal(feature["base_y"], y)
    assert all(r["role"] == "train" for r in rows)
    assert np.array_equal(old["parent"], teacher)
    for k in parent_ck["state_dict"]:
        if not k.startswith("head."):
            assert torch.equal(parent_ck["state_dict"][k], ref_ck["state_dict"][k]), k
    predictions = {
        "b8": E.G.predict_prefix(parent, base),
        "832": E.G.predict_prefix(ref, base),
    }  # Original batch_size=32, fixed original weights. No optimizer or training.
    assert np.array_equal(predictions["b8"]["pooled"], predictions["832"]["pooled"])
    outputs = {}
    for name, ck in (("b8", parent_ck), ("832", ref_ck)):
        p = predictions[name]["pooled"]
        actual = predictions[name]["logits"].astype(np.float64)
        w = ck["state_dict"]["head.weight"].numpy()
        b = ck["state_dict"]["head.bias"].numpy()
        baseline = feature["base_x"].astype(np.float64) @ w.astype(np.float64).T + b.astype(np.float64)
        prefix_affine = p.astype(np.float64) @ w.astype(np.float64).T + b.astype(np.float64)
        cache_part = prefix_affine - baseline
        head_part = actual - prefix_affine
        total = actual - baseline
        # Replay only the same head on the same already-computed pooled rows, original chunks of 32.
        with torch.no_grad():
            replay32 = np.concatenate([
                torch.nn.functional.linear(torch.from_numpy(p[s:s + 32].copy()),
                                           ck["state_dict"]["head.weight"], ck["state_dict"]["head.bias"]).numpy()
                for s in range(0, len(p), 32)])
        assert np.array_equal(replay32, predictions[name]["logits"])
        residual = total - (cache_part + head_part)
        assert np.max(np.abs(residual)) < 1e-12
        maximum = tuple(delta_stats(total)["maximum_index"])
        u = np.finfo(np.float32).eps / 2
        gamma = (65 * u) / (1 - 65 * u)
        round_bound = gamma * (np.abs(p.astype(np.float64)) @ np.abs(w.astype(np.float64)).T
                               + np.abs(b.astype(np.float64)))
        outputs[name] = {
            "total_prefix_float32_vs_cached_feature_float64": delta_stats(total),
            "same_pooled_float32_head_vs_float64_head": delta_stats(head_part),
            "pooled_cache_difference_projected_by_same_float64_head": delta_stats(cache_part),
            "decomposition_residual_max_abs": float(np.max(np.abs(residual))),
            "same_head_same_chunks_bitwise_reproduces_prefix_logits": True,
            "argmax_differences_total": int((actual.argmax(1) != baseline.argmax(1)).sum()),
            "argmax_differences_head_arithmetic_only": int((actual.argmax(1) != prefix_affine.argmax(1)).sum()),
            "frozen_tolerance_exceeding_rows": int((np.abs(total).max(1) > 1e-5).sum()),
            "conservative_float32_65_term_roundoff_bound_violations": int((np.abs(head_part) > round_bound).sum()),
            "at_total_maximum": {"row": maximum[0], "class_index": maximum[1], "id": ids[maximum[0]],
                                 "actual_logit": float(actual[maximum]), "float64_teacher": float(baseline[maximum]),
                                 "total": float(total[maximum]), "cache_part": float(cache_part[maximum]),
                                 "head_arithmetic_part": float(head_part[maximum]),
                                 "float32_spacing_near_logit": float(abs(np.spacing(np.float32(actual[maximum]))))},
            "affine_teacher_sha256": arrsha(baseline), "prefix_pooled_sha256": arrsha(p),
        }
        if name == "b8":
            outputs[name]["stored_teacher_vs_reconstructed_affine"] = delta_stats(teacher - baseline)
    assert sha(cfg_path) == cfgsha
    assert all(sha(p) == h for p, h in pinned.items()), "input/protected changed during diagnosis"
    for model, ck in ((parent, parent_ck), (ref, ref_ck)):
        assert all(torch.equal(v, ck["state_dict"][k]) for k, v in model.state_dict().items())
    result = {
        "status": "NUMERICAL_DIAGNOSIS_COMPLETE_EXPERIMENT_REMAINS_STOPPED",
        "config_sha256": cfgsha, "script_sha256": sha(__file__), "pinned_files_verified_before_after": len(pinned),
        "all_encoder_parameters_buffers_bitwise_equal": True,
        "classes_labels_ids_cache_teacher_provenance_match": True,
        "base_schedule_shape": list(base_schedule.shape),
        "pooled_difference": delta_stats(predictions["b8"]["pooled"] - feature["base_x"]),
        "models": outputs,
        "interpretation": "Observed discrepancy separates exactly into float32 head arithmetic plus tiny existing pooled-cache differences; this does not independently prove the upstream source of every pooled-cache rounding difference. Frozen threshold is unchanged and failed. No accuracy evaluation or main training occurred.",
        "execution": {"main_training": 0, "optimizer_steps": 0, "prefix_predict_calls": 2,
                      "prefix_rows_per_call": 2122, "batch_size": 32, "new_raw_extraction": 0,
                      "camera": 0, "user_data": 0, "changed_model_config_tolerance": 0},
        "sources": {str(module_path): sha(module_path),
                    str(Path(E.G.__file__)): sha(E.G.__file__),
                    **{str(tp[k]): sha(tp[k]) for k in ("base_pooled_path", "historical_teacher_scores_path", "metadata_path", "cached_teacher_path")}},
    }
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(OUT), "models": outputs}, ensure_ascii=True))


if __name__ == "__main__":
    run()
