"""Head-only control vs last-block(blocks.3)+head candidate on the same MP113 + KSL-fold extra data
(FROZEN_PLAN.md).

    python run_pair.py preflight --config PREPARED_CONFIG.json   # pins, inputs, eval prefix cache, 832 reproduction
    python run_pair.py train     --config PREPARED_CONFIG.json   # needs PREFLIGHT + TEST_RECEIPT; 6 x 660 steps
    python run_pair.py evaluate  --config PREPARED_CONFIG.json   # needs TRAINING.json; prefix -> each arm's last block

Old modules are loaded read-only: gradient_projection_pair_20261004/run_pair.py as R (only its
load_inputs / input_summary / verify_hashes / required_pins helpers), R.P.load_experiment() as E, E.T, E.G.
R.preflight/train/evaluate and the projection learner are never called.
Writes only to FRONT/runs/mp_feature_adaptation_pair_v1 (create-exclusive) and FAILED_*.json.
Nothing is promoted; no camera/user session is read.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FRONT = Path(__file__).resolve().parents[2]
OLD_OPS = FRONT / "ops" / "gradient_projection_pair_20261004"
RUN_DIR = FRONT / "runs" / "mp_feature_adaptation_pair_v1"
OLD_SUPPORT_RUN = FRONT / "runs" / "mp_support_addition_v1"
RECEIPT = HERE / "TEST_RECEIPT.json"
NEW_FILES = ("feature_train.py", "run_pair.py", "test_feature.py")

STEPS, EXTRA_W, N_FOLDS, N_BLOCKS, THREADS = 660, 0.25, 3, 4, 2
FOLD_TRAIN, FOLD_HELD = (22, 26, 24), (14, 10, 12)
N_SUPPORT = 113
TRAIN_KW = dict(learning_rate=1e-3, weight_decay=1e-3, anchor_coefficient=1e-2, output_coefficient=1.0)
GROUP_SIZES = {"annotation_dev": 699, "whole_dev": 74, "public": 32, "mp": 14}
HEAD = "head."
LAST = f"blocks.{N_BLOCKS - 1}."
HEADS = [f"head_fold{f}" for f in range(N_FOLDS)]
BLOCKS = [f"block_fold{f}" for f in range(N_FOLDS)]
ARMS = HEADS + BLOCKS
REF = "ref832"
TOL_STEP, TOL_832, TOL_NEW, TOL_AFFINE, TOL_CACHED_AFFINE = 1e-6, 1e-4, 1e-5, 1e-4, 1e-6
GATE_PRIMARY_TOTAL = 31
PREFLIGHT_FILES = sorted(["PREFLIGHT.json", "PREFIX_READY.json", "eval_prefixes.npz"])
PREFLIGHT_STATUS = "PREFLIGHT_PASSED_PREFIX_CACHE_FROZEN_NO_TRAINING"
TRAINED_STATUS = "TRAINING_COMPLETE_CHECKPOINTS_FROZEN_BEFORE_EVALUATION"
PASS = "PASS_LOCAL_DEVELOPMENT_GATES_NOT_PRODUCTION_READY"
REJECT = "REJECTED_BY_PREDECLARED_GATES"


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def need(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def maxabs(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    need(a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _load(name, path):
    path = Path(path).resolve()
    mod = sys.modules.get(name)
    if mod is not None:
        need(Path(mod.__file__).resolve() == path, f"{name} already bound to {mod.__file__}")
        return mod
    need(path.is_file(), f"module not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def load_modules():
    """(R, E, FT): old gradient_projection run_pair (read-only helpers), old experiment, new learner."""
    need(not os.environ.get("TIA_PRIOR_GENERATED"), "TIA_PRIOR_GENERATED override is not allowed")
    R = _load("mfa_old_gradient_run_pair", OLD_OPS / "run_pair.py")
    need(Path(R.FRONT).resolve() == FRONT, "old run_pair FRONT differs")
    E = R.P.load_experiment()
    need(Path(E.T.PRIOR_GENERATED).resolve() == (E.PRIOR / "generated_v1.py").resolve(), "generated_v1 override")
    FT = _load("mfa_feature_train", HERE / "feature_train.py")
    return R, E, FT


def _rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(FRONT).as_posix()
    except ValueError:
        return str(p)


def code_hashes():
    R, E, _ = load_modules()
    files = [HERE / n for n in NEW_FILES] + [R.__file__, R.P.__file__, E.__file__, E.T.__file__,
                                             E.T.PRIOR_GENERATED, E.PRIOR / "local_io.py"]
    return {_rel(p): sha_file(p) for p in files}


# --------------------------------------------------------------------------- #
# config / pins / receipts
# --------------------------------------------------------------------------- #
def load_cfg(path):
    R, E, FT = load_modules()
    raw = Path(path).read_bytes()
    cfg = json.loads(raw.decode("utf-8-sig"))
    need({"root", "out_dir", "ops_dir", "pins", "protected_files"} <= set(cfg), "config keys missing")
    for k, want in (("root", FRONT), ("ops_dir", HERE), ("out_dir", RUN_DIR)):
        need(Path(cfg[k]).is_absolute() and Path(cfg[k]).resolve() == want.resolve(), f"config {k} != {want}")
    cfg["pins"] = {R.norm_path(p): h for p, h in cfg["pins"].items()}
    cfg["protected_files"] = {R.norm_path(p): h for p, h in cfg["protected_files"].items()}
    return R, E, FT, cfg, RUN_DIR.resolve(), hashlib.sha256(raw).hexdigest()


def required_pins(R, E, pol, neural_path):
    ps = R.required_pins(E, pol) + [
        OLD_OPS / "run_pair.py", R.P.__file__, E.__file__, E.T.__file__, E.T.PRIOR_GENERATED,
        E.PRIOR / "local_io.py", E.POLICY, E.CACHE_RECEIPT, E.META, pol["paths"]["metadata"], neural_path]
    return sorted({R.norm_path(p) for p in ps})


def check_required(R, E, cfg, pol, neural_path):
    missing = [p for p in required_pins(R, E, pol, neural_path) if p not in cfg["pins"]]
    need(not missing, f"config pins lack required sources/inputs: {missing}")


def verify_receipt(E):
    need(RECEIPT.is_file(), "TEST_RECEIPT.json missing; run test_feature.py first")
    r = E.read(RECEIPT)
    need(r.get("status") == "PASSED" and r.get("synthetic_only") is True, "TEST_RECEIPT status is not PASSED")
    need(r.get("code_hashes") == code_hashes(), "TEST_RECEIPT code hashes differ from current code")
    return r


def verify_preflight(E, out, cfg_sha):
    pre = E.read(out / "PREFLIGHT.json")
    need(pre["status"] == PREFLIGHT_STATUS and pre["config_sha256"] == cfg_sha, "PREFLIGHT status/config differ")
    need(pre["code_hashes"] == code_hashes(), "code changed since preflight")
    for n in ("PREFIX_READY.json", "eval_prefixes.npz"):
        need(E.sha(out / n) == pre["files"][n], f"preflight file changed: {n}")
    ready = E.read(out / "PREFIX_READY.json")
    need(ready["eval_prefixes_sha256"] == pre["files"]["eval_prefixes.npz"], "PREFIX_READY cache sha differs")
    return pre, ready


def check_four_blocks(G, model):
    G._inspect_model(model)
    need(len(model.blocks) == N_BLOCKS, f"expected exactly {N_BLOCKS} blocks, got {len(model.blocks)}")
    need(any(n.startswith(LAST) for n, _ in model.named_parameters()), f"no {LAST}* parameters")


def check_inputs(R, S):
    s = R.input_summary(S)
    need(s["classes"] == 93 and s["base"] == 2122 and s["context"] == 224 and s["old_schedule_shape"] == [STEPS, 16]
         and s["new36"] == 36 and s["exposed"] == 2 and len(S["sup_ids"]) == N_SUPPORT, f"input counts differ: {s}")
    need(R.STEPS == STEPS and R.EXTRA_W == EXTRA_W and tuple(R.FOLD_TRAIN) == FOLD_TRAIN
         and tuple(R.FOLD_HELD) == FOLD_HELD, "old runner constants differ from the frozen plan")
    for f, d in S["folds"].items():
        need(d["xsch"].shape == (STEPS, 8) and len(d["rows"]) == FOLD_TRAIN[f] and len(d["held"]) == FOLD_HELD[f]
             and len(d["xp"]) == FOLD_TRAIN[f] + N_SUPPORT and d["n_ksl"] == FOLD_TRAIN[f], f"fold {f} sizes")
    return s


def check_recipe(E):
    need(E.T.TRAIN_KW == TRAIN_KW, f"T.TRAIN_KW differs from frozen plan: {E.T.TRAIN_KW}")
    need(E.PARENT_SHA.startswith("b8d6045c"), "parent is not b8d6045c")


def ref_model(parent, ref_sd):
    m = copy.deepcopy(parent)
    m.load_state_dict(ref_sd, strict=True)
    return m.eval()


def row_key(E, r):
    return (str(r["id"]), r.get("condition", "intact"), r.get("signer_id"), E.label_of(r))


def prefix_scores(G, model, prefixes):
    """Runtime engine G.predict_prefix (last block -> masked mean -> head, float32) plus the
    standardized float64 affine score pooled64 @ W64.T + b64 from the same model's head."""
    pr = G.predict_prefix(model, prefixes)
    need(pr["pooled"].dtype == np.float32 and pr["logits"].dtype == np.float32, "predict_prefix dtype")
    w = model.head.weight.detach().numpy().astype(np.float64)
    b = model.head.bias.detach().numpy().astype(np.float64)
    scores = pr["pooled"].astype(np.float64) @ w.T + b
    d = maxabs(scores, pr["logits"])
    need(d <= TOL_AFFINE, f"float32 logits vs float64 affine scores differ ({d})")
    need(np.array_equal(scores.argmax(1), pr["logits"].argmax(1)), "float32/float64 argmax differ")
    return dict(pooled=pr["pooled"], logits=pr["logits"], scores=scores, logits_vs_scores=d)


def state_diff(a, b):
    if set(a) != set(b) or any(a[k].shape != b[k].shape or a[k].dtype != b[k].dtype for k in a):
        return {"keys_shapes_dtypes_equal": False}
    changed = sorted(k for k in a if not torch.equal(a[k], b[k]))
    return {"keys_shapes_dtypes_equal": True, "bitwise_equal_all": not changed, "changed_keys": changed,
            "max_abs_all": max([float((a[k].double() - b[k].double()).abs().max()) for k in changed] or [0.0])}


def loss_diff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return {"shape_equal": False}
    m = ~np.isnan(a)
    return {"shape_equal": True, "nan_pattern_equal": bool(np.array_equal(m, ~np.isnan(b))),
            "bitwise_equal": bool(np.array_equal(a, b, equal_nan=True)),
            "max_abs": float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0}


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def preflight(R, E, FT, cfg, out, cfg_sha):
    need(not out.exists(), f"refusing: {out} exists")
    hashes0 = code_hashes()
    R.verify_hashes(E, cfg)
    check_recipe(E)
    G = E.G
    S = R.load_inputs(E, cfg)
    parent, classes, pol = S["parent"], S["classes"], S["pol"]
    target_idx, neural_path = E.project18(classes)
    check_required(R, E, cfg, pol, neural_path)
    summary = check_inputs(R, S)
    check_four_blocks(G, parent)
    m832 = ref_model(parent, S["ref"]["state_dict"])
    io = E.load_io()

    # ---- 819 evaluation inputs from provided/public coordinates (no MediaPipe, no video)
    groups, meta_io = io.evaluation_inputs(pol)
    meta = E.read(E.META)
    for g, n in GROUP_SIZES.items():
        need(len(groups[g]) == len(meta_io[g]) == len(meta[g]) == n, f"{g}: expected {n} rows")
        need([row_key(E, r) for r in meta_io[g]] == [row_key(E, r) for r in meta[g]], f"{g}: metadata differs from META")
    need(list(E.GROUPS) == list(GROUP_SIZES), "group order differs")
    prefixes, input_hashes = {}, {}
    for g in E.GROUPS:
        input_hashes[g] = [R.arr_sha(x) for x in groups[g]]
        prefixes[g] = G.extract_prefix(parent, groups[g])
        need(all(p.dtype == np.float32 and len(p) == len(x) for p, x in zip(prefixes[g], groups[g])), f"{g}: prefix")
    del groups

    # ---- 832 reproduction from the new prefixes (before any training)
    w = S["ref"]["state_dict"]["head.weight"].numpy().astype(np.float64)
    b = S["ref"]["state_dict"]["head.bias"].numpy().astype(np.float64)
    checks, ref_out = {}, {}
    with np.load(E.OLD_RUN / "evaluation_arrays.npz", allow_pickle=False) as ev:
        need([str(c) for c in ev["classes"].tolist()] == classes, "evaluation_arrays classes")
        for g in E.GROUPS:
            cs, cp = ev[g + "__control__scores"], ev[g + "__control__pooled"]
            r = prefix_scores(G, m832, prefixes[g])
            c = {"new_scores_vs_cached_scores": maxabs(r["scores"], cs),
                 "new_pooled_vs_cached_pooled": maxabs(r["pooled"], cp),
                 "cached_pooled_affine_vs_cached_scores": maxabs(cp.astype(np.float64) @ w.T + b, cs),
                 "float32_logits_vs_float64_scores": r["logits_vs_scores"],
                 "argmax_full93_equal": bool(np.array_equal(r["scores"].argmax(1), cs.argmax(1))),
                 "argmax_project18_equal": bool(np.array_equal(r["scores"][:, target_idx].argmax(1),
                                                               cs[:, target_idx].argmax(1)))}
            need(c["new_scores_vs_cached_scores"] <= TOL_832 and c["argmax_full93_equal"]
                 and c["argmax_project18_equal"] and c["cached_pooled_affine_vs_cached_scores"] <= TOL_CACHED_AFFINE,
                 f"{g}: 832 not reproduced from new prefixes {c}")
            checks[g] = c
            ref_out[g] = r
    new = S["new"]
    r = prefix_scores(G, m832, new["prefixes"])
    checks["ksl36"] = {"scores_vs_cached_logits": maxabs(r["scores"], new["logits_cached"]),
                       "pooled_vs_pooled_832": maxabs(r["pooled"], new["pooled_832"]),
                       "argmax_equal": bool(np.array_equal(r["scores"].argmax(1), new["logits_cached"].argmax(1)))}
    need(checks["ksl36"]["scores_vs_cached_logits"] <= TOL_NEW and checks["ksl36"]["pooled_vs_pooled_832"] <= TOL_NEW
         and checks["ksl36"]["argmax_equal"], f"KSL36 832 not reproduced {checks['ksl36']}")
    need(all(torch.equal(v, S["psd0"][k]) for k, v in parent.state_dict().items()), "parent modified")
    R.verify_hashes(E, cfg)
    need(code_hashes() == hashes0, "code changed during preflight")

    # ---- freeze (nothing written before this point)
    out.mkdir(parents=False, exist_ok=False)
    arrays = {"classes": np.array(classes), "project18_idx": np.array(target_idx, dtype=np.int64)}
    for g in E.GROUPS:
        v, o = io.pack(prefixes[g])
        need(all(same(x, y) for x, y in zip(io.unpack(v, o), prefixes[g])), f"{g}: pack round trip")
        arrays.update({f"{g}__prefix": v, f"{g}__offsets": o,
                       f"{g}__ids": np.array([str(x["id"]) for x in meta[g]]),
                       f"{g}__conditions": np.array([x.get("condition", "intact") for x in meta[g]]),
                       f"{g}__ref832_pooled": ref_out[g]["pooled"], f"{g}__ref832_logits": ref_out[g]["logits"],
                       f"{g}__ref832_scores": ref_out[g]["scores"]})
    E.save_npz(out / "eval_prefixes.npz", arrays)
    with np.load(out / "eval_prefixes.npz", allow_pickle=False) as z:
        need(all(same(z[k], v) for k, v in arrays.items()), "eval_prefixes.npz reload differs")
    cache_sha = E.sha(out / "eval_prefixes.npz")
    E.write_json(out / "PREFIX_READY.json", {
        "status": "EVAL_PREFIX_CACHE_READY", "created_utc": now().isoformat(), "eval_prefixes_sha256": cache_sha,
        "construction": "local_io.evaluation_inputs(policy) -> generated_v1.extract_prefix(parent) "
                        "(proj + blocks.0..2, one unpadded clip at a time); no MediaPipe/video re-extraction",
        "valid_only_if": "proj and blocks.0..2 bitwise equal parent b8d6045c",
        "group_order": E.GROUPS, "group_sizes": GROUP_SIZES,
        "ids": {g: [str(x["id"]) for x in meta[g]] for g in E.GROUPS},
        "conditions": {g: [x.get("condition", "intact") for x in meta[g]] for g in E.GROUPS},
        "signers": {g: [x.get("signer_id") for x in meta[g]] for g in E.GROUPS},
        "input_array_sha256": input_hashes, "classes": classes, "project18_idx": target_idx,
        "project18_classes": [classes[i] for i in target_idx], "project18_module_sha256": E.sha(neural_path),
        "metadata": {"path": str(E.META), "sha256": E.sha(E.META),
                     "policy_metadata_path": str(pol["paths"]["metadata"]),
                     "policy_metadata_sha256": E.sha(pol["paths"]["metadata"])},
        "source_sha256": {_rel(p): E.sha(p) for p in (E.PRIOR / "local_io.py", E.T.PRIOR_GENERATED, E.POLICY)},
        "parent_sha256": E.PARENT_SHA, "reference_sha256": E.REF_SHA,
        "ref832_reproduction": checks,
        "tolerances": {"ref832_vs_cached_scores": TOL_832, "ksl36": TOL_NEW, "logits_vs_affine": TOL_AFFINE,
                       "cached_pooled_affine": TOL_CACHED_AFFINE},
        "evaluation_labels_used_for_training": False})
    E.write_json(out / "PREFLIGHT.json", {
        "status": PREFLIGHT_STATUS, "created_utc": now().isoformat(), "config_sha256": cfg_sha,
        "code_hashes": hashes0, "inputs": E.plain(summary),
        "files": {"PREFIX_READY.json": E.sha(out / "PREFIX_READY.json"), "eval_prefixes.npz": cache_sha},
        "required_pins": required_pins(R, E, pol, neural_path), "pins_verified_before_and_after": True,
        "no_training_performed": True, "torch": torch.__version__, "threads": torch.get_num_threads()})
    print("preflight passed:", out)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def check_arm(r, parent, psd, S, d, block_arm):
    sd, losses = r["state_dict"], np.asarray(r["losses"])
    allowed = (HEAD, LAST) if block_arm else (HEAD,)
    pnames = [n for n, _ in parent.named_parameters()]
    bnames = [n for n, _ in parent.named_buffers() if n in psd]
    c = {"updates": int(r["updates"]) == STEPS,
         "losses_shape_finite": losses.shape == (STEPS, 6) and bool(np.isfinite(losses).all()),
         "old_schedule_identical": same(r["base_indices"], S["cache"][5]) and same(r["context_indices"], S["cache"][6]),
         "extra_schedule_identical": same(r["extra_indices"], d["xsch"]),
         "keys_dtype_shape_equal_parent": set(sd) == set(psd)
                                          and all(sd[k].dtype == psd[k].dtype and sd[k].shape == psd[k].shape for k in psd),
         "trainable_names_exact": sorted(r["trainable_names"]) == sorted(n for n in pnames if n.startswith(allowed)),
         "untrained_bitwise_equal_parent": all(torch.equal(sd[k], psd[k]) for k in psd if not k.startswith(allowed)),
         "all_buffers_bitwise_equal_parent": all(torch.equal(sd[k], psd[k]) for k in bnames),
         "head_changed": any(not torch.equal(sd[k], psd[k]) for k in psd if k.startswith(HEAD)),
         "state_finite": all(bool(torch.isfinite(v).all()) for v in sd.values() if v.is_floating_point()),
         "frozen_parameters_unchanged": bool(r["frozen_parameters_unchanged"]),
         "buffers_unchanged": bool(r["buffers_unchanged"])}
    if block_arm:
        c["last_block_params_changed"] = any(not torch.equal(sd[n], psd[n]) for n in pnames if n.startswith(LAST))
    else:
        c["last_block_bitwise_equal_parent"] = all(torch.equal(sd[k], psd[k]) for k in psd if k.startswith(LAST))
    return c


def first_step_parity(E, FT, S):
    T, G, c = E.T, E.G, S["cache"]
    one = (c[5][:1].copy(), c[6][:1].copy())
    res = {}
    for f, d in S["folds"].items():
        kw = dict(extra_prefixes=d["xp"], extra_y=d["xy"], extra_indices=d["xsch"][:1].copy(), extra_weight=EXTRA_W,
                  **T.TRAIN_KW)
        a = T.train_arm(S["parent"], *c[:5], *one, arm=f"parity_old_fold{f}", **kw)
        b = FT.train_arm(S["parent"], *c[:5], *one, arm=f"parity_clone_fold{f}", train_last_block=False, G=G, **kw)
        sdf, ldf = state_diff(a["state_dict"], b["state_dict"]), loss_diff(a["losses"], b["losses"])
        res[f"fold{f}"] = {"state": sdf, "losses": ldf}
        need(sdf["keys_shapes_dtypes_equal"] and sdf["max_abs_all"] <= TOL_STEP and ldf["shape_equal"]
             and ldf["nan_pattern_equal"] and ldf["max_abs"] <= TOL_STEP, f"first-step parity failed fold {f}: {res}")
    return res


def train(R, E, FT, cfg, out, cfg_sha):
    need(out.is_dir() and sorted(os.listdir(out)) == PREFLIGHT_FILES,
         f"refusing: {out} must hold exactly the preflight outputs {PREFLIGHT_FILES}")
    hashes0 = code_hashes()
    pre, ready = verify_preflight(E, out, cfg_sha)
    receipt = verify_receipt(E)
    R.verify_hashes(E, cfg)
    check_recipe(E)
    T, G = E.T, E.G
    S = R.load_inputs(E, cfg)
    _, neural_path = E.project18(S["classes"])
    check_required(R, E, cfg, S["pol"], neural_path)
    need(json.loads(json.dumps(E.plain(check_inputs(R, S)))) == pre["inputs"], "inputs differ from preflight")
    parent, psd = S["parent"], S["psd0"]
    check_four_blocks(G, parent)

    parity = first_step_parity(E, FT, S)
    E.write_json(out / "FIRST_STEP_PARITY.json", {
        "created_utc": now().isoformat(), "tolerance": TOL_STEP, "folds": parity, "diagnostic_runs": 2 * N_FOLDS,
        "note": "one step on schedule[:1]: train_addition.train_arm vs feature_train.train_arm(train_last_block=False); "
                "diagnostic only, not counted as experiment runs"})
    need(all(torch.equal(v, psd[k]) for k, v in parent.state_dict().items()), "parity modified parent")

    def progress(arm, done, total):
        print(f"[{now().isoformat()}] {arm} {done}/{total}", flush=True)

    meta_ck = {k: v for k, v in S["ck"].items() if k != "state_dict"}
    recipe = {"steps": STEPS, "extra_weight": EXTRA_W, "train_kw": T.TRAIN_KW, "threads": THREADS,
              "optimizer": "AdamW betas(0.9,0.999) eps1e-8, unchanged", "module_mode": "eval (all buffers frozen)",
              "objective": "CE32 + 0.5*0.01*sum((theta-theta_parent)^2 over trainable) + 0.5*1.0*centred teacher "
                           "residual on base16 + 0.25*CE(4 KSL + 4 MP support, separate forward); one total.backward()",
              "head_arm": "head.* only; train_addition.train_arm unchanged",
              "block_arm": f"{LAST}* + head.*; feature_train.train_arm(train_last_block=True)",
              "loss_columns": FT.LOSS_COLUMNS, "projection": None, "warm_start": None}
    arrays = {"loss_columns": np.array(FT.LOSS_COLUMNS), "base_indices": S["cache"][5],
              "context_indices": S["cache"][6], "support_ids": S["sup_ids"]}
    files, checks, results = {}, {}, {}
    for f, d in S["folds"].items():
        arrays.update({f"extra_schedule__fold{f}": d["xsch"], f"train_rows__fold{f}": d["rows"],
                       f"heldout_rows__fold{f}": d["held"]})
        kw = dict(extra_prefixes=d["xp"], extra_y=d["xy"], extra_indices=d["xsch"], extra_weight=EXTRA_W,
                  progress=progress, **T.TRAIN_KW)
        for arm, block_arm in ((f"head_fold{f}", False), (f"block_fold{f}", True)):
            if block_arm:
                r = FT.train_arm(parent, *S["cache"], arm=arm, train_last_block=True, G=G, **kw)
            else:
                r = T.train_arm(parent, *S["cache"], arm=arm, **kw)
            checks[arm] = check_arm(r, parent, psd, S, d, block_arm)
            need(all(checks[arm].values()), f"{arm}: training checks failed {checks[arm]}")
            need(all(torch.equal(v, psd[k]) for k, v in parent.state_dict().items()), f"{arm} modified parent")
            p = out / f"{arm}_final.pt"
            with p.open("xb") as fh:
                torch.save(dict(meta_ck, state_dict=r["state_dict"], mp_feature_adaptation_pair={
                    "arm": arm, "fold": f, "train_last_block": block_arm, "trainable_names": r["trainable_names"],
                    "recipe": E.plain(recipe), "parent_sha256": E.PARENT_SHA, "classes": S["classes"],
                    "training_ids": [S["ids"][int(i)] for i in d["rows"]],
                    "heldout_ids": [S["ids"][int(i)] for i in d["held"]], "support_ids": S["sup_ids"].tolist(),
                    "code_hashes": hashes0, "config_sha256": cfg_sha,
                    "eval_prefixes_sha256": ready["eval_prefixes_sha256"]}), fh)
            files[p.name] = E.sha(p)
            arrays[f"losses__{arm}"] = r["losses"]
            results[arm] = r

    head_vs_old, block_vs_head = {}, {}
    for f in range(N_FOLDS):
        h, b = results[f"head_fold{f}"], results[f"block_fold{f}"]
        osd = torch.load(OLD_SUPPORT_RUN / f"candidate_fold{f}_final.pt", map_location="cpu", weights_only=True)["state_dict"]
        head_vs_old[f"head_fold{f}"] = {"state": state_diff(h["state_dict"], osd),
                                        "losses": loss_diff(h["losses"], S["old"][f"losses__candidate_fold{f}"])}
        block_vs_head[f"block_fold{f}"] = {"state": state_diff(b["state_dict"], h["state_dict"]),
                                           "losses": loss_diff(b["losses"], h["losses"]),
                                           "first_step_losses": loss_diff(b["losses"][:1], h["losses"][:1])}
        need(block_vs_head[f"block_fold{f}"]["first_step_losses"]["max_abs"] <= TOL_STEP,
             f"fold {f}: first pre-update losses differ between arms")
    E.save_npz(out / "training_arrays.npz", arrays)
    files["training_arrays.npz"] = E.sha(out / "training_arrays.npz")
    R.verify_hashes(E, cfg)
    need(code_hashes() == hashes0, "code changed during training")
    E.write_json(out / "TRAINING.json", {
        "status": TRAINED_STATUS, "created_utc": now().isoformat(), "files": files, "config_sha256": cfg_sha,
        "preflight_sha256": E.sha(out / "PREFLIGHT.json"), "prefix_ready_sha256": E.sha(out / "PREFIX_READY.json"),
        "eval_prefixes_sha256": ready["eval_prefixes_sha256"],
        "first_step_parity_sha256": E.sha(out / "FIRST_STEP_PARITY.json"),
        "test_receipt_sha256": E.sha(RECEIPT), "test_receipt_status": receipt["status"], "code_hashes": hashes0,
        "recipe": recipe, "parent_sha256": E.PARENT_SHA, "reference_sha256": E.REF_SHA, "classes": S["classes"],
        "checks": checks, "trainable_names": {a: r["trainable_names"] for a, r in results.items()},
        "head_vs_old_mp_support_candidate": head_vs_old, "block_vs_head": block_vs_head,
        "loss_first_last": {a: {"first": r["losses"][0], "last": r["losses"][-1]} for a, r in results.items()},
        "experiment_runs": len(results), "diagnostic_parity_runs": 2 * N_FOLDS,
        "warm_start": False, "projection": False, "retrained_on_all36": False, "hyperparameter_search": False,
        "evaluation_performed": False, "improvement_claimed": False,
        "threads": torch.get_num_threads(), "torch": torch.__version__, "python": platform.python_version(),
        "pins_verified_before_and_after": True})
    print("training frozen:", out)


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def changes_between(rows, groups, cand, comp, scope):
    out = {}
    for g in groups:
        rs = [r for r in rows if r["group"] == g and scope in r["models"][cand]]
        out[g] = {"gains": [dict(id=r["id"], condition=r["condition"]) for r in rs
                            if r["models"][cand][scope]["correct"] and not r["models"][comp][scope]["correct"]],
                  "regressions": [dict(id=r["id"], condition=r["condition"], true=r["true"],
                                       pred=r["models"][cand][scope]["pred"]) for r in rs
                                  if not r["models"][cand][scope]["correct"] and r["models"][comp][scope]["correct"]]}
    return out


def evaluate(R, E, FT, cfg, out, cfg_sha):
    for n in ("EVALUATION.json", "evaluation_arrays.npz"):
        need(not (out / n).exists(), f"refusing to overwrite {out / n}")
    hashes0 = code_hashes()
    tr = E.read(out / "TRAINING.json")
    need(tr["status"] == TRAINED_STATUS and tr["config_sha256"] == cfg_sha, "TRAINING.json status/config")
    need(tr["preflight_sha256"] == E.sha(out / "PREFLIGHT.json"), "PREFLIGHT changed since training")
    for n, h in tr["files"].items():
        need(E.sha(out / n) == h, f"training file changed: {n}")
    need(tr["code_hashes"] == hashes0, "code changed since training")
    _, ready = verify_preflight(E, out, cfg_sha)
    need(tr["eval_prefixes_sha256"] == ready["eval_prefixes_sha256"], "prefix cache differs from training record")
    R.verify_hashes(E, cfg)
    G = E.G
    torch.set_num_threads(THREADS)
    pol, ck, parent, classes, ref, _ = E.load_models()
    torch.set_num_threads(THREADS)
    classes = [str(c) for c in classes]
    target_idx, neural_path = E.project18(classes)
    check_required(R, E, cfg, pol, neural_path)
    check_four_blocks(G, parent)
    need(classes == ready["classes"] and target_idx == ready["project18_idx"], "classes/project18 differ from preflight")
    target_names = {classes[i] for i in target_idx}

    # ---- accept the prefix cache only if proj + blocks.0..2 (and every buffer) equal the parent in every arm
    psd = ck["state_dict"]
    pnames = {n for n, _ in parent.named_parameters()}
    sds = {REF: ref["state_dict"]}
    for a in ARMS:
        sds[a] = torch.load(out / f"{a}_final.pt", map_location="cpu", weights_only=True)["state_dict"]
    for a, sd in sds.items():
        need(set(sd) == set(psd), f"{a}: state_dict keys differ")
        for k in psd:
            trainable = k.startswith(HEAD) or (a in BLOCKS and k.startswith(LAST) and k in pnames)
            need(trainable or torch.equal(sd[k], psd[k]), f"{a}: {k} differs from parent; prefix cache invalid")
    models = [REF] + ARMS
    mods = {a: ref_model(parent, sds[a]) for a in models}
    checks = {"prefix_cache_valid_all_non_trainable_tensors_bitwise_equal_parent": True,
              "project18_module_sha256": E.sha(neural_path)}

    io = E.load_io()
    meta, old = E.read(E.META), E.read(E.OLD_RUN / "evaluation.json")
    oldmap = {(r["group"], r["id"], r.get("condition", "intact")): r for r in old["rows"]}
    prefixes, pre_ref = {}, {}
    with np.load(out / "eval_prefixes.npz", allow_pickle=False) as z:
        need([str(c) for c in z["classes"].tolist()] == classes, "eval_prefixes classes")
        for g in E.GROUPS:
            need(z[f"{g}__ids"].tolist() == [str(x["id"]) for x in meta[g]] and len(meta[g]) == GROUP_SIZES[g]
                 and z[f"{g}__conditions"].tolist() == [x.get("condition", "intact") for x in meta[g]],
                 f"{g}: prefix cache order differs from metadata")
            prefixes[g] = io.unpack(z[f"{g}__prefix"], z[f"{g}__offsets"])
            pre_ref[g] = z[f"{g}__ref832_scores"].copy()

    # ---- every arm: cached prefix -> its OWN last block -> masked mean -> head
    scores, pooled, logits = {}, {}, {}
    checks["ref832_reproduction"], checks["pooled_vs_ref832_max_abs"] = {}, {}
    with np.load(E.OLD_RUN / "evaluation_arrays.npz", allow_pickle=False) as ev:
        need([str(c) for c in ev["classes"].tolist()] == classes, "evaluation_arrays classes")
        for g in E.GROUPS:
            for a in models:
                r = prefix_scores(G, mods[a], prefixes[g])
                scores[g, a], pooled[g, a], logits[g, a] = r["scores"], r["pooled"], r["logits"]
            cs = ev[g + "__control__scores"]
            c = {"vs_cached_scores": maxabs(scores[g, REF], cs),
                 "vs_preflight_scores": maxabs(scores[g, REF], pre_ref[g]),
                 "argmax_full93_equal": bool(np.array_equal(scores[g, REF].argmax(1), cs.argmax(1))),
                 "argmax_project18_equal": bool(np.array_equal(scores[g, REF][:, target_idx].argmax(1),
                                                               cs[:, target_idx].argmax(1)))}
            need(c["vs_cached_scores"] <= TOL_832 and c["argmax_full93_equal"] and c["argmax_project18_equal"],
                 f"{g}: 832 not reproduced {c}")
            checks["ref832_reproduction"][g] = c
            checks["pooled_vs_ref832_max_abs"][g] = {a: maxabs(pooled[g, a], pooled[g, REF]) for a in ARMS}
            for a in HEADS:
                need(same(pooled[g, a], pooled[g, REF]), f"{g}/{a}: head arm pooled differs from 832 (shared last block)")
            for a in BLOCKS:
                need(checks["pooled_vs_ref832_max_abs"][g][a] > 0, f"{g}/{a}: candidate last block not used")
    checks["head_arm_pooled_bitwise_equal_ref832"] = True
    checks["block_arm_pooled_differs_from_ref832"] = True
    del prefixes

    rows = []
    for g in E.GROUPS:
        for i, spec in enumerate(meta[g]):
            label, cond = E.label_of(spec), spec.get("condition", "intact")
            yi = classes.index(label)
            row = dict(group=g, id=spec["id"], condition=cond, true=label, models={})
            if "signer_id" in spec:
                row["signer_id"] = spec["signer_id"]
            for a in models:
                row["models"][a] = {sc: E.score_entry(scores[g, a][i], yi, inds, classes)
                                    for sc, inds in (("full93", list(range(len(classes)))), ("project18", target_idx))
                                    if yi in inds}
            oldrow = oldmap[g, spec["id"], cond]
            need(oldrow["true"] == label, f"{g}/{spec['id']}: label differs from cached evaluation")
            for sc, e in row["models"][REF].items():
                o = oldrow["models"]["control"][sc]
                need(e["pred"] == o["pred"] and e["correct"] == o["correct"], f"{g}/{spec['id']}: 832 prediction differs")
            rows.append(row)
    checks["ref832_cached_predictions_reproduced"] = True
    summary = {g: {a: {cond: {sc: E.summarize([r for r in rows if r["group"] == g and r["condition"] == cond], a, sc)
                              for sc in ("full93", "project18")}
                       for cond in sorted({r["condition"] for r in rows if r["group"] == g})}
                   for a in models} for g in E.GROUPS}
    changes = {}
    for f in range(N_FOLDS):
        b, h = f"block_fold{f}", f"head_fold{f}"
        for cand, comp in ((b, h), (b, REF), (h, REF)):
            changes[f"{cand}_vs_{comp}"] = {sc: changes_between(rows, E.GROUPS, cand, comp, sc)
                                            for sc in ("full93", "project18")}

    # ---- KSL 36: cached KSL prefixes through each arm's last block; OOF by new.folds only
    new = E.load_new()
    need([str(c) for c in new["classes"].tolist()] == classes, "new classes differ")
    ids, y, folds, exposed = new["id_list"], new["y"], new["folds"], new["exposed"].astype(bool)
    ns, npooled = {}, {}
    for a in models:
        r = prefix_scores(G, mods[a], new["prefixes"])
        ns[a], npooled[a] = r["scores"], r["pooled"]
    checks["ksl36"] = {"ref832_scores_vs_cached_logits": maxabs(ns[REF], new["logits_cached"]),
                       "ref832_pooled_vs_pooled_832": maxabs(npooled[REF], new["pooled_832"]),
                       "pooled_vs_ref832_max_abs": {a: maxabs(npooled[a], npooled[REF]) for a in ARMS}}
    need(checks["ksl36"]["ref832_scores_vs_cached_logits"] <= TOL_NEW
         and checks["ksl36"]["ref832_pooled_vs_pooled_832"] <= TOL_NEW
         and np.array_equal(ns[REF].argmax(1), new["logits_cached"].argmax(1)), f"KSL36 832 not reproduced {checks['ksl36']}")
    for kind in ("head", "block"):
        ns[f"{kind}_oof"] = np.stack([ns[f"{kind}_fold{int(f)}"][i] for i, f in enumerate(folds)])
    view = [REF, "head_oof", "block_oof"]
    all_idx = list(range(len(classes)))
    preds = {sc: {a: (np.array(inds)[s[:, inds].argmax(1)] if inds is not None else None) for a, s in ns.items()}
             for sc, inds in (("full93", all_idx),
                              ("project18", target_idx if all(classes[c] in target_names for c in set(y.tolist()))
                               else None))}
    subsets = {"primary34": ~exposed, "previously_exposed00": exposed, "all36_supplementary": np.ones(len(ids), bool)}
    need(int((~exposed).sum()) == 34, "primary subset must hold 34 clips")
    new_summary = {k: {a: {sc: (E.clip_stats(y, preds[sc][a], msk, classes) if preds[sc][a] is not None
                                else "labels not in PROJECT_18") for sc in preds} for a in view}
                   for k, msk in subsets.items()}
    per_fold = {}
    for f in range(N_FOLDS):
        held, arms = folds == f, (f"block_fold{f}", f"head_fold{f}", REF)
        per_fold[f"fold{f}"] = {"heldout": {a: E.clip_stats(y, preds["full93"][a], held, classes) for a in arms},
                                "heldout_primary": {a: E.clip_stats(y, preds["full93"][a], held & ~exposed, classes)
                                                    for a in arms}}
    clips, fp = [], preds["full93"]
    for i, cid in enumerate(ids):
        clips.append(dict(id=cid, label=classes[int(y[i])], fold=int(folds[i]), bucket=str(new["buckets"][i]),
                          previously_exposed=bool(exposed[i]),
                          pred={a: classes[int(fp[a][i])] for a in view},
                          correct={a: bool(fp[a][i] == y[i]) for a in view},
                          margin={a: float(ns[a][i, y[i]] - np.max(np.delete(ns[a][i], y[i]))) for a in view}))
    clip_changes = {f"block_oof_vs_{c}": {
        "gains": [r["id"] for r in clips if r["correct"]["block_oof"] and not r["correct"][c]],
        "losses": [r["id"] for r in clips if not r["correct"]["block_oof"] and r["correct"][c]]}
        for c in (REF, "head_oof")}

    # ---- predeclared gates (class thresholds from primary34 class counts 18 -> 17, 16 -> 14)
    prim = new_summary["primary34"]
    bo, ho = prim["block_oof"]["full93"], prim["head_oof"]["full93"]
    counts = {k: v["n"] for k, v in bo["per_class"].items()}
    need(sorted(counts.values()) == [16, 18], f"primary34 class counts not 18/16: {counts}")
    need(set(counts) == {classes[c] for c in sorted(set(y.tolist()))}, "primary34 classes differ from new y")
    threshold = {k: (17 if n == 18 else 14) for k, n in counts.items()}
    gates = {"primary": {
        "block_oof_total_ge_head_oof": bo["correct"] >= ho["correct"],
        f"block_oof_total_ge_{GATE_PRIMARY_TOTAL}": bo["correct"] >= GATE_PRIMARY_TOTAL,
        **{f"class_{k}_ge_{threshold[k]}_of_{counts[k]}": bo["per_class"][k]["correct"] >= threshold[k] for k in counts},
        **{f"class_{k}_ge_head_oof": bo["per_class"][k]["correct"] >= ho["per_class"][k]["correct"] for k in counts}},
        "retention": {f"block_fold{f}_vs_{comp}": E.retention(summary, f"block_fold{f}", comp)
                      for f in range(N_FOLDS) for comp in (REF, f"head_fold{f}")}}
    info_retention = {f"head_fold{f}_vs_{REF}": E.retention(summary, f"head_fold{f}", REF) for f in range(N_FOLDS)}
    passed = all(gates["primary"].values()) and all(all(v.values()) for v in gates["retention"].values())
    decision = PASS if passed else REJECT

    arrays = {"classes": np.array(classes), "project18_idx": np.array(target_idx, dtype=np.int64),
              "new__ids": np.array(ids), "new__y": y, "new__folds": folds, "new__exposed": exposed}
    for a in models:
        arrays[f"new__{a}__pooled"] = npooled[a]
    for a, s in ns.items():
        arrays[f"new__{a}__scores"] = s
    for g in E.GROUPS:
        for a in models:
            arrays.update({f"{g}__{a}__pooled": pooled[g, a], f"{g}__{a}__logits": logits[g, a],
                           f"{g}__{a}__scores": scores[g, a]})
    E.save_npz(out / "evaluation_arrays.npz", arrays)
    R.verify_hashes(E, cfg)
    need(code_hashes() == hashes0, "code changed during evaluation")
    E.write_json(out / "EVALUATION.json", {
        "status": "LOCAL_DEVELOPMENT_EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT", "decision": decision,
        "production_ready": False, "default_promoted": False, "user_camera_sessions_used": 0,
        "default_model_changes": 0, "gates": gates, "informational_retention_not_gated": info_retention,
        "class_thresholds": threshold, "checks": checks,
        "new36": {"summary": new_summary, "per_fold": per_fold, "clips": clips, "changes": clip_changes},
        "regression": {"summary": summary, "changes": changes}, "rows": rows,
        "training_comparisons": {"head_vs_old_mp_support_candidate": tr["head_vs_old_mp_support_candidate"],
                                 "block_vs_head": tr["block_vs_head"]},
        "scoring": "cached blocks.0..2 prefix -> each arm's own blocks.3 -> masked mean -> head via "
                   "generated_v1.predict_prefix; scores are float64 pooled @ head; old pooled features never used",
        "created_utc": now().isoformat(), "arrays_sha256": E.sha(out / "evaluation_arrays.npz"),
        "training_record_sha256": E.sha(out / "TRAINING.json"), "config_sha256": cfg_sha,
        "eval_prefixes_sha256": ready["eval_prefixes_sha256"], "code_hashes": hashes0,
        "parent_sha256": E.PARENT_SHA, "reference_sha256": E.REF_SHA, "pins_verified_before_and_after": True,
        "best_fold_selected": False, "retrained_on_all36": False, "hyperparameter_search": False,
        "limits": [
            "local development comparison only; not independent signer validation or new-person generalization",
            "primary34 is the repeatedly analysed development subset excluding the two older-exposed 00 clips; "
            "it is not pristine unseen data; previously_exposed00 is reported separately",
            "KSL OOF combines three fold models; each clip is excluded only from its own fold's training; "
            "signer identity unknown",
            "project18 scope is a narrowed ranking and is not counted as full93 success",
            "not evidence of user personalization or signing correctness",
            "nothing is production ready and the default model is not replaced"]})
    print("decision:", decision)
    for k in subsets:
        print(k, {a: new_summary[k][a]["full93"]["correct"] for a in view})


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["preflight", "train", "evaluate"])
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    out = None
    try:
        R, E, FT, cfg, out, cfg_sha = load_cfg(a.config)
        {"preflight": preflight, "train": train, "evaluate": evaluate}[a.command](R, E, FT, cfg, out, cfg_sha)
    except BaseException as e:
        where = out if out is not None and out.is_dir() else HERE
        t = now()
        p = where / f"FAILED_{a.command}_{t.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        with p.open("x", encoding="utf-8") as fh:
            json.dump({"command": a.command, "created_utc": t.isoformat(), "error": repr(e),
                       "traceback": traceback.format_exc(), "note": "first failure preserved; do not rerun to improve scores"},
                      fh, ensure_ascii=False, indent=2)
        raise


if __name__ == "__main__":
    main()
