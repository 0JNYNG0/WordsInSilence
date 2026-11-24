"""Original b8 preservation teacher vs current-832 preservation teacher, head-only, fixed recipe (FROZEN_PLAN.md).

    python teacher_pair.py required-pins                     # print paths the config must pin (reads, writes nothing)
    python teacher_pair.py preflight --config CONFIG.json    # verify + freeze teachers.npz, no training
    python teacher_pair.py train     --config CONFIG.json    # needs PREFLIGHT + TEST_RECEIPT; 6 x 660 steps
    python teacher_pair.py evaluate  --config CONFIG.json    # needs TRAINING.json; cached pooled features only

Both arms call the existing train_addition.train_arm (E.T.train_arm) unchanged. The ONLY difference
between control_foldF and current_teacher_foldF is the positional teacher_base_logits array:
    control         : the original cached b8 teacher (train_prefixes.npz teacher_base_logits), as before
    current_teacher : float64 affine of the old training base_x pooled features with the 832 head
Start weights and anchor stay b8. No projection, no last-block training, no new learner.
Old modules are read-only and loaded under private names; nothing old is modified or written.
Writes only FRONT/runs/current_teacher_pair_v1 (create-exclusive) and FAILED_*.json. Nothing is promoted.
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
UNIT = FRONT / "ops" / "current_teacher_pair_20261004"
RUN = FRONT / "runs" / "current_teacher_pair_v1"
OLD_GPP = FRONT / "ops" / "gradient_projection_pair_20261004"
RECEIPT = HERE / "TEST_RECEIPT.json"
SCRIPTS = ("teacher_pair.py", "test_teacher.py")
OLD_HELPERS = {
    "gpp/run_pair.py": OLD_GPP / "run_pair.py",
    "gpp/projected_train.py": OLD_GPP / "projected_train.py",
    "tia/experiment.py": FRONT / "ops" / "target_input_adaptation_20260928" / "experiment.py",
    "tia/train_addition.py": FRONT / "ops" / "target_input_adaptation_20260928" / "train_addition.py",
    "pep/generated_v1.py": FRONT / "ops" / "partial_encoder_pair_20260922" / "generated_v1.py",
    "pep/local_io.py": FRONT / "ops" / "partial_encoder_pair_20260922" / "local_io.py",
}
_R_NAME = "ctp_old_gpp_run_pair"

PARENT_SHA = "b8d6045c0c154697a5d6d3440927a48cf1e9db3146421db0a920da9fa684870d"
REF_SHA = "83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91"
TRAIN_KW = dict(learning_rate=1e-3, weight_decay=1e-3, anchor_coefficient=1e-2, output_coefficient=1.0)
LOSS_COLUMNS = ["old_ce", "extra_ce", "weighted_extra", "weighted_anchor", "weighted_output", "total"]
STEPS, EXTRA_W, N_FOLDS, HALF, N_BLOCKS, KSL_PER_STEP, SUPPORT_PER_STEP = 660, 0.25, 3, 16, 4, 4, 4
N_BASE, N_CTX, N_CLASSES = 2122, 224, 93
FOLD_TRAIN, FOLD_HELD = (22, 26, 24), (14, 10, 12)
HEAD_KEYS = ["head.bias", "head.weight"]
CONTROL = [f"control_fold{f}" for f in range(N_FOLDS)]
TEACHER = [f"current_teacher_fold{f}" for f in range(N_FOLDS)]
ARMS = CONTROL + TEACHER
REF = "ref832"
GROUP_N = {"annotation_dev": 699, "whole_dev": 74, "public": 32, "mp": 14}
# base-id overlap roles: heldout groups must not overlap the 2122 training base rows; public/mp are
# pre-existing, documented training-exposed regression references (reported, never an independent test)
HELDOUT_GROUPS = ("annotation_dev", "whole_dev")
TRAINING_EXPOSED_GROUPS = ("public", "mp")
ANNOTATION_MIN, GATE_PRIMARY_TOTAL = 691, 31
TOL_TEACHER, TOL_832, TOL_NEW = 1e-5, 1e-6, 1e-5
POLICY_KEYS = ("old_features", "metadata", "teacher_scores")
FEATURE_KEYS = ("base_x", "base_y", "classes")
TS_ID_KEYS = ("ids", "base_ids")
TS_SCORE_KEY = "parent"           # existing b8 teacher in the authoritative teacher_scores npz; no fallback
PRE_FILES = ["PREFLIGHT.json", "teachers.npz"]
PREFLIGHT_STATUS = "PREFLIGHT_PASSED_NO_TRAINING"
TRAINED_STATUS = "TRAINING_COMPLETE_CHECKPOINTS_FROZEN_BEFORE_EVALUATION"
PASS = "PASS_LOCAL_DEVELOPMENT_GATES_NOT_PRODUCTION_READY"
REJECT = "REJECTED_BY_PREDECLARED_GATES"


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def need(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hashes():
    """Own scripts + every read-only old helper source this runner imports."""
    out = {n: _sha(HERE / n) for n in SCRIPTS}
    out.update({k: _sha(p) for k, p in OLD_HELPERS.items()})
    return out


def load_modules():
    """(R, E): old gradient_projection run_pair.py under a private name and its experiment module.
    Module source only; no data, model or cache is read here."""
    R = sys.modules.get(_R_NAME)
    if R is None:
        path = OLD_HELPERS["gpp/run_pair.py"]
        if not path.is_file():
            raise ImportError(f"old run_pair.py not found at {path}")
        spec = importlib.util.spec_from_file_location(_R_NAME, path)
        R = importlib.util.module_from_spec(spec)
        sys.modules[_R_NAME] = R
        try:
            spec.loader.exec_module(R)
        except BaseException:
            sys.modules.pop(_R_NAME, None)
            raise
    E = R.P.load_experiment()
    bound = {"gpp/run_pair.py": R.__file__, "gpp/projected_train.py": R.P.__file__,
             "tia/experiment.py": E.__file__, "tia/train_addition.py": E.T.__file__,
             "pep/generated_v1.py": E.G.__file__}
    bad = {k: str(v) for k, v in bound.items() if Path(v).resolve() != OLD_HELPERS[k].resolve()}
    if bad:
        raise ImportError(f"old modules bound to unexpected files: {bad}")
    fixed = {"front": (Path(R.FRONT).resolve(), FRONT), "steps": (R.STEPS, STEPS), "extra_weight": (R.EXTRA_W, EXTRA_W),
             "ksl_per_step": (R.KSL_PER_STEP, KSL_PER_STEP), "support_per_step": (R.SUPPORT_PER_STEP, SUPPORT_PER_STEP),
             "n_folds": (R.N_FOLDS, N_FOLDS), "fold_train": (tuple(R.FOLD_TRAIN), FOLD_TRAIN),
             "fold_held": (tuple(R.FOLD_HELD), FOLD_HELD), "train_kw": (dict(E.T.TRAIN_KW), TRAIN_KW),
             "head": (E.T.HEAD, "head."), "loss_columns": (list(E.T.LOSS_COLUMNS), LOSS_COLUMNS),
             "parent_sha": (E.PARENT_SHA, PARENT_SHA), "ref_sha": (E.REF_SHA, REF_SHA),
             "groups": (list(E.GROUPS), list(GROUP_N))}
    bad = [k for k, (a, b) in fixed.items() if a != b]
    if bad:
        raise ImportError(f"old modules differ from the frozen plan: {bad}")
    return R, E


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def make_teacher(pooled, weight, bias):
    """float64 affine  pooled64 @ weight64.T + bias64  ->  [rows, classes].

    pooled [N, D], weight [C, D], bias [C]: finite, non-empty, real floating numpy arrays.
    Inputs are never modified. Raises TypeError / ValueError on bad input or non-finite output."""
    for name, a, nd in (("pooled", pooled, 2), ("weight", weight, 2), ("bias", bias, 1)):
        if not isinstance(a, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        if not np.issubdtype(a.dtype, np.floating):
            raise TypeError(f"{name} must be a real floating array, got {a.dtype}")
        if a.ndim != nd or a.size == 0:
            raise ValueError(f"{name} must be a non-empty {nd}-D array, got shape {a.shape}")
        if not bool(np.isfinite(a).all()):
            raise ValueError(f"{name} contains non-finite values")
    if pooled.shape[1] != weight.shape[1]:
        raise ValueError(f"pooled width {pooled.shape[1]} != weight width {weight.shape[1]}")
    if bias.shape[0] != weight.shape[0]:
        raise ValueError(f"bias length {bias.shape[0]} != weight rows {weight.shape[0]}")
    out = pooled.astype(np.float64) @ weight.astype(np.float64).T + bias.astype(np.float64)
    if not bool(np.isfinite(out).all()):
        raise ValueError("affine output is non-finite")
    return out


def head_np(sd):
    return sd["head.weight"].detach().cpu().numpy(), sd["head.bias"].detach().cpu().numpy()


def encoder_checks(G, parent, psd, ck_sd, ref_sd, n_classes=N_CLASSES):
    """Exactly 4 blocks; parent == b8 checkpoint bitwise; every non-head tensor (params AND buffers)
    of the reference equals b8 bitwise; heads differ."""
    width, _, ncls = G._inspect_model(parent)
    need(len(parent.blocks) == N_BLOCKS, f"model must have exactly {N_BLOCKS} blocks, has {len(parent.blocks)}")
    need(ncls == n_classes, f"model has {ncls} classes, expected {n_classes}")
    need(set(psd) == set(ck_sd) == set(ref_sd), "state_dict keys differ between parent / b8 checkpoint / reference")
    need(sorted(k for k in psd if k.startswith("head.")) == HEAD_KEYS, "head tensors must be exactly head.weight/bias")
    pnames = {n for n, _ in parent.named_parameters()}
    bnames = {n for n, _ in parent.named_buffers()}
    need(pnames | bnames <= set(psd), "model parameters/buffers missing from state_dict")
    non_head = [k for k in psd if k not in HEAD_KEYS]
    need(all(torch.equal(psd[k], ck_sd[k]) for k in psd), "loaded parent differs from b8 checkpoint")
    need(all(torch.equal(ref_sd[k], psd[k]) for k in non_head), "reference non-head tensor differs from b8")
    need(any(not torch.equal(ref_sd[k], psd[k]) for k in HEAD_KEYS), "reference head equals b8 head")
    return {"width": width, "blocks": len(parent.blocks), "classes": ncls, "non_head_tensors": len(non_head),
            "buffers": len(bnames), "non_head_params_and_buffers_bitwise_equal_b8": True, "heads_differ": True}


def run_arm(T, parent, cache, teacher, d, arm, progress=None):
    """The one training call for BOTH arms: existing train_arm, only `teacher` varies."""
    base, by, ctx, cy, _cached_teacher, bidx, cidx = cache
    return T.train_arm(parent, base, by, ctx, cy, teacher, bidx, cidx,
                       extra_prefixes=d["xp"], extra_y=d["xy"], extra_indices=d["xsch"], extra_weight=EXTRA_W,
                       arm=arm, progress=progress, **T.TRAIN_KW)


def teacher_diff_stats(a, b, y):
    """Diagnostics only (b - a); never used to choose anything."""
    d = b - a
    dc = d - d.mean(axis=1, keepdims=True)
    rms = np.sqrt((dc ** 2).mean(axis=1))
    return {"rows": int(len(d)), "max_abs": float(np.abs(d).max()), "mean_abs": float(np.abs(d).mean()),
            "centered_rms_mean": float(rms.mean()), "centered_rms_max": float(rms.max()),
            "argmax_agreement": int((a.argmax(1) == b.argmax(1)).sum()),
            "first_argmax_eq_train_label": int((a.argmax(1) == y).sum()),
            "second_argmax_eq_train_label": int((b.argmax(1) == y).sum()),
            "note": "first=b8 cached teacher, second=current832 teacher; diagnostic only, no coefficient chosen"}


# --------------------------------------------------------------------------- #
# config / pins
# --------------------------------------------------------------------------- #
def load_cfg(R, path):
    raw = Path(path).read_bytes()
    cfg = json.loads(raw.decode("utf-8-sig"))
    need({"root", "ops_dir", "out_dir", "pins", "protected_files"} <= set(cfg), "config keys missing")
    for k in ("root", "ops_dir", "out_dir"):
        need(Path(cfg[k]).is_absolute(), f"config {k} must be absolute")
    need(Path(cfg["root"]).resolve() == FRONT, "config root != FRONT")
    need(HERE == UNIT.resolve(), f"script must live in {UNIT}")
    need(Path(cfg["ops_dir"]).resolve() == UNIT.resolve(), "config ops_dir != FRONT/ops/current_teacher_pair_20261004")
    need(Path(cfg["out_dir"]).resolve() == RUN.resolve(), "config out_dir != FRONT/runs/current_teacher_pair_v1")
    cfg["pins"] = {R.norm_path(p): h for p, h in cfg["pins"].items()}
    cfg["protected_files"] = {R.norm_path(p): h for p, h in cfg["protected_files"].items()}
    return cfg, RUN.resolve(), hashlib.sha256(raw).hexdigest()


def policy_paths(E, pol):
    paths = pol.get("paths") if isinstance(pol, dict) else None
    need(isinstance(paths, dict), "policy has no 'paths' mapping")
    out = {}
    for k in POLICY_KEYS:
        need(k in paths, f"policy paths lack '{k}' (available: {sorted(paths)})")
        p = Path(paths[k])
        cands = [p] if p.is_absolute() else [E.PRIOR / p, FRONT / p]
        hits = sorted({str(c.resolve()) for c in cands if c.is_file()})
        need(len(hits) == 1, f"policy path '{k}' not found or ambiguous: {paths[k]} -> {hits}")
        out[k] = Path(hits[0])
    return out


def required_pins(R, E, pol):
    ps = list(R.required_pins(E, pol))
    ps += [R.norm_path(p) for p in OLD_HELPERS.values()]
    ps += [R.norm_path(p) for p in policy_paths(E, pol).values()]
    return list(dict.fromkeys(ps))


def guard(R, E, cfg, want):
    need(code_hashes() == want, "own code or old helper source changed during/between stages")
    R.verify_hashes(E, cfg)


def verify_receipt():
    r = json.loads(RECEIPT.read_text(encoding="utf-8-sig"))
    need(r.get("status") == "PASSED", "TEST_RECEIPT status is not PASSED")
    need(r.get("code_hashes") == code_hashes(), "TEST_RECEIPT code hashes differ from current code")
    return r


def check_plan_inputs(S):
    c = S["cache"]
    need(len(c[0]) == N_BASE and len(c[2]) == N_CTX, "base/context counts differ from plan")
    need(c[5].shape == c[6].shape == (STEPS, HALF), "old base/context schedules are not [660,16]")
    for f, d in S["folds"].items():
        need(len(d["rows"]) == FOLD_TRAIN[f] and len(d["held"]) == FOLD_HELD[f], f"fold {f} sizes differ from plan")
        need(d["xsch"].shape == (STEPS, KSL_PER_STEP + SUPPORT_PER_STEP), f"fold {f} extra schedule shape")


# --------------------------------------------------------------------------- #
# old training base rows and the two teachers
# --------------------------------------------------------------------------- #
def load_teacher_scores(path):
    """Authoritative teacher_scores npz: ids, classes and the existing b8 teacher under `parent`.
    Missing `parent` refuses; control/treatment are never used as a fallback."""
    path = Path(path)
    need(path.suffix == ".npz", f"teacher_scores must be the authoritative .npz: {path}")
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        idk = next((k for k in TS_ID_KEYS if k in keys), None)
        need(idk is not None and "classes" in keys, f"teacher_scores lacks ids/classes; keys {keys}")
        need(TS_SCORE_KEY in keys, f"teacher_scores lacks '{TS_SCORE_KEY}' teacher; keys {keys}")
        return dict(ids=[str(i) for i in z[idk].tolist()], classes=[str(c) for c in z["classes"].tolist()],
                    scores=z[TS_SCORE_KEY].copy(), keys=keys, id_key=idk, score_key=TS_SCORE_KEY)


def check_teacher_scores(ts, ids, classes, cached, maxabs):
    """Class order equal, ids unique, base ids a strictly order-preserving subset, and the `parent`
    teacher (rows taken in exact base order) equal to the cached teacher within TOL_TEACHER."""
    need(ts["score_key"] == TS_SCORE_KEY and ts["scores"] is not None, "teacher_scores parent teacher not loaded")
    need(ts["classes"] == classes, "original teacher_scores classes differ (order)")
    need(len(set(ts["ids"])) == len(ts["ids"]), "original teacher_scores ids not unique")
    pos_of = {i: k for k, i in enumerate(ts["ids"])}
    need(all(i in pos_of for i in ids), "base ids missing from original teacher_scores")
    pos = np.array([pos_of[i] for i in ids], dtype=np.int64)
    need(bool((np.diff(pos) > 0).all()), "base ids are not in original teacher_scores order")
    sc = np.asarray(ts["scores"])
    need(sc.shape == (len(ts["ids"]), len(classes)), f"teacher_scores parent shape {sc.shape}")
    need(np.shape(cached) == (len(ids), len(classes)), f"cached teacher shape {np.shape(cached)}")
    d = maxabs(sc[pos], cached)
    need(d <= TOL_TEACHER, f"original teacher_scores parent != cache teacher ({d})")
    return {"keys": ts["keys"], "id_key": ts["id_key"], "score_key": ts["score_key"], "n_ids": len(ts["ids"]),
            "identical_order": ts["ids"] == ids, "relative_order_preserved": True,
            "parent_vs_cache_teacher_max_abs": d, "parent_compared": True}


def id_overlap_audit(base_ids, eval_meta, new_ids, support_ids):
    """Exact-id overlap of the training base rows with every group. Heldout groups and new36 must be
    zero; public/mp overlaps are reported as pre-existing training-exposed regression references."""
    need(not set(HELDOUT_GROUPS) & set(TRAINING_EXPOSED_GROUPS)
         and set(HELDOUT_GROUPS) | set(TRAINING_EXPOSED_GROUPS) == set(GROUP_N), "every group needs exactly one role")
    need(set(GROUP_N) <= set(eval_meta), f"evaluation metadata lacks groups {sorted(set(GROUP_N) - set(eval_meta))}")
    base = set(base_ids)
    gid = {g: {str(s.get("id")) for s in eval_meta[g]} for g in GROUP_N}
    heldout = {g: len(gid[g] & base) for g in HELDOUT_GROUPS}
    heldout["new36"] = len({str(i) for i in new_ids} & base)
    need(all(v == 0 for v in heldout.values()), f"base rows overlap heldout rows: {heldout}")
    exposed = {g: {"rows": len(eval_meta[g]), "unique_ids": len(gid[g]), "overlap_ids": len(gid[g] & base),
                   "overlapping_ids": sorted(gid[g] & base)} for g in TRAINING_EXPOSED_GROUPS}
    return {"heldout_must_be_zero": heldout,
            "training_exposed_regression_reference": exposed,
            "training_exposed_note": "public/mp ids overlapping the fixed base are pre-existing, documented "
                                     "training exposure; these groups are regression references, not an "
                                     "independent test. No new evaluation rows are introduced: teacher rows are "
                                     "exclusively the fixed 2122 base.",
            "support113_informational": len({str(s) for s in support_ids} & base)}


def load_base_rows(E, S, paths, width):
    """Old training base_x/base_y/classes and base ids, aligned to the cache. Reads only training-row
    arrays and metadata ids/labels; no evaluation or user data is read."""
    classes, cache = S["classes"], S["cache"]
    with np.load(paths["old_features"], allow_pickle=False) as z:
        files = list(z.files)
        need(set(FEATURE_KEYS) <= set(files), f"old_features lacks {FEATURE_KEYS}; keys {files}")
        x, y, cl = z["base_x"].copy(), z["base_y"].copy(), z["classes"].copy()
        fid = [str(i) for i in z["base_ids"].tolist()] if "base_ids" in files else None
    need([str(c) for c in cl.tolist()] == classes, "old_features classes != model classes (order)")
    need(x.ndim == 2 and x.shape == (N_BASE, width), f"base_x shape {x.shape} != ({N_BASE}, {width})")
    need(np.issubdtype(x.dtype, np.floating) and bool(np.isfinite(x).all()), "base_x not finite floating")
    need(np.issubdtype(y.dtype, np.integer) and y.shape == (N_BASE,), "base_y shape/dtype")
    need(np.array_equal(y.astype(np.int64), cache[1]), "old_features base_y != original cache base_y")

    meta = E.read(paths["metadata"])
    rows = meta["base"]
    need(len(rows) == N_BASE, "metadata base count != 2122")
    ids = [str(r.get("id")) for r in rows]
    need(len(set(ids)) == N_BASE, "metadata base ids not unique")
    need(all(E.label_of(r) == classes[int(c)] for r, c in zip(rows, cache[1])), "metadata base labels != cache base_y")
    need(all(E.approved_training_row(r) for r in rows), "metadata base row not tagged as approved training")
    if fid is not None:
        need(fid == ids, "old_features base_ids != metadata base ids")

    ts = load_teacher_scores(paths["teacher_scores"])
    ts_check = check_teacher_scores(ts, ids, classes, cache[4], E.maxabs)

    overlap = id_overlap_audit(ids, E.read(E.META), S["ids"], S["sup_ids"].tolist())
    return dict(base_x=x, ids=ids, checks={
        "old_features_keys": files, "base_x_dtype": str(x.dtype), "old_features_file_sha256": E.sha(paths["old_features"]),
        "metadata_is_experiment_META": Path(paths["metadata"]).resolve() == Path(E.META).resolve(),
        "base_y_equals_cache": True, "labels_and_training_tags_ok": True, "teacher_scores": ts_check,
        "id_overlap": overlap})


def build_teachers(E, S, base_x, ref_sd):
    """(b8 affine, current832 affine) from base_x; b8 affine must reproduce the cached teacher."""
    cached = S["cache"][4]
    b8 = make_teacher(base_x, *head_np(S["psd0"]))
    t832 = make_teacher(base_x, *head_np(ref_sd))
    need(b8.shape == t832.shape == cached.shape == (N_BASE, len(S["classes"])), "teacher shapes")
    need(t832.dtype == cached.dtype, f"candidate dtype {t832.dtype} != cached teacher dtype {cached.dtype}")
    d = E.maxabs(b8, cached)
    need(d <= TOL_TEACHER, f"b8 affine does not reproduce cached teacher ({d})")
    need(not same(t832, cached), "current832 teacher identical to b8 teacher")
    return b8, t832, d


def build_832(parent, ref_sd):
    m = copy.deepcopy(parent)
    m.load_state_dict(ref_sd, strict=True)
    return m.eval()


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def preflight(R, E, cfg, out, cfg_sha):
    need(not out.exists(), f"refusing: {out} exists")
    h0 = code_hashes()
    R.verify_hashes(E, cfg)
    S = R.load_inputs(E, cfg)
    G = E.G
    check_plan_inputs(S)
    missing = [p for p in required_pins(R, E, S["pol"]) if p not in cfg["pins"]]
    need(not missing, f"config pins lack required inputs: {missing}")
    parent, ref_sd, cache = S["parent"], S["ref"]["state_dict"], S["cache"]
    enc = encoder_checks(G, parent, S["psd0"], S["ck"]["state_dict"], ref_sd)
    paths = policy_paths(E, S["pol"])
    base = load_base_rows(E, S, paths, enc["width"])
    base_x = base["base_x"]

    b8, t832, d_b8 = build_teachers(E, S, base_x, ref_sd)
    m832 = build_832(parent, ref_sd)
    pb = G.predict_prefix(parent, cache[0])
    pc = G.predict_prefix(m832, cache[0])
    checks = {"b8_affine_vs_cached_teacher_max_abs": d_b8,
              "parent_prefix_pooled_vs_base_x_max_abs": E.maxabs(pb["pooled"], base_x),
              "pooled_b8_vs_832_bitwise_equal": same(pb["pooled"], pc["pooled"]),
              "m832_prefix_logits_vs_current832_teacher_max_abs": E.maxabs(pc["logits"], t832),
              "m832_prefix_argmax_equal_teacher": bool((pc["logits"].argmax(1) == t832.argmax(1)).all()),
              "parent_prefix_logits_vs_b8_affine_max_abs_informational": E.maxabs(pb["logits"], b8)}
    need(checks["parent_prefix_pooled_vs_base_x_max_abs"] <= TOL_TEACHER, f"base_x not reproduced: {checks}")
    need(checks["pooled_b8_vs_832_bitwise_equal"], "b8 and 832 pooled base features differ")
    need(checks["m832_prefix_logits_vs_current832_teacher_max_abs"] <= TOL_TEACHER, f"832 logits mismatch: {checks}")
    need(checks["m832_prefix_argmax_equal_teacher"], "832 prefix argmax != current832 teacher argmax")
    need(all(torch.equal(v, S["psd0"][k]) for k, v in parent.state_dict().items()), "parent modified in preflight")
    used = np.unique(cache[5])
    guard(R, E, cfg, h0)

    out.mkdir(parents=False, exist_ok=False)
    E.save_npz(out / "teachers.npz", {
        "classes": np.array(S["classes"]), "base_ids": np.array(base["ids"]), "base_y": cache[1],
        "control_teacher_b8_cached": cache[4], "b8_affine_teacher": b8, "current832_teacher": t832,
        "base_x_sha256": np.array(R.arr_sha(base_x)), "base_pooled_b8_sha256": np.array(R.arr_sha(pb["pooled"]))})
    E.write_json(out / "PREFLIGHT.json", {
        "status": PREFLIGHT_STATUS, "created_utc": now().isoformat(), "config_sha256": cfg_sha,
        "code_hashes": h0, "inputs": R.input_summary(S), "files": {"teachers.npz": E.sha(out / "teachers.npz")},
        "policy_paths": {k: str(p) for k, p in paths.items()}, "encoder": enc, "base_rows": base["checks"],
        "teachers": {"control": {"source": "train_prefixes.npz teacher_base_logits (original b8)",
                                 "sha256": R.arr_sha(cache[4]), "dtype": str(cache[4].dtype)},
                     "current_teacher": {"source": "make_teacher(old_features base_x, 832 head) float64",
                                         "sha256": R.arr_sha(t832), "dtype": str(t832.dtype)},
                     "b8_affine_sha256": R.arr_sha(b8), "base_x_sha256": R.arr_sha(base_x),
                     "shape": list(t832.shape), "tolerance": TOL_TEACHER, "checks": checks},
        "teacher_rows_used_by_base_schedule": int(len(used)),
        "teacher_difference_diagnostics": teacher_diff_stats(cache[4].astype(np.float64), t832, cache[1]),
        "frozen_recipe": {"steps": STEPS, "extra_weight": EXTRA_W, "train_kw": TRAIN_KW, "half_batch": HALF,
                          "ksl_per_step": KSL_PER_STEP, "support_per_step": SUPPORT_PER_STEP,
                          "start_and_anchor": "b8 parent", "trainable": "head.* only", "projection": None},
        "pins_and_code_verified_before_and_after": True, "no_training_performed": True,
        "torch": torch.__version__, "threads": torch.get_num_threads()})
    print("preflight passed:", out)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def check_out(T, r, psd, S, d, tag):
    sd, losses = r["state_dict"], np.asarray(r["losses"])
    c = {"updates": int(r["updates"]) == STEPS,
         "losses_shape_finite": losses.shape == (STEPS, len(T.LOSS_COLUMNS)) and bool(np.isfinite(losses).all()),
         "extra_weight": float(r["extra_weight"]) == EXTRA_W,
         "old_schedule_identical": same(r["base_indices"], S["cache"][5]) and same(r["context_indices"], S["cache"][6]),
         "extra_schedule_identical": same(r["extra_indices"], d["xsch"]),
         "keys_equal_parent": set(sd) == set(psd),
         "trainable_exactly_head": sorted(r["trainable_names"]) == HEAD_KEYS,
         "non_head_bitwise_equal_parent": all(torch.equal(sd[k], psd[k]) for k in psd if k not in HEAD_KEYS),
         "head_changed": any(not torch.equal(sd[k], psd[k]) for k in HEAD_KEYS),
         "dtype_shape_equal_parent": all(sd[k].dtype == psd[k].dtype and sd[k].shape == psd[k].shape for k in psd),
         "state_finite": all(bool(torch.isfinite(v).all()) for v in sd.values() if v.is_floating_point()),
         "frozen_parameters_unchanged": bool(r["frozen_parameters_unchanged"]),
         "buffers_unchanged": bool(r["buffers_unchanged"])}
    need(all(c.values()), f"{tag}: training checks failed {c}")
    return c


def train(R, E, cfg, out, cfg_sha):
    need((out / "PREFLIGHT.json").is_file(), "PREFLIGHT.json missing; run preflight first")
    need(sorted(os.listdir(out)) == PRE_FILES, f"refusing: {out} holds files besides {PRE_FILES}")
    pre = E.read(out / "PREFLIGHT.json")
    need(pre["status"] == PREFLIGHT_STATUS and pre["config_sha256"] == cfg_sha, "PREFLIGHT status/config differ")
    h0 = code_hashes()
    need(pre["code_hashes"] == h0, "code changed since preflight")
    receipt = verify_receipt()
    R.verify_hashes(E, cfg)
    need(E.sha(out / "teachers.npz") == pre["files"]["teachers.npz"], "teachers.npz changed since preflight")
    S = R.load_inputs(E, cfg)
    need(json.loads(json.dumps(E.plain(R.input_summary(S)))) == pre["inputs"], "inputs differ from preflight")
    check_plan_inputs(S)
    T, G = E.T, E.G
    parent, psd, cache = S["parent"], S["psd0"], S["cache"]
    encoder_checks(G, parent, psd, S["ck"]["state_dict"], S["ref"]["state_dict"])

    # frozen teachers: control is the cache array itself; candidate must equal a fresh affine bitwise
    paths = policy_paths(E, S["pol"])
    need({k: str(p) for k, p in paths.items()} == pre["policy_paths"], "policy paths differ from preflight")
    base = load_base_rows(E, S, paths, G._inspect_model(parent)[0])
    with np.load(out / "teachers.npz", allow_pickle=False) as z:
        tz = {k: z[k].copy() for k in z.files}
    need(same(tz["control_teacher_b8_cached"], cache[4]), "frozen control teacher != cache teacher")
    need([str(i) for i in tz["base_ids"].tolist()] == base["ids"], "frozen base ids differ")
    need([str(c) for c in tz["classes"].tolist()] == S["classes"], "frozen classes differ")
    t832 = tz["current832_teacher"]
    need(same(t832, make_teacher(base["base_x"], *head_np(S["ref"]["state_dict"]))), "frozen 832 teacher not reproduced")
    need(R.arr_sha(t832) == pre["teachers"]["current_teacher"]["sha256"], "832 teacher sha differs from preflight")
    teachers = {"control": cache[4], "current_teacher": t832}
    tsha = {k: R.arr_sha(v) for k, v in teachers.items()}

    def progress(arm, done, total):
        print(f"[{now().isoformat()}] {arm} {done}/{total}", flush=True)

    meta_ck = {k: v for k, v in S["ck"].items() if k != "state_dict"}
    recipe = {"call": "E.T.train_arm(parent, base, base_y, context, context_y, TEACHER, base_idx, context_idx, "
                      "extra_prefixes, extra_y, extra_indices, extra_weight=0.25, **T.TRAIN_KW)",
              "steps": STEPS, "extra_weight": EXTRA_W, "train_kw": T.TRAIN_KW, "trainable": "head.* only",
              "start_and_anchor": "b8 parent for both arms", "projection": None, "threads": R.THREADS,
              "only_difference": "teacher_base_logits", "loss_columns": T.LOSS_COLUMNS}
    arrays = {"loss_columns": np.array(T.LOSS_COLUMNS), "base_indices": cache[5], "context_indices": cache[6],
              "support_ids": S["sup_ids"]}
    files, checks, results = {}, {}, {}
    for f, d in S["folds"].items():
        arrays.update({f"extra_schedule__fold{f}": d["xsch"], f"train_rows__fold{f}": d["rows"],
                       f"heldout_rows__fold{f}": d["held"]})
        for arm, tkey in ((CONTROL[f], "control"), (TEACHER[f], "current_teacher")):
            r = run_arm(T, parent, cache, teachers[tkey], d, arm, progress)
            checks[arm] = check_out(T, r, psd, S, d, arm)
            need(all(torch.equal(v, psd[k]) for k, v in parent.state_dict().items()), f"{arm} modified parent")
            p = out / f"{arm}_final.pt"
            with p.open("xb") as fh:
                torch.save(dict(meta_ck, state_dict=r["state_dict"], current_teacher_pair={
                    "arm": arm, "fold": f, "teacher": tkey, "teacher_sha256": tsha[tkey],
                    "teachers_npz_sha256": pre["files"]["teachers.npz"], "recipe": E.plain(recipe),
                    "parent_sha256": E.PARENT_SHA, "classes": S["classes"],
                    "training_ids": [S["ids"][int(i)] for i in d["rows"]],
                    "heldout_ids": [S["ids"][int(i)] for i in d["held"]], "support_ids": S["sup_ids"].tolist(),
                    "code_hashes": h0, "config_sha256": cfg_sha}), fh)
            files[p.name] = E.sha(p)
            arrays[f"losses__{arm}"] = r["losses"]
            results[arm] = r
    E.save_npz(out / "training_arrays.npz", arrays)
    files["training_arrays.npz"] = E.sha(out / "training_arrays.npz")

    vs_old, vs_control = {}, {}
    for f in range(N_FOLDS):
        osd = torch.load(R.OLD_RUN / f"candidate_fold{f}_final.pt", map_location="cpu", weights_only=True)["state_dict"]
        cl, ol = results[CONTROL[f]]["losses"], S["old"][f"losses__candidate_fold{f}"]
        vs_old[CONTROL[f]] = dict(R.head_diff(results[CONTROL[f]]["state_dict"], osd),
                                  losses_bitwise_equal=bool(cl.shape == ol.shape and np.array_equal(cl, ol, equal_nan=True)),
                                  max_abs_loss_diff=float(np.nanmax(np.abs(cl - ol))) if cl.shape == ol.shape else None)
        tl = results[TEACHER[f]]["losses"]
        vs_control[TEACHER[f]] = dict(R.head_diff(results[TEACHER[f]]["state_dict"], results[CONTROL[f]]["state_dict"]),
                                      max_abs_loss_diff_per_column=dict(zip(T.LOSS_COLUMNS,
                                                                            np.abs(tl - cl).max(axis=0).tolist())))
    guard(R, E, cfg, h0)
    E.write_json(out / "TRAINING.json", {
        "status": TRAINED_STATUS, "created_utc": now().isoformat(), "files": files, "config_sha256": cfg_sha,
        "preflight_sha256": E.sha(out / "PREFLIGHT.json"), "teachers_npz_sha256": pre["files"]["teachers.npz"],
        "teacher_sha256": tsha, "test_receipt_sha256": E.sha(RECEIPT), "test_receipt_status": receipt["status"],
        "code_hashes": h0, "recipe": recipe, "parent_sha256": E.PARENT_SHA, "reference_sha256": E.REF_SHA,
        "classes": S["classes"], "checks": checks,
        "control_vs_old_mp_support_candidate": vs_old,
        "control_vs_old_note": "same train_arm call/inputs as mp_support_addition_v1 candidate_fold*; reported, not "
                               "required. If not bitwise equal, the new controls remain the predeclared comparator.",
        "current_teacher_vs_control": vs_control,
        "loss_first_last": {a: {"first": r["losses"][0], "last": r["losses"][-1]} for a, r in results.items()},
        "evaluation_performed": False, "hyperparameter_search": False, "diagnostic_training_runs": 0,
        "threads": torch.get_num_threads(), "torch": torch.__version__, "python": platform.python_version(),
        "pins_and_code_verified_before_and_after": True})
    print("training frozen:", out)


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def changes_between(rows, cand, comp):
    out = {}
    for sc in ("full93", "project18"):
        rs = [r for r in rows if sc in r["models"][cand]]
        out[sc] = {g: {"gains": [dict(id=r["id"], condition=r["condition"], true=r["true"])
                                 for r in rs if r["group"] == g and r["models"][cand][sc]["correct"]
                                 and not r["models"][comp][sc]["correct"]],
                       "regressions": [dict(id=r["id"], condition=r["condition"], true=r["true"],
                                            pred=r["models"][cand][sc]["pred"])
                                       for r in rs if r["group"] == g and not r["models"][cand][sc]["correct"]
                                       and r["models"][comp][sc]["correct"]]} for g in GROUP_N}
    return out


def evaluate(R, E, cfg, out, cfg_sha):
    expected = sorted(PRE_FILES + [f"{a}_final.pt" for a in ARMS] + ["training_arrays.npz", "TRAINING.json"])
    need(sorted(os.listdir(out)) == expected, f"refusing: {out} contents differ from {expected}")
    tr = E.read(out / "TRAINING.json")
    need(tr["status"] == TRAINED_STATUS and tr["config_sha256"] == cfg_sha, "TRAINING.json status/config")
    need(tr["preflight_sha256"] == E.sha(out / "PREFLIGHT.json"), "PREFLIGHT changed since training")
    need(set(tr["files"]) == set(expected) - {"PREFLIGHT.json", "teachers.npz", "TRAINING.json"}, "training file list")
    for n, h in tr["files"].items():
        need(E.sha(out / n) == h, f"training file changed: {n}")
    need(E.sha(out / "teachers.npz") == tr["teachers_npz_sha256"], "teachers.npz changed")
    h0 = code_hashes()
    need(tr["code_hashes"] == h0, "code changed since training")
    need(E.sha(RECEIPT) == tr["test_receipt_sha256"], "TEST_RECEIPT changed since training")
    R.verify_hashes(E, cfg)
    G = E.G
    torch.set_num_threads(R.THREADS)
    pol, ck, parent, classes, ref, _ = E.load_models()
    classes = [str(c) for c in classes]
    missing = [p for p in required_pins(R, E, pol) if p not in cfg["pins"]]
    need(not missing, f"config pins lack required inputs: {missing}")
    target_idx, neural_path = E.project18(classes)
    target_names = {classes[i] for i in target_idx}
    sds = {REF: ref["state_dict"]}
    for a in ARMS:
        sds[a] = torch.load(out / f"{a}_final.pt", map_location="cpu", weights_only=True)["state_dict"]
    E.shared_encoder(dict(parent=ck["state_dict"], **sds))       # whole encoder + buffers bitwise equal
    models = [REF] + ARMS
    heads = {a: head_np(sds[a]) for a in models}
    checks = {"all_non_head_tensors_bitwise_equal_parent": True, "project18_module_sha256": E.sha(neural_path)}

    # ---- regression groups from cached pooled features (encoder frozen; no encoder run)
    meta, old = E.read(E.META), E.read(E.OLD_RUN / "evaluation.json")
    oldmap = {(r["group"], r["id"], r.get("condition", "intact")): r for r in old["rows"]}
    scores, pooled, diffs = {}, {}, {}
    with np.load(E.OLD_RUN / "evaluation_arrays.npz", allow_pickle=False) as ev:
        need([str(c) for c in ev["classes"].tolist()] == classes, "evaluation_arrays classes")
        for g in GROUP_N:
            pooled[g] = ev[g + "__control__pooled"].copy()
            for a in models:
                scores[g, a] = make_teacher(pooled[g], *heads[a])     # same float64 affine as the teacher
            diffs[g] = E.maxabs(scores[g, REF], ev[g + "__control__scores"])
            need(diffs[g] <= TOL_832, f"{g}: 832 scores not reproduced ({diffs[g]})")
    checks["ref832_cached_scores_max_abs"] = diffs
    rows = []
    for g, n in GROUP_N.items():
        need(len(meta[g]) == len(pooled[g]) == n, f"{g}: metadata/pooled count != {n}")
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
                   for a in models} for g in GROUP_N}
    changes = {}
    for f in range(N_FOLDS):
        t, c = TEACHER[f], CONTROL[f]
        for cand, comp in ((t, c), (t, REF), (c, REF)):
            changes[f"{cand}_vs_{comp}"] = changes_between(rows, cand, comp)

    # ---- KSL 36 from cached 832 pooled (encoder-equal); OOF by new.folds only
    new = E.load_new()
    need([str(c) for c in new["classes"].tolist()] == classes, "new classes differ")
    ids, y, folds, exposed = new["id_list"], new["y"], new["folds"], new["exposed"].astype(bool)
    need(len(ids) == 36 and y.shape == folds.shape == exposed.shape == (36,), "new36 shapes")
    need([int((folds == f).sum()) for f in range(N_FOLDS)] == list(FOLD_HELD), "new36 fold sizes")
    d_pool = E.maxabs(G.predict_prefix(parent, new["prefixes"])["pooled"], new["pooled_832"])
    need(d_pool <= TOL_NEW, f"new pooled features not reproduced ({d_pool})")
    ns = {a: make_teacher(new["pooled_832"], *heads[a]) for a in models}
    d_new = E.maxabs(ns[REF], new["logits_cached"])
    need(d_new <= TOL_NEW, f"832 logits on new clips not reproduced ({d_new})")
    need(bool((ns[REF].argmax(1) == new["logits_cached"].argmax(1)).all()), "832 new-clip predictions differ")
    checks.update(new_pooled_max_abs=d_pool, new_ref832_vs_cached_logits_max_abs=d_new)
    for kind, arms in (("control", CONTROL), ("current_teacher", TEACHER)):
        ns[f"{kind}_oof"] = np.stack([ns[arms[int(f)]][i] for i, f in enumerate(folds)])
    view = [REF, "control_oof", "current_teacher_oof"]
    preds = {sc: {a: (np.array(inds)[s[:, inds].argmax(1)] if inds is not None else None) for a, s in ns.items()}
             for sc, inds in (("full93", list(range(len(classes)))),
                              ("project18", target_idx if all(classes[c] in target_names for c in set(y.tolist()))
                               else None))}
    subsets = {"primary34": ~exposed, "previously_exposed2": exposed, "all36_supplementary": np.ones(len(ids), bool)}
    need(int((~exposed).sum()) == 34, "primary subset must hold 34 clips")
    new_summary = {k: {a: {sc: (E.clip_stats(y, preds[sc][a], msk, classes) if preds[sc][a] is not None
                                else "labels not in PROJECT_18") for sc in preds} for a in view}
                   for k, msk in subsets.items()}
    per_fold = {}
    for f in range(N_FOLDS):
        held, arms = folds == f, (TEACHER[f], CONTROL[f], REF)
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
    clip_changes = {f"current_teacher_oof_vs_{c}": {
        "gains": [r["id"] for r in clips if r["correct"]["current_teacher_oof"] and not r["correct"][c]],
        "losses": [r["id"] for r in clips if not r["correct"]["current_teacher_oof"] and r["correct"][c]]}
        for c in (REF, "control_oof")}

    # ---- predeclared gates (class thresholds from primary34 class counts, not names)
    prim = new_summary["primary34"]
    to, co = prim["current_teacher_oof"]["full93"], prim["control_oof"]["full93"]
    counts = {k: v["n"] for k, v in to["per_class"].items()}
    need(sorted(counts.values()) == [16, 18], f"primary34 class counts not 18/16: {counts}")
    need(set(counts) == {classes[c] for c in sorted(set(y.tolist()))}, "primary34 classes differ from new y")
    threshold = {k: (17 if n == 18 else 14) for k, n in counts.items()}
    gates = {"primary": {
        "teacher_total_ge_control": to["correct"] >= co["correct"],
        f"teacher_total_ge_{GATE_PRIMARY_TOTAL}": to["correct"] >= GATE_PRIMARY_TOTAL,
        **{f"class_{k}_ge_{threshold[k]}_of_{counts[k]}": to["per_class"][k]["correct"] >= threshold[k] for k in counts},
        **{f"class_{k}_ge_control": to["per_class"][k]["correct"] >= co["per_class"][k]["correct"] for k in counts}},
        "retention": {}, "annotation_minimum": {}}
    for f in range(N_FOLDS):
        for comp in (REF, CONTROL[f]):
            gates["retention"][f"{TEACHER[f]}_vs_{comp}"] = E.retention(summary, TEACHER[f], comp)
        gates["annotation_minimum"][f"{TEACHER[f]}_annotation_dev_full93_ge_{ANNOTATION_MIN}"] = \
            summary["annotation_dev"][TEACHER[f]]["intact"]["full93"]["correct"] >= ANNOTATION_MIN
    info_retention = {f"{CONTROL[f]}_vs_{REF}": E.retention(summary, CONTROL[f], REF) for f in range(N_FOLDS)}
    passed = (all(gates["primary"].values()) and all(all(v.values()) for v in gates["retention"].values())
              and all(gates["annotation_minimum"].values()))
    decision = PASS if passed else REJECT

    arrays = {"classes": np.array(classes), "new__ids": np.array(ids), "new__y": y, "new__folds": folds,
              "new__exposed": exposed, "new__pooled": new["pooled_832"]}
    for a, s in ns.items():
        arrays[f"new__{a}__scores"] = s
    for g in GROUP_N:
        arrays[f"{g}__ids"] = np.array([str(s["id"]) for s in meta[g]])
        arrays[f"{g}__conditions"] = np.array([str(s.get("condition", "intact")) for s in meta[g]])
        arrays[f"{g}__y"] = np.array([classes.index(E.label_of(s)) for s in meta[g]], dtype=np.int64)
        for a in models:
            arrays[f"{g}__{a}__scores"] = scores[g, a]
    E.save_npz(out / "evaluation_arrays.npz", arrays)
    guard(R, E, cfg, h0)
    E.write_json(out / "EVALUATION.json", {
        "status": "LOCAL_DEVELOPMENT_EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT", "decision": decision,
        "production_ready": False, "default_promoted": False, "gates": gates,
        "informational_retention_not_gated": info_retention, "class_thresholds": threshold, "checks": checks,
        "reference_annotation_dev_full93_correct": summary["annotation_dev"][REF]["intact"]["full93"]["correct"],
        "new36": {"summary": new_summary, "per_fold": per_fold, "clips": clips, "changes": clip_changes},
        "regression": {"summary": summary, "changes": changes}, "rows": rows,
        "control_vs_old_mp_support_candidate": tr["control_vs_old_mp_support_candidate"],
        "current_teacher_vs_control_training": tr["current_teacher_vs_control"],
        "claims": {
            "implemented": "both arms via existing E.T.train_arm; only teacher_base_logits differs; head-only; "
                           "b8 start/anchor; cached-feature float64 affine evaluation",
            "tested": {"scope": "synthetic unit tests only (test_teacher.py); no real data in tests",
                       "test_receipt_sha256": tr["test_receipt_sha256"]},
            "actual_performance": {"decision": decision, "basis": "this run on local development groups only"}},
        "created_utc": now().isoformat(), "arrays_sha256": E.sha(out / "evaluation_arrays.npz"),
        "training_record_sha256": E.sha(out / "TRAINING.json"), "config_sha256": cfg_sha,
        "code_hashes": h0, "parent_sha256": E.PARENT_SHA, "reference_sha256": E.REF_SHA,
        "best_fold_selected": False, "retrained_on_all36": False, "hyperparameter_search": False,
        "teacher_blend": False,
        "limits": [
            "local development comparison only; not independent signer validation or new-person generalization",
            "primary34 is the repeatedly analysed development subset excluding the two older-exposed 00 clips; "
            "it is not pristine unseen data; previously_exposed2 and all36 are reported separately",
            "each clip is excluded only from its own fold's training; signer identity unknown",
            "public/mp share ids with the fixed training base (pre-existing, documented); they are training-exposed "
            "regression references, not an independent test (see PREFLIGHT base_rows.id_overlap)",
            "project18 scope is a narrowed ranking and is not counted as full93 success",
            "a teacher-only difference does not establish the cause of earlier retention failures; not a bug fix",
            "centred squared-logit replay loss is related to, not a reproduction of, Learning without Forgetting",
            "nothing is production ready and the default model is not replaced"]})
    print("decision:", decision)
    for k in subsets:
        print(k, {a: new_summary[k][a]["full93"]["correct"] for a in view})


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["required-pins", "preflight", "train", "evaluate"])
    ap.add_argument("--config")
    a = ap.parse_args(argv)
    if a.command == "required-pins":
        R, E = load_modules()
        pol = E.load_models()[0]
        print(json.dumps(required_pins(R, E, pol), ensure_ascii=False, indent=2))
        return
    need(a.config, "--config is required")
    out = None
    try:
        R, E = load_modules()
        cfg, out, cfg_sha = load_cfg(R, a.config)
        {"preflight": preflight, "train": train, "evaluate": evaluate}[a.command](R, E, cfg, out, cfg_sha)
    except BaseException as e:
        where = out if a.command != "preflight" and out is not None and out.is_dir() else HERE
        t = now()
        p = where / f"FAILED_{a.command}_{t.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        with p.open("x", encoding="utf-8") as fh:
            json.dump({"command": a.command, "created_utc": t.isoformat(), "error": repr(e),
                       "traceback": traceback.format_exc(), "note": "first failure preserved; do not rerun to improve scores"},
                      fh, ensure_ascii=False, indent=2)
        raise


if __name__ == "__main__":
    main()
