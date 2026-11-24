"""Local DEVELOPMENT comparison (not independent signer validation):
head-only control  vs  control + 0.125*CE on added eat/study inputs, 3 bookkeeping-bucket folds.

    python experiment.py prepare    # verify pins/provenance, reproduce reference, freeze manifest (no training)
    python experiment.py train      # requires PREPARED + TEST_RECEIPT; trains control + 3 fold candidates
    python experiment.py evaluate   # requires TRAINING.json; regression + OOF evaluation, gates

Writes only to  <this ops folder>/prepared, <this ops folder>/FAILED_*.json  and
FRONT/runs/target_input_adaptation_v1. Never overwrites. Nothing is promoted.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import platform
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FRONT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(FRONT), str(FRONT / "src"), str(FRONT.parent / "sign_model_pilot_v1" / "src")]
import train_addition as T  # noqa: E402

G = T.G
PRIOR = FRONT / "ops" / "partial_encoder_pair_20260922"
CACHE = PRIOR / "train_prefixes.npz"
CACHE_RECEIPT = PRIOR / "TRAIN_PREFIX_READY.json"
POLICY = PRIOR / "comparison_policy.json"
PROBE_OPS = FRONT / "ops" / "ksl77_development_probe_20260926"
PROBE_RUN = FRONT / "runs" / "ksl77_development_probe_v1"
OLD_RUN = FRONT / "runs" / "partial_encoder_pair_v1"
REF_CKPT = OLD_RUN / "checkpoint_control_final.pt"
META = FRONT / "ops" / "context_training_pair_20260921" / "feature_metadata.json"
PREP = HERE / "prepared"
RUN = FRONT / "runs" / "target_input_adaptation_v1"
RECEIPT = HERE / "TEST_RECEIPT.json"
SCRIPTS = ("experiment.py", "train_addition.py", "test_addition.py")
PARENT_SHA = "b8d6045c0c154697a5d6d3440927a48cf1e9db3146421db0a920da9fa684870d"
REF_SHA = "83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91"
GROUPS = ["annotation_dev", "whole_dev", "public", "mp"]
ARMS = T.RECIPE["arms"]
REF = "ref832"
TOL = 1e-5
PREPARED_STATUS = "PREPARED_AND_FROZEN_BEFORE_ANY_TRAINING"
TRAINED_STATUS = "TRAINING_COMPLETE_CHECKPOINTS_FROZEN_BEFORE_EVALUATION"


# --------------------------------------------------------------------------- #
# small IO helpers
# --------------------------------------------------------------------------- #
def need(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def plain(v):
    """JSON-safe copy; non-finite floats (e.g. absent extra-loss columns of control) become null."""
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    if isinstance(v, np.generic):
        return plain(v.item())
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [plain(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    return v


def write_json(p, v):
    with Path(p).open("x", encoding="utf-8") as f:
        json.dump(plain(v), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def save_npz(p, arrays):
    with Path(p).open("xb") as f:
        np.savez(f, **arrays)


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def rows_of(obj):
    return obj if isinstance(obj, list) else obj["rows"]


def unpack(values, offsets):
    return [values[a:b].copy() for a, b in zip(offsets[:-1], offsets[1:])]


def maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def script_hashes():
    return {n: sha(HERE / n) for n in SCRIPTS}


def label_of(spec):
    return spec.get("label", spec.get("true", spec.get("reference_label")))


def load_io():
    name = "tia_prior_local_io"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PRIOR / "local_io.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def project18(classes):
    from wis_lab.service import neural
    vals = list(neural.PROJECT_18.values()) if isinstance(neural.PROJECT_18, dict) else list(neural.PROJECT_18)
    need(len(set(vals)) == 18 and all(v in classes for v in vals), "PROJECT_18 must be 18 known class names")
    return [i for i, c in enumerate(classes) if c in set(vals)], Path(neural.__file__).resolve()


def shared_encoder(state_dicts):
    names = list(state_dicts)
    ref = state_dicts[names[0]]
    for n in names[1:]:
        sd = state_dicts[n]
        need(set(sd) == set(ref), f"{n}: state_dict keys differ")
        for k in ref:
            if not k.startswith(T.HEAD):
                need(torch.equal(sd[k], ref[k]), f"{n}: non-head tensor {k} differs from {names[0]}")
    return True


def load_models():
    need(T.PRIOR_GENERATED == (PRIOR / "generated_v1.py").resolve(), "generated_v1 path override is not allowed here")
    io = load_io()
    pol = io.policy()                                        # verifies the existing pins
    ck, parent, classes = io.parent_model(pol)
    torch.set_num_threads(T.RECIPE["cpu_threads"])
    need(sha(pol["parent"]["path"]) == PARENT_SHA, "parent checkpoint sha256 mismatch")
    need(sha(REF_CKPT) == REF_SHA, "reference 832 checkpoint sha256 mismatch")
    ref = torch.load(REF_CKPT, map_location="cpu", weights_only=True)
    m832 = copy.deepcopy(parent)
    m832.load_state_dict(ref["state_dict"], strict=True)
    m832.eval()
    shared_encoder({"parent": ck["state_dict"], REF: ref["state_dict"]})
    return pol, ck, parent, list(classes), ref, m832


def load_prior_cache(classes):
    with np.load(CACHE, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    need(d["classes"].tolist() == classes, "prior cache classes differ from parent")
    base, ctx = unpack(d["base_prefix"], d["base_offsets"]), unpack(d["pair_context_prefix"], d["pair_context_offsets"])
    need(len(base) == 2122 and d["base_y"].shape == (2122,), "prior base count != 2122")
    need(len(ctx) == 224 and d["pair_context_y"].shape == (224,), "prior pair_context count != 224")
    need(d["teacher_base_logits"].shape == (2122, len(classes)), "teacher shape")
    need(d["base_indices"].shape == d["context_indices"].shape == (T.RECIPE["steps"], 16), "schedule shape")
    for k in ("base_y", "pair_context_y", "base_indices", "context_indices"):
        need(d[k].dtype == np.int64, f"{k} must be int64")
    return (base, d["base_y"], ctx, d["pair_context_y"], d["teacher_base_logits"],
            d["base_indices"], d["context_indices"])


def approved_training_row(r):
    """Approved old training rows: signer 1..9 or NIKL public; never reserved/user/dev/test roles."""
    role = " ".join(str(r.get(k, "")).lower() for k in ("role", "split", "group", "kind"))
    if any(w in role for w in ("reserved", "user", "dev", "test", "eval", "holdout")):
        return False
    sid = r.get("signer_id")
    if sid is not None and str(sid).isdigit() and 1 <= int(sid) <= 9:
        return True
    origin = " ".join(str(r.get(k, "")).lower() for k in ("source", "dataset", "corpus", "origin", "id"))
    return "nikl" in origin


def load_new():
    with np.load(PREP / "new_prefixes.npz", allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    d["prefixes"] = unpack(d["prefix"], d["offsets"])
    d["id_list"] = d["ids"].tolist()
    return d


def verify_prepared():
    prep = read(PREP / "PREPARED.json")
    need(prep["status"] == PREPARED_STATUS, "PREPARED.json status")
    for n, h in prep["files"].items():
        need(sha(PREP / n) == h, f"prepared file changed: {n}")
    pins = read(PREP / "PINS.json")
    for p, h in pins["inputs"].items():
        need(sha(p) == h, f"pinned input changed: {p}")
    need(script_hashes() == pins["scripts"] == prep["scripts"], "scripts changed since prepare")
    need(read(PREP / "RECIPE.json") == plain(T.RECIPE), "frozen recipe differs from code recipe")
    return prep, pins


def verify_receipt():
    r = read(RECEIPT)
    need(r["status"] == "PASSED", "test receipt not PASSED")
    need(r["scripts"] == script_hashes(), "test receipt script hashes differ from current scripts")
    need(r["prior_generated_sha256"] == sha(PRIOR / "generated_v1.py"), "test receipt generated_v1 hash differs")
    return sha(RECEIPT)


def full_logits(model, x):
    """MaskedSignTCN.forward(x[B,T,280], pad_mask[B,T] float32, 1 = real frame); single clip, all frames real."""
    xt = torch.from_numpy(x).unsqueeze(0)
    pad_mask = torch.ones(xt.shape[:2], dtype=torch.float32)
    with torch.no_grad():
        out = model(xt, pad_mask)
    if isinstance(out, dict):
        out = out["logits"]
    elif isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().float().numpy()[0]


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def prepare():
    need(not PREP.exists(), f"refusing to overwrite {PREP}")
    pol, ck, parent, classes, ref, m832 = load_models()
    width, in_ch, ncls = G._inspect_model(parent)
    need(ncls == len(classes) == 93, "expected 93 classes")
    target_idx, neural_path = project18(classes)
    inputs = {}                                               # path -> sha256, re-verified later

    def pin(p):
        p = Path(p).resolve()
        inputs[str(p)] = sha(p)
        return inputs[str(p)]

    for p in (PRIOR / "generated_v1.py", PRIOR / "local_io.py", CACHE, pol["parent"]["path"], REF_CKPT,
              OLD_RUN / "evaluation_arrays.npz", OLD_RUN / "evaluation.json", META, neural_path,
              PROBE_OPS / "FROZEN_PLAN.json", PROBE_OPS / "RESULT.json", PROBE_RUN / "logits.npz"):
        pin(p)
    if (PRIOR / "local_policy.json").is_file():
        pin(PRIOR / "local_policy.json")
    # comparison_policy.json predates train_prefixes.npz and cannot pin it; the cache is pinned by the
    # frozen TRAIN_PREFIX_READY.json receipt, which in turn pins the policy, module and metadata.
    rc = read(CACHE_RECEIPT)
    for k in ("status", "policy_sha256", "module_sha256", "cache_sha256", "metadata_sha256"):
        need(k in rc, f"TRAIN_PREFIX_READY.json lacks {k}")
    cache_receipt = {
        "receipt_sha256": pin(CACHE_RECEIPT), "receipt_status": rc["status"],
        "cache": pin(CACHE) == rc["cache_sha256"],
        "comparison_policy": pin(POLICY) == rc["policy_sha256"],
        "generated_v1": pin(PRIOR / "generated_v1.py") == rc["module_sha256"],
        "metadata": pin(META) == rc["metadata_sha256"]}
    bad = [k for k in ("cache", "comparison_policy", "generated_v1", "metadata") if not cache_receipt[k]]
    need(not bad, f"TRAIN_PREFIX_READY.json sha256 mismatch: {bad}")

    # ---- prior cache and its approved training rows
    _, by, _, cy, _, _, _ = load_prior_cache(classes)
    meta = read(META)
    roles, old_ids = {}, set()
    for key, ys in (("base", by), ("pair_context", cy)):
        rows = meta[key]
        need(len(rows) == len(ys), f"metadata {key} count mismatch")
        for r, yi in zip(rows, ys):
            need(label_of(r) == classes[int(yi)], f"metadata {key} label mismatch for {r.get('id')}")
            need(approved_training_row(r), f"metadata {key} row not approved: {r.get('id')}")
            old_ids.add(str(r.get("id")))
        roles[key] = dict(Counter(str(r.get("role", r.get("split"))) for r in rows))

    # ---- the 36 new clips (labels from the plan; predictions never used for selection/split)
    plan = rows_of(read(PROBE_OPS / "FROZEN_PLAN.json"))
    result = {r["id"]: r for r in rows_of(read(PROBE_OPS / "RESULT.json"))}
    ids = [str(r["id"]) for r in plan]
    need(len(plan) == 36 and len(set(ids)) == 36, "FROZEN_PLAN must hold exactly 36 unique ids")
    need(not (set(ids) & old_ids), "a new id already appears in the prior training metadata")
    labels = [r["candidate_label"] for r in plan]
    need(all(lab in classes for lab in labels), "candidate_label not in classes")
    counts = Counter(labels)
    need(sorted(counts.values()) == [17, 19], f"expected 17/19 clips per class, got {dict(counts)}")
    xs, input_hashes, bases, frames = [], [], [], []
    for r in plan:
        need(r.get("signer_id") is None and r.get("usage_terms") == "UNKNOWN", f"{r['id']}: signer/usage terms changed")
        need(isinstance(r.get("previous_diagnostic_exposure"), bool), f"{r['id']}: exposure flag must be bool")
        src = Path(r["path"]) if Path(r["path"]).is_absolute() else FRONT / r["path"]
        need(pin(src) == r["source_sha256"], f"{r['id']}: source video sha256 mismatch")
        p = (PROBE_RUN / r["id"] / "model_input.npz").resolve()
        need(p.parent.parent == PROBE_RUN.resolve(), f"{r['id']}: unexpected input path")
        with np.load(p, allow_pickle=False) as z:
            x = z["x"].copy()
        need(x.dtype == np.float32 and x.ndim == 2 and x.shape[0] >= 1 and x.shape[1] == in_ch == 280,
             f"{r['id']}: bad input shape/dtype {x.shape} {x.dtype}")
        need(bool(np.isfinite(x).all()), f"{r['id']}: non-finite input")
        fh, xh, want = pin(p), hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(), result[r["id"]]["input_sha256"]
        bases.append("file" if fh == want else "x_bytes" if xh == want else None)
        need(bases[-1] is not None, f"{r['id']}: model_input does not match RESULT input_sha256")
        xs.append(x)
        input_hashes.append(fh)
        frames.append(int(x.shape[0]))
    need(len(set(bases)) == 1, "inconsistent input hash basis across rows")

    y = np.array([classes.index(lab) for lab in labels], dtype=np.int64)
    buckets, folds, order = T.assign_folds([r["member"] for r in plan])
    split = T.check_split(ids, y, folds, buckets,
                          {"source_sha256": [r["source_sha256"] for r in plan], "input_sha256": input_hashes})
    exposed = np.array([r["previous_diagnostic_exposure"] for r in plan], dtype=bool)
    need(int(exposed.sum()) == 2 and {buckets[i] for i in np.flatnonzero(exposed)} == {"00"},
         "expected exactly the two previously exposed 00 clips")

    # ---- reference 832 reproduction: individual full inputs and batched prefix path
    with np.load(PROBE_RUN / "logits.npz", allow_pickle=False) as z:
        lids, lcls, lg = z["ids"].tolist(), z["classes"].tolist(), z["logits"].copy()
    need(lcls == classes and len(set(lids)) == len(lids), "logits.npz classes/ids")
    cached = np.stack([lg[lids.index(i)] for i in ids]).astype(np.float32)
    full = np.stack([full_logits(m832, x) for x in xs]).astype(np.float32)
    prefixes = G.extract_prefix(m832, xs)                    # encoder == parent encoder (asserted)
    pr = G.predict_prefix(m832, prefixes, batch_size=32)
    w, b = ref["state_dict"]["head.weight"].numpy(), ref["state_dict"]["head.bias"].numpy()
    affine = pr["pooled"].astype(np.float64) @ w.T.astype(np.float64) + b.astype(np.float64)
    diffs = {"full_vs_cached": maxabs(full, cached), "prefix_batched_vs_cached": maxabs(pr["logits"], cached),
             "affine_vs_prefix": maxabs(affine, pr["logits"])}
    need(all(v <= TOL for v in diffs.values()), f"reference reproduction failed: {diffs}")
    pred_labels = [classes[int(k)] for k in cached.argmax(1)]
    recorded = [(i, result[i].get("prediction")) for i in ids if isinstance(result[i].get("prediction"), str)]
    need(all(p == pred_labels[ids.index(i)] for i, p in recorded), "RESULT predictions not reproduced")

    # ---- freeze (all checks passed; nothing has been written until here)
    PREP.mkdir(parents=False, exist_ok=False)
    offsets = np.cumsum([0] + [len(p) for p in prefixes], dtype=np.int64)
    save_npz(PREP / "new_prefixes.npz", {
        "classes": np.array(classes), "ids": np.array(ids), "labels": np.array(labels), "y": y, "folds": folds,
        "buckets": np.array(buckets), "exposed": exposed, "frames": np.array(frames, dtype=np.int64),
        "prefix": np.concatenate(prefixes, axis=0).astype(np.float32), "offsets": offsets,
        "pooled_832": pr["pooled"], "logits_832_prefix": pr["logits"], "logits_832_full": full,
        "logits_cached": cached})
    manifest_rows = [dict(id=i, member=r["member"], bucket=buckets[k], fold=int(folds[k]),
                          candidate_label=labels[k], label_index=int(y[k]),
                          previous_diagnostic_exposure=bool(exposed[k]), signer_id=None, usage_terms="UNKNOWN",
                          source_path=str(r["path"]), source_sha256=r["source_sha256"],
                          input_path=str((PROBE_RUN / i / "model_input.npz").resolve()),
                          input_file_sha256=input_hashes[k], frames=frames[k], plan_row=r)
                     for k, (i, r) in enumerate(zip(ids, plan))]
    write_json(PREP / "MANIFEST.json", {
        "rows": manifest_rows, "bucket_order": order, "fold_rule": T.RECIPE["fold_rule"], "split_checks": split,
        "label_note": "candidate_label copied from FROZEN_PLAN; source-provided, NOT expert annotation; "
                      "never relabelled; predictions never used for selection or split",
        "bucket_note": "filename-prefix bucket is conservative bookkeeping, NOT a confirmed signer identity",
        "exposure_note": "all 36 clips are already development-exposed (ksl77 development probe result); "
                         "previous_diagnostic_exposure only separates the 2 older-exposed 00 clips from the 34 "
                         "newer ones (technical name 'primary_unexposed34'), NOT pristine unseen data; each "
                         "evaluated clip is excluded from its own fold's candidate TRAINING only",
        "input_hash_basis": bases[0], "class_counts": dict(counts)})
    write_json(PREP / "RECIPE.json", T.RECIPE)
    write_json(PREP / "PINS.json", {"inputs": inputs, "scripts": script_hashes(),
                                    "parent_sha256": PARENT_SHA, "reference_sha256": REF_SHA,
                                    "prior_generated_sha256": T.prior_sha256()})
    files = {n: sha(PREP / n) for n in ("new_prefixes.npz", "MANIFEST.json", "RECIPE.json", "PINS.json")}
    write_json(PREP / "PREPARED.json", {
        "status": PREPARED_STATUS, "created_utc": now().isoformat(), "files": files, "scripts": script_hashes(),
        "checks": {"policy_pins_verified": True, "cache_verified_by_train_prefix_receipt": cache_receipt,
                   "encoder_parent_equals_832": True, "reference_reproduction_max_abs": diffs,
                   "result_predictions_reproduced": len(recorded), "prior_training_roles": roles,
                   "prior_counts": {"base": 2122, "pair_context": 224, "schedule": [T.RECIPE["steps"], 16]},
                   "split": split},
        "counts": {"n": 36, "classes": dict(counts), "exposed": int(exposed.sum()), "folds":
                   {f: int((folds == f).sum()) for f in range(3)}},
        "no_training_performed": True, "torch": torch.__version__, "threads": torch.get_num_threads()})
    print("prepared:", PREP)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train():
    need(not RUN.exists(), f"refusing to overwrite {RUN}")
    prep, _ = verify_prepared()
    receipt_sha = verify_receipt()
    pol, ck, parent, classes, ref, _ = load_models()
    base, by, ctx, cy, teacher, bidx, cidx = load_prior_cache(classes)
    new = load_new()
    need(new["classes"].tolist() == classes, "prepared classes differ")
    ids, y, folds = new["id_list"], new["y"], new["folds"]
    class_order = sorted(set(y.tolist()))
    steps, per = T.RECIPE["steps"], T.RECIPE["extra_per_class"]
    RUN.mkdir(parents=False, exist_ok=False)

    def progress(arm, done, total):
        print(f"[{now().isoformat()}] {arm} {done}/{total}", flush=True)

    common = (parent, base, by, ctx, cy, teacher, bidx, cidx)
    res = {"control": T.train_arm(*common, arm="control", progress=progress, **T.TRAIN_KW)}
    fold_plan = {}
    for f in range(T.N_FOLDS):
        arm = f"candidate_fold{f}"
        groups = T.fold_training_rows(ids, y, folds, f, class_order)
        rows = np.array([i for g in groups for i in g], dtype=np.int64)
        held = np.flatnonzero(folds == f)
        sched = T.cyclic_schedule([len(g) for g in groups], steps, per)
        need(not np.isin(rows, held).any() and not np.isin(rows[sched], held).any(), f"{arm}: held-out leakage")
        # the learner receives ONLY this fold's training rows
        res[arm] = T.train_arm(*common, extra_prefixes=[new["prefixes"][i] for i in rows], extra_y=y[rows].copy(),
                               extra_indices=sched, extra_weight=T.RECIPE["extra_weight"], arm=arm,
                               progress=progress, **T.TRAIN_KW)
        fold_plan[arm] = {"fold": f, "train_rows": rows, "heldout_rows": held, "schedule": sched}

    psd, rsd = ck["state_dict"], ref["state_dict"]
    checks = {}
    for arm, r in res.items():
        sd = r["state_dict"]
        cols = [0, 3, 4, 5] if arm == "control" else list(range(len(T.LOSS_COLUMNS)))
        checks[arm] = {
            "keys_equal_parent": set(sd) == set(psd),
            "non_head_bitwise_equal_parent": all(torch.equal(sd[k], psd[k]) for k in sd if not k.startswith(T.HEAD)),
            "head_changed": any(not torch.equal(sd[k], psd[k]) for k in sd if k.startswith(T.HEAD)),
            "trainable_head_only": all(n.startswith(T.HEAD) for n in r["trainable_names"]),
            "frozen_parameters_unchanged": r["frozen_parameters_unchanged"], "buffers_unchanged": r["buffers_unchanged"],
            "old_schedule_identical": bool(np.array_equal(r["base_indices"], bidx)
                                           and np.array_equal(r["context_indices"], cidx)),
            "losses_finite": bool(np.isfinite(r["losses"][:, cols]).all()),
            "updates": r["updates"] == steps}
        need(all(checks[arm].values()), f"{arm}: training checks failed {checks[arm]}")
    csd = res["control"]["state_dict"]
    control_vs_832 = {"bitwise_equal_all_tensors": all(torch.equal(csd[k], rsd[k]) for k in csd),
                      "max_abs_head_diff": max(float((csd[k] - rsd[k]).abs().max()) for k in csd if k.startswith(T.HEAD)),
                      "note": "equality not required; control is evaluated as its own arm"}
    verify_prepared()                                          # pins unchanged after training

    files, arrays = {}, {"loss_columns": np.array(T.LOSS_COLUMNS), "base_indices": bidx, "context_indices": cidx}
    extra_meta = {k: v for k, v in ck.items() if k != "state_dict"}
    for arm, r in res.items():
        fp = fold_plan.get(arm)
        info = {"arm": arm, "parent_sha256": PARENT_SHA, "classes": classes, "recipe": plain(T.RECIPE),
                "prepared_sha256": sha(PREP / "PREPARED.json"), "fold": None if fp is None else fp["fold"],
                "training_ids": [] if fp is None else [ids[i] for i in fp["train_rows"]],
                "heldout_ids": [] if fp is None else [ids[i] for i in fp["heldout_rows"]]}
        p = RUN / f"{arm}_final.pt"
        with p.open("xb") as fh:
            torch.save(dict(extra_meta, state_dict=r["state_dict"], target_input_adaptation=info), fh)
        files[p.name] = sha(p)
        arrays[f"losses__{arm}"] = r["losses"]
        if fp is not None:
            arrays[f"extra_schedule__{arm}"] = fp["schedule"]
            arrays[f"train_rows__{arm}"] = fp["train_rows"]
            arrays[f"heldout_rows__{arm}"] = fp["heldout_rows"]
    save_npz(RUN / "training_arrays.npz", arrays)
    files["training_arrays.npz"] = sha(RUN / "training_arrays.npz")
    write_json(RUN / "TRAINING.json", {
        "status": TRAINED_STATUS, "created_utc": now().isoformat(), "files": files,
        "prepared_sha256": sha(PREP / "PREPARED.json"), "test_receipt_sha256": receipt_sha,
        "scripts": script_hashes(), "recipe": T.RECIPE, "train_kwargs": T.TRAIN_KW,
        "parent_sha256": PARENT_SHA, "reference_sha256": REF_SHA, "classes": classes,
        "checks": checks, "control_vs_832": control_vs_832,
        "loss_first_last": {a: {"first": r["losses"][0], "last": r["losses"][-1]} for a, r in res.items()},
        "folds": {a: {"fold": fp["fold"], "n_train": len(fp["train_rows"]), "n_heldout": len(fp["heldout_rows"]),
                      "heldout_ids": [ids[i] for i in fp["heldout_rows"]]} for a, fp in fold_plan.items()},
        "threads": torch.get_num_threads(), "torch": torch.__version__, "python": platform.python_version(),
        "pins_verified_before_and_after": True, "evaluation_performed": False, "reruns": 0})
    print("training frozen:", RUN)


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def summarize(rows, arm, scope):
    """Same counting as the prior evaluate scripts."""
    rs = [r for r in rows if scope in r["models"][arm]]
    labels = sorted({r["true"] for r in rs})
    detail, recalls, f1s = {}, [], []
    for label in labels:
        support = sum(r["true"] == label for r in rs)
        tp = sum(r["true"] == label and r["models"][arm][scope]["pred"] == label for r in rs)
        predicted = sum(r["models"][arm][scope]["pred"] == label for r in rs)
        detail[label] = dict(n=support, correct=tp, predicted=predicted, recall=tp / support,
                             precision=tp / predicted if predicted else 0., f1=2 * tp / (support + predicted))
        recalls.append(tp / support)
        f1s.append(2 * tp / (support + predicted))
    people = {str(s): dict(n=sum(r.get("signer_id") == s for r in rs),
                           correct=sum(r.get("signer_id") == s and r["models"][arm][scope]["correct"] for r in rs))
              for s in sorted({r["signer_id"] for r in rs if "signer_id" in r}, key=str)}
    return dict(n=len(rs), correct=sum(r["models"][arm][scope]["correct"] for r in rs),
                ground_truth_classes=len(labels),
                macro_recall=float(np.mean(recalls)) if recalls else None,
                macro_f1=float(np.mean(f1s)) if f1s else None, per_class=detail, per_signer=people)


def score_entry(s, yi, inds, classes):
    order = sorted(inds, key=lambda j: -float(s[j]))
    rival = next(j for j in order if j != yi)
    return dict(pred=classes[order[0]], correct=bool(order[0] == yi), rank=order.index(yi) + 1,
                margin=float(s[yi] - s[rival]))


def retention(summary, cand, comp):
    def s(g, arm, cond="intact", scope="full93"):
        return summary[g][arm][cond][scope]
    a, c = s("annotation_dev", cand), s("annotation_dev", comp)
    gates = {
        "annotation_total": a["correct"] >= c["correct"],
        "annotation_macro_recall": a["macro_recall"] >= c["macro_recall"] - 1e-12,
        "annotation_signers_11_12_13": all(a["per_signer"][str(k)]["correct"] >= c["per_signer"][str(k)]["correct"]
                                           for k in (11, 12, 13)),
        "annotation_project18": s("annotation_dev", cand, scope="project18")["correct"]
                                >= s("annotation_dev", comp, scope="project18")["correct"],
        "whole_no_loss": all(s("whole_dev", cand, scope=sc)["correct"] >= s("whole_dev", comp, scope=sc)["correct"]
                             for sc in ("full93", "project18")),
    }
    for g in ("public", "mp"):
        for cond in sorted(summary[g][cand]):
            gates[f"{g}_no_loss_{cond}"] = all(
                s(g, cand, cond, sc)["correct"] >= s(g, comp, cond, sc)["correct"] for sc in ("full93", "project18"))
    return gates


def clip_stats(y, pred, mask, classes):
    per = {}
    for c in sorted(set(y[mask].tolist())):
        m = mask & (y == c)
        per[classes[c]] = dict(n=int(m.sum()), correct=int((pred[m] == c).sum()), recall=float((pred[m] == c).mean()))
    return dict(n=int(mask.sum()), correct=int((pred[mask] == y[mask]).sum()),
                macro_recall=float(np.mean([v["recall"] for v in per.values()])) if per else None, per_class=per)


def evaluate():
    for n in ("evaluation.json", "evaluation_arrays.npz"):
        need(not (RUN / n).exists(), f"refusing to overwrite {RUN / n}")
    verify_prepared()
    tr = read(RUN / "TRAINING.json")
    need(tr["status"] == TRAINED_STATUS and tr["prepared_sha256"] == sha(PREP / "PREPARED.json"), "TRAINING.json")
    for n, h in tr["files"].items():
        need(sha(RUN / n) == h, f"training file changed: {n}")
    need(tr["scripts"] == script_hashes(), "scripts changed since training")
    pol, ck, parent, classes, ref, _ = load_models()
    target_idx, _ = project18(classes)
    target_names = {classes[i] for i in target_idx}
    sds = {REF: ref["state_dict"]}
    for arm in ARMS:
        sds[arm] = torch.load(RUN / f"{arm}_final.pt", map_location="cpu", weights_only=True)["state_dict"]
    shared_encoder(dict(parent=ck["state_dict"], **sds))
    models = [REF] + ARMS
    heads = {a: (sds[a]["head.weight"].numpy().astype(np.float64), sds[a]["head.bias"].numpy().astype(np.float64))
             for a in models}
    checks = {"all_non_head_tensors_bitwise_equal": True}

    # ---- regression groups from frozen pooled caches (no encoder run)
    meta = read(META)
    old = read(OLD_RUN / "evaluation.json")
    oldmap = {(r["group"], r["id"], r.get("condition", "intact")): r for r in old["rows"]}
    scores, pooled, diffs = {}, {}, {}
    with np.load(OLD_RUN / "evaluation_arrays.npz", allow_pickle=False) as ev:
        need(ev["classes"].tolist() == classes, "evaluation_arrays classes")
        for g in GROUPS:
            pooled[g] = ev[g + "__control__pooled"].copy()
            for a in models:
                scores[g, a] = pooled[g].astype(np.float64) @ heads[a][0].T + heads[a][1]
            diffs[g] = maxabs(scores[g, REF], ev[g + "__control__scores"])
            need(diffs[g] <= 1e-6, f"{g}: 832 scores not reproduced ({diffs[g]})")
    checks["ref832_cached_scores_max_abs"] = diffs
    rows = []
    for g in GROUPS:
        need(len(meta[g]) == len(pooled[g]), f"{g}: metadata/pooled count mismatch")
        for i, spec in enumerate(meta[g]):
            label, cond = label_of(spec), spec.get("condition", "intact")
            yi = classes.index(label)
            row = dict(group=g, id=spec["id"], condition=cond, true=label, models={})
            if "signer_id" in spec:
                row["signer_id"] = spec["signer_id"]
            for a in models:
                row["models"][a] = {sc: score_entry(scores[g, a][i], yi, inds, classes)
                                    for sc, inds in (("full93", list(range(len(classes)))), ("project18", target_idx))
                                    if yi in inds}
            oldrow = oldmap[g, spec["id"], cond]
            need(oldrow["true"] == label, f"{g}/{spec['id']}: label differs from cached evaluation")
            for sc, e in row["models"][REF].items():
                o = oldrow["models"]["control"][sc]
                need(e["pred"] == o["pred"] and e["correct"] == o["correct"], f"{g}/{spec['id']}: 832 prediction differs")
            rows.append(row)
    checks["ref832_cached_predictions_reproduced"] = True
    summary = {g: {a: {cond: {sc: summarize([r for r in rows if r["group"] == g and r["condition"] == cond], a, sc)
                              for sc in ("full93", "project18")}
                       for cond in sorted({r["condition"] for r in rows if r["group"] == g})}
                   for a in models} for g in GROUPS}
    changes = {}
    for cand in ARMS[1:]:
        for comp in (REF, "control"):
            changes[f"{cand}_vs_{comp}"] = {g: {
                "gains": [r["id"] for r in rows if r["group"] == g and r["models"][cand]["full93"]["correct"]
                          and not r["models"][comp]["full93"]["correct"]],
                "regressions": [dict(id=r["id"], condition=r["condition"], true=r["true"],
                                     pred=r["models"][cand]["full93"]["pred"])
                                for r in rows if r["group"] == g and not r["models"][cand]["full93"]["correct"]
                                and r["models"][comp]["full93"]["correct"]]} for g in GROUPS}

    # ---- added 36 clips: frozen pooled features, affine heads, out-of-fold candidate
    new = load_new()
    ids, y, folds, exposed = new["id_list"], new["y"], new["folds"], new["exposed"].astype(bool)
    d_pool = maxabs(G.predict_prefix(parent, new["prefixes"])["pooled"], new["pooled_832"])
    need(d_pool <= TOL, f"new pooled features not reproduced ({d_pool})")
    P = new["pooled_832"].astype(np.float64)
    ns = {a: P @ heads[a][0].T + heads[a][1] for a in models}
    d_new = maxabs(ns[REF], new["logits_cached"])
    need(d_new <= TOL, f"832 logits on new clips not reproduced ({d_new})")
    checks.update(new_pooled_max_abs=d_pool, new_ref832_vs_cached_logits_max_abs=d_new)
    ns["oof"] = np.stack([ns[f"candidate_fold{int(f)}"][i] for i, f in enumerate(folds)])
    all_idx = np.arange(len(classes))
    preds = {sc: {a: (np.array(inds)[s[:, inds].argmax(1)] if inds is not None else None)
                  for a, s in ns.items()}
             for sc, inds in (("full93", list(all_idx)),
                              ("project18", target_idx if all(classes[c] in target_names for c in set(y.tolist()))
                               else None))}
    subsets = {"primary_unexposed34": ~exposed, "previously_exposed00": exposed,
               "all36_supplementary": np.ones(len(ids), dtype=bool)}
    need(int((~exposed).sum()) == 34, "primary subset must hold 34 clips")
    new_summary = {k: {a: {sc: (clip_stats(y, preds[sc][a], m, classes) if preds[sc][a] is not None
                                else "labels not in PROJECT_18") for sc in preds}
                       for a in (REF, "control", "oof")} for k, m in subsets.items()}
    per_fold = {}
    for f in range(T.N_FOLDS):
        arm, held = f"candidate_fold{f}", folds == f
        per_fold[arm] = {
            "heldout": {a: clip_stats(y, preds["full93"][a], held, classes) for a in (arm, "control", REF)},
            "heldout_unexposed": {a: clip_stats(y, preds["full93"][a], held & ~exposed, classes)
                                  for a in (arm, "control", REF)},
            "training_rows_sanity_not_generalization": {a: clip_stats(y, preds["full93"][a], ~held, classes)
                                                        for a in (arm, "control", REF)}}
    clips, fp = [], preds["full93"]
    for i, cid in enumerate(ids):
        clips.append(dict(id=cid, label=classes[int(y[i])], fold=int(folds[i]), bucket=str(new["buckets"][i]),
                          previously_exposed=bool(exposed[i]),
                          pred={a: classes[int(fp[a][i])] for a in (REF, "control", "oof")},
                          correct={a: bool(fp[a][i] == y[i]) for a in (REF, "control", "oof")},
                          margin={a: float(ns[a][i, y[i]] - np.max(np.delete(ns[a][i], y[i])))
                                  for a in (REF, "control", "oof")}))
    clip_changes = {f"oof_vs_{c}": {"gains": [r["id"] for r in clips if r["correct"]["oof"] and not r["correct"][c]],
                                    "losses": [r["id"] for r in clips if not r["correct"]["oof"] and r["correct"][c]]}
                    for c in (REF, "control")}

    # ---- predeclared gates
    prim = new_summary["primary_unexposed34"]
    gates = {"primary": {}, "retention": {}}
    for comp in (REF, "control"):
        o, c = prim["oof"]["full93"], prim[comp]["full93"]
        gates["primary"][f"macro_recall_gt_{comp}"] = o["macro_recall"] > c["macro_recall"]
        gates["primary"][f"each_class_correct_ge_{comp}"] = all(o["per_class"][k]["correct"] >= v["correct"]
                                                                for k, v in c["per_class"].items())
    for cand in ARMS[1:]:
        for comp in (REF, "control"):
            gates["retention"][f"{cand}_vs_{comp}"] = retention(summary, cand, comp)
    passed = all(gates["primary"].values()) and all(all(v.values()) for v in gates["retention"].values())
    decision = ("PASS_LOCAL_DEVELOPMENT_GATES_NOT_PRODUCTION_READY_PENDING_INDEPENDENT_AUDIT" if passed
                else "REJECTED_BY_PREDECLARED_GATES")

    arrays = {"classes": np.array(classes), "new__ids": np.array(ids), "new__y": y, "new__folds": folds,
              "new__exposed": exposed, "new__pooled": new["pooled_832"]}
    for a, s in ns.items():
        arrays[f"new__{a}__scores"] = s
    for g in GROUPS:
        for a in models:
            arrays[f"{g}__{a}__scores"] = scores[g, a]
    save_npz(RUN / "evaluation_arrays.npz", arrays)
    write_json(RUN / "evaluation.json", {
        "status": "LOCAL_DEVELOPMENT_EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT", "decision": decision,
        "production_ready": False, "default_promoted": False, "gates": gates, "checks": checks,
        "new36": {"summary": new_summary, "per_fold": per_fold, "clips": clips, "changes": clip_changes},
        "regression": {"summary": summary, "changes": changes}, "rows": rows,
        "created_utc": now().isoformat(), "arrays_sha256": sha(RUN / "evaluation_arrays.npz"),
        "training_record_sha256": sha(RUN / "TRAINING.json"), "prepared_sha256": sha(PREP / "PREPARED.json"),
        "scripts": script_hashes(), "parent_sha256": PARENT_SHA, "reference_sha256": REF_SHA,
        "encoder_runs_on_regression_caches": 0, "last_block_runs_on_new36_for_verification": len(ids),
        "best_fold_selected": False, "retrained_on_all36": False,
        "limits": [
            "local development comparison only; not independent signer validation",
            "all 36 clips were already development-exposed by the earlier ksl77 development probe; none is "
            "pristine unseen data",
            "'primary_unexposed34' is a technical subset name: previous_diagnostic_exposure separates the 2 "
            "older-exposed 00 clips from the 34 newer clips; it does not mean those 34 are unexposed",
            "each evaluated clip is excluded only from its own fold's candidate training; it is used to train "
            "the other folds' candidates and the whole set was seen in prior development",
            "source identity and signer identity are UNKNOWN; filename-prefix buckets are bookkeeping, "
            "not confirmed signers",
            "external usage terms UNKNOWN and signer_id null are retained as recorded",
            "labels are source-provided candidate labels, not expert annotations",
            "all 36 clips may share a source; a same-source two-class shortcut cannot be excluded",
            "eat/study are absent from annotation699, so regression groups only test retention",
            "annotation699 and whole74 share source recordings; public/mp use training signers",
            "no execution-correctness, performance-quality or generalization claim; nothing is production ready"]})
    print("decision:", decision)
    for k in subsets:
        print(k, {a: new_summary[k][a]["full93"]["macro_recall"] for a in (REF, "control", "oof")})


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "train", "evaluate"])
    a = ap.parse_args(argv)
    try:
        {"prepare": prepare, "train": train, "evaluate": evaluate}[a.command]()
    except BaseException as e:
        where = RUN if a.command != "prepare" and RUN.is_dir() else HERE
        t = now()
        write_json(where / f"FAILED_{a.command}_{t.strftime('%Y%m%dT%H%M%S%fZ')}.json",
                   {"command": a.command, "created_utc": t.isoformat(), "error": repr(e),
                    "traceback": traceback.format_exc(), "note": "do not rerun to improve scores"})
        raise


if __name__ == "__main__":
    main()
