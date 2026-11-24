"""Matched split-gradient control vs one-sided extra-gradient projection (FROZEN_PLAN.md).

    python run_pair.py preflight --config PREPARED_CONFIG.json   # verify + build inputs, no training
    python run_pair.py train     --config PREPARED_CONFIG.json   # needs PREFLIGHT + TEST_RECEIPT; 6 x 660 steps
    python run_pair.py evaluate  --config PREPARED_CONFIG.json   # needs TRAINING.json; cached-feature evaluation

Reads cached prefixes/schedules/pooled features only (no raw inputs, videos or user sessions).
Writes only to cfg out_dir (create-exclusive) and FAILED_*.json there or in this folder.
Never overwrites. Nothing is promoted.
"""
from __future__ import annotations

import argparse
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


def _load_local(name, filename):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / filename)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


P = _load_local("gpp_projected_train", "projected_train.py")

SUPPORT_OPS = FRONT / "ops" / "mp_support_addition_20260929"
SUPPORT_NPZ = SUPPORT_OPS / "SUPPORT_PREPARED.npz"
SUPPORT_RECEIPT = SUPPORT_OPS / "SUPPORT_PREPARED_RECEIPT.json"
OLD_RUN = FRONT / "runs" / "mp_support_addition_v1"
RECEIPT = HERE / "TEST_RECEIPT.json"
STEPS, EXTRA_W, KSL_PER_STEP, SUPPORT_PER_STEP, THREADS = 660, 0.25, 4, 4, 2
N_FOLDS = 3
FOLD_TRAIN, FOLD_HELD = (22, 26, 24), (14, 10, 12)
N_SUPPORT, N_SUPPORT_CLIPS, N_SUPPORT_LABELS = 113, 105, 69
MATCHED = [f"matched_fold{f}" for f in range(N_FOLDS)]
PROJECTED = [f"projected_fold{f}" for f in range(N_FOLDS)]
ARMS = MATCHED + PROJECTED
OLDM = [f"oldmethod_fold{f}" for f in range(N_FOLDS)]          # mp_support_addition_v1 candidate_fold*
REF = "ref832"
TOL_STEP, TOL_832, TOL_NEW = 1e-6, 1e-6, 1e-5
GATE_PRIMARY_TOTAL = 31
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


def norm_path(p):
    return str(Path(p).resolve())


def arr_sha(a):
    a =np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# config / pins
# --------------------------------------------------------------------------- #
def load_cfg(path):
    E = P.load_experiment()                                    # refuses TIA_PRIOR_GENERATED
    raw = Path(path).read_bytes()
    cfg = json.loads(raw.decode("utf-8-sig"))
    need({"root", "out_dir", "pins", "protected_files"} <= set(cfg), "config keys missing")
    need(Path(cfg["root"]).is_absolute() and Path(cfg["root"]).resolve() == FRONT, "config root != FRONT")
    out = Path(cfg["out_dir"])
    need(out.is_absolute() and out.resolve().parent == (FRONT / "runs").resolve()
         and out.name.startswith("gradient_projection_pair_v"), "out_dir must be FRONT/runs/gradient_projection_pair_v*")
    need(out.resolve() != OLD_RUN.resolve(), "out_dir equals an old run")
    cfg["pins"] = {norm_path(p): h for p, h in cfg["pins"].items()}
    cfg["protected_files"] = {norm_path(p): h for p, h in cfg["protected_files"].items()}
    return E, cfg, out.resolve(), hashlib.sha256(raw).hexdigest()


def verify_hashes(E, cfg):
    for group in ("pins", "protected_files"):
        for p, h in cfg[group].items():
            need(Path(p).is_file() and E.sha(p) == h, f"{group} sha256 mismatch: {p}")
    old_tr = E.read(OLD_RUN / "TRAINING.json")
    for n, h in old_tr["files"].items():
        need(E.sha(OLD_RUN / n) == h, f"old mp_support TRAINING file changed: {n}")
    return True


def required_pins(E, pol):
    ps = [P.OLD_EXPERIMENT, P.OLD_TRAIN, P.OLD_GENERATED, E.PRIOR / "local_io.py", E.CACHE,
          E.PREP / "new_prefixes.npz", pol["parent"]["path"], E.REF_CKPT, E.META,
          E.OLD_RUN / "evaluation_arrays.npz", E.OLD_RUN / "evaluation.json",
          SUPPORT_NPZ, SUPPORT_RECEIPT, OLD_RUN / "training_arrays.npz", OLD_RUN / "TRAINING.json"]
    ps += [OLD_RUN / f"candidate_fold{f}_final.pt" for f in range(N_FOLDS)]
    return [norm_path(p) for p in ps]


def verify_receipt():
    r = json.loads(RECEIPT.read_text(encoding="utf-8-sig"))
    need(r.get("status") == "PASSED", "TEST_RECEIPT status is not PASSED")
    need(r.get("code_hashes") == P.code_hashes(), "TEST_RECEIPT code hashes differ from current code")
    return r


# --------------------------------------------------------------------------- #
# inputs (shared by every command; read-only)
# --------------------------------------------------------------------------- #
def load_inputs(E, cfg):
    G, T = E.G, E.T
    torch.set_num_threads(THREADS)
    pol, ck, parent, classes, ref, _ = E.load_models()
    torch.set_num_threads(THREADS)
    need(torch.get_num_threads() == THREADS, "cpu threads != 2")
    classes = [str(c) for c in classes]
    width, _, ncls = G._inspect_model(parent)
    need(ncls == len(classes) == 93, "expected 93 classes")
    missing = [p for p in required_pins(E, pol) if p not in cfg["pins"]]
    need(not missing, f"config pins lack required inputs: {missing}")
    psd0 = {k: v.detach().clone() for k, v in parent.state_dict().items()}

    cache = E.load_prior_cache(classes)                        # 2122 / 224 / [660,16] checked there
    for name, xs in (("base", cache[0]), ("context", cache[2])):
        G._check_matrix_list(xs, width, name)
    need(bool(np.isfinite(cache[4]).all()), "teacher logits non-finite")

    new = E.load_new()
    need([str(c) for c in new["classes"].tolist()] == classes, "new classes differ")
    ny, nf, ids = new["y"], new["folds"], list(new["id_list"])
    exposed = new["exposed"].astype(bool)
    need(ny.dtype == np.int64 and ny.shape == (36,) and nf.shape == (36,) and len(set(ids)) == 36, "new36 shape")
    need(int(exposed.sum()) == 2, "expected two previously exposed clips")
    G._check_matrix_list(new["prefixes"], width, "new prefixes")
    class_order = sorted(set(ny.tolist()))
    need(len(class_order) == 2, "new36 is not two-label")

    with np.load(OLD_RUN / "training_arrays.npz", allow_pickle=False) as z:
        old = {k: z[k].copy() for k in z.files}
    need(same(old["base_indices"], cache[5]) and same(old["context_indices"], cache[6]), "old schedules differ from cache")

    with np.load(SUPPORT_NPZ, allow_pickle=False) as z:
        sup = {k: z[k].copy() for k in z.files}
    pkeys = sorted(k for k in sup if k.startswith("support_prefix__"))
    need(pkeys == [f"support_prefix__{i:03d}" for i in range(N_SUPPORT)], "support prefix keys")
    sup_prefix = G._check_matrix_list([sup[k] for k in pkeys], width, "support prefixes")
    sup_y, sup_ids, sup_sched = sup["support_y"], sup["support_ids"], sup["support_schedule"]
    need(sup_y.dtype == np.int64 and sup_y.shape == (N_SUPPORT,), "support_y")
    need(sup_ids.shape == (N_SUPPORT,) and len(set(sup_ids.tolist())) == N_SUPPORT, "support_ids")
    need(sup_sched.dtype == np.int64 and sup_sched.shape == (STEPS, SUPPORT_PER_STEP), "support_schedule")
    need(same(old["support_ids"], sup_ids) and same(old["support_schedule"], sup_sched), "support arrays differ from old run")
    rc = E.read(SUPPORT_RECEIPT)
    rrows = rc["rows"]
    need(rc["npz_sha256"] == E.sha(SUPPORT_NPZ), "support receipt npz sha256")
    need([str(r["sample_id"]) for r in rrows] == [str(s) for s in sup_ids.tolist()], "support receipt order/ids")
    need(len({r["clip_id"] for r in rrows}) == N_SUPPORT_CLIPS, "support clip count != 105")
    need(len({r["label"] for r in rrows}) == N_SUPPORT_LABELS, "support label count != 69")
    need([r["label"] for r in rrows] == [classes[int(c)] for c in sup_y], "support labels differ from receipt")
    need(not ({int(c) for c in sup_y} & set(class_order)), "support labels overlap new36 labels")

    folds, all_held = {}, []
    for f in range(N_FOLDS):
        k = f"candidate_fold{f}"
        groups = T.fold_training_rows(ids, ny, nf, f, class_order)
        rows = np.concatenate([np.asarray(g, dtype=np.int64) for g in groups])
        held = np.flatnonzero(nf == f)
        need(same(rows, old[f"train_rows__{k}"]) and same(held, old[f"heldout_rows__{k}"]), f"fold {f} rows differ")
        need(len(rows) == FOLD_TRAIN[f] and len(held) == FOLD_HELD[f], f"fold {f} sizes")
        need(not (set(rows.tolist()) & set(held.tolist())) and not (nf[rows] == f).any(), f"fold {f} overlap")
        need(len(rows) + len(held) == 36, f"fold {f} does not cover 36")
        all_held += held.tolist()
        xsch = old[f"extra_schedule__{k}"]
        n = len(rows)
        osch = T.cyclic_schedule([len(g) for g in groups], STEPS, per_class=2)
        need(xsch.dtype == np.int64 and xsch.shape == (STEPS, KSL_PER_STEP + SUPPORT_PER_STEP), f"fold {f} schedule shape")
        need(same(xsch, sup[f"extra_schedule__{k}"]), f"fold {f} schedule differs from SUPPORT_PREPARED")
        need(same(xsch[:, :KSL_PER_STEP], osch) and same(xsch[:, :KSL_PER_STEP], old[f"old_extra_schedule__{k}"]),
             f"fold {f} first4 != KSL schedule")
        need(int(xsch[:, :KSL_PER_STEP].max()) < n and int(xsch[:, KSL_PER_STEP:].min()) >= n, f"fold {f} ksl/support split")
        need(same(xsch[:, KSL_PER_STEP:] - n, sup_sched), f"fold {f} last4 != support schedule")
        xp = [new["prefixes"][int(i)] for i in rows] + sup_prefix
        xy = np.concatenate([ny[rows], sup_y])
        need(int(xsch.max()) < len(xp) and xy.shape == (len(xp),), f"fold {f} index range")
        folds[f] = dict(rows=rows, held=held, xp=xp, xy=xy, xsch=xsch.copy(), n_ksl=n)
    need(sorted(all_held) == list(range(36)), "held-out folds overlap or miss rows")
    need(all(torch.equal(v, psd0[k]) for k, v in parent.state_dict().items()), "parent modified during load")
    return dict(pol=pol, ck=ck, parent=parent, classes=classes, ref=ref, cache=cache, new=new, ids=ids,
                class_order=class_order, exposed=exposed, old=old, sup_ids=sup_ids, folds=folds, psd0=psd0)


def input_summary(S):
    return {"classes": len(S["classes"]), "base": len(S["cache"][0]), "context": len(S["cache"][2]),
            "old_schedule_shape": list(S["cache"][5].shape), "new36": len(S["ids"]),
            "new_class_order": [S["classes"][c] for c in S["class_order"]], "exposed": int(S["exposed"].sum()),
            "support": {"rows": N_SUPPORT, "clips": N_SUPPORT_CLIPS, "labels": N_SUPPORT_LABELS},
            "folds": {f: {"n_ksl_train": d["n_ksl"], "n_heldout": len(d["held"]), "n_extra_pool": len(d["xp"]),
                          "schedule_sha": arr_sha(d["xsch"]),
                          "heldout_ids": [S["ids"][int(i)] for i in d["held"]]} for f, d in S["folds"].items()}}


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def preflight(E, cfg, out, cfg_sha):
    need(not out.exists(), f"refusing: {out} exists")
    verify_hashes(E, cfg)
    S = load_inputs(E, cfg)
    verify_hashes(E, cfg)
    out.mkdir(parents=False, exist_ok=False)
    E.write_json(out / "PREFLIGHT.json", {
        "status": PREFLIGHT_STATUS, "created_utc": now().isoformat(), "config_sha256": cfg_sha,
        "code_hashes": P.code_hashes(), "inputs": input_summary(S), "pins_verified_before_and_after": True,
        "no_training_performed": True, "torch": torch.__version__, "threads": torch.get_num_threads()})
    print("preflight passed:", out)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def head_diff(a, b):
    keys_ok = set(a) == set(b)
    return {"keys_equal": keys_ok,
            "bitwise_equal_all": keys_ok and all(torch.equal(a[k], b[k]) for k in a),
            "max_abs_head_diff": max(float((a[k].double() - b[k].double()).abs().max())
                                     for k in a if k.startswith(P.HEAD)) if keys_ok else None}


def check_out(out, psd, S, d, tag, project):
    sd, losses, diags = out["state_dict"], np.asarray(out["losses"]), np.asarray(out["diagnostics"])
    c = {"updates": int(out["updates"]) == STEPS,
         "losses_shape_finite": losses.shape == (STEPS, len(P.LOSS_COLUMNS)) and bool(np.isfinite(losses).all()),
         "diagnostics_shape_finite": diags.shape == (STEPS, len(P.DIAG_COLUMNS)) and bool(np.isfinite(diags).all()),
         "projection_flag": bool(out["project_extra"]) == project
                            and (project or not diags[:, P.DIAG_COLUMNS.index("projected")].any()),
         "old_schedule_identical": same(out["base_indices"], S["cache"][5]) and same(out["context_indices"], S["cache"][6]),
         "extra_schedule_identical": same(out["extra_indices"], d["xsch"]),
         "keys_equal_parent": set(sd) == set(psd),
         "trainable_head_only": all(n.startswith(P.HEAD) for n in out["trainable_names"]),
         "non_head_bitwise_equal_parent": all(torch.equal(sd[k], psd[k]) for k in psd if not k.startswith(P.HEAD)),
         "head_changed": any(not torch.equal(sd[k], psd[k]) for k in psd if k.startswith(P.HEAD)),
         "dtype_shape_equal_parent": all(sd[k].dtype == psd[k].dtype and sd[k].shape == psd[k].shape for k in psd),
         "frozen_parameters_unchanged": bool(out["frozen_parameters_unchanged"]),
         "buffers_unchanged": bool(out["buffers_unchanged"])}
    need(all(c.values()), f"{tag}: training checks failed {c}")
    return c


def diag_summary(d):
    col = {c: d[:, i] for i, c in enumerate(P.DIAG_COLUMNS)}
    pj = col["projected"] > 0
    neg = col["dot_before"] < 0
    rel = np.divide(col["removed_norm"], col["new_norm"], out=np.zeros_like(col["new_norm"]), where=col["new_norm"] > 0)
    return {"steps": int(len(d)), "negative_dot_steps": int(neg.sum()), "projected_steps": int(pj.sum()),
            "first_projected_step": int(np.flatnonzero(pj)[0]) if pj.any() else None,
            "old_norm_mean": float(col["old_norm"].mean()), "new_norm_mean": float(col["new_norm"].mean()),
            "dot_before_mean": float(col["dot_before"].mean()), "dot_after_mean": float(col["dot_after"].mean()),
            "dot_after_abs_max_on_projected": float(np.abs(col["dot_after"][pj]).max()) if pj.any() else None,
            "removed_norm_mean": float(col["removed_norm"].mean()), "removed_norm_max": float(col["removed_norm"].max()),
            "removed_over_new_norm_mean_on_projected": float(rel[pj].mean()) if pj.any() else None,
            "old_dot_actual_delta_mean": float(col["old_dot_actual_delta"].mean()),
            "old_dot_actual_delta_sum": float(col["old_dot_actual_delta"].sum()),
            "old_dot_actual_delta_positive_steps": int((col["old_dot_actual_delta"] > 0).sum()),
            "note": "old_dot_actual_delta = <pre-update g_old, AdamW parameter change>; negative means the step "
                    "moved against g_old to first order. Direction diagnostics are technical, not accuracy gains."}


def first_step_parity(E, S):
    T, G, c = E.T, E.G, S["cache"]
    one = (c[5][:1].copy(), c[6][:1].copy())
    res = {}
    for f, d in S["folds"].items():
        kw = dict(extra_prefixes=d["xp"], extra_y=d["xy"], extra_indices=d["xsch"][:1].copy(), extra_weight=EXTRA_W,
                  **T.TRAIN_KW)
        a = T.train_arm(S["parent"], *c[:5], *one, arm=f"parity_old_fold{f}", **kw)
        b = P.train_arm(S["parent"], *c[:5], *one, arm=f"parity_split_fold{f}", project_extra=False, G=G, **kw)
        hd = head_diff(b["state_dict"], a["state_dict"])
        res[f"fold{f}"] = dict(hd, losses_bitwise_equal=bool(np.array_equal(a["losses"], b["losses"])),
                               max_abs_loss_diff=float(np.abs(a["losses"] - b["losses"]).max()),
                               split_diagnostics=b["diagnostics"][0])
        need(hd["keys_equal"] and hd["max_abs_head_diff"] <= TOL_STEP and res[f"fold{f}"]["max_abs_loss_diff"] <= TOL_STEP,
             f"first-step parity failed fold {f}: {res[f'fold{f}']}")
    return res


def train(E, cfg, out, cfg_sha):
    need((out / "PREFLIGHT.json").is_file(), "PREFLIGHT.json missing; run preflight first")
    need(sorted(os.listdir(out)) == ["PREFLIGHT.json"], f"refusing: {out} already holds outputs besides PREFLIGHT.json")
    pre = E.read(out / "PREFLIGHT.json")
    need(pre["status"] == PREFLIGHT_STATUS and pre["config_sha256"] == cfg_sha, "PREFLIGHT status/config differ")
    need(pre["code_hashes"] == P.code_hashes(), "code changed since preflight")
    receipt = verify_receipt()
    verify_hashes(E, cfg)
    S = load_inputs(E, cfg)
    need(json.loads(json.dumps(E.plain(input_summary(S)))) == pre["inputs"], "inputs differ from preflight")
    T, G = E.T, E.G
    parent, psd = S["parent"], S["psd0"]

    parity = first_step_parity(E, S)
    E.write_json(out / "FIRST_STEP_PARITY.json", {
        "created_utc": now().isoformat(), "tolerance": TOL_STEP, "folds": parity,
        "note": "one step on schedule[:1]: original train_addition.train_arm (total.backward) vs split-gradient "
                "path with projection OFF; first-step agreement does not imply full-run identity"})
    need(all(torch.equal(v, psd[k]) for k, v in parent.state_dict().items()), "parity modified parent")

    def progress(arm, done, total):
        print(f"[{now().isoformat()}] {arm} {done}/{total}", flush=True)

    meta_ck = {k: v for k, v in S["ck"].items() if k != "state_dict"}
    recipe = {"steps": STEPS, "extra_weight": EXTRA_W, "train_kw": T.TRAIN_KW, "trainable": "head.* only",
              "optimizer": "AdamW betas(0.9,0.999) eps1e-8, unchanged", "threads": THREADS,
              "old_objective": "CE32 + weighted anchor + weighted output (gradient g_old)",
              "extra_objective": "0.25 * CE(4 KSL + 4 MP support) (gradient g_extra)",
              "projection": "candidate only: global head weight+bias float64; if <g_old,g_extra> < 0 and |g_old|^2 > 0, "
                            "g_extra -= <g_old,g_extra>/|g_old|^2 * g_old; cast to param dtype; no epsilon/threshold",
              "loss_columns": P.LOSS_COLUMNS, "diagnostic_columns": P.DIAG_COLUMNS}
    arrays = {"loss_columns": np.array(P.LOSS_COLUMNS), "diagnostic_columns": np.array(P.DIAG_COLUMNS),
              "base_indices": S["cache"][5], "context_indices": S["cache"][6], "support_ids": S["sup_ids"]}
    files, checks, results = {}, {}, {}
    for f, d in S["folds"].items():
        arrays.update({f"extra_schedule__fold{f}": d["xsch"], f"train_rows__fold{f}": d["rows"],
                       f"heldout_rows__fold{f}": d["held"]})
        for arm, project in ((f"matched_fold{f}", False), (f"projected_fold{f}", True)):
            r = P.train_arm(parent, *S["cache"], extra_prefixes=d["xp"], extra_y=d["xy"], extra_indices=d["xsch"],
                            extra_weight=EXTRA_W, arm=arm, progress=progress, project_extra=project, G=G, **T.TRAIN_KW)
            checks[arm] = check_out(r, psd, S, d, arm, project)
            p = out / f"{arm}_final.pt"
            with p.open("xb") as fh:
                torch.save(dict(meta_ck, state_dict=r["state_dict"], gradient_projection_pair={
                    "arm": arm, "fold": f, "project_extra": project, "recipe": E.plain(recipe),
                    "parent_sha256": E.PARENT_SHA, "classes": S["classes"],
                    "training_ids": [S["ids"][int(i)] for i in d["rows"]],
                    "heldout_ids": [S["ids"][int(i)] for i in d["held"]], "support_ids": S["sup_ids"].tolist(),
                    "code_hashes": P.code_hashes(), "config_sha256": cfg_sha}), fh)
            files[p.name] = E.sha(p)
            arrays[f"losses__{arm}"] = r["losses"]
            arrays[f"diagnostics__{arm}"] = r["diagnostics"]
            results[arm] = r
    need(all(torch.equal(v, psd[k]) for k, v in parent.state_dict().items()), "training modified parent")
    E.save_npz(out / "training_arrays.npz", arrays)
    files["training_arrays.npz"] = E.sha(out / "training_arrays.npz")

    vs_old, vs_matched = {}, {}
    for f in range(N_FOLDS):
        osd = torch.load(OLD_RUN / f"candidate_fold{f}_final.pt", map_location="cpu", weights_only=True)["state_dict"]
        ml, ol = results[f"matched_fold{f}"]["losses"], S["old"][f"losses__candidate_fold{f}"]
        vs_old[f"matched_fold{f}"] = dict(head_diff(results[f"matched_fold{f}"]["state_dict"], osd),
                                          losses_bitwise_equal=bool(np.array_equal(ml, ol)),
                                          max_abs_loss_diff=float(np.abs(ml - ol).max()) if ml.shape == ol.shape else None)
        vs_matched[f"projected_fold{f}"] = dict(head_diff(results[f"projected_fold{f}"]["state_dict"],
                                                          results[f"matched_fold{f}"]["state_dict"]),
                                                max_abs_loss_diff=float(np.abs(results[f"projected_fold{f}"]["losses"] - ml).max()))
    verify_hashes(E, cfg)
    E.write_json(out / "TRAINING.json", {
        "status": TRAINED_STATUS, "created_utc": now().isoformat(), "files": files, "config_sha256": cfg_sha,
        "preflight_sha256": E.sha(out / "PREFLIGHT.json"), "first_step_parity_sha256": E.sha(out / "FIRST_STEP_PARITY.json"),
        "test_receipt_sha256": E.sha(RECEIPT), "test_receipt_status": receipt["status"], "code_hashes": P.code_hashes(),
        "recipe": recipe, "parent_sha256": E.PARENT_SHA, "reference_sha256": E.REF_SHA, "classes": S["classes"],
        "checks": checks, "matched_vs_old_mp_support_candidate": vs_old, "projected_vs_matched": vs_matched,
        "old_vs_matched_note": "old candidate used total.backward(); matched uses split gradients; roundoff "
                               "differences are reported, not required to be zero",
        "diagnostic_summary": {a: diag_summary(r["diagnostics"]) for a, r in results.items()},
        "loss_first_last": {a: {"first": r["losses"][0], "last": r["losses"][-1]} for a, r in results.items()},
        "no_extra_control_copied": False, "evaluation_performed": False, "hyperparameter_search": False,
        "threads": torch.get_num_threads(), "torch": torch.__version__, "python": platform.python_version(),
        "pins_verified_before_and_after": True})
    print("training frozen:", out)


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def changes_between(rows, groups, cand, comp):
    return {g: {"gains": [r["id"] for r in rows if r["group"] == g and r["models"][cand]["full93"]["correct"]
                          and not r["models"][comp]["full93"]["correct"]],
                "regressions": [dict(id=r["id"], condition=r["condition"], true=r["true"],
                                     pred=r["models"][cand]["full93"]["pred"])
                                for r in rows if r["group"] == g and not r["models"][cand]["full93"]["correct"]
                                and r["models"][comp]["full93"]["correct"]]} for g in groups}


def evaluate(E, cfg, out, cfg_sha):
    for n in ("EVALUATION.json", "evaluation_arrays.npz"):
        need(not (out / n).exists(), f"refusing to overwrite {out / n}")
    tr = E.read(out / "TRAINING.json")
    need(tr["status"] == TRAINED_STATUS and tr["config_sha256"] == cfg_sha, "TRAINING.json status/config")
    need(tr["preflight_sha256"] == E.sha(out / "PREFLIGHT.json"), "PREFLIGHT changed since training")
    for n, h in tr["files"].items():
        need(E.sha(out / n) == h, f"training file changed: {n}")
    need(tr["code_hashes"] == P.code_hashes(), "code changed since training")
    verify_hashes(E, cfg)
    G = E.G
    torch.set_num_threads(THREADS)
    pol, ck, parent, classes, ref, _ = E.load_models()
    classes = [str(c) for c in classes]
    missing = [p for p in required_pins(E, pol) if p not in cfg["pins"]]
    need(not missing, f"config pins lack required inputs: {missing}")
    target_idx, neural_path = E.project18(classes)
    target_names = {classes[i] for i in target_idx}
    sds = {REF: ref["state_dict"]}
    for a in ARMS:
        sds[a] = torch.load(out / f"{a}_final.pt", map_location="cpu", weights_only=True)["state_dict"]
    for f, a in enumerate(OLDM):
        sds[a] = torch.load(OLD_RUN / f"candidate_fold{f}_final.pt", map_location="cpu", weights_only=True)["state_dict"]
    E.shared_encoder(dict(parent=ck["state_dict"], **sds))
    models = [REF] + ARMS + OLDM
    heads = {a: (sds[a]["head.weight"].numpy().astype(np.float64), sds[a]["head.bias"].numpy().astype(np.float64))
             for a in models}
    checks = {"all_non_head_tensors_bitwise_equal_parent": True, "project18_module_sha256": E.sha(neural_path)}

    # ---- regression groups from cached pooled features (no encoder run)
    meta, old = E.read(E.META), E.read(E.OLD_RUN / "evaluation.json")
    oldmap = {(r["group"], r["id"], r.get("condition", "intact")): r for r in old["rows"]}
    scores, pooled, diffs = {}, {}, {}
    with np.load(E.OLD_RUN / "evaluation_arrays.npz", allow_pickle=False) as ev:
        need([str(c) for c in ev["classes"].tolist()] == classes, "evaluation_arrays classes")
        for g in E.GROUPS:
            pooled[g] = ev[g + "__control__pooled"].copy()
            for a in models:
                scores[g, a] = pooled[g].astype(np.float64) @ heads[a][0].T + heads[a][1]
            diffs[g] = E.maxabs(scores[g, REF], ev[g + "__control__scores"])
            need(diffs[g] <= TOL_832, f"{g}: 832 scores not reproduced ({diffs[g]})")
    checks["ref832_cached_scores_max_abs"] = diffs
    rows = []
    for g in E.GROUPS:
        need(len(meta[g]) == len(pooled[g]), f"{g}: metadata/pooled count mismatch")
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
        p, m, o = f"projected_fold{f}", f"matched_fold{f}", f"oldmethod_fold{f}"
        for cand, comp in ((p, m), (p, REF), (m, REF), (m, o), (p, o)):
            changes[f"{cand}_vs_{comp}"] = changes_between(rows, E.GROUPS, cand, comp)

    # ---- KSL 36 from cached pooled features; OOF by new.folds only
    new = E.load_new()
    need([str(c) for c in new["classes"].tolist()] == classes, "new classes differ")
    ids, y, folds, exposed = new["id_list"], new["y"], new["folds"], new["exposed"].astype(bool)
    d_pool = E.maxabs(G.predict_prefix(parent, new["prefixes"])["pooled"], new["pooled_832"])
    need(d_pool <= TOL_NEW, f"new pooled features not reproduced ({d_pool})")
    Pn = new["pooled_832"].astype(np.float64)
    ns = {a: Pn @ heads[a][0].T + heads[a][1] for a in models}
    d_new = E.maxabs(ns[REF], new["logits_cached"])
    need(d_new <= TOL_NEW, f"832 logits on new clips not reproduced ({d_new})")
    checks.update(new_pooled_max_abs=d_pool, new_ref832_vs_cached_logits_max_abs=d_new)
    for kind in ("matched", "projected", "oldmethod"):
        ns[f"{kind}_oof"] = np.stack([ns[f"{kind}_fold{int(f)}"][i] for i, f in enumerate(folds)])
    view = [REF, "matched_oof", "projected_oof", "oldmethod_oof"]
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
        held, arms = folds == f, (f"projected_fold{f}", f"matched_fold{f}", f"oldmethod_fold{f}", REF)
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
    clip_changes = {f"projected_oof_vs_{c}": {
        "gains": [r["id"] for r in clips if r["correct"]["projected_oof"] and not r["correct"][c]],
        "losses": [r["id"] for r in clips if not r["correct"]["projected_oof"] and r["correct"][c]]}
        for c in (REF, "matched_oof", "oldmethod_oof")}

    # ---- predeclared gates (class thresholds derived from primary34 class counts, not from names)
    prim = new_summary["primary34"]
    po, mo = prim["projected_oof"]["full93"], prim["matched_oof"]["full93"]
    counts = {k: v["n"] for k, v in po["per_class"].items()}
    need(sorted(counts.values()) == [16, 18], f"primary34 class counts not 18/16: {counts}")
    need(set(counts) == {classes[c] for c in sorted(set(y.tolist()))}, "primary34 classes differ from new y")
    threshold = {k: (17 if n == 18 else 14) for k, n in counts.items()}
    gates = {"primary": {
        "projected_total_ge_matched": po["correct"] >= mo["correct"],
        f"projected_total_ge_{GATE_PRIMARY_TOTAL}": po["correct"] >= GATE_PRIMARY_TOTAL,
        **{f"class_{k}_ge_{threshold[k]}_of_{counts[k]}": po["per_class"][k]["correct"] >= threshold[k] for k in counts},
        **{f"class_{k}_ge_matched": po["per_class"][k]["correct"] >= mo["per_class"][k]["correct"] for k in counts}},
        "retention": {}}
    for f in range(N_FOLDS):
        for comp in (REF, f"matched_fold{f}"):
            gates["retention"][f"projected_fold{f}_vs_{comp}"] = E.retention(summary, f"projected_fold{f}", comp)
    info_retention = {f"{a}_fold{f}_vs_{c}": E.retention(summary, f"{a}_fold{f}", c)
                      for f in range(N_FOLDS) for a, c in (("matched", REF), ("projected", f"oldmethod_fold{f}"))}
    passed = all(gates["primary"].values()) and all(all(v.values()) for v in gates["retention"].values())
    decision = PASS if passed else REJECT

    with np.load(out / "training_arrays.npz", allow_pickle=False) as z:
        diag = {a: diag_summary(z[f"diagnostics__{a}"]) for a in ARMS}

    arrays = {"classes": np.array(classes), "new__ids": np.array(ids), "new__y": y, "new__folds": folds,
              "new__exposed": exposed, "new__pooled": new["pooled_832"]}
    for a, s in ns.items():
        arrays[f"new__{a}__scores"] = s
    for g in E.GROUPS:
        for a in models:
            arrays[f"{g}__{a}__scores"] = scores[g, a]
    E.save_npz(out / "evaluation_arrays.npz", arrays)
    verify_hashes(E, cfg)
    E.write_json(out / "EVALUATION.json", {
        "status": "LOCAL_DEVELOPMENT_EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT", "decision": decision,
        "production_ready": False, "default_promoted": False, "gates": gates,
        "informational_retention_not_gated": info_retention, "class_thresholds": threshold, "checks": checks,
        "new36": {"summary": new_summary, "per_fold": per_fold, "clips": clips, "changes": clip_changes},
        "regression": {"summary": summary, "changes": changes}, "rows": rows,
        "training_diagnostic_summary": diag, "matched_vs_old_mp_support_candidate": tr["matched_vs_old_mp_support_candidate"],
        "created_utc": now().isoformat(), "arrays_sha256": E.sha(out / "evaluation_arrays.npz"),
        "training_record_sha256": E.sha(out / "TRAINING.json"), "config_sha256": cfg_sha,
        "code_hashes": P.code_hashes(), "parent_sha256": E.PARENT_SHA, "reference_sha256": E.REF_SHA,
        "best_fold_selected": False, "retrained_on_all36": False, "hyperparameter_search": False,
        "limits": [
            "local development comparison only; not independent signer validation or new-person generalization",
            "primary34 is the repeatedly analysed development subset excluding the two older-exposed 00 clips; "
            "it is not pristine unseen data; previously_exposed00 is reported separately",
            "each clip is excluded only from its own fold's training; signer identity unknown",
            "project18 scope is a narrowed ranking and is not counted as full93 success",
            "one-sided projection is A-GEM-inspired, not original A-GEM/PCGrad; no retention guarantee under AdamW",
            "a negative gradient dot product can arise from ordinary objective trade-offs and is not a diagnosed cause",
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
        E, cfg, out, cfg_sha = load_cfg(a.config)
        {"preflight": preflight, "train": train, "evaluate": evaluate}[a.command](E, cfg, out, cfg_sha)
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
