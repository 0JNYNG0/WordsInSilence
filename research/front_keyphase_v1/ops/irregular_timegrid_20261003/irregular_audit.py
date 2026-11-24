"""Irregular-time inference diagnostic: same frozen TCN, predeclared synthetic clock corruptions
restored by the audited, unmodified timegrid.restore of the old unit (imported from helper_dir).

Diagnostic only. No training, no threshold/parameter search, no camera, no user data, no network.

Arms per row and case (same model, same rows; no best-method selection):
  original   stored scores (native original_logits / MP arm 'mediapipe'), identical in every case
  corrupted  build_channels(corruptions.corrupt(raw))
  previous   build_channels(TG.restore(corrupted, 'previous'))   mechanism control
  linear     build_channels(TG.restore(corrupted, 'linear'))     predeclared primary
restore() only receives the corrupted xy/present/ts - never the dense original or its length.

This module imports only stdlib, NumPy and corruptions at top level; the helper, timegrid, torch
and the pilot/loader APIs are imported inside _run(), so the pure helpers here are unit-testable.

Run: python irregular_audit.py --config <config.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import corruptions as C

ARMS = ("original", "corrupted", "previous", "linear")
CASES = tuple(C.CASES)
GROUPS = ("native", "mediapipe")
PRIMARY_CASE = "isolated_drop"
NATIVE_MIN_NET_GAIN = 7             # practical ~1 pp of 699 rows, NOT a significance test
REPRO_INDICES = {"native": [0, 349, 698], "mediapipe": [0, 56, 112]}
COMPARISONS = (("corrupted", "linear"), ("corrupted", "previous"), ("previous", "linear"),
               ("original", "corrupted"), ("original", "previous"), ("original", "linear"))
HELPER_FILES = ("run_audit.py", "timegrid.py")
API_MODULES = ("tools_prep_union93_candidate", "pilot.data", "pilot.web_input", "framerate_impl")
CAMERA_MODULES = ("cv2", "mediapipe")
LONG_GAP_30 = 2 * 1_000_000 + 60    # 30*gap > this  <=>  gap > 2P + 2 us
LIMITS = [
    "Synthetic stress diagnostic on the same frozen model; no training, no deployment or "
    "production authorisation, no web, new-people or target-18 generalisation claim.",
    "native = reused development rows (699, 81 classes, 621 source recordings, signers 11/12/13); "
    "mediapipe = training-exposed cached rows (113, 69 classes, 105 source recordings, signers 1/2). "
    "Groups are never pooled; neither is an independent test.",
    "'linear' is the predeclared primary; 'previous' is a mechanism control and is never "
    "substituted for 'linear' based on scores.",
    "clock_noise simulates timestamp measurement noise only, not motion resampling or speed change; "
    "grid points may be estimated there although no frame was dropped.",
    "Estimated points are interpolations on the corrupted clock, not observations; brackets longer "
    "than 2P + 2 us (burst_drop spans) stay unsupported.",
    "The native net-gain gate (>= 7 rows) is a practical threshold, not statistical significance.",
]


class IrregularError(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


# --- pure helpers (no helper/model import) ----------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def norm(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def under(path, root) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    eq = np.issubdtype(a.dtype, np.floating)
    return bool(np.array_equal(a, b, equal_nan=True) if eq else np.array_equal(a, b))


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o, key=str)
    return str(o)


def write_json_exclusive(path, value, strict=True):
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=1, allow_nan=not strict,
                  default=None if strict else _plain)
        fh.write("\n")


def check_corruption(xy, pr, ts, c, case, sample_id) -> dict:
    """Independent contract check of one corrupt() result against its source arrays."""
    if set(c) != {"xy", "present", "ts", "indices", "clock_offset_us", "case"} or c["case"] != case:
        raise IrregularError(f"{sample_id}/{case}: corrupt() result keys/case wrong")
    ts = np.asarray(ts)
    t, idx = len(ts), np.asarray(c["indices"])
    if idx.dtype != np.int64 or idx.ndim != 1 or len(idx) < 1 or idx[0] != 0 or idx[-1] != t - 1:
        raise IrregularError(f"{sample_id}/{case}: indices must run 0..T-1 (first/last retained)")
    gaps = np.diff(idx)
    if (gaps < 1).any() or (len(gaps) and int(gaps.max()) > C.MAX_SOURCE_GAP[case]):
        raise IrregularError(f"{sample_id}/{case}: source index gaps outside [1,{C.MAX_SOURCE_GAP[case]}]")
    if case == "clock_noise" and not np.array_equal(idx, np.arange(t)):
        raise IrregularError(f"{sample_id}/{case}: clock_noise must keep every frame")
    if not (same(c["xy"], np.asarray(xy)[idx]) and same(c["present"], np.asarray(pr)[idx])):
        raise IrregularError(f"{sample_id}/{case}: xy/present are not verbatim source frames")
    cts, off = np.asarray(c["ts"]), np.asarray(c["clock_offset_us"])
    if cts.dtype != ts.dtype or cts.shape != idx.shape or off.shape != idx.shape:
        raise IrregularError(f"{sample_id}/{case}: ts/offset shape or dtype wrong")
    src, got, o = [int(v) for v in ts[idx]], [int(v) for v in cts], [int(v) for v in off]
    if [g - s for g, s in zip(got, src)] != o:
        raise IrregularError(f"{sample_id}/{case}: ts != source ts + clock_offset_us")
    if o[0] or o[-1] or got[0] != int(ts[0]) or got[-1] != int(ts[-1]):
        raise IrregularError(f"{sample_id}/{case}: endpoints moved")
    if (case != "clock_noise" and any(o)) or any(abs(v) > C.MAX_OFFSET_US for v in o):
        raise IrregularError(f"{sample_id}/{case}: clock offsets outside contract")
    if any(b <= a for a, b in zip(got, got[1:])):
        raise IrregularError(f"{sample_id}/{case}: corrupted ts not strictly increasing")
    again = C.corrupt(xy, pr, ts, sample_id, case)
    if not all(same(again[k], c[k]) for k in ("xy", "present", "ts", "indices", "clock_offset_us")):
        raise IrregularError(f"{sample_id}/{case}: corrupt() not deterministic")
    d = [b - a for a, b in zip(got, got[1:])]
    return {"max_source_index_gap": int(gaps.max()) if len(gaps) else 0,
            "clock_offset_range_us": [min(o), max(o)],
            "corrupted_delta_range_us": [min(d), max(d)] if d else None}


def bracket_check(cts, cpresent, out_ts, src, est) -> dict:
    """No estimate where the bracket gap > 2P + 2 us or either bracket lacks the joint."""
    ti, g = [int(v) for v in cts], [int(v) for v in out_ts]
    pb, src, est = np.asarray(cpresent).astype(bool), np.asarray(src), np.asarray(est, dtype=bool)
    if len(src) != len(g) or est.shape != (len(g), pb.shape[1]):
        raise IrregularError("restored grid / provenance shapes inconsistent")
    if g[0] != ti[0] or g[-1] != ti[-1] or any(b <= a for a, b in zip(g, g[1:])):
        raise IrregularError("restored grid endpoints or order wrong")
    if est[src >= 0].any():
        raise IrregularError("estimated slot on an exact source frame")
    bad, long_frames, uns_frames, uns_slots = [], 0, 0, 0
    for k in np.flatnonzero(src < 0).tolist():
        lo = bisect_right(ti, g[k]) - 1
        if 0 <= lo < len(ti) - 1:
            long_ = 30 * (ti[lo + 1] - ti[lo]) > LONG_GAP_30
            allowed = np.zeros(pb.shape[1], bool) if long_ else pb[lo] & pb[lo + 1]
        else:
            long_, allowed = False, np.zeros(pb.shape[1], bool)
        long_frames += long_
        v = int((est[k] & ~allowed).sum())
        if v:
            bad.append({"grid_index": k, "slots": v, "long_bracket": bool(long_)})
        uns_slots += int((~est[k]).sum())
        uns_frames += not bool(est[k].any())
    if bad:
        raise IrregularError("estimates outside supported brackets", {"n": len(bad), "first": bad[:20]})
    return {"grid_frames": len(g), "exact_frames": int((src >= 0).sum()),
            "estimated_frames": int(est.any(axis=1).sum()), "estimated_slots": int(est.sum()),
            "unsupported_frames": uns_frames, "unsupported_slots": uns_slots,
            "frames_with_long_bracket": long_frames}


def breakdown(rows, key) -> dict:
    out = {}
    for r in rows:
        g = out.setdefault(str(r[key]), {"n": 0, **{a: 0 for a in ARMS}})
        g["n"] += 1
        for a in ARMS:
            g[a] += int(r["arms"][a]["correct"])
    return dict(sorted(out.items()))


def recoveries(rows) -> dict:
    sub = [r for r in rows if r["arms"]["original"]["correct"] and not r["arms"]["corrupted"]["correct"]]
    return {"n": len(sub), **{f"correct_in_{a}": sum(r["arms"][a]["correct"] for r in sub)
                              for a in ("previous", "linear")},
            "rows": [{"id": r["id"], "true": r["true"], "corrupted_pred": r["arms"]["corrupted"]["pred"],
                      **{a: {k: r["arms"][a][k] for k in ("pred", "correct", "rank")} for a in ("previous", "linear")}}
                     for r in sub]}


def _vs(st):
    net = st["correct"]["linear"] - st["correct"]["corrupted"]
    sig = {s: {"corrupted": v["corrupted"], "linear": v["linear"], "nondecrease": v["linear"] >= v["corrupted"]}
           for s, v in st["per_signer"].items()}
    return net, sig, bool(sig) and all(v["nondecrease"] for v in sig.values())


def admission(stats, contracts) -> dict:
    """Frozen admission conditions; each reported separately. stats[group][case] =
    {'correct': {arm: n}, 'per_signer': {s: {arm: n}}, 'newly_wrong': corrupted-correct->linear-wrong}."""
    net, sig, ok = _vs(stats["native"][PRIMARY_CASE])
    c1 = {"case": PRIMARY_CASE, "group": "native", "net_linear_vs_corrupted": net,
          "net_ge_7": net >= NATIVE_MIN_NET_GAIN, "per_signer": sig, "all_signers_nondecrease": ok}
    c1["pass"] = bool(c1["net_ge_7"] and ok)
    net, sig, ok = _vs(stats["mediapipe"][PRIMARY_CASE])
    c2 = {"case": PRIMARY_CASE, "group": "mediapipe", "net_linear_vs_corrupted": net,
          "total_nondecrease": net >= 0, "per_signer": sig, "all_signers_nondecrease": ok,
          "pass": bool(net >= 0 and ok)}
    c3 = {}
    for g in GROUPS:
        for c in CASES:
            net, sig, ok = _vs(stats[g][c])
            nw = int(stats[g][c]["newly_wrong"])
            c3[f"{g}/{c}"] = {"newly_wrong_corrupted_correct_linear_wrong": nw, "no_newly_wrong": nw == 0,
                              "net_linear_vs_corrupted": net, "total_nondecrease": net >= 0,
                              "per_signer": sig, "all_signers_nondecrease": ok,
                              "pass": bool(nw == 0 and net >= 0 and ok)}
    c4 = {k: bool(v) for k, v in contracts.items()}
    c4_pass = bool(c4) and all(c4.values())
    guard_fail = [k for k, v in c3.items() if not v["pass"]]
    passes = [c1["pass"], c2["pass"], not guard_fail, c4_pass]
    return {"condition_1_native_isolated_drop": c1, "condition_2_mediapipe_isolated_drop": c2,
            "condition_3_all_cases_no_newly_wrong": {"pass": not guard_fail, "failing": guard_fail, "cells": c3},
            "condition_4_contracts": {"pass": c4_pass, **c4},
            "all_conditions_hold": all(passes),
            "decision": "ADMISSION_CONDITIONS_MET_PENDING_CODEX_VERIFICATION" if all(passes) else "HOLD",
            "web_integration_halted": not all(passes),
            "production_authorised": False,
            "note": "linear is never replaced by previous; a failing case is never removed; HOLD is never "
                    "promoted. Synthetic stress on reused/training-exposed rows; no significance claim."}


# --- data preparation (needs R/TG/api, passed in) -----------------------------------
def raw_hash(R, *arrays) -> str:
    return hashlib.sha256("|".join(R.sha256_array(a) for a in arrays).encode()).hexdigest()


def prepare_base(R, api, group, row, tol) -> dict:
    if group == "native":
        nat = api["F"].load_native(row["meta"], {R.NATIVE_ROLE}, set(R.NATIVE_SIGNERS), api["native_roots"])
        if nat["sha256"] != row["files"]["native"][1]:
            raise IrregularError(f"{row['id']}: loader sha differs from metadata")
        xy, pr, ts, diff = nat["xy"], nat["present"], nat["ts"], None
        x0 = api["build_channels"](xy, pr, ts)["x"]
    else:
        xy, pr, ts, x0, diff = R.load_mp(api, row, tol)
    clock = R.clock_report(ts)
    if not (clock["strictly_increasing"] and clock["nominal_30hz"]):
        raise IrregularError(f"{row['id']}: original clock is not nominal 30 Hz", {"clock": clock})
    return {"xy": xy, "pr": pr, "ts": ts, "x": x0,
            "meta": {"frames": len(ts), "original_clock": clock, "mp_prepared_vs_rebuilt_max_abs_diff": diff,
                     "source_raw_sha256": raw_hash(R, xy, pr, ts), "original_input_sha256": R.sha256_array(x0)}}


def prepare_case(R, TG, api, base, sample_id, case):
    xy, pr, ts = base["xy"], base["pr"], base["ts"]
    src_hash = base["meta"]["source_raw_sha256"]
    c = C.corrupt(xy, pr, ts, sample_id, case)
    info = check_corruption(xy, pr, ts, c, case, sample_id)
    cxy, cpr, cts = c["xy"], c["present"], c["ts"]
    c_hash = raw_hash(R, cxy, cpr, cts)
    res = {m: TG.restore(cxy, cpr, cts, m) for m in ("previous", "linear")}      # corrupted arrays only
    if raw_hash(R, xy, pr, ts) != src_hash or raw_hash(R, cxy, cpr, cts) != c_hash:
        raise IrregularError(f"{sample_id}/{case}: source or corrupted arrays were mutated")
    restored = R.check_restored(res, cxy, cts)      # H/L same grid/masks, disjoint cover, verbatim exact
    lin = res["linear"]
    brackets = bracket_check(cts, cpr, lin["ts"], lin["exact_source_index"], lin["estimated"])
    bc = api["build_channels"]
    x = {"corrupted": bc(cxy, cpr, cts)["x"]}
    for m in ("previous", "linear"):
        x[m] = bc(res[m]["xy"], res[m]["present"], res[m]["ts"])["x"]
    t, idx, rts = len(ts), c["indices"], lin["ts"]
    ep = {"source": [int(ts[0]), int(ts[-1])], "corrupted": [int(cts[0]), int(cts[-1])],
          "restored": [int(rts[0]), int(rts[-1])]}
    meta = {"case": case, "indices": idx.tolist(), "source_frames": t, "retained_frames": len(idx),
            "dropped_frames": t - len(idx), "restored_frames": len(rts), **info,
            "endpoints_us": ep, "duration_us": {k: v[1] - v[0] for k, v in ep.items()},
            "restored": restored, "brackets": brackets,
            "raw_sha256": {"source": src_hash, "corrupted": c_hash,
                           **{m: raw_hash(R, res[m]["xy"], res[m]["present"], res[m]["ts"]) for m in res}},
            "input_sha256": {"original": base["meta"]["original_input_sha256"],
                             **{a: R.sha256_array(v) for a, v in x.items()}}}
    return x, meta


def timegrid_summary(rows) -> dict:
    rk = ("exact_frames", "interior_frames", "interior_frames_long_gap", "observed_slots",
          "estimated_slots", "unsupported_interior_slots")
    bk = ("estimated_frames", "unsupported_frames", "unsupported_slots", "frames_with_long_bracket")
    fk = ("source_frames", "retained_frames", "dropped_frames", "restored_frames")
    return {"rows_identity": sum(r["case"]["restored"]["identity"] for r in rows),
            **{k: sum(r["case"][k] for r in rows) for k in fk},
            **{k: sum(r["case"]["restored"][k] for r in rows) for k in rk},
            **{f"bracket_{k}": sum(r["case"]["brackets"][k] for r in rows) for k in bk}}


# --- main ----------------------------------------------------------------------------
def _precheck(cfg):
    if not isinstance(cfg, dict):
        raise IrregularError("config must be a JSON object")
    for k in ("helper_dir", "output_dir"):
        if not isinstance(cfg.get(k), str) or not os.path.isabs(cfg[k]):
            raise IrregularError(f"config {k} must be an absolute string")
    roots = []
    for k in ("native_roots", "mp_roots"):
        v = cfg.get(k)
        if not isinstance(v, list) or not v or any(not isinstance(r, str) or not os.path.isabs(r) for r in v):
            raise IrregularError(f"{k} must be a non-empty list of absolute paths")
        roots += v
    out = norm(cfg["output_dir"])
    if any(under(out, norm(r)) for r in roots + [cfg["helper_dir"]]):
        raise IrregularError("output_dir lies inside a source root or helper_dir")
    pins = cfg.get("pins")
    if not isinstance(pins, dict) or not pins or any(not os.path.isabs(p) for p in pins):
        raise IrregularError("pins must be a non-empty object of absolute paths")
    pinned = {norm(p) for p in pins}
    miss = [f for f in HELPER_FILES if norm(os.path.join(cfg["helper_dir"], f)) not in pinned]
    if miss:
        raise IrregularError(f"helper files not pinned: {miss}")
    if cfg.get("repro_indices") != REPRO_INDICES:
        raise IrregularError(f"repro_indices must be exactly {REPRO_INDICES}")


def main(argv=None):
    started = time.time()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg_path = os.path.abspath(args.config)
    cfg_bytes = Path(cfg_path).read_bytes()
    cfg = json.loads(cfg_bytes.decode("utf-8-sig"))
    _precheck(cfg)
    if norm(os.path.dirname(C.__file__)) != norm(os.path.dirname(os.path.abspath(__file__))):
        raise IrregularError(f"corruptions imported from unexpected location {C.__file__}")
    out_dir = Path(cfg["output_dir"])
    if os.path.lexists(out_dir):
        raise IrregularError(f"output_dir {out_dir} exists; outputs are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=False)
    code = {"irregular_audit.py": sha256_file(__file__), "corruptions.py": sha256_file(C.__file__)}
    state = {"stage": "start", "group": None, "row": None, "case": None, "prepared": {g: 0 for g in GROUPS},
             "forwards_repro": 0, "forwards_new": 0, "details": None, "pins_before": None}
    try:
        return _run(cfg, cfg_path, cfg_bytes, out_dir, code, state, started)
    except BaseException as exc:
        failure = {"status": "FAILED", "stage": state["stage"], "group": state["group"], "row": state["row"],
                   "case": state["case"], "error_class": type(exc).__name__, "error": str(exc),
                   "details": getattr(exc, "details", None) or state["details"],
                   "counts": {"rows_prepared": state["prepared"], "forwards_repro": state["forwards_repro"],
                              "forwards_new": state["forwards_new"], "training": 0, "camera": 0,
                              "user_data": 0, "downloads": 0},
                   "pins_before": state["pins_before"], "code_sha256": code,
                   "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
                   "at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            write_json_exclusive(out_dir / "FAILURE.json", failure, strict=False)
        except Exception as wexc:                                   # original failure still raised
            print(f"could not write FAILURE.json: {type(wexc).__name__}: {wexc}", file=sys.stderr)
        raise


def _run(cfg, cfg_path, cfg_bytes, out_dir, code, state, started):
    # 1. every pin (incl. helper, timegrid, model, manifests, plan, APIs) before anything is imported/read
    state["stage"] = "pins"
    pins_before = {p: sha256_file(p) for p in cfg["pins"]}
    state["pins_before"] = pins_before
    bad = {p: got for p, got in pins_before.items() if got != cfg["pins"][p]}
    if bad:
        raise IrregularError("pin mismatch", {"mismatched": bad})
    pin_of = {norm(p): d for p, d in cfg["pins"].items()}

    # 2. audited helper + timegrid, unmodified, from helper_dir only
    state["stage"] = "helper_import"
    helper = cfg["helper_dir"]
    if any(n in sys.modules for n in ("run_audit", "timegrid")):
        raise IrregularError("run_audit/timegrid already imported from elsewhere")
    sys.path.insert(0, helper)
    import run_audit as R
    import timegrid as TG
    for mod, f in ((R, "run_audit.py"), (TG, "timegrid.py")):
        p = os.path.join(helper, f)
        if norm(mod.__file__) != norm(p) or sha256_file(p) != pin_of[norm(p)]:
            raise IrregularError(f"{f} not the pinned file in helper_dir ({mod.__file__})")
    if R.TG is not TG:
        raise IrregularError("run_audit uses a different timegrid module")
    code.update({f"helper/{f}": pin_of[norm(os.path.join(helper, f))] for f in HELPER_FILES})
    frozen = {"N_CLASSES": (R.N_CLASSES, 93), "NATIVE_N": (R.NATIVE_N, 699), "MP_N": (R.MP_N, 113),
              "NOMINAL_STEP_US": (TG.NOMINAL_STEP_US, 1e6 / 30.0), "EXACT_TOL_US": (TG.EXACT_TOL_US, 1),
              "CLOCK_TOL_US": (TG.CLOCK_TOL_US, 2), "N_JOINTS": (TG.N_JOINTS, C.N_JOINTS)}
    if any(a != b for a, b in frozen.values()):
        raise IrregularError("helper/timegrid constants differ from the frozen method", {"constants": frozen})

    state["stage"] = "config"
    R.validate_config(cfg)              # old keys; extra helper_dir is ignored there, checked in _precheck
    tol = float(cfg["tolerance"])
    roots = {"native": cfg["native_roots"], "mediapipe": cfg["mp_roots"]}
    lab = cfg["lab_root"]
    sys.path[1:1] = [lab, os.path.join(lab, "src"), cfg["pilot_src"], cfg["loader_dir"]]

    # 3. row lists and every listed row file hash, before any array or model load
    state["stage"] = "metadata"
    rows = {"native": R.load_native_rows(cfg["native_metadata"]), "mediapipe": R.load_mp_rows(cfg["mp_manifest"])}
    state["stage"] = "source_hashes_before"
    sources_before = {g: R.hash_sources(rows[g], roots[g]) for g in GROUPS}
    for g in GROUPS:
        R.check_listed(rows[g], sources_before[g], g)

    state["stage"] = "saved_material"
    classes, a_native, _ = R.load_saved_native(cfg["saved_native_logits"], rows["native"])
    a_mp, mp_arms = R.load_saved_mp(cfg["saved_mp_logits"], rows["mediapipe"], classes)
    saved_a = {"native": a_native, "mediapipe": a_mp}

    # 4. local APIs; their files must be pinned (normalisation/kernel unchanged)
    state["stage"] = "imports"
    import torch
    import tools_prep_union93_candidate as u
    from pilot.data import build_channels, collate
    from pilot.web_input import to_native_like
    import framerate_impl as F
    module_files = {n: os.path.abspath(sys.modules[n].__file__) for n in API_MODULES}
    modules_before = {k: sha256_file(p) for k, p in module_files.items()}
    unpinned = {k: p for k, p in module_files.items() if pin_of.get(norm(p)) != modules_before[k]}
    if unpinned:
        raise IrregularError("imported API modules are not pinned or differ from pins", {"modules": unpinned})
    camera_loaded = [m for m in CAMERA_MODULES if m in sys.modules]
    api = {"F": F, "build_channels": build_channels, "to_native_like": to_native_like,
           "native_roots": roots["native"], "mp_roots": roots["mediapipe"]}

    # 5. preflight ALL rows x cases; every failure recorded, nothing filtered, no inference
    metas, failures = {g: [] for g in GROUPS}, []

    def fail(g, i, r, c, stage, exc):
        failures.append({"group": g, "index": i, "id": r["id"], "case": c, "stage": stage,
                         "error": f"{type(exc).__name__}: {exc}", "details": getattr(exc, "details", None)})

    state["stage"] = "preflight"
    for g in GROUPS:
        state["group"] = g
        for i, r in enumerate(rows[g]):
            state["row"], state["case"] = r["id"], None
            try:
                base = prepare_base(R, api, g, r, tol)
            except Exception as exc:
                fail(g, i, r, None, "base", exc)
                metas[g].append(None)
                continue
            m = {"base": base["meta"], "cases": {}}
            for c in CASES:
                state["case"] = c
                try:
                    m["cases"][c] = prepare_case(R, TG, api, base, r["id"], c)[1]
                except Exception as exc:
                    fail(g, i, r, c, "case", exc)
                    m["cases"][c] = None
            metas[g].append(m)
            state["prepared"][g] += 1
    state.update(group=None, row=None, case=None, stage="preflight_gate")
    if failures:
        counts = {}
        for f in failures:
            k = f"{f['group']}/{f['case'] or 'base'}"
            counts[k] = counts.get(k, 0) + 1
        raise IrregularError("preflight failed before any scoring; no rows filtered",
                             {"failures_n": counts, "failures": failures})

    # 6. model
    state["stage"] = "model"
    torch.set_num_threads(R.FIXED_THREADS)
    payload = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    model, model_classes = u.rebuild_from_payload(payload)
    model.eval()
    if [str(c) for c in model_classes] != classes:
        raise IrregularError("model class order differs from saved class order")
    n_cls = R.N_CLASSES

    def forward(x):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 280 or x.dtype != np.float32 \
                or not np.isfinite(x).all():
            raise IrregularError(f"bad forward input {getattr(x, 'shape', None)}")
        bx, pad, _ = collate([{"x": x, "frames": int(len(x)), "label": 0}])   # label unused by model
        with torch.inference_mode():
            lg = model(torch.from_numpy(bx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
        if lg.shape != (n_cls,) or not np.isfinite(lg).all():
            raise IrregularError(f"model output shape {lg.shape} or non-finite")
        return lg

    # 7. reproduce stored original scores (6 forwards) before any new inference
    state["stage"] = "reproduction"
    repro = []
    for g in GROUPS:
        for i in cfg["repro_indices"][g]:
            r = rows[g][i]
            state["group"], state["row"] = g, r["id"]
            base = prepare_base(R, api, g, r, tol)
            if base["meta"] != metas[g][i]["base"]:
                raise IrregularError(f"{r['id']}: original input differs from preflight")
            lg = forward(base["x"])
            state["forwards_repro"] += 1
            diff = float(np.max(np.abs(lg - saved_a[g][i])))
            ok = diff <= tol and int(np.argmax(lg)) == int(np.argmax(saved_a[g][i]))
            repro.append({"group": g, "index": i, "id": r["id"], "arm": "original", "max_abs_diff": diff,
                          "top1_now": classes[int(np.argmax(lg))],
                          "top1_saved": classes[int(np.argmax(saved_a[g][i]))], "ok": bool(ok)})
    reproduction = {"indices": cfg["repro_indices"], "atol": tol, "rows": repro}
    if len(repro) != 6 or not all(x["ok"] for x in repro):
        raise IrregularError("stored score reproduction failed; no new inference performed",
                             {"reproduction": reproduction})

    # 8. new inference: 3 arms x 3 cases per row; every input re-prepared and compared to preflight
    state["stage"] = "inference"
    logits, results = {}, {}
    for g in GROUPS:
        n = len(rows[g])
        lg = np.full((n, len(CASES), len(ARMS), n_cls), np.nan, dtype=np.float64)
        lg[:, :, 0] = saved_a[g][:, None, :]
        per_case = {c: [] for c in CASES}
        for i, r in enumerate(rows[g]):
            state["group"], state["row"], state["case"] = g, r["id"], None
            base = prepare_base(R, api, g, r, tol)
            if base["meta"] != metas[g][i]["base"]:
                raise IrregularError(f"{r['id']}: original input differs from preflight")
            for ci, c in enumerate(CASES):
                state["case"] = c
                x, cm = prepare_case(R, TG, api, base, r["id"], c)
                if cm != metas[g][i]["cases"][c]:
                    raise IrregularError(f"{r['id']}/{c}: inputs/metadata differ from preflight")
                for ai in range(1, len(ARMS)):
                    lg[i, ci, ai] = forward(x[ARMS[ai]])
                    state["forwards_new"] += 1
                per_case[c].append({"index": i, "id": r["id"], "signer_id": r["signer_id"], "true": r["label"],
                                    "true_index": classes.index(r["label"]), "recording_id": R.recording_id(r["id"]),
                                    "arms": {a: R.score_row(lg[i, ci, k], classes, r["label"])
                                             for k, a in enumerate(ARMS)},
                                    "base": base["meta"], "case": cm})
        if not np.isfinite(lg).all():
            raise IrregularError(f"{g}: non-finite logits in score tensor")
        if not all(np.array_equal(lg[:, ci, 0], saved_a[g]) for ci in range(len(CASES))):
            raise IrregularError(f"{g}: original logits are not the saved scores in every case")
        logits[g], results[g] = lg, per_case
    state.update(group=None, row=None, case=None)
    expected_new = sum(len(rows[g]) for g in GROUPS) * len(CASES) * (len(ARMS) - 1)
    if state["forwards_new"] != expected_new or expected_new != 7308 or state["forwards_repro"] != 6:
        raise IrregularError("forward count mismatch", {"new": state["forwards_new"], "expected_new": expected_new,
                                                        "repro": state["forwards_repro"]})

    # 9. scores
    state["stage"] = "write_scores"
    scores_sha = {}
    for g in GROUPS:
        p = out_dir / f"SCORES_{g}.npz"
        ids = [r["id"] for r in results[g][CASES[0]]]
        with open(p, "xb") as fh:
            np.savez_compressed(fh, logits=logits[g], cases=np.array(CASES), arms=np.array(ARMS),
                                ids=np.array(ids), classes=np.array(classes),
                                true_indices=np.array([classes.index(r["label"]) for r in rows[g]], dtype=np.int64))
        scores_sha[g] = sha256_file(p)

    # 10. hashes after
    state["stage"] = "hashes_after"
    pins_after = {p: sha256_file(p) for p in cfg["pins"]}
    sources_after = {g: R.hash_sources(rows[g], roots[g]) for g in GROUPS}
    modules_after = {k: sha256_file(p) for k, p in module_files.items()}
    code_after = {"irregular_audit.py": sha256_file(__file__), "corruptions.py": sha256_file(C.__file__),
                  **{f"helper/{f}": sha256_file(os.path.join(helper, f)) for f in HELPER_FILES}}
    config_sha = hashlib.sha256(cfg_bytes).hexdigest()
    config_after = sha256_file(cfg_path)
    changed = {"pins": [p for p in pins_before if pins_before[p] != pins_after[p]],
               "sources": {g: [k for k in sources_before[g] if sources_before[g][k] != sources_after[g][k]]
                           for g in GROUPS},
               "modules": [k for k in modules_before if modules_before[k] != modules_after[k]],
               "code": code_after != code, "config": config_after != config_sha}
    write_json_exclusive(out_dir / "SOURCE_HASHES.json", {
        "pins": {"before": pins_before, "after": pins_after},
        "sources": {"before": sources_before, "after": sources_after},
        "imported_modules": {"files": module_files, "before": modules_before, "after": modules_after},
        "code": {"before": code, "after": code_after},
        "config": {"path": cfg_path, "before": config_sha, "after": config_after}, "changed": changed})
    if changed["pins"] or any(changed["sources"].values()) or changed["modules"] or changed["code"] or changed["config"]:
        raise IrregularError("hashes changed during run", {"changed": changed})

    # 11. result
    state["stage"] = "write_result"
    groups_out, stats = {}, {g: {} for g in GROUPS}
    for g in GROUPS:
        cases_out = {}
        for c in CASES:
            rs = results[g][c]
            comps = {f"{s}->{t}": R.compare_arms(rs, s, t) for s, t in COMPARISONS}
            correct = {a: sum(r["arms"][a]["correct"] for r in rs) for a in ARMS}
            per_signer = breakdown(rs, "signer_id")
            stats[g][c] = {"correct": correct, "per_signer": per_signer,
                           "newly_wrong": comps["corrupted->linear"]["losses_n"]}
            cases_out[c] = {"n": len(rs), "per_arm_correct": correct, "per_signer": per_signer,
                            "per_class": breakdown(rs, "true"), "comparisons": comps,
                            "original_correct_corrupted_wrong_recoveries": recoveries(rs),
                            "timegrid_summary": timegrid_summary(rs), "rows": rs}
        groups_out[g] = {"description": ("native provided-coordinate development rows (annotation_dev, validation)"
                                         if g == "native" else "cached MediaPipe training-exposed diagnostic rows"),
                         "n_rows": len(rows[g]), "source_recordings": R.recording_counts(results[g][CASES[0]]),
                         "cases": cases_out}
    contracts = {"preflight_input_source_longgap_provenance_failures_zero": not failures,
                 "reproduction_within_tolerance": all(x["ok"] for x in repro),
                 "original_logits_equal_saved_every_case": True, "forward_counts_match": True,
                 "pins_unchanged": not changed["pins"], "sources_unchanged": not any(changed["sources"].values()),
                 "imported_modules_unchanged": not changed["modules"], "code_unchanged": not changed["code"],
                 "config_unchanged": not changed["config"]}
    result = {
        "status": "COMPLETE_PENDING_CODEX_VERIFICATION",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "nature": "synthetic irregular-time inference diagnostic on a frozen model; no training, no parameter search",
        "cases": list(CASES), "arm_order": list(ARMS), "primary": "linear", "mechanism_control": "previous",
        "primary_case": PRIMARY_CASE,
        "timegrid_rules": {"nominal_step_us": TG.NOMINAL_STEP_US, "exact_tol_us": TG.EXACT_TOL_US,
                           "clock_tol_us": TG.CLOCK_TOL_US, "max_bracket_gap": "2P + 2 us"},
        "corruption_rules": {"domain": C.DOMAIN, "max_source_gap": C.MAX_SOURCE_GAP,
                             "max_offset_us": C.MAX_OFFSET_US},
        "model": {"checkpoint": cfg["checkpoint"], "sha256": pin_of[norm(cfg["checkpoint"])], "classes": len(classes)},
        "saved_mp_arms": mp_arms, "reproduction": reproduction,
        "admission": admission(stats, contracts),
        "groups": groups_out, "interpretation_limits": LIMITS,
        "execution_counts": {"forwards_repro": state["forwards_repro"], "forwards_new": state["forwards_new"],
                             "forwards_total": state["forwards_repro"] + state["forwards_new"],
                             "training": 0, "camera": 0, "user_data": 0, "downloads": 0},
        "camera_modules_in_sys_modules": camera_loaded,
        "hashes": {"code": code, "config": config_sha, "plan_file": pin_of[norm(cfg["plan_file"])],
                   "scores": scores_sha, "imported_modules": modules_before},
        "seconds": round(time.time() - started, 2),
    }
    write_json_exclusive(out_dir / "RESULT.json", result)
    print(json.dumps({"decision": result["admission"]["decision"],
                      **{f"{g}/{c}": stats[g][c]["correct"] for g in GROUPS for c in CASES}},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                 # IrregularError or helper AuditError; FAILURE.json already written
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
