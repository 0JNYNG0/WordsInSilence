"""Partial-hand reconstruction diagnostic on the frozen 832 TCN (offline, FROZEN_PLAN.json).

Four arms per row; groups are never pooled:
  original         stored original scores (native original_logits / MP arm 'mediapipe')
  original_repair  build_channels(handgap.restore(original raw))
  masked           build_channels(handgap.mask_one_hand(original raw, row id))
  masked_repair    build_channels(handgap.restore(masked raw))   -- restore never sees the hidden originals
Coordinate metrics (evaluator only) compare masked_repair estimates and a previous-original-frame copy with
the hidden original points, in units of the ORIGINAL build_channels body_scale. Every arm is prepared and
verified for all 812 rows before the model is loaded. No training, no threshold/parameter search, no camera,
no network, no deployment. Prior code is imported unmodified from pins:
  run_audit (R), timegrid (TG)   from cfg.helper_dir

Run: python run_handgap.py --config <absolute config.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

import handgap as HG

GROUPS = ("native", "mediapipe")
ARMS = ("original", "original_repair", "masked", "masked_repair")
PAIRS = (("original", "original_repair"), ("masked", "masked_repair"), ("original", "masked"),
         ("original", "masked_repair"))
REPRO_INDICES = {"native": [0, 349, 698], "mediapipe": [0, 56, 112]}
GROUP_N = {"native": 699, "mediapipe": 113}
N_FEATURES = 280
HELPER_FILES = ("run_audit.py", "timegrid.py")
API_MODULES = ("tools_prep_union93_candidate", "pilot.data", "pilot.web_input", "framerate_impl")
MODEL_API_MODULE = "model_masked_v1"
CAMERA_MODULES = ("cv2", "mediapipe")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = ("run_handgap.py", "handgap.py")
PATH_FIELDS = ("lab_root", "pilot_src", "loader_dir", "helper_dir", "native_metadata", "mp_manifest",
               "saved_native_logits", "saved_mp_logits", "checkpoint", "plan_file", "output_dir")
PINNED_FIELDS = ("native_metadata", "mp_manifest", "saved_native_logits", "saved_mp_logits", "checkpoint",
                 "plan_file")
# evaluator's own copy of the frozen rule (independent of handgap.py; equality checked at run time)
MASK_SALT, MASK_PERIOD, MASK_FIRST, MAX_SPAN_US = "WIS_HAND_GAP_V1|", 6, 2, 66668
HAND_SLICES = {"left": slice(25, 46), "right": slice(46, 67)}
SIDES = ("left", "right")
SUBSETS = {"all": tuple(range(21)), "wrist": (0,), "palm": (1, 5, 9, 13, 17),
           "fingers": tuple(k for k in range(21) if k not in (0, 1, 5, 9, 13, 17))}
COORD_METHODS = ("linear", "previous")
LIMITS = [
    "Offline synthetic diagnostic on the same frozen model with one-frame lookahead; not a causal live tracker. "
    "No training, no parameter search, no operational admission, no webcam or deployment claim even if every "
    "gate holds.",
    "native = reused development rows, mediapipe = training-exposed cached rows; groups are never pooled and "
    "neither establishes independent signer or signed-form generalisation. No significance claim.",
    "The mask is a repeated isolated hand-only dropout stress schedule, not an estimate of real dropout frequency.",
    "Original observations are detector estimates, not anatomical ground truth. Natural gaps have no target, so "
    "only artificially hidden points carry coordinate error.",
    "masked_repair changes both injected and naturally absent hand slots; recognition changes cannot be "
    "attributed exclusively to recovery of injected points.",
    "A same-channel fill is not evidence of correct hand identity; swapped hands are not detected or fixed.",
    "build_channels normalisation is unchanged code but re-estimated per arm (including estimated points); raw "
    "observation preservation does not imply preserved normalised features.",
    "The previous-frame copy is a coordinate comparator only; it is never scored by the model.",
    "The net diagnostic gate may pass with individual masked regressions; strict no-newly-wrong is reported "
    "separately and is not waived.",
]


class RunError(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


# --- pure helpers (no prior helper / torch import) -----------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha_array(x) -> str:
    x = np.ascontiguousarray(x)
    return hashlib.sha256(f"{x.shape}|{x.dtype}|".encode() + x.tobytes()).hexdigest()


def raw_hash(*arrays) -> str:
    return hashlib.sha256("|".join(sha_array(a) for a in arrays).encode()).hexdigest()


def norm(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def under(path, root) -> bool:
    try:
        return os.path.commonpath([norm(path), norm(root)]) == norm(root)
    except ValueError:
        return False


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json_exclusive(path, value, strict=True):
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=1, allow_nan=not strict,
                  default=None if strict else _plain)
        fh.write("\n")


def jsonable(v):
    return json.loads(json.dumps(v, default=_plain, allow_nan=False))


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=bool(np.issubdtype(a.dtype, np.floating))))


def _pb(m):
    m = np.asarray(m)
    return m.copy() if m.dtype == np.bool_ else m == 1


def features_ok(x, frames) -> bool:
    return isinstance(x, np.ndarray) and x.dtype == np.float32 and x.shape == (frames, N_FEATURES) \
        and bool(np.isfinite(x).all())


def precheck(cfg, code_dir=CODE_DIR):
    """Config shape, required pins and output location. Touches no file."""
    if not isinstance(cfg, dict):
        raise RunError("config must be a JSON object")
    for k in PATH_FIELDS:
        if not isinstance(cfg.get(k), str) or not os.path.isabs(cfg[k]):
            raise RunError(f"config {k} must be an absolute string")
    for k in ("native_roots", "mp_roots"):
        v = cfg.get(k)
        if not isinstance(v, list) or not v or any(not isinstance(r, str) or not os.path.isabs(r) for r in v):
            raise RunError(f"{k} must be a non-empty list of absolute paths")
    pins = cfg.get("pins")
    if not isinstance(pins, dict) or not pins or any(
            not isinstance(p, str) or not os.path.isabs(p) or not isinstance(d, str) or len(d) != 64
            or any(c not in "0123456789abcdef" for c in d) for p, d in pins.items()):
        raise RunError("pins must be a non-empty object {absolute path: sha256 hex}")
    pinned = {norm(p) for p in pins}
    need = [cfg[k] for k in PINNED_FIELDS] + [os.path.join(cfg["helper_dir"], f) for f in HELPER_FILES] + \
        [os.path.join(code_dir, "handgap.py")]
    miss = [p for p in need if norm(p) not in pinned]
    if miss:
        raise RunError("required files not pinned", {"missing": miss})
    if cfg.get("repro_indices") != REPRO_INDICES:
        raise RunError(f"repro_indices must be exactly {REPRO_INDICES}")
    if type(cfg.get("threads")) is not int or cfg["threads"] != 2:
        raise RunError("threads must be 2")
    if not isinstance(cfg.get("tolerance"), (int, float)) or float(cfg["tolerance"]) != 1e-5:
        raise RunError("tolerance must be 1e-5")
    out, lab = cfg["output_dir"], cfg["lab_root"]
    ops = os.path.join(lab, "ops")
    if not under(out, ops) or norm(out) == norm(ops):
        raise RunError("output_dir must be a fresh diagnostic subdirectory under lab/ops")
    if under(out, code_dir):
        raise RunError("output_dir must not be the code directory or inside it")
    protected = cfg["native_roots"] + cfg["mp_roots"] + [
        cfg["helper_dir"], code_dir, cfg["pilot_src"], cfg["loader_dir"], os.path.join(lab, "src"),
        os.path.join(lab, "web"), os.path.join(lab, "tests"),
        *(os.path.dirname(cfg[k]) for k in ("saved_native_logits", "saved_mp_logits", "checkpoint",
                                            "native_metadata", "mp_manifest"))]
    # a protected directory that contains lab/ops itself (e.g. a checkpoint directly in lab_root) cannot be
    # excluded by location; such files are protected by pins and exclusive file creation instead
    bad = [r for r in protected if not under(ops, r) and under(out, r)]
    if bad:
        raise RunError("output_dir lies inside a protected source/helper/code/prior-output/model directory",
                       {"protected": bad})


def verify_mask(xy, pr, ts, sample_id, m) -> dict:
    """Independent re-derivation of the frozen mask schedule; every untouched value must be identical."""
    xy, pr, ts = np.asarray(xy), np.asarray(pr), np.asarray(ts)
    if set(m) != {"xy", "present", "ts", "injected", "events", "schedule", "offset", "first_side"}:
        raise RunError("mask result keys wrong")
    if not (m["xy"].shape == xy.shape and m["xy"].dtype == xy.dtype and m["present"].shape == pr.shape
            and m["present"].dtype == pr.dtype and m["injected"].shape == pr.shape and m["injected"].dtype == np.bool_):
        raise RunError("mask output shape/dtype differs from input")
    if not _same(m["ts"], ts):
        raise RunError("mask changed timestamps")
    pb, ti, t = _pb(pr), [int(v) for v in ts], len(ts)
    d = hashlib.sha256((MASK_SALT + sample_id).encode("utf-8")).digest()
    offset, first = d[0] % MASK_PERIOD, d[1] % 2
    exp, events, scheduled = np.zeros(pb.shape, dtype=bool), [], 0
    for k, i in enumerate(range(MASK_FIRST + offset, t - 2, MASK_PERIOD)):
        scheduled += 1
        side = SIDES[(first + k) % 2]
        s = HAND_SLICES[side]
        if pb[i - 1:i + 2, s].all() and ti[i + 1] - ti[i - 1] <= MAX_SPAN_US:
            exp[i, s] = True
            events.append({"frame": i, "side": side})
    if not np.array_equal(m["injected"], exp) or m["events"] != events or m["offset"] != offset \
            or m["first_side"] != SIDES[first] or len(m["schedule"]) != scheduled:
        raise RunError("mask differs from the frozen schedule")
    keep = ~exp
    if not (_same(m["xy"][keep], xy[keep]) and _same(m["present"][keep], pr[keep])):
        raise RunError("mask changed values outside injected hand slots")
    if np.any(m["xy"][exp] != 0) or _pb(m["present"])[exp].any():
        raise RunError("injected slots are not xy 0 / present 0")
    return {"offset": offset, "first_side": SIDES[first], "scheduled_n": scheduled, "events_n": len(events),
            "events": events, "injected_slots": {sd: int(exp[:, HAND_SLICES[sd]].sum()) for sd in SIDES}}


def verify_restore(xy, pr, ts, res) -> dict:
    """Independent check of one handgap.restore result against exactly the arrays it received."""
    xy, pr, ts = np.asarray(xy), np.asarray(pr), np.asarray(ts)
    if set(res) != {"xy", "present", "ts", "observed", "estimated", "identity"}:
        raise RunError("restore result keys wrong")
    oxy, opr, obs, est = (np.asarray(res[k]) for k in ("xy", "present", "observed", "estimated"))
    if (oxy.dtype, opr.dtype, obs.dtype, est.dtype) != (xy.dtype, pr.dtype, np.bool_, np.bool_) \
            or oxy.shape != xy.shape or opr.shape != pr.shape or obs.shape != pr.shape or est.shape != pr.shape:
        raise RunError("restore output shape/dtype differs from contract")
    if not _same(res["ts"], ts):
        raise RunError("restore changed timestamps")
    pb, ti = _pb(pr), [int(v) for v in ts]
    if not np.array_equal(obs, pb):
        raise RunError("observed is not the received present mask")
    exp, fills = np.zeros(pb.shape, dtype=bool), []
    for side in SIDES:
        s = HAND_SLICES[side]
        for i in range(1, len(ti) - 1):
            if not pb[i, s].any() and pb[i - 1, s].all() and pb[i + 1, s].all() and ti[i + 1] - ti[i - 1] <= MAX_SPAN_US:
                exp[i, s] = True
                fills.append({"frame": i, "side": side})
    if not np.array_equal(est, exp):
        raise RunError("estimated set differs from the frozen fill rule")
    if (obs & est).any() or not np.array_equal(obs | est, _pb(opr)):
        raise RunError("observed/estimated are not a disjoint cover of output present")
    keep = ~est
    if not (_same(oxy[keep], xy[keep]) and _same(opr[keep], pr[keep])):
        raise RunError("restore changed values outside filled hand slots")
    eps = float(np.finfo(xy.dtype).eps)
    for f in fills:
        i, s = f["frame"], HAND_SLICES[f["side"]]
        w = float(Fraction(ti[i] - ti[i - 1], ti[i + 1] - ti[i - 1]))
        lo, hi = xy[i - 1, s].astype(np.float64), xy[i + 1, s].astype(np.float64)
        got = oxy[i, s].astype(np.float64)
        bound = 4 * eps * np.maximum(1.0, np.maximum(np.abs(lo), np.abs(hi)))
        if not np.isfinite(got).all() or np.any(np.abs(got - ((1 - w) * lo + w * hi)) > bound):
            raise RunError("filled coordinates are not the time-weighted interpolation", {"fill": f})
    if res["identity"] is not (not fills):
        raise RunError("identity flag inconsistent with fills")
    return {"fills_n": len(fills), "fills": fills, "slots": int(est.sum()),
            "slots_by_side": {sd: int(est[:, HAND_SLICES[sd]].sum()) for sd in SIDES}}


def event_errors(orig_xy, est_xy, events, body_scale) -> list:
    """Per injected hand event: 21 point errors / body_scale for the linear estimate and previous-frame copy."""
    s = float(body_scale) if body_scale is not None else float("nan")
    if not (math.isfinite(s) and s > 0):
        raise RunError(f"original body_scale {body_scale!r} is not positive finite")
    out = []
    for e in events:
        i, sl = e["frame"], HAND_SLICES[e["side"]]
        o = np.asarray(orig_xy[i, sl], np.float64)
        lin = np.linalg.norm(np.asarray(est_xy[i, sl], np.float64) - o, axis=1) / s
        prev = np.linalg.norm(np.asarray(orig_xy[i - 1, sl], np.float64) - o, axis=1) / s
        if lin.shape != (21,) or not (np.isfinite(lin).all() and np.isfinite(prev).all()):
            raise RunError("non-finite coordinate error", {"event": e})
        out.append({"frame": int(i), "side": e["side"], "linear": [float(v) for v in lin],
                    "previous": [float(v) for v in prev]})
    return out


def _stats(points, clips) -> dict:
    p, c = np.asarray(points, np.float64), np.asarray(clips, np.float64)
    return {"points": int(p.size), "point_mean": float(p.mean()), "point_median": float(np.median(p)),
            "point_p95": float(np.percentile(p, 95)), "clips": int(c.size),
            "clip_mean_mean": float(c.mean()), "clip_mean_median": float(np.median(c))}


def coord_summary(rows_events) -> dict:
    """rows_events: one list of event_errors() dicts per row. Null statistics when nothing was injected."""
    n_ev = sum(len(e) for e in rows_events)
    out = {"available": n_ev > 0, "events": n_ev, "clips_with_events": sum(bool(e) for e in rows_events),
           "units": "Euclidean error / original build_channels body_scale"}
    for meth in COORD_METHODS:
        out[meth] = {}
        for name, idx in SUBSETS.items():
            if not n_ev:
                out[meth][name] = None
                continue
            pts, clips = [], []
            for evs in rows_events:
                if evs:
                    v = np.concatenate([np.asarray(e[meth], np.float64)[list(idx)] for e in evs])
                    pts.append(v)
                    clips.append(float(v.mean()))
            out[meth][name] = _stats(np.concatenate(pts), clips)
    return out


def coord_gate(summary) -> bool:
    return bool(summary["available"] and summary["linear"]["all"]["point_mean"]
                < summary["previous"]["all"]["point_mean"])


def cache_plan(hx) -> dict:
    """Fixed order: an arm reuses the first earlier arm of the same row with a byte-identical built input."""
    plan, seen = {"original": "saved_original"}, {hx["original"]: "original"}
    for a in ARMS[1:]:
        if hx[a] in seen:
            plan[a] = f"reuse:{seen[hx[a]]}"
        else:
            plan[a] = "new"
            seen[hx[a]] = a
    return plan


def breakdown(rows, key) -> dict:
    out = {}
    for r in rows:
        g = out.setdefault(str(r[key]), {"n": 0, **{a: 0 for a in ARMS}})
        g["n"] += 1
        for a in ARMS:
            g[a] += int(r["arms"][a]["correct"])
    return dict(sorted(out.items()))


def state_digest(model) -> dict:
    import torch

    def h(t):
        t = t.detach().cpu().contiguous()
        return sha_array((t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy())
    tensors = {f"state:{k}": h(v) for k, v in model.state_dict().items()}
    tensors.update({f"param:{k}": h(v) for k, v in model.named_parameters()})
    tensors.update({f"buffer:{k}": h(v) for k, v in model.named_buffers()})
    return {"tensors": tensors, "combined": hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest(),
            "training_modules": sorted(n or "<root>" for n, m in model.named_modules() if m.training)}


def code_hashes(cfg) -> dict:
    out = {f: sha256_file(os.path.join(CODE_DIR, f)) for f in CODE_FILES}
    out.update({f"helper/{f}": sha256_file(os.path.join(cfg["helper_dir"], f)) for f in HELPER_FILES})
    return out


# --- main ----------------------------------------------------------------------------------------
def main(argv=None):
    started = time.time()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    if not os.path.isabs(args.config):
        raise RunError("--config must be an absolute path")
    cfg_path = os.path.abspath(args.config)
    cfg_bytes = Path(cfg_path).read_bytes()
    cfg = json.loads(cfg_bytes.decode("utf-8-sig"))
    precheck(cfg)
    out_dir = Path(cfg["output_dir"])
    if os.path.lexists(out_dir):
        raise RunError(f"output_dir {out_dir} exists; outputs are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=False)
    code = code_hashes(cfg)
    state = {"stage": "start", "group": None, "id": None, "prepared_rows": {g: 0 for g in GROUPS},
             "forwards": {"reproduction": 0, "new": 0}, "reused": {}, "details": None, "pins_before": None}
    try:
        return _run(cfg, cfg_path, cfg_bytes, out_dir, code, state, started)
    except BaseException as exc:
        failure = {"status": "FAILED", "stage": state["stage"], "group": state["group"], "id": state["id"],
                   "error_class": type(exc).__name__, "error": str(exc),
                   "details": getattr(exc, "details", None) or state["details"],
                   "counts": {"rows_prepared": state["prepared_rows"], "forwards": state["forwards"],
                              "reused": state["reused"], "training": 0, "camera": 0, "user_data": 0,
                              "downloads": 0},
                   "pins_before": state["pins_before"], "code_sha256": code,
                   "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
                   "seconds": round(time.time() - started, 2),
                   "at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            write_json_exclusive(out_dir / "FAILURE.json", failure, strict=False)
        except Exception as wexc:
            print(f"could not write FAILURE.json: {type(wexc).__name__}: {wexc}", file=sys.stderr)
        raise


def _import_pinned(name, directory, pin_of):
    if name in sys.modules:
        raise RunError(f"{name} already imported from elsewhere")
    sys.path.insert(0, directory)
    try:
        mod = __import__(name)
    finally:
        sys.path.remove(directory)
    p = os.path.join(directory, f"{name}.py")
    if norm(mod.__file__) != norm(p) or sha256_file(p) != pin_of.get(norm(p)):
        raise RunError(f"{name} is not the pinned file in {directory} ({mod.__file__})")
    return mod


def _camera_check():
    loaded = [m for m in CAMERA_MODULES if m in sys.modules]
    if loaded:
        raise RunError(f"camera modules imported: {loaded}")


def _run(cfg, cfg_path, cfg_bytes, out_dir, code, state, started):
    # 1. every pin before anything else is read; the candidate must be the pinned file next to this runner
    state["stage"] = "pins"
    pins_before = {p: sha256_file(p) for p in cfg["pins"]}
    state["pins_before"] = pins_before
    bad = {p: got for p, got in pins_before.items() if got != cfg["pins"][p]}
    if bad:
        raise RunError("pin mismatch", {"mismatched": bad})
    pin_of = {norm(p): d for p, d in cfg["pins"].items()}
    hg_file = os.path.join(CODE_DIR, "handgap.py")
    if norm(HG.__file__) != norm(hg_file) or sha256_file(hg_file) != pin_of.get(norm(hg_file)):
        raise RunError(f"handgap is not the pinned file next to run_handgap.py ({HG.__file__})")

    # 2. frozen plan (pinned) must describe this exact protocol
    state["stage"] = "plan"
    plan = json.loads(Path(cfg["plan_file"]).read_text(encoding="utf-8-sig"))
    plan_expect = {"arms": list(ARMS), "salt": MASK_SALT, "tolerance": 1e-5, "indices": REPRO_INDICES,
                   "native_rows": 699, "mediapipe_rows": 113, "model_sha256": pin_of[norm(cfg["checkpoint"])]}
    plan_got = {"arms": plan.get("arms"), "salt": (plan.get("mask") or {}).get("salt"),
                "tolerance": (plan.get("reproduction") or {}).get("tolerance"),
                "indices": (plan.get("reproduction") or {}).get("indices"),
                "native_rows": ((plan.get("groups") or {}).get("native") or {}).get("rows"),
                "mediapipe_rows": ((plan.get("groups") or {}).get("mediapipe") or {}).get("rows"),
                "model_sha256": plan.get("model_sha256")}
    if plan_got != plan_expect:
        raise RunError("plan file differs from the implemented protocol", {"expected": plan_expect, "got": plan_got})

    # 3. prior helpers, unmodified
    state["stage"] = "helper_import"
    pre = [m for m in ("run_audit", "timegrid") if m in sys.modules]
    if pre:
        raise RunError(f"prior modules already imported from elsewhere: {pre}")
    R = _import_pinned("run_audit", cfg["helper_dir"], pin_of)
    TG = sys.modules.get("timegrid")
    if TG is None or R.TG is not TG or norm(TG.__file__) != norm(os.path.join(cfg["helper_dir"], "timegrid.py")) \
            or sha256_file(TG.__file__) != pin_of.get(norm(TG.__file__)):
        raise RunError("timegrid is not the pinned helper_dir file used by run_audit")

    def frozen():
        return {"R.N_CLASSES": R.N_CLASSES, "R.NATIVE_N": R.NATIVE_N, "R.MP_N": R.MP_N,
                "R.FIXED_THREADS": R.FIXED_THREADS, "R.FIXED_TOLERANCE": R.FIXED_TOLERANCE,
                "TG.N_JOINTS": TG.N_JOINTS, "TG.NOMINAL_STEP_US": TG.NOMINAL_STEP_US, "TG.CLOCK_TOL_US": TG.CLOCK_TOL_US,
                "HG.N_JOINTS": HG.N_JOINTS, "HG.SALT": HG.SALT, "HG.PERIOD": HG.PERIOD, "HG.FIRST": HG.FIRST,
                "HG.MAX_SPAN_US": HG.MAX_SPAN_US, "HG.SIDES": list(HG.SIDES),
                "HG.HANDS": {k: [v.start, v.stop] for k, v in HG.HANDS.items()}}
    frozen_before = frozen()
    expect = {"R.N_CLASSES": 93, "R.NATIVE_N": 699, "R.MP_N": 113, "R.FIXED_THREADS": 2, "R.FIXED_TOLERANCE": 1e-5,
              "TG.N_JOINTS": 67, "TG.NOMINAL_STEP_US": 1e6 / 30.0, "TG.CLOCK_TOL_US": 2,
              "HG.N_JOINTS": 67, "HG.SALT": MASK_SALT, "HG.PERIOD": MASK_PERIOD, "HG.FIRST": MASK_FIRST,
              "HG.MAX_SPAN_US": MAX_SPAN_US, "HG.SIDES": list(SIDES),
              "HG.HANDS": {k: [v.start, v.stop] for k, v in HAND_SLICES.items()}}
    off = {k: frozen_before[k] for k, v in expect.items() if frozen_before[k] != v}
    if off:
        raise RunError("helper/candidate constants differ from the frozen method", {"constants": off})

    state["stage"] = "config"
    R.validate_config(cfg)
    tol = float(cfg["tolerance"])
    roots = {"native": cfg["native_roots"], "mediapipe": cfg["mp_roots"]}
    lab = cfg["lab_root"]
    sys.path[1:1] = [lab, os.path.join(lab, "src"), cfg["pilot_src"], cfg["loader_dir"]]

    # 4. rows + listed source hashes before any array/model load; nothing filtered
    state["stage"] = "metadata"
    rows = {"native": R.load_native_rows(cfg["native_metadata"]), "mediapipe": R.load_mp_rows(cfg["mp_manifest"])}
    if {g: len(rows[g]) for g in GROUPS} != GROUP_N:
        raise RunError("row counts differ from 699 + 113")
    state["stage"] = "source_hashes_before"
    sources_before = {g: R.hash_sources(rows[g], roots[g]) for g in GROUPS}
    for g in GROUPS:
        R.check_listed(rows[g], sources_before[g], g)

    state["stage"] = "saved_material"
    classes, a_native, _unused_decimated = R.load_saved_native(cfg["saved_native_logits"], rows["native"])
    a_mp, mp_arms = R.load_saved_mp(cfg["saved_mp_logits"], rows["mediapipe"], classes)
    saved_a = {"native": a_native, "mediapipe": a_mp}

    # 5. local APIs, pinned
    state["stage"] = "imports"
    _camera_check()
    import torch
    import tools_prep_union93_candidate as u
    from pilot.data import build_channels, collate
    from pilot.web_input import to_native_like
    import framerate_impl as F
    module_files = {n: os.path.abspath(sys.modules[n].__file__) for n in API_MODULES}
    modules_before = {k: sha256_file(p) for k, p in module_files.items()}
    unpinned = {k: p for k, p in module_files.items() if pin_of.get(norm(p)) != modules_before[k]}
    if unpinned:
        raise RunError("API modules not pinned or differ from pins", {"modules": unpinned})
    _camera_check()
    api = {"F": F, "build_channels": build_channels, "to_native_like": to_native_like,
           "native_roots": roots["native"], "mp_roots": roots["mediapipe"]}

    def prepare_row(g, i):
        """-> (x by arm, record, coordinate events). Evaluator alone sees the hidden originals."""
        r = rows[g][i]
        if g == "native":
            nat = F.load_native(r["meta"], {R.NATIVE_ROLE}, set(R.NATIVE_SIGNERS), roots["native"])
            if nat["sha256"] != r["files"]["native"][1]:
                raise RunError(f"{r['id']}: loader sha differs from metadata")
            xy, pr, ts, x_saved, diff = nat["xy"], nat["present"], nat["ts"], None, None
        else:
            xy, pr, ts, x_saved, diff = R.load_mp(api, r, tol)   # exact prepared frames/times, diff <= tol
        t = len(ts)
        src = raw_hash(xy, pr, ts)
        clock = R.clock_report(ts)
        rep_o = HG.restore(xy, pr, ts)
        m = HG.mask_one_hand(xy, pr, ts, r["id"])
        masked_raw = raw_hash(m["xy"], m["present"], m["ts"])
        rep_m = HG.restore(m["xy"], m["present"], m["ts"])            # masked arrays only
        checks = {"clock_strictly_increasing": clock["strictly_increasing"], "clock_nominal_30hz": clock["nominal_30hz"],
                  "source_unmodified_by_candidate": raw_hash(xy, pr, ts) == src,
                  "masked_unmodified_by_restore": raw_hash(m["xy"], m["present"], m["ts"]) == masked_raw}
        vm = verify_mask(xy, pr, ts, r["id"], m)
        vo = verify_restore(xy, pr, ts, rep_o)
        vr = verify_restore(m["xy"], m["present"], m["ts"], rep_m)
        inj, est_o, est_m = m["injected"], rep_o["estimated"], rep_m["estimated"]
        checks["all_injected_repaired"] = bool(est_m[inj].all())
        checks["original_fills_disjoint_from_injected"] = not bool((est_o & inj).any())
        checks["masked_repair_fills_are_original_fills_plus_injected"] = bool(np.array_equal(est_m, est_o | inj))
        raws = {"original": (xy, pr, ts), "original_repair": (rep_o["xy"], rep_o["present"], rep_o["ts"]),
                "masked": (m["xy"], m["present"], m["ts"]), "masked_repair": (rep_m["xy"], rep_m["present"], rep_m["ts"])}
        rh = {a: raw_hash(*v) for a, v in raws.items()}
        built = {a: build_channels(*v) for a, v in raws.items()}      # unchanged normalisation, re-run per arm
        checks["arms_unmodified_by_build"] = {a: raw_hash(*v) for a, v in raws.items()} == rh
        checks["source_raw_is_original_arm"] = rh["original"] == src
        x = {a: b["x"] for a, b in built.items()}
        if x_saved is not None:
            x["original"] = x_saved                                   # prepared x behind the saved MP scores
        checks["features_float32_finite"] = all(features_ok(v, t) for v in x.values())
        if rep_o["identity"]:
            checks["clean_identity_same_built_input"] = _same(built["original_repair"]["x"], built["original"]["x"])
        if not vm["events"]:
            checks["no_event_mask_identity"] = rh["masked"] == src
        body = R.num_or_null(built["original"]["body_scale"])
        checks["body_scale_positive_finite_if_events"] = not vm["events"] or (body is not None and body > 0)
        if not all(checks.values()):
            raise RunError(f"{r['id']}: row invariants failed", {"checks": checks})
        events = event_errors(xy, rep_m["xy"], vm["events"], body) if vm["events"] else []
        hx = {a: sha_array(v) for a, v in x.items()}
        rec = {"index": i, "id": r["id"], "signer_id": r["signer_id"], "true": r["label"],
               "true_index": classes.index(r["label"]), "recording_id": R.recording_id(r["id"]), "frames": t,
               "clock": clock, "raw_sha256": rh, "input_sha256": hx,
               "scales": {a: R.scales_of(b) for a, b in built.items()}, "original_body_scale": body,
               "mp_prepared_vs_rebuilt_max_abs_diff": diff, "mask": vm,
               "repairs": {"original_repair": {"natural_events": vo["fills_n"], "natural_slots": vo["slots"],
                                               "slots_by_side": vo["slots_by_side"], "fills": vo["fills"]},
                           "masked_repair": {"artificial_events": vm["events_n"],
                                             "artificial_slots": int((est_m & inj).sum()),
                                             "natural_events": vr["fills_n"] - vm["events_n"],
                                             "natural_slots": int((est_m & ~inj).sum()),
                                             "slots_by_side": vr["slots_by_side"]}},
               "identity": {"original_repair": rep_o["identity"], "masked": not vm["events"],
                            "masked_repair_raw_equals_original_repair_raw": rh["masked_repair"] == rh["original_repair"]},
               "checks": checks, "cache": cache_plan(hx)}
        return x, jsonable(rec), events

    # 6. preflight ALL rows and arms before model loading; any failure stops everything, nothing filtered
    state["stage"] = "preflight"
    recs, coord, failures = {g: [] for g in GROUPS}, {g: [] for g in GROUPS}, []
    for g in GROUPS:
        state["group"] = g
        for i, r in enumerate(rows[g]):
            state["id"] = r["id"]
            try:
                _, rec, ev = prepare_row(g, i)
            except Exception as exc:
                failures.append({"group": g, "index": i, "id": r["id"], "error": f"{type(exc).__name__}: {exc}",
                                 "details": getattr(exc, "details", None)})
                recs[g].append(None)
                coord[g].append(None)
                continue
            recs[g].append(rec)
            coord[g].append(ev)
            state["prepared_rows"][g] += 1
    state.update(group=None, id=None, stage="preflight_gate")
    if failures:
        raise RunError("preflight failed before model load; no rows filtered",
                       {"failures_n": {g: sum(f["group"] == g for f in failures) for g in GROUPS},
                        "failures": failures})
    plan_counts = {}
    for g in GROUPS:
        for rec in recs[g]:
            for a in ARMS[1:]:
                plan_counts[rec["cache"][a]] = plan_counts.get(rec["cache"][a], 0) + 1
    expected_new = plan_counts.get("new", 0)
    write_json_exclusive(out_dir / "PREFLIGHT.json", jsonable({
        "status": "PREFLIGHT_COMPLETE", "rows": {g: recs[g] for g in GROUPS}, "cache_plan_counts": plan_counts,
        "expected_new_forwards": expected_new, "at_utc": datetime.now(timezone.utc).isoformat()}))
    _camera_check()

    # 7. model (frozen, eval, CPU, 2 threads); official forward only
    state["stage"] = "model"
    torch.set_num_threads(R.FIXED_THREADS)
    payload = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    model, model_classes = u.rebuild_from_payload(payload)
    model.eval()
    if [str(c) for c in model_classes] != classes:
        raise RunError("model class order differs from saved class order")
    model_mods = sorted({type(m).__module__ for m in model.modules() if not type(m).__module__.startswith("torch.")})
    if not any(n.split(".")[-1] == MODEL_API_MODULE for n in model_mods):
        raise RunError(f"model classes not from {MODEL_API_MODULE}", {"modules": model_mods})
    for n in model_mods:
        module_files[n] = os.path.abspath(sys.modules[n].__file__)
        modules_before[n] = sha256_file(module_files[n])
        if pin_of.get(norm(module_files[n])) != modules_before[n]:
            raise RunError(f"model API module {n} not pinned or differs", {"file": module_files[n]})
    _camera_check()
    state_before = state_digest(model)
    if state_before["training_modules"]:
        raise RunError("model not in eval mode", {"training": state_before["training_modules"]})
    n_cls = R.N_CLASSES

    def forward(x, kind):
        if not features_ok(x, len(x)):
            raise RunError(f"bad forward input {getattr(x, 'shape', None)}")
        bx, pad, _ = collate([{"x": x, "frames": int(len(x)), "label": 0}])   # label unused by model
        with torch.inference_mode():
            lg = model(torch.from_numpy(bx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
        if lg.shape != (n_cls,) or not np.isfinite(lg).all():
            raise RunError(f"model output shape {lg.shape} or non-finite")
        state["forwards"][kind] += 1
        return lg

    def top1(v):
        return classes[int(np.argmax(v))]

    def reprepare(g, i):
        x, rec, ev = prepare_row(g, i)
        if rec != recs[g][i] or ev != coord[g][i]:
            raise RunError(f"{rows[g][i]['id']}: rebuilt row differs from preflight")
        return x

    # 8. reproduce stored original all-93 scores BEFORE any candidate scoring or reuse
    state["stage"] = "reproduction"
    repro = []
    for g in GROUPS:
        for i in cfg["repro_indices"][g]:
            state["group"], state["id"] = g, rows[g][i]["id"]
            lg = forward(reprepare(g, i)["original"], "reproduction")
            ref = saved_a[g][i]
            d = float(np.max(np.abs(lg - ref)))
            repro.append({"group": g, "index": i, "id": rows[g][i]["id"], "arm": "original", "max_abs_diff": d,
                          "top1_saved": top1(ref), "top1_now": top1(lg), "ok": bool(d <= tol and top1(ref) == top1(lg))})
    if len(repro) != 6 or not all(x["ok"] for x in repro):
        raise RunError("reproduction failed; no candidate scored", {"reproduction": repro})

    # 9. candidate scores: every distinct new built input of a row is scored exactly once
    state["stage"] = "inference"
    logits, results = {}, {}
    for g in GROUPS:
        n = len(rows[g])
        lg = np.full((n, len(ARMS), n_cls), np.nan, dtype=np.float64)
        lg[:, 0] = saved_a[g]
        out_rows = []
        for i, r in enumerate(rows[g]):
            state["group"], state["id"] = g, r["id"]
            x = reprepare(g, i)
            rec = recs[g][i]
            for k, a in enumerate(ARMS[1:], start=1):
                src = rec["cache"][a]
                if src == "new":
                    lg[i, k] = forward(x[a], "new")
                else:
                    lg[i, k] = lg[i, ARMS.index(src.split(":", 1)[1])]
                    state["reused"][src] = state["reused"].get(src, 0) + 1
            inp = {k: v for k, v in rec.items() if k not in ("index", "id", "signer_id", "true", "true_index",
                                                             "recording_id", "cache")}
            out_rows.append({"id": r["id"], "index": i, "recording_id": rec["recording_id"], "signer_id": r["signer_id"],
                             "true": r["label"], "true_index": rec["true_index"],
                             "arms": {a: R.score_row(lg[i, k], classes, r["label"]) for k, a in enumerate(ARMS)},
                             "logits_source": rec["cache"], "input": inp, "coordinate_events": coord[g][i]})
        if not np.isfinite(lg).all() or not np.array_equal(lg[:, 0], saved_a[g]):
            raise RunError(f"{g}: non-finite logits or original logits not the saved scores")
        logits[g], results[g] = lg, out_rows
    state.update(group=None, id=None)
    reused_expected = {k: v for k, v in plan_counts.items() if k != "new"}
    counts_ok = state["forwards"]["new"] == expected_new and state["forwards"]["reproduction"] == 6 \
        and state["reused"] == reused_expected
    if not counts_ok:
        raise RunError("forward/reuse count mismatch", {"forwards": state["forwards"], "expected_new": expected_new,
                                                        "reused": state["reused"], "reused_expected": reused_expected})

    # 10. model state unchanged, still eval
    state["stage"] = "model_state_after"
    state_after = state_digest(model)
    if state_after != state_before:
        raise RunError("model state changed", {
            "changed": [k for k in state_before["tensors"] if state_before["tensors"][k] != state_after["tensors"].get(k)],
            "training": state_after["training_modules"]})
    _camera_check()

    # 11. scores
    state["stage"] = "write_scores"
    scores_sha = {}
    for g in GROUPS:
        p = out_dir / f"SCORES_{g}.npz"
        with open(p, "xb") as fh:
            np.savez_compressed(fh, logits=logits[g], arms=np.array(ARMS), ids=np.array([r["id"] for r in rows[g]]),
                                classes=np.array(classes),
                                true_indices=np.array([classes.index(r["label"]) for r in rows[g]], dtype=np.int64))
        scores_sha[g] = sha256_file(p)

    # 12. hashes after
    state["stage"] = "hashes_after"
    pins_after = {p: sha256_file(p) for p in cfg["pins"]}
    sources_after = {g: R.hash_sources(rows[g], roots[g]) for g in GROUPS}
    modules_after = {k: sha256_file(p) for k, p in module_files.items()}
    code_after = code_hashes(cfg)
    config_sha, config_after = hashlib.sha256(cfg_bytes).hexdigest(), sha256_file(cfg_path)
    changed = {"pins": [p for p in pins_before if pins_before[p] != pins_after[p]],
               "sources": {g: [k for k in sources_before[g] if sources_before[g][k] != sources_after[g][k]]
                           for g in GROUPS},
               "modules": [k for k in modules_before if modules_before[k] != modules_after[k]],
               "code": code_after != code, "config": config_after != config_sha, "constants": frozen() != frozen_before}
    write_json_exclusive(out_dir / "SOURCE_HASHES.json", {
        "pins": {"before": pins_before, "after": pins_after},
        "sources": {"before": sources_before, "after": sources_after},
        "imported_modules": {"files": module_files, "before": modules_before, "after": modules_after},
        "code": {"before": code, "after": code_after}, "model_state": {"before": state_before, "after": state_after},
        "config": {"path": cfg_path, "before": config_sha, "after": config_after}, "changed": changed}, strict=False)
    if changed["pins"] or any(changed["sources"].values()) or changed["modules"] or changed["code"] \
            or changed["config"] or changed["constants"]:
        raise RunError("hashes changed during run", {"changed": changed})

    # 13. result
    state["stage"] = "write_result"
    contracts = {"preflight_all_rows_no_failures": not failures and sum(state["prepared_rows"].values()) == 812,
                 "row_contracts_all": all(all(rec["checks"].values()) for g in GROUPS for rec in recs[g]),
                 "all_injected_points_repaired": all(rec["checks"]["all_injected_repaired"]
                                                     for g in GROUPS for rec in recs[g]),
                 "reproduction_within_tolerance": all(x["ok"] for x in repro),
                 "forward_and_reuse_counts_match": counts_ok,
                 "model_state_and_eval_unchanged": state_after == state_before and not state_after["training_modules"],
                 "pins_unchanged": not changed["pins"], "sources_unchanged": not any(changed["sources"].values()),
                 "imported_modules_unchanged": not changed["modules"], "code_unchanged": not changed["code"],
                 "config_unchanged": not changed["config"], "constants_unchanged": not changed["constants"],
                 "no_camera_modules": not any(m in sys.modules for m in CAMERA_MODULES)}
    groups_out, gates = {}, {}
    for g in GROUPS:
        rs = results[g]
        comps = {}
        for s, t in PAIRS:
            c = R.compare_arms(rs, s, t)
            c["gain_recordings"] = len({R.recording_id(x["id"]) for x in c["gains"]})
            c["loss_recordings"] = len({R.recording_id(x["id"]) for x in c["losses"]})
            c["gain_ids"], c["loss_ids"] = [x["id"] for x in c["gains"]], [x["id"] for x in c["losses"]]
            comps[f"{s}->{t}"] = c
        per_signer = breakdown(rs, "signer_id")
        cs = coord_summary(coord[g])
        mm = comps["masked->masked_repair"]
        subsets = {
            "rows_with_injected_mask": [r for r in rs if r["input"]["mask"]["events_n"] > 0],
            "rows_with_natural_fill": [r for r in rs if r["input"]["repairs"]["original_repair"]["natural_events"] > 0],
            "rows_original_repair_top1_changed": [r for r in rs if r["arms"]["original"]["pred"]
                                                  != r["arms"]["original_repair"]["pred"]],
            "rows_masked_repair_top1_changed": [r for r in rs if r["arms"]["masked"]["pred"]
                                                != r["arms"]["masked_repair"]["pred"]]}
        provenance = {
            "original_repair": {k: sum(r["input"]["repairs"]["original_repair"][k] for r in rs)
                                for k in ("natural_events", "natural_slots")},
            "masked_repair": {k: sum(r["input"]["repairs"]["masked_repair"][k] for r in rs)
                              for k in ("artificial_events", "artificial_slots", "natural_events", "natural_slots")},
            "mask_scheduled": sum(r["input"]["mask"]["scheduled_n"] for r in rs),
            "mask_injected_events": sum(r["input"]["mask"]["events_n"] for r in rs),
            "mask_injected_slots_by_side": {sd: sum(r["input"]["mask"]["injected_slots"][sd] for r in rs) for sd in SIDES}}
        scale_changed = {a: sum(r["input"]["scales"][a] != r["input"]["scales"]["original"] for r in rs) for a in ARMS[1:]}
        gates[g] = {
            "coordinate_linear_mean_lt_previous": coord_gate(cs),
            "coordinate_available": cs["available"],
            "recognition_masked_repair_vs_masked": (mm["net"] > 0) if g == "native" else (mm["net"] >= 0),
            "recognition_rule": "strictly increases" if g == "native" else "nondecrease",
            "every_signer_nondecrease": all(v["masked_repair"] >= v["masked"] for v in per_signer.values()),
            "clean_original_repair_no_newly_wrong": comps["original->original_repair"]["losses_n"] == 0,
            "strict_no_newly_wrong_original_and_masked_pairs":
                comps["original->original_repair"]["losses_n"] == 0 and mm["losses_n"] == 0}
        groups_out[g] = {
            "description": ("native provided-coordinate development rows (annotation_dev, validation; reused)"
                            if g == "native" else "cached MediaPipe training-exposed diagnostic rows"),
            "n_rows": len(rs), "source_recordings": R.recording_counts(rs),
            "totals": {a: sum(int(r["arms"][a]["correct"]) for r in rs) for a in ARMS},
            "per_signer": per_signer, "per_class": breakdown(rs, "true"), "comparisons": comps,
            "affected": {k: {**R.recording_counts(v), "ids": [r["id"] for r in v]} for k, v in subsets.items()},
            "coordinate": cs, "provenance": provenance,
            "normalization_rows_scale_differs_from_original": scale_changed,
            "original_clock_rows_nominal_30hz": sum(r["input"]["clock"]["nominal_30hz"] for r in rs),
            "logits_sources": {a: {s: sum(r["logits_source"][a] == s for r in rs)
                                   for s in sorted({r["logits_source"][a] for r in rs})} for a in ARMS},
            "rows": rs}
    diag_keys = ("coordinate_linear_mean_lt_previous", "recognition_masked_repair_vs_masked",
                 "every_signer_nondecrease", "clean_original_repair_no_newly_wrong")
    contracts_ok = all(contracts.values())
    diagnostic_pass = contracts_ok and all(gates[g][k] for g in GROUPS for k in diag_keys)
    strict_all = all(gates[g]["strict_no_newly_wrong_original_and_masked_pairs"] for g in GROUPS)
    decision = {"contracts": contracts, "contracts_all_valid": contracts_ok, "gates": gates,
                "diagnostic_gate_keys": list(diag_keys), "diagnostic_gate_pass": bool(diagnostic_pass),
                "strict_no_newwrong_all_pairs": bool(strict_all),
                "decision": "DIAGNOSTIC_GATES_MET_PENDING_CODEX_VERIFICATION" if diagnostic_pass
                else "DIAGNOSTIC_GATES_NOT_MET",
                "operational_admission": False, "webcam_claim": False, "production_authorised": False,
                "model_promoted": False,
                "note": "Frozen gates, not relaxed. A failed scientific gate is a valid reported outcome, not a runner "
                        "failure. Net diagnostic may pass with individual masked regressions; strict is separate."}
    result = {
        "status": "COMPLETE_PENDING_CODEX_VERIFICATION",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "nature": "offline isolated whole-hand dropout reconstruction diagnostic on a frozen model; no training, no search",
        "arm_order": list(ARMS), "candidate": "handgap.restore (time-weighted linear, same channel)",
        "coordinate_comparator": "previous original frame copy (not scored)",
        "rules": {"mask": {"salt": MASK_SALT, "offset": "digest[0] % 6", "first_side": "digest[1] % 2 (0 left)",
                           "positions": "range(2+offset, T-2, 6)", "max_span_us": MAX_SPAN_US},
                  "restore": {"max_span_us": MAX_SPAN_US, "whole_hand_only": True, "endpoints": "received input only"},
                  "cache": "fixed order " + ",".join(ARMS) + "; reuse only for a byte-identical built input of the same row"},
        "model": {"checkpoint": cfg["checkpoint"], "sha256": pin_of[norm(cfg["checkpoint"])], "classes": len(classes),
                  "api_modules": model_mods, "state_combined_sha256": state_before["combined"], "state_unchanged": True},
        "saved_mp_arms": mp_arms, "reproduction": {"atol": tol, "indices": cfg["repro_indices"], "rows": repro},
        "decision": decision, "groups": groups_out, "interpretation_limits": LIMITS,
        "execution_counts": {"rows_prepared": state["prepared_rows"], "forwards": state["forwards"],
                             "forwards_total": sum(state["forwards"].values()), "expected_new_forwards": expected_new,
                             "reused": state["reused"], "training": 0, "camera": 0, "user_data": 0, "downloads": 0},
        "hashes": {"code": code, "config": config_sha, "plan_file": pin_of[norm(cfg["plan_file"])],
                   "scores": scores_sha, "imported_modules": modules_before},
        "seconds": round(time.time() - started, 2),
    }
    write_json_exclusive(out_dir / "RESULT.json", jsonable(result))
    print(json.dumps({"decision": decision["decision"], "strict": strict_all, "forwards": state["forwards"],
                      **{g: groups_out[g]["totals"] for g in GROUPS}}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                 # FAILURE.json already written when output_dir existed
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
