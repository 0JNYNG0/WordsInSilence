"""Observation-preserving local insertion evaluation on the frozen TCN (offline diagnostic).

Five cases x four report arms per row; groups are never pooled:
  cases  clean            unchanged base clip
         regular_half     raw arrays [::2] (last dense frame not forced)
         isolated_drop, burst_drop, clock_noise   corruptions.corrupt(base, row id, case), unmodified
  arms   original         stored original scores (same in every case; context, not extra samples)
         unrestored       stored scores of the clean / sparse / corrupted input
         global_linear    stored scores of the prior timegrid.restore(..., 'linear') input (comparison only)
         local_insert     build_channels(local_insert.restore(unrestored arrays)) -> official forward, or the
                          stored score of a byte-identical built input of the same row/case
local_insert.restore only receives the unrestored xy/present/ts of the cell. No training, no threshold or
parameter search, no camera, no network, no deployment. Prior code is imported unmodified from pins:
  run_audit (R), timegrid (TG)            from cfg.helper_dir
  irregular_audit (I), corruptions (C)    from cfg.irregular_dir

Run: python run_local.py --config <absolute config.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

import local_insert as LI

GROUPS = ("native", "mediapipe")
CASES = ("clean", "regular_half", "isolated_drop", "burst_drop", "clock_noise")
IRREGULAR = CASES[2:]
ARMS = ("original", "unrestored", "global_linear", "local_insert")
CACHE_ORDER = ("unrestored", "global_linear", "original")
PRIOR_ARMS = ("original", "corrupted", "previous", "linear")
REGULAR_ARMS = ("original", "sparse", "endpoint_reference", "previous", "linear")
STORED_NAME = {"regular_half": {"original": "original", "unrestored": "sparse", "global_linear": "linear"},
               "irregular": {"original": "original", "unrestored": "corrupted", "global_linear": "linear"}}
COMPARISONS = (("original", "local_insert"), ("unrestored", "local_insert"), ("global_linear", "local_insert"),
               ("unrestored", "global_linear"))
HELPER_FILES = ("run_audit.py", "timegrid.py")
IRREGULAR_FILES = ("irregular_audit.py", "corruptions.py")
API_MODULES = ("tools_prep_union93_candidate", "pilot.data", "pilot.web_input", "framerate_impl")
MODEL_API_MODULE = "model_masked_v1"
CAMERA_MODULES = ("cv2", "mediapipe")
REPRO_INDICES = {"native": [0, 349, 698], "mediapipe": [0, 56, 112]}
N_FEATURES = 280
CELLS_EXPECTED = (699 + 113) * len(CASES)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = ("run_local.py", "local_insert.py")
LIMITS = [
    "Offline synthetic diagnostic with one-frame lookahead on the same frozen model; not a causal live tracker. "
    "No training, no threshold/parameter search, no web integration or deployment even if every condition holds.",
    "native = reused development rows, mediapipe = training-exposed cached rows; groups are never pooled and "
    "neither shows webcam or new-person performance. No statistical significance claim.",
    "The original arm is the same stored score in every case: context, not additional samples.",
    "Inserted midpoints are interpolations, never observations. Gaps below 1.5P (incl. clock_noise jitter) and "
    "above 2P + 2 us (e.g. burst_drop ~100 ms spans) are left unchanged, so such cells can equal unrestored.",
    "build_channels normalisation is unchanged code but is re-estimated from the candidate sequence including "
    "inserted points; raw observation preservation does not imply unchanged normalised features.",
    "Admission reuses the frozen prior gates exactly. HOLD is never promoted. A failing strict gate (e.g. a "
    "single-row isolated_drop regression as seen before) is a reported outcome, not by itself evidence of an "
    "implementation bug; condition-specific net recovery is distinct from operational admission.",
    "global_linear is a comparison arm only and is never selected by score.",
]


class RunError(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


# --- pure helpers (no prior helper / torch import) ---------------------------------------
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
    return json.loads(json.dumps(v, default=_plain))


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=bool(np.issubdtype(a.dtype, np.floating))))


def _mask(m):
    m = np.asarray(m)
    return m.astype(bool) if m.dtype == np.bool_ else m == 1


def precheck(cfg):
    if not isinstance(cfg, dict):
        raise RunError("config must be a JSON object")
    for k in ("helper_dir", "irregular_dir", "output_dir", "prior_result", "regular_result", "plan_file", "lab_root"):
        if not isinstance(cfg.get(k), str) or not os.path.isabs(cfg[k]):
            raise RunError(f"config {k} must be an absolute string")
    for k in ("prior_scores", "regular_scores"):
        v = cfg.get(k)
        if not isinstance(v, dict) or set(v) != set(GROUPS) or any(
                not isinstance(p, str) or not os.path.isabs(p) for p in v.values()):
            raise RunError(f"{k} must be {{native: abs path, mediapipe: abs path}}")
    pins = cfg.get("pins")
    if not isinstance(pins, dict) or not pins or any(not os.path.isabs(p) for p in pins):
        raise RunError("pins must be a non-empty object of absolute paths")
    pinned = {norm(p) for p in pins}
    need = [os.path.join(cfg["helper_dir"], f) for f in HELPER_FILES] + \
           [os.path.join(cfg["irregular_dir"], f) for f in IRREGULAR_FILES] + \
           [cfg["prior_result"], cfg["regular_result"], cfg["plan_file"], *cfg["prior_scores"].values(),
            *cfg["regular_scores"].values(), os.path.join(CODE_DIR, "local_insert.py")]
    miss = [p for p in need if norm(p) not in pinned]
    if miss:
        raise RunError("required files not pinned", {"missing": miss})
    if cfg.get("repro_indices") != REPRO_INDICES:
        raise RunError(f"repro_indices must be exactly {REPRO_INDICES}")
    out, lab = cfg["output_dir"], cfg["lab_root"]
    roots = list(cfg.get("native_roots") or []) + list(cfg.get("mp_roots") or [])
    forbidden = roots + [cfg["helper_dir"], cfg["irregular_dir"], CODE_DIR,
                         os.path.join(lab, "src"), os.path.join(lab, "web"), os.path.join(lab, "tests"),
                         os.path.dirname(cfg["prior_result"]), os.path.dirname(cfg["regular_result"]),
                         *(os.path.dirname(v) for v in cfg["prior_scores"].values()),
                         *(os.path.dirname(v) for v in cfg["regular_scores"].values())]
    ops = os.path.join(lab, "ops")
    if not under(out, ops) or norm(out) == norm(ops):
        raise RunError("output_dir must be a fresh diagnostic subdirectory under lab/ops")
    if any(isinstance(r, str) and os.path.isabs(r) and under(out, r) for r in forbidden):
        raise RunError("output_dir lies inside a protected source/helper/prior-output/code directory")


def choose_cache(candidate_sha, stored_sha) -> str | None:
    """Stored arm whose built input of this same row/case is byte-identical; fixed preference order."""
    for a in CACHE_ORDER:
        if stored_sha.get(a) is not None and stored_sha[a] == candidate_sha:
            return a
    return None


def verify_insertion(li, xy, pr, ts) -> dict:
    """Independent check of one local_insert.restore result against the arrays it received."""
    if set(li) != {"xy", "present", "ts", "observed", "estimated", "exact_source_index", "identity"}:
        raise RunError("candidate result keys wrong")
    xy, pr, ts = np.asarray(xy), np.asarray(pr), np.asarray(ts)
    pb, ti = _mask(pr), [int(v) for v in ts]
    t = len(ti)
    oxy, opr, ots = np.asarray(li["xy"]), np.asarray(li["present"]), np.asarray(li["ts"])
    obs, est, src = np.asarray(li["observed"]), np.asarray(li["estimated"]), np.asarray(li["exact_source_index"])
    n = len(ots)
    if (oxy.dtype, opr.dtype, ots.dtype, src.dtype, obs.dtype, est.dtype) != \
            (xy.dtype, pr.dtype, ts.dtype, np.int64, np.bool_, np.bool_):
        raise RunError("candidate output dtypes differ from input / contract")
    if oxy.shape != (n,) + xy.shape[1:] or opr.shape != (n,) + pr.shape[1:] or obs.shape != opr.shape \
            or est.shape != opr.shape or src.shape != (n,) or ots.shape != (n,):
        raise RunError("candidate output shapes inconsistent")
    pos, ins = np.flatnonzero(src >= 0), np.flatnonzero(src < 0)
    if (src < -1).any() or not np.array_equal(src[pos], np.arange(t)):
        raise RunError("source frames are not each present exactly once, in order")
    oti = [int(v) for v in ots]
    if not (_same(oxy[pos], xy) and _same(opr[pos], pr) and [oti[p] for p in pos.tolist()] == ti):
        raise RunError("source frames are not verbatim (xy/present/ts)")
    if not np.array_equal(obs[pos], pb) or obs[ins].any() or est[pos].any():
        raise RunError("observed/estimated provenance wrong")
    if any(b <= a for a, b in zip(oti, oti[1:])) or oti[0] != ti[0] or oti[-1] != ti[-1]:
        raise RunError("candidate timestamps not increasing or endpoints moved")
    at = {}
    for p in ins.tolist():
        if p == 0 or p == n - 1 or src[p - 1] < 0 or src[p + 1] != src[p - 1] + 1:
            raise RunError("inserted frame is not between two adjacent source frames", {"position": p})
        at[int(src[p - 1])] = p
    classes = {"below_window": 0, "inserted": 0, "window_no_shared_joint": 0, "long_gap": 0}
    gaps = []
    for k in range(t - 1):
        gap = ti[k + 1] - ti[k]
        gaps.append(gap)
        shared = pb[k] & pb[k + 1]
        if 30 * gap < 1_500_000:
            cls = "below_window"
        elif 30 * gap > 2 * 1_000_000 + 60:
            cls = "long_gap"
        else:
            cls = "inserted" if shared.any() else "window_no_shared_joint"
        classes[cls] += 1
        p = at.get(k)
        if (p is not None) != (cls == "inserted"):
            raise RunError("insertion set differs from the rule", {"gap_index": k, "gap_us": gap, "class": cls})
        if p is None:
            continue
        mid = round(Fraction(ti[k] + ti[k + 1], 2))           # exact rational, ties to even
        if oti[p] != mid:
            raise RunError("inserted timestamp is not the exact midpoint", {"position": p})
        if not np.array_equal(est[p], shared) or not np.all(opr[p][shared] == 1) or np.any(opr[p][~shared] != 0):
            raise RunError("inserted joints are not exactly the jointly present joints", {"position": p})
        if np.any(oxy[p][~shared] != 0):
            raise RunError("unsupported slots of an inserted frame are not zero", {"position": p})
        w = float(Fraction(mid - ti[k], gap))
        lo, hi = xy[k][shared].astype(np.float64), xy[k + 1][shared].astype(np.float64)
        got = oxy[p][shared].astype(np.float64)
        bound = 4 * float(np.finfo(xy.dtype).eps) * np.maximum(1.0, np.maximum(np.abs(lo), np.abs(hi)))
        if not np.isfinite(got).all() or np.any(np.abs(got - ((1 - w) * lo + w * hi)) > bound):
            raise RunError("inserted coordinates are not the midpoint interpolation", {"position": p})
    if li["identity"] is not (n == t):
        raise RunError("identity flag inconsistent with insertion count")
    return {"frames_in": t, "frames_out": n, "inserted_frames": int(len(ins)), "inserted_joints": int(est.sum()),
            "source_frames_covered_once": t, "identity": n == t, "inserted_positions": ins.tolist(),
            "gap_classes": classes, "gap_us": {"min": min(gaps), "max": max(gaps)} if gaps else None}


def breakdown(rows, key) -> dict:
    out = {}
    for r in rows:
        g = out.setdefault(str(r[key]), {"n": 0, **{a: 0 for a in ARMS}})
        g["n"] += 1
        for a in ARMS:
            g[a] += int(r["arms"][a]["correct"])
    return dict(sorted(out.items()))


def state_digest(model, sha_array) -> dict:
    import torch

    def h(t):
        t = t.detach().cpu().contiguous()
        return sha_array((t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy())
    tensors = {f"state:{k}": h(v) for k, v in model.state_dict().items()}
    tensors.update({f"buffer:{k}": h(v) for k, v in model.named_buffers()})
    return {"tensors": tensors, "combined": hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest(),
            "training_modules": sorted(n or "<root>" for n, m in model.named_modules() if m.training)}


def code_hashes(cfg) -> dict:
    out = {f: sha256_file(os.path.join(CODE_DIR, f)) for f in CODE_FILES}
    out.update({f"helper/{f}": sha256_file(os.path.join(cfg["helper_dir"], f)) for f in HELPER_FILES})
    out.update({f"irregular/{f}": sha256_file(os.path.join(cfg["irregular_dir"], f)) for f in IRREGULAR_FILES})
    return out


# --- main ----------------------------------------------------------------------------------
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
    state = {"stage": "start", "group": None, "id": None, "case": None, "prepared_cells": {g: 0 for g in GROUPS},
             "forwards": {"reproduction": 0, "new_candidate": 0}, "reused_cells": {a: 0 for a in CACHE_ORDER},
             "details": None, "pins_before": None}
    try:
        return _run(cfg, cfg_path, cfg_bytes, out_dir, code, state, started)
    except BaseException as exc:
        failure = {"status": "FAILED", "stage": state["stage"], "group": state["group"], "id": state["id"],
                   "case": state["case"], "error_class": type(exc).__name__, "error": str(exc),
                   "details": getattr(exc, "details", None) or state["details"],
                   "counts": {"cells_prepared": state["prepared_cells"], "forwards": state["forwards"],
                              "reused_cells": state["reused_cells"], "training": 0, "camera": 0,
                              "user_data": 0, "downloads": 0},
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
    # 1. pins first; the candidate module must be the pinned file next to this runner
    state["stage"] = "pins"
    pins_before = {p: sha256_file(p) for p in cfg["pins"]}
    state["pins_before"] = pins_before
    bad = {p: got for p, got in pins_before.items() if got != cfg["pins"][p]}
    if bad:
        raise RunError("pin mismatch", {"mismatched": bad})
    pin_of = {norm(p): d for p, d in cfg["pins"].items()}
    li_file = os.path.join(CODE_DIR, "local_insert.py")
    if norm(LI.__file__) != norm(li_file) or sha256_file(li_file) != pin_of.get(norm(li_file)):
        raise RunError(f"local_insert is not the pinned file next to run_local.py ({LI.__file__})")

    # 2. prior helpers, unmodified (corruptions must be importable before irregular_audit)
    state["stage"] = "helper_import"
    pre = [m for m in ("run_audit", "timegrid", "irregular_audit", "corruptions") if m in sys.modules]
    if pre:
        raise RunError(f"prior modules already imported from elsewhere: {pre}")
    R = _import_pinned("run_audit", cfg["helper_dir"], pin_of)     # imports timegrid from helper_dir
    TG = sys.modules.get("timegrid")
    if TG is None or R.TG is not TG or norm(TG.__file__) != norm(os.path.join(cfg["helper_dir"], "timegrid.py")) \
            or sha256_file(TG.__file__) != pin_of.get(norm(TG.__file__)):
        raise RunError("timegrid is not the pinned helper_dir file used by run_audit")
    C = _import_pinned("corruptions", cfg["irregular_dir"], pin_of)
    I = _import_pinned("irregular_audit", cfg["irregular_dir"], pin_of)
    if I.C is not C:
        raise RunError("irregular_audit uses a different corruptions module")

    def frozen():
        return {"R.N_CLASSES": R.N_CLASSES, "R.NATIVE_N": R.NATIVE_N, "R.MP_N": R.MP_N, "R.ARMS": list(R.ARMS),
                "R.FIXED_THREADS": R.FIXED_THREADS, "R.FIXED_TOLERANCE": R.FIXED_TOLERANCE,
                "TG.NOMINAL_STEP_US": TG.NOMINAL_STEP_US, "TG.EXACT_TOL_US": TG.EXACT_TOL_US,
                "TG.CLOCK_TOL_US": TG.CLOCK_TOL_US, "TG.N_JOINTS": TG.N_JOINTS, "C.N_JOINTS": C.N_JOINTS,
                "C.CASES": list(C.CASES), "C.DOMAIN": C.DOMAIN, "C.MAX_SOURCE_GAP": dict(C.MAX_SOURCE_GAP),
                "C.MAX_OFFSET_US": C.MAX_OFFSET_US, "I.ARMS": list(I.ARMS), "I.CASES": list(I.CASES),
                "I.GROUPS": list(I.GROUPS), "I.PRIMARY_CASE": I.PRIMARY_CASE,
                "I.NATIVE_MIN_NET_GAIN": I.NATIVE_MIN_NET_GAIN, "I.REPRO_INDICES": I.REPRO_INDICES,
                "LI.N_JOINTS": LI.N_JOINTS, "LI.MIN_GAP_30": LI.MIN_GAP_30, "LI.MAX_GAP_30": LI.MAX_GAP_30}
    frozen_before = frozen()
    expect = {"R.N_CLASSES": 93, "R.NATIVE_N": 699, "R.MP_N": 113, "R.ARMS": list(REGULAR_ARMS),
              "R.FIXED_THREADS": 2, "R.FIXED_TOLERANCE": 1e-5, "TG.NOMINAL_STEP_US": 1e6 / 30.0,
              "TG.EXACT_TOL_US": 1, "TG.CLOCK_TOL_US": 2, "TG.N_JOINTS": 67, "C.N_JOINTS": 67,
              "C.CASES": list(IRREGULAR), "I.ARMS": list(PRIOR_ARMS), "I.CASES": list(IRREGULAR),
              "I.GROUPS": list(GROUPS), "I.PRIMARY_CASE": "isolated_drop", "I.NATIVE_MIN_NET_GAIN": 7,
              "I.REPRO_INDICES": REPRO_INDICES, "LI.N_JOINTS": 67, "LI.MIN_GAP_30": 1_500_000,
              "LI.MAX_GAP_30": 2_000_060}
    off = {k: frozen_before[k] for k, v in expect.items() if frozen_before[k] != v}
    if off:
        raise RunError("prior/candidate constants differ from the frozen method", {"constants": off})

    state["stage"] = "config"
    R.validate_config(cfg)
    tol = float(cfg["tolerance"])
    roots = {"native": cfg["native_roots"], "mediapipe": cfg["mp_roots"]}
    lab = cfg["lab_root"]
    sys.path[1:1] = [lab, os.path.join(lab, "src"), cfg["pilot_src"], cfg["loader_dir"]]

    # 3. rows + listed source hashes before any array/model load
    state["stage"] = "metadata"
    rows = {"native": R.load_native_rows(cfg["native_metadata"]), "mediapipe": R.load_mp_rows(cfg["mp_manifest"])}
    if sum(len(rows[g]) for g in GROUPS) * len(CASES) != CELLS_EXPECTED:
        raise RunError("row counts differ from 699 + 113")
    state["stage"] = "source_hashes_before"
    sources_before = {g: R.hash_sources(rows[g], roots[g]) for g in GROUPS}
    for g in GROUPS:
        R.check_listed(rows[g], sources_before[g], g)

    state["stage"] = "saved_material"
    classes, a_native, b_native = R.load_saved_native(cfg["saved_native_logits"], rows["native"])
    a_mp, mp_arms = R.load_saved_mp(cfg["saved_mp_logits"], rows["mediapipe"], classes)
    saved_a = {"native": a_native, "mediapipe": a_mp}

    def load_scores(p, g, arms, shape_mid, what):
        with np.load(p, allow_pickle=False) as z:
            if any(k not in z.files for k in ("logits", "arms", "ids", "classes", "true_indices")):
                raise RunError(f"{what} {g} lacks keys; has {z.files}")
            lg = np.asarray(z["logits"], np.float64)
            za = [str(v) for v in z["arms"].tolist()]
            zc = [str(v) for v in z["cases"].tolist()] if "cases" in z.files else None
            zid, zcl = [str(v) for v in z["ids"].tolist()], [str(v) for v in z["classes"].tolist()]
            zti = np.asarray(z["true_indices"]).astype(np.int64).tolist()
        n = len(rows[g])
        if za != list(arms) or zcl != classes:
            raise RunError(f"{what} {g}: arms/classes differ")
        if zid != [r["id"] for r in rows[g]] or zti != [classes.index(r["label"]) for r in rows[g]]:
            raise RunError(f"{what} {g}: ids or true labels differ from manifest")
        if lg.shape != (n, *shape_mid, len(arms), R.N_CLASSES) or not np.isfinite(lg).all():
            raise RunError(f"{what} {g}: logits shape {lg.shape} or non-finite")
        return lg, zc

    # 4a. prior irregular-time scores + result (arms addressed by name only)
    state["stage"] = "prior_material"
    prior_lg, prior_rows = {}, {}
    pres = json.loads(Path(cfg["prior_result"]).read_text(encoding="utf-8"))
    if list(pres.get("arm_order", [])) != list(PRIOR_ARMS) or list(pres.get("cases", [])) != list(IRREGULAR):
        raise RunError("prior result arm/case order differs")
    for g in GROUPS:
        p = cfg["prior_scores"][g]
        lg, zc = load_scores(p, g, PRIOR_ARMS, (len(IRREGULAR),), "prior scores")
        if zc != list(IRREGULAR):
            raise RunError(f"prior scores {g}: cases differ")
        if not all(np.array_equal(lg[:, ci, PRIOR_ARMS.index("original")], saved_a[g]) for ci in range(len(IRREGULAR))):
            raise RunError(f"prior scores {g}: original logits differ from saved scores")
        prior_sha = (pres.get("hashes") or {}).get("scores", {}).get(g)
        if prior_sha is not None and prior_sha != pin_of[norm(p)]:
            raise RunError(f"prior scores {g}: sha differs from prior RESULT hashes.scores")
        pr_rows, n = {}, len(rows[g])
        for c in IRREGULAR:
            rs = pres["groups"][g]["cases"][c]["rows"]
            if len(rs) != n or [r["id"] for r in rs] != [r["id"] for r in rows[g]] \
                    or [r["index"] for r in rs] != list(range(n)):
                raise RunError(f"prior result {g}/{c}: row ids/order differ from manifest")
            pr_rows[c] = rs
        prior_lg[g], prior_rows[g] = lg, pr_rows

    # 4b. prior regular [::2] scores + result
    reg_lg, reg_rows = {}, {}
    rres = json.loads(Path(cfg["regular_result"]).read_text(encoding="utf-8"))
    if list(rres.get("arm_order", [])) != list(REGULAR_ARMS):
        raise RunError("regular result arm order differs")
    for g in GROUPS:
        p = cfg["regular_scores"][g]
        lg, _ = load_scores(p, g, REGULAR_ARMS, (), "regular scores")
        if not np.array_equal(lg[:, REGULAR_ARMS.index("original")], saved_a[g]):
            raise RunError(f"regular scores {g}: original logits differ from saved scores")
        if g == "native" and not np.array_equal(lg[:, REGULAR_ARMS.index("sparse")], b_native):
            raise RunError("regular scores native: sparse logits differ from saved decimated logits")
        reg_sha = (rres.get("hashes") or {}).get("scores", {}).get(g)
        if reg_sha is not None and reg_sha != pin_of[norm(p)]:
            raise RunError(f"regular scores {g}: sha differs from regular RESULT hashes.scores")
        rr, n = rres["groups"][g]["rows"], len(rows[g])
        if len(rr) != n or [r["id"] for r in rr] != [r["id"] for r in rows[g]] \
                or [r["index"] for r in rr] != list(range(n)):
            raise RunError(f"regular result {g}: row ids/order differ from manifest")
        if any(any(a not in r.get("input_sha256", {}) for a in ("original", "sparse", "previous", "linear"))
               for r in rr):
            raise RunError(f"regular result {g}: input_sha256 records incomplete")
        reg_lg[g], reg_rows[g] = lg, rr

    def stored(g, i, c, arm):
        if c == "clean":
            return saved_a[g][i]
        if c == "regular_half":
            return reg_lg[g][i, REGULAR_ARMS.index(STORED_NAME[c][arm])]
        return prior_lg[g][i, IRREGULAR.index(c), PRIOR_ARMS.index(STORED_NAME["irregular"][arm])]

    # 5. local APIs, pinned
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
        raise RunError("API modules not pinned or differ from pins", {"modules": unpinned})
    _camera_check()
    api = {"F": F, "build_channels": build_channels, "to_native_like": to_native_like,
           "native_roots": roots["native"], "mp_roots": roots["mediapipe"]}
    sha = R.sha256_array

    def raw(*arrays):
        return I.raw_hash(R, *arrays)

    def prepare_base(g, i):
        r = rows[g][i]
        base = I.prepare_base(R, api, g, r, tol)
        pb = prior_rows[g][IRREGULAR[0]][i]["base"]
        for k in ("source_raw_sha256", "original_input_sha256"):
            if base["meta"][k] != pb[k]:
                raise RunError(f"{r['id']}: base {k} differs from prior result")
        if base["meta"]["original_input_sha256"] != reg_rows[g][i]["input_sha256"]["original"]:
            raise RunError(f"{r['id']}: base original input differs from regular result")
        return base

    def features_ok(x, frames):
        return isinstance(x, np.ndarray) and x.dtype == np.float32 and x.shape == (frames, N_FEATURES) \
            and bool(np.isfinite(x).all())

    def prepare_cell(g, i, base, c):
        """-> ({unrestored, global_linear, local_insert: x}, cell). Stored-record checks before any scoring."""
        sid = rows[g][i]["id"]
        xy, pr, ts = base["xy"], base["pr"], base["ts"]
        src_hash = base["meta"]["source_raw_sha256"]
        hx = {"original": base["meta"]["original_input_sha256"]}
        rh = {"source": src_hash}
        checks, extra = {}, {}
        if c == "clean":
            uxy, upr, uts = xy, pr, ts
            x_un = x_gl = base["x"]
            hx["unrestored"] = hx["global_linear"] = hx["original"]
            gl = TG.restore(xy, pr, ts, "linear")
            rh["unrestored"], rh["global_linear"] = src_hash, raw(gl["xy"], gl["present"], gl["ts"])
            n_gl = len(gl["ts"])
            rebuilt = build_channels(xy, pr, ts)["x"]
            checks["global_linear_clean_identity"] = bool(R.clean_identity(api, xy, pr, ts, rebuilt)["ok"])
            checks["original_input_matches_base"] = sha(base["x"]) == hx["original"]
        elif c == "regular_half":
            uxy, upr, uts = xy[::2], pr[::2], ts[::2]
            rh["unrestored"] = raw(uxy, upr, uts)
            gl = TG.restore(uxy, upr, uts, "linear")
            rh["global_linear"], n_gl = raw(gl["xy"], gl["present"], gl["ts"]), len(gl["ts"])
            x_un = build_channels(uxy, upr, uts)["x"]
            x_gl = build_channels(gl["xy"], gl["present"], gl["ts"])["x"]
            hx["unrestored"], hx["global_linear"] = sha(x_un), sha(x_gl)
            rec = reg_rows[g][i]["input_sha256"]
            checks["kept_intervals_are_sums"] = bool(F.kept_intervals_are_sums(ts, uts, 2, 0))
            checks["inputs_match_regular_result"] = (hx["original"], hx["unrestored"], hx["global_linear"]) == \
                (rec["original"], rec["sparse"], rec["linear"])
            checks["sliced_unmodified_by_global"] = raw(uxy, upr, uts) == rh["unrestored"]
            extra["last_dense_frame_dropped_by_slice"] = (len(ts) - 1) % 2 == 1
        else:
            xs, meta = I.prepare_case(R, TG, api, base, sid, c)
            prior = prior_rows[g][c][i]["case"]
            cr = C.corrupt(xy, pr, ts, sid, c)
            uxy, upr, uts = cr["xy"], cr["present"], cr["ts"]
            x_un, x_gl = xs["corrupted"], xs["linear"]
            hx["unrestored"], hx["global_linear"] = sha(x_un), sha(x_gl)
            rh["unrestored"], rh["global_linear"] = raw(uxy, upr, uts), meta["raw_sha256"]["linear"]
            n_gl = int(meta["restored_frames"])
            checks["source_raw_matches_prior"] = src_hash == meta["raw_sha256"]["source"] == \
                prior["raw_sha256"]["source"]
            checks["corrupted_raw_matches_prior"] = rh["unrestored"] == meta["raw_sha256"]["corrupted"] == \
                prior["raw_sha256"]["corrupted"]
            checks["global_raw_matches_prior"] = rh["global_linear"] == prior["raw_sha256"]["linear"]
            checks["inputs_match_prior"] = all(
                hx[a] == meta["input_sha256"][s] == prior["input_sha256"][s]
                for a, s in STORED_NAME["irregular"].items())
            extra["prior_case_meta_identical"] = jsonable(meta) == prior
        # candidate: ONLY the unrestored arrays of this cell
        before = raw(uxy, upr, uts)
        li = LI.restore(uxy, upr, uts)
        checks["candidate_input_unmodified"] = raw(uxy, upr, uts) == before == rh["unrestored"]
        checks["source_unmodified"] = raw(xy, pr, ts) == src_hash
        ins = verify_insertion(li, uxy, upr, uts)
        li_snap = {k: sha(li[k]) for k in ("xy", "present", "ts", "observed", "estimated", "exact_source_index")}
        x_li = build_channels(li["xy"], li["present"], li["ts"])["x"]
        checks["candidate_unmodified_by_build"] = li_snap == {k: sha(li[k]) for k in li_snap}
        rh["local_insert"], hx["local_insert"] = raw(li["xy"], li["present"], li["ts"]), sha(x_li)
        checks["features_finite_float32"] = features_ok(x_un, len(uts)) and features_ok(x_gl, n_gl) \
            and features_ok(x_li, len(li["ts"]))
        if c == "clean":
            checks["clean_candidate_raw_identity"] = bool(li["identity"]) and _same(li["xy"], xy) \
                and _same(li["present"], pr) and _same(li["ts"], ts)
            checks["clean_candidate_input_identity"] = _same(x_li, rebuilt)
        if not all(checks.values()):
            raise RunError(f"{sid}/{c}: cell invariants failed", {"checks": checks})
        cell = {"case": c, "raw_sha256": rh, "input_sha256": hx, "checks": checks, **extra,
                "endpoints_us": {"unrestored": [int(uts[0]), int(uts[-1])],
                                 "local_insert": [int(li["ts"][0]), int(li["ts"][-1])]},
                "frames": {"source": len(ts), "unrestored": len(uts), "global_linear": n_gl,
                           "local_insert": len(li["ts"])},
                "insertion": ins, "source_index_sha256": sha(li["exact_source_index"]),
                "cache": choose_cache(hx["local_insert"], hx)}
        return {"unrestored": x_un, "global_linear": x_gl, "local_insert": x_li}, jsonable(cell)

    def regular_crosscheck(g, i, cell):
        """Prior R.prepare_row reproduces the regular-result inputs that this run sliced itself."""
        x, meta = R.prepare_row(api, g, rows[g][i], tol)
        got = {a: sha(x[a]) for a in ("original", "sparse", "previous", "linear")}
        rec = {a: reg_rows[g][i]["input_sha256"][a] for a in got}
        mine = {"original": cell["input_sha256"]["original"], "sparse": cell["input_sha256"]["unrestored"],
                "linear": cell["input_sha256"]["global_linear"]}
        if got != rec or any(got[a] != v for a, v in mine.items()) or not meta["clean_identity"]["ok"]:
            raise RunError(f"{rows[g][i]['id']}: R.prepare_row differs from regular result / this slice",
                           {"prepare_row": got, "regular_result": rec, "this_run": mine})

    # 6. preflight ALL 4060 cells before model loading; any failure stops everything, nothing filtered
    state["stage"] = "preflight"
    cells, failures = {g: [] for g in GROUPS}, []
    for g in GROUPS:
        state["group"] = g
        for i, r in enumerate(rows[g]):
            state["id"], state["case"] = r["id"], None
            try:
                base = prepare_base(g, i)
            except Exception as exc:
                failures.append({"group": g, "index": i, "id": r["id"], "case": None,
                                 "error": f"{type(exc).__name__}: {exc}", "details": getattr(exc, "details", None)})
                cells[g].append(None)
                continue
            row_cells = {}
            for c in CASES:
                state["case"] = c
                try:
                    row_cells[c] = prepare_cell(g, i, base, c)[1]
                    if c == "regular_half":
                        regular_crosscheck(g, i, row_cells[c])
                    state["prepared_cells"][g] += 1
                except Exception as exc:
                    failures.append({"group": g, "index": i, "id": r["id"], "case": c,
                                     "error": f"{type(exc).__name__}: {exc}", "details": getattr(exc, "details", None)})
                    row_cells[c] = None
            cells[g].append(row_cells)
    state.update(group=None, id=None, case=None, stage="preflight_gate")
    if failures:
        counts = {}
        for f in failures:
            k = f"{f['group']}/{f['case'] or 'base'}"
            counts[k] = counts.get(k, 0) + 1
        raise RunError("preflight failed before model load; no cells filtered",
                       {"failures_n": counts, "failures": failures})
    n_cells = sum(state["prepared_cells"].values())
    if n_cells != CELLS_EXPECTED:
        raise RunError(f"prepared {n_cells} cells, expected {CELLS_EXPECTED}")
    expected_new = sum(cells[g][i][c]["cache"] is None for g in GROUPS for i in range(len(rows[g])) for c in CASES)
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
    state_before = state_digest(model, sha)
    if state_before["training_modules"]:
        raise RunError("model not in eval mode", {"training": state_before["training_modules"]})
    n_cls = R.N_CLASSES

    def forward(x, kind):
        if not features_ok(x, len(x)):
            raise RunError(f"bad forward input {getattr(x, 'shape', None)}")
        bx, pad, _ = collate([{"x": x, "frames": int(len(x)), "label": 0}])   # label unused by model
        if np.asarray(bx).shape[0] != 1 or not np.all(np.asarray(pad) == 1):
            raise RunError("collate did not give one all-valid sequence")
        with torch.inference_mode():
            lg = model(torch.from_numpy(bx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
        if lg.shape != (n_cls,) or not np.isfinite(lg).all():
            raise RunError(f"model output shape {lg.shape} or non-finite")
        state["forwards"][kind] += 1
        return lg

    def top1(v):
        return classes[int(np.argmax(v))]

    def reprepare(g, i, c, base):
        xs, cell = prepare_cell(g, i, base, c)
        if cell != cells[g][i][c]:
            raise RunError(f"{rows[g][i]['id']}/{c}: cell differs from preflight")
        return xs

    # 8. reproduction of stored unrestored/global_linear scores (fixed rows x all cases) BEFORE any reuse
    state["stage"] = "reproduction"
    repro, dedup = [], 0
    for g in GROUPS:
        for i in cfg["repro_indices"][g]:
            state["group"], state["id"] = g, rows[g][i]["id"]
            base, memo = prepare_base(g, i), {}
            for c in CASES:
                state["case"] = c
                xs = reprepare(g, i, c, base)
                for a in ("unrestored", "global_linear"):
                    h = cells[g][i][c]["input_sha256"][a]
                    if h in memo:
                        dedup += 1
                    else:
                        memo[h] = forward(xs[a], "reproduction")
                    ref = stored(g, i, c, a)
                    d = float(np.max(np.abs(memo[h] - ref)))
                    repro.append({"group": g, "index": i, "id": rows[g][i]["id"], "case": c, "arm": a,
                                  "input_sha256": h, "max_abs_diff": d, "top1_stored": top1(ref),
                                  "top1_now": top1(memo[h]), "ok": bool(d <= tol and top1(ref) == top1(memo[h]))})
    if len(repro) != 6 * len(CASES) * 2 or not all(x["ok"] for x in repro):
        raise RunError("reproduction failed; no stored score reused and no candidate scored", {"reproduction": repro})

    # 9. candidate scores: byte-identical built input of the same row/case reuses its stored score
    state["stage"] = "inference"
    logits, results = {}, {}
    for g in GROUPS:
        n = len(rows[g])
        lg = np.full((n, len(CASES), len(ARMS), n_cls), np.nan, dtype=np.float64)
        per_case = {c: [] for c in CASES}
        for i, r in enumerate(rows[g]):
            state["group"], state["id"], state["case"] = g, r["id"], None
            base = prepare_base(g, i)
            for ci, c in enumerate(CASES):
                state["case"] = c
                xs = reprepare(g, i, c, base)
                cell = cells[g][i][c]
                for ai, a in enumerate(ARMS[:3]):
                    lg[i, ci, ai] = stored(g, i, c, a)
                if cell["cache"] is None:
                    lg[i, ci, 3] = forward(xs["local_insert"], "new_candidate")
                    source = "new_forward"
                else:
                    lg[i, ci, 3] = stored(g, i, c, cell["cache"])
                    state["reused_cells"][cell["cache"]] += 1
                    source = f"stored_{cell['cache']}_identical_input"
                per_case[c].append({"index": i, "id": r["id"], "signer_id": r["signer_id"], "true": r["label"],
                                    "true_index": classes.index(r["label"]), "recording_id": R.recording_id(r["id"]),
                                    "local_insert_logits_source": source,
                                    "arms": {a: R.score_row(lg[i, ci, k], classes, r["label"])
                                             for k, a in enumerate(ARMS)},
                                    "cell": cell})
        if not np.isfinite(lg).all():
            raise RunError(f"{g}: non-finite logits")
        if not all(np.array_equal(lg[:, ci, 0], saved_a[g]) for ci in range(len(CASES))):
            raise RunError(f"{g}: original logits are not the saved scores in every case")
        logits[g], results[g] = lg, per_case
    state.update(group=None, id=None, case=None)
    fw = state["forwards"]
    counts_ok = fw["new_candidate"] == expected_new and sum(state["reused_cells"].values()) == n_cells - expected_new
    if not counts_ok:
        raise RunError("forward/reuse count mismatch", {"forwards": fw, "expected_new": expected_new,
                                                        "reused": state["reused_cells"]})

    # 10. model state unchanged, still eval
    state["stage"] = "model_state_after"
    state_after = state_digest(model, sha)
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
            np.savez_compressed(fh, logits=logits[g], cases=np.array(CASES), arms=np.array(ARMS),
                                ids=np.array([r["id"] for r in rows[g]]), classes=np.array(classes),
                                true_indices=np.array([classes.index(r["label"]) for r in rows[g]], dtype=np.int64))
        scores_sha[g] = sha256_file(p)

    # 12. hashes after
    state["stage"] = "hashes_after"
    pins_after = {p: sha256_file(p) for p in cfg["pins"]}
    sources_after = {g: R.hash_sources(rows[g], roots[g]) for g in GROUPS}
    modules_after = {k: sha256_file(p) for k, p in module_files.items()}
    code_after = code_hashes(cfg)
    config_sha, config_after = hashlib.sha256(cfg_bytes).hexdigest(), sha256_file(cfg_path)
    frozen_after = frozen()
    changed = {"pins": [p for p in pins_before if pins_before[p] != pins_after[p]],
               "sources": {g: [k for k in sources_before[g] if sources_before[g][k] != sources_after[g][k]]
                           for g in GROUPS},
               "modules": [k for k in modules_before if modules_before[k] != modules_after[k]],
               "code": code_after != code, "config": config_after != config_sha,
               "constants": frozen_after != frozen_before}
    write_json_exclusive(out_dir / "SOURCE_HASHES.json", {
        "pins": {"before": pins_before, "after": pins_after},
        "sources": {"before": sources_before, "after": sources_after},
        "imported_modules": {"files": module_files, "before": modules_before, "after": modules_after},
        "code": {"before": code, "after": code_after}, "model_state": {"before": state_before, "after": state_after},
        "config": {"path": cfg_path, "before": config_sha, "after": config_after}, "changed": changed},
        strict=False)
    if changed["pins"] or any(changed["sources"].values()) or changed["modules"] or changed["code"] \
            or changed["config"] or changed["constants"]:
        raise RunError("hashes changed during run", {"changed": changed})

    # 13. result
    state["stage"] = "write_result"
    groups_out, stats, net_recovery, vs_global = {}, {g: {} for g in GROUPS}, {}, {}
    clean_gate, regular_gate = {}, {}
    for g in GROUPS:
        cases_out = {}
        for c in CASES:
            rs = results[g][c]
            comps = {f"{s}->{t}": R.compare_arms(rs, s, t) for s, t in COMPARISONS}
            correct = {a: sum(int(r["arms"][a]["correct"]) for r in rs) for a in ARMS}
            per_signer = breakdown(rs, "signer_id")
            ul, gl = comps["unrestored->local_insert"], comps["global_linear->local_insert"]
            ins = [r["cell"]["insertion"] for r in rs]
            cases_out[c] = {
                "n": len(rs), "per_arm_correct": correct, "per_signer": per_signer,
                "per_class": breakdown(rs, "true"), "comparisons": comps,
                "regression_vs_global_linear": {"losses_n": gl["losses_n"], "ids": [x["id"] for x in gl["losses"]]},
                "insertion_summary": {
                    "identity_cells": sum(x["identity"] for x in ins),
                    "frames_in": sum(x["frames_in"] for x in ins), "frames_out": sum(x["frames_out"] for x in ins),
                    "inserted_frames": sum(x["inserted_frames"] for x in ins),
                    "inserted_joints": sum(x["inserted_joints"] for x in ins),
                    "gap_classes": {k: sum(x["gap_classes"][k] for x in ins) for k in ins[0]["gap_classes"]}},
                "calls": {"new_forward": sum(r["cell"]["cache"] is None for r in rs),
                          **{f"reused_{a}": sum(r["cell"]["cache"] == a for r in rs) for a in CACHE_ORDER}},
                "rows": rs}
            net_recovery.setdefault(g, {})[c] = {"net_local_vs_unrestored": ul["net"], "gains": ul["gains_n"],
                                                 "losses": ul["losses_n"]}
            vs_global.setdefault(g, {})[c] = cases_out[c]["regression_vs_global_linear"]
            if c in IRREGULAR:
                stats[g][c] = {"correct": {"corrupted": correct["unrestored"], "linear": correct["local_insert"]},
                               "per_signer": {s: {"corrupted": v["unrestored"], "linear": v["local_insert"]}
                                              for s, v in per_signer.items()},
                               "newly_wrong": ul["losses_n"]}
            if c == "clean":
                raw_ok = sum(r["cell"]["checks"]["clean_candidate_raw_identity"] for r in rs)
                in_ok = sum(r["cell"]["checks"]["clean_candidate_input_identity"] for r in rs)
                clean_gate[g] = {"cells": len(rs), "raw_identity": raw_ok, "input_identity": in_ok,
                                 "pass": raw_ok == in_ok == len(rs)}
            if c == "regular_half":
                regular_gate[g] = {"newly_wrong_unrestored_correct_local_wrong": ul["losses_n"],
                                   "ids": [x["id"] for x in ul["losses"]], "pass": ul["losses_n"] == 0}
        groups_out[g] = {"description": ("native provided-coordinate development rows (annotation_dev, validation)"
                                         if g == "native" else "cached MediaPipe training-exposed diagnostic rows"),
                         "n_rows": len(rows[g]), "source_recordings": R.recording_counts(results[g][CASES[0]]),
                         "totals": {c: cases_out[c]["per_arm_correct"] for c in CASES}, "cases": cases_out}
    contracts = {"preflight_all_cells_no_failures": not failures and n_cells == CELLS_EXPECTED,
                 "candidate_contract_all_cells": all(all(cells[g][i][c]["checks"].values()) for g in GROUPS
                                                     for i in range(len(rows[g])) for c in CASES),
                 "reproduction_within_tolerance": all(x["ok"] for x in repro),
                 "forward_and_reuse_counts_match": counts_ok,
                 "model_state_and_eval_unchanged": state_after == state_before and not state_after["training_modules"],
                 "pins_unchanged": not changed["pins"], "sources_unchanged": not any(changed["sources"].values()),
                 "imported_modules_unchanged": not changed["modules"], "code_unchanged": not changed["code"],
                 "config_unchanged": not changed["config"], "constants_unchanged": not changed["constants"]}
    adm = I.admission(stats, contracts)
    hold = adm["all_conditions_hold"] and all(v["pass"] for v in clean_gate.values()) \
        and all(v["pass"] for v in regular_gate.values())
    decision = {
        "prior_admission_reused_exactly": adm,
        "added_gate_clean_raw_and_input_identity": clean_gate,
        "added_gate_regular_half_no_newly_wrong_vs_unrestored": regular_gate,
        "all_conditions_hold": bool(hold),
        "decision": "ADMISSION_CONDITIONS_MET_PENDING_CODEX_VERIFICATION" if hold else "HOLD",
        "condition_specific_net_recovery_not_admission": net_recovery,
        "regressions_vs_global_linear_reported_not_selected": vs_global,
        "web_deployment": False, "production_authorised": False, "model_promoted": False,
        "note": "Frozen gates, not relaxed. HOLD is never promoted. A failing strict gate is a reported outcome "
                "and not by itself an implementation bug. Synthetic stress on reused/training-exposed rows; "
                "no significance claim."}
    result = {
        "status": "COMPLETE_PENDING_CODEX_VERIFICATION",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "nature": "offline observation-preserving local insertion diagnostic on a frozen model; no training, no search",
        "cases": list(CASES), "arm_order": list(ARMS), "candidate": "local_insert",
        "candidate_rule": {"insert_one_midpoint_iff": "1_500_000 <= 30*gap <= 2_000_060 and a joint present at both "
                                                      "adjacent source frames",
                           "midpoint": "round((t0+t1)/2) exact ties-to-even", "recursion": False,
                           "source_frames": "verbatim, each once, in order"},
        "cache_rule": {"order": list(CACHE_ORDER), "requires": "byte-identical built input of the same row/case "
                                                              "after successful reproduction"},
        "model": {"checkpoint": cfg["checkpoint"], "sha256": pin_of[norm(cfg["checkpoint"])], "classes": len(classes),
                  "api_modules": model_mods, "state_combined_sha256": state_before["combined"],
                  "state_unchanged": True},
        "saved_mp_arms": mp_arms,
        "reproduction": {"atol": tol, "indices": cfg["repro_indices"], "deduplicated_identical_inputs": dedup,
                         "rows": repro},
        "admission": decision, "groups": groups_out, "failures": failures, "interpretation_limits": LIMITS,
        "execution_counts": {"cells_total": n_cells, "cells_new_forward": expected_new,
                             "cells_reused": state["reused_cells"], "forwards": fw,
                             "forwards_total": sum(fw.values()), "regular_prepare_row_crosschecks":
                                 sum(len(rows[g]) for g in GROUPS),
                             "training": 0, "camera": 0, "user_data": 0, "downloads": 0},
        "camera_modules_in_sys_modules": [m for m in CAMERA_MODULES if m in sys.modules],
        "hashes": {"code": code, "config": config_sha, "plan_file": pin_of[norm(cfg["plan_file"])],
                   "prior_result": pin_of[norm(cfg["prior_result"])],
                   "regular_result": pin_of[norm(cfg["regular_result"])],
                   "prior_scores": {g: pin_of[norm(cfg["prior_scores"][g])] for g in GROUPS},
                   "regular_scores": {g: pin_of[norm(cfg["regular_scores"][g])] for g in GROUPS},
                   "scores": scores_sha, "imported_modules": modules_before},
        "seconds": round(time.time() - started, 2),
    }
    write_json_exclusive(out_dir / "RESULT.json", result)
    print(json.dumps({"decision": decision["decision"], "forwards": fw, "reused": state["reused_cells"],
                      **{f"{g}/{c}": groups_out[g]["cases"][c]["per_arm_correct"] for g in GROUPS for c in CASES}},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                 # FAILURE.json already written when output_dir existed
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
