"""Time-grid restoration audit: same frozen TCN, raw frames[::2] restored onto a nominal 30 Hz grid.

Diagnostic only. No training, no threshold/parameter search, no camera, no user data, no network.

Arms per row (same model, same rows):
  original            A  stored scores (native original_logits / MP arm 'mediapipe')
  sparse              B  build_channels(raw xy/present/ts[::2])  (native stored, MP new)
  endpoint_reference  E  build_channels(original[:last_retained+1]); stored A when identical
  previous            H  build_channels(timegrid.restore(B raw, 'previous'))  mechanism control
  linear              L  build_channels(timegrid.restore(B raw, 'linear'))    predeclared primary
restore() never receives E, A or any dropped frame. H/L normalisation scales are re-estimated
from their own restored sequences (including estimated points), never from A/E.

Groups are never pooled: 'native' = provided-coordinate development rows (annotation_dev,
validation, signers 11/12/13); 'mediapipe' = cached MediaPipe training-exposed diagnostic rows
(signers 1/2). Neither is a new independent test.

Run: python run_audit.py --config <config.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import timegrid as TG

ARMS = ("original", "sparse", "endpoint_reference", "previous", "linear")
NEW_ARMS = {"native": ("endpoint_reference", "previous", "linear"),
            "mediapipe": ("sparse", "endpoint_reference", "previous", "linear")}
COMPARISONS = (("sparse", "linear"), ("sparse", "previous"), ("previous", "linear"),
               ("original", "endpoint_reference"), ("endpoint_reference", "linear"))
CONTEXT_COMPARISONS = (("original", "sparse"), ("original", "linear"))
GROUPS = ("native", "mediapipe")
N_CLASSES = 93
NATIVE_N, NATIVE_ROLE, NATIVE_SIGNERS = 699, "validation", {11, 12, 13}
MP_N, MP_ROLE, MP_SIGNERS = 113, "PREPARED_TRAINING_CANDIDATE_NOT_REGISTERED", {1, 2}
RESERVED_SIGNERS = {14, 15, 16}
FIXED_THREADS, FIXED_TOLERANCE = 2, 1e-5
NATIVE_MIN_NET_GAIN = 7          # practical ~1 pp of 699 gate, NOT a significance test
REGIONS = {"body": slice(0, 25), "left_hand": slice(25, 46), "right_hand": slice(46, 67)}
PATH_KEYS = ("lab_root", "pilot_src", "loader_dir", "checkpoint", "native_metadata", "mp_manifest",
             "saved_native_logits", "saved_mp_logits", "output_dir", "plan_file")
REQUIRED_PINS = ("checkpoint", "native_metadata", "mp_manifest", "saved_native_logits",
                 "saved_mp_logits", "plan_file")
NATIVE_KEYS = ("id", "label", "signer_id", "role", "path", "sha256")
MP_KEYS = ("sample_id", "label", "signer_id", "role", "input_path", "input_sha256",
           "observation_path", "observation_sha256")
LIMITS = [
    "Diagnostic of a retained-frame time-grid restoration on the same frozen model; no training, "
    "no deployment authorisation, no webcam or target-18 generalisation claim.",
    "native and mediapipe groups are reported separately and never pooled; neither is a new "
    "independent test (mediapipe rows are training-exposed).",
    "'previous' is a mechanism control and is never substituted for 'linear' based on scores.",
    "Estimated points are interpolations on the retained clock, not observations; the original "
    "final frame dropped by an even-length slice is not restored (output ends at the last "
    "retained timestamp).",
    "previous/linear body and hand scales are re-estimated from each restored sequence "
    "(estimated points included), so normalisation can differ from both sparse and original.",
    "The native net-gain gate (>= 7 rows) is a practical ~1 pp threshold, not statistical "
    "significance.",
]


class AuditError(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


# --- helpers -------------------------------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_array(x) -> str:
    x = np.ascontiguousarray(x)
    return hashlib.sha256(f"{x.shape}|{x.dtype}|".encode() + x.tobytes()).hexdigest()


def norm_path(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def under(path, root) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def num_or_null(v):
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def same(a, b) -> bool:
    """Exact equality incl. shape and dtype; NaN == NaN for floating arrays."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.floating):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def recording_id(sample_id: str) -> str:
    return re.sub(r"_a\d\d_\d\d\d$", "", sample_id)


def is_int(v) -> bool:
    return type(v) is int


def write_json_exclusive(path, value):
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write("\n")


def scales_of(built) -> dict:
    hs = built["hand_scales"]
    return {"body": num_or_null(built["body_scale"]), "left_hand": num_or_null(hs.get("left")),
            "right_hand": num_or_null(hs.get("right"))}


# --- config --------------------------------------------------------------------
def validate_config(cfg: dict) -> dict:
    need = PATH_KEYS + ("pins", "native_roots", "mp_roots", "threads", "tolerance", "repro_indices")
    missing = [k for k in need if k not in cfg]
    if missing:
        raise AuditError(f"config lacks keys {missing}")
    for k in PATH_KEYS:
        if not isinstance(cfg[k], str) or not os.path.isabs(cfg[k]):
            raise AuditError(f"config path {k} must be an absolute string")
    pins = cfg["pins"]
    if not isinstance(pins, dict) or not pins:
        raise AuditError("pins must be a non-empty object")
    for p, d in pins.items():
        if not os.path.isabs(p) or not isinstance(d, str) or not re.fullmatch(r"[0-9a-f]{64}", d):
            raise AuditError(f"invalid pin {p!r}")
    pinned = {norm_path(p) for p in pins}
    for k in REQUIRED_PINS:
        if norm_path(cfg[k]) not in pinned:
            raise AuditError(f"required pin missing for {k}")
    for k in ("native_roots", "mp_roots"):
        rs = cfg[k]
        if not isinstance(rs, list) or not rs or any(not isinstance(r, str) or not os.path.isabs(r) for r in rs):
            raise AuditError(f"{k} must be a non-empty list of absolute paths")
    if not is_int(cfg["threads"]) or cfg["threads"] != FIXED_THREADS:
        raise AuditError(f"threads must be {FIXED_THREADS}")
    if not isinstance(cfg["tolerance"], (int, float)) or float(cfg["tolerance"]) != FIXED_TOLERANCE:
        raise AuditError(f"tolerance must be {FIXED_TOLERANCE}")
    ri = cfg["repro_indices"]
    if not isinstance(ri, dict) or set(ri) != set(GROUPS):
        raise AuditError("repro_indices must have exactly keys native, mediapipe")
    for g, n in (("native", NATIVE_N), ("mediapipe", MP_N)):
        v = ri[g]
        if not isinstance(v, list) or len(v) != 3 or len(set(v)) != 3 or any(not is_int(i) or not 0 <= i < n for i in v):
            raise AuditError(f"repro_indices.{g} must be 3 unique ints in [0,{n})")
    out = norm_path(cfg["output_dir"])
    for r in cfg["native_roots"] + cfg["mp_roots"]:
        if under(out, norm_path(r)):
            raise AuditError("output_dir lies inside a source root")
    return cfg


# --- metadata ------------------------------------------------------------------
def locate(path, roots, what) -> Path:
    """Listed .npz under an allowed root -> resolved path. Nothing is opened here."""
    p = Path(path).resolve()
    if p.suffix.lower() != ".npz":
        raise AuditError(f"{what}: not an .npz file: {path}")
    if not any(under(norm_path(p), norm_path(Path(r).resolve())) for r in roots):
        raise AuditError(f"{what}: {path} outside allowed roots")
    return p


def check_file(path, sha, roots, what):
    """Root and suffix check, then hash, BEFORE any open by np.load. -> resolved path."""
    p = locate(path, roots, what)
    got = sha256_file(p)
    if got != sha:
        raise AuditError(f"{what}: sha256 mismatch", {"path": str(path), "got": got, "listed": sha})
    return p


def load_native_rows(path) -> list:
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))["annotation_dev"]
    if not isinstance(rows, list) or len(rows) != NATIVE_N:
        raise AuditError(f"native metadata: {len(rows)} rows, expected {NATIVE_N}")
    out = []
    for r in rows:
        if any(k not in r for k in NATIVE_KEYS):
            raise AuditError(f"native row lacks keys: {r.get('id')!r}")
        if r["role"] != NATIVE_ROLE or not is_int(r["signer_id"]) or r["signer_id"] in RESERVED_SIGNERS \
                or r["signer_id"] not in NATIVE_SIGNERS:
            raise AuditError(f"native row {r['id']!r}: role/signer not allowed")
        out.append({"id": str(r["id"]), "label": r["label"], "signer_id": r["signer_id"], "meta": r,
                    "files": {"native": (r["path"], r["sha256"])}})
    check_ids_signers(out, NATIVE_SIGNERS, "native")
    return out


def load_mp_rows(path) -> list:
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))["rows"]
    if not isinstance(rows, list) or len(rows) != MP_N:
        raise AuditError(f"mp manifest: {len(rows)} rows, expected {MP_N}")
    out = []
    for r in rows:
        if any(k not in r for k in MP_KEYS):
            raise AuditError(f"mp row lacks keys: {r.get('sample_id')!r}")
        if r["role"] != MP_ROLE or not is_int(r["signer_id"]) or r["signer_id"] in RESERVED_SIGNERS \
                or r["signer_id"] not in MP_SIGNERS:
            raise AuditError(f"mp row {r['sample_id']!r}: role/signer not allowed")
        out.append({"id": str(r["sample_id"]), "label": r["label"], "signer_id": r["signer_id"], "meta": r,
                    "files": {"input": (r["input_path"], r["input_sha256"]),
                              "observation": (r["observation_path"], r["observation_sha256"])}})
    check_ids_signers(out, MP_SIGNERS, "mediapipe")
    return out


def check_ids_signers(rows, signers, group):
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise AuditError(f"{group}: duplicate ids")
    if {r["signer_id"] for r in rows} != signers:
        raise AuditError(f"{group}: signer set {sorted({r['signer_id'] for r in rows})} != {sorted(signers)}")


def hash_sources(rows, roots) -> dict:
    """{id: {file_kind: sha256}} after root checks; comparison with the listing is the caller's."""
    return {r["id"]: {k: sha256_file(locate(p, roots, f"{r['id']}/{k}")) for k, (p, _) in r["files"].items()}
            for r in rows}


def check_listed(rows, hashes, group):
    bad = [{"id": r["id"], "file": k} for r in rows for k, (_, s) in r["files"].items() if hashes[r["id"]][k] != s]
    if bad:
        raise AuditError(f"{group}: {len(bad)} row files differ from their listed sha256", {"mismatched": bad})


def hash_pins(pins) -> dict:
    return {p: sha256_file(p) for p in pins}


# --- saved material ------------------------------------------------------------
def load_saved_native(path, rows):
    with np.load(path, allow_pickle=False) as z:
        need = ("ids", "classes", "original_logits", "decimated_logits")
        if any(k not in z.files for k in need):
            raise AuditError(f"saved native lacks keys; has {z.files}")
        ids, classes = [str(i) for i in z["ids"].tolist()], [str(c) for c in z["classes"].tolist()]
        a, b = np.asarray(z["original_logits"], np.float64), np.asarray(z["decimated_logits"], np.float64)
    check_classes(classes)
    if ids != [r["id"] for r in rows]:
        raise AuditError("saved native id order differs from metadata")
    for name, arr in (("original_logits", a), ("decimated_logits", b)):
        if arr.shape != (NATIVE_N, N_CLASSES) or not np.isfinite(arr).all():
            raise AuditError(f"saved native {name} shape {arr.shape} or non-finite")
    for r in rows:
        if r["label"] not in classes:
            raise AuditError(f"{r['id']}: label not in classes")
    return classes, a, b


def load_saved_mp(path, rows, classes):
    with np.load(path, allow_pickle=False) as z:
        need = ("sample_ids", "classes", "arms", "logits", "true_indices")
        if any(k not in z.files for k in need):
            raise AuditError(f"saved mp lacks keys; has {z.files}")
        ids, cls = [str(i) for i in z["sample_ids"].tolist()], [str(c) for c in z["classes"].tolist()]
        arms, lg = [str(a) for a in z["arms"].tolist()], np.asarray(z["logits"], np.float64)
        ti = np.asarray(z["true_indices"]).astype(np.int64).tolist()
    if cls != classes:
        raise AuditError("saved mp class order differs from saved native class order")
    if ids != [r["id"] for r in rows]:
        raise AuditError("saved mp sample order differs from manifest")
    if len(arms) != 4 or arms.count("mediapipe") != 1:
        raise AuditError(f"saved mp arms {arms}: need 4 with exactly one 'mediapipe'")
    if lg.shape != (MP_N, 4, N_CLASSES) or not np.isfinite(lg).all():
        raise AuditError(f"saved mp logits shape {lg.shape} or non-finite")
    for r, t in zip(rows, ti):
        if r["label"] not in classes or classes.index(r["label"]) != t:
            raise AuditError(f"{r['id']}: saved true index {t} inconsistent with label {r['label']!r}")
    return lg[:, arms.index("mediapipe")].copy(), arms


def check_classes(classes):
    if len(classes) != N_CLASSES or len(set(classes)) != N_CLASSES:
        raise AuditError(f"classes: {len(classes)} entries, expected {N_CLASSES} unique")


# --- per-row inputs (no inference) ---------------------------------------------
def to_bool_mask(m, what):
    m = np.asarray(m)
    if m.dtype == np.bool_:
        return m.copy()
    if (np.issubdtype(m.dtype, np.integer) or np.issubdtype(m.dtype, np.floating)) and np.all((m == 0) | (m == 1)):
        return m == 1
    raise AuditError(f"{what}: mask must be bool or exactly 0/1")


def unpack_native_like(out, t):
    """to_native_like result -> (xy [T,67,2], present [T,67]); shape-checked (see HANDOFF)."""
    if isinstance(out, dict):
        xy = out.get("xy", out.get("xy_native"))
        pr = out.get("present", out.get("coordinate_present"))
    elif isinstance(out, (tuple, list)) and len(out) >= 2:
        xy, pr = out[0], out[1]
    else:
        raise AuditError(f"to_native_like returned unsupported {type(out).__name__}")
    xy, pr = np.asarray(xy), np.asarray(pr)
    if xy.shape != (t, TG.N_JOINTS, 2) or pr.shape != (t, TG.N_JOINTS):
        raise AuditError(f"to_native_like shapes {xy.shape} {pr.shape}")
    return xy, pr


def load_mp(api, row, tol):
    (ip, isha), (op, osha) = row["files"]["input"], row["files"]["observation"]
    ip = check_file(ip, isha, api["mp_roots"], f"{row['id']}/input")
    op = check_file(op, osha, api["mp_roots"], f"{row['id']}/observation")
    with np.load(ip, allow_pickle=False) as z:
        if any(k not in z.files for k in ("x", "frame_id", "timestamp_us")):
            raise AuditError(f"{row['id']}: prepared npz lacks keys; has {z.files}")
        x, fid, ts = z["x"].copy(), z["frame_id"].copy(), z["timestamp_us"].copy()
    with np.load(op, allow_pickle=False) as z:
        if any(k not in z.files for k in ("xy_px", "usable", "frame_id", "timestamp_us")):
            raise AuditError(f"{row['id']}: observation npz lacks keys; has {z.files}")
        oxy, ous, ofid, ots = z["xy_px"].copy(), z["usable"].copy(), z["frame_id"].copy(), z["timestamp_us"].copy()
    t, n = len(fid), len(ofid)
    if x.ndim != 2 or x.shape != (t, 280) or ts.shape != (t,) or t < 1:
        raise AuditError(f"{row['id']}: prepared shapes x{x.shape} fid{fid.shape} ts{ts.shape}")
    if oxy.shape != (n, 75, 2) or ous.shape != (n, 75) or ots.shape != (n,):
        raise AuditError(f"{row['id']}: observation shapes {oxy.shape} {ous.shape} {ots.shape}")
    fl, ofl = [int(v) for v in fid], [int(v) for v in ofid]
    if len(set(fl)) != t or len(set(ofl)) != n:
        raise AuditError(f"{row['id']}: frame_id not unique")
    pos = {f: i for i, f in enumerate(ofl)}
    miss = [f for f in fl if f not in pos]
    if miss:
        raise AuditError(f"{row['id']}: {len(miss)} prepared frame_ids absent from observation")
    idx = np.array([pos[f] for f in fl], dtype=np.int64)
    if not np.array_equal(ots[idx].astype(np.int64), ts.astype(np.int64)):
        raise AuditError(f"{row['id']}: observation timestamps differ from prepared")
    usable = to_bool_mask(ous, f"{row['id']}/usable")
    xy, pr = unpack_native_like(api["to_native_like"](oxy[idx], usable[idx]), t)
    rebuilt = api["build_channels"](xy, pr, ts)["x"]
    if rebuilt.shape != x.shape:
        raise AuditError(f"{row['id']}: rebuilt x shape {rebuilt.shape} != prepared {x.shape}")
    diff = float(np.max(np.abs(rebuilt.astype(np.float64) - x.astype(np.float64))))
    if not diff <= tol:
        raise AuditError(f"{row['id']}: rebuilt x differs from prepared x by {diff}")
    return xy, pr, ts, np.asarray(x, dtype=np.float32), diff


def clock_report(ts) -> dict:
    ti = [int(v) for v in ts]
    d = [b - a for a, b in zip(ti, ti[1:])]
    return {"frames": len(ti), "first_us": ti[0], "last_us": ti[-1], "duration_us": ti[-1] - ti[0],
            "min_interval_us": min(d) if d else None, "max_interval_us": max(d) if d else None,
            "strictly_increasing": all(x > 0 for x in d), "nominal_30hz": TG.is_nominal_clock(ti)}


def clean_identity(api, xy, pr, ts, x_ref) -> dict:
    out = {}
    for m in ("previous", "linear"):
        r = TG.restore(xy, pr, ts, m)
        c = {"identity_flag": bool(r["identity"]), "xy": same(r["xy"], xy), "present": same(r["present"], pr),
             "ts": same(r["ts"], ts), "observed_is_present": bool(np.array_equal(r["observed"], to_bool_mask(pr, "present"))),
             "no_estimated": not bool(r["estimated"].any())}
        c["built_x"] = all(c.values()) and same(api["build_channels"](r["xy"], r["present"], r["ts"])["x"], x_ref)
        out[m] = c
    out["ok"] = all(all(out[m].values()) for m in ("previous", "linear"))
    return out


def check_restored(res, xy2, ts2) -> dict:
    h, l = res["previous"], res["linear"]
    for k in ("ts", "present", "observed", "estimated", "exact_source_index"):
        if not same(h[k], l[k]):
            raise AuditError(f"previous/linear differ in {k}")
    if h["identity"] != l["identity"]:
        raise AuditError("previous/linear identity flags differ")
    pb, obs, est, src = h["present"].astype(bool), h["observed"], h["estimated"], h["exact_source_index"]
    exact = src >= 0
    if (obs & est).any() or not np.array_equal(obs | est, pb):
        raise AuditError("observed/estimated are not a disjoint cover of present")
    if est[exact].any() or obs[~exact].any():
        raise AuditError("provenance inconsistent with exact_source_index")
    if not (same(h["xy"][exact], xy2[src[exact]]) and same(l["xy"][exact], xy2[src[exact]])):
        raise AuditError("exact grid frames are not verbatim retained frames")
    rest = ~exact[:, None] & ~est
    if h["xy"][rest].any() or l["xy"][rest].any():
        raise AuditError("unsupported points are not zero")
    out_ts = [int(v) for v in h["ts"]]
    if out_ts[0] != int(ts2[0]) or out_ts[-1] != int(ts2[-1]):
        raise AuditError("restored endpoints differ from retained endpoints")
    t2 = [int(v) for v in ts2]
    interior = np.flatnonzero(~exact)
    long_gap = sum(30 * (t2[bisect_right(t2, out_ts[k])] - t2[bisect_right(t2, out_ts[k]) - 1])
                   > 2 * 1_000_000 + 60 for k in interior)
    d = np.abs(h["xy"][est].astype(np.float64) - l["xy"][est].astype(np.float64))
    return {"identity": bool(h["identity"]), "frames": len(out_ts), "exact_frames": int(exact.sum()),
            "interior_frames": int(len(interior)), "interior_frames_long_gap": int(long_gap),
            "observed_slots": int(obs.sum()), "estimated_slots": int(est.sum()),
            "estimated_slots_by_region": {k: int(est[:, s].sum()) for k, s in REGIONS.items()},
            "unsupported_interior_slots": int(len(interior) * TG.N_JOINTS - est[~exact].sum()),
            "max_abs_previous_vs_linear_estimate": float(d.max()) if d.size else 0.0}


def prepare_row(api, group, row, tol):
    """-> (x_by_arm, meta). x for 'original' is the input behind the stored A scores."""
    if group == "native":
        nat = api["F"].load_native(row["meta"], [NATIVE_ROLE], sorted(NATIVE_SIGNERS), api["native_roots"])
        if nat["sha256"] != row["files"]["native"][1]:
            raise AuditError(f"{row['id']}: loader sha differs from metadata")
        xy, pr, ts, x_saved, mp_diff = nat["xy"], nat["present"], nat["ts"], None, None
    else:
        xy, pr, ts, x_saved, mp_diff = load_mp(api, row, tol)
    bc, t = api["build_channels"], len(ts)
    meta = {"original_clock": clock_report(ts)}
    built_a = bc(xy, pr, ts)
    x = {"original": built_a["x"] if x_saved is None else x_saved}
    meta["clean_identity"] = clean_identity(api, xy, pr, ts, built_a["x"])
    xy2, pr2, ts2 = api["F"].decimate(xy, pr, ts, stride=2, offset=0)
    lri = 2 * ((t - 1) // 2)
    if not api["F"].kept_intervals_are_sums(ts, ts2, 2, 0) or len(ts2) != lri // 2 + 1 or int(ts2[-1]) != int(ts[lri]):
        raise AuditError(f"{row['id']}: retained frames are not raw [::2]")
    built_b = bc(xy2, pr2, ts2)
    x["sparse"] = built_b["x"]
    e_needed = lri != t - 1
    built_e = bc(xy[:lri + 1], pr[:lri + 1], ts[:lri + 1]) if e_needed else built_a
    if e_needed:
        x["endpoint_reference"] = built_e["x"]
    snap = (xy2.copy(), pr2.copy(), ts2.copy())
    res = {m: TG.restore(xy2, pr2, ts2, m) for m in ("previous", "linear")}
    if not all(same(a, b) for a, b in zip(snap, (xy2, pr2, ts2))):
        raise AuditError(f"{row['id']}: restore mutated its input")
    meta["restored"] = check_restored(res, xy2, ts2)
    built_h = bc(res["previous"]["xy"], res["previous"]["present"], res["previous"]["ts"])
    built_l = bc(res["linear"]["xy"], res["linear"]["present"], res["linear"]["ts"])
    x["previous"], x["linear"] = built_h["x"], built_l["x"]
    rts = res["linear"]["ts"]
    meta.update({
        "frames": {"original": t, "retained": len(ts2), "endpoint_reference": lri + 1, "restored": len(rts)},
        "durations_us": {"original": int(ts[-1]) - int(ts[0]), "retained": int(ts2[-1]) - int(ts2[0]),
                         "restored": int(rts[-1]) - int(rts[0])},
        "endpoints_us": {"original": [int(ts[0]), int(ts[-1])], "retained": [int(ts2[0]), int(ts2[-1])],
                         "restored": [int(rts[0]), int(rts[-1])]},
        "original_final_frame_dropped_by_slice": e_needed,
        "endpoint_reference_source": "inferred" if e_needed else "stored_original",
        "scales": {"original": scales_of(built_a), "sparse": scales_of(built_b),
                   "endpoint_reference": scales_of(built_e), "previous": scales_of(built_h),
                   "linear": scales_of(built_l)},
        "mp_prepared_vs_rebuilt_max_abs_diff": mp_diff,
    })
    return x, meta


# --- scoring -------------------------------------------------------------------
def score_row(logits, classes, true_label) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(classes),) or not np.isfinite(logits).all():
        raise AuditError(f"logits shape {logits.shape} or non-finite")
    yi = classes.index(true_label)
    order = np.argsort(-logits, kind="stable")
    rival = next(int(j) for j in order if int(j) != yi)
    return {"pred": classes[int(order[0])], "correct": bool(int(order[0]) == yi),
            "rank": int(np.where(order == yi)[0][0]) + 1, "margin": float(logits[yi] - logits[rival])}


def compare_arms(rows, src, dst) -> dict:
    """Paired transition table src -> dst; arithmetic is self-checked."""
    gains, losses, w2w_changed = [], [], []
    c2c = w2w_same = 0
    for r in rows:
        a, b = r["arms"][src], r["arms"][dst]
        item = {"id": r["id"], "signer_id": r["signer_id"], "true": r["true"],
                "src_pred": a["pred"], "dst_pred": b["pred"]}
        if a["correct"] and b["correct"]:
            c2c += 1
        elif b["correct"]:
            gains.append(item)
        elif a["correct"]:
            losses.append(item)
        elif a["pred"] != b["pred"]:
            w2w_changed.append(item)
        else:
            w2w_same += 1
    n = len(rows)
    src_ok = sum(r["arms"][src]["correct"] for r in rows)
    dst_ok = sum(r["arms"][dst]["correct"] for r in rows)
    changed = sum(r["arms"][src]["pred"] != r["arms"][dst]["pred"] for r in rows)
    if dst_ok - src_ok != len(gains) - len(losses):
        raise AuditError(f"{src}->{dst}: net arithmetic mismatch")
    if changed != len(gains) + len(losses) + len(w2w_changed):
        raise AuditError(f"{src}->{dst}: top1-change arithmetic mismatch")
    if c2c + len(gains) + len(losses) + len(w2w_changed) + w2w_same != n:
        raise AuditError(f"{src}->{dst}: partition arithmetic mismatch")
    return {"src": src, "dst": dst, "n": n, "src_correct": src_ok, "dst_correct": dst_ok,
            "net": dst_ok - src_ok, "top1_changed": changed, "correct_to_correct": c2c,
            "gains_n": len(gains), "losses_n": len(losses),
            "wrong_to_wrong_changed_n": len(w2w_changed), "wrong_to_wrong_same": w2w_same,
            "gains": gains, "losses": losses, "wrong_to_wrong_changed": w2w_changed}


def breakdown(rows, key) -> dict:
    out = {}
    for r in rows:
        g = out.setdefault(str(r[key]), {"n": 0, **{a: 0 for a in ARMS}})
        g["n"] += 1
        for a in ARMS:
            g[a] += r["arms"][a]["correct"]
    return dict(sorted(out.items()))


def recoveries(rows) -> dict:
    sub = [r for r in rows if r["arms"]["original"]["correct"] and not r["arms"]["sparse"]["correct"]]
    later = ("endpoint_reference", "previous", "linear")
    return {"n": len(sub), **{f"correct_in_{a}": sum(r["arms"][a]["correct"] for r in sub) for a in later},
            "rows": [{"id": r["id"], "true": r["true"], "sparse_pred": r["arms"]["sparse"]["pred"],
                      **{a: {k: r["arms"][a][k] for k in ("pred", "correct", "rank")} for a in later}}
                     for r in sub]}


def recording_counts(rows) -> dict:
    per = {}
    for r in rows:
        per.setdefault(str(r["signer_id"]), set()).add(r["recording_id"])
    return {"rows": len(rows), "recordings": len({r["recording_id"] for r in rows}),
            "per_signer_recordings": {s: len(v) for s, v in sorted(per.items())}}


def normalization_summary(rows) -> dict:
    out = {}
    for arm in ("previous", "linear", "endpoint_reference"):
        rel, fin = [], {"left_hand": 0, "right_hand": 0}
        for r in rows:
            sa, sb = r["scales"][arm], r["scales"]["sparse"]
            for k in ("body", "left_hand", "right_hand"):
                if sa[k] is not None and sb[k] is not None:
                    rel.append(abs(sa[k] - sb[k]) / sb[k])
            for h in fin:
                fin[h] += (sa[h] is None) != (sb[h] is None)
        out[f"{arm}_vs_sparse"] = {"max_rel_scale_diff": max(rel) if rel else 0.0,
                                   "rows_with_any_scale_diff": sum(
                                       r["scales"][arm] != r["scales"]["sparse"] for r in rows),
                                   "rows_hand_scale_finiteness_changed": fin}
    return out


def outcome_checks(group, rows, comps, identity_ok_n, input_failures) -> dict:
    sl = comps["sparse->linear"]
    sig = breakdown(rows, "signer_id")
    per_signer = {s: {"sparse": v["sparse"], "linear": v["linear"], "nondecrease": v["linear"] >= v["sparse"]}
                  for s, v in sig.items()}
    base = {"linear_net_vs_sparse": sl["net"], "gains": sl["gains_n"], "losses": sl["losses_n"],
            "per_signer": per_signer, "all_signers_nondecrease": all(v["nondecrease"] for v in per_signer.values()),
            "clean_identity_rows": identity_ok_n, "clean_identity_all": identity_ok_n == len(rows),
            "input_failures": input_failures,
            "newly_wrong_sparse_correct_linear_wrong": [x["id"] for x in sl["losses"]],
            "newly_wrong_original_correct_linear_wrong": [x["id"] for x in comps["original->linear"]["losses"]]}
    if group == "native":
        base["gate_net_gain_ge_7"] = sl["net"] >= NATIVE_MIN_NET_GAIN
        base["primary_pass"] = bool(base["gate_net_gain_ge_7"] and base["all_signers_nondecrease"]
                                    and base["clean_identity_all"] and input_failures == 0)
        base["note"] = "practical 1 pp gate, not statistical significance; no deployment authorisation"
    else:
        base["direction"] = "positive" if sl["net"] > 0 else ("zero" if sl["net"] == 0 else "negative")
        base["corroboration_conditions_met"] = bool(sl["net"] >= 0 and base["all_signers_nondecrease"]
                                                    and base["clean_identity_all"] and input_failures == 0)
        base["note"] = "training-exposed cross-extractor corroboration only; not an independent test"
    return base


# --- main ------------------------------------------------------------------------
def main(argv=None):
    started = time.time()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg_path = os.path.abspath(args.config)
    cfg_bytes = Path(cfg_path).read_bytes()
    cfg = validate_config(json.loads(cfg_bytes.decode("utf-8-sig")))
    out_dir = Path(cfg["output_dir"])
    if os.path.lexists(out_dir):
        raise AuditError(f"output_dir {out_dir} exists; outputs are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=False)
    code = {"run_audit.py": sha256_file(__file__), "timegrid.py": sha256_file(TG.__file__)}
    state = {"stage": "start", "group": None, "row": None, "prepared": {g: 0 for g in GROUPS},
             "forwards_repro": 0, "forwards_new": 0, "details": None, "pins_before": None}
    try:
        return _run(cfg, cfg_path, cfg_bytes, out_dir, code, state, started)
    except BaseException as exc:
        failure = {"status": "FAILED", "stage": state["stage"], "group": state["group"], "row": state["row"],
                   "error_class": type(exc).__name__, "error": str(exc),
                   "details": getattr(exc, "details", None) or state["details"],
                   "counts": {"rows_prepared": state["prepared"], "forwards_repro": state["forwards_repro"],
                              "forwards_new": state["forwards_new"], "training": 0, "camera": 0,
                              "user_data": 0, "downloads": 0},
                   "pins_before": state["pins_before"],
                   "code_sha256": code, "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
                   "at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            write_json_exclusive(out_dir / "FAILURE.json", failure)
        except FileExistsError:
            pass
        raise


def _run(cfg, cfg_path, cfg_bytes, out_dir, code, state, started):
    tol = float(cfg["tolerance"])
    roots = {"native": cfg["native_roots"], "mediapipe": cfg["mp_roots"]}

    # 1. every pin before anything else is read
    state["stage"] = "pins"
    pins_before = hash_pins(cfg["pins"])
    state["pins_before"] = pins_before
    bad = {p: got for p, got in pins_before.items() if got != cfg["pins"][p]}
    if bad:
        raise AuditError("pin mismatch", {"mismatched": bad})
    pin_of = {norm_path(p): d for p, d in cfg["pins"].items()}

    # 2. row lists and every row file hash (roots checked first), before any data or model load
    state["stage"] = "metadata"
    rows = {"native": load_native_rows(cfg["native_metadata"]), "mediapipe": load_mp_rows(cfg["mp_manifest"])}
    state["stage"] = "source_hashes_before"
    sources_before = {g: hash_sources(rows[g], roots[g]) for g in GROUPS}
    for g in GROUPS:
        check_listed(rows[g], sources_before[g], g)

    # 3. saved scores
    state["stage"] = "saved_material"
    classes, a_native, b_native = load_saved_native(cfg["saved_native_logits"], rows["native"])
    a_mp, mp_arms = load_saved_mp(cfg["saved_mp_logits"], rows["mediapipe"], classes)
    saved_a = {"native": a_native, "mediapipe": a_mp}

    # 4. local APIs (explicit), no model yet
    state["stage"] = "imports"
    lab = cfg["lab_root"]
    sys.path[:0] = [lab, os.path.join(lab, "src"), cfg["pilot_src"], cfg["loader_dir"]]
    import torch
    import tools_prep_union93_candidate as u
    from pilot.data import build_channels, collate
    from pilot.web_input import to_native_like
    import framerate_impl as F
    module_files = {n: os.path.abspath(sys.modules[n].__file__) for n in
                    ("tools_prep_union93_candidate", "pilot.data", "pilot.web_input", "framerate_impl")}
    modules_before = {k: sha256_file(p) for k, p in module_files.items()}
    api = {"F": F, "build_channels": build_channels, "to_native_like": to_native_like,
           "native_roots": roots["native"], "mp_roots": roots["mediapipe"]}

    # 5. preflight: clocks, clean identity, all inputs built; no inference. Stop on any failure.
    metas, identity_fail, input_fail = {g: [] for g in GROUPS}, [], []
    for g in GROUPS:
        state["stage"], state["group"] = "preflight", g
        for i, r in enumerate(rows[g]):
            state["row"] = r["id"]
            try:
                x, meta = prepare_row(api, g, r, tol)
            except Exception as exc:                                  # recorded, run stops below
                input_fail.append({"group": g, "index": i, "id": r["id"],
                                   "error": f"{type(exc).__name__}: {exc}"})
                metas[g].append(None)
                continue
            meta["input_sha256"] = {a: sha256_array(v) for a, v in x.items()}
            if not meta["clean_identity"]["ok"]:
                identity_fail.append({"group": g, "index": i, "id": r["id"],
                                      "clean_identity": meta["clean_identity"],
                                      "original_clock": meta["original_clock"]})
            metas[g].append(meta)
            state["prepared"][g] += 1
    state["row"] = state["group"] = None
    state["stage"] = "preflight_gate"
    if identity_fail or input_fail:
        raise AuditError("preflight failed before any new scoring; scope decision needed",
                         {"identity_failures_n": {g: sum(f["group"] == g for f in identity_fail) for g in GROUPS},
                          "input_failures_n": {g: sum(f["group"] == g for f in input_fail) for g in GROUPS},
                          "identity_failures": identity_fail, "input_failures": input_fail})

    # 6. model
    state["stage"] = "model"
    torch.set_num_threads(FIXED_THREADS)
    payload = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    model, model_classes = u.rebuild_from_payload(payload)
    model.eval()
    if [str(c) for c in model_classes] != classes:
        raise AuditError("model class order differs from saved class order")

    def forward(x):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 280 or x.dtype != np.float32 \
                or not np.isfinite(x).all():
            raise AuditError(f"bad forward input {getattr(x, 'shape', None)}")
        bx, pad, _ = collate([{"x": x, "frames": int(len(x)), "label": 0}])
        with torch.inference_mode():
            lg = model(torch.from_numpy(bx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
        if lg.shape != (N_CLASSES,) or not np.isfinite(lg).all():
            raise AuditError(f"model output shape {lg.shape} or non-finite")
        return lg

    # 7. reproduce stored scores before new inference
    state["stage"] = "reproduction"
    repro = []
    for g in GROUPS:
        checks = [("original", saved_a[g])] + ([("sparse", b_native)] if g == "native" else [])
        for i in cfg["repro_indices"][g]:
            r = rows[g][i]
            state["group"], state["row"] = g, r["id"]
            x, _ = prepare_row(api, g, r, tol)
            for arm, saved in checks:
                lg = forward(x[arm])
                state["forwards_repro"] += 1
                diff = float(np.max(np.abs(lg - saved[i])))
                ok = diff <= tol and int(np.argmax(lg)) == int(np.argmax(saved[i]))
                repro.append({"group": g, "index": i, "id": r["id"], "arm": arm, "max_abs_diff": diff,
                              "top1_now": classes[int(np.argmax(lg))], "top1_saved": classes[int(np.argmax(saved[i]))],
                              "ok": bool(ok)})
    reproduction = {"indices": cfg["repro_indices"], "atol": tol, "rows": repro}
    if not all(x["ok"] for x in repro):
        raise AuditError("stored score reproduction failed; no new inference performed",
                         {"reproduction": reproduction})

    # 8. new inference, one forward per new arm per row; inputs must match the preflight hashes
    state["stage"] = "inference"
    logits, results = {}, {}
    for g in GROUPS:
        n = len(rows[g])
        lg = np.full((n, len(ARMS), N_CLASSES), np.nan, dtype=np.float64)
        lg[:, 0] = saved_a[g]
        if g == "native":
            lg[:, 1] = b_native
        out_rows = []
        for i, r in enumerate(rows[g]):
            state["group"], state["row"] = g, r["id"]
            x, meta = prepare_row(api, g, r, tol)
            if {a: sha256_array(v) for a, v in x.items()} != metas[g][i]["input_sha256"]:
                raise AuditError(f"{r['id']}: inputs differ from preflight")
            for arm in NEW_ARMS[g]:
                if arm == "endpoint_reference" and arm not in x:
                    lg[i, ARMS.index(arm)] = saved_a[g][i]
                    continue
                lg[i, ARMS.index(arm)] = forward(x[arm])
                state["forwards_new"] += 1
            meta["input_sha256"] = metas[g][i]["input_sha256"]
            out_rows.append({"index": i, "id": r["id"], "signer_id": r["signer_id"], "true": r["label"],
                             "true_index": classes.index(r["label"]), "recording_id": recording_id(r["id"]),
                             "arms": {a: score_row(lg[i, k], classes, r["label"]) for k, a in enumerate(ARMS)},
                             **meta})
        if not np.isfinite(lg).all():
            raise AuditError(f"{g}: non-finite logits in score tensor")
        logits[g], results[g] = lg, out_rows
    state["group"] = state["row"] = None
    expected_new = sum(len(NEW_ARMS[g]) - (0 if m["original_final_frame_dropped_by_slice"] else 1)
                       for g in GROUPS for m in metas[g])
    expected_repro = 2 * len(cfg["repro_indices"]["native"]) + len(cfg["repro_indices"]["mediapipe"])
    if state["forwards_new"] != expected_new or state["forwards_repro"] != expected_repro:
        raise AuditError("forward count mismatch")

    # 9. scores
    state["stage"] = "write_scores"
    scores_sha = {}
    for g in GROUPS:
        p = out_dir / f"SCORES_{g}.npz"
        with open(p, "xb") as fh:
            np.savez_compressed(fh, logits=logits[g], arms=np.array(ARMS), ids=np.array([r["id"] for r in results[g]]),
                                classes=np.array(classes),
                                true_indices=np.array([r["true_index"] for r in results[g]], dtype=np.int64))
        scores_sha[g] = sha256_file(p)

    # 10. hashes after
    state["stage"] = "hashes_after"
    pins_after = hash_pins(cfg["pins"])
    sources_after = {g: hash_sources(rows[g], roots[g]) for g in GROUPS}
    modules_after = {k: sha256_file(p) for k, p in module_files.items()}
    code_after = {"run_audit.py": sha256_file(__file__), "timegrid.py": sha256_file(TG.__file__)}
    config_sha = hashlib.sha256(cfg_bytes).hexdigest()
    config_after = sha256_file(cfg_path)
    changed = {"pins": [p for p in pins_before if pins_before[p] != pins_after[p]],
               "sources": {g: [k for k in sources_before[g] if sources_before[g][k] != sources_after[g][k]] for g in GROUPS},
               "modules": [k for k in modules_before if modules_before[k] != modules_after[k]],
               "code": code_after != code, "config": config_after != config_sha}
    write_json_exclusive(out_dir / "SOURCE_HASHES.json", {
        "pins": {"before": pins_before, "after": pins_after},
        "sources": {"before": sources_before, "after": sources_after},
        "imported_modules": {"files": module_files, "before": modules_before, "after": modules_after},
        "code": {"before": code, "after": code_after},
        "config": {"path": cfg_path, "before": config_sha, "after": config_after}, "changed": changed})
    if changed["pins"] or any(changed["sources"].values()) or changed["modules"] or changed["code"] or changed["config"]:
        raise AuditError("hashes changed during run", {"changed": changed})

    # 11. result
    state["stage"] = "write_result"
    groups_out = {}
    for g in GROUPS:
        rs = results[g]
        comps = {f"{s}->{t}": compare_arms(rs, s, t) for s, t in COMPARISONS + CONTEXT_COMPARISONS}
        identity_ok = sum(r["clean_identity"]["ok"] for r in rs)
        groups_out[g] = {
            "description": ("native provided-coordinate development rows (annotation_dev, validation)"
                            if g == "native" else "cached MediaPipe training-exposed diagnostic rows"),
            "n_rows": len(rs), "source_recordings": recording_counts(rs),
            "per_arm_correct": {a: sum(r["arms"][a]["correct"] for r in rs) for a in ARMS},
            "per_signer": breakdown(rs, "signer_id"), "per_class": breakdown(rs, "true"),
            "comparisons": {k: comps[k] for k in (f"{s}->{t}" for s, t in COMPARISONS)},
            "context_comparisons": {k: comps[k] for k in (f"{s}->{t}" for s, t in CONTEXT_COMPARISONS)},
            "original_correct_sparse_wrong_recoveries": recoveries(rs),
            "durations": {"rows_final_frame_dropped_by_slice": sum(r["original_final_frame_dropped_by_slice"] for r in rs),
                          "sum_original_duration_us": sum(r["durations_us"]["original"] for r in rs),
                          "sum_retained_duration_us": sum(r["durations_us"]["retained"] for r in rs),
                          "all_restored_end_at_last_retained": all(
                              r["endpoints_us"]["restored"] == r["endpoints_us"]["retained"] for r in rs)},
            "restoration_summary": {k: sum(r["restored"][k] for r in rs) for k in
                                    ("exact_frames", "interior_frames", "interior_frames_long_gap",
                                     "observed_slots", "estimated_slots", "unsupported_interior_slots")},
            "original_clock_rows_nominal_30hz": sum(r["original_clock"]["nominal_30hz"] for r in rs),
            "normalization": normalization_summary(rs),
            "outcome_checks": outcome_checks(g, rs, comps, identity_ok, 0),
            "rows": rs,
        }
    result = {
        "status": "COMPLETE_PENDING_CODEX_VERIFICATION",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "nature": "time-grid restoration diagnostic on a frozen model; no training, no parameter search",
        "arm_order": list(ARMS), "primary": "linear", "mechanism_control": "previous",
        "timegrid_rules": {"nominal_step_us": TG.NOMINAL_STEP_US, "exact_tol_us": TG.EXACT_TOL_US,
                           "clock_tol_us": TG.CLOCK_TOL_US, "max_bracket_gap": "2P + 2 us"},
        "model": {"checkpoint": cfg["checkpoint"], "sha256": pin_of[norm_path(cfg["checkpoint"])],
                  "classes": len(classes)},
        "saved_mp_arms": mp_arms, "reproduction": reproduction,
        "groups": groups_out, "interpretation_limits": LIMITS,
        "execution_counts": {"forwards_repro": state["forwards_repro"], "forwards_new": state["forwards_new"],
                             "forwards_total": state["forwards_repro"] + state["forwards_new"],
                             "training": 0, "camera": 0, "user_data": 0, "downloads": 0},
        "hashes": {"code": code, "config": config_sha, "plan_file": pin_of[norm_path(cfg["plan_file"])],
                   "scores": scores_sha, "imported_modules": modules_before},
        "seconds": round(time.time() - started, 2),
    }
    write_json_exclusive(out_dir / "RESULT.json", result)
    print(json.dumps({g: groups_out[g]["per_arm_correct"] for g in GROUPS}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuditError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
