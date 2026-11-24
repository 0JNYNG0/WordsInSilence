#!/usr/bin/env python3
"""pose_roi_probe.py - Stage-1 pose-guided ROI re-detection observability probe (diagnostic only).

Usage:
  python pose_roi_probe.py --config PATH --out NEW_DIRECTORY
  python pose_roi_probe.py --self-test        (synthetic data + mock backend/decoder only)

A = stored VIDEO observation (read only, never re-run). B = one full-frame RGB IMAGE call.
C = one pose-guided rotated-crop RGB IMAGE call. B and C use separate HandLandmarker IMAGE
instances (num_hands=2, detection/presence/tracking 0.5). Rules follow FROZEN_PLAN.md exactly;
there are no tuning knobs. No coordinates are merged or promoted. 'potential' is an automatic
hypothesis for later visual review, never a correctness label. Pose validity is heuristic only.
"""
from __future__ import annotations

import argparse
import codecs
import copy
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback
from collections import Counter
from fractions import Fraction

import numpy as np

SCHEMA = "pose_roi_probe/v1"
SRC = os.path.abspath(__file__)
EXPECTED_CLIPS = 6
N_POINTS = 75
ROLES = ("failure", "control")
SIDES = ("L", "R")
OTHER = {"L": "R", "R": "L"}
HAND_SLICE = {"L": (33, 54), "R": (54, 75)}
POSE_WRIST = {"L": 15, "R": 16}
POSE_REFS = {"L": (15, 19, 17), "R": (16, 20, 18)}  # wrist, index, pinky (plan step 2)
# Frozen plan step 3/4 constants - not adjustable.
ROI_CENTER_SHIFT = 0.2
ROI_SIDE_SCALE = 5.4
ROI_MIN_SIDE_PX = 2.0
ROI_MAX_N_FACTOR = 2
# Frozen B/C IMAGE options and potential rule.
NUM_HANDS = 2
MIN_CONF = 0.5
BOUNDS_MARGIN = 0.05
HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))
SKIP_REASONS = ("outside_active_span", "pose_wrist_invalid", "partial_hand_observation",
                "both_hands_observed", "no_hand_observed", "occupied_hand_not_all_usable_finite",
                "empty_side_usable_without_observed", "empty_side_pose_refs_invalid",
                "roi_nonfinite", "roi_degenerate_zero_axis", "roi_side_below_2px", "roi_oversized")
FROZEN_RULES = {
    "roi": "c=(2*index+pinky)/3, v=c-wrist, d=|v|, center=c+0.2*v, side=5.4*d, N=ceil(side)",
    "roi_skip": "d==0 or nonfinite -> skip; side<2px -> skip; N>2*max(H,W) -> skip",
    "crop_to_source": "p=center+right*(x-q)+down*(y-q), q=(N-1)/2, up=v/d, right=(-up_y,up_x), down=-up",
    "warp": "cv2.warpAffine WARP_INVERSE_MAP|INTER_LINEAR, BORDER_CONSTANT RGB 0",
    "output_mapping": "B: (nx*W, ny*H); C: (nx*N, ny*N) then crop_to_source; no clipping",
    "potential": "all 21 finite, within [-0.05,1.05] of W/H, wrist strictly closer to empty-side pose "
                 "wrist than occupied-side pose wrist (ties excluded); exactly one qualifying -> index",
    "image_options": {"num_hands": NUM_HANDS, "min_hand_detection_confidence": MIN_CONF,
                      "min_hand_presence_confidence": MIN_CONF, "min_tracking_confidence": MIN_CONF},
    "calls_per_eligible_row": {"B": 1, "C": 1}, "retries": 0,
}
NOTES = [
    "Stage 1 only: raw B/C outputs are stored separately; nothing is merged into observations.",
    "potential is an automatic hypothesis for visual review, not a correctness/sign label.",
    "handedness label/score is metadata only and never used for selection.",
    "sidecar unassigned totals are logged only and never used to exclude rows.",
    "Eligible rows imply stored VIDEO returned exactly one raw hand under the pinned code (plan correction); "
    "this is code inference, not raw-log evidence.",
    "Pose references may be wrong; ROI validity is heuristic.",
]
HEX64 = re.compile(r"[0-9a-f]{64}")
CLIP_ID = re.compile(r"[A-Za-z0-9._-]{1,80}")
CONFIG_KEYS = ("schema", "plan_path", "plan_sha256", "hand_model_path", "hand_model_sha256",
               "code_pins", "clips")
CLIP_KEYS = ("id", "video", "video_sha256", "observation", "observation_sha256", "sidecar",
             "sidecar_sha256", "active_span", "role")
OBS_KEYS = ("xy_px", "observed", "usable", "timestamp_us", "frame_id", "hand_score", "conf")


class ProbeError(Exception):
    """Fail-closed validation/technical error."""


def need(cond, msg):
    if not cond:
        raise ProbeError(msg)


# ----------------------------------------------------------------------------- JSON / hashing
def to_jsonable(obj):
    """Recursively convert numpy types; NaN/inf -> None. Result passes allow_nan=False."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Fraction):
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    raise TypeError(f"not JSON-serialisable: {type(obj)!r}")


def dumps_strict(obj, compact=False):
    return json.dumps(to_jsonable(obj), allow_nan=False, ensure_ascii=True,
                      separators=(",", ":") if compact else None, indent=None if compact else 1)


def write_json_new(path, obj):
    with open(path, "x", encoding="utf-8") as f:  # 'x': never overwrite
        f.write(dumps_strict(obj))


def write_bytes_new(path, data):
    with open(path, "xb") as f:
        f.write(data)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rgb_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _no_dup_pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ProbeError(f"duplicate JSON key {k!r}")
        d[k] = v
    return d


def load_json_bytes(data, what):
    """UTF-8 (optional BOM) JSON; rejects NaN/Infinity constants and duplicate keys."""
    had_bom = data.startswith(codecs.BOM_UTF8)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ProbeError(f"{what}: not UTF-8: {e}") from e

    def bad_constant(name):
        raise ProbeError(f"{what}: non-standard JSON constant {name}")

    try:
        obj = json.loads(text, object_pairs_hook=_no_dup_pairs, parse_constant=bad_constant)
    except json.JSONDecodeError as e:
        raise ProbeError(f"{what}: invalid JSON: {e}") from e
    return obj, had_bom


# ----------------------------------------------------------------------------- config / pins
def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _hex(v, what):
    need(isinstance(v, str) and HEX64.fullmatch(v.lower()) is not None, f"{what}: not a sha256 hex digest")
    return v.lower()


def _abs(v, what, ext=None):
    need(isinstance(v, str) and v and os.path.isabs(v), f"{what}: absolute path required, got {v!r}")
    if ext:
        need(v.lower().endswith(ext), f"{what}: must end with {ext}")
    return v


def validate_config(cfg):
    need(isinstance(cfg, dict), "config must be a JSON object")
    need(set(cfg) == set(CONFIG_KEYS), f"config keys must be exactly {list(CONFIG_KEYS)}; missing "
         f"{sorted(set(CONFIG_KEYS) - set(cfg))}, unknown {sorted(set(cfg) - set(CONFIG_KEYS))}")
    need(cfg["schema"] == SCHEMA, f"schema {cfg['schema']!r} != {SCHEMA!r}")
    out = {"plan_path": _abs(cfg["plan_path"], "plan_path"),
           "plan_sha256": _hex(cfg["plan_sha256"], "plan_sha256"),
           "hand_model_path": _abs(cfg["hand_model_path"], "hand_model_path", ".task"),
           "hand_model_sha256": _hex(cfg["hand_model_sha256"], "hand_model_sha256")}
    pins = cfg["code_pins"]
    need(isinstance(pins, dict) and pins, "code_pins must be a non-empty object")
    out["code_pins"] = {_abs(p, "code_pins key"): _hex(s, f"code_pins[{p}]") for p, s in pins.items()}
    clips = cfg["clips"]
    need(isinstance(clips, list) and len(clips) == EXPECTED_CLIPS,
         f"clips must be a list of exactly {EXPECTED_CLIPS}")
    seen, vc = set(), []
    for i, c in enumerate(clips):
        need(isinstance(c, dict) and set(c) == set(CLIP_KEYS),
             f"clip[{i}] keys must be exactly {list(CLIP_KEYS)}")
        cid = c["id"]
        need(isinstance(cid, str) and CLIP_ID.fullmatch(cid) is not None, f"clip[{i}] id must be [A-Za-z0-9._-]{{1,80}}")
        need(cid.lower() not in seen, f"duplicate clip id {cid!r}")
        seen.add(cid.lower())
        span = c["active_span"]
        need(isinstance(span, list) and len(span) == 2 and all(_is_int(v) for v in span)
             and 0 <= span[0] <= span[1], f"{cid}: active_span must be [start,end] ints, 0<=start<=end")
        need(c["role"] in ROLES, f"{cid}: role must be one of {ROLES}")
        vc.append({"id": cid,
                   "video": _abs(c["video"], f"{cid}.video", ".mp4"),
                   "video_sha256": _hex(c["video_sha256"], f"{cid}.video_sha256"),
                   "observation": _abs(c["observation"], f"{cid}.observation", ".npz"),
                   "observation_sha256": _hex(c["observation_sha256"], f"{cid}.observation_sha256"),
                   "sidecar": _abs(c["sidecar"], f"{cid}.sidecar", ".json"),
                   "sidecar_sha256": _hex(c["sidecar_sha256"], f"{cid}.sidecar_sha256"),
                   "active_span": [int(span[0]), int(span[1])], "role": c["role"]})
    out["clips"] = vc
    return out


def pin_targets(cfg):
    t = [("plan", cfg["plan_path"], cfg["plan_sha256"]),
         ("hand_model", cfg["hand_model_path"], cfg["hand_model_sha256"])]
    t += [(f"code_pin:{i}", p, s) for i, (p, s) in enumerate(sorted(cfg["code_pins"].items()))]
    for c in cfg["clips"]:
        for k in ("video", "observation", "sidecar"):
            t.append((f"clip:{c['id']}:{k}", c[k], c[f"{k}_sha256"]))
    return t


def _keep_bytes(label):
    return label == "hand_model" or label.endswith(":observation") or label.endswith(":sidecar")


def hash_targets(targets, keep=False):
    """Hash every pinned input (read-only). With keep=True, return the hashed bytes for parsing."""
    checks, data = [], {}
    for label, path, exp in targets:
        rec = {"label": label, "path": path, "expected": exp, "actual": None, "ok": False}
        try:
            if keep and _keep_bytes(label):
                with open(path, "rb") as f:
                    b = f.read()
                rec["actual"] = sha256_bytes(b)
                data[label] = b
            else:
                rec["actual"] = sha256_file(path)
        except OSError as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["ok"] = rec["actual"] == exp
        checks.append(rec)
    return checks, data


# ----------------------------------------------------------------------------- observation / sidecar
def _flag_array(a, name, shape):
    need(a.shape == shape, f"{name} must have shape {shape}, got {a.shape}")
    if a.dtype == np.bool_:
        return a.copy()
    need(np.issubdtype(a.dtype, np.integer) and np.isin(a, (0, 1)).all(), f"{name} must be bool or 0/1 ints")
    return a.astype(bool)


def load_observation(data):
    """Load npz from already-hashed bytes (allow_pickle=False). Returned arrays are read-only copies."""
    try:
        z = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as e:
        raise ProbeError(f"observation is not a loadable npz: {e}") from e
    need(isinstance(z, np.lib.npyio.NpzFile), "observation is not an npz archive")
    with z:
        missing = [k for k in OBS_KEYS if k not in z.files]
        need(not missing, f"observation missing arrays {missing} (frame_id is required for ordinal alignment)")
        raw = {k: np.array(z[k], copy=True) for k in OBS_KEYS}
    xy = raw["xy_px"]
    need(xy.ndim == 3 and xy.shape[0] >= 1 and xy.shape[1:] == (N_POINTS, 2),
         f"xy_px must be [T,{N_POINTS},2], got {xy.shape}")
    need(np.issubdtype(xy.dtype, np.floating), "xy_px must be floating")
    T = xy.shape[0]
    observed = _flag_array(raw["observed"], "observed", (T, N_POINTS))
    usable = _flag_array(raw["usable"], "usable", (T, N_POINTS))
    ts, fid = raw["timestamp_us"], raw["frame_id"]
    need(ts.shape == (T,) and np.issubdtype(ts.dtype, np.integer), "timestamp_us must be integer [T]")
    need(T == 1 or bool(np.all(np.diff(ts.astype(np.int64)) > 0)), "timestamp_us must be strictly increasing")
    need(fid.shape == (T,) and np.issubdtype(fid.dtype, np.integer), "frame_id must be integer [T]")
    need(int(fid.min()) >= 0 and (T == 1 or bool(np.all(np.diff(fid.astype(np.int64)) > 0))),
         "frame_id must be non-negative and strictly increasing")
    for k, shape in (("hand_score", (T, 2)), ("conf", (T, N_POINTS))):
        need(raw[k].shape == shape and np.issubdtype(raw[k].dtype, np.floating), f"{k} must be floating {shape}")
    obs = {"xy": xy.astype(np.float64), "observed": observed, "usable": usable,
           "ts": ts.astype(np.int64), "frame_id": fid.astype(np.int64),
           "hand_score": raw["hand_score"].copy(), "conf": raw["conf"].copy()}
    for a in obs.values():
        a.setflags(write=False)
    return obs


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}/{k}"
            yield p, str(k), v
            yield from _walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def _num(v):
    return (_is_int(v) or isinstance(v, float)) and not isinstance(v, bool)


def summarize_sidecar(obj):
    """Generic scan (sidecar schema is not specified): log unassigned counts, require any num_hands==2,
    and fail on any key containing 'drop' that reports a nonzero/true/non-empty value."""
    need(isinstance(obj, dict), "sidecar must be a JSON object")
    unassigned, num_hands, drops = [], [], []
    for p, k, v in _walk(obj):
        lk = k.lower()
        if "unassigned" in lk:
            if _num(v):
                unassigned.append({"path": p, "value": v})
            elif isinstance(v, list) and all(_num(x) for x in v):
                unassigned.append({"path": p, "list_len": len(v), "list_sum": sum(v)})
        if lk == "num_hands":
            num_hands.append({"path": p, "value": v})
        if "drop" in lk and not isinstance(v, dict):
            bad = (v is True or (_num(v) and v != 0) or (isinstance(v, list) and len(v) > 0))
            drops.append({"path": p, "value": v if not isinstance(v, list) else {"list_len": len(v)},
                          "reports_drop": bool(bad)})
    bad_nh = [e for e in num_hands if not (_is_int(e["value"]) and e["value"] == NUM_HANDS)]
    need(not bad_nh, f"sidecar num_hands != {NUM_HANDS}: {bad_nh}")
    bad_drop = [e for e in drops if e["reports_drop"]]
    need(not bad_drop, f"sidecar reports dropped frames/PTS: {bad_drop}")
    any_nonzero = any(e.get("value", e.get("list_sum", 0)) for e in unassigned)
    return {"top_level_keys": sorted(obj), "unassigned_entries": unassigned,
            "unassigned_any_nonzero": bool(any_nonzero), "num_hands_entries": num_hands,
            "num_hands_source": "sidecar" if num_hands else "code_pins_only", "drop_entries": drops,
            "note": "unassigned values are logged only; never used to exclude rows"}


# ----------------------------------------------------------------------------- decoding
def pts_to_us(pts, time_base):
    """Exact rational PTS -> integer microseconds (Python round: half-to-even)."""
    return round(Fraction(pts) * Fraction(time_base) * 1_000_000)


def pyav_decode(video_path):
    """Real file decoder (PyAV). Yields one record per decoded frame; frames without PTS or with
    non-increasing rounded-us time are reported as ignored. Origin = first kept PTS."""
    import av
    with av.open(video_path, mode="r") as c:
        st = c.streams.video[0]
        tb = st.time_base
        need(tb is not None, f"{video_path}: video stream has no time_base")
        last = origin = None
        ordinal = -1
        for fr in c.decode(st):
            if fr.pts is None:
                yield {"ignored": "pts_none"}
                continue
            abs_us = pts_to_us(fr.pts, tb)
            if last is not None and abs_us <= last:
                yield {"ignored": "non_increasing_pts"}
                continue
            last = abs_us
            if origin is None:
                origin = abs_us
            ordinal += 1
            yield {"ignored": None, "ordinal": ordinal, "pts": int(fr.pts), "time_base": str(tb),
                   "abs_us": abs_us, "rel_us": abs_us - origin, "width": int(fr.width),
                   "height": int(fr.height),
                   "to_rgb": (lambda f=fr: f.to_ndarray(format="rgb24"))}


def probe_alignment(decoder, video, ts, fid):
    """Pre-model pass: every stored row must match decoded ordinal==frame_id and rel_us==timestamp_us."""
    kept, ignored, dims, tbs = {}, Counter(), set(), set()
    gen = decoder(video)
    try:
        for rec in gen:
            if rec["ignored"]:
                ignored[rec["ignored"]] += 1
                continue
            kept[rec["ordinal"]] = (rec["rel_us"], rec["pts"], rec["abs_us"])
            dims.add((rec["width"], rec["height"]))
            tbs.add(rec["time_base"])
    finally:
        gen.close()
    need(kept, f"{video}: no decodable frames")
    need(not ignored, f"{video}: decoder ignored frames (PTS missing/non-increasing): {dict(ignored)}")
    need(len(dims) == 1, f"{video}: frame size varies: {sorted(dims)}")
    need(len(tbs) == 1, f"{video}: time_base varies")
    bad = []
    for r in range(len(ts)):
        o = int(fid[r])
        if o not in kept:
            bad.append({"row": r, "frame_id": o, "problem": "no decoded ordinal"})
        elif kept[o][0] != int(ts[r]):
            bad.append({"row": r, "frame_id": o, "problem": "rel_us mismatch",
                        "decoded_rel_us": kept[o][0], "stored_us": int(ts[r])})
    need(not bad, f"{video}: {len(bad)} rows misaligned with decode, first: {bad[:5]}")
    W, H = dims.pop()
    return {"width": W, "height": H, "time_base": tbs.pop(), "decoded_frames": len(kept),
            "ignored_frames": 0, "rows_aligned": len(ts),
            "frame_id_gaps": int(np.sum(np.diff(fid) > 1)) if len(fid) > 1 else 0,
            "decoded_frames_not_in_observation": len(kept) - len(ts)}


def iter_rows(decoder, video, rows, ts, fid, shape_hw):
    """Yield exactly the requested rows in order; raise on any mismatch/missing frame (no nearest frame)."""
    k = 0
    gen = decoder(video)
    try:
        for rec in gen:
            if k == len(rows):
                break
            need(not rec["ignored"], f"{video}: decoder ignored a frame ({rec['ignored']}) during row pass")
            r = rows[k]
            want = int(fid[r])
            if rec["ordinal"] < want:
                continue
            need(rec["ordinal"] == want, f"row {r}: no decoded frame with ordinal {want}")
            need(rec["rel_us"] == int(ts[r]), f"row {r}: decoded rel_us {rec['rel_us']} != stored {int(ts[r])}")
            t0 = time.perf_counter_ns()
            rgb = rec["to_rgb"]()
            dt = time.perf_counter_ns() - t0
            need(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8
                 and rgb.shape == (shape_hw[0], shape_hw[1], 3), f"row {r}: decoded RGB has unexpected shape/dtype")
            info = {key: rec[key] for key in ("ordinal", "pts", "time_base", "abs_us", "rel_us")}
            info["to_rgb_ns"] = dt
            yield r, rgb, info
            k += 1
    finally:
        gen.close()
    need(k == len(rows), f"{video}: stream ended before row {rows[k] if k < len(rows) else None}")


# ----------------------------------------------------------------------------- ROI geometry
def crop_to_src(M, pts):
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ M[:, :2].T + M[:, 2]


def src_to_crop(M, pts):
    pts = np.asarray(pts, dtype=np.float64)
    return (pts - M[:, 2]) @ np.linalg.inv(M[:, :2]).T


def compute_roi(wrist, index, pinky, W, H):
    """Frozen plan step 3/4. Returns a record with ok/skip_reason; never mutates inputs."""
    w = np.array(wrist, dtype=np.float64)
    i = np.array(index, dtype=np.float64)
    p = np.array(pinky, dtype=np.float64)
    rec = {"wrist_px": w, "index_px": i, "pinky_px": p}

    def skip(reason):
        rec.update(ok=False, skip_reason=reason)
        return rec

    if not (np.isfinite(w).all() and np.isfinite(i).all() and np.isfinite(p).all()):
        return skip("roi_nonfinite")
    with np.errstate(over="ignore", invalid="ignore"):  # overflow is reported as roi_nonfinite
        c = (2.0 * i + p) / 3.0
        v = c - w
    rec.update(c_px=c, v_px=v)
    if not np.isfinite(v).all():
        return skip("roi_nonfinite")
    d = math.hypot(float(v[0]), float(v[1]))
    rec["d_px"] = d
    if not math.isfinite(d):
        return skip("roi_nonfinite")
    if d == 0.0:
        return skip("roi_degenerate_zero_axis")
    center = c + ROI_CENTER_SHIFT * v
    side = ROI_SIDE_SCALE * d
    rec.update(center_px=center, side_px=side)
    if not (math.isfinite(side) and np.isfinite(center).all()):
        return skip("roi_nonfinite")
    if side < ROI_MIN_SIDE_PX:
        return skip("roi_side_below_2px")
    n = math.ceil(side)
    max_n = ROI_MAX_N_FACTOR * max(int(W), int(H))
    rec.update(N=n, max_N=max_n, ceil_extra_px=n - side)
    if n > max_n:
        return skip("roi_oversized")
    up = v / d
    right = np.array([-up[1], up[0]])
    down = -up
    q = (n - 1) / 2.0
    M = np.array([[right[0], down[0], center[0] - q * (right[0] + down[0])],
                  [right[1], down[1], center[1] - q * (right[1] + down[1])]], dtype=np.float64)
    corners = crop_to_src(M, np.array([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]], dtype=np.float64))
    inside = bool((corners[:, 0] >= 0).all() and (corners[:, 0] <= W - 1).all()
                  and (corners[:, 1] >= 0).all() and (corners[:, 1] <= H - 1).all())
    rec.update(ok=True, skip_reason=None, q=q, up=up, right=right, down=down, crop_to_src_affine=M,
               corners_src_px_tl_tr_br_bl=corners, fully_inside_source=inside,
               black_padding_present=not inside)
    return rec


def warp_crop(img, M, n):
    import cv2
    return cv2.warpAffine(img, np.asarray(M, dtype=np.float64), (int(n), int(n)),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


# ----------------------------------------------------------------------------- eligibility
def evaluate_rows(xy, observed, usable, span, W, H):
    """Deterministic per-row eligibility (frozen plan step 1-4). Inputs are only read."""
    xy = np.asarray(xy)
    observed = np.asarray(observed, dtype=bool)
    usable = np.asarray(usable, dtype=bool)
    pv = observed & usable & np.isfinite(xy).all(axis=-1)
    lo, hi = span
    return [_evaluate_row(r, lo, hi, xy[r], observed[r], usable[r], pv[r], W, H) for r in range(xy.shape[0])]


def _evaluate_row(r, lo, hi, xy, obs, use, pv, W, H):
    rec = {"row": r, "eligible": False, "skip_reason": None, "empty_side": None, "occupied_side": None}

    def skip(reason):
        rec["skip_reason"] = reason
        return rec

    if not lo <= r <= hi:
        return skip("outside_active_span")
    if not (pv[POSE_WRIST["L"]] and pv[POSE_WRIST["R"]]):
        return skip("pose_wrist_invalid")
    full = {s: bool(obs[a:b].all()) for s, (a, b) in HAND_SLICE.items()}
    none = {s: not bool(obs[a:b].any()) for s, (a, b) in HAND_SLICE.items()}
    if any(not (full[s] or none[s]) for s in SIDES):
        return skip("partial_hand_observation")
    if full["L"] and full["R"]:
        return skip("both_hands_observed")
    if none["L"] and none["R"]:
        return skip("no_hand_observed")
    occ = "L" if full["L"] else "R"
    emp = OTHER[occ]
    rec.update(occupied_side=occ, empty_side=emp)
    a, b = HAND_SLICE[occ]
    if not pv[a:b].all():
        return skip("occupied_hand_not_all_usable_finite")
    a, b = HAND_SLICE[emp]
    if use[a:b].any():
        return skip("empty_side_usable_without_observed")
    refs = POSE_REFS[emp]
    if not all(pv[k] for k in refs):
        return skip("empty_side_pose_refs_invalid")
    roi = compute_roi(xy[refs[0]], xy[refs[1]], xy[refs[2]], W, H)
    rec["roi"] = roi
    if not roi["ok"]:
        return skip(roi["skip_reason"])
    rec["eligible"] = True
    rec["pose_wrist_px"] = {"empty": np.array(xy[POSE_WRIST[emp]], dtype=np.float64),
                            "occupied": np.array(xy[POSE_WRIST[occ]], dtype=np.float64)}
    rec["A_raw_hand_count_inferred"] = 1
    return rec


# ----------------------------------------------------------------------------- outputs / potential
def full_norm_to_src(lm, W, H):
    return np.asarray(lm, dtype=np.float64)[:, :2] * np.array([W, H], dtype=np.float64)


def crop_norm_to_src(lm, n, M):
    return crop_to_src(M, np.asarray(lm, dtype=np.float64)[:, :2] * float(n))


def convert_output(raw, tag, W, H, roi):
    """Keep every returned raw hand; map its normalized landmarks to original-image pixels."""
    need(isinstance(raw, list), f"{tag} backend must return a list")
    hands, src = [], []
    for i, h in enumerate(raw):
        need(isinstance(h, dict) and isinstance(h.get("landmarks_norm"), list), f"{tag} raw hand {i} malformed")
        lm = np.array(h["landmarks_norm"], dtype=np.float64)
        px = None
        if lm.ndim == 2 and lm.shape[0] > 0 and lm.shape[1] >= 2:
            px = full_norm_to_src(lm, W, H) if tag == "B" else crop_norm_to_src(lm, roi["N"], roi["crop_to_src_affine"])
        hands.append({"raw_index": i, "landmarks_norm": lm, "landmarks_src_px": px,
                      "handedness_metadata_only": copy.deepcopy(h.get("handedness", []))})
        src.append(px)
    return hands, src


def select_potential(src_list, empty_wrist, occupied_wrist, W, H):
    """Per-hand flags + exactly-one rule. Two or more qualifying hands stay ambiguous (no winner)."""
    ew = np.asarray(empty_wrist, dtype=np.float64)
    ow = np.asarray(occupied_wrist, dtype=np.float64)
    lo_x, hi_x = -BOUNDS_MARGIN * W, (1.0 + BOUNDS_MARGIN) * W
    lo_y, hi_y = -BOUNDS_MARGIN * H, (1.0 + BOUNDS_MARGIN) * H
    flags = []
    for pts in src_list:
        causes = []
        n21 = pts is not None and np.asarray(pts).shape == (21, 2)
        if not n21:
            causes.append("not_21_landmarks")
        finite = n21 and bool(np.isfinite(pts).all())
        if n21 and not finite:
            causes.append("nonfinite_landmark")
        within = finite and bool((pts[:, 0] >= lo_x).all() and (pts[:, 0] <= hi_x).all()
                                 and (pts[:, 1] >= lo_y).all() and (pts[:, 1] <= hi_y).all())
        if finite and not within:
            causes.append("outside_5pct_frame_margin")
        de = do = None
        closer = tie = False
        if finite:
            de = math.dist(pts[0], ew)
            do = math.dist(pts[0], ow)
            closer, tie = de < do, de == do
            if tie:
                causes.append("distance_tie")
            elif not closer:
                causes.append("not_closer_to_empty_side_pose_wrist")
        flags.append({"flags": {"has_21": n21, "all_finite": finite, "within_5pct_margin": within,
                                "wrist_dist_to_empty_pose_wrist_px": de,
                                "wrist_dist_to_occupied_pose_wrist_px": do,
                                "strictly_closer_to_empty": closer, "distance_tie": tie},
                      "qualifies": bool(n21 and finite and within and closer), "rejection_causes": causes})
    qi = [i for i, f in enumerate(flags) if f["qualifies"]]
    status = "none" if not qi else ("single" if len(qi) == 1 else "ambiguous_multiple")
    return flags, {"index": qi[0] if len(qi) == 1 else None, "status": status, "qualifying_indices": qi}


# ----------------------------------------------------------------------------- real backend
class _MPImageDetector:
    def __init__(self, task, mp):
        self._task, self._mp = task, mp

    def detect(self, rgb):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._task.detect(image)
        hl, hd = res.hand_landmarks or [], res.handedness or []
        return [{"landmarks_norm": [[float(p.x), float(p.y), float(p.z)] for p in hl[i]],
                 "handedness": [{"label": c.category_name, "score": float(c.score)}
                                for c in (hd[i] if i < len(hd) else [])]}
                for i in range(len(hl))]

    def close(self):
        self._task.close()


def mediapipe_backend(model_bytes, model_path):
    """Real IMAGE backend. MediaPipe is imported only here (never during --self-test).
    The model is passed as the exact bytes that were hashed in preflight."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    def create(tag):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=mp_vision.RunningMode.IMAGE, num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_CONF, min_hand_presence_confidence=MIN_CONF,
            min_tracking_confidence=MIN_CONF)
        return _MPImageDetector(mp_vision.HandLandmarker.create_from_options(opts), mp)

    info = {"kind": "mediapipe_tasks_HandLandmarker_IMAGE", "mediapipe_version": getattr(mp, "__version__", None),
            "model_path": model_path, "model_sha256_of_buffer": sha256_bytes(model_bytes),
            "options": FROZEN_RULES["image_options"], "video_tasks_instantiated": 0}
    return create, info


# ----------------------------------------------------------------------------- rendering
PANEL_FULL = (480, 270)
PANEL_ROI = 270
GAP = 4
ROW_LABEL_H = 32
CAPTION_H = 30
HEADER_H = 48
ROWS_PER_PAGE = 4
SHEET_W = GAP + PANEL_FULL[0] + 3 * (GAP + PANEL_ROI) + GAP
ROW_H = ROW_LABEL_H + PANEL_ROI + CAPTION_H
COL = {"L": (0, 255, 255), "R": (255, 165, 0), "pose": (0, 200, 0), "roi": (255, 255, 255)}
RAW_COLORS = ((255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 80, 80))


def letterbox(rgb, size):
    from PIL import Image
    H, W = rgb.shape[:2]
    pw, ph = size
    s = min(pw / W, ph / H)
    nw, nh = max(1, round(W * s)), max(1, round(H * s))
    ox, oy = (pw - nw) // 2, (ph - nh) // 2
    img = Image.new("RGB", size, (0, 0, 0))
    img.paste(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR), (ox, oy))
    return img, (nw / W, nh / H, ox, oy)


def _draw_hand(d, pts, ok, color, width, r, label, font):
    pts = np.asarray(pts, dtype=np.float64)
    ok = np.asarray(ok, dtype=bool) & np.isfinite(pts).all(axis=1) & (np.abs(pts) < 1e5).all(axis=1)
    if pts.shape[0] == 21:
        for a, b in HAND_EDGES:
            if ok[a] and ok[b]:
                d.line([tuple(pts[a]), tuple(pts[b])], fill=color, width=width)
    for i in range(pts.shape[0]):
        if ok[i]:
            x, y = pts[i]
            d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=color)
    if pts.shape[0] and ok[0]:
        x, y = pts[0]
        d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=width)
        if label:
            d.text((x + r + 2, y - r - 10), label, fill=color, font=font)


def _pot_text(p):
    if p["index"] is not None:
        return f"#{p['index']}"
    return "ambiguous" if p["status"] == "ambiguous_multiple" else "none"


def _hand_list(res):
    if not res["hands"]:
        return "no raw hands"
    return " ".join(f"#{h['raw_index']}{'(P)' if h['is_potential'] else ''}" for h in res["hands"]) + \
        f" [{res['potential']['status']}]"


def make_row_image(rgb, crop, rec, xy_row, pv_row, clip_id, font):
    import cv2
    from PIL import Image, ImageDraw
    roi = rec["roi"]
    n, M, S = roi["N"], roi["crop_to_src_affine"], PANEL_ROI
    row = Image.new("RGB", (SHEET_W, ROW_H), (25, 25, 25))
    d = ImageDraw.Draw(row)
    lines = [f"{clip_id} | row {rec['row']} | frame_id {rec['frame_id']} | t={rec['timestamp_us']}us | "
             f"empty side {rec['empty_side']} (stored hand on {rec['occupied_side']})",
             f"ROI N={n}px side={roi['side_px']:.2f}px"
             f"{'' if roi['fully_inside_source'] else ' | ROI exceeds frame (black pad)'} | "
             f"B raw={rec['B']['raw_count']} potential={_pot_text(rec['B']['potential'])} | "
             f"C raw={rec['C']['raw_count']} potential={_pot_text(rec['C']['potential'])}"]
    for j, t in enumerate(lines):
        d.text((GAP, 3 + 14 * j), t, fill=(255, 255, 255), font=font)
    y0 = ROW_LABEL_H
    full, (sx, sy, ox, oy) = letterbox(rgb, PANEL_FULL)
    fd = ImageDraw.Draw(full)

    def tf(p):
        return (float(p[0]) * sx + ox, float(p[1]) * sy + oy)

    for side, (a, b) in HAND_SLICE.items():  # stored baseline A only
        _draw_hand(fd, [tf(p) for p in xy_row[a:b]], pv_row[a:b], COL[side], 1, 2, None, font)
    for k in (POSE_WRIST["L"], POSE_WRIST["R"]) + POSE_REFS[rec["empty_side"]][1:]:
        if pv_row[k]:
            x, y = tf(xy_row[k])
            fd.ellipse([x - 2, y - 2, x + 2, y + 2], fill=COL["pose"])
    cs = [tf(p) for p in roi["corners_src_px_tl_tr_br_bl"]]
    fd.polygon(cs, outline=COL["roi"])
    fd.line([cs[0], cs[1]], fill=(255, 255, 0), width=2)
    base = Image.fromarray(cv2.resize(crop, (S, S), interpolation=cv2.INTER_LINEAR))
    panels = [full, base]
    for tag in ("B", "C"):
        im = base.copy()
        dd = ImageDraw.Draw(im)
        for h in rec[tag]["hands"]:
            if h["landmarks_src_px"] is None:
                continue
            pts = src_to_crop(M, h["landmarks_src_px"]) * (S / n)  # same inverse mapping for B and C
            pot = h["is_potential"]
            _draw_hand(dd, pts, np.ones(len(pts), bool), RAW_COLORS[h["raw_index"] % len(RAW_COLORS)],
                       3 if pot else 1, 5 if pot else 3, f"{tag}#{h['raw_index']}{' P' if pot else ''}", font)
        panels.append(im)
    captions = [["A stored VIDEO baseline + ROI outline", "(yellow edge = crop top)"],
                ["ROI unmarked (exact C input)", ""],
                ["ROI + B full-IMAGE raw (inverse-mapped)", _hand_list(rec["B"])],
                ["ROI + C crop-IMAGE raw", _hand_list(rec["C"])]]
    x = GAP
    for im, cap in zip(panels, captions):
        row.paste(im, (x, y0))
        for j, t in enumerate(cap):
            d.text((x, y0 + S + 3 + 13 * j), t, fill=(220, 220, 220), font=font)
        x += im.size[0] + GAP
    return row


class SheetWriter:
    def __init__(self, out_dir, clip_id, n_eligible, skip_counts, font):
        self.rel_dir = os.path.join("sheets", clip_id)
        self.abs_dir = os.path.join(out_dir, self.rel_dir)
        os.makedirs(self.abs_dir)
        self.clip_id, self.font, self.skip_counts = clip_id, font, skip_counts
        self.n_eligible, self.n_pages = n_eligible, math.ceil(n_eligible / ROWS_PER_PAGE)
        self.pending, self.pages, self.entries = [], [], []

    def add(self, row_img, entry):
        self.pending.append((row_img, entry))
        if len(self.pending) == ROWS_PER_PAGE:
            self._flush()

    def finish(self):
        if self.pending:
            self._flush()

    def _flush(self):
        from PIL import Image, ImageDraw
        p = len(self.pages) + 1
        rel = os.path.join(self.rel_dir, f"page_{p:03d}.png").replace(os.sep, "/")
        page = Image.new("RGB", (SHEET_W, HEADER_H + ROWS_PER_PAGE * (ROW_H + GAP)), (55, 55, 55))
        d = ImageDraw.Draw(page)
        hdr = [f"{self.clip_id} | sheet {p}/{self.n_pages} | eligible rows {self.n_eligible} (all rendered once)",
               f"skipped rows by reason: {dict(sorted(self.skip_counts.items()))}",
               "Raw IMAGE outputs only. P = automatic potential hypothesis (finite, 5% margin, strictly closer to "
               "empty-side pose wrist, exactly one); not a correctness label. Handedness unused."]
        for j, t in enumerate(hdr):
            d.text((GAP, 3 + 14 * j), t, fill=(255, 255, 255), font=self.font)
        for k, (im, e) in enumerate(self.pending):
            page.paste(im, (0, HEADER_H + k * (ROW_H + GAP)))
            self.entries.append({**e, "page": rel, "slot": k})
        with open(os.path.join(self.abs_dir, os.path.basename(rel)), "xb") as f:
            page.save(f, format="PNG")
        self.pages.append(rel)
        self.pending.clear()


# ----------------------------------------------------------------------------- pipeline
def preflight(config_path, decoder):
    """All validation before any model creation: config, pins, arrays, sidecars, decode alignment, rows."""
    with open(config_path, "rb") as f:
        raw = f.read()
    cfg_obj, had_bom = load_json_bytes(raw, "config")
    cfg = validate_config(cfg_obj)
    targets = pin_targets(cfg)
    checks, data = hash_targets(targets, keep=True)
    bad = [{k: c.get(k) for k in ("label", "path", "expected", "actual", "error")} for c in checks if not c["ok"]]
    need(not bad, f"pin check failed (missing or changed): {bad}")
    clips = []
    for c in cfg["clips"]:
        cid = c["id"]
        try:
            obs = load_observation(data[f"clip:{cid}:observation"])
            side, _ = load_json_bytes(data[f"clip:{cid}:sidecar"], f"{cid} sidecar")
            sidecar = summarize_sidecar(side)
        except ProbeError as e:
            raise ProbeError(f"{cid}: {e}") from e
        T = obs["xy"].shape[0]
        lo, hi = c["active_span"]
        need(hi <= T - 1, f"{cid}: active_span {c['active_span']} outside rows 0..{T - 1}")
        dec = probe_alignment(decoder, c["video"], obs["ts"], obs["frame_id"])
        rows = evaluate_rows(obs["xy"], obs["observed"], obs["usable"], (lo, hi), dec["width"], dec["height"])
        for rec in rows:
            rec["frame_id"] = int(obs["frame_id"][rec["row"]])
            rec["timestamp_us"] = int(obs["ts"][rec["row"]])
        skip = Counter(r["skip_reason"] for r in rows if not r["eligible"])
        clips.append({"cfg": c, "obs": obs, "sidecar": sidecar, "decode": dec, "rows": rows, "T": T,
                      "eligible_rows": [r["row"] for r in rows if r["eligible"]],
                      "skip_counts": dict(sorted(skip.items()))})
    return {"config_path": config_path, "config_bytes": raw, "config_sha256": sha256_bytes(raw),
            "config_had_bom": had_bom, "cfg": cfg, "targets": targets, "pin_checks_before": checks,
            "model_bytes": data["hand_model"], "clips": clips}


def _clip_summary(cl):
    return {"id": cl["cfg"]["id"], "role": cl["cfg"]["role"], "active_span": cl["cfg"]["active_span"],
            "T": cl["T"], "decode_preflight": cl["decode"], "sidecar": cl["sidecar"],
            "eligible_count": len(cl["eligible_rows"]), "eligible_rows": cl["eligible_rows"],
            "skip_counts": cl["skip_counts"]}


def new_counters():
    z = lambda: {"B": 0, "C": 0}  # noqa: E731
    return {"backend_factory_calls": 0, "image_tasks_created": z(), "image_tasks_closed_ok": z(),
            "video_tasks_created": 0, "detect_calls_attempted": z(), "detect_calls_completed": z(),
            "raw_hands_returned": z(), "potential_single": z(), "potential_ambiguous_multiple": z(),
            "potential_none": z(), "retries": 0, "frames_converted_to_rgb": 0,
            "coordinates_merged_into_observations": 0, "prediction_changes": 0, "training_runs": 0,
            "tcn_forwards": 0, "camera_opened": 0, "network_calls": 0}


def _infer_row(rec, rgb, dets, W, H, counters):
    roi = rec["roi"]
    sha0 = _rgb_sha(rgb)
    t0 = time.perf_counter_ns()
    b_in = np.ascontiguousarray(rgb, dtype=np.uint8)
    t1 = time.perf_counter_ns()
    counters["detect_calls_attempted"]["B"] += 1
    b_raw = dets["B"].detect(b_in)
    t2 = time.perf_counter_ns()
    counters["detect_calls_completed"]["B"] += 1
    t3 = time.perf_counter_ns()
    crop = np.ascontiguousarray(warp_crop(rgb, roi["crop_to_src_affine"], roi["N"]), dtype=np.uint8)
    t4 = time.perf_counter_ns()
    crop_sha = _rgb_sha(crop)
    counters["detect_calls_attempted"]["C"] += 1
    c_raw = dets["C"].detect(crop)
    t5 = time.perf_counter_ns()
    counters["detect_calls_completed"]["C"] += 1
    need(_rgb_sha(rgb) == sha0, f"row {rec['row']}: source frame changed during IMAGE calls")
    need(_rgb_sha(crop) == crop_sha, f"row {rec['row']}: crop changed during IMAGE call")
    for tag, raw, shape, pre_ns, det_ns in (("B", b_raw, b_in.shape, t1 - t0, t2 - t1),
                                            ("C", c_raw, crop.shape, t4 - t3, t5 - t4)):
        t6 = time.perf_counter_ns()
        hands, src = convert_output(raw, tag, W, H, roi)
        flags, pot = select_potential(src, rec["pose_wrist_px"]["empty"], rec["pose_wrist_px"]["occupied"], W, H)
        for h, f in zip(hands, flags):
            h.update(f)
            h["is_potential"] = pot["index"] == h["raw_index"]
        rec[tag] = {"api": "HandLandmarker.detect (IMAGE)",
                    "input": "full_frame_rgb" if tag == "B" else "pose_roi_rotated_crop_rgb",
                    "input_shape": list(shape), "preprocess_ns": pre_ns, "detect_ns": det_ns,
                    "postprocess_ns": time.perf_counter_ns() - t6, "raw_count": len(hands),
                    "raw_count_exceeds_num_hands": len(hands) > NUM_HANDS, "hands": hands, "potential": pot}
        counters["raw_hands_returned"][tag] += len(hands)
        counters[f"potential_{pot['status']}"][tag] += 1
    rec["rgb_sha256"] = sha0
    rec["crop_sha256"] = crop_sha
    return crop


def _run_clip(cl, dets, decoder, out, state, font):
    c, obs, rows = cl["cfg"], cl["obs"], cl["rows"]
    W, H = cl["decode"]["width"], cl["decode"]["height"]
    elig = cl["eligible_rows"]
    entry = {"clip_id": c["id"], "eligible_rows": len(elig), "skip_counts": cl["skip_counts"],
             "pages": [], "entries": []}
    if not elig:
        return entry
    sheets = SheetWriter(out, c["id"], len(elig), cl["skip_counts"], font)
    pv = obs["observed"] & obs["usable"] & np.isfinite(obs["xy"]).all(axis=-1)
    it = iter_rows(decoder, c["video"], elig, obs["ts"], obs["frame_id"], (H, W))
    try:
        for r, rgb, info in it:
            state["stage"] = f"clip:{c['id']}:row:{r}"
            state["counters"]["frames_converted_to_rgb"] += 1
            rec = rows[r]
            rec["decode"] = info
            crop = _infer_row(rec, rgb, dets, W, H, state["counters"])
            sheets.add(make_row_image(rgb, crop, rec, obs["xy"][r], pv[r], c["id"], font),
                       {"clip_id": c["id"], "row": r, "frame_id": rec["frame_id"],
                        "timestamp_us": rec["timestamp_us"], "empty_side": rec["empty_side"]})
    finally:
        it.close()
    sheets.finish()
    entry["pages"], entry["entries"] = sheets.pages, sheets.entries
    need(sorted(e["row"] for e in entry["entries"]) == elig, f"{c['id']}: render index does not cover eligible rows once")
    return entry


def run_inference(pre, backend_factory, decoder, out, state):
    from PIL import ImageFont
    font = ImageFont.load_default()
    counters, lifecycle = state["counters"], state["tasks"]
    state["stage"] = "backend_factory"
    create, info = backend_factory(pre["model_bytes"], pre["cfg"]["hand_model_path"])
    counters["backend_factory_calls"] += 1
    state["backend_info"] = info
    dets = {}
    index = {"schema": SCHEMA + "/render_index", "config_sha256": pre["config_sha256"], "clips": []}
    try:
        for tag in ("B", "C"):
            state["stage"] = f"create_{tag}"
            lifecycle[tag] = {"created": False, "create_ns": None, "close_attempted": False,
                              "close_ok": False, "close_error": None}
            t0 = time.perf_counter_ns()
            dets[tag] = create(tag)
            lifecycle[tag].update(created=True, create_ns=time.perf_counter_ns() - t0)
            counters["image_tasks_created"][tag] += 1
        for cl in pre["clips"]:
            index["clips"].append(_run_clip(cl, dets, decoder, out, state, font))
    finally:
        for tag in ("C", "B"):
            if tag in dets:
                lifecycle[tag]["close_attempted"] = True
                try:
                    dets[tag].close()
                    lifecycle[tag]["close_ok"] = True
                    counters["image_tasks_closed_ok"][tag] += 1
                except Exception as exc:  # recorded; raised below if nothing else failed
                    lifecycle[tag]["close_error"] = f"{type(exc).__name__}: {exc}"
    bad = [t for t, r in lifecycle.items() if r["created"] and not r["close_ok"]]
    need(not bad, f"IMAGE task close failed: {bad}")
    n_el = sum(len(cl["eligible_rows"]) for cl in pre["clips"])
    n_ix = sum(len(e["entries"]) for e in index["clips"])
    need(n_ix == n_el, f"render index entries {n_ix} != eligible rows {n_el}")
    need(counters["detect_calls_completed"] == {"B": n_el, "C": n_el}, "IMAGE call count != 1B+1C per eligible row")
    index.update(total_eligible_rows=n_el, total_rendered_rows=n_ix,
                 note="Local-only images; eligible rows only; skipped rows counted, not rendered.")
    return index


def _pct(v):
    v = np.asarray(v, dtype=np.float64)
    if not v.size:
        return {"n": 0, "total": 0, "p50": None, "p95": None, "max": None}
    return {"n": int(v.size), "total": float(v.sum()), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def timing_summary(pre, state):
    rows = [r for cl in pre["clips"] for r in cl["rows"] if "B" in r and "C" in r]
    out = {"unit": "ns", "scope": "offline wall time on this machine; not a speed claim",
           "task_create_ns": {t: r["create_ns"] for t, r in state["tasks"].items()},
           "decode_to_rgb": _pct([r["decode"]["to_rgb_ns"] for r in rows])}
    for tag in ("B", "C"):
        out[tag] = {k: _pct([r[tag][k] for r in rows]) for k in ("preprocess_ns", "detect_ns", "postprocess_ns")}
    return out


def _safe_after(pre):
    try:
        checks, _ = hash_targets(pre["targets"])
    except Exception as e:  # noqa: BLE001 - recorded only on the failure path
        checks = [{"error": repr(e)}]
    try:
        cfg_after = sha256_file(pre["config_path"])
    except OSError as e:
        cfg_after = f"error: {e!r}"
    return checks, cfg_after


def execute(config_path, out_dir, backend_factory, decoder):
    """Shared by the real run and the self-test mocks. Returns (exit_code, result)."""
    out = os.path.abspath(out_dir)
    config_path = os.path.abspath(config_path)
    need(not os.path.lexists(out), f"output path already exists: {out}")
    script_sha = sha256_file(SRC)
    pre = preflight(config_path, decoder)  # raises before any backend/model creation
    norm = lambda p: os.path.normcase(os.path.abspath(p))  # noqa: E731
    need(all(norm(p) != norm(out) and not norm(p).startswith(norm(out) + os.sep)
             for _, p, _ in pre["targets"]), "an input path lies under the output directory")
    need(not os.path.lexists(out), f"output path already exists: {out}")
    os.makedirs(out)
    write_bytes_new(os.path.join(out, "config_snapshot.json"), pre["config_bytes"])
    write_json_new(os.path.join(out, "preflight.json"), {
        "schema": SCHEMA + "/preflight", "status": "PREFLIGHT_OK_BEFORE_MODEL_CREATION",
        "config_path": config_path, "config_sha256": pre["config_sha256"],
        "config_had_utf8_bom": pre["config_had_bom"], "script_sha256": script_sha,
        "pin_checks_before": pre["pin_checks_before"], "clips": [_clip_summary(cl) for cl in pre["clips"]]})
    state = {"stage": "start", "counters": new_counters(), "tasks": {}, "backend_info": None}
    try:
        index = run_inference(pre, backend_factory, decoder, out, state)
    except BaseException as exc:
        try:
            after, cfg_after = _safe_after(pre)
            write_json_new(os.path.join(out, "failure.json"), {
                "status": "TECHNICAL_FAILURE_NO_JUDGEMENT", "stage": state["stage"],
                "error": f"{type(exc).__name__}: {exc}", "counters": state["counters"],
                "task_lifecycle": state["tasks"], "backend": state["backend_info"],
                "pin_checks_before": pre["pin_checks_before"], "pin_checks_after": after,
                "config_sha256_before": pre["config_sha256"], "config_sha256_after": cfg_after})
        except Exception:  # noqa: BLE001 - never mask the original error
            pass
        raise
    state["stage"] = "final_pins"
    after, cfg_after = _safe_after(pre)
    script_after = sha256_file(SRC)
    pins_ok = (all(c.get("ok") for c in after) and cfg_after == pre["config_sha256"]
               and script_after == script_sha)
    rows_all = [r for cl in pre["clips"] for r in cl["rows"]]
    skip_total = Counter(r["skip_reason"] for r in rows_all if not r["eligible"])
    result = {
        "schema": SCHEMA + "/result",
        "status": "COMPLETED_PENDING_VISUAL_REVIEW" if pins_ok else "INVALID_INPUT_CHANGED_DURING_RUN",
        "stage": "stage1_observability_only",
        "script": {"path": SRC, "sha256_before": script_sha, "sha256_after": script_after},
        "config": {"path": config_path, "sha256_before": pre["config_sha256"], "sha256_after": cfg_after,
                   "had_utf8_bom": pre["config_had_bom"], "snapshot": "config_snapshot.json"},
        "pin_checks_before": pre["pin_checks_before"], "pin_checks_after": after, "pins_unchanged": pins_ok,
        "frozen_rules": FROZEN_RULES, "skip_reason_vocabulary": SKIP_REASONS,
        "backend": state["backend_info"], "task_lifecycle": state["tasks"], "counters": state["counters"],
        "timing": timing_summary(pre, state),
        "totals": {"rows": len(rows_all), "eligible_rows": sum(r["eligible"] for r in rows_all),
                   "skipped_rows": len(rows_all) - sum(r["eligible"] for r in rows_all),
                   "skip_counts": dict(sorted(skip_total.items()))},
        "prediction_changes": 0, "training_runs": 0, "tcn_forwards": 0,
        "coordinates_merged_into_observations": 0,
        "clips": [{**_clip_summary(cl), "rows": cl["rows"]} for cl in pre["clips"]],
        "render_index": "render_index.json", "notes": NOTES}
    write_json_new(os.path.join(out, "render_index.json"), index)
    write_json_new(os.path.join(out, "result.json"), result)
    return (0 if pins_ok else 3), result


# ----------------------------------------------------------------------------- self-test (synthetic/mock)
RED, GREEN = (255, 0, 0), (0, 255, 0)
K = 25.0 / 6.0  # axis length giving 1.2*K == 5 px (integer centre) and side 22.5 -> N=23, q=11


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def close_to(a, b, tol=1e-9):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=tol, rtol=0))


def _redness(img, ch):
    a = np.asarray(img, dtype=np.int32)
    s = 2 * a[..., ch] - a.sum(axis=-1)
    y, x = divmod(int(np.argmax(s)), a.shape[1])
    return (x, y), int(s[y, x])


def _marker_hands(rgb):
    h, w = rgb.shape[:2]
    out = []
    for ch in (1, 0):  # green (stored-hand marker) first, then red (empty-side marker)
        (x, y), s = _redness(rgb, ch)
        if s > 200:
            out.append({"landmarks_norm": [[x / w, y / h, 0.0]] * 21,
                        "handedness": [{"label": "Left", "score": 0.9}]})
    return out


class _FakeDetector:
    def __init__(self, tag, log, fail_detect):
        self.tag, self.log, self.fail = tag, log, fail_detect

    def detect(self, rgb):
        self.log["calls"][self.tag] += 1
        self.log["shapes"][self.tag].append(tuple(rgb.shape))
        if self.fail:
            raise RuntimeError(f"injected {self.tag} detect failure")
        return _marker_hands(rgb)

    def close(self):
        self.log["closed"].append(self.tag)


def _fake_factory(log, fail=None):
    def factory(model_bytes, model_path):
        log["factory_calls"] += 1

        def create(tag):
            if fail == ("create", tag):
                raise RuntimeError(f"injected {tag} create failure")
            log["created"].append(tag)
            return _FakeDetector(tag, log, fail == ("detect", tag))
        return create, {"kind": "fake"}
    return factory


def _new_log():
    return {"factory_calls": 0, "created": [], "closed": [], "calls": {"B": 0, "C": 0},
            "shapes": {"B": [], "C": []}}


def _synth_clip():
    W, H, T = 160, 120, 10
    xy = np.full((T, N_POINTS, 2), np.nan)
    obs = np.zeros((T, N_POINTS), dtype=bool)
    use = obs.copy()
    frames = [np.full((H, W, 3), 30, dtype=np.uint8) for _ in range(T)]

    def pt(r, i, p):
        xy[r, i] = p
        obs[r, i] = use[r, i] = True

    def hand(r, side, n=21):
        a = HAND_SLICE[side][0]
        bx, by = (50.0, 50.0) if side == "L" else (110.0, 50.0)
        for j in range(n):
            pt(r, a + j, (bx + j % 5, by - j // 5))

    for r in range(T):
        pt(r, 15, (50.0, 60.0)); pt(r, 19, (50.0, 60.0 - K)); pt(r, 17, (50.0, 60.0 - K))  # noqa: E702
        pt(r, 16, (110.0, 60.0)); pt(r, 20, (110.0, 60.0 + K)); pt(r, 18, (110.0, 60.0 + K))  # noqa: E702
    for r in (0, 1, 5, 6, 7, 8, 9):
        hand(r, "R")
    hand(2, "L"); hand(3, "L"); hand(3, "R"); hand(5, "L", 10)  # noqa: E702
    use[6, 16] = False
    use[7, 19] = False
    xy[8, 19] = xy[8, 15]
    xy[8, 17] = xy[8, 15]
    for r, p, col in ((1, (52, 51), RED), (1, (110, 50), GREEN), (2, (108, 69), RED), (2, (50, 50), GREEN)):
        frames[r][p[1], p[0]] = col
    reasons = ["outside_active_span", None, None, "both_hands_observed", "no_hand_observed",
               "partial_hand_observation", "pose_wrist_invalid", "empty_side_pose_refs_invalid",
               "roi_degenerate_zero_axis", "outside_active_span"]
    return {"W": W, "H": H, "T": T, "xy": xy, "observed": obs, "usable": use, "frames": frames,
            "ts": np.arange(T, dtype=np.int64) * 40000, "fid": np.arange(T, dtype=np.int64),
            "span": [1, 8], "reasons": reasons, "red_src": {1: (52.0, 51.0), 2: (108.0, 69.0)}}


def _mock_decoder(frames_by_path, ts_by_path, W, H, tamper_ts=False):
    def decoder(path):
        for i, fr in enumerate(frames_by_path[path]):
            rel = int(ts_by_path[path][i]) + (1 if tamper_ts and i == 3 else 0)
            yield {"ignored": None, "ordinal": i, "pts": i * 512, "time_base": "1/12800", "abs_us": rel,
                   "rel_us": rel, "width": W, "height": H, "to_rgb": (lambda f=fr: f.copy())}
    return decoder


def _build_env(tmp, mutate=()):
    s = _synth_clip()

    def wr(name, data):
        p = os.path.join(tmp, name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    plan = wr("FROZEN_PLAN.md", b"synthetic plan")
    model = wr("hand.task", b"synthetic task bytes")
    code = wr("landmarker.py", b"# synthetic pinned code")
    clips, frames_by, ts_by = [], {}, {}
    for i in range(EXPECTED_CLIPS):
        cid = f"SYN_{i}"
        video = wr(f"{cid}.mp4", f"synthetic video {i}".encode())
        arrays = {"xy_px": s["xy"], "observed": s["observed"], "usable": s["usable"].astype(np.uint8),
                  "timestamp_us": s["ts"], "frame_id": s["fid"],
                  "hand_score": np.zeros((s["T"], 2), np.float32), "conf": np.zeros((s["T"], N_POINTS), np.float32)}
        if i == 1 and "xy_shape" in mutate:
            arrays["xy_px"] = s["xy"][:, :74]
        if i == 1 and "no_frame_id" in mutate:
            del arrays["frame_id"]
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        npz = wr(f"{cid}.npz", buf.getvalue())
        side = {"num_hands": 1 if (i == 1 and "num_hands" in mutate) else 2,
                "unassigned_hands_total": 3 if i == 2 else 0,
                "dropped_frames": 2 if (i == 1 and "drop" in mutate) else 0}
        sc = wr(f"{cid}.json", json.dumps(side).encode())
        span = [1, s["T"]] if (i == 1 and "span" in mutate) else list(s["span"])
        clips.append({"id": cid, "video": video, "video_sha256": sha256_file(video), "observation": npz,
                      "observation_sha256": sha256_file(npz), "sidecar": sc,
                      "sidecar_sha256": "0" * 64 if (i == 1 and "pin" in mutate) else sha256_file(sc),
                      "active_span": span, "role": "failure" if i < 3 else "control"})
        frames_by[video], ts_by[video] = s["frames"], s["ts"]
    cfg = {"schema": SCHEMA, "plan_path": plan, "plan_sha256": sha256_file(plan), "hand_model_path": model,
           "hand_model_sha256": sha256_file(model),
           "code_pins": {code: "f" * 64 if "code_pin" in mutate else sha256_file(code)}, "clips": clips}
    cfg_path = wr("config.json", codecs.BOM_UTF8 + json.dumps(cfg).encode("utf-8"))
    return cfg_path, _mock_decoder(frames_by, ts_by, s["W"], s["H"], "ts" in mutate), s


DIRS = {  # unit axis u, src offset expected at crop (q, q-4), src offset expected at crop (q+3, q)
    "up": ((0.0, -1.0), (0, -4), (3, 0)),
    "down": ((0.0, 1.0), (0, 4), (-3, 0)),
    "right": ((1.0, 0.0), (4, 0), (0, 3)),
    "left": ((-1.0, 0.0), (-4, 0), (0, -3)),
}


def t_roi_directions():
    W, H = 160, 120
    for name, (u, ax_off, r_off) in DIRS.items():
        w = np.array([80.0, 60.0])
        c = w + K * np.array(u)
        roi = compute_roi(w, c, c, W, H)
        check(roi["ok"] and roi["N"] == 23 and roi["q"] == 11.0, f"{name}: N/q")
        center = w + 5.0 * np.array(u)
        check(close_to(roi["center_px"], center), f"{name}: centre")
        M = roi["crop_to_src_affine"]
        check(abs(np.linalg.det(M[:, :2]) - 1.0) < 1e-12, f"{name}: mapping is not a proper rotation (mirror?)")
        img = np.full((H, W, 3), 30, np.uint8)
        pa, pr = (center + ax_off).round().astype(int), (center + r_off).round().astype(int)
        img[pa[1], pa[0]] = RED
        img[pr[1], pr[0]] = GREEN
        crop = warp_crop(img, M, roi["N"])
        check(_redness(crop, 0)[0] == (11, 7), f"{name}: hand axis not toward crop top: {_redness(crop, 0)[0]}")
        check(_redness(crop, 1)[0] == (14, 11), f"{name}: crop right wrong: {_redness(crop, 1)[0]}")
        check(close_to(src_to_crop(M, center + ax_off), [11, 7], 1e-9), f"{name}: src_to_crop")
        check(close_to(crop_to_src(M, [14, 11]), center + r_off, 1e-9), f"{name}: crop_to_src")


def t_inverse_mapping_and_warp():
    W, H = 160, 120
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lin = np.dstack([xx, yy, xx + yy]).astype(np.float32)  # bilinear reproduces linear images exactly
    rng = np.random.default_rng(0)
    for ang in (0.1, 1.0, 2.5, -2.0, math.pi / 4):
        u = np.array([math.cos(ang), math.sin(ang)])
        w = np.array([80.2, 60.7])
        roi = compute_roi(w, w + 7.3 * u, w + 7.3 * u, W, H)
        check(roi["ok"], "roi ok")
        M, n, q = roi["crop_to_src_affine"], roi["N"], roi["q"]
        check(close_to(crop_to_src(M, [q, q - 3.0]), roi["center_px"] + 3.0 * u), "crop top != hand axis")
        pts = rng.uniform(-50, 200, size=(50, 2))
        check(close_to(crop_to_src(M, src_to_crop(M, pts)), pts, 1e-9), "src->crop->src")
        check(close_to(src_to_crop(M, crop_to_src(M, pts)), pts, 1e-9), "crop->src->crop")
        crop = warp_crop(lin, M, n)
        gy, gx = np.mgrid[0:n, 0:n].astype(np.float64)
        sp = crop_to_src(M, np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(n, n, 2)
        m = (sp[..., 0] > 1) & (sp[..., 0] < W - 2) & (sp[..., 1] > 1) & (sp[..., 1] < H - 2)
        check(m.sum() > 100, "warp test coverage")
        check(np.abs(crop[..., 0][m] - sp[..., 0][m]).max() < 0.05
              and np.abs(crop[..., 1][m] - sp[..., 1][m]).max() < 0.05, f"warp sampling != mapping at {ang}")


def t_landmark_roundtrip():
    W, H = 160, 120
    w = np.array([70.0, 65.0])
    u = np.array([math.cos(0.7), math.sin(0.7)])
    roi = compute_roi(w, w + 6.0 * u, w + 6.0 * u, W, H)
    M, n = roi["crop_to_src_affine"], roi["N"]
    src = np.array([[60.0 + 1.5 * j, 50.0 + (j % 4) * 2.25] for j in range(21)])
    full_raw = [{"landmarks_norm": np.column_stack([src / [W, H], np.zeros(21)]).tolist(), "handedness": []}]
    crop_px = src_to_crop(M, src)
    crop_raw = [{"landmarks_norm": np.column_stack([crop_px / n, np.zeros(21)]).tolist(), "handedness": []}]
    hb, sb = convert_output(full_raw, "B", W, H, roi)
    hc, scr = convert_output(crop_raw, "C", W, H, roi)
    check(close_to(sb[0], src, 1e-9) and close_to(hb[0]["landmarks_src_px"], src, 1e-9), "B roundtrip")
    check(close_to(scr[0], src, 1e-9), "C roundtrip")


def t_centered_and_padded():
    W, H = 160, 120
    img = np.full((H, W, 3), 30, np.uint8)
    w = np.array([80.0, 60.0])
    roi = compute_roi(w, w + [0, -K], w + [0, -K], W, H)
    crop = warp_crop(img, roi["crop_to_src_affine"], roi["N"])
    check(roi["fully_inside_source"] and (crop == 30).all(), "centred crop must have no padding")
    w = np.array([5.0, 8.0])  # centre (5,3): src y = y-8, src x = x-6
    roi = compute_roi(w, w + [0, -K], w + [0, -K], W, H)
    crop = warp_crop(img, roi["crop_to_src_affine"], roi["N"])
    check(not roi["fully_inside_source"] and roi["black_padding_present"], "padded flag")
    check((crop[0:8, 11] == 0).all() and (crop[11, 0:6] == 0).all(), "out-of-image pixels must be RGB 0")
    check((crop[8:, 6:] == 30).all(), "in-image pixels must be copied")


def t_degenerate_nonfinite_oversized():
    W, H = 160, 120
    w = np.array([80.0, 60.0])
    check(compute_roi(w, w, w, W, H)["skip_reason"] == "roi_degenerate_zero_axis", "d=0")
    check(compute_roi([np.inf, 0], w, w, W, H)["skip_reason"] == "roi_nonfinite", "inf wrist")
    check(compute_roi(w, [np.nan, 1], w, W, H)["skip_reason"] == "roi_nonfinite", "nan index")
    check(compute_roi([0, 0], [1e308, 0], [1e308, 0], W, H)["skip_reason"] == "roi_nonfinite", "overflow")
    check(compute_roi(w, w + [0.3, 0], w + [0.3, 0], W, H)["skip_reason"] == "roi_side_below_2px", "tiny")
    a = compute_roi(w, w + [59.2, 0], w + [59.2, 0], W, H)  # side 319.68 -> N=320 == 2*max(W,H)
    b = compute_roi(w, w + [59.3, 0], w + [59.3, 0], W, H)  # side 320.22 -> N=321
    check(a["ok"] and a["N"] == 320, "N==limit must be allowed")
    check(b["skip_reason"] == "roi_oversized" and b["N"] == 321, "N>limit must skip")


def t_eligibility_variants():
    s = _synth_clip()
    rows = evaluate_rows(s["xy"], s["observed"], s["usable"], s["span"], s["W"], s["H"])
    check([r["skip_reason"] for r in rows] == s["reasons"], f"reasons {[r['skip_reason'] for r in rows]}")
    check([r["row"] for r in rows if r["eligible"]] == [1, 2], "eligible rows")
    check(rows[1]["empty_side"] == "L" and rows[2]["empty_side"] == "R", "empty sides")
    u = s["usable"].copy()
    u[1, 33] = True
    check(evaluate_rows(s["xy"], s["observed"], u, s["span"], 160, 120)[1]["skip_reason"]
          == "empty_side_usable_without_observed", "usable without observed")
    u = s["usable"].copy()
    u[1, 54] = False
    check(evaluate_rows(s["xy"], s["observed"], u, s["span"], 160, 120)[1]["skip_reason"]
          == "occupied_hand_not_all_usable_finite", "observed but not usable")
    side = summarize_sidecar({"unassigned_hands_total": 3, "num_hands": 2})
    check(side["unassigned_any_nonzero"], "unassigned logged")
    for bad in ({"num_hands": 1}, {"dropped_frames": 1}, {"x": {"pts_drop": True}}, []):
        try:
            summarize_sidecar(bad)
            raise AssertionError(f"sidecar {bad} accepted")
        except ProbeError:
            pass


def _hand_at(wrist, last=None, n=21):
    pts = [[wrist[0] + 0.5 * j, wrist[1] - 0.5 * j] for j in range(n)]
    if last is not None:
        pts[-1] = list(last)
    return np.array(pts, dtype=np.float64)


def t_potential_rules():
    W, H, ew, ow = 100, 100, (20.0, 50.0), (80.0, 50.0)
    _, p = select_potential([], ew, ow, W, H)
    check(p["index"] is None and p["status"] == "none", "zero hands")
    f, p = select_potential([_hand_at((50.0, 50.0))], ew, ow, W, H)
    check(p["index"] is None and "distance_tie" in f[0]["rejection_causes"], "tie must not qualify")
    _, p = select_potential([_hand_at((25.0, 50.0)), _hand_at((30.0, 40.0))], ew, ow, W, H)
    check(p["index"] is None and p["status"] == "ambiguous_multiple" and p["qualifying_indices"] == [0, 1], "two")
    f, p = select_potential([_hand_at((75.0, 50.0)), _hand_at((25.0, 50.0))], ew, ow, W, H)
    check(p["index"] == 1 and "not_closer_to_empty_side_pose_wrist" in f[0]["rejection_causes"], "one of two")
    _, p = select_potential([_hand_at((25.0, 50.0), last=(-5.0, 50.0))], ew, ow, W, H)
    check(p["index"] == 0, "point at -5% margin is inside")
    f, p = select_potential([_hand_at((25.0, 50.0), last=(-5.01, 50.0))], ew, ow, W, H)
    check(p["index"] is None and "outside_5pct_frame_margin" in f[0]["rejection_causes"], "beyond margin")
    f, p = select_potential([_hand_at((25.0, 50.0), last=(np.nan, 1.0)), _hand_at((25.0, 50.0), n=20), None],
                            ew, ow, W, H)
    check(p["index"] is None and "nonfinite_landmark" in f[0]["rejection_causes"]
          and "not_21_landmarks" in f[1]["rejection_causes"] and "not_21_landmarks" in f[2]["rejection_causes"],
          "nonfinite / not 21")


def t_fail_before_backend():
    for mutate in ("xy_shape", "no_frame_id", "pin", "code_pin", "ts", "span", "num_hands", "drop", "out_exists"):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, dec, _ = _build_env(tmp, (mutate,))
            out = os.path.join(tmp, "out")
            if mutate == "out_exists":
                os.mkdir(out)
            log = _new_log()
            try:
                execute(cfg, out, _fake_factory(log), dec)
                raise AssertionError(f"{mutate}: run was not rejected")
            except ProbeError:
                pass
            check(log["factory_calls"] == 0 and not log["created"], f"{mutate}: backend touched before validation")
            check(mutate == "out_exists" or not os.path.exists(out), f"{mutate}: output created")


def _strict_load(path):
    def bad(c):
        raise AssertionError(f"non-standard JSON constant {c}")
    with open(path, encoding="utf-8") as f:
        return json.loads(f.read(), parse_constant=bad)


def t_fake_backend_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, dec, s = _build_env(tmp)
        before = {n: sha256_file(os.path.join(tmp, n)) for n in os.listdir(tmp)}
        out = os.path.join(tmp, "out")
        log = _new_log()
        rc, _ = execute(cfg, out, _fake_factory(log), dec)
        n_el = 2 * EXPECTED_CLIPS
        check(rc == 0, f"rc {rc}")
        check(log["created"] == ["B", "C"] and log["closed"] == ["C", "B"], "one B and one C task, both closed")
        check(log["calls"] == {"B": n_el, "C": n_el}, f"calls {log['calls']}")
        check(set(log["shapes"]["B"]) == {(s["H"], s["W"], 3)} and set(log["shapes"]["C"]) == {(23, 23, 3)},
              "B gets full frame, C gets N x N crop")
        res = _strict_load(os.path.join(out, "result.json"))
        idx = _strict_load(os.path.join(out, "render_index.json"))
        check(res["pins_unchanged"] and res["counters"]["video_tasks_created"] == 0
              and res["counters"]["retries"] == 0 and res["prediction_changes"] == 0, "pins/counters")
        for cl in res["clips"]:
            check(cl["eligible_rows"] == [1, 2], f"{cl['id']} eligible rows (unassigned must not exclude)")
            for r in (1, 2):
                row = cl["rows"][r]
                check(row["B"]["potential"]["index"] == 1 and row["C"]["potential"]["index"] == 0, "potentials")
                check(close_to(row["B"]["hands"][1]["landmarks_src_px"][0], s["red_src"][r], 1e-6), "B src px")
                check(close_to(row["C"]["hands"][0]["landmarks_src_px"][0], s["red_src"][r], 1e-6), "C src px")
                check("not_closer_to_empty_side_pose_wrist" in row["B"]["hands"][0]["rejection_causes"], "dup hand")
        un = res["clips"][2]["sidecar"]["unassigned_entries"]
        check(un and un[0]["value"] == 3, "unassigned logged")
        entries = [(e["clip_id"], e["row"]) for c in idx["clips"] for e in c["entries"]]
        check(sorted(entries) == sorted((f"SYN_{i}", r) for i in range(EXPECTED_CLIPS) for r in (1, 2)),
              "render index covers each eligible row once")
        check(all(os.path.isfile(os.path.join(out, p)) for c in idx["clips"] for p in c["pages"]), "pages exist")
        check(all(sha256_file(os.path.join(tmp, n)) == h for n, h in before.items()), "inputs modified")


def t_close_on_failure():
    for fail, created, calls in ((("detect", "C"), ["B", "C"], {"B": 1, "C": 1}),
                                 (("create", "C"), ["B"], {"B": 0, "C": 0})):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, dec, _ = _build_env(tmp)
            out = os.path.join(tmp, "out")
            log = _new_log()
            try:
                execute(cfg, out, _fake_factory(log, fail), dec)
                raise AssertionError(f"{fail}: failure not propagated")
            except RuntimeError:
                pass
            check(log["created"] == created and sorted(log["closed"]) == sorted(created), f"{fail}: close")
            check(log["calls"] == calls, f"{fail}: no retries {log['calls']}")
            fj = _strict_load(os.path.join(out, "failure.json"))
            check(all(v["close_ok"] for v in fj["task_lifecycle"].values() if v["created"]), "failure close record")


def t_nonmutation():
    s = _synth_clip()
    xy, o, u = s["xy"].copy(), s["observed"].copy(), s["usable"].copy()
    snap = (xy.tobytes(), o.tobytes(), u.tobytes())
    rows = evaluate_rows(xy, o, u, s["span"], s["W"], s["H"])
    check((xy.tobytes(), o.tobytes(), u.tobytes()) == snap, "evaluate_rows mutated inputs")
    roi = rows[1]["roi"]
    raw = [{"landmarks_norm": [[0.5, 0.5, 0.0]] * 21, "handedness": [{"label": "Right", "score": 0.7}]}]
    raw_copy = copy.deepcopy(raw)
    convert_output(raw, "C", s["W"], s["H"], roi)
    check(raw == raw_copy, "convert_output mutated raw output")
    buf = io.BytesIO()
    np.savez(buf, xy_px=s["xy"], observed=s["observed"], usable=s["usable"], timestamp_us=s["ts"],
             frame_id=s["fid"], hand_score=np.zeros((s["T"], 2)), conf=np.zeros((s["T"], N_POINTS)))
    obs = load_observation(buf.getvalue())
    check(all(not a.flags.writeable for a in obs.values()), "loaded observation arrays must be read-only")


SELF_TESTS = (t_roi_directions, t_inverse_mapping_and_warp, t_landmark_roundtrip, t_centered_and_padded,
              t_degenerate_nonfinite_oversized, t_eligibility_variants, t_potential_rules,
              t_fail_before_backend, t_fake_backend_end_to_end, t_close_on_failure, t_nonmutation)


def run_self_tests():
    failed = []
    for t in SELF_TESTS:
        try:
            t()
        except Exception as e:  # noqa: BLE001 - every failure is reported
            failed.append({"test": t.__name__, "error": f"{type(e).__name__}: {e}"})
    return {"schema": SCHEMA + "/self_test", "status": "PASS" if not failed else "FAIL",
            "tests_run": len(SELF_TESTS), "failed": failed, "script_sha256": sha256_file(SRC),
            "mediapipe_imported": "mediapipe" in sys.modules}


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config")
    ap.add_argument("--out", help="new directory; must not exist")
    ap.add_argument("--self-test", action="store_true", help="synthetic + mock only; prints compact JSON")
    a = ap.parse_args(argv)
    if a.self_test:
        if a.config or a.out:
            ap.error("--self-test takes no --config/--out")
        rep = run_self_tests()
        print(dumps_strict(rep, compact=True))
        return 0 if rep["status"] == "PASS" else 1
    if not (a.config and a.out):
        ap.error("--config and --out are required")
    try:
        rc, res = execute(a.config, a.out, mediapipe_backend, pyav_decode)
    except ProbeError as e:
        print(dumps_strict({"status": "FAILED_CLOSED", "error": str(e)}, compact=True), file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 4
    print(dumps_strict({"status": res["status"], "rc": rc, "totals": res["totals"],
                        "detect_calls": res["counters"]["detect_calls_completed"],
                        "out": os.path.abspath(a.out)}, compact=True))
    return rc


if __name__ == "__main__":
    sys.exit(main())
