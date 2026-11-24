#!/usr/bin/env python3
"""
Step 7B v1.0: Full 45-sample preprocessing using approved Step 7A v1.12 definitions.

Step 7A v1.12 EXTERNAL_REVIEW_PASS. Step 7B extends pilot 9 to full 45 with:
  - All 45 samples processed (train=27, val=9, test=9)
  - Pilot 9 strict NPZ/transform/timing regression against v1.12
  - Scaled coverage: dynamics 720, aggregate 4860+1620, full-cell 50400, etc.
  - Same production verifiers, synthetic regression, coverage metadata
  - No algorithm/threshold/setting changes from v1.12

Input: Step 6B v1.2.4 full 45 (frozen read-only)
Output: data/processed/sessions/S001_P001/step7b_full_v1_0/
"""

import csv
import hashlib
import json
import math
import os
import sys
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ================================================================
#  Paths & Constants
# ================================================================

PROJECT = Path(__file__).resolve().parent.parent
STEP6B_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "landmarks_3d_v1" / "full_v1_2_4"
RAW3D_DIR = STEP6B_DIR / "raw_3d"
OVERLAY_DIR = STEP6B_DIR / "overlays"
TA_CSV = STEP6B_DIR / "temporal_anomalies.csv"
OUT_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7b_full_v1_0"
V112_PILOT_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7a_pilot_v1_12"
V112_PILOT_NPZ_DIR = V112_PILOT_DIR / "npz"
V110_NPZ_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7a_pilot_v1_10" / "npz"
V111_NPZ_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7a_pilot_v1_11" / "npz"
V14_NPZ_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7a_pilot_v1_4" / "npz"
V15_NPZ_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7a_pilot_v1_5" / "npz"
V16_NPZ_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7a_pilot_v1_6" / "npz"
V17_NPZ_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7a_pilot_v1_7" / "npz"
V18_NPZ_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7a_pilot_v1_8" / "npz"
V19_NPZ_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7a_pilot_v1_9" / "npz"

VERSION = "1.0"
STEP = "7B"
RUN_ID = str(uuid.uuid4())
VERSION_TAG = f"7b_v{VERSION}"

WORDS = ["GREETING", "LOVE", "THANK_YOU"]
SPEEDS = ["FAST", "NORMAL", "SLOW"]
REPS = ["R01", "R02", "R03", "R04", "R05"]

LM_NOSE = 0
LM_LS, LM_RS = 11, 12
LM_LE, LM_RE = 13, 14
LM_LW, LM_RW = 15, 16
LM_LH, LM_RH = 23, 24

POSE_KEY_LMS = [LM_LE, LM_RE, LM_LW, LM_RW]
HAND_KEY_LMS = [0, 4, 8, 12, 16, 20]

SEG_ENERGY_WINDOW = 5
SEG_IDLE_THRESHOLD = 0.008
SEG_PAD_START = 5
SEG_PAD_END = 5
SEG_MIN_ACTIVE = 10

INTERP_CANDIDATES = [1, 2, 3]

ONE_EURO_CANDIDATES = [
    {"name": "conservative", "min_cutoff": 3.0, "beta": 0.001, "d_cutoff": 1.0},
    {"name": "moderate",     "min_cutoff": 1.7, "beta": 0.004, "d_cutoff": 1.0},
    {"name": "aggressive",   "min_cutoff": 1.0, "beta": 0.007, "d_cutoff": 1.0},
    {"name": "very_light",   "min_cutoff": 5.0, "beta": 0.0005, "d_cutoff": 1.0},
]

SAVGOL_WINDOW = 7
SAVGOL_ORDER = 3

RSN_NONE = np.int8(0)
RSN_INITIAL = np.int8(1)
RSN_TIMESTAMP_DISC = np.int8(2)
RSN_REAPPEAR = np.int8(3)
RSN_INVALID_OBS = np.int8(4)
RSN_STRONG_DEPTH = np.int8(5)
RSN_OTHER = np.int8(6)

RSN_CODE_MAP = {
    0: "none", 1: "initial", 2: "timestamp_discontinuity",
    3: "reappearance", 4: "invalid_observation",
    5: "strong_depth_boundary", 6: "other_boundary",
}

SMOOTHING_CONCLUSION = "no_final_smoothing_candidate_due_to_insufficient_verified_idle"
SMOOTHING_DEFAULT = "no_smoothing"
SMOOTHING_SAVGOL_NOTE = "savgol_offline_reference_only"
SELECTED_NORM_TRANSFORM = "raw_framewise"
SELECTED_MAX_GAP = 1

def interp_npz_keys():
    keys = []
    for c in ["pose", "lh", "rh"]:
        for s in ["interp", "norm_interp", "original_valid", "mask", "rejected", "boundary", "strong_forbidden"]:
            keys.append(f"{c}_{s}")
    return keys

def smooth_npz_keys():
    keys = []
    for c in ["pose", "lh", "rh"]:
        for s in ["filtered", "norm_filtered", "filter_valid", "reset_mask", "first_after_reset", "reset_reason"]:
            keys.append(f"{c}_{s}")
    return keys

def norm_npz_keys():
    return ["origins", "rotations", "scales", "dets", "ortho_errs",
            "norm_pose", "norm_lh", "norm_rh",
            "stab_origins", "stab_scales", "stab_rotations"]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _cmpv_csv(csv_str, expected, tol=1e-4):
    """Compare a CSV string cell to an expected value (numeric or sentinel string)."""
    if isinstance(expected, str): return csv_str == expected
    if isinstance(expected, bool): return csv_str == str(expected)
    if isinstance(expected, (int, float)):
        try: return abs(float(csv_str) - expected) < tol
        except (ValueError, TypeError): return False
    return csv_str == str(expected)

def sp(s):
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode())

# ================================================================
#  Dataset Split / Pilot Selection (unchanged)
# ================================================================

def build_dataset_split():
    split = {}
    pattern = ["train", "train", "val", "train", "train", "test",
               "train", "train", "val", "train", "train", "test",
               "train", "val", "test"]
    for word in WORDS:
        samples = []
        for si, speed in enumerate(SPEEDS):
            for ri, rep in enumerate(REPS):
                sid = f"S001_P001_{word}_{speed}_{rep}"
                samples.append((si * 5 + ri, sid))
        samples.sort()
        for idx, (_, sid) in enumerate(samples):
            split[sid] = {"sample_id": sid, "word": word, "split": pattern[idx], "position_in_word": idx}
    return split

def compute_sample_quality(sid):
    short = sid.replace("S001_P001_", "")
    with open(RAW3D_DIR / f"landmarks_3d_{short}.json", encoding="utf-8") as f:
        data = json.load(f)
    frames = data["frames"]; N = len(frames); total_lm = N * 75
    te_valid = sum(1 for fr in frames for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                   for lm in fr[k] if lm["template_eligible"])
    n_anom = sum(len(lm["qa_flags"]) for fr in frames for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                 for lm in fr[k])
    te_rate = te_valid / total_lm if total_lm else 0
    return {"sample_id": sid, "n_frames": N, "total_landmarks": total_lm,
            "te_valid": te_valid, "te_rate": te_rate, "n_anomaly_flags": n_anom,
            "quality_score": te_rate - 0.001 * n_anom / max(N, 1)}

def select_pilots(split):
    train_ids = [s for s, i in split.items() if i["split"] == "train"]
    by_word = defaultdict(list)
    for sid in train_ids: by_word[split[sid]["word"]].append(sid)
    pilot = {}
    for word in WORDS:
        qs = sorted([compute_sample_quality(s) for s in by_word[word]],
                     key=lambda x: x["quality_score"], reverse=True)
        for tier, idx in [("high", 0), ("mid", len(qs) // 2), ("low", -1)]:
            q = qs[idx]; pilot[q["sample_id"]] = {"tier": tier, "word": word, **q}
    return pilot

# ================================================================
#  Load Step 6B data (unchanged)
# ================================================================

def load_sample(sid):
    short = sid.replace("S001_P001_", "")
    with open(RAW3D_DIR / f"landmarks_3d_{short}.json", encoding="utf-8") as f:
        return json.load(f)

def extract_arrays(data):
    frames = data["frames"]; N = len(frames)
    ts = np.zeros(N, dtype=np.float64)
    COMPS = [("pose", 33), ("lh", 21), ("rh", 21)]
    KEYS_MAP = {"pose": "pose_landmarks_3d", "lh": "left_hand_landmarks_3d", "rh": "right_hand_landmarks_3d"}
    arrs = {}
    for comp, nlm in COMPS:
        arrs[f"{comp}_xyz"] = np.full((N, nlm, 3), np.nan)
        arrs[f"{comp}_tri"] = np.zeros((N, nlm), dtype=bool)
        arrs[f"{comp}_qa"] = np.zeros((N, nlm), dtype=bool)
        arrs[f"{comp}_te"] = np.zeros((N, nlm), dtype=bool)
        arrs[f"{comp}_reappear"] = np.zeros((N, nlm), dtype=bool)
        arrs[f"{comp}_flags"] = [[[] for _ in range(nlm)] for _ in range(N)]
        arrs[f"{comp}_invalid_reason"] = [["" for _ in range(nlm)] for _ in range(N)]
    for fi, fr in enumerate(frames):
        ts[fi] = fr["front_time_s"]
        for comp, nlm in COMPS:
            for li, lm in enumerate(fr[KEYS_MAP[comp]]):
                if lm["triangulation_valid"] and lm["xyz"] is not None:
                    arrs[f"{comp}_xyz"][fi, li] = lm["xyz"]
                arrs[f"{comp}_tri"][fi, li] = lm["triangulation_valid"]
                arrs[f"{comp}_qa"][fi, li] = lm["qa_valid"]
                arrs[f"{comp}_te"][fi, li] = lm["template_eligible"]
                arrs[f"{comp}_reappear"][fi, li] = lm.get("is_reappearance", False)
                arrs[f"{comp}_flags"][fi][li] = lm["qa_flags"]
                arrs[f"{comp}_invalid_reason"][fi][li] = lm.get("invalid_reason", "")
    arrs["timestamps"] = ts
    return arrs

# ================================================================
#  Strong-depth loading (unchanged)
# ================================================================

def load_strong_depth_data(pilot_ids):
    boundaries = defaultdict(set); records = []
    cat_to_comp = {"pose": "pose", "left_hand": "lh", "right_hand": "rh"}
    with open(TA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["sample_id"] not in pilot_ids: continue
            if row.get("strong_depth_condition") != "True": continue
            sid = row["sample_id"]; fi = int(row["frame_index"]); li = int(row["landmark_index"])
            comp = cat_to_comp.get(row["category"], row["category"])
            boundaries[sid].add((fi, comp, li))
            records.append({"sample_id": sid, "frame_index": fi, "component": comp,
                "landmark_index": li, "category": row["category"], "strong_depth_condition": True,
                "qa_invalidated_by_depth": row.get("qa_invalidated_by_depth", ""),
                "qa_already_invalid_before_depth": row.get("qa_already_invalid_before_depth", "")})
    return boundaries, records

# ================================================================
#  Normalization (unchanged)
# ================================================================

def compute_torso_frame(pose_xyz, pose_te, fi):
    ls = pose_xyz[fi, LM_LS]; rs = pose_xyz[fi, LM_RS]
    lh = pose_xyz[fi, LM_LH]; rh = pose_xyz[fi, LM_RH]
    ls_v = pose_te[fi, LM_LS] and not np.any(np.isnan(ls))
    rs_v = pose_te[fi, LM_RS] and not np.any(np.isnan(rs))
    lh_v = pose_te[fi, LM_LH] and not np.any(np.isnan(lh))
    rh_v = pose_te[fi, LM_RH] and not np.any(np.isnan(rh))
    if not (ls_v and rs_v): return None, None, None, "degenerate_shoulders"
    origin = (ls + rs) / 2.0; right_vec = rs - ls; sw = np.linalg.norm(right_vec)
    if sw < 1e-6: return origin, None, 0.0, "degenerate_shoulder_width"
    right = right_vec / sw
    if lh_v and rh_v: up_vec = origin - (lh + rh) / 2.0
    elif lh_v: up_vec = origin - lh
    elif rh_v: up_vec = origin - rh
    else: up_vec = np.array([0.0, -1.0, 0.0])
    up_n = np.linalg.norm(up_vec)
    if up_n < 1e-6: up_vec = np.array([0.0, -1.0, 0.0]); up_n = 1.0
    up = up_vec / up_n; fwd = np.cross(right, up); fn = np.linalg.norm(fwd)
    if fn < 1e-6: return origin, None, sw, "degenerate_axes"
    fwd /= fn; up = np.cross(fwd, right); up /= np.linalg.norm(up)
    R = np.column_stack([right, up, fwd])
    return origin, R, sw, "ok"

def normalize_sequence(arrs):
    N = len(arrs["timestamps"])
    origins = np.full((N, 3), np.nan); rotations = np.full((N, 3, 3), np.nan)
    scales = np.full(N, np.nan); dets = np.full(N, np.nan); ortho_errs = np.full(N, np.nan)
    statuses = []; fallback_src = [None] * N
    for fi in range(N):
        o, R, s, st = compute_torso_frame(arrs["pose_xyz"], arrs["pose_te"], fi)
        statuses.append(st)
        if o is not None: origins[fi] = o
        if R is not None: rotations[fi] = R; dets[fi] = np.linalg.det(R); ortho_errs[fi] = np.linalg.norm(R.T @ R - np.eye(3))
        if s is not None and s > 0: scales[fi] = s
    last_valid = None
    for fi in range(N):
        if statuses[fi] == "ok": last_valid = fi
        elif last_valid is not None:
            origins[fi] = origins[last_valid]; rotations[fi] = rotations[last_valid]
            scales[fi] = scales[last_valid]; dets[fi] = dets[last_valid]; ortho_errs[fi] = ortho_errs[last_valid]
            statuses[fi] = "fallback_prev"; fallback_src[fi] = last_valid
    first_valid = None
    for fi in range(N):
        if statuses[fi] in ("ok", "fallback_prev"): first_valid = fi; break
    if first_valid is not None and first_valid > 0:
        for fi in range(first_valid):
            origins[fi] = origins[first_valid]; rotations[fi] = rotations[first_valid]
            scales[fi] = scales[first_valid]; dets[fi] = dets[first_valid]; ortho_errs[fi] = ortho_errs[first_valid]
            statuses[fi] = "fallback_next"; fallback_src[fi] = first_valid
    def norm_pts(xyz, valid):
        out = np.full_like(xyz, np.nan)
        for fi in range(N):
            if np.isnan(scales[fi]) or scales[fi] < 1e-6 or np.any(np.isnan(rotations[fi])): continue
            R = rotations[fi]
            for li in range(xyz.shape[1]):
                if valid[fi, li] and not np.any(np.isnan(xyz[fi, li])):
                    out[fi, li] = R.T @ (xyz[fi, li] - origins[fi]) / scales[fi]
        return out
    return {"norm_pose": norm_pts(arrs["pose_xyz"], arrs["pose_te"]),
            "norm_lh": norm_pts(arrs["lh_xyz"], arrs["lh_te"]),
            "norm_rh": norm_pts(arrs["rh_xyz"], arrs["rh_te"]),
            "origins": origins, "rotations": rotations, "scales": scales,
            "dets": dets, "ortho_errs": ortho_errs, "statuses": statuses, "fallback_src": fallback_src}

def normalize_metric_coords(xyz_metric, valid, norm):
    N = xyz_metric.shape[0]; out = np.full_like(xyz_metric, np.nan)
    for fi in range(N):
        s = norm["scales"][fi]
        if np.isnan(s) or s < 1e-6 or np.any(np.isnan(norm["rotations"][fi])): continue
        R = norm["rotations"][fi]
        for li in range(xyz_metric.shape[1]):
            if valid[fi, li] and not np.any(np.isnan(xyz_metric[fi, li])):
                out[fi, li] = R.T @ (xyz_metric[fi, li] - norm["origins"][fi]) / s
    return out

def normalize_with_transform(xyz, valid, origins, rotations, scales):
    N = xyz.shape[0]; out = np.full_like(xyz, np.nan)
    for fi in range(N):
        s = scales[fi]
        if np.isnan(s) or s < 1e-6 or np.any(np.isnan(rotations[fi])): continue
        R = rotations[fi]
        for li in range(xyz.shape[1]):
            if valid[fi, li] and not np.any(np.isnan(xyz[fi, li])):
                out[fi, li] = R.T @ (xyz[fi, li] - origins[fi]) / s
    return out

def _normalize_with_transform_fast(xyz, valid, origins, rotations, scales):
    """Vectorized version for assertion use only (not byte-exact with original)."""
    N, L, _ = xyz.shape; out = np.full_like(xyz, np.nan)
    for fi in range(N):
        s = scales[fi]
        if np.isnan(s) or s < 1e-6 or np.any(np.isnan(rotations[fi])): continue
        mask = valid[fi] & ~np.any(np.isnan(xyz[fi]), axis=1)
        if not np.any(mask): continue
        centered = xyz[fi, mask] - origins[fi]
        out[fi, mask] = (centered @ rotations[fi]) / s
    return out

# ================================================================
#  Causal Stabilized Transform (unchanged)
# ================================================================

class OneEuroFilter1D:
    def __init__(self, mc=1.0, beta=0.007, dc=1.0):
        self.mc = mc; self.beta = beta; self.dc = dc
        self.x_prev = None; self.dx_prev = 0.0; self.t_prev = None
    def __call__(self, x, t):
        if self.x_prev is None: self.x_prev = x; self.t_prev = t; return x
        dt = t - self.t_prev
        if dt <= 0: return self.x_prev
        dx = (x - self.x_prev) / dt
        tau_d = 1.0 / (2 * math.pi * self.dc); a_d = 1.0 / (1.0 + tau_d / dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mc + self.beta * abs(dx_hat)
        tau = 1.0 / (2 * math.pi * cutoff); a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat; self.dx_prev = dx_hat; self.t_prev = t
        return x_hat

def compute_causal_stabilized_transform(norm, ts):
    N = len(ts)
    stab_origins = np.full((N, 3), np.nan); stab_scales = np.full(N, np.nan)
    stab_rotations = np.full((N, 3, 3), np.nan)
    params = {"mc": 3.0, "beta": 0.001, "dc": 1.0}
    o_filts = [OneEuroFilter1D(**params) for _ in range(3)]
    for fi in range(N):
        if not np.any(np.isnan(norm["origins"][fi])):
            for d in range(3): stab_origins[fi, d] = o_filts[d](norm["origins"][fi, d], ts[fi])
    s_filt = OneEuroFilter1D(**params)
    for fi in range(N):
        if not np.isnan(norm["scales"][fi]): stab_scales[fi] = s_filt(norm["scales"][fi], ts[fi])
    r_filts = [[OneEuroFilter1D(**params) for _ in range(3)] for _ in range(3)]
    for fi in range(N):
        R = norm["rotations"][fi]
        if np.any(np.isnan(R)): continue
        fc = np.zeros((3, 3))
        for col in range(3):
            for d in range(3): fc[d, col] = r_filts[col][d](R[d, col], ts[fi])
        right = fc[:, 0]; rn = np.linalg.norm(right)
        if rn < 1e-8: continue
        right /= rn
        up = fc[:, 1]; up -= np.dot(up, right) * right; un = np.linalg.norm(up)
        if un < 1e-8: continue
        up /= un; fwd = np.cross(right, up)
        stab_rotations[fi] = np.column_stack([right, up, fwd])
    return {"stab_origins": stab_origins, "stab_scales": stab_scales, "stab_rotations": stab_rotations}

# ================================================================
#  Shared timing helpers (v1.5)
# ================================================================

def _find_peaks_with_prominence(signal, min_prom_frac=0.2):
    """Returns list of (index, peak_height, prominence)."""
    peaks = []
    if len(signal) < 3: return peaks
    max_val = np.max(signal)
    if max_val <= 0: return peaks
    min_prom = max_val * min_prom_frac
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            left_min = np.min(signal[max(0, i - 15):i]) if i > 0 else signal[i]
            right_min = np.min(signal[i + 1:min(len(signal), i + 16)])
            prom = signal[i] - max(left_min, right_min)
            if prom >= min_prom: peaks.append((i, float(signal[i]), float(prom)))
    return peaks

def _segment_xcorr_lag(sr, sf):
    if len(sr) < 5: return {"lag_frames": 0, "status": "too_short"}
    sr_c = sr - np.mean(sr); sf_c = sf - np.mean(sf)
    sr_std = np.std(sr_c); sf_std = np.std(sf_c)
    if sr_std < 1e-12 or sf_std < 1e-12: return {"lag_frames": 0, "status": "constant_signal"}
    corr = np.correlate(sr_c / sr_std, sf_c / sf_std, mode="full") / len(sr)
    max_lag = min(10, len(sr) // 2); center = len(sr) - 1
    region = corr[center - max_lag:center + max_lag + 1]
    if len(region) == 0: return {"lag_frames": 0, "status": "no_region"}
    # Convention: lag = filtered_frame - raw_frame; positive = causal delay
    # np.correlate(raw, filt) peaks at offset -d for delay d, so negate
    lag = -int(np.argmax(region) - max_lag)
    status = "negative_inconsistent_with_causality" if lag < 0 else "nonnegative"
    return {"lag_frames": lag, "status": status}

def run_xcorr_regression_tests():
    """3 synthetic regression tests verifying cross-correlation lag sign convention."""
    failures = []
    N = 100; t = np.linspace(0, 2 * np.pi, N)
    # Test 1: identity (no delay) → lag = 0
    sig = np.sin(t)
    r1 = _segment_xcorr_lag(sig, sig)
    if r1["lag_frames"] != 0:
        failures.append(f"xcorr_test1_identity: expected lag=0, got {r1['lag_frames']}")
    # Test 2: filtered delayed by 3 frames (causal) → lag > 0
    raw = np.sin(t)
    filt = np.zeros(N)
    filt[3:] = raw[:-3]
    r2 = _segment_xcorr_lag(raw, filt)
    if r2["lag_frames"] <= 0:
        failures.append(f"xcorr_test2_causal_delay: expected lag>0, got {r2['lag_frames']}")
    if r2["status"] != "nonnegative":
        failures.append(f"xcorr_test2_causal_status: expected nonnegative, got {r2['status']}")
    # Test 3: filtered leads by 3 frames (anticausal) → lag < 0
    filt3 = np.zeros(N)
    filt3[:-3] = raw[3:]
    r3 = _segment_xcorr_lag(raw, filt3)
    if r3["lag_frames"] >= 0:
        failures.append(f"xcorr_test3_anticausal: expected lag<0, got {r3['lag_frames']}")
    if r3["status"] != "negative_inconsistent_with_causality":
        failures.append(f"xcorr_test3_anticausal_status: expected negative, got {r3['status']}")
    return failures

def run_timing_aggregate_regression_tests():
    """Synthetic timing aggregate test: nonneg, negative, and empty events."""
    failures = []
    # Build synthetic per_landmark list with mixed timing statuses
    lm1 = {"timing": {
        "n_nonneg_matched": 2, "n_negative_inconsistent": 1,
        "n_unmatched_raw": 1, "n_unmatched_filt": 0,
        "matched_events": [
            {"status": "matched", "lag_frames": 3, "lag_ms": 100.0},
            {"status": "matched", "lag_frames": 5, "lag_ms": 166.7},
        ],
        "segments": [
            {"onset_lag": {"status": "nonnegative", "lag_frames": 2, "lag_ms": 66.7},
             "end_lag": {"status": "negative_inconsistent_with_causality", "lag_frames": -1, "lag_ms": -33.3},
             "xcorr_lag": {"status": "nonnegative", "lag_frames": 4}},
            {"onset_lag": {"status": "negative_inconsistent_with_causality", "lag_frames": -2, "lag_ms": -66.7},
             "end_lag": {"status": "nonnegative", "lag_frames": 1, "lag_ms": 33.3},
             "xcorr_lag": {"status": "nonnegative", "lag_frames": 6}},
        ]
    }}
    lm2 = {"timing": {
        "n_nonneg_matched": 0, "n_negative_inconsistent": 0,
        "n_unmatched_raw": 0, "n_unmatched_filt": 1,
        "matched_events": [],
        "segments": [
            {"onset_lag": {"status": "too_short"},
             "end_lag": {"status": "threshold_not_reached"},
             "xcorr_lag": {"status": "no_region"}},
        ]
    }}
    # The _build_timing_aggregate is a local function inside compare_transforms.
    # Replicate the logic here for testing.
    peak_nonneg, peak_neg, peak_unmatched_raw, peak_unmatched_filt = 0, 0, 0, 0
    peak_lags_frames, peak_lags_ms = [], []
    onset_nonneg, onset_neg, end_nonneg, end_neg = 0, 0, 0, 0
    xcorr_nonneg, xcorr_neg = 0, 0
    onset_frames, onset_ms, end_frames, end_ms, xcorr_frames = [], [], [], [], []
    for pl in [lm1, lm2]:
        t = pl.get("timing", {})
        peak_nonneg += t.get("n_nonneg_matched", 0)
        peak_neg += t.get("n_negative_inconsistent", 0)
        peak_unmatched_raw += t.get("n_unmatched_raw", 0)
        peak_unmatched_filt += t.get("n_unmatched_filt", 0)
        for m in t.get("matched_events", []):
            if m.get("status") == "matched":
                peak_lags_frames.append(m["lag_frames"])
                if "lag_ms" in m: peak_lags_ms.append(m["lag_ms"])
        for seg_t in t.get("segments", []):
            ol = seg_t.get("onset_lag", {}); ol_s = ol.get("status", "")
            if ol_s == "nonnegative":
                onset_nonneg += 1
                if "lag_frames" in ol: onset_frames.append(ol["lag_frames"])
                if "lag_ms" in ol: onset_ms.append(ol["lag_ms"])
            elif ol_s.startswith("negative"): onset_neg += 1
            el = seg_t.get("end_lag", {}); el_s = el.get("status", "")
            if el_s == "nonnegative":
                end_nonneg += 1
                if "lag_frames" in el: end_frames.append(el["lag_frames"])
                if "lag_ms" in el: end_ms.append(el["lag_ms"])
            elif el_s.startswith("negative"): end_neg += 1
            xl = seg_t.get("xcorr_lag", {}); xl_s = xl.get("status", "")
            if xl_s == "nonnegative":
                xcorr_nonneg += 1
                if "lag_frames" in xl: xcorr_frames.append(xl["lag_frames"])
            elif xl_s.startswith("negative"): xcorr_neg += 1
    # Verify counts
    if peak_nonneg != 2: failures.append(f"peak_nonneg expected 2, got {peak_nonneg}")
    if peak_neg != 1: failures.append(f"peak_neg expected 1, got {peak_neg}")
    if peak_unmatched_raw != 1: failures.append(f"peak_unmatched_raw expected 1, got {peak_unmatched_raw}")
    if peak_unmatched_filt != 1: failures.append(f"peak_unmatched_filt expected 1, got {peak_unmatched_filt}")
    if onset_nonneg != 1: failures.append(f"onset_nonneg expected 1, got {onset_nonneg}")
    if onset_neg != 1: failures.append(f"onset_neg expected 1, got {onset_neg}")
    if end_nonneg != 1: failures.append(f"end_nonneg expected 1, got {end_nonneg}")
    if end_neg != 1: failures.append(f"end_neg expected 1, got {end_neg}")
    if xcorr_nonneg != 2: failures.append(f"xcorr_nonneg expected 2, got {xcorr_nonneg}")
    if xcorr_neg != 0: failures.append(f"xcorr_neg expected 0, got {xcorr_neg}")
    # Verify pooled means
    expected_peak_mean_frames = (3 + 5) / 2
    actual_peak_mean = float(np.mean(peak_lags_frames)) if peak_lags_frames else None
    if actual_peak_mean is None or abs(actual_peak_mean - expected_peak_mean_frames) > 1e-6:
        failures.append(f"peak_pooled_mean expected {expected_peak_mean_frames}, got {actual_peak_mean}")
    expected_onset_mean_frames = 2.0  # only one nonneg onset
    actual_onset_mean = float(np.mean(onset_frames)) if onset_frames else None
    if actual_onset_mean is None or abs(actual_onset_mean - expected_onset_mean_frames) > 1e-6:
        failures.append(f"onset_pooled_mean expected {expected_onset_mean_frames}, got {actual_onset_mean}")
    expected_xcorr_mean = (4 + 6) / 2
    actual_xcorr_mean = float(np.mean(xcorr_frames)) if xcorr_frames else None
    if actual_xcorr_mean is None or abs(actual_xcorr_mean - expected_xcorr_mean) > 1e-6:
        failures.append(f"xcorr_pooled_mean expected {expected_xcorr_mean}, got {actual_xcorr_mean}")
    return failures

def _onset_lag(sr, sf, ta):
    if len(sr) < 3: return {"lag_frames": 0, "lag_ms": 0.0, "status": "too_short"}
    thr_r = np.max(sr) * 0.2; thr_f = np.max(sf) * 0.2
    onset_r = next((i for i in range(len(sr)) if sr[i] > thr_r), None)
    onset_f = next((i for i in range(len(sf)) if sf[i] > thr_f), None)
    if onset_r is not None and onset_f is not None:
        lag_f = onset_f - onset_r; lag_ms = float((ta[onset_f] - ta[onset_r]) * 1000)
        status = "negative_inconsistent_with_causality" if lag_f < 0 else "nonnegative"
        return {"lag_frames": lag_f, "lag_ms": lag_ms, "status": status}
    return {"lag_frames": 0, "lag_ms": 0.0, "status": "threshold_not_reached"}

def _end_lag(sr, sf, ta):
    if len(sr) < 3: return {"lag_frames": 0, "lag_ms": 0.0, "status": "too_short"}
    thr_r = np.max(sr) * 0.2; thr_f = np.max(sf) * 0.2
    end_r = next((i for i in range(len(sr) - 1, -1, -1) if sr[i] > thr_r), None)
    end_f = next((i for i in range(len(sf) - 1, -1, -1) if sf[i] > thr_f), None)
    if end_r is not None and end_f is not None:
        lag_f = end_f - end_r; lag_ms = float((ta[end_f] - ta[end_r]) * 1000)
        status = "negative_inconsistent_with_causality" if lag_f < 0 else "nonnegative"
        return {"lag_frames": lag_f, "lag_ms": lag_ms, "status": status}
    return {"lag_frames": 0, "lag_ms": 0.0, "status": "threshold_not_reached"}

def _match_peaks(raw_peaks, filt_peaks, ta, seg_start):
    """Match raw to filtered peaks. Returns (matches, unmatched_raw, unmatched_filt)."""
    matches, unmatched_raw, unmatched_filt = [], [], []
    used = set()
    for r_idx, r_height, r_prom in sorted(raw_peaks, key=lambda x: -x[2]):
        best_j, best_d = None, 11
        for j, (f_idx, _, _) in enumerate(filt_peaks):
            if j in used: continue
            d = f_idx - r_idx
            if abs(d) < best_d: best_d = abs(d); best_j = j
        if best_j is not None and best_d <= 10:
            used.add(best_j)
            f_idx, f_height, f_prom = filt_peaks[best_j]
            lag_f = f_idx - r_idx
            lag_ms = float((ta[f_idx] - ta[r_idx]) * 1000) if f_idx < len(ta) and r_idx < len(ta) else 0
            if lag_f < 0:
                status = "negative_match_inconsistent_with_causality"
            else:
                status = "matched"
            matches.append({
                "raw_peak_frame": int(r_idx + seg_start), "filt_peak_frame": int(f_idx + seg_start),
                "raw_peak_height": float(r_height), "raw_prominence": float(r_prom),
                "filt_peak_height": float(f_height), "filt_prominence": float(f_prom),
                "lag_frames": int(lag_f), "lag_ms": float(lag_ms), "status": status})
        else:
            unmatched_raw.append({
                "raw_peak_frame": int(r_idx + seg_start),
                "raw_peak_height": float(r_height), "raw_prominence": float(r_prom),
                "reason": "no_filt_peak_within_window"})
    for j, (f_idx, f_height, f_prom) in enumerate(filt_peaks):
        if j not in used:
            unmatched_filt.append({
                "filt_peak_frame": int(f_idx + seg_start),
                "filt_peak_height": float(f_height), "filt_prominence": float(f_prom),
                "reason": "no_raw_peak_matched"})
    return matches, unmatched_raw, unmatched_filt

def _compute_timing_for_speeds(speeds_raw, speeds_filt, bd_flags, ts, key_lms_for_comp):
    """Compute per-landmark timing analysis for a pair of speed arrays."""
    # This is called for both smoothing and transform timing
    N = len(speeds_raw)  # speeds_raw shape: (N, n_lms)
    n_lms = speeds_raw.shape[1] if speeds_raw.ndim > 1 else 1
    all_lm_results = []
    nonneg_lags_f, nonneg_lags_ms = [], []

    lm_list = key_lms_for_comp if key_lms_for_comp is not None else range(n_lms)
    for li_idx, li in enumerate(lm_list):
        if speeds_raw.ndim == 1:
            sr_full = speeds_raw; sf_full = speeds_filt; bd_full = bd_flags
        else:
            sr_full = speeds_raw[:, li_idx]; sf_full = speeds_filt[:, li_idx]
            bd_full = bd_flags[:, li_idx] if bd_flags.ndim > 1 else bd_flags

        segments = _split_contiguous_valid(sr_full, bd_full, min_len=10)
        seg_results = []
        lm_matches_all, lm_unmatched_raw_all, lm_unmatched_filt_all = [], [], []

        for seg_s, seg_e in segments:
            sr = sr_full[seg_s:seg_e]; sf = sf_full[seg_s:seg_e]; ta = ts[seg_s:seg_e]
            raw_peaks = _find_peaks_with_prominence(sr, min_prom_frac=0.2)
            filt_peaks = _find_peaks_with_prominence(sf, min_prom_frac=0.2)
            matches, unmatched_r, unmatched_f = _match_peaks(raw_peaks, filt_peaks, ta, seg_s)
            xcorr = _segment_xcorr_lag(sr, sf)
            onset = _onset_lag(sr, sf, ta)
            end = _end_lag(sr, sf, ta)

            raw_peak_list = [{"frame": int(idx + seg_s), "timestamp": float(ts[idx + seg_s]) if idx + seg_s < len(ts) else 0,
                              "peak_height": float(h), "prominence": float(p)} for idx, h, p in raw_peaks]
            filt_peak_list = [{"frame": int(idx + seg_s), "timestamp": float(ts[idx + seg_s]) if idx + seg_s < len(ts) else 0,
                               "peak_height": float(h), "prominence": float(p)} for idx, h, p in filt_peaks]

            seg_results.append({
                "seg_start": int(seg_s), "seg_end": int(seg_e), "seg_len": int(seg_e - seg_s),
                "raw_peaks": raw_peak_list, "filt_peaks": filt_peak_list,
                "n_raw_peaks": len(raw_peaks), "n_filt_peaks": len(filt_peaks),
                "xcorr_lag": xcorr, "onset_lag": onset, "end_lag": end})
            lm_matches_all.extend(matches)
            lm_unmatched_raw_all.extend(unmatched_r)
            lm_unmatched_filt_all.extend(unmatched_f)

        # Per-landmark aggregate (nonnegative matches only)
        nn_matches = [m for m in lm_matches_all if m["status"] == "matched"]
        neg_matches = [m for m in lm_matches_all if m["status"] != "matched"]

        lm_entry = {
            "landmark": int(li), "n_segments": len(segments),
            "n_nonneg_matched": len(nn_matches), "n_negative_inconsistent": len(neg_matches),
            "n_unmatched_raw": len(lm_unmatched_raw_all), "n_unmatched_filt": len(lm_unmatched_filt_all),
            "segments": seg_results, "matched_events": lm_matches_all,
            "unmatched_raw_events": lm_unmatched_raw_all, "unmatched_filt_events": lm_unmatched_filt_all}
        if nn_matches:
            lags_f = [m["lag_frames"] for m in nn_matches]
            lags_ms = [m["lag_ms"] for m in nn_matches]
            lm_entry["mean_lag_frames"] = float(np.mean(lags_f))
            lm_entry["mean_lag_ms"] = float(np.mean(lags_ms))
            lm_entry["status"] = "matched"
            nonneg_lags_f.extend(lags_f); nonneg_lags_ms.extend(lags_ms)
        else:
            lm_entry["status"] = "no_nonneg_matches"
        all_lm_results.append(lm_entry)

    # Collect per-landmark counts for aggregate
    total_neg = sum(lm.get("n_negative_inconsistent", 0) for lm in all_lm_results)
    total_unmatched_raw = sum(lm.get("n_unmatched_raw", 0) for lm in all_lm_results)
    total_unmatched_filt = sum(lm.get("n_unmatched_filt", 0) for lm in all_lm_results)

    result = {"per_landmark": all_lm_results,
              "total_negative_inconsistent": total_neg,
              "total_unmatched_raw": total_unmatched_raw,
              "total_unmatched_filt": total_unmatched_filt}
    if nonneg_lags_f:
        result["aggregate_lag_frames"] = float(np.mean(nonneg_lags_f))
        result["aggregate_lag_ms"] = float(np.mean(nonneg_lags_ms))
        result["total_nonneg_matched"] = len(nonneg_lags_f)
        result["status"] = "matched"
    else:
        result["total_nonneg_matched"] = 0
        result["status"] = "peak_correspondence_failed"
    return result

def _split_contiguous_valid(speeds, bd_flags, min_len=10):
    segments = []; N = len(speeds); ss = None
    for fi in range(N):
        if bd_flags[fi]:
            if ss is not None and fi - ss >= min_len: segments.append((ss, fi))
            ss = None
        else:
            if ss is None: ss = fi
    if ss is not None and N - ss >= min_len: segments.append((ss, N))
    return segments

# ================================================================
#  Transform Comparison (v1.5: boundary-aware, physical paths, timing)
# ================================================================

def compare_transforms(arrs, norm, stab, ts, seg_info, boundaries):
    per_landmark = []
    boundary_edges_skipped = 0

    for comp, nlm, key_lms, comp_name in [
        ("pose", 33, POSE_KEY_LMS, "pose"),
        ("lh", 21, HAND_KEY_LMS, "left_hand"),
        ("rh", 21, HAND_KEY_LMS, "right_hand"),
    ]:
        xyz = arrs[f"{comp}_xyz"]; valid = arrs[f"{comp}_te"]
        bd = boundaries[comp]
        raw_norm = normalize_with_transform(xyz, valid, norm["origins"], norm["rotations"], norm["scales"])
        stab_norm = normalize_with_transform(xyz, valid, stab["stab_origins"], stab["stab_rotations"], stab["stab_scales"])
        N = len(ts)

        # Compute per-landmark speeds for timing analysis
        n_key = len(key_lms)
        speeds_raw_arr = np.zeros((N, n_key)); speeds_filt_arr = np.zeros((N, n_key))
        bd_flags_arr = np.zeros((N, n_key), dtype=bool)
        bd_flags_arr[0, :] = True

        for ki, li in enumerate(key_lms):
            devs_m = []
            path_raw_phys, path_stab_phys = 0.0, 0.0
            vel_vec_errs, accel_vec_errs = [], []
            prev_rv, prev_sv = None, None

            for fi in range(N):
                if valid[fi, li] and not np.any(np.isnan(raw_norm[fi, li])) and not np.any(np.isnan(stab_norm[fi, li])):
                    s = norm["scales"][fi] if not np.isnan(norm["scales"][fi]) else 0
                    devs_m.append(np.linalg.norm(stab_norm[fi, li] - raw_norm[fi, li]) * s)

                # Check boundary for edge fi-1 -> fi
                if fi > 0:
                    dt = ts[fi] - ts[fi - 1]
                    is_boundary_edge = bd[fi, li] or dt <= 0 or dt > 0.1 or not valid[fi, li] or not valid[fi - 1, li]
                    if is_boundary_edge:
                        bd_flags_arr[fi, ki] = True
                        prev_rv = prev_sv = None
                        if bd[fi, li]: boundary_edges_skipped += 1
                        continue

                    r0, r1 = raw_norm[fi - 1, li], raw_norm[fi, li]
                    s0, s1 = stab_norm[fi - 1, li], stab_norm[fi, li]
                    if np.any(np.isnan(r0)) or np.any(np.isnan(r1)) or np.any(np.isnan(s0)) or np.any(np.isnan(s1)):
                        bd_flags_arr[fi, ki] = True; prev_rv = prev_sv = None; continue

                    sc = norm["scales"][fi] if not np.isnan(norm["scales"][fi]) else 0
                    # Physical path lengths
                    path_raw_phys += np.linalg.norm(r1 - r0) * sc
                    path_stab_phys += np.linalg.norm(s1 - s0) * sc

                    # Speeds for timing
                    speeds_raw_arr[fi, ki] = np.linalg.norm(r1 - r0) * sc / dt
                    speeds_filt_arr[fi, ki] = np.linalg.norm(s1 - s0) * sc / dt

                    # Velocity/acceleration vector errors
                    rv = (r1 - r0) * sc / dt; sv = (s1 - s0) * sc / dt
                    vel_vec_errs.append(np.linalg.norm(sv - rv))
                    if prev_rv is not None and prev_sv is not None:
                        ra = (rv - prev_rv) / dt; sa = (sv - prev_sv) / dt
                        accel_vec_errs.append(np.linalg.norm(sa - ra))
                    prev_rv, prev_sv = rv, sv

            if devs_m:
                da = np.array(devs_m)
                entry = {
                    "component": comp_name, "landmark": int(li),
                    "traj_dev_mean_mm": float(np.mean(da) * 1000),
                    "traj_dev_p95_mm": float(np.percentile(da, 95) * 1000),
                    "traj_dev_max_mm": float(np.max(da) * 1000),
                    "path_length_raw_m": float(path_raw_phys),
                    "path_length_stab_m": float(path_stab_phys),
                    "path_length_ratio": float(path_stab_phys / path_raw_phys) if path_raw_phys > 0 else 1.0,
                    "velocity_vector_rmse_m_per_s": float(np.sqrt(np.mean(np.array(vel_vec_errs) ** 2))) if vel_vec_errs else 0.0,
                    "acceleration_vector_rmse_m_per_s2": float(np.sqrt(np.mean(np.array(accel_vec_errs) ** 2))) if accel_vec_errs else 0.0,
                    # Canonical counts and sums (kept in output for CSV)
                    "_n_traj": len(devs_m), "_traj_sum_m": float(sum(devs_m)),
                    "_n_vel": len(vel_vec_errs), "_vel_sum": float(sum(vel_vec_errs)),
                    "_vel_sum_sq": float(sum(v**2 for v in vel_vec_errs)),
                    "_n_accel": len(accel_vec_errs), "_accel_sum": float(sum(accel_vec_errs)),
                    "_accel_sum_sq": float(sum(v**2 for v in accel_vec_errs)),
                    # Internal: raw error arrays for pooled aggregate (removed before return)
                    "_raw_devs_m": devs_m,
                    "_raw_vel_vec_errs": vel_vec_errs,
                    "_raw_accel_vec_errs": accel_vec_errs,
                }
                per_landmark.append(entry)

        # Per-landmark timing using shared helper
        timing = _compute_timing_for_speeds(speeds_raw_arr, speeds_filt_arr, bd_flags_arr, ts, list(range(n_key)))
        # Map back to actual landmark indices
        for ki, li in enumerate(key_lms):
            for lm_t in timing["per_landmark"]:
                if lm_t["landmark"] == ki:
                    lm_t["landmark"] = int(li)
                    # Find matching per_landmark entry and add timing
                    for pl in per_landmark:
                        if pl["component"] == comp_name and pl["landmark"] == int(li):
                            pl["timing"] = {
                                "n_segments": lm_t["n_segments"],
                                "n_nonneg_matched": lm_t["n_nonneg_matched"],
                                "n_negative_inconsistent": lm_t.get("n_negative_inconsistent", 0),
                                "n_unmatched_raw": lm_t.get("n_unmatched_raw", 0),
                                "n_unmatched_filt": lm_t.get("n_unmatched_filt", 0),
                                "mean_lag_frames": lm_t.get("mean_lag_frames", "no_nonneg_matches"),
                                "mean_lag_ms": lm_t.get("mean_lag_ms", "no_nonneg_matches"),
                                "status": lm_t["status"],
                                "matched_events": lm_t.get("matched_events", []),
                                "unmatched_raw_events": lm_t.get("unmatched_raw_events", []),
                                "unmatched_filt_events": lm_t.get("unmatched_filt_events", []),
                                "segments": lm_t.get("segments", []),
                            }

    # Idle jitter
    idle_jitter = _compute_transform_idle_jitter(arrs, norm, stab, ts, seg_info)

    # Helper: build timing/onset/end/xcorr aggregate from per-landmark timing data
    def _build_timing_aggregate(lm_list):
        peak_nonneg, peak_neg, peak_unmatched_raw, peak_unmatched_filt = 0, 0, 0, 0
        peak_lags_frames, peak_lags_ms = [], []
        onset_nonneg, onset_neg, end_nonneg, end_neg = 0, 0, 0, 0
        xcorr_nonneg, xcorr_neg = 0, 0
        onset_frames, onset_ms, end_frames, end_ms, xcorr_frames = [], [], [], [], []
        for pl in lm_list:
            t = pl.get("timing", {})
            peak_nonneg += t.get("n_nonneg_matched", 0)
            peak_neg += t.get("n_negative_inconsistent", 0)
            peak_unmatched_raw += t.get("n_unmatched_raw", 0)
            peak_unmatched_filt += t.get("n_unmatched_filt", 0)
            for m in t.get("matched_events", []):
                if m.get("status") == "matched":
                    peak_lags_frames.append(m["lag_frames"])
                    if "lag_ms" in m: peak_lags_ms.append(m["lag_ms"])
            for seg_t in t.get("segments", []):
                ol = seg_t.get("onset_lag", {})
                ol_s = ol.get("status", "")
                if ol_s == "nonnegative":
                    onset_nonneg += 1
                    if "lag_frames" in ol: onset_frames.append(ol["lag_frames"])
                    if "lag_ms" in ol: onset_ms.append(ol["lag_ms"])
                elif ol_s.startswith("negative"): onset_neg += 1
                el = seg_t.get("end_lag", {})
                el_s = el.get("status", "")
                if el_s == "nonnegative":
                    end_nonneg += 1
                    if "lag_frames" in el: end_frames.append(el["lag_frames"])
                    if "lag_ms" in el: end_ms.append(el["lag_ms"])
                elif el_s.startswith("negative"): end_neg += 1
                xl = seg_t.get("xcorr_lag", {})
                xl_s = xl.get("status", "")
                if xl_s == "nonnegative":
                    xcorr_nonneg += 1
                    if "lag_frames" in xl: xcorr_frames.append(xl["lag_frames"])
                elif xl_s.startswith("negative"): xcorr_neg += 1
        ta = {
            "peak_nonneg": peak_nonneg, "peak_negative": peak_neg,
            "peak_unmatched_raw": peak_unmatched_raw, "peak_unmatched_filt": peak_unmatched_filt,
            "peak_pooled_mean_frames": float(np.mean(peak_lags_frames)) if peak_lags_frames else "no_valid_nonnegative_events",
            "peak_pooled_mean_ms": float(np.mean(peak_lags_ms)) if peak_lags_ms else "no_valid_nonnegative_events",
            "onset_nonneg": onset_nonneg, "onset_negative": onset_neg,
            "onset_status": "has_nonnegative" if onset_nonneg > 0 else "no_valid_nonnegative_events",
            "onset_pooled_mean_frames": float(np.mean(onset_frames)) if onset_frames else "no_valid_nonnegative_events",
            "onset_pooled_mean_ms": float(np.mean(onset_ms)) if onset_ms else "no_valid_nonnegative_events",
            "end_nonneg": end_nonneg, "end_negative": end_neg,
            "end_status": "has_nonnegative" if end_nonneg > 0 else "no_valid_nonnegative_events",
            "end_pooled_mean_frames": float(np.mean(end_frames)) if end_frames else "no_valid_nonnegative_events",
            "end_pooled_mean_ms": float(np.mean(end_ms)) if end_ms else "no_valid_nonnegative_events",
            "xcorr_nonneg": xcorr_nonneg, "xcorr_negative": xcorr_neg,
            "xcorr_status": "has_nonnegative" if xcorr_nonneg > 0 else "no_valid_nonnegative_events",
            "xcorr_pooled_mean_frames": float(np.mean(xcorr_frames)) if xcorr_frames else "no_valid_nonnegative_events",
        }
        return ta

    # Per-component aggregates from raw distributions
    comp_aggregates = {}
    for comp_name_agg in ["pose", "left_hand", "right_hand"]:
        comp_lms = [r for r in per_landmark if r["component"] == comp_name_agg]
        all_d = [v for r in comp_lms for v in r.get("_raw_devs_m", [])]
        all_v = [v for r in comp_lms for v in r.get("_raw_vel_vec_errs", [])]
        all_a = [v for r in comp_lms for v in r.get("_raw_accel_vec_errs", [])]
        ca = {}
        if all_d:
            da = np.array(all_d)
            ca.update({"traj_n": len(da), "traj_sum_m": float(np.sum(da)),
                        "traj_mean_mm": float(np.mean(da) * 1000),
                        "traj_p95_mm": float(np.percentile(da, 95) * 1000), "traj_max_mm": float(np.max(da) * 1000)})
        if all_v:
            va = np.array(all_v)
            ca.update({"vel_n": len(va), "vel_sum": float(np.sum(va)), "vel_sum_sq": float(np.sum(va**2)),
                        "vel_rmse": float(np.sqrt(np.mean(va**2)))})
        if all_a:
            aa = np.array(all_a)
            ca.update({"accel_n": len(aa), "accel_sum": float(np.sum(aa)), "accel_sum_sq": float(np.sum(aa**2)),
                        "accel_rmse": float(np.sqrt(np.mean(aa**2)))})
        # Path sums from per-landmark
        path_raw_sum = sum(r.get("path_length_raw_m", 0) for r in comp_lms)
        path_stab_sum = sum(r.get("path_length_stab_m", 0) for r in comp_lms)
        ca["path_raw_sum_m"] = float(path_raw_sum)
        ca["path_stab_sum_m"] = float(path_stab_sum)
        ca["path_ratio"] = float(path_stab_sum / path_raw_sum) if path_raw_sum > 0 else 1.0
        # Timing aggregate
        ca["timing"] = _build_timing_aggregate(comp_lms)
        comp_aggregates[comp_name_agg] = ca

    # Sample-level pooled aggregate from raw distributions
    all_raw_devs = [v for r in per_landmark for v in r.get("_raw_devs_m", [])]
    all_raw_vel = [v for r in per_landmark for v in r.get("_raw_vel_vec_errs", [])]
    all_raw_accel = [v for r in per_landmark for v in r.get("_raw_accel_vec_errs", [])]

    aggregate = {}
    if all_raw_devs:
        da = np.array(all_raw_devs)
        aggregate["traj_n"] = len(da)
        aggregate["traj_sum_m"] = float(np.sum(da))
        aggregate["traj_dev_mean_mm"] = float(np.mean(da) * 1000)
        aggregate["traj_dev_p95_mm"] = float(np.percentile(da, 95) * 1000)
        aggregate["traj_dev_max_mm"] = float(np.max(da) * 1000)
    else:
        aggregate["traj_n"] = 0; aggregate["traj_sum_m"] = 0
        aggregate["traj_dev_mean_mm"] = 0; aggregate["traj_dev_p95_mm"] = 0; aggregate["traj_dev_max_mm"] = 0
    if all_raw_vel:
        va = np.array(all_raw_vel)
        aggregate["vel_n"] = len(va)
        aggregate["vel_sum"] = float(np.sum(va))
        aggregate["vel_sum_sq"] = float(np.sum(va**2))
        aggregate["velocity_vector_rmse_m_per_s"] = float(np.sqrt(np.mean(va**2)))
    else:
        aggregate["vel_n"] = 0; aggregate["vel_sum"] = 0; aggregate["vel_sum_sq"] = 0
        aggregate["velocity_vector_rmse_m_per_s"] = 0
    if all_raw_accel:
        aa = np.array(all_raw_accel)
        aggregate["accel_n"] = len(aa)
        aggregate["accel_sum"] = float(np.sum(aa))
        aggregate["accel_sum_sq"] = float(np.sum(aa**2))
        aggregate["acceleration_vector_rmse_m_per_s2"] = float(np.sqrt(np.mean(aa**2)))
    else:
        aggregate["accel_n"] = 0; aggregate["accel_sum"] = 0; aggregate["accel_sum_sq"] = 0
        aggregate["acceleration_vector_rmse_m_per_s2"] = 0
    # Path sums (sample-level = sum of all component path sums)
    aggregate["path_raw_sum_m"] = float(sum(ca.get("path_raw_sum_m", 0) for ca in comp_aggregates.values()))
    aggregate["path_stab_sum_m"] = float(sum(ca.get("path_stab_sum_m", 0) for ca in comp_aggregates.values()))
    aggregate["path_ratio"] = float(aggregate["path_stab_sum_m"] / aggregate["path_raw_sum_m"]) if aggregate["path_raw_sum_m"] > 0 else 1.0
    # Timing aggregate (sample-level)
    sample_timing = _build_timing_aggregate(per_landmark)
    aggregate["timing"] = sample_timing
    # Legacy flat fields for backward compatibility
    aggregate["timing_aggregate_lag_frames"] = sample_timing["peak_pooled_mean_frames"]
    aggregate["timing_n_nonneg_matched"] = sample_timing["peak_nonneg"]
    aggregate["selected_normalization_transform"] = SELECTED_NORM_TRANSFORM
    aggregate["boundary_edges_skipped"] = boundary_edges_skipped

    # NOTE: _raw_devs_m, _raw_vel_vec_errs, _raw_accel_vec_errs kept in per_landmark
    # for assertion recomputation (P95/max from raw arrays). Cleaned after assertions in main().

    return {"per_landmark": per_landmark, "aggregate": aggregate,
            "component_aggregates": comp_aggregates, "idle_jitter": idle_jitter}


def _compute_transform_idle_jitter(arrs, norm, stab, ts, seg_info):
    idle_jitter = {}
    ss, se = seg_info["seg_start"], seg_info["seg_end"]; N = len(ts)
    for region, r_start, r_end, verified in [("start", 0, ss, seg_info["idle_verified_start"]),
                                              ("end", se + 1, N, seg_info["idle_verified_end"])]:
        if not verified or r_end <= r_start:
            idle_jitter[f"{region}_raw_mm"] = "not_available_no_verified_idle"
            idle_jitter[f"{region}_stabilized_mm"] = "not_available_no_verified_idle"
            continue
        jr, js = [], []
        for fi in range(max(1, r_start), min(r_end, N)):
            dt = ts[fi] - ts[fi - 1]
            if dt <= 0 or dt > 0.1: continue
            valid_c = arrs["pose_te"]
            for li in POSE_KEY_LMS:
                if valid_c[fi, li] and valid_c[fi - 1, li]:
                    xyz = arrs["pose_xyz"]
                    if not np.any(np.isnan(xyz[fi, li])) and not np.any(np.isnan(xyz[fi - 1, li])):
                        sc = norm["scales"][fi] if not np.isnan(norm["scales"][fi]) else 0
                        r0 = normalize_with_transform(xyz[fi-1:fi, li:li+1], valid_c[fi-1:fi, li:li+1],
                                                      norm["origins"][fi-1:fi], norm["rotations"][fi-1:fi], norm["scales"][fi-1:fi])
                        r1 = normalize_with_transform(xyz[fi:fi+1, li:li+1], valid_c[fi:fi+1, li:li+1],
                                                      norm["origins"][fi:fi+1], norm["rotations"][fi:fi+1], norm["scales"][fi:fi+1])
                        if not np.any(np.isnan(r0)) and not np.any(np.isnan(r1)):
                            jr.append(np.linalg.norm(r1[0,0] - r0[0,0]) * sc)
                        s0 = normalize_with_transform(xyz[fi-1:fi, li:li+1], valid_c[fi-1:fi, li:li+1],
                                                      stab["stab_origins"][fi-1:fi], stab["stab_rotations"][fi-1:fi], stab["stab_scales"][fi-1:fi])
                        s1 = normalize_with_transform(xyz[fi:fi+1, li:li+1], valid_c[fi:fi+1, li:li+1],
                                                      stab["stab_origins"][fi:fi+1], stab["stab_rotations"][fi:fi+1], stab["stab_scales"][fi:fi+1])
                        if not np.any(np.isnan(s0)) and not np.any(np.isnan(s1)):
                            js.append(np.linalg.norm(s1[0,0] - s0[0,0]) * sc)
        idle_jitter[f"{region}_raw_mm"] = float(np.mean(jr) * 1000) if jr else "not_available"
        idle_jitter[f"{region}_stabilized_mm"] = float(np.mean(js) * 1000) if js else "not_available"
    return idle_jitter

# ================================================================
#  Segmentation (unchanged)
# ================================================================

def detect_segment(arrs, norm):
    N = len(arrs["timestamps"]); ts = arrs["timestamps"]
    energy = np.zeros(N)
    motion_lms = [LM_LE, LM_RE, LM_LW, LM_RW]; tips = [4, 8, 12, 16, 20]
    for fi in range(1, N):
        dt = ts[fi] - ts[fi - 1]
        if dt <= 0 or dt > 0.2: continue
        e, c = 0.0, 0
        for li in motion_lms:
            if not np.any(np.isnan(norm["norm_pose"][fi, li])) and not np.any(np.isnan(norm["norm_pose"][fi - 1, li])):
                e += np.linalg.norm(norm["norm_pose"][fi, li] - norm["norm_pose"][fi - 1, li]); c += 1
        for hk in ["norm_lh", "norm_rh"]:
            w0, w1 = norm[hk][fi - 1, 0], norm[hk][fi, 0]
            if not np.any(np.isnan(w0)) and not np.any(np.isnan(w1)): e += np.linalg.norm(w1 - w0); c += 1
            for t in tips:
                t0, t1 = norm[hk][fi - 1, t], norm[hk][fi, t]
                if not np.any(np.isnan(t0)) and not np.any(np.isnan(t1)): e += np.linalg.norm(t1 - t0) * 0.5; c += 0.5
        if c > 0: energy[fi] = e / c
    kernel = np.ones(SEG_ENERGY_WINDOW) / SEG_ENERGY_WINDOW
    energy_sm = np.convolve(energy, kernel, mode="same")
    active = energy_sm > SEG_IDLE_THRESHOLD; ai = np.where(active)[0]
    if len(ai) < SEG_MIN_ACTIVE:
        return {"seg_start": 0, "seg_end": N - 1, "seg_frames": N, "total_frames": N,
                "raw_start": 0, "raw_end": N - 1, "method": "source_already_trimmed_no_idle_available",
                "energy": energy, "energy_smooth": energy_sm, "threshold": SEG_IDLE_THRESHOLD,
                "idle_verified_start": False, "idle_verified_end": False}
    rs, re = int(ai[0]), int(ai[-1])
    ss = max(0, rs - SEG_PAD_START); se = min(N - 1, re + SEG_PAD_END)
    return {"seg_start": ss, "seg_end": se, "seg_frames": se - ss + 1, "total_frames": N,
            "raw_start": rs, "raw_end": re, "method": "motion_energy",
            "energy": energy, "energy_smooth": energy_sm, "threshold": SEG_IDLE_THRESHOLD,
            "idle_verified_start": ss > 0, "idle_verified_end": se < N - 1}

# ================================================================
#  Boundaries + Reasons (unchanged)
# ================================================================

def build_all_boundaries(arrs, strong_depth_set, sid):
    N = len(arrs["timestamps"]); ts = arrs["timestamps"]
    frame_disc = np.zeros(N, dtype=bool); frame_disc[0] = True
    frame_ts_disc = np.zeros(N, dtype=bool)
    for fi in range(1, N):
        dt = ts[fi] - ts[fi - 1]
        if dt <= 0 or dt > 0.1: frame_disc[fi] = True; frame_ts_disc[fi] = True
    boundaries = {}; boundary_reasons = {}; strong_forbidden = {}
    for comp, nlm in [("pose", 33), ("lh", 21), ("rh", 21)]:
        bd = np.zeros((N, nlm), dtype=bool); br = np.full((N, nlm), RSN_NONE, dtype=np.int8)
        bd[0, :] = True; br[0, :] = RSN_INITIAL
        for fi in range(1, N):
            if frame_ts_disc[fi]:
                bd[fi, :] = True; br[fi, :] = np.where(br[fi, :] == RSN_NONE, RSN_TIMESTAMP_DISC, br[fi, :])
        reapp = arrs[f"{comp}_reappear"]
        for fi in range(1, N):
            for li in range(nlm):
                if reapp[fi, li]: bd[fi, li] = True; br[fi, li] = RSN_REAPPEAR
        for (sfi, scomp, sli) in strong_depth_set:
            if scomp == comp and 0 <= sfi < N and 0 <= sli < nlm:
                bd[sfi, sli] = True; br[sfi, sli] = RSN_STRONG_DEPTH
        boundaries[comp] = bd; boundary_reasons[comp] = br
        sf = np.zeros((N, nlm), dtype=bool)
        for (sfi, scomp, sli) in strong_depth_set:
            if scomp == comp and 0 <= sfi < N and 0 <= sli < nlm: sf[sfi, sli] = True
        strong_forbidden[comp] = sf
    return frame_disc, boundaries, boundary_reasons, strong_forbidden

# ================================================================
#  Interpolation (unchanged)
# ================================================================

def interpolate_gaps(xyz, valid, boundary, strong_forbidden, max_gap,
                     invalid_reasons, qa_flags_arr, sd_prov_map, comp_name, sid):
    N, nlm, _ = xyz.shape
    out = xyz.copy(); mask = np.zeros((N, nlm), dtype=bool); rejected = np.zeros((N, nlm), dtype=bool)
    records = []
    for li in range(nlm):
        segs = []; ss = 0
        for fi in range(1, N):
            if boundary[fi, li]: segs.append((ss, fi - 1)); ss = fi
        segs.append((ss, N - 1))
        for s0, s1 in segs:
            fi = s0
            while fi <= s1:
                if not valid[fi, li]:
                    gs = fi
                    while fi <= s1 and not valid[fi, li]: fi += 1
                    ge = fi - 1; gl = ge - gs + 1
                    has_before = gs > s0 and valid[gs - 1, li] and not np.any(np.isnan(xyz[gs - 1, li]))
                    has_after = fi <= s1 and valid[fi, li] and not np.any(np.isnan(xyz[fi, li]))
                    touches_strong = any(strong_forbidden[gs + g, li] for g in range(gl))
                    reasons = []
                    if gl > max_gap: reasons.append(f"exceeds_max_gap_{max_gap}")
                    if not has_before: reasons.append("no_valid_predecessor")
                    if not has_after: reasons.append("no_valid_successor")
                    if touches_strong: reasons.append("strong_depth_forbidden")
                    gap_provenance = []
                    for g in range(gl):
                        gfi = gs + g
                        prov = {"frame_index": int(gfi), "landmark_index": int(li),
                                "invalid_reason": invalid_reasons[gfi][li], "qa_flags": qa_flags_arr[gfi][li]}
                        sd_key = (gfi, comp_name, li)
                        if sd_key in sd_prov_map: prov["strong_depth"] = sd_prov_map[sd_key]
                        gap_provenance.append(prov)
                    if not reasons and has_before and has_after and gl <= max_gap:
                        p0 = out[gs - 1, li]; p1 = out[fi, li]
                        for g in range(gl):
                            t = (g + 1) / (gl + 1)
                            out[gs + g, li] = p0 * (1 - t) + p1 * t; mask[gs + g, li] = True
                        records.append({"lm_index": int(li), "gap_start": int(gs), "gap_end": int(ge),
                            "gap_length": int(gl), "method": "linear", "max_gap": int(max_gap),
                            "seg_start": int(s0), "seg_end": int(s1),
                            "endpoint_before_fi": int(gs - 1), "endpoint_after_fi": int(fi),
                            "endpoint_before_xyz": p0.tolist(), "endpoint_after_xyz": p1.tolist(),
                            "interp_distance_m": float(np.linalg.norm(p1 - p0)),
                            "gap_provenance": gap_provenance})
                    else:
                        for g in range(gl): rejected[gs + g, li] = True
                        records.append({"lm_index": int(li), "gap_start": int(gs), "gap_end": int(ge),
                            "gap_length": int(gl), "method": "rejected", "max_gap": int(max_gap),
                            "rejection_reason": "|".join(reasons) if reasons else "unknown",
                            "seg_start": int(s0), "seg_end": int(s1), "gap_provenance": gap_provenance})
                else: fi += 1
    return out, mask, rejected, records

# ================================================================
#  OneEuro / SavGol / filter masks (unchanged)
# ================================================================

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff; self.beta = beta; self.d_cutoff = d_cutoff
        self.x_prev = None; self.dx_prev = None; self.t_prev = None
    def reset(self): self.x_prev = None; self.dx_prev = None; self.t_prev = None
    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x.copy(); self.dx_prev = np.zeros_like(x); self.t_prev = t; return x.copy()
        dt = t - self.t_prev
        if dt <= 0: return self.x_prev.copy()
        dx = (x - self.x_prev) / dt
        tau_d = 1.0 / (2 * math.pi * self.d_cutoff); a_d = 1.0 / (1.0 + tau_d / dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        speed = np.linalg.norm(dx_hat); cutoff = self.min_cutoff + self.beta * speed
        tau = 1.0 / (2 * math.pi * cutoff); a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat.copy(); self.dx_prev = dx_hat.copy(); self.t_prev = t; return x_hat

def apply_one_euro(xyz, valid, ts, boundary, boundary_reason, params):
    N, nlm, _ = xyz.shape; out = xyz.copy()
    reset_mask = np.zeros((N, nlm), dtype=bool); far_mask = np.zeros((N, nlm), dtype=bool)
    reset_rsn = np.full((N, nlm), RSN_NONE, dtype=np.int8)
    for li in range(nlm):
        filt = OneEuroFilter(params["min_cutoff"], params["beta"], params["d_cutoff"]); pending = True
        for fi in range(N):
            if boundary[fi, li]:
                filt.reset(); reset_mask[fi, li] = True
                reset_rsn[fi, li] = boundary_reason[fi, li] if boundary_reason[fi, li] > 0 else RSN_OTHER; pending = True
            if valid[fi, li] and not np.any(np.isnan(xyz[fi, li])):
                result = filt(xyz[fi, li], ts[fi])
                if pending: far_mask[fi, li] = True; pending = False
                out[fi, li] = result
            else:
                filt.reset()
                if not reset_mask[fi, li]: reset_mask[fi, li] = True; reset_rsn[fi, li] = RSN_INVALID_OBS
                pending = True
    return out, reset_mask, far_mask, reset_rsn

def apply_savgol(xyz, valid, boundary, window=SAVGOL_WINDOW, order=SAVGOL_ORDER):
    if not HAS_SCIPY: return xyz.copy()
    N, nlm, _ = xyz.shape; out = xyz.copy()
    for li in range(nlm):
        ss = None
        for fi in range(N + 1):
            is_break = fi == N or boundary[fi, li] or not valid[fi if fi < N else 0, li] or np.any(np.isnan(xyz[fi if fi < N else 0, li]))
            if fi < N and not is_break:
                if ss is None: ss = fi
            else:
                if ss is not None:
                    slen = fi - ss
                    if slen >= window:
                        for dim in range(3): out[ss:fi, li, dim] = savgol_filter(xyz[ss:fi, li, dim], window, order)
                    ss = None
    return out

def compute_filter_masks(xyz, valid, boundary, boundary_reason):
    N, nlm = valid.shape
    reset_mask = np.zeros((N, nlm), dtype=bool); far_mask = np.zeros((N, nlm), dtype=bool)
    reset_rsn = np.full((N, nlm), RSN_NONE, dtype=np.int8)
    for li in range(nlm):
        pending = True
        for fi in range(N):
            is_bd = boundary[fi, li]; is_ok = valid[fi, li] and not np.any(np.isnan(xyz[fi, li]))
            if is_bd:
                reset_mask[fi, li] = True
                reset_rsn[fi, li] = boundary_reason[fi, li] if boundary_reason[fi, li] > 0 else RSN_OTHER; pending = True
            if is_ok:
                if pending: far_mask[fi, li] = True; pending = False
            else:
                if not is_bd: reset_mask[fi, li] = True; reset_rsn[fi, li] = RSN_INVALID_OBS
                pending = True
    return reset_mask, far_mask, reset_rsn

def recompute_first_after_reset(xyz, valid, boundary, boundary_reason):
    N, nlm = valid.shape
    reset_mask = np.zeros((N, nlm), dtype=bool); far_mask = np.zeros((N, nlm), dtype=bool)
    reset_rsn = np.full((N, nlm), RSN_NONE, dtype=np.int8)
    for li in range(nlm):
        pending = True
        for fi in range(N):
            is_bd = boundary[fi, li]; is_ok = valid[fi, li] and not np.any(np.isnan(xyz[fi, li]))
            if is_bd:
                reset_mask[fi, li] = True
                reset_rsn[fi, li] = boundary_reason[fi, li] if boundary_reason[fi, li] > 0 else RSN_OTHER; pending = True
            if is_ok:
                if pending: far_mask[fi, li] = True; pending = False
            else:
                if not is_bd: reset_mask[fi, li] = True; reset_rsn[fi, li] = RSN_INVALID_OBS
                pending = True
    return reset_mask, far_mask, reset_rsn

# ================================================================
#  Extended Metrics (v1.5: uses shared timing helper)
# ================================================================

def compute_extended_metrics(norm_base, norm_filt, valid, scales, ts, boundary, seg_info,
                             label, word, comp_nlm):
    N, nlm, _ = norm_base.shape
    key_lms = POSE_KEY_LMS if nlm == 33 else HAND_KEY_LMS
    phys_devs = []
    for fi in range(N):
        s = scales[fi] if not np.isnan(scales[fi]) else 0
        for li in range(nlm):
            if valid[fi, li] and not np.any(np.isnan(norm_base[fi, li])) and not np.any(np.isnan(norm_filt[fi, li])):
                phys_devs.append(np.linalg.norm(norm_filt[fi, li] - norm_base[fi, li]) * s)
    if not phys_devs: return _empty_metrics(label)
    phys_devs = np.array(phys_devs)

    j_start = _compute_idle_jitter(norm_base, norm_filt, valid, scales, 0, seg_info["seg_start"]) \
        if seg_info["idle_verified_start"] else None
    j_end = _compute_idle_jitter(norm_base, norm_filt, valid, scales, seg_info["seg_end"] + 1, N) \
        if seg_info["idle_verified_end"] else None

    vel_raw_all, vel_filt_all, vel_diffs = [], [], []
    accel_diffs = []; path_raw, path_filt = 0.0, 0.0
    per_lm_results = []
    # Build speed arrays for timing
    n_key = len(key_lms)
    speeds_raw_arr = np.zeros((N, n_key)); speeds_filt_arr = np.zeros((N, n_key))
    bd_flags_arr = np.zeros((N, n_key), dtype=bool); bd_flags_arr[0, :] = True

    for ki, li in enumerate(key_lms):
        lm_path_raw, lm_path_filt = 0.0, 0.0; prev_rv, prev_fv = None, None
        for fi in range(1, N):
            dt = ts[fi] - ts[fi - 1]
            if dt <= 0 or dt > 0.1 or boundary[fi, li]:
                prev_rv = prev_fv = None; bd_flags_arr[fi, ki] = True; continue
            if not valid[fi, li] or not valid[fi - 1, li]:
                prev_rv = prev_fv = None; bd_flags_arr[fi, ki] = True; continue
            s = scales[fi] if not np.isnan(scales[fi]) else 0
            r0, r1 = norm_base[fi - 1, li], norm_base[fi, li]
            f0, f1 = norm_filt[fi - 1, li], norm_filt[fi, li]
            if np.any(np.isnan(r0)) or np.any(np.isnan(r1)) or np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                prev_rv = prev_fv = None; bd_flags_arr[fi, ki] = True; continue
            rv = np.linalg.norm(r1 - r0) * s / dt; fv = np.linalg.norm(f1 - f0) * s / dt
            vel_raw_all.append(rv); vel_filt_all.append(fv); vel_diffs.append(fv - rv)
            lm_path_raw += np.linalg.norm(r1 - r0) * s; lm_path_filt += np.linalg.norm(f1 - f0) * s
            speeds_raw_arr[fi, ki] = rv; speeds_filt_arr[fi, ki] = fv
            rv_vec = (r1 - r0) * s / dt; fv_vec = (f1 - f0) * s / dt
            if prev_rv is not None:
                ra = np.linalg.norm(rv_vec - prev_rv) / dt; fa = np.linalg.norm(fv_vec - prev_fv) / dt
                accel_diffs.append(fa - ra)
            prev_rv, prev_fv = rv_vec, fv_vec
        path_raw += lm_path_raw; path_filt += lm_path_filt
        per_lm_results.append({"landmark": li, "path_raw_m": lm_path_raw, "path_filt_m": lm_path_filt,
            "path_ratio": lm_path_filt / lm_path_raw if lm_path_raw > 0 else 1.0})

    vel_rmse = float(np.sqrt(np.mean(np.array(vel_diffs) ** 2))) if vel_diffs else 0
    vel_dist = float(abs(np.mean(vel_filt_all) - np.mean(vel_raw_all)) / np.mean(vel_raw_all) * 100) \
        if vel_raw_all and np.mean(vel_raw_all) > 0 else 0
    accel_rmse = float(np.sqrt(np.mean(np.array(accel_diffs) ** 2))) if accel_diffs else 0

    # Timing via shared helper (v1.5)
    peak_lag = _compute_timing_for_speeds(speeds_raw_arr, speeds_filt_arr, bd_flags_arr, ts, list(range(n_key)))
    # Map landmark indices back
    for lm_t in peak_lag.get("per_landmark", []):
        if lm_t["landmark"] < len(key_lms): lm_t["landmark"] = key_lms[lm_t["landmark"]]

    new_nans = sum(1 for fi in range(N) for li in range(nlm) if valid[fi, li]
                   and not np.any(np.isnan(norm_base[fi, li])) and np.any(np.isnan(norm_filt[fi, li])))
    new_infs = sum(1 for fi in range(N) for li in range(nlm) if valid[fi, li]
                   and not np.any(np.isnan(norm_base[fi, li])) and np.any(np.isinf(norm_filt[fi, li])))

    return {
        "label": label, "n_valid": len(phys_devs),
        "phys_dev_mean_mm": float(np.mean(phys_devs) * 1000),
        "phys_dev_p95_mm": float(np.percentile(phys_devs, 95) * 1000),
        "phys_dev_max_mm": float(np.max(phys_devs) * 1000),
        "vel_distortion_pct": vel_dist, "vel_rmse_m_per_s": vel_rmse,
        "accel_rmse_m_per_s2": accel_rmse,
        "path_length_ratio": float(path_filt / path_raw) if path_raw > 0 else 1.0,
        "jitter_raw_start_idle_mm": j_start[0] if j_start else "not_available_no_verified_idle",
        "jitter_filt_start_idle_mm": j_start[1] if j_start else "not_available_no_verified_idle",
        "jitter_reduction_start_idle": j_start[2] if j_start else "not_available_no_verified_idle",
        "jitter_raw_end_idle_mm": j_end[0] if j_end else "not_available_no_verified_idle",
        "jitter_filt_end_idle_mm": j_end[1] if j_end else "not_available_no_verified_idle",
        "jitter_reduction_end_idle": j_end[2] if j_end else "not_available_no_verified_idle",
        "peak_vel_lag_frames": peak_lag.get("aggregate_lag_frames", "peak_correspondence_failed"),
        "peak_vel_lag_ms": peak_lag.get("aggregate_lag_ms", "peak_correspondence_failed"),
        "peak_match_status": peak_lag.get("status", "no_data"),
        "new_nans": new_nans, "new_infs": new_infs,
        "per_landmark": per_lm_results, "peak_lag_detail": peak_lag}

def _empty_metrics(label):
    return {"label": label, "n_valid": 0,
            "phys_dev_mean_mm": 0, "phys_dev_p95_mm": 0, "phys_dev_max_mm": 0,
            "vel_distortion_pct": 0, "vel_rmse_m_per_s": 0, "accel_rmse_m_per_s2": 0,
            "path_length_ratio": 1.0, "new_nans": 0, "new_infs": 0,
            "jitter_raw_start_idle_mm": "not_available_no_verified_idle",
            "jitter_filt_start_idle_mm": "not_available_no_verified_idle",
            "jitter_reduction_start_idle": "not_available_no_verified_idle",
            "jitter_raw_end_idle_mm": "not_available_no_verified_idle",
            "jitter_filt_end_idle_mm": "not_available_no_verified_idle",
            "jitter_reduction_end_idle": "not_available_no_verified_idle",
            "peak_vel_lag_frames": "no_data", "peak_vel_lag_ms": "no_data",
            "peak_match_status": "no_data", "per_landmark": [],
            "peak_lag_detail": {"status": "no_data"}}

def _compute_idle_jitter(norm_base, norm_filt, valid, scales, start, end):
    N = norm_base.shape[0]; nlm = norm_base.shape[1]; jr, jf = [], []
    for fi in range(max(1, start), min(end, N)):
        s = scales[fi] if not np.isnan(scales[fi]) else 0
        for li in range(nlm):
            if valid[fi, li] and valid[fi - 1, li]:
                r0, r1 = norm_base[fi - 1, li], norm_base[fi, li]
                f0, f1 = norm_filt[fi - 1, li], norm_filt[fi, li]
                if not np.any(np.isnan(r0)) and not np.any(np.isnan(r1)): jr.append(np.linalg.norm(r1 - r0) * s)
                if not np.any(np.isnan(f0)) and not np.any(np.isnan(f1)): jf.append(np.linalg.norm(f1 - f0) * s)
    if jr and jf:
        return (float(np.mean(jr) * 1000), float(np.mean(jf) * 1000),
                float(1 - np.mean(jf) / np.mean(jr)) if np.mean(jr) > 0 else 0.0)
    return None

# ================================================================
#  Sign-specific Metrics (unchanged)
# ================================================================

def compute_sign_metrics(norm_base, norm_filt, valid, arrs, norm_dict, seg, word, ts, scales, boundary):
    N = norm_base.shape[0]; result = {}; ss, se = seg["seg_start"], seg["seg_end"]
    if word == "LOVE" and norm_base.shape[1] >= 33:
        for label, lw, rw in [("pose_wrists", LM_LW, LM_RW)]:
            dists_raw, dists_filt, dists_raw_fi, dists_filt_fi = [], [], [], []
            for fi in range(ss, min(se + 1, N)):
                if valid[fi, lw] and valid[fi, rw]:
                    s = scales[fi] if not np.isnan(scales[fi]) else 0
                    lr, rr = norm_base[fi, lw], norm_base[fi, rw]; lf, rf = norm_filt[fi, lw], norm_filt[fi, rw]
                    if not np.any(np.isnan(lr)) and not np.any(np.isnan(rr)): dists_raw.append(np.linalg.norm(rr - lr) * s); dists_raw_fi.append(fi)
                    if not np.any(np.isnan(lf)) and not np.any(np.isnan(rf)): dists_filt.append(np.linalg.norm(rf - lf) * s); dists_filt_fi.append(fi)
            if dists_raw and dists_filt:
                mr_fi = dists_raw_fi[int(np.argmin(dists_raw))]; mf_fi = dists_filt_fi[int(np.argmin(dists_filt))]
                result["love_min_hand_dist_raw_mm"] = float(np.min(dists_raw) * 1000)
                result["love_min_hand_dist_filt_mm"] = float(np.min(dists_filt) * 1000)
                result["love_min_dist_timing_shift_ms"] = float((ts[mf_fi] - ts[mr_fi]) * 1000)
    elif word == "GREETING" and norm_base.shape[1] >= 33:
        bd = boundary
        for wname, wli in [("left_wrist", LM_LW), ("right_wrist", LM_RW)]:
            raw_reversals, filt_reversals = [], []; prev_vy_raw, prev_vy_filt = None, None
            for fi in range(ss + 1, min(se + 1, N)):
                if bd[fi, wli] or not valid[fi, wli] or not valid[fi - 1, wli]: prev_vy_raw = prev_vy_filt = None; continue
                r0y = norm_base[fi - 1, wli, 1]; r1y = norm_base[fi, wli, 1]
                f0y = norm_filt[fi - 1, wli, 1]; f1y = norm_filt[fi, wli, 1]
                vy_raw = (r1y - r0y) if not (np.isnan(r0y) or np.isnan(r1y)) else None
                vy_filt = (f1y - f0y) if not (np.isnan(f0y) or np.isnan(f1y)) else None
                if vy_raw is not None and prev_vy_raw is not None and vy_raw * prev_vy_raw < 0:
                    raw_reversals.append({"frame": fi, "time_s": float(ts[fi])})
                if vy_filt is not None and prev_vy_filt is not None and vy_filt * prev_vy_filt < 0:
                    filt_reversals.append({"frame": fi, "time_s": float(ts[fi])})
                if vy_raw is not None: prev_vy_raw = vy_raw
                if vy_filt is not None: prev_vy_filt = vy_filt
            result[f"greeting_{wname}_reversals_raw"] = len(raw_reversals)
            result[f"greeting_{wname}_reversals_filt"] = len(filt_reversals)
            result[f"greeting_{wname}_reversal_times_raw"] = raw_reversals
            result[f"greeting_{wname}_reversal_times_filt"] = filt_reversals
            if raw_reversals and filt_reversals:
                timing_diffs = []
                for rr in raw_reversals:
                    best = min(filt_reversals, key=lambda x: abs(x["frame"] - rr["frame"]))
                    if abs(best["frame"] - rr["frame"]) <= 5: timing_diffs.append((best["time_s"] - rr["time_s"]) * 1000)
                result[f"greeting_{wname}_reversal_timing_diff_ms"] = timing_diffs
    elif word == "THANK_YOU" and norm_base.shape[1] >= 33:
        for wname, wli in [("left_wrist", LM_LW), ("right_wrist", LM_RW)]:
            if ss < N and valid[ss, wli] and valid[ss, LM_NOSE]:
                nose_r = norm_base[ss, LM_NOSE]; wr = norm_base[ss, wli]; wf = norm_filt[ss, wli]
                s = scales[ss] if not np.isnan(scales[ss]) else 0
                if not np.any(np.isnan(nose_r)) and not np.any(np.isnan(wr)):
                    result[f"thankyou_{wname}_start_face_dist_raw_mm"] = float(np.linalg.norm(wr - nose_r) * s * 1000)
                if not np.any(np.isnan(nose_r)) and not np.any(np.isnan(wf)):
                    result[f"thankyou_{wname}_start_face_dist_filt_mm"] = float(np.linalg.norm(wf - nose_r) * s * 1000)
            z_raw = [norm_base[fi, wli, 2] for fi in range(ss, min(se + 1, N)) if valid[fi, wli] and not np.isnan(norm_base[fi, wli, 2])]
            z_filt = [norm_filt[fi, wli, 2] for fi in range(ss, min(se + 1, N)) if valid[fi, wli] and not np.isnan(norm_filt[fi, wli, 2])]
            s_mean = np.nanmean(scales[ss:se + 1])
            if z_raw: result[f"thankyou_{wname}_fwd_amplitude_raw_mm"] = float((max(z_raw) - min(z_raw)) * s_mean * 1000) if not np.isnan(s_mean) else 0
            if z_filt: result[f"thankyou_{wname}_fwd_amplitude_filt_mm"] = float((max(z_filt) - min(z_filt)) * s_mean * 1000) if not np.isnan(s_mean) else 0
    return result

# ================================================================
#  Masking Experiment (v1.5: returns raw errors for pooled aggregate)
# ================================================================

def run_masking_experiment(arrs, norm, boundaries, strong_forbidden, ts, sid):
    results = []; raw_errors = {}
    for comp, nlm in [("pose", 33), ("lh", 21), ("rh", 21)]:
        xyz = arrs[f"{comp}_xyz"]; valid = arrs[f"{comp}_te"]
        bd = boundaries[comp]; sf = strong_forbidden[comp]; N = xyz.shape[0]
        for gap_len in [1, 2, 3]:
            pos_errs, speed_mag_errs, accel_mag_errs = [], [], []
            vel_vec_errs, accel_vec_errs = [], []
            entry_vd, exit_vd = [], []; n_windows = 0
            for li in range(nlm):
                spans = []; ss = None
                for fi in range(N):
                    if valid[fi, li] and not bd[fi, li] and not sf[fi, li] and not np.any(np.isnan(xyz[fi, li])):
                        if ss is None: ss = fi
                    else:
                        if ss is not None: spans.append((ss, fi - 1)); ss = None
                if ss is not None: spans.append((ss, N - 1))
                for s0, s1 in spans:
                    if s1 - s0 + 1 < gap_len + 2: continue
                    for start in range(s0 + 1, s1 - gap_len + 1, max(1, gap_len)):
                        end = start + gap_len - 1
                        if end >= s1: break
                        p0 = xyz[start - 1, li]; p1 = xyz[end + 1, li]; n_windows += 1
                        for g in range(gap_len):
                            t = (g + 1) / (gap_len + 1)
                            recon = p0 * (1 - t) + p1 * t
                            pos_errs.append(np.linalg.norm(recon - xyz[start + g, li]))
                        recon_pts = [p0] + [p0 * (1 - (g+1)/(gap_len+1)) + p1 * ((g+1)/(gap_len+1)) for g in range(gap_len)] + [p1]
                        actual_pts = [xyz[start - 1 + i, li] for i in range(gap_len + 2)]
                        for i in range(1, len(recon_pts)):
                            fi_a = start - 1 + i; fi_b = fi_a - 1
                            dt = ts[fi_a] - ts[fi_b] if fi_a < N and fi_b >= 0 else 1/30
                            if dt > 0 and dt < 0.2:
                                rv = (recon_pts[i] - recon_pts[i-1]) / dt; av = (actual_pts[i] - actual_pts[i-1]) / dt
                                speed_mag_errs.append(abs(np.linalg.norm(rv) - np.linalg.norm(av)))
                                vel_vec_errs.append(np.linalg.norm(rv - av))
                        for i in range(1, len(recon_pts) - 1):
                            fi_1 = start - 1 + i; dt1 = ts[fi_1] - ts[fi_1 - 1] if fi_1 < N and fi_1 - 1 >= 0 else 1/30
                            dt2 = ts[fi_1 + 1] - ts[fi_1] if fi_1 + 1 < N else 1/30
                            if dt1 > 0 and dt2 > 0 and dt1 < 0.2 and dt2 < 0.2:
                                ra = ((recon_pts[i+1]-recon_pts[i])/dt2 - (recon_pts[i]-recon_pts[i-1])/dt1) / ((dt1+dt2)/2)
                                aa = ((actual_pts[i+1]-actual_pts[i])/dt2 - (actual_pts[i]-actual_pts[i-1])/dt1) / ((dt1+dt2)/2)
                                accel_mag_errs.append(abs(np.linalg.norm(ra) - np.linalg.norm(aa)))
                                accel_vec_errs.append(np.linalg.norm(ra - aa))
                        if start >= 2:
                            dt_e = ts[start] - ts[start-1] if start < N else 1/30
                            dt_p = ts[start-1] - ts[start-2] if start-1 >= 1 else 1/30
                            if dt_e > 0 and dt_p > 0 and dt_e < 0.2 and dt_p < 0.2:
                                entry_vd.append(abs(np.linalg.norm(recon_pts[1]-recon_pts[0])/dt_e - np.linalg.norm(xyz[start-1,li]-xyz[start-2,li])/dt_p))
                        if end + 2 < N:
                            dt_x = ts[end+2] - ts[end+1] if end+2 < N else 1/30
                            dt_l = ts[end+1] - ts[end] if end+1 < N else 1/30
                            if dt_x > 0 and dt_l > 0 and dt_x < 0.2 and dt_l < 0.2:
                                exit_vd.append(abs(np.linalg.norm(recon_pts[-1]-recon_pts[-2])/dt_l - np.linalg.norm(xyz[end+2,li]-xyz[end+1,li])/dt_x))
            if pos_errs:
                pe = np.array(pos_errs)
                row = {"sample_id": sid, "component": comp, "gap_length": gap_len,
                       "n_windows": n_windows, "n_masked_points": len(pe),
                       "recon_err_mean_mm": float(np.mean(pe) * 1000),
                       "recon_err_p95_mm": float(np.percentile(pe, 95) * 1000),
                       "recon_err_max_mm": float(np.max(pe) * 1000)}
                raw_key = (comp, gap_len)
                raw_errors.setdefault(raw_key, {"pos": [], "speed_mag": [], "vel_vec": [], "accel_mag": [], "accel_vec": []})
                raw_errors[raw_key]["pos"].extend(pos_errs)
                if speed_mag_errs:
                    sme = np.array(speed_mag_errs)
                    row["speed_magnitude_error_rmse"] = float(np.sqrt(np.mean(sme**2)))
                    row["speed_magnitude_error_p95"] = float(np.percentile(sme, 95))
                    row["speed_magnitude_error_max"] = float(np.max(sme))
                    raw_errors[raw_key]["speed_mag"].extend(speed_mag_errs)
                if vel_vec_errs:
                    vve = np.array(vel_vec_errs)
                    row["velocity_vector_error_rmse"] = float(np.sqrt(np.mean(vve**2)))
                    row["velocity_vector_error_p95"] = float(np.percentile(vve, 95))
                    row["velocity_vector_error_max"] = float(np.max(vve))
                    raw_errors[raw_key]["vel_vec"].extend(vel_vec_errs)
                if accel_mag_errs:
                    ame = np.array(accel_mag_errs)
                    row["acceleration_magnitude_error_rmse"] = float(np.sqrt(np.mean(ame**2)))
                    row["acceleration_magnitude_error_p95"] = float(np.percentile(ame, 95))
                    row["acceleration_magnitude_error_max"] = float(np.max(ame))
                    raw_errors[raw_key]["accel_mag"].extend(accel_mag_errs)
                if accel_vec_errs:
                    ave = np.array(accel_vec_errs)
                    row["acceleration_vector_error_rmse"] = float(np.sqrt(np.mean(ave**2)))
                    row["acceleration_vector_error_p95"] = float(np.percentile(ave, 95))
                    row["acceleration_vector_error_max"] = float(np.max(ave))
                    raw_errors[raw_key]["accel_vec"].extend(accel_vec_errs)
                if entry_vd: row["entry_vel_disc_mean_m_per_s"] = float(np.mean(entry_vd))
                if exit_vd: row["exit_vel_disc_mean_m_per_s"] = float(np.mean(exit_vd))
                results.append(row)
    return results, raw_errors

def compute_masking_pooled_aggregate(all_raw_errors, all_results):
    agg_rows = []
    for comp in ["pose", "lh", "rh"]:
        for gl in [1, 2, 3]:
            key = (comp, gl)
            if key not in all_raw_errors: continue
            errs = all_raw_errors[key]
            rows = [r for r in all_results if r["component"] == comp and r["gap_length"] == gl]
            total_pts = sum(r["n_masked_points"] for r in rows)
            total_win = sum(r["n_windows"] for r in rows)
            if not errs["pos"]: continue
            pe = np.array(errs["pos"])
            agg = {"sample_id": "POOLED_AGGREGATE", "component": comp, "gap_length": gl,
                   "n_windows": total_win, "n_masked_points": total_pts,
                   "n_position_errors": len(pe),
                   "recon_err_mean_mm": float(np.mean(pe) * 1000),
                   "recon_err_rmse_mm": float(np.sqrt(np.mean(pe**2)) * 1000),
                   "recon_err_p95_mm": float(np.percentile(pe, 95) * 1000),
                   "recon_err_max_mm": float(np.max(pe) * 1000),
                   "max_sample_p95_mm": float(max(r["recon_err_p95_mm"] for r in rows))}
            for metric_name, err_key in [("speed_magnitude_error", "speed_mag"), ("velocity_vector_error", "vel_vec"),
                                          ("acceleration_magnitude_error", "accel_mag"), ("acceleration_vector_error", "accel_vec")]:
                vals = errs[err_key]
                if vals:
                    va = np.array(vals)
                    agg[f"{metric_name}_rmse"] = float(np.sqrt(np.mean(va**2)))
                    agg[f"{metric_name}_mean"] = float(np.mean(va))
                    agg[f"{metric_name}_p95"] = float(np.percentile(va, 95))
                    agg[f"{metric_name}_max"] = float(np.max(va))
                    agg[f"n_{err_key}_errors"] = len(va)
            # Denominator aliases
            if "n_vel_vec_errors" in agg: agg["n_velocity_errors"] = agg["n_vel_vec_errors"]
            if "n_accel_vec_errors" in agg: agg["n_acceleration_errors"] = agg["n_accel_vec_errors"]
            agg_rows.append(agg)
    return agg_rows

# ================================================================
#  Contact sheet / NPZ Save/Verify / Full Sequence / Plot (unchanged from v1.4)
# ================================================================

def make_seg_contact_sheet(sid, seg_info, out_dir):
    short = sid.replace("S001_P001_", "")
    vid_path = OVERLAY_DIR / f"reproj_{short}_front.mp4"
    if not vid_path.exists(): return None
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened(): return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    N = seg_info["total_frames"]; ss = seg_info["seg_start"]; se = seg_info["seg_end"]; mid = (ss + se) // 2
    fi_list = [max(0, ss - 3) if ss > 2 else 0, ss, mid, se, min(N - 1, se + 3) if se < N - 3 else N - 1]
    lb_list = [f"before(f{fi_list[0]})" if ss > 2 else f"start_edge(f0)", f"seg_start(f{ss})",
               f"middle(f{mid})", f"seg_end(f{se})",
               f"after(f{fi_list[4]})" if se < N - 3 else f"end_edge(f{N-1})"]
    frames = []
    for idx, fi in enumerate(fi_list):
        if fi < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ret, frame = cap.read()
            if ret:
                cv2.putText(frame, lb_list[idx], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                esm = seg_info.get("energy_smooth", np.zeros(1))
                cv2.putText(frame, f"e={esm[min(fi, len(esm)-1)]:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                frames.append(frame)
            else: frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        else: frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
    cap.release()
    if not frames: return None
    th = 360; resized = [cv2.resize(f, (int(f.shape[1] * th / f.shape[0]), th)) for f in frames]
    sheet = np.concatenate(resized, axis=1)
    out_path = out_dir / f"seg_contact_{short}.png"; cv2.imwrite(str(out_path), sheet); return out_path

def save_sample_npz(sid, arrs, norm, interp_data, smooth_data, norm_interp_cache, stab, out_dir):
    short = sid.replace("S001_P001_", ""); npz_dir = out_dir / "npz"; npz_dir.mkdir(exist_ok=True)
    prov_dir = out_dir / "provenance"; prov_dir.mkdir(exist_ok=True)
    for mg in INTERP_CANDIDATES:
        ir = interp_data[mg]; arrays = {}
        for comp in ["pose", "lh", "rh"]:
            arrays[f"{comp}_interp"] = ir[f"{comp}_interp"]; arrays[f"{comp}_norm_interp"] = norm_interp_cache[f"norm_interp_{comp}_g{mg}"]
            arrays[f"{comp}_original_valid"] = arrs[f"{comp}_te"]; arrays[f"{comp}_mask"] = ir[f"{comp}_mask"]
            arrays[f"{comp}_rejected"] = ir[f"{comp}_rejected"]; arrays[f"{comp}_boundary"] = ir[f"{comp}_boundary"]
            arrays[f"{comp}_strong_forbidden"] = ir[f"{comp}_strong_forbidden"]
        np.savez_compressed(npz_dir / f"interp_{short}_g{mg}.npz", **arrays)
        prov = {comp: ir[f"{comp}_records"] for comp in ["pose", "lh", "rh"]}
        with open(prov_dir / f"interp_provenance_{short}_g{mg}.json", "w", encoding="utf-8") as f: json.dump(prov, f)
    for fname, sm in smooth_data.items():
        arrays = {}
        for comp in ["pose", "lh", "rh"]:
            for sfx in ["filtered", "norm_filtered", "filter_valid", "reset_mask", "first_after_reset", "reset_reason"]:
                arrays[f"{comp}_{sfx}"] = sm[f"{comp}_{sfx}"]
        np.savez_compressed(npz_dir / f"smooth_{short}_{fname}.npz", **arrays)
        schema = {"filter": fname, "sample_id": sid, "interp_candidate_used": 2,
                  "reset_reason_codes": RSN_CODE_MAP, "first_after_reset_includes_initial": True}
        if fname.startswith("oe_"):
            pname = fname[3:]
            for c in ONE_EURO_CANDIDATES:
                if c["name"] == pname: schema["filter_params"] = c; break
        elif fname == "savgol": schema["filter_params"] = {"window": SAVGOL_WINDOW, "order": SAVGOL_ORDER}
        elif fname == "none": schema["filter_params"] = {"type": "identity"}
        for comp in ["pose", "lh", "rh"]:
            schema[f"{comp}_metrics"] = sm[f"{comp}_metrics"]
            if f"{comp}_sign_metrics" in sm: schema[f"{comp}_sign_metrics"] = sm[f"{comp}_sign_metrics"]
        with open(prov_dir / f"smooth_schema_{short}_{fname}.json", "w", encoding="utf-8") as f: json.dump(schema, f, indent=2, default=str)
    arrays = {"origins": norm["origins"], "rotations": norm["rotations"], "scales": norm["scales"],
              "dets": norm["dets"], "ortho_errs": norm["ortho_errs"],
              "norm_pose": norm["norm_pose"], "norm_lh": norm["norm_lh"], "norm_rh": norm["norm_rh"],
              "stab_origins": stab["stab_origins"], "stab_scales": stab["stab_scales"], "stab_rotations": stab["stab_rotations"]}
    np.savez_compressed(npz_dir / f"norm_{short}.npz", **arrays)

def verify_sample_npz(sid, arrs, norm, interp_data, smooth_data, norm_interp_cache, stab, out_dir):
    short = sid.replace("S001_P001_", ""); npz_dir = out_dir / "npz"; fails = []
    expected_interp = set(interp_npz_keys()); expected_smooth = set(smooth_npz_keys()); expected_norm = set(norm_npz_keys())
    for mg in INTERP_CANDIDATES:
        path = npz_dir / f"interp_{short}_g{mg}.npz"
        if not path.exists(): fails.append(f"missing_interp_g{mg}"); continue
        loaded = np.load(path, allow_pickle=False); loaded_keys = set(loaded.files)
        if loaded_keys != expected_interp:
            for ek in expected_interp - loaded_keys: fails.append(f"missing_key_interp_g{mg}_{ek}")
            for ek in loaded_keys - expected_interp: fails.append(f"extra_key_interp_g{mg}_{ek}")
        ir = interp_data[mg]; mem_map = {}
        for comp in ["pose", "lh", "rh"]:
            mem_map[f"{comp}_interp"] = ir[f"{comp}_interp"]; mem_map[f"{comp}_norm_interp"] = norm_interp_cache[f"norm_interp_{comp}_g{mg}"]
            mem_map[f"{comp}_original_valid"] = arrs[f"{comp}_te"]; mem_map[f"{comp}_mask"] = ir[f"{comp}_mask"]
            mem_map[f"{comp}_rejected"] = ir[f"{comp}_rejected"]; mem_map[f"{comp}_boundary"] = ir[f"{comp}_boundary"]
            mem_map[f"{comp}_strong_forbidden"] = ir[f"{comp}_strong_forbidden"]
        for k, mem in mem_map.items():
            if k in loaded_keys:
                if mem.dtype in (np.float32, np.float64):
                    if not np.allclose(loaded[k], mem, equal_nan=True, atol=1e-12): fails.append(f"mismatch_interp_g{mg}_{k}")
                else:
                    if not np.array_equal(loaded[k], mem): fails.append(f"mismatch_interp_g{mg}_{k}")
        loaded.close()
    for fname, sm in smooth_data.items():
        path = npz_dir / f"smooth_{short}_{fname}.npz"
        if not path.exists(): fails.append(f"missing_smooth_{fname}"); continue
        loaded = np.load(path, allow_pickle=False); loaded_keys = set(loaded.files)
        if loaded_keys != expected_smooth:
            for ek in expected_smooth - loaded_keys: fails.append(f"missing_key_smooth_{fname}_{ek}")
            for ek in loaded_keys - expected_smooth: fails.append(f"extra_key_smooth_{fname}_{ek}")
        for comp in ["pose", "lh", "rh"]:
            for sfx in ["filtered", "norm_filtered", "filter_valid", "reset_mask", "first_after_reset", "reset_reason"]:
                k = f"{comp}_{sfx}"
                if k in loaded_keys and k in sm:
                    mem = sm[k]
                    if mem.dtype in (np.float32, np.float64):
                        if not np.allclose(loaded[k], mem, equal_nan=True, atol=1e-12): fails.append(f"mismatch_smooth_{fname}_{k}")
                    else:
                        if not np.array_equal(loaded[k], mem): fails.append(f"mismatch_smooth_{fname}_{k}")
        loaded.close()
    path = npz_dir / f"norm_{short}.npz"
    if not path.exists(): fails.append("missing_norm")
    else:
        loaded = np.load(path, allow_pickle=False); loaded_keys = set(loaded.files)
        if loaded_keys != expected_norm:
            for ek in expected_norm - loaded_keys: fails.append(f"missing_key_norm_{ek}")
            for ek in loaded_keys - expected_norm: fails.append(f"extra_key_norm_{ek}")
        nm = {"origins": norm["origins"], "rotations": norm["rotations"], "scales": norm["scales"],
              "dets": norm["dets"], "ortho_errs": norm["ortho_errs"],
              "norm_pose": norm["norm_pose"], "norm_lh": norm["norm_lh"], "norm_rh": norm["norm_rh"],
              "stab_origins": stab["stab_origins"], "stab_scales": stab["stab_scales"], "stab_rotations": stab["stab_rotations"]}
        for k, mem in nm.items():
            if k in loaded_keys:
                if not np.allclose(loaded[k], mem, equal_nan=True, atol=1e-12): fails.append(f"mismatch_norm_{k}")
        loaded.close()
    return fails

def _byte_exact_npz(path_a, path_b, label):
    """Compare two NPZ files strict byte-level: key sets, dtype, shape, full raw bytes including NaN bit payload."""
    fails = []
    if not path_a.exists(): fails.append(f"{label}_ref_missing_{path_a.name}"); return fails
    if not path_b.exists(): fails.append(f"{label}_cur_missing_{path_b.name}"); return fails
    a = np.load(path_a, allow_pickle=False); b = np.load(path_b, allow_pickle=False)
    keys_a = set(a.files); keys_b = set(b.files)
    if keys_a != keys_b:
        for ek in keys_a - keys_b: fails.append(f"{label}_missing_key_{ek}")
        for ek in keys_b - keys_a: fails.append(f"{label}_extra_key_{ek}")
    for k in keys_a & keys_b:
        aa, bb = a[k], b[k]
        if aa.dtype != bb.dtype: fails.append(f"{label}_dtype_{k}_{aa.dtype}_vs_{bb.dtype}"); continue
        if aa.shape != bb.shape: fails.append(f"{label}_shape_{k}_{aa.shape}_vs_{bb.shape}"); continue
        # Strict full raw bytes comparison (including NaN bit payload)
        a_bytes = np.ascontiguousarray(aa).tobytes()
        b_bytes = np.ascontiguousarray(bb).tobytes()
        if a_bytes != b_bytes:
            fails.append(f"{label}_bytes_{k}")
    a.close(); b.close()
    return fails

def verify_npz_byte_exact(sid, out_dir, ref_npz_dir, ref_label):
    """Verify current NPZ arrays match reference directory byte-exactly."""
    short = sid.replace("S001_P001_", ""); fails = []
    if not ref_npz_dir.exists(): return [f"{ref_label}_npz_dir_not_found"]
    npz_dir = out_dir / "npz"
    for mg in INTERP_CANDIDATES:
        ref_path = ref_npz_dir / f"interp_{short}_g{mg}.npz"
        cur_path = npz_dir / f"interp_{short}_g{mg}.npz"
        fails.extend(_byte_exact_npz(ref_path, cur_path, f"{ref_label}_interp_g{mg}"))
    for suffix in ["none", "savgol"] + [f"oe_{c['name']}" for c in ONE_EURO_CANDIDATES]:
        ref_path = ref_npz_dir / f"smooth_{short}_{suffix}.npz"
        cur_path = npz_dir / f"smooth_{short}_{suffix}.npz"
        if not ref_path.exists(): continue
        fails.extend(_byte_exact_npz(ref_path, cur_path, f"{ref_label}_smooth_{suffix}"))
    ref_path = ref_npz_dir / f"norm_{short}.npz"; cur_path = npz_dir / f"norm_{short}.npz"
    if ref_path.exists():
        fails.extend(_byte_exact_npz(ref_path, cur_path, f"{ref_label}_norm"))
    return fails

def save_full_sequence(sid, arrs, norm, seg, out_dir):
    short = sid.replace("S001_P001_", ""); N = len(arrs["timestamps"])
    seq = {"sample_id": sid, "version": VERSION_TAG, "n_frames": N,
           "coordinate_system": {"raw": "FRONT_camera_meters", "normalized": "body_centric_shoulder_width_scaled"},
           "segmentation": {"start": seg["seg_start"], "end": seg["seg_end"], "method": seg["method"],
                             "idle_verified_start": seg["idle_verified_start"], "idle_verified_end": seg["idle_verified_end"]}, "frames": []}
    for fi in range(N):
        frame = {"frame_index": fi, "timestamp_s": float(arrs["timestamps"][fi]),
                 "in_segment": seg["seg_start"] <= fi <= seg["seg_end"],
                 "transform": {"origin": norm["origins"][fi].tolist() if not np.any(np.isnan(norm["origins"][fi])) else None,
                     "rotation": norm["rotations"][fi].tolist() if not np.any(np.isnan(norm["rotations"][fi])) else None,
                     "scale": float(norm["scales"][fi]) if not np.isnan(norm["scales"][fi]) else None,
                     "status": norm["statuses"][fi], "fallback_src": norm["fallback_src"][fi],
                     "determinant": float(norm["dets"][fi]) if not np.isnan(norm["dets"][fi]) else None,
                     "ortho_error": float(norm["ortho_errs"][fi]) if not np.isnan(norm["ortho_errs"][fi]) else None}}
        for comp_full, jkey, nlm in [("pose","pose_landmarks_3d",33),("left_hand","left_hand_landmarks_3d",21),("right_hand","right_hand_landmarks_3d",21)]:
            cs = {"pose":"pose","left_hand":"lh","right_hand":"rh"}[comp_full]; lms = []
            for li in range(nlm):
                rp = arrs[f"{cs}_xyz"][fi, li]; np_v = norm[f"norm_{cs}"][fi, li]
                lms.append({"raw_xyz": rp.tolist() if not np.any(np.isnan(rp)) else None,
                             "norm_xyz": np_v.tolist() if not np.any(np.isnan(np_v)) else None,
                             "tri_valid": bool(arrs[f"{cs}_tri"][fi, li]), "qa_valid": bool(arrs[f"{cs}_qa"][fi, li]),
                             "te": bool(arrs[f"{cs}_te"][fi, li]), "qa_flags": arrs[f"{cs}_flags"][fi][li],
                             "invalid_reason": arrs[f"{cs}_invalid_reason"][fi][li],
                             "is_reappearance": bool(arrs[f"{cs}_reappear"][fi, li])})
            frame[comp_full] = lms
        seq["frames"].append(frame)
    with open(out_dir / "sequences" / f"full_{short}.json", "w", encoding="utf-8") as f: json.dump(seq, f)
    sp(f"      Saved full_{short}.json ({os.path.getsize(out_dir / 'sequences' / f'full_{short}.json') / 1e6:.1f} MB)")

def plot_seg(sid, arrs, norm, seg, viz_dir):
    if not HAS_MPL: return
    short = sid.replace("S001_P001_", ""); t = arrs["timestamps"] - arrs["timestamps"][0]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, seg["energy"], alpha=0.3, label="raw"); ax.plot(t, seg["energy_smooth"], label="smooth")
    ax.axhline(seg["threshold"], color="r", ls="--", alpha=0.5)
    ax.axvline(t[seg["seg_start"]], color="g", ls="-", label="start"); ax.axvline(t[seg["seg_end"]], color="g", ls="--", label="end")
    ax.set_title(f"Seg: {short}"); ax.legend(fontsize=7); ax.set_xlabel("Time (s)")
    fig.savefig(viz_dir / f"seg_{short}.png", dpi=120); plt.close(fig)

# ================================================================
#  Output manifest classifier (unchanged from v1.4, version updated)
# ================================================================

def classify_artifact(rel_path):
    meta = {"artifact_type": "unknown"}; p = rel_path.replace("\\", "/")
    if p.startswith("npz/interp_"):
        meta["artifact_type"] = "interpolation_npz"
        parts = p.replace("npz/interp_", "").replace(".npz", "").split("_g")
        if len(parts) == 2: meta["sample_id"] = "S001_P001_" + parts[0]; meta["max_gap"] = int(parts[1])
    elif p.startswith("npz/smooth_"):
        meta["artifact_type"] = "smoothing_npz"
        inner = p.replace("npz/smooth_", "").replace(".npz", "")
        for filt_name in ["none", "savgol"] + [f"oe_{c['name']}" for c in ONE_EURO_CANDIDATES]:
            if inner.endswith(f"_{filt_name}"): meta["sample_id"] = "S001_P001_" + inner[:-(len(filt_name)+1)]; meta["filter"] = filt_name; break
    elif p.startswith("npz/norm_"): meta["artifact_type"] = "normalization_npz"; meta["sample_id"] = "S001_P001_" + p.replace("npz/norm_","").replace(".npz","")
    elif p.startswith("provenance/interp_provenance_"):
        meta["artifact_type"] = "interpolation_provenance"
        parts = p.replace("provenance/interp_provenance_", "").replace(".json", "").split("_g")
        if len(parts) == 2: meta["sample_id"] = "S001_P001_" + parts[0]; meta["max_gap"] = int(parts[1])
    elif p.startswith("provenance/smooth_schema_"):
        meta["artifact_type"] = "smoothing_schema"
        inner = p.replace("provenance/smooth_schema_", "").replace(".json", "")
        for filt_name in ["none", "savgol"] + [f"oe_{c['name']}" for c in ONE_EURO_CANDIDATES]:
            if inner.endswith(f"_{filt_name}"): meta["sample_id"] = "S001_P001_" + inner[:-(len(filt_name)+1)]; meta["filter"] = filt_name; break
    elif p.startswith("sequences/full_"): meta["artifact_type"] = "full_sequence_json"; meta["sample_id"] = "S001_P001_" + p.replace("sequences/full_","").replace(".json","")
    elif p.startswith("contact_sheets/"): meta["artifact_type"] = "segmentation_contact_sheet"
    elif p.startswith("visualizations/"): meta["artifact_type"] = "segmentation_visualization"
    elif "config" in p and p.endswith(".json"): meta["artifact_type"] = "execution_config"
    elif p == "input_file_manifest.json": meta["artifact_type"] = "input_manifest"
    elif p == "run_manifest.json": meta["artifact_type"] = "run_manifest"
    elif p == "assertion_results.json": meta["artifact_type"] = "assertion_results"
    elif p.endswith("_REPORT.md"): meta["artifact_type"] = "report"
    elif "step7b_preprocess_full" in p or "step7a_preprocess_pilot" in p: meta["artifact_type"] = "execution_script"
    return meta

# ================================================================
#  Assertions (v1.5: enhanced)
# ================================================================

def run_assertions(all_results, split, all_sample_ids, pilot_ids, input_hashes, strong_depth, out_dir,
                   boundary_reasons_all, transform_detail_all, masking_agg, all_raw_errors):
    results = []; np_ = [0]; nf_ = [0]
    def chk(cond, name, detail="", **meta):
        if cond: np_[0] += 1
        else: nf_[0] += 1; sp(f"    [FAIL] {name}: {detail}")
        rec = {"name": name, "status": "PASS" if cond else "FAIL", "detail": detail}
        for mk, mv in meta.items():
            rec[mk] = mv
        results.append(rec)

    for sid in all_sample_ids:
        short = sid.replace("S001_P001_", "")
        h = sha256_file(RAW3D_DIR / f"landmarks_3d_{short}.json"); chk(h == input_hashes[sid], f"input_hash_{short}")
    chk(len(all_sample_ids) == 45, "sample_count_45", f"actual={len(all_sample_ids)}")
    splits_count = {"train": 0, "val": 0, "test": 0}
    for sid in all_sample_ids: splits_count[split[sid]["split"]] += 1
    chk(splits_count == {"train": 27, "val": 9, "test": 9}, "split_27_9_9", f"actual={splits_count}")
    chk(len(pilot_ids) == 9, "pilot_count_9")
    for w in WORDS: chk(len([s for s in pilot_ids if w in s]) == 3, f"pilot_{w}_3")
    for sid in pilot_ids: chk(split[sid]["split"] == "train", f"pilot_train_{sid.replace('S001_P001_','')}")
    chk(all(p in set(all_sample_ids) for p in pilot_ids), "pilot_in_full_set")

    for sid, sr in all_results.items():
        short = sid.replace("S001_P001_", "")
        arrs = sr["arrs"]; norm = sr["norm"]; raw = sr["raw_data"]; N = len(arrs["timestamps"])
        for fi, fr in enumerate(raw["frames"]): chk(abs(arrs["timestamps"][fi] - fr["front_time_s"]) < 1e-12, f"ts_{short}_f{fi}")
        KEYS_MAP = {"pose": ("pose_landmarks_3d", 33), "lh": ("left_hand_landmarks_3d", 21), "rh": ("right_hand_landmarks_3d", 21)}
        for comp, (jkey, nlm) in KEYS_MAP.items():
            for fi in range(N):
                for li in range(nlm):
                    lm = raw["frames"][fi][jkey][li]
                    chk(arrs[f"{comp}_tri"][fi, li] == lm["triangulation_valid"], f"tri_{short}_{comp}_f{fi}_l{li}")
                    chk(arrs[f"{comp}_qa"][fi, li] == lm["qa_valid"], f"qa_{short}_{comp}_f{fi}_l{li}")
                    chk(arrs[f"{comp}_te"][fi, li] == lm["template_eligible"], f"te_{short}_{comp}_f{fi}_l{li}")
                    if lm["triangulation_valid"] and lm["xyz"] is not None:
                        chk(np.allclose(np.array(lm["xyz"]), arrs[f"{comp}_xyz"][fi, li], atol=1e-12), f"xyz_{short}_{comp}_f{fi}_l{li}")
        for fi in range(N):
            if norm["statuses"][fi] in ("ok", "fallback_prev", "fallback_next"):
                R = norm["rotations"][fi]
                if not np.any(np.isnan(R)):
                    chk(abs(norm["dets"][fi] - 1.0) < 1e-6, f"det_{short}_f{fi}")
                    chk(norm["ortho_errs"][fi] < 1e-6, f"ortho_{short}_f{fi}")
        for fi in range(1, N):
            if norm["statuses"][fi] in ("ok", "fallback_prev", "fallback_next") and norm["statuses"][fi-1] in ("ok", "fallback_prev", "fallback_next"):
                R0, R1 = norm["rotations"][fi-1], norm["rotations"][fi]
                if not np.any(np.isnan(R0)) and not np.any(np.isnan(R1)):
                    for col, name in [(0,"right"),(1,"up"),(2,"forward")]:
                        chk(np.dot(R0[:,col], R1[:,col]) >= 0, f"no_{name}_flip_{short}_f{fi}")
        vs = norm["scales"][~np.isnan(norm["scales"])]
        chk(len(vs) > 0, f"scale_exists_{short}")
        if len(vs) > 0: chk(np.all(vs > 0.05), f"scale_pos_{short}"); chk(np.all(vs < 2.0), f"scale_range_{short}")
        for mg, ir in sr["interp"].items():
            for comp in ["pose", "lh", "rh"]:
                chk(int(np.sum(ir[f"{comp}_mask"] & ir[f"{comp}_strong_forbidden"])) == 0, f"no_sd_interp_{short}_{comp}_g{mg}")
                for li in range(ir[f"{comp}_mask"].shape[1]):
                    for fi in range(ir[f"{comp}_mask"].shape[0]):
                        if ir[f"{comp}_mask"][fi, li] and ir[f"{comp}_boundary"][fi, li]:
                            chk(False, f"bd_cross_{short}_{comp}_g{mg}_f{fi}_l{li}")
        if sid == "S001_P001_LOVE_FAST_R04":
            for mg in INTERP_CANDIDATES:
                for fi in [37, 38]:
                    if fi < N: chk(not sr["interp"][mg]["pose_mask"][fi, 20], f"regress_LFR04_p20_f{fi}_g{mg}")
        ns = sr["smooth"].get("none", {})
        for comp in ["pose", "lh", "rh"]:
            m = ns.get(f"{comp}_metrics", {})
            chk(m.get("phys_dev_mean_mm", -1) == 0, f"no_smooth_zero_{short}_{comp}_mean")
            chk(m.get("phys_dev_p95_mm", -1) == 0, f"no_smooth_zero_{short}_{comp}_p95")
            chk(m.get("vel_distortion_pct", -1) == 0, f"no_smooth_vel_zero_{short}_{comp}")
        for mg, ir in sr["interp"].items():
            for comp in ["pose", "lh", "rh"]:
                n_interp = int(np.sum(ir[f"{comp}_mask"]))
                recs = [r for r in ir[f"{comp}_records"] if r["method"] == "linear"]
                chk(n_interp == sum(r["gap_length"] for r in recs), f"interp_prov_{short}_{comp}_g{mg}")
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                far = sm[f"{comp}_first_after_reset"]; filtered = sm[f"{comp}_filtered"]; base = sr["interp"][2][f"{comp}_interp"]
                for fi in range(far.shape[0]):
                    for li in range(far.shape[1]):
                        if far[fi, li] and not np.any(np.isnan(filtered[fi, li])) and not np.any(np.isnan(base[fi, li])):
                            chk(np.allclose(filtered[fi, li], base[fi, li], atol=1e-10), f"far_eq_input_{short}_{comp}_{fname}_f{fi}_l{li}")
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                rm = sm[f"{comp}_reset_mask"]; bd = sr["interp"][2][f"{comp}_boundary"]
                for li in range(rm.shape[1]):
                    for fi in range(rm.shape[0]):
                        if bd[fi, li]: chk(rm[fi, li], f"smooth_reset_{short}_{comp}_{fname}_f{fi}_l{li}")
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                m = sm[f"{comp}_metrics"]
                chk(m["new_nans"] == 0, f"no_nan_{short}_{comp}_{fname}"); chk(m["new_infs"] == 0, f"no_inf_{short}_{comp}_{fname}")
        for fname, sm in sr["smooth"].items():
            if not fname.startswith("oe_"): continue
            total_far = sum(int(np.sum(sm[f"{c}_first_after_reset"])) for c in ["pose", "lh", "rh"])
            chk(total_far > 0, f"far_nonzero_{short}_{fname}")
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                chk(f"{comp}_filter_valid" in sm, f"has_filter_valid_{short}_{comp}_{fname}")
                chk(f"{comp}_reset_reason" in sm, f"has_reset_reason_{short}_{comp}_{fname}")
                rm = sm[f"{comp}_reset_mask"]; rsn = sm[f"{comp}_reset_reason"]
                chk(np.array_equal(rm, rsn != RSN_NONE), f"reset_mask_eq_reason_{short}_{comp}_{fname}")
                chk(np.all(sm[f"{comp}_filter_valid"][sm[f"{comp}_first_after_reset"]]), f"far_implies_valid_{short}_{comp}_{fname}")
                valid_sm = sm[f"{comp}_filter_valid"]; bd = sr["interp"][2][f"{comp}_boundary"]; br = boundary_reasons_all[sid][comp]
                base_xyz = sr["interp"][2][f"{comp}_interp"]
                ref_rm, ref_far, ref_rsn = recompute_first_after_reset(base_xyz, valid_sm, bd, br)
                chk(int(np.sum(ref_far & ~sm[f"{comp}_first_after_reset"])) == 0, f"far_recomp_missing_{short}_{comp}_{fname}")
                chk(int(np.sum(sm[f"{comp}_first_after_reset"] & ~ref_far)) == 0, f"far_recomp_extra_{short}_{comp}_{fname}")
        for fail_name in sr.get("npz_verify_fails", []): chk(False, f"npz_{short}_{fail_name}")
        if not sr.get("npz_verify_fails"): chk(True, f"npz_{short}_all_pass")
        # Pilot-only: v1.4-v1.11 byte-exact NPZ regression
        if sr.get("is_pilot"):
            for fail_name in sr.get("v14_npz_fails", []): chk(False, f"v14npz_{short}_{fail_name}")
            if not sr.get("v14_npz_fails"): chk(True, f"v14npz_{short}_byte_exact")
            for fail_name in sr.get("v15_npz_fails", []): chk(False, f"v15npz_{short}_{fail_name}")
            if not sr.get("v15_npz_fails"): chk(True, f"v15npz_{short}_byte_exact")
            for fail_name in sr.get("v16_npz_fails", []): chk(False, f"v16npz_{short}_{fail_name}")
            if not sr.get("v16_npz_fails"): chk(True, f"v16npz_{short}_byte_exact")
            for fail_name in sr.get("v17_npz_fails", []): chk(False, f"v17npz_{short}_{fail_name}")
            if not sr.get("v17_npz_fails"): chk(True, f"v17npz_{short}_byte_exact")
            for fail_name in sr.get("v18_npz_fails", []): chk(False, f"v18npz_{short}_{fail_name}")
            if not sr.get("v18_npz_fails"): chk(True, f"v18npz_{short}_byte_exact")
            for fail_name in sr.get("v19_npz_fails", []): chk(False, f"v19npz_{short}_{fail_name}")
            if not sr.get("v19_npz_fails"): chk(True, f"v19npz_{short}_byte_exact")
            for fail_name in sr.get("v110_npz_fails", []): chk(False, f"v110npz_{short}_{fail_name}")
            if not sr.get("v110_npz_fails"): chk(True, f"v110npz_{short}_byte_exact")
            for fail_name in sr.get("v111_npz_fails", []): chk(False, f"v111npz_{short}_{fail_name}")
            if not sr.get("v111_npz_fails"): chk(True, f"v111npz_{short}_byte_exact")
            # v1.12 pilot byte-exact regression
            for fail_name in sr.get("v112_pilot_fails", []): chk(False, f"v112pilotnpz_{short}_{fail_name}")
            if not sr.get("v112_pilot_fails"): chk(True, f"v112pilotnpz_{short}_byte_exact")

    # v1.6: smoothing timing - no negative events in causal delay aggregate
    for sid, sr in all_results.items():
        short = sid.replace("S001_P001_", "")
        for fname, sm in sr["smooth"].items():
            if fname == "none": continue
            for comp in ["pose", "lh", "rh"]:
                m = sm[f"{comp}_metrics"]; pld = m.get("peak_lag_detail", {})
                agg_lag = pld.get("aggregate_lag_frames")
                if isinstance(agg_lag, (int, float)):
                    chk(agg_lag >= 0, f"nonneg_agg_lag_{short}_{comp}_{fname}", f"lag={agg_lag}")
                # Check onset/end/xcorr exist in segments
                for lm_t in pld.get("per_landmark", []):
                    for seg_t in lm_t.get("segments", []):
                        seg_s = seg_t.get("seg_start", "?"); lm_idx = lm_t["landmark"]
                        chk("onset_lag" in seg_t, f"has_onset_{short}_{comp}_{fname}_l{lm_idx}_s{seg_s}")
                        chk("end_lag" in seg_t, f"has_end_{short}_{comp}_{fname}_l{lm_idx}_s{seg_s}")
                        chk("xcorr_lag" in seg_t, f"has_xcorr_{short}_{comp}_{fname}_l{lm_idx}_s{seg_s}")
                        # peak_height and prominence as separate fields in each raw peak
                        for pk_idx, pk in enumerate(seg_t.get("raw_peaks", [])):
                            pk_frame = pk.get("frame", pk_idx)
                            chk("peak_height" in pk, f"has_peak_height_{short}_{comp}_{fname}_l{lm_idx}_s{seg_s}_pk{pk_frame}")
                            chk("prominence" in pk, f"has_prominence_{short}_{comp}_{fname}_l{lm_idx}_s{seg_s}_pk{pk_frame}")

    # v1.6: transform timing fields exist for all key landmarks, with matched/unmatched/negative counts
    for td in transform_detail_all:
        sid_short = td["sample_id"].replace("S001_P001_", "")
        for pl in td.get("per_landmark", []):
            comp_lbl = pl["component"]; lm_lbl = pl["landmark"]
            chk("timing" in pl, f"transform_timing_exists_{sid_short}_{comp_lbl}_l{lm_lbl}")
            t = pl.get("timing", {})
            if t:
                chk("n_nonneg_matched" in t, f"transform_timing_nonneg_{sid_short}_{comp_lbl}_l{lm_lbl}")
                chk("n_negative_inconsistent" in t, f"transform_timing_neg_{sid_short}_{comp_lbl}_l{lm_lbl}")
                chk("n_unmatched_raw" in t, f"transform_timing_unmatched_raw_{sid_short}_{comp_lbl}_l{lm_lbl}")
                chk("n_unmatched_filt" in t, f"transform_timing_unmatched_filt_{sid_short}_{comp_lbl}_l{lm_lbl}")
                chk("matched_events" in t, f"transform_timing_matched_events_{sid_short}_{comp_lbl}_l{lm_lbl}")

    # v1.6: masking aggregate pooled RMSE recomputation
    for agg in masking_agg:
        key = (agg["component"], agg["gap_length"])
        if key in all_raw_errors:
            pe = np.array(all_raw_errors[key]["pos"])
            if len(pe) > 0:
                expected_rmse = float(np.sqrt(np.mean(pe**2)) * 1000)
                chk(abs(agg["recon_err_rmse_mm"] - expected_rmse) < 1e-6, f"pooled_rmse_{agg['component']}_g{agg['gap_length']}")
                expected_p95 = float(np.percentile(pe, 95) * 1000)
                chk(abs(agg["recon_err_p95_mm"] - expected_p95) < 1e-6, f"pooled_p95_{agg['component']}_g{agg['gap_length']}")

    # Input manifest checks
    inp_man = out_dir / "input_file_manifest.json"
    if inp_man.exists():
        with open(inp_man, encoding="utf-8") as f: inp_data = json.load(f)
        inputs = inp_data.get("inputs", [])
        has_script = any(e.get("artifact_type") == "execution_script" for e in inputs)
        has_config = any(e.get("artifact_type") == "execution_config" for e in inputs)
        chk(has_script, "input_manifest_has_script"); chk(has_config, "input_manifest_has_config")
        script_entry = next((e for e in inputs if e.get("artifact_type") == "execution_script"), None)
        if script_entry: chk(sha256_file(PROJECT / script_entry["file"]) == script_entry["sha256"], "input_manifest_script_hash_match")
        config_entry = next((e for e in inputs if e.get("artifact_type") == "execution_config"), None)
        if config_entry: chk(sha256_file(PROJECT / config_entry["file"]) == config_entry["sha256"], "input_manifest_config_hash_match")
    rm_path = out_dir / "run_manifest.json"
    if rm_path.exists():
        with open(rm_path, encoding="utf-8") as f: rm_data = json.load(f)
        chk(rm_data.get("smoothing_recommendation") == SMOOTHING_CONCLUSION, "run_manifest_smoothing_conclusion")
        chk(rm_data.get("selected_normalization_transform") == SELECTED_NORM_TRANSFORM, "run_manifest_norm_transform")
        chk(rm_data.get("selected_max_gap") == SELECTED_MAX_GAP, "run_manifest_max_gap")
        chk(rm_data.get("savgol_note") == SMOOTHING_SAVGOL_NOTE, "run_manifest_savgol_note")
        chk(rm_data.get("run_id") == RUN_ID, "run_manifest_run_id_matches")
    chk((out_dir / "assertion_results.json").exists(), "assertion_results_placeholder_exists")

    # v1.10: Real dynamics edge assertions with independent sum recomputation
    total_used_edges, total_excluded_edges, total_boundary_edges = 0, 0, 0
    total_first_vel_after_reset, total_accel_edges = 0, 0
    dynamics_targets_visited = set()
    reset_targets_visited = set()
    _norm_cache = {}  # Cache normalize_with_transform results per (sid, comp_key)
    for td in transform_detail_all:
        sid_val = td["sample_id"]; sid_short = sid_val.replace("S001_P001_", "")
        sr = all_results[sid_val]; arrs = sr["arrs"]; norm_d = sr["norm"]; bd_all = sr["boundaries"]
        stab_d = sr["stab"]; ts_arr = arrs["timestamps"]; N = len(ts_arr)
        for pl in td.get("per_landmark", []):
            comp_lbl = pl["component"]; lm_lbl = pl["landmark"]
            comp_key = {"pose": "pose", "left_hand": "lh", "right_hand": "rh"}[comp_lbl]
            li = lm_lbl; lm_key = f"{sid_short}_{comp_lbl}_{li}"
            xyz = arrs[f"{comp_key}_xyz"]; valid = arrs[f"{comp_key}_te"]; bd = bd_all[comp_key]
            cache_key = (sid_val, comp_key)
            if cache_key not in _norm_cache:
                _norm_cache[cache_key] = (
                    _normalize_with_transform_fast(xyz, valid, norm_d["origins"], norm_d["rotations"], norm_d["scales"]),
                    _normalize_with_transform_fast(xyz, valid, stab_d["stab_origins"], stab_d["stab_rotations"], stab_d["stab_scales"]))
            raw_norm, stab_norm = _norm_cache[cache_key]
            n_vel_retrace, n_accel_retrace = 0, 0
            n_excluded_retrace, n_boundary_retrace = 0, 0
            n_first_vel_retrace = 0
            prev_vel_valid = False
            vel_sum_retrace, vel_sum_sq_retrace = 0.0, 0.0
            accel_sum_retrace, accel_sum_sq_retrace = 0.0, 0.0
            prev_rv, prev_sv = None, None
            no_forbidden_cross = True
            for fi in range(1, N):
                dt = ts_arr[fi] - ts_arr[fi - 1]
                is_bd = bd[fi, li]
                is_invalid_obs = not valid[fi, li] or not valid[fi - 1, li]
                is_ts_disc = dt <= 0 or dt > 0.1
                r0 = raw_norm[fi - 1, li]; r1 = raw_norm[fi, li]
                s0 = stab_norm[fi - 1, li]; s1 = stab_norm[fi, li]
                has_nan_r = np.any(np.isnan(r0)) or np.any(np.isnan(r1))
                has_nan_s = np.any(np.isnan(s0)) or np.any(np.isnan(s1))
                is_excluded = is_bd or is_ts_disc or is_invalid_obs or has_nan_r or has_nan_s
                if is_excluded:
                    if is_bd: n_boundary_retrace += 1
                    n_excluded_retrace += 1
                    prev_vel_valid = False; prev_rv = prev_sv = None
                else:
                    # Used edge: verify no forbidden crossing
                    if is_bd: no_forbidden_cross = False
                    sc = norm_d["scales"][fi] if not np.isnan(norm_d["scales"][fi]) else 0
                    rv = (r1 - r0) * sc / dt; sv = (s1 - s0) * sc / dt
                    vel_err = float(np.linalg.norm(sv - rv))
                    vel_sum_retrace += vel_err; vel_sum_sq_retrace += vel_err ** 2
                    n_vel_retrace += 1
                    if not prev_vel_valid:
                        n_first_vel_retrace += 1
                    else:
                        if prev_rv is not None and prev_sv is not None:
                            ra = (rv - prev_rv) / dt; sa = (sv - prev_sv) / dt
                            accel_err = float(np.linalg.norm(sa - ra))
                            accel_sum_retrace += accel_err; accel_sum_sq_retrace += accel_err ** 2
                            n_accel_retrace += 1
                    prev_vel_valid = True; prev_rv = rv; prev_sv = sv
            # Summary assertions
            dyn_tk = f"{sid_short}_{comp_lbl}_{li}"
            chk(no_forbidden_cross, f"dynamics_edge_no_forbidden_cross_{dyn_tk}",
                family="dynamics_edge", target_key=dyn_tk, field="edge_no_forbidden_cross", source="coordinate_recomputation")
            dynamics_targets_visited.add(lm_key)
            vel_ok = n_vel_retrace - n_accel_retrace == n_first_vel_retrace
            chk(vel_ok, f"boundary_vel_reset_before_accel_{dyn_tk}",
                f"vel={n_vel_retrace} accel={n_accel_retrace} first_vel={n_first_vel_retrace}",
                family="boundary_reset", target_key=dyn_tk, field="boundary_reset_before_accel", source="coordinate_recomputation")
            reset_targets_visited.add(lm_key)
            # Count and sum verification vs stored
            chk(n_vel_retrace == pl.get("_n_vel", 0), f"retrace_vel_count_{sid_short}_{comp_lbl}_l{li}")
            chk(n_accel_retrace == pl.get("_n_accel", 0), f"retrace_accel_count_{sid_short}_{comp_lbl}_l{li}")
            chk(abs(vel_sum_retrace - pl.get("_vel_sum", 0)) < 1e-6, f"retrace_vel_sum_{sid_short}_{comp_lbl}_l{li}")
            chk(abs(vel_sum_sq_retrace - pl.get("_vel_sum_sq", 0)) < 1e-6, f"retrace_vel_sumsq_{sid_short}_{comp_lbl}_l{li}")
            chk(abs(accel_sum_retrace - pl.get("_accel_sum", 0)) < 1e-6, f"retrace_accel_sum_{sid_short}_{comp_lbl}_l{li}")
            chk(abs(accel_sum_sq_retrace - pl.get("_accel_sum_sq", 0)) < 1e-6, f"retrace_accel_sumsq_{sid_short}_{comp_lbl}_l{li}")
            total_used_edges += n_vel_retrace
            total_excluded_edges += n_excluded_retrace
            total_boundary_edges += n_boundary_retrace
            total_first_vel_after_reset += n_first_vel_retrace
            total_accel_edges += n_accel_retrace
    del _norm_cache  # Free memory
    # Build expected landmark key set
    dynamics_expected = set()
    for td in transform_detail_all:
        sid_short = td["sample_id"].replace("S001_P001_", "")
        for pl in td.get("per_landmark", []):
            dynamics_expected.add(f"{sid_short}_{pl['component']}_{pl['landmark']}")
    chk(len(dynamics_expected) == 720, "dynamics_expected_720", f"actual={len(dynamics_expected)}")
    chk(dynamics_targets_visited == dynamics_expected, "dynamics_coverage_complete",
        f"missing={dynamics_expected - dynamics_targets_visited} unexpected={dynamics_targets_visited - dynamics_expected}")
    chk(reset_targets_visited == dynamics_expected, "reset_coverage_complete")

    # v1.11 Section 1: Component 135x36=4860 + Sample 45x36=1620 field-target assertions
    # Common comparison function for numeric and sentinel values
    def _cmpv(stored, expected, tol=1e-6):
        """Compare values: both numeric → tolerance, both str → exact, mismatch → FAIL."""
        s_num = isinstance(stored, (int, float)) and not isinstance(stored, bool)
        e_num = isinstance(expected, (int, float)) and not isinstance(expected, bool)
        if s_num and e_num:
            if math.isnan(stored) and math.isnan(expected): return True
            return abs(stored - expected) < tol
        s_str = isinstance(stored, str)
        e_str = isinstance(expected, str)
        if s_str and e_str: return stored == expected
        # Try numeric conversion for string values
        if s_str and e_num:
            try: return abs(float(stored) - expected) < tol
            except (ValueError, TypeError): return False
        if s_num and e_str:
            try: return abs(stored - float(expected)) < tol
            except (ValueError, TypeError): return False
        return False

    def _recompute_timing_agg(per_lms):
        """Recompute all 20 timing fields from per-landmark events."""
        rc_peak_nn = sum(pl.get("timing", {}).get("n_nonneg_matched", 0) for pl in per_lms)
        rc_peak_neg = sum(pl.get("timing", {}).get("n_negative_inconsistent", 0) for pl in per_lms)
        rc_umr = sum(pl.get("timing", {}).get("n_unmatched_raw", 0) for pl in per_lms)
        rc_umf = sum(pl.get("timing", {}).get("n_unmatched_filt", 0) for pl in per_lms)
        rc_on_nn, rc_on_neg, rc_en_nn, rc_en_neg, rc_xc_nn, rc_xc_neg = 0, 0, 0, 0, 0, 0
        peak_fr, peak_ms = [], []
        on_fr, on_ms, en_fr, en_ms, xc_fr = [], [], [], [], []
        for pl in per_lms:
            for m in pl.get("timing", {}).get("matched_events", []):
                if m.get("status") == "matched":
                    peak_fr.append(m["lag_frames"])
                    if "lag_ms" in m: peak_ms.append(m["lag_ms"])
            for seg_t in pl.get("timing", {}).get("segments", []):
                os_ = seg_t.get("onset_lag", {}).get("status", "")
                if os_ == "nonnegative":
                    rc_on_nn += 1
                    ol = seg_t["onset_lag"]
                    if "lag_frames" in ol: on_fr.append(ol["lag_frames"])
                    if "lag_ms" in ol: on_ms.append(ol["lag_ms"])
                elif os_.startswith("negative"): rc_on_neg += 1
                es_ = seg_t.get("end_lag", {}).get("status", "")
                if es_ == "nonnegative":
                    rc_en_nn += 1
                    el = seg_t["end_lag"]
                    if "lag_frames" in el: en_fr.append(el["lag_frames"])
                    if "lag_ms" in el: en_ms.append(el["lag_ms"])
                elif es_.startswith("negative"): rc_en_neg += 1
                xs_ = seg_t.get("xcorr_lag", {}).get("status", "")
                if xs_ == "nonnegative":
                    rc_xc_nn += 1; xc_fr.append(seg_t["xcorr_lag"].get("lag_frames", 0))
                elif xs_.startswith("negative"): rc_xc_neg += 1
        SENTINEL = "no_valid_nonnegative_events"
        return {
            "peak_nonneg": rc_peak_nn, "peak_negative": rc_peak_neg,
            "peak_unmatched_raw": rc_umr, "peak_unmatched_filt": rc_umf,
            "peak_pooled_mean_frames": float(np.mean(peak_fr)) if peak_fr else SENTINEL,
            "peak_pooled_mean_ms": float(np.mean(peak_ms)) if peak_ms else SENTINEL,
            "onset_nonneg": rc_on_nn, "onset_negative": rc_on_neg,
            "onset_status": "has_nonnegative" if rc_on_nn > 0 else SENTINEL,
            "onset_pooled_mean_frames": float(np.mean(on_fr)) if on_fr else SENTINEL,
            "onset_pooled_mean_ms": float(np.mean(on_ms)) if on_ms else SENTINEL,
            "end_nonneg": rc_en_nn, "end_negative": rc_en_neg,
            "end_status": "has_nonnegative" if rc_en_nn > 0 else SENTINEL,
            "end_pooled_mean_frames": float(np.mean(en_fr)) if en_fr else SENTINEL,
            "end_pooled_mean_ms": float(np.mean(en_ms)) if en_ms else SENTINEL,
            "xcorr_nonneg": rc_xc_nn, "xcorr_negative": rc_xc_neg,
            "xcorr_status": "has_nonnegative" if rc_xc_nn > 0 else SENTINEL,
            "xcorr_pooled_mean_frames": float(np.mean(xc_fr)) if xc_fr else SENTINEL,
        }

    # 36 field definitions: 16 basic stats + 20 timing
    STAT_FIELDS_16 = [
        "traj_n", "traj_sum_m", "traj_mean_mm", "traj_p95_mm", "traj_max_mm",
        "vel_n", "vel_sum", "vel_sum_sq", "vel_rmse",
        "accel_n", "accel_sum", "accel_sum_sq", "accel_rmse",
        "path_raw_sum_m", "path_stab_sum_m", "path_ratio"
    ]
    TIMING_FIELDS_20 = [
        "peak_nonneg", "peak_negative", "peak_unmatched_raw", "peak_unmatched_filt",
        "peak_pooled_mean_frames", "peak_pooled_mean_ms",
        "onset_nonneg", "onset_negative", "onset_status", "onset_pooled_mean_frames", "onset_pooled_mean_ms",
        "end_nonneg", "end_negative", "end_status", "end_pooled_mean_frames", "end_pooled_mean_ms",
        "xcorr_nonneg", "xcorr_negative", "xcorr_status", "xcorr_pooled_mean_frames"
    ]
    ALL_36_FIELDS = STAT_FIELDS_16 + TIMING_FIELDS_20
    comp_agg_field_targets = set(); sample_agg_field_targets = set()

    def _compute_expected_stats(raw_d, raw_v, raw_a, comp_lms):
        """Compute expected 16 basic stat fields from raw distributions."""
        expected = {}
        if raw_d:
            da = np.array(raw_d)
            expected["traj_n"] = len(da); expected["traj_sum_m"] = float(np.sum(da))
            expected["traj_mean_mm"] = float(np.mean(da) * 1000)
            expected["traj_p95_mm"] = float(np.percentile(da, 95) * 1000)
            expected["traj_max_mm"] = float(np.max(da) * 1000)
        else:
            expected["traj_n"] = 0; expected["traj_sum_m"] = 0.0
            expected["traj_mean_mm"] = 0.0; expected["traj_p95_mm"] = 0.0; expected["traj_max_mm"] = 0.0
        if raw_v:
            va = np.array(raw_v)
            expected["vel_n"] = len(va); expected["vel_sum"] = float(np.sum(va))
            expected["vel_sum_sq"] = float(np.sum(va**2)); expected["vel_rmse"] = float(np.sqrt(np.mean(va**2)))
        else:
            expected["vel_n"] = 0; expected["vel_sum"] = 0.0; expected["vel_sum_sq"] = 0.0; expected["vel_rmse"] = 0.0
        if raw_a:
            aa = np.array(raw_a)
            expected["accel_n"] = len(aa); expected["accel_sum"] = float(np.sum(aa))
            expected["accel_sum_sq"] = float(np.sum(aa**2)); expected["accel_rmse"] = float(np.sqrt(np.mean(aa**2)))
        else:
            expected["accel_n"] = 0; expected["accel_sum"] = 0.0; expected["accel_sum_sq"] = 0.0; expected["accel_rmse"] = 0.0
        lm_pr = sum(pl.get("path_length_raw_m", 0) for pl in comp_lms)
        lm_ps = sum(pl.get("path_length_stab_m", 0) for pl in comp_lms)
        expected["path_raw_sum_m"] = lm_pr; expected["path_stab_sum_m"] = lm_ps
        expected["path_ratio"] = float(lm_ps / lm_pr) if lm_pr > 0 else 1.0
        return expected

    for td in transform_detail_all:
        sid_short = td["sample_id"].replace("S001_P001_", "")
        agg = td["aggregate"]
        # Component aggregate: 135 targets x 36 fields = 4860
        for comp_name_a, ca in td.get("component_aggregates", {}).items():
            ckey = f"{sid_short}_{comp_name_a}"
            comp_lms = [pl for pl in td["per_landmark"] if pl["component"] == comp_name_a]
            raw_d = [v for pl_c in comp_lms for v in pl_c.get("_raw_devs_m", [])]
            raw_v = [v for pl_c in comp_lms for v in pl_c.get("_raw_vel_vec_errs", [])]
            raw_a = [v for pl_c in comp_lms for v in pl_c.get("_raw_accel_vec_errs", [])]
            expected_stats = _compute_expected_stats(raw_d, raw_v, raw_a, comp_lms)
            timing_rc = _recompute_timing_agg(comp_lms)
            for field in ALL_36_FIELDS:
                tgt_key = f"{ckey}|{field}"
                if field in STAT_FIELDS_16:
                    stored = ca.get(field); exp = expected_stats[field]
                else:
                    stored = ca.get("timing", {}).get(field); exp = timing_rc[field]
                ok = _cmpv(stored, exp, tol=1e-6)
                results.append({"name": f"caf_{field}_{ckey}", "status": "PASS" if ok else "FAIL",
                    "detail": f"stored={stored} expected={exp}" if not ok else "",
                    "family": "component_aggregate_field", "target_key": tgt_key, "field": field, "source": "raw_distribution"})
                if ok: np_[0] += 1
                else: nf_[0] += 1
                comp_agg_field_targets.add(tgt_key)
        # Sample aggregate: 45 targets x 36 fields = 1620
        all_lms = td["per_landmark"]
        raw_d = [v for pl_s in all_lms for v in pl_s.get("_raw_devs_m", [])]
        raw_v = [v for pl_s in all_lms for v in pl_s.get("_raw_vel_vec_errs", [])]
        raw_a = [v for pl_s in all_lms for v in pl_s.get("_raw_accel_vec_errs", [])]
        expected_stats = _compute_expected_stats(raw_d, raw_v, raw_a, all_lms)
        # Remap sample aggregate field names
        SA_STAT_REMAP = {"traj_mean_mm": "traj_dev_mean_mm", "traj_p95_mm": "traj_dev_p95_mm",
                         "traj_max_mm": "traj_dev_max_mm", "vel_rmse": "velocity_vector_rmse_m_per_s",
                         "accel_rmse": "acceleration_vector_rmse_m_per_s2"}
        timing_rc = _recompute_timing_agg(all_lms)
        for field in ALL_36_FIELDS:
            tgt_key = f"{sid_short}|{field}"
            if field in STAT_FIELDS_16:
                sa_key = SA_STAT_REMAP.get(field, field)
                stored = agg.get(sa_key); exp = expected_stats[field]
            else:
                stored = agg.get("timing", {}).get(field); exp = timing_rc[field]
            ok = _cmpv(stored, exp, tol=1e-6)
            results.append({"name": f"saf_{field}_{sid_short}", "status": "PASS" if ok else "FAIL",
                "detail": f"stored={stored} expected={exp}" if not ok else "",
                "family": "sample_aggregate_field", "target_key": tgt_key, "field": field, "source": "raw_distribution"})
            if ok: np_[0] += 1
            else: nf_[0] += 1
            sample_agg_field_targets.add(tgt_key)
    chk(len(comp_agg_field_targets) == 4860, "comp_agg_field_coverage_4860", f"actual={len(comp_agg_field_targets)}")
    chk(len(sample_agg_field_targets) == 1620, "sample_agg_field_coverage_1620", f"actual={len(sample_agg_field_targets)}")

    # v1.11 Section 2: CSV 180x56=10,080 per-cell JSON-CSV comparison (re-read saved files)
    # Re-read saved transform_comparison_detail.json and transform_comparison.csv
    tc_detail_path = out_dir / "transform_comparison_detail.json"
    tc_csv_path = out_dir / "transform_comparison.csv"
    chk(tc_detail_path.exists(), "transform_detail_json_exists")
    chk(tc_csv_path.exists(), "transform_csv_exists")
    json_csv_cell_targets = set()
    status_membership_targets = set()
    canonical_alias_targets = set()
    idle_jitter_targets = set()
    if tc_detail_path.exists() and tc_csv_path.exists():
        with open(tc_detail_path, encoding="utf-8") as f: saved_detail = json.load(f)
        with open(tc_csv_path, encoding="utf-8") as f: csv_rows = list(csv.DictReader(f))
        n_total_csv_rows = len(csv_rows)
        chk(n_total_csv_rows == 900, "csv_total_rows_900", f"actual={n_total_csv_rows}")
        n_lm_rows = len([r for r in csv_rows if r.get("row_type") == "landmark"])
        n_ca_rows = len([r for r in csv_rows if r.get("row_type") == "component_aggregate"])
        n_sa_rows = len([r for r in csv_rows if r.get("row_type") == "sample_aggregate"])
        chk(n_lm_rows == 720, "csv_landmark_rows_720", f"actual={n_lm_rows}")
        chk(n_ca_rows == 135, "csv_component_rows_135", f"actual={n_ca_rows}")
        chk(n_sa_rows == 45, "csv_sample_rows_45", f"actual={n_sa_rows}")
        blank_count = sum(1 for cr in csv_rows for v in cr.values() if v == "" or v is None)
        chk(blank_count == 0, "csv_zero_blanks", f"blank_cells={blank_count}")
        n_csv_cols = len(csv_rows[0]) if csv_rows else 0
        chk(n_csv_cols == 56, "csv_56_columns", f"actual={n_csv_cols}")
        # Build expected rows from saved JSON detail
        def _build_expected_rows(saved_td_list):
            """Build expected CSV rows from saved transform_comparison_detail.json."""
            expected = {}
            for td in saved_td_list:
                sid_val = td["sample_id"]
                # Landmark rows
                for pl in td.get("per_landmark", []):
                    comp_lbl = pl["component"]; lm_lbl = pl["landmark"]
                    rk = f"{sid_val}|landmark|{comp_lbl}|{lm_lbl}"
                    row = {"sample_id": sid_val, "row_type": "landmark", "component": comp_lbl, "landmark": str(lm_lbl)}
                    row["path_length_raw_m"] = pl.get("path_length_raw_m", 0)
                    row["path_length_stab_m"] = pl.get("path_length_stab_m", 0)
                    row["path_length_ratio"] = pl.get("path_length_ratio", 0)
                    n_traj = pl.get("_n_traj", 0); n_vel = pl.get("_n_vel", 0); n_accel = pl.get("_n_accel", 0)
                    row["n_trajectory_deviations"] = n_traj; row["trajectory_error_sum_m"] = pl.get("_traj_sum_m", 0)
                    row["traj_dev_mean_mm"] = pl.get("traj_dev_mean_mm", 0); row["traj_dev_p95_mm"] = pl.get("traj_dev_p95_mm", 0)
                    row["traj_dev_max_mm"] = pl.get("traj_dev_max_mm", 0)
                    row["n_velocity_errors"] = n_vel; row["velocity_error_sum"] = pl.get("_vel_sum", 0)
                    row["velocity_error_sum_squares"] = pl.get("_vel_sum_sq", 0); row["velocity_vector_rmse_m_per_s"] = pl.get("velocity_vector_rmse_m_per_s", 0)
                    row["n_acceleration_errors"] = n_accel; row["acceleration_error_sum"] = pl.get("_accel_sum", 0)
                    row["acceleration_error_sum_squares"] = pl.get("_accel_sum_sq", 0); row["acceleration_vector_rmse_m_per_s2"] = pl.get("acceleration_vector_rmse_m_per_s2", 0)
                    t = pl.get("timing", {})
                    row["timing_status"] = t.get("status", "missing")
                    row["timing_n_nonneg_matched"] = t.get("n_nonneg_matched", 0)
                    row["n_nonneg_matched"] = t.get("n_nonneg_matched", 0)
                    row["n_negative_inconsistent"] = t.get("n_negative_inconsistent", 0)
                    row["n_unmatched_raw"] = t.get("n_unmatched_raw", 0)
                    row["n_unmatched_filt"] = t.get("n_unmatched_filt", 0)
                    row["matched_peak_lag_frames"] = t.get("mean_lag_frames", "no_valid_nonnegative_events")
                    row["matched_peak_lag_ms"] = t.get("mean_lag_ms", "no_valid_nonnegative_events")
                    # Recompute onset/end/xcorr from nested segments
                    all_onset_ms, all_onset_fr, all_end_ms, all_end_fr, all_xcorr = [], [], [], [], []
                    onset_nn, onset_neg, end_nn, end_neg, xcorr_nn, xcorr_neg = 0, 0, 0, 0, 0, 0
                    for seg_t in t.get("segments", []):
                        ol = seg_t.get("onset_lag", {}); ol_s = ol.get("status", "")
                        if ol_s == "nonnegative":
                            onset_nn += 1
                            if "lag_ms" in ol: all_onset_ms.append(ol["lag_ms"])
                            if "lag_frames" in ol: all_onset_fr.append(ol["lag_frames"])
                        elif ol_s.startswith("negative"): onset_neg += 1
                        el = seg_t.get("end_lag", {}); el_s = el.get("status", "")
                        if el_s == "nonnegative":
                            end_nn += 1
                            if "lag_ms" in el: all_end_ms.append(el["lag_ms"])
                            if "lag_frames" in el: all_end_fr.append(el["lag_frames"])
                        elif el_s.startswith("negative"): end_neg += 1
                        xl = seg_t.get("xcorr_lag", {}); xl_s = xl.get("status", "")
                        if xl_s == "nonnegative":
                            xcorr_nn += 1; all_xcorr.append(xl.get("lag_frames", 0))
                        elif xl_s.startswith("negative"): xcorr_neg += 1
                    SENT = "no_valid_nonnegative_events"
                    row["onset_nonneg"] = onset_nn; row["onset_negative"] = onset_neg
                    row["onset_status"] = "has_nonnegative" if onset_nn > 0 else SENT
                    row["onset_lag_ms_pooled"] = float(np.mean(all_onset_ms)) if all_onset_ms else SENT
                    row["onset_lag_frames_pooled"] = float(np.mean(all_onset_fr)) if all_onset_fr else SENT
                    row["end_nonneg"] = end_nn; row["end_negative"] = end_neg
                    row["end_status"] = "has_nonnegative" if end_nn > 0 else SENT
                    row["end_lag_ms_pooled"] = float(np.mean(all_end_ms)) if all_end_ms else SENT
                    row["end_lag_frames_pooled"] = float(np.mean(all_end_fr)) if all_end_fr else SENT
                    row["xcorr_nonneg"] = xcorr_nn; row["xcorr_negative"] = xcorr_neg
                    row["xcorr_status"] = "has_nonnegative" if xcorr_nn > 0 else SENT
                    row["xcorr_lag_frames_pooled"] = float(np.mean(all_xcorr)) if all_xcorr else SENT
                    # Alias columns
                    row["path_raw_sum_m"] = row["path_length_raw_m"]; row["path_stab_sum_m"] = row["path_length_stab_m"]
                    row["traj_n"] = row["n_trajectory_deviations"]; row["traj_sum_m"] = row["trajectory_error_sum_m"]
                    row["vel_n"] = row["n_velocity_errors"]; row["vel_sum"] = row["velocity_error_sum"]
                    row["vel_sum_sq"] = row["velocity_error_sum_squares"]
                    row["accel_n"] = row["n_acceleration_errors"]; row["accel_sum"] = row["acceleration_error_sum"]
                    row["accel_sum_sq"] = row["acceleration_error_sum_squares"]
                    # Idle-jitter
                    row["idle_jitter_start_raw_mm"] = "not_applicable_landmark"
                    row["idle_jitter_start_stab_mm"] = "not_applicable_landmark"
                    row["idle_jitter_end_raw_mm"] = "not_applicable_landmark"
                    row["idle_jitter_end_stab_mm"] = "not_applicable_landmark"
                    expected[rk] = row
                # Component aggregate rows
                for comp_name_csv, ca in td.get("component_aggregates", {}).items():
                    if not ca: continue
                    rk = f"{sid_val}|component_aggregate|{comp_name_csv}|not_applicable_aggregate"
                    ct = ca.get("timing", {})
                    SENT = "no_valid_nonnegative_events"
                    row = {"sample_id": sid_val, "row_type": "component_aggregate", "component": comp_name_csv, "landmark": "not_applicable_aggregate",
                           "path_length_raw_m": ca.get("path_raw_sum_m", 0), "path_length_stab_m": ca.get("path_stab_sum_m", 0),
                           "path_length_ratio": ca.get("path_ratio", 0),
                           "n_trajectory_deviations": ca.get("traj_n", 0), "trajectory_error_sum_m": ca.get("traj_sum_m", 0),
                           "traj_dev_mean_mm": ca.get("traj_mean_mm", 0), "traj_dev_p95_mm": ca.get("traj_p95_mm", 0), "traj_dev_max_mm": ca.get("traj_max_mm", 0),
                           "n_velocity_errors": ca.get("vel_n", 0), "velocity_error_sum": ca.get("vel_sum", 0),
                           "velocity_error_sum_squares": ca.get("vel_sum_sq", 0), "velocity_vector_rmse_m_per_s": ca.get("vel_rmse", 0),
                           "n_acceleration_errors": ca.get("accel_n", 0), "acceleration_error_sum": ca.get("accel_sum", 0),
                           "acceleration_error_sum_squares": ca.get("accel_sum_sq", 0), "acceleration_vector_rmse_m_per_s2": ca.get("accel_rmse", 0),
                           "timing_status": ct.get("onset_status", SENT), "timing_n_nonneg_matched": ct.get("peak_nonneg", 0),
                           "n_nonneg_matched": ct.get("peak_nonneg", 0), "n_negative_inconsistent": ct.get("peak_negative", 0),
                           "n_unmatched_raw": ct.get("peak_unmatched_raw", 0), "n_unmatched_filt": ct.get("peak_unmatched_filt", 0),
                           "matched_peak_lag_frames": ct.get("peak_pooled_mean_frames", SENT), "matched_peak_lag_ms": ct.get("peak_pooled_mean_ms", SENT),
                           "onset_nonneg": ct.get("onset_nonneg", 0), "onset_negative": ct.get("onset_negative", 0),
                           "onset_status": ct.get("onset_status", SENT), "onset_lag_ms_pooled": ct.get("onset_pooled_mean_ms", SENT),
                           "onset_lag_frames_pooled": ct.get("onset_pooled_mean_frames", SENT),
                           "end_nonneg": ct.get("end_nonneg", 0), "end_negative": ct.get("end_negative", 0),
                           "end_status": ct.get("end_status", SENT), "end_lag_ms_pooled": ct.get("end_pooled_mean_ms", SENT),
                           "end_lag_frames_pooled": ct.get("end_pooled_mean_frames", SENT),
                           "xcorr_nonneg": ct.get("xcorr_nonneg", 0), "xcorr_negative": ct.get("xcorr_negative", 0),
                           "xcorr_status": ct.get("xcorr_status", SENT), "xcorr_lag_frames_pooled": ct.get("xcorr_pooled_mean_frames", SENT),
                           "path_raw_sum_m": ca.get("path_raw_sum_m", 0), "path_stab_sum_m": ca.get("path_stab_sum_m", 0),
                           "traj_n": ca.get("traj_n", 0), "traj_sum_m": ca.get("traj_sum_m", 0),
                           "vel_n": ca.get("vel_n", 0), "vel_sum": ca.get("vel_sum", 0), "vel_sum_sq": ca.get("vel_sum_sq", 0),
                           "accel_n": ca.get("accel_n", 0), "accel_sum": ca.get("accel_sum", 0), "accel_sum_sq": ca.get("accel_sum_sq", 0),
                           "idle_jitter_start_raw_mm": "not_applicable_component_aggregate",
                           "idle_jitter_start_stab_mm": "not_applicable_component_aggregate",
                           "idle_jitter_end_raw_mm": "not_applicable_component_aggregate",
                           "idle_jitter_end_stab_mm": "not_applicable_component_aggregate"}
                    expected[rk] = row
                # Sample aggregate row
                agg_td = td["aggregate"]; ij = td.get("idle_jitter", {}); st = agg_td.get("timing", {})
                SENT = "no_valid_nonnegative_events"
                rk = f"{sid_val}|sample_aggregate|ALL|not_applicable_aggregate"
                row = {"sample_id": sid_val, "row_type": "sample_aggregate", "component": "ALL", "landmark": "not_applicable_aggregate",
                       "path_length_raw_m": agg_td.get("path_raw_sum_m", 0), "path_length_stab_m": agg_td.get("path_stab_sum_m", 0),
                       "path_length_ratio": agg_td.get("path_ratio", 0),
                       "n_trajectory_deviations": agg_td.get("traj_n", 0), "trajectory_error_sum_m": agg_td.get("traj_sum_m", 0),
                       "traj_dev_mean_mm": agg_td["traj_dev_mean_mm"], "traj_dev_p95_mm": agg_td.get("traj_dev_p95_mm", 0),
                       "traj_dev_max_mm": agg_td["traj_dev_max_mm"],
                       "n_velocity_errors": agg_td.get("vel_n", 0), "velocity_error_sum": agg_td.get("vel_sum", 0),
                       "velocity_error_sum_squares": agg_td.get("vel_sum_sq", 0), "velocity_vector_rmse_m_per_s": agg_td.get("velocity_vector_rmse_m_per_s", 0),
                       "n_acceleration_errors": agg_td.get("accel_n", 0), "acceleration_error_sum": agg_td.get("accel_sum", 0),
                       "acceleration_error_sum_squares": agg_td.get("accel_sum_sq", 0), "acceleration_vector_rmse_m_per_s2": agg_td.get("acceleration_vector_rmse_m_per_s2", 0),
                       "timing_status": st.get("onset_status", SENT), "timing_n_nonneg_matched": st.get("peak_nonneg", 0),
                       "n_nonneg_matched": st.get("peak_nonneg", 0), "n_negative_inconsistent": st.get("peak_negative", 0),
                       "n_unmatched_raw": st.get("peak_unmatched_raw", 0), "n_unmatched_filt": st.get("peak_unmatched_filt", 0),
                       "matched_peak_lag_frames": st.get("peak_pooled_mean_frames", SENT), "matched_peak_lag_ms": st.get("peak_pooled_mean_ms", SENT),
                       "onset_nonneg": st.get("onset_nonneg", 0), "onset_negative": st.get("onset_negative", 0),
                       "onset_status": st.get("onset_status", SENT), "onset_lag_ms_pooled": st.get("onset_pooled_mean_ms", SENT),
                       "onset_lag_frames_pooled": st.get("onset_pooled_mean_frames", SENT),
                       "end_nonneg": st.get("end_nonneg", 0), "end_negative": st.get("end_negative", 0),
                       "end_status": st.get("end_status", SENT), "end_lag_ms_pooled": st.get("end_pooled_mean_ms", SENT),
                       "end_lag_frames_pooled": st.get("end_pooled_mean_frames", SENT),
                       "xcorr_nonneg": st.get("xcorr_nonneg", 0), "xcorr_negative": st.get("xcorr_negative", 0),
                       "xcorr_status": st.get("xcorr_status", SENT), "xcorr_lag_frames_pooled": st.get("xcorr_pooled_mean_frames", SENT),
                       "path_raw_sum_m": agg_td.get("path_raw_sum_m", 0), "path_stab_sum_m": agg_td.get("path_stab_sum_m", 0),
                       "traj_n": agg_td.get("traj_n", 0), "traj_sum_m": agg_td.get("traj_sum_m", 0),
                       "vel_n": agg_td.get("vel_n", 0), "vel_sum": agg_td.get("vel_sum", 0), "vel_sum_sq": agg_td.get("vel_sum_sq", 0),
                       "accel_n": agg_td.get("accel_n", 0), "accel_sum": agg_td.get("accel_sum", 0), "accel_sum_sq": agg_td.get("accel_sum_sq", 0),
                       "idle_jitter_start_raw_mm": ij.get("start_raw_mm", "not_available_no_verified_idle"),
                       "idle_jitter_start_stab_mm": ij.get("start_stabilized_mm", "not_available_no_verified_idle"),
                       "idle_jitter_end_raw_mm": ij.get("end_raw_mm", "not_available_no_verified_idle"),
                       "idle_jitter_end_stab_mm": ij.get("end_stabilized_mm", "not_available_no_verified_idle")}
                expected[rk] = row
            return expected
        expected_rows = _build_expected_rows(saved_detail)
        # Build CSV row lookup
        csv_lookup = {}
        for cr in csv_rows:
            sid_c = cr.get("sample_id", ""); rt_c = cr.get("row_type", ""); comp_c = cr.get("component", ""); lm_c = cr.get("landmark", "")
            crk = f"{sid_c}|{rt_c}|{comp_c}|{lm_c}"
            csv_lookup[crk] = cr
        expected_keys = set(expected_rows.keys()); actual_keys = set(csv_lookup.keys())
        missing_rows = expected_keys - actual_keys; unexpected_rows = actual_keys - expected_keys
        chk(len(missing_rows) == 0, "csv_no_missing_rows", f"missing={len(missing_rows)}")
        chk(len(unexpected_rows) == 0, "csv_no_unexpected_rows", f"unexpected={len(unexpected_rows)}")
        dup_csv_rows = len(csv_rows) - len(csv_lookup)
        chk(dup_csv_rows == 0, "csv_no_duplicate_rows", f"dup={dup_csv_rows}")
        # Per-cell comparison: 10,080 cells
        LM_TIMING_STATUS = {"matched", "no_nonneg_matches"}
        AGG_TIMING_STATUS = {"has_nonnegative", "no_valid_nonnegative_events"}
        ONSET_END_XCORR_STATUS = {"has_nonnegative", "no_valid_nonnegative_events"}
        ALIAS_PAIRS = [("path_length_raw_m", "path_raw_sum_m"), ("path_length_stab_m", "path_stab_sum_m"),
                       ("n_trajectory_deviations", "traj_n"), ("trajectory_error_sum_m", "traj_sum_m"),
                       ("n_velocity_errors", "vel_n"), ("velocity_error_sum", "vel_sum"),
                       ("velocity_error_sum_squares", "vel_sum_sq"),
                       ("n_acceleration_errors", "accel_n"), ("acceleration_error_sum", "accel_sum"),
                       ("acceleration_error_sum_squares", "accel_sum_sq")]
        IDLE_JITTER_COLS = ["idle_jitter_start_raw_mm", "idle_jitter_start_stab_mm",
                            "idle_jitter_end_raw_mm", "idle_jitter_end_stab_mm"]
        for rk in sorted(expected_keys & actual_keys):
            exp_row = expected_rows[rk]; csv_row = csv_lookup[rk]
            rt = csv_row.get("row_type", "")
            sid_s = csv_row.get("sample_id", "").replace("S001_P001_", "")
            comp_s = csv_row.get("component", ""); lm_s = csv_row.get("landmark", "")
            row_id = f"{sid_s}_{rt}_{comp_s}" + (f"_l{lm_s}" if rt == "landmark" else "")
            for col in sorted(csv_row.keys()):
                cell_key = f"{rk}|{col}"
                csv_val = csv_row[col]; exp_val = exp_row.get(col)
                # Convert CSV string to comparable type
                ok = False
                if exp_val is None:
                    ok = csv_val != ""  # unexpected col, just check non-blank
                else:
                    ok = _cmpv_csv(csv_val, exp_val)
                results.append({"name": f"cell_{col}_{row_id}", "status": "PASS" if ok else "FAIL",
                    "detail": f"csv={csv_val} exp={exp_val}" if not ok else "",
                    "family": "json_csv_full_cell", "target_key": cell_key, "field": col, "source": "saved_json_vs_csv"})
                if ok: np_[0] += 1
                else: nf_[0] += 1
                json_csv_cell_targets.add(cell_key)
            # Status membership checks (separate family)
            ts_val = csv_row.get("timing_status", "")
            ts_set = LM_TIMING_STATUS if rt == "landmark" else AGG_TIMING_STATUS
            sm_ok = ts_val in ts_set
            results.append({"name": f"ts_membership_{row_id}", "status": "PASS" if sm_ok else "FAIL",
                "detail": f"val={ts_val}" if not sm_ok else "", "family": "status_membership", "target_key": f"{rk}|timing_status"})
            if sm_ok: np_[0] += 1
            else: nf_[0] += 1
            status_membership_targets.add(f"{rk}|timing_status")
            for scol in ["onset_status", "end_status", "xcorr_status"]:
                sv = csv_row.get(scol, ""); sm2 = sv in ONSET_END_XCORR_STATUS
                results.append({"name": f"{scol}_membership_{row_id}", "status": "PASS" if sm2 else "FAIL",
                    "detail": f"val={sv}" if not sm2 else "", "family": "status_membership", "target_key": f"{rk}|{scol}"})
                if sm2: np_[0] += 1
                else: nf_[0] += 1
                status_membership_targets.add(f"{rk}|{scol}")
            # Alias 10 pairs
            for canon, alias in ALIAS_PAIRS:
                cv = csv_row.get(canon, ""); av = csv_row.get(alias, "")
                try: alias_ok = abs(float(cv) - float(av)) < 1e-9
                except (ValueError, TypeError): alias_ok = False
                results.append({"name": f"alias_{canon}_eq_{alias}_{row_id}", "status": "PASS" if alias_ok else "FAIL",
                    "detail": f"canon={cv} alias={av}" if not alias_ok else "", "family": "canonical_alias", "target_key": f"{rk}|{canon}={alias}"})
                if alias_ok: np_[0] += 1
                else: nf_[0] += 1
                canonical_alias_targets.add(f"{rk}|{canon}={alias}")
            # Idle-jitter 4 cols
            for ijcol in IDLE_JITTER_COLS:
                ijv = csv_row.get(ijcol, "")
                if rt == "landmark": ij_ok = ijv == "not_applicable_landmark"
                elif rt == "component_aggregate": ij_ok = ijv == "not_applicable_component_aggregate"
                else: ij_ok = ijv != ""
                results.append({"name": f"ij_{ijcol}_{row_id}", "status": "PASS" if ij_ok else "FAIL",
                    "detail": f"val={ijv}" if not ij_ok else "", "family": "idle_jitter", "target_key": f"{rk}|{ijcol}"})
                if ij_ok: np_[0] += 1
                else: nf_[0] += 1
                idle_jitter_targets.add(f"{rk}|{ijcol}")
        chk(len(json_csv_cell_targets) == 50400, "csv_cell_total_50400", f"actual={len(json_csv_cell_targets)}")
        chk(len(status_membership_targets) == 3600, "status_membership_3600", f"actual={len(status_membership_targets)}")
        chk(len(canonical_alias_targets) == 9000, "canonical_alias_9000", f"actual={len(canonical_alias_targets)}")
        chk(len(idle_jitter_targets) == 3600, "idle_jitter_3600", f"actual={len(idle_jitter_targets)}")

    # v1.9: config/run_manifest settings consistency
    config_files = list(out_dir.glob("step7b_v*_config.json"))
    for cf in config_files:
        with open(cf, encoding="utf-8") as f: cfg_data = json.load(f)
        chk(cfg_data.get("run_id") == RUN_ID, f"config_run_id_{cf.name}")
        chk(cfg_data.get("smoothing_recommendation") == SMOOTHING_CONCLUSION, f"config_smoothing_{cf.name}")
        chk(cfg_data.get("selected_normalization_transform") == SELECTED_NORM_TRANSFORM, f"config_norm_{cf.name}")
        chk(cfg_data.get("selected_max_gap") == SELECTED_MAX_GAP, f"config_gap_{cf.name}")
        chk(cfg_data.get("savgol_note") == SMOOTHING_SAVGOL_NOTE, f"config_savgol_{cf.name}")
        chk(cfg_data.get("version") == VERSION, f"config_version_{cf.name}")

    sp(f"\n  Assertions: {np_[0]} PASS, {nf_[0]} FAIL")
    sp(f"  Dynamics: used_edges={total_used_edges} excluded={total_excluded_edges} boundary={total_boundary_edges}")
    sp(f"  Dynamics: first_vel_after_reset={total_first_vel_after_reset} accel_edges={total_accel_edges}")
    coverage_targets = {
        "dynamics_targets_visited": dynamics_targets_visited,
        "reset_targets_visited": reset_targets_visited,
        "dynamics_expected": dynamics_expected,
        "comp_agg_field_targets": comp_agg_field_targets,
        "sample_agg_field_targets": sample_agg_field_targets,
        "json_csv_cell_targets": json_csv_cell_targets,
        "status_membership_targets": status_membership_targets,
        "canonical_alias_targets": canonical_alias_targets,
        "idle_jitter_targets": idle_jitter_targets,
    }
    return results, np_[0], nf_[0], coverage_targets

# ================================================================
#  Report (v1.6)
# ================================================================

def write_report(out_dir, all_sample_ids, pilot_ids, pilot, seg_rows, masking_rows, masking_agg, np_, nf_, ar_records=None):
    status = f"STEP7B_FULL_V{VERSION.replace('.','_')}_COMPLETE" if nf_ == 0 else f"STEP7B_V{VERSION.replace('.','_')}_BLOCKED"
    total_frames = sum(r.get("total_frames", 0) for r in seg_rows)
    rpt = [f"# Step 7B v{VERSION} Full 45-sample Preprocessing Report", "", f"## Status: {status}", "",
           f"**Run ID**: `{RUN_ID}`", f"**Version**: {VERSION}",
           f"**Assertions**: {np_} PASS / {nf_} FAIL", "",
           f"## Dataset: {len(all_sample_ids)} samples, {total_frames} frames, {total_frames*75} landmarks", "",
           "## Pilot Samples (regression baseline)", "", "| Sample | Tier | TE Rate |", "|--------|------|---------|"]
    for sid in pilot_ids:
        p = pilot[sid]; rpt.append(f"| {sid.replace('S001_P001_','')} | {p['tier']} | {p['te_rate']:.1%} |")
    rpt += ["", "## Masking Experiment (pooled aggregate)", "",
            "| Comp | Gap | N_pos | Mean(mm) | RMSE(mm) | P95(mm) | Max(mm) |",
            "|------|-----|-------|----------|----------|---------|---------|"]
    for r in masking_agg:
        rpt.append(f"| {r['component']} | {r['gap_length']} | {r['n_position_errors']} | "
                   f"{r['recon_err_mean_mm']:.2f} | {r['recon_err_rmse_mm']:.2f} | "
                   f"{r['recon_err_p95_mm']:.2f} | {r['recon_err_max_mm']:.2f} |")
    rpt += ["", "## Selected Defaults", "",
            f"- **Normalization transform**: `{SELECTED_NORM_TRANSFORM}`",
            f"- **Smoothing**: `{SMOOTHING_DEFAULT}`",
            f"- **Smoothing conclusion**: `{SMOOTHING_CONCLUSION}`",
            f"- **SavGol**: `{SMOOTHING_SAVGOL_NOTE}`",
            f"- **Max gap**: `{SELECTED_MAX_GAP}` (gap=1 has lowest position and velocity error; acceleration-vector RMSE is lowest at gap=3, but gap=1 chosen for minimal interpolation and conservative gap policy)", ""]
    # v1.10: assertion family coverage counts + coverage registry
    if ar_records:
        from collections import Counter
        families = Counter()
        for r in ar_records:
            name = r["name"]
            parts = name.split("_")
            family = "_".join(parts[:3]) if len(parts) >= 3 else name
            families[family] += 1
        rpt += ["## Assertion Family Coverage", "",
                "| Family Prefix | Count |", "|---------------|-------|"]
        for fam, cnt in sorted(families.items(), key=lambda x: -x[1])[:30]:
            rpt.append(f"| {fam} | {cnt} |")
        rpt.append("")
    # Coverage registry summary from assertion_results.json
    ar_json_path = out_dir / "assertion_results.json"
    if ar_json_path.exists():
        with open(ar_json_path, encoding="utf-8") as f: ar_cov = json.load(f)
        cov = ar_cov.get("coverage", {})
        if cov:
            rpt += ["## Coverage Registry", "",
                    "| Family | Expected | Actual | Missing | Unexpected | Status |",
                    "|--------|----------|--------|---------|------------|--------|"]
            for fn, fd in cov.items():
                rpt.append(f"| {fn} | {fd['expected_count']} | {fd['actual_count']} | "
                           f"{len(fd['missing_targets'])} | {len(fd['unexpected_targets'])} | {fd['coverage_status']} |")
            rpt.append("")
    with open(out_dir / f"STEP7B_V{VERSION.replace('.','_')}_REPORT.md", "w", encoding="utf-8") as f: f.write("\n".join(rpt))

# ================================================================
#  Helpers
# ================================================================

def _write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def _write_csv_flexible(path, rows):
    if not rows: return
    all_keys = []; seen = set()
    for r in rows:
        for k in r:
            if k not in seen: all_keys.append(k); seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

# ================================================================
#  Main
# ================================================================

def main():
    sp(f"Step 7B v{VERSION}: Full 45-sample Preprocessing")
    sp(f"  Run ID: {RUN_ID}")
    sp(f"  Status: STEP7B_V{VERSION.replace('.','_')}_IN_PROGRESS\n")

    # v1.6: synthetic cross-correlation regression tests
    sp("  0. Cross-correlation regression tests...")
    xcorr_fails = run_xcorr_regression_tests()
    if xcorr_fails:
        for xf in xcorr_fails: sp(f"    [XCORR FAIL] {xf}")
        sp("  ABORTING: cross-correlation sign convention test failed")
        sys.exit(1)
    sp("    3/3 synthetic xcorr tests PASS")

    # v1.8: synthetic timing aggregate regression test
    sp("  0b. Timing aggregate regression tests...")
    timing_fails = run_timing_aggregate_regression_tests()
    if timing_fails:
        for tf in timing_fails: sp(f"    [TIMING FAIL] {tf}")
        sp("  ABORTING: timing aggregate regression test failed")
        sys.exit(1)
    sp("    timing aggregate regression PASS")

    assert STEP6B_DIR.exists()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sd in ["sequences", "contact_sheets", "visualizations", "npz", "provenance"]: (OUT_DIR / sd).mkdir(exist_ok=True)

    sp("  1. Dataset split..."); split = build_dataset_split()
    split_rows = sorted(split.values(), key=lambda x: x["sample_id"])
    with open(OUT_DIR / "step7_dataset_split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["sample_id","word","split","position_in_word"]); w.writeheader(); w.writerows(split_rows)
    with open(OUT_DIR / "step7_dataset_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"samples": split_rows, "counts": {"train": 27, "val": 9, "test": 9}}, f, indent=2)

    # Step 7B: process ALL 45 samples, keep pilot_ids for regression
    all_sample_ids = sorted(split.keys())
    sp(f"  2. Sample selection: {len(all_sample_ids)} samples (all)")
    pilot = select_pilots(split); pilot_ids = sorted(pilot.keys())
    sp(f"    Pilot 9 (for regression): {[s.replace('S001_P001_','') for s in pilot_ids]}")
    with open(OUT_DIR / "pilot_selection_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"pilot_samples": [pilot[s] for s in pilot_ids]}, f, indent=2, default=str)
    for sid in all_sample_ids:
        sp_name = split[sid]["split"]
        sp(f"    {sid.replace('S001_P001_',''):30s} split={sp_name}")

    config_name = f"step7b_v{VERSION.replace('.','_')}_config.json"
    config = {"step": "7B", "version": VERSION, "run_id": RUN_ID, "input_dir": str(STEP6B_DIR.relative_to(PROJECT)),
              "segmentation": {"energy_window": SEG_ENERGY_WINDOW, "idle_threshold": SEG_IDLE_THRESHOLD,
                                "pad_start": SEG_PAD_START, "pad_end": SEG_PAD_END, "min_active": SEG_MIN_ACTIVE},
              "interpolation_candidates": INTERP_CANDIDATES, "one_euro_candidates": ONE_EURO_CANDIDATES,
              "savgol": {"window": SAVGOL_WINDOW, "order": SAVGOL_ORDER},
              "normalization": {"origin": "shoulder_midpoint", "scale": "shoulder_width", "axes": "right_up_forward"},
              "reset_reason_codes": RSN_CODE_MAP, "smoothing_recommendation": SMOOTHING_CONCLUSION,
              "smoothing_default": SMOOTHING_DEFAULT, "savgol_note": SMOOTHING_SAVGOL_NOTE,
              "selected_normalization_transform": SELECTED_NORM_TRANSFORM,
              "selected_max_gap": SELECTED_MAX_GAP}
    with open(OUT_DIR / config_name, "w", encoding="utf-8") as f: json.dump(config, f, indent=2)

    sp("\n  3. Input hashes..."); input_hashes = {}; input_manifest = []
    for sid in all_sample_ids:
        short = sid.replace("S001_P001_", ""); p = RAW3D_DIR / f"landmarks_3d_{short}.json"
        h = sha256_file(p); input_hashes[sid] = h
        input_manifest.append({"sample_id": sid, "file": p.relative_to(PROJECT).as_posix(), "size_bytes": os.path.getsize(p), "sha256": h, "artifact_type": "raw_3d_json"})
    if TA_CSV.exists():
        input_manifest.append({"file": TA_CSV.relative_to(PROJECT).as_posix(), "size_bytes": os.path.getsize(TA_CSV), "sha256": sha256_file(TA_CSV), "artifact_type": "temporal_anomalies_csv"})
    for sid in all_sample_ids:
        short = sid.replace("S001_P001_", ""); mp4 = OVERLAY_DIR / f"reproj_{short}_front.mp4"
        if mp4.exists():
            input_manifest.append({"sample_id": sid, "file": mp4.relative_to(PROJECT).as_posix(), "size_bytes": os.path.getsize(mp4), "sha256": sha256_file(mp4), "artifact_type": "overlay_mp4_front"})
    input_manifest.append({"file": Path(__file__).relative_to(PROJECT).as_posix(), "size_bytes": os.path.getsize(__file__), "sha256": sha256_file(__file__), "artifact_type": "execution_script"})
    cfg_path = OUT_DIR / config_name
    input_manifest.append({"file": cfg_path.relative_to(PROJECT).as_posix(), "size_bytes": os.path.getsize(cfg_path), "sha256": sha256_file(cfg_path), "artifact_type": "execution_config"})
    with open(OUT_DIR / "input_file_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"self_reference_excluded": True, "inputs": input_manifest}, f, indent=2)

    sp("  4. Loading strong-depth boundaries...")
    strong_depth, sd_prov_records = load_strong_depth_data(set(all_sample_ids))
    sp(f"    Total: {sum(len(v) for v in strong_depth.values())}")
    sd_prov_map = {}
    for rec in sd_prov_records: sd_prov_map.setdefault(rec["sample_id"], {})[(rec["frame_index"], rec["component"], rec["landmark_index"])] = rec
    with open(OUT_DIR / "strong_depth_provenance.json", "w", encoding="utf-8") as f: json.dump(sd_prov_records, f, indent=2)

    sp(f"\n  5. Processing all {len(all_sample_ids)} samples...")
    all_results = {}; seg_rows, norm_rows, interp_rows, smooth_rows = [], [], [], []
    masking_all = []; all_raw_errors = {}
    transform_rows, transform_detail = [], []
    boundary_reasons_all = {}
    is_pilot = set(pilot_ids)

    for sid in all_sample_ids:
        short = sid.replace("S001_P001_", ""); word = split[sid]["word"]
        sp(f"\n    === {short} ===")
        raw_data = load_sample(sid); arrs = extract_arrays(raw_data)
        N = len(arrs["timestamps"]); ts = arrs["timestamps"]
        sd_set = strong_depth.get(sid, set()); sd_prov_sid = sd_prov_map.get(sid, {})
        norm = normalize_sequence(arrs)
        sc = defaultdict(int)
        for s in norm["statuses"]: sc[s] += 1
        sp(f"      Norm: {dict(sc)}, scale_mean={np.nanmean(norm['scales']):.4f}")
        vs = norm["scales"][~np.isnan(norm["scales"])]
        norm_rows.append({"sample_id": sid, "n_frames": N, "n_ok": sc.get("ok", 0), "scale_mean_m": float(np.mean(vs)) if len(vs) else None})
        stab = compute_causal_stabilized_transform(norm, ts)
        seg = detect_segment(arrs, norm)
        sp(f"      Seg: [{seg['seg_start']}-{seg['seg_end']}] ({seg['seg_frames']}/{N}f) idle_end={seg['idle_verified_end']}")
        seg_rows.append({"sample_id": sid, "total_frames": N, "seg_start": seg["seg_start"], "seg_end": seg["seg_end"],
                          "seg_frames": seg["seg_frames"], "method": seg["method"],
                          "idle_verified_start": seg["idle_verified_start"], "idle_verified_end": seg["idle_verified_end"]})
        make_seg_contact_sheet(sid, seg, OUT_DIR / "contact_sheets")
        frame_disc, boundaries, boundary_reasons, strong_forbidden = build_all_boundaries(arrs, sd_set, sid)
        boundary_reasons_all[sid] = boundary_reasons

        # Transform comparison (v1.5: boundary-aware)
        tc = compare_transforms(arrs, norm, stab, ts, seg, boundaries)
        transform_rows.append({"sample_id": sid, **tc["aggregate"]})
        transform_detail.append({"sample_id": sid, **tc})
        sp(f"      Transform: traj_dev_mean={tc['aggregate']['traj_dev_mean_mm']:.2f}mm bd_skip={tc['aggregate']['boundary_edges_skipped']}")

        sp(f"      Interpolation...")
        interp_data = {}; norm_interp_cache = {}
        for mg in INTERP_CANDIDATES:
            ir = {}
            for comp, nlm in [("pose", 33), ("lh", 21), ("rh", 21)]:
                out_xyz, mask, rej, records = interpolate_gaps(
                    arrs[f"{comp}_xyz"], arrs[f"{comp}_te"], boundaries[comp], strong_forbidden[comp], mg,
                    arrs[f"{comp}_invalid_reason"], arrs[f"{comp}_flags"], sd_prov_sid, comp, sid)
                ir[f"{comp}_interp"] = out_xyz; ir[f"{comp}_mask"] = mask; ir[f"{comp}_rejected"] = rej
                ir[f"{comp}_records"] = records; ir[f"{comp}_boundary"] = boundaries[comp]; ir[f"{comp}_strong_forbidden"] = strong_forbidden[comp]
                valid_sm = arrs[f"{comp}_te"] | mask
                ni = normalize_metric_coords(out_xyz, valid_sm, norm); norm_interp_cache[f"norm_interp_{comp}_g{mg}"] = ni
                interp_rows.append({"sample_id": sid, "component": comp, "max_gap": mg,
                                    "n_filled": int(np.sum(mask)), "n_rejected": int(np.sum(rej)),
                                    "n_strong_depth_interp": int(np.sum(mask & strong_forbidden[comp]))})
            interp_data[mg] = ir
            sp(f"        g={mg}: pose={int(np.sum(ir['pose_mask']))} lh={int(np.sum(ir['lh_mask']))} rh={int(np.sum(ir['rh_mask']))}")

        sp(f"      Masking experiment...")
        me, me_raw = run_masking_experiment(arrs, norm, boundaries, strong_forbidden, ts, sid)
        masking_all.extend(me)
        for k, v in me_raw.items():
            all_raw_errors.setdefault(k, {"pos": [], "speed_mag": [], "vel_vec": [], "accel_mag": [], "accel_vec": []})
            for ek in v: all_raw_errors[k][ek].extend(v[ek])

        sp(f"      Smoothing sweep...")
        base = interp_data[2]; smooth_data = {}
        ns = {}
        for comp in ["pose", "lh", "rh"]:
            valid_sm = arrs[f"{comp}_te"] | base[f"{comp}_mask"]; ni = norm_interp_cache[f"norm_interp_{comp}_g2"]
            rm, far, rsn = compute_filter_masks(base[f"{comp}_interp"], valid_sm, base[f"{comp}_boundary"], boundary_reasons[comp])
            ns[f"{comp}_filtered"] = base[f"{comp}_interp"].copy(); ns[f"{comp}_norm_filtered"] = ni.copy()
            ns[f"{comp}_filter_valid"] = valid_sm; ns[f"{comp}_reset_mask"] = rm; ns[f"{comp}_first_after_reset"] = far; ns[f"{comp}_reset_reason"] = rsn
            ns[f"{comp}_metrics"] = {"label": "none", "n_valid": int(np.sum(valid_sm)),
                "phys_dev_mean_mm": 0.0, "phys_dev_p95_mm": 0.0, "phys_dev_max_mm": 0.0,
                "vel_distortion_pct": 0.0, "vel_rmse_m_per_s": 0.0, "accel_rmse_m_per_s2": 0.0,
                "path_length_ratio": 1.0, "new_nans": 0, "new_infs": 0,
                "jitter_raw_start_idle_mm": "not_available_no_verified_idle", "jitter_filt_start_idle_mm": "not_available_no_verified_idle",
                "jitter_reduction_start_idle": "not_available_no_verified_idle", "jitter_raw_end_idle_mm": "not_available_no_verified_idle",
                "jitter_filt_end_idle_mm": "not_available_no_verified_idle", "jitter_reduction_end_idle": "not_available_no_verified_idle",
                "peak_vel_lag_frames": 0, "peak_vel_lag_ms": 0, "peak_match_status": "identity",
                "per_landmark": [], "peak_lag_detail": {"status": "identity"}}
            smooth_rows.append({"sample_id": sid, "component": comp, "filter": "none", **ns[f"{comp}_metrics"]})
        smooth_data["none"] = ns

        for oe_params in ONE_EURO_CANDIDATES:
            oe = {}
            for comp in ["pose", "lh", "rh"]:
                valid_sm = arrs[f"{comp}_te"] | base[f"{comp}_mask"]; bd = base[f"{comp}_boundary"]; br = boundary_reasons[comp]
                filtered, rm, far, rsn = apply_one_euro(base[f"{comp}_interp"], valid_sm, ts, bd, br, oe_params)
                norm_filt = normalize_metric_coords(filtered, valid_sm, norm); norm_base = norm_interp_cache[f"norm_interp_{comp}_g2"]
                oe[f"{comp}_filtered"] = filtered; oe[f"{comp}_norm_filtered"] = norm_filt; oe[f"{comp}_filter_valid"] = valid_sm
                oe[f"{comp}_reset_mask"] = rm; oe[f"{comp}_first_after_reset"] = far; oe[f"{comp}_reset_reason"] = rsn
                oe[f"{comp}_metrics"] = compute_extended_metrics(norm_base, norm_filt, valid_sm, norm["scales"], ts, bd, seg,
                    f"oe_{oe_params['name']}", word, comp_nlm=(comp, None))
                if comp == "pose":
                    oe[f"{comp}_sign_metrics"] = compute_sign_metrics(norm_base, norm_filt, valid_sm, arrs, norm, seg, word, ts, norm["scales"], bd)
                smooth_rows.append({"sample_id": sid, "component": comp, "filter": f"one_euro_{oe_params['name']}", **oe[f"{comp}_metrics"]})
            smooth_data[f"oe_{oe_params['name']}"] = oe
            pm = oe["pose_metrics"]
            sp(f"        {oe_params['name']}: dev={pm['phys_dev_mean_mm']:.1f}mm path={pm['path_length_ratio']:.3f} "
               f"far={sum(int(np.sum(oe[f'{c}_first_after_reset'])) for c in ['pose','lh','rh'])}")

        if HAS_SCIPY:
            sg = {}
            for comp in ["pose", "lh", "rh"]:
                valid_sm = arrs[f"{comp}_te"] | base[f"{comp}_mask"]; bd = base[f"{comp}_boundary"]; br = boundary_reasons[comp]
                filtered = apply_savgol(base[f"{comp}_interp"], valid_sm, bd)
                rm, far, rsn = compute_filter_masks(base[f"{comp}_interp"], valid_sm, bd, br)
                base_interp = base[f"{comp}_interp"]
                for fi in range(far.shape[0]):
                    for li in range(far.shape[1]):
                        if far[fi, li] and not np.any(np.isnan(base_interp[fi, li])): filtered[fi, li] = base_interp[fi, li].copy()
                norm_filt = normalize_metric_coords(filtered, valid_sm, norm); norm_base = norm_interp_cache[f"norm_interp_{comp}_g2"]
                for fi in range(far.shape[0]):
                    for li in range(far.shape[1]):
                        if far[fi, li] and not np.any(np.isnan(norm_base[fi, li])): norm_filt[fi, li] = norm_base[fi, li].copy()
                sg[f"{comp}_filtered"] = filtered; sg[f"{comp}_norm_filtered"] = norm_filt; sg[f"{comp}_filter_valid"] = valid_sm
                sg[f"{comp}_reset_mask"] = rm; sg[f"{comp}_first_after_reset"] = far; sg[f"{comp}_reset_reason"] = rsn
                sg[f"{comp}_metrics"] = compute_extended_metrics(norm_base, norm_filt, valid_sm, norm["scales"], ts, bd, seg, "savgol", word, comp_nlm=(comp, None))
                smooth_rows.append({"sample_id": sid, "component": comp, "filter": "savgol", **sg[f"{comp}_metrics"]})
            smooth_data["savgol"] = sg

        save_sample_npz(sid, arrs, norm, interp_data, smooth_data, norm_interp_cache, stab, OUT_DIR)
        npz_fails = verify_sample_npz(sid, arrs, norm, interp_data, smooth_data, norm_interp_cache, stab, OUT_DIR)
        sp(f"      NPZ: {'PASS' if not npz_fails else 'FAIL ' + str(npz_fails[:3])}")

        # v1.4-v1.11 byte-exact only for pilot samples (prior versions only have pilot NPZs)
        v14_fails, v15_fails, v16_fails, v17_fails = [], [], [], []
        v18_fails, v19_fails, v110_fails, v111_fails = [], [], [], []
        if sid in is_pilot:
            v14_fails = verify_npz_byte_exact(sid, OUT_DIR, V14_NPZ_DIR, "v14")
            v15_fails = verify_npz_byte_exact(sid, OUT_DIR, V15_NPZ_DIR, "v15")
            v16_fails = verify_npz_byte_exact(sid, OUT_DIR, V16_NPZ_DIR, "v16")
            v17_fails = verify_npz_byte_exact(sid, OUT_DIR, V17_NPZ_DIR, "v17")
            sp(f"      v1.4 byte-exact: {'PASS' if not v14_fails else 'FAIL ' + str(v14_fails[:3])}")
            sp(f"      v1.5 byte-exact: {'PASS' if not v15_fails else 'FAIL ' + str(v15_fails[:3])}")
            sp(f"      v1.6 byte-exact: {'PASS' if not v16_fails else 'FAIL ' + str(v16_fails[:3])}")
            sp(f"      v1.7 byte-exact: {'PASS' if not v17_fails else 'FAIL ' + str(v17_fails[:3])}")

        if HAS_MPL: plot_seg(sid, arrs, norm, seg, OUT_DIR / "visualizations")
        save_full_sequence(sid, arrs, norm, seg, OUT_DIR)

        if sid in is_pilot:
            v18_fails = verify_npz_byte_exact(sid, OUT_DIR, V18_NPZ_DIR, "v18")
            sp(f"      v1.8 byte-exact: {'PASS' if not v18_fails else 'FAIL ' + str(v18_fails[:3])}")
            v19_fails = verify_npz_byte_exact(sid, OUT_DIR, V19_NPZ_DIR, "v19")
            sp(f"      v1.9 byte-exact: {'PASS' if not v19_fails else 'FAIL ' + str(v19_fails[:3])}")
            v110_fails = verify_npz_byte_exact(sid, OUT_DIR, V110_NPZ_DIR, "v110")
            sp(f"      v1.10 byte-exact: {'PASS' if not v110_fails else 'FAIL ' + str(v110_fails[:3])}")
            v111_fails = verify_npz_byte_exact(sid, OUT_DIR, V111_NPZ_DIR, "v111")
            sp(f"      v1.11 byte-exact: {'PASS' if not v111_fails else 'FAIL ' + str(v111_fails[:3])}")

        # v1.12 pilot byte-exact regression (pilot samples only)
        v112_pilot_fails = []
        if sid in is_pilot:
            v112_pilot_fails = verify_npz_byte_exact(sid, OUT_DIR, V112_PILOT_NPZ_DIR, "v112pilot")
            sp(f"      v1.12 pilot byte-exact: {'PASS' if not v112_pilot_fails else 'FAIL ' + str(v112_pilot_fails[:3])}")

        all_results[sid] = {"arrs": arrs, "raw_data": raw_data, "norm": norm, "seg": seg,
                            "interp": interp_data, "smooth": smooth_data, "boundaries": boundaries,
                            "stab": stab, "is_pilot": sid in is_pilot,
                            "npz_verify_fails": npz_fails, "v14_npz_fails": v14_fails, "v15_npz_fails": v15_fails,
                            "v16_npz_fails": v16_fails, "v17_npz_fails": v17_fails, "v18_npz_fails": v18_fails,
                            "v19_npz_fails": v19_fails, "v110_npz_fails": v110_fails, "v111_npz_fails": v111_fails,
                            "v112_pilot_fails": v112_pilot_fails}

    sp("\n  6. Saving CSVs...")
    masking_agg = compute_masking_pooled_aggregate(all_raw_errors, masking_all)
    _write_csv(OUT_DIR / "segmentation_summary.csv", seg_rows)
    _write_csv(OUT_DIR / "normalization_summary.csv", norm_rows)
    _write_csv(OUT_DIR / "interpolation_comparison.csv", interp_rows)
    _write_csv_flexible(OUT_DIR / "smoothing_comparison.csv", smooth_rows)
    _write_csv_flexible(OUT_DIR / "masking_experiment.csv", masking_all + masking_agg)

    tc_csv_rows = []
    for td in transform_detail:
        sid_val = td["sample_id"]
        for lm in td["per_landmark"]:
            t = lm.get("timing", {})
            n_traj = len(lm.get("_raw_devs_m", [])) if "_raw_devs_m" in lm else lm.get("_n_traj", 0)
            n_vel = len(lm.get("_raw_vel_vec_errs", [])) if "_raw_vel_vec_errs" in lm else lm.get("_n_vel", 0)
            n_accel = len(lm.get("_raw_accel_vec_errs", [])) if "_raw_accel_vec_errs" in lm else lm.get("_n_accel", 0)
            row = {"sample_id": sid_val, "row_type": "landmark",
                   "component": lm["component"], "landmark": lm["landmark"],
                   "path_length_raw_m": lm["path_length_raw_m"],
                   "path_length_stab_m": lm["path_length_stab_m"],
                   "path_length_ratio": lm["path_length_ratio"],
                   "n_trajectory_deviations": n_traj,
                   "trajectory_error_sum_m": lm.get("_traj_sum_m", 0),
                   "traj_dev_mean_mm": lm["traj_dev_mean_mm"], "traj_dev_p95_mm": lm["traj_dev_p95_mm"],
                   "traj_dev_max_mm": lm["traj_dev_max_mm"],
                   "n_velocity_errors": n_vel,
                   "velocity_error_sum": lm.get("_vel_sum", 0),
                   "velocity_error_sum_squares": lm.get("_vel_sum_sq", 0),
                   "velocity_vector_rmse_m_per_s": lm["velocity_vector_rmse_m_per_s"],
                   "n_acceleration_errors": n_accel,
                   "acceleration_error_sum": lm.get("_accel_sum", 0),
                   "acceleration_error_sum_squares": lm.get("_accel_sum_sq", 0),
                   "acceleration_vector_rmse_m_per_s2": lm["acceleration_vector_rmse_m_per_s2"],
                   "timing_status": t.get("status", "missing"),
                   "timing_n_nonneg_matched": t.get("n_nonneg_matched", 0),
                   "n_nonneg_matched": t.get("n_nonneg_matched", 0),
                   "n_negative_inconsistent": t.get("n_negative_inconsistent", 0),
                   "n_unmatched_raw": t.get("n_unmatched_raw", 0),
                   "n_unmatched_filt": t.get("n_unmatched_filt", 0),
                   "matched_peak_lag_frames": t.get("mean_lag_frames", "no_valid_nonnegative_events"),
                   "matched_peak_lag_ms": t.get("mean_lag_ms", "no_valid_nonnegative_events")}
            # Pooled onset/end/xcorr from all segments
            all_onset_ms, all_onset_fr, all_end_ms, all_end_fr, all_xcorr = [], [], [], [], []
            onset_nn, onset_neg, end_nn, end_neg, xcorr_nn, xcorr_neg = 0, 0, 0, 0, 0, 0
            for seg_t in t.get("segments", []):
                ol = seg_t.get("onset_lag", {}); ol_s = ol.get("status", "")
                if ol_s == "nonnegative":
                    onset_nn += 1
                    if "lag_ms" in ol: all_onset_ms.append(ol["lag_ms"])
                    if "lag_frames" in ol: all_onset_fr.append(ol["lag_frames"])
                elif ol_s.startswith("negative"): onset_neg += 1
                el = seg_t.get("end_lag", {}); el_s = el.get("status", "")
                if el_s == "nonnegative":
                    end_nn += 1
                    if "lag_ms" in el: all_end_ms.append(el["lag_ms"])
                    if "lag_frames" in el: all_end_fr.append(el["lag_frames"])
                elif el_s.startswith("negative"): end_neg += 1
                xl = seg_t.get("xcorr_lag", {}); xl_s = xl.get("status", "")
                if xl_s == "nonnegative":
                    xcorr_nn += 1; all_xcorr.append(xl.get("lag_frames", 0))
                elif xl_s.startswith("negative"): xcorr_neg += 1
            row["onset_nonneg"] = onset_nn; row["onset_negative"] = onset_neg
            row["onset_status"] = "has_nonnegative" if onset_nn > 0 else "no_valid_nonnegative_events"
            row["onset_lag_ms_pooled"] = float(np.mean(all_onset_ms)) if all_onset_ms else "no_valid_nonnegative_events"
            row["onset_lag_frames_pooled"] = float(np.mean(all_onset_fr)) if all_onset_fr else "no_valid_nonnegative_events"
            row["end_nonneg"] = end_nn; row["end_negative"] = end_neg
            row["end_status"] = "has_nonnegative" if end_nn > 0 else "no_valid_nonnegative_events"
            row["end_lag_ms_pooled"] = float(np.mean(all_end_ms)) if all_end_ms else "no_valid_nonnegative_events"
            row["end_lag_frames_pooled"] = float(np.mean(all_end_fr)) if all_end_fr else "no_valid_nonnegative_events"
            row["xcorr_nonneg"] = xcorr_nn; row["xcorr_negative"] = xcorr_neg
            row["xcorr_status"] = "has_nonnegative" if xcorr_nn > 0 else "no_valid_nonnegative_events"
            row["xcorr_lag_frames_pooled"] = float(np.mean(all_xcorr)) if all_xcorr else "no_valid_nonnegative_events"
            # v1.9 Section 5: fill alias columns for landmark rows (same as canonical)
            row["path_raw_sum_m"] = row["path_length_raw_m"]
            row["path_stab_sum_m"] = row["path_length_stab_m"]
            row["traj_n"] = row["n_trajectory_deviations"]
            row["traj_sum_m"] = row["trajectory_error_sum_m"]
            row["vel_n"] = row["n_velocity_errors"]
            row["vel_sum"] = row["velocity_error_sum"]
            row["vel_sum_sq"] = row["velocity_error_sum_squares"]
            row["accel_n"] = row["n_acceleration_errors"]
            row["accel_sum"] = row["acceleration_error_sum"]
            row["accel_sum_sq"] = row["acceleration_error_sum_squares"]
            # v1.9 Section 5: idle-jitter not applicable for landmark rows
            row["idle_jitter_start_raw_mm"] = "not_applicable_landmark"
            row["idle_jitter_start_stab_mm"] = "not_applicable_landmark"
            row["idle_jitter_end_raw_mm"] = "not_applicable_landmark"
            row["idle_jitter_end_stab_mm"] = "not_applicable_landmark"
            tc_csv_rows.append(row)
        # Per-component aggregate rows
        for comp_name_csv, ca in td.get("component_aggregates", {}).items():
            if not ca: continue
            ct = ca.get("timing", {})
            tc_csv_rows.append({"sample_id": sid_val, "row_type": "component_aggregate",
                                "component": comp_name_csv, "landmark": "not_applicable_aggregate",
                                "path_length_raw_m": ca.get("path_raw_sum_m", 0),
                                "path_length_stab_m": ca.get("path_stab_sum_m", 0),
                                "path_length_ratio": ca.get("path_ratio", 0),
                                "n_trajectory_deviations": ca.get("traj_n", 0),
                                "trajectory_error_sum_m": ca.get("traj_sum_m", 0),
                                "traj_dev_mean_mm": ca.get("traj_mean_mm", 0),
                                "traj_dev_p95_mm": ca.get("traj_p95_mm", 0),
                                "traj_dev_max_mm": ca.get("traj_max_mm", 0),
                                "n_velocity_errors": ca.get("vel_n", 0),
                                "velocity_error_sum": ca.get("vel_sum", 0),
                                "velocity_error_sum_squares": ca.get("vel_sum_sq", 0),
                                "velocity_vector_rmse_m_per_s": ca.get("vel_rmse", 0),
                                "n_acceleration_errors": ca.get("accel_n", 0),
                                "acceleration_error_sum": ca.get("accel_sum", 0),
                                "acceleration_error_sum_squares": ca.get("accel_sum_sq", 0),
                                "acceleration_vector_rmse_m_per_s2": ca.get("accel_rmse", 0),
                                "timing_status": ct.get("onset_status", "no_valid_nonnegative_events"),
                                "timing_n_nonneg_matched": ct.get("peak_nonneg", 0),
                                "n_nonneg_matched": ct.get("peak_nonneg", 0),
                                "n_negative_inconsistent": ct.get("peak_negative", 0),
                                "n_unmatched_raw": ct.get("peak_unmatched_raw", 0),
                                "n_unmatched_filt": ct.get("peak_unmatched_filt", 0),
                                "matched_peak_lag_frames": ct.get("peak_pooled_mean_frames", "no_valid_nonnegative_events"),
                                "matched_peak_lag_ms": ct.get("peak_pooled_mean_ms", "no_valid_nonnegative_events"),
                                "onset_nonneg": ct.get("onset_nonneg", 0), "onset_negative": ct.get("onset_negative", 0),
                                "onset_status": ct.get("onset_status", "no_valid_nonnegative_events"),
                                "onset_lag_ms_pooled": ct.get("onset_pooled_mean_ms", "no_valid_nonnegative_events"),
                                "onset_lag_frames_pooled": ct.get("onset_pooled_mean_frames", "no_valid_nonnegative_events"),
                                "end_nonneg": ct.get("end_nonneg", 0), "end_negative": ct.get("end_negative", 0),
                                "end_status": ct.get("end_status", "no_valid_nonnegative_events"),
                                "end_lag_ms_pooled": ct.get("end_pooled_mean_ms", "no_valid_nonnegative_events"),
                                "end_lag_frames_pooled": ct.get("end_pooled_mean_frames", "no_valid_nonnegative_events"),
                                "xcorr_nonneg": ct.get("xcorr_nonneg", 0), "xcorr_negative": ct.get("xcorr_negative", 0),
                                "xcorr_status": ct.get("xcorr_status", "no_valid_nonnegative_events"),
                                "xcorr_lag_frames_pooled": ct.get("xcorr_pooled_mean_frames", "no_valid_nonnegative_events"),
                                "path_raw_sum_m": ca.get("path_raw_sum_m", 0), "path_stab_sum_m": ca.get("path_stab_sum_m", 0),
                                "traj_n": ca.get("traj_n", 0), "traj_sum_m": ca.get("traj_sum_m", 0),
                                "vel_n": ca.get("vel_n", 0), "vel_sum": ca.get("vel_sum", 0), "vel_sum_sq": ca.get("vel_sum_sq", 0),
                                "accel_n": ca.get("accel_n", 0), "accel_sum": ca.get("accel_sum", 0), "accel_sum_sq": ca.get("accel_sum_sq", 0),
                                "idle_jitter_start_raw_mm": "not_applicable_component_aggregate",
                                "idle_jitter_start_stab_mm": "not_applicable_component_aggregate",
                                "idle_jitter_end_raw_mm": "not_applicable_component_aggregate",
                                "idle_jitter_end_stab_mm": "not_applicable_component_aggregate"})
        # Sample aggregate row
        agg = td["aggregate"]; ij = td.get("idle_jitter", {}); st = agg.get("timing", {})
        tc_csv_rows.append({"sample_id": sid_val, "row_type": "sample_aggregate",
                            "component": "ALL", "landmark": "not_applicable_aggregate",
                            "path_length_raw_m": agg.get("path_raw_sum_m", 0),
                            "path_length_stab_m": agg.get("path_stab_sum_m", 0),
                            "path_length_ratio": agg.get("path_ratio", 0),
                            "n_trajectory_deviations": agg.get("traj_n", 0),
                            "trajectory_error_sum_m": agg.get("traj_sum_m", 0),
                            "traj_dev_mean_mm": agg["traj_dev_mean_mm"],
                            "traj_dev_p95_mm": agg.get("traj_dev_p95_mm", 0),
                            "traj_dev_max_mm": agg["traj_dev_max_mm"],
                            "n_velocity_errors": agg.get("vel_n", 0),
                            "velocity_error_sum": agg.get("vel_sum", 0),
                            "velocity_error_sum_squares": agg.get("vel_sum_sq", 0),
                            "velocity_vector_rmse_m_per_s": agg.get("velocity_vector_rmse_m_per_s", 0),
                            "n_acceleration_errors": agg.get("accel_n", 0),
                            "acceleration_error_sum": agg.get("accel_sum", 0),
                            "acceleration_error_sum_squares": agg.get("accel_sum_sq", 0),
                            "acceleration_vector_rmse_m_per_s2": agg.get("acceleration_vector_rmse_m_per_s2", 0),
                            "timing_status": st.get("onset_status", "no_valid_nonnegative_events"),
                            "timing_n_nonneg_matched": st.get("peak_nonneg", 0),
                            "n_nonneg_matched": st.get("peak_nonneg", 0),
                            "n_negative_inconsistent": st.get("peak_negative", 0),
                            "n_unmatched_raw": st.get("peak_unmatched_raw", 0),
                            "n_unmatched_filt": st.get("peak_unmatched_filt", 0),
                            "matched_peak_lag_frames": st.get("peak_pooled_mean_frames", "no_valid_nonnegative_events"),
                            "matched_peak_lag_ms": st.get("peak_pooled_mean_ms", "no_valid_nonnegative_events"),
                            "onset_nonneg": st.get("onset_nonneg", 0), "onset_negative": st.get("onset_negative", 0),
                            "onset_status": st.get("onset_status", "no_valid_nonnegative_events"),
                            "onset_lag_ms_pooled": st.get("onset_pooled_mean_ms", "no_valid_nonnegative_events"),
                            "onset_lag_frames_pooled": st.get("onset_pooled_mean_frames", "no_valid_nonnegative_events"),
                            "end_nonneg": st.get("end_nonneg", 0), "end_negative": st.get("end_negative", 0),
                            "end_status": st.get("end_status", "no_valid_nonnegative_events"),
                            "end_lag_ms_pooled": st.get("end_pooled_mean_ms", "no_valid_nonnegative_events"),
                            "end_lag_frames_pooled": st.get("end_pooled_mean_frames", "no_valid_nonnegative_events"),
                            "xcorr_nonneg": st.get("xcorr_nonneg", 0), "xcorr_negative": st.get("xcorr_negative", 0),
                            "xcorr_status": st.get("xcorr_status", "no_valid_nonnegative_events"),
                            "xcorr_lag_frames_pooled": st.get("xcorr_pooled_mean_frames", "no_valid_nonnegative_events"),
                            "path_raw_sum_m": agg.get("path_raw_sum_m", 0), "path_stab_sum_m": agg.get("path_stab_sum_m", 0),
                            "traj_n": agg.get("traj_n", 0), "traj_sum_m": agg.get("traj_sum_m", 0),
                            "vel_n": agg.get("vel_n", 0), "vel_sum": agg.get("vel_sum", 0), "vel_sum_sq": agg.get("vel_sum_sq", 0),
                            "accel_n": agg.get("accel_n", 0), "accel_sum": agg.get("accel_sum", 0), "accel_sum_sq": agg.get("accel_sum_sq", 0),
                            "idle_jitter_start_raw_mm": ij.get("start_raw_mm", "not_available_no_verified_idle"),
                            "idle_jitter_start_stab_mm": ij.get("start_stabilized_mm", "not_available_no_verified_idle"),
                            "idle_jitter_end_raw_mm": ij.get("end_raw_mm", "not_available_no_verified_idle"),
                            "idle_jitter_end_stab_mm": ij.get("end_stabilized_mm", "not_available_no_verified_idle")})
    _write_csv_flexible(OUT_DIR / "transform_comparison.csv", tc_csv_rows)
    # v1.11: Write detail JSON BEFORE assertions so Section 2 per-cell comparison can re-read it
    # Raw arrays (_raw_devs_m etc.) still present here — stripped AFTER assertions use them
    with open(OUT_DIR / "transform_comparison_detail.json", "w", encoding="utf-8") as f:
        json.dump(transform_detail, f, indent=2, default=str)

    # Peak lag detail (smoothing)
    peak_lag_details = {}
    for sid, sr in all_results.items():
        for fname, sm in sr["smooth"].items():
            if fname == "none": continue
            for comp in ["pose", "lh", "rh"]:
                pld = sm[f"{comp}_metrics"].get("peak_lag_detail", {})
                if pld.get("per_landmark"):
                    peak_lag_details.setdefault(sid, {})[f"{fname}_{comp}"] = pld
    with open(OUT_DIR / "peak_lag_detail.json", "w", encoding="utf-8") as f: json.dump(peak_lag_details, f, indent=2, default=str)

    # === v1.5 PACKAGING ORDER ===
    sp("\n  7. Running assertions...")
    with open(OUT_DIR / "assertion_results.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": {"run_id": RUN_ID, "version": VERSION}, "summary": {}, "records": []}, f)
    run_manifest = {"step": "7B", "version": VERSION, "run_id": RUN_ID,
        "prior_status": ["STEP6B_COMPLETE", "STEP7A_V1_12_EXTERNAL_REVIEW_PASS", "STEP7B_READY"],
        "all_samples": all_sample_ids, "pilot_samples": pilot_ids,
        "dataset_split": {"train": 27, "val": 9, "test": 9},
        "smoothing_recommendation": SMOOTHING_CONCLUSION, "smoothing_default": SMOOTHING_DEFAULT,
        "savgol_note": SMOOTHING_SAVGOL_NOTE,
        "selected_normalization_transform": SELECTED_NORM_TRANSFORM, "selected_max_gap": SELECTED_MAX_GAP,
        "self_reference_excluded": True}
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f: json.dump(run_manifest, f, indent=2)

    ar, np_count, nf_count, cov_targets = run_assertions(all_results, split, all_sample_ids, pilot_ids, input_hashes, strong_depth,
                                             OUT_DIR, boundary_reasons_all, transform_detail, masking_agg, all_raw_errors)

    # v1.9: Clean raw arrays from per_landmark AFTER assertions used them, then overwrite detail JSON
    for td in transform_detail:
        for pl in td.get("per_landmark", []):
            pl.pop("_raw_devs_m", None); pl.pop("_raw_vel_vec_errs", None); pl.pop("_raw_accel_vec_errs", None)
    with open(OUT_DIR / "transform_comparison_detail.json", "w", encoding="utf-8") as f:
        json.dump(transform_detail, f, indent=2, default=str)

    # v1.12: Coverage registry — all 8 families use family+target_key metadata (no name_pattern fallback)
    import re as _re
    import copy as _copy
    REQUIRED_FAMILIES = {
        "dynamics_edge": cov_targets["dynamics_expected"],
        "boundary_reset": cov_targets["dynamics_expected"],
        "component_aggregate_field": cov_targets["comp_agg_field_targets"],
        "sample_aggregate_field": cov_targets["sample_agg_field_targets"],
        "json_csv_full_cell": cov_targets["json_csv_cell_targets"],
        "status_membership": cov_targets["status_membership_targets"],
        "canonical_alias": cov_targets["canonical_alias_targets"],
        "idle_jitter": cov_targets["idle_jitter_targets"],
    }
    REQUIRED_FAMILY_NAMES = set(REQUIRED_FAMILIES.keys())

    def _verify_coverage(assertion_records, required_families):
        """Production coverage verifier: returns {family: {expected_targets, actual_targets, missing, unexpected, status, counts}}."""
        result = {}
        for fam_name, expected_set in required_families.items():
            actual_set = set()
            for r in assertion_records:
                if r.get("family") == fam_name:
                    tk = r.get("target_key", "")
                    if tk:
                        actual_set.add(tk)
            missing = sorted(expected_set - actual_set)
            unexpected = sorted(actual_set - expected_set)
            cov_status = "PASS" if len(missing) == 0 and len(unexpected) == 0 else "FAIL"
            result[fam_name] = {
                "expected_targets": sorted(expected_set),
                "actual_targets": sorted(actual_set),
                "missing_targets": missing,
                "unexpected_targets": unexpected,
                "expected_count": len(expected_set),
                "actual_count": len(actual_set),
                "n_missing": len(missing),
                "n_unexpected": len(unexpected),
                "coverage_status": cov_status,
            }
        return result

    coverage_result = _verify_coverage(ar, REQUIRED_FAMILIES)
    for fam_name, fam_data in coverage_result.items():
        cov_status = fam_data["coverage_status"]
        if cov_status == "FAIL":
            ar.append({"name": f"coverage_fail_{fam_name}", "status": "FAIL",
                "detail": f"missing={fam_data['n_missing']} unexpected={fam_data['n_unexpected']}", "family": "coverage"})
            nf_count += 1
        else:
            ar.append({"name": f"coverage_pass_{fam_name}", "status": "PASS", "detail": "", "family": "coverage"})
            np_count += 1
    # Check required family set itself
    actual_fam_names = set(coverage_result.keys())
    fam_set_ok = actual_fam_names == REQUIRED_FAMILY_NAMES
    ar.append({"name": "coverage_required_family_set", "status": "PASS" if fam_set_ok else "FAIL",
        "detail": f"missing={REQUIRED_FAMILY_NAMES - actual_fam_names}" if not fam_set_ok else ""})
    if fam_set_ok: np_count += 1
    else: nf_count += 1

    # Duplicate assertion name and record check (AFTER coverage additions)
    all_ar_names = [r["name"] for r in ar]
    ar_unique_names = list(dict.fromkeys(all_ar_names))
    from collections import Counter as _Counter
    _name_counts = _Counter(all_ar_names)
    ar_dup_names = [n for n, c in _name_counts.items() if c > 1]
    if ar_dup_names:
        ar.append({"name": "zero_duplicate_assertion_names", "status": "FAIL", "detail": f"duplicates: {ar_dup_names[:10]}"})
        nf_count += 1
    else:
        ar.append({"name": "zero_duplicate_assertion_names", "status": "PASS", "detail": ""})
        np_count += 1
    ar_seen_records = set(); ar_dup_records = []; dup_found = False
    for r in ar:
        key = (r["name"], r["status"], r.get("detail", ""))
        if key in ar_seen_records: ar_dup_records.append(r["name"]); dup_found = True
        ar_seen_records.add(key)
    ar.append({"name": "zero_duplicate_assertion_records", "status": "PASS" if not dup_found else "FAIL", "detail": ""})
    if not dup_found: np_count += 1
    else: nf_count += 1

    # Build assertion container
    assertion_container = {
        "metadata": {"run_id": RUN_ID, "version": VERSION},
        "summary": {"PASS": np_count, "FAIL": nf_count, "total": np_count + nf_count,
                     "unique_names": len(set(r["name"] for r in ar)), "duplicate_names": ar_dup_names,
                     "duplicate_records": ar_dup_records},
        "coverage": coverage_result,
        "records": ar
    }
    with open(OUT_DIR / "assertion_results.json", "w", encoding="utf-8") as f:
        json.dump(assertion_container, f, indent=2)

    final_status = f"STEP7B_FULL_V{VERSION.replace('.','_')}_COMPLETE" if nf_count == 0 else f"STEP7B_V{VERSION.replace('.','_')}_BLOCKED"
    run_manifest["assertions"] = {"pass": np_count, "fail": nf_count}; run_manifest["status"] = final_status
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f: json.dump(run_manifest, f, indent=2)
    write_report(OUT_DIR, all_sample_ids, pilot_ids, pilot, seg_rows, masking_all, masking_agg, np_count, nf_count, ar_records=ar)

    sp("  8. Output manifest...")
    script_arc = f"s001_step7b_preprocess_full_v{VERSION.replace('.','_')}.py"
    output_manifest = []
    for root, _, fnames in os.walk(OUT_DIR):
        for fn in fnames:
            fp = Path(root) / fn; rel = fp.relative_to(OUT_DIR).as_posix()
            if rel == "output_file_manifest.json": continue
            meta = classify_artifact(rel)
            entry = {"relative_path": rel, "size_bytes": os.path.getsize(fp), "sha256": sha256_file(fp), "schema_version": VERSION_TAG}
            for k in ["artifact_type", "sample_id", "component", "filter", "max_gap"]:
                if k in meta and meta[k] != "unknown": entry[k] = meta[k]
            output_manifest.append(entry)
    output_manifest.append({"relative_path": script_arc, "size_bytes": os.path.getsize(__file__),
                             "sha256": sha256_file(__file__), "schema_version": VERSION_TAG, "artifact_type": "execution_script"})
    with open(OUT_DIR / "output_file_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"self_reference_excluded": True, "schema_version": VERSION_TAG, "run_id": RUN_ID,
                    "version": VERSION,
                    "outputs": output_manifest, "total_files": len(output_manifest)}, f, indent=2)

    sp("  9. ZIP...")
    zip_path = PROJECT / f"WordsInSilence_step7b_v{VERSION.replace('.','_')}_full_delivery.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for root, _, fnames in os.walk(OUT_DIR):
            for fn in fnames:
                fp = Path(root) / fn; arc = fp.relative_to(OUT_DIR).as_posix()
                comp = zipfile.ZIP_STORED if fp.suffix.lower() in (".png", ".npz") else zipfile.ZIP_DEFLATED
                zf.write(fp, arc, compress_type=comp)
        zf.write(Path(__file__), script_arc, compress_type=zipfile.ZIP_DEFLATED)

    zs = os.path.getsize(zip_path); zh = sha256_file(zip_path)
    sp(f"\n    ZIP: {zip_path.name} ({zs / 1e6:.1f} MB)")
    sp(f"    SHA-256: {zh}")

    # v1.6: detached .sha256 file
    sha_path = zip_path.with_suffix(".zip.sha256")
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(f"{zh}  {zip_path.name}\n")
    sp(f"    Detached hash: {sha_path.name}")

    # v1.12: Production verification functions (reusable by real and synthetic tests)

    def _verify_report_semantic(report_text, expected_run_id, expected_version, expected_settings):
        """Production report semantic verifier. Returns list of {name, status, detail, source}."""
        checks = []
        def _rchk(cond, name, detail=""):
            checks.append({"name": name, "status": "PASS" if cond else "FAIL", "detail": detail, "source": "report"})
        rid_match = _re.search(r'\*\*Run ID\*\*:\s*`([^`]+)`', report_text)
        _rchk(rid_match is not None and rid_match.group(1) == expected_run_id, "pp_report_run_id",
              f"found={rid_match.group(1) if rid_match else 'NONE'}")
        ver_match = _re.search(r'\*\*Version\*\*:\s*(\S+)', report_text)
        _rchk(ver_match is not None and ver_match.group(1) == expected_version, "pp_report_version",
              f"found={ver_match.group(1) if ver_match else 'NONE'}")
        norm_match = _re.search(r'Normalization transform\*\*:\s*`([^`]+)`', report_text)
        _rchk(norm_match is not None and norm_match.group(1) == expected_settings["normalization"], "pp_report_normalization")
        smooth_match = _re.search(r'\*\*Smoothing\*\*:\s*`([^`]+)`', report_text)
        _rchk(smooth_match is not None and smooth_match.group(1) == expected_settings["smoothing_default"], "pp_report_smoothing_default")
        concl_match = _re.search(r'Smoothing conclusion\*\*:\s*`([^`]+)`', report_text)
        _rchk(concl_match is not None and concl_match.group(1) == expected_settings["smoothing_conclusion"], "pp_report_smoothing_conclusion")
        savgol_match = _re.search(r'\*\*SavGol\*\*:\s*`([^`]+)`', report_text)
        _rchk(savgol_match is not None and savgol_match.group(1) == expected_settings["savgol"], "pp_report_savgol")
        gap_match = _re.search(r'\*\*Max gap\*\*:\s*`(\d+)`', report_text)
        _rchk(gap_match is not None and int(gap_match.group(1)) == expected_settings["max_gap"], "pp_report_max_gap")
        return checks

    def _verify_manifest_payload(manifest_data, payload_reader):
        """Production manifest payload verifier. payload_reader(rel_path) → bytes or None. Returns {size_fail, hash_fail, details}."""
        size_fail, hash_fail = 0, 0
        details = []
        for oe in manifest_data.get("outputs", []):
            rel_p = oe["relative_path"]
            payload = payload_reader(rel_p)
            if payload is None:
                size_fail += 1; hash_fail += 1
                details.append({"entry": rel_p, "error": "missing"})
                continue
            if len(payload) != oe["size_bytes"]:
                size_fail += 1
                details.append({"entry": rel_p, "error": "size_mismatch", "expected": oe["size_bytes"], "actual": len(payload)})
            actual_hash = hashlib.sha256(payload).hexdigest()
            if actual_hash != oe["sha256"]:
                hash_fail += 1
                details.append({"entry": rel_p, "error": "hash_mismatch", "expected_hash": oe["sha256"][:16], "actual_hash": actual_hash[:16]})
        return {"size_fail": size_fail, "hash_fail": hash_fail, "details": details}

    def _verify_full_cell(saved_detail_list, csv_row_list):
        """Production JSON↔CSV full-cell verifier. Returns {n_mismatch, mismatches: [{target_key, col, csv_val, exp_val}]}."""
        expected_rows = {}
        for td in saved_detail_list:
            sid_val = td["sample_id"]
            for pl in td.get("per_landmark", []):
                comp_lbl = pl["component"]; lm_lbl = pl["landmark"]
                rk = f"{sid_val}|landmark|{comp_lbl}|{lm_lbl}"
                row = {"sample_id": sid_val, "row_type": "landmark", "component": comp_lbl, "landmark": str(lm_lbl)}
                row["path_length_raw_m"] = pl.get("path_length_raw_m", 0)
                row["path_length_stab_m"] = pl.get("path_length_stab_m", 0)
                row["path_length_ratio"] = pl.get("path_length_ratio", 0)
                n_traj = pl.get("_n_traj", 0); n_vel = pl.get("_n_vel", 0); n_accel = pl.get("_n_accel", 0)
                row["n_trajectory_deviations"] = n_traj; row["trajectory_error_sum_m"] = pl.get("_traj_sum_m", 0)
                row["traj_dev_mean_mm"] = pl.get("traj_dev_mean_mm", 0); row["traj_dev_p95_mm"] = pl.get("traj_dev_p95_mm", 0)
                row["traj_dev_max_mm"] = pl.get("traj_dev_max_mm", 0)
                row["n_velocity_errors"] = n_vel; row["velocity_error_sum"] = pl.get("_vel_sum", 0)
                row["velocity_error_sum_squares"] = pl.get("_vel_sum_sq", 0); row["velocity_vector_rmse_m_per_s"] = pl.get("velocity_vector_rmse_m_per_s", 0)
                row["n_acceleration_errors"] = n_accel; row["acceleration_error_sum"] = pl.get("_accel_sum", 0)
                row["acceleration_error_sum_squares"] = pl.get("_accel_sum_sq", 0); row["acceleration_vector_rmse_m_per_s2"] = pl.get("acceleration_vector_rmse_m_per_s2", 0)
                t = pl.get("timing", {})
                row["timing_status"] = t.get("status", "missing")
                row["timing_n_nonneg_matched"] = t.get("n_nonneg_matched", 0)
                row["n_nonneg_matched"] = t.get("n_nonneg_matched", 0)
                row["n_negative_inconsistent"] = t.get("n_negative_inconsistent", 0)
                row["n_unmatched_raw"] = t.get("n_unmatched_raw", 0)
                row["n_unmatched_filt"] = t.get("n_unmatched_filt", 0)
                row["matched_peak_lag_frames"] = t.get("mean_lag_frames", "no_valid_nonnegative_events")
                row["matched_peak_lag_ms"] = t.get("mean_lag_ms", "no_valid_nonnegative_events")
                SENT = "no_valid_nonnegative_events"
                all_onset_ms, all_onset_fr, all_end_ms, all_end_fr, all_xcorr = [], [], [], [], []
                onset_nn, onset_neg, end_nn, end_neg, xcorr_nn, xcorr_neg = 0, 0, 0, 0, 0, 0
                for seg_t in t.get("segments", []):
                    ol = seg_t.get("onset_lag", {}); ol_s = ol.get("status", "")
                    if ol_s == "nonnegative":
                        onset_nn += 1
                        if "lag_ms" in ol: all_onset_ms.append(ol["lag_ms"])
                        if "lag_frames" in ol: all_onset_fr.append(ol["lag_frames"])
                    elif ol_s.startswith("negative"): onset_neg += 1
                    el = seg_t.get("end_lag", {}); el_s = el.get("status", "")
                    if el_s == "nonnegative":
                        end_nn += 1
                        if "lag_ms" in el: all_end_ms.append(el["lag_ms"])
                        if "lag_frames" in el: all_end_fr.append(el["lag_frames"])
                    elif el_s.startswith("negative"): end_neg += 1
                    xl = seg_t.get("xcorr_lag", {}); xl_s = xl.get("status", "")
                    if xl_s == "nonnegative":
                        xcorr_nn += 1; all_xcorr.append(xl.get("lag_frames", 0))
                    elif xl_s.startswith("negative"): xcorr_neg += 1
                row["onset_nonneg"] = onset_nn; row["onset_negative"] = onset_neg
                row["onset_status"] = "has_nonnegative" if onset_nn > 0 else SENT
                row["onset_lag_ms_pooled"] = float(np.mean(all_onset_ms)) if all_onset_ms else SENT
                row["onset_lag_frames_pooled"] = float(np.mean(all_onset_fr)) if all_onset_fr else SENT
                row["end_nonneg"] = end_nn; row["end_negative"] = end_neg
                row["end_status"] = "has_nonnegative" if end_nn > 0 else SENT
                row["end_lag_ms_pooled"] = float(np.mean(all_end_ms)) if all_end_ms else SENT
                row["end_lag_frames_pooled"] = float(np.mean(all_end_fr)) if all_end_fr else SENT
                row["xcorr_nonneg"] = xcorr_nn; row["xcorr_negative"] = xcorr_neg
                row["xcorr_status"] = "has_nonnegative" if xcorr_nn > 0 else SENT
                row["xcorr_lag_frames_pooled"] = float(np.mean(all_xcorr)) if all_xcorr else SENT
                row["path_raw_sum_m"] = row["path_length_raw_m"]; row["path_stab_sum_m"] = row["path_length_stab_m"]
                row["traj_n"] = row["n_trajectory_deviations"]; row["traj_sum_m"] = row["trajectory_error_sum_m"]
                row["vel_n"] = row["n_velocity_errors"]; row["vel_sum"] = row["velocity_error_sum"]
                row["vel_sum_sq"] = row["velocity_error_sum_squares"]
                row["accel_n"] = row["n_acceleration_errors"]; row["accel_sum"] = row["acceleration_error_sum"]
                row["accel_sum_sq"] = row["acceleration_error_sum_squares"]
                row["idle_jitter_start_raw_mm"] = "not_applicable_landmark"
                row["idle_jitter_start_stab_mm"] = "not_applicable_landmark"
                row["idle_jitter_end_raw_mm"] = "not_applicable_landmark"
                row["idle_jitter_end_stab_mm"] = "not_applicable_landmark"
                expected_rows[rk] = row
            for comp_name_csv, ca in td.get("component_aggregates", {}).items():
                if not ca: continue
                rk = f"{sid_val}|component_aggregate|{comp_name_csv}|not_applicable_aggregate"
                ct = ca.get("timing", {}); SENT = "no_valid_nonnegative_events"
                row = {"sample_id": sid_val, "row_type": "component_aggregate", "component": comp_name_csv, "landmark": "not_applicable_aggregate",
                       "path_length_raw_m": ca.get("path_raw_sum_m", 0), "path_length_stab_m": ca.get("path_stab_sum_m", 0),
                       "path_length_ratio": ca.get("path_ratio", 0),
                       "n_trajectory_deviations": ca.get("traj_n", 0), "trajectory_error_sum_m": ca.get("traj_sum_m", 0),
                       "traj_dev_mean_mm": ca.get("traj_mean_mm", 0), "traj_dev_p95_mm": ca.get("traj_p95_mm", 0), "traj_dev_max_mm": ca.get("traj_max_mm", 0),
                       "n_velocity_errors": ca.get("vel_n", 0), "velocity_error_sum": ca.get("vel_sum", 0),
                       "velocity_error_sum_squares": ca.get("vel_sum_sq", 0), "velocity_vector_rmse_m_per_s": ca.get("vel_rmse", 0),
                       "n_acceleration_errors": ca.get("accel_n", 0), "acceleration_error_sum": ca.get("accel_sum", 0),
                       "acceleration_error_sum_squares": ca.get("accel_sum_sq", 0), "acceleration_vector_rmse_m_per_s2": ca.get("accel_rmse", 0),
                       "timing_status": ct.get("onset_status", SENT), "timing_n_nonneg_matched": ct.get("peak_nonneg", 0),
                       "n_nonneg_matched": ct.get("peak_nonneg", 0), "n_negative_inconsistent": ct.get("peak_negative", 0),
                       "n_unmatched_raw": ct.get("peak_unmatched_raw", 0), "n_unmatched_filt": ct.get("peak_unmatched_filt", 0),
                       "matched_peak_lag_frames": ct.get("peak_pooled_mean_frames", SENT), "matched_peak_lag_ms": ct.get("peak_pooled_mean_ms", SENT),
                       "onset_nonneg": ct.get("onset_nonneg", 0), "onset_negative": ct.get("onset_negative", 0),
                       "onset_status": ct.get("onset_status", SENT), "onset_lag_ms_pooled": ct.get("onset_pooled_mean_ms", SENT),
                       "onset_lag_frames_pooled": ct.get("onset_pooled_mean_frames", SENT),
                       "end_nonneg": ct.get("end_nonneg", 0), "end_negative": ct.get("end_negative", 0),
                       "end_status": ct.get("end_status", SENT), "end_lag_ms_pooled": ct.get("end_pooled_mean_ms", SENT),
                       "end_lag_frames_pooled": ct.get("end_pooled_mean_frames", SENT),
                       "xcorr_nonneg": ct.get("xcorr_nonneg", 0), "xcorr_negative": ct.get("xcorr_negative", 0),
                       "xcorr_status": ct.get("xcorr_status", SENT), "xcorr_lag_frames_pooled": ct.get("xcorr_pooled_mean_frames", SENT),
                       "path_raw_sum_m": ca.get("path_raw_sum_m", 0), "path_stab_sum_m": ca.get("path_stab_sum_m", 0),
                       "traj_n": ca.get("traj_n", 0), "traj_sum_m": ca.get("traj_sum_m", 0),
                       "vel_n": ca.get("vel_n", 0), "vel_sum": ca.get("vel_sum", 0), "vel_sum_sq": ca.get("vel_sum_sq", 0),
                       "accel_n": ca.get("accel_n", 0), "accel_sum": ca.get("accel_sum", 0), "accel_sum_sq": ca.get("accel_sum_sq", 0),
                       "idle_jitter_start_raw_mm": "not_applicable_component_aggregate",
                       "idle_jitter_start_stab_mm": "not_applicable_component_aggregate",
                       "idle_jitter_end_raw_mm": "not_applicable_component_aggregate",
                       "idle_jitter_end_stab_mm": "not_applicable_component_aggregate"}
                expected_rows[rk] = row
            agg_td = td["aggregate"]; ij = td.get("idle_jitter", {}); st = agg_td.get("timing", {}); SENT = "no_valid_nonnegative_events"
            rk = f"{sid_val}|sample_aggregate|ALL|not_applicable_aggregate"
            row = {"sample_id": sid_val, "row_type": "sample_aggregate", "component": "ALL", "landmark": "not_applicable_aggregate",
                   "path_length_raw_m": agg_td.get("path_raw_sum_m", 0), "path_length_stab_m": agg_td.get("path_stab_sum_m", 0),
                   "path_length_ratio": agg_td.get("path_ratio", 0),
                   "n_trajectory_deviations": agg_td.get("traj_n", 0), "trajectory_error_sum_m": agg_td.get("traj_sum_m", 0),
                   "traj_dev_mean_mm": agg_td["traj_dev_mean_mm"], "traj_dev_p95_mm": agg_td.get("traj_dev_p95_mm", 0),
                   "traj_dev_max_mm": agg_td["traj_dev_max_mm"],
                   "n_velocity_errors": agg_td.get("vel_n", 0), "velocity_error_sum": agg_td.get("vel_sum", 0),
                   "velocity_error_sum_squares": agg_td.get("vel_sum_sq", 0), "velocity_vector_rmse_m_per_s": agg_td.get("velocity_vector_rmse_m_per_s", 0),
                   "n_acceleration_errors": agg_td.get("accel_n", 0), "acceleration_error_sum": agg_td.get("accel_sum", 0),
                   "acceleration_error_sum_squares": agg_td.get("accel_sum_sq", 0), "acceleration_vector_rmse_m_per_s2": agg_td.get("acceleration_vector_rmse_m_per_s2", 0),
                   "timing_status": st.get("onset_status", SENT), "timing_n_nonneg_matched": st.get("peak_nonneg", 0),
                   "n_nonneg_matched": st.get("peak_nonneg", 0), "n_negative_inconsistent": st.get("peak_negative", 0),
                   "n_unmatched_raw": st.get("peak_unmatched_raw", 0), "n_unmatched_filt": st.get("peak_unmatched_filt", 0),
                   "matched_peak_lag_frames": st.get("peak_pooled_mean_frames", SENT), "matched_peak_lag_ms": st.get("peak_pooled_mean_ms", SENT),
                   "onset_nonneg": st.get("onset_nonneg", 0), "onset_negative": st.get("onset_negative", 0),
                   "onset_status": st.get("onset_status", SENT), "onset_lag_ms_pooled": st.get("onset_pooled_mean_ms", SENT),
                   "onset_lag_frames_pooled": st.get("onset_pooled_mean_frames", SENT),
                   "end_nonneg": st.get("end_nonneg", 0), "end_negative": st.get("end_negative", 0),
                   "end_status": st.get("end_status", SENT), "end_lag_ms_pooled": st.get("end_pooled_mean_ms", SENT),
                   "end_lag_frames_pooled": st.get("end_pooled_mean_frames", SENT),
                   "xcorr_nonneg": st.get("xcorr_nonneg", 0), "xcorr_negative": st.get("xcorr_negative", 0),
                   "xcorr_status": st.get("xcorr_status", SENT), "xcorr_lag_frames_pooled": st.get("xcorr_pooled_mean_frames", SENT),
                   "path_raw_sum_m": agg_td.get("path_raw_sum_m", 0), "path_stab_sum_m": agg_td.get("path_stab_sum_m", 0),
                   "traj_n": agg_td.get("traj_n", 0), "traj_sum_m": agg_td.get("traj_sum_m", 0),
                   "vel_n": agg_td.get("vel_n", 0), "vel_sum": agg_td.get("vel_sum", 0), "vel_sum_sq": agg_td.get("vel_sum_sq", 0),
                   "accel_n": agg_td.get("accel_n", 0), "accel_sum": agg_td.get("accel_sum", 0), "accel_sum_sq": agg_td.get("accel_sum_sq", 0),
                   "idle_jitter_start_raw_mm": ij.get("start_raw_mm", "not_available_no_verified_idle"),
                   "idle_jitter_start_stab_mm": ij.get("start_stabilized_mm", "not_available_no_verified_idle"),
                   "idle_jitter_end_raw_mm": ij.get("end_raw_mm", "not_available_no_verified_idle"),
                   "idle_jitter_end_stab_mm": ij.get("end_stabilized_mm", "not_available_no_verified_idle")}
            expected_rows[rk] = row
        csv_lookup = {}
        for cr in csv_row_list:
            sid_c = cr.get("sample_id", ""); rt_c = cr.get("row_type", ""); comp_c = cr.get("component", ""); lm_c = cr.get("landmark", "")
            crk = f"{sid_c}|{rt_c}|{comp_c}|{lm_c}"
            csv_lookup[crk] = cr
        mismatches = []
        for rk in sorted(set(expected_rows.keys()) & set(csv_lookup.keys())):
            exp_row = expected_rows[rk]; csv_row_c = csv_lookup[rk]
            for col in sorted(csv_row_c.keys()):
                cell_key = f"{rk}|{col}"
                csv_val = csv_row_c[col]; exp_val = exp_row.get(col)
                if exp_val is None:
                    ok = csv_val != ""
                else:
                    ok = _cmpv_csv(csv_val, exp_val)
                if not ok:
                    mismatches.append({"target_key": cell_key, "col": col, "csv_val": csv_val, "exp_val": str(exp_val)})
        return {"n_mismatch": len(mismatches), "mismatches": mismatches}

    # v1.12: ZIP payload-based post-package verification
    post_pkg_checks = []
    def ppchk(cond, name, detail="", source=""):
        post_pkg_checks.append({"name": name, "status": "PASS" if cond else "FAIL", "detail": detail, "source": source})
        if not cond: sp(f"    [POST-PKG FAIL] {name}: {detail}")
    FIXED_SETTINGS = {
        "normalization": ("selected_normalization_transform", SELECTED_NORM_TRANSFORM),
        "smoothing_default": ("smoothing_default", SMOOTHING_DEFAULT),
        "smoothing_conclusion": ("smoothing_recommendation", SMOOTHING_CONCLUSION),
        "savgol": ("savgol_note", SMOOTHING_SAVGOL_NOTE),
        "max_gap": ("selected_max_gap", SELECTED_MAX_GAP),
    }
    EXPECTED_REPORT_SETTINGS = {
        "normalization": SELECTED_NORM_TRANSFORM,
        "smoothing_default": SMOOTHING_DEFAULT,
        "smoothing_conclusion": SMOOTHING_CONCLUSION,
        "savgol": SMOOTHING_SAVGOL_NOTE,
        "max_gap": SELECTED_MAX_GAP,
    }
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_info_list = zf.infolist()
        zip_entry_count = len(zip_info_list)
        zip_entry_names = [zi.filename for zi in zip_info_list]
        zip_dup_entries = [n for n in set(zip_entry_names) if zip_entry_names.count(n) > 1]
        zip_unsafe_paths = [n for n in zip_entry_names if n.startswith("/") or ".." in n]
        zip_symlinks = [zi.filename for zi in zip_info_list if (zi.external_attr >> 16) & 0o120000 == 0o120000]
        zip_encrypted = [zi.filename for zi in zip_info_list if zi.flag_bits & 0x1]
        bad_crc = zf.testzip()
        ppchk(bad_crc is None, "zip_crc", f"bad={bad_crc}" if bad_crc else "", "zip")
        ppchk(len(zip_dup_entries) == 0, "zip_no_duplicates", str(zip_dup_entries) if zip_dup_entries else "", "zip")
        ppchk(len(zip_unsafe_paths) == 0, "zip_no_unsafe_paths", source="zip")
        ppchk(len(zip_symlinks) == 0, "zip_no_symlinks", source="zip")
        ppchk(len(zip_encrypted) == 0, "zip_no_encrypted", source="zip")

        def _zip_json(name):
            try: return json.loads(zf.read(name))
            except (KeyError, json.JSONDecodeError): return None

        config_name_z = f"step7b_v{VERSION.replace('.','_')}_config.json"
        z_cfg = _zip_json(config_name_z)
        if z_cfg:
            ppchk(z_cfg.get("run_id") == RUN_ID, "pp_config_run_id", source=config_name_z)
            ppchk(z_cfg.get("version") == VERSION, "pp_config_version", source=config_name_z)
            for sname, (skey, sval) in FIXED_SETTINGS.items():
                ppchk(z_cfg.get(skey) == sval, f"pp_config_{sname}", f"expected={sval} actual={z_cfg.get(skey)}", source=config_name_z)
        else:
            ppchk(False, "pp_config_exists", source=config_name_z)

        z_rm = _zip_json("run_manifest.json")
        if z_rm:
            ppchk(z_rm.get("run_id") == RUN_ID, "pp_run_manifest_run_id", source="run_manifest.json")
            ppchk(z_rm.get("version") == VERSION, "pp_run_manifest_version", source="run_manifest.json")
            for sname, (skey, sval) in FIXED_SETTINGS.items():
                ppchk(z_rm.get(skey) == sval, f"pp_run_manifest_{sname}", f"expected={sval} actual={z_rm.get(skey)}", source="run_manifest.json")
        else:
            ppchk(False, "pp_run_manifest_exists", source="run_manifest.json")

        report_name_z = f"STEP7B_V{VERSION.replace('.','_')}_REPORT.md"
        try:
            z_rpt = zf.read(report_name_z).decode("utf-8")
            rpt_checks = _verify_report_semantic(z_rpt, RUN_ID, VERSION, EXPECTED_REPORT_SETTINGS)
            for rc in rpt_checks:
                ppchk(rc["status"] == "PASS", rc["name"], rc["detail"], source=report_name_z)
        except KeyError:
            ppchk(False, "pp_report_exists", source=report_name_z)

        z_ar = _zip_json("assertion_results.json")
        if z_ar:
            ppchk(z_ar.get("metadata", {}).get("run_id") == RUN_ID, "pp_assertion_metadata_run_id", source="assertion_results.json")
            ppchk(z_ar.get("metadata", {}).get("version") == VERSION, "pp_assertion_metadata_version", source="assertion_results.json")
            z_sum = z_ar.get("summary", {})
            ppchk(z_sum.get("PASS") == np_count, "pp_assertion_pass_count", f"expected={np_count} actual={z_sum.get('PASS')}", source="assertion_results.json")
            ppchk(z_sum.get("FAIL") == nf_count, "pp_assertion_fail_count", f"expected={nf_count} actual={z_sum.get('FAIL')}", source="assertion_results.json")
            ppchk(z_sum.get("total") == np_count + nf_count, "pp_assertion_total", source="assertion_results.json")
            ppchk(len(z_sum.get("duplicate_names", [])) == 0, "pp_assertion_no_dup_names", source="assertion_results.json")
            ppchk(len(z_sum.get("duplicate_records", [])) == 0, "pp_assertion_no_dup_records", source="assertion_results.json")
            z_cov = z_ar.get("coverage", {})
            ppchk(set(z_cov.keys()) == REQUIRED_FAMILY_NAMES, "pp_coverage_family_set",
                  f"actual={set(z_cov.keys())}", source="assertion_results.json")
            for fn_pp, fd_pp in z_cov.items():
                ppchk(fd_pp.get("n_missing", -1) == 0 and fd_pp.get("n_unexpected", -1) == 0,
                      f"pp_coverage_{fn_pp}", f"miss={fd_pp.get('n_missing')} unexp={fd_pp.get('n_unexpected')}", source="assertion_results.json")
        else:
            ppchk(False, "pp_assertion_results_exists", source="assertion_results.json")

        z_om = _zip_json("output_file_manifest.json")
        if z_om:
            ppchk(z_om.get("run_id") == RUN_ID, "pp_output_manifest_run_id", source="output_file_manifest.json")
            ppchk(z_om.get("version") == VERSION, "pp_output_manifest_version",
                  f"expected={VERSION} actual={z_om.get('version')}", source="output_file_manifest.json")
            ppchk(z_om.get("self_reference_excluded") is True, "pp_output_manifest_self_ref", source="output_file_manifest.json")
            om_entries = set(e["relative_path"] for e in z_om.get("outputs", []))
            zip_content_entries = set(zip_entry_names) - {"output_file_manifest.json"}
            ppchk(om_entries == zip_content_entries, "pp_zip_manifest_match",
                  f"only_manifest={sorted(om_entries - zip_content_entries)[:5]} only_zip={sorted(zip_content_entries - om_entries)[:5]}", source="output_file_manifest.json")
            def _zip_payload_reader(rel_p):
                try: return zf.read(rel_p)
                except KeyError: return None
            mv_result = _verify_manifest_payload(z_om, _zip_payload_reader)
            z_size_fail = mv_result["size_fail"]; z_hash_fail = mv_result["hash_fail"]
            ppchk(z_size_fail == 0, "pp_zip_payload_sizes", f"mismatches={z_size_fail}", source="output_file_manifest.json")
            ppchk(z_hash_fail == 0, "pp_zip_payload_hashes", f"mismatches={z_hash_fail}", source="output_file_manifest.json")
        else:
            ppchk(False, "pp_output_manifest_exists", source="output_file_manifest.json")
            z_size_fail = -1; z_hash_fail = -1

    # Parse detached .sha256 file
    sha_content = sha_path.read_text(encoding="utf-8").strip()
    sha_parts = sha_content.split("  ", 1)
    sha_parse_ok = len(sha_parts) == 2
    ppchk(sha_parse_ok, "detached_sha256_format", f"parts={len(sha_parts)}", source=sha_path.name)
    if sha_parse_ok:
        recorded_hash, recorded_fn = sha_parts[0], sha_parts[1]
        recomputed_zh = sha256_file(zip_path)
        ppchk(recorded_hash == recomputed_zh, "detached_sha256_hash_match",
              f"recorded={recorded_hash[:16]}... recomputed={recomputed_zh[:16]}...", source=sha_path.name)
        ppchk(recorded_fn == zip_path.name, "detached_sha256_filename_match",
              f"recorded={recorded_fn} expected={zip_path.name}", source=sha_path.name)

    # v1.12: 5 synthetic negative regression tests with actual mutation + production verifier
    sp("  11. Synthetic negative regression tests...")
    def _zip_json_standalone(zf_handle, name):
        try: return json.loads(zf_handle.read(name))
        except (KeyError, json.JSONDecodeError): return None

    # Test A: Aggregate assertion removal → coverage FAIL
    ar_copy_a = _copy.deepcopy(ar)
    removed_rec = None
    for i, r in enumerate(ar_copy_a):
        if r.get("family") == "component_aggregate_field":
            removed_rec = ar_copy_a.pop(i); break
    if removed_rec:
        synth_cov = _verify_coverage(ar_copy_a, REQUIRED_FAMILIES)
        caf_result = synth_cov["component_aggregate_field"]
        removed_tk = removed_rec.get("target_key", "")
        detected = caf_result["coverage_status"] == "FAIL" and removed_tk in caf_result["missing_targets"]
        ppchk(detected, "synth_agg_field_removal_detected",
              f"removed_target={removed_tk} verifier=_verify_coverage missing_targets_has_it={removed_tk in caf_result['missing_targets']} family_status={caf_result['coverage_status']} expected_failure=FAIL",
              source="synthetic_regression")
    else:
        ppchk(False, "synth_agg_field_removal_detected", "no component_aggregate_field record found", source="synthetic_regression")

    # Test B: CSV cell mutation → full-cell FAIL
    tc_detail_path_s = OUT_DIR / "transform_comparison_detail.json"
    tc_csv_path_s = OUT_DIR / "transform_comparison.csv"
    if tc_detail_path_s.exists() and tc_csv_path_s.exists():
        with open(tc_detail_path_s, encoding="utf-8") as f: saved_detail_s = json.load(f)
        with open(tc_csv_path_s, encoding="utf-8") as f: csv_rows_s = list(csv.DictReader(f))
        # Mutate one CSV cell
        mutated_csv = _copy.deepcopy(csv_rows_s)
        mutated_row = mutated_csv[0]
        mutate_col = "traj_dev_mean_mm"
        original_val = mutated_row[mutate_col]
        mutated_row[mutate_col] = "999999.99"
        # Identify target key for the mutated row
        mut_rk = f"{mutated_row['sample_id']}|{mutated_row['row_type']}|{mutated_row['component']}|{mutated_row['landmark']}|{mutate_col}"
        fc_result = _verify_full_cell(saved_detail_s, mutated_csv)
        cell_detected = fc_result["n_mismatch"] > 0 and any(m["col"] == mutate_col for m in fc_result["mismatches"])
        ppchk(cell_detected, "synth_csv_cell_mutation_detected",
              f"mutated_target={mut_rk} original={original_val} mutated=999999.99 verifier=_verify_full_cell n_mismatch={fc_result['n_mismatch']} expected_failure=FAIL",
              source="synthetic_regression")
    else:
        ppchk(False, "synth_csv_cell_mutation_detected", "saved files not found", source="synthetic_regression")

    # Test C: Report Version changed → semantic FAIL
    with zipfile.ZipFile(zip_path, "r") as zf:
        z_rpt_c = zf.read(report_name_z).decode("utf-8")
    rpt_mutated_c = z_rpt_c.replace(f"**Version**: {VERSION}", "**Version**: 99.99")
    rpt_checks_c = _verify_report_semantic(rpt_mutated_c, RUN_ID, VERSION, EXPECTED_REPORT_SETTINGS)
    ver_check_c = [c for c in rpt_checks_c if c["name"] == "pp_report_version"]
    ver_failed_c = len(ver_check_c) > 0 and ver_check_c[0]["status"] == "FAIL"
    ppchk(ver_failed_c, "synth_report_version_changed_detected",
          f"mutated_version=99.99 verifier=_verify_report_semantic version_check_status={ver_check_c[0]['status'] if ver_check_c else 'MISSING'} found={ver_check_c[0]['detail'] if ver_check_c else 'N/A'} expected_failure=FAIL",
          source="synthetic_regression")

    # Test D: Report Version line removed → semantic FAIL
    rpt_lines = z_rpt_c.split("\n")
    rpt_mutated_d = "\n".join(line for line in rpt_lines if not line.startswith("**Version**:"))
    rpt_checks_d = _verify_report_semantic(rpt_mutated_d, RUN_ID, VERSION, EXPECTED_REPORT_SETTINGS)
    ver_check_d = [c for c in rpt_checks_d if c["name"] == "pp_report_version"]
    ver_failed_d = len(ver_check_d) > 0 and ver_check_d[0]["status"] == "FAIL"
    ppchk(ver_failed_d, "synth_report_version_missing_detected",
          f"mutation=removed_version_line verifier=_verify_report_semantic version_check_status={ver_check_d[0]['status'] if ver_check_d else 'MISSING'} found={ver_check_d[0]['detail'] if ver_check_d else 'N/A'} expected_failure=FAIL",
          source="synthetic_regression")

    # Test E: ZIP payload byte mutation → manifest hash FAIL
    with zipfile.ZipFile(zip_path, "r") as zf:
        z_om_e = _zip_json_standalone(zf, "output_file_manifest.json")
        if z_om_e:
            # Pick first entry that has content
            target_entry = z_om_e["outputs"][0]["relative_path"]
            original_payload = zf.read(target_entry)
            mutated_payload = bytearray(original_payload)
            mutated_payload[0] = (mutated_payload[0] + 1) % 256
            mutated_payload = bytes(mutated_payload)
            payload_cache = {target_entry: mutated_payload}
            def _mutated_reader(rel_p):
                if rel_p in payload_cache: return payload_cache[rel_p]
                try: return zf.read(rel_p)
                except KeyError: return None
            mv_result_e = _verify_manifest_payload(z_om_e, _mutated_reader)
            hash_detected = mv_result_e["hash_fail"] > 0 and any(d.get("entry") == target_entry and d.get("error") == "hash_mismatch" for d in mv_result_e["details"])
            ppchk(hash_detected, "synth_zip_payload_mutation_detected",
                  f"mutated_entry={target_entry} verifier=_verify_manifest_payload hash_fail={mv_result_e['hash_fail']} detected_entry={target_entry} expected_failure=FAIL",
                  source="synthetic_regression")
        else:
            ppchk(False, "synth_zip_payload_mutation_detected", "output manifest not found", source="synthetic_regression")

    # v1.12: Sidecar check uniqueness verification
    all_check_names = [c["name"] for c in post_pkg_checks]
    unique_check_names = list(dict.fromkeys(all_check_names))
    dup_check_names = [n for n in set(all_check_names) if all_check_names.count(n) > 1]
    check_seen = set(); dup_check_records = []
    for c in post_pkg_checks:
        ck = (c["name"], c["status"], c.get("detail", ""))
        if ck in check_seen: dup_check_records.append(c["name"])
        check_seen.add(ck)
    check_uniqueness_ok = len(dup_check_names) == 0 and len(dup_check_records) == 0

    pp_pass = sum(1 for c in post_pkg_checks if c["status"] == "PASS")
    pp_fail = sum(1 for c in post_pkg_checks if c["status"] == "FAIL")
    if not check_uniqueness_ok:
        pp_fail += 1  # count uniqueness failure
    verify_ok = pp_fail == 0 and check_uniqueness_ok
    sidecar = {
        "run_id": RUN_ID, "version": VERSION,
        "zip_file": zip_path.name, "zip_size_bytes": zs, "zip_sha256": zh,
        "zip_crc_result": "PASS" if bad_crc is None else f"FAIL_{bad_crc}",
        "entry_count": zip_entry_count,
        "duplicate_entries": zip_dup_entries,
        "unsafe_paths": zip_unsafe_paths,
        "symlinks": zip_symlinks,
        "encrypted_entries": zip_encrypted,
        "size_mismatches": z_size_fail if z_om else -1,
        "hash_mismatches": z_hash_fail if z_om else -1,
        "in_package_assertion_pass": np_count, "in_package_assertion_fail": nf_count,
        "post_package_checks": post_pkg_checks,
        "post_package_pass": pp_pass, "post_package_fail": pp_fail,
        "check_uniqueness": {
            "total_checks": len(post_pkg_checks),
            "unique_names": len(unique_check_names),
            "duplicate_names": dup_check_names,
            "unique_records": len(check_seen),
            "duplicate_records": dup_check_records,
        },
        "verification_status": "PASS" if verify_ok else "FAIL",
        "self_reference_excluded": True
    }
    sidecar_path = zip_path.with_name(zip_path.stem + "_post_zip_verification.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    sp(f"    Sidecar: {sidecar_path.name}")

    sp(f"\n  ============================================================")
    sp(f"  STEP 7B v{VERSION} SUMMARY")
    sp(f"  ============================================================")
    sp(f"  Run ID: {RUN_ID}")
    sp(f"  Samples: {len(all_sample_ids)} (pilot={len(pilot_ids)})")
    sp(f"  Assertions: {np_count} PASS / {nf_count} FAIL")
    sp(f"  Post-ZIP verification: {'PASS' if verify_ok else 'FAIL'}")
    sp(f"  Output: {OUT_DIR}")
    if nf_count == 0 and verify_ok:
        sp(f"  Prior: STEP6B_COMPLETE, STEP7A_V1_12_EXTERNAL_REVIEW_PASS")
        sp(f"  Status: STEP7B_COMPLETE_BLOCKED_PENDING_EXTERNAL_REVIEW")
    else:
        sp(f"  Status: STEP7B_V{VERSION.replace('.','_')}_BLOCKED")

if __name__ == "__main__":
    main()
