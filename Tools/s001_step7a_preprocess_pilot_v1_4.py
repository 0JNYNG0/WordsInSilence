#!/usr/bin/env python3
"""
Step 7A v1.4: Recognition preprocessing pilot - packaging & evaluation fixes per v1.3 partial-pass review.

Fixes from v1.3 review:
  - 2: Packaging order: assertions before ZIP, finalized files in ZIP
  - 3: Assertion regression checks strengthened
  - 4: SavGol first_after_reset overwrite (output=input at reset points)
  - 5: Smoothing delay evaluation: per-segment per-landmark peak matching
  - 6: Transform comparison: full CSV/JSON, selected_normalization_transform=raw_framewise
  - 7: Masking dynamics: magnitude vs vector error separation
  - 8: Output manifest metadata (artifact_type, sample_id, etc.)
  - 9: Final defaults: raw_framewise, no_smoothing, max_gap=1

Input: Step 6B v1.2.4 full 45 (frozen read-only)
Output: data/processed/sessions/S001_P001/step7a_pilot_v1_4/
"""

import csv
import hashlib
import json
import math
import os
import sys
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
OUT_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7a_pilot_v1_4"

VERSION = "1.4"
VERSION_TAG = f"7a_v{VERSION}"

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

# Reset reason codes
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

# NPZ expected key sets
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


def sp(s):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode())


# ================================================================
#  Dataset Split / Pilot Selection (unchanged from v1.3)
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
    frames = data["frames"]
    N = len(frames)
    total_lm = N * 75
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
    for sid in train_ids:
        by_word[split[sid]["word"]].append(sid)
    pilot = {}
    for word in WORDS:
        qs = sorted([compute_sample_quality(s) for s in by_word[word]],
                     key=lambda x: x["quality_score"], reverse=True)
        for tier, idx in [("high", 0), ("mid", len(qs) // 2), ("low", -1)]:
            q = qs[idx]
            pilot[q["sample_id"]] = {"tier": tier, "word": word, **q}
    return pilot


# ================================================================
#  Load Step 6B data (unchanged)
# ================================================================

def load_sample(sid):
    short = sid.replace("S001_P001_", "")
    with open(RAW3D_DIR / f"landmarks_3d_{short}.json", encoding="utf-8") as f:
        return json.load(f)


def extract_arrays(data):
    frames = data["frames"]
    N = len(frames)
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
    boundaries = defaultdict(set)
    records = []
    cat_to_comp = {"pose": "pose", "left_hand": "lh", "right_hand": "rh"}
    with open(TA_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["sample_id"] not in pilot_ids:
                continue
            if row.get("strong_depth_condition") != "True":
                continue
            sid = row["sample_id"]
            fi = int(row["frame_index"])
            li = int(row["landmark_index"])
            comp = cat_to_comp.get(row["category"], row["category"])
            boundaries[sid].add((fi, comp, li))
            records.append({
                "sample_id": sid, "frame_index": fi, "component": comp,
                "landmark_index": li, "category": row["category"],
                "strong_depth_condition": True,
                "qa_invalidated_by_depth": row.get("qa_invalidated_by_depth", ""),
                "qa_already_invalid_before_depth": row.get("qa_already_invalid_before_depth", ""),
            })
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
    if not (ls_v and rs_v):
        return None, None, None, "degenerate_shoulders"
    origin = (ls + rs) / 2.0
    right_vec = rs - ls
    sw = np.linalg.norm(right_vec)
    if sw < 1e-6:
        return origin, None, 0.0, "degenerate_shoulder_width"
    right = right_vec / sw
    if lh_v and rh_v: up_vec = origin - (lh + rh) / 2.0
    elif lh_v: up_vec = origin - lh
    elif rh_v: up_vec = origin - rh
    else: up_vec = np.array([0.0, -1.0, 0.0])
    up_n = np.linalg.norm(up_vec)
    if up_n < 1e-6: up_vec = np.array([0.0, -1.0, 0.0]); up_n = 1.0
    up = up_vec / up_n
    fwd = np.cross(right, up)
    fn = np.linalg.norm(fwd)
    if fn < 1e-6:
        return origin, None, sw, "degenerate_axes"
    fwd /= fn
    up = np.cross(fwd, right); up /= np.linalg.norm(up)
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
        if R is not None:
            rotations[fi] = R; dets[fi] = np.linalg.det(R)
            ortho_errs[fi] = np.linalg.norm(R.T @ R - np.eye(3))
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
            "dets": dets, "ortho_errs": ortho_errs,
            "statuses": statuses, "fallback_src": fallback_src}


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


# ================================================================
#  Causal Stabilized Transform
# ================================================================

class OneEuroFilter1D:
    def __init__(self, mc=1.0, beta=0.007, dc=1.0):
        self.mc = mc; self.beta = beta; self.dc = dc
        self.x_prev = None; self.dx_prev = 0.0; self.t_prev = None
    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x; self.t_prev = t; return x
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
    stab_origins = np.full((N, 3), np.nan)
    stab_scales = np.full(N, np.nan)
    stab_rotations = np.full((N, 3, 3), np.nan)
    params = {"mc": 3.0, "beta": 0.001, "dc": 1.0}
    o_filts = [OneEuroFilter1D(**params) for _ in range(3)]
    for fi in range(N):
        if not np.any(np.isnan(norm["origins"][fi])):
            for d in range(3): stab_origins[fi, d] = o_filts[d](norm["origins"][fi, d], ts[fi])
    s_filt = OneEuroFilter1D(**params)
    for fi in range(N):
        if not np.isnan(norm["scales"][fi]):
            stab_scales[fi] = s_filt(norm["scales"][fi], ts[fi])
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
        up /= un
        fwd = np.cross(right, up)
        stab_rotations[fi] = np.column_stack([right, up, fwd])
    return {"stab_origins": stab_origins, "stab_scales": stab_scales, "stab_rotations": stab_rotations}


# ================================================================
#  Transform Comparison (v1.4: enhanced with vel/accel RMSE, idle jitter, per-lm detail)
# ================================================================

def compare_transforms(arrs, norm, stab, ts, seg_info):
    """Compare raw vs stabilized transform trajectories on key landmarks.
    Returns per-landmark detail and aggregate for CSV/JSON output."""
    per_landmark = []
    for comp, nlm, key_lms, comp_name in [
        ("pose", 33, POSE_KEY_LMS, "pose"),
        ("lh", 21, HAND_KEY_LMS, "left_hand"),
        ("rh", 21, HAND_KEY_LMS, "right_hand"),
    ]:
        xyz = arrs[f"{comp}_xyz"]; valid = arrs[f"{comp}_te"]
        raw_norm = normalize_with_transform(xyz, valid, norm["origins"], norm["rotations"], norm["scales"])
        stab_norm = normalize_with_transform(xyz, valid, stab["stab_origins"], stab["stab_rotations"], stab["stab_scales"])

        for li in key_lms:
            devs_m = []
            path_raw, path_stab = 0.0, 0.0
            vel_vec_errs = []
            accel_vec_errs = []
            prev_rv, prev_sv = None, None
            for fi in range(len(ts)):
                if valid[fi, li] and not np.any(np.isnan(raw_norm[fi, li])) and not np.any(np.isnan(stab_norm[fi, li])):
                    s = norm["scales"][fi] if not np.isnan(norm["scales"][fi]) else 0
                    devs_m.append(np.linalg.norm(stab_norm[fi, li] - raw_norm[fi, li]) * s)
                if fi > 0 and valid[fi, li] and valid[fi - 1, li]:
                    r0, r1 = raw_norm[fi - 1, li], raw_norm[fi, li]
                    s0, s1 = stab_norm[fi - 1, li], stab_norm[fi, li]
                    dt = ts[fi] - ts[fi - 1]
                    if not np.any(np.isnan(r0)) and not np.any(np.isnan(r1)):
                        path_raw += np.linalg.norm(r1 - r0)
                    if not np.any(np.isnan(s0)) and not np.any(np.isnan(s1)):
                        path_stab += np.linalg.norm(s1 - s0)
                    if dt > 0 and dt < 0.1 and not np.any(np.isnan(r0)) and not np.any(np.isnan(r1)) \
                       and not np.any(np.isnan(s0)) and not np.any(np.isnan(s1)):
                        sc = norm["scales"][fi] if not np.isnan(norm["scales"][fi]) else 0
                        rv = (r1 - r0) * sc / dt
                        sv = (s1 - s0) * sc / dt
                        vel_vec_errs.append(np.linalg.norm(sv - rv))
                        if prev_rv is not None and prev_sv is not None:
                            ra = (rv - prev_rv) / dt
                            sa = (sv - prev_sv) / dt
                            accel_vec_errs.append(np.linalg.norm(sa - ra))
                        prev_rv, prev_sv = rv, sv
                    else:
                        prev_rv = prev_sv = None
                else:
                    prev_rv = prev_sv = None

            if devs_m:
                da = np.array(devs_m)
                entry = {
                    "component": comp_name, "landmark": int(li),
                    "traj_dev_mean_mm": float(np.mean(da) * 1000),
                    "traj_dev_p95_mm": float(np.percentile(da, 95) * 1000),
                    "traj_dev_max_mm": float(np.max(da) * 1000),
                    "path_length_ratio": float(path_stab / path_raw) if path_raw > 0 else 1.0,
                    "velocity_vector_rmse_m_per_s": float(np.sqrt(np.mean(np.array(vel_vec_errs) ** 2))) if vel_vec_errs else 0.0,
                    "acceleration_vector_rmse_m_per_s2": float(np.sqrt(np.mean(np.array(accel_vec_errs) ** 2))) if accel_vec_errs else 0.0,
                }
                per_landmark.append(entry)

    # Idle jitter comparison (raw vs stabilized transform)
    idle_jitter = {}
    ss, se = seg_info["seg_start"], seg_info["seg_end"]
    N = len(ts)
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
            for comp in ["pose"]:
                valid_c = arrs[f"{comp}_te"]
                for li in POSE_KEY_LMS:
                    if valid_c[fi, li] and valid_c[fi - 1, li]:
                        xyz = arrs[f"{comp}_xyz"]
                        if not np.any(np.isnan(xyz[fi, li])) and not np.any(np.isnan(xyz[fi - 1, li])):
                            # Raw transform
                            r0 = normalize_with_transform(xyz[fi - 1:fi, li:li + 1], valid_c[fi - 1:fi, li:li + 1],
                                                          norm["origins"][fi - 1:fi], norm["rotations"][fi - 1:fi], norm["scales"][fi - 1:fi])
                            r1 = normalize_with_transform(xyz[fi:fi + 1, li:li + 1], valid_c[fi:fi + 1, li:li + 1],
                                                          norm["origins"][fi:fi + 1], norm["rotations"][fi:fi + 1], norm["scales"][fi:fi + 1])
                            if not np.any(np.isnan(r0)) and not np.any(np.isnan(r1)):
                                s = norm["scales"][fi] if not np.isnan(norm["scales"][fi]) else 0
                                jr.append(np.linalg.norm(r1[0, 0] - r0[0, 0]) * s)
                            # Stabilized transform
                            s0 = normalize_with_transform(xyz[fi - 1:fi, li:li + 1], valid_c[fi - 1:fi, li:li + 1],
                                                          stab["stab_origins"][fi - 1:fi], stab["stab_rotations"][fi - 1:fi], stab["stab_scales"][fi - 1:fi])
                            s1 = normalize_with_transform(xyz[fi:fi + 1, li:li + 1], valid_c[fi:fi + 1, li:li + 1],
                                                          stab["stab_origins"][fi:fi + 1], stab["stab_rotations"][fi:fi + 1], stab["stab_scales"][fi:fi + 1])
                            if not np.any(np.isnan(s0)) and not np.any(np.isnan(s1)):
                                s = norm["scales"][fi] if not np.isnan(norm["scales"][fi]) else 0
                                js.append(np.linalg.norm(s1[0, 0] - s0[0, 0]) * s)
        idle_jitter[f"{region}_raw_mm"] = float(np.mean(jr) * 1000) if jr else "not_available"
        idle_jitter[f"{region}_stabilized_mm"] = float(np.mean(js) * 1000) if js else "not_available"

    # Aggregate
    all_devs = [r["traj_dev_mean_mm"] for r in per_landmark]
    aggregate = {
        "traj_dev_mean_mm": float(np.mean(all_devs)) if all_devs else 0,
        "traj_dev_max_mm": float(max(r["traj_dev_max_mm"] for r in per_landmark)) if per_landmark else 0,
        "selected_normalization_transform": SELECTED_NORM_TRANSFORM,
    }

    return {"per_landmark": per_landmark, "aggregate": aggregate, "idle_jitter": idle_jitter}


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
        bd = np.zeros((N, nlm), dtype=bool)
        br = np.full((N, nlm), RSN_NONE, dtype=np.int8)
        bd[0, :] = True; br[0, :] = RSN_INITIAL
        for fi in range(1, N):
            if frame_ts_disc[fi]:
                bd[fi, :] = True
                br[fi, :] = np.where(br[fi, :] == RSN_NONE, RSN_TIMESTAMP_DISC, br[fi, :])
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
#  Interpolation (unchanged from v1.3)
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
#  OneEuro Filter (unchanged)
# ================================================================

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff; self.beta = beta; self.d_cutoff = d_cutoff
        self.x_prev = None; self.dx_prev = None; self.t_prev = None
    def reset(self):
        self.x_prev = None; self.dx_prev = None; self.t_prev = None
    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x.copy(); self.dx_prev = np.zeros_like(x); self.t_prev = t
            return x.copy()
        dt = t - self.t_prev
        if dt <= 0: return self.x_prev.copy()
        dx = (x - self.x_prev) / dt
        tau_d = 1.0 / (2 * math.pi * self.d_cutoff); a_d = 1.0 / (1.0 + tau_d / dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        speed = np.linalg.norm(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        tau = 1.0 / (2 * math.pi * cutoff); a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat.copy(); self.dx_prev = dx_hat.copy(); self.t_prev = t
        return x_hat


def apply_one_euro(xyz, valid, ts, boundary, boundary_reason, params):
    """Apply OneEuro with pending first_after_reset state."""
    N, nlm, _ = xyz.shape
    out = xyz.copy()
    reset_mask = np.zeros((N, nlm), dtype=bool)
    far_mask = np.zeros((N, nlm), dtype=bool)
    reset_rsn = np.full((N, nlm), RSN_NONE, dtype=np.int8)

    for li in range(nlm):
        filt = OneEuroFilter(params["min_cutoff"], params["beta"], params["d_cutoff"])
        pending = True  # initial state
        for fi in range(N):
            if boundary[fi, li]:
                filt.reset()
                reset_mask[fi, li] = True
                reset_rsn[fi, li] = boundary_reason[fi, li] if boundary_reason[fi, li] > 0 else RSN_OTHER
                pending = True
            if valid[fi, li] and not np.any(np.isnan(xyz[fi, li])):
                result = filt(xyz[fi, li], ts[fi])
                if pending:
                    far_mask[fi, li] = True
                    pending = False
                out[fi, li] = result
            else:
                filt.reset()
                if not reset_mask[fi, li]:
                    reset_mask[fi, li] = True
                    reset_rsn[fi, li] = RSN_INVALID_OBS
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
    """Compute reset/first-after-reset/reason for segment-based filters (no-smoothing, SavGol)."""
    N, nlm = valid.shape
    reset_mask = np.zeros((N, nlm), dtype=bool)
    far_mask = np.zeros((N, nlm), dtype=bool)
    reset_rsn = np.full((N, nlm), RSN_NONE, dtype=np.int8)
    for li in range(nlm):
        pending = True
        for fi in range(N):
            is_bd = boundary[fi, li]
            is_ok = valid[fi, li] and not np.any(np.isnan(xyz[fi, li]))
            if is_bd:
                reset_mask[fi, li] = True
                reset_rsn[fi, li] = boundary_reason[fi, li] if boundary_reason[fi, li] > 0 else RSN_OTHER
                pending = True
            if is_ok:
                if pending: far_mask[fi, li] = True; pending = False
            else:
                if not is_bd:
                    reset_mask[fi, li] = True; reset_rsn[fi, li] = RSN_INVALID_OBS
                pending = True
    return reset_mask, far_mask, reset_rsn


# ================================================================
#  Independent first_after_reset recomputation (for assertion)
# ================================================================

def recompute_first_after_reset(xyz, valid, boundary, boundary_reason):
    """Independently recompute first_after_reset and reset_mask for verification."""
    N, nlm = valid.shape
    reset_mask = np.zeros((N, nlm), dtype=bool)
    far_mask = np.zeros((N, nlm), dtype=bool)
    reset_rsn = np.full((N, nlm), RSN_NONE, dtype=np.int8)
    for li in range(nlm):
        pending = True
        for fi in range(N):
            is_bd = boundary[fi, li]
            is_ok = valid[fi, li] and not np.any(np.isnan(xyz[fi, li]))
            if is_bd:
                reset_mask[fi, li] = True
                reset_rsn[fi, li] = boundary_reason[fi, li] if boundary_reason[fi, li] > 0 else RSN_OTHER
                pending = True
            if is_ok:
                if pending: far_mask[fi, li] = True; pending = False
            else:
                if not is_bd:
                    reset_mask[fi, li] = True; reset_rsn[fi, li] = RSN_INVALID_OBS
                pending = True
    return reset_mask, far_mask, reset_rsn


# ================================================================
#  Extended Metrics (v1.4: enhanced matched-peak lag)
# ================================================================

def compute_extended_metrics(norm_base, norm_filt, valid, scales, ts, boundary, seg_info,
                             label, word, comp_nlm):
    N, nlm, _ = norm_base.shape
    key_lms = POSE_KEY_LMS if nlm == 33 else HAND_KEY_LMS

    # Position deviation
    phys_devs = []
    for fi in range(N):
        s = scales[fi] if not np.isnan(scales[fi]) else 0
        for li in range(nlm):
            if valid[fi, li] and not np.any(np.isnan(norm_base[fi, li])) and not np.any(np.isnan(norm_filt[fi, li])):
                phys_devs.append(np.linalg.norm(norm_filt[fi, li] - norm_base[fi, li]) * s)
    if not phys_devs:
        return _empty_metrics(label)
    phys_devs = np.array(phys_devs)

    # Jitter (start and end idle)
    j_start = _compute_idle_jitter(norm_base, norm_filt, valid, scales, 0, seg_info["seg_start"]) \
        if seg_info["idle_verified_start"] else None
    j_end = _compute_idle_jitter(norm_base, norm_filt, valid, scales, seg_info["seg_end"] + 1, N) \
        if seg_info["idle_verified_end"] else None

    # Boundary-aware velocity/acceleration on key landmarks
    vel_raw_all, vel_filt_all, vel_diffs = [], [], []
    accel_raw_all, accel_filt_all, accel_diffs = [], [], []
    path_raw, path_filt = 0.0, 0.0
    per_lm_results = []

    for li in key_lms:
        lm_vel_raw, lm_vel_filt = [], []
        lm_path_raw, lm_path_filt = 0.0, 0.0
        prev_rv, prev_fv = None, None
        for fi in range(1, N):
            dt = ts[fi] - ts[fi - 1]
            if dt <= 0 or dt > 0.1 or boundary[fi, li]: prev_rv = prev_fv = None; continue
            if not valid[fi, li] or not valid[fi - 1, li]: prev_rv = prev_fv = None; continue
            s = scales[fi] if not np.isnan(scales[fi]) else 0
            r0, r1 = norm_base[fi - 1, li], norm_base[fi, li]
            f0, f1 = norm_filt[fi - 1, li], norm_filt[fi, li]
            if np.any(np.isnan(r0)) or np.any(np.isnan(r1)) or np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                prev_rv = prev_fv = None; continue
            rv = np.linalg.norm(r1 - r0) * s / dt; fv = np.linalg.norm(f1 - f0) * s / dt
            vel_raw_all.append(rv); vel_filt_all.append(fv); vel_diffs.append(fv - rv)
            lm_vel_raw.append(rv); lm_vel_filt.append(fv)
            lm_path_raw += np.linalg.norm(r1 - r0) * s; lm_path_filt += np.linalg.norm(f1 - f0) * s
            # Acceleration
            rv_vec = (r1 - r0) * s / dt; fv_vec = (f1 - f0) * s / dt
            if prev_rv is not None:
                ra = np.linalg.norm(rv_vec - prev_rv) / dt; fa = np.linalg.norm(fv_vec - prev_fv) / dt
                accel_raw_all.append(ra); accel_filt_all.append(fa); accel_diffs.append(fa - ra)
            prev_rv, prev_fv = rv_vec, fv_vec
        path_raw += lm_path_raw; path_filt += lm_path_filt
        per_lm_results.append({"landmark": li, "n_vel": len(lm_vel_raw),
            "path_raw_m": lm_path_raw, "path_filt_m": lm_path_filt,
            "path_ratio": lm_path_filt / lm_path_raw if lm_path_raw > 0 else 1.0})

    vel_rmse = float(np.sqrt(np.mean(np.array(vel_diffs) ** 2))) if vel_diffs else 0
    vel_dist = float(abs(np.mean(vel_filt_all) - np.mean(vel_raw_all)) / np.mean(vel_raw_all) * 100) \
        if vel_raw_all and np.mean(vel_raw_all) > 0 else 0
    accel_rmse = float(np.sqrt(np.mean(np.array(accel_diffs) ** 2))) if accel_diffs else 0

    # Matched-peak timing lag (v1.4: per-segment per-landmark)
    peak_lag = _matched_peak_lag_v14(norm_base, norm_filt, valid, scales, ts, boundary, key_lms)

    # New NaN/Inf
    new_nans = sum(1 for fi in range(N) for li in range(nlm) if valid[fi, li]
                   and not np.any(np.isnan(norm_base[fi, li])) and np.any(np.isnan(norm_filt[fi, li])))
    new_infs = sum(1 for fi in range(N) for li in range(nlm) if valid[fi, li]
                   and not np.any(np.isnan(norm_base[fi, li])) and np.any(np.isinf(norm_filt[fi, li])))

    return {
        "label": label, "n_valid": len(phys_devs),
        "phys_dev_mean_mm": float(np.mean(phys_devs) * 1000),
        "phys_dev_p95_mm": float(np.percentile(phys_devs, 95) * 1000),
        "phys_dev_max_mm": float(np.max(phys_devs) * 1000),
        "vel_distortion_pct": vel_dist,
        "vel_rmse_m_per_s": vel_rmse,
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
        "per_landmark": per_lm_results,
        "peak_lag_detail": peak_lag,
    }


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
#  Matched-Peak Lag v1.4: per-segment per-landmark
# ================================================================

def _split_contiguous_valid_segments(speeds, boundary_flags, min_len=10):
    """Split speed array into contiguous valid segments (no boundary crossing)."""
    segments = []
    N = len(speeds)
    ss = None
    for fi in range(N):
        if boundary_flags[fi]:
            if ss is not None and fi - ss >= min_len:
                segments.append((ss, fi))
            ss = None
        else:
            if ss is None:
                ss = fi
    if ss is not None and N - ss >= min_len:
        segments.append((ss, N))
    return segments


def _matched_peak_lag_v14(norm_base, norm_filt, valid, scales, ts, boundary, key_lms):
    """Per-segment per-landmark peak matching with full event recording."""
    N = norm_base.shape[0]
    all_landmark_results = []
    all_matched_lags_frames = []
    all_matched_lags_ms = []

    for li in key_lms:
        # Compute per-frame speeds and boundary flags
        speeds_raw = np.zeros(N)
        speeds_filt = np.zeros(N)
        bd_flags = np.zeros(N, dtype=bool)
        bd_flags[0] = True
        for fi in range(1, N):
            dt = ts[fi] - ts[fi - 1]
            if dt <= 0 or dt > 0.1 or boundary[fi, li] or not valid[fi, li] or not valid[fi - 1, li]:
                bd_flags[fi] = True
                continue
            r0, r1 = norm_base[fi - 1, li], norm_base[fi, li]
            f0, f1 = norm_filt[fi - 1, li], norm_filt[fi, li]
            s = scales[fi] if not np.isnan(scales[fi]) else 0
            if np.any(np.isnan(r0)) or np.any(np.isnan(r1)) or np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                bd_flags[fi] = True; continue
            speeds_raw[fi] = np.linalg.norm(r1 - r0) * s / dt
            speeds_filt[fi] = np.linalg.norm(f1 - f0) * s / dt

        # Split into contiguous valid segments
        segments = _split_contiguous_valid_segments(speeds_raw, bd_flags, min_len=10)
        if not segments:
            all_landmark_results.append({
                "landmark": int(li), "n_segments": 0, "status": "no_valid_segments",
                "segments": [], "matched_events": [], "unmatched_events": []})
            continue

        lm_matches = []
        lm_unmatched = []
        seg_results = []

        for seg_start, seg_end in segments:
            sr = speeds_raw[seg_start:seg_end]
            sf = speeds_filt[seg_start:seg_end]
            ta = ts[seg_start:seg_end]

            # Find peaks in segment
            raw_peaks = _find_peaks_with_prominence(sr, min_prom_frac=0.2)
            filt_peaks = _find_peaks_with_prominence(sf, min_prom_frac=0.2)

            # Cross-correlation lag for segment
            xcorr_lag = _segment_xcorr_lag(sr, sf)

            # Onset timing: first frame above 20% of max speed
            onset_lag = _onset_lag(sr, sf, ta)

            seg_entry = {
                "seg_start": int(seg_start), "seg_end": int(seg_end), "seg_len": int(seg_end - seg_start),
                "n_raw_peaks": len(raw_peaks), "n_filt_peaks": len(filt_peaks),
                "xcorr_lag_frames": xcorr_lag,
                "onset_lag_ms": onset_lag,
            }

            # Match peaks within +-10 frame window
            used = set()
            for ri, rprom, rval in sorted(raw_peaks, key=lambda x: -x[2]):
                best_j, best_d = None, 11
                for j, (fp, _, _) in enumerate(filt_peaks):
                    if j in used: continue
                    d = fp - ri
                    if abs(d) < best_d: best_d = abs(d); best_j = j
                if best_j is not None and best_d <= 10:
                    used.add(best_j)
                    fp = filt_peaks[best_j][0]
                    lag_f = fp - ri
                    lag_ms = float((ta[fp] - ta[ri]) * 1000) if fp < len(ta) and ri < len(ta) else 0
                    if lag_f < -2:
                        status = "negative_match_inconsistent_with_causality"
                    else:
                        status = "matched"
                        all_matched_lags_frames.append(lag_f)
                        all_matched_lags_ms.append(lag_ms)
                    lm_matches.append({
                        "seg_start": int(seg_start), "raw_peak_idx": int(ri + seg_start),
                        "filt_peak_idx": int(fp + seg_start),
                        "lag_frames": int(lag_f), "lag_ms": float(lag_ms), "status": status,
                        "raw_prominence": float(rprom), "filt_prominence": float(filt_peaks[best_j][2])})
                else:
                    lm_unmatched.append({
                        "seg_start": int(seg_start), "raw_peak_idx": int(ri + seg_start),
                        "reason": "no_filt_peak_within_window", "raw_prominence": float(rprom)})
            # Unmatched filtered peaks
            for j, (fp, _, fprom) in enumerate(filt_peaks):
                if j not in used:
                    lm_unmatched.append({
                        "seg_start": int(seg_start), "filt_peak_idx": int(fp + seg_start),
                        "reason": "no_raw_peak_matched", "filt_prominence": float(fprom)})
            seg_results.append(seg_entry)

        lm_entry = {
            "landmark": int(li), "n_segments": len(segments),
            "n_matched": len([m for m in lm_matches if m["status"] == "matched"]),
            "n_negative_causal": len([m for m in lm_matches if m["status"] == "negative_match_inconsistent_with_causality"]),
            "n_unmatched": len(lm_unmatched),
            "segments": seg_results, "matched_events": lm_matches, "unmatched_events": lm_unmatched,
        }
        matched_only = [m for m in lm_matches if m["status"] == "matched"]
        if matched_only:
            lags_f = [m["lag_frames"] for m in matched_only]
            lags_ms = [m["lag_ms"] for m in matched_only]
            lm_entry["mean_lag_frames"] = float(np.mean(lags_f))
            lm_entry["mean_lag_ms"] = float(np.mean(lags_ms))
            lm_entry["status"] = "matched"
        else:
            lm_entry["status"] = "no_valid_matches"
        all_landmark_results.append(lm_entry)

    # Component aggregate
    result = {"per_landmark": all_landmark_results}
    if all_matched_lags_frames:
        result["aggregate_lag_frames"] = float(np.mean(all_matched_lags_frames))
        result["aggregate_lag_ms"] = float(np.mean(all_matched_lags_ms))
        result["total_matched"] = len(all_matched_lags_frames)
        result["status"] = "matched"
    else:
        result["status"] = "peak_correspondence_failed"
    return result


def _segment_xcorr_lag(sr, sf):
    """Cross-correlation lag between raw and filtered speed within a segment."""
    if len(sr) < 5: return 0
    sr_c = sr - np.mean(sr); sf_c = sf - np.mean(sf)
    sr_std = np.std(sr_c); sf_std = np.std(sf_c)
    if sr_std < 1e-12 or sf_std < 1e-12: return 0
    corr = np.correlate(sr_c / sr_std, sf_c / sf_std, mode="full") / len(sr)
    max_lag = min(10, len(sr) // 2)
    center = len(sr) - 1
    region = corr[center - max_lag:center + max_lag + 1]
    if len(region) == 0: return 0
    return int(np.argmax(region) - max_lag)


def _onset_lag(sr, sf, ta):
    """Timing difference for onset (first frame above 20% of peak speed)."""
    if len(sr) < 3: return 0.0
    threshold_r = np.max(sr) * 0.2; threshold_f = np.max(sf) * 0.2
    onset_r = next((i for i in range(len(sr)) if sr[i] > threshold_r), None)
    onset_f = next((i for i in range(len(sf)) if sf[i] > threshold_f), None)
    if onset_r is not None and onset_f is not None:
        return float((ta[onset_f] - ta[onset_r]) * 1000)
    return 0.0


def _find_peaks_with_prominence(signal, min_prom_frac=0.2):
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


# ================================================================
#  Sign-specific Metrics (unchanged from v1.3)
# ================================================================

def compute_sign_metrics(norm_base, norm_filt, valid, arrs, norm_dict, seg, word, ts, scales, boundary):
    N = norm_base.shape[0]; result = {}
    ss, se = seg["seg_start"], seg["seg_end"]
    if word == "LOVE" and norm_base.shape[1] >= 33:
        for label, lw, rw in [("pose_wrists", LM_LW, LM_RW)]:
            dists_raw, dists_filt, dists_raw_fi, dists_filt_fi = [], [], [], []
            for fi in range(ss, min(se + 1, N)):
                if valid[fi, lw] and valid[fi, rw]:
                    s = scales[fi] if not np.isnan(scales[fi]) else 0
                    lr, rr = norm_base[fi, lw], norm_base[fi, rw]
                    lf, rf = norm_filt[fi, lw], norm_filt[fi, rw]
                    if not np.any(np.isnan(lr)) and not np.any(np.isnan(rr)):
                        dists_raw.append(np.linalg.norm(rr - lr) * s); dists_raw_fi.append(fi)
                    if not np.any(np.isnan(lf)) and not np.any(np.isnan(rf)):
                        dists_filt.append(np.linalg.norm(rf - lf) * s); dists_filt_fi.append(fi)
            if dists_raw and dists_filt:
                mr_fi = dists_raw_fi[int(np.argmin(dists_raw))]; mf_fi = dists_filt_fi[int(np.argmin(dists_filt))]
                result["love_min_hand_dist_raw_mm"] = float(np.min(dists_raw) * 1000)
                result["love_min_hand_dist_filt_mm"] = float(np.min(dists_filt) * 1000)
                result["love_min_dist_timing_shift_ms"] = float((ts[mf_fi] - ts[mr_fi]) * 1000)
    elif word == "GREETING" and norm_base.shape[1] >= 33:
        bd = boundary
        for wname, wli in [("left_wrist", LM_LW), ("right_wrist", LM_RW)]:
            raw_reversals, filt_reversals = [], []
            prev_vy_raw, prev_vy_filt = None, None
            for fi in range(ss + 1, min(se + 1, N)):
                if bd[fi, wli] or not valid[fi, wli] or not valid[fi - 1, wli]:
                    prev_vy_raw = prev_vy_filt = None; continue
                r0y = norm_base[fi - 1, wli, 1]; r1y = norm_base[fi, wli, 1]
                f0y = norm_filt[fi - 1, wli, 1]; f1y = norm_filt[fi, wli, 1]
                if np.isnan(r0y) or np.isnan(r1y): vy_raw = None
                else: vy_raw = r1y - r0y
                if np.isnan(f0y) or np.isnan(f1y): vy_filt = None
                else: vy_filt = f1y - f0y
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
                    if abs(best["frame"] - rr["frame"]) <= 5:
                        timing_diffs.append((best["time_s"] - rr["time_s"]) * 1000)
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
            z_raw = [norm_base[fi, wli, 2] for fi in range(ss, min(se + 1, N))
                     if valid[fi, wli] and not np.isnan(norm_base[fi, wli, 2])]
            z_filt = [norm_filt[fi, wli, 2] for fi in range(ss, min(se + 1, N))
                      if valid[fi, wli] and not np.isnan(norm_filt[fi, wli, 2])]
            s_mean = np.nanmean(scales[ss:se + 1])
            if z_raw:
                result[f"thankyou_{wname}_fwd_amplitude_raw_mm"] = float((max(z_raw) - min(z_raw)) * s_mean * 1000) if not np.isnan(s_mean) else 0
            if z_filt:
                result[f"thankyou_{wname}_fwd_amplitude_filt_mm"] = float((max(z_filt) - min(z_filt)) * s_mean * 1000) if not np.isnan(s_mean) else 0
    return result


# ================================================================
#  Masking Experiment (v1.4: magnitude vs vector error separation)
# ================================================================

def run_masking_experiment(arrs, norm, boundaries, strong_forbidden, ts, sid):
    results = []
    for comp, nlm in [("pose", 33), ("lh", 21), ("rh", 21)]:
        xyz = arrs[f"{comp}_xyz"]; valid = arrs[f"{comp}_te"]
        bd = boundaries[comp]; sf = strong_forbidden[comp]; N = xyz.shape[0]
        for gap_len in [1, 2, 3]:
            pos_errs = []
            speed_mag_errs, accel_mag_errs = [], []
            vel_vec_errs, accel_vec_errs = [], []
            entry_vd, exit_vd = [], []
            n_windows = 0
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
                        p0 = xyz[start - 1, li]; p1 = xyz[end + 1, li]
                        n_windows += 1
                        for g in range(gap_len):
                            t = (g + 1) / (gap_len + 1)
                            recon = p0 * (1 - t) + p1 * t
                            actual = xyz[start + g, li]
                            pos_errs.append(np.linalg.norm(recon - actual))
                        # Velocity/acceleration comparison within window
                        recon_pts = [p0]
                        for g in range(gap_len):
                            t = (g + 1) / (gap_len + 1); recon_pts.append(p0 * (1 - t) + p1 * t)
                        recon_pts.append(p1)
                        actual_pts = [xyz[start - 1 + i, li] for i in range(gap_len + 2)]
                        for i in range(1, len(recon_pts)):
                            fi_a = start - 1 + i; fi_b = start - 1 + i - 1
                            dt = ts[fi_a] - ts[fi_b] if fi_a < N and fi_b >= 0 else 1 / 30
                            if dt > 0 and dt < 0.2:
                                recon_vel_vec = (recon_pts[i] - recon_pts[i - 1]) / dt
                                actual_vel_vec = (actual_pts[i] - actual_pts[i - 1]) / dt
                                # Magnitude error (speed)
                                speed_mag_errs.append(abs(np.linalg.norm(recon_vel_vec) - np.linalg.norm(actual_vel_vec)))
                                # Vector error
                                vel_vec_errs.append(np.linalg.norm(recon_vel_vec - actual_vel_vec))
                        # Acceleration within window
                        for i in range(1, len(recon_pts) - 1):
                            fi_1 = start - 1 + i; fi_0 = fi_1 - 1; fi_2 = fi_1 + 1
                            dt1 = ts[fi_1] - ts[fi_0] if fi_1 < N and fi_0 >= 0 else 1 / 30
                            dt2 = ts[fi_2] - ts[fi_1] if fi_2 < N and fi_1 >= 0 else 1 / 30
                            if dt1 > 0 and dt2 > 0 and dt1 < 0.2 and dt2 < 0.2:
                                recon_a_vec = ((recon_pts[i + 1] - recon_pts[i]) / dt2 - (recon_pts[i] - recon_pts[i - 1]) / dt1) / ((dt1 + dt2) / 2)
                                actual_a_vec = ((actual_pts[i + 1] - actual_pts[i]) / dt2 - (actual_pts[i] - actual_pts[i - 1]) / dt1) / ((dt1 + dt2) / 2)
                                # Magnitude error
                                accel_mag_errs.append(abs(np.linalg.norm(recon_a_vec) - np.linalg.norm(actual_a_vec)))
                                # Vector error
                                accel_vec_errs.append(np.linalg.norm(recon_a_vec - actual_a_vec))
                        # Entry/exit velocity discontinuity
                        if start >= 2:
                            dt_e = ts[start] - ts[start - 1] if start < N else 1 / 30
                            dt_p = ts[start - 1] - ts[start - 2] if start - 1 >= 1 else 1 / 30
                            if dt_e > 0 and dt_p > 0 and dt_e < 0.2 and dt_p < 0.2:
                                v_pre = np.linalg.norm(xyz[start - 1, li] - xyz[start - 2, li]) / dt_p
                                v_entry = np.linalg.norm(recon_pts[1] - recon_pts[0]) / dt_e
                                entry_vd.append(abs(v_entry - v_pre))
                        if end + 2 < N:
                            dt_x = ts[end + 2] - ts[end + 1] if end + 2 < N else 1 / 30
                            dt_l = ts[end + 1] - ts[end] if end + 1 < N else 1 / 30
                            if dt_x > 0 and dt_l > 0 and dt_x < 0.2 and dt_l < 0.2:
                                v_post = np.linalg.norm(xyz[end + 2, li] - xyz[end + 1, li]) / dt_x
                                v_exit = np.linalg.norm(recon_pts[-1] - recon_pts[-2]) / dt_l
                                exit_vd.append(abs(v_exit - v_post))
            if pos_errs:
                pe = np.array(pos_errs)
                row = {"sample_id": sid, "component": comp, "gap_length": gap_len,
                       "n_windows": n_windows, "n_masked_points": len(pe),
                       "recon_err_mean_mm": float(np.mean(pe) * 1000),
                       "recon_err_p95_mm": float(np.percentile(pe, 95) * 1000),
                       "recon_err_max_mm": float(np.max(pe) * 1000)}
                if speed_mag_errs:
                    sme = np.array(speed_mag_errs)
                    row["speed_magnitude_error_rmse"] = float(np.sqrt(np.mean(sme ** 2)))
                    row["speed_magnitude_error_p95"] = float(np.percentile(sme, 95))
                    row["speed_magnitude_error_max"] = float(np.max(sme))
                if vel_vec_errs:
                    vve = np.array(vel_vec_errs)
                    row["velocity_vector_error_rmse"] = float(np.sqrt(np.mean(vve ** 2)))
                    row["velocity_vector_error_p95"] = float(np.percentile(vve, 95))
                    row["velocity_vector_error_max"] = float(np.max(vve))
                if accel_mag_errs:
                    ame = np.array(accel_mag_errs)
                    row["acceleration_magnitude_error_rmse"] = float(np.sqrt(np.mean(ame ** 2)))
                    row["acceleration_magnitude_error_p95"] = float(np.percentile(ame, 95))
                    row["acceleration_magnitude_error_max"] = float(np.max(ame))
                if accel_vec_errs:
                    ave = np.array(accel_vec_errs)
                    row["acceleration_vector_error_rmse"] = float(np.sqrt(np.mean(ave ** 2)))
                    row["acceleration_vector_error_p95"] = float(np.percentile(ave, 95))
                    row["acceleration_vector_error_max"] = float(np.max(ave))
                if entry_vd:
                    row["entry_vel_disc_mean_m_per_s"] = float(np.mean(entry_vd))
                if exit_vd:
                    row["exit_vel_disc_mean_m_per_s"] = float(np.mean(exit_vd))
                results.append(row)
    return results


def compute_masking_weighted_aggregate(masking_all):
    """Compute weighted aggregate rows per component per gap_length."""
    agg_rows = []
    for comp in ["pose", "lh", "rh"]:
        for gl in [1, 2, 3]:
            rows = [r for r in masking_all if r["component"] == comp and r["gap_length"] == gl]
            if not rows: continue
            total_pts = sum(r["n_masked_points"] for r in rows)
            total_win = sum(r["n_windows"] for r in rows)
            if total_pts == 0: continue
            w_mean = sum(r["recon_err_mean_mm"] * r["n_masked_points"] for r in rows) / total_pts
            w_p95 = max(r["recon_err_p95_mm"] for r in rows)
            w_max = max(r["recon_err_max_mm"] for r in rows)
            agg = {"sample_id": "WEIGHTED_AGGREGATE", "component": comp, "gap_length": gl,
                   "n_windows": total_win, "n_masked_points": total_pts,
                   "recon_err_mean_mm": float(w_mean), "recon_err_p95_mm": float(w_p95),
                   "recon_err_max_mm": float(w_max)}
            # Aggregate vector metrics if available
            for metric_key in ["speed_magnitude_error_rmse", "velocity_vector_error_rmse",
                               "acceleration_magnitude_error_rmse", "acceleration_vector_error_rmse"]:
                vals = [r[metric_key] for r in rows if metric_key in r]
                if vals:
                    weights = [r["n_masked_points"] for r in rows if metric_key in r]
                    agg[metric_key] = float(sum(v * w for v, w in zip(vals, weights)) / sum(weights))
            agg_rows.append(agg)
    return agg_rows


# ================================================================
#  Contact sheet (unchanged)
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


# ================================================================
#  NPZ Save / Verify (strict schema, unchanged logic)
# ================================================================

def save_sample_npz(sid, arrs, norm, interp_data, smooth_data, norm_interp_cache, stab, out_dir):
    short = sid.replace("S001_P001_", ""); npz_dir = out_dir / "npz"; npz_dir.mkdir(exist_ok=True)
    prov_dir = out_dir / "provenance"; prov_dir.mkdir(exist_ok=True)

    for mg in INTERP_CANDIDATES:
        ir = interp_data[mg]; arrays = {}
        for comp in ["pose", "lh", "rh"]:
            arrays[f"{comp}_interp"] = ir[f"{comp}_interp"]
            arrays[f"{comp}_norm_interp"] = norm_interp_cache[f"norm_interp_{comp}_g{mg}"]
            arrays[f"{comp}_original_valid"] = arrs[f"{comp}_te"]
            arrays[f"{comp}_mask"] = ir[f"{comp}_mask"]
            arrays[f"{comp}_rejected"] = ir[f"{comp}_rejected"]
            arrays[f"{comp}_boundary"] = ir[f"{comp}_boundary"]
            arrays[f"{comp}_strong_forbidden"] = ir[f"{comp}_strong_forbidden"]
        np.savez_compressed(npz_dir / f"interp_{short}_g{mg}.npz", **arrays)
        prov = {comp: ir[f"{comp}_records"] for comp in ["pose", "lh", "rh"]}
        with open(prov_dir / f"interp_provenance_{short}_g{mg}.json", "w", encoding="utf-8") as f:
            json.dump(prov, f)

    for fname, sm in smooth_data.items():
        arrays = {}
        for comp in ["pose", "lh", "rh"]:
            arrays[f"{comp}_filtered"] = sm[f"{comp}_filtered"]
            arrays[f"{comp}_norm_filtered"] = sm[f"{comp}_norm_filtered"]
            arrays[f"{comp}_filter_valid"] = sm[f"{comp}_filter_valid"]
            arrays[f"{comp}_reset_mask"] = sm[f"{comp}_reset_mask"]
            arrays[f"{comp}_first_after_reset"] = sm[f"{comp}_first_after_reset"]
            arrays[f"{comp}_reset_reason"] = sm[f"{comp}_reset_reason"]
        np.savez_compressed(npz_dir / f"smooth_{short}_{fname}.npz", **arrays)
        schema = {"filter": fname, "sample_id": sid, "interp_candidate_used": 2,
                  "reset_reason_codes": RSN_CODE_MAP,
                  "first_after_reset_includes_initial": True}
        if fname.startswith("oe_"):
            pname = fname[3:]
            for c in ONE_EURO_CANDIDATES:
                if c["name"] == pname: schema["filter_params"] = c; break
        elif fname == "savgol": schema["filter_params"] = {"window": SAVGOL_WINDOW, "order": SAVGOL_ORDER}
        elif fname == "none": schema["filter_params"] = {"type": "identity"}
        for comp in ["pose", "lh", "rh"]:
            schema[f"{comp}_metrics"] = sm[f"{comp}_metrics"]
            if f"{comp}_sign_metrics" in sm: schema[f"{comp}_sign_metrics"] = sm[f"{comp}_sign_metrics"]
        with open(prov_dir / f"smooth_schema_{short}_{fname}.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, default=str)

    arrays = {"origins": norm["origins"], "rotations": norm["rotations"], "scales": norm["scales"],
              "dets": norm["dets"], "ortho_errs": norm["ortho_errs"],
              "norm_pose": norm["norm_pose"], "norm_lh": norm["norm_lh"], "norm_rh": norm["norm_rh"],
              "stab_origins": stab["stab_origins"], "stab_scales": stab["stab_scales"],
              "stab_rotations": stab["stab_rotations"]}
    np.savez_compressed(npz_dir / f"norm_{short}.npz", **arrays)


def verify_sample_npz(sid, arrs, norm, interp_data, smooth_data, norm_interp_cache, stab, out_dir):
    short = sid.replace("S001_P001_", ""); npz_dir = out_dir / "npz"; fails = []
    expected_interp = set(interp_npz_keys())
    expected_smooth = set(smooth_npz_keys())
    expected_norm = set(norm_npz_keys())

    for mg in INTERP_CANDIDATES:
        path = npz_dir / f"interp_{short}_g{mg}.npz"
        if not path.exists(): fails.append(f"missing_interp_g{mg}"); continue
        loaded = np.load(path, allow_pickle=False)
        loaded_keys = set(loaded.files)
        # Strict: exact key match
        if loaded_keys != expected_interp:
            for ek in expected_interp - loaded_keys: fails.append(f"missing_key_interp_g{mg}_{ek}")
            for ek in loaded_keys - expected_interp: fails.append(f"extra_key_interp_g{mg}_{ek}")
        ir = interp_data[mg]
        mem_map = {}
        for comp in ["pose", "lh", "rh"]:
            mem_map[f"{comp}_interp"] = ir[f"{comp}_interp"]
            mem_map[f"{comp}_norm_interp"] = norm_interp_cache[f"norm_interp_{comp}_g{mg}"]
            mem_map[f"{comp}_original_valid"] = arrs[f"{comp}_te"]
            mem_map[f"{comp}_mask"] = ir[f"{comp}_mask"]
            mem_map[f"{comp}_rejected"] = ir[f"{comp}_rejected"]
            mem_map[f"{comp}_boundary"] = ir[f"{comp}_boundary"]
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
        loaded = np.load(path, allow_pickle=False)
        loaded_keys = set(loaded.files)
        # Strict: exact key match
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
        loaded = np.load(path, allow_pickle=False)
        loaded_keys = set(loaded.files)
        if loaded_keys != expected_norm:
            for ek in expected_norm - loaded_keys: fails.append(f"missing_key_norm_{ek}")
            for ek in loaded_keys - expected_norm: fails.append(f"extra_key_norm_{ek}")
        norm_map = {"origins": norm["origins"], "rotations": norm["rotations"], "scales": norm["scales"],
                    "dets": norm["dets"], "ortho_errs": norm["ortho_errs"],
                    "norm_pose": norm["norm_pose"], "norm_lh": norm["norm_lh"], "norm_rh": norm["norm_rh"],
                    "stab_origins": stab["stab_origins"], "stab_scales": stab["stab_scales"],
                    "stab_rotations": stab["stab_rotations"]}
        for k, mem in norm_map.items():
            if k in loaded_keys:
                if not np.allclose(loaded[k], mem, equal_nan=True, atol=1e-12): fails.append(f"mismatch_norm_{k}")
        loaded.close()
    return fails


# ================================================================
#  Assertions (v1.4: enhanced with section 3 checks)
# ================================================================

def run_assertions(all_results, split, pilot_ids, input_hashes, strong_depth, out_dir,
                   boundary_reasons_all):
    results = []; np_ = [0]; nf_ = [0]
    def chk(cond, name, detail=""):
        if cond: np_[0] += 1
        else: nf_[0] += 1; sp(f"    [FAIL] {name}: {detail}")
        results.append({"name": name, "status": "PASS" if cond else "FAIL", "detail": detail})

    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        h = sha256_file(RAW3D_DIR / f"landmarks_3d_{short}.json")
        chk(h == input_hashes[sid], f"input_hash_{short}")
    chk(len(pilot_ids) == 9, "pilot_count_9")
    for w in WORDS: chk(len([s for s in pilot_ids if w in s]) == 3, f"pilot_{w}_3")
    for sid in pilot_ids: chk(split[sid]["split"] == "train", f"pilot_train_{sid.replace('S001_P001_','')}")

    for sid, sr in all_results.items():
        short = sid.replace("S001_P001_", "")
        arrs = sr["arrs"]; norm = sr["norm"]; raw = sr["raw_data"]; N = len(arrs["timestamps"])

        # Timestamp preservation
        for fi, fr in enumerate(raw["frames"]):
            chk(abs(arrs["timestamps"][fi] - fr["front_time_s"]) < 1e-12, f"ts_{short}_f{fi}")

        # Coordinate + validity (all frames, all landmarks)
        KEYS_MAP = {"pose": ("pose_landmarks_3d", 33), "lh": ("left_hand_landmarks_3d", 21), "rh": ("right_hand_landmarks_3d", 21)}
        for comp, (jkey, nlm) in KEYS_MAP.items():
            for fi in range(N):
                for li in range(nlm):
                    lm = raw["frames"][fi][jkey][li]
                    chk(arrs[f"{comp}_tri"][fi, li] == lm["triangulation_valid"], f"tri_{short}_{comp}_f{fi}_l{li}")
                    chk(arrs[f"{comp}_qa"][fi, li] == lm["qa_valid"], f"qa_{short}_{comp}_f{fi}_l{li}")
                    chk(arrs[f"{comp}_te"][fi, li] == lm["template_eligible"], f"te_{short}_{comp}_f{fi}_l{li}")
                    if lm["triangulation_valid"] and lm["xyz"] is not None:
                        chk(np.allclose(np.array(lm["xyz"]), arrs[f"{comp}_xyz"][fi, li], atol=1e-12),
                            f"xyz_{short}_{comp}_f{fi}_l{li}")

        # Rotation checks
        for fi in range(N):
            if norm["statuses"][fi] in ("ok", "fallback_prev", "fallback_next"):
                R = norm["rotations"][fi]
                if not np.any(np.isnan(R)):
                    chk(abs(norm["dets"][fi] - 1.0) < 1e-6, f"det_{short}_f{fi}")
                    chk(norm["ortho_errs"][fi] < 1e-6, f"ortho_{short}_f{fi}")
        # Axis continuity
        for fi in range(1, N):
            if norm["statuses"][fi] in ("ok", "fallback_prev", "fallback_next") and norm["statuses"][fi-1] in ("ok", "fallback_prev", "fallback_next"):
                R0, R1 = norm["rotations"][fi-1], norm["rotations"][fi]
                if not np.any(np.isnan(R0)) and not np.any(np.isnan(R1)):
                    for col, name in [(0,"right"),(1,"up"),(2,"forward")]:
                        chk(np.dot(R0[:,col], R1[:,col]) >= 0, f"no_{name}_flip_{short}_f{fi}")

        # Scale
        vs = norm["scales"][~np.isnan(norm["scales"])]
        chk(len(vs) > 0, f"scale_exists_{short}")
        if len(vs) > 0:
            chk(np.all(vs > 0.05), f"scale_pos_{short}"); chk(np.all(vs < 2.0), f"scale_range_{short}")

        # Strong-depth interpolation = 0
        for mg, ir in sr["interp"].items():
            for comp in ["pose", "lh", "rh"]:
                chk(int(np.sum(ir[f"{comp}_mask"] & ir[f"{comp}_strong_forbidden"])) == 0,
                    f"no_sd_interp_{short}_{comp}_g{mg}")
                # No boundary crossing
                for li in range(ir[f"{comp}_mask"].shape[1]):
                    for fi in range(ir[f"{comp}_mask"].shape[0]):
                        if ir[f"{comp}_mask"][fi, li] and ir[f"{comp}_boundary"][fi, li]:
                            chk(False, f"bd_cross_{short}_{comp}_g{mg}_f{fi}_l{li}")

        # Known regression: LOVE_FAST_R04 pose lm20 f37-38
        if sid == "S001_P001_LOVE_FAST_R04":
            for mg in INTERP_CANDIDATES:
                for fi in [37, 38]:
                    if fi < N: chk(not sr["interp"][mg]["pose_mask"][fi, 20], f"regress_LFR04_p20_f{fi}_g{mg}")

        # No-smoothing deviation = 0
        ns = sr["smooth"].get("none", {})
        for comp in ["pose", "lh", "rh"]:
            m = ns.get(f"{comp}_metrics", {})
            chk(m.get("phys_dev_mean_mm", -1) == 0, f"no_smooth_zero_{short}_{comp}_mean")
            chk(m.get("phys_dev_p95_mm", -1) == 0, f"no_smooth_zero_{short}_{comp}_p95")
            chk(m.get("vel_distortion_pct", -1) == 0, f"no_smooth_vel_zero_{short}_{comp}")

        # Interpolation provenance count
        for mg, ir in sr["interp"].items():
            for comp in ["pose", "lh", "rh"]:
                n_interp = int(np.sum(ir[f"{comp}_mask"]))
                recs = [r for r in ir[f"{comp}_records"] if r["method"] == "linear"]
                chk(n_interp == sum(r["gap_length"] for r in recs), f"interp_prov_{short}_{comp}_g{mg}")

        # Smoothing: first-after-reset == input at ALL marked positions (ALL filters including SavGol)
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                far = sm[f"{comp}_first_after_reset"]
                filtered = sm[f"{comp}_filtered"]
                base = sr["interp"][2][f"{comp}_interp"]
                for fi in range(far.shape[0]):
                    for li in range(far.shape[1]):
                        if far[fi, li]:
                            if not np.any(np.isnan(filtered[fi, li])) and not np.any(np.isnan(base[fi, li])):
                                chk(np.allclose(filtered[fi, li], base[fi, li], atol=1e-10),
                                    f"far_eq_input_{short}_{comp}_{fname}_f{fi}_l{li}")

        # Smoothing: reset at every boundary
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                rm = sm[f"{comp}_reset_mask"]
                bd = sr["interp"][2][f"{comp}_boundary"]
                for li in range(rm.shape[1]):
                    for fi in range(rm.shape[0]):
                        if bd[fi, li]: chk(rm[fi, li], f"smooth_reset_{short}_{comp}_{fname}_f{fi}_l{li}")

        # No new NaN/Inf
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                m = sm[f"{comp}_metrics"]
                chk(m["new_nans"] == 0, f"no_nan_{short}_{comp}_{fname}")
                chk(m["new_infs"] == 0, f"no_inf_{short}_{comp}_{fname}")

        # -- v1.3 regression: first_after_reset mask count > 0 for OneEuro --
        for fname, sm in sr["smooth"].items():
            if not fname.startswith("oe_"): continue
            total_far = sum(int(np.sum(sm[f"{c}_first_after_reset"])) for c in ["pose", "lh", "rh"])
            chk(total_far > 0, f"far_nonzero_{short}_{fname}")

        # -- v1.3 regression: filter_valid / reset_reason present --
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                chk(f"{comp}_filter_valid" in sm, f"has_filter_valid_{short}_{comp}_{fname}")
                chk(f"{comp}_reset_reason" in sm, f"has_reset_reason_{short}_{comp}_{fname}")

        # -- v1.4: reset_mask == (reset_reason != 0) --
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                rm = sm[f"{comp}_reset_mask"]
                rsn = sm[f"{comp}_reset_reason"]
                chk(np.array_equal(rm, rsn != RSN_NONE),
                    f"reset_mask_eq_reason_{short}_{comp}_{fname}")

        # -- v1.4: first_after_reset -> filter_valid --
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                far = sm[f"{comp}_first_after_reset"]
                fv = sm[f"{comp}_filter_valid"]
                # Every far position must have filter_valid=True
                chk(np.all(fv[far]), f"far_implies_valid_{short}_{comp}_{fname}")

        # -- v1.4: independent first_after_reset recomputation --
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                valid_sm = sm[f"{comp}_filter_valid"]
                bd = sr["interp"][2][f"{comp}_boundary"]
                br = boundary_reasons_all[sid][comp]
                base_xyz = sr["interp"][2][f"{comp}_interp"]
                ref_rm, ref_far, ref_rsn = recompute_first_after_reset(base_xyz, valid_sm, bd, br)
                stored_far = sm[f"{comp}_first_after_reset"]
                n_missing = int(np.sum(ref_far & ~stored_far))
                n_extra = int(np.sum(stored_far & ~ref_far))
                chk(n_missing == 0, f"far_recomp_missing_{short}_{comp}_{fname}", f"missing={n_missing}")
                chk(n_extra == 0, f"far_recomp_extra_{short}_{comp}_{fname}", f"extra={n_extra}")

        # NPZ round-trip
        for fail_name in sr.get("npz_verify_fails", []):
            chk(False, f"npz_{short}_{fail_name}")
        if not sr.get("npz_verify_fails"): chk(True, f"npz_{short}_all_pass")

    # -- v1.4: input manifest checks --
    inp_man = out_dir / "input_file_manifest.json"
    if inp_man.exists():
        with open(inp_man, encoding="utf-8") as f: inp_data = json.load(f)
        inputs = inp_data.get("inputs", [])
        has_script = any(e.get("artifact_type") == "execution_script" for e in inputs)
        has_config = any(e.get("artifact_type") == "execution_config" for e in inputs)
        chk(has_script, "input_manifest_has_script")
        chk(has_config, "input_manifest_has_config")
        # SHA-256 match for script
        script_entry = next((e for e in inputs if e.get("artifact_type") == "execution_script"), None)
        if script_entry:
            actual_hash = sha256_file(PROJECT / script_entry["file"])
            chk(actual_hash == script_entry["sha256"], "input_manifest_script_hash_match")
        # SHA-256 match for config
        config_entry = next((e for e in inputs if e.get("artifact_type") == "execution_config"), None)
        if config_entry:
            actual_hash = sha256_file(PROJECT / config_entry["file"])
            chk(actual_hash == config_entry["sha256"], "input_manifest_config_hash_match")

    # -- v1.4: run manifest checks --
    rm_path = out_dir / "run_manifest.json"
    if rm_path.exists():
        with open(rm_path, encoding="utf-8") as f: rm_data = json.load(f)
        chk(rm_data.get("smoothing_recommendation") == SMOOTHING_CONCLUSION, "run_manifest_smoothing_conclusion")
        chk(rm_data.get("selected_normalization_transform") == SELECTED_NORM_TRANSFORM, "run_manifest_norm_transform")
        chk(rm_data.get("selected_max_gap") == SELECTED_MAX_GAP, "run_manifest_max_gap")

    # -- v1.4: assertion_results.json exists (written before this check re-run) --
    chk((out_dir / "assertion_results.json").exists(), "assertion_results_file_exists")

    sp(f"\n  Assertions: {np_[0]} PASS, {nf_[0]} FAIL")
    return results, np_[0], nf_[0]


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

def save_full_sequence(sid, arrs, norm, seg, out_dir):
    short = sid.replace("S001_P001_", ""); N = len(arrs["timestamps"])
    seq = {"sample_id": sid, "version": VERSION_TAG, "n_frames": N,
           "coordinate_system": {"raw": "FRONT_camera_meters", "normalized": "body_centric_shoulder_width_scaled"},
           "segmentation": {"start": seg["seg_start"], "end": seg["seg_end"], "method": seg["method"],
                             "idle_verified_start": seg["idle_verified_start"], "idle_verified_end": seg["idle_verified_end"]},
           "frames": []}
    for fi in range(N):
        frame = {"frame_index": fi, "timestamp_s": float(arrs["timestamps"][fi]),
                 "in_segment": seg["seg_start"] <= fi <= seg["seg_end"],
                 "transform": {
                     "origin": norm["origins"][fi].tolist() if not np.any(np.isnan(norm["origins"][fi])) else None,
                     "rotation": norm["rotations"][fi].tolist() if not np.any(np.isnan(norm["rotations"][fi])) else None,
                     "scale": float(norm["scales"][fi]) if not np.isnan(norm["scales"][fi]) else None,
                     "status": norm["statuses"][fi], "fallback_src": norm["fallback_src"][fi],
                     "determinant": float(norm["dets"][fi]) if not np.isnan(norm["dets"][fi]) else None,
                     "ortho_error": float(norm["ortho_errs"][fi]) if not np.isnan(norm["ortho_errs"][fi]) else None}}
        for comp_full, jkey, nlm in [("pose","pose_landmarks_3d",33),("left_hand","left_hand_landmarks_3d",21),("right_hand","right_hand_landmarks_3d",21)]:
            cs = {"pose":"pose","left_hand":"lh","right_hand":"rh"}[comp_full]
            lms = []
            for li in range(nlm):
                rp = arrs[f"{cs}_xyz"][fi, li]; np_v = norm[f"norm_{cs}"][fi, li]
                lms.append({"raw_xyz": rp.tolist() if not np.any(np.isnan(rp)) else None,
                             "norm_xyz": np_v.tolist() if not np.any(np.isnan(np_v)) else None,
                             "tri_valid": bool(arrs[f"{cs}_tri"][fi, li]), "qa_valid": bool(arrs[f"{cs}_qa"][fi, li]),
                             "te": bool(arrs[f"{cs}_te"][fi, li]),
                             "qa_flags": arrs[f"{cs}_flags"][fi][li],
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
#  Report (v1.4: enhanced with sample IDs, defaults section)
# ================================================================

def write_report(out_dir, pilot_ids, pilot, seg_rows, interp_rows, smooth_rows,
                 masking_rows, masking_agg, transform_rows, np_, nf_):
    status = "STEP7A_PREPROCESSING_PILOT_V1_4_COMPLETE" if nf_ == 0 else "STEP7A_V1_4_BLOCKED"
    rpt = [f"# Step 7A v{VERSION} Preprocessing Pilot Report", "",
           f"## Status: {status}", "",
           f"**Assertions**: {np_} PASS / {nf_} FAIL", "",
           "## Pilot Samples", "", "| Sample | Tier | TE Rate |", "|--------|------|---------|"]
    for sid in pilot_ids:
        p = pilot[sid]; rpt.append(f"| {sid.replace('S001_P001_','')} | {p['tier']} | {p['te_rate']:.1%} |")

    # Masking experiment with sample IDs
    rpt += ["", "## Masking Experiment (per sample)", "",
            "| Sample | Comp | Gap | Windows | Points | Mean(mm) | P95(mm) | Max(mm) | SpeedMagRMSE | VelVecRMSE |",
            "|--------|------|-----|---------|--------|----------|---------|---------|--------------|------------|"]
    for r in masking_rows:
        smr = f"{r.get('speed_magnitude_error_rmse', 0):.4f}" if 'speed_magnitude_error_rmse' in r else "N/A"
        vvr = f"{r.get('velocity_vector_error_rmse', 0):.4f}" if 'velocity_vector_error_rmse' in r else "N/A"
        sid_short = r.get('sample_id', '?').replace('S001_P001_', '')
        rpt.append(f"| {sid_short} | {r['component']} | {r['gap_length']} | {r['n_windows']} | {r['n_masked_points']} | "
                   f"{r['recon_err_mean_mm']:.2f} | {r['recon_err_p95_mm']:.2f} | {r['recon_err_max_mm']:.2f} | "
                   f"{smr} | {vvr} |")

    rpt += ["", "## Masking Experiment (weighted aggregate)", "",
            "| Comp | Gap | Windows | Points | Mean(mm) | P95(mm) | Max(mm) |",
            "|------|-----|---------|--------|----------|---------|---------|"]
    for r in masking_agg:
        rpt.append(f"| {r['component']} | {r['gap_length']} | {r['n_windows']} | {r['n_masked_points']} | "
                   f"{r['recon_err_mean_mm']:.2f} | {r['recon_err_p95_mm']:.2f} | {r['recon_err_max_mm']:.2f} |")

    rpt += ["", "## Selected Defaults", "",
            f"- **Normalization transform**: `{SELECTED_NORM_TRANSFORM}`",
            f"- **Smoothing**: `{SMOOTHING_DEFAULT}`",
            f"- **Smoothing conclusion**: `{SMOOTHING_CONCLUSION}`",
            f"- **SavGol**: `{SMOOTHING_SAVGOL_NOTE}`",
            f"- **Max gap**: `{SELECTED_MAX_GAP}`", ""]
    with open(out_dir / f"STEP7A_V{VERSION.replace('.','_')}_REPORT.md", "w", encoding="utf-8") as f: f.write("\n".join(rpt))


# ================================================================
#  Output Manifest with metadata (v1.4)
# ================================================================

def classify_artifact(rel_path):
    """Classify an output file and extract metadata."""
    meta = {"artifact_type": "unknown"}
    p = rel_path.replace("\\", "/")

    if p.startswith("npz/interp_"):
        meta["artifact_type"] = "interpolation_npz"
        parts = p.replace("npz/interp_", "").replace(".npz", "").split("_g")
        if len(parts) == 2:
            meta["sample_id"] = "S001_P001_" + parts[0]
            meta["max_gap"] = int(parts[1])
    elif p.startswith("npz/smooth_"):
        meta["artifact_type"] = "smoothing_npz"
        inner = p.replace("npz/smooth_", "").replace(".npz", "")
        # Find filter name (last part after sample)
        # Pattern: smooth_{sample}_{filter}.npz
        # sample can have underscores, filter is after last known pattern
        for filt_name in ["none", "savgol"] + [f"oe_{c['name']}" for c in ONE_EURO_CANDIDATES]:
            if inner.endswith(f"_{filt_name}"):
                meta["sample_id"] = "S001_P001_" + inner[:-(len(filt_name) + 1)]
                meta["filter"] = filt_name
                break
    elif p.startswith("npz/norm_"):
        meta["artifact_type"] = "normalization_npz"
        meta["sample_id"] = "S001_P001_" + p.replace("npz/norm_", "").replace(".npz", "")
    elif p.startswith("provenance/interp_provenance_"):
        meta["artifact_type"] = "interpolation_provenance"
        parts = p.replace("provenance/interp_provenance_", "").replace(".json", "").split("_g")
        if len(parts) == 2:
            meta["sample_id"] = "S001_P001_" + parts[0]
            meta["max_gap"] = int(parts[1])
    elif p.startswith("provenance/smooth_schema_"):
        meta["artifact_type"] = "smoothing_schema"
        inner = p.replace("provenance/smooth_schema_", "").replace(".json", "")
        for filt_name in ["none", "savgol"] + [f"oe_{c['name']}" for c in ONE_EURO_CANDIDATES]:
            if inner.endswith(f"_{filt_name}"):
                meta["sample_id"] = "S001_P001_" + inner[:-(len(filt_name) + 1)]
                meta["filter"] = filt_name
                break
    elif p.startswith("sequences/full_"):
        meta["artifact_type"] = "full_sequence_json"
        meta["sample_id"] = "S001_P001_" + p.replace("sequences/full_", "").replace(".json", "")
    elif p.startswith("contact_sheets/"):
        meta["artifact_type"] = "segmentation_contact_sheet"
        meta["sample_id"] = "S001_P001_" + p.replace("contact_sheets/seg_contact_", "").replace(".png", "")
    elif p.startswith("visualizations/"):
        meta["artifact_type"] = "segmentation_visualization"
        meta["sample_id"] = "S001_P001_" + p.replace("visualizations/seg_", "").replace(".png", "")
    elif p == f"step7a_v{VERSION.replace('.','_')}_config.json":
        meta["artifact_type"] = "execution_config"
    elif p == "input_file_manifest.json":
        meta["artifact_type"] = "input_manifest"
    elif p == "run_manifest.json":
        meta["artifact_type"] = "run_manifest"
    elif p == "assertion_results.json":
        meta["artifact_type"] = "assertion_results"
    elif p.endswith("_REPORT.md"):
        meta["artifact_type"] = "report"
    elif p == "step7_dataset_split_manifest.csv":
        meta["artifact_type"] = "dataset_split_csv"
    elif p == "step7_dataset_split_manifest.json":
        meta["artifact_type"] = "dataset_split_json"
    elif p == "pilot_selection_manifest.json":
        meta["artifact_type"] = "pilot_selection"
    elif p == "strong_depth_provenance.json":
        meta["artifact_type"] = "strong_depth_provenance"
    elif p == "interpolation_comparison.csv":
        meta["artifact_type"] = "interpolation_comparison"
    elif p == "smoothing_comparison.csv":
        meta["artifact_type"] = "smoothing_comparison"
    elif p == "segmentation_summary.csv":
        meta["artifact_type"] = "segmentation_summary"
    elif p == "normalization_summary.csv":
        meta["artifact_type"] = "normalization_summary"
    elif p == "masking_experiment.csv":
        meta["artifact_type"] = "masking_experiment"
    elif p == "transform_comparison.csv":
        meta["artifact_type"] = "transform_comparison"
    elif p == "transform_comparison_detail.json":
        meta["artifact_type"] = "transform_comparison_detail"
    elif p == "peak_lag_detail.json":
        meta["artifact_type"] = "peak_lag_detail"
    elif "step7a_preprocess_pilot" in p:
        meta["artifact_type"] = "execution_script"
    return meta


# ================================================================
#  Main
# ================================================================

def main():
    sp(f"Step 7A v{VERSION}: Preprocessing Pilot (packaging & evaluation fix)")
    sp(f"  Status: STEP7A_V{VERSION.replace('.','_')}_REVISION_REQUIRED\n")
    assert STEP6B_DIR.exists()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sd in ["sequences", "contact_sheets", "visualizations", "npz", "provenance"]:
        (OUT_DIR / sd).mkdir(exist_ok=True)

    # 1. Split & pilot (unchanged)
    sp("  1. Dataset split..."); split = build_dataset_split()
    split_rows = sorted(split.values(), key=lambda x: x["sample_id"])
    with open(OUT_DIR / "step7_dataset_split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["sample_id","word","split","position_in_word"]); w.writeheader(); w.writerows(split_rows)
    with open(OUT_DIR / "step7_dataset_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"samples": split_rows, "counts": {"train": 27, "val": 9, "test": 9}}, f, indent=2)

    sp("  2. Pilot selection..."); pilot = select_pilots(split); pilot_ids = sorted(pilot.keys())
    for sid in pilot_ids:
        p = pilot[sid]; sp(f"    {sid.replace('S001_P001_',''):30s} tier={p['tier']:4s} te={p['te_rate']:.1%}")
    with open(OUT_DIR / "pilot_selection_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"pilot_samples": [pilot[s] for s in pilot_ids]}, f, indent=2, default=str)

    # 2. Config (saved FIRST for manifest inclusion)
    config_name = f"step7a_v{VERSION.replace('.','_')}_config.json"
    config = {"step": "7A", "version": VERSION, "input_dir": str(STEP6B_DIR.relative_to(PROJECT)),
              "segmentation": {"energy_window": SEG_ENERGY_WINDOW, "idle_threshold": SEG_IDLE_THRESHOLD,
                                "pad_start": SEG_PAD_START, "pad_end": SEG_PAD_END, "min_active": SEG_MIN_ACTIVE},
              "interpolation_candidates": INTERP_CANDIDATES, "one_euro_candidates": ONE_EURO_CANDIDATES,
              "savgol": {"window": SAVGOL_WINDOW, "order": SAVGOL_ORDER},
              "normalization": {"origin": "shoulder_midpoint", "scale": "shoulder_width", "axes": "right_up_forward"},
              "reset_reason_codes": RSN_CODE_MAP,
              "smoothing_recommendation": SMOOTHING_CONCLUSION,
              "smoothing_default": SMOOTHING_DEFAULT,
              "selected_normalization_transform": SELECTED_NORM_TRANSFORM,
              "selected_max_gap": SELECTED_MAX_GAP}
    with open(OUT_DIR / config_name, "w", encoding="utf-8") as f: json.dump(config, f, indent=2)

    # 3. Input manifest (includes config + script)
    sp("\n  3. Input hashes..."); input_hashes = {}; input_manifest = []
    for sid in pilot_ids:
        short = sid.replace("S001_P001_", ""); p = RAW3D_DIR / f"landmarks_3d_{short}.json"
        h = sha256_file(p); input_hashes[sid] = h
        input_manifest.append({"sample_id": sid, "file": p.relative_to(PROJECT).as_posix(),
                                "size_bytes": os.path.getsize(p), "sha256": h, "artifact_type": "raw_3d_json"})
    if TA_CSV.exists():
        input_manifest.append({"file": TA_CSV.relative_to(PROJECT).as_posix(), "size_bytes": os.path.getsize(TA_CSV),
                                "sha256": sha256_file(TA_CSV), "artifact_type": "temporal_anomalies_csv"})
    for sid in pilot_ids:
        short = sid.replace("S001_P001_", ""); mp4 = OVERLAY_DIR / f"reproj_{short}_front.mp4"
        if mp4.exists():
            input_manifest.append({"sample_id": sid, "file": mp4.relative_to(PROJECT).as_posix(),
                                    "size_bytes": os.path.getsize(mp4), "sha256": sha256_file(mp4), "artifact_type": "overlay_mp4_front"})
    input_manifest.append({"file": Path(__file__).relative_to(PROJECT).as_posix(), "size_bytes": os.path.getsize(__file__),
                            "sha256": sha256_file(__file__), "artifact_type": "execution_script"})
    cfg_path = OUT_DIR / config_name
    input_manifest.append({"file": cfg_path.relative_to(PROJECT).as_posix(), "size_bytes": os.path.getsize(cfg_path),
                            "sha256": sha256_file(cfg_path), "artifact_type": "execution_config"})
    with open(OUT_DIR / "input_file_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"self_reference_excluded": True, "inputs": input_manifest}, f, indent=2)

    # 4. Strong-depth
    sp("  4. Loading strong-depth boundaries...")
    strong_depth, sd_prov_records = load_strong_depth_data(set(pilot_ids))
    sp(f"    Total: {sum(len(v) for v in strong_depth.values())}")
    sd_prov_map = {}
    for rec in sd_prov_records: sd_prov_map.setdefault(rec["sample_id"], {})[(rec["frame_index"], rec["component"], rec["landmark_index"])] = rec
    with open(OUT_DIR / "strong_depth_provenance.json", "w", encoding="utf-8") as f: json.dump(sd_prov_records, f, indent=2)

    # 5. Process samples
    sp("\n  5. Processing pilot samples...")
    all_results = {}; seg_rows, norm_rows, interp_rows, smooth_rows = [], [], [], []
    masking_all, transform_rows, transform_detail = [], [], []
    peak_lag_details = {}
    boundary_reasons_all = {}

    for sid in pilot_ids:
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
        norm_rows.append({"sample_id": sid, "n_frames": N, "n_ok": sc.get("ok", 0),
                          "scale_mean_m": float(np.mean(vs)) if len(vs) else None})

        stab = compute_causal_stabilized_transform(norm, ts)

        seg = detect_segment(arrs, norm)
        sp(f"      Seg: [{seg['seg_start']}-{seg['seg_end']}] ({seg['seg_frames']}/{N}f) idle_end={seg['idle_verified_end']}")
        seg_rows.append({"sample_id": sid, "total_frames": N, "seg_start": seg["seg_start"], "seg_end": seg["seg_end"],
                          "seg_frames": seg["seg_frames"], "method": seg["method"],
                          "idle_verified_start": seg["idle_verified_start"], "idle_verified_end": seg["idle_verified_end"]})
        make_seg_contact_sheet(sid, seg, OUT_DIR / "contact_sheets")

        # Transform comparison (v1.4: with seg_info)
        tc = compare_transforms(arrs, norm, stab, ts, seg)
        transform_rows.append({"sample_id": sid, **tc["aggregate"]})
        transform_detail.append({"sample_id": sid, **tc})
        sp(f"      Transform: traj_dev_mean={tc['aggregate']['traj_dev_mean_mm']:.2f}mm")

        frame_disc, boundaries, boundary_reasons, strong_forbidden = build_all_boundaries(arrs, sd_set, sid)
        boundary_reasons_all[sid] = boundary_reasons

        # Interpolation
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
                ni = normalize_metric_coords(out_xyz, valid_sm, norm)
                norm_interp_cache[f"norm_interp_{comp}_g{mg}"] = ni
                interp_rows.append({"sample_id": sid, "component": comp, "max_gap": mg,
                                    "n_filled": int(np.sum(mask)), "n_rejected": int(np.sum(rej)),
                                    "n_strong_depth_interp": int(np.sum(mask & strong_forbidden[comp]))})
            interp_data[mg] = ir
            sp(f"        g={mg}: pose={int(np.sum(ir['pose_mask']))} lh={int(np.sum(ir['lh_mask']))} rh={int(np.sum(ir['rh_mask']))}")

        # Masking experiment (v1.4: with sid)
        sp(f"      Masking experiment...")
        me = run_masking_experiment(arrs, norm, boundaries, strong_forbidden, ts, sid)
        masking_all.extend(me)

        # Smoothing
        sp(f"      Smoothing sweep...")
        base = interp_data[2]; smooth_data = {}

        # No-smoothing
        ns = {}
        for comp in ["pose", "lh", "rh"]:
            valid_sm = arrs[f"{comp}_te"] | base[f"{comp}_mask"]
            ni = norm_interp_cache[f"norm_interp_{comp}_g2"]
            rm, far, rsn = compute_filter_masks(base[f"{comp}_interp"], valid_sm, base[f"{comp}_boundary"], boundary_reasons[comp])
            ns[f"{comp}_filtered"] = base[f"{comp}_interp"].copy()
            ns[f"{comp}_norm_filtered"] = ni.copy()
            ns[f"{comp}_filter_valid"] = valid_sm
            ns[f"{comp}_reset_mask"] = rm; ns[f"{comp}_first_after_reset"] = far; ns[f"{comp}_reset_reason"] = rsn
            ns[f"{comp}_metrics"] = {"label": "none", "n_valid": int(np.sum(valid_sm)),
                "phys_dev_mean_mm": 0.0, "phys_dev_p95_mm": 0.0, "phys_dev_max_mm": 0.0,
                "vel_distortion_pct": 0.0, "vel_rmse_m_per_s": 0.0, "accel_rmse_m_per_s2": 0.0,
                "path_length_ratio": 1.0, "new_nans": 0, "new_infs": 0,
                "jitter_raw_start_idle_mm": "not_available_no_verified_idle",
                "jitter_filt_start_idle_mm": "not_available_no_verified_idle",
                "jitter_reduction_start_idle": "not_available_no_verified_idle",
                "jitter_raw_end_idle_mm": "not_available_no_verified_idle",
                "jitter_filt_end_idle_mm": "not_available_no_verified_idle",
                "jitter_reduction_end_idle": "not_available_no_verified_idle",
                "peak_vel_lag_frames": 0, "peak_vel_lag_ms": 0, "peak_match_status": "identity",
                "per_landmark": [], "peak_lag_detail": {"status": "identity"}}
            smooth_rows.append({"sample_id": sid, "component": comp, "filter": "none", **ns[f"{comp}_metrics"]})
        smooth_data["none"] = ns

        # OneEuro
        for oe_params in ONE_EURO_CANDIDATES:
            oe = {}
            for comp in ["pose", "lh", "rh"]:
                valid_sm = arrs[f"{comp}_te"] | base[f"{comp}_mask"]
                bd = base[f"{comp}_boundary"]; br = boundary_reasons[comp]
                filtered, rm, far, rsn = apply_one_euro(base[f"{comp}_interp"], valid_sm, ts, bd, br, oe_params)
                norm_filt = normalize_metric_coords(filtered, valid_sm, norm)
                norm_base = norm_interp_cache[f"norm_interp_{comp}_g2"]
                oe[f"{comp}_filtered"] = filtered; oe[f"{comp}_norm_filtered"] = norm_filt
                oe[f"{comp}_filter_valid"] = valid_sm
                oe[f"{comp}_reset_mask"] = rm; oe[f"{comp}_first_after_reset"] = far; oe[f"{comp}_reset_reason"] = rsn
                oe[f"{comp}_metrics"] = compute_extended_metrics(
                    norm_base, norm_filt, valid_sm, norm["scales"], ts, bd, seg,
                    f"oe_{oe_params['name']}", word, comp_nlm=(comp, {("pose",33),("lh",21),("rh",21)}))
                if comp == "pose":
                    oe[f"{comp}_sign_metrics"] = compute_sign_metrics(
                        norm_base, norm_filt, valid_sm, arrs, norm, seg, word, ts, norm["scales"], bd)
                smooth_rows.append({"sample_id": sid, "component": comp,
                                     "filter": f"one_euro_{oe_params['name']}", **oe[f"{comp}_metrics"]})
                # Collect peak lag detail
                pld = oe[f"{comp}_metrics"].get("peak_lag_detail", {})
                if pld.get("per_landmark"):
                    peak_lag_details.setdefault(sid, {})[f"oe_{oe_params['name']}_{comp}"] = pld
            smooth_data[f"oe_{oe_params['name']}"] = oe
            pm = oe["pose_metrics"]
            sp(f"        {oe_params['name']}: dev={pm['phys_dev_mean_mm']:.1f}mm path={pm['path_length_ratio']:.3f} "
               f"far={sum(int(np.sum(oe[f'{c}_first_after_reset'])) for c in ['pose','lh','rh'])}")

        # SavGol (v1.4: overwrite first_after_reset positions with original)
        if HAS_SCIPY:
            sg = {}
            for comp in ["pose", "lh", "rh"]:
                valid_sm = arrs[f"{comp}_te"] | base[f"{comp}_mask"]
                bd = base[f"{comp}_boundary"]; br = boundary_reasons[comp]
                filtered = apply_savgol(base[f"{comp}_interp"], valid_sm, bd)
                rm, far, rsn = compute_filter_masks(base[f"{comp}_interp"], valid_sm, bd, br)
                # v1.4: overwrite first_after_reset positions with original input
                base_interp = base[f"{comp}_interp"]
                for fi in range(far.shape[0]):
                    for li in range(far.shape[1]):
                        if far[fi, li] and not np.any(np.isnan(base_interp[fi, li])):
                            filtered[fi, li] = base_interp[fi, li].copy()
                norm_filt = normalize_metric_coords(filtered, valid_sm, norm)
                norm_base = norm_interp_cache[f"norm_interp_{comp}_g2"]
                # Also overwrite norm_filt at first_after_reset with norm_base
                for fi in range(far.shape[0]):
                    for li in range(far.shape[1]):
                        if far[fi, li] and not np.any(np.isnan(norm_base[fi, li])):
                            norm_filt[fi, li] = norm_base[fi, li].copy()
                sg[f"{comp}_filtered"] = filtered; sg[f"{comp}_norm_filtered"] = norm_filt
                sg[f"{comp}_filter_valid"] = valid_sm
                sg[f"{comp}_reset_mask"] = rm; sg[f"{comp}_first_after_reset"] = far; sg[f"{comp}_reset_reason"] = rsn
                sg[f"{comp}_metrics"] = compute_extended_metrics(
                    norm_base, norm_filt, valid_sm, norm["scales"], ts, bd, seg, "savgol", word, comp_nlm=(comp, None))
                smooth_rows.append({"sample_id": sid, "component": comp, "filter": "savgol", **sg[f"{comp}_metrics"]})
            smooth_data["savgol"] = sg

        # Save NPZ
        save_sample_npz(sid, arrs, norm, interp_data, smooth_data, norm_interp_cache, stab, OUT_DIR)
        npz_fails = verify_sample_npz(sid, arrs, norm, interp_data, smooth_data, norm_interp_cache, stab, OUT_DIR)
        sp(f"      NPZ: {'PASS' if not npz_fails else 'FAIL ' + str(npz_fails[:3])}")

        if HAS_MPL: plot_seg(sid, arrs, norm, seg, OUT_DIR / "visualizations")
        save_full_sequence(sid, arrs, norm, seg, OUT_DIR)

        all_results[sid] = {"arrs": arrs, "raw_data": raw_data, "norm": norm, "seg": seg,
                            "interp": interp_data, "smooth": smooth_data, "npz_verify_fails": npz_fails}

    # 6. CSVs
    sp("\n  6. Saving CSVs...")
    masking_agg = compute_masking_weighted_aggregate(masking_all)
    _write_csv(OUT_DIR / "segmentation_summary.csv", seg_rows)
    _write_csv(OUT_DIR / "normalization_summary.csv", norm_rows)
    _write_csv(OUT_DIR / "interpolation_comparison.csv", interp_rows)
    _write_csv_flexible(OUT_DIR / "smoothing_comparison.csv", smooth_rows)
    _write_csv_flexible(OUT_DIR / "masking_experiment.csv", masking_all + masking_agg)

    # Transform comparison: per-landmark CSV and full JSON detail
    tc_csv_rows = []
    for td in transform_detail:
        sid_val = td["sample_id"]
        for lm in td["per_landmark"]:
            tc_csv_rows.append({"sample_id": sid_val, **lm})
        tc_csv_rows.append({"sample_id": sid_val, "component": "AGGREGATE", "landmark": "",
                            **td["aggregate"], **td.get("idle_jitter", {})})
    _write_csv_flexible(OUT_DIR / "transform_comparison.csv", tc_csv_rows)
    with open(OUT_DIR / "transform_comparison_detail.json", "w", encoding="utf-8") as f:
        json.dump(transform_detail, f, indent=2, default=str)
    with open(OUT_DIR / "peak_lag_detail.json", "w", encoding="utf-8") as f:
        json.dump(peak_lag_details, f, indent=2, default=str)

    # ============================================================
    # v1.4 PACKAGING ORDER: assertions BEFORE ZIP
    # ============================================================

    # 7. Write placeholder assertion_results.json (so assertion can check its existence)
    sp("\n  7. Running assertions...")
    with open(OUT_DIR / "assertion_results.json", "w", encoding="utf-8") as f:
        json.dump([], f)

    # 8. Run manifest placeholder (needed for assertion checks)
    run_manifest = {"step": "7A", "version": VERSION,
        "prior_status": ["STEP6B_COMPLETE", "STEP7A_V1_3_EXTERNAL_REVIEW_PARTIAL_PASS"],
        "pilot_samples": pilot_ids, "dataset_split": {"train": 27, "val": 9, "test": 9},
        "smoothing_recommendation": SMOOTHING_CONCLUSION,
        "smoothing_default": SMOOTHING_DEFAULT,
        "selected_normalization_transform": SELECTED_NORM_TRANSFORM,
        "selected_max_gap": SELECTED_MAX_GAP,
        "self_reference_excluded": True}
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f: json.dump(run_manifest, f, indent=2)

    # 9. Run assertions
    ar, np_count, nf_count = run_assertions(all_results, split, pilot_ids, input_hashes, strong_depth,
                                             OUT_DIR, boundary_reasons_all)

    # 10. Save FINAL assertion_results.json
    with open(OUT_DIR / "assertion_results.json", "w", encoding="utf-8") as f:
        json.dump(ar, f, indent=2)

    # 11. Save FINAL run manifest with assertion counts + status
    final_status = f"STEP7A_PREPROCESSING_PILOT_V{VERSION.replace('.','_')}_COMPLETE" if nf_count == 0 else f"STEP7A_V{VERSION.replace('.','_')}_BLOCKED"
    run_manifest["assertions"] = {"pass": np_count, "fail": nf_count}
    run_manifest["status"] = final_status
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f: json.dump(run_manifest, f, indent=2)

    # 12. Save FINAL report with assertion counts
    write_report(OUT_DIR, pilot_ids, pilot, seg_rows, interp_rows, smooth_rows,
                 masking_all, masking_agg, transform_rows, np_count, nf_count)

    # 13. Output manifest (LAST, after all files finalized)
    sp("  8. Output manifest...")
    script_arc = f"s001_step7a_preprocess_pilot_v{VERSION.replace('.','_')}.py"
    output_manifest = []
    for root, _, fnames in os.walk(OUT_DIR):
        for fn in fnames:
            fp = Path(root) / fn; rel = fp.relative_to(OUT_DIR).as_posix()
            if rel == "output_file_manifest.json": continue
            meta = classify_artifact(rel)
            entry = {"relative_path": rel, "size_bytes": os.path.getsize(fp),
                     "sha256": sha256_file(fp), "schema_version": VERSION_TAG}
            # Add metadata fields where applicable
            for k in ["artifact_type", "sample_id", "component", "filter", "max_gap"]:
                if k in meta and meta[k] != "unknown":
                    entry[k] = meta[k]
            output_manifest.append(entry)
    # Add script entry (will be in ZIP)
    output_manifest.append({"relative_path": script_arc, "size_bytes": os.path.getsize(__file__),
                             "sha256": sha256_file(__file__), "schema_version": VERSION_TAG,
                             "artifact_type": "execution_script"})
    with open(OUT_DIR / "output_file_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"self_reference_excluded": True, "schema_version": VERSION_TAG,
                    "outputs": output_manifest, "total_files": len(output_manifest)}, f, indent=2)

    # 14. ZIP
    sp("  9. ZIP...")
    zip_path = PROJECT / f"WordsInSilence_step7a_v{VERSION.replace('.','_')}_pilot_delivery.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for root, _, fnames in os.walk(OUT_DIR):
            for fn in fnames:
                fp = Path(root) / fn; arc = fp.relative_to(OUT_DIR).as_posix()
                comp = zipfile.ZIP_STORED if fp.suffix.lower() in (".png", ".npz") else zipfile.ZIP_DEFLATED
                zf.write(fp, arc, compress_type=comp)
        zf.write(Path(__file__), script_arc, compress_type=zipfile.ZIP_DEFLATED)

    # 15. Post-ZIP verification (no file updates after this)
    sp("  10. ZIP verification...")
    verify_ok = True
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_entries = set(zi.filename for zi in zf.infolist())
        # CRC check
        bad_crc = zf.testzip()
        if bad_crc:
            sp(f"    CRC FAIL: {bad_crc}"); verify_ok = False
    with open(OUT_DIR / "output_file_manifest.json", encoding="utf-8") as f:
        om = json.load(f)
    manifest_entries = set(e["relative_path"] for e in om["outputs"])
    zip_minus_self = zip_entries - {"output_file_manifest.json"}
    if manifest_entries != zip_minus_self:
        sp(f"    ZIP/manifest MISMATCH: only_manifest={manifest_entries - zip_minus_self} only_zip={zip_minus_self - manifest_entries}")
        verify_ok = False
    else:
        sp(f"    ZIP entries match manifest ({len(zip_entries)} entries)")

    # Verify output manifest hashes match actual files
    hash_mismatches = 0
    for entry in om["outputs"]:
        rel = entry["relative_path"]
        if rel == script_arc:
            actual_hash = sha256_file(Path(__file__))
        else:
            fp = OUT_DIR / rel
            if not fp.exists(): hash_mismatches += 1; continue
            actual_hash = sha256_file(fp)
        if actual_hash != entry["sha256"]:
            hash_mismatches += 1
    if hash_mismatches > 0:
        sp(f"    Output manifest hash mismatches: {hash_mismatches}"); verify_ok = False
    else:
        sp(f"    Output manifest hashes: all match")

    # Verify assertion counts match across files
    with open(OUT_DIR / "assertion_results.json", encoding="utf-8") as f:
        ar_data = json.load(f)
    ar_pass = sum(1 for r in ar_data if r["status"] == "PASS")
    ar_fail = sum(1 for r in ar_data if r["status"] == "FAIL")
    with open(OUT_DIR / "run_manifest.json", encoding="utf-8") as f:
        rm_data = json.load(f)
    rm_pass = rm_data.get("assertions", {}).get("pass", -1)
    rm_fail = rm_data.get("assertions", {}).get("fail", -1)
    if ar_pass != rm_pass or ar_fail != rm_fail:
        sp(f"    Count mismatch: assertion_results={ar_pass}/{ar_fail} run_manifest={rm_pass}/{rm_fail}")
        verify_ok = False

    zs = os.path.getsize(zip_path); zh = sha256_file(zip_path)
    sp(f"\n    ZIP: {zip_path.name} ({zs / 1e6:.1f} MB)")
    sp(f"    SHA-256: {zh}")
    sp(f"    Verification: {'PASS' if verify_ok else 'FAIL'}")
    sp(f"\n  ============================================================")
    sp(f"  STEP 7A v{VERSION} SUMMARY")
    sp(f"  ============================================================")
    sp(f"  Assertions: {np_count} PASS / {nf_count} FAIL")
    sp(f"  Post-ZIP verification: {'PASS' if verify_ok else 'FAIL'}")
    sp(f"  Output: {OUT_DIR}")
    if nf_count == 0 and verify_ok:
        sp(f"  Status: STEP6B_COMPLETE")
        sp(f"  Status: STEP7A_PREPROCESSING_PILOT_V{VERSION.replace('.','_')}_COMPLETE")
        sp(f"  Status: STEP7A_V{VERSION.replace('.','_')}_BLOCKED_PENDING_EXTERNAL_REVIEW")
    else:
        sp(f"  Status: STEP7A_V{VERSION.replace('.','_')}_BLOCKED")


if __name__ == "__main__":
    main()
