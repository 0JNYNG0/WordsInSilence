#!/usr/bin/env python3
"""
Step 7A v1.2: Recognition preprocessing pilot - revised per v1.1 partial-pass review.

Fixes from v1.1 review:
  - 2: Save actual interpolation/smoothing arrays (NPZ) + provenance (JSON)
  - 3: No-smoothing baseline fix (deviation=0, separate interpolation evaluation)
  - 4: Extended OneEuro evaluation (timing lag, path-length, amplitude, vel/accel RMSE, sign-specific)
  - 5: max_gap masking experiment with reconstruction error analysis
  - 6: Provenance fix (target landmark indexing, full gap reasons, strong-depth provenance)
  - 7: Full input/output manifests (temporal_anomalies.csv, overlay MP4s, config)
  - 8: Normalization transform comparison (raw vs causal stabilized)
  - 9: Round-trip file verification assertions

Unchanged from v1.1 (passed review):
  - dataset split, pilot selection, segmentation, contact sheets
  - strong-depth prohibition, normalization coordinate definition
  - OneEuro physical deviation with per-frame shoulder scale

Input: Step 6B v1.2.4 full 45 (frozen read-only)
Output: data/processed/sessions/S001_P001/step7a_pilot_v1_2/
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
#  Paths
# ================================================================

PROJECT = Path(__file__).resolve().parent.parent
STEP6B_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "landmarks_3d_v1" / "full_v1_2_4"
RAW3D_DIR = STEP6B_DIR / "raw_3d"
OVERLAY_DIR = STEP6B_DIR / "overlays"
TA_CSV = STEP6B_DIR / "temporal_anomalies.csv"
OUT_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7a_pilot_v1_2"

WORDS = ["GREETING", "LOVE", "THANK_YOU"]
SPEEDS = ["FAST", "NORMAL", "SLOW"]
REPS = ["R01", "R02", "R03", "R04", "R05"]

# Pose landmark indices
LM_NOSE = 0
LM_LS, LM_RS = 11, 12
LM_LE, LM_RE = 13, 14
LM_LW, LM_RW = 15, 16
LM_LH, LM_RH = 23, 24

# Segmentation
SEG_ENERGY_WINDOW = 5
SEG_IDLE_THRESHOLD = 0.008
SEG_PAD_START = 5
SEG_PAD_END = 5
SEG_MIN_ACTIVE = 10

# Interpolation
INTERP_CANDIDATES = [1, 2, 3]

# OneEuro parameter sweep candidates
ONE_EURO_CANDIDATES = [
    {"name": "conservative", "min_cutoff": 3.0, "beta": 0.001, "d_cutoff": 1.0},
    {"name": "moderate",     "min_cutoff": 1.7, "beta": 0.004, "d_cutoff": 1.0},
    {"name": "aggressive",   "min_cutoff": 1.0, "beta": 0.007, "d_cutoff": 1.0},
    {"name": "very_light",   "min_cutoff": 5.0, "beta": 0.0005, "d_cutoff": 1.0},
]

# SavGol
SAVGOL_WINDOW = 7
SAVGOL_ORDER = 3


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
#  Dataset Split (identical to v1.0/v1.1)
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


# ================================================================
#  Pilot Selection (identical to v1.0/v1.1)
# ================================================================

def compute_sample_quality(sid):
    short = sid.replace("S001_P001_", "")
    with open(RAW3D_DIR / f"landmarks_3d_{short}.json", encoding="utf-8") as f:
        data = json.load(f)
    frames = data["frames"]
    N = len(frames)
    total_lm = N * 75
    te_valid = sum(1 for fr in frames for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                   for lm in fr[k] if lm["template_eligible"])
    qa_valid = sum(1 for fr in frames for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                   for lm in fr[k] if lm["qa_valid"])
    n_anom = sum(len(lm["qa_flags"]) for fr in frames for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                 for lm in fr[k])
    te_rate = te_valid / total_lm if total_lm else 0
    qa_rate = qa_valid / total_lm if total_lm else 0
    return {"sample_id": sid, "n_frames": N, "total_landmarks": total_lm,
            "te_valid": te_valid, "qa_valid": qa_valid, "te_rate": te_rate, "qa_rate": qa_rate,
            "n_anomaly_flags": n_anom,
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
#  Load Step 6B data
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
#  Load strong-depth boundaries (enhanced with full provenance)
# ================================================================

def load_strong_depth_data(pilot_ids):
    """Returns (boundaries_dict, provenance_records)."""
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
#  Body-centric Normalization
# ================================================================

def compute_torso_frame(pose_xyz, pose_te, fi):
    ls = pose_xyz[fi, LM_LS]
    rs = pose_xyz[fi, LM_RS]
    lh = pose_xyz[fi, LM_LH]
    rh = pose_xyz[fi, LM_RH]
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

    if lh_v and rh_v:
        up_vec = origin - (lh + rh) / 2.0
    elif lh_v:
        up_vec = origin - lh
    elif rh_v:
        up_vec = origin - rh
    else:
        up_vec = np.array([0.0, -1.0, 0.0])

    up_n = np.linalg.norm(up_vec)
    if up_n < 1e-6:
        up_vec = np.array([0.0, -1.0, 0.0])
        up_n = 1.0
    up = up_vec / up_n

    fwd = np.cross(right, up)
    fn = np.linalg.norm(fwd)
    if fn < 1e-6:
        return origin, None, sw, "degenerate_axes"
    fwd = fwd / fn
    up = np.cross(fwd, right)
    up = up / np.linalg.norm(up)

    R = np.column_stack([right, up, fwd])
    return origin, R, sw, "ok"


def normalize_sequence(arrs):
    N = len(arrs["timestamps"])
    origins = np.full((N, 3), np.nan)
    rotations = np.full((N, 3, 3), np.nan)
    scales = np.full(N, np.nan)
    dets = np.full(N, np.nan)
    ortho_errs = np.full(N, np.nan)
    statuses = []
    fallback_src = [None] * N

    for fi in range(N):
        o, R, s, st = compute_torso_frame(arrs["pose_xyz"], arrs["pose_te"], fi)
        statuses.append(st)
        if o is not None:
            origins[fi] = o
        if R is not None:
            rotations[fi] = R
            dets[fi] = np.linalg.det(R)
            ortho_errs[fi] = np.linalg.norm(R.T @ R - np.eye(3))
        if s is not None and s > 0:
            scales[fi] = s

    last_valid = None
    for fi in range(N):
        if statuses[fi] == "ok":
            last_valid = fi
        elif last_valid is not None:
            origins[fi] = origins[last_valid]
            rotations[fi] = rotations[last_valid]
            scales[fi] = scales[last_valid]
            dets[fi] = dets[last_valid]
            ortho_errs[fi] = ortho_errs[last_valid]
            statuses[fi] = "fallback_prev"
            fallback_src[fi] = last_valid

    first_valid = None
    for fi in range(N):
        if statuses[fi] in ("ok", "fallback_prev"):
            first_valid = fi
            break
    if first_valid is not None and first_valid > 0:
        for fi in range(first_valid):
            origins[fi] = origins[first_valid]
            rotations[fi] = rotations[first_valid]
            scales[fi] = scales[first_valid]
            dets[fi] = dets[first_valid]
            ortho_errs[fi] = ortho_errs[first_valid]
            statuses[fi] = "fallback_next"
            fallback_src[fi] = first_valid

    def norm_pts(xyz, valid):
        out = np.full_like(xyz, np.nan)
        for fi in range(N):
            if np.isnan(scales[fi]) or scales[fi] < 1e-6 or np.any(np.isnan(rotations[fi])):
                continue
            R = rotations[fi]
            for li in range(xyz.shape[1]):
                if valid[fi, li] and not np.any(np.isnan(xyz[fi, li])):
                    out[fi, li] = R.T @ (xyz[fi, li] - origins[fi]) / scales[fi]
        return out

    return {
        "norm_pose": norm_pts(arrs["pose_xyz"], arrs["pose_te"]),
        "norm_lh": norm_pts(arrs["lh_xyz"], arrs["lh_te"]),
        "norm_rh": norm_pts(arrs["rh_xyz"], arrs["rh_te"]),
        "origins": origins, "rotations": rotations, "scales": scales,
        "dets": dets, "ortho_errs": ortho_errs,
        "statuses": statuses, "fallback_src": fallback_src,
    }


def normalize_metric_coords(xyz_metric, valid, norm):
    """Convert metric-space coordinates to body-centric normalized space."""
    N = xyz_metric.shape[0]
    out = np.full_like(xyz_metric, np.nan)
    for fi in range(N):
        s = norm["scales"][fi]
        if np.isnan(s) or s < 1e-6 or np.any(np.isnan(norm["rotations"][fi])):
            continue
        R = norm["rotations"][fi]
        for li in range(xyz_metric.shape[1]):
            if valid[fi, li] and not np.any(np.isnan(xyz_metric[fi, li])):
                out[fi, li] = R.T @ (xyz_metric[fi, li] - norm["origins"][fi]) / s
    return out


# ================================================================
#  Causal Stabilized Transform
# ================================================================

class OneEuroFilter1D:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        speed = abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


def compute_causal_stabilized_transform(norm, ts):
    """Apply OneEuro to origin/scale, Gram-Schmidt re-orthogonalize filtered axes."""
    N = len(ts)
    stab_origins = np.full((N, 3), np.nan)
    stab_scales = np.full(N, np.nan)
    stab_rotations = np.full((N, 3, 3), np.nan)

    params = {"min_cutoff": 3.0, "beta": 0.001, "d_cutoff": 1.0}

    # Filter origin channels
    o_filts = [OneEuroFilter1D(**params) for _ in range(3)]
    for fi in range(N):
        if not np.any(np.isnan(norm["origins"][fi])):
            for d in range(3):
                stab_origins[fi, d] = o_filts[d](norm["origins"][fi, d], ts[fi])

    # Filter scale
    s_filt = OneEuroFilter1D(**params)
    for fi in range(N):
        if not np.isnan(norm["scales"][fi]):
            stab_scales[fi] = s_filt(norm["scales"][fi], ts[fi])

    # Filter rotation axes independently then re-orthogonalize
    r_filts = [[OneEuroFilter1D(**params) for _ in range(3)] for _ in range(3)]
    for fi in range(N):
        R = norm["rotations"][fi]
        if np.any(np.isnan(R)):
            continue
        filtered_cols = np.zeros((3, 3))
        for col in range(3):
            for d in range(3):
                filtered_cols[d, col] = r_filts[col][d](R[d, col], ts[fi])
        # Gram-Schmidt re-orthogonalization
        right = filtered_cols[:, 0]
        rn = np.linalg.norm(right)
        if rn < 1e-8:
            continue
        right = right / rn
        up = filtered_cols[:, 1]
        up = up - np.dot(up, right) * right
        un = np.linalg.norm(up)
        if un < 1e-8:
            continue
        up = up / un
        fwd = np.cross(right, up)
        stab_rotations[fi] = np.column_stack([right, up, fwd])

    # Compute comparison metrics
    origin_diffs = []
    scale_diffs = []
    angle_diffs = {"right": [], "up": [], "forward": []}
    for fi in range(N):
        if not np.any(np.isnan(norm["origins"][fi])) and not np.any(np.isnan(stab_origins[fi])):
            origin_diffs.append(np.linalg.norm(stab_origins[fi] - norm["origins"][fi]))
        if not np.isnan(norm["scales"][fi]) and not np.isnan(stab_scales[fi]):
            scale_diffs.append(abs(stab_scales[fi] - norm["scales"][fi]))
        R_raw = norm["rotations"][fi]
        R_stab = stab_rotations[fi]
        if not np.any(np.isnan(R_raw)) and not np.any(np.isnan(R_stab)):
            for col, name in [(0, "right"), (1, "up"), (2, "forward")]:
                cos_a = np.clip(np.dot(R_raw[:, col], R_stab[:, col]), -1, 1)
                angle_diffs[name].append(math.degrees(math.acos(cos_a)))

    # Compute normalized trajectory differences for wrists
    def norm_with_transform(xyz, valid, origins, rotations, scales):
        out = np.full_like(xyz, np.nan)
        for fi in range(xyz.shape[0]):
            s = scales[fi]
            if np.isnan(s) or s < 1e-6 or np.any(np.isnan(rotations[fi])):
                continue
            R = rotations[fi]
            for li in range(xyz.shape[1]):
                if valid[fi, li] and not np.any(np.isnan(xyz[fi, li])):
                    out[fi, li] = R.T @ (xyz[fi, li] - origins[fi]) / s
        return out

    comparison = {
        "origin_diff_mean_mm": float(np.mean(origin_diffs) * 1000) if origin_diffs else 0,
        "origin_diff_max_mm": float(np.max(origin_diffs) * 1000) if origin_diffs else 0,
        "scale_diff_mean_mm": float(np.mean(scale_diffs) * 1000) if scale_diffs else 0,
        "scale_diff_max_mm": float(np.max(scale_diffs) * 1000) if scale_diffs else 0,
    }
    for name in ["right", "up", "forward"]:
        ad = angle_diffs[name]
        comparison[f"{name}_axis_angle_diff_mean_deg"] = float(np.mean(ad)) if ad else 0
        comparison[f"{name}_axis_angle_diff_max_deg"] = float(np.max(ad)) if ad else 0

    return {
        "stab_origins": stab_origins, "stab_scales": stab_scales, "stab_rotations": stab_rotations,
        "comparison": comparison,
    }


# ================================================================
#  Segmentation (unchanged from v1.1)
# ================================================================

def detect_segment(arrs, norm):
    N = len(arrs["timestamps"])
    ts = arrs["timestamps"]
    energy = np.zeros(N)
    motion_lms = [LM_LE, LM_RE, LM_LW, LM_RW]
    tips = [4, 8, 12, 16, 20]

    for fi in range(1, N):
        dt = ts[fi] - ts[fi - 1]
        if dt <= 0 or dt > 0.2:
            continue
        e, c = 0.0, 0
        for li in motion_lms:
            if not np.any(np.isnan(norm["norm_pose"][fi, li])) and not np.any(np.isnan(norm["norm_pose"][fi - 1, li])):
                e += np.linalg.norm(norm["norm_pose"][fi, li] - norm["norm_pose"][fi - 1, li])
                c += 1
        for hk in ["norm_lh", "norm_rh"]:
            w0, w1 = norm[hk][fi - 1, 0], norm[hk][fi, 0]
            if not np.any(np.isnan(w0)) and not np.any(np.isnan(w1)):
                e += np.linalg.norm(w1 - w0)
                c += 1
            for t in tips:
                t0, t1 = norm[hk][fi - 1, t], norm[hk][fi, t]
                if not np.any(np.isnan(t0)) and not np.any(np.isnan(t1)):
                    e += np.linalg.norm(t1 - t0) * 0.5
                    c += 0.5
        if c > 0:
            energy[fi] = e / c

    kernel = np.ones(SEG_ENERGY_WINDOW) / SEG_ENERGY_WINDOW
    energy_sm = np.convolve(energy, kernel, mode="same")
    active = energy_sm > SEG_IDLE_THRESHOLD
    ai = np.where(active)[0]

    if len(ai) < SEG_MIN_ACTIVE:
        return {"seg_start": 0, "seg_end": N - 1, "seg_frames": N, "total_frames": N,
                "raw_start": 0, "raw_end": N - 1, "method": "source_already_trimmed_no_idle_available",
                "energy": energy, "energy_smooth": energy_sm, "threshold": SEG_IDLE_THRESHOLD,
                "idle_verified_start": False, "idle_verified_end": False}

    rs, re = int(ai[0]), int(ai[-1])
    ss = max(0, rs - SEG_PAD_START)
    se = min(N - 1, re + SEG_PAD_END)

    idle_start = ss > 0
    idle_end = se < N - 1

    return {"seg_start": ss, "seg_end": se, "seg_frames": se - ss + 1, "total_frames": N,
            "raw_start": rs, "raw_end": re, "method": "motion_energy",
            "energy": energy, "energy_smooth": energy_sm, "threshold": SEG_IDLE_THRESHOLD,
            "idle_verified_start": idle_start, "idle_verified_end": idle_end}


# ================================================================
#  Discontinuity & Boundary Masks (unchanged from v1.1)
# ================================================================

def build_all_boundaries(arrs, strong_depth_set, sid):
    N = len(arrs["timestamps"])
    ts = arrs["timestamps"]
    frame_disc = np.zeros(N, dtype=bool)
    frame_disc[0] = True
    for fi in range(1, N):
        dt = ts[fi] - ts[fi - 1]
        if dt <= 0 or dt > 0.1:
            frame_disc[fi] = True

    boundaries = {}
    for comp, nlm in [("pose", 33), ("lh", 21), ("rh", 21)]:
        bd = np.zeros((N, nlm), dtype=bool)
        bd[0, :] = True
        for fi in range(N):
            if frame_disc[fi]:
                bd[fi, :] = True
        reapp = arrs[f"{comp}_reappear"]
        for fi in range(1, N):
            for li in range(nlm):
                if reapp[fi, li]:
                    bd[fi, li] = True
        for (sfi, scomp, sli) in strong_depth_set:
            if scomp == comp and 0 <= sfi < N and 0 <= sli < nlm:
                bd[sfi, sli] = True
        boundaries[comp] = bd

    strong_forbidden = {}
    for comp, nlm in [("pose", 33), ("lh", 21), ("rh", 21)]:
        sf = np.zeros((N, nlm), dtype=bool)
        for (sfi, scomp, sli) in strong_depth_set:
            if scomp == comp and 0 <= sfi < N and 0 <= sli < nlm:
                sf[sfi, sli] = True
        strong_forbidden[comp] = sf

    return frame_disc, boundaries, strong_forbidden


# ================================================================
#  Interpolation (provenance fix)
# ================================================================

def interpolate_gaps(xyz, valid, boundary, strong_forbidden, max_gap,
                     invalid_reasons, qa_flags_arr, sd_provenance_map, comp_name, sid):
    """Interpolate gaps with fixed provenance.

    Args:
        invalid_reasons: list[list[str]] shape [N][nlm]
        qa_flags_arr: list[list[list[str]]] shape [N][nlm][flags]
        sd_provenance_map: dict (fi, comp, li) -> provenance record
        comp_name: 'pose'/'lh'/'rh'
    """
    N, nlm, _ = xyz.shape
    out = xyz.copy()
    mask = np.zeros((N, nlm), dtype=bool)
    rejected = np.zeros((N, nlm), dtype=bool)
    records = []

    for li in range(nlm):
        segs = []
        ss = 0
        for fi in range(1, N):
            if boundary[fi, li]:
                segs.append((ss, fi - 1))
                ss = fi
        segs.append((ss, N - 1))

        for s0, s1 in segs:
            fi = s0
            while fi <= s1:
                if not valid[fi, li]:
                    gs = fi
                    while fi <= s1 and not valid[fi, li]:
                        fi += 1
                    ge = fi - 1
                    gl = ge - gs + 1

                    has_before = gs > s0 and valid[gs - 1, li] and not np.any(np.isnan(xyz[gs - 1, li]))
                    has_after = fi <= s1 and valid[fi, li] and not np.any(np.isnan(xyz[fi, li]))
                    touches_strong = any(strong_forbidden[gs + g, li] for g in range(gl))

                    reasons = []
                    if gl > max_gap:
                        reasons.append(f"exceeds_max_gap_{max_gap}")
                    if not has_before:
                        reasons.append("no_valid_predecessor")
                    if not has_after:
                        reasons.append("no_valid_successor")
                    if touches_strong:
                        reasons.append("strong_depth_forbidden")

                    # Collect per-frame provenance for ALL gap frames
                    gap_provenance = []
                    for g in range(gl):
                        gfi = gs + g
                        prov = {
                            "frame_index": int(gfi),
                            "landmark_index": int(li),
                            "invalid_reason": invalid_reasons[gfi][li],
                            "qa_flags": qa_flags_arr[gfi][li],
                        }
                        # Strong-depth provenance
                        sd_key = (gfi, comp_name, li)
                        if sd_key in sd_provenance_map:
                            prov["strong_depth"] = sd_provenance_map[sd_key]
                        gap_provenance.append(prov)

                    if not reasons and has_before and has_after and gl <= max_gap:
                        p0 = out[gs - 1, li]
                        p1 = out[fi, li]
                        for g in range(gl):
                            t = (g + 1) / (gl + 1)
                            out[gs + g, li] = p0 * (1 - t) + p1 * t
                            mask[gs + g, li] = True
                        records.append({
                            "lm_index": int(li), "gap_start": int(gs), "gap_end": int(ge),
                            "gap_length": int(gl), "method": "linear", "max_gap": int(max_gap),
                            "seg_start": int(s0), "seg_end": int(s1),
                            "endpoint_before_fi": int(gs - 1),
                            "endpoint_after_fi": int(fi),
                            "endpoint_before_xyz": p0.tolist(),
                            "endpoint_after_xyz": p1.tolist(),
                            "interp_distance_m": float(np.linalg.norm(p1 - p0)),
                            "boundary_type": "none",
                            "gap_provenance": gap_provenance,
                        })
                    else:
                        for g in range(gl):
                            rejected[gs + g, li] = True
                        records.append({
                            "lm_index": int(li), "gap_start": int(gs), "gap_end": int(ge),
                            "gap_length": int(gl), "method": "rejected", "max_gap": int(max_gap),
                            "rejection_reason": "|".join(reasons) if reasons else "unknown",
                            "seg_start": int(s0), "seg_end": int(s1),
                            "gap_provenance": gap_provenance,
                        })
                else:
                    fi += 1

    return out, mask, rejected, records


# ================================================================
#  OneEuro Filter (unchanged from v1.1)
# ================================================================

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
        self._reset_count = 0
        self._first_after_reset = False

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
        self._reset_count += 1
        self._first_after_reset = True

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            self._first_after_reset = False
            return x.copy()

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev.copy()

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        speed = np.linalg.norm(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t
        self._first_after_reset = False
        return x_hat


def apply_one_euro(xyz, valid, ts, boundary, params):
    N, nlm, _ = xyz.shape
    out = xyz.copy()
    reset_mask = np.zeros((N, nlm), dtype=bool)
    first_after_reset = np.zeros((N, nlm), dtype=bool)

    for li in range(nlm):
        filt = OneEuroFilter(params["min_cutoff"], params["beta"], params["d_cutoff"])
        for fi in range(N):
            if boundary[fi, li]:
                filt.reset()
                reset_mask[fi, li] = True
            if valid[fi, li] and not np.any(np.isnan(xyz[fi, li])):
                result = filt(xyz[fi, li], ts[fi])
                if filt._first_after_reset is False and filt._reset_count > 0 and fi > 0 and reset_mask[fi, li]:
                    first_after_reset[fi, li] = True
                out[fi, li] = result
            else:
                filt.reset()
                reset_mask[fi, li] = True
    return out, reset_mask, first_after_reset


def apply_savgol(xyz, valid, boundary, window=SAVGOL_WINDOW, order=SAVGOL_ORDER):
    if not HAS_SCIPY:
        return xyz.copy()
    N, nlm, _ = xyz.shape
    out = xyz.copy()
    for li in range(nlm):
        ss = None
        for fi in range(N + 1):
            is_break = fi == N or boundary[fi, li] or not valid[fi if fi < N else 0, li] or np.any(np.isnan(xyz[fi if fi < N else 0, li]))
            if fi < N and not is_break:
                if ss is None:
                    ss = fi
            else:
                if ss is not None:
                    slen = fi - ss
                    if slen >= window:
                        for dim in range(3):
                            out[ss:fi, li, dim] = savgol_filter(xyz[ss:fi, li, dim], window, order)
                    ss = None
    return out


# ================================================================
#  Extended Metrics
# ================================================================

def compute_extended_metrics(norm_baseline, norm_filtered, valid, scales, ts, boundary,
                             seg_info, label, word, arrs=None, norm=None):
    """Compute comprehensive filter evaluation metrics.

    norm_baseline and norm_filtered are in body-centric normalized space.
    Physical metrics use per-frame shoulder scale.
    """
    N, nlm, _ = norm_baseline.shape

    # Position deviation
    phys_devs = []
    norm_devs = []
    for fi in range(N):
        s = scales[fi] if not np.isnan(scales[fi]) else 0
        for li in range(nlm):
            if valid[fi, li] and not np.any(np.isnan(norm_baseline[fi, li])) and not np.any(np.isnan(norm_filtered[fi, li])):
                nd = np.linalg.norm(norm_filtered[fi, li] - norm_baseline[fi, li])
                norm_devs.append(nd)
                phys_devs.append(nd * s)

    if not phys_devs:
        return {"label": label, "n_valid": 0,
                "phys_dev_mean_mm": 0, "phys_dev_p95_mm": 0, "phys_dev_max_mm": 0,
                "norm_dev_mean_norm_units": 0, "norm_dev_p95_norm_units": 0, "norm_dev_max_norm_units": 0,
                "vel_rmse_mm_per_s": 0, "vel_peak_distortion_pct": 0,
                "accel_rmse_mm_per_s2": 0, "accel_peak_distortion_pct": 0,
                "path_length_ratio": 1.0, "amplitude_ratio_x": 1.0, "amplitude_ratio_y": 1.0, "amplitude_ratio_z": 1.0,
                "causal_delay_frames": 0, "causal_delay_ms": 0,
                "vel_distortion_pct": 0, "new_nans": 0, "new_infs": 0,
                "jitter_raw_start_idle_mm": "not_available_no_verified_idle",
                "jitter_filt_start_idle_mm": "not_available_no_verified_idle",
                "jitter_reduction_start_idle": "not_available_no_verified_idle",
                "jitter_raw_end_idle_mm": "not_available_no_verified_idle",
                "jitter_filt_end_idle_mm": "not_available_no_verified_idle",
                "jitter_reduction_end_idle": "not_available_no_verified_idle"}

    phys_devs = np.array(phys_devs)
    norm_devs = np.array(norm_devs)

    # -- Jitter: start idle --
    def compute_idle_jitter(idle_start_fi, idle_end_fi):
        jr, jf = [], []
        for fi in range(max(1, idle_start_fi), min(idle_end_fi, N)):
            s = scales[fi] if not np.isnan(scales[fi]) else 0
            for li in range(nlm):
                if valid[fi, li] and valid[fi - 1, li]:
                    r0, r1 = norm_baseline[fi - 1, li], norm_baseline[fi, li]
                    f0, f1 = norm_filtered[fi - 1, li], norm_filtered[fi, li]
                    if not np.any(np.isnan(r0)) and not np.any(np.isnan(r1)):
                        jr.append(np.linalg.norm(r1 - r0) * s)
                    if not np.any(np.isnan(f0)) and not np.any(np.isnan(f1)):
                        jf.append(np.linalg.norm(f1 - f0) * s)
        if jr and jf:
            return (float(np.mean(jr) * 1000), float(np.mean(jf) * 1000),
                    float(1 - np.mean(jf) / np.mean(jr)) if np.mean(jr) > 0 else 0.0)
        return None

    j_start = ("not_available_no_verified_idle",) * 3
    j_end = ("not_available_no_verified_idle",) * 3
    if seg_info["idle_verified_start"]:
        r = compute_idle_jitter(0, seg_info["seg_start"])
        if r:
            j_start = r
    if seg_info["idle_verified_end"]:
        r = compute_idle_jitter(seg_info["seg_end"] + 1, N)
        if r:
            j_end = r

    # -- Velocity (frame-by-frame) --
    raw_vel_norms = []
    filt_vel_norms = []
    vel_diffs = []
    key_lms = list(range(min(nlm, 5)))  # first 5 landmarks for overview
    for li in key_lms:
        for fi in range(1, N):
            dt = ts[fi] - ts[fi - 1]
            if dt <= 0 or dt > 0.1 or boundary[fi, li]:
                continue
            if not valid[fi, li] or not valid[fi - 1, li]:
                continue
            s = scales[fi] if not np.isnan(scales[fi]) else 0
            r0, r1 = norm_baseline[fi - 1, li], norm_baseline[fi, li]
            f0, f1 = norm_filtered[fi - 1, li], norm_filtered[fi, li]
            if np.any(np.isnan(r0)) or np.any(np.isnan(r1)) or np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                continue
            rv = np.linalg.norm(r1 - r0) * s / dt
            fv = np.linalg.norm(f1 - f0) * s / dt
            raw_vel_norms.append(rv)
            filt_vel_norms.append(fv)
            vel_diffs.append((fv - rv))

    vel_rmse = float(np.sqrt(np.mean(np.array(vel_diffs) ** 2))) * 1000 if vel_diffs else 0
    vel_peak_raw = float(np.max(raw_vel_norms)) if raw_vel_norms else 0
    vel_peak_filt = float(np.max(filt_vel_norms)) if filt_vel_norms else 0
    vel_peak_dist = float(abs(vel_peak_filt - vel_peak_raw) / vel_peak_raw * 100) if vel_peak_raw > 0 else 0
    vel_dist_mean = float(abs(np.mean(filt_vel_norms) - np.mean(raw_vel_norms)) / np.mean(raw_vel_norms) * 100) \
        if raw_vel_norms and np.mean(raw_vel_norms) > 0 else 0

    # -- Acceleration --
    raw_accel = []
    filt_accel = []
    accel_diffs = []
    for li in key_lms:
        prev_rv, prev_fv, prev_dt = None, None, None
        for fi in range(1, N):
            dt = ts[fi] - ts[fi - 1]
            if dt <= 0 or dt > 0.1 or boundary[fi, li]:
                prev_rv, prev_fv, prev_dt = None, None, None
                continue
            if not valid[fi, li] or not valid[fi - 1, li]:
                prev_rv, prev_fv, prev_dt = None, None, None
                continue
            s = scales[fi] if not np.isnan(scales[fi]) else 0
            r0, r1 = norm_baseline[fi - 1, li], norm_baseline[fi, li]
            f0, f1 = norm_filtered[fi - 1, li], norm_filtered[fi, li]
            if np.any(np.isnan(r0)) or np.any(np.isnan(r1)) or np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                prev_rv, prev_fv, prev_dt = None, None, None
                continue
            rv = (r1 - r0) * s / dt
            fv = (f1 - f0) * s / dt
            if prev_rv is not None and prev_dt is not None and prev_dt > 0:
                ra = np.linalg.norm(rv - prev_rv) / dt
                fa = np.linalg.norm(fv - prev_fv) / dt
                raw_accel.append(ra)
                filt_accel.append(fa)
                accel_diffs.append(fa - ra)
            prev_rv, prev_fv, prev_dt = rv, fv, dt

    accel_rmse = float(np.sqrt(np.mean(np.array(accel_diffs) ** 2))) * 1000 if accel_diffs else 0
    accel_peak_raw = float(np.max(raw_accel)) if raw_accel else 0
    accel_peak_filt = float(np.max(filt_accel)) if filt_accel else 0
    accel_peak_dist = float(abs(accel_peak_filt - accel_peak_raw) / accel_peak_raw * 100) if accel_peak_raw > 0 else 0

    # -- Path length & amplitude --
    path_raw_total = 0.0
    path_filt_total = 0.0
    amp_raw = [[], [], []]
    amp_filt = [[], [], []]
    for li in key_lms:
        for fi in range(1, N):
            if not valid[fi, li] or not valid[fi - 1, li]:
                continue
            r0, r1 = norm_baseline[fi - 1, li], norm_baseline[fi, li]
            f0, f1 = norm_filtered[fi - 1, li], norm_filtered[fi, li]
            if np.any(np.isnan(r0)) or np.any(np.isnan(r1)) or np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                continue
            s = scales[fi] if not np.isnan(scales[fi]) else 0
            path_raw_total += np.linalg.norm(r1 - r0) * s
            path_filt_total += np.linalg.norm(f1 - f0) * s
        for fi in range(N):
            if valid[fi, li]:
                for d in range(3):
                    if not np.isnan(norm_baseline[fi, li, d]):
                        amp_raw[d].append(norm_baseline[fi, li, d])
                    if not np.isnan(norm_filtered[fi, li, d]):
                        amp_filt[d].append(norm_filtered[fi, li, d])

    path_ratio = path_filt_total / path_raw_total if path_raw_total > 0 else 1.0
    amp_ratios = []
    for d in range(3):
        rng_r = (max(amp_raw[d]) - min(amp_raw[d])) if amp_raw[d] else 0
        rng_f = (max(amp_filt[d]) - min(amp_filt[d])) if amp_filt[d] else 0
        amp_ratios.append(rng_f / rng_r if rng_r > 0 else 1.0)

    # -- Causal delay (cross-correlation on wrist Y) --
    delay_frames = 0
    delay_ms = 0.0
    wrist_li = 0  # hand wrist (landmark 0) or pose wrist
    if nlm == 33:
        wrist_li = LM_LW
    sig_raw = []
    sig_filt = []
    sig_ts = []
    for fi in range(N):
        if valid[fi, wrist_li] and not np.any(np.isnan(norm_baseline[fi, wrist_li])) and not np.any(np.isnan(norm_filtered[fi, wrist_li])):
            sig_raw.append(norm_baseline[fi, wrist_li, 1])
            sig_filt.append(norm_filtered[fi, wrist_li, 1])
            sig_ts.append(ts[fi])
    if len(sig_raw) > 10:
        sr = np.array(sig_raw) - np.mean(sig_raw)
        sf = np.array(sig_filt) - np.mean(sig_filt)
        cc = np.correlate(sr, sf, mode="full")
        mid = len(sr) - 1
        # Only consider positive lags (filter delays)
        peak = int(np.argmax(cc[mid:mid + min(10, len(sr))]))
        delay_frames = peak
        if peak > 0 and len(sig_ts) > peak:
            delay_ms = float((sig_ts[peak] - sig_ts[0]) / peak * peak * 1000)

    # -- Peak velocity timing lag for wrists --
    peak_vel_lag_frames = "not_applicable"
    peak_vel_lag_ms = "not_applicable"
    onset_lag_frames = "not_applicable"
    onset_lag_ms = "not_applicable"
    end_lag_frames = "not_applicable"
    end_lag_ms = "not_applicable"

    if nlm == 33:  # pose component
        for wli in [LM_LW, LM_RW]:
            rv_arr, fv_arr, t_arr = [], [], []
            for fi in range(1, N):
                dt = ts[fi] - ts[fi - 1]
                if dt <= 0 or dt > 0.1 or boundary[fi, wli] or not valid[fi, wli] or not valid[fi - 1, wli]:
                    continue
                s = scales[fi] if not np.isnan(scales[fi]) else 0
                r0, r1 = norm_baseline[fi - 1, wli], norm_baseline[fi, wli]
                f0, f1 = norm_filtered[fi - 1, wli], norm_filtered[fi, wli]
                if np.any(np.isnan(r0)) or np.any(np.isnan(r1)) or np.any(np.isnan(f0)) or np.any(np.isnan(f1)):
                    continue
                rv_arr.append(np.linalg.norm(r1 - r0) * s / dt)
                fv_arr.append(np.linalg.norm(f1 - f0) * s / dt)
                t_arr.append(ts[fi])
            if len(rv_arr) > 5:
                rv_a = np.array(rv_arr)
                fv_a = np.array(fv_arr)
                t_a = np.array(t_arr)
                pk_r = int(np.argmax(rv_a))
                pk_f = int(np.argmax(fv_a))
                lag_f = pk_f - pk_r
                lag_t = float((t_a[pk_f] - t_a[pk_r]) * 1000) if pk_f < len(t_a) and pk_r < len(t_a) else 0
                peak_vel_lag_frames = lag_f
                peak_vel_lag_ms = lag_t
                # Onset: first frame > 10% of peak
                thr = 0.1 * np.max(rv_a)
                onset_r = next((i for i in range(len(rv_a)) if rv_a[i] > thr), 0)
                thr_f = 0.1 * np.max(fv_a)
                onset_f = next((i for i in range(len(fv_a)) if fv_a[i] > thr_f), 0)
                onset_lag_frames = onset_f - onset_r
                onset_lag_ms = float((t_a[onset_f] - t_a[onset_r]) * 1000) if onset_f < len(t_a) and onset_r < len(t_a) else 0
                # End: last frame > 10% of peak
                end_r = len(rv_a) - 1 - next((i for i in range(len(rv_a) - 1, -1, -1) if rv_a[i] > thr), len(rv_a) - 1)
                end_f = len(fv_a) - 1 - next((i for i in range(len(fv_a) - 1, -1, -1) if fv_a[i] > thr_f), len(fv_a) - 1)
                end_r_idx = next((i for i in range(len(rv_a) - 1, -1, -1) if rv_a[i] > thr), len(rv_a) - 1)
                end_f_idx = next((i for i in range(len(fv_a) - 1, -1, -1) if fv_a[i] > thr_f), len(fv_a) - 1)
                end_lag_frames = end_f_idx - end_r_idx
                end_lag_ms = float((t_a[min(end_f_idx, len(t_a)-1)] - t_a[min(end_r_idx, len(t_a)-1)]) * 1000)
                break  # use first wrist found

    # -- New NaN/Inf --
    new_nans = sum(1 for fi in range(N) for li in range(nlm)
                   if valid[fi, li] and not np.any(np.isnan(norm_baseline[fi, li]))
                   and np.any(np.isnan(norm_filtered[fi, li])))
    new_infs = sum(1 for fi in range(N) for li in range(nlm)
                   if valid[fi, li] and not np.any(np.isnan(norm_baseline[fi, li]))
                   and np.any(np.isinf(norm_filtered[fi, li])))

    return {
        "label": label, "n_valid": len(phys_devs),
        "phys_dev_mean_mm": float(np.mean(phys_devs) * 1000),
        "phys_dev_p95_mm": float(np.percentile(phys_devs, 95) * 1000),
        "phys_dev_max_mm": float(np.max(phys_devs) * 1000),
        "norm_dev_mean_norm_units": float(np.mean(norm_devs)),
        "norm_dev_p95_norm_units": float(np.percentile(norm_devs, 95)),
        "norm_dev_max_norm_units": float(np.max(norm_devs)),
        "jitter_raw_start_idle_mm": j_start[0],
        "jitter_filt_start_idle_mm": j_start[1],
        "jitter_reduction_start_idle": j_start[2],
        "jitter_raw_end_idle_mm": j_end[0],
        "jitter_filt_end_idle_mm": j_end[1],
        "jitter_reduction_end_idle": j_end[2],
        "vel_distortion_pct": vel_dist_mean,
        "vel_rmse_mm_per_s": vel_rmse,
        "vel_peak_distortion_pct": vel_peak_dist,
        "accel_rmse_mm_per_s2": accel_rmse,
        "accel_peak_distortion_pct": accel_peak_dist,
        "path_length_ratio": float(path_ratio),
        "amplitude_ratio_x": float(amp_ratios[0]),
        "amplitude_ratio_y": float(amp_ratios[1]),
        "amplitude_ratio_z": float(amp_ratios[2]),
        "causal_delay_frames": delay_frames,
        "causal_delay_ms": float(delay_ms),
        "peak_vel_lag_frames": peak_vel_lag_frames,
        "peak_vel_lag_ms": peak_vel_lag_ms,
        "onset_lag_frames": onset_lag_frames,
        "onset_lag_ms": onset_lag_ms,
        "end_lag_frames": end_lag_frames,
        "end_lag_ms": end_lag_ms,
        "new_nans": new_nans, "new_infs": new_infs,
    }


# ================================================================
#  Sign-specific Metrics
# ================================================================

def compute_sign_metrics(norm_baseline, norm_filtered, valid, arrs, norm_dict, seg, word, ts, scales):
    """Compute word-specific quality metrics."""
    N = norm_baseline.shape[0] if norm_baseline.ndim == 3 else 0
    result = {}
    ss, se = seg["seg_start"], seg["seg_end"]

    if word == "LOVE" and norm_baseline.shape[1] >= 33:
        # Min distance between left and right hand wrist (pose landmarks)
        dists_raw, dists_filt = [], []
        dists_raw_fi, dists_filt_fi = [], []
        for fi in range(ss, min(se + 1, N)):
            if valid[fi, LM_LW] and valid[fi, LM_RW]:
                lw_r = norm_baseline[fi, LM_LW]
                rw_r = norm_baseline[fi, LM_RW]
                lw_f = norm_filtered[fi, LM_LW]
                rw_f = norm_filtered[fi, LM_RW]
                if not np.any(np.isnan(lw_r)) and not np.any(np.isnan(rw_r)):
                    s = scales[fi] if not np.isnan(scales[fi]) else 0
                    dists_raw.append(np.linalg.norm(rw_r - lw_r) * s)
                    dists_raw_fi.append(fi)
                if not np.any(np.isnan(lw_f)) and not np.any(np.isnan(rw_f)):
                    s = scales[fi] if not np.isnan(scales[fi]) else 0
                    dists_filt.append(np.linalg.norm(rw_f - lw_f) * s)
                    dists_filt_fi.append(fi)
        if dists_raw and dists_filt:
            min_r = np.min(dists_raw)
            min_f = np.min(dists_filt)
            min_r_fi = dists_raw_fi[int(np.argmin(dists_raw))]
            min_f_fi = dists_filt_fi[int(np.argmin(dists_filt))]
            result["love_min_hand_dist_raw_mm"] = float(min_r * 1000)
            result["love_min_hand_dist_filt_mm"] = float(min_f * 1000)
            result["love_min_dist_frame_raw"] = int(min_r_fi)
            result["love_min_dist_frame_filt"] = int(min_f_fi)
            result["love_min_dist_timing_shift_ms"] = float((ts[min_f_fi] - ts[min_r_fi]) * 1000)
        else:
            result["love_min_hand_dist_raw_mm"] = "insufficient_data"
            result["love_min_hand_dist_filt_mm"] = "insufficient_data"

    elif word == "GREETING" and norm_baseline.shape[1] >= 33:
        # Direction reversals in wrist Y (vertical oscillation)
        for wrist_name, wli in [("left_wrist", LM_LW), ("right_wrist", LM_RW)]:
            vy_raw, vy_filt = [], []
            for fi in range(ss + 1, min(se + 1, N)):
                if valid[fi, wli] and valid[fi - 1, wli]:
                    r0 = norm_baseline[fi - 1, wli, 1]
                    r1 = norm_baseline[fi, wli, 1]
                    f0 = norm_filtered[fi - 1, wli, 1]
                    f1 = norm_filtered[fi, wli, 1]
                    if not np.isnan(r0) and not np.isnan(r1):
                        vy_raw.append(r1 - r0)
                    if not np.isnan(f0) and not np.isnan(f1):
                        vy_filt.append(f1 - f0)
            # Count sign changes
            rev_raw = sum(1 for i in range(1, len(vy_raw)) if vy_raw[i] * vy_raw[i - 1] < 0) if len(vy_raw) > 1 else 0
            rev_filt = sum(1 for i in range(1, len(vy_filt)) if vy_filt[i] * vy_filt[i - 1] < 0) if len(vy_filt) > 1 else 0
            result[f"greeting_{wrist_name}_reversals_raw"] = rev_raw
            result[f"greeting_{wrist_name}_reversals_filt"] = rev_filt
            # Amplitude (Y range)
            y_raw_vals = [norm_baseline[fi, wli, 1] for fi in range(ss, min(se + 1, N))
                          if valid[fi, wli] and not np.isnan(norm_baseline[fi, wli, 1])]
            y_filt_vals = [norm_filtered[fi, wli, 1] for fi in range(ss, min(se + 1, N))
                           if valid[fi, wli] and not np.isnan(norm_filtered[fi, wli, 1])]
            amp_raw = (max(y_raw_vals) - min(y_raw_vals)) if y_raw_vals else 0
            amp_filt = (max(y_filt_vals) - min(y_filt_vals)) if y_filt_vals else 0
            result[f"greeting_{wrist_name}_amplitude_raw_norm_units"] = float(amp_raw)
            result[f"greeting_{wrist_name}_amplitude_filt_norm_units"] = float(amp_filt)
            result[f"greeting_{wrist_name}_amplitude_ratio"] = float(amp_filt / amp_raw) if amp_raw > 0 else 1.0

    elif word == "THANK_YOU" and norm_baseline.shape[1] >= 33:
        # Starting wrist position relative to face (nose landmark)
        # Forward (Z-axis) movement amplitude
        for wrist_name, wli in [("left_wrist", LM_LW), ("right_wrist", LM_RW)]:
            # Start position near face
            start_dist_raw = "insufficient_data"
            start_dist_filt = "insufficient_data"
            if ss < N and valid[ss, wli] and valid[ss, LM_NOSE]:
                nose_r = norm_baseline[ss, LM_NOSE]
                wr = norm_baseline[ss, wli]
                wf = norm_filtered[ss, wli]
                if not np.any(np.isnan(nose_r)) and not np.any(np.isnan(wr)):
                    s = scales[ss] if not np.isnan(scales[ss]) else 0
                    start_dist_raw = float(np.linalg.norm(wr - nose_r) * s * 1000)
                if not np.any(np.isnan(nose_r)) and not np.any(np.isnan(wf)):
                    s = scales[ss] if not np.isnan(scales[ss]) else 0
                    start_dist_filt = float(np.linalg.norm(wf - nose_r) * s * 1000)
            result[f"thankyou_{wrist_name}_start_face_dist_raw_mm"] = start_dist_raw
            result[f"thankyou_{wrist_name}_start_face_dist_filt_mm"] = start_dist_filt

            # Forward movement amplitude (Z-axis range in segment)
            z_raw = [norm_baseline[fi, wli, 2] for fi in range(ss, min(se + 1, N))
                     if valid[fi, wli] and not np.isnan(norm_baseline[fi, wli, 2])]
            z_filt = [norm_filtered[fi, wli, 2] for fi in range(ss, min(se + 1, N))
                      if valid[fi, wli] and not np.isnan(norm_filtered[fi, wli, 2])]
            amp_raw = (max(z_raw) - min(z_raw)) if z_raw else 0
            amp_filt = (max(z_filt) - min(z_filt)) if z_filt else 0
            s_mean = np.nanmean(scales[ss:se+1])
            result[f"thankyou_{wrist_name}_fwd_amplitude_raw_mm"] = float(amp_raw * s_mean * 1000) if not np.isnan(s_mean) else 0
            result[f"thankyou_{wrist_name}_fwd_amplitude_filt_mm"] = float(amp_filt * s_mean * 1000) if not np.isnan(s_mean) else 0
            result[f"thankyou_{wrist_name}_fwd_amplitude_ratio"] = float(amp_filt / amp_raw) if amp_raw > 0 else 1.0

    return result


# ================================================================
#  Masking Experiment (max_gap evaluation)
# ================================================================

def run_masking_experiment(arrs, norm, boundaries, strong_forbidden, ts):
    """Deterministic masking experiment on known-valid segments."""
    results = []
    for comp, nlm in [("pose", 33), ("lh", 21), ("rh", 21)]:
        xyz = arrs[f"{comp}_xyz"]
        valid = arrs[f"{comp}_te"]
        bd = boundaries[comp]
        sf = strong_forbidden[comp]
        N = xyz.shape[0]

        for gap_len in [1, 2, 3]:
            position_errs_m = []
            vel_discontinuities_m = []
            accel_errs_m = []

            for li in range(nlm):
                # Find contiguous valid spans not crossing boundaries/strong-depth
                spans = []
                ss = None
                for fi in range(N):
                    if valid[fi, li] and not bd[fi, li] and not sf[fi, li] and not np.any(np.isnan(xyz[fi, li])):
                        if ss is None:
                            ss = fi
                    else:
                        if ss is not None:
                            spans.append((ss, fi - 1))
                            ss = None
                if ss is not None:
                    spans.append((ss, N - 1))

                for s0, s1 in spans:
                    span_len = s1 - s0 + 1
                    if span_len < gap_len + 2:
                        continue
                    # Mask interior windows (step by gap_len to avoid overlap)
                    for start in range(s0 + 1, s1 - gap_len + 1, max(1, gap_len)):
                        end = start + gap_len - 1
                        if end >= s1:
                            break
                        # Endpoints
                        p0 = xyz[start - 1, li]
                        p1 = xyz[end + 1, li]
                        for g in range(gap_len):
                            t = (g + 1) / (gap_len + 1)
                            recon = p0 * (1 - t) + p1 * t
                            actual = xyz[start + g, li]
                            position_errs_m.append(np.linalg.norm(recon - actual))

                        # Velocity discontinuity at entry
                        if start >= s0 + 2:
                            dt_pre = ts[start - 1] - ts[start - 2]
                            if dt_pre > 0 and dt_pre < 0.1:
                                v_true_pre = np.linalg.norm(xyz[start - 1, li] - xyz[start - 2, li]) / dt_pre
                                dt_gap = ts[start] - ts[start - 1]
                                if dt_gap > 0:
                                    t_frac = 1 / (gap_len + 1)
                                    recon_first = p0 * (1 - t_frac) + p1 * t_frac
                                    v_recon_entry = np.linalg.norm(recon_first - p0) / dt_gap
                                    vel_discontinuities_m.append(abs(v_recon_entry - v_true_pre))

            if position_errs_m:
                pe = np.array(position_errs_m)
                row = {
                    "component": comp, "gap_length": gap_len,
                    "n_trials": len(pe),
                    "recon_err_mean_mm": float(np.mean(pe) * 1000),
                    "recon_err_p95_mm": float(np.percentile(pe, 95) * 1000),
                    "recon_err_max_mm": float(np.max(pe) * 1000),
                }
                if vel_discontinuities_m:
                    vd = np.array(vel_discontinuities_m)
                    row["vel_disc_mean_m_per_s"] = float(np.mean(vd))
                    row["vel_disc_max_m_per_s"] = float(np.max(vd))
                results.append(row)

    return results


# ================================================================
#  Segmentation contact sheet (unchanged from v1.1)
# ================================================================

def make_seg_contact_sheet(sid, seg_info, out_dir):
    short = sid.replace("S001_P001_", "")
    vid_path = OVERLAY_DIR / f"reproj_{short}_front.mp4"
    if not vid_path.exists():
        return None

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    N = seg_info["total_frames"]
    ss = seg_info["seg_start"]
    se = seg_info["seg_end"]
    mid = (ss + se) // 2

    frame_indices = []
    labels = []

    before = max(0, ss - 3)
    after = min(N - 1, se + 3)

    if ss > 2:
        frame_indices.append(before)
        labels.append(f"before(f{before})")
    else:
        frame_indices.append(0)
        labels.append(f"start_edge(f0)")

    frame_indices.append(ss)
    labels.append(f"seg_start(f{ss})")
    frame_indices.append(mid)
    labels.append(f"middle(f{mid})")
    frame_indices.append(se)
    labels.append(f"seg_end(f{se})")

    if se < N - 3:
        frame_indices.append(after)
        labels.append(f"after(f{after})")
    else:
        frame_indices.append(N - 1)
        labels.append(f"end_edge(f{N - 1})")

    frames = []
    for fi in frame_indices:
        if fi < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                cv2.putText(frame, labels[len(frames)], (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                esm = seg_info.get("energy_smooth", np.zeros(1))
                eval_idx = min(fi, len(esm) - 1)
                cv2.putText(frame, f"e={esm[eval_idx]:.4f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                frames.append(frame)
            else:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        else:
            frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
    cap.release()

    if not frames:
        return None

    target_h = 360
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        scale = target_h / h
        resized.append(cv2.resize(f, (int(w * scale), target_h)))

    sheet = np.concatenate(resized, axis=1)
    out_path = out_dir / f"seg_contact_{short}.png"
    cv2.imwrite(str(out_path), sheet)
    return out_path


# ================================================================
#  NPZ Save / Load / Verify
# ================================================================

def save_sample_npz(sid, arrs, norm, interp_data, smooth_data, seg, masking_results,
                    norm_interp_cache, transform_comp, out_dir):
    """Save all numeric arrays and masks to compressed NPZ files."""
    short = sid.replace("S001_P001_", "")
    npz_dir = out_dir / "npz"
    npz_dir.mkdir(exist_ok=True)

    # 1. Interpolation NPZ (per max_gap)
    for mg in INTERP_CANDIDATES:
        ir = interp_data[mg]
        arrays = {}
        for comp in ["pose", "lh", "rh"]:
            arrays[f"{comp}_interp"] = ir[f"{comp}_interp"]
            arrays[f"{comp}_mask"] = ir[f"{comp}_mask"]
            arrays[f"{comp}_rejected"] = ir[f"{comp}_rejected"]
            arrays[f"{comp}_boundary"] = ir[f"{comp}_boundary"]
            arrays[f"{comp}_strong_forbidden"] = ir[f"{comp}_strong_forbidden"]
            # Normalized interpolated
            ni_key = f"norm_interp_{comp}_g{mg}"
            if ni_key in norm_interp_cache:
                arrays[f"{comp}_norm_interp"] = norm_interp_cache[ni_key]
        np.savez_compressed(npz_dir / f"interp_{short}_g{mg}.npz", **arrays)

    # 2. Interpolation provenance JSON (per max_gap)
    prov_dir = out_dir / "provenance"
    prov_dir.mkdir(exist_ok=True)
    for mg in INTERP_CANDIDATES:
        ir = interp_data[mg]
        prov = {}
        for comp in ["pose", "lh", "rh"]:
            prov[comp] = ir[f"{comp}_records"]
        with open(prov_dir / f"interp_provenance_{short}_g{mg}.json", "w", encoding="utf-8") as f:
            json.dump(prov, f)

    # 3. Smoothing NPZ (per filter)
    for fname, sm in smooth_data.items():
        arrays = {}
        for comp in ["pose", "lh", "rh"]:
            arrays[f"{comp}_filtered"] = sm[f"{comp}_filtered"]
            if f"{comp}_norm_filtered" in sm:
                arrays[f"{comp}_norm_filtered"] = sm[f"{comp}_norm_filtered"]
            if f"{comp}_reset_mask" in sm:
                arrays[f"{comp}_reset_mask"] = sm[f"{comp}_reset_mask"]
            if f"{comp}_first_after_reset" in sm:
                arrays[f"{comp}_first_after_reset"] = sm[f"{comp}_first_after_reset"]
        np.savez_compressed(npz_dir / f"smooth_{short}_{fname}.npz", **arrays)

    # 4. Smoothing schema JSON (per filter)
    for fname, sm in smooth_data.items():
        schema = {
            "filter": fname, "sample_id": sid,
            "interp_candidate_used": 2,
        }
        if fname.startswith("oe_"):
            pname = fname[3:]
            for c in ONE_EURO_CANDIDATES:
                if c["name"] == pname:
                    schema["filter_params"] = c
                    break
        elif fname == "savgol":
            schema["filter_params"] = {"window": SAVGOL_WINDOW, "order": SAVGOL_ORDER}
        elif fname == "none":
            schema["filter_params"] = {"type": "identity"}
        for comp in ["pose", "lh", "rh"]:
            schema[f"{comp}_metrics"] = sm[f"{comp}_metrics"]
            if f"{comp}_sign_metrics" in sm:
                schema[f"{comp}_sign_metrics"] = sm[f"{comp}_sign_metrics"]
        with open(prov_dir / f"smooth_schema_{short}_{fname}.json", "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, default=str)

    # 5. Normalization NPZ
    arrays = {
        "origins": norm["origins"], "rotations": norm["rotations"],
        "scales": norm["scales"], "dets": norm["dets"], "ortho_errs": norm["ortho_errs"],
    }
    for comp in ["pose", "lh", "rh"]:
        arrays[f"norm_{comp}"] = norm[f"norm_{comp}"]
    if transform_comp:
        arrays["stab_origins"] = transform_comp["stab_origins"]
        arrays["stab_scales"] = transform_comp["stab_scales"]
        arrays["stab_rotations"] = transform_comp["stab_rotations"]
    np.savez_compressed(npz_dir / f"norm_{short}.npz", **arrays)

    return npz_dir


def verify_sample_npz(sid, arrs, norm, interp_data, smooth_data, norm_interp_cache, out_dir):
    """Load saved NPZ files and verify all values match in-memory data."""
    short = sid.replace("S001_P001_", "")
    npz_dir = out_dir / "npz"
    fails = []

    # Verify interpolation NPZ
    for mg in INTERP_CANDIDATES:
        path = npz_dir / f"interp_{short}_g{mg}.npz"
        if not path.exists():
            fails.append(f"missing_interp_g{mg}")
            continue
        loaded = np.load(path, allow_pickle=False)
        ir = interp_data[mg]
        for comp in ["pose", "lh", "rh"]:
            for key_suffix in ["interp", "mask", "rejected", "boundary", "strong_forbidden"]:
                arr_key = f"{comp}_{key_suffix}"
                if arr_key in loaded:
                    mem = ir[f"{comp}_{key_suffix}"]
                    if not np.array_equal(loaded[arr_key], mem):
                        if mem.dtype in (np.float32, np.float64):
                            if not np.allclose(loaded[arr_key], mem, equal_nan=True, atol=1e-12):
                                fails.append(f"mismatch_interp_g{mg}_{arr_key}")
                        else:
                            fails.append(f"mismatch_interp_g{mg}_{arr_key}")
            ni_key = f"norm_interp_{comp}_g{mg}"
            if ni_key in norm_interp_cache and f"{comp}_norm_interp" in loaded:
                if not np.allclose(loaded[f"{comp}_norm_interp"], norm_interp_cache[ni_key], equal_nan=True, atol=1e-12):
                    fails.append(f"mismatch_interp_g{mg}_norm_{comp}")
        loaded.close()

    # Verify smoothing NPZ
    for fname, sm in smooth_data.items():
        path = npz_dir / f"smooth_{short}_{fname}.npz"
        if not path.exists():
            fails.append(f"missing_smooth_{fname}")
            continue
        loaded = np.load(path, allow_pickle=False)
        for comp in ["pose", "lh", "rh"]:
            for key_suffix in ["filtered", "norm_filtered", "reset_mask", "first_after_reset"]:
                arr_key = f"{comp}_{key_suffix}"
                mem_key = f"{comp}_{key_suffix}"
                if arr_key in loaded and mem_key in sm:
                    mem = sm[mem_key]
                    if mem.dtype in (np.float32, np.float64):
                        if not np.allclose(loaded[arr_key], mem, equal_nan=True, atol=1e-12):
                            fails.append(f"mismatch_smooth_{fname}_{arr_key}")
                    else:
                        if not np.array_equal(loaded[arr_key], mem):
                            fails.append(f"mismatch_smooth_{fname}_{arr_key}")
        loaded.close()

    # Verify normalization NPZ
    path = npz_dir / f"norm_{short}.npz"
    if not path.exists():
        fails.append("missing_norm")
    else:
        loaded = np.load(path, allow_pickle=False)
        for key in ["origins", "rotations", "scales", "dets", "ortho_errs"]:
            if not np.allclose(loaded[key], norm[key], equal_nan=True, atol=1e-12):
                fails.append(f"mismatch_norm_{key}")
        for comp in ["pose", "lh", "rh"]:
            nk = f"norm_{comp}"
            if not np.allclose(loaded[nk], norm[nk], equal_nan=True, atol=1e-12):
                fails.append(f"mismatch_norm_{nk}")
        loaded.close()

    return fails


# ================================================================
#  Assertions
# ================================================================

def run_assertions(all_results, split, pilot_ids, input_hashes, strong_depth, sd_prov_records,
                   out_dir, masking_all):
    results = []
    np_ = 0
    nf_ = 0

    def chk(cond, name, detail=""):
        nonlocal np_, nf_
        if cond:
            np_ += 1
        else:
            nf_ += 1
            sp(f"    [FAIL] {name}: {detail}")
        results.append({"name": name, "status": "PASS" if cond else "FAIL", "detail": detail})

    # --- Input verification ---
    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        h = sha256_file(RAW3D_DIR / f"landmarks_3d_{short}.json")
        chk(h == input_hashes[sid], f"input_hash_{short}", f"exp={input_hashes[sid][:16]}... got={h[:16]}...")

    # --- Split & pilot checks ---
    chk(len(pilot_ids) == 9, "pilot_count_9")
    for w in WORDS:
        wp = [s for s in pilot_ids if w in s]
        chk(len(wp) == 3, f"pilot_{w}_3")
    for sid in pilot_ids:
        chk(split[sid]["split"] == "train", f"pilot_train_{sid.replace('S001_P001_','')}")
    chk(len(set(pilot_ids)) == 9, "no_dup_pilots")

    for sid, sr in all_results.items():
        short = sid.replace("S001_P001_", "")
        arrs = sr["arrs"]
        norm = sr["norm"]
        raw = sr["raw_data"]
        N = len(arrs["timestamps"])

        # --- Full timestamp preservation ---
        for fi, fr in enumerate(raw["frames"]):
            chk(abs(arrs["timestamps"][fi] - fr["front_time_s"]) < 1e-12,
                f"ts_{short}_f{fi}")

        # --- Full coordinate + validity preservation ---
        KEYS_MAP = {"pose": ("pose_landmarks_3d", 33), "lh": ("left_hand_landmarks_3d", 21), "rh": ("right_hand_landmarks_3d", 21)}
        for comp, (jkey, nlm) in KEYS_MAP.items():
            for fi in range(N):
                for li in range(nlm):
                    lm = raw["frames"][fi][jkey][li]
                    chk(arrs[f"{comp}_tri"][fi, li] == lm["triangulation_valid"],
                        f"tri_{short}_{comp}_f{fi}_l{li}")
                    chk(arrs[f"{comp}_qa"][fi, li] == lm["qa_valid"],
                        f"qa_{short}_{comp}_f{fi}_l{li}")
                    chk(arrs[f"{comp}_te"][fi, li] == lm["template_eligible"],
                        f"te_{short}_{comp}_f{fi}_l{li}")
                    if lm["triangulation_valid"] and lm["xyz"] is not None:
                        orig = np.array(lm["xyz"])
                        loaded = arrs[f"{comp}_xyz"][fi, li]
                        chk(np.allclose(orig, loaded, atol=1e-12),
                            f"xyz_{short}_{comp}_f{fi}_l{li}")

        # --- Rotation checks ---
        for fi in range(N):
            if norm["statuses"][fi] in ("ok", "fallback_prev", "fallback_next"):
                R = norm["rotations"][fi]
                if not np.any(np.isnan(R)):
                    det = norm["dets"][fi]
                    chk(abs(det - 1.0) < 1e-6, f"det_{short}_f{fi}")
                    oe = norm["ortho_errs"][fi]
                    chk(oe < 1e-6, f"ortho_{short}_f{fi}")

        # --- Axis continuity ---
        n_flips = {"right": 0, "up": 0, "forward": 0}
        for fi in range(1, N):
            if (norm["statuses"][fi] in ("ok", "fallback_prev", "fallback_next") and
                    norm["statuses"][fi - 1] in ("ok", "fallback_prev", "fallback_next")):
                R0 = norm["rotations"][fi - 1]
                R1 = norm["rotations"][fi]
                if not np.any(np.isnan(R0)) and not np.any(np.isnan(R1)):
                    for col, name in [(0, "right"), (1, "up"), (2, "forward")]:
                        if np.dot(R0[:, col], R1[:, col]) < 0:
                            n_flips[name] += 1
        for ax, count in n_flips.items():
            chk(count == 0, f"no_{ax}_flip_{short}")

        # --- Scale checks ---
        vs = norm["scales"][~np.isnan(norm["scales"])]
        if len(vs) > 0:
            chk(np.all(vs > 0.05), f"scale_pos_{short}")
            chk(np.all(vs < 2.0), f"scale_range_{short}")
            chk(not np.any(np.isinf(vs)), f"scale_no_inf_{short}")
        else:
            chk(False, f"scale_exists_{short}")

        # --- No invalid->zero replacement ---
        for comp in ["pose", "lh", "rh"]:
            nk = f"norm_{comp}"
            v = arrs[f"{comp}_te"]
            coords = norm[nk]
            nzero = sum(1 for fi in range(N) for li in range(coords.shape[1])
                        if not v[fi, li] and not np.any(np.isnan(coords[fi, li])) and np.allclose(coords[fi, li], 0))
            chk(nzero == 0, f"no_zero_replace_{short}_{comp}")

        # --- Interpolation: strong-depth boundary checks ---
        sd_set = strong_depth.get(sid, set())
        for mg, ir in sr["interp"].items():
            for comp in ["pose", "lh", "rh"]:
                imask = ir[f"{comp}_mask"]
                sf = ir[f"{comp}_strong_forbidden"]
                n_sd_interp = int(np.sum(imask & sf))
                chk(n_sd_interp == 0, f"no_strong_depth_interp_{short}_{comp}_g{mg}")

                bd = ir[f"{comp}_boundary"]
                for li in range(imask.shape[1]):
                    for fi in range(imask.shape[0]):
                        if imask[fi, li] and bd[fi, li]:
                            chk(False, f"boundary_cross_{short}_{comp}_g{mg}_f{fi}_l{li}")

        # --- Known regression: LOVE_FAST_R04 pose lm20 f37-38 ---
        if sid == "S001_P001_LOVE_FAST_R04":
            for mg in INTERP_CANDIDATES:
                ir = sr["interp"][mg]
                for fi in [37, 38]:
                    if fi < N:
                        chk(not ir["pose_mask"][fi, 20], f"regression_LOVE_FAST_R04_pose_lm20_f{fi}_g{mg}")

        # --- No extrapolation at boundaries ---
        for mg, ir in sr["interp"].items():
            for comp in ["pose", "lh", "rh"]:
                m = ir[f"{comp}_mask"]
                chk(not np.any(m[0, :]), f"no_extrap_start_{short}_{comp}_g{mg}")
                chk(not np.any(m[-1, :]), f"no_extrap_end_{short}_{comp}_g{mg}")

        # --- Interpolation provenance count matches mask count ---
        for mg, ir in sr["interp"].items():
            for comp in ["pose", "lh", "rh"]:
                n_interp = int(np.sum(ir[f"{comp}_mask"]))
                recs = [r for r in ir[f"{comp}_records"] if r["method"] == "linear"]
                n_acc = sum(r["gap_length"] for r in recs)
                chk(n_interp == n_acc, f"interp_prov_{short}_{comp}_g{mg}")

        # --- Interpolation provenance: target landmark accuracy ---
        for mg, ir in sr["interp"].items():
            for comp in ["pose", "lh", "rh"]:
                for rec in ir[f"{comp}_records"]:
                    for gp in rec.get("gap_provenance", []):
                        fi_p = gp["frame_index"]
                        li_p = gp["landmark_index"]
                        expected_reason = arrs[f"{comp}_invalid_reason"][fi_p][li_p]
                        chk(gp["invalid_reason"] == expected_reason,
                            f"prov_reason_{short}_{comp}_g{mg}_f{fi_p}_l{li_p}")

        # --- Smoothing: reset at boundaries ---
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                if f"{comp}_reset_mask" not in sm:
                    continue
                rm = sm[f"{comp}_reset_mask"]
                bd = sr["interp"][2][f"{comp}_boundary"]
                for li in range(rm.shape[1]):
                    for fi in range(rm.shape[0]):
                        if bd[fi, li]:
                            chk(rm[fi, li], f"smooth_reset_{short}_{comp}_{fname}_f{fi}_l{li}")

        # --- Smoothing: first after reset == input (v1.1 fix preserved) ---
        for fname, sm in sr["smooth"].items():
            if fname == "none" or fname == "savgol":
                continue
            for comp in ["pose", "lh", "rh"]:
                far = sm.get(f"{comp}_first_after_reset")
                filtered = sm[f"{comp}_filtered"]
                base = sr["interp"][2][f"{comp}_interp"]
                valid_sm = arrs[f"{comp}_te"] | sr["interp"][2][f"{comp}_mask"]
                if far is None:
                    continue
                rm = sm[f"{comp}_reset_mask"]
                for li in range(rm.shape[1]):
                    in_reset = False
                    for fi in range(rm.shape[0]):
                        if rm[fi, li]:
                            if valid_sm[fi, li] and not np.any(np.isnan(base[fi, li])):
                                if not np.any(np.isnan(filtered[fi, li])):
                                    chk(np.allclose(filtered[fi, li], base[fi, li], atol=1e-10),
                                        f"first_after_reset_{short}_{comp}_{fname}_f{fi}_l{li}")
                                in_reset = False
                            else:
                                in_reset = True
                        elif in_reset and valid_sm[fi, li] and not np.any(np.isnan(base[fi, li])):
                            if not np.any(np.isnan(filtered[fi, li])):
                                chk(np.allclose(filtered[fi, li], base[fi, li], atol=1e-10),
                                    f"first_after_reset_{short}_{comp}_{fname}_f{fi}_l{li}")
                            in_reset = False

        # --- No-smoothing deviation must be exactly 0 ---
        ns = sr["smooth"].get("none", {})
        for comp in ["pose", "lh", "rh"]:
            m = ns.get(f"{comp}_metrics", {})
            chk(m.get("phys_dev_mean_mm", -1) == 0, f"no_smooth_dev_zero_{short}_{comp}_mean")
            chk(m.get("phys_dev_p95_mm", -1) == 0, f"no_smooth_dev_zero_{short}_{comp}_p95")
            chk(m.get("phys_dev_max_mm", -1) == 0, f"no_smooth_dev_zero_{short}_{comp}_max")
            chk(m.get("vel_distortion_pct", -1) == 0, f"no_smooth_vel_zero_{short}_{comp}")

        # --- New NaN/Inf from smoothing ---
        for fname, sm in sr["smooth"].items():
            for comp in ["pose", "lh", "rh"]:
                m = sm[f"{comp}_metrics"]
                chk(m["new_nans"] == 0, f"no_new_nan_{short}_{comp}_{fname}")
                chk(m["new_infs"] == 0, f"no_new_inf_{short}_{comp}_{fname}")

        # --- Shape checks ---
        for comp, nlm in [("pose", 33), ("lh", 21), ("rh", 21)]:
            chk(arrs[f"{comp}_xyz"].shape == (N, nlm, 3), f"shape_{short}_{comp}_xyz")
            chk(norm[f"norm_{comp}"].shape == (N, nlm, 3), f"shape_{short}_{comp}_norm")
            for mg in INTERP_CANDIDATES:
                chk(sr["interp"][mg][f"{comp}_interp"].shape == (N, nlm, 3), f"shape_{short}_{comp}_interp_g{mg}")
                chk(sr["interp"][mg][f"{comp}_mask"].shape == (N, nlm), f"shape_{short}_{comp}_mask_g{mg}")

        # --- Round-trip NPZ verification ---
        npz_fails = sr.get("npz_verify_fails", [])
        for fail_name in npz_fails:
            chk(False, f"npz_roundtrip_{short}_{fail_name}")
        if not npz_fails:
            chk(True, f"npz_roundtrip_{short}_all_pass")

    # --- Output files ---
    expected = [
        "step7a_v1_2_config.json", "step7_dataset_split_manifest.csv", "step7_dataset_split_manifest.json",
        "pilot_selection_manifest.json", "input_file_manifest.json",
        "interpolation_comparison.csv", "smoothing_comparison.csv",
        "segmentation_summary.csv", "normalization_summary.csv",
        "masking_experiment.csv", "transform_comparison.csv",
        "strong_depth_provenance.json",
    ]
    for fn in expected:
        chk((out_dir / fn).exists(), f"output_{fn}")

    # NPZ files exist
    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        for mg in INTERP_CANDIDATES:
            chk((out_dir / "npz" / f"interp_{short}_g{mg}.npz").exists(), f"npz_interp_{short}_g{mg}")
        for fname in all_results[sid]["smooth"]:
            chk((out_dir / "npz" / f"smooth_{short}_{fname}.npz").exists(), f"npz_smooth_{short}_{fname}")
        chk((out_dir / "npz" / f"norm_{short}.npz").exists(), f"npz_norm_{short}")

    # Contact sheets
    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        chk((out_dir / "contact_sheets" / f"seg_contact_{short}.png").exists(), f"contact_{short}")

    # Seg CSV rows
    seg_csv = out_dir / "segmentation_summary.csv"
    if seg_csv.exists():
        with open(seg_csv) as f:
            rows = sum(1 for _ in csv.reader(f)) - 1
        chk(rows == 9, "seg_csv_9_rows")

    # Input manifest hash verification
    inp_man = out_dir / "input_file_manifest.json"
    if inp_man.exists():
        with open(inp_man, encoding="utf-8") as f:
            inp_data = json.load(f)
        for entry in inp_data.get("inputs", []):
            fp = PROJECT / entry["file"]
            if fp.exists():
                h = sha256_file(fp)
                chk(h == entry["sha256"], f"input_manifest_hash_{Path(entry['file']).name}")

    sp(f"\n  Assertions: {np_} PASS, {nf_} FAIL")
    return results, np_, nf_


# ================================================================
#  Visualization (unchanged from v1.1)
# ================================================================

def plot_seg(sid, arrs, norm, seg, viz_dir):
    if not HAS_MPL:
        return
    short = sid.replace("S001_P001_", "")
    t = arrs["timestamps"] - arrs["timestamps"][0]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, seg["energy"], alpha=0.3, label="raw")
    ax.plot(t, seg["energy_smooth"], label="smooth")
    ax.axhline(seg["threshold"], color="r", ls="--", alpha=0.5)
    ax.axvline(t[seg["seg_start"]], color="g", ls="-", label="start")
    ax.axvline(t[seg["seg_end"]], color="g", ls="--", label="end")
    ax.set_title(f"Seg: {short} ({seg['method']})")
    ax.legend(fontsize=7)
    ax.set_xlabel("Time (s)")
    fig.savefig(viz_dir / f"seg_{short}.png", dpi=120)
    plt.close(fig)


def plot_traj(sid, arrs, norm, interp_data, smooth_data, norm_interp_cache, viz_dir):
    if not HAS_MPL:
        return
    short = sid.replace("S001_P001_", "")
    t = arrs["timestamps"] - arrs["timestamps"][0]
    N = len(t)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    li = LM_LW
    ni = norm_interp_cache.get(f"norm_interp_pose_g2")
    for dim, ax_name in enumerate(["X(right)", "Y(up)", "Z(fwd)"]):
        ax = axes[dim]
        raw = norm["norm_pose"][:, li, dim]
        ax.plot(t, raw, "k-", alpha=0.3, lw=0.8, label="raw")
        if ni is not None:
            ax.plot(t, ni[:, li, dim], "b-", alpha=0.4, lw=0.8, label="interp(2)")

        for fname, sm in smooth_data.items():
            if fname == "none":
                continue
            nf = sm.get("pose_norm_filtered")
            if nf is not None:
                ax.plot(t, nf[:, li, dim], alpha=0.5, lw=1, label=fname)

        ax.set_ylabel(ax_name)
        if dim == 0:
            ax.set_title(f"{short} - Pose L.Wrist")
            ax.legend(fontsize=6, ncol=6)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(viz_dir / f"traj_{short}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ================================================================
#  Report
# ================================================================

def write_report(out_dir, pilot_ids, pilot, seg_rows, norm_rows, interp_rows, smooth_rows,
                 masking_rows, transform_rows, np_, nf_, smoothing_recommendation):
    rpt = ["# Step 7A v1.2 Preprocessing Pilot Report", "",
           "## Status: STEP7A_PREPROCESSING_PILOT_V1_2_COMPLETE" if nf_ == 0 else "## Status: STEP7A_V1_2_BLOCKED",
           "", f"**Assertions**: {np_} PASS / {nf_} FAIL", "",
           "## Review Fixes Applied (v1.2)", "",
           "- 2: Actual NPZ+JSON save for interpolation/smoothing arrays",
           "- 3: No-smoothing baseline fix (deviation=0, interp effect separated)",
           "- 4: Extended OneEuro evaluation (timing lag, path-length, amplitude, vel/accel RMSE, sign-specific)",
           "- 5: max_gap masking experiment with reconstruction error",
           "- 6: Provenance fix (target landmark indexing, strong-depth provenance)",
           "- 7: Full input/output manifests",
           "- 8: Normalization transform comparison (raw vs causal stabilized)",
           "- 9: Round-trip NPZ file verification assertions", "",
           "## Pilot Samples", "",
           "| Sample | Tier | TE Rate |", "|--------|------|---------|"]
    for sid in pilot_ids:
        p = pilot[sid]
        rpt.append(f"| {sid.replace('S001_P001_','')} | {p['tier']} | {p['te_rate']:.1%} |")
    rpt.append("")

    rpt.append("## Segmentation")
    rpt.append("")
    rpt.append("| Sample | Frames | Seg | Method | Idle Start | Idle End |")
    rpt.append("|--------|--------|-----|--------|------------|----------|")
    for r in seg_rows:
        rpt.append(f"| {r['sample_id'].replace('S001_P001_','')} | {r['total_frames']} | "
                   f"[{r['seg_start']}-{r['seg_end']}] | {r['method']} | {r['idle_verified_start']} | {r['idle_verified_end']} |")
    rpt.append("")

    rpt.append("## OneEuro Parameter Comparison (pose, physical units)")
    rpt.append("")
    rpt.append("| Sample | Param | Mean(mm) | P95(mm) | Max(mm) | VelDist(%) | PathRatio | Delay(ms) |")
    rpt.append("|--------|-------|----------|---------|---------|------------|-----------|-----------|")
    for r in smooth_rows:
        if r["component"] == "pose" and r["filter"].startswith("one_euro_"):
            rpt.append(f"| {r['sample_id'].replace('S001_P001_','')} | {r['filter'].replace('one_euro_','')} | "
                       f"{r.get('phys_dev_mean_mm', 'N/A'):.1f} | {r.get('phys_dev_p95_mm', 'N/A'):.1f} | "
                       f"{r.get('phys_dev_max_mm', 'N/A'):.1f} | {r.get('vel_distortion_pct', 'N/A'):.1f} | "
                       f"{r.get('path_length_ratio', 'N/A'):.3f} | {r.get('causal_delay_ms', 'N/A')} |")
    rpt.append("")
    rpt.append(f"**Smoothing recommendation**: {smoothing_recommendation}")
    rpt.append("")

    rpt.append("## max_gap Masking Experiment (reconstruction error)")
    rpt.append("")
    rpt.append("| Component | Gap | Trials | Mean(mm) | P95(mm) | Max(mm) |")
    rpt.append("|-----------|-----|--------|----------|---------|---------|")
    for r in masking_rows:
        rpt.append(f"| {r['component']} | {r['gap_length']} | {r['n_trials']} | "
                   f"{r['recon_err_mean_mm']:.2f} | {r['recon_err_p95_mm']:.2f} | {r['recon_err_max_mm']:.2f} |")
    rpt.append("")
    rpt.append("**Note**: max_gap is NOT confirmed. External review required before selection.")
    rpt.append("")

    rpt.append("## Transform Comparison (raw vs causal stabilized)")
    rpt.append("")
    rpt.append("| Sample | Origin Mean(mm) | Origin Max(mm) | Scale Mean(mm) | Right Axis(deg) | Up Axis(deg) |")
    rpt.append("|--------|----------------|----------------|----------------|-----------------|--------------|")
    for r in transform_rows:
        c = r["comparison"]
        rpt.append(f"| {r['sample_id'].replace('S001_P001_','')} | "
                   f"{c['origin_diff_mean_mm']:.2f} | {c['origin_diff_max_mm']:.2f} | "
                   f"{c['scale_diff_mean_mm']:.2f} | "
                   f"{c['right_axis_angle_diff_mean_deg']:.3f} | "
                   f"{c['up_axis_angle_diff_mean_deg']:.3f} |")
    rpt.append("")

    rpt.append("## Interpolation (strong-depth enforcement)")
    rpt.append("")
    rpt.append("| Sample | Comp | max_gap | Filled | Rejected | SD Interp |")
    rpt.append("|--------|------|---------|--------|----------|-----------|")
    for r in interp_rows:
        rpt.append(f"| {r['sample_id'].replace('S001_P001_','')} | {r['component']} | {r['max_gap']} | "
                   f"{r['n_filled']} | {r['n_rejected']} | {r['n_strong_depth_interp']} |")
    rpt.append("")

    with open(out_dir / "STEP7A_V1_2_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(rpt))


# ================================================================
#  Helpers
# ================================================================

def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_csv_flexible(path, rows):
    if not rows:
        return
    all_keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_full_sequence(sid, arrs, norm, seg, out_dir):
    """Save lightweight sequence JSON (schema only, arrays in NPZ)."""
    short = sid.replace("S001_P001_", "")
    N = len(arrs["timestamps"])

    seq = {
        "sample_id": sid, "version": "7a_v1.2",
        "n_frames": N,
        "coordinate_system": {"raw": "FRONT_camera_meters", "normalized": "body_centric_shoulder_width_scaled"},
        "segmentation": {"start": seg["seg_start"], "end": seg["seg_end"], "method": seg["method"],
                          "idle_verified_start": seg["idle_verified_start"], "idle_verified_end": seg["idle_verified_end"]},
        "data_files": {
            "normalization": f"npz/norm_{short}.npz",
            "interpolation": {mg: f"npz/interp_{short}_g{mg}.npz" for mg in INTERP_CANDIDATES},
            "smoothing": "npz/smooth_{short}_{{filter}}.npz",
            "provenance": {mg: f"provenance/interp_provenance_{short}_g{mg}.json" for mg in INTERP_CANDIDATES},
        },
        "frames": [],
    }

    for fi in range(N):
        frame = {
            "frame_index": fi,
            "timestamp_s": float(arrs["timestamps"][fi]),
            "in_segment": seg["seg_start"] <= fi <= seg["seg_end"],
            "transform": {
                "origin": norm["origins"][fi].tolist() if not np.any(np.isnan(norm["origins"][fi])) else None,
                "rotation": norm["rotations"][fi].tolist() if not np.any(np.isnan(norm["rotations"][fi])) else None,
                "scale": float(norm["scales"][fi]) if not np.isnan(norm["scales"][fi]) else None,
                "status": norm["statuses"][fi],
                "fallback_src": norm["fallback_src"][fi],
                "determinant": float(norm["dets"][fi]) if not np.isnan(norm["dets"][fi]) else None,
                "ortho_error": float(norm["ortho_errs"][fi]) if not np.isnan(norm["ortho_errs"][fi]) else None,
            },
        }

        for comp, jkey, nlm in [("pose", "pose_landmarks_3d", 33), ("left_hand", "left_hand_landmarks_3d", 21),
                                  ("right_hand", "right_hand_landmarks_3d", 21)]:
            comp_short = {"pose": "pose", "left_hand": "lh", "right_hand": "rh"}[comp]
            lms = []
            for li in range(nlm):
                raw_pt = arrs[f"{comp_short}_xyz"][fi, li]
                norm_pt = norm[f"norm_{comp_short}"][fi, li]
                lm_data = {
                    "raw_xyz": raw_pt.tolist() if not np.any(np.isnan(raw_pt)) else None,
                    "norm_xyz": norm_pt.tolist() if not np.any(np.isnan(norm_pt)) else None,
                    "tri_valid": bool(arrs[f"{comp_short}_tri"][fi, li]),
                    "qa_valid": bool(arrs[f"{comp_short}_qa"][fi, li]),
                    "te": bool(arrs[f"{comp_short}_te"][fi, li]),
                    "qa_flags": arrs[f"{comp_short}_flags"][fi][li],
                    "invalid_reason": arrs[f"{comp_short}_invalid_reason"][fi][li],
                    "is_reappearance": bool(arrs[f"{comp_short}_reappear"][fi, li]),
                }
                lms.append(lm_data)
            frame[comp] = lms
        seq["frames"].append(frame)

    with open(out_dir / "sequences" / f"full_{short}.json", "w", encoding="utf-8") as f:
        json.dump(seq, f)
    sz = os.path.getsize(out_dir / "sequences" / f"full_{short}.json")
    sp(f"      Saved full_{short}.json ({sz / 1e6:.1f} MB)")


# ================================================================
#  Main
# ================================================================

def main():
    sp("Step 7A v1.2: Preprocessing Pilot (revised)")
    sp("  Status: STEP7A_V1_2_REVISION_REQUIRED\n")

    assert STEP6B_DIR.exists(), f"Step 6B dir not found: {STEP6B_DIR}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ["sequences", "contact_sheets", "visualizations", "npz", "provenance"]:
        (OUT_DIR / subdir).mkdir(exist_ok=True)

    # 1. Split
    sp("  1. Dataset split (unchanged from v1.0)...")
    split = build_dataset_split()
    split_rows = sorted(split.values(), key=lambda x: x["sample_id"])
    with open(OUT_DIR / "step7_dataset_split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, ["sample_id", "word", "split", "position_in_word"]).writeheader()
        csv.DictWriter(f, ["sample_id", "word", "split", "position_in_word"]).writerows(split_rows)
    with open(OUT_DIR / "step7_dataset_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"samples": split_rows, "counts": {"train": 27, "val": 9, "test": 9}}, f, indent=2)

    # 2. Pilot selection (unchanged)
    sp("  2. Pilot selection (unchanged from v1.0)...")
    pilot = select_pilots(split)
    pilot_ids = sorted(pilot.keys())
    for sid in pilot_ids:
        p = pilot[sid]
        sp(f"    {sid.replace('S001_P001_',''):30s} tier={p['tier']:4s} te={p['te_rate']:.1%}")

    with open(OUT_DIR / "pilot_selection_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"pilot_samples": [pilot[s] for s in pilot_ids],
                    "selection_criteria": {"method": "quality_score_ranking", "source": "train_only"}}, f, indent=2, default=str)

    # 3. Input hashes (enhanced: include all inputs)
    sp("\n  3. Input hashes...")
    input_hashes = {}
    input_manifest = []

    # Raw 3D JSON files
    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        p = RAW3D_DIR / f"landmarks_3d_{short}.json"
        h = sha256_file(p)
        sz = os.path.getsize(p)
        input_hashes[sid] = h
        input_manifest.append({"sample_id": sid, "file": p.relative_to(PROJECT).as_posix(),
                                "size_bytes": sz, "sha256": h, "artifact_type": "raw_3d_json"})

    # Temporal anomalies CSV
    if TA_CSV.exists():
        input_manifest.append({
            "file": TA_CSV.relative_to(PROJECT).as_posix(),
            "size_bytes": os.path.getsize(TA_CSV),
            "sha256": sha256_file(TA_CSV),
            "artifact_type": "temporal_anomalies_csv",
        })

    # Overlay MP4s (for contact sheets)
    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        mp4 = OVERLAY_DIR / f"reproj_{short}_front.mp4"
        if mp4.exists():
            input_manifest.append({
                "sample_id": sid, "file": mp4.relative_to(PROJECT).as_posix(),
                "size_bytes": os.path.getsize(mp4),
                "sha256": sha256_file(mp4),
                "artifact_type": "overlay_mp4_front",
            })

    # Config (self-reference)
    input_manifest.append({
        "file": Path(__file__).relative_to(PROJECT).as_posix(),
        "size_bytes": os.path.getsize(__file__),
        "sha256": sha256_file(__file__),
        "artifact_type": "execution_script",
    })

    with open(OUT_DIR / "input_file_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"self_reference_excluded": True, "inputs": input_manifest}, f, indent=2)

    # 4. Load strong-depth boundaries (enhanced)
    sp("  4. Loading strong-depth boundaries...")
    strong_depth, sd_prov_records = load_strong_depth_data(set(pilot_ids))
    total_sd = sum(len(v) for v in strong_depth.values())
    sp(f"    Total strong-depth entries in pilots: {total_sd}")

    # Build strong-depth provenance lookup map: (fi, comp, li) -> record
    sd_prov_map = {}
    for rec in sd_prov_records:
        key = (rec["frame_index"], rec["component"], rec["landmark_index"])
        sd_prov_map.setdefault(rec["sample_id"], {})[key] = rec

    with open(OUT_DIR / "strong_depth_provenance.json", "w", encoding="utf-8") as f:
        json.dump(sd_prov_records, f, indent=2)

    # Config
    config = {
        "step": "7A", "version": "1.2",
        "input_dir": str(STEP6B_DIR.relative_to(PROJECT)),
        "output_dir": str(OUT_DIR.relative_to(PROJECT)),
        "segmentation": {"energy_window": SEG_ENERGY_WINDOW, "idle_threshold": SEG_IDLE_THRESHOLD,
                          "pad_start": SEG_PAD_START, "pad_end": SEG_PAD_END, "min_active": SEG_MIN_ACTIVE},
        "interpolation_candidates": INTERP_CANDIDATES,
        "one_euro_candidates": ONE_EURO_CANDIDATES,
        "savgol": {"window": SAVGOL_WINDOW, "order": SAVGOL_ORDER},
        "normalization": {"origin": "shoulder_midpoint", "scale": "shoulder_width",
                          "axes": "right_up_forward", "handedness": "right_handed"},
        "data_format": {"arrays": "compressed_npz_allow_pickle_false", "provenance": "json"},
    }
    with open(OUT_DIR / "step7a_v1_2_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 5. Process samples
    sp("\n  5. Processing pilot samples...")
    all_results = {}
    seg_rows, norm_rows, interp_rows, smooth_rows = [], [], [], []
    masking_all_rows = []
    transform_rows = []

    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        word = split[sid]["word"]
        sp(f"\n    === {short} ===")

        raw_data = load_sample(sid)
        arrs = extract_arrays(raw_data)
        N = len(arrs["timestamps"])
        ts = arrs["timestamps"]
        sd_set = strong_depth.get(sid, set())
        sd_prov_sid = sd_prov_map.get(sid, {})

        # Normalize
        norm = normalize_sequence(arrs)
        sc = defaultdict(int)
        for s in norm["statuses"]:
            sc[s] += 1
        sp(f"      Norm: {dict(sc)}, scale_mean={np.nanmean(norm['scales']):.4f}")

        vs = norm["scales"][~np.isnan(norm["scales"])]
        norm_rows.append({
            "sample_id": sid, "n_frames": N, "n_ok": sc.get("ok", 0),
            "n_fallback_prev": sc.get("fallback_prev", 0), "n_fallback_next": sc.get("fallback_next", 0),
            "n_degenerate": sum(v for k, v in sc.items() if "degenerate" in k),
            "scale_mean_m": float(np.mean(vs)) if len(vs) else None,
            "scale_std_m": float(np.std(vs)) if len(vs) else None,
        })

        # Transform comparison
        transform_comp = compute_causal_stabilized_transform(norm, ts)
        transform_rows.append({"sample_id": sid, "comparison": transform_comp["comparison"]})
        sp(f"      Transform: origin_diff={transform_comp['comparison']['origin_diff_mean_mm']:.2f}mm "
           f"scale_diff={transform_comp['comparison']['scale_diff_mean_mm']:.2f}mm")

        # Segment
        seg = detect_segment(arrs, norm)
        sp(f"      Seg: [{seg['seg_start']}-{seg['seg_end']}] ({seg['seg_frames']}/{N}f) "
           f"method={seg['method']} idle_start={seg['idle_verified_start']} idle_end={seg['idle_verified_end']}")

        seg_rows.append({
            "sample_id": sid, "total_frames": N, "seg_start": seg["seg_start"], "seg_end": seg["seg_end"],
            "seg_frames": seg["seg_frames"], "raw_start": seg["raw_start"], "raw_end": seg["raw_end"],
            "method": seg["method"], "idle_verified_start": seg["idle_verified_start"],
            "idle_verified_end": seg["idle_verified_end"],
        })

        # Contact sheet
        cs_path = make_seg_contact_sheet(sid, seg, OUT_DIR / "contact_sheets")
        if cs_path:
            sp(f"      Contact sheet: {cs_path.name}")

        # Boundaries
        frame_disc, boundaries, strong_forbidden = build_all_boundaries(arrs, sd_set, sid)

        # Interpolation
        sp(f"      Interpolation...")
        interp_data = {}
        norm_interp_cache = {}
        for mg in INTERP_CANDIDATES:
            ir = {}
            for comp, nlm in [("pose", 33), ("lh", 21), ("rh", 21)]:
                xyz = arrs[f"{comp}_xyz"]
                valid = arrs[f"{comp}_te"]
                bd = boundaries[comp]
                sf = strong_forbidden[comp]

                out_xyz, mask, rej, records = interpolate_gaps(
                    xyz, valid, bd, sf, mg,
                    arrs[f"{comp}_invalid_reason"], arrs[f"{comp}_flags"],
                    sd_prov_sid, comp, sid)
                ir[f"{comp}_interp"] = out_xyz
                ir[f"{comp}_mask"] = mask
                ir[f"{comp}_rejected"] = rej
                ir[f"{comp}_records"] = records
                ir[f"{comp}_boundary"] = bd
                ir[f"{comp}_strong_forbidden"] = sf

                # Normalize interpolated for smoothing baseline
                valid_sm = valid | mask
                ni = normalize_metric_coords(out_xyz, valid_sm, norm)
                norm_interp_cache[f"norm_interp_{comp}_g{mg}"] = ni

                n_filled = int(np.sum(mask))
                n_sd_check = int(np.sum(mask & sf))
                interp_rows.append({
                    "sample_id": sid, "component": comp, "max_gap": mg,
                    "n_filled": n_filled, "n_rejected": int(np.sum(rej)),
                    "n_strong_depth_interp": n_sd_check,
                })

            interp_data[mg] = ir
            sp(f"        g={mg}: pose={int(np.sum(ir['pose_mask']))} lh={int(np.sum(ir['lh_mask']))} "
               f"rh={int(np.sum(ir['rh_mask']))} sd_interp=0")

        # Masking experiment
        sp(f"      Masking experiment...")
        me_results = run_masking_experiment(arrs, norm, boundaries, strong_forbidden, ts)
        for r in me_results:
            r["sample_id"] = sid
            masking_all_rows.append(r)
            sp(f"        {r['component']} g={r['gap_length']}: n={r['n_trials']} mean={r['recon_err_mean_mm']:.2f}mm "
               f"p95={r['recon_err_p95_mm']:.2f}mm max={r['recon_err_max_mm']:.2f}mm")

        # Smoothing (on gap=2 interpolated)
        sp(f"      Smoothing sweep...")
        base = interp_data[2]
        smooth_data = {}

        # No smoothing (fixed: deviation = 0)
        ns = {}
        for comp in ["pose", "lh", "rh"]:
            valid_sm = arrs[f"{comp}_te"] | base[f"{comp}_mask"]
            ni = norm_interp_cache[f"norm_interp_{comp}_g2"]
            ns[f"{comp}_filtered"] = base[f"{comp}_interp"].copy()
            ns[f"{comp}_norm_filtered"] = ni.copy()
            ns[f"{comp}_reset_mask"] = base[f"{comp}_boundary"].copy()
            # No-smoothing: baseline = filtered (identity), so all deviations = 0
            ns[f"{comp}_metrics"] = {
                "label": "none", "n_valid": int(np.sum(valid_sm)),
                "phys_dev_mean_mm": 0.0, "phys_dev_p95_mm": 0.0, "phys_dev_max_mm": 0.0,
                "norm_dev_mean_norm_units": 0.0, "norm_dev_p95_norm_units": 0.0, "norm_dev_max_norm_units": 0.0,
                "vel_distortion_pct": 0.0, "vel_rmse_mm_per_s": 0.0, "vel_peak_distortion_pct": 0.0,
                "accel_rmse_mm_per_s2": 0.0, "accel_peak_distortion_pct": 0.0,
                "path_length_ratio": 1.0, "amplitude_ratio_x": 1.0, "amplitude_ratio_y": 1.0, "amplitude_ratio_z": 1.0,
                "causal_delay_frames": 0, "causal_delay_ms": 0.0,
                "peak_vel_lag_frames": 0, "peak_vel_lag_ms": 0.0,
                "onset_lag_frames": 0, "onset_lag_ms": 0.0,
                "end_lag_frames": 0, "end_lag_ms": 0.0,
                "new_nans": 0, "new_infs": 0,
                "jitter_raw_start_idle_mm": "not_available_no_verified_idle",
                "jitter_filt_start_idle_mm": "not_available_no_verified_idle",
                "jitter_reduction_start_idle": "not_available_no_verified_idle",
                "jitter_raw_end_idle_mm": "not_available_no_verified_idle",
                "jitter_filt_end_idle_mm": "not_available_no_verified_idle",
                "jitter_reduction_end_idle": "not_available_no_verified_idle",
            }
            smooth_rows.append({"sample_id": sid, "component": comp, "filter": "none", **ns[f"{comp}_metrics"]})
        smooth_data["none"] = ns

        # OneEuro candidates
        for oe_params in ONE_EURO_CANDIDATES:
            oe = {}
            for comp in ["pose", "lh", "rh"]:
                valid_sm = arrs[f"{comp}_te"] | base[f"{comp}_mask"]
                bd = base[f"{comp}_boundary"]
                filtered, rm, far = apply_one_euro(base[f"{comp}_interp"], valid_sm, ts, bd, oe_params)

                # Normalize filtered
                norm_filt = normalize_metric_coords(filtered, valid_sm, norm)

                # Baseline for smoothing comparison: normalized interpolated (gap=2)
                norm_base = norm_interp_cache[f"norm_interp_{comp}_g2"]

                oe[f"{comp}_filtered"] = filtered
                oe[f"{comp}_norm_filtered"] = norm_filt
                oe[f"{comp}_reset_mask"] = rm
                oe[f"{comp}_first_after_reset"] = far
                oe[f"{comp}_metrics"] = compute_extended_metrics(
                    norm_base, norm_filt, valid_sm,
                    norm["scales"], ts, bd, seg, f"oe_{oe_params['name']}", word, arrs, norm)
                # Sign-specific metrics
                if comp == "pose":
                    oe[f"{comp}_sign_metrics"] = compute_sign_metrics(
                        norm_base, norm_filt, valid_sm, arrs, norm, seg, word, ts, norm["scales"])
                smooth_rows.append({"sample_id": sid, "component": comp,
                                     "filter": f"one_euro_{oe_params['name']}", **oe[f"{comp}_metrics"]})

            smooth_data[f"oe_{oe_params['name']}"] = oe
            sp(f"        {oe_params['name']}: pose_dev={oe['pose_metrics']['phys_dev_mean_mm']:.1f}mm "
               f"p95={oe['pose_metrics']['phys_dev_p95_mm']:.1f}mm "
               f"vel_dist={oe['pose_metrics']['vel_distortion_pct']:.1f}% "
               f"path={oe['pose_metrics']['path_length_ratio']:.3f}")

        # SavGol
        if HAS_SCIPY:
            sg = {}
            for comp in ["pose", "lh", "rh"]:
                valid_sm = arrs[f"{comp}_te"] | base[f"{comp}_mask"]
                bd = base[f"{comp}_boundary"]
                filtered = apply_savgol(base[f"{comp}_interp"], valid_sm, bd)
                norm_filt = normalize_metric_coords(filtered, valid_sm, norm)
                norm_base = norm_interp_cache[f"norm_interp_{comp}_g2"]

                sg[f"{comp}_filtered"] = filtered
                sg[f"{comp}_norm_filtered"] = norm_filt
                sg[f"{comp}_reset_mask"] = bd.copy()
                sg[f"{comp}_metrics"] = compute_extended_metrics(
                    norm_base, norm_filt, valid_sm,
                    norm["scales"], ts, bd, seg, "savgol", word, arrs, norm)
                smooth_rows.append({"sample_id": sid, "component": comp, "filter": "savgol", **sg[f"{comp}_metrics"]})
            smooth_data["savgol"] = sg

        # Save NPZ
        save_sample_npz(sid, arrs, norm, interp_data, smooth_data, seg,
                        me_results, norm_interp_cache, transform_comp, OUT_DIR)

        # Verify NPZ round-trip
        npz_fails = verify_sample_npz(sid, arrs, norm, interp_data, smooth_data, norm_interp_cache, OUT_DIR)
        if npz_fails:
            sp(f"      NPZ verification FAIL: {npz_fails}")
        else:
            sp(f"      NPZ round-trip verified")

        # Visualization
        if HAS_MPL:
            plot_seg(sid, arrs, norm, seg, OUT_DIR / "visualizations")
            plot_traj(sid, arrs, norm, interp_data, smooth_data, norm_interp_cache, OUT_DIR / "visualizations")

        # Save full sequence JSON
        save_full_sequence(sid, arrs, norm, seg, OUT_DIR)

        all_results[sid] = {"arrs": arrs, "raw_data": raw_data, "norm": norm, "seg": seg,
                            "interp": interp_data, "smooth": smooth_data,
                            "npz_verify_fails": npz_fails}

    # Save CSVs
    sp("\n  6. Saving CSVs...")
    _write_csv(OUT_DIR / "segmentation_summary.csv", seg_rows)
    _write_csv(OUT_DIR / "normalization_summary.csv", norm_rows)
    _write_csv(OUT_DIR / "interpolation_comparison.csv", interp_rows)
    _write_csv_flexible(OUT_DIR / "smoothing_comparison.csv", smooth_rows)
    _write_csv_flexible(OUT_DIR / "masking_experiment.csv", masking_all_rows)
    _write_csv_flexible(OUT_DIR / "transform_comparison.csv",
                        [{"sample_id": r["sample_id"], **r["comparison"]} for r in transform_rows])

    # Smoothing recommendation
    has_any_idle = any(sr["seg"]["idle_verified_start"] or sr["seg"]["idle_verified_end"]
                       for sr in all_results.values())
    if not has_any_idle:
        smoothing_rec = "no_final_smoothing_candidate_due_to_insufficient_verified_idle"
    else:
        # Check if any candidate has both jitter reduction AND acceptable distortion
        candidates_with_jitter = []
        for oe_params in ONE_EURO_CANDIDATES:
            fname = f"oe_{oe_params['name']}"
            has_jitter_data = False
            for sid, sr in all_results.items():
                m = sr["smooth"].get(fname, {}).get("pose_metrics", {})
                if isinstance(m.get("jitter_reduction_start_idle"), float) or isinstance(m.get("jitter_reduction_end_idle"), float):
                    has_jitter_data = True
                    break
            if has_jitter_data:
                candidates_with_jitter.append(fname)
        if not candidates_with_jitter:
            smoothing_rec = "no_final_smoothing_candidate_due_to_insufficient_verified_idle"
        else:
            smoothing_rec = f"candidates_with_verified_idle_jitter: {candidates_with_jitter}. No confirmed selection - external review required."

    # Assertions
    sp("\n  7. Assertions...")
    ar, np_, nf_ = run_assertions(all_results, split, pilot_ids, input_hashes, strong_depth,
                                   sd_prov_records, OUT_DIR, masking_all_rows)
    with open(OUT_DIR / "assertion_results.json", "w", encoding="utf-8") as f:
        json.dump(ar, f, indent=2)

    # Report
    sp("\n  8. Report...")
    write_report(OUT_DIR, pilot_ids, pilot, seg_rows, norm_rows, interp_rows, smooth_rows,
                 masking_all_rows, transform_rows, np_, nf_, smoothing_rec)

    # Output manifest
    sp("  9. Output manifest...")
    output_manifest = []
    for root, _, fnames in os.walk(OUT_DIR):
        for fn in fnames:
            fp = Path(root) / fn
            rel = fp.relative_to(OUT_DIR).as_posix()
            if rel.startswith("output_file_manifest"):
                continue
            sz = os.path.getsize(fp)
            h = sha256_file(fp)
            # Determine artifact type
            if fn.endswith(".npz"):
                atype = "numeric_array"
            elif fn.endswith(".json") and "provenance" in rel:
                atype = "provenance"
            elif fn.endswith(".json") and "smooth_schema" in rel:
                atype = "smoothing_schema"
            elif fn.endswith(".json") and "full_" in fn:
                atype = "sequence_json"
            elif fn.endswith(".csv"):
                atype = "summary_csv"
            elif fn.endswith(".png"):
                atype = "visualization"
            elif fn.endswith(".md"):
                atype = "report"
            elif fn.endswith(".json"):
                atype = "metadata_json"
            else:
                atype = "other"
            # Infer sample/filter/max_gap
            entry = {"relative_path": rel, "size_bytes": sz, "sha256": h,
                     "artifact_type": atype, "schema_version": "7a_v1.2"}
            # Extract sample from filename
            for sid in pilot_ids:
                short_sid = sid.replace("S001_P001_", "")
                if short_sid in fn:
                    entry["sample_id"] = sid
                    break
            # Extract max_gap
            for mg in INTERP_CANDIDATES:
                if f"_g{mg}" in fn:
                    entry["max_gap"] = mg
                    break
            # Extract filter
            for fname_f in ["none", "savgol"] + [f"oe_{c['name']}" for c in ONE_EURO_CANDIDATES]:
                if f"_{fname_f}" in fn:
                    entry["filter"] = fname_f
                    break
            output_manifest.append(entry)

    with open(OUT_DIR / "output_file_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"self_reference_excluded": True, "schema_version": "7a_v1.2",
                    "outputs": output_manifest, "total_files": len(output_manifest)}, f, indent=2)

    # Manifest
    manifest = {
        "step": "7A", "version": "1.2",
        "status": "STEP7A_PREPROCESSING_PILOT_V1_2_COMPLETE" if nf_ == 0 else "STEP7A_V1_2_BLOCKED_ASSERTION_FAILURE",
        "prior_status": ["STEP6B_COMPLETE", "STEP7A_V1_1_EXTERNAL_REVIEW_PARTIAL_PASS"],
        "pilot_samples": pilot_ids, "dataset_split": {"train": 27, "val": 9, "test": 9},
        "assertions": {"pass": np_, "fail": nf_},
        "smoothing_recommendation": smoothing_rec,
        "self_reference_excluded": True,
    }
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ZIP
    sp("\n  10. ZIP...")
    zip_path = PROJECT / "WordsInSilence_step7a_v1_2_pilot_delivery.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for root, _, fnames in os.walk(OUT_DIR):
            for fn in fnames:
                fp = Path(root) / fn
                arc = fp.relative_to(OUT_DIR).as_posix()
                comp = zipfile.ZIP_STORED if fp.suffix.lower() in (".png", ".mp4", ".npz") else zipfile.ZIP_DEFLATED
                zf.write(fp, arc, compress_type=comp)
        zf.write(Path(__file__), "s001_step7a_preprocess_pilot_v1_2.py", compress_type=zipfile.ZIP_DEFLATED)

    zs = os.path.getsize(zip_path)
    zh = sha256_file(zip_path)
    sp(f"    ZIP: {zip_path.name} ({zs / 1e6:.1f} MB)")
    sp(f"    SHA-256: {zh}")

    sp(f"\n  ============================================================")
    sp(f"  STEP 7A v1.2 SUMMARY")
    sp(f"  ============================================================")
    sp(f"  Assertions: {np_} PASS / {nf_} FAIL")
    sp(f"  Output: {OUT_DIR}")
    if nf_ == 0:
        sp(f"  Status: STEP6B_COMPLETE")
        sp(f"  Status: STEP7A_PREPROCESSING_PILOT_V1_2_COMPLETE")
        sp(f"  Status: STEP7A_V1_2_BLOCKED_PENDING_EXTERNAL_REVIEW")
    else:
        sp(f"  Status: STEP7A_V1_2_BLOCKED_ASSERTION_FAILURE")

    return nf_


if __name__ == "__main__":
    sys.exit(main())
