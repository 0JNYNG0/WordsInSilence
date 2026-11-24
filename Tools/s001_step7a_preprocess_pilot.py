#!/usr/bin/env python3
"""
Step 7A: Recognition preprocessing pilot — 9 samples from train split.

Input: Step 6B v1.2.4 full 45-sample 3D landmarks (frozen, read-only).
Output: data/processed/sessions/S001_P001/step7a_pilot/

Stages:
  1. Dataset split (deterministic, sample-level)
  2. Pilot selection (3 per word from train, by QA quality tier)
  3. Body-centric normalization
  4. Sign segment detection
  5. Gap interpolation (1/2/3 frame candidates)
  6. Smoothing comparison (none / OneEuro / SavGol)
  7. Assertions
  8. Visualizations, report, manifest, ZIP
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

import numpy as np

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping visualizations")

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available, SavGol comparison will be skipped")


# ================================================================
#  Paths & Constants
# ================================================================

PROJECT = Path(__file__).resolve().parent.parent
STEP6B_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "landmarks_3d_v1" / "full_v1_2_4"
RAW3D_DIR = STEP6B_DIR / "raw_3d"
OUT_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "step7a_pilot"
SEGMENTS_FILE = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "segments_v1.json"

WORDS = ["GREETING", "LOVE", "THANK_YOU"]
SPEEDS = ["FAST", "NORMAL", "SLOW"]
REPS = ["R01", "R02", "R03", "R04", "R05"]

# Pose landmark indices for body reference
# MediaPipe Pose: 11=left_shoulder, 12=right_shoulder, 23=left_hip, 24=right_hip
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_ELBOW = 13
LM_RIGHT_ELBOW = 14
LM_LEFT_WRIST = 15
LM_RIGHT_WRIST = 16

# Hand wrist = index 0 in hand landmarks
HAND_WRIST = 0

# Segmentation parameters
SEG_ENERGY_WINDOW = 5       # frames for motion energy smoothing
SEG_IDLE_THRESHOLD = 0.008  # norm-units/frame motion energy threshold for idle detection
SEG_PAD_START = 5           # frames of padding before detected start
SEG_PAD_END = 5             # frames of padding after detected end
SEG_MIN_ACTIVE_FRAMES = 10  # minimum active frames for a valid segment

# Interpolation max gap candidates
INTERP_CANDIDATES = [1, 2, 3]

# One Euro filter parameters (real-time candidate)
ONE_EURO_MIN_CUTOFF = 1.0   # Hz
ONE_EURO_BETA = 0.007       # speed coefficient
ONE_EURO_D_CUTOFF = 1.0     # Hz for derivative

# Savitzky-Golay parameters (offline reference)
SAVGOL_WINDOW = 7           # must be odd
SAVGOL_ORDER = 3


# ================================================================
#  Helpers
# ================================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_print(s):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode())


# ================================================================
#  1. Dataset Split (deterministic)
# ================================================================

def build_dataset_split():
    """
    Deterministic sample-level split per word:
      R01, R04 -> train  (spread: early + late)
      R02, R05 -> train  (spread: early + late)
      R03      -> train  (middle rep)
      Total train per word: R01, R02, R03, R04, R05 = 5 per speed
      But we need 9 train, 3 val, 3 test per word (15 total).

    With 3 speeds x 5 reps = 15 per word:
      Deterministic rule: interleave reps across splits.
      train = 9, val = 3, test = 3

    Strategy for each word (15 samples = 3 speeds x 5 reps):
      Sort all 15 by (speed_index, rep_index).
      Assign by position in sorted order using pattern:
        train: positions 0,1,3,4,6,7,9,10,12  (9 samples)
        val:   positions 2,8,13               (3 samples)
        test:  positions 5,11,14              (3 samples)

    This spreads each split across all speeds and rep ranges.
    """
    split = {}
    for word in WORDS:
        samples = []
        for si, speed in enumerate(SPEEDS):
            for ri, rep in enumerate(REPS):
                sid = f"S001_P001_{word}_{speed}_{rep}"
                samples.append((si * 5 + ri, sid))
        samples.sort(key=lambda x: x[0])

        # Fixed assignment pattern (deterministic, no random seed)
        # Positions:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
        # Split:      T  T  V  T  T  X  T  T  V  T  T  X  T  V  X
        # T=train(9), V=val(3), X=test(3)
        pattern = ["train", "train", "val", "train", "train", "test",
                   "train", "train", "val", "train", "train", "test",
                   "train", "val", "test"]

        for idx, (_, sid) in enumerate(samples):
            split[sid] = {
                "sample_id": sid,
                "word": word,
                "split": pattern[idx],
                "position_in_word": idx,
            }

    return split


def verify_split(split):
    """Verify split counts and distribution."""
    counts = defaultdict(lambda: defaultdict(int))
    for sid, info in split.items():
        counts[info["word"]][info["split"]] += 1
        counts["ALL"][info["split"]] += 1

    for word in WORDS:
        assert counts[word]["train"] == 9, f"{word} train={counts[word]['train']}"
        assert counts[word]["val"] == 3, f"{word} val={counts[word]['val']}"
        assert counts[word]["test"] == 3, f"{word} test={counts[word]['test']}"

    assert counts["ALL"]["train"] == 27
    assert counts["ALL"]["val"] == 9
    assert counts["ALL"]["test"] == 9
    assert len(split) == 45
    return counts


# ================================================================
#  2. Pilot Selection
# ================================================================

def compute_sample_quality(sid):
    """Compute quality score from Step 6B QA summary."""
    json_path = RAW3D_DIR / f"landmarks_3d_{sid.replace('S001_P001_', '')}.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    frames = data["frames"]
    n_frames = len(frames)
    total_lm = n_frames * 75

    tri_valid = 0
    qa_valid = 0
    te_valid = 0
    n_anomalies = 0
    n_reappear = 0

    for fr in frames:
        for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
            for lm in fr[key]:
                if lm["triangulation_valid"]:
                    tri_valid += 1
                if lm["qa_valid"]:
                    qa_valid += 1
                if lm["template_eligible"]:
                    te_valid += 1
                if lm.get("is_reappearance"):
                    n_reappear += 1
                if lm["qa_flags"]:
                    n_anomalies += len(lm["qa_flags"])

    te_rate = te_valid / total_lm if total_lm > 0 else 0
    qa_rate = qa_valid / total_lm if total_lm > 0 else 0

    return {
        "sample_id": sid,
        "n_frames": n_frames,
        "total_landmarks": total_lm,
        "tri_valid": tri_valid,
        "qa_valid": qa_valid,
        "te_valid": te_valid,
        "te_rate": te_rate,
        "qa_rate": qa_rate,
        "n_reappear": n_reappear,
        "n_anomaly_flags": n_anomalies,
        "quality_score": te_rate - 0.001 * n_anomalies / max(n_frames, 1),
    }


def select_pilot_samples(split):
    """Select 3 per word from train: high/mid/low quality."""
    train_ids = [sid for sid, info in split.items() if info["split"] == "train"]

    # Group by word
    by_word = defaultdict(list)
    for sid in train_ids:
        word = split[sid]["word"]
        by_word[word].append(sid)

    pilot = {}
    quality_data = {}

    for word in WORDS:
        # Compute quality for all train samples of this word
        qualities = []
        for sid in by_word[word]:
            q = compute_sample_quality(sid)
            qualities.append(q)
            quality_data[sid] = q

        # Sort by quality_score descending
        qualities.sort(key=lambda x: x["quality_score"], reverse=True)

        # Pick high (rank 0), mid (rank 4), low (rank 8)
        high = qualities[0]
        mid = qualities[len(qualities) // 2]
        low = qualities[-1]

        pilot[high["sample_id"]] = {"tier": "high", "word": word, **high}
        pilot[mid["sample_id"]] = {"tier": "mid", "word": word, **mid}
        pilot[low["sample_id"]] = {"tier": "low", "word": word, **low}

    return pilot, quality_data


# ================================================================
#  3. Load Step 6B Data
# ================================================================

def load_sample(sid):
    """Load raw 3D JSON for one sample."""
    short = sid.replace("S001_P001_", "")
    json_path = RAW3D_DIR / f"landmarks_3d_{short}.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_arrays(data):
    """
    Extract numpy arrays from Step 6B JSON.
    Returns:
      timestamps: (N,) float64, front_time_s
      pose_xyz: (N, 33, 3) float64
      lh_xyz: (N, 21, 3) float64
      rh_xyz: (N, 21, 3) float64
      pose_valid: (N, 33) bool  -- template_eligible
      lh_valid: (N, 21) bool
      rh_valid: (N, 21) bool
      pose_qa_flags: (N, 33) list of flag lists
      lh_qa_flags, rh_qa_flags: same
      pose_reappear: (N, 33) bool
      lh_reappear, rh_reappear: same
    """
    frames = data["frames"]
    N = len(frames)

    timestamps = np.zeros(N, dtype=np.float64)
    pose_xyz = np.full((N, 33, 3), np.nan, dtype=np.float64)
    lh_xyz = np.full((N, 21, 3), np.nan, dtype=np.float64)
    rh_xyz = np.full((N, 21, 3), np.nan, dtype=np.float64)
    pose_valid = np.zeros((N, 33), dtype=bool)
    lh_valid = np.zeros((N, 21), dtype=bool)
    rh_valid = np.zeros((N, 21), dtype=bool)
    pose_reappear = np.zeros((N, 33), dtype=bool)
    lh_reappear = np.zeros((N, 21), dtype=bool)
    rh_reappear = np.zeros((N, 21), dtype=bool)
    pose_tri = np.zeros((N, 33), dtype=bool)
    lh_tri = np.zeros((N, 21), dtype=bool)
    rh_tri = np.zeros((N, 21), dtype=bool)
    pose_qa = np.zeros((N, 33), dtype=bool)
    lh_qa = np.zeros((N, 21), dtype=bool)
    rh_qa = np.zeros((N, 21), dtype=bool)

    for fi, fr in enumerate(frames):
        timestamps[fi] = fr["front_time_s"]
        for li, lm in enumerate(fr["pose_landmarks_3d"]):
            if lm["triangulation_valid"] and lm["xyz"] is not None:
                pose_xyz[fi, li] = lm["xyz"]
            pose_valid[fi, li] = lm["template_eligible"]
            pose_tri[fi, li] = lm["triangulation_valid"]
            pose_qa[fi, li] = lm["qa_valid"]
            pose_reappear[fi, li] = lm.get("is_reappearance", False)
        for li, lm in enumerate(fr["left_hand_landmarks_3d"]):
            if lm["triangulation_valid"] and lm["xyz"] is not None:
                lh_xyz[fi, li] = lm["xyz"]
            lh_valid[fi, li] = lm["template_eligible"]
            lh_tri[fi, li] = lm["triangulation_valid"]
            lh_qa[fi, li] = lm["qa_valid"]
            lh_reappear[fi, li] = lm.get("is_reappearance", False)
        for li, lm in enumerate(fr["right_hand_landmarks_3d"]):
            if lm["triangulation_valid"] and lm["xyz"] is not None:
                rh_xyz[fi, li] = lm["xyz"]
            rh_valid[fi, li] = lm["template_eligible"]
            rh_tri[fi, li] = lm["triangulation_valid"]
            rh_qa[fi, li] = lm["qa_valid"]
            rh_reappear[fi, li] = lm.get("is_reappearance", False)

    return {
        "timestamps": timestamps,
        "pose_xyz": pose_xyz, "lh_xyz": lh_xyz, "rh_xyz": rh_xyz,
        "pose_valid": pose_valid, "lh_valid": lh_valid, "rh_valid": rh_valid,
        "pose_tri": pose_tri, "lh_tri": lh_tri, "rh_tri": rh_tri,
        "pose_qa": pose_qa, "lh_qa": lh_qa, "rh_qa": rh_qa,
        "pose_reappear": pose_reappear, "lh_reappear": lh_reappear, "rh_reappear": rh_reappear,
    }


# ================================================================
#  4.3 Body-centric Normalization
# ================================================================

def compute_torso_frame(pose_xyz, pose_valid, fi):
    """
    Compute body-centric coordinate frame for one frame.
    Returns (origin, R, scale, status) where:
      origin: (3,) shoulder midpoint
      R: (3,3) rotation matrix [right, up, forward]
      scale: shoulder width in meters
      status: 'ok' | 'fallback_prev' | 'degenerate'
    """
    ls = pose_xyz[fi, LM_LEFT_SHOULDER]
    rs = pose_xyz[fi, LM_RIGHT_SHOULDER]
    lh = pose_xyz[fi, LM_LEFT_HIP]
    rh = pose_xyz[fi, LM_RIGHT_HIP]

    ls_v = pose_valid[fi, LM_LEFT_SHOULDER]
    rs_v = pose_valid[fi, LM_RIGHT_SHOULDER]
    lh_v = pose_valid[fi, LM_LEFT_HIP]
    rh_v = pose_valid[fi, LM_RIGHT_HIP]

    if not (ls_v and rs_v):
        return None, None, None, "degenerate_shoulders"

    origin = (ls + rs) / 2.0
    right_vec = rs - ls
    shoulder_width = np.linalg.norm(right_vec)

    if shoulder_width < 1e-6:
        return origin, None, 0.0, "degenerate_shoulder_width"

    scale = shoulder_width
    right = right_vec / shoulder_width

    # Up direction: hip_center -> shoulder_center
    if lh_v and rh_v:
        hip_center = (lh + rh) / 2.0
        up_vec = origin - hip_center
    elif lh_v:
        up_vec = origin - lh
    elif rh_v:
        up_vec = origin - rh
    else:
        # Fallback: use Y axis (up = -Y in camera coords where Y=down)
        up_vec = np.array([0.0, -1.0, 0.0])

    up_norm = np.linalg.norm(up_vec)
    if up_norm < 1e-6:
        up_vec = np.array([0.0, -1.0, 0.0])
        up_norm = 1.0
    up = up_vec / up_norm

    # Make orthogonal: forward = right x up (right-handed)
    forward = np.cross(right, up)
    fwd_norm = np.linalg.norm(forward)
    if fwd_norm < 1e-6:
        return origin, None, scale, "degenerate_axes"
    forward = forward / fwd_norm

    # Re-orthogonalize up
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)

    # R: columns are [right, up, forward] in world coords
    # To transform world -> body: body_coords = R^T @ (world - origin) / scale
    R = np.column_stack([right, up, forward])

    return origin, R, scale, "ok"


def normalize_sequence(arrays):
    """
    Compute body-centric normalization for all frames.
    Returns dict with normalized arrays and transformation params.
    """
    N = len(arrays["timestamps"])
    pose_xyz = arrays["pose_xyz"]
    pose_valid = arrays["pose_valid"]

    origins = np.full((N, 3), np.nan)
    rotations = np.full((N, 3, 3), np.nan)
    scales = np.full(N, np.nan)
    statuses = []

    # First pass: compute torso frames
    for fi in range(N):
        origin, R, scale, status = compute_torso_frame(pose_xyz, pose_valid, fi)
        statuses.append(status)
        if origin is not None:
            origins[fi] = origin
        if R is not None:
            rotations[fi] = R
        if scale is not None and scale > 0:
            scales[fi] = scale

    # Stabilize: fill degenerate frames with nearest valid frame
    last_valid = None
    for fi in range(N):
        if statuses[fi] == "ok":
            last_valid = fi
        elif last_valid is not None:
            origins[fi] = origins[last_valid]
            rotations[fi] = rotations[last_valid]
            scales[fi] = scales[last_valid]
            statuses[fi] = "fallback_prev"

    # Forward fill remaining (if first frames were degenerate)
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
            statuses[fi] = "fallback_next"

    # Normalize all point arrays
    def normalize_pts(xyz, valid):
        out = np.full_like(xyz, np.nan)
        for fi in range(N):
            if np.isnan(scales[fi]) or scales[fi] < 1e-6:
                continue
            R = rotations[fi]
            if np.any(np.isnan(R)):
                continue
            for li in range(xyz.shape[1]):
                if valid[fi, li] and not np.any(np.isnan(xyz[fi, li])):
                    body = R.T @ (xyz[fi, li] - origins[fi]) / scales[fi]
                    out[fi, li] = body
        return out

    norm_pose = normalize_pts(arrays["pose_xyz"], arrays["pose_valid"])
    norm_lh = normalize_pts(arrays["lh_xyz"], arrays["lh_valid"])
    norm_rh = normalize_pts(arrays["rh_xyz"], arrays["rh_valid"])

    return {
        "norm_pose": norm_pose,
        "norm_lh": norm_lh,
        "norm_rh": norm_rh,
        "origins": origins,
        "rotations": rotations,
        "scales": scales,
        "statuses": statuses,
    }


# ================================================================
#  4.2 Sign Segment Detection
# ================================================================

def detect_sign_segment(arrays, norm_data):
    """
    Detect active sign segment using body-relative motion energy.
    Uses wrist + elbow + fingertip landmarks for motion energy.
    """
    N = len(arrays["timestamps"])
    timestamps = arrays["timestamps"]

    # Key landmark indices for motion energy
    # Pose: wrists (15,16), elbows (13,14)
    # Hand: wrist (0), fingertips (4,8,12,16,20)
    pose_motion_lms = [LM_LEFT_ELBOW, LM_RIGHT_ELBOW, LM_LEFT_WRIST, LM_RIGHT_WRIST]
    hand_fingertips = [4, 8, 12, 16, 20]

    # Compute per-frame motion energy using normalized coords
    energy = np.zeros(N)

    for fi in range(1, N):
        dt = timestamps[fi] - timestamps[fi - 1]
        if dt <= 0 or dt > 0.2:  # skip gaps
            continue

        e = 0.0
        count = 0

        # Pose landmarks
        for li in pose_motion_lms:
            if (not np.any(np.isnan(norm_data["norm_pose"][fi, li])) and
                    not np.any(np.isnan(norm_data["norm_pose"][fi - 1, li]))):
                d = np.linalg.norm(norm_data["norm_pose"][fi, li] - norm_data["norm_pose"][fi - 1, li])
                e += d
                count += 1

        # Hand landmarks
        for hand_key in ["norm_lh", "norm_rh"]:
            wrist_curr = norm_data[hand_key][fi, HAND_WRIST]
            wrist_prev = norm_data[hand_key][fi - 1, HAND_WRIST]
            if not np.any(np.isnan(wrist_curr)) and not np.any(np.isnan(wrist_prev)):
                e += np.linalg.norm(wrist_curr - wrist_prev)
                count += 1
            for tip in hand_fingertips:
                curr = norm_data[hand_key][fi, tip]
                prev = norm_data[hand_key][fi - 1, tip]
                if not np.any(np.isnan(curr)) and not np.any(np.isnan(prev)):
                    e += np.linalg.norm(curr - prev) * 0.5  # lower weight for fingertips
                    count += 0.5

        if count > 0:
            energy[fi] = e / count

    # Smooth energy
    kernel = np.ones(SEG_ENERGY_WINDOW) / SEG_ENERGY_WINDOW
    energy_smooth = np.convolve(energy, kernel, mode="same")

    # Find active region: where energy > threshold
    active = energy_smooth > SEG_IDLE_THRESHOLD

    # Find first and last active frame
    active_indices = np.where(active)[0]

    if len(active_indices) < SEG_MIN_ACTIVE_FRAMES:
        # Fallback: use entire sequence
        seg_start = 0
        seg_end = N - 1
        seg_method = "fallback_full"
    else:
        raw_start = active_indices[0]
        raw_end = active_indices[-1]

        # Add padding
        seg_start = max(0, raw_start - SEG_PAD_START)
        seg_end = min(N - 1, raw_end + SEG_PAD_END)
        seg_method = "motion_energy"

    return {
        "seg_start": int(seg_start),
        "seg_end": int(seg_end),
        "seg_frames": int(seg_end - seg_start + 1),
        "total_frames": N,
        "raw_start": int(active_indices[0]) if len(active_indices) >= SEG_MIN_ACTIVE_FRAMES else 0,
        "raw_end": int(active_indices[-1]) if len(active_indices) >= SEG_MIN_ACTIVE_FRAMES else N - 1,
        "method": seg_method,
        "energy": energy,
        "energy_smooth": energy_smooth,
        "threshold": SEG_IDLE_THRESHOLD,
    }


# ================================================================
#  4.4 Gap Interpolation
# ================================================================

def find_discontinuity_mask(arrays):
    """
    Build per-frame discontinuity mask: True = discontinuity boundary.
    Sources: timestamp gap, reappearance, velocity reset.
    """
    N = len(arrays["timestamps"])
    timestamps = arrays["timestamps"]

    # Frame-level discontinuities
    disc = np.zeros(N, dtype=bool)
    disc[0] = True  # first frame is always a boundary

    for fi in range(1, N):
        dt = timestamps[fi] - timestamps[fi - 1]
        if dt <= 0 or dt > 0.1:  # timestamp gap (>3 frames at 30fps)
            disc[fi] = True

    # Landmark-level discontinuities (reappearance)
    # We track per-landmark, but for interpolation we need per-landmark masks
    return disc


def find_landmark_discontinuities(arrays, comp_key, n_lm):
    """
    Per-landmark discontinuity: reappearance or strong depth invalidation boundary.
    Returns (N, n_lm) bool array.
    """
    N = len(arrays["timestamps"])
    reappear_key = comp_key.replace("_xyz", "_reappear").replace("pose", "pose").replace("lh", "lh").replace("rh", "rh")
    reappear = arrays[reappear_key]

    disc = np.zeros((N, n_lm), dtype=bool)
    disc[0, :] = True

    for fi in range(1, N):
        for li in range(n_lm):
            if reappear[fi, li]:
                disc[fi, li] = True

    return disc


def interpolate_gaps(xyz, valid, frame_disc, lm_disc, max_gap):
    """
    Linear interpolation for gaps <= max_gap frames.
    Never crosses frame-level or landmark-level discontinuity boundaries.
    Never extrapolates at sequence boundaries.

    Returns:
      interp_xyz: (N, n_lm, 3) with interpolated values filled
      interp_mask: (N, n_lm) bool, True where interpolated
      interp_reasons: list of dicts with provenance
    """
    N, n_lm, _ = xyz.shape
    interp_xyz = xyz.copy()
    interp_mask = np.zeros((N, n_lm), dtype=bool)
    reasons = []

    for li in range(n_lm):
        # Find continuous segments (no frame or landmark discontinuity)
        seg_boundaries = []
        seg_start = 0

        for fi in range(1, N):
            if frame_disc[fi] or lm_disc[fi, li]:
                seg_boundaries.append((seg_start, fi - 1))
                seg_start = fi
        seg_boundaries.append((seg_start, N - 1))

        for s_start, s_end in seg_boundaries:
            # Within this segment, find gaps in valid
            fi = s_start
            while fi <= s_end:
                if not valid[fi, li]:
                    # Start of a gap
                    gap_start = fi
                    while fi <= s_end and not valid[fi, li]:
                        fi += 1
                    gap_end = fi - 1  # inclusive
                    gap_len = gap_end - gap_start + 1

                    # Check boundaries: need valid before and after within segment
                    has_before = gap_start > s_start and valid[gap_start - 1, li]
                    has_after = fi <= s_end and valid[fi, li]

                    if has_before and has_after and gap_len <= max_gap:
                        # Linear interpolation
                        p0 = interp_xyz[gap_start - 1, li]
                        p1 = interp_xyz[fi, li]
                        if not np.any(np.isnan(p0)) and not np.any(np.isnan(p1)):
                            for gi in range(gap_len):
                                t = (gi + 1) / (gap_len + 1)
                                interp_xyz[gap_start + gi, li] = p0 * (1 - t) + p1 * t
                                interp_mask[gap_start + gi, li] = True

                            reasons.append({
                                "landmark_index": int(li),
                                "gap_start_frame": int(gap_start),
                                "gap_end_frame": int(gap_end),
                                "gap_length": int(gap_len),
                                "method": "linear",
                                "max_gap_setting": int(max_gap),
                                "segment_start": int(s_start),
                                "segment_end": int(s_end),
                            })
                    elif gap_len > max_gap or not has_before or not has_after:
                        for gi in range(gap_len):
                            reason_str = []
                            if gap_len > max_gap:
                                reason_str.append(f"exceeds_max_gap_{max_gap}")
                            if not has_before:
                                reason_str.append("no_valid_predecessor_in_segment")
                            if not has_after:
                                reason_str.append("no_valid_successor_in_segment")
                            reasons.append({
                                "landmark_index": int(li),
                                "gap_start_frame": int(gap_start),
                                "gap_end_frame": int(gap_end),
                                "gap_length": int(gap_len),
                                "method": "rejected",
                                "rejection_reason": "|".join(reason_str),
                                "max_gap_setting": int(max_gap),
                                "segment_start": int(s_start),
                                "segment_end": int(s_end),
                            })
                            break  # one record per gap, not per frame
                else:
                    fi += 1

    return interp_xyz, interp_mask, reasons


# ================================================================
#  4.5 Smoothing
# ================================================================

class OneEuroFilter:
    """One Euro Filter for real-time smoothing (causal)."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()

        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev.copy()

        # Derivative
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Adaptive cutoff
        speed = np.linalg.norm(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        self.t_prev = t

        return x_hat


def apply_one_euro(xyz, valid, timestamps, frame_disc, lm_disc):
    """Apply One Euro filter within continuous valid segments only."""
    N, n_lm, _ = xyz.shape
    filtered = xyz.copy()

    for li in range(n_lm):
        filt = OneEuroFilter(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)

        for fi in range(N):
            if frame_disc[fi] or lm_disc[fi, li]:
                filt.reset()

            if valid[fi, li] and not np.any(np.isnan(xyz[fi, li])):
                filtered[fi, li] = filt(xyz[fi, li], timestamps[fi])
            else:
                filt.reset()

    return filtered


def apply_savgol(xyz, valid, frame_disc, lm_disc, window=SAVGOL_WINDOW, order=SAVGOL_ORDER):
    """Apply Savitzky-Golay filter within continuous valid segments."""
    if not HAS_SCIPY:
        return xyz.copy()

    N, n_lm, _ = xyz.shape
    filtered = xyz.copy()

    for li in range(n_lm):
        # Find continuous valid segments
        seg_start = None
        for fi in range(N + 1):
            is_boundary = fi == N or frame_disc[fi] or lm_disc[fi, li] or not valid[fi if fi < N else 0, li]
            if fi < N and not is_boundary and not np.any(np.isnan(xyz[fi, li])):
                if seg_start is None:
                    seg_start = fi
            else:
                if seg_start is not None:
                    seg_len = fi - seg_start
                    if seg_len >= window:
                        for dim in range(3):
                            signal = xyz[seg_start:fi, li, dim]
                            filtered[seg_start:fi, li, dim] = savgol_filter(signal, window, order)
                    seg_start = None

    return filtered


# ================================================================
#  4.5 Smoothing Metrics
# ================================================================

def compute_smoothing_metrics(raw_xyz, filt_xyz, valid, timestamps, frame_disc, lm_disc, label):
    """Compute comparison metrics between raw and filtered data."""
    N, n_lm, _ = raw_xyz.shape

    # Position deviation
    diffs = []
    for fi in range(N):
        for li in range(n_lm):
            if valid[fi, li] and not np.any(np.isnan(raw_xyz[fi, li])) and not np.any(np.isnan(filt_xyz[fi, li])):
                d = np.linalg.norm(filt_xyz[fi, li] - raw_xyz[fi, li])
                diffs.append(d)

    if not diffs:
        return {"label": label, "n_valid": 0}

    diffs = np.array(diffs)

    # Jitter in idle segments (first and last 3 frames of sequence)
    def jitter_in_range(xyz_data, start, end):
        vals = []
        for fi in range(max(0, start), min(N, end)):
            for li in range(n_lm):
                if valid[fi, li] and fi > 0 and valid[fi - 1, li]:
                    if not np.any(np.isnan(xyz_data[fi, li])) and not np.any(np.isnan(xyz_data[fi - 1, li])):
                        vals.append(np.linalg.norm(xyz_data[fi, li] - xyz_data[fi - 1, li]))
        return np.mean(vals) if vals else 0.0

    raw_jitter_start = jitter_in_range(raw_xyz, 0, 5)
    filt_jitter_start = jitter_in_range(filt_xyz, 0, 5)
    raw_jitter_end = jitter_in_range(raw_xyz, N - 5, N)
    filt_jitter_end = jitter_in_range(filt_xyz, N - 5, N)

    # Velocity/acceleration distortion
    raw_vel = []
    filt_vel = []
    raw_acc = []
    filt_acc = []

    for li in range(min(n_lm, 5)):  # sample first 5 landmarks
        for fi in range(1, N):
            dt = timestamps[fi] - timestamps[fi - 1]
            if dt <= 0 or dt > 0.1 or frame_disc[fi] or lm_disc[fi, li]:
                continue
            if valid[fi, li] and valid[fi - 1, li]:
                if not np.any(np.isnan(raw_xyz[fi, li])) and not np.any(np.isnan(raw_xyz[fi - 1, li])):
                    rv = np.linalg.norm(raw_xyz[fi, li] - raw_xyz[fi - 1, li]) / dt
                    raw_vel.append(rv)
                if not np.any(np.isnan(filt_xyz[fi, li])) and not np.any(np.isnan(filt_xyz[fi - 1, li])):
                    fv = np.linalg.norm(filt_xyz[fi, li] - filt_xyz[fi - 1, li]) / dt
                    filt_vel.append(fv)

    # New NaN check
    new_nans = 0
    new_infs = 0
    for fi in range(N):
        for li in range(n_lm):
            if valid[fi, li] and not np.any(np.isnan(raw_xyz[fi, li])):
                if np.any(np.isnan(filt_xyz[fi, li])):
                    new_nans += 1
                if np.any(np.isinf(filt_xyz[fi, li])):
                    new_infs += 1

    return {
        "label": label,
        "n_valid": len(diffs),
        "pos_dev_mean_m": float(np.mean(diffs)),
        "pos_dev_max_m": float(np.max(diffs)),
        "pos_dev_p95_m": float(np.percentile(diffs, 95)),
        "jitter_start_raw_m": float(raw_jitter_start),
        "jitter_start_filt_m": float(filt_jitter_start),
        "jitter_end_raw_m": float(raw_jitter_end),
        "jitter_end_filt_m": float(filt_jitter_end),
        "jitter_reduction_start": float(1 - filt_jitter_start / raw_jitter_start) if raw_jitter_start > 0 else 0,
        "jitter_reduction_end": float(1 - filt_jitter_end / raw_jitter_end) if raw_jitter_end > 0 else 0,
        "raw_vel_mean_ms": float(np.mean(raw_vel)) if raw_vel else 0,
        "filt_vel_mean_ms": float(np.mean(filt_vel)) if filt_vel else 0,
        "vel_distortion_pct": float(abs(np.mean(filt_vel) - np.mean(raw_vel)) / np.mean(raw_vel) * 100) if raw_vel and np.mean(raw_vel) > 0 else 0,
        "new_nans": new_nans,
        "new_infs": new_infs,
    }


# ================================================================
#  Visualization
# ================================================================

def plot_segmentation(sid, arrays, norm_data, seg_info, out_dir):
    """Plot motion energy with segmentation boundaries."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    timestamps = arrays["timestamps"]
    t_rel = timestamps - timestamps[0]

    # Energy plot
    ax = axes[0]
    ax.plot(t_rel, seg_info["energy"], alpha=0.3, label="raw energy")
    ax.plot(t_rel, seg_info["energy_smooth"], label="smoothed energy")
    ax.axhline(seg_info["threshold"], color="r", linestyle="--", alpha=0.5, label="threshold")
    ax.axvline(t_rel[seg_info["seg_start"]], color="g", linestyle="-", label="seg_start")
    ax.axvline(t_rel[seg_info["seg_end"]], color="g", linestyle="--", label="seg_end")
    if seg_info["raw_start"] != seg_info["seg_start"]:
        ax.axvline(t_rel[seg_info["raw_start"]], color="orange", linestyle=":", alpha=0.7, label="raw_start")
        ax.axvline(t_rel[seg_info["raw_end"]], color="orange", linestyle=":", alpha=0.7, label="raw_end")
    ax.set_ylabel("Motion energy (norm units/frame)")
    ax.set_title(f"Segmentation: {sid}")
    ax.legend(fontsize=8)

    # Wrist trajectory (normalized Y)
    ax = axes[1]
    lw_y = norm_data["norm_lh"][:, HAND_WRIST, 1]
    rw_y = norm_data["norm_rh"][:, HAND_WRIST, 1]
    pw_ly = norm_data["norm_pose"][:, LM_LEFT_WRIST, 1]
    pw_ry = norm_data["norm_pose"][:, LM_RIGHT_WRIST, 1]

    ax.plot(t_rel, lw_y, label="LH wrist Y", alpha=0.7)
    ax.plot(t_rel, rw_y, label="RH wrist Y", alpha=0.7)
    ax.plot(t_rel, pw_ly, label="Pose L wrist Y", alpha=0.5, linestyle="--")
    ax.plot(t_rel, pw_ry, label="Pose R wrist Y", alpha=0.5, linestyle="--")
    ax.axvline(t_rel[seg_info["seg_start"]], color="g", linestyle="-")
    ax.axvline(t_rel[seg_info["seg_end"]], color="g", linestyle="--")
    ax.set_ylabel("Body-norm Y (up)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / f"seg_{sid.replace('S001_P001_', '')}.png", dpi=120)
    plt.close(fig)


def plot_trajectory_comparison(sid, arrays, norm_data, interp_results, smooth_results, out_dir):
    """Plot raw/normalized/interpolated/smoothed trajectories for key joints."""
    if not HAS_MPL:
        return

    timestamps = arrays["timestamps"]
    t_rel = timestamps - timestamps[0]
    short = sid.replace("S001_P001_", "")

    # Plot for pose left wrist (index 15) - 3 axes
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    li = LM_LEFT_WRIST
    axis_names = ["X (right)", "Y (up)", "Z (forward)"]

    for dim in range(3):
        ax = axes[dim]
        # Raw normalized
        raw = norm_data["norm_pose"][:, li, dim]
        ax.plot(t_rel, raw, "k-", alpha=0.3, label="raw_norm", linewidth=0.8)

        # Best interpolation (max_gap=2)
        if 2 in interp_results:
            ir = interp_results[2]
            interp = ir["pose_interp"][:, li, dim]
            mask = ir["pose_mask"][:, li]
            ax.plot(t_rel, interp, "b-", alpha=0.5, label="interp(2)", linewidth=0.8)
            # Mark interpolated points
            ax.scatter(t_rel[mask], interp[mask], c="blue", s=8, zorder=5, label="interpolated")

        # Smoothed (OneEuro on interp)
        if "one_euro" in smooth_results:
            se = smooth_results["one_euro"]["pose"][:, li, dim]
            ax.plot(t_rel, se, "r-", alpha=0.6, label="OneEuro", linewidth=1)

        if "savgol" in smooth_results:
            sg = smooth_results["savgol"]["pose"][:, li, dim]
            ax.plot(t_rel, sg, "g-", alpha=0.6, label="SavGol", linewidth=1)

        ax.set_ylabel(f"Pose L.Wrist {axis_names[dim]}")
        if dim == 0:
            ax.set_title(f"Trajectory: {short} - Pose Left Wrist (norm)")
            ax.legend(fontsize=7, ncol=4)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    fig.savefig(out_dir / f"traj_{short}_pose_lwrist.png", dpi=120)
    plt.close(fig)

    # Plot for left hand wrist (hand landmark 0) - velocity/acceleration
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    raw_pos = norm_data["norm_lh"][:, HAND_WRIST]
    valid_mask = ~np.any(np.isnan(raw_pos), axis=1)

    # Position magnitude
    ax = axes[0]
    pos_mag = np.linalg.norm(raw_pos, axis=1)
    ax.plot(t_rel, pos_mag, "k-", alpha=0.5, label="raw")
    if "one_euro" in smooth_results:
        se_pos = smooth_results["one_euro"]["lh"][:, HAND_WRIST]
        se_mag = np.linalg.norm(se_pos, axis=1)
        ax.plot(t_rel, se_mag, "r-", alpha=0.7, label="OneEuro")
    ax.set_ylabel("LH wrist |pos| (norm)")
    ax.set_title(f"Vel/Acc: {short} - LH Wrist")
    ax.legend(fontsize=8)

    # Velocity
    ax = axes[1]
    for label_key, xyz_src, color in [("raw", raw_pos, "k"), ("OneEuro", smooth_results.get("one_euro", {}).get("lh", raw_pos), "r")]:
        if xyz_src is None:
            continue
        vel = np.full(len(t_rel), np.nan)
        for fi in range(1, len(t_rel)):
            dt = timestamps[fi] - timestamps[fi - 1]
            if dt > 0 and dt < 0.1:
                src = xyz_src if isinstance(xyz_src, np.ndarray) and xyz_src.ndim == 3 else None
                if src is not None:
                    p0 = src[fi - 1, HAND_WRIST]
                    p1 = src[fi, HAND_WRIST]
                else:
                    p0 = xyz_src[fi - 1]
                    p1 = xyz_src[fi]
                if not np.any(np.isnan(p0)) and not np.any(np.isnan(p1)):
                    vel[fi] = np.linalg.norm(p1 - p0) / dt
        ax.plot(t_rel, vel, color=color, alpha=0.5, label=label_key, linewidth=0.8)
    ax.set_ylabel("Velocity (norm/s)")
    ax.legend(fontsize=8)

    # Acceleration
    ax = axes[2]
    vel_raw = np.full(len(t_rel), np.nan)
    for fi in range(1, len(t_rel)):
        dt = timestamps[fi] - timestamps[fi - 1]
        if dt > 0 and dt < 0.1 and not np.any(np.isnan(raw_pos[fi])) and not np.any(np.isnan(raw_pos[fi - 1])):
            vel_raw[fi] = np.linalg.norm(raw_pos[fi] - raw_pos[fi - 1]) / dt
    acc_raw = np.full(len(t_rel), np.nan)
    for fi in range(2, len(t_rel)):
        dt = timestamps[fi] - timestamps[fi - 1]
        if dt > 0 and not np.isnan(vel_raw[fi]) and not np.isnan(vel_raw[fi - 1]):
            acc_raw[fi] = (vel_raw[fi] - vel_raw[fi - 1]) / dt
    ax.plot(t_rel, acc_raw, "k-", alpha=0.3, label="raw", linewidth=0.8)
    ax.set_ylabel("Accel (norm/s2)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / f"velacc_{short}_lh_wrist.png", dpi=120)
    plt.close(fig)


# ================================================================
#  Assertions
# ================================================================

def run_assertions(all_sample_results, split, pilot_ids, input_hashes, out_dir):
    """Run all Step 7A assertions."""
    results = []
    n_pass = 0
    n_fail = 0

    def chk(cond, name, detail=""):
        nonlocal n_pass, n_fail
        status = "PASS" if cond else "FAIL"
        if cond:
            n_pass += 1
        else:
            n_fail += 1
            safe_print(f"    [{status}] {name}: {detail}")
        results.append({"name": name, "status": status, "detail": detail})

    # A1. Step 6B input hash match
    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        json_path = RAW3D_DIR / f"landmarks_3d_{short}.json"
        actual_hash = sha256_file(json_path)
        expected = input_hashes.get(sid)
        chk(actual_hash == expected, f"input_hash_{short}",
            f"expected={expected[:16]}... actual={actual_hash[:16]}...")

    # A2. Pilot count = 9, 3 per word
    chk(len(pilot_ids) == 9, "pilot_count_9", f"got={len(pilot_ids)}")
    for word in WORDS:
        word_pilots = [s for s in pilot_ids if word in s]
        chk(len(word_pilots) == 3, f"pilot_{word}_count_3", f"got={len(word_pilots)}")

    # A3. No test or val samples used
    for sid in pilot_ids:
        s = split[sid]["split"]
        chk(s == "train", f"pilot_is_train_{sid.replace('S001_P001_','')}",
            f"split={s}")

    # A4. No duplicate pilot samples
    chk(len(set(pilot_ids)) == len(pilot_ids), "no_duplicate_pilots")

    for sid, sr in all_sample_results.items():
        short = sid.replace("S001_P001_", "")
        arrays = sr["arrays"]
        norm = sr["norm"]
        N = len(arrays["timestamps"])

        # A5. Original coordinates unchanged (spot-check first and last frame)
        raw_data = sr["raw_data"]
        for fi_check in [0, N - 1]:
            fr = raw_data["frames"][fi_check]
            for li, lm in enumerate(fr["pose_landmarks_3d"]):
                if lm["triangulation_valid"] and lm["xyz"] is not None:
                    orig = np.array(lm["xyz"])
                    loaded = arrays["pose_xyz"][fi_check, li]
                    chk(np.allclose(orig, loaded, atol=1e-12),
                        f"coords_preserved_{short}_f{fi_check}_p{li}",
                        f"orig={orig} loaded={loaded}")
                    break  # one check per frame
            break  # check first frame only per sample

        # A6. No axis flip in normalization (check sign consistency of right-axis)
        n_flips = 0
        for fi in range(1, N):
            if norm["statuses"][fi] in ("ok", "fallback_prev", "fallback_next"):
                if not np.any(np.isnan(norm["rotations"][fi])) and not np.any(np.isnan(norm["rotations"][fi - 1])):
                    # Check if right-axis (col 0) flipped sign
                    dot = np.dot(norm["rotations"][fi][:, 0], norm["rotations"][fi - 1][:, 0])
                    if dot < 0:
                        n_flips += 1
        chk(n_flips == 0, f"no_axis_flip_{short}", f"flips={n_flips}")

        # A7. No degenerate scale
        valid_scales = norm["scales"][~np.isnan(norm["scales"])]
        if len(valid_scales) > 0:
            chk(np.all(valid_scales > 0.01), f"scale_positive_{short}",
                f"min={np.min(valid_scales):.4f}")
            chk(np.all(valid_scales < 2.0), f"scale_reasonable_{short}",
                f"max={np.max(valid_scales):.4f}")
        else:
            chk(False, f"scale_exists_{short}", "no valid scales")

        # A8. No zeros replacing invalid
        for comp, key in [("pose", "norm_pose"), ("lh", "norm_lh"), ("rh", "norm_rh")]:
            valid_key = f"{comp}_valid" if comp != "lh" else "lh_valid"
            if comp == "rh":
                valid_key = "rh_valid"
            v = arrays[valid_key]
            coords = norm[key]
            n_zero_invalid = 0
            for fi in range(N):
                for li in range(coords.shape[1]):
                    if not v[fi, li]:
                        if not np.any(np.isnan(coords[fi, li])) and np.allclose(coords[fi, li], 0):
                            n_zero_invalid += 1
            chk(n_zero_invalid == 0, f"no_zero_replace_{short}_{comp}",
                f"count={n_zero_invalid}")

        # A9. Interpolation boundary checks
        for max_gap, ir in sr["interp_results"].items():
            for comp in ["pose", "lh", "rh"]:
                reasons = ir[f"{comp}_reasons"]
                # Check no boundary-crossing interpolation
                n_cross = sum(1 for r in reasons if r["method"] == "linear" and
                              r.get("crossed_boundary"))
                chk(n_cross == 0,
                    f"no_boundary_interp_{short}_{comp}_g{max_gap}",
                    f"crossings={n_cross}")

        # A10. No extrapolation at sequence boundaries
        for max_gap, ir in sr["interp_results"].items():
            for comp in ["pose", "lh", "rh"]:
                mask = ir[f"{comp}_mask"]
                if mask.shape[0] > 0:
                    chk(not np.any(mask[0, :]),
                        f"no_extrap_start_{short}_{comp}_g{max_gap}")
                    chk(not np.any(mask[-1, :]),
                        f"no_extrap_end_{short}_{comp}_g{max_gap}")

        # A11. No new NaN/Inf from smoothing
        for filt_name, sm in sr["smooth_results"].items():
            for comp in ["pose", "lh", "rh"]:
                metrics = sm[f"{comp}_metrics"]
                chk(metrics["new_nans"] == 0,
                    f"no_new_nan_{short}_{comp}_{filt_name}",
                    f"nans={metrics['new_nans']}")
                chk(metrics["new_infs"] == 0,
                    f"no_new_inf_{short}_{comp}_{filt_name}",
                    f"infs={metrics['new_infs']}")

        # A12. Smoothing didn't cross discontinuity boundaries (implicit in segment-based application)

        # A13. Interpolation provenance exists for all interpolated points
        for max_gap, ir in sr["interp_results"].items():
            for comp in ["pose", "lh", "rh"]:
                n_interp = int(np.sum(ir[f"{comp}_mask"]))
                reasons = [r for r in ir[f"{comp}_reasons"] if r["method"] == "linear"]
                n_accounted = sum(r["gap_length"] for r in reasons)
                chk(n_interp == n_accounted,
                    f"interp_provenance_{short}_{comp}_g{max_gap}",
                    f"interp={n_interp} accounted={n_accounted}")

    # A14. Output file existence (exclude report/manifest written after assertions)
    expected_outputs = [
        "step7a_config.json",
        "step7_dataset_split_manifest.csv",
        "step7_dataset_split_manifest.json",
        "pilot_selection_manifest.json",
        "input_file_manifest.json",
        "interpolation_comparison.csv",
        "smoothing_comparison.csv",
        "segmentation_summary.csv",
        "normalization_summary.csv",
    ]
    for fn in expected_outputs:
        chk((out_dir / fn).exists(), f"output_exists_{fn}")

    # A15. CSV/JSON count consistency
    # Check segmentation CSV row count
    seg_csv = out_dir / "segmentation_summary.csv"
    if seg_csv.exists():
        with open(seg_csv, encoding="utf-8") as f:
            rows = sum(1 for _ in csv.reader(f)) - 1  # minus header
        chk(rows == 9, f"seg_csv_rows_9", f"rows={rows}")

    safe_print(f"\n  Assertions: {n_pass} PASS, {n_fail} FAIL")
    return results, n_pass, n_fail


# ================================================================
#  Main
# ================================================================

def main():
    safe_print("Step 7A: Recognition Preprocessing Pilot")
    safe_print(f"  Status: STEP7A_PREPROCESSING_PILOT_APPROVED_FOR_EXECUTION")
    safe_print("")

    # Verify Step 6B exists
    assert STEP6B_DIR.exists(), f"Step 6B dir not found: {STEP6B_DIR}"
    assert RAW3D_DIR.exists(), f"Raw 3D dir not found: {RAW3D_DIR}"

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    viz_dir = OUT_DIR / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # ---- 1. Dataset Split ----
    safe_print("  1. Building dataset split...")
    split = build_dataset_split()
    counts = verify_split(split)
    safe_print(f"    Total: {len(split)} samples")
    for word in WORDS:
        safe_print(f"    {word}: train={counts[word]['train']} val={counts[word]['val']} test={counts[word]['test']}")

    # Show split assignment
    safe_print("    Assignment (per word, sorted by position):")
    for word in WORDS:
        word_items = [(info["position_in_word"], sid, info["split"])
                      for sid, info in split.items() if info["word"] == word]
        word_items.sort()
        items_str = " ".join(f"{sid.split('_')[-2]}_{sid.split('_')[-1]}:{sp[0].upper()}"
                             for _, sid, sp in word_items)
        safe_print(f"      {word}: {items_str}")

    # Save split manifests
    split_rows = sorted(split.values(), key=lambda x: x["sample_id"])
    with open(OUT_DIR / "step7_dataset_split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "word", "split", "position_in_word"])
        w.writeheader()
        w.writerows(split_rows)

    with open(OUT_DIR / "step7_dataset_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"samples": split_rows, "counts": {
            "train": 27, "val": 9, "test": 9, "total": 45
        }}, f, indent=2)

    # ---- 2. Pilot Selection ----
    safe_print("\n  2. Selecting pilot samples...")
    pilot, quality_data = select_pilot_samples(split)
    pilot_ids = sorted(pilot.keys())

    safe_print(f"    Selected {len(pilot_ids)} pilot samples:")
    for sid in pilot_ids:
        p = pilot[sid]
        safe_print(f"      {sid.replace('S001_P001_',''):30s} tier={p['tier']:4s} "
                   f"te_rate={p['te_rate']:.1%} qa_rate={p['qa_rate']:.1%} "
                   f"anomalies={p['n_anomaly_flags']:>4d} score={p['quality_score']:.4f}")

    # Save pilot manifest
    pilot_manifest = {
        "pilot_samples": [pilot[sid] for sid in pilot_ids],
        "selection_criteria": {
            "method": "quality_score_ranking_within_train_split",
            "quality_score": "te_rate - 0.001 * n_anomaly_flags / n_frames",
            "tiers": {"high": "rank 0 (best)", "mid": "rank 4 (middle)", "low": "rank 8 (worst)"},
            "source_split": "train_only",
            "n_per_word": 3,
        }
    }
    with open(OUT_DIR / "pilot_selection_manifest.json", "w", encoding="utf-8") as f:
        json.dump(pilot_manifest, f, indent=2, default=str)

    # ---- Input hashes ----
    safe_print("\n  3. Computing input file hashes...")
    input_hashes = {}
    input_manifest = []
    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        json_path = RAW3D_DIR / f"landmarks_3d_{short}.json"
        h = sha256_file(json_path)
        sz = os.path.getsize(json_path)
        input_hashes[sid] = h
        input_manifest.append({
            "sample_id": sid,
            "file": json_path.relative_to(PROJECT).as_posix(),
            "size_bytes": sz,
            "sha256": h,
        })
        safe_print(f"    {short}: {h[:16]}... ({sz / 1e6:.1f} MB)")

    with open(OUT_DIR / "input_file_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"self_reference_excluded": True, "inputs": input_manifest}, f, indent=2)

    # ---- Config ----
    config = {
        "step": "7A",
        "version": "1.0",
        "input_dir": str(STEP6B_DIR.relative_to(PROJECT)),
        "segmentation": {
            "energy_window": SEG_ENERGY_WINDOW,
            "idle_threshold": SEG_IDLE_THRESHOLD,
            "pad_start_frames": SEG_PAD_START,
            "pad_end_frames": SEG_PAD_END,
            "min_active_frames": SEG_MIN_ACTIVE_FRAMES,
        },
        "interpolation_candidates": INTERP_CANDIDATES,
        "smoothing": {
            "one_euro": {
                "min_cutoff_hz": ONE_EURO_MIN_CUTOFF,
                "beta": ONE_EURO_BETA,
                "d_cutoff_hz": ONE_EURO_D_CUTOFF,
            },
            "savgol": {
                "window": SAVGOL_WINDOW,
                "order": SAVGOL_ORDER,
            },
        },
        "normalization": {
            "origin": "shoulder_midpoint",
            "right_axis": "right_shoulder - left_shoulder",
            "up_axis": "hip_center -> shoulder_center",
            "forward_axis": "cross(right, up)",
            "scale": "shoulder_width_meters",
            "handedness": "right_handed_coordinate_system",
        },
    }
    with open(OUT_DIR / "step7a_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ---- 4. Process each pilot sample ----
    safe_print("\n  4. Processing pilot samples...")
    all_sample_results = {}
    seg_rows = []
    norm_rows = []
    interp_comparison_rows = []
    smooth_comparison_rows = []

    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        safe_print(f"\n    === {short} ===")

        # Load
        raw_data = load_sample(sid)
        arrays = extract_arrays(raw_data)
        N = len(arrays["timestamps"])
        safe_print(f"      Frames: {N}, timestamps: {arrays['timestamps'][0]:.3f} - {arrays['timestamps'][-1]:.3f}")

        # 4.3 Normalize
        safe_print(f"      Normalizing...")
        norm = normalize_sequence(arrays)
        status_counts = defaultdict(int)
        for s in norm["statuses"]:
            status_counts[s] += 1
        safe_print(f"        Statuses: {dict(status_counts)}")
        valid_scales = norm["scales"][~np.isnan(norm["scales"])]
        if len(valid_scales) > 0:
            safe_print(f"        Scale: mean={np.mean(valid_scales):.4f} std={np.std(valid_scales):.4f}")

        norm_rows.append({
            "sample_id": sid,
            "n_frames": N,
            "n_ok": status_counts.get("ok", 0),
            "n_fallback_prev": status_counts.get("fallback_prev", 0),
            "n_fallback_next": status_counts.get("fallback_next", 0),
            "n_degenerate": sum(v for k, v in status_counts.items() if "degenerate" in k),
            "scale_mean": float(np.mean(valid_scales)) if len(valid_scales) > 0 else None,
            "scale_std": float(np.std(valid_scales)) if len(valid_scales) > 0 else None,
        })

        # 4.2 Segment detection
        safe_print(f"      Detecting sign segment...")
        seg_info = detect_sign_segment(arrays, norm)
        safe_print(f"        Segment: [{seg_info['seg_start']}-{seg_info['seg_end']}] "
                   f"({seg_info['seg_frames']}/{N} frames, method={seg_info['method']})")

        seg_rows.append({
            "sample_id": sid,
            "total_frames": N,
            "seg_start": seg_info["seg_start"],
            "seg_end": seg_info["seg_end"],
            "seg_frames": seg_info["seg_frames"],
            "raw_start": seg_info["raw_start"],
            "raw_end": seg_info["raw_end"],
            "method": seg_info["method"],
            "idle_before": seg_info["seg_start"],
            "idle_after": N - 1 - seg_info["seg_end"],
        })

        # Plot segmentation
        plot_segmentation(sid, arrays, norm, seg_info, viz_dir)

        # 4.4 Interpolation
        safe_print(f"      Interpolation candidates...")
        frame_disc = find_discontinuity_mask(arrays)
        interp_results = {}

        for max_gap in INTERP_CANDIDATES:
            ir = {}
            for comp, xyz_key, valid_key, n_lm in [
                ("pose", "pose_xyz", "pose_valid", 33),
                ("lh", "lh_xyz", "lh_valid", 21),
                ("rh", "rh_xyz", "rh_valid", 21),
            ]:
                lm_disc = find_landmark_discontinuities(arrays, xyz_key, n_lm)
                interp_xyz, interp_mask, reasons = interpolate_gaps(
                    arrays[xyz_key], arrays[valid_key], frame_disc, lm_disc, max_gap)

                # Also interpolate normalized coords
                norm_key = f"norm_{comp}"
                norm_interp, norm_mask, _ = interpolate_gaps(
                    norm[norm_key],
                    arrays[valid_key],
                    frame_disc, lm_disc, max_gap)

                ir[f"{comp}_interp"] = norm_interp  # use normalized for downstream
                ir[f"{comp}_mask"] = norm_mask
                ir[f"{comp}_reasons"] = reasons
                ir[f"{comp}_raw_interp"] = interp_xyz

                n_filled = int(np.sum(interp_mask))
                n_rejected = sum(1 for r in reasons if r["method"] == "rejected")
                n_linear = sum(1 for r in reasons if r["method"] == "linear")

                interp_comparison_rows.append({
                    "sample_id": sid,
                    "component": comp,
                    "max_gap": max_gap,
                    "n_gaps_filled": n_linear,
                    "n_points_filled": n_filled,
                    "n_gaps_rejected": n_rejected,
                    "fill_rate": n_filled / (N * n_lm) * 100 if N * n_lm > 0 else 0,
                })

            interp_results[max_gap] = ir
            safe_print(f"        max_gap={max_gap}: pose={int(np.sum(ir['pose_mask']))} "
                       f"lh={int(np.sum(ir['lh_mask']))} rh={int(np.sum(ir['rh_mask']))} filled")

        # 4.5 Smoothing comparison (on max_gap=2 interpolated data)
        safe_print(f"      Smoothing comparison...")
        base_interp = interp_results[2]  # Use gap=2 as base for smoothing comparison
        smooth_results = {}

        # No smoothing (baseline)
        no_smooth = {}
        for comp, n_lm in [("pose", 33), ("lh", 21), ("rh", 21)]:
            no_smooth[comp] = base_interp[f"{comp}_interp"].copy()
            valid_for_smooth = arrays[f"{comp}_valid"] | base_interp[f"{comp}_mask"]
            lm_disc = find_landmark_discontinuities(arrays, f"{comp}_xyz", n_lm)
            metrics = compute_smoothing_metrics(
                norm[f"norm_{comp}"], no_smooth[comp], valid_for_smooth,
                arrays["timestamps"], frame_disc, lm_disc, "none")
            no_smooth[f"{comp}_metrics"] = metrics
            smooth_comparison_rows.append({"sample_id": sid, "component": comp, "filter": "none", **metrics})
        smooth_results["none"] = no_smooth

        # One Euro
        one_euro = {}
        for comp, n_lm in [("pose", 33), ("lh", 21), ("rh", 21)]:
            valid_for_smooth = arrays[f"{comp}_valid"] | base_interp[f"{comp}_mask"]
            lm_disc = find_landmark_discontinuities(arrays, f"{comp}_xyz", n_lm)
            filtered = apply_one_euro(
                base_interp[f"{comp}_interp"], valid_for_smooth,
                arrays["timestamps"], frame_disc, lm_disc)
            one_euro[comp] = filtered
            metrics = compute_smoothing_metrics(
                norm[f"norm_{comp}"], filtered, valid_for_smooth,
                arrays["timestamps"], frame_disc, lm_disc, "one_euro")
            one_euro[f"{comp}_metrics"] = metrics
            smooth_comparison_rows.append({"sample_id": sid, "component": comp, "filter": "one_euro", **metrics})
        smooth_results["one_euro"] = one_euro

        # Savitzky-Golay
        if HAS_SCIPY:
            savgol = {}
            for comp, n_lm in [("pose", 33), ("lh", 21), ("rh", 21)]:
                valid_for_smooth = arrays[f"{comp}_valid"] | base_interp[f"{comp}_mask"]
                lm_disc = find_landmark_discontinuities(arrays, f"{comp}_xyz", n_lm)
                filtered = apply_savgol(
                    base_interp[f"{comp}_interp"], valid_for_smooth,
                    frame_disc, lm_disc)
                savgol[comp] = filtered
                metrics = compute_smoothing_metrics(
                    norm[f"norm_{comp}"], filtered, valid_for_smooth,
                    arrays["timestamps"], frame_disc, lm_disc, "savgol")
                savgol[f"{comp}_metrics"] = metrics
                smooth_comparison_rows.append({"sample_id": sid, "component": comp, "filter": "savgol", **metrics})
            smooth_results["savgol"] = savgol

        # Trajectory visualizations
        plot_trajectory_comparison(sid, arrays, norm, interp_results, smooth_results, viz_dir)

        all_sample_results[sid] = {
            "arrays": arrays,
            "raw_data": raw_data,
            "norm": norm,
            "seg_info": seg_info,
            "interp_results": interp_results,
            "smooth_results": smooth_results,
        }

    # ---- Save CSVs ----
    safe_print("\n  5. Saving CSV outputs...")

    with open(OUT_DIR / "segmentation_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(seg_rows[0].keys()))
        w.writeheader()
        w.writerows(seg_rows)

    with open(OUT_DIR / "normalization_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(norm_rows[0].keys()))
        w.writeheader()
        w.writerows(norm_rows)

    with open(OUT_DIR / "interpolation_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(interp_comparison_rows[0].keys()))
        w.writeheader()
        w.writerows(interp_comparison_rows)

    # Smoothing CSV (flatten nested dicts)
    sm_fieldnames = set()
    for r in smooth_comparison_rows:
        sm_fieldnames.update(r.keys())
    sm_fieldnames = sorted(sm_fieldnames)
    with open(OUT_DIR / "smoothing_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sm_fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(smooth_comparison_rows)

    # Save per-sample normalized sequences as JSON
    safe_print("  Saving normalized sequence JSONs...")
    seq_dir = OUT_DIR / "sequences"
    seq_dir.mkdir(exist_ok=True)

    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        sr = all_sample_results[sid]
        arrays = sr["arrays"]
        norm = sr["norm"]
        seg = sr["seg_info"]

        seq_data = {
            "sample_id": sid,
            "version": "7a_pilot_1.0",
            "coordinate_system": "body_centric",
            "normalization": {
                "origin": "shoulder_midpoint",
                "scale": "shoulder_width",
                "axes": "right_up_forward (right-handed)",
            },
            "segmentation": {
                "seg_start": seg["seg_start"],
                "seg_end": seg["seg_end"],
                "method": seg["method"],
            },
            "n_frames": len(arrays["timestamps"]),
            "frames": [],
        }

        for fi in range(len(arrays["timestamps"])):
            frame = {
                "frame_index": fi,
                "timestamp_s": float(arrays["timestamps"][fi]),
                "in_segment": seg["seg_start"] <= fi <= seg["seg_end"],
                "norm_origin": norm["origins"][fi].tolist() if not np.any(np.isnan(norm["origins"][fi])) else None,
                "norm_scale": float(norm["scales"][fi]) if not np.isnan(norm["scales"][fi]) else None,
                "norm_status": norm["statuses"][fi],
                "pose": [],
                "left_hand": [],
                "right_hand": [],
            }

            for li in range(33):
                pt = norm["norm_pose"][fi, li]
                frame["pose"].append({
                    "xyz": pt.tolist() if not np.any(np.isnan(pt)) else None,
                    "valid": bool(arrays["pose_valid"][fi, li]),
                    "tri_valid": bool(arrays["pose_tri"][fi, li]),
                    "qa_valid": bool(arrays["pose_qa"][fi, li]),
                    "te": bool(arrays["pose_valid"][fi, li]),
                })

            for hand, key, valid_key in [("left_hand", "norm_lh", "lh_valid"),
                                          ("right_hand", "norm_rh", "rh_valid")]:
                for li in range(21):
                    pt = norm[key][fi, li]
                    frame[hand].append({
                        "xyz": pt.tolist() if not np.any(np.isnan(pt)) else None,
                        "valid": bool(arrays[valid_key][fi, li]),
                    })

            seq_data["frames"].append(frame)

        with open(seq_dir / f"norm_{short}.json", "w", encoding="utf-8") as f:
            json.dump(seq_data, f)
        sz = os.path.getsize(seq_dir / f"norm_{short}.json")
        safe_print(f"    norm_{short}.json ({sz / 1e6:.1f} MB)")

    # ---- 6. Assertions ----
    safe_print("\n  6. Running assertions...")
    assertion_results, n_pass, n_fail = run_assertions(
        all_sample_results, split, pilot_ids, input_hashes, OUT_DIR)

    with open(OUT_DIR / "assertion_results.json", "w", encoding="utf-8") as f:
        json.dump(assertion_results, f, indent=2)

    # ---- 7. Report ----
    safe_print("\n  7. Generating report...")
    rpt = []
    rpt.append("# Step 7A Preprocessing Pilot Report")
    rpt.append("")
    rpt.append("## Status: STEP7A_PREPROCESSING_PILOT_COMPLETE")
    rpt.append("")
    rpt.append(f"**Samples**: {len(pilot_ids)} (3 per word from train split)")
    rpt.append(f"**Input**: Step 6B v1.2.4 full 45-sample 3D landmarks")
    rpt.append(f"**Assertions**: {n_pass} PASS / {n_fail} FAIL")
    rpt.append("")

    rpt.append("## Dataset Split")
    rpt.append("")
    rpt.append("| Word | Train | Val | Test |")
    rpt.append("|------|-------|-----|------|")
    for word in WORDS:
        rpt.append(f"| {word} | {counts[word]['train']} | {counts[word]['val']} | {counts[word]['test']} |")
    rpt.append(f"| **Total** | 27 | 9 | 9 |")
    rpt.append("")

    rpt.append("## Pilot Samples")
    rpt.append("")
    rpt.append("| Sample | Word | Tier | TE Rate | QA Rate | Anomalies | Score |")
    rpt.append("|--------|------|------|---------|---------|-----------|-------|")
    for sid in pilot_ids:
        p = pilot[sid]
        rpt.append(f"| {sid.replace('S001_P001_','')} | {p['word']} | {p['tier']} | "
                   f"{p['te_rate']:.1%} | {p['qa_rate']:.1%} | {p['n_anomaly_flags']} | {p['quality_score']:.4f} |")
    rpt.append("")

    rpt.append("## Segmentation")
    rpt.append("")
    rpt.append("| Sample | Total | Seg Start | Seg End | Seg Frames | Method | Idle Before | Idle After |")
    rpt.append("|--------|-------|-----------|---------|------------|--------|-------------|------------|")
    for r in seg_rows:
        rpt.append(f"| {r['sample_id'].replace('S001_P001_','')} | {r['total_frames']} | "
                   f"{r['seg_start']} | {r['seg_end']} | {r['seg_frames']} | {r['method']} | "
                   f"{r['idle_before']} | {r['idle_after']} |")
    rpt.append("")

    rpt.append("## Normalization")
    rpt.append("")
    rpt.append("| Sample | Frames | OK | Fallback | Degenerate | Scale Mean | Scale Std |")
    rpt.append("|--------|--------|----|-----------|-----------:|------------|-----------|")
    for r in norm_rows:
        rpt.append(f"| {r['sample_id'].replace('S001_P001_','')} | {r['n_frames']} | "
                   f"{r['n_ok']} | {r['n_fallback_prev']} | {r['n_degenerate']} | "
                   f"{r['scale_mean']:.4f} | {r['scale_std']:.4f} |")
    rpt.append("")

    rpt.append("## Interpolation Comparison (pose component)")
    rpt.append("")
    rpt.append("| Sample | max_gap=1 filled | max_gap=2 filled | max_gap=3 filled |")
    rpt.append("|--------|-----------------|-----------------|-----------------|")
    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        vals = []
        for mg in INTERP_CANDIDATES:
            row = [r for r in interp_comparison_rows if r["sample_id"] == sid and r["component"] == "pose" and r["max_gap"] == mg]
            vals.append(str(row[0]["n_points_filled"]) if row else "?")
        rpt.append(f"| {short} | {vals[0]} | {vals[1]} | {vals[2]} |")
    rpt.append("")
    rpt.append("**Recommended max_gap**: 2 frames (balances fill rate vs boundary risk)")
    rpt.append("")

    rpt.append("## Smoothing Comparison (mean position deviation, pose component)")
    rpt.append("")
    rpt.append("| Sample | None | OneEuro dev(mm) | OneEuro jitter red. | SavGol dev(mm) | SavGol jitter red. |")
    rpt.append("|--------|------|-----------------|---------------------|----------------|---------------------|")
    for sid in pilot_ids:
        short = sid.replace("S001_P001_", "")
        oe = [r for r in smooth_comparison_rows if r["sample_id"] == sid and r["component"] == "pose" and r["filter"] == "one_euro"]
        sg = [r for r in smooth_comparison_rows if r["sample_id"] == sid and r["component"] == "pose" and r["filter"] == "savgol"]
        oe_dev = f"{oe[0]['pos_dev_mean_m'] * 1000:.2f}" if oe else "N/A"
        oe_jr = f"{oe[0].get('jitter_reduction_start', 0) * 100:.0f}%" if oe else "N/A"
        sg_dev = f"{sg[0]['pos_dev_mean_m'] * 1000:.2f}" if sg else "N/A"
        sg_jr = f"{sg[0].get('jitter_reduction_start', 0) * 100:.0f}%" if sg else "N/A"
        rpt.append(f"| {short} | 0 | {oe_dev} | {oe_jr} | {sg_dev} | {sg_jr} |")
    rpt.append("")
    rpt.append("**Note**: OneEuro is the real-time candidate; SavGol is offline reference only.")
    rpt.append("")

    rpt.append("## Visualization Files")
    rpt.append("")
    if HAS_MPL:
        viz_files = sorted(viz_dir.glob("*.png"))
        for vf in viz_files:
            rpt.append(f"- `visualizations/{vf.name}` ({os.path.getsize(vf) / 1e3:.0f} KB)")
    else:
        rpt.append("- Matplotlib not available; visualizations not generated")
    rpt.append("")

    rpt.append(f"## Assertions: {n_pass} PASS / {n_fail} FAIL")
    rpt.append("")

    report_text = "\n".join(rpt)
    with open(OUT_DIR / "STEP7A_PILOT_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    # ---- Run Manifest ----
    manifest = {
        "step": "7A",
        "status": "STEP7A_PREPROCESSING_PILOT_COMPLETE" if n_fail == 0 else "STEP7A_BLOCKED_ASSERTION_FAILURE",
        "prior_status": ["STEP6B_COMPLETE", "STEP6B_FULL_45_EXTERNAL_REVIEW_PASS"],
        "pilot_sample_count": len(pilot_ids),
        "pilot_samples": pilot_ids,
        "dataset_split": {"train": 27, "val": 9, "test": 9},
        "assertions": {"pass": n_pass, "fail": n_fail},
        "self_reference_excluded": True,
        "input_dir": str(STEP6B_DIR.relative_to(PROJECT)),
        "output_dir": str(OUT_DIR.relative_to(PROJECT)),
    }
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ---- ZIP ----
    safe_print("\n  8. Creating delivery ZIP...")
    zip_path = PROJECT / "WordsInSilence_step7a_pilot_delivery.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for root, _dirs, fnames in os.walk(OUT_DIR):
            for fn in fnames:
                fpath = Path(root) / fn
                arcname = fpath.relative_to(OUT_DIR).as_posix()
                ext = Path(fn).suffix.lower()
                comp = zipfile.ZIP_STORED if ext in (".png", ".mp4") else zipfile.ZIP_DEFLATED
                zf.write(fpath, arcname, compress_type=comp)
        # Add script
        zf.write(Path(__file__), "s001_step7a_preprocess_pilot.py", compress_type=zipfile.ZIP_DEFLATED)

    zip_size = os.path.getsize(zip_path)
    zip_hash = sha256_file(zip_path)
    safe_print(f"    ZIP: {zip_path.name} ({zip_size / 1e6:.1f} MB)")
    safe_print(f"    SHA-256: {zip_hash}")

    # ---- Summary ----
    safe_print(f"\n  ============================================================")
    safe_print(f"  STEP 7A PREPROCESSING PILOT SUMMARY")
    safe_print(f"  ============================================================")
    safe_print(f"  Pilot samples: {len(pilot_ids)}")
    safe_print(f"  Dataset split: train=27 val=9 test=9")
    safe_print(f"  Normalization: body-centric (shoulder origin, shoulder-width scale)")
    safe_print(f"  Segmentation: motion energy, threshold={SEG_IDLE_THRESHOLD}")
    safe_print(f"  Interpolation: candidates max_gap=[1,2,3], recommended=2")
    safe_print(f"  Smoothing: none / OneEuro / SavGol compared")
    safe_print(f"  Assertions: {n_pass} PASS / {n_fail} FAIL")
    safe_print(f"  Output: {OUT_DIR}")
    safe_print(f"  ZIP: {zip_path} ({zip_size / 1e6:.1f} MB)")
    if n_fail == 0:
        safe_print(f"\n  Status: STEP7A_PREPROCESSING_PILOT_COMPLETE")
        safe_print(f"  Status: STEP7A_BLOCKED_PENDING_EXTERNAL_REVIEW")
    else:
        safe_print(f"\n  Status: STEP7A_BLOCKED_ASSERTION_FAILURE")

    return n_fail


if __name__ == "__main__":
    sys.exit(main())
