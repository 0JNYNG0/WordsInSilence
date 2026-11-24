"""
Tools/s04_phase2_prototype.py — Phase 2: ExpectedSignPrototype & Canonical Motion

1. Load front landmarks + accepted rep boundaries (0-10)
2. Extract normalized features per rep:
   - Body-relative wrist positions (shoulder-width normalized)
   - Finger flex angles & spread
   - Forearm/palm orientation vectors
   - Hand trajectory path & velocity
   - Two-hand distance & contact state
3. Compute mean template + stddev tolerances
4. Select cleanest rep (lowest MSE vs mean)
5. Export prototype JSON + canonical motion JSON + report
"""

import csv
import datetime
import json
import math
import os
import sys
import numpy as np
from typing import List, Dict, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(_THIS_DIR, "s04_out_v3a")

# MediaPipe Pose landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

# MediaPipe Hand landmark indices
WRIST_H = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = [
    ("thumb",  [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP]),
    ("index",  [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP]),
    ("middle", [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP]),
    ("ring",   [RING_MCP, RING_PIP, RING_DIP, RING_TIP]),
    ("pinky",  [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP]),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def vec3(pt):
    return np.array(pt[:3], dtype=np.float64)

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)

def angle_between(a, b):
    """Angle in degrees between two vectors."""
    cos = np.clip(np.dot(normalize(a), normalize(b)), -1, 1)
    return math.degrees(math.acos(cos))

def joint_angle(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3, in degrees."""
    v1 = vec3(p1) - vec3(p2)
    v2 = vec3(p3) - vec3(p2)
    return angle_between(v1, v2)


# ── Body Coordinate System ──────────────────────────────────────────────────

def body_coordinate_system(pose_world):
    """
    Build body-local coordinate system from pose world landmarks.
    Origin = hip midpoint, X = right (L_shoulder -> R_shoulder),
    Y = up (orthogonalized), Z = forward (X cross Y).
    Returns (origin, axes_3x3) where axes[0]=X, axes[1]=Y, axes[2]=Z.
    """
    l_hip = vec3(pose_world[L_HIP])
    r_hip = vec3(pose_world[R_HIP])
    l_sh = vec3(pose_world[L_SHOULDER])
    r_sh = vec3(pose_world[R_SHOULDER])

    origin = (l_hip + r_hip) / 2
    spine = ((l_sh + r_sh) / 2) - origin

    x_raw = r_sh - l_sh
    x_axis = normalize(x_raw)

    # Y = up, orthogonalized against X
    y_raw = spine - np.dot(spine, x_axis) * x_axis
    y_axis = normalize(y_raw)

    # Z = X cross Y (forward, toward camera in MediaPipe world)
    z_axis = normalize(np.cross(x_axis, y_axis))

    return origin, np.array([x_axis, y_axis, z_axis])


def to_body_local(point_world, origin, axes):
    """Transform a world point to body-local coordinates."""
    delta = vec3(point_world) - origin
    return np.array([np.dot(delta, axes[i]) for i in range(3)])


def shoulder_width(pose_world):
    return np.linalg.norm(vec3(pose_world[R_SHOULDER]) - vec3(pose_world[L_SHOULDER]))


# ── Feature Extraction Per Frame ─────────────────────────────────────────────

def extract_frame_features(fr):
    """
    Extract feature vector from a single frame.
    Returns dict of named features, or None if critical data missing.
    """
    pose_w = fr.get("pose_world")
    hands = fr.get("hands") or []
    if pose_w is None or len(pose_w) < 25:
        return None

    origin, axes = body_coordinate_system(pose_w)
    sw = shoulder_width(pose_w)
    if sw < 1e-6:
        return None

    features = {}

    # ── Body-relative wrist positions (normalized by shoulder width) ──
    l_wrist_w = vec3(pose_w[L_WRIST])
    r_wrist_w = vec3(pose_w[R_WRIST])
    l_wrist_local = to_body_local(l_wrist_w, origin, axes) / sw
    r_wrist_local = to_body_local(r_wrist_w, origin, axes) / sw

    features["l_wrist_pos"] = l_wrist_local.tolist()
    features["r_wrist_pos"] = r_wrist_local.tolist()

    # ── Forearm orientation vectors (body-local) ──
    l_elbow_w = vec3(pose_w[L_ELBOW])
    r_elbow_w = vec3(pose_w[R_ELBOW])
    l_forearm = normalize(to_body_local(l_wrist_w, origin, axes) - to_body_local(l_elbow_w, origin, axes))
    r_forearm = normalize(to_body_local(r_wrist_w, origin, axes) - to_body_local(r_elbow_w, origin, axes))
    features["l_forearm_dir"] = l_forearm.tolist()
    features["r_forearm_dir"] = r_forearm.tolist()

    # ── Two-hand distance (normalized) & contact ──
    two_hand_dist = np.linalg.norm(l_wrist_w - r_wrist_w) / sw
    features["two_hand_dist"] = float(two_hand_dist)
    features["contact_state"] = two_hand_dist < 0.3  # threshold

    # ── Hand landmarks (if detected) ──
    left_hand = None
    right_hand = None
    for h in hands:
        label = h.get("handedness", "")
        world = h.get("world")
        if world and len(world) >= 21:
            if label == "Left" and left_hand is None:
                left_hand = world
            elif label == "Right" and right_hand is None:
                right_hand = world

    for side, hand_lm in [("l", left_hand), ("r", right_hand)]:
        if hand_lm is None:
            # Fill with NaN so we can handle missing hands
            for finger_name, _ in FINGERS:
                features[f"{side}_{finger_name}_flex"] = float("nan")
            features[f"{side}_finger_spread"] = float("nan")
            features[f"{side}_palm_normal"] = [float("nan")] * 3
            continue

        # Finger flex angles (joint angle at PIP/IP)
        for finger_name, indices in FINGERS:
            if finger_name == "thumb":
                # Thumb: angle at IP joint
                ang = joint_angle(hand_lm[indices[1]], hand_lm[indices[2]], hand_lm[indices[3]])
            else:
                # Other fingers: angle at PIP joint
                ang = joint_angle(hand_lm[indices[0]], hand_lm[indices[1]], hand_lm[indices[2]])
            features[f"{side}_{finger_name}_flex"] = float(ang)

        # Finger spread: angle between index MCP-tip and pinky MCP-tip vectors
        idx_dir = vec3(hand_lm[INDEX_TIP]) - vec3(hand_lm[INDEX_MCP])
        pnk_dir = vec3(hand_lm[PINKY_TIP]) - vec3(hand_lm[PINKY_MCP])
        features[f"{side}_finger_spread"] = float(angle_between(idx_dir, pnk_dir))

        # Palm normal: cross product of (MCP5-wrist) x (MCP17-wrist)
        v1 = vec3(hand_lm[INDEX_MCP]) - vec3(hand_lm[WRIST_H])
        v2 = vec3(hand_lm[PINKY_MCP]) - vec3(hand_lm[WRIST_H])
        palm_n = normalize(np.cross(v1, v2))
        # Transform to body-local
        palm_n_local = np.array([np.dot(palm_n, axes[i]) for i in range(3)])
        features[f"{side}_palm_normal"] = palm_n_local.tolist()

    return features


# ── Trajectory & Velocity ────────────────────────────────────────────────────

def compute_trajectory_features(frame_features, pts_ms_list):
    """
    Compute trajectory path length and velocity curve from per-frame features.
    """
    n = len(frame_features)
    if n < 2:
        return {}

    # Wrist trajectory path length (body-local, normalized)
    l_path = 0.0
    r_path = 0.0
    l_velocities = []
    r_velocities = []

    for i in range(1, n):
        prev = frame_features[i - 1]
        curr = frame_features[i]
        if prev is None or curr is None:
            l_velocities.append(0.0)
            r_velocities.append(0.0)
            continue

        dt = (pts_ms_list[i] - pts_ms_list[i - 1]) / 1000.0  # seconds
        if dt < 1e-6:
            l_velocities.append(0.0)
            r_velocities.append(0.0)
            continue

        l_delta = np.linalg.norm(
            np.array(curr["l_wrist_pos"]) - np.array(prev["l_wrist_pos"])
        )
        r_delta = np.linalg.norm(
            np.array(curr["r_wrist_pos"]) - np.array(prev["r_wrist_pos"])
        )
        l_path += l_delta
        r_path += r_delta
        l_velocities.append(l_delta / dt)
        r_velocities.append(r_delta / dt)

    return {
        "l_wrist_path_length": float(l_path),
        "r_wrist_path_length": float(r_path),
        "l_wrist_velocity_mean": float(np.mean(l_velocities)) if l_velocities else 0.0,
        "r_wrist_velocity_mean": float(np.mean(r_velocities)) if r_velocities else 0.0,
        "l_wrist_velocity_peak": float(np.max(l_velocities)) if l_velocities else 0.0,
        "r_wrist_velocity_peak": float(np.max(r_velocities)) if r_velocities else 0.0,
    }


# ── Rep Feature Extraction ──────────────────────────────────────────────────

def extract_rep_features(frames, start_fi, end_fi):
    """Extract all per-frame features for a single rep."""
    rep_frames = [fr for fr in frames if start_fi <= fr["fi"] <= end_fi]
    if not rep_frames:
        return None, None, None

    frame_features = []
    pts_list = []
    raw_frames = []

    for fr in rep_frames:
        ff = extract_frame_features(fr)
        frame_features.append(ff)
        pts_list.append(fr["source_pts_ms"])
        raw_frames.append(fr)

    # Trajectory features
    traj = compute_trajectory_features(frame_features, pts_list)

    return frame_features, traj, raw_frames


# ── Feature Vectorization (for MSE comparison) ──────────────────────────────

SCALAR_KEYS = [
    "two_hand_dist",
    "l_thumb_flex", "l_index_flex", "l_middle_flex", "l_ring_flex", "l_pinky_flex",
    "r_thumb_flex", "r_index_flex", "r_middle_flex", "r_ring_flex", "r_pinky_flex",
    "l_finger_spread", "r_finger_spread",
]

VEC3_KEYS = [
    "l_wrist_pos", "r_wrist_pos",
    "l_forearm_dir", "r_forearm_dir",
    "l_palm_normal", "r_palm_normal",
]


def features_to_vector(ff):
    """Convert a feature dict to a flat numpy vector for comparison."""
    vals = []
    for k in SCALAR_KEYS:
        v = ff.get(k, float("nan"))
        if isinstance(v, bool):
            v = float(v)
        vals.append(float(v))
    for k in VEC3_KEYS:
        v = ff.get(k, [float("nan")] * 3)
        vals.extend([float(x) for x in v])
    return np.array(vals)


def time_normalize_features(frame_features, n_steps=30):
    """
    Time-normalize a sequence of per-frame features to n_steps via linear interpolation.
    Returns list of n_steps feature vectors.
    """
    valid = [(i, ff) for i, ff in enumerate(frame_features) if ff is not None]
    if len(valid) < 2:
        return None

    # Convert to matrix
    indices = [v[0] for v in valid]
    vectors = np.array([features_to_vector(v[1]) for v in valid])
    n_features = vectors.shape[1]

    # Interpolate each feature dimension
    normalized = np.zeros((n_steps, n_features))
    target_positions = np.linspace(indices[0], indices[-1], n_steps)
    for d in range(n_features):
        col = vectors[:, d]
        if np.all(np.isnan(col)):
            normalized[:, d] = np.nan
        else:
            # Fill NaN with linear interp
            valid_mask = ~np.isnan(col)
            if np.sum(valid_mask) >= 2:
                normalized[:, d] = np.interp(
                    target_positions,
                    np.array(indices)[valid_mask],
                    col[valid_mask],
                )
            else:
                normalized[:, d] = np.nanmean(col)

    return normalized


# ── Mean Template & Tolerances ───────────────────────────────────────────────

def compute_prototype(all_rep_normalized):
    """
    Compute mean template and stddev tolerances across all reps.
    all_rep_normalized: list of (n_steps x n_features) arrays.
    """
    stack = np.array(all_rep_normalized)  # (n_reps, n_steps, n_features)
    mean_template = np.nanmean(stack, axis=0)
    std_template = np.nanstd(stack, axis=0)
    return mean_template, std_template


def compute_rep_mse(rep_norm, mean_template):
    """MSE of a single rep against the mean template, ignoring NaN."""
    diff = rep_norm - mean_template
    mask = ~np.isnan(diff)
    if np.sum(mask) == 0:
        return float("inf")
    return float(np.mean(diff[mask] ** 2))


# ── Canonical Motion Export ──────────────────────────────────────────────────

def export_canonical_motion(raw_frames, rep_id, out_path):
    """
    Export normalized landmarks for the selected rep in blend_animate-compatible format.
    Schema: list of frames with pose_world (33 landmarks) and hands (world landmarks).
    """
    canonical = {
        "schema": "canonical_motion_v1",
        "sign_id": "SIGN001",
        "sign_name": "greeting",
        "source_rep_id": rep_id,
        "source_view": "front",
        "frame_count": len(raw_frames),
        "fps": 30,
        "coordinate_system": "mediapipe_world_hip_centered_y_down",
        "frames": [],
    }

    for fr in raw_frames:
        frame_data = {
            "fi": fr["fi"],
            "source_pts_ms": fr["source_pts_ms"],
            "pose_world": fr.get("pose_world"),
            "hands": [],
        }
        for h in (fr.get("hands") or []):
            frame_data["hands"].append({
                "handedness": h.get("handedness"),
                "world": h.get("world"),
            })
        canonical["frames"].append(frame_data)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(canonical, f, indent=2, ensure_ascii=False)

    return canonical


# ── Prototype JSON Export ────────────────────────────────────────────────────

def export_prototype(mean_template, std_template, traj_stats, rep_mses,
                     selected_rep, out_path):
    """Export the ExpectedSignPrototype JSON."""

    # Build readable feature names
    feature_names = list(SCALAR_KEYS)
    for k in VEC3_KEYS:
        feature_names.extend([f"{k}_x", f"{k}_y", f"{k}_z"])

    n_steps, n_feat = mean_template.shape

    # Per-step mean and std as lists
    steps = []
    for t in range(n_steps):
        step = {"t_normalized": round(t / (n_steps - 1), 4)}
        for i, name in enumerate(feature_names):
            m = mean_template[t, i]
            s = std_template[t, i]
            step[name] = {
                "mean": None if np.isnan(m) else round(float(m), 6),
                "std": None if np.isnan(s) else round(float(s), 6),
            }
        steps.append(step)

    # Summary tolerances (aggregated across time)
    tolerances = {}
    for i, name in enumerate(feature_names):
        col_mean = mean_template[:, i]
        col_std = std_template[:, i]
        if np.all(np.isnan(col_mean)):
            tolerances[name] = {"range_mean": None, "range_std": None}
        else:
            tolerances[name] = {
                "range_mean": [
                    round(float(np.nanmin(col_mean)), 4),
                    round(float(np.nanmax(col_mean)), 4),
                ],
                "range_std": [
                    round(float(np.nanmin(col_std)), 4),
                    round(float(np.nanmax(col_std)), 4),
                ],
            }

    prototype = {
        "schema": "expected_sign_prototype_v1",
        "sign_id": "SIGN001",
        "sign_name": "greeting",
        "source_view": "front",
        "source_reps_used": 11,
        "time_steps": n_steps,
        "feature_count": n_feat,
        "feature_names": feature_names,
        "selected_canonical_rep": selected_rep,
        "rep_mse_scores": {str(k): round(v, 6) for k, v in rep_mses.items()},
        "trajectory_stats": traj_stats,
        "tolerances_summary": tolerances,
        "time_normalized_template": steps,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prototype, f, indent=2, ensure_ascii=False)

    return prototype


# ── Report Generation ────────────────────────────────────────────────────────

def generate_report(prototype, canonical, rep_mses, traj_all, out_path):
    lines = []
    a = lines.append

    a("# Phase 2: ExpectedSignPrototype & Canonical Motion Report")
    a("")
    a(f"Generated: {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    a("")

    # 1. Selected Rep
    sel = prototype["selected_canonical_rep"]
    a("## 1. Selected Canonical Repetition")
    a("")
    a(f"- **Rep ID**: {sel['rep_id']}")
    a(f"- **MSE vs mean**: {sel['mse']:.6f} (lowest of 11 reps)")
    a(f"- **Frame range**: fi={sel['start_fi']}-{sel['end_fi']} ({sel['frame_count']} frames)")
    a(f"- **Duration**: {sel['duration_ms']:.0f} ms")
    a(f"- **Hand detection rate**: {sel['hand_detection_rate']:.1%}")
    a("")

    # MSE ranking
    a("### Rep MSE Ranking")
    a("")
    a("| Rep | MSE | Rank |")
    a("|-----|-----|------|")
    sorted_reps = sorted(rep_mses.items(), key=lambda x: x[1])
    for rank, (rep_id, mse) in enumerate(sorted_reps, 1):
        marker = " **<--**" if rep_id == sel["rep_id"] else ""
        a(f"| {rep_id} | {mse:.6f} | {rank}{marker} |")
    a("")

    # 2. Feature Tolerances
    a("## 2. Normalized Feature Tolerances")
    a("")

    tol = prototype["tolerances_summary"]
    a("### Wrist Positions (body-local, shoulder-width normalized)")
    a("")
    a("| Feature | Mean Range | Std Range |")
    a("|---------|-----------|-----------|")
    for k in ["l_wrist_pos_x", "l_wrist_pos_y", "l_wrist_pos_z",
              "r_wrist_pos_x", "r_wrist_pos_y", "r_wrist_pos_z"]:
        t = tol.get(k, {})
        mr = t.get("range_mean", None)
        sr = t.get("range_std", None)
        mr_s = f"[{mr[0]:.4f}, {mr[1]:.4f}]" if mr else "N/A"
        sr_s = f"[{sr[0]:.4f}, {sr[1]:.4f}]" if sr else "N/A"
        a(f"| {k} | {mr_s} | {sr_s} |")
    a("")

    a("### Finger Flex Angles (degrees)")
    a("")
    a("| Feature | Mean Range | Std Range |")
    a("|---------|-----------|-----------|")
    for side in ["l", "r"]:
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            k = f"{side}_{finger}_flex"
            t = tol.get(k, {})
            mr = t.get("range_mean", None)
            sr = t.get("range_std", None)
            mr_s = f"[{mr[0]:.1f}, {mr[1]:.1f}]" if mr else "N/A"
            sr_s = f"[{sr[0]:.1f}, {sr[1]:.1f}]" if sr else "N/A"
            a(f"| {k} | {mr_s} | {sr_s} |")
    a("")

    a("### Two-Hand Distance & Spread")
    a("")
    for k in ["two_hand_dist", "l_finger_spread", "r_finger_spread"]:
        t = tol.get(k, {})
        mr = t.get("range_mean", None)
        sr = t.get("range_std", None)
        mr_s = f"[{mr[0]:.4f}, {mr[1]:.4f}]" if mr else "N/A"
        sr_s = f"[{sr[0]:.4f}, {sr[1]:.4f}]" if sr else "N/A"
        a(f"- **{k}**: mean={mr_s}, std={sr_s}")
    a("")

    # 3. Trajectory Stats
    a("## 3. Trajectory Statistics (across 11 reps)")
    a("")
    a("| Metric | Mean | Std |")
    a("|--------|------|-----|")
    ts = prototype["trajectory_stats"]
    for key in sorted(ts.keys()):
        val = ts[key]
        a(f"| {key} | {val['mean']:.4f} | {val['std']:.4f} |")
    a("")

    # 4. Schema Validation
    a("## 4. Schema Validation")
    a("")
    proto_keys = ["schema", "sign_id", "sign_name", "source_view", "source_reps_used",
                  "time_steps", "feature_count", "feature_names", "selected_canonical_rep",
                  "rep_mse_scores", "trajectory_stats", "tolerances_summary",
                  "time_normalized_template"]
    canon_keys = ["schema", "sign_id", "sign_name", "source_rep_id", "source_view",
                  "frame_count", "fps", "coordinate_system", "frames"]

    proto_valid = all(k in prototype for k in proto_keys)
    canon_valid = all(k in canonical for k in canon_keys)

    a(f"- **SIGN001_greeting_prototype.json**: {'PASS' if proto_valid else 'FAIL'} ({len(proto_keys)} required keys)")
    a(f"- **SIGN001_canonical_motion.json**: {'PASS' if canon_valid else 'FAIL'} ({len(canon_keys)} required keys)")
    a(f"- Prototype time steps: {prototype['time_steps']}")
    a(f"- Prototype features: {prototype['feature_count']}")
    a(f"- Canonical frames: {canonical['frame_count']}")
    a("")

    # 5. Output Files
    a("## 5. Output Files")
    a("")
    a("| File | Description |")
    a("|------|-------------|")
    a("| `SIGN001_greeting_prototype.json` | Mean feature template + stddev tolerances (30 time-normalized steps x 31 features) |")
    a("| `SIGN001_canonical_motion.json` | Raw landmarks for cleanest rep (Blender retargeting input) |")
    a("| `phase2_prototype_report.md` | This report |")
    a("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 2: ExpectedSignPrototype & Canonical Motion ===\n")

    # Load data
    print("[1/6] Loading front landmarks...")
    with open(os.path.join(OUT_DIR, "landmarks_raw_front.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data if isinstance(data, list) else data.get("frames", data)
    print(f"  {len(frames)} frames")

    # Load boundaries
    print("[2/6] Loading accepted rep boundaries...")
    boundaries = []
    with open(os.path.join(OUT_DIR, "boundaries_override_template.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["user_status"] == "accepted":
                boundaries.append({
                    "rep_id": int(row["rep_id"]),
                    "start_fi": int(row["front_start_fi"]),
                    "end_fi": int(row["front_end_fi"]),
                    "duration_ms": float(row["duration_ms"]),
                })
    print(f"  {len(boundaries)} accepted reps")

    # Extract features per rep
    print("[3/6] Extracting features per rep...")
    all_rep_normalized = []
    all_traj = []
    rep_raw_frames = {}
    rep_hand_rates = {}

    for b in boundaries:
        rep_id = b["rep_id"]
        frame_features, traj, raw_frames = extract_rep_features(
            frames, b["start_fi"], b["end_fi"]
        )

        # Hand detection rate
        valid_hands = sum(1 for ff in frame_features if ff is not None and
                         not np.isnan(ff.get("l_thumb_flex", float("nan"))) and
                         not np.isnan(ff.get("r_thumb_flex", float("nan"))))
        hand_rate = valid_hands / len(frame_features) if frame_features else 0
        rep_hand_rates[rep_id] = hand_rate

        # Time normalize
        norm = time_normalize_features(frame_features, n_steps=30)
        if norm is not None:
            all_rep_normalized.append(norm)
            all_traj.append(traj)
            rep_raw_frames[rep_id] = raw_frames
            print(f"  rep {rep_id}: {len(frame_features)} frames, "
                  f"hand_rate={hand_rate:.1%}, "
                  f"path_L={traj['l_wrist_path_length']:.3f}, "
                  f"path_R={traj['r_wrist_path_length']:.3f}")
        else:
            print(f"  rep {rep_id}: SKIPPED (insufficient valid frames)")

    # Compute mean template
    print(f"\n[4/6] Computing mean template ({len(all_rep_normalized)} reps)...")
    mean_template, std_template = compute_prototype(all_rep_normalized)
    print(f"  template shape: {mean_template.shape}")

    # Compute MSE per rep and select cleanest
    print("[5/6] Selecting cleanest repetition...")
    rep_mses = {}
    for i, b in enumerate(boundaries):
        if i < len(all_rep_normalized):
            mse = compute_rep_mse(all_rep_normalized[i], mean_template)
            rep_mses[b["rep_id"]] = mse

    best_rep_id = min(rep_mses, key=rep_mses.get)
    best_mse = rep_mses[best_rep_id]
    best_boundary = [b for b in boundaries if b["rep_id"] == best_rep_id][0]
    print(f"  selected: rep {best_rep_id} (MSE={best_mse:.6f})")

    # Trajectory stats
    traj_keys = list(all_traj[0].keys()) if all_traj else []
    traj_stats = {}
    for k in traj_keys:
        vals = [t[k] for t in all_traj if k in t]
        traj_stats[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
        }

    selected_info = {
        "rep_id": best_rep_id,
        "mse": round(best_mse, 6),
        "start_fi": best_boundary["start_fi"],
        "end_fi": best_boundary["end_fi"],
        "frame_count": len(rep_raw_frames[best_rep_id]),
        "duration_ms": best_boundary["duration_ms"],
        "hand_detection_rate": rep_hand_rates[best_rep_id],
    }

    # Export
    print("\n[6/6] Exporting...")
    proto_path = os.path.join(OUT_DIR, "SIGN001_greeting_prototype.json")
    prototype = export_prototype(
        mean_template, std_template, traj_stats, rep_mses,
        selected_info, proto_path,
    )
    print(f"  SIGN001_greeting_prototype.json ({os.path.getsize(proto_path) // 1024} KB)")

    canon_path = os.path.join(OUT_DIR, "SIGN001_canonical_motion.json")
    canonical = export_canonical_motion(
        rep_raw_frames[best_rep_id], best_rep_id, canon_path,
    )
    print(f"  SIGN001_canonical_motion.json ({os.path.getsize(canon_path) // 1024} KB)")

    report_path = os.path.join(OUT_DIR, "phase2_prototype_report.md")
    generate_report(prototype, canonical, rep_mses, all_traj, report_path)
    print(f"  phase2_prototype_report.md ({os.path.getsize(report_path)} bytes)")

    print("\n=== Phase 2 Complete ===")


if __name__ == "__main__":
    main()
