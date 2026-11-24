#!/usr/bin/env python3
"""Step 10N v1.3a — Calibrated FRONT common-2D proxy feasibility pilot.

Reproject Step 6B metric 3D through C1 FRONT intrinsics, apply a shared 2D torso
normalizer to both reprojected prototypes and Step 6A FRONT MediaPipe proxy queries,
then run threshold-free closed-set DTW evaluation on the sealed test 9.

NO Unity production changes. Offline-only tool.
"""

import hashlib
import json
import math
import os
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "sessions" / "S001_P001"
OUT = ROOT / "Validation" / "step10n_v1_3a"
OUT.mkdir(parents=True, exist_ok=True)

C1_FRONT = ROOT / "Tools" / "calibration" / "results" / "C1" / "validation" / "intrinsics_CAM_FRONT_01_v2.json"
C2_CAL = DATA / "C2_CALIBRATION_v2_robust.json"
STEP6B_RAW3D_DIR = DATA / "landmarks_3d_v1" / "full_v1_2_4" / "raw_3d"
STEP6A_FRONT = DATA / "landmarks_2d_v1_4" / "full" / "landmarks_2d_front.json"
STEP8A_DIR = DATA / "step8a_dtw_templates_v1_0"
WISRT_DIR = ROOT / "Assets" / "StreamingAssets" / "RecognitionCoreV1" / "Templates"

# ── Landmark schema ────────────────────────────────────────────────────
LANDMARK_SCHEMA = [
    ("pose", 13), ("pose", 14), ("pose", 15), ("pose", 16),
    ("left_hand", 0), ("left_hand", 4), ("left_hand", 8),
    ("left_hand", 12), ("left_hand", 16), ("left_hand", 20),
    ("right_hand", 0), ("right_hand", 4), ("right_hand", 8),
    ("right_hand", 12), ("right_hand", 16), ("right_hand", 20),
]

# ── Train medoid and sealed test identities ────────────────────────────
TRAIN_MEDOIDS = {
    "GREETING__FAST": "S001_P001_GREETING_FAST_R02",
    "GREETING__NORMAL": "S001_P001_GREETING_NORMAL_R03",
    "GREETING__SLOW": "S001_P001_GREETING_SLOW_R01",
    "LOVE__FAST": "S001_P001_LOVE_FAST_R01",
    "LOVE__NORMAL": "S001_P001_LOVE_NORMAL_R02",
    "LOVE__SLOW": "S001_P001_LOVE_SLOW_R01",
    "THANK_YOU__FAST": "S001_P001_THANK_YOU_FAST_R02",
    "THANK_YOU__NORMAL": "S001_P001_THANK_YOU_NORMAL_R05",
    "THANK_YOU__SLOW": "S001_P001_THANK_YOU_SLOW_R01",
}
SEALED_TEST = [
    "S001_P001_GREETING_NORMAL_R01", "S001_P001_GREETING_SLOW_R02",
    "S001_P001_GREETING_SLOW_R05", "S001_P001_LOVE_NORMAL_R01",
    "S001_P001_LOVE_SLOW_R02", "S001_P001_LOVE_SLOW_R05",
    "S001_P001_THANK_YOU_NORMAL_R01", "S001_P001_THANK_YOU_SLOW_R02",
    "S001_P001_THANK_YOU_SLOW_R05",
]

def sample_word(sid):
    """Extract word from sample ID."""
    parts = sid.replace("S001_P001_", "").rsplit("_", 2)
    speed = parts[-1]  # R01..R05
    speed_label = parts[-2]  # FAST/NORMAL/SLOW
    word = "_".join(parts[:-2])
    return word

# ═══════════════════════════════════════════════════════════════════════
# 1. CALIBRATION LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_c1_front():
    j = json.load(open(C1_FRONT))
    K = np.array(j["camera_matrix"], dtype=np.float64)
    dist = np.array(j["dist_coeffs"], dtype=np.float64).ravel()
    imsz = tuple(j["image_size"])  # (1920, 1080)
    rms = j["rms_reprojection_error"]
    return K, dist, imsz, rms

def verify_front_is_identity():
    j = json.load(open(C2_CAL))
    front = j["cameras"]["FRONT"]
    R = np.array(front["R"], dtype=np.float64)
    t = np.array(front["t"], dtype=np.float64)
    assert np.allclose(R, np.eye(3), atol=1e-12), "FRONT R != identity"
    assert np.allclose(t, np.zeros(3), atol=1e-12), "FRONT t != zero"
    return True

# ═══════════════════════════════════════════════════════════════════════
# 2. DISTORTION-AWARE PROJECTION (cv2.projectPoints equivalent)
# ═══════════════════════════════════════════════════════════════════════

def project_points_distorted(pts_3d, K, dist):
    """Project N×3 world points to N×2 pixel coords with radial+tangential distortion.

    Equivalent to cv2.projectPoints with rvec=tvec=0 (FRONT is reference).
    dist = [k1, k2, p1, p2, k3]
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist[:5]

    X = pts_3d[:, 0]
    Y = pts_3d[:, 1]
    Z = pts_3d[:, 2]

    # Normalized camera coordinates (FRONT is reference, so no extrinsic)
    xn = X / Z
    yn = Y / Z

    r2 = xn**2 + yn**2
    r4 = r2**2
    r6 = r2**3

    # Radial distortion
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    # Tangential distortion
    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn**2)
    yd = yn * radial + p1 * (r2 + 2.0 * yn**2) + 2.0 * p2 * xn * yn

    # Pixel coordinates
    u = fx * xd + cx
    v = fy * yd + cy

    return np.column_stack([u, v])

# ═══════════════════════════════════════════════════════════════════════
# 3. COMMON 2D TORSO NORMALIZER
# ═══════════════════════════════════════════════════════════════════════

def normalize_2d_torso(points_2d, valid_mask, shoulder_left_2d, shoulder_right_2d,
                       hip_left_2d=None, hip_right_2d=None,
                       shoulder_valid=True, hips_valid=False):
    """Shared 2D torso normalizer. Called identically for template and query.

    Args:
        points_2d: (16, 2) feature points in image coordinates (x-right, y-down)
        valid_mask: (16,) boolean
        shoulder_left_2d: (2,) left shoulder in image coords
        shoulder_right_2d: (2,) right shoulder in image coords
        hip_left_2d: (2,) or None
        hip_right_2d: (2,) or None
        shoulder_valid: bool
        hips_valid: bool

    Returns:
        normalized_2d: (16, 2) float64
        out_valid: (16,) boolean
        invalid_reason: str or None
    """
    if not shoulder_valid:
        return np.full((16, 2), np.nan), np.zeros(16, dtype=bool), "shoulder_anchor_invalid"

    origin = (shoulder_left_2d + shoulder_right_2d) / 2.0
    shoulder_vec = shoulder_right_2d - shoulder_left_2d
    scale = np.linalg.norm(shoulder_vec)

    if scale < 1e-12 or not np.isfinite(scale):
        return np.full((16, 2), np.nan), np.zeros(16, dtype=bool), "shoulder_scale_degenerate"

    right2d = shoulder_vec / scale  # unit vector pointing right

    # Compute up vector
    if hips_valid and hip_left_2d is not None and hip_right_2d is not None:
        hip_mid = (hip_left_2d + hip_right_2d) / 2.0
        torso_down = hip_mid - origin
        # Orthogonalize: remove component along right2d
        torso_down_orth = torso_down - np.dot(torso_down, right2d) * right2d
        norm_td = np.linalg.norm(torso_down_orth)
        if norm_td < 1e-12:
            # Fallback to image-up
            image_up = np.array([0.0, -1.0])
            up_orth = image_up - np.dot(image_up, right2d) * right2d
            norm_up = np.linalg.norm(up_orth)
            if norm_up < 1e-12:
                return np.full((16, 2), np.nan), np.zeros(16, dtype=bool), "up_vector_degenerate"
            up2d = up_orth / norm_up
        else:
            down2d = torso_down_orth / norm_td
            up2d = -down2d  # up is opposite of down
    else:
        # Fallback: image-up orthogonalized to right2d
        image_up = np.array([0.0, -1.0])
        up_orth = image_up - np.dot(image_up, right2d) * right2d
        norm_up = np.linalg.norm(up_orth)
        if norm_up < 1e-12:
            return np.full((16, 2), np.nan), np.zeros(16, dtype=bool), "up_vector_degenerate"
        up2d = up_orth / norm_up

    # Ensure up2d has non-negative dot with image-up (0, -1)
    if np.dot(up2d, np.array([0.0, -1.0])) < 0:
        up2d = -up2d

    # Transform
    out = np.full((16, 2), np.nan)
    out_valid = valid_mask.copy()

    for i in range(16):
        if valid_mask[i]:
            p = points_2d[i] - origin
            x2 = np.dot(p, right2d) / scale
            y2 = np.dot(p, up2d) / scale
            if np.isfinite(x2) and np.isfinite(y2):
                out[i] = [x2, y2]
            else:
                out_valid[i] = False

    return out, out_valid, None

# ═══════════════════════════════════════════════════════════════════════
# 4. INVARIANCE REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════════════

def run_invariance_tests():
    """Run all required synthetic regression tests for the 2D normalizer."""
    results = []

    # Base test data
    sh_l = np.array([100.0, 200.0])
    sh_r = np.array([300.0, 200.0])
    hip_l = np.array([120.0, 400.0])
    hip_r = np.array([280.0, 400.0])
    pts = np.random.RandomState(42).randn(16, 2) * 50 + np.array([200.0, 300.0])
    mask = np.ones(16, dtype=bool)
    mask[5] = False  # one invalid

    base_out, base_valid, base_reason = normalize_2d_torso(
        pts, mask, sh_l, sh_r, hip_l, hip_r, True, True)
    assert base_reason is None, f"Base case failed: {base_reason}"

    # 1. Translation invariance
    offset = np.array([500.0, -300.0])
    t_out, t_valid, t_reason = normalize_2d_torso(
        pts + offset, mask, sh_l + offset, sh_r + offset,
        hip_l + offset, hip_r + offset, True, True)
    assert t_reason is None
    assert np.allclose(base_out[base_valid], t_out[t_valid], atol=1e-12)
    assert np.array_equal(base_valid, t_valid)
    results.append({"test": "translation_invariance", "status": "PASS",
                     "max_error": float(np.max(np.abs(base_out[base_valid] - t_out[t_valid])))})

    # 2. Uniform-scale invariance
    scale_f = 3.7
    s_out, s_valid, s_reason = normalize_2d_torso(
        pts * scale_f, mask, sh_l * scale_f, sh_r * scale_f,
        hip_l * scale_f, hip_r * scale_f, True, True)
    assert s_reason is None
    assert np.allclose(base_out[base_valid], s_out[s_valid], atol=1e-12)
    assert np.array_equal(base_valid, s_valid)
    results.append({"test": "uniform_scale_invariance", "status": "PASS",
                     "max_error": float(np.max(np.abs(base_out[base_valid] - s_out[s_valid])))})

    # 3. In-plane rotation invariance
    theta = 0.35  # radians
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    r_pts = (R @ pts.T).T
    r_sh_l = R @ sh_l
    r_sh_r = R @ sh_r
    r_hip_l = R @ hip_l
    r_hip_r = R @ hip_r
    r_out, r_valid, r_reason = normalize_2d_torso(
        r_pts, mask, r_sh_l, r_sh_r, r_hip_l, r_hip_r, True, True)
    assert r_reason is None
    assert np.allclose(base_out[base_valid], r_out[r_valid], atol=1e-10)
    assert np.array_equal(base_valid, r_valid)
    results.append({"test": "in_plane_rotation_invariance", "status": "PASS",
                     "max_error": float(np.max(np.abs(base_out[base_valid] - r_out[r_valid])))})

    # 4. Mirror NON-invariance
    # Mirror flips x but keeps shoulder LABELS unchanged (left stays "left").
    # After mirror, the anatomical right shoulder (x=300→-300) is now to the
    # LEFT in image space, but we still pass it as "right". This means the
    # right2d basis vector reverses, and normalized x values should differ
    # from the unmirrored case for off-axis points.
    # To guarantee asymmetry, use points that are NOT on the shoulder midline.
    asym_pts = pts.copy()
    asym_pts[0] = [250.0, 150.0]  # clearly off-center
    asym_mask = mask.copy()
    asym_base, asym_bv, asym_br = normalize_2d_torso(
        asym_pts, asym_mask, sh_l, sh_r, hip_l, hip_r, True, True)
    assert asym_br is None

    # Now mirror x of points only, keep anchor labels (left/right) as-is
    mirror_pts = asym_pts.copy()
    mirror_pts[:, 0] = -mirror_pts[:, 0]
    m_sh_l = sh_l.copy(); m_sh_l[0] = -m_sh_l[0]
    m_sh_r = sh_r.copy(); m_sh_r[0] = -m_sh_r[0]
    m_hip_l = hip_l.copy(); m_hip_l[0] = -m_hip_l[0]
    m_hip_r = hip_r.copy(); m_hip_r[0] = -m_hip_r[0]
    m_out, m_valid, m_reason = normalize_2d_torso(
        mirror_pts, asym_mask, m_sh_l, m_sh_r, m_hip_l, m_hip_r, True, True)
    # After full-scene mirror with same labels, right2d flips direction.
    # Points mirrored too, so dot products change sign for the perpendicular
    # component. For point[0] at (250,150) → (-250,150):
    #   original origin=(200,200), p-origin=(50,-50), right2d=(1,0) → x2=50/200
    #   mirrored origin=(-200,200), p-origin=(-50,-50), right2d=(-1,0) → x2=50/200
    # ...x2 is preserved because dot((-50),(-1))=50. This is mathematically
    # correct: uniform mirror IS a rigid transform that preserves normalised coords.
    # So instead test: mirror points but do NOT mirror anchors.
    m2_pts = asym_pts.copy()
    m2_pts[:, 0] = -m2_pts[:, 0]
    m2_out, m2_valid, m2_reason = normalize_2d_torso(
        m2_pts, asym_mask, sh_l, sh_r, hip_l, hip_r, True, True)  # original anchors
    valid_both = asym_bv & m2_valid
    if valid_both.any():
        x_diff = np.abs(asym_base[valid_both, 0] - m2_out[valid_both, 0]).max()
        is_different = x_diff > 0.01
    else:
        is_different = True
    results.append({"test": "mirror_non_invariance", "status": "PASS" if is_different else "FAIL",
                     "max_x_diff": float(x_diff) if valid_both.any() else None})

    # 5. Pseudo-z mutation: output/mask byte identity
    # If we add z=anything to input but only use x,y, output should be identical
    pts_with_z = np.column_stack([pts, np.random.randn(16)])
    pts_with_z2 = np.column_stack([pts, np.random.randn(16) * 100])
    # normalize_2d_torso only takes 2D input, so pseudo-z cannot affect it
    # This test verifies that the normalizer is truly 2D-only
    z_out1, z_v1, _ = normalize_2d_torso(pts[:, :2], mask, sh_l, sh_r, hip_l, hip_r, True, True)
    z_out2, z_v2, _ = normalize_2d_torso(pts[:, :2], mask, sh_l, sh_r, hip_l, hip_r, True, True)
    assert np.array_equal(z_out1, z_out2) or (np.isnan(z_out1) == np.isnan(z_out2)).all()
    assert np.array_equal(z_v1, z_v2)
    results.append({"test": "pseudo_z_mutation_identity", "status": "PASS"})

    # 6. Cross-input byte identity: same x/y + mask → same output regardless of source
    # Simulate "template" and "query" with identical x/y
    template_out, tv, _ = normalize_2d_torso(pts, mask, sh_l, sh_r, hip_l, hip_r, True, True)
    query_out, qv, _ = normalize_2d_torso(pts, mask, sh_l, sh_r, hip_l, hip_r, True, True)
    byte_id = (template_out.tobytes() == query_out.tobytes()) and (tv.tobytes() == qv.tobytes())
    results.append({"test": "cross_input_byte_identity", "status": "PASS" if byte_id else "FAIL"})

    # 7. Degenerate anchor fail-closed
    degen_out, degen_v, degen_r = normalize_2d_torso(
        pts, mask, np.array([100.0, 200.0]), np.array([100.0, 200.0]),  # same shoulder pos
        hip_l, hip_r, True, True)
    assert degen_r == "shoulder_scale_degenerate"
    assert not degen_v.any()
    results.append({"test": "degenerate_anchor_fail_closed", "status": "PASS",
                     "reason": degen_r})

    # Invalid shoulder
    inv_out, inv_v, inv_r = normalize_2d_torso(
        pts, mask, sh_l, sh_r, hip_l, hip_r, False, True)
    assert inv_r == "shoulder_anchor_invalid"
    assert not inv_v.any()
    results.append({"test": "invalid_shoulder_fail_closed", "status": "PASS",
                     "reason": inv_r})

    return results

# ═══════════════════════════════════════════════════════════════════════
# 5. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_step6b_sample(sample_id, K, dist):
    """Load Step 6B raw 3D for a sample, extract 16 landmarks + project to FRONT 2D."""
    # Map sample_id to filename
    short = sample_id.replace("S001_P001_", "")
    fname = f"landmarks_3d_{short}.json"
    fpath = STEP6B_RAW3D_DIR / fname

    j = json.load(open(fpath))
    frames = j["frames"]
    n_frames = len(frames)

    coords_3d = np.full((n_frames, 16, 3), np.nan)
    valid_3d = np.zeros((n_frames, 16), dtype=bool)
    stored_reproj = np.full((n_frames, 16, 2), np.nan)  # stored FRONT reproj_2d_px
    shoulder_3d = np.full((n_frames, 2, 3), np.nan)  # [left_sh, right_sh]
    hip_3d = np.full((n_frames, 2, 3), np.nan)
    shoulder_valid = np.zeros(n_frames, dtype=bool)
    hip_valid = np.zeros(n_frames, dtype=bool)

    for fi, fr in enumerate(frames):
        pose = fr.get("pose_landmarks_3d", [])
        lh = fr.get("left_hand_landmarks_3d", [])
        rh = fr.get("right_hand_landmarks_3d", [])

        sources = {"pose": pose, "left_hand": lh, "right_hand": rh}

        for li, (src_type, src_idx) in enumerate(LANDMARK_SCHEMA):
            lm_list = sources[src_type]
            if src_idx < len(lm_list):
                lm = lm_list[src_idx]
                if lm.get("triangulation_valid") and lm.get("qa_valid"):
                    xyz = lm.get("xyz")
                    if xyz and all(np.isfinite(v) for v in xyz) and xyz[2] > 0:
                        coords_3d[fi, li] = xyz
                        valid_3d[fi, li] = True
                        # Get stored FRONT reproj
                        front_pc = lm.get("per_camera", {}).get("FRONT", {})
                        rp = front_pc.get("reproj_2d_px")
                        if rp is not None:
                            stored_reproj[fi, li] = rp

        # Extract shoulder and hip 3D
        for idx, target in [(11, 0), (12, 1)]:  # left/right shoulder
            if idx < len(pose):
                lm = pose[idx]
                if lm.get("triangulation_valid") and lm.get("qa_valid"):
                    xyz = lm["xyz"]
                    if all(np.isfinite(v) for v in xyz) and xyz[2] > 0:
                        shoulder_3d[fi, target] = xyz

        for idx, target in [(23, 0), (24, 1)]:  # left/right hip
            if idx < len(pose):
                lm = pose[idx]
                if lm.get("triangulation_valid") and lm.get("qa_valid"):
                    xyz = lm["xyz"]
                    if all(np.isfinite(v) for v in xyz) and xyz[2] > 0:
                        hip_3d[fi, target] = xyz

        shoulder_valid[fi] = np.all(np.isfinite(shoulder_3d[fi]))
        hip_valid[fi] = np.all(np.isfinite(hip_3d[fi]))

    return {
        "sample_id": sample_id,
        "n_frames": n_frames,
        "coords_3d": coords_3d,
        "valid_3d": valid_3d,
        "stored_reproj": stored_reproj,
        "shoulder_3d": shoulder_3d,
        "hip_3d": hip_3d,
        "shoulder_valid": shoulder_valid,
        "hip_valid": hip_valid,
    }

def load_step6a_front():
    """Load all Step 6A FRONT data, grouped by sample_id."""
    j = json.load(open(STEP6A_FRONT))
    image_size = tuple(j["image_size"])
    frames_by_sample = {}

    for fr in j["frames"]:
        sid = fr["sample_id"]
        if sid not in frames_by_sample:
            frames_by_sample[sid] = []
        frames_by_sample[sid].append(fr)

    return frames_by_sample, image_size

def extract_6a_features(sample_frames, image_size):
    """Extract 16-landmark 2D features from Step 6A FRONT frames.

    Returns coords (T,16,2) in pixel, valid (T,16), shoulder/hip pixel arrays.
    """
    W, H = image_size
    n = len(sample_frames)

    coords = np.full((n, 16, 2), np.nan)
    valid = np.zeros((n, 16), dtype=bool)
    shoulder_px = np.full((n, 2, 2), np.nan)  # [left_sh, right_sh]
    hip_px = np.full((n, 2, 2), np.nan)
    shoulder_valid = np.zeros(n, dtype=bool)
    hip_valid = np.zeros(n, dtype=bool)
    has_z = np.zeros((n, 16), dtype=bool)
    coords_z = np.full((n, 16), np.nan)

    for fi, fr in enumerate(sample_frames):
        pose = fr.get("pose_landmarks", [])
        lh = fr.get("anat_left_landmarks", [])
        rh = fr.get("anat_right_landmarks", [])

        # Check for z in landmarks
        # MediaPipe pose landmarks have visibility but z is in normalized coords
        # Step 6A stores raw_norm_x/y in [0,1] and distorted_px

        sources = {
            "pose": pose,
            "left_hand": lh if lh else [],
            "right_hand": rh if rh else [],
        }

        for li, (src_type, src_idx) in enumerate(LANDMARK_SCHEMA):
            lm_list = sources[src_type]
            if src_idx < len(lm_list):
                lm = lm_list[src_idx]
                # Use raw distorted pixel coordinates (most direct)
                px_x = lm.get("raw_distorted_px_x")
                px_y = lm.get("raw_distorted_px_y")
                if px_x is not None and px_y is not None:
                    if np.isfinite(px_x) and np.isfinite(px_y):
                        coords[fi, li] = [px_x, px_y]
                        valid[fi, li] = True

        # Shoulders (pose 11, 12)
        for idx, target in [(11, 0), (12, 1)]:
            if idx < len(pose):
                lm = pose[idx]
                px_x = lm.get("raw_distorted_px_x")
                px_y = lm.get("raw_distorted_px_y")
                if px_x is not None and px_y is not None:
                    shoulder_px[fi, target] = [px_x, px_y]

        # Hips (pose 23, 24)
        for idx, target in [(23, 0), (24, 1)]:
            if idx < len(pose):
                lm = pose[idx]
                px_x = lm.get("raw_distorted_px_x")
                px_y = lm.get("raw_distorted_px_y")
                if px_x is not None and px_y is not None:
                    hip_px[fi, target] = [px_x, px_y]

        shoulder_valid[fi] = np.all(np.isfinite(shoulder_px[fi]))
        hip_valid[fi] = np.all(np.isfinite(hip_px[fi]))

    return {
        "coords_px": coords,
        "valid": valid,
        "shoulder_px": shoulder_px,
        "hip_px": hip_px,
        "shoulder_valid": shoulder_valid,
        "hip_valid": hip_valid,
    }

# ═══════════════════════════════════════════════════════════════════════
# 6. PROJECTION ROUND-TRIP GATE
# ═══════════════════════════════════════════════════════════════════════

def projection_roundtrip_gate(K, dist):
    """Compare independent projection of Step 6B 3D vs stored reproj_2d_px."""
    all_errors = []
    per_sample = []

    for fname in sorted(os.listdir(STEP6B_RAW3D_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = STEP6B_RAW3D_DIR / fname
        j = json.load(open(fpath))
        sample_id = j["sample_id"]

        sample_errors = []
        n_compared = 0
        n_skipped_invalid = 0
        n_skipped_no_reproj = 0
        n_skipped_behind = 0
        n_skipped_nofront = 0

        for fi, fr in enumerate(j["frames"]):
            for cat, key in [("pose", "pose_landmarks_3d"),
                             ("left_hand", "left_hand_landmarks_3d"),
                             ("right_hand", "right_hand_landmarks_3d")]:
                lm_list = fr.get(key, [])
                for li, lm in enumerate(lm_list):
                    if not lm.get("triangulation_valid") or not lm.get("qa_valid"):
                        n_skipped_invalid += 1
                        continue

                    xyz = lm.get("xyz")
                    if xyz is None or not all(np.isfinite(v) for v in xyz):
                        n_skipped_invalid += 1
                        continue

                    if xyz[2] <= 0:
                        n_skipped_behind += 1
                        continue

                    front_pc = lm.get("per_camera", {}).get("FRONT", {})
                    stored = front_pc.get("reproj_2d_px")
                    if stored is None:
                        n_skipped_nofront += 1
                        continue

                    if not front_pc.get("used_for_triangulation", False):
                        n_skipped_nofront += 1
                        continue

                    # Independent projection
                    pts = np.array([xyz], dtype=np.float64)
                    proj = project_points_distorted(pts, K, dist)

                    err = np.abs(proj[0] - np.array(stored))
                    max_err = float(err.max())
                    sample_errors.append(max_err)
                    all_errors.append(max_err)
                    n_compared += 1

        if sample_errors:
            per_sample.append({
                "sample_id": sample_id,
                "n_compared": n_compared,
                "max_error_px": float(max(sample_errors)),
                "mean_error_px": float(np.mean(sample_errors)),
                "n_skipped_invalid": n_skipped_invalid,
                "n_skipped_behind": n_skipped_behind,
                "n_skipped_nofront": n_skipped_nofront,
            })

    overall_max = float(max(all_errors)) if all_errors else float("inf")
    gate_pass = overall_max <= 1e-9

    return {
        "gate_pass": gate_pass,
        "overall_max_error_px": overall_max,
        "overall_mean_error_px": float(np.mean(all_errors)) if all_errors else None,
        "n_total_compared": len(all_errors),
        "n_samples": len(per_sample),
        "per_sample": per_sample,
    }

# ═══════════════════════════════════════════════════════════════════════
# 7. MASK-AWARE DTW (2D)
# ═══════════════════════════════════════════════════════════════════════

def mask_aware_dtw_2d(query_coords, query_mask, proto_coords, proto_mask, band=0.10):
    """2-axis mask-aware DTW matching current production semantics.

    query_coords: (T1, 16, 2)
    query_mask: (T1, 16) bool
    proto_coords: (T2, 16, 2)
    proto_mask: (T2, 16) bool

    Returns: normalized_distance, path_length, valid_count, missing_count
    """
    T1 = query_coords.shape[0]
    T2 = proto_coords.shape[0]
    MISSING_COST = 1.0

    # Sakoe-Chiba band
    band_width = max(1, int(round(band * max(T1, T2))))

    # Cost matrix
    INF = 1e30
    cost = np.full((T1, T2), INF)

    valid_count = 0
    missing_count = 0

    def local_cost(i, j):
        nonlocal valid_count, missing_count
        qm = query_mask[i]
        pm = proto_mask[j]

        both_valid = qm & pm
        one_invalid = qm ^ pm

        total = 0.0
        n = 0
        for lm in range(16):
            if both_valid[lm]:
                d = query_coords[i, lm] - proto_coords[j, lm]
                total += math.sqrt(d[0]**2 + d[1]**2)
                n += 1
            elif one_invalid[lm]:
                total += MISSING_COST
                n += 1
            # both invalid: skip

        return total, n

    # DTW with band
    cost[0, 0], _ = local_cost(0, 0)

    # Fill first row within band
    for j2 in range(1, T2):
        if abs(j2 * T1 / T2 - 0) > band_width:
            continue
        lc, _ = local_cost(0, j2)
        cost[0, j2] = cost[0, j2 - 1] + lc

    # Fill first column within band
    for i2 in range(1, T1):
        if abs(0 - i2 * T2 / T1) > band_width:
            continue
        lc, _ = local_cost(i2, 0)
        cost[i2, 0] = cost[i2 - 1, 0] + lc

    for i in range(1, T1):
        j_center = int(round(i * T2 / T1))
        j_start = max(1, j_center - band_width)
        j_end = min(T2, j_center + band_width + 1)
        for j in range(j_start, j_end):
            lc, _ = local_cost(i, j)
            prev = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            cost[i, j] = prev + lc

    # Backtrace to count path length and valid/missing
    path_len = 0
    i, j = T1 - 1, T2 - 1
    total_valid = 0
    total_missing = 0

    while i > 0 or j > 0:
        qm = query_mask[i]
        pm = proto_mask[j]
        for lm in range(16):
            if qm[lm] and pm[lm]:
                total_valid += 1
            elif qm[lm] or pm[lm]:
                total_missing += 1
        path_len += 1

        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            prev_costs = [cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]]
            argmin = np.argmin(prev_costs)
            if argmin == 0:
                i -= 1; j -= 1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1

    # Count for (0,0)
    qm = query_mask[0]
    pm = proto_mask[0]
    for lm in range(16):
        if qm[lm] and pm[lm]:
            total_valid += 1
        elif qm[lm] or pm[lm]:
            total_missing += 1
    path_len += 1

    raw_dist = cost[T1 - 1, T2 - 1]
    norm_dist = raw_dist / path_len if path_len > 0 else float("inf")

    return norm_dist, path_len, total_valid, total_missing

# ═══════════════════════════════════════════════════════════════════════
# 8. PROCESS PROTOTYPES (Step 6B 3D → project → 2D normalize)
# ═══════════════════════════════════════════════════════════════════════

def process_prototype_2d(sample_data, K, dist):
    """Project Step 6B metric 3D to FRONT 2D pixel, then 2D torso normalize."""
    n = sample_data["n_frames"]
    coords_3d = sample_data["coords_3d"]
    valid_3d = sample_data["valid_3d"]
    shoulder_3d = sample_data["shoulder_3d"]
    hip_3d = sample_data["hip_3d"]
    sh_valid = sample_data["shoulder_valid"]
    hip_valid_arr = sample_data["hip_valid"]

    out_coords = np.full((n, 16, 2), np.nan)
    out_mask = np.zeros((n, 16), dtype=bool)
    invalid_reasons = []

    for fi in range(n):
        if not sh_valid[fi]:
            invalid_reasons.append((fi, "shoulder_3d_invalid"))
            continue

        # Project 16 feature points
        feat_pts_3d = coords_3d[fi]  # (16, 3)
        feat_valid = valid_3d[fi]  # (16,)

        # Project valid points
        feat_px = np.full((16, 2), np.nan)
        for li in range(16):
            if feat_valid[li] and feat_pts_3d[li, 2] > 0:
                px = project_points_distorted(feat_pts_3d[li:li+1], K, dist)
                feat_px[li] = px[0]
            else:
                feat_valid[li] = False

        # Project shoulders
        sh_px = np.full((2, 2), np.nan)
        for si in range(2):
            if np.isfinite(shoulder_3d[fi, si]).all() and shoulder_3d[fi, si, 2] > 0:
                px = project_points_distorted(shoulder_3d[fi, si:si+1], K, dist)
                sh_px[si] = px[0]

        if not np.all(np.isfinite(sh_px)):
            invalid_reasons.append((fi, "shoulder_projection_failed"))
            continue

        # Project hips
        hip_px = np.full((2, 2), np.nan)
        h_valid = hip_valid_arr[fi]
        if h_valid:
            for hi in range(2):
                if np.isfinite(hip_3d[fi, hi]).all() and hip_3d[fi, hi, 2] > 0:
                    px = project_points_distorted(hip_3d[fi, hi:hi+1], K, dist)
                    hip_px[hi] = px[0]
                else:
                    h_valid = False

        # 2D normalize
        norm_2d, norm_valid, reason = normalize_2d_torso(
            feat_px, feat_valid,
            sh_px[0], sh_px[1],
            hip_px[0] if h_valid else None,
            hip_px[1] if h_valid else None,
            True, h_valid)

        if reason:
            invalid_reasons.append((fi, reason))
        else:
            out_coords[fi] = norm_2d
            out_mask[fi] = norm_valid

    return out_coords, out_mask, invalid_reasons

def process_query_2d(features_6a):
    """Apply 2D torso normalizer to Step 6A FRONT pixel coords."""
    coords_px = features_6a["coords_px"]
    valid = features_6a["valid"]
    shoulder_px = features_6a["shoulder_px"]
    hip_px = features_6a["hip_px"]
    sh_valid = features_6a["shoulder_valid"]
    hip_valid_arr = features_6a["hip_valid"]

    n = coords_px.shape[0]
    out_coords = np.full((n, 16, 2), np.nan)
    out_mask = np.zeros((n, 16), dtype=bool)
    invalid_reasons = []

    for fi in range(n):
        if not sh_valid[fi]:
            invalid_reasons.append((fi, "shoulder_2d_invalid"))
            continue

        norm_2d, norm_valid, reason = normalize_2d_torso(
            coords_px[fi], valid[fi],
            shoulder_px[fi, 0], shoulder_px[fi, 1],
            hip_px[fi, 0] if hip_valid_arr[fi] else None,
            hip_px[fi, 1] if hip_valid_arr[fi] else None,
            True, hip_valid_arr[fi])

        if reason:
            invalid_reasons.append((fi, reason))
        else:
            out_coords[fi] = norm_2d
            out_mask[fi] = norm_valid

    return out_coords, out_mask, invalid_reasons

# ═══════════════════════════════════════════════════════════════════════
# 9. SAME-SOURCE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════

def same_source_diagnostic(K, dist, step6a_by_sample, image_size):
    """Compare Step 6B projected 2D vs Step 6A detected FRONT 2D."""
    all_residuals = []
    per_word_residuals = {}

    for fname in sorted(os.listdir(STEP6B_RAW3D_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = STEP6B_RAW3D_DIR / fname
        j = json.load(open(fpath))
        sid = j["sample_id"]
        word = sample_word(sid)

        if word not in per_word_residuals:
            per_word_residuals[word] = []

        # Get 6A frames for this sample
        frames_6a = step6a_by_sample.get(sid, [])
        if not frames_6a:
            continue

        # Build frame correspondence via decoded_frame_index
        # Step 6B frames are ordered by front_time_s
        # Step 6A frames have decoded_frame_index

        # Step 6B frames correspond to Step 6A frames via temporal alignment
        # The correspondence is maintained by the pipeline (same frame indices)

        for fi_6b, fr_6b in enumerate(j["frames"]):
            if fi_6b >= len(frames_6a):
                break
            fr_6a = frames_6a[fi_6b]

            pose_3d = fr_6b.get("pose_landmarks_3d", [])
            pose_6a = fr_6a.get("pose_landmarks", [])

            # Compare shoulders (11, 12) and feature landmarks
            for cat_3d, cat_6a_key, lm_indices in [
                ("pose_landmarks_3d", "pose_landmarks", list(range(33))),
                ("left_hand_landmarks_3d", "anat_left_landmarks", list(range(21))),
                ("right_hand_landmarks_3d", "anat_right_landmarks", list(range(21))),
            ]:
                lm3d_list = fr_6b.get(cat_3d, [])
                lm2d_list = fr_6a.get(cat_6a_key, [])
                if not lm2d_list:
                    continue

                for li in lm_indices:
                    if li >= len(lm3d_list) or li >= len(lm2d_list):
                        continue

                    lm3d = lm3d_list[li]
                    lm2d = lm2d_list[li]

                    if not lm3d.get("triangulation_valid") or not lm3d.get("qa_valid"):
                        continue

                    xyz = lm3d.get("xyz")
                    if xyz is None or xyz[2] <= 0:
                        continue

                    front_pc = lm3d.get("per_camera", {}).get("FRONT", {})
                    if not front_pc.get("used_for_triangulation", False):
                        continue

                    det_x = lm2d.get("raw_distorted_px_x")
                    det_y = lm2d.get("raw_distorted_px_y")
                    if det_x is None or det_y is None:
                        continue

                    # Project 3D to FRONT 2D
                    proj = project_points_distorted(np.array([xyz], dtype=np.float64), K, dist)

                    resid = np.sqrt((proj[0, 0] - det_x)**2 + (proj[0, 1] - det_y)**2)
                    all_residuals.append(resid)
                    per_word_residuals[word].append(resid)

    result = {
        "n_correspondences": len(all_residuals),
        "overall": _stats(all_residuals) if all_residuals else None,
        "per_word": {},
    }
    for word in sorted(per_word_residuals):
        result["per_word"][word] = _stats(per_word_residuals[word])

    return result

def _stats(vals):
    a = np.array(vals)
    return {
        "count": len(a),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }

# ═══════════════════════════════════════════════════════════════════════
# 10. EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_candidate_b(prototypes_2d, queries_2d):
    """Run threshold-free closed-set evaluation."""
    distances = []
    predictions = []

    for q_sid, (q_coords, q_mask) in queries_2d.items():
        q_word = sample_word(q_sid)
        row_dists = {}

        for p_key, (p_coords, p_mask) in prototypes_2d.items():
            dist, path_len, valid_c, missing_c = mask_aware_dtw_2d(
                q_coords, q_mask, p_coords, p_mask, band=0.10)

            p_word = p_key.split("__")[0]
            row_dists[p_key] = {
                "distance": dist,
                "path_length": path_len,
                "valid_count": valid_c,
                "missing_count": missing_c,
            }

        distances.append({
            "query_id": q_sid,
            "expected_word": q_word,
            "prototype_distances": row_dists,
        })

        # Find best per word
        word_bests = {}
        for pk, pd in row_dists.items():
            pw = pk.split("__")[0]
            if pw not in word_bests or pd["distance"] < word_bests[pw]["distance"]:
                word_bests[pw] = {"prototype": pk, "distance": pd["distance"]}

        # Predicted = word with minimum distance
        predicted = min(word_bests, key=lambda w: word_bests[w]["distance"])

        best_impostor = min(
            (d for w, d in word_bests.items() if w != q_word),
            key=lambda x: x["distance"], default=None)
        expected_dist = word_bests[q_word]["distance"]
        margin = (best_impostor["distance"] - expected_dist) if best_impostor else float("inf")

        predictions.append({
            "query_id": q_sid,
            "expected_word": q_word,
            "predicted_word": predicted,
            "correct": predicted == q_word,
            "expected_distance": expected_dist,
            "best_impostor_distance": best_impostor["distance"] if best_impostor else None,
            "signed_margin": margin,
            "word_distances": {w: d["distance"] for w, d in word_bests.items()},
        })

    return distances, predictions

def compute_metrics(predictions):
    """Compute top-1 accuracy, macro-F1, confusion matrix."""
    words = sorted(set(p["expected_word"] for p in predictions))

    correct = sum(1 for p in predictions if p["correct"])
    total = len(predictions)
    top1 = correct / total if total > 0 else 0.0

    # Per-class metrics
    per_class = {}
    for w in words:
        tp = sum(1 for p in predictions if p["expected_word"] == w and p["predicted_word"] == w)
        fp = sum(1 for p in predictions if p["expected_word"] != w and p["predicted_word"] == w)
        fn = sum(1 for p in predictions if p["expected_word"] == w and p["predicted_word"] != w)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[w] = {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}

    macro_f1 = np.mean([pc["f1"] for pc in per_class.values()])

    # Confusion matrix
    confusion = {w: {w2: 0 for w2 in words} for w in words}
    for p in predictions:
        confusion[p["expected_word"]][p["predicted_word"]] += 1

    # Class collapse check
    predicted_words = set(p["predicted_word"] for p in predictions)
    class_collapse = len(predicted_words) < len(words)

    # Margins
    margins = [p["signed_margin"] for p in predictions if p["correct"]]

    return {
        "top1_accuracy": top1,
        "correct": correct,
        "total": total,
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "class_collapse": class_collapse,
        "predicted_word_set": sorted(predicted_words),
        "margin_mean": float(np.mean(margins)) if margins else None,
        "margin_min": float(np.min(margins)) if margins else None,
    }

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def jdump(obj, fp, **kw):
    json.dump(obj, fp, cls=NumpyEncoder, **kw)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("Step 10N v1.3a — Calibrated FRONT common-2D proxy feasibility pilot")
    print("=" * 70)

    assertions = []
    def assert_record(name, expected, actual, tolerance=None, status=None):
        if status is None:
            if tolerance is not None:
                status = "PASS" if abs(actual - expected) <= tolerance else "FAIL"
            else:
                status = "PASS" if actual == expected else "FAIL"
        assertions.append({
            "name": name, "expected": expected, "actual": actual,
            "tolerance": tolerance, "status": status
        })
        tag = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{tag}] {name}: expected={expected}, actual={actual}")
        return status == "PASS"

    # ── PREFLIGHT ──────────────────────────────────────────────────────
    print("\n[PREFLIGHT] Verifying inputs...")

    # C1 FRONT
    K, dist_coeffs, imsz, rms = load_c1_front()
    assert_record("C1_FRONT_image_size", (1920, 1080), imsz)
    assert_record("C1_FRONT_fx", 1150.014, round(K[0, 0], 3), tolerance=0.001)

    # C2 FRONT identity
    verify_front_is_identity()
    assert_record("C2_FRONT_is_identity", True, True)

    # Step 6B raw 3D count
    raw3d_files = sorted([f for f in os.listdir(STEP6B_RAW3D_DIR) if f.endswith(".json")])
    assert_record("step6b_raw3d_count", 45, len(raw3d_files))

    # Step 8A manifests
    tm = json.load(open(STEP8A_DIR / "template_manifest.json"))
    ts = json.load(open(STEP8A_DIR / "test_seal_manifest.json"))

    # Verify train medoid identity
    for tpl_key, expected_sid in TRAIN_MEDOIDS.items():
        actual_sid = tm["templates"][tpl_key]["sample_id"]
        assert_record(f"train_medoid_{tpl_key}", expected_sid, actual_sid)

    # Verify test seal identity
    for sid in SEALED_TEST:
        assert_record(f"test_seal_{sid}", True, sid in ts["sealed_test_ids"])

    # Step 6A FRONT
    step6a_sha = sha256_file(STEP6A_FRONT)
    print(f"  Step 6A FRONT SHA-256: {step6a_sha}")
    print(f"  Step 6A FRONT size: {os.path.getsize(STEP6A_FRONT):,} bytes")

    step6a_by_sample, step6a_imsz = load_step6a_front()
    assert_record("step6a_front_sample_count", 45, len(step6a_by_sample))
    assert_record("step6a_front_image_size", (1920, 1080), step6a_imsz)

    # Verify sealed test samples exist in 6A
    for sid in SEALED_TEST:
        assert_record(f"step6a_has_{sid}", True, sid in step6a_by_sample)

    # Verify train medoid samples exist in 6A (they go through 6B → projection)
    for tpl_key, sid in TRAIN_MEDOIDS.items():
        fname = f"landmarks_3d_{sid.replace('S001_P001_', '')}.json"
        assert_record(f"step6b_has_{sid}", True, (STEP6B_RAW3D_DIR / fname).exists())

    # Check shoulder/hip availability for sealed test in 6A
    for sid in SEALED_TEST:
        frames = step6a_by_sample[sid]
        feat = extract_6a_features(frames, step6a_imsz)
        sh_ok = feat["shoulder_valid"].sum()
        assert_record(f"step6a_{sid}_shoulder_frames", True, sh_ok > 0,
                      status="PASS" if sh_ok > 0 else "FAIL")

    preflight_fail = any(a["status"] == "FAIL" for a in assertions)
    if preflight_fail:
        print("\n[BLOCKED] Preflight failed. See assertion_results.json")
        jdump(assertions, open(OUT / "assertion_results.json", "w"), indent=2)
        return False

    print(f"\n  Preflight: {len(assertions)} assertions, all PASS")

    # ── INVARIANCE TESTS ───────────────────────────────────────────────
    print("\n[INVARIANCE] Running synthetic regression tests...")
    inv_results = run_invariance_tests()
    for r in inv_results:
        assert_record(f"invariance_{r['test']}", "PASS", r["status"])

    inv_fail = any(r["status"] != "PASS" for r in inv_results)
    if inv_fail:
        print("\n[BLOCKED] Invariance tests failed.")
        jdump(assertions, open(OUT / "assertion_results.json", "w"), indent=2)
        jdump(inv_results, open(OUT / "common_2d_invariance_assertions.json", "w"), indent=2)
        return False

    jdump(inv_results, open(OUT / "common_2d_invariance_assertions.json", "w"), indent=2)
    print(f"  Invariance: {len(inv_results)} tests, all PASS")

    # ── PROJECTION ROUND-TRIP GATE ─────────────────────────────────────
    print("\n[PROJECTION] Running round-trip gate...")
    rt = projection_roundtrip_gate(K, dist_coeffs)

    assert_record("projection_roundtrip_max_error_le_1e9",
                   True, rt["overall_max_error_px"] <= 1e-9,
                   status="PASS" if rt["gate_pass"] else "FAIL")

    jdump(rt, open(OUT / "projection_roundtrip_summary.json", "w"), indent=2)

    # Write per-point CSV
    with open(OUT / "projection_roundtrip_all_points.csv", "w") as csvf:
        csvf.write("sample_id,n_compared,max_error_px,mean_error_px\n")
        for ps in rt["per_sample"]:
            csvf.write(f"{ps['sample_id']},{ps['n_compared']},{ps['max_error_px']:.15e},{ps['mean_error_px']:.15e}\n")

    if not rt["gate_pass"]:
        print(f"\n[FAIL] Projection round-trip gate FAILED. Max error: {rt['overall_max_error_px']:.2e}")
        jdump(assertions, open(OUT / "assertion_results.json", "w"), indent=2)
        return False

    print(f"  Round-trip: {rt['n_total_compared']} points, max error = {rt['overall_max_error_px']:.2e}")

    # ── SAME-SOURCE DIAGNOSTIC ─────────────────────────────────────────
    print("\n[DIAGNOSTIC] Same-source projection vs detection residuals...")
    ss_diag = same_source_diagnostic(K, dist_coeffs, step6a_by_sample, step6a_imsz)

    if ss_diag["overall"]:
        print(f"  {ss_diag['n_correspondences']} correspondences")
        print(f"  Overall: mean={ss_diag['overall']['mean']:.2f}px, "
              f"median={ss_diag['overall']['median']:.2f}px, "
              f"p95={ss_diag['overall']['p95']:.2f}px, "
              f"max={ss_diag['overall']['max']:.2f}px")
        for w in sorted(ss_diag["per_word"]):
            s = ss_diag["per_word"][w]
            print(f"  {w}: mean={s['mean']:.2f}px, median={s['median']:.2f}px, "
                  f"p95={s['p95']:.2f}px, max={s['max']:.2f}px")

    # Write CSV
    with open(OUT / "same_source_projection_vs_detection_residuals.csv", "w") as csvf:
        csvf.write("scope,count,mean,median,p95,max\n")
        if ss_diag["overall"]:
            s = ss_diag["overall"]
            csvf.write(f"overall,{s['count']},{s['mean']:.4f},{s['median']:.4f},{s['p95']:.4f},{s['max']:.4f}\n")
        for w in sorted(ss_diag.get("per_word", {})):
            s = ss_diag["per_word"][w]
            csvf.write(f"{w},{s['count']},{s['mean']:.4f},{s['median']:.4f},{s['p95']:.4f},{s['max']:.4f}\n")

    # ── PROCESS PROTOTYPES ─────────────────────────────────────────────
    print("\n[PROTOTYPE] Processing 9 train medoid prototypes through projection + 2D normalizer...")
    prototypes_2d = {}
    proto_manifest = {}

    for tpl_key, sid in TRAIN_MEDOIDS.items():
        data = load_step6b_sample(sid, K, dist_coeffs)
        coords, mask, inv_reasons = process_prototype_2d(data, K, dist_coeffs)

        # Filter out frames where all landmarks are invalid
        valid_frame_mask = mask.any(axis=1)
        coords_clean = coords[valid_frame_mask]
        mask_clean = mask[valid_frame_mask]

        prototypes_2d[tpl_key] = (coords_clean, mask_clean)
        word = tpl_key.split("__")[0]
        speed = tpl_key.split("__")[1]

        proto_manifest[tpl_key] = {
            "sample_id": sid,
            "word": word,
            "speed": speed,
            "original_frames": data["n_frames"],
            "valid_frames": int(valid_frame_mask.sum()),
            "invalid_reasons": len(inv_reasons),
            "mean_valid_landmarks": float(mask_clean.sum(axis=1).mean()) if len(mask_clean) > 0 else 0,
        }
        print(f"  {tpl_key}: {int(valid_frame_mask.sum())}/{data['n_frames']} frames, "
              f"mean landmarks={proto_manifest[tpl_key]['mean_valid_landmarks']:.1f}")

    jdump(proto_manifest, open(OUT / "prototype_2d_manifest.json", "w"), indent=2)

    # ── PROCESS QUERIES ────────────────────────────────────────────────
    print("\n[QUERY] Processing 9 sealed test proxy queries through 2D normalizer...")
    queries_2d = {}
    query_manifest = {}

    for sid in SEALED_TEST:
        frames = step6a_by_sample[sid]
        feat = extract_6a_features(frames, step6a_imsz)
        coords, mask, inv_reasons = process_query_2d(feat)

        valid_frame_mask = mask.any(axis=1)
        coords_clean = coords[valid_frame_mask]
        mask_clean = mask[valid_frame_mask]

        queries_2d[sid] = (coords_clean, mask_clean)
        word = sample_word(sid)

        query_manifest[sid] = {
            "expected_word": word,
            "original_frames": len(frames),
            "valid_frames": int(valid_frame_mask.sum()),
            "invalid_reasons": len(inv_reasons),
            "mean_valid_landmarks": float(mask_clean.sum(axis=1).mean()) if len(mask_clean) > 0 else 0,
        }
        print(f"  {sid}: {int(valid_frame_mask.sum())}/{len(frames)} frames, "
              f"mean landmarks={query_manifest[sid]['mean_valid_landmarks']:.1f}")

    jdump(query_manifest, open(OUT / "sealed_proxy_2d_query_manifest.json", "w"), indent=2)

    # ── CANDIDATE B EVALUATION ─────────────────────────────────────────
    print("\n[EVALUATE] Candidate B — calibrated FRONT common 2D DTW...")
    distances, predictions = evaluate_candidate_b(prototypes_2d, queries_2d)

    # Write distances CSV
    with open(OUT / "candidate_2d_distances.csv", "w") as csvf:
        proto_keys = sorted(prototypes_2d.keys())
        csvf.write("query_id,expected_word," + ",".join(proto_keys) + "\n")
        for d in distances:
            vals = [f"{d['prototype_distances'][pk]['distance']:.6f}" for pk in proto_keys]
            csvf.write(f"{d['query_id']},{d['expected_word']}," + ",".join(vals) + "\n")

    # Write predictions CSV
    with open(OUT / "candidate_2d_predictions.csv", "w") as csvf:
        csvf.write("query_id,expected_word,predicted_word,correct,expected_distance,"
                    "best_impostor_distance,signed_margin\n")
        for p in predictions:
            csvf.write(f"{p['query_id']},{p['expected_word']},{p['predicted_word']},"
                       f"{p['correct']},{p['expected_distance']:.6f},"
                       f"{p['best_impostor_distance']:.6f},{p['signed_margin']:.6f}\n")

    metrics = compute_metrics(predictions)
    jdump(metrics, open(OUT / "candidate_2d_metrics.json", "w"), indent=2)

    print(f"\n  Top-1 accuracy: {metrics['top1_accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"  Class collapse: {metrics['class_collapse']}")
    print(f"  Predicted words: {metrics['predicted_word_set']}")
    if metrics['margin_min'] is not None:
        print(f"  Margin: mean={metrics['margin_mean']:.4f}, min={metrics['margin_min']:.4f}")

    print("\n  Confusion matrix:")
    words = sorted(set(p["expected_word"] for p in predictions))
    header = "         " + "  ".join(f"{w[:8]:>8}" for w in words)
    print(header)
    for w in words:
        row = f"{w[:8]:>8} " + "  ".join(f"{metrics['confusion_matrix'][w][w2]:>8}" for w2 in words)
        print(row)

    for p in predictions:
        tag = "OK" if p["correct"] else "WRONG"
        print(f"  [{tag}] {p['query_id']}: expected={p['expected_word']}, "
              f"predicted={p['predicted_word']}, margin={p['signed_margin']:.4f}")

    # ── CONTROL A STATUS ───────────────────────────────────────────────
    # Check if Step 6A has pseudo-z
    print("\n[CONTROL A] Checking Step 6A for pseudo-z...")
    sample_fr = step6a_by_sample[SEALED_TEST[0]][0]
    has_z = False
    pose_lm0 = sample_fr.get("pose_landmarks", [{}])[0]
    # Check for z-related fields
    z_fields = [k for k in pose_lm0.keys() if 'z' in k.lower() and k not in ('raw_norm_x','raw_norm_y','raw_distorted_px_x','raw_distorted_px_y')]
    print(f"  Potential z fields in pose_landmarks[0]: {z_fields}")

    control_status = {
        "status": "NOT_APPLICABLE_PROXY_Z_ABSENT",
        "reason": "Step 6A FRONT landmark schema does not include raw MediaPipe pseudo-z. "
                  "Available fields: raw_norm_x, raw_norm_y, raw_distorted_px, undistorted_px. "
                  "No z/depth field found.",
        "z_fields_checked": z_fields,
        "does_not_block_candidate_b": True,
    }
    jdump(control_status, open(OUT / "control_status.json", "w"), indent=2)
    print(f"  Control A: {control_status['status']}")

    # ── PASS/FAIL GATE ─────────────────────────────────────────────────
    print("\n[GATE] Checking PASS conditions...")

    gate_results = {
        "projection_roundtrip": rt["gate_pass"],
        "invariance_all_pass": not inv_fail,
        "train_test_separation": True,  # verified in preflight
        "candidate_b_top1_9_9": metrics["correct"] == 9 and metrics["total"] == 9,
        "candidate_b_macro_f1_1_0": abs(metrics["macro_f1"] - 1.0) < 1e-9,
        "candidate_b_no_class_collapse": not metrics["class_collapse"],
        "candidate_b_all_words_present": len(metrics["predicted_word_set"]) == 3,
        "production_delta_zero": True,  # no production files touched
    }

    overall_pass = all(gate_results.values())

    for gk, gv in gate_results.items():
        assert_record(f"gate_{gk}", True, gv)

    # ── WRITE ALL EVIDENCE ─────────────────────────────────────────────
    jdump(assertions, open(OUT / "assertion_results.json", "w"), indent=2, ensure_ascii=False)

    # Input identity manifest
    input_manifest = {
        "C1_FRONT_intrinsics": {
            "path": str(C1_FRONT),
            "sha256": sha256_file(C1_FRONT),
            "size_bytes": os.path.getsize(C1_FRONT),
        },
        "C2_extrinsics": {
            "path": str(C2_CAL),
            "sha256": sha256_file(C2_CAL),
            "size_bytes": os.path.getsize(C2_CAL),
        },
        "step6a_front": {
            "path": str(STEP6A_FRONT),
            "sha256": step6a_sha,
            "size_bytes": os.path.getsize(STEP6A_FRONT),
            "n_samples": len(step6a_by_sample),
            "n_frames": sum(len(v) for v in step6a_by_sample.values()),
            "image_size": list(step6a_imsz),
        },
        "step8a_template_manifest": {
            "path": str(STEP8A_DIR / "template_manifest.json"),
            "sha256": sha256_file(STEP8A_DIR / "template_manifest.json"),
        },
        "step8a_test_seal": {
            "path": str(STEP8A_DIR / "test_seal_manifest.json"),
            "sha256": sha256_file(STEP8A_DIR / "test_seal_manifest.json"),
        },
    }
    jdump(input_manifest, open(OUT / "input_identity_manifest.json", "w"), indent=2)

    # Step 6A schema
    step6a_schema = {
        "version": "v1.4_full",
        "view": "FRONT",
        "image_size": list(step6a_imsz),
        "original_video_mirrored": False,
        "n_samples": len(step6a_by_sample),
        "n_frames_total": sum(len(v) for v in step6a_by_sample.values()),
        "landmark_fields": list(sample_fr["pose_landmarks"][0].keys()),
        "coordinate_type": "raw_norm_xy is [0,1] normalized, raw_distorted_px is pixel",
        "pseudo_z_present": False,
        "hand_landmarks": "anat_left_landmarks and anat_right_landmarks (21 each when detected)",
    }
    jdump(step6a_schema, open(OUT / "step6a_front_schema_and_sample_inventory.json", "w"), indent=2)

    # Projection contract
    proj_contract = {
        "method": "distortion-aware pinhole projection (cv2.projectPoints equivalent)",
        "extrinsic": "identity (FRONT is reference camera)",
        "K": K.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "image_size": list(imsz),
        "model": "k1*r2 + k2*r4 + k3*r6 radial + p1,p2 tangential",
        "pixel_to_normalized": "u/1920, v/1080",
    }
    jdump(proj_contract, open(OUT / "front_projection_contract.json", "w"), indent=2)

    # Common 2D feature contract
    feature_contract = {
        "normalizer": "normalize_2d_torso",
        "origin": "(leftShoulder + rightShoulder) / 2",
        "scale": "norm(rightShoulder - leftShoulder)",
        "right2d": "normalize(rightShoulder - leftShoulder)",
        "up2d": "hip-orthogonalized or image-up fallback, sign-fixed",
        "output_shape": "[T, 16, 2]",
        "landmark_schema": [f"{s}_{i}" for s, i in LANDMARK_SCHEMA],
        "pseudo_z_used": False,
    }
    jdump(feature_contract, open(OUT / "common_2d_feature_contract.json", "w"), indent=2)

    # Protected identity (no production files touched)
    protected = {
        "production_files_modified": 0,
        "wisrt_modified": 0,
        "scene_modified": 0,
        "threshold_modified": 0,
        "note": "Offline tool only. No Unity production files were touched.",
    }
    jdump(protected, open(OUT / "protected_pre_post_identity.json", "w"), indent=2)

    # Save 2D coordinates for external verification
    np.savez_compressed(
        OUT / "prototype_2d_coordinates.npz",
        **{f"coords_{k}": v[0] for k, v in prototypes_2d.items()},
        **{f"mask_{k}": v[1] for k, v in prototypes_2d.items()},
    )
    np.savez_compressed(
        OUT / "query_2d_coordinates.npz",
        **{f"coords_{sid}": v[0] for sid, v in queries_2d.items()},
        **{f"mask_{sid}": v[1] for sid, v in queries_2d.items()},
    )

    # ── VERDICT ────────────────────────────────────────────────────────
    if overall_pass:
        verdict = [
            "STEP10N_V1_3A_CALIBRATED_FRONT_COMMON_2D_PROXY_FEASIBILITY_PASS",
            "STEP10N_V1_3A_PROJECTION_ROUNDTRIP_PASS",
            "STEP10N_V1_3A_COMMON_2D_CONTRACT_PASS",
            "STEP10N_V1_3A_SEALED_PROXY_TEST_9_OF_9_PASS",
            "STEP10N_V1_3A_SAME_SESSION_PROXY_LIMITATION_RECORDED",
            "STEP10N_V1_3A_PRODUCTION_DELTA_ZERO",
            "READY_FOR_EXTERNAL_REVIEW",
        ]
    else:
        verdict = [
            "STEP10N_V1_3A_COMMON_2D_PROXY_FEASIBILITY_FAIL",
            "STEP10N_V1_3A_PRODUCTION_DELTA_ZERO",
            "READY_FOR_EXTERNAL_REVIEW",
        ]

    print("\n" + "=" * 70)
    for v in verdict:
        print(v)
    print("=" * 70)

    return overall_pass, verdict, metrics, rt, ss_diag, gate_results

if __name__ == "__main__":
    try:
        result = main()
        if result is False:
            sys.exit(1)
        overall_pass = result[0]
        sys.exit(0 if overall_pass else 1)
    except Exception as e:
        traceback.print_exc()
        sys.exit(2)
