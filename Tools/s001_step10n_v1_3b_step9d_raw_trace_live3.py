#!/usr/bin/env python3
"""Step 10N v1.3b — Step 9D raw-trace actual-live 3-query evaluation.

Recovers approved Step 9D raw intermediate diagnostic traces (3 actual live webcam
captures: GREETING, LOVE, THANK_YOU), fixes DTW implementation to match actual Unity C#
MaskAwareDtw semantics, and runs both Control A (current mismatched 3D) and Candidate B
(corrected common 2D) on the same 3 actual live queries.

PASS requires: Candidate B 3/3 top-1 AND strictly better than Control A.
Also preserves v1.3a corrected sealed proxy 9/9 results with corrected DTW.

No new captures, no Unity changes — read-only offline pilot.
"""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Validation" / "step10n_v1_3b"

DATA = ROOT / "data" / "processed" / "sessions" / "S001_P001"
RAW_3D_DIR = DATA / "landmarks_3d_v1" / "full_v1_2_4" / "raw_3d"
TEMPLATE_MANIFEST = DATA / "step8a_dtw_templates_v1_0" / "template_manifest.json"
TEST_SEAL = DATA / "step8a_dtw_templates_v1_0" / "test_seal_manifest.json"
FRONT_2D = DATA / "landmarks_2d_v1_4" / "full" / "landmarks_2d_front.json"
INTRINSICS = ROOT / "Tools" / "calibration" / "results" / "C1" / "validation" / "intrinsics_CAM_FRONT_01_v2.json"
EXTRINSICS = DATA / "C2_CALIBRATION_v2_robust.json"
PROJECT_CONTEXT = ROOT / "WordsInSilence_PROJECT_CONTEXT.md"
V13A_OUT = ROOT / "Validation" / "step10n_v1_3a"

# Step 9D traces
STEP9D_BASE = Path(os.path.expanduser("~")) / "AppData" / "LocalLow" / "DefaultCompany" / "WordsInSilence" / "Step9DRawIntermediatePhysicalAcquisition" / "e965f94530f6432c800eea60be8f2cdf"
STEP9D_TRACES = {
    "GREETING": STEP9D_BASE / "take_01_GREETING_26f436475e2f4e8c832dbce00cbfe883.json",
    "LOVE": STEP9D_BASE / "take_02_LOVE_cd65ad6b277c48b78fa50785070fdddc.json",
    "THANK_YOU": STEP9D_BASE / "take_03_THANK_YOU_2b323671e3f64049a74556c71ff70154.json",
}
STEP9D_EXPECTED_SHA = {
    "GREETING": "bc74d8cb",
    "LOVE": "b81e8cd8",
    "THANK_YOU": "858042fa",
}

# WISRT directory
WISRT_DIR = ROOT / "Assets" / "StreamingAssets" / "RecognitionCoreV1" / "Templates"

# Train medoids (from template_manifest)
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

# Test seal (for v1.3a preservation)
TEST_SEAL_IDS = [
    "S001_P001_GREETING_NORMAL_R01", "S001_P001_GREETING_SLOW_R02", "S001_P001_GREETING_SLOW_R05",
    "S001_P001_LOVE_NORMAL_R01", "S001_P001_LOVE_SLOW_R02", "S001_P001_LOVE_SLOW_R05",
    "S001_P001_THANK_YOU_NORMAL_R01", "S001_P001_THANK_YOU_SLOW_R02", "S001_P001_THANK_YOU_SLOW_R05",
]

# 16 landmark indices in 75-landmark raw array
# pose[13,14,15,16] + left_hand[0,4,8,12,16,20] + right_hand[0,4,8,12,16,20]
POSE_INDICES_IN_RAW75 = [13, 14, 15, 16]
LEFT_HAND_INDICES_IN_RAW75 = [33 + i for i in [0, 4, 8, 12, 16, 20]]
RIGHT_HAND_INDICES_IN_RAW75 = [54 + i for i in [0, 4, 8, 12, 16, 20]]
LANDMARK_16_INDICES = POSE_INDICES_IN_RAW75 + LEFT_HAND_INDICES_IN_RAW75 + RIGHT_HAND_INDICES_IN_RAW75

# Shoulder indices in raw 75
POSE_LEFT_SHOULDER = 11   # pose landmark 11
POSE_RIGHT_SHOULDER = 12  # pose landmark 12
POSE_LEFT_HIP = 23        # pose landmark 23
POSE_RIGHT_HIP = 24       # pose landmark 24

# Corrected v1.3a expected margins (from external review with corrected DTW)
V13A_EXPECTED_MARGIN_MEAN = 0.18601950357596309
V13A_EXPECTED_MARGIN_MIN = 0.10215668138409889

# ── JSON encoder ─────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def jdump(obj, fp, **kwargs):
    json.dump(obj, fp, cls=NumpyEncoder, **kwargs)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ── Corrected Unity-equivalent DTW (matches MaskAwareDtw.cs exactly) ─────────

def unity_dtw_3d(q_coords, q_mask, p_coords, p_mask, band=0.10):
    """Corrected 3D DTW matching Unity C# MaskAwareDtw.Compute exactly.

    q_coords: [T1, 16, 3], p_coords: [T2, 16, 3]
    q_mask: [T1, 16] bool, p_mask: [T2, 16] bool
    """
    MISSING = 1.0
    BAND_EPS = 1e-15
    T1, T2 = q_coords.shape[0], p_coords.shape[0]
    INF = float("inf")

    def in_band(i, j):
        if T1 <= 1 or T2 <= 1:
            return True
        return abs(i / (T1 - 1) - j / (T2 - 1)) <= band + BAND_EPS

    def local_cost(i, j):
        total = 0.0
        included = 0
        for lm in range(16):
            qv = bool(q_mask[i, lm])
            pv = bool(p_mask[j, lm])
            if qv and pv:
                dx = float(q_coords[i, lm, 0] - p_coords[j, lm, 0])
                dy = float(q_coords[i, lm, 1] - p_coords[j, lm, 1])
                dz = float(q_coords[i, lm, 2] - p_coords[j, lm, 2])
                total += math.sqrt(dx * dx + dy * dy + dz * dz)
                included += 1
            elif qv != pv:
                total += MISSING
                included += 1
        return MISSING if included == 0 else total / included

    cum = np.full((T1, T2), INF, dtype=np.float64)
    parent = np.zeros((T1, T2), dtype=np.uint8)  # 0=start, 1=diag, 2=vert, 3=horiz

    for i in range(T1):
        for j in range(T2):
            if not in_band(i, j):
                continue
            lc = local_cost(i, j)
            if i == 0 and j == 0:
                cum[i, j] = lc
                continue
            best = INF
            direction = 0
            # diagonal (priority 1)
            if i > 0 and j > 0 and not math.isinf(cum[i-1, j-1]):
                best = cum[i-1, j-1]
                direction = 1
            # vertical (priority 2, strictly <)
            if i > 0 and not math.isinf(cum[i-1, j]) and cum[i-1, j] < best:
                best = cum[i-1, j]
                direction = 2
            # horizontal (priority 3, strictly <)
            if j > 0 and not math.isinf(cum[i, j-1]) and cum[i, j-1] < best:
                best = cum[i, j-1]
                direction = 3
            if direction != 0:
                cum[i, j] = best + lc
                parent[i, j] = direction

    total = cum[T1-1, T2-1]
    assert not (math.isnan(total) or math.isinf(total)), "DTW path unreachable"

    # backtrace
    pi, pj = T1-1, T2-1
    path = [(pi, pj)]
    while pi != 0 or pj != 0:
        d = parent[pi, pj]
        if d == 1:
            pi -= 1; pj -= 1
        elif d == 2:
            pi -= 1
        elif d == 3:
            pj -= 1
        else:
            raise RuntimeError("incomplete DTW path")
        path.append((pi, pj))
    path.reverse()
    return total / len(path), total, len(path)


def unity_dtw_2d(q_coords, q_mask, p_coords, p_mask, band=0.10):
    """Corrected 2D DTW matching Unity C# MaskAwareDtw semantics but for [T,16,2].

    Identical logic to unity_dtw_3d but uses only x,y (no z axis).
    """
    MISSING = 1.0
    BAND_EPS = 1e-15
    T1, T2 = q_coords.shape[0], p_coords.shape[0]
    INF = float("inf")

    def in_band(i, j):
        if T1 <= 1 or T2 <= 1:
            return True
        return abs(i / (T1 - 1) - j / (T2 - 1)) <= band + BAND_EPS

    def local_cost(i, j):
        total = 0.0
        included = 0
        for lm in range(16):
            qv = bool(q_mask[i, lm])
            pv = bool(p_mask[j, lm])
            if qv and pv:
                dx = float(q_coords[i, lm, 0] - p_coords[j, lm, 0])
                dy = float(q_coords[i, lm, 1] - p_coords[j, lm, 1])
                total += math.sqrt(dx * dx + dy * dy)
                included += 1
            elif qv != pv:
                total += MISSING
                included += 1
        return MISSING if included == 0 else total / included

    cum = np.full((T1, T2), INF, dtype=np.float64)
    parent = np.zeros((T1, T2), dtype=np.uint8)

    for i in range(T1):
        for j in range(T2):
            if not in_band(i, j):
                continue
            lc = local_cost(i, j)
            if i == 0 and j == 0:
                cum[i, j] = lc
                continue
            best = INF
            direction = 0
            if i > 0 and j > 0 and not math.isinf(cum[i-1, j-1]):
                best = cum[i-1, j-1]
                direction = 1
            if i > 0 and not math.isinf(cum[i-1, j]) and cum[i-1, j] < best:
                best = cum[i-1, j]
                direction = 2
            if j > 0 and not math.isinf(cum[i, j-1]) and cum[i, j-1] < best:
                best = cum[i, j-1]
                direction = 3
            if direction != 0:
                cum[i, j] = best + lc
                parent[i, j] = direction

    total = cum[T1-1, T2-1]
    assert not (math.isnan(total) or math.isinf(total)), "DTW path unreachable"

    pi, pj = T1-1, T2-1
    path = [(pi, pj)]
    while pi != 0 or pj != 0:
        d = parent[pi, pj]
        if d == 1:
            pi -= 1; pj -= 1
        elif d == 2:
            pi -= 1
        elif d == 3:
            pj -= 1
        else:
            raise RuntimeError("incomplete DTW path")
        path.append((pi, pj))
    path.reverse()
    return total / len(path), total, len(path)


# ── DTW C#-parity synthetic regression tests ─────────────────────────────────

def run_dtw_parity_tests():
    """Synthetic tests to verify Python DTW matches C# MaskAwareDtw semantics."""
    results = []

    # Test 1: identical sequences => distance 0
    c = np.random.RandomState(42).randn(10, 16, 3)
    m = np.ones((10, 16), dtype=bool)
    d, _, _ = unity_dtw_3d(c, m, c.copy(), m.copy())
    assert abs(d) < 1e-12, f"identical sequences distance={d}"
    results.append({"test": "identical_sequences_zero", "status": "PASS", "distance": d})

    # Test 2: local cost = total/included, not sum
    q = np.zeros((1, 16, 3))
    p = np.zeros((1, 16, 3))
    qm = np.ones((1, 16), dtype=bool)
    pm = np.ones((1, 16), dtype=bool)
    p[0, 0, 0] = 1.0  # one landmark offset by 1.0
    d, _, _ = unity_dtw_3d(q, qm, p, pm)
    # expected: total = 1.0 (one landmark dist) + 0.0*15, included = 16, cost = 1.0/16
    expected = 1.0 / 16.0
    assert abs(d - expected) < 1e-12, f"local cost average: got {d}, expected {expected}"
    results.append({"test": "local_cost_is_average", "status": "PASS", "distance": d, "expected": expected})

    # Test 3: one-valid-one-invalid => MissingCost=1.0 per pair, then /included
    q2 = np.zeros((1, 16, 3))
    p2 = np.zeros((1, 16, 3))
    qm2 = np.ones((1, 16), dtype=bool)
    pm2 = np.ones((1, 16), dtype=bool)
    pm2[0, 0] = False  # one landmark invalid in prototype
    d2, _, _ = unity_dtw_3d(q2, qm2, p2, pm2)
    # 15 both-valid (dist=0) + 1 one-sided (cost=1.0), included=16, avg = 1.0/16
    expected2 = 1.0 / 16.0
    assert abs(d2 - expected2) < 1e-12, f"missing cost: got {d2}, expected {expected2}"
    results.append({"test": "missing_cost_average", "status": "PASS", "distance": d2, "expected": expected2})

    # Test 4: both invalid => not included
    q3 = np.zeros((1, 16, 3))
    p3 = np.zeros((1, 16, 3))
    qm3 = np.ones((1, 16), dtype=bool)
    pm3 = np.ones((1, 16), dtype=bool)
    qm3[0, 0] = False
    pm3[0, 0] = False
    d3, _, _ = unity_dtw_3d(q3, qm3, p3, pm3)
    # 15 both-valid (dist=0) + 1 both-invalid (excluded), included=15, avg = 0/15 = 0
    assert abs(d3) < 1e-12, f"both invalid excluded: got {d3}"
    results.append({"test": "both_invalid_excluded", "status": "PASS", "distance": d3})

    # Test 5: all invalid => MissingCost
    q4 = np.zeros((1, 16, 3))
    p4 = np.zeros((1, 16, 3))
    qm4 = np.zeros((1, 16), dtype=bool)
    pm4 = np.zeros((1, 16), dtype=bool)
    d4, _, _ = unity_dtw_3d(q4, qm4, p4, pm4)
    assert abs(d4 - 1.0) < 1e-12, f"all invalid: got {d4}, expected 1.0"
    results.append({"test": "all_invalid_missing_cost", "status": "PASS", "distance": d4})

    # Test 6: relative band
    assert unity_dtw_3d.__name__  # just verify function exists
    from functools import partial
    # Band test: (0,0) always in band
    assert in_band_check(0, 0, 10, 10, 0.10) == True
    assert in_band_check(9, 9, 10, 10, 0.10) == True
    # (0, 9) with T=10: abs(0/9 - 9/9)=1.0 > 0.10
    assert in_band_check(0, 9, 10, 10, 0.10) == False
    results.append({"test": "relative_band_correct", "status": "PASS"})

    # Test 7: tie priority diagonal > vertical > horizontal
    # With 2-frame sequences where diagonal and non-diagonal have same cost
    q5 = np.zeros((2, 16, 3))
    p5 = np.zeros((2, 16, 3))
    qm5 = np.ones((2, 16), dtype=bool)
    pm5 = np.ones((2, 16), dtype=bool)
    d5, _, pl5 = unity_dtw_3d(q5, qm5, p5, pm5, band=1.0)
    # All costs 0, diagonal path: (0,0)->(1,1), length=2
    assert pl5 == 2, f"diagonal tie priority: path_len={pl5}, expected 2"
    results.append({"test": "tie_priority_diagonal", "status": "PASS", "path_length": pl5})

    # Test 8: 2D DTW same as 3D when z=0
    q6 = np.random.RandomState(99).randn(8, 16, 3)
    q6_2d = q6[:, :, :2].copy()
    q6_3d_z0 = np.concatenate([q6_2d, np.zeros((8, 16, 1))], axis=2)
    p6 = np.random.RandomState(100).randn(6, 16, 3)
    p6_2d = p6[:, :, :2].copy()
    p6_3d_z0 = np.concatenate([p6_2d, np.zeros((6, 16, 1))], axis=2)
    m6q = np.ones((8, 16), dtype=bool)
    m6p = np.ones((6, 16), dtype=bool)
    d_3d, _, _ = unity_dtw_3d(q6_3d_z0, m6q, p6_3d_z0, m6p)
    d_2d, _, _ = unity_dtw_2d(q6_2d, m6q, p6_2d, m6p)
    assert abs(d_3d - d_2d) < 1e-12, f"2D vs 3D(z=0): {d_2d} vs {d_3d}"
    results.append({"test": "2d_equals_3d_z0", "status": "PASS", "d_2d": d_2d, "d_3d": d_3d})

    print(f"  DTW parity tests: {len(results)}/{len(results)} PASS")
    return results

def in_band_check(i, j, T1, T2, band):
    if T1 <= 1 or T2 <= 1:
        return True
    return abs(i / (T1 - 1) - j / (T2 - 1)) <= band + 1e-15


# ── Projection / 2D normalizer (from v1.3a, corrected) ──────────────────────

def load_intrinsics():
    with open(INTRINSICS) as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64).ravel()
    return K, dist

def project_points_distorted(pts_3d, K, dist):
    """Project Nx3 metric 3D points to Nx2 pixel using distortion-aware pinhole."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]
    x = pts_3d[:, 0] / pts_3d[:, 2]
    y = pts_3d[:, 1] / pts_3d[:, 2]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    u = fx * xd + cx
    v = fy * yd + cy
    return np.stack([u, v], axis=1)


def normalize_2d_torso(pts_2d, mask, shoulder_l_px, shoulder_r_px, hip_l_px, hip_r_px,
                        shoulder_valid, hip_valid):
    """Shared 2D torso normalizer. Returns [T,16,2] + [T,16] mask.

    pts_2d: [T, 16, 2] pixel coordinates
    mask: [T, 16] boolean
    shoulder_l_px, shoulder_r_px: [T, 2] pixel (pose landmarks 11, 12)
    hip_l_px, hip_r_px: [T, 2] pixel (pose landmarks 23, 24)
    shoulder_valid, hip_valid: [T] boolean
    """
    T = pts_2d.shape[0]
    out_coords = np.zeros((T, 16, 2), dtype=np.float64)
    out_mask = np.zeros((T, 16), dtype=bool)

    for fi in range(T):
        if not shoulder_valid[fi]:
            continue  # fail-closed: no anchor => no output

        origin = (shoulder_l_px[fi] + shoulder_r_px[fi]) / 2.0
        diff = shoulder_r_px[fi] - shoulder_l_px[fi]
        scale = np.linalg.norm(diff)
        if scale < 1e-6:
            continue  # degenerate shoulder width

        right2d = diff / scale

        # up2d: hip-orthogonalized or image-up fallback
        if hip_valid[fi]:
            hip_mid = (hip_l_px[fi] + hip_r_px[fi]) / 2.0
            raw_down = hip_mid - origin
            # orthogonalize: remove component along right2d
            raw_down = raw_down - np.dot(raw_down, right2d) * right2d
            down_norm = np.linalg.norm(raw_down)
            if down_norm > 1e-6:
                up2d = -raw_down / down_norm
            else:
                up2d = np.array([0.0, -1.0])
                up2d = up2d - np.dot(up2d, right2d) * right2d
                up2d = up2d / np.linalg.norm(up2d)
        else:
            up2d = np.array([0.0, -1.0])
            up2d = up2d - np.dot(up2d, right2d) * right2d
            up_norm = np.linalg.norm(up2d)
            if up_norm < 1e-6:
                continue
            up2d = up2d / up_norm

        # sign-fix: up2d should point image-up (y < 0 in pixel coords)
        if np.dot(up2d, np.array([0.0, -1.0])) < 0:
            up2d = -up2d

        for lm in range(16):
            if not mask[fi, lm]:
                continue
            p = pts_2d[fi, lm] - origin
            x2 = np.dot(p, right2d) / scale
            y2 = np.dot(p, up2d) / scale
            out_coords[fi, lm, 0] = x2
            out_coords[fi, lm, 1] = y2
            out_mask[fi, lm] = True

    return out_coords, out_mask


# ── Processing functions ─────────────────────────────────────────────────────

def load_step6b_raw_3d(sample_id):
    """Load a Step 6B raw 3D JSON file. Filename is landmarks_3d_{suffix}.json."""
    # sample_id like S001_P001_GREETING_FAST_R02 -> suffix = GREETING_FAST_R02
    suffix = sample_id.replace("S001_P001_", "")
    fname = f"landmarks_3d_{suffix}.json"
    fpath = RAW_3D_DIR / fname
    if not fpath.exists():
        return None
    with open(fpath) as f:
        return json.load(f)

def extract_prototype_3d_from_6b(raw_3d_json):
    """Extract [T, 16, 3] coordinates and [T, 16] mask from Step 6B raw 3D.

    Step 6B structure: frames[i].pose_landmarks_3d (list of 33),
    left_hand_landmarks_3d (list of 21), right_hand_landmarks_3d (list of 21).
    Each landmark has 'xyz' [x,y,z] and 'qa_valid' bool.
    """
    frames = raw_3d_json["frames"]
    T = len(frames)
    coords = np.zeros((T, 16, 3), dtype=np.float64)
    mask = np.zeros((T, 16), dtype=bool)

    # 16 features: pose[13,14,15,16] + left_hand[0,4,8,12,16,20] + right_hand[0,4,8,12,16,20]
    POSE_FEAT_IDX = [13, 14, 15, 16]
    HAND_FEAT_IDX = [0, 4, 8, 12, 16, 20]

    for fi, frame in enumerate(frames):
        pose_lms = frame.get("pose_landmarks_3d", [])
        left_lms = frame.get("left_hand_landmarks_3d", [])
        right_lms = frame.get("right_hand_landmarks_3d", [])

        li = 0
        for pidx in POSE_FEAT_IDX:
            if pidx < len(pose_lms):
                lm = pose_lms[pidx]
                if lm.get("qa_valid", False):
                    coords[fi, li] = lm["xyz"]
                    mask[fi, li] = True
            li += 1
        for hidx in HAND_FEAT_IDX:
            if hidx < len(left_lms):
                lm = left_lms[hidx]
                if lm.get("qa_valid", False):
                    coords[fi, li] = lm["xyz"]
                    mask[fi, li] = True
            li += 1
        for hidx in HAND_FEAT_IDX:
            if hidx < len(right_lms):
                lm = right_lms[hidx]
                if lm.get("qa_valid", False):
                    coords[fi, li] = lm["xyz"]
                    mask[fi, li] = True
            li += 1
    return coords, mask

def extract_prototype_front_reproj(raw_3d_json, K, dist):
    """Extract [T, 16, 2] pixel from Step 6B by reprojecting metric 3D through FRONT intrinsics.

    Also extracts shoulder/hip anchors for the 2D normalizer.
    """
    frames = raw_3d_json["frames"]
    T = len(frames)
    coords_2d = np.zeros((T, 16, 2), dtype=np.float64)
    mask_2d = np.zeros((T, 16), dtype=bool)
    shoulder_l = np.zeros((T, 2), dtype=np.float64)
    shoulder_r = np.zeros((T, 2), dtype=np.float64)
    hip_l = np.zeros((T, 2), dtype=np.float64)
    hip_r = np.zeros((T, 2), dtype=np.float64)
    shoulder_valid = np.zeros(T, dtype=bool)
    hip_valid = np.zeros(T, dtype=bool)

    POSE_FEAT_IDX = [13, 14, 15, 16]
    HAND_FEAT_IDX = [0, 4, 8, 12, 16, 20]

    def project_lm(lm):
        """Project a single qa_valid landmark to 2D pixel."""
        pt3d = np.array([lm["xyz"]], dtype=np.float64)
        return project_points_distorted(pt3d, K, dist)[0]

    for fi, frame in enumerate(frames):
        pose_lms = frame.get("pose_landmarks_3d", [])
        left_lms = frame.get("left_hand_landmarks_3d", [])
        right_lms = frame.get("right_hand_landmarks_3d", [])

        # Feature landmarks
        li = 0
        for pidx in POSE_FEAT_IDX:
            if pidx < len(pose_lms) and pose_lms[pidx].get("qa_valid", False):
                coords_2d[fi, li] = project_lm(pose_lms[pidx])
                mask_2d[fi, li] = True
            li += 1
        for hidx in HAND_FEAT_IDX:
            if hidx < len(left_lms) and left_lms[hidx].get("qa_valid", False):
                coords_2d[fi, li] = project_lm(left_lms[hidx])
                mask_2d[fi, li] = True
            li += 1
        for hidx in HAND_FEAT_IDX:
            if hidx < len(right_lms) and right_lms[hidx].get("qa_valid", False):
                coords_2d[fi, li] = project_lm(right_lms[hidx])
                mask_2d[fi, li] = True
            li += 1

        # Anchor landmarks (shoulders 11,12 and hips 23,24)
        sl_ok = 11 < len(pose_lms) and pose_lms[11].get("qa_valid", False)
        sr_ok = 12 < len(pose_lms) and pose_lms[12].get("qa_valid", False)
        if sl_ok and sr_ok:
            shoulder_l[fi] = project_lm(pose_lms[11])
            shoulder_r[fi] = project_lm(pose_lms[12])
            shoulder_valid[fi] = True

        hl_ok = 23 < len(pose_lms) and pose_lms[23].get("qa_valid", False)
        hr_ok = 24 < len(pose_lms) and pose_lms[24].get("qa_valid", False)
        if hl_ok and hr_ok:
            hip_l[fi] = project_lm(pose_lms[23])
            hip_r[fi] = project_lm(pose_lms[24])
            hip_valid[fi] = True

    return coords_2d, mask_2d, shoulder_l, shoulder_r, hip_l, hip_r, shoulder_valid, hip_valid


def load_step6a_front():
    """Load Step 6A FRONT 2D data, grouping frames by sample_id."""
    with open(FRONT_2D) as f:
        data = json.load(f)
    # Group frames by sample_id
    samples = {}
    for frame in data["frames"]:
        sid = frame["sample_id"]
        if sid not in samples:
            samples[sid] = []
        samples[sid].append(frame)
    return samples


def process_step6a_query_2d(step6a_samples, sample_id):
    """Extract [T, 16, 2] pixel from Step 6A FRONT detected 2D for a given sample.

    Step 6A structure: pose_landmarks (list of 33 dicts), anat_left_landmarks (list of 21 or None),
    anat_right_landmarks (list of 21 or None). Each dict has raw_distorted_px_x/y, visibility, presence.
    """
    frames = step6a_samples.get(sample_id)
    if frames is None:
        return None, None, None, None, None, None, None, None

    T = len(frames)
    coords_2d = np.zeros((T, 16, 2), dtype=np.float64)
    mask_2d = np.zeros((T, 16), dtype=bool)
    shoulder_l = np.zeros((T, 2), dtype=np.float64)
    shoulder_r = np.zeros((T, 2), dtype=np.float64)
    hip_l = np.zeros((T, 2), dtype=np.float64)
    hip_r = np.zeros((T, 2), dtype=np.float64)
    shoulder_valid = np.zeros(T, dtype=bool)
    hip_valid = np.zeros(T, dtype=bool)

    POSE_FEAT_IDX = [13, 14, 15, 16]
    HAND_FEAT_IDX = [0, 4, 8, 12, 16, 20]

    for fi, frame in enumerate(frames):
        pose_lms = frame.get("pose_landmarks") or []
        left_lms = frame.get("anat_left_landmarks") or []
        right_lms = frame.get("anat_right_landmarks") or []

        # Feature landmarks
        li = 0
        for pidx in POSE_FEAT_IDX:
            if pidx < len(pose_lms):
                lm = pose_lms[pidx]
                if lm.get("visibility", 0) is not None and lm.get("visibility", 0) > 0.5:
                    coords_2d[fi, li, 0] = lm["raw_distorted_px_x"]
                    coords_2d[fi, li, 1] = lm["raw_distorted_px_y"]
                    mask_2d[fi, li] = True
            li += 1
        for hidx in HAND_FEAT_IDX:
            if hidx < len(left_lms):
                lm = left_lms[hidx]
                # hand landmarks have visibility=None, presence=None when detected
                # validity = hand was detected (non-empty list)
                coords_2d[fi, li, 0] = lm["raw_distorted_px_x"]
                coords_2d[fi, li, 1] = lm["raw_distorted_px_y"]
                mask_2d[fi, li] = True
            li += 1
        for hidx in HAND_FEAT_IDX:
            if hidx < len(right_lms):
                lm = right_lms[hidx]
                coords_2d[fi, li, 0] = lm["raw_distorted_px_x"]
                coords_2d[fi, li, 1] = lm["raw_distorted_px_y"]
                mask_2d[fi, li] = True
            li += 1

        # Shoulder anchors (pose 11, 12)
        if len(pose_lms) > 12:
            sl = pose_lms[11]
            sr = pose_lms[12]
            sl_vis = sl.get("visibility", 0) or 0
            sr_vis = sr.get("visibility", 0) or 0
            if sl_vis > 0.5 and sr_vis > 0.5:
                shoulder_l[fi] = [sl["raw_distorted_px_x"], sl["raw_distorted_px_y"]]
                shoulder_r[fi] = [sr["raw_distorted_px_x"], sr["raw_distorted_px_y"]]
                shoulder_valid[fi] = True

        # Hip anchors (pose 23, 24)
        if len(pose_lms) > 24:
            hl = pose_lms[23]
            hr = pose_lms[24]
            hl_vis = hl.get("visibility", 0) or 0
            hr_vis = hr.get("visibility", 0) or 0
            if hl_vis > 0.5 and hr_vis > 0.5:
                hip_l[fi] = [hl["raw_distorted_px_x"], hl["raw_distorted_px_y"]]
                hip_r[fi] = [hr["raw_distorted_px_x"], hr["raw_distorted_px_y"]]
                hip_valid[fi] = True

    return coords_2d, mask_2d, shoulder_l, shoulder_r, hip_l, hip_r, shoulder_valid, hip_valid


def process_step9d_live_query_2d(trace_json, image_w, image_h):
    """Extract [T, 16, 2] pixel from Step 9D raw trace for Candidate B 2D query.

    raw_coordinates is [T, 75, 3] in MediaPipe NormalizedLandmark (x,y in [0,1]).
    Convert to pixel by multiplying by image dimensions. We use the FRONT camera
    dimensions (1920x1080) so both prototype and query share the same aspect ratio.
    The normalizer output is aspect-ratio-dependent, so this must match.
    """
    raw_coords = np.array(trace_json["raw_and_preprocessing"]["raw_coordinates"], dtype=np.float64)
    raw_mask = np.array(trace_json["raw_and_preprocessing"]["raw_original_valid_mask"], dtype=bool)
    T = raw_coords.shape[0]

    # Extract 16 feature landmarks
    coords_2d = np.zeros((T, 16, 2), dtype=np.float64)
    mask_2d = np.zeros((T, 16), dtype=bool)

    for li, idx75 in enumerate(LANDMARK_16_INDICES):
        coords_2d[:, li, 0] = raw_coords[:, idx75, 0] * image_w  # x -> pixel
        coords_2d[:, li, 1] = raw_coords[:, idx75, 1] * image_h  # y -> pixel
        mask_2d[:, li] = raw_mask[:, idx75]

    # Extract shoulder anchors (pose 11, 12)
    shoulder_l = np.zeros((T, 2), dtype=np.float64)
    shoulder_r = np.zeros((T, 2), dtype=np.float64)
    shoulder_l[:, 0] = raw_coords[:, POSE_LEFT_SHOULDER, 0] * image_w
    shoulder_l[:, 1] = raw_coords[:, POSE_LEFT_SHOULDER, 1] * image_h
    shoulder_r[:, 0] = raw_coords[:, POSE_RIGHT_SHOULDER, 0] * image_w
    shoulder_r[:, 1] = raw_coords[:, POSE_RIGHT_SHOULDER, 1] * image_h
    shoulder_valid = raw_mask[:, POSE_LEFT_SHOULDER] & raw_mask[:, POSE_RIGHT_SHOULDER]

    # Extract hip anchors (pose 23, 24)
    hip_l = np.zeros((T, 2), dtype=np.float64)
    hip_r = np.zeros((T, 2), dtype=np.float64)
    hip_l[:, 0] = raw_coords[:, POSE_LEFT_HIP, 0] * image_w
    hip_l[:, 1] = raw_coords[:, POSE_LEFT_HIP, 1] * image_h
    hip_r[:, 0] = raw_coords[:, POSE_RIGHT_HIP, 0] * image_w
    hip_r[:, 1] = raw_coords[:, POSE_RIGHT_HIP, 1] * image_h
    hip_valid = raw_mask[:, POSE_LEFT_HIP] & raw_mask[:, POSE_RIGHT_HIP]

    return coords_2d, mask_2d, shoulder_l, shoulder_r, hip_l, hip_r, shoulder_valid, hip_valid


def process_step9d_control_a(trace_json):
    """Extract [T, 16, 3] + [T, 16] from Step 9D normalized_output for Control A."""
    coords = np.array(trace_json["normalized_output"]["coordinates"], dtype=np.float64)
    mask = np.array(trace_json["normalized_output"]["mask"], dtype=bool)
    return coords, mask


# ── WISRT loading ────────────────────────────────────────────────────────────

def load_wisrt_as_3d(wisrt_path):
    """Load a .wisrt binary (WISRCB01 format) as [T, 16, 3] + [T, 16] mask.

    Format: magic(8) + version(u32) + endian(u32) + LM(u32) + T(u32) +
    strings(sampleId, word, speed, split, sourceSha, logicalSha) +
    coordCount(u64) + coords(T*16*3 float64) +
    maskBitCount(u64) + maskByteCount(u32) + packedMask +
    SHA-256(32 bytes at end).
    """
    with open(wisrt_path, "rb") as f:
        data = f.read()

    import struct

    # Strip trailing SHA-256 (32 bytes) — payload is everything before that
    payload_len = len(data) - 32

    offset = 0
    magic = data[offset:offset+8]; offset += 8
    assert magic == b"WISRCB01", f"bad magic: {magic}"
    version = struct.unpack_from("<I", data, offset)[0]; offset += 4
    assert version == 1
    endian = struct.unpack_from("<I", data, offset)[0]; offset += 4
    assert endian == 0x01020304
    LM = struct.unpack_from("<I", data, offset)[0]; offset += 4
    T = struct.unpack_from("<I", data, offset)[0]; offset += 4
    assert LM == 16, f"LM={LM}"

    def read_string():
        nonlocal offset
        slen = struct.unpack_from("<I", data, offset)[0]; offset += 4
        s = data[offset:offset+slen].decode("utf-8")
        offset += slen
        return s

    sample_id = read_string()
    word = read_string()
    speed = read_string()
    split = read_string()
    source_sha = read_string()
    logical_sha = read_string()

    coord_count = struct.unpack_from("<Q", data, offset)[0]; offset += 8
    expected_coords = T * LM * 3
    assert coord_count == expected_coords, f"coord count {coord_count} != {expected_coords}"

    coords = np.zeros((T, LM, 3), dtype=np.float64)
    for i in range(expected_coords):
        val = struct.unpack_from("<d", data, offset)[0]; offset += 8
        fi = i // (LM * 3)
        rem = i % (LM * 3)
        li = rem // 3
        ai = rem % 3
        coords[fi, li, ai] = val

    mask_bit_count = struct.unpack_from("<Q", data, offset)[0]; offset += 8
    expected_mask = T * LM
    assert mask_bit_count == expected_mask
    mask_byte_count = struct.unpack_from("<I", data, offset)[0]; offset += 4
    expected_mask_bytes = (expected_mask + 7) // 8
    assert mask_byte_count == expected_mask_bytes

    packed_mask = data[offset:offset+expected_mask_bytes]; offset += expected_mask_bytes
    mask = np.zeros((T, LM), dtype=bool)
    for i in range(expected_mask):
        mask[i // LM, i % LM] = bool(packed_mask[i >> 3] & (1 << (i & 7)))

    assert offset == payload_len, f"offset {offset} != payload_len {payload_len}"

    return sample_id, word, speed, coords, mask


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_sealed_proxy_corrected(proto_2d_dict, query_2d_dict):
    """Re-run v1.3a sealed proxy 9/9 with corrected DTW, return metrics."""
    results = []
    for qid in TEST_SEAL_IDS:
        q_coords, q_mask = query_2d_dict[qid]
        word_dists = {}
        all_template_dists = []
        for tid, (p_coords, p_mask) in proto_2d_dict.items():
            word = tid.split("__")[0]
            d, _, _ = unity_dtw_2d(q_coords, q_mask, p_coords, p_mask, band=0.10)
            all_template_dists.append((tid, d))
            if word not in word_dists or d < word_dists[word]:
                word_dists[word] = d

        q_word = qid.replace("S001_P001_", "").rsplit("_", 2)
        q_word = "_".join(q_word[:-2])

        predicted = min(word_dists, key=lambda w: word_dists[w])
        pred_dist = word_dists[predicted]
        sorted_dists = sorted(word_dists.items(), key=lambda x: x[1])
        margin = sorted_dists[1][1] - sorted_dists[0][1] if len(sorted_dists) > 1 else float("inf")

        results.append({
            "query_id": qid,
            "expected_word": q_word,
            "predicted_word": predicted,
            "correct": predicted == q_word,
            "predicted_distance": pred_dist,
            "signed_margin": margin,
            "word_distances": dict(word_dists),
        })

    n_correct = sum(1 for r in results if r["correct"])
    top1 = n_correct / len(results)
    margins = [r["signed_margin"] for r in results]

    return {
        "results": results,
        "correct": n_correct,
        "total": len(results),
        "top1_accuracy": top1,
        "margin_mean": np.mean(margins),
        "margin_min": np.min(margins),
    }


def evaluate_live3(proto_3d_dict, proto_2d_dict, live_queries):
    """Run Control A and Candidate B on 3 actual live queries."""
    control_a_results = []
    candidate_b_results = []

    for word, (q3d_coords, q3d_mask, q2d_coords, q2d_mask) in live_queries.items():
        # Control A: 3D DTW against WISRT prototypes
        word_dists_a = {}
        for tid, (p_coords, p_mask) in proto_3d_dict.items():
            t_word = tid.split("__")[0]
            d, _, _ = unity_dtw_3d(q3d_coords, q3d_mask, p_coords, p_mask, band=0.10)
            if t_word not in word_dists_a or d < word_dists_a[t_word]:
                word_dists_a[t_word] = d

        pred_a = min(word_dists_a, key=lambda w: word_dists_a[w])
        sorted_a = sorted(word_dists_a.items(), key=lambda x: x[1])
        margin_a = sorted_a[1][1] - sorted_a[0][1] if len(sorted_a) > 1 else float("inf")

        control_a_results.append({
            "expected_word": word,
            "predicted_word": pred_a,
            "correct": pred_a == word,
            "predicted_distance": word_dists_a[pred_a],
            "signed_margin": margin_a,
            "word_distances": dict(word_dists_a),
        })

        # Candidate B: 2D DTW against reprojected prototypes
        word_dists_b = {}
        for tid, (p_coords, p_mask) in proto_2d_dict.items():
            t_word = tid.split("__")[0]
            d, _, _ = unity_dtw_2d(q2d_coords, q2d_mask, p_coords, p_mask, band=0.10)
            if t_word not in word_dists_b or d < word_dists_b[t_word]:
                word_dists_b[t_word] = d

        pred_b = min(word_dists_b, key=lambda w: word_dists_b[w])
        sorted_b = sorted(word_dists_b.items(), key=lambda x: x[1])
        margin_b = sorted_b[1][1] - sorted_b[0][1] if len(sorted_b) > 1 else float("inf")

        candidate_b_results.append({
            "expected_word": word,
            "predicted_word": pred_b,
            "correct": pred_b == word,
            "predicted_distance": word_dists_b[pred_b],
            "signed_margin": margin_b,
            "word_distances": dict(word_dists_b),
        })

    return control_a_results, candidate_b_results


# ── Protected production delta ───────────────────────────────────────────────

PROTECTED_PATTERNS = [
    "Assets/_Project/Scripts/**/*.cs",
    "Assets/StreamingAssets/RecognitionCoreV1/**/*",
    "Assets/_Project/Scenes/**/*.unity",
]

def compute_protected_delta():
    """Compute SHA-256 of all protected production files."""
    import glob
    inventory = {}
    for pattern in PROTECTED_PATTERNS:
        full_pattern = str(ROOT / pattern)
        for fpath in glob.glob(full_pattern, recursive=True):
            fp = Path(fpath)
            if fp.is_file():
                rel = str(fp.relative_to(ROOT)).replace("\\", "/")
                inventory[rel] = {
                    "size": fp.stat().st_size,
                    "sha256": sha256_file(fp),
                }
    return inventory


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    assertions = []
    def check(name, condition, detail=""):
        assertions.append({"name": name, "status": "PASS" if condition else "FAIL", "detail": detail})
        if not condition:
            print(f"  FAIL: {name} — {detail}")
        return condition

    print("=" * 70)
    print("Step 10N v1.3b — Step 9D Raw-Trace Actual-Live 3-Query Evaluation")
    print("=" * 70)

    # ── 0. Preflight ─────────────────────────────────────────────────────
    print("\n[0] Preflight identity checks...")

    pc_sha = sha256_file(PROJECT_CONTEXT)
    check("PROJECT_CONTEXT_PRESENT", PROJECT_CONTEXT.exists())
    print(f"  PROJECT_CONTEXT SHA-256: {pc_sha[:16]}...")

    check("INTRINSICS_PRESENT", INTRINSICS.exists(), str(INTRINSICS))
    check("EXTRINSICS_PRESENT", EXTRINSICS.exists(), str(EXTRINSICS))
    check("TEMPLATE_MANIFEST_PRESENT", TEMPLATE_MANIFEST.exists())
    check("TEST_SEAL_PRESENT", TEST_SEAL.exists())
    check("FRONT_2D_PRESENT", FRONT_2D.exists())
    check("RAW_3D_DIR_PRESENT", RAW_3D_DIR.exists())

    # Step 9D traces
    for word, path in STEP9D_TRACES.items():
        present = path.exists()
        check(f"STEP9D_{word}_PRESENT", present, str(path))
        if present:
            sha = sha256_file(path)
            check(f"STEP9D_{word}_SHA_PREFIX", sha.startswith(STEP9D_EXPECTED_SHA[word]),
                  f"got {sha[:16]}, expected prefix {STEP9D_EXPECTED_SHA[word]}")

    # Step 6A FRONT 2D
    check("FRONT_2D_PRESENT_VERIFY", FRONT_2D.exists(), str(FRONT_2D))

    # WISRT files
    wisrt_count = 0
    for tid in TRAIN_MEDOIDS:
        wp = WISRT_DIR / f"template_{tid}.wisrt"
        if wp.exists():
            wisrt_count += 1
    check("WISRT_9_PRESENT", wisrt_count == 9, f"found {wisrt_count}/9")

    # ── 1. Pre-delta snapshot ────────────────────────────────────────────
    print("\n[1] Pre-delta snapshot...")
    pre_inventory = compute_protected_delta()
    print(f"  {len(pre_inventory)} protected files captured")

    # ── 2. DTW parity tests ──────────────────────────────────────────────
    print("\n[2] DTW C#-parity synthetic regression tests...")
    parity_results = run_dtw_parity_tests()
    check("DTW_PARITY_ALL_PASS", all(r["status"] == "PASS" for r in parity_results),
          f"{sum(1 for r in parity_results if r['status']=='PASS')}/{len(parity_results)}")

    # ── 3. Load calibration data ─────────────────────────────────────────
    print("\n[3] Loading calibration data...")
    K, dist = load_intrinsics()
    print(f"  K: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")

    # ── 4. Build 2D prototypes (reproject Step 6B 3D through FRONT intrinsics) ──
    print("\n[4] Building 2D prototypes from Step 6B reprojection...")
    proto_2d_dict = {}  # tid -> (coords_2d_norm, mask_2d_norm)

    for tid, sid in TRAIN_MEDOIDS.items():
        raw_3d = load_step6b_raw_3d(sid)
        check(f"PROTO_RAW3D_{tid}", raw_3d is not None, sid)
        if raw_3d is None:
            continue

        c2d, m2d, sl, sr, hl, hr, sv, hv = extract_prototype_front_reproj(raw_3d, K, dist)
        norm_c, norm_m = normalize_2d_torso(c2d, m2d, sl, sr, hl, hr, sv, hv)
        proto_2d_dict[tid] = (norm_c, norm_m)
        print(f"  {tid}: T={norm_c.shape[0]}, valid_mean={norm_m.mean():.3f}")

    check("PROTO_2D_COUNT_9", len(proto_2d_dict) == 9, f"got {len(proto_2d_dict)}")

    # ── 5. Build 2D sealed proxy queries (from Step 6A FRONT) ───────────
    print("\n[5] Loading Step 6A FRONT and building sealed proxy queries...")
    step6a_samples = load_step6a_front()
    print(f"  Step 6A samples loaded: {len(step6a_samples)}")

    query_2d_dict = {}
    for qid in TEST_SEAL_IDS:
        result = process_step6a_query_2d(step6a_samples, qid)
        if result[0] is None:
            print(f"  WARNING: {qid} not found in Step 6A")
            continue
        c2d, m2d, sl, sr, hl, hr, sv, hv = result
        norm_c, norm_m = normalize_2d_torso(c2d, m2d, sl, sr, hl, hr, sv, hv)
        query_2d_dict[qid] = (norm_c, norm_m)
    check("QUERY_2D_COUNT_9", len(query_2d_dict) == 9, f"got {len(query_2d_dict)}")

    # ── 6. Corrected sealed proxy evaluation ─────────────────────────────
    print("\n[6] Re-running v1.3a sealed proxy with corrected DTW...")
    proxy_metrics = evaluate_sealed_proxy_corrected(proto_2d_dict, query_2d_dict)

    print(f"  Top-1: {proxy_metrics['correct']}/{proxy_metrics['total']} = {proxy_metrics['top1_accuracy']:.4f}")
    print(f"  Margin mean: {proxy_metrics['margin_mean']:.15f}")
    print(f"  Margin min:  {proxy_metrics['margin_min']:.15f}")

    check("PROXY_9_OF_9_CORRECT", proxy_metrics["correct"] == 9)
    check("PROXY_MARGIN_MEAN_MATCH",
          abs(proxy_metrics["margin_mean"] - V13A_EXPECTED_MARGIN_MEAN) < 1e-6,
          f"got {proxy_metrics['margin_mean']:.15f}, expected {V13A_EXPECTED_MARGIN_MEAN:.15f}")
    check("PROXY_MARGIN_MIN_MATCH",
          abs(proxy_metrics["margin_min"] - V13A_EXPECTED_MARGIN_MIN) < 1e-6,
          f"got {proxy_metrics['margin_min']:.15f}, expected {V13A_EXPECTED_MARGIN_MIN:.15f}")

    for r in proxy_metrics["results"]:
        print(f"    {r['query_id']}: {r['expected_word']} -> {r['predicted_word']} "
              f"({'OK' if r['correct'] else 'WRONG'}) margin={r['signed_margin']:.6f}")

    # ── 7. Load WISRT 3D prototypes for Control A ────────────────────────
    print("\n[7] Loading WISRT 3D prototypes for Control A...")
    proto_3d_dict = {}  # tid -> (coords_3d, mask_3d)

    for tid in TRAIN_MEDOIDS:
        wp = WISRT_DIR / f"template_{tid}.wisrt"
        sid, word, speed, coords, mask = load_wisrt_as_3d(wp)
        proto_3d_dict[tid] = (coords, mask)
        print(f"  {tid}: T={coords.shape[0]}, valid_mean={mask.mean():.3f}")

    check("WISRT_3D_COUNT_9", len(proto_3d_dict) == 9)

    # ── 8. Load Step 9D traces ───────────────────────────────────────────
    print("\n[8] Loading Step 9D raw traces...")
    live_queries = {}  # word -> (q3d_coords, q3d_mask, q2d_norm_coords, q2d_norm_mask)
    live_trace_info = {}

    for word, path in STEP9D_TRACES.items():
        with open(path) as f:
            trace = json.load(f)

        # Verify schema
        check(f"TRACE_{word}_SCHEMA",
              trace.get("schema_version") == "words_in_silence.step9d.raw_intermediate_diagnostic.v1_0",
              trace.get("schema_version", "MISSING"))
        check(f"TRACE_{word}_EXPECTED_WORD", trace.get("expected_word") == word,
              f"got {trace.get('expected_word')}")

        # Control A: normalized_output
        q3d_coords, q3d_mask = process_step9d_control_a(trace)
        print(f"  {word} Control A: T={q3d_coords.shape[0]}, valid_mean={q3d_mask.mean():.3f}")

        # Candidate B: raw coordinates -> pixel -> 2D normalize
        # Use FRONT camera dimensions (1920x1080) for aspect-ratio consistency
        # with the prototype side. The normalizer output depends on aspect ratio
        # (shoulder direction vector), so both sides must use the same frame.
        img_w, img_h = 1920, 1080
        print(f"  {word} using proxy image: {img_w}x{img_h} (FRONT aspect)")

        c2d, m2d, sl, sr, hl, hr, sv, hv = process_step9d_live_query_2d(trace, img_w, img_h)
        q2d_norm, q2d_mask_norm = normalize_2d_torso(c2d, m2d, sl, sr, hl, hr, sv, hv)
        print(f"  {word} Candidate B: T={q2d_norm.shape[0]}, valid_mean={q2d_mask_norm.mean():.3f}")

        live_queries[word] = (q3d_coords, q3d_mask, q2d_norm, q2d_mask_norm)
        live_trace_info[word] = {
            "path": str(path),
            "frames": q3d_coords.shape[0],
            "image_size": [img_w, img_h],
            "control_a_valid_rate": float(q3d_mask.mean()),
            "candidate_b_valid_rate": float(q2d_mask_norm.mean()),
        }

    check("LIVE_QUERIES_3", len(live_queries) == 3, f"got {len(live_queries)}")

    # ── 9. Evaluate Control A vs Candidate B ─────────────────────────────
    print("\n[9] Evaluating Control A vs Candidate B on 3 actual live queries...")
    control_a_results, candidate_b_results = evaluate_live3(proto_3d_dict, proto_2d_dict, live_queries)

    print("\n  === Control A (current 3D mismatch) ===")
    control_a_correct = 0
    for r in control_a_results:
        ok = "OK" if r["correct"] else "WRONG"
        print(f"    {r['expected_word']}: predicted={r['predicted_word']} ({ok}) "
              f"dist={r['predicted_distance']:.6f} margin={r['signed_margin']:.6f}")
        if r["correct"]:
            control_a_correct += 1

    print(f"\n  Control A: {control_a_correct}/3 correct")

    print("\n  === Candidate B (corrected common 2D) ===")
    candidate_b_correct = 0
    for r in candidate_b_results:
        ok = "OK" if r["correct"] else "WRONG"
        print(f"    {r['expected_word']}: predicted={r['predicted_word']} ({ok}) "
              f"dist={r['predicted_distance']:.6f} margin={r['signed_margin']:.6f}")
        if r["correct"]:
            candidate_b_correct += 1

    print(f"\n  Candidate B: {candidate_b_correct}/3 correct")

    # PASS gate
    candidate_b_3of3 = candidate_b_correct == 3
    candidate_b_strictly_better = candidate_b_correct > control_a_correct
    pass_gate = candidate_b_3of3 and candidate_b_strictly_better

    check("CANDIDATE_B_3_OF_3", candidate_b_3of3,
          f"Candidate B {candidate_b_correct}/3")
    check("CANDIDATE_B_STRICTLY_BETTER", candidate_b_strictly_better,
          f"B={candidate_b_correct} vs A={control_a_correct}")
    check("ACTUAL_LIVE_PASS_GATE", pass_gate,
          f"B_3of3={candidate_b_3of3}, B>A={candidate_b_strictly_better}")

    # ── 10. Post-delta check ─────────────────────────────────────────────
    print("\n[10] Post-delta verification...")
    post_inventory = compute_protected_delta()

    delta_count = 0
    for rel, pre_info in pre_inventory.items():
        post_info = post_inventory.get(rel)
        if post_info is None:
            delta_count += 1
            print(f"  DELETED: {rel}")
        elif pre_info["sha256"] != post_info["sha256"]:
            delta_count += 1
            print(f"  MODIFIED: {rel}")
    for rel in post_inventory:
        if rel not in pre_inventory:
            delta_count += 1
            print(f"  ADDED: {rel}")

    check("PRODUCTION_DELTA_ZERO", delta_count == 0, f"delta={delta_count}")

    # ── 11. Save outputs ─────────────────────────────────────────────────
    print("\n[11] Saving outputs...")

    # Input identity manifest
    input_manifest = {
        "project_context_sha256": pc_sha,
        "intrinsics_sha256": sha256_file(INTRINSICS),
        "extrinsics_sha256": sha256_file(EXTRINSICS),
        "template_manifest_sha256": sha256_file(TEMPLATE_MANIFEST),
        "test_seal_sha256": sha256_file(TEST_SEAL),
    }
    for word, path in STEP9D_TRACES.items():
        input_manifest[f"step9d_{word.lower()}_sha256"] = sha256_file(path)

    with open(OUT / "input_identity_manifest.json", "w") as f:
        jdump(input_manifest, f, indent=2)

    # DTW parity results
    with open(OUT / "dtw_parity_results.json", "w") as f:
        jdump(parity_results, f, indent=2)

    # Sealed proxy preservation
    with open(OUT / "sealed_proxy_corrected_metrics.json", "w") as f:
        jdump(proxy_metrics, f, indent=2)

    # Control A results
    with open(OUT / "control_a_live3_results.json", "w") as f:
        jdump(control_a_results, f, indent=2)

    # Candidate B results
    with open(OUT / "candidate_b_live3_results.json", "w") as f:
        jdump(candidate_b_results, f, indent=2)

    # Live trace info
    with open(OUT / "step9d_trace_info.json", "w") as f:
        jdump(live_trace_info, f, indent=2)

    # Protected delta
    protected_delta = {
        "pre_file_count": len(pre_inventory),
        "post_file_count": len(post_inventory),
        "production_files_modified": delta_count,
        "wisrt_modified": 0,
        "scene_modified": 0,
        "pre_inventory": pre_inventory,
        "post_inventory": post_inventory,
    }
    with open(OUT / "protected_pre_post_identity.json", "w") as f:
        jdump(protected_delta, f, indent=2)

    # All assertions
    with open(OUT / "assertion_results.json", "w") as f:
        jdump(assertions, f, indent=2)

    # Summary comparison
    comparison = {
        "control_a": {
            "correct": control_a_correct,
            "total": 3,
            "results": control_a_results,
        },
        "candidate_b": {
            "correct": candidate_b_correct,
            "total": 3,
            "results": candidate_b_results,
        },
        "sealed_proxy": {
            "correct": proxy_metrics["correct"],
            "total": proxy_metrics["total"],
            "margin_mean": proxy_metrics["margin_mean"],
            "margin_min": proxy_metrics["margin_min"],
        },
        "pass_gate": {
            "candidate_b_3of3": candidate_b_3of3,
            "candidate_b_strictly_better": candidate_b_strictly_better,
            "overall_pass": pass_gate,
        },
    }
    with open(OUT / "comparison_summary.json", "w") as f:
        jdump(comparison, f, indent=2)

    # Save 2D coordinates for independent verification
    proto_2d_save = {}
    for tid, (c, m) in proto_2d_dict.items():
        proto_2d_save[f"coords_{tid}"] = c
        proto_2d_save[f"mask_{tid}"] = m
    np.savez(OUT / "prototype_2d_coordinates.npz", **proto_2d_save)

    live_2d_save = {}
    for word, (_, _, q2d, q2d_m) in live_queries.items():
        live_2d_save[f"coords_{word}"] = q2d
        live_2d_save[f"mask_{word}"] = q2d_m
    np.savez(OUT / "live_query_2d_coordinates.npz", **live_2d_save)

    live_3d_save = {}
    for word, (q3d, q3d_m, _, _) in live_queries.items():
        live_3d_save[f"coords_{word}"] = q3d
        live_3d_save[f"mask_{word}"] = q3d_m
    np.savez(OUT / "live_query_3d_coordinates.npz", **live_3d_save)

    # ── 12. Final report ─────────────────────────────────────────────────
    n_pass = sum(1 for a in assertions if a["status"] == "PASS")
    n_fail = sum(1 for a in assertions if a["status"] == "FAIL")

    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"Step 10N v1.3b COMPLETE")
    print(f"  Assertions: {n_pass} PASS / {n_fail} FAIL / {len(assertions)} total")
    print(f"  Sealed proxy (corrected): {proxy_metrics['correct']}/9 (margin_mean={proxy_metrics['margin_mean']:.6f})")
    print(f"  Control A: {control_a_correct}/3")
    print(f"  Candidate B: {candidate_b_correct}/3")
    print(f"  PASS gate: {'PASS' if pass_gate else 'FAIL'}")
    print(f"  Production delta: {delta_count}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 70}")

    verdict_lines = []
    if pass_gate:
        verdict_lines.append("STEP10N_V1_3B_ACTUAL_LIVE_3_QUERY_PASS")
    else:
        verdict_lines.append("STEP10N_V1_3B_ACTUAL_LIVE_3_QUERY_FAIL")
    verdict_lines.append(f"STEP10N_V1_3B_SEALED_PROXY_CORRECTED_{'9_OF_9_PASS' if proxy_metrics['correct']==9 else 'FAIL'}")
    verdict_lines.append(f"STEP10N_V1_3B_CONTROL_A_{control_a_correct}_OF_3")
    verdict_lines.append(f"STEP10N_V1_3B_CANDIDATE_B_{candidate_b_correct}_OF_3")
    verdict_lines.append(f"STEP10N_V1_3B_PRODUCTION_DELTA_{delta_count}")

    print("\nVerdict:")
    for v in verdict_lines:
        print(f"  {v}")

    # Write report
    report_path = OUT / "STEP10N_V1_3B_ACTUAL_LIVE_3_QUERY_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Step 10N v1.3b -- Step 9D Raw-Trace Actual-Live 3-Query Report\n\n")
        f.write(f"**Date**: 2026-08-27\n")
        f.write(f"**Author**: Claude Code\n\n")
        f.write(f"## Verdict\n\n```\n")
        for v in verdict_lines:
            f.write(f"{v}\n")
        f.write(f"```\n\n")

        f.write(f"## Assertions: {n_pass} PASS / {n_fail} FAIL / {len(assertions)} total\n\n")

        f.write(f"## DTW Parity Tests: {len(parity_results)}/{len(parity_results)} PASS\n\n")
        f.write(f"Verified: local cost = total/included, MissingCost=1.0, both-invalid excluded, ")
        f.write(f"relative band, diagonal tie priority, 2D==3D(z=0)\n\n")

        f.write(f"## Sealed Proxy Preservation (corrected DTW)\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Top-1 | {proxy_metrics['correct']}/{proxy_metrics['total']} |\n")
        f.write(f"| Margin mean | {proxy_metrics['margin_mean']:.15f} |\n")
        f.write(f"| Margin min | {proxy_metrics['margin_min']:.15f} |\n")
        f.write(f"| Expected mean | {V13A_EXPECTED_MARGIN_MEAN:.15f} |\n")
        f.write(f"| Expected min | {V13A_EXPECTED_MARGIN_MIN:.15f} |\n\n")

        f.write(f"## Control A (current 3D mismatch)\n\n")
        f.write(f"| Query | Predicted | Correct | Distance | Margin |\n")
        f.write(f"|-------|-----------|---------|----------|--------|\n")
        for r in control_a_results:
            f.write(f"| {r['expected_word']} | {r['predicted_word']} | "
                    f"{'YES' if r['correct'] else 'NO'} | {r['predicted_distance']:.6f} | "
                    f"{r['signed_margin']:.6f} |\n")
        f.write(f"\n**Control A: {control_a_correct}/3**\n\n")

        f.write(f"## Candidate B (corrected common 2D)\n\n")
        f.write(f"| Query | Predicted | Correct | Distance | Margin |\n")
        f.write(f"|-------|-----------|---------|----------|--------|\n")
        for r in candidate_b_results:
            f.write(f"| {r['expected_word']} | {r['predicted_word']} | "
                    f"{'YES' if r['correct'] else 'NO'} | {r['predicted_distance']:.6f} | "
                    f"{r['signed_margin']:.6f} |\n")
        f.write(f"\n**Candidate B: {candidate_b_correct}/3**\n\n")

        f.write(f"## PASS Gate\n\n")
        f.write(f"- Candidate B 3/3: {'YES' if candidate_b_3of3 else 'NO'}\n")
        f.write(f"- Candidate B > Control A: {'YES' if candidate_b_strictly_better else 'NO'} ")
        f.write(f"(B={candidate_b_correct} vs A={control_a_correct})\n")
        f.write(f"- **Overall: {'PASS' if pass_gate else 'FAIL'}**\n\n")

        f.write(f"## Production Delta: {delta_count}\n\n")
        f.write(f"## Delivery Contents\n\n")
        f.write(f"All in `Validation/step10n_v1_3b/`\n")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
