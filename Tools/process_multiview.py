"""
Unified multiview processing pipeline: S1~S5.5
Takes raw landmark JSONs and produces Blender-ready filtered 3D JSON.

Usage:
  python process_multiview.py --sign-id SIGN001_THANKS
  python process_multiview.py --sign-id SIGN003_LOVE
"""

import argparse, json, csv, os, sys, math, copy
import numpy as np

PROJECT = r"C:\MyUnityProjects\WordsInSilence"
DATASET = os.path.join(PROJECT, "Tools", "dataset")
VIEWS = ["front", "left45", "right45"]

POSE_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]
UPPER_BODY_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


# ══════════════════════════════════════════════════════════════
# S1: Sync & Segmentation
# ══════════════════════════════════════════════════════════════

def load_raw_landmarks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    result = []
    for fr in frames:
        fi = fr.get("fi")
        pts = fr.get("source_pts_ms", 0.0)
        pn = fr.get("pose_norm")
        lw_x, lw_y, rw_x, rw_y = None, None, None, None
        lw_wy, rw_wy = None, None
        if pn and len(pn) > 16:
            lw = pn[15]; rw = pn[16]
            if lw and len(lw) >= 2: lw_x, lw_y = lw[0], lw[1]
            if rw and len(rw) >= 2: rw_x, rw_y = rw[0], rw[1]
        pw = fr.get("pose_world")
        if pw and len(pw) > 16:
            lww = pw[15]; rww = pw[16]
            if lww and len(lww) >= 3: lw_wy = lww[1]
            if rww and len(rww) >= 3: rw_wy = rww[1]
        result.append({
            "fi": fi, "pts_ms": pts,
            "lw_x": lw_x, "lw_y": lw_y, "rw_x": rw_x, "rw_y": rw_y,
            "lw_wy": lw_wy, "rw_wy": rw_wy,
        })
    return result, data


def find_clap_start(landmarks_minimal):
    """Find first clap (wrist distance minimum) in first 30% of video."""
    from scipy.signal import find_peaks
    fis, dists = [], []
    for fr in landmarks_minimal:
        if fr["lw_x"] is not None and fr["rw_x"] is not None:
            d = math.sqrt((fr["lw_x"] - fr["rw_x"])**2 + (fr["lw_y"] - fr["rw_y"])**2)
        else:
            d = float("nan")
        fis.append(fr["fi"])
        dists.append(d)
    fis = np.array(fis); dists = np.array(dists)
    n = len(fis)
    valid = ~np.isnan(dists)
    if valid.sum() < 10: return None
    dists_clean = np.interp(np.arange(n), np.where(valid)[0], dists[valid])
    kernel = 7
    dists_smooth = np.convolve(dists_clean, np.ones(kernel)/kernel, mode="same")
    cutoff = int(n * 0.30)
    inv = -dists_smooth[:cutoff]
    peaks, props = find_peaks(inv, prominence=0.01, distance=15)
    if len(peaks) == 0: return None
    best = peaks[np.argmax(props["prominences"])]
    return {"fi": int(fis[best]), "pts_ms": float(landmarks_minimal[best]["pts_ms"])}


def find_end_clap(landmarks_minimal):
    """Find last clap in last 30% of video."""
    from scipy.signal import find_peaks
    fis, dists = [], []
    for fr in landmarks_minimal:
        if fr["lw_x"] is not None and fr["rw_x"] is not None:
            d = math.sqrt((fr["lw_x"] - fr["rw_x"])**2 + (fr["lw_y"] - fr["rw_y"])**2)
        else:
            d = float("nan")
        fis.append(fr["fi"])
        dists.append(d)
    fis = np.array(fis); dists = np.array(dists)
    n = len(fis)
    valid = ~np.isnan(dists)
    if valid.sum() < 10: return None
    dists_clean = np.interp(np.arange(n), np.where(valid)[0], dists[valid])
    dists_smooth = np.convolve(dists_clean, np.ones(7)/7, mode="same")
    back_start = int(n * 0.70)
    inv = -dists_smooth[back_start:]
    peaks, props = find_peaks(inv, prominence=0.01, distance=15)
    if len(peaks) == 0: return None
    best = peaks[np.argmax(props["prominences"])]
    actual_idx = back_start + best
    return {"fi": int(fis[actual_idx]), "pts_ms": float(landmarks_minimal[actual_idx]["pts_ms"])}


def find_repetition_boundaries(landmarks_minimal, start_fi, end_fi):
    """Find repetition boundaries using wrist height peaks across full range."""
    from scipy.signal import find_peaks
    # Extract wrist world Y for ALL frames (not just start_fi~end_fi)
    fis, heights = [], []
    for fr in landmarks_minimal:
        rw_wy = fr.get("rw_wy")
        lw_wy = fr.get("lw_wy")
        if rw_wy is not None and lw_wy is not None:
            h = min(rw_wy, lw_wy)  # lowest wrist Y (most negative = highest)
        elif rw_wy is not None:
            h = rw_wy
        elif lw_wy is not None:
            h = lw_wy
        else:
            continue
        fis.append(fr["fi"])
        heights.append(h)

    if len(fis) < 30:
        return []

    fis = np.array(fis)
    heights = np.array(heights)
    # Heavy smooth to find gesture envelope
    kernel = 30
    heights_smooth = np.convolve(heights, np.ones(kernel)/kernel, mode="same")
    # Find peaks (wrist going UP = more negative Y in MediaPipe world)
    inv = -heights_smooth
    # Use adaptive prominence: median of inv range * 0.15
    inv_range = inv.max() - inv.min()
    min_prom = max(inv_range * 0.15, 0.005)
    peaks, props = find_peaks(inv, prominence=min_prom, distance=50)

    if len(peaks) == 0:
        return []

    # Skip first and last peaks (likely setup/cleanup, not gesture)
    if len(peaks) > 4:
        peaks = peaks[1:-1]

    # Group peaks into repetitions (each rep = one raise)
    reps = []
    for i, pk in enumerate(peaks):
        # Find valley before and after this peak
        if i == 0:
            search_start = max(0, pk - 100)
        else:
            search_start = peaks[i-1]
        segment_before = heights_smooth[search_start:pk]
        if len(segment_before) > 0:
            valley_start = search_start + np.argmax(segment_before)
        else:
            valley_start = max(0, pk - 50)

        if i == len(peaks) - 1:
            search_end = min(len(heights_smooth), pk + 100)
        else:
            search_end = peaks[i+1]
        segment_after = heights_smooth[pk:search_end]
        if len(segment_after) > 0:
            valley_end = pk + np.argmax(segment_after)
        else:
            valley_end = min(len(heights_smooth) - 1, pk + 50)

        reps.append({
            "rep_idx": i + 1,
            "front_start_fi": int(fis[valley_start]),
            "front_end_fi": int(fis[valley_end]),
            "peak_fi": int(fis[pk]),
            "n_frames": int(fis[valley_end]) - int(fis[valley_start]) + 1,
        })

    return reps


# ══════════════════════════════════════════════════════════════
# S2: 2D Landmark Enrichment
# ══════════════════════════════════════════════════════════════

def enrich_landmarks_2d(raw_data, resolution):
    """Convert raw landmarks to enriched 2D format with anatomical hand identity."""
    frames = raw_data["frames"]
    w_px, h_px = resolution
    enriched = []

    for fr in frames:
        entry = {"fi": fr["fi"], "source_pts_ms": fr.get("source_pts_ms", 0)}
        pn = fr.get("pose_norm")
        if pn and len(pn) == 33:
            pose_2d = []
            for j, lm in enumerate(pn):
                if lm and len(lm) >= 4:
                    pose_2d.append({
                        "norm_x": lm[0], "norm_y": lm[1], "norm_z": lm[2],
                        "visibility": lm[3],
                        "pixel_x": lm[0] * w_px, "pixel_y": lm[1] * h_px,
                    })
                else:
                    pose_2d.append(None)
            entry["pose"] = pose_2d
        else:
            entry["pose"] = None

        hands = fr.get("hands", [])
        enriched_hands = {"left": None, "right": None}
        if hands:
            pose_lw_xy, pose_rw_xy = None, None
            if pn and len(pn) > 16:
                if pn[15] and len(pn[15]) >= 2: pose_lw_xy = np.array([pn[15][0], pn[15][1]])
                if pn[16] and len(pn[16]) >= 2: pose_rw_xy = np.array([pn[16][0], pn[16][1]])
            for hand in hands:
                hw = hand["norm"][0]
                hand_xy = np.array([hw[0], hw[1]])
                if pose_lw_xy is not None and pose_rw_xy is not None:
                    dl = np.linalg.norm(hand_xy - pose_lw_xy)
                    dr = np.linalg.norm(hand_xy - pose_rw_xy)
                    anat_label = "left" if dl < dr else "right"
                else:
                    anat_label = hand["handedness"].lower()
                hand_2d = [{"norm_x": lm[0], "norm_y": lm[1], "norm_z": lm[2],
                            "pixel_x": lm[0]*w_px, "pixel_y": lm[1]*h_px} for lm in hand["norm"]]
                enriched_hands[anat_label] = {
                    "mp_handedness": hand["handedness"], "mp_score": hand["score"],
                    "anatomical_side": anat_label, "landmarks": hand_2d,
                }
        entry["hands"] = enriched_hands
        enriched.append(entry)
    return enriched


# ══════════════════════════════════════════════════════════════
# S4: 2.5D Triangulation
# ══════════════════════════════════════════════════════════════

def reconstruct_3d(s2_data, frame_pts, offsets, reps):
    """Reconstruct 3D joints using 2.5D weak-perspective for all active reps."""

    def find_nearest_fi(view, front_pts_ms):
        target = front_pts_ms + offsets[view]
        best_fi, best_diff = 0, float("inf")
        for fi, pts in frame_pts[view]:
            d = abs(pts - target)
            if d < best_diff: best_diff = d; best_fi = fi
        return best_fi

    def get_pose_norm(view, fi, j_idx):
        fr = s2_data[view].get(fi)
        if not fr or not fr.get("pose"): return None, 0
        pose = fr["pose"]
        if j_idx >= len(pose) or pose[j_idx] is None: return None, 0
        pt = pose[j_idx]
        vis = pt.get("visibility", 0)
        if vis < 0.3: return None, vis
        return np.array([pt["norm_x"], pt["norm_y"]]), vis

    def get_hand_norm(view, fi, side, lm_idx):
        fr = s2_data[view].get(fi)
        if not fr or not fr.get("hands"): return None
        hands = fr["hands"]
        if isinstance(hands, dict):
            hand = hands.get(side)
        else: return None
        if not hand or not hand.get("landmarks"): return None
        lms = hand["landmarks"]
        if lm_idx >= len(lms): return None
        lm = lms[lm_idx]
        return np.array([lm["norm_x"], lm["norm_y"]])

    def compute_body_frame(view, fi):
        ls, _ = get_pose_norm(view, fi, 11)
        rs, _ = get_pose_norm(view, fi, 12)
        if ls is None or rs is None: return None, None
        mid = (ls + rs) / 2
        sw = np.linalg.norm(rs - ls)
        return mid, sw

    def reconstruct_joint(front_fi, left_fi, right_fi, get_2d_func):
        f_mid, f_sw = compute_body_frame("front", front_fi)
        if f_mid is None or f_sw < 0.01: return None
        l_mid, l_sw = compute_body_frame("left45", left_fi)
        r_mid, r_sw = compute_body_frame("right45", right_fi)
        f_pt, f_vis = get_2d_func("front", front_fi)
        l_pt, l_vis = get_2d_func("left45", left_fi)
        r_pt, r_vis = get_2d_func("right45", right_fi)
        if f_pt is None: return None
        body_x = (f_pt[0] - f_mid[0]) / f_sw
        body_z = -(f_pt[1] - f_mid[1]) / f_sw
        used_views = ["front"]; depth_vals = []; confidences = [f_vis]
        if l_pt is not None and l_mid is not None and l_sw > 0.01:
            l_depth = -(l_pt[0] - l_mid[0]) / l_sw / math.sin(math.radians(45))
            depth_vals.append(l_depth); used_views.append("left45"); confidences.append(l_vis)
        if r_pt is not None and r_mid is not None and r_sw > 0.01:
            r_depth = (r_pt[0] - r_mid[0]) / r_sw / math.sin(math.radians(45))
            depth_vals.append(r_depth); used_views.append("right45"); confidences.append(r_vis)
        body_y = float(np.mean(depth_vals)) if depth_vals else 0.0
        return {
            "x": round(body_x, 5), "y": round(body_y, 5), "z": round(body_z, 5),
            "used_views": used_views, "confidence": round(float(np.mean(confidences)), 4),
            "n_depth_views": len(depth_vals),
        }

    frames_3d = []
    for r in reps:
        rep_idx = r["rep_idx"]
        front_start = r["front_start_fi"]
        front_end = r["front_end_fi"]
        for front_fi in range(front_start, front_end + 1):
            front_pts_ms = None
            for fi, pts in frame_pts["front"]:
                if fi == front_fi: front_pts_ms = pts; break
            if front_pts_ms is None: continue
            left_fi = find_nearest_fi("left45", front_pts_ms)
            right_fi = find_nearest_fi("right45", front_pts_ms)
            fj = {"front_fi": front_fi, "left45_fi": left_fi, "right45_fi": right_fi,
                  "rep_idx": rep_idx, "front_pts_ms": front_pts_ms,
                  "pose": {}, "left_hand": {}, "right_hand": {}}
            for j_idx in UPPER_BODY_INDICES:
                j_name = POSE_NAMES[j_idx]
                def gf(v, fi, idx=j_idx): return get_pose_norm(v, fi, idx)
                fj["pose"][j_name] = reconstruct_joint(front_fi, left_fi, right_fi, gf)
            for side in ["left", "right"]:
                for lm_idx in range(21):
                    def gf(v, fi, s=side, li=lm_idx):
                        pt = get_hand_norm(v, fi, s, li)
                        return (pt, 0.8) if pt is not None else (None, 0)
                    key = f"{side}_hand"
                    fj[key][f"lm{lm_idx}"] = reconstruct_joint(front_fi, left_fi, right_fi, gf)
            frames_3d.append(fj)
    return frames_3d


# ══════════════════════════════════════════════════════════════
# S5: Canonical Rep Selection
# ══════════════════════════════════════════════════════════════

def select_canonical_rep(frames_3d, reps):
    rep_frames = {}
    for fr in frames_3d:
        ri = fr["rep_idx"]
        if ri not in rep_frames: rep_frames[ri] = []
        rep_frames[ri].append(fr)

    def bone_len(fr, a, b):
        ja = fr["pose"].get(a); jb = fr["pose"].get(b)
        if ja and jb: return math.sqrt((ja["x"]-jb["x"])**2+(ja["y"]-jb["y"])**2+(ja["z"]-jb["z"])**2)
        return None

    scores = []
    for ri, frames in sorted(rep_frames.items()):
        n = len(frames)
        full_pose = sum(1 for fr in frames if all(fr["pose"].get(j) for j in
                        ["left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist"]))
        pose_cov = full_pose / n * 100
        both_hands = sum(1 for fr in frames
                         if any(fr["left_hand"].get(f"lm{i}") for i in range(21))
                         and any(fr["right_hand"].get(f"lm{i}") for i in range(21)))
        hand_cov = both_hands / n * 100
        depth_2 = sum(1 for fr in frames
                      if fr["pose"].get("left_wrist") and fr["pose"]["left_wrist"]["n_depth_views"] >= 2)
        depth_cov = depth_2 / n * 100
        bone_cvs = []
        for a, b in [("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
                     ("right_shoulder","right_elbow"),("right_elbow","right_wrist")]:
            lengths = [bl for bl in (bone_len(fr, a, b) for fr in frames) if bl is not None]
            if len(lengths) > 2:
                arr = np.array(lengths)
                bone_cvs.append(arr.std()/arr.mean()*100 if arr.mean()>0 else 999)
        avg_cv = np.mean(bone_cvs) if bone_cvs else 999
        teleports = 0
        for wrist in ["left_wrist","right_wrist"]:
            prev = None
            for fr in frames:
                w = fr["pose"].get(wrist)
                if w and prev:
                    d = math.sqrt((w["x"]-prev["x"])**2+(w["y"]-prev["y"])**2+(w["z"]-prev["z"])**2)
                    if d > 0.5: teleports += 1
                prev = w if w else prev
        confs = [fr["pose"][j]["confidence"] for fr in frames for j in fr["pose"] if fr["pose"][j]]
        mean_conf = np.mean(confs) if confs else 0
        score = (pose_cov*0.2 + hand_cov*0.2 + depth_cov*0.15 +
                 (100-min(avg_cv,100))*0.15 + (100-teleports*20)*0.1 + mean_conf*100*0.1 + 10*0.1)
        scores.append({"rep_idx": ri, "n_frames": n, "score": round(score, 1),
                        "pose_cov": round(pose_cov,1), "hand_cov": round(hand_cov,1)})
        print(f"  rep{ri:02d}: frames={n}, pose={pose_cov:.0f}%, hands={hand_cov:.0f}%, score={score:.1f}")

    scores.sort(key=lambda x: x["score"], reverse=True)
    best = scores[0]
    print(f"\n  Selected: rep{best['rep_idx']:02d} (score={best['score']})")
    return best["rep_idx"], rep_frames[best["rep_idx"]], scores


# ══════════════════════════════════════════════════════════════
# S5.5: Filtering
# ══════════════════════════════════════════════════════════════

def extract_series(frames, joint_path):
    xs, ys, zs, valid = [], [], [], []
    for fr in frames:
        category, name = joint_path
        j = fr.get(category, {}).get(name)
        if j:
            xs.append(j["x"]); ys.append(j["y"]); zs.append(j["z"]); valid.append(True)
        else:
            xs.append(0.0); ys.append(0.0); zs.append(0.0); valid.append(False)
    return np.array(xs), np.array(ys), np.array(zs), np.array(valid)

def apply_series(frames, joint_path, xs, ys, zs, valid):
    for i, fr in enumerate(frames):
        category, name = joint_path
        if not valid[i]: continue
        if category not in fr: continue
        j = fr[category].get(name)
        if j:
            j["x"] = round(float(xs[i]), 5)
            j["y"] = round(float(ys[i]), 5)
            j["z"] = round(float(zs[i]), 5)

def one_euro_1d(signal, valid, min_cutoff=1.5, beta=0.005, d_cutoff=1.0, fps=30):
    out = signal.copy()
    dt = 1.0 / fps
    def sf(te, c): r = 2*math.pi*c*te; return r/(r+1)
    x_prev = None; dx_prev = 0.0
    for i in range(len(signal)):
        if not valid[i]: continue
        if x_prev is None: x_prev = signal[i]; out[i] = signal[i]; continue
        dx = (signal[i]-x_prev)/dt
        a_d = sf(dt, d_cutoff); dx_hat = a_d*dx + (1-a_d)*dx_prev
        cutoff = min_cutoff + beta*abs(dx_hat)
        a = sf(dt, cutoff); x_hat = a*signal[i] + (1-a)*x_prev
        out[i] = x_hat; x_prev = x_hat; dx_prev = dx_hat
    return out

def filter_frames(canonical_frames):
    """Apply One Euro filter + bone length constraint."""
    frames = copy.deepcopy(canonical_frames)
    joint_paths = []
    for name in ["nose","left_shoulder","right_shoulder","left_elbow","right_elbow",
                 "left_wrist","right_wrist","left_hip","right_hip",
                 "left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb"]:
        joint_paths.append(("pose", name))
    for side in ["left","right"]:
        for i in range(21): joint_paths.append((f"{side}_hand", f"lm{i}"))

    for jp in joint_paths:
        xs, ys, zs, valid = extract_series(frames, jp)
        if valid.sum() < 3: continue
        xs_f = one_euro_1d(xs, valid)
        ys_f = one_euro_1d(ys, valid)
        zs_f = one_euro_1d(zs, valid)
        apply_series(frames, jp, xs_f, ys_f, zs_f, valid)

    # Bone length constraint
    BONES = [("pose","left_shoulder","left_elbow"),("pose","left_elbow","left_wrist"),
             ("pose","right_shoulder","right_elbow"),("pose","right_elbow","right_wrist")]
    for cat, a, b in BONES:
        lengths = []
        for fr in frames:
            ja = fr.get(cat,{}).get(a); jb = fr.get(cat,{}).get(b)
            if ja and jb:
                l = math.sqrt((ja["x"]-jb["x"])**2+(ja["y"]-jb["y"])**2+(ja["z"]-jb["z"])**2)
                if l > 0.001: lengths.append(l)
        if not lengths: continue
        med = float(np.median(lengths))
        for fr in frames:
            ja = fr.get(cat,{}).get(a); jb = fr.get(cat,{}).get(b)
            if ja and jb:
                dx,dy,dz = jb["x"]-ja["x"], jb["y"]-ja["y"], jb["z"]-ja["z"]
                cl = math.sqrt(dx*dx+dy*dy+dz*dz)
                if cl > 0.001:
                    s = 1.0 + (med/cl - 1.0)*0.7
                    jb["x"] = round(ja["x"]+dx*s, 5)
                    jb["y"] = round(ja["y"]+dy*s, 5)
                    jb["z"] = round(ja["z"]+dz*s, 5)
    return frames


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign-id", required=True)
    args = parser.parse_args()
    sign_id = args.sign_id

    pipeline_dir = os.path.join(PROJECT, "_multiview_rebuild_v3", f"pipeline_{sign_id}")
    out_dir = pipeline_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"Multiview Processing Pipeline: {sign_id}")
    print("=" * 60)

    # Check landmark files exist
    for v in VIEWS:
        lm_path = os.path.join(pipeline_dir, f"landmarks_raw_{v}.json")
        if not os.path.exists(lm_path):
            print(f"ERROR: {lm_path} not found! Run extract_multiview_landmarks.py first.")
            sys.exit(1)

    # ── S1: Load & Sync ──
    print("\n=== S1: Sync & Segmentation ===")
    landmarks = {}
    raw_data = {}
    resolutions = {}
    for v in VIEWS:
        lm_path = os.path.join(pipeline_dir, f"landmarks_raw_{v}.json")
        lm_min, lm_raw = load_raw_landmarks(lm_path)
        landmarks[v] = lm_min
        raw_data[v] = lm_raw
        res = lm_raw.get("resolution", [1920, 1080])
        resolutions[v] = tuple(res)
        print(f"  {v}: {len(lm_min)} frames, resolution={res}")

    # Find clap sync
    print("\n  Finding clap sync...")
    claps = {}
    for v in VIEWS:
        clap = find_clap_start(landmarks[v])
        claps[v] = clap
        if clap:
            print(f"  {v}: start clap fi={clap['fi']}, pts={clap['pts_ms']:.1f}ms")
        else:
            print(f"  {v}: no clap found!")

    if not all(claps.values()):
        print("ERROR: Could not find clap in all views!")
        sys.exit(1)

    # Compute offsets (front = reference)
    ref_pts = claps["front"]["pts_ms"]
    offsets = {}
    for v in VIEWS:
        offsets[v] = claps[v]["pts_ms"] - ref_pts
    print(f"\n  Sync offsets: {offsets}")

    # Find end claps
    end_claps = {}
    for v in VIEWS:
        ec = find_end_clap(landmarks[v])
        end_claps[v] = ec
        if ec:
            print(f"  {v}: end clap fi={ec['fi']}")

    # Find repetition boundaries using front view (full range, not clap-limited)
    front_start = 0
    front_end = len(landmarks["front"]) - 1
    print(f"\n  Full range: fi {front_start} ~ {front_end}")

    reps = find_repetition_boundaries(landmarks["front"], front_start, front_end)
    print(f"  Found {len(reps)} repetitions")

    # Add side view frame indices
    frame_pts = {}
    for v in VIEWS:
        frame_pts[v] = [(fr["fi"], fr["pts_ms"]) for fr in landmarks[v]]

    def find_nearest_fi_for_view(view, front_pts_ms):
        target = front_pts_ms + offsets[view]
        best_fi, best_diff = 0, float("inf")
        for fi, pts in frame_pts[view]:
            d = abs(pts - target)
            if d < best_diff: best_diff = d; best_fi = fi
        return best_fi

    for r in reps:
        start_pts = None
        end_pts = None
        for fi, pts in frame_pts["front"]:
            if fi == r["front_start_fi"]: start_pts = pts
            if fi == r["front_end_fi"]: end_pts = pts
        if start_pts and end_pts:
            for sv in ["left45", "right45"]:
                r[f"{sv}_start_fi"] = find_nearest_fi_for_view(sv, start_pts)
                r[f"{sv}_end_fi"] = find_nearest_fi_for_view(sv, end_pts)
            r["duration_ms"] = end_pts - start_pts

    for r in reps:
        print(f"  rep{r['rep_idx']:02d}: front fi {r['front_start_fi']}~{r['front_end_fi']} ({r['n_frames']} frames)")

    # Save rep boundaries
    if reps:
        with open(os.path.join(out_dir, "repetition_boundaries.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=reps[0].keys())
            w.writeheader()
            w.writerows(reps)

    # Save frame_pts
    for v in VIEWS:
        with open(os.path.join(out_dir, f"frame_pts_{v}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "pts_ms", "pts_s"])
            for fr in landmarks[v]:
                w.writerow([fr["fi"], f'{fr["pts_ms"]:.3f}', f'{fr["pts_ms"]/1000:.6f}'])

    # Save sync mapping
    with open(os.path.join(out_dir, "sync_mapping.json"), "w") as f:
        json.dump({"offsets": offsets, "start_claps": {v: c for v, c in claps.items()},
                   "end_claps": {v: c for v, c in end_claps.items() if c}}, f, indent=2)

    # ── S2: 2D Landmark Enrichment ──
    print("\n=== S2: 2D Landmark Enrichment ===")
    s2_data = {}
    for v in VIEWS:
        enriched = enrich_landmarks_2d(raw_data[v], resolutions[v])
        s2_data[v] = {fr["fi"]: fr for fr in enriched}
        print(f"  {v}: {len(enriched)} frames enriched")

    # ── S4: 2.5D Triangulation ──
    print("\n=== S4: 2.5D Triangulation ===")
    frames_3d = reconstruct_3d(s2_data, frame_pts, offsets, reps)
    print(f"  Total frames reconstructed: {len(frames_3d)}")

    # ── S5: Canonical Rep Selection ──
    print("\n=== S5: Canonical Rep Selection ===")
    best_rep_idx, canonical_frames, all_scores = select_canonical_rep(frames_3d, reps)
    print(f"  Canonical: rep{best_rep_idx:02d}, {len(canonical_frames)} frames")

    # ── S5.5: Filtering ──
    print("\n=== S5.5: Filtering ===")
    filtered_frames = filter_frames(canonical_frames)
    print(f"  Filtered {len(filtered_frames)} frames")

    # ── Save output ──
    print("\n=== Saving ===")
    filtered_path = os.path.join(out_dir, f"{sign_id}_canonical_3d_filtered.json")
    coord_sys = {"+X": "right", "+Y": "forward", "+Z": "up",
                 "origin": "shoulder_mid", "scale": "shoulder_width=1.0"}
    with open(filtered_path, "w") as f:
        json.dump({
            "sign_id": sign_id, "rep_idx": best_rep_idx,
            "method": "2.5D_weak_perspective", "coordinate_system": coord_sys,
            "n_frames": len(filtered_frames),
            "score": all_scores[0]["score"],
            "filter": "one_euro", "bone_constraint": "median_blend_0.7",
            "frames": filtered_frames,
        }, f)
    size_mb = os.path.getsize(filtered_path) / 1024 / 1024
    print(f"  Saved: {filtered_path} ({size_mb:.1f}MB)")

    print("\n" + "=" * 60)
    print(f"Pipeline complete for {sign_id}")
    print(f"  Filtered JSON: {filtered_path}")
    print(f"  Next: Run Blender retargeting scripts")
    print("=" * 60)


if __name__ == "__main__":
    main()
