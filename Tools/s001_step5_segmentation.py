"""
Step 5: Action Segmentation for S001_P001

Detects 9 blocks x 5 reps = 45 action segments using wrist motion energy.
Uses FRONT camera PTS (VFR-safe), maps to LEFT/RIGHT via sync offsets + drift.
Does NOT cut/reencode originals. Outputs time ranges only.
"""
import cv2
import numpy as np
import json
import csv
from pathlib import Path

try:
    import mediapipe as mp
    HAS_MP = True
except ImportError:
    HAS_MP = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sessions" / "S001_P001"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "sessions" / "S001_P001"

FRONT_VIDEO = RAW_DIR / "S001_P001_FRONT.mp4"

# ── Block order (fixed, from user specification) ──
BLOCKS = [
    {"block_idx": 0, "sign_id": "GREETING",  "sign_kr": "인사",     "speed": "NORMAL", "n_reps": 5},
    {"block_idx": 1, "sign_id": "GREETING",  "sign_kr": "인사",     "speed": "SLOW",   "n_reps": 5},
    {"block_idx": 2, "sign_id": "GREETING",  "sign_kr": "인사",     "speed": "FAST",   "n_reps": 5},
    {"block_idx": 3, "sign_id": "THANK_YOU", "sign_kr": "감사합니다", "speed": "NORMAL", "n_reps": 5},
    {"block_idx": 4, "sign_id": "THANK_YOU", "sign_kr": "감사합니다", "speed": "SLOW",   "n_reps": 5},
    {"block_idx": 5, "sign_id": "THANK_YOU", "sign_kr": "감사합니다", "speed": "FAST",   "n_reps": 5},
    {"block_idx": 6, "sign_id": "LOVE",      "sign_kr": "사랑",     "speed": "NORMAL", "n_reps": 5},
    {"block_idx": 7, "sign_id": "LOVE",      "sign_kr": "사랑",     "speed": "SLOW",   "n_reps": 5},
    {"block_idx": 8, "sign_id": "LOVE",      "sign_kr": "사랑",     "speed": "FAST",   "n_reps": 5},
]

# Sync
SYNC = {
    "FRONT": {"offset_s": 0.0, "drift_rate": 0.0},
    "LEFT":  {"offset_s": 2.300333, "drift_rate": -19.3e-6},
    "RIGHT": {"offset_s": 9.388633, "drift_rate": -0.3e-6},
}

# Load sync data
sync_path = OUT_DIR / "sync_report.json"
if sync_path.exists():
    sd = json.load(open(sync_path))
    for view in ["LEFT", "RIGHT"]:
        cam = sd["cameras"].get(view, {})
        SYNC[view]["offset_s"] = cam.get("offset_s_final", SYNC[view]["offset_s"])


def front_to_camera_time(front_time_s, view):
    """Convert FRONT PTS to camera-local time using offset + drift."""
    s = SYNC[view]
    return front_time_s + s["offset_s"] + s["drift_rate"] * front_time_s


def extract_wrist_motion_energy(video_path, sample_interval=3):
    """
    Extract wrist motion energy from FRONT video using optical flow or
    simple frame differencing in the wrist region.

    Returns: list of (pts_s, energy) tuples.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Total frames: {total}")

    # Strategy: compute frame difference energy in the central region
    # (where hands/arms are most active for sign language)
    pts_energy = []
    prev_gray = None
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            pts_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Focus on central 60% width, 30%-80% height (torso+arms region)
            y1, y2 = int(h * 0.25), int(h * 0.85)
            x1, x2 = int(w * 0.2), int(w * 0.8)
            roi = gray[y1:y2, x1:x2]

            if prev_gray is not None:
                diff = cv2.absdiff(roi, prev_gray)
                energy = float(np.mean(diff))
                pts_energy.append((pts_s, energy))

            prev_gray = roi.copy()
            processed += 1

        frame_idx += 1

        if processed % 1000 == 0:
            print(f"    Processed {processed} frames (idx={frame_idx})...")

    cap.release()
    print(f"  Extracted {len(pts_energy)} energy samples")
    return pts_energy


def smooth_energy(pts_energy, window_s=1.0):
    """Smooth energy signal with a time-based moving average."""
    times = np.array([t for t, e in pts_energy])
    energies = np.array([e for t, e in pts_energy])

    # Estimate median dt
    dt = np.median(np.diff(times))
    if dt <= 0:
        dt = 0.1
    win = max(3, int(window_s / dt))
    if win % 2 == 0:
        win += 1

    # Pad and convolve
    kernel = np.ones(win) / win
    smoothed = np.convolve(energies, kernel, mode="same")

    return times, energies, smoothed


def find_neutral_zones(times, smoothed, threshold_factor=0.3):
    """
    Find neutral (idle) zones where smoothed energy is below threshold.
    Returns list of (start_s, end_s) for each neutral zone.
    """
    threshold = np.median(smoothed) * threshold_factor + np.percentile(smoothed, 10) * (1 - threshold_factor)

    # Actually use a fixed low percentile
    threshold = np.percentile(smoothed, 25) * 0.8 + np.percentile(smoothed, 5) * 0.2

    is_low = smoothed < threshold
    zones = []
    in_zone = False
    start = 0

    for i in range(len(is_low)):
        if is_low[i] and not in_zone:
            start = i
            in_zone = True
        elif not is_low[i] and in_zone:
            zones.append((times[start], times[i - 1]))
            in_zone = False

    if in_zone:
        zones.append((times[start], times[-1]))

    # Filter: neutral zones should be at least 0.5s long
    zones = [(s, e) for s, e in zones if e - s >= 0.5]
    return zones, threshold


def find_action_bursts(times, smoothed, threshold):
    """
    Find action bursts (periods of sustained high energy).
    Returns list of (start_s, end_s, peak_energy).
    """
    is_high = smoothed > threshold
    bursts = []
    in_burst = False
    start = 0

    for i in range(len(is_high)):
        if is_high[i] and not in_burst:
            start = i
            in_burst = True
        elif not is_high[i] and in_burst:
            peak = float(np.max(smoothed[start:i]))
            bursts.append((times[start], times[i - 1], peak))
            in_burst = False

    if in_burst:
        peak = float(np.max(smoothed[start:]))
        bursts.append((times[start], times[-1], peak))

    return bursts


def segment_actions(times, energies, smoothed):
    """
    Segment 45 actions from 9 blocks using energy analysis.

    Strategy:
    1. Find the C2 region end and clap region start to define the action window
    2. Within the action window, find neutral zones (inter-block gaps are longer)
    3. Find 9 blocks separated by long neutral zones
    4. Within each block, find 5 action bursts separated by shorter neutral zones
    """
    # Load sync for clap times to define action window boundaries
    sync_data = json.load(open(OUT_DIR / "sync_report.json"))
    front_start_claps = sync_data["cameras"]["FRONT"].get("start_clap_times_s", [])
    front_end_claps = sync_data["cameras"]["FRONT"].get("end_clap_times_s", [])

    # C2 region ends around the start claps (start claps mark beginning of action region)
    # Actually, start claps at ~25-26s, C2 region 33-161s in FRONT
    # The action window starts AFTER C2 region and any start signals
    c2_report = json.load(open(OUT_DIR / "C2_CALIBRATION.json"))
    c2_end_s = c2_report["c2_timestamp_ranges"]["FRONT"]["end_pts_ms"] / 1000.0

    action_start = c2_end_s + 5.0  # 5s buffer after last C2 detection
    action_end = front_end_claps[0] - 5.0 if front_end_claps else times[-1] - 10.0

    print(f"\n  Action window: {action_start:.1f}s - {action_end:.1f}s "
          f"({action_end - action_start:.1f}s)")

    # Mask to action window
    mask = (times >= action_start) & (times <= action_end)
    act_times = times[mask]
    act_smooth = smoothed[mask]
    act_energy = energies[mask]

    if len(act_times) < 100:
        print("  ERROR: Too few samples in action window!")
        return []

    # Compute adaptive threshold
    # Use percentile-based approach: action energy is typically above 30th percentile
    p15 = np.percentile(act_smooth, 15)
    p50 = np.percentile(act_smooth, 50)
    action_thresh = p15 + (p50 - p15) * 0.3

    print(f"  Energy stats: p15={p15:.2f}, p50={p50:.2f}, threshold={action_thresh:.2f}")

    # Find all action bursts
    bursts = find_action_bursts(act_times, act_smooth, action_thresh)
    print(f"  Raw action bursts: {len(bursts)}")

    # Merge bursts that are very close together (< 0.3s gap)
    if bursts:
        merged = [list(bursts[0])]
        for s, e, p in bursts[1:]:
            if s - merged[-1][1] < 0.3:
                merged[-1][1] = e
                merged[-1][2] = max(merged[-1][2], p)
            else:
                merged.append([s, e, p])
        bursts = [(s, e, p) for s, e, p in merged]
        print(f"  After merging close bursts: {len(bursts)}")

    # Filter out very short bursts (< 0.3s)
    bursts = [(s, e, p) for s, e, p in bursts if e - s >= 0.3]
    print(f"  After filtering short (<0.3s): {len(bursts)}")

    if len(bursts) < 2:
        print("  ERROR: Too few bursts found!")
        return []

    # ── Hierarchical block detection ──
    # Inter-burst gaps
    gaps = []
    for i in range(len(bursts) - 1):
        gap_dur = bursts[i + 1][0] - bursts[i][1]
        gaps.append((i, gap_dur))

    gap_durations_sorted = sorted([g[1] for g in gaps], reverse=True)
    print(f"  Top 15 inter-burst gaps: {[round(g, 2) for g in gap_durations_sorted[:15]]}")

    # Step 1: Find 2 longest gaps = inter-sign boundaries (GREETING|THANK_YOU|LOVE)
    gaps_by_dur = sorted(gaps, key=lambda x: x[1], reverse=True)
    sign_gap_indices = sorted([gaps_by_dur[0][0], gaps_by_dur[1][0]])
    print(f"  Sign boundaries at burst indices: {sign_gap_indices} "
          f"(gaps: {round(gaps_by_dur[0][1],2)}s, {round(gaps_by_dur[1][1],2)}s)")

    sign_groups = [
        (0, sign_gap_indices[0] + 1),
        (sign_gap_indices[0] + 1, sign_gap_indices[1] + 1),
        (sign_gap_indices[1] + 1, len(bursts)),
    ]
    sign_names = ["GREETING", "THANK_YOU", "LOVE"]

    for si, (gs, ge) in enumerate(sign_groups):
        gb = bursts[gs:ge]
        print(f"  {sign_names[si]}: {ge-gs} bursts, "
              f"t={gb[0][0]:.1f}-{gb[-1][1]:.1f}s ({gb[-1][1]-gb[0][0]:.1f}s)")

    # Step 2: Within each sign group, find 2 gaps to split into NORMAL/SLOW/FAST
    block_time_ranges = []
    for si, (gs, ge) in enumerate(sign_groups):
        group_bursts = bursts[gs:ge]

        if len(group_bursts) <= 2:
            # Can't split
            block_time_ranges.append((group_bursts[0][0], group_bursts[-1][1]))
            block_time_ranges.append((group_bursts[-1][1] + 0.01, group_bursts[-1][1] + 0.02))
            block_time_ranges.append((group_bursts[-1][1] + 0.02, group_bursts[-1][1] + 0.03))
            continue

        # Gaps within this sign group
        group_gaps = []
        for i in range(len(group_bursts) - 1):
            gd = group_bursts[i + 1][0] - group_bursts[i][1]
            group_gaps.append((i, gd))

        group_gaps_sorted = sorted(group_gaps, key=lambda x: x[1], reverse=True)
        if len(group_gaps_sorted) >= 2:
            speed_splits = sorted([group_gaps_sorted[0][0], group_gaps_sorted[1][0]])
            print(f"    {sign_names[si]} speed splits at local idx: {speed_splits} "
                  f"(gaps: {round(group_gaps_sorted[0][1],2)}s, {round(group_gaps_sorted[1][1],2)}s)")

            ranges = [
                (0, speed_splits[0] + 1),
                (speed_splits[0] + 1, speed_splits[1] + 1),
                (speed_splits[1] + 1, len(group_bursts)),
            ]
            for ss, se in ranges:
                sb = group_bursts[ss:se]
                block_time_ranges.append((sb[0][0], sb[-1][1]))
        else:
            # Only 1 gap available
            gi = group_gaps_sorted[0][0]
            block_time_ranges.append((group_bursts[0][0], group_bursts[gi][1]))
            block_time_ranges.append((group_bursts[gi+1][0], group_bursts[-1][1]))
            block_time_ranges.append((group_bursts[-1][1] + 0.01, group_bursts[-1][1] + 0.02))

    # Filter degenerate blocks
    block_time_ranges = [(s, e) for s, e in block_time_ranges if e - s > 0.1]

    print(f"  Detected {len(block_time_ranges)} blocks (expected 9)")
    for bi, (bs, be) in enumerate(block_time_ranges):
        print(f"    Block {bi}: {bs:.1f}s - {be:.1f}s ({be-bs:.1f}s)")

    # ── Within-block segmentation using local minima ──
    from scipy.signal import find_peaks as scipy_find_peaks

    segments = []

    for block_idx, (blk_start, blk_end) in enumerate(block_time_ranges):
        if block_idx >= len(BLOCKS):
            print(f"  WARNING: More blocks than expected!")
            break

        block_info = BLOCKS[block_idx]
        n_expected = block_info["n_reps"]

        # Extract energy within this block's time range (with small padding)
        pad = 0.2
        bmask = (act_times >= blk_start - pad) & (act_times <= blk_end + pad)
        blk_times = act_times[bmask]
        blk_smooth = act_smooth[bmask]

        if len(blk_times) < 5:
            for ri in range(n_expected):
                sample_id = f"S001_P001_{block_info['sign_id']}_{block_info['speed']}_R{ri+1:02d}"
                segments.append({
                    "sample_id": sample_id, "block_idx": block_idx, "rep_idx": ri + 1,
                    "sign_id": block_info["sign_id"], "sign_kr": block_info["sign_kr"],
                    "speed": block_info["speed"],
                    "front_start_s": None, "front_end_s": None, "duration_s": None,
                    "peak_energy": None, "camera_ranges": {},
                    "status": "REVIEW_REQUIRED", "review_reason": "Insufficient energy data in block",
                })
            continue

        # Strategy: find local minima in the smoothed energy.
        # We need n_expected-1 = 4 valleys to split into 5 reps.
        # Invert energy to find valleys as peaks.
        inverted = -blk_smooth
        blk_duration = blk_end - blk_start

        # Try multiple prominence levels to find exactly n_expected-1 valleys
        found_reps = None
        for prom_factor in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
            energy_range = np.max(blk_smooth) - np.min(blk_smooth)
            min_prom = energy_range * prom_factor
            min_dist = max(3, int(0.5 / np.median(np.diff(blk_times))))  # at least 0.5s apart

            valley_idx, valley_props = scipy_find_peaks(inverted, prominence=min_prom, distance=min_dist)

            if len(valley_idx) == n_expected - 1:
                # Perfect split
                split_times = blk_times[valley_idx]
                reps = []
                prev_t = blk_start
                for st in split_times:
                    reps.append((prev_t, st))
                    prev_t = st
                reps.append((prev_t, blk_end))
                found_reps = reps
                break

        if found_reps is None:
            # Fallback: try equal-time splitting if block has enough duration
            # for n_expected reps, but mark as REVIEW_REQUIRED
            avg_rep = blk_duration / n_expected
            fallback_reps = []
            for ri in range(n_expected):
                rs = blk_start + ri * avg_rep
                re = blk_start + (ri + 1) * avg_rep
                fallback_reps.append((rs, re))

            # Or just use the burst-based results
            block_bursts_orig = [b for b in bursts if b[0] >= blk_start - 0.5 and b[1] <= blk_end + 0.5]

            if len(block_bursts_orig) == n_expected:
                found_reps = [(b[0], b[1]) for b in block_bursts_orig]
                rep_status = "OK"
            else:
                # Use valley approach with closest count
                best_valleys = None
                best_count_diff = float("inf")
                for prom_factor in np.arange(0.02, 0.6, 0.02):
                    energy_range = np.max(blk_smooth) - np.min(blk_smooth)
                    min_prom = energy_range * prom_factor
                    min_dist = max(3, int(0.3 / np.median(np.diff(blk_times))))
                    vi, _ = scipy_find_peaks(inverted, prominence=min_prom, distance=min_dist)
                    diff = abs(len(vi) - (n_expected - 1))
                    if diff < best_count_diff:
                        best_count_diff = diff
                        best_valleys = vi

                if best_valleys is not None and len(best_valleys) >= 1:
                    # Use best valleys, take top n_expected-1 by time
                    if len(best_valleys) > n_expected - 1:
                        # Take the n_expected-1 most prominent
                        inv_vals = inverted[best_valleys]
                        top_k = np.argsort(inv_vals)[:n_expected-1]  # most negative = deepest valleys
                        best_valleys = np.sort(best_valleys[top_k])

                    split_times = blk_times[best_valleys]
                    reps = []
                    prev_t = blk_start
                    for st in split_times:
                        reps.append((prev_t, st))
                        prev_t = st
                    reps.append((prev_t, blk_end))
                    found_reps = reps
                else:
                    found_reps = fallback_reps

        # Determine status
        if found_reps and len(found_reps) == n_expected:
            # Check if durations are reasonable
            durations = [e - s for s, e in found_reps]
            cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 999
            if cv < 0.6:  # coefficient of variation < 60%
                rep_status = "OK"
                rep_reason = ""
            else:
                rep_status = "REVIEW_REQUIRED"
                rep_reason = f"High duration variability (CV={cv:.2f})"
        else:
            rep_status = "REVIEW_REQUIRED"
            rep_reason = f"Found {len(found_reps) if found_reps else 0} reps instead of {n_expected}"

        if found_reps:
            for ri, (rs, re) in enumerate(found_reps):
                sample_id = f"S001_P001_{block_info['sign_id']}_{block_info['speed']}_R{ri+1:02d}"
                duration = round(re - rs, 3)

                cam_ranges = {}
                for view in ["FRONT", "LEFT", "RIGHT"]:
                    local_start = front_to_camera_time(rs, view)
                    local_end = front_to_camera_time(re, view)
                    cam_ranges[view] = {
                        "start_pts_s": round(local_start, 4),
                        "end_pts_s": round(local_end, 4),
                    }

                # Per-rep energy stats
                rmask = (act_times >= rs) & (act_times <= re)
                peak_e = float(np.max(act_smooth[rmask])) if np.any(rmask) else 0.0

                segments.append({
                    "sample_id": sample_id,
                    "block_idx": block_idx,
                    "rep_idx": ri + 1,
                    "sign_id": block_info["sign_id"],
                    "sign_kr": block_info["sign_kr"],
                    "speed": block_info["speed"],
                    "front_start_s": round(rs, 4),
                    "front_end_s": round(re, 4),
                    "duration_s": duration,
                    "peak_energy": round(peak_e, 4),
                    "camera_ranges": cam_ranges,
                    "status": rep_status,
                    "review_reason": rep_reason,
                })

    return segments, bursts, act_times, act_smooth, action_thresh


def generate_contact_sheet(video_path, segments, out_dir, n_cols=5, thumb_w=320):
    """Generate a contact sheet thumbnail for each block."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    thumb_h = int(thumb_w * h_orig / w_orig)

    # Group segments by block
    blocks = {}
    for seg in segments:
        bi = seg["block_idx"]
        if bi not in blocks:
            blocks[bi] = []
        blocks[bi].append(seg)

    for bi, segs in sorted(blocks.items()):
        valid_segs = [s for s in segs if s["front_start_s"] is not None]
        if not valid_segs:
            continue

        n_rows = (len(valid_segs) + n_cols - 1) // n_cols
        sheet = np.zeros((n_rows * (thumb_h + 30), n_cols * thumb_w, 3), dtype=np.uint8)

        for idx, seg in enumerate(valid_segs):
            mid_s = (seg["front_start_s"] + seg["front_end_s"]) / 2
            # Seek by time (VFR safe)
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_s * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            thumb = cv2.resize(frame, (thumb_w, thumb_h))
            row = idx // n_cols
            col = idx % n_cols
            y = row * (thumb_h + 30)
            x = col * thumb_w
            sheet[y:y + thumb_h, x:x + thumb_w] = thumb

            # Label
            label = f"{seg['sample_id'].split('_')[-1]} {seg['duration_s']:.1f}s"
            cv2.putText(sheet, label, (x + 5, y + thumb_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        block_info = BLOCKS[bi] if bi < len(BLOCKS) else {"sign_id": "?", "speed": "?"}
        sheet_path = out_dir / f"contact_block{bi}_{block_info['sign_id']}_{block_info['speed']}.jpg"
        cv2.imwrite(str(sheet_path), sheet)
        print(f"  Saved: {sheet_path.name}")

    cap.release()


def validate_segments(segments):
    """Validate segment consistency."""
    issues = []

    # Check total count
    n_total = len(segments)
    if n_total != 45:
        issues.append(f"Total segments: {n_total} (expected 45)")

    # Check per-block counts
    for bi, block in enumerate(BLOCKS):
        block_segs = [s for s in segments if s["block_idx"] == bi]
        if len(block_segs) != block["n_reps"]:
            issues.append(f"Block {bi} ({block['sign_id']}_{block['speed']}): "
                          f"{len(block_segs)} reps (expected {block['n_reps']})")

    # Check for overlaps
    valid = [s for s in segments if s["front_start_s"] is not None]
    valid_sorted = sorted(valid, key=lambda s: s["front_start_s"])
    for i in range(len(valid_sorted) - 1):
        if valid_sorted[i]["front_end_s"] > valid_sorted[i + 1]["front_start_s"]:
            issues.append(f"OVERLAP: {valid_sorted[i]['sample_id']} end={valid_sorted[i]['front_end_s']:.2f} > "
                          f"{valid_sorted[i+1]['sample_id']} start={valid_sorted[i+1]['front_start_s']:.2f}")

    # Check temporal order
    for i in range(len(valid_sorted) - 1):
        if valid_sorted[i]["block_idx"] > valid_sorted[i + 1]["block_idx"]:
            issues.append(f"ORDER: {valid_sorted[i]['sample_id']} (block {valid_sorted[i]['block_idx']}) "
                          f"after {valid_sorted[i+1]['sample_id']} (block {valid_sorted[i+1]['block_idx']})")

    # Check durations
    for s in valid:
        if s["duration_s"] is not None:
            if s["duration_s"] < 0.3:
                issues.append(f"SHORT: {s['sample_id']} duration={s['duration_s']:.2f}s")
            if s["duration_s"] > 10.0:
                issues.append(f"LONG: {s['sample_id']} duration={s['duration_s']:.2f}s")

    return issues


def main():
    print("Step 5: Action Segmentation -- S001_P001")
    print(f"  Video: {FRONT_VIDEO}")

    # Extract motion energy from FRONT
    print("\n  Extracting wrist motion energy from FRONT...")
    pts_energy = extract_wrist_motion_energy(FRONT_VIDEO, sample_interval=2)

    times, energies, smoothed = smooth_energy(pts_energy, window_s=0.8)
    print(f"  Smoothed: {len(smoothed)} samples, range [{np.min(smoothed):.2f}, {np.max(smoothed):.2f}]")

    # Segment
    result = segment_actions(times, energies, smoothed)
    if not result:
        print("  FAILED: Could not segment actions!")
        return

    segments, bursts, act_times, act_smooth, thresh = result

    # Validate
    print("\n  Validation:")
    issues = validate_segments(segments)
    if issues:
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("    All checks passed!")

    # Summary
    ok_segs = [s for s in segments if s["status"] == "OK"]
    review_segs = [s for s in segments if s["status"] == "REVIEW_REQUIRED"]
    print(f"\n  Total: {len(segments)}, OK: {len(ok_segs)}, REVIEW_REQUIRED: {len(review_segs)}")

    if review_segs:
        print("  REVIEW_REQUIRED samples:")
        for s in review_segs:
            print(f"    {s['sample_id']}: {s['review_reason']}")

    # Per-block summary
    print("\n  Per-block detection:")
    for bi, block in enumerate(BLOCKS):
        block_segs = [s for s in segments if s["block_idx"] == bi and s["front_start_s"] is not None]
        durations = [s["duration_s"] for s in block_segs if s["duration_s"]]
        status_counts = {}
        for s in [seg for seg in segments if seg["block_idx"] == bi]:
            status_counts[s["status"]] = status_counts.get(s["status"], 0) + 1
        dur_str = f"dur: {min(durations):.1f}-{max(durations):.1f}s" if durations else "no durations"
        print(f"    Block {bi} ({block['sign_id']}_{block['speed']}): "
              f"{len(block_segs)} detected, {dur_str}, statuses={status_counts}")

    # Save JSON
    output = {
        "session_id": "S001",
        "participant_id": "P001",
        "n_blocks": len(BLOCKS),
        "n_segments_expected": 45,
        "n_segments_detected": len(segments),
        "n_ok": len(ok_segs),
        "n_review_required": len(review_segs),
        "segmentation_method": "frame_differencing_energy",
        "energy_threshold": round(float(thresh), 4),
        "smoothing_window_s": 0.8,
        "front_video_vfr": True,
        "sync_offsets_applied": True,
        "drift_correction_applied": True,
        "validation_issues": issues,
        "blocks": [
            {
                "block_idx": bi,
                "sign_id": BLOCKS[bi]["sign_id"],
                "sign_kr": BLOCKS[bi]["sign_kr"],
                "speed": BLOCKS[bi]["speed"],
                "n_detected": len([s for s in segments if s["block_idx"] == bi and s["front_start_s"] is not None]),
                "n_expected": BLOCKS[bi]["n_reps"],
            }
            for bi in range(len(BLOCKS))
        ],
        "segments": segments,
    }

    json_path = OUT_DIR / "action_segments.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")

    # Save CSV
    csv_path = OUT_DIR / "action_segments.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "block_idx", "rep_idx", "sign_id", "sign_kr", "speed",
            "front_start_s", "front_end_s", "duration_s", "peak_energy",
            "left_start_s", "left_end_s", "right_start_s", "right_end_s",
            "status", "review_reason"
        ])
        for s in segments:
            lr = s.get("camera_ranges", {})
            left = lr.get("LEFT", {})
            right = lr.get("RIGHT", {})
            writer.writerow([
                s["sample_id"], s["block_idx"], s["rep_idx"],
                s["sign_id"], s["sign_kr"], s["speed"],
                s.get("front_start_s"), s.get("front_end_s"),
                s.get("duration_s"), s.get("peak_energy"),
                left.get("start_pts_s"), left.get("end_pts_s"),
                right.get("start_pts_s"), right.get("end_pts_s"),
                s["status"], s.get("review_reason", "")
            ])
    print(f"  Saved: {csv_path}")

    # Contact sheets
    print("\n  Generating contact sheets...")
    contact_dir = OUT_DIR / "contact_sheets"
    contact_dir.mkdir(exist_ok=True)
    generate_contact_sheet(FRONT_VIDEO, segments, contact_dir)

    # Save energy profile for debugging
    energy_path = OUT_DIR / "energy_profile.json"
    # Subsample for file size (every 10th point)
    energy_sub = [(round(float(times[i]), 4), round(float(smoothed[i]), 4))
                  for i in range(0, len(times), 10)]
    with open(energy_path, "w") as f:
        json.dump({"energy": energy_sub, "threshold": round(float(thresh), 4)}, f)
    print(f"  Saved: {energy_path}")

    print("\n  Step 5 complete.")


if __name__ == "__main__":
    main()
