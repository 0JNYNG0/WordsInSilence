"""
Step 5 v2: Action Segmentation for S001_P001

Improvements over v1:
- Multi-camera energy fusion (FRONT+LEFT+RIGHT) via PTS alignment
- PyAV for accurate VFR PTS on FRONT
- Proper action window detection (not C2+5s)
- Per-block adaptive smoothing (short for FAST, longer for SLOW)
- Motion onset/offset boundaries (not midpoints of neutral zones)
- Per-sample quality checks (not just block-level CV)
- 5-frame contact sheets, energy timeline plots, preview videos
- Always outputs 45 canonical rows
- No MediaPipe, no equal-time fallback
"""

import cv2
import numpy as np
import json
import csv
import av
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sessions" / "S001_P001"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "sessions" / "S001_P001"

VIDEOS = {
    "FRONT": RAW_DIR / "S001_P001_FRONT.mp4",
    "LEFT":  RAW_DIR / "S001_P001_LEFT.mp4",
    "RIGHT": RAW_DIR / "S001_P001_RIGHT.mp4",
}

SYNC = {
    "FRONT": {"offset_s": 0.0, "drift_rate": 0.0},
    "LEFT":  {"offset_s": 2.300333, "drift_rate": -19.3e-6},
    "RIGHT": {"offset_s": 9.388633, "drift_rate": -0.3e-6},
}

# Load sync from file
sync_path = OUT_DIR / "sync_report.json"
if sync_path.exists():
    sd = json.load(open(sync_path))
    for view in ["LEFT", "RIGHT"]:
        cam = sd["cameras"].get(view, {})
        SYNC[view]["offset_s"] = cam.get("offset_s_final", SYNC[view]["offset_s"])
        drift_ppm = cam.get("drift_ppm", 0) or 0
        SYNC[view]["drift_rate"] = drift_ppm * 1e-6

BLOCKS = [
    {"block_idx": 0, "sign_id": "GREETING",  "sign_kr": "인사",      "speed": "NORMAL"},
    {"block_idx": 1, "sign_id": "GREETING",  "sign_kr": "인사",      "speed": "SLOW"},
    {"block_idx": 2, "sign_id": "GREETING",  "sign_kr": "인사",      "speed": "FAST"},
    {"block_idx": 3, "sign_id": "THANK_YOU", "sign_kr": "감사합니다", "speed": "NORMAL"},
    {"block_idx": 4, "sign_id": "THANK_YOU", "sign_kr": "감사합니다", "speed": "SLOW"},
    {"block_idx": 5, "sign_id": "THANK_YOU", "sign_kr": "감사합니다", "speed": "FAST"},
    {"block_idx": 6, "sign_id": "LOVE",      "sign_kr": "사랑",      "speed": "NORMAL"},
    {"block_idx": 7, "sign_id": "LOVE",      "sign_kr": "사랑",      "speed": "SLOW"},
    {"block_idx": 8, "sign_id": "LOVE",      "sign_kr": "사랑",      "speed": "FAST"},
]

N_REPS = 5
CANONICAL_IDS = []
for b in BLOCKS:
    for r in range(1, N_REPS + 1):
        CANONICAL_IDS.append(f"S001_P001_{b['sign_id']}_{b['speed']}_R{r:02d}")


def camera_to_front_time(cam_time_s, view):
    s = SYNC[view]
    # cam_time = front_time + offset + drift_rate * front_time
    # cam_time = front_time * (1 + drift_rate) + offset
    # front_time = (cam_time - offset) / (1 + drift_rate)
    return (cam_time_s - s["offset_s"]) / (1.0 + s["drift_rate"])


def front_to_camera_time(front_time_s, view):
    s = SYNC[view]
    return front_time_s * (1.0 + s["drift_rate"]) + s["offset_s"]


# ═══════════════════════════════════════════════════════════
#  Energy extraction with proper PTS
# ═══════════════════════════════════════════════════════════
def extract_energy_pyav(video_path, view, sample_interval=2, scale=0.25):
    """
    Extract frame-differencing energy using PyAV for accurate PTS.
    Downscales frames for speed. Returns [(front_time_s, energy), ...].
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    tb = float(stream.time_base)
    pts_energy = []
    prev_gray = None
    count = 0
    skip = 0

    for frame in container.decode(stream):
        if skip < sample_interval - 1:
            skip += 1
            continue
        skip = 0

        # Accurate PTS
        cam_time_s = float(frame.pts) * tb if frame.pts is not None else count * (1.0 / 30.0)

        # Convert to front time
        front_time_s = camera_to_front_time(cam_time_s, view)

        # Downscale
        img = frame.to_ndarray(format="gray")
        h, w = img.shape
        sh, sw = int(h * scale), int(w * scale)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)

        # ROI: central 60% width, 20%-85% height
        y1, y2 = int(sh * 0.20), int(sh * 0.85)
        x1, x2 = int(sw * 0.20), int(sw * 0.80)
        roi = small[y1:y2, x1:x2]

        if prev_gray is not None:
            diff = cv2.absdiff(roi, prev_gray)
            energy = float(np.mean(diff))
            pts_energy.append((front_time_s, energy))

        prev_gray = roi.copy()
        count += 1

        if count % 2000 == 0:
            print(f"    {view}: {count} frames, t={front_time_s:.1f}s")

    container.close()
    print(f"    {view}: {len(pts_energy)} energy samples, "
          f"t=[{pts_energy[0][0]:.1f}, {pts_energy[-1][0]:.1f}]s")
    return pts_energy


def fuse_multi_camera_energy(all_energies, bin_size_s=0.1):
    """
    Fuse energy from multiple cameras by binning on front_time and averaging.
    """
    # Collect all times
    all_times = []
    for elist in all_energies.values():
        all_times.extend([t for t, e in elist])

    t_min = min(all_times)
    t_max = max(all_times)
    n_bins = int((t_max - t_min) / bin_size_s) + 1
    bins = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for elist in all_energies.values():
        for t, e in elist:
            bi = int((t - t_min) / bin_size_s)
            if 0 <= bi < n_bins:
                bins[bi] += e
                counts[bi] += 1

    counts[counts == 0] = 1
    fused = bins / counts
    times = np.array([t_min + i * bin_size_s for i in range(n_bins)])

    return times, fused


def multi_smooth(energy, dt_s, window_s_list):
    """Return dict of smoothed versions at different window sizes."""
    results = {}
    for ws in window_s_list:
        win = max(1, int(ws / dt_s))
        results[ws] = uniform_filter1d(energy, size=win, mode="nearest")
    return results


# ═══════════════════════════════════════════════════════════
#  Action window detection
# ═══════════════════════════════════════════════════════════
def find_action_window(times, energy_short, c2_end_s, clap_start_s):
    """
    Find the actual action start: after C2 region, after the participant
    settles into a stable neutral pose, and before the first real action.
    """
    # After C2 ends, look for a sustained low-energy period (neutral pose)
    # then the first significant energy rise (first action onset)
    search_start = c2_end_s
    search_end = min(search_start + 60, times[-1])

    mask = (times >= search_start) & (times <= search_end)
    seg_t = times[mask]
    seg_e = energy_short[mask]

    if len(seg_e) < 10:
        return c2_end_s + 10, clap_start_s - 5

    # Find first stable low period (energy < p25 for at least 2s)
    p25 = np.percentile(seg_e, 25)
    low_mask = seg_e < p25

    # Find runs of low energy
    stable_start = None
    run_start = None
    for i in range(len(low_mask)):
        if low_mask[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_dur = seg_t[i - 1] - seg_t[run_start]
                if run_dur >= 1.5:
                    stable_start = seg_t[run_start]
                    stable_end_idx = i - 1
                    break
            run_start = None

    if stable_start is None:
        # Fallback: first time energy drops below p25
        low_idx = np.where(low_mask)[0]
        if len(low_idx) > 0:
            stable_start = seg_t[low_idx[0]]
        else:
            return c2_end_s + 10, clap_start_s - 5

    # Now find the first significant rise after the neutral period
    post_stable = (times >= stable_start) & (times <= search_end)
    ps_t = times[post_stable]
    ps_e = energy_short[post_stable]

    # First point where energy exceeds 2x the neutral median
    neutral_median = np.median(ps_e[:max(1, len(ps_e)//4)])
    rise_thresh = neutral_median * 2.5

    for i in range(len(ps_e)):
        if ps_e[i] > rise_thresh:
            action_start = ps_t[max(0, i - 2)]  # slightly before onset
            break
    else:
        action_start = stable_start + 5

    # Action end: shortly before end claps
    action_end = clap_start_s - 3 if clap_start_s else times[-1] - 10

    return action_start, action_end


# ═══════════════════════════════════════════════════════════
#  Block and rep segmentation
# ═══════════════════════════════════════════════════════════
def find_motion_bursts(times, energy, threshold):
    """Find contiguous above-threshold regions. Returns [(start_s, end_s, peak), ...]."""
    is_high = energy > threshold
    bursts = []
    in_burst = False
    start_i = 0
    for i in range(len(is_high)):
        if is_high[i] and not in_burst:
            start_i = i
            in_burst = True
        elif not is_high[i] and in_burst:
            peak = float(np.max(energy[start_i:i]))
            bursts.append((times[start_i], times[i - 1], peak))
            in_burst = False
    if in_burst:
        peak = float(np.max(energy[start_i:]))
        bursts.append((times[start_i], times[-1], peak))
    return bursts


def merge_close_bursts(bursts, min_gap_s):
    """Merge bursts separated by less than min_gap_s."""
    if not bursts:
        return bursts
    merged = [list(bursts[0])]
    for s, e, p in bursts[1:]:
        if s - merged[-1][1] < min_gap_s:
            merged[-1][1] = e
            merged[-1][2] = max(merged[-1][2], p)
        else:
            merged.append([s, e, p])
    return [(s, e, p) for s, e, p in merged]


def find_onset_offset(times, energy, approx_start, approx_end, neutral_level):
    """
    Refine burst boundaries to motion onset (energy rises above neutral)
    and motion offset (energy returns to neutral).
    """
    onset_thresh = neutral_level * 1.5

    # Find onset: scan backwards from approx_start to find where energy first exceeds threshold
    mask_before = (times >= approx_start - 2.0) & (times <= approx_start + 0.5)
    if np.any(mask_before):
        idx_before = np.where(mask_before)[0]
        onset_idx = idx_before[0]
        for i in reversed(idx_before):
            if energy[i] <= onset_thresh:
                onset_idx = i + 1 if i + 1 < len(times) else i
                break
        onset = times[min(onset_idx, len(times) - 1)]
    else:
        onset = approx_start

    # Find offset: scan forward from approx_end
    mask_after = (times >= approx_end - 0.5) & (times <= approx_end + 2.0)
    if np.any(mask_after):
        idx_after = np.where(mask_after)[0]
        offset_idx = idx_after[-1]
        for i in idx_after:
            if energy[i] <= onset_thresh:
                offset_idx = i
                break
        offset = times[min(offset_idx, len(times) - 1)]
    else:
        offset = approx_end

    return onset, offset


def segment_block_reps(times, energy, blk_start, blk_end, speed, neutral_level):
    """
    Within a block time range, find 5 repetitions by detecting 4 inter-rep
    neutral valleys and 5 motion bursts with onset/offset boundaries.
    Returns list of (onset_s, offset_s, peak_energy, confidence) or None items.
    """
    mask = (times >= blk_start) & (times <= blk_end)
    bt = times[mask]
    be = energy[mask]

    if len(bt) < 10:
        return [None] * 5

    dt = float(np.median(np.diff(bt)))

    # Adaptive smoothing per speed (include heavy options for multi-peak signs)
    block_dur = blk_end - blk_start
    if speed == "FAST":
        smooth_wins = [0.1, 0.15, 0.2, 0.4, 0.8, 1.0]
    elif speed == "SLOW":
        smooth_wins = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    else:
        smooth_wins = [0.15, 0.25, 0.4, 0.8, 1.0, 1.5, 2.0]
    # For very long blocks (>30s), prioritize heavy smoothing
    if block_dur > 30:
        smooth_wins = sorted(set(smooth_wins + [1.5, 2.0, 2.5, 3.0, 4.0]), reverse=True)

    best_reps = None
    best_score = -1

    for sw in smooth_wins:
        win = max(1, int(sw / dt))
        be_smooth = uniform_filter1d(be, size=win, mode="nearest")

        # Try multiple threshold factors
        for thresh_pct in [20, 30, 40]:
            thresh = np.percentile(be_smooth, thresh_pct)
            if thresh < neutral_level * 1.2:
                thresh = neutral_level * 1.2

            bursts = find_motion_bursts(bt, be_smooth, thresh)

            # Merge very close bursts
            min_gap = 0.15 if speed == "FAST" else 0.3
            bursts = merge_close_bursts(bursts, min_gap)

            # Filter very short bursts
            min_dur = 0.2 if speed == "FAST" else 0.4
            bursts = [(s, e, p) for s, e, p in bursts if e - s >= min_dur]

            if len(bursts) == 5:
                # Perfect! Score based on consistency
                durations = [e - s for s, e, p in bursts]
                cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 999
                score = 10.0 - cv  # lower CV = better
                if score > best_score:
                    best_score = score
                    best_reps = bursts

    # If no perfect 5 found, try inverted peak approach (find 4 valleys)
    if best_reps is None:
        for sw in smooth_wins:
            win = max(1, int(sw / dt))
            be_smooth = uniform_filter1d(be, size=win, mode="nearest")

            inverted = -be_smooth
            min_dist = max(3, int(0.5 / dt))

            for prom_f in np.arange(0.03, 0.5, 0.03):
                erange = np.max(be_smooth) - np.min(be_smooth)
                if erange <= 0:
                    continue
                min_prom = erange * prom_f
                valleys, _ = find_peaks(inverted, prominence=min_prom, distance=min_dist)

                if len(valleys) == 4:
                    # Split at valleys
                    splits = [0] + valleys.tolist() + [len(bt) - 1]
                    reps = []
                    for i in range(5):
                        s_idx = splits[i]
                        e_idx = splits[i + 1]
                        if e_idx > s_idx:
                            peak = float(np.max(be_smooth[s_idx:e_idx + 1]))
                            reps.append((bt[s_idx], bt[e_idx], peak))
                        else:
                            reps.append(None)

                    valid_reps = [r for r in reps if r is not None]
                    if len(valid_reps) == 5:
                        durations = [e - s for s, e, p in valid_reps]
                        cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 999
                        score = 5.0 - cv
                        if score > best_score:
                            best_score = score
                            best_reps = valid_reps
                    break  # found 4 valleys

    if best_reps is None:
        return [None] * 5

    # Refine to onset/offset
    refined = []
    for bs, be_t, bp in best_reps:
        onset, offset = find_onset_offset(times, energy, bs, be_t, neutral_level)
        refined.append((onset, offset, bp, round(best_score, 3)))

    return refined


# ═══════════════════════════════════════════════════════════
#  Per-sample quality assessment
# ═══════════════════════════════════════════════════════════
def assess_sample_quality(seg, all_segments, times, energy, neutral_level):
    """
    Per-sample quality check. Returns (status, reasons).
    Checks: neutral evidence, internal energy, overlap, duration, confidence.
    """
    reasons = []
    if seg["front_start_s"] is None:
        return "REVIEW_REQUIRED", ["No boundaries detected"]

    s, e = seg["front_start_s"], seg["front_end_s"]
    dur = e - s

    # Duration checks
    if dur < 0.3:
        reasons.append(f"Very short duration: {dur:.2f}s")
    if dur > 15.0:
        reasons.append(f"Very long duration: {dur:.2f}s (likely merged reps)")

    # Speed-specific duration bounds
    speed = seg["speed"]
    if speed == "FAST" and dur > 5.0:
        reasons.append(f"FAST rep too long: {dur:.2f}s")
    if speed == "SLOW" and dur < 0.5:
        reasons.append(f"SLOW rep too short: {dur:.2f}s")

    # Internal energy: should have significant motion
    mask = (times >= s) & (times <= e)
    if np.any(mask):
        internal_energy = np.mean(energy[mask])
        if internal_energy < neutral_level * 1.3:
            reasons.append(f"Low internal energy: {internal_energy:.2f} (neutral={neutral_level:.2f})")
        peak_internal = np.max(energy[mask])
        seg["internal_energy_mean"] = round(float(internal_energy), 4)
        seg["internal_energy_peak"] = round(float(peak_internal), 4)

    # Neutral evidence at boundaries: energy near start and end should be low
    pre_mask = (times >= s - 0.5) & (times < s)
    post_mask = (times > e) & (times <= e + 0.5)

    if np.any(pre_mask):
        pre_energy = np.mean(energy[pre_mask])
        if pre_energy > neutral_level * 2.0:
            reasons.append(f"High energy before onset: {pre_energy:.2f}")
        seg["pre_boundary_energy"] = round(float(pre_energy), 4)

    if np.any(post_mask):
        post_energy = np.mean(energy[post_mask])
        if post_energy > neutral_level * 2.0:
            reasons.append(f"High energy after offset: {post_energy:.2f}")
        seg["post_boundary_energy"] = round(float(post_energy), 4)

    # Overlap with adjacent segments
    for other in all_segments:
        if other["sample_id"] == seg["sample_id"]:
            continue
        if other["front_start_s"] is None:
            continue
        os, oe = other["front_start_s"], other["front_end_s"]
        if s < oe and e > os:
            overlap = min(e, oe) - max(s, os)
            reasons.append(f"Overlaps with {other['sample_id']} by {overlap:.2f}s")

    # Confidence from segmentation
    conf = seg.get("seg_confidence", 0)
    if conf < 3.0:
        reasons.append(f"Low segmentation confidence: {conf:.2f}")

    if reasons:
        return "REVIEW_REQUIRED", reasons
    return "AUTO_OK", []


# ═══════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════
def make_block_contact_sheet(video_path, segments, block_idx, out_dir, thumb_w=256):
    """5 reps x 5 timepoints (start, 25%, 50%, 75%, end) contact sheet."""
    block_segs = [s for s in segments if s["block_idx"] == block_idx and s["front_start_s"] is not None]
    if not block_segs:
        return

    cap = cv2.VideoCapture(str(video_path))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    thumb_h = int(thumb_w * h_orig / w_orig)

    n_rows = len(block_segs)
    n_cols = 5
    label_h = 25
    sheet = np.zeros((n_rows * (thumb_h + label_h), n_cols * thumb_w, 3), dtype=np.uint8)

    timepoint_labels = ["Start", "25%", "50%", "75%", "End"]

    for row, seg in enumerate(block_segs):
        s, e = seg["front_start_s"], seg["front_end_s"]
        timepoints = [s, s + (e - s) * 0.25, s + (e - s) * 0.5, s + (e - s) * 0.75, e]

        for col, tp in enumerate(timepoints):
            cap.set(cv2.CAP_PROP_POS_MSEC, tp * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            thumb = cv2.resize(frame, (thumb_w, thumb_h))
            y = row * (thumb_h + label_h)
            x = col * thumb_w
            sheet[y:y + thumb_h, x:x + thumb_w] = thumb

            # Label
            if row == 0:
                cv2.putText(sheet, timepoint_labels[col], (x + 5, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if col == 0:
                label = f"{seg['sample_id'].split('_')[-1]} {seg['duration_s']:.1f}s"
                cv2.putText(sheet, label, (x + 5, y + thumb_h + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cap.release()

    bi = BLOCKS[block_idx]
    fname = f"contact_block{block_idx}_{bi['sign_id']}_{bi['speed']}.jpg"
    cv2.imwrite(str(out_dir / fname), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"    Saved: {fname}")


def make_block_timeline(times, energy_short, segments, block_idx, blk_start, blk_end, out_dir):
    """Energy curve with onset/offset markers and neutral zones."""
    pad = 3.0
    mask = (times >= blk_start - pad) & (times <= blk_end + pad)
    bt = times[mask]
    be = energy_short[mask]

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(bt, be, color="steelblue", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("FRONT time (s)")
    ax.set_ylabel("Energy")

    block_segs = [s for s in segments if s["block_idx"] == block_idx]
    colors = plt.cm.Set1(np.linspace(0, 1, 5))

    for i, seg in enumerate(block_segs):
        if seg["front_start_s"] is None:
            continue
        s, e = seg["front_start_s"], seg["front_end_s"]
        ax.axvspan(s, e, alpha=0.2, color=colors[i % 5])
        ax.axvline(s, color=colors[i % 5], linewidth=1.5, linestyle="--")
        ax.axvline(e, color=colors[i % 5], linewidth=1.5, linestyle="--")
        ax.text((s + e) / 2, ax.get_ylim()[1] * 0.9,
                seg["sample_id"].split("_")[-1],
                ha="center", fontsize=8, color="black")

    bi = BLOCKS[block_idx]
    ax.set_title(f"Block {block_idx}: {bi['sign_id']} {bi['speed']}")
    fig.tight_layout()
    fname = f"timeline_block{block_idx}_{bi['sign_id']}_{bi['speed']}.png"
    fig.savefig(str(out_dir / fname), dpi=120)
    plt.close(fig)
    print(f"    Saved: {fname}")


def make_preview_video(video_path, segments, block_idx, out_dir, fps_out=10, scale=0.5):
    """Low-framerate preview video with sample ID overlay."""
    block_segs = [s for s in segments if s["block_idx"] == block_idx and s["front_start_s"] is not None]
    if not block_segs:
        return

    blk_start = min(s["front_start_s"] for s in block_segs) - 2.0
    blk_end = max(s["front_end_s"] for s in block_segs) + 2.0

    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

    bi = BLOCKS[block_idx]
    fname = f"preview_block{block_idx}_{bi['sign_id']}_{bi['speed']}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_dir / fname), fourcc, fps_out, (w, h))

    cap.set(cv2.CAP_PROP_POS_MSEC, blk_start * 1000)
    frame_interval_ms = 1000.0 / fps_out
    next_grab_ms = blk_start * 1000

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_s = current_ms / 1000.0

        if current_s > blk_end:
            break
        if current_ms < next_grab_ms:
            continue
        next_grab_ms = current_ms + frame_interval_ms

        small = cv2.resize(frame, (w, h))

        # Overlay current segment info
        label = ""
        for seg in block_segs:
            if seg["front_start_s"] <= current_s <= seg["front_end_s"]:
                label = seg["sample_id"].split("_", 3)[-1]
                cv2.rectangle(small, (0, 0), (w, 30), (0, 100, 0), -1)
                break

        if not label:
            cv2.rectangle(small, (0, 0), (w, 30), (80, 80, 80), -1)
            label = "NEUTRAL"

        cv2.putText(small, f"{label}  t={current_s:.1f}s", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(small)

    writer.release()
    cap.release()
    print(f"    Saved: {fname}")


# ═══════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════
def main():
    print("Step 5 v2: Action Segmentation -- S001_P001")

    out_contacts = OUT_DIR / "contact_sheets_v2"
    out_timelines = OUT_DIR / "segmentation_timelines_v2"
    out_contacts.mkdir(exist_ok=True)
    out_timelines.mkdir(exist_ok=True)

    # ── 1. Extract multi-camera energy ──
    print("\n  Extracting energy from all cameras (PyAV PTS)...")
    all_energies = {}
    for view, path in VIDEOS.items():
        print(f"  {view}:")
        all_energies[view] = extract_energy_pyav(path, view, sample_interval=2, scale=0.25)

    # ── 2. Fuse and smooth ──
    print("\n  Fusing multi-camera energy...")
    times, fused = fuse_multi_camera_energy(all_energies, bin_size_s=0.05)
    dt = float(np.median(np.diff(times)))
    print(f"  Fused: {len(times)} bins, dt={dt*1000:.1f}ms, range=[{times[0]:.1f}, {times[-1]:.1f}]s")

    # Multiple smoothing scales
    smoothed = multi_smooth(fused, dt, [0.1, 0.15, 0.25, 0.4, 0.8])
    energy_short = smoothed[0.15]  # default for block detection
    energy_fine = smoothed[0.1]    # for FAST blocks

    # Neutral level: robust estimate from low percentile
    neutral_level = float(np.percentile(fused, 10))
    print(f"  Neutral energy level: {neutral_level:.3f}")

    # ── 3. Find action window ──
    c2_report = json.load(open(OUT_DIR / "C2_CALIBRATION.json"))
    c2_end_s = c2_report["c2_timestamp_ranges"]["FRONT"]["end_pts_ms"] / 1000.0
    sync_data = json.load(open(OUT_DIR / "sync_report.json"))
    front_end_claps = sync_data["cameras"]["FRONT"].get("end_clap_times_s", [])
    clap_start = front_end_claps[0] if front_end_claps else times[-1] - 10

    action_start, action_end = find_action_window(times, energy_short, c2_end_s, clap_start)
    print(f"  Action window: {action_start:.1f}s - {action_end:.1f}s ({action_end - action_start:.1f}s)")

    # ── 4. Block detection ──
    # Strategy: heavy smoothing (1.5s) to merge multi-peak reps into single bumps,
    # then find ~45 rep-level bursts, group into 9 blocks via top-8 gaps.
    act_mask = (times >= action_start) & (times <= action_end)
    act_t = times[act_mask]

    # Try multiple heavy smoothing scales to get close to 45 bursts
    best_bursts = None
    best_n_diff = 999
    for heavy_sw in [1.0, 1.5, 2.0, 2.5, 0.8]:
        heavy_win = max(1, int(heavy_sw / dt))
        heavy_e = uniform_filter1d(fused, size=heavy_win, mode="nearest")
        act_heavy = heavy_e[act_mask]

        for thresh_pct in [15, 20, 25, 30, 35]:
            thresh = float(np.percentile(act_heavy, thresh_pct))
            if thresh < neutral_level * 1.1:
                thresh = neutral_level * 1.1

            bursts = find_motion_bursts(act_t, act_heavy, thresh)
            bursts = merge_close_bursts(bursts, 0.8)  # merge sub-0.8s gaps
            bursts = [(s, e, p) for s, e, p in bursts if e - s >= 0.5]

            n_diff = abs(len(bursts) - 45)
            if n_diff < best_n_diff:
                best_n_diff = n_diff
                best_bursts = bursts
            if len(bursts) == 45:
                break
        if best_n_diff == 0:
            break

    bursts = best_bursts
    print(f"  Rep-level bursts: {len(bursts)} (target=45, best_diff={best_n_diff})")

    if len(bursts) < 9:
        print("  ERROR: Too few bursts!")
        return

    # Inter-burst gaps
    gaps = [(i, bursts[i + 1][0] - bursts[i][1]) for i in range(len(bursts) - 1)]
    gaps_desc = sorted(gaps, key=lambda x: x[1], reverse=True)
    print(f"  Top 15 gaps: {[round(g[1], 2) for g in gaps_desc[:15]]}")

    # Find 8 block boundaries from the 8 largest gaps
    if len(gaps_desc) < 8:
        print(f"  ERROR: Only {len(gaps_desc)} inter-burst gaps, need at least 8 for 9 blocks!")
        return

    top8_indices = sorted([g[0] for g in gaps_desc[:8]])
    # Build 9 block ranges from burst groups
    block_burst_ranges = []
    prev = 0
    for gi in top8_indices:
        block_burst_ranges.append((prev, gi + 1))
        prev = gi + 1
    block_burst_ranges.append((prev, len(bursts)))

    block_time_ranges = []
    for bs_idx, be_idx in block_burst_ranges:
        bb = bursts[bs_idx:be_idx]
        if bb:
            block_time_ranges.append((bb[0][0], bb[-1][1]))
        else:
            block_time_ranges.append((None, None))

    # Validate: each block should have ~5 bursts
    print(f"\n  Block burst counts: {[be-bs for bs, be in block_burst_ranges]}")

    # Filter degenerate
    valid_blocks = [(s, e) for s, e in block_time_ranges if s is not None and e is not None and e - s > 0.5]

    print(f"\n  Detected {len(valid_blocks)} blocks:")
    for i, (bs, be) in enumerate(valid_blocks):
        print(f"    Block {i}: {bs:.1f}s - {be:.1f}s ({be - bs:.1f}s)")

    # ── 5. Within-block rep segmentation ──
    print("\n  Segmenting reps within blocks...")
    segments = []

    for bi in range(len(BLOCKS)):
        block_info = BLOCKS[bi]

        if bi < len(valid_blocks):
            blk_s, blk_e = valid_blocks[bi]
            speed = block_info["speed"]

            # Use appropriate energy for this block's speed
            if speed == "FAST":
                seg_energy = smoothed[0.1][act_mask] if len(smoothed[0.1]) == len(times) else energy_fine[act_mask]
                # Actually use full-length arrays
                e_for_seg = smoothed[0.1]
            elif speed == "SLOW":
                e_for_seg = smoothed[0.4]
            else:
                e_for_seg = smoothed[0.25]

            reps = segment_block_reps(times, e_for_seg, blk_s, blk_e, speed, neutral_level)

            for ri in range(N_REPS):
                sample_id = f"S001_P001_{block_info['sign_id']}_{block_info['speed']}_R{ri + 1:02d}"
                if ri < len(reps) and reps[ri] is not None:
                    onset, offset, peak, conf = reps[ri]
                    cam_ranges = {}
                    for view in ["FRONT", "LEFT", "RIGHT"]:
                        cam_ranges[view] = {
                            "start_pts_s": round(front_to_camera_time(onset, view), 4),
                            "end_pts_s": round(front_to_camera_time(offset, view), 4),
                        }
                    segments.append({
                        "sample_id": sample_id,
                        "block_idx": bi,
                        "rep_idx": ri + 1,
                        "sign_id": block_info["sign_id"],
                        "sign_kr": block_info["sign_kr"],
                        "speed": speed,
                        "front_start_s": round(onset, 4),
                        "front_end_s": round(offset, 4),
                        "duration_s": round(offset - onset, 3),
                        "peak_energy": round(peak, 4),
                        "seg_confidence": conf,
                        "camera_ranges": cam_ranges,
                        "status": "PENDING",
                        "review_reasons": [],
                    })
                else:
                    segments.append({
                        "sample_id": sample_id,
                        "block_idx": bi,
                        "rep_idx": ri + 1,
                        "sign_id": block_info["sign_id"],
                        "sign_kr": block_info["sign_kr"],
                        "speed": speed,
                        "front_start_s": None,
                        "front_end_s": None,
                        "duration_s": None,
                        "peak_energy": None,
                        "seg_confidence": 0,
                        "camera_ranges": {},
                        "status": "REVIEW_REQUIRED",
                        "review_reasons": [f"Rep {ri+1} not detected in block {bi}"],
                    })
        else:
            for ri in range(N_REPS):
                sample_id = f"S001_P001_{block_info['sign_id']}_{block_info['speed']}_R{ri + 1:02d}"
                segments.append({
                    "sample_id": sample_id,
                    "block_idx": bi,
                    "rep_idx": ri + 1,
                    "sign_id": block_info["sign_id"],
                    "sign_kr": block_info["sign_kr"],
                    "speed": block_info["speed"],
                    "front_start_s": None,
                    "front_end_s": None,
                    "duration_s": None,
                    "peak_energy": None,
                    "seg_confidence": 0,
                    "camera_ranges": {},
                    "status": "REVIEW_REQUIRED",
                    "review_reasons": [f"Block {bi} not detected"],
                })

    # ── 6. Per-sample quality assessment ──
    print("\n  Quality assessment...")
    for seg in segments:
        if seg["status"] == "PENDING":
            status, reasons = assess_sample_quality(seg, segments, times, fused, neutral_level)
            seg["status"] = status
            seg["review_reasons"] = reasons

    # ── 7. Verify 45 canonical IDs ──
    seg_ids = [s["sample_id"] for s in segments]
    missing = [cid for cid in CANONICAL_IDS if cid not in seg_ids]
    if missing:
        print(f"  WARNING: Missing canonical IDs: {missing}")
        for mid in missing:
            # Parse block info from ID
            parts = mid.split("_")
            bi_search = [i for i, b in enumerate(BLOCKS) if b["sign_id"] == parts[2] + ("_" + parts[3] if parts[3] not in ["NORMAL","SLOW","FAST"] else "") and b["speed"] == parts[-2]]
            segments.append({
                "sample_id": mid, "block_idx": -1, "rep_idx": -1,
                "sign_id": "", "sign_kr": "", "speed": "",
                "front_start_s": None, "front_end_s": None,
                "duration_s": None, "peak_energy": None, "seg_confidence": 0,
                "camera_ranges": {}, "status": "REVIEW_REQUIRED",
                "review_reasons": ["Missing from segmentation"],
            })

    assert len(segments) == 45, f"Expected 45 segments, got {len(segments)}"

    # ── 8. Summary ──
    ok_count = sum(1 for s in segments if s["status"] == "AUTO_OK")
    review_count = sum(1 for s in segments if s["status"] == "REVIEW_REQUIRED")
    print(f"\n  Results: {ok_count} AUTO_OK, {review_count} REVIEW_REQUIRED (total {len(segments)})")

    if review_count > 0:
        print("  REVIEW_REQUIRED details:")
        for s in segments:
            if s["status"] == "REVIEW_REQUIRED":
                reasons_str = "; ".join(s["review_reasons"]) if s["review_reasons"] else "unknown"
                print(f"    {s['sample_id']}: {reasons_str}")

    # Per-block summary
    print("\n  Per-block:")
    for bi in range(len(BLOCKS)):
        b = BLOCKS[bi]
        bsegs = [s for s in segments if s["block_idx"] == bi]
        detected = [s for s in bsegs if s["front_start_s"] is not None]
        ok = sum(1 for s in bsegs if s["status"] == "AUTO_OK")
        rev = sum(1 for s in bsegs if s["status"] == "REVIEW_REQUIRED")
        durs = [s["duration_s"] for s in detected if s["duration_s"]]
        dur_str = f"{min(durs):.1f}-{max(durs):.1f}s" if durs else "N/A"
        print(f"    Block {bi} ({b['sign_id']}_{b['speed']}): "
              f"{len(detected)}/5 detected, {ok} OK, {rev} REVIEW, dur={dur_str}")

    # ── 9. Save results ──
    output = {
        "version": "v2",
        "session_id": "S001",
        "participant_id": "P001",
        "n_blocks": 9,
        "n_segments_expected": 45,
        "n_segments": 45,
        "n_auto_ok": ok_count,
        "n_review_required": review_count,
        "method": "multi_camera_frame_diff_energy_fusion",
        "energy_bin_size_s": 0.05,
        "smoothing_windows_s": [0.1, 0.15, 0.25, 0.4, 0.8],
        "neutral_energy_level": round(neutral_level, 4),
        "action_window": {"start_s": round(action_start, 2), "end_s": round(action_end, 2)},
        "front_vfr": True,
        "pts_source": "PyAV (av.open, frame.pts * time_base)",
        "segments": segments,
    }

    json_path = OUT_DIR / "action_segments_v2_auto.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")

    # CSV
    csv_path = OUT_DIR / "action_segments_v2_auto.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "block_idx", "rep_idx", "sign_id", "sign_kr", "speed",
            "front_start_s", "front_end_s", "duration_s", "peak_energy",
            "left_start_s", "left_end_s", "right_start_s", "right_end_s",
            "status", "review_reasons", "seg_confidence",
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
                s["status"], "; ".join(s.get("review_reasons", [])),
                s.get("seg_confidence"),
            ])
    print(f"  Saved: {csv_path}")

    # ── 10. Visual QA ──
    print("\n  Generating contact sheets (5x5 per block)...")
    for bi in range(len(BLOCKS)):
        make_block_contact_sheet(FRONT_VIDEO, segments, bi, out_contacts)

    print("\n  Generating energy timelines...")
    for bi in range(min(len(BLOCKS), len(valid_blocks))):
        blk_s, blk_e = valid_blocks[bi]
        make_block_timeline(times, energy_short, segments, bi, blk_s, blk_e, out_timelines)

    print("\n  Generating preview videos...")
    for bi in range(min(len(BLOCKS), len(valid_blocks))):
        make_preview_video(FRONT_VIDEO, segments, bi, out_timelines, fps_out=8)

    # ── 11. v1 vs v2 comparison ──
    v1_path = OUT_DIR / "action_segments_v1.json"
    if v1_path.exists():
        v1 = json.load(open(v1_path, encoding="utf-8"))
        v1_segs = {s["sample_id"]: s for s in v1.get("segments", [])}
        print("\n  v1 vs v2 comparison:")
        print(f"    {'sample_id':<45} {'v1_start':>10} {'v1_end':>10} {'v2_start':>10} {'v2_end':>10} {'delta_s':>8}")
        for s in segments:
            sid = s["sample_id"]
            v1s = v1_segs.get(sid, {})
            v1_start = v1s.get("front_start_s")
            v1_end = v1s.get("front_end_s")
            v2_start = s.get("front_start_s")
            v2_end = s.get("front_end_s")

            v1s_str = f"{v1_start:.1f}" if v1_start else "N/A"
            v1e_str = f"{v1_end:.1f}" if v1_end else "N/A"
            v2s_str = f"{v2_start:.1f}" if v2_start else "N/A"
            v2e_str = f"{v2_end:.1f}" if v2_end else "N/A"

            delta = ""
            if v1_start and v2_start:
                d = abs(v2_start - v1_start) + abs(v2_end - v1_end)
                delta = f"{d:.1f}"

            print(f"    {sid:<45} {v1s_str:>10} {v1e_str:>10} {v2s_str:>10} {v2e_str:>10} {delta:>8}")

    print("\n  Step 5 v2 complete.")


FRONT_VIDEO = VIDEOS["FRONT"]

if __name__ == "__main__":
    main()
