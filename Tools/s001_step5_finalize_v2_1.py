"""
Step 5 v2.1: Finalize verified segmentation for S001_P001

- Copies v2 FRONT boundaries exactly (no changes)
- Least-squares affine time transform from start+end clap correspondences
- Recalculates LEFT/RIGHT local PTS
- PyAV PTS-based contact sheets and preview videos
- Adds MANUAL_VERIFIED status
- Saves energy profile
- Renames seg_confidence -> selection_score
"""

import cv2
import numpy as np
import json
import csv
import av
import copy
import datetime
from pathlib import Path
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

BLOCKS = [
    {"block_idx": 0, "sign_id": "GREETING",  "sign_kr": "\uc778\uc0ac",      "speed": "NORMAL"},
    {"block_idx": 1, "sign_id": "GREETING",  "sign_kr": "\uc778\uc0ac",      "speed": "SLOW"},
    {"block_idx": 2, "sign_id": "GREETING",  "sign_kr": "\uc778\uc0ac",      "speed": "FAST"},
    {"block_idx": 3, "sign_id": "THANK_YOU", "sign_kr": "\uac10\uc0ac\ud569\ub2c8\ub2e4", "speed": "NORMAL"},
    {"block_idx": 4, "sign_id": "THANK_YOU", "sign_kr": "\uac10\uc0ac\ud569\ub2c8\ub2e4", "speed": "SLOW"},
    {"block_idx": 5, "sign_id": "THANK_YOU", "sign_kr": "\uac10\uc0ac\ud569\ub2c8\ub2e4", "speed": "FAST"},
    {"block_idx": 6, "sign_id": "LOVE",      "sign_kr": "\uc0ac\ub791",      "speed": "NORMAL"},
    {"block_idx": 7, "sign_id": "LOVE",      "sign_kr": "\uc0ac\ub791",      "speed": "SLOW"},
    {"block_idx": 8, "sign_id": "LOVE",      "sign_kr": "\uc0ac\ub791",      "speed": "FAST"},
]


# ===================================================================
#  1. Least-squares affine time transform from clap correspondences
# ===================================================================
def compute_affine_transform(sync_data):
    """
    Compute camera_pts = a * front_pts + b for each secondary camera
    using start+end clap correspondences via least-squares fit.
    Returns dict: {camera: {a, b, residuals, max_residual_ms, clap_pairs}}
    """
    front_cam = sync_data["cameras"]["FRONT"]
    front_end_claps = front_cam["end_clap_times_s"]
    front_start_claps = front_cam["start_clap_times_s"]

    transforms = {"FRONT": {"a": 1.0, "b": 0.0, "residuals": [], "max_residual_ms": 0.0,
                             "clap_pairs": [], "method": "identity (reference)"}}

    for view in ["LEFT", "RIGHT"]:
        cam = sync_data["cameras"][view]
        cam_end_claps = cam["end_clap_times_s"]
        cam_start_claps = cam.get("matched_start_clap_times_s", cam["start_clap_times_s"])

        # Build correspondence pairs: (front_pts, camera_pts)
        pairs = []
        for ft, ct in zip(front_start_claps, cam_start_claps):
            pairs.append((ft, ct))
        for ft, ct in zip(front_end_claps, cam_end_claps):
            pairs.append((ft, ct))

        front_pts = np.array([p[0] for p in pairs])
        cam_pts = np.array([p[1] for p in pairs])

        # Least-squares fit: camera_pts = a * front_pts + b
        A_matrix = np.column_stack([front_pts, np.ones(len(front_pts))])
        result = np.linalg.lstsq(A_matrix, cam_pts, rcond=None)
        a, b = result[0]

        # Residuals
        predicted = a * front_pts + b
        residuals = cam_pts - predicted
        max_res_ms = float(np.max(np.abs(residuals))) * 1000

        transforms[view] = {
            "a": float(a),
            "b": float(b),
            "residuals": [round(float(r) * 1000, 4) for r in residuals],
            "max_residual_ms": round(max_res_ms, 4),
            "n_clap_pairs": len(pairs),
            "clap_pairs": [{"front_s": round(float(f), 6), "camera_s": round(float(c), 6)}
                           for f, c in pairs],
            "method": "least_squares_linear_fit_from_6_clap_pairs",
        }

    return transforms


def front_to_camera_affine(front_s, transform):
    """camera_pts = a * front_pts + b"""
    return transform["a"] * front_s + transform["b"]


def camera_to_front_affine(cam_s, transform):
    """front_pts = (camera_pts - b) / a"""
    return (cam_s - transform["b"]) / transform["a"]


# ===================================================================
#  2. PyAV PTS-based frame seeking
# ===================================================================
def build_pts_index(video_path, max_frames=None):
    """Build a list of (pts_s, frame_idx) for the video using PyAV."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    tb = float(stream.time_base)

    pts_list = []
    for i, frame in enumerate(container.decode(stream)):
        if max_frames and i >= max_frames:
            break
        pts_s = float(frame.pts) * tb if frame.pts is not None else i / 30.0
        pts_list.append(pts_s)

    container.close()
    return np.array(pts_list)


def find_nearest_frame_idx(pts_array, target_s):
    """Find the frame index with PTS closest to target_s."""
    idx = int(np.argmin(np.abs(pts_array - target_s)))
    return idx, float(pts_array[idx])


def read_frame_by_index_pyav(video_path, frame_idx):
    """Read a specific frame by index using PyAV."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    for i, frame in enumerate(container.decode(stream)):
        if i == frame_idx:
            img = frame.to_ndarray(format="bgr24")
            container.close()
            return img
    container.close()
    return None


def read_frames_by_indices_pyav(video_path, frame_indices):
    """Read multiple frames by index in a single pass (indices must be sorted)."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    sorted_indices = sorted(set(frame_indices))
    idx_set = set(sorted_indices)
    max_idx = sorted_indices[-1] if sorted_indices else -1

    frames = {}
    for i, frame in enumerate(container.decode(stream)):
        if i > max_idx:
            break
        if i in idx_set:
            frames[i] = frame.to_ndarray(format="bgr24")

    container.close()
    return frames


# ===================================================================
#  3. Energy extraction (reused from v2 with improvements)
# ===================================================================
def extract_energy_pyav(video_path, view, transform, sample_interval=2, scale=0.25):
    """Extract frame-differencing energy, returning per-camera AND front-aligned times."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    tb = float(stream.time_base)

    cam_energy = []  # (cam_pts_s, energy)
    front_energy = []  # (front_pts_s, energy)
    prev_gray = None
    count = 0
    skip = 0

    for frame in container.decode(stream):
        if skip < sample_interval - 1:
            skip += 1
            continue
        skip = 0

        cam_time_s = float(frame.pts) * tb if frame.pts is not None else count * (1.0 / 30.0)
        front_time_s = camera_to_front_affine(cam_time_s, transform)

        img = frame.to_ndarray(format="gray")
        h, w = img.shape
        sh, sw = int(h * scale), int(w * scale)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)

        y1, y2 = int(sh * 0.20), int(sh * 0.85)
        x1, x2 = int(sw * 0.20), int(sw * 0.80)
        roi = small[y1:y2, x1:x2]

        if prev_gray is not None:
            diff = cv2.absdiff(roi, prev_gray)
            energy = float(np.mean(diff))
            cam_energy.append((cam_time_s, energy))
            front_energy.append((front_time_s, energy))

        prev_gray = roi.copy()
        count += 1
        if count % 2000 == 0:
            print(f"    {view}: {count} frames, front_t={front_time_s:.1f}s")

    container.close()
    print(f"    {view}: {len(front_energy)} samples, "
          f"front_t=[{front_energy[0][0]:.1f}, {front_energy[-1][0]:.1f}]s")
    return cam_energy, front_energy


def fuse_multi_camera_energy(all_front_energies, bin_size_s=0.05):
    """
    Fuse per-camera energy onto common front-time grid.
    Uses NaN for missing bins instead of 0.
    Per-camera energy is individually normalized by robust median before averaging.
    """
    all_times = []
    for elist in all_front_energies.values():
        all_times.extend([t for t, e in elist])

    t_min = min(all_times)
    t_max = max(all_times)
    n_bins = int((t_max - t_min) / bin_size_s) + 1
    times = np.array([t_min + i * bin_size_s for i in range(n_bins)])

    # Per-camera binning with NaN for missing
    per_cam_binned = {}
    for view, elist in all_front_energies.items():
        cam_bins = np.full(n_bins, np.nan)
        cam_counts = np.zeros(n_bins)
        for t, e in elist:
            bi = int((t - t_min) / bin_size_s)
            if 0 <= bi < n_bins:
                if np.isnan(cam_bins[bi]):
                    cam_bins[bi] = e
                    cam_counts[bi] = 1
                else:
                    cam_bins[bi] += e
                    cam_counts[bi] += 1
        valid = cam_counts > 0
        cam_bins[valid] /= cam_counts[valid]

        # Robust normalization: divide by p75 to equalize cameras
        p75 = np.nanpercentile(cam_bins, 75)
        if p75 > 0:
            cam_bins /= p75
        per_cam_binned[view] = cam_bins

    # Fuse: nanmean across cameras (NaN remains where ALL cameras missing)
    stacked = np.array(list(per_cam_binned.values()))
    fused = np.nanmean(stacked, axis=0)
    all_missing_mask = np.isnan(fused)

    return times, fused, per_cam_binned, all_missing_mask


# ===================================================================
#  4. Contact sheets with PyAV PTS
# ===================================================================
def make_contact_sheet_pyav(video_path, pts_array, segments, block_idx, out_dir, thumb_w=256):
    """5 reps x 5 timepoints contact sheet using PyAV PTS-based seeking."""
    block_segs = [s for s in segments if s["block_idx"] == block_idx and s["front_start_s"] is not None]
    if not block_segs:
        return None

    # Collect all needed frame indices
    frame_requests = []  # (row, col, target_s, frame_idx, actual_pts_s)
    for row, seg in enumerate(block_segs):
        s, e = seg["front_start_s"], seg["front_end_s"]
        timepoints = [s, s + (e - s) * 0.25, s + (e - s) * 0.5, s + (e - s) * 0.75, e]
        for col, tp in enumerate(timepoints):
            fi, actual_pts = find_nearest_frame_idx(pts_array, tp)
            frame_requests.append((row, col, tp, fi, actual_pts))

    indices_needed = sorted(set(r[3] for r in frame_requests))
    frames_map = read_frames_by_indices_pyav(video_path, indices_needed)

    # Get dimensions from first frame
    sample_frame = next(iter(frames_map.values()))
    h_orig, w_orig = sample_frame.shape[:2]
    thumb_h = int(thumb_w * h_orig / w_orig)

    n_rows = len(block_segs)
    n_cols = 5
    label_h = 25
    sheet = np.zeros((n_rows * (thumb_h + label_h), n_cols * thumb_w, 3), dtype=np.uint8)
    timepoint_labels = ["Start", "25%", "50%", "75%", "End"]
    max_pts_error = 0.0

    for row, col, target_s, fi, actual_pts in frame_requests:
        pts_err = abs(actual_pts - target_s)
        max_pts_error = max(max_pts_error, pts_err)

        frame_img = frames_map.get(fi)
        if frame_img is None:
            continue

        thumb = cv2.resize(frame_img, (thumb_w, thumb_h))
        y = row * (thumb_h + label_h)
        x = col * thumb_w
        sheet[y:y + thumb_h, x:x + thumb_w] = thumb

        if row == 0:
            cv2.putText(sheet, timepoint_labels[col], (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        if col == 0:
            seg = block_segs[row]
            label = f"{seg['sample_id'].split('_')[-1]} {seg['duration_s']:.1f}s"
            cv2.putText(sheet, label, (x + 5, y + thumb_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    bi = BLOCKS[block_idx]
    fname = f"contact_block{block_idx}_{bi['sign_id']}_{bi['speed']}.jpg"
    cv2.imwrite(str(out_dir / fname), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"    Saved: {fname} (max PTS error: {max_pts_error*1000:.1f}ms)")
    return max_pts_error


# ===================================================================
#  5. Preview videos with PyAV PTS and overlay
# ===================================================================
def make_preview_video_pyav(video_path, pts_array, segments, block_idx, out_dir, fps_out=10, scale=0.5):
    """Preview video with PTS overlay: requested time, actual PTS, delta."""
    block_segs = [s for s in segments if s["block_idx"] == block_idx and s["front_start_s"] is not None]
    if not block_segs:
        return None

    blk_start = min(s["front_start_s"] for s in block_segs) - 2.0
    blk_end = max(s["front_end_s"] for s in block_segs) + 2.0

    # Find frame index range
    start_idx, _ = find_nearest_frame_idx(pts_array, blk_start)
    end_idx, _ = find_nearest_frame_idx(pts_array, blk_end)

    # Sample frames at fps_out rate
    dt = 1.0 / fps_out
    sample_times = np.arange(blk_start, blk_end, dt)
    frame_indices_pts = []
    for st in sample_times:
        fi, actual = find_nearest_frame_idx(pts_array, st)
        frame_indices_pts.append((fi, st, actual))

    indices_needed = sorted(set(r[0] for r in frame_indices_pts))
    frames_map = read_frames_by_indices_pyav(video_path, indices_needed)

    if not frames_map:
        return None

    sample_frame = next(iter(frames_map.values()))
    h_orig, w_orig = sample_frame.shape[:2]
    w = int(w_orig * scale)
    h = int(h_orig * scale)

    bi = BLOCKS[block_idx]
    fname = f"preview_block{block_idx}_{bi['sign_id']}_{bi['speed']}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_dir / fname), fourcc, fps_out, (w, h))
    max_pts_error = 0.0

    for fi, req_s, actual_s in frame_indices_pts:
        frame_img = frames_map.get(fi)
        if frame_img is None:
            continue

        small = cv2.resize(frame_img, (w, h))
        pts_err = abs(actual_s - req_s)
        max_pts_error = max(max_pts_error, pts_err)

        # Determine segment label
        label = ""
        for seg in block_segs:
            if seg["front_start_s"] <= actual_s <= seg["front_end_s"]:
                label = seg["sample_id"].split("_", 3)[-1]
                cv2.rectangle(small, (0, 0), (w, 40), (0, 100, 0), -1)
                break
        if not label:
            cv2.rectangle(small, (0, 0), (w, 40), (80, 80, 80), -1)
            label = "NEUTRAL"

        # Overlay with PTS info
        line1 = f"{label}  req={req_s:.2f}s  pts={actual_s:.2f}s  d={pts_err*1000:.0f}ms"
        cv2.putText(small, line1, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        writer.write(small)

    writer.release()
    print(f"    Saved: {fname} (max PTS error: {max_pts_error*1000:.1f}ms)")
    return max_pts_error


# ===================================================================
#  6. Timeline plots (reused from v2, no changes needed)
# ===================================================================
def make_block_timeline(times, energy, segments, block_idx, blk_start, blk_end, out_dir):
    pad = 3.0
    mask = (times >= blk_start - pad) & (times <= blk_end + pad)
    bt = times[mask]
    be = energy[mask]

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


# ===================================================================
#  Main
# ===================================================================
def main():
    print("Step 5 v2.1: Finalize verified segmentation -- S001_P001\n")

    # ── Load v2 auto segments ──
    v2_path = OUT_DIR / "action_segments_v2_auto.json"
    v2_data = json.load(open(v2_path, encoding="utf-8"))
    v2_segments = v2_data["segments"]
    print(f"  Loaded {len(v2_segments)} segments from v2_auto")

    # ── Load sync data ──
    sync_data = json.load(open(OUT_DIR / "sync_report.json", encoding="utf-8"))

    # ── 1. Compute affine transforms ──
    print("\n  Computing affine time transforms from clap correspondences...")
    transforms = compute_affine_transform(sync_data)

    for view in ["LEFT", "RIGHT"]:
        t = transforms[view]
        old_offset = sync_data["cameras"][view]["offset_s_final"]
        old_drift_ppm = sync_data["cameras"][view].get("drift_ppm", 0) or 0
        old_a = 1.0 + old_drift_ppm * 1e-6
        old_b = old_offset
        print(f"  {view}:")
        print(f"    Old model: cam = {old_a:.10f} * front + {old_b:.6f}")
        print(f"    New model: cam = {t['a']:.10f} * front + {t['b']:.6f}")
        print(f"    Delta a: {(t['a'] - old_a):.2e}")
        print(f"    Delta b: {(t['b'] - old_b)*1000:.3f} ms")
        print(f"    Fit residuals (ms): {t['residuals']}")
        print(f"    Max residual: {t['max_residual_ms']:.3f} ms")

    # ── 2. Build new segments with corrected camera ranges ──
    print("\n  Building v2.1 segments with corrected camera ranges...")
    segments = []
    max_left_change_ms = 0.0
    max_right_change_ms = 0.0

    for v2seg in v2_segments:
        seg = copy.deepcopy(v2seg)

        # Preserve auto status/reasons
        seg["auto_status"] = seg.pop("status")
        seg["auto_review_reasons"] = seg.pop("review_reasons")

        # Add verification
        seg["verification_status"] = "MANUAL_VERIFIED"
        seg["verification_basis"] = "preview_video_and_energy_timeline"
        seg["verification_date"] = "2026-08-05"

        # Rename seg_confidence -> selection_score
        if "seg_confidence" in seg:
            seg["selection_score"] = seg.pop("seg_confidence")

        # Recalculate camera ranges with affine model
        if seg["front_start_s"] is not None:
            fs = seg["front_start_s"]
            fe = seg["front_end_s"]

            new_ranges = {}
            for view in ["FRONT", "LEFT", "RIGHT"]:
                new_start = front_to_camera_affine(fs, transforms[view])
                new_end = front_to_camera_affine(fe, transforms[view])
                new_ranges[view] = {
                    "start_pts_s": round(new_start, 6),
                    "end_pts_s": round(new_end, 6),
                }

            # Track max change
            old_ranges = seg.get("camera_ranges", {})
            for view in ["LEFT", "RIGHT"]:
                if view in old_ranges and view in new_ranges:
                    old_s = old_ranges[view]["start_pts_s"]
                    new_s = new_ranges[view]["start_pts_s"]
                    old_e = old_ranges[view]["end_pts_s"]
                    new_e = new_ranges[view]["end_pts_s"]
                    change_ms = max(abs(new_s - old_s), abs(new_e - old_e)) * 1000
                    if view == "LEFT":
                        max_left_change_ms = max(max_left_change_ms, change_ms)
                    else:
                        max_right_change_ms = max(max_right_change_ms, change_ms)

            seg["camera_ranges"] = new_ranges

        segments.append(seg)

    print(f"  Max LEFT change: {max_left_change_ms:.3f} ms")
    print(f"  Max RIGHT change: {max_right_change_ms:.3f} ms")

    # ── 3. Fix action_window to include first segment ──
    first_start = min(s["front_start_s"] for s in segments if s["front_start_s"] is not None)
    last_end = max(s["front_end_s"] for s in segments if s["front_end_s"] is not None)
    action_window = {
        "start_s": round(first_start, 4),
        "end_s": round(v2_data["action_window"]["end_s"], 4),
        "note": "start_s adjusted to include first verified segment (was 178.51, first seg starts 178.4614)"
    }
    print(f"  Action window: {action_window['start_s']}s - {action_window['end_s']}s")

    # ── 4. Verify 45 IDs, no overlaps, correct order ──
    print("\n  Verification checks...")
    seg_ids = [s["sample_id"] for s in segments]
    all_present = len(set(seg_ids)) == 45
    print(f"    45 unique IDs: {all_present}")

    overlap_count = 0
    order_violation = 0
    for i in range(len(segments) - 1):
        s1, s2 = segments[i], segments[i + 1]
        if s1["front_start_s"] and s2["front_start_s"]:
            if s1["front_start_s"] > s2["front_start_s"]:
                order_violation += 1
            overlap = max(0, min(s1["front_end_s"], s2["front_end_s"]) - max(s1["front_start_s"], s2["front_start_s"]))
            if overlap > 0:
                overlap_count += 1
    print(f"    Overlaps: {overlap_count}")
    print(f"    Order violations: {order_violation}")
    print(f"    All MANUAL_VERIFIED: {all(s['verification_status'] == 'MANUAL_VERIFIED' for s in segments)}")

    # ── 5. Extract energy (for energy_profile) ──
    print("\n  Extracting energy from all cameras...")
    all_cam_energies = {}
    all_front_energies = {}
    for view in ["FRONT", "LEFT", "RIGHT"]:
        print(f"  {view}:")
        cam_e, front_e = extract_energy_pyav(VIDEOS[view], view, transforms[view])
        all_cam_energies[view] = cam_e
        all_front_energies[view] = front_e

    print("\n  Fusing energy (with robust normalization and NaN handling)...")
    times, fused, per_cam_binned, all_missing_mask = fuse_multi_camera_energy(all_front_energies, bin_size_s=0.05)
    dt = float(np.median(np.diff(times)))

    # Count missing bins
    n_missing_total = int(np.sum(all_missing_mask))
    aw_start = action_window["start_s"]
    aw_end = action_window["end_s"]
    aw_mask = (times >= aw_start) & (times <= aw_end)
    n_missing_in_aw = int(np.sum(all_missing_mask & aw_mask))
    missing_indices = [int(i) for i in np.where(all_missing_mask)[0]]
    print(f"  Fused: {len(times)} bins, dt={dt*1000:.1f}ms")
    print(f"  All-camera-missing bins: {n_missing_total} total, {n_missing_in_aw} in action window")

    # Interpolate NaN bins for timeline display only
    fused_for_display = fused.copy()
    valid_bins = ~all_missing_mask
    if np.any(valid_bins) and np.any(all_missing_mask):
        fused_for_display[all_missing_mask] = np.interp(
            times[all_missing_mask], times[valid_bins], fused[valid_bins]
        )

    # Neutral level computed from valid bins only
    valid_fused = fused[valid_bins]
    neutral_level = round(float(np.percentile(valid_fused, 10)), 4)

    # ── 6. Posthoc diagnostic: re-derive per-block selection params ──
    # These are NOT the original v2 params (which were not logged at generation time).
    # This is a diagnostic reanalysis using v2.1 corrected energy and verified block ranges.
    print("\n  Posthoc diagnostic: re-deriving per-block selection params...")
    from scipy.signal import find_peaks

    def find_motion_bursts(t, e, threshold):
        is_high = e > threshold
        bursts = []
        in_burst = False
        start_i = 0
        for i in range(len(is_high)):
            if is_high[i] and not in_burst:
                start_i = i
                in_burst = True
            elif not is_high[i] and in_burst:
                peak = float(np.max(e[start_i:i]))
                bursts.append((t[start_i], t[i - 1], peak))
                in_burst = False
        if in_burst:
            peak = float(np.max(e[start_i:]))
            bursts.append((t[start_i], t[-1], peak))
        return bursts

    def merge_close_bursts(bursts, min_gap_s):
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

    # Use interpolated energy for segmentation (same as v2 used 0-filled)
    act_mask_diag = (times >= aw_start) & (times <= aw_end)
    act_t = times[act_mask_diag]

    per_block_params = []

    for bi in range(9):
        block_info = BLOCKS[bi]
        speed = block_info["speed"]
        bsegs = [s for s in segments if s["block_idx"] == bi and s["front_start_s"] is not None]
        if not bsegs:
            per_block_params.append({"block_idx": bi, "detected": False})
            continue

        blk_s = min(s["front_start_s"] for s in bsegs)
        blk_e = max(s["front_end_s"] for s in bsegs)
        block_dur = blk_e - blk_s

        bmask = (times >= blk_s) & (times <= blk_e)
        bt = times[bmask]
        be_raw = fused_for_display[bmask]

        if len(bt) < 10:
            per_block_params.append({"block_idx": bi, "detected": False})
            continue

        # Adaptive smoothing per speed
        if speed == "FAST":
            smooth_wins = [0.1, 0.15, 0.2, 0.4, 0.8, 1.0]
        elif speed == "SLOW":
            smooth_wins = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
        else:
            smooth_wins = [0.15, 0.25, 0.4, 0.8, 1.0, 1.5, 2.0]
        if block_dur > 30:
            smooth_wins = sorted(set(smooth_wins + [1.5, 2.0, 2.5, 3.0, 4.0]), reverse=True)

        best_info = {"method": "not_found", "smooth_win_s": None, "thresh_pct": None, "score": None}
        best_reps = None
        best_score = -1

        # Try burst detection
        for sw in smooth_wins:
            win = max(1, int(sw / dt))
            be_smooth = uniform_filter1d(be_raw, size=win, mode="nearest")
            for thresh_pct in [20, 30, 40]:
                thresh = np.percentile(be_smooth, thresh_pct)
                if thresh < neutral_level * 1.2:
                    thresh = neutral_level * 1.2
                bursts = find_motion_bursts(bt, be_smooth, thresh)
                min_gap = 0.15 if speed == "FAST" else 0.3
                bursts = merge_close_bursts(bursts, min_gap)
                min_dur = 0.2 if speed == "FAST" else 0.4
                bursts = [(s, e, p) for s, e, p in bursts if e - s >= min_dur]
                if len(bursts) == 5:
                    durations = [e - s for s, e, p in bursts]
                    cv_val = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 999
                    score = 10.0 - cv_val
                    if score > best_score:
                        best_score = score
                        best_reps = bursts
                        best_info = {
                            "method": "5_burst_detection",
                            "smooth_win_s": sw,
                            "thresh_pct": thresh_pct,
                            "threshold_value": round(float(thresh), 4),
                            "score": round(float(score), 3),
                            "cv": round(float(cv_val), 4),
                        }

        # Try valley detection if no burst match
        if best_reps is None:
            for sw in smooth_wins:
                win = max(1, int(sw / dt))
                be_smooth = uniform_filter1d(be_raw, size=win, mode="nearest")
                inverted = -be_smooth
                min_dist = max(3, int(0.5 / dt))
                for prom_f in np.arange(0.03, 0.5, 0.03):
                    erange = np.max(be_smooth) - np.min(be_smooth)
                    if erange <= 0:
                        continue
                    min_prom = erange * prom_f
                    valleys, _ = find_peaks(inverted, prominence=min_prom, distance=min_dist)
                    if len(valleys) == 4:
                        splits = [0] + valleys.tolist() + [len(bt) - 1]
                        reps = []
                        for i in range(5):
                            s_idx = splits[i]
                            e_idx = splits[i + 1]
                            if e_idx > s_idx:
                                peak = float(np.max(be_smooth[s_idx:e_idx + 1]))
                                reps.append((bt[s_idx], bt[e_idx], peak))
                        valid_reps = [r for r in reps if r is not None]
                        if len(valid_reps) == 5:
                            durations = [e - s for s, e, p in valid_reps]
                            cv_val = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 999
                            score = 5.0 - cv_val
                            if score > best_score:
                                best_score = score
                                best_reps = valid_reps
                                best_info = {
                                    "method": "4_valley_detection",
                                    "smooth_win_s": sw,
                                    "prominence_factor": round(float(prom_f), 3),
                                    "score": round(float(score), 3),
                                    "cv": round(float(cv_val), 4),
                                }
                        break

        per_block_params.append({
            "block_idx": bi,
            "sign_id": block_info["sign_id"],
            "speed": speed,
            "block_range_s": [round(blk_s, 4), round(blk_e, 4)],
            "block_duration_s": round(block_dur, 3),
            "detected": best_reps is not None,
            "selection_params": best_info,
        })

    # Assert FRONT boundaries unchanged
    print("\n  Asserting FRONT boundaries match v2...")
    for v2seg, seg in zip(v2_segments, segments):
        assert v2seg["front_start_s"] == seg["front_start_s"], \
            f"FRONT start mismatch: {v2seg['sample_id']} v2={v2seg['front_start_s']} v2.1={seg['front_start_s']}"
        assert v2seg["front_end_s"] == seg["front_end_s"], \
            f"FRONT end mismatch: {v2seg['sample_id']} v2={v2seg['front_end_s']} v2.1={seg['front_end_s']}"
    print("  PASS: 45/45 FRONT boundaries identical to v2")

    # ── 7. Save energy_profile_v2_1.json ──
    energy_profile = {
        "version": "v2.1",
        "session_id": "S001",
        "participant_id": "P001",
        "bin_size_s": 0.05,
        "time_grid_start_s": round(float(times[0]), 4),
        "time_grid_end_s": round(float(times[-1]), 4),
        "n_bins": len(times),
        "neutral_energy_level": neutral_level,
        "all_cameras_missing": {
            "total_count": n_missing_total,
            "in_action_window_count": n_missing_in_aw,
            "bin_indices": missing_indices,
            "note": "Bins where all 3 cameras had no energy sample. "
                    "Stored as null in fused_energy. Not energy=0, but truly missing data.",
        },
        "per_camera_energy": {},
        "fused_energy": [round(float(v), 6) if not np.isnan(v) else None for v in fused],
        "fusion_method": "per_camera_robust_p75_normalization_then_nanmean",
        "missing_bin_handling": "NaN_preserved_as_null_in_json",
        "timeline_display_method": "NaN_bins_interpolated_from_adjacent_valid_values",
        "historical_v2_block_detection_summary": {
            "source": "preserved_from_v2_segmentation_run_and_code",
            "note": "These values describe the original v2 block detection, not re-derived from v2.1 energy.",
            "method": "heavy_smoothing_rep_level_bursts_top8_gaps",
            "smoothing_s": 1.5,
            "target_bursts": 45,
            "initial_burst_counts_per_block": [5, 5, 5, 5, 6, 5, 5, 4, 5],
            "boundary_selection": "top_8_inter_burst_gaps_from_45_rep_level_bursts",
        },
        "original_v2_per_block_params_logged": False,
        "original_v2_params_recovery_status": "not_claimed",
        "posthoc_v2_1_diagnostic_params": {
            "used_to_generate_verified_boundaries": False,
            "purpose": "diagnostic_reanalysis_after_boundary_verification",
            "energy_source": "corrected_affine_p75_normalized_nan_interpolated_v2_1_energy",
            "block_range_source": "first_verified_onset_to_last_verified_offset",
            "per_block": per_block_params,
        },
        "affine_transforms": transforms,
    }

    for view in per_cam_binned:
        vals = per_cam_binned[view]
        energy_profile["per_camera_energy"][view] = {
            "n_bins": len(vals),
            "energy": [round(float(v), 6) if not np.isnan(v) else None for v in vals],
        }

    ep_path = OUT_DIR / "energy_profile_v2_1.json"
    with open(ep_path, "w", encoding="utf-8") as f:
        json.dump(energy_profile, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {ep_path.name}")

    # ── 8. Generate energy timelines (using interpolated display energy) ──
    out_timelines = OUT_DIR / "segmentation_timelines_v2_1"
    out_timelines.mkdir(exist_ok=True)

    smooth_win = max(1, int(0.15 / dt))
    energy_display = uniform_filter1d(fused_for_display, size=smooth_win, mode="nearest")

    print("\n  Generating energy timelines...")
    block_ranges = []
    for bi in range(9):
        bsegs = [s for s in segments if s["block_idx"] == bi and s["front_start_s"] is not None]
        if bsegs:
            bs = min(s["front_start_s"] for s in bsegs)
            be = max(s["front_end_s"] for s in bsegs)
            block_ranges.append((bs, be))
        else:
            block_ranges.append((0, 0))

    for bi in range(9):
        bs, be = block_ranges[bi]
        make_block_timeline(times, energy_display, segments, bi, bs, be, out_timelines)

    # ── 9. Skip contact sheets and preview videos (reuse v2.1 outputs) ──
    print("\n  Skipping contact sheet and preview regeneration (reusing v2.1 PTS-based outputs)")
    # Load existing max errors from previous run
    prev_v21 = OUT_DIR / "action_segments_v2_1_verified.json"
    if prev_v21.exists():
        prev_data = json.load(open(prev_v21, encoding="utf-8"))
        global_max_contact_err = prev_data.get("contact_sheet_max_pts_error_ms", 0) / 1000.0
        global_max_preview_err = prev_data.get("preview_video_max_pts_error_ms", 0) / 1000.0
    else:
        global_max_contact_err = 0.0
        global_max_preview_err = 0.0

    # ── 10. Build and save final JSON ──
    output_data = {
        "version": "v2.1_verified",
        "session_id": "S001",
        "participant_id": "P001",
        "finalization_date": "2026-08-05",
        "n_blocks": 9,
        "n_segments_expected": 45,
        "n_manual_verified": sum(1 for s in segments if s["verification_status"] == "MANUAL_VERIFIED"),
        "method": "protocol_constrained_energy_segmentation_with_manual_visual_verification",
        "method_notes": "Block structure (9 blocks x 5 reps) from recording protocol. "
                        "Multi-camera frame-differencing energy fused on common FRONT time grid. "
                        "Heavy smoothing (1.5s) to find 45 rep-level bursts, top-8 gaps for 9 blocks. "
                        "Within-block segmentation with adaptive smoothing. "
                        "All 45 segments manually verified via preview video and energy timeline.",
        "protocol_constraint": "5_reps_per_block",
        "energy_bin_size_s": 0.05,
        "neutral_energy_level": neutral_level,
        "action_window": action_window,
        "front_vfr": True,
        "pts_source": "PyAV (av.open, frame.pts * time_base)",
        "affine_transforms": transforms,
        "contact_sheet_pts_method": "PyAV_PTS_nearest_frame",
        "contact_sheet_max_pts_error_ms": round(global_max_contact_err * 1000, 2),
        "preview_video_pts_method": "PyAV_PTS_nearest_frame",
        "preview_video_max_pts_error_ms": round(global_max_preview_err * 1000, 2),
        "qa_notes": {
            "v2_contact_sheets_used_cv2_random_seeking": True,
            "v2_contact_sheets_vfr_accuracy": "Unreliable due to OpenCV CAP_PROP_POS_MSEC seeking on VFR FRONT video. "
                                               "v2.1 contact sheets use PyAV PTS-based nearest-frame selection.",
            "fused_energy_missing_bins": f"{n_missing_total} total ({n_missing_in_aw} in action window) stored as null, "
                                          "not 0. Timeline display uses time-axis interpolation for these bins.",
        },
        "segments": segments,
    }

    json_path = OUT_DIR / "action_segments_v2_1_verified.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path.name}")

    # ── 12. Save CSV ──
    csv_path = OUT_DIR / "action_segments_v2_1_verified.csv"
    fieldnames = [
        "sample_id", "block_idx", "rep_idx", "sign_id", "sign_kr", "speed",
        "front_start_s", "front_end_s", "duration_s",
        "left_start_pts_s", "left_end_pts_s", "right_start_pts_s", "right_end_pts_s",
        "peak_energy", "selection_score",
        "auto_status", "verification_status",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seg in segments:
            row = {k: seg.get(k) for k in ["sample_id", "block_idx", "rep_idx", "sign_id",
                                             "sign_kr", "speed", "front_start_s", "front_end_s",
                                             "duration_s", "peak_energy", "selection_score",
                                             "auto_status", "verification_status"]}
            cr = seg.get("camera_ranges", {})
            row["left_start_pts_s"] = cr.get("LEFT", {}).get("start_pts_s")
            row["left_end_pts_s"] = cr.get("LEFT", {}).get("end_pts_s")
            row["right_start_pts_s"] = cr.get("RIGHT", {}).get("start_pts_s")
            row["right_end_pts_s"] = cr.get("RIGHT", {}).get("end_pts_s")
            writer.writerow(row)
    print(f"  Saved: {csv_path.name}")

    # ── Final report ──
    print(f"""
  ═══════════════════════════════════════════════════════════
  Step 5 v2.1 Finalization Report (QA metadata fix)
  ═══════════════════════════════════════════════════════════
  FRONT 45 boundaries: IDENTICAL to v2_auto (assertion PASS)
  LEFT  affine: cam = {transforms['LEFT']['a']:.10f} * front + {transforms['LEFT']['b']:.6f}
        method: least_squares_linear_fit_from_6_clap_pairs
        max residual: {transforms['LEFT']['max_residual_ms']:.3f} ms
        max change vs v2: {max_left_change_ms:.3f} ms
  RIGHT affine: cam = {transforms['RIGHT']['a']:.10f} * front + {transforms['RIGHT']['b']:.6f}
        method: least_squares_linear_fit_from_6_clap_pairs
        max residual: {transforms['RIGHT']['max_residual_ms']:.3f} ms
        max change vs v2: {max_right_change_ms:.3f} ms
  All 45 MANUAL_VERIFIED: {all(s['verification_status'] == 'MANUAL_VERIFIED' for s in segments)}
  All-camera-missing bins: {n_missing_total} total, {n_missing_in_aw} in action window
    (stored as null in JSON, interpolated for timeline display only)
  Contact sheet max PTS error: {global_max_contact_err*1000:.2f} ms (reused from prev run)
  Preview video max PTS error: {global_max_preview_err*1000:.2f} ms (reused from prev run)
  Overlaps: {overlap_count}
  Order violations: {order_violation}
  Missing IDs: {45 - len(set(seg_ids))}
  Per-block selection params: {sum(1 for p in per_block_params if p['detected'])}/9 blocks with params

  Updated files:
    {json_path.name}
    {csv_path.name}
    {ep_path.name}
    segmentation_timelines_v2_1/  (9 timelines regenerated)
  Unchanged:
    contact_sheets_v2_1_pts/  (9 sheets, not regenerated)
  """)


if __name__ == "__main__":
    main()
