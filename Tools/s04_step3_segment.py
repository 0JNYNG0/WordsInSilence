"""
Tools/s04_step3_segment.py -- Phase 1 v3a Step 3
Motion-based repetition segmentation + review artifacts.

Reads:
  s04_out_v3a/landmarks_raw_front.json
  s04_out_v3a/sync_candidates.json
  s04_out_v3a/manifest.json

Outputs to s04_out_v3a/:
  segmentation_candidates.json
  segmentation_review.mp4
  boundaries_override_template.csv
  rep_contact_*.png  (per-rep contact sheets)
"""

import csv
import cv2
import json
import math
import os
import sys
import time

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# ---- paths -----------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "s04_out_v3a")

# ---- pose landmark indices -------------------------------------------------

NOSE            = 0
LEFT_SHOULDER   = 11
RIGHT_SHOULDER  = 12
LEFT_ELBOW      = 13
RIGHT_ELBOW     = 14
LEFT_WRIST      = 15
RIGHT_WRIST     = 16
LEFT_HIP        = 23
RIGHT_HIP       = 24

# ---- load helpers ----------------------------------------------------------

def load_json(name):
    p = os.path.join(OUT_DIR, name)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def save_json(data, name):
    p = os.path.join(OUT_DIR, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    print(f"  saved: {name} ({os.path.getsize(p)//1024}KB)", file=sys.stderr)

# ---- 1. motion signal computation -----------------------------------------

def compute_motion_signals(frames):
    """Compute per-frame motion signals from front view pose+hand landmarks.

    Returns dict of 1-D numpy arrays, length = len(frames).
    All values in normalised image coordinates.
    """
    n = len(frames)

    # raw per-frame values (NaN where missing)
    nose_y      = np.full(n, np.nan)
    wrist_l_x   = np.full(n, np.nan)
    wrist_l_y   = np.full(n, np.nan)
    wrist_r_x   = np.full(n, np.nan)
    wrist_r_y   = np.full(n, np.nan)
    shoulder_cx = np.full(n, np.nan)
    shoulder_cy = np.full(n, np.nan)
    shoulder_w  = np.full(n, np.nan)
    hand_dist   = np.full(n, np.nan)  # inter-wrist distance (2-hand frames)

    for i, f in enumerate(frames):
        pn = f.get("pose_norm")
        if pn and len(pn) >= 25:
            nose_y[i]    = pn[NOSE][1]
            wrist_l_x[i] = pn[LEFT_WRIST][0]
            wrist_l_y[i] = pn[LEFT_WRIST][1]
            wrist_r_x[i] = pn[RIGHT_WRIST][0]
            wrist_r_y[i] = pn[RIGHT_WRIST][1]
            lsh = pn[LEFT_SHOULDER]
            rsh = pn[RIGHT_SHOULDER]
            shoulder_cx[i] = (lsh[0] + rsh[0]) * 0.5
            shoulder_cy[i] = (lsh[1] + rsh[1]) * 0.5
            sw = math.sqrt((lsh[0]-rsh[0])**2 + (lsh[1]-rsh[1])**2)
            shoulder_w[i]  = sw

        hs = f.get("hands")
        if hs and len(hs) >= 2:
            w0 = hs[0]["norm"][0]
            w1 = hs[1]["norm"][0]
            hand_dist[i] = math.sqrt((w0[0]-w1[0])**2 + (w0[1]-w1[1])**2)

    # ---- derived signals ----

    # body-relative wrist positions (normalised by shoulder width)
    with np.errstate(divide="ignore", invalid="ignore"):
        lwr_brel_x = (wrist_l_x - shoulder_cx) / shoulder_w
        lwr_brel_y = (wrist_l_y - shoulder_cy) / shoulder_w
        rwr_brel_x = (wrist_r_x - shoulder_cx) / shoulder_w
        rwr_brel_y = (wrist_r_y - shoulder_cy) / shoulder_w

    # velocity (frame-to-frame displacement magnitude)
    def velocity(x, y):
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        return np.sqrt(dx**2 + dy**2)

    vel_wrist_l = velocity(lwr_brel_x, lwr_brel_y)
    vel_wrist_r = velocity(rwr_brel_x, rwr_brel_y)
    vel_nose    = velocity(np.zeros(n), nose_y)  # vertical only

    # combined motion magnitude: max of left/right wrist velocity
    motion_mag = np.fmax(vel_wrist_l, vel_wrist_r)
    # NaN-fill: carry-forward
    for i in range(1, n):
        if np.isnan(motion_mag[i]):
            motion_mag[i] = motion_mag[i-1] if not np.isnan(motion_mag[i-1]) else 0.0
    motion_mag = np.nan_to_num(motion_mag, nan=0.0)

    return {
        "nose_y": nose_y,
        "hand_dist": hand_dist,
        "vel_wrist_l": vel_wrist_l,
        "vel_wrist_r": vel_wrist_r,
        "vel_nose": vel_nose,
        "motion_mag": motion_mag,
        "lwr_brel_y": lwr_brel_y,
        "rwr_brel_y": rwr_brel_y,
    }


# ---- 2. repetition boundary detection -------------------------------------

def detect_repetitions(frames, signals, clap_fi, clap_pts_ms):
    """Detect repetition boundaries from motion envelope.

    Uses smoothed motion magnitude to find activity bursts separated by
    rest periods (low motion). Does NOT force a specific count.

    Returns list of rep dicts with PTS-based boundaries.
    """
    motion = signals["motion_mag"]
    n = len(motion)

    # smooth the motion envelope
    sm_short = uniform_filter1d(motion, size=9)   # fine detail
    sm_long  = uniform_filter1d(motion, size=31)  # envelope

    # post-clap only
    post_start = clap_fi + 1
    if post_start >= n:
        return [], sm_short, sm_long

    post_motion = sm_long[post_start:]

    # adaptive threshold: median + 0.5 * std of post-clap motion
    valid = post_motion[~np.isnan(post_motion)]
    if len(valid) < 10:
        return [], sm_short, sm_long
    thresh = float(np.median(valid) + 0.3 * np.std(valid))

    # find active regions (above threshold) separated by rest
    active = post_motion > thresh
    # pad with False to detect edges
    padded = np.concatenate([[False], active, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]   # rising edges
    ends   = np.where(diff == -1)[0]  # falling edges

    # merge bursts that are close together (< 15 frames gap = 0.5s)
    MIN_GAP = 15
    MIN_DURATION = 30  # minimum 1s at 30fps

    merged_starts = []
    merged_ends = []
    for s, e in zip(starts, ends):
        if merged_starts and (s - merged_ends[-1]) < MIN_GAP:
            merged_ends[-1] = e  # extend previous
        else:
            merged_starts.append(s)
            merged_ends.append(e)

    # filter by minimum duration
    reps = []
    for s, e in zip(merged_starts, merged_ends):
        if (e - s) < MIN_DURATION:
            continue
        abs_s = s + post_start
        abs_e = min(e + post_start, n - 1)
        pts_s = frames[abs_s]["source_pts_ms"]
        pts_e = frames[abs_e]["source_pts_ms"]

        # find peak motion within this segment
        seg = sm_short[abs_s:abs_e+1]
        peak_rel = int(np.argmax(seg))
        peak_abs = abs_s + peak_rel
        pts_peak = frames[peak_abs]["source_pts_ms"]

        reps.append({
            "start_fi": int(abs_s),
            "end_fi": int(abs_e),
            "peak_fi": int(peak_abs),
            "start_pts_ms": round(float(pts_s), 3),
            "end_pts_ms": round(float(pts_e), 3),
            "peak_pts_ms": round(float(pts_peak), 3),
            "duration_ms": round(float(pts_e - pts_s), 1),
            "duration_frames": int(abs_e - abs_s),
            "peak_motion": round(float(np.max(seg)), 5),
            "mean_motion": round(float(np.mean(seg)), 5),
        })

    # also try nose-Y (head bob) as secondary signal for validation
    nose_sm = uniform_filter1d(np.nan_to_num(signals["nose_y"], nan=0.5),
                                size=15)
    post_nose = -nose_sm[post_start:]  # invert: bow = min → peak
    nose_peaks, nose_props = find_peaks(
        post_nose, prominence=0.005, distance=50)
    nose_peak_count = len(nose_peaks)

    return reps, sm_short, sm_long, thresh, nose_peak_count


# ---- 3. PTS mapping to side views -----------------------------------------

def map_pts_to_view(front_pts_ms, front_clap_pts, view_clap_pts):
    """Map a front PTS to a side-view PTS using clap offsets.

    Provisional -- NOT verified. Returns mapped PTS in ms.
    """
    elapsed = front_pts_ms - front_clap_pts
    return round(view_clap_pts + elapsed, 3)


def pts_to_fi(frames, target_pts_ms):
    """Find closest frame index to a target PTS."""
    best_fi = 0
    best_diff = abs(frames[0]["source_pts_ms"] - target_pts_ms)
    for f in frames:
        diff = abs(f["source_pts_ms"] - target_pts_ms)
        if diff < best_diff:
            best_diff = diff
            best_fi = f["fi"]
    return best_fi


# ---- 4. review MP4 --------------------------------------------------------

def make_segmentation_review_mp4(video_path, frames, reps, sm_short, sm_long,
                                  thresh, clap_fi, out_path):
    """Front video with motion envelope overlay + rep IDs + start/end flags."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w, out_h = 1280, 720
    bar_h = 140
    video_h = out_h - bar_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    n = len(frames)

    # build rep lookup: fi -> rep_id
    fi_to_rep = {}
    rep_starts = set()
    rep_ends = set()
    for ri, rep in enumerate(reps):
        rep_starts.add(rep["start_fi"])
        rep_ends.add(rep["end_fi"])
        for fi in range(rep["start_fi"], rep["end_fi"] + 1):
            fi_to_rep[fi] = ri

    # signal range for visualisation
    sig_all = sm_long[:n]
    sig_min = float(np.nanmin(sig_all))
    sig_max = float(np.nanmax(sig_all)) + 0.001
    sig_range = sig_max - sig_min

    fi = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(bgr, (out_w, video_h))
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:video_h, :] = frame_resized

        # signal bar background
        bar_y0 = video_h
        canvas[bar_y0:, :] = (30, 30, 30)

        # draw full waveform
        for x in range(out_w):
            si = int(x * n / out_w)
            if 0 <= si < n:
                # long envelope (yellow line)
                val_l = (sm_long[si] - sig_min) / sig_range
                y_l = int(bar_y0 + bar_h - val_l * bar_h)
                y_l = max(bar_y0, min(out_h - 1, y_l))
                canvas[y_l, x] = (0, 200, 200)

                # short signal (white)
                val_s = (sm_short[si] - sig_min) / sig_range
                y_s = int(bar_y0 + bar_h - val_s * bar_h)
                y_s = max(bar_y0, min(out_h - 1, y_s))
                canvas[y_s, x] = (200, 200, 200)

        # threshold line (red dashed)
        thr_val = (thresh - sig_min) / sig_range
        thr_y = int(bar_y0 + bar_h - thr_val * bar_h)
        thr_y = max(bar_y0, min(out_h - 1, thr_y))
        for x in range(0, out_w, 6):
            canvas[thr_y, x:x+3] = (0, 0, 200)

        # current position indicator (cyan vertical line)
        cx = int(fi * out_w / n)
        cv2.line(canvas, (cx, bar_y0), (cx, out_h - 1), (255, 255, 0), 1)

        # rep start/end markers in waveform
        for ri, rep in enumerate(reps):
            sx = int(rep["start_fi"] * out_w / n)
            ex = int(rep["end_fi"] * out_w / n)
            cv2.line(canvas, (sx, bar_y0), (sx, out_h-1), (0, 255, 0), 1)
            cv2.line(canvas, (ex, bar_y0), (ex, out_h-1), (0, 0, 255), 1)
            mx = (sx + ex) // 2
            cv2.putText(canvas, str(ri), (mx - 4, bar_y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # clap marker
        clap_x = int(clap_fi * out_w / n)
        cv2.line(canvas, (clap_x, bar_y0), (clap_x, out_h-1), (255, 0, 255), 2)

        # overlay on video: rep ID and start/end flags
        rep_id = fi_to_rep.get(fi, None)
        if rep_id is not None:
            cv2.putText(canvas, f"Rep {rep_id}",
                        (out_w - 150, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if fi in rep_starts:
            cv2.putText(canvas, "START",
                        (out_w - 150, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if fi in rep_ends:
            cv2.putText(canvas, "END",
                        (out_w - 150, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # frame info
        pts_ms = frames[fi]["source_pts_ms"] if fi < len(frames) else 0
        cv2.putText(canvas, f"fi={fi} pts={pts_ms:.0f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        writer.write(canvas)
        fi += 1

    cap.release()
    writer.release()
    sz = os.path.getsize(out_path) // 1024
    print(f"  segmentation_review.mp4: {sz}KB", file=sys.stderr)


# ---- 5. per-rep contact sheets (3-view) -----------------------------------

def make_rep_contact_sheets(manifest_videos, front_frames, side_data,
                             reps, sync_info, max_reps=20):
    """Generate one contact sheet per rep showing Start/Peak/End across 3 views."""
    views = ["front", "left45", "right45"]
    front_clap_pts = sync_info["corrected_best_candidates"]["front"]["source_pts_ms"]
    view_clap_pts = {
        "front": front_clap_pts,
        "left45": sync_info["corrected_best_candidates"]["left45"]["source_pts_ms"],
        "right45": sync_info["corrected_best_candidates"]["right45"]["source_pts_ms"],
    }

    thumb_w, thumb_h = 220, 165
    # 3 columns (Start, Peak, End) × 3 rows (views)

    for ri, rep in enumerate(reps[:max_reps]):
        canvas = np.zeros((3 * (thumb_h + 25), 3 * thumb_w, 3), np.uint8)

        # front frame indices
        front_fis = [rep["start_fi"], rep["peak_fi"], rep["end_fi"]]
        front_pts_list = [rep["start_pts_ms"], rep["peak_pts_ms"], rep["end_pts_ms"]]
        col_labels = ["Start", "Peak", "End"]

        for row, view in enumerate(views):
            vid_path = manifest_videos[view]["absolute_path"]
            cap = cv2.VideoCapture(vid_path)
            total = manifest_videos[view]["frame_count"]

            for col in range(3):
                # map front PTS to this view
                mapped_pts = map_pts_to_view(
                    front_pts_list[col], front_clap_pts, view_clap_pts[view])
                # find closest fi in this view
                if view == "front":
                    fi = front_fis[col]
                else:
                    side_frames = side_data[view]["frames"]
                    fi = pts_to_fi(side_frames, mapped_pts)

                fi = max(0, min(total - 1, fi))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, bgr = cap.read()
                if not ret:
                    continue
                thumb = cv2.resize(bgr, (thumb_w, thumb_h))

                y0 = row * (thumb_h + 25)
                x0 = col * thumb_w
                canvas[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb

                label = f"{view} fi={fi} [{col_labels[col]}]"
                cv2.putText(canvas, label,
                            (x0 + 3, y0 + thumb_h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

            cap.release()

        out_path = os.path.join(OUT_DIR, f"rep_contact_{ri:02d}.png")
        cv2.imwrite(out_path, canvas)

    print(f"  rep contact sheets: {min(len(reps), max_reps)} files", file=sys.stderr)


# ---- 6. CSV template -------------------------------------------------------

def write_boundaries_csv(reps, sync_info, out_path):
    front_clap = sync_info["corrected_best_candidates"]["front"]["source_pts_ms"]
    l45_clap   = sync_info["corrected_best_candidates"]["left45"]["source_pts_ms"]
    r45_clap   = sync_info["corrected_best_candidates"]["right45"]["source_pts_ms"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rep_id",
            "front_start_pts_ms", "front_end_pts_ms",
            "front_start_fi", "front_end_fi",
            "front_peak_fi", "duration_ms",
            "left45_mapped_start_pts_ms", "left45_mapped_end_pts_ms",
            "right45_mapped_start_pts_ms", "right45_mapped_end_pts_ms",
            "peak_motion", "mean_motion",
            "user_status", "user_notes",
        ])
        for ri, rep in enumerate(reps):
            l45_s = map_pts_to_view(rep["start_pts_ms"], front_clap, l45_clap)
            l45_e = map_pts_to_view(rep["end_pts_ms"],   front_clap, l45_clap)
            r45_s = map_pts_to_view(rep["start_pts_ms"], front_clap, r45_clap)
            r45_e = map_pts_to_view(rep["end_pts_ms"],   front_clap, r45_clap)
            w.writerow([
                ri,
                round(rep["start_pts_ms"], 1), round(rep["end_pts_ms"], 1),
                rep["start_fi"], rep["end_fi"],
                rep["peak_fi"], round(rep["duration_ms"], 1),
                round(l45_s, 1), round(l45_e, 1),
                round(r45_s, 1), round(r45_e, 1),
                rep["peak_motion"], rep["mean_motion"],
                "pending", "",
            ])
    print(f"  CSV: {out_path}", file=sys.stderr)


# ---- main ------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = load_json("manifest.json")
    sync     = load_json("sync_candidates.json")
    sync_corr = sync["automated_analysis"]

    # front clap
    front_clap_fi  = sync_corr["corrected_best_candidates"]["front"]["frame_index"]
    front_clap_pts = sync_corr["corrected_best_candidates"]["front"]["source_pts_ms"]

    # load front landmarks
    print("[load] front landmarks...", file=sys.stderr)
    front_data = load_json("landmarks_raw_front.json")
    front_frames = front_data["frames"]
    print(f"  {len(front_frames)} frames loaded", file=sys.stderr)

    # ---- 1. motion signals ----
    print("[motion] computing signals...", file=sys.stderr)
    signals = compute_motion_signals(front_frames)

    # ---- 2. detect reps ----
    print("[segment] detecting repetitions...", file=sys.stderr)
    result = detect_repetitions(front_frames, signals, front_clap_fi,
                                 front_clap_pts)
    reps, sm_short, sm_long, thresh, nose_peak_count = result

    print(f"  detected: {len(reps)} repetition candidates", file=sys.stderr)
    print(f"  nose-Y validation: {nose_peak_count} head-bob peaks", file=sys.stderr)
    for ri, rep in enumerate(reps):
        print(f"  rep[{ri:2d}]: fi=[{rep['start_fi']},{rep['end_fi']}] "
              f"dur={rep['duration_ms']:.0f}ms peak_motion={rep['peak_motion']:.4f}",
              file=sys.stderr)

    # ---- 3. segmentation_candidates.json ----
    seg_report = {
        "step": "Step 3: Repetition Segmentation",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "motion_envelope_thresholding",
        "parameters": {
            "smoothing_short": 9,
            "smoothing_long": 31,
            "threshold_formula": "median + 0.3 * std (post-clap motion)",
            "threshold_value": round(thresh, 6),
            "min_gap_frames": 15,
            "min_duration_frames": 30,
        },
        "clap_reference": {
            "front_fi": front_clap_fi,
            "front_pts_ms": front_clap_pts,
        },
        "detected_count": len(reps),
        "nose_y_peak_count": nose_peak_count,
        "count_agreement": abs(len(reps) - nose_peak_count) <= 2,
        "note": (
            f"Detected {len(reps)} motion bursts. "
            f"Nose-Y head-bob analysis found {nose_peak_count} peaks. "
            f"{'Counts are consistent.' if abs(len(reps) - nose_peak_count) <= 2 else 'Counts diverge -- manual review recommended.'} "
            "No count was forced. Boundaries are data-driven."
        ),
        "repetitions": reps,
        "status": "PENDING_HUMAN_REVIEW",
    }
    save_json(seg_report, "segmentation_candidates.json")

    # ---- 4. review MP4 ----
    print("[video] segmentation_review.mp4...", file=sys.stderr)
    front_video = manifest["videos"]["front"]["absolute_path"]
    make_segmentation_review_mp4(
        front_video, front_frames, reps, sm_short, sm_long,
        thresh, front_clap_fi,
        os.path.join(OUT_DIR, "segmentation_review.mp4"))

    # ---- 5. per-rep contact sheets ----
    print("[contact] per-rep sheets...", file=sys.stderr)
    # load side-view frame lists (just fi + pts, skip full landmarks)
    side_data = {}
    for view in ["left45", "right45"]:
        print(f"  loading {view} (pts only)...", file=sys.stderr)
        vd = load_json(f"landmarks_raw_{view}.json")
        side_data[view] = {"frames": [{"fi": f["fi"], "source_pts_ms": f["source_pts_ms"]}
                                       for f in vd["frames"]]}

    make_rep_contact_sheets(manifest["videos"], front_frames, side_data,
                             reps, sync_corr)

    # ---- 6. CSV template ----
    print("[csv] boundaries template...", file=sys.stderr)
    write_boundaries_csv(reps, sync_corr,
                          os.path.join(OUT_DIR, "boundaries_override_template.csv"))

    # ---- summary ----
    print(f"\n=== Step 3 complete ===", file=sys.stderr)
    print(f"  {len(reps)} repetition candidates detected", file=sys.stderr)
    print(f"  status: PENDING_HUMAN_REVIEW", file=sys.stderr)


if __name__ == "__main__":
    main()
