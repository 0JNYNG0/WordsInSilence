"""
S001_P001 Session Pipeline — Steps 1-4
  1. Media manifest (SHA-256, resolution, codec, fps, VFR check, etc.)
  2. Audio-based synchronization (clap detection)
  3. C2 extrinsic calibration (ChArUco in first ~2-3 min)
  4. Interim judgment

Uses C1 intrinsics from Tools/calibration/results/C1/validation/intrinsics_*_v2.json
Does NOT modify original videos.
"""

import cv2
import numpy as np
import json
import hashlib
import os
import sys
import subprocess
import struct
import wave
import tempfile
from pathlib import Path
from scipy.signal import find_peaks, correlate
from scipy.optimize import least_squares

# ─── Paths ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sessions" / "S001_P001"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "sessions" / "S001_P001"
C1_DIR = PROJECT_ROOT / "Tools" / "calibration" / "results" / "C1" / "validation"

VIDEOS = {
    "FRONT": RAW_DIR / "S001_P001_FRONT.mp4",
    "LEFT":  RAW_DIR / "S001_P001_LEFT.mp4",
    "RIGHT": RAW_DIR / "S001_P001_RIGHT.mp4",
}

CAMERA_IDS = {"FRONT": "CAM_FRONT_01", "LEFT": "CAM_LEFT_01", "RIGHT": "CAM_RIGHT_01"}

# Board spec (from C1_APPROVAL.json)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD = cv2.aruco.CharucoBoard((7, 5), 0.035, 0.026, ARUCO_DICT)
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

# ffmpeg from imageio_ffmpeg
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_EXE = "ffmpeg"


def sha256_file(path):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_video_ffmpeg(path):
    """Use ffmpeg to probe video metadata (since ffprobe is not available)."""
    result = {}
    try:
        # Get video stream info via ffmpeg -i
        proc = subprocess.run(
            [FFMPEG_EXE, "-i", str(path)],
            capture_output=True, text=True, timeout=30
        )
        stderr = proc.stderr
        result["ffmpeg_stderr_snippet"] = stderr[:2000]

        # Parse video stream
        import re
        # Video line: Stream #0:0... Video: h264 ..., 1920x1080, 30 fps, ...
        vid_match = re.search(
            r"Stream #\d+:\d+.*Video:\s*(\S+).*?,\s*(\d+)x(\d+)", stderr
        )
        if vid_match:
            result["codec"] = vid_match.group(1).rstrip(",")
            result["width"] = int(vid_match.group(2))
            result["height"] = int(vid_match.group(3))

        # FPS patterns
        fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", stderr)
        if fps_match:
            result["fps_reported"] = float(fps_match.group(1))

        tbr_match = re.search(r"(\d+(?:\.\d+)?)\s*tbr", stderr)
        if tbr_match:
            result["tbr"] = float(tbr_match.group(1))

        tbn_match = re.search(r"(\d+(?:\.\d+)?k?)\s*tbn", stderr)
        if tbn_match:
            result["tbn"] = tbn_match.group(1)

        # Duration
        dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", stderr)
        if dur_match:
            h, m, s = float(dur_match.group(1)), float(dur_match.group(2)), float(dur_match.group(3))
            result["duration_s"] = round(h * 3600 + m * 60 + s, 3)

        # Audio
        aud_match = re.search(r"Stream #\d+:\d+.*Audio:\s*(\S+)", stderr)
        result["has_audio"] = aud_match is not None
        if aud_match:
            result["audio_codec"] = aud_match.group(1).rstrip(",")
            sr_match = re.search(r"(\d+)\s*Hz", stderr[aud_match.start():])
            if sr_match:
                result["audio_sample_rate"] = int(sr_match.group(1))

        # Rotation metadata
        rot_match = re.search(r"rotate\s*:\s*(\d+)", stderr)
        if rot_match:
            result["rotation"] = int(rot_match.group(1))
        else:
            result["rotation"] = 0

    except Exception as e:
        result["probe_error"] = str(e)

    return result


def check_vfr_opencv(path, max_frames=500):
    """Check VFR by reading PTS via OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "cannot_open"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    expected_interval_ms = 1000.0 / fps if fps > 0 else 33.333

    pts_list = []
    for i in range(max_frames):
        ret = cap.grab()
        if not ret:
            break
        pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        pts_list.append(pts_ms)

    cap.release()

    if len(pts_list) < 10:
        return {"error": "too_few_frames", "n_frames_read": len(pts_list)}

    intervals = np.diff(pts_list)
    intervals = intervals[intervals > 0]  # filter zero intervals

    if len(intervals) < 5:
        return {"error": "too_few_valid_intervals"}

    result = {
        "n_frames_sampled": len(pts_list),
        "interval_mean_ms": round(float(np.mean(intervals)), 3),
        "interval_std_ms": round(float(np.std(intervals)), 3),
        "interval_min_ms": round(float(np.min(intervals)), 3),
        "interval_max_ms": round(float(np.max(intervals)), 3),
        "expected_interval_ms": round(expected_interval_ms, 3),
        "opencv_fps": round(fps, 6),
    }

    # VFR heuristic: std > 2ms or max-min > 10ms
    is_vfr = result["interval_std_ms"] > 2.0 or (result["interval_max_ms"] - result["interval_min_ms"]) > 10.0
    result["vfr_detected"] = is_vfr
    result["vfr_status"] = "VFR_DETECTED" if is_vfr else "CFR_LIKELY"

    return result


# ═══════════════════════════════════════════════════════════
#  STEP 1: Media Manifest
# ═══════════════════════════════════════════════════════════
def step1_media_manifest():
    print("\n" + "=" * 70)
    print("  STEP 1: Media Manifest")
    print("=" * 70)

    manifest = {
        "session_id": "S001",
        "participant_id": "P001",
        "n_cameras": 3,
        "simultaneous_recording": True,
        "n_actions": 45,
        "note": "Three cameras recorded simultaneously. 45 actions = 1 sample each (NOT 3×45).",
        "cameras": {}
    }

    for view, path in VIDEOS.items():
        print(f"\n  Processing {view}: {path.name}")
        if not path.exists():
            print(f"    ERROR: File not found!")
            manifest["cameras"][view] = {"error": "FILE_NOT_FOUND", "path": str(path)}
            continue

        # SHA-256
        print(f"    Computing SHA-256...")
        sha = sha256_file(path)
        print(f"    SHA-256: {sha[:16]}...")

        # File size
        fsize = path.stat().st_size

        # OpenCV probe
        cap = cv2.VideoCapture(str(path))
        cv_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cv_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv_fps = cap.get(cv2.CAP_PROP_FPS)
        cv_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv_duration = cv_frame_count / cv_fps if cv_fps > 0 else 0
        cv_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((cv_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()

        # ffmpeg probe
        ff = probe_video_ffmpeg(path)

        # VFR check
        vfr = check_vfr_opencv(path)

        cam_info = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "file_size_bytes": fsize,
            "sha256": sha,
            "width": ff.get("width", cv_width),
            "height": ff.get("height", cv_height),
            "codec": ff.get("codec", fourcc_str),
            "opencv_fps": round(cv_fps, 6),
            "fps_reported": ff.get("fps_reported"),
            "tbr": ff.get("tbr"),
            "tbn": ff.get("tbn"),
            "frame_count_opencv": cv_frame_count,
            "duration_s": ff.get("duration_s", round(cv_duration, 3)),
            "rotation": ff.get("rotation", 0),
            "has_audio": ff.get("has_audio", False),
            "audio_codec": ff.get("audio_codec"),
            "audio_sample_rate": ff.get("audio_sample_rate"),
            "vfr_check": vfr,
        }

        # Remove ffmpeg stderr snippet from final output
        manifest["cameras"][view] = cam_info
        print(f"    {cam_info['width']}x{cam_info['height']}, "
              f"fps={cam_info['opencv_fps']}, "
              f"frames={cam_info['frame_count_opencv']}, "
              f"duration={cam_info['duration_s']}s, "
              f"audio={cam_info['has_audio']}, "
              f"VFR={vfr.get('vfr_status','?')}")

    out_path = OUT_DIR / "session_media_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")
    return manifest


# ═══════════════════════════════════════════════════════════
#  STEP 2: Audio-based Synchronization
# ═══════════════════════════════════════════════════════════
def extract_audio_wav(video_path, out_wav, sample_rate=16000):
    """Extract audio from video to mono WAV using ffmpeg."""
    cmd = [
        FFMPEG_EXE, "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate),
        "-ac", "1", out_wav
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {proc.stderr[:500]}")
    return out_wav


def read_wav_samples(wav_path):
    """Read mono 16-bit WAV and return float32 array + sample rate."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def detect_claps(audio, sr, min_prominence=0.15, min_distance_s=0.3):
    """Detect clap peaks in audio signal using envelope."""
    # Compute RMS envelope with ~20ms window
    win = int(sr * 0.02)
    envelope = np.sqrt(np.convolve(audio ** 2, np.ones(win) / win, mode="same"))

    # Normalize
    envelope = envelope / (np.max(envelope) + 1e-10)

    # Find peaks
    min_dist = int(min_distance_s * sr)
    peaks, props = find_peaks(envelope, prominence=min_prominence, distance=min_dist)

    return peaks, props["prominences"], envelope


def find_clap_cluster(peaks, prominences, sr, n_claps=3,
                       min_interval_s=0.2, max_interval_s=1.5):
    """
    Find a cluster of n_claps consecutive peaks with consistent intervals.
    Returns list of (peak_sample, prominence) tuples for the best cluster,
    or None if no valid cluster found.
    """
    if len(peaks) < n_claps:
        return None

    best_cluster = None
    best_score = -1  # higher is better (sum of prominences)

    for i in range(len(peaks) - n_claps + 1):
        cluster_peaks = peaks[i:i + n_claps]
        cluster_proms = prominences[i:i + n_claps]
        intervals = np.diff(cluster_peaks) / sr

        # Check all intervals are within acceptable range
        if np.all(intervals >= min_interval_s) and np.all(intervals <= max_interval_s):
            # Check intervals are reasonably consistent (max/min ratio < 3)
            if np.min(intervals) > 0 and np.max(intervals) / np.min(intervals) < 3.0:
                score = float(np.sum(cluster_proms))
                if score > best_score:
                    best_score = score
                    best_cluster = list(zip(cluster_peaks.tolist(), cluster_proms.tolist()))

    return best_cluster


def step2_synchronization(manifest):
    print("\n" + "=" * 70)
    print("  STEP 2: Audio-based Synchronization")
    print("=" * 70)

    SR = 16000  # Common sample rate for sync
    audio_data = {}
    sync_report = {
        "method": "audio_cross_correlation",
        "reference_camera": "FRONT",
        "sample_rate_hz": SR,
        "cameras": {},
    }

    # Extract audio from all videos
    tmp_dir = OUT_DIR / "tmp_audio"
    tmp_dir.mkdir(exist_ok=True)

    for view, path in VIDEOS.items():
        has_audio = manifest["cameras"].get(view, {}).get("has_audio", False)
        if not has_audio:
            print(f"  {view}: No audio track — sync not possible via audio!")
            sync_report["cameras"][view] = {"error": "NO_AUDIO_TRACK"}
            continue

        print(f"  Extracting audio: {view}...")
        wav_path = str(tmp_dir / f"{view}.wav")
        try:
            extract_audio_wav(path, wav_path, SR)
            samples, sr = read_wav_samples(wav_path)
            audio_data[view] = samples
            print(f"    Samples: {len(samples)}, Duration: {len(samples)/sr:.2f}s")
            sync_report["cameras"][view] = {
                "audio_samples": len(samples),
                "audio_duration_s": round(len(samples) / sr, 3),
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            sync_report["cameras"][view] = {"error": str(e)}

    if len(audio_data) < 2:
        print("  ERROR: Need audio from at least 2 cameras for sync!")
        sync_report["status"] = "FAIL_NO_AUDIO"
        out_path = OUT_DIR / "sync_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sync_report, f, indent=2, ensure_ascii=False)
        return sync_report

    # Detect claps in each audio
    clap_info = {}
    for view, audio in audio_data.items():
        peaks, proms, envelope = detect_claps(audio, SR, min_prominence=0.10)
        clap_info[view] = {
            "peaks_samples": peaks,
            "prominences": proms,
            "envelope": envelope,
        }
        peak_times = [round(p / SR, 3) for p in peaks]
        print(f"  {view}: {len(peaks)} audio peaks detected")
        if len(peaks) > 0:
            print(f"    Top peaks (time_s): {peak_times[:10]}...")
            sync_report["cameras"][view]["n_peaks"] = len(peaks)
            sync_report["cameras"][view]["peak_times_s"] = peak_times[:30]

    # Cross-correlation for each pair relative to FRONT
    ref_view = "FRONT"
    if ref_view not in audio_data:
        ref_view = list(audio_data.keys())[0]

    ref_audio = audio_data[ref_view]
    sync_report["reference_camera"] = ref_view

    # Use last 20% of audio for end-clap correlation
    # Also try full audio cross-correlation
    offsets = {}
    for view, audio in audio_data.items():
        if view == ref_view:
            offsets[view] = 0.0
            sync_report["cameras"][view]["offset_s"] = 0.0
            sync_report["cameras"][view]["offset_samples"] = 0
            continue

        # Full cross-correlation (computationally heavy for long audio, use last 30s)
        # Use last portion for clap sync
        last_n = min(SR * 60, len(ref_audio))  # last 60s
        ref_tail = ref_audio[-last_n:]
        other_tail = audio[-last_n:]

        # Pad shorter to same length
        max_len = max(len(ref_tail), len(other_tail))
        ref_padded = np.pad(ref_tail, (0, max_len - len(ref_tail)))
        other_padded = np.pad(other_tail, (0, max_len - len(other_tail)))

        # Cross-correlation
        corr = correlate(ref_padded, other_padded, mode="full")
        lag_samples = np.argmax(corr) - (max_len - 1)
        lag_s = lag_samples / SR

        # Also try with first 60s for start-of-video sync
        first_n = min(SR * 60, len(ref_audio))
        ref_head = ref_audio[:first_n]
        other_head = audio[:first_n]
        max_len_h = max(len(ref_head), len(other_head))
        ref_h_pad = np.pad(ref_head, (0, max_len_h - len(ref_head)))
        other_h_pad = np.pad(other_head, (0, max_len_h - len(other_head)))
        corr_head = correlate(ref_h_pad, other_h_pad, mode="full")
        lag_head_samples = np.argmax(corr_head) - (max_len_h - 1)
        lag_head_s = lag_head_samples / SR

        # Peak correlation values
        corr_peak_tail = float(np.max(corr))
        corr_peak_head = float(np.max(corr_head))

        offsets[view] = lag_s

        # Drift estimation: if both head and tail sync are available
        drift_s = None
        if abs(lag_head_s) < 30 and abs(lag_s) < 30:  # sanity
            drift_s = round(lag_s - lag_head_s, 6)
            ref_dur = len(ref_audio) / SR
            drift_ppm = round(drift_s / ref_dur * 1e6, 2) if ref_dur > 0 else None

        cam_sync = {
            "offset_s_tail_xcorr": round(lag_s, 6),
            "offset_samples_tail": int(lag_samples),
            "xcorr_peak_tail": round(corr_peak_tail, 2),
            "offset_s_head_xcorr": round(lag_head_s, 6),
            "offset_samples_head": int(lag_head_samples),
            "xcorr_peak_head": round(corr_peak_head, 2),
        }
        if drift_s is not None:
            cam_sync["drift_s"] = drift_s
            cam_sync["drift_ppm"] = drift_ppm

        # Use head offset as primary (closer to start of useful data)
        cam_sync["offset_s"] = round(lag_head_s, 6)
        cam_sync["offset_samples"] = int(lag_head_samples)

        sync_report["cameras"][view].update(cam_sync)
        print(f"  {view} vs {ref_view}:")
        print(f"    Head xcorr offset: {lag_head_s:.4f}s ({lag_head_samples} samples)")
        print(f"    Tail xcorr offset: {lag_s:.4f}s ({lag_samples} samples)")
        if drift_s is not None:
            print(f"    Drift: {drift_s:.6f}s ({drift_ppm:.2f} ppm)")

    # Attempt clap-based refinement: find 3 end claps using cluster detection
    print("\n  Clap-based refinement (cluster detection)...")
    end_clap_times = {}
    start_clap_times = {}
    for view in audio_data:
        peaks = clap_info[view]["peaks_samples"]
        proms = clap_info[view]["prominences"]
        audio_dur = len(audio_data[view]) / SR

        # Find ALL valid 3-clap clusters in the full audio
        all_clusters = []
        for i in range(len(peaks) - 2):
            cluster_p = peaks[i:i + 3]
            cluster_pr = proms[i:i + 3]
            intervals = np.diff(cluster_p) / SR
            if np.all(intervals >= 0.2) and np.all(intervals <= 1.5):
                if np.min(intervals) > 0 and np.max(intervals) / np.min(intervals) < 3.0:
                    center_time = float(cluster_p[1]) / SR
                    score = float(np.sum(cluster_pr))
                    all_clusters.append((center_time, score, cluster_p.tolist(), cluster_pr.tolist()))

        if all_clusters:
            # End cluster: highest center_time
            end_cl = max(all_clusters, key=lambda x: x[0])
            end_times = [round(p / SR, 4) for p in end_cl[2]]
            end_clap_times[view] = end_times
            print(f"    {view} end claps: {end_times} (score={end_cl[1]:.3f})")
            sync_report["cameras"][view]["end_clap_times_s"] = end_times

            # Start cluster: lowest center_time (if distinct from end)
            start_cl = min(all_clusters, key=lambda x: x[0])
            if start_cl[0] < end_cl[0] - 30:  # at least 30s apart
                start_times = [round(p / SR, 4) for p in start_cl[2]]
                start_clap_times[view] = start_times
                print(f"    {view} start claps: {start_times}")
                sync_report["cameras"][view]["start_clap_times_s"] = start_times
        else:
            print(f"    {view}: No valid 3-clap cluster found")

    # Refine offset using clap clusters.
    # Key insight: each camera started recording at a different time, so the
    # FIRST cluster in each camera is NOT necessarily the same physical event.
    # Strategy: use end claps (reliable, same event) as primary offset,
    # then find the MATCHING start claps using that offset to compute drift.

    if len(end_clap_times) >= 2 and ref_view in end_clap_times:
        ref_end = end_clap_times[ref_view]

        for view in audio_data:
            if view == ref_view:
                continue
            cam = sync_report["cameras"][view]
            other_end = end_clap_times.get(view)

            if not (other_end and len(other_end) == 3 and len(ref_end) == 3):
                cam["offset_s_final"] = cam.get("offset_s", 0.0)
                cam["offset_method"] = "xcorr_head"
                continue

            # End clap offset: other_time = ref_time + offset
            end_diffs = [c - r for c, r in zip(other_end, ref_end)]
            end_offset = float(np.mean(end_diffs))
            end_offset_std = float(np.std(end_diffs))
            cam["end_clap_offset_s"] = round(end_offset, 6)
            cam["end_clap_offset_std_s"] = round(end_offset_std, 6)
            print(f"    {view} end-clap offset: {end_offset:.4f}s (std={end_offset_std:.6f}s)")

            # Use end-clap offset as primary
            cam["offset_s_final"] = round(end_offset, 6)
            cam["offset_method"] = "end_clap"
            cam["residual_error_estimate_s"] = round(end_offset_std, 6)

            # Now find matching start claps: look for a cluster in `view` that
            # matches ref start claps when adjusted by end_offset.
            ref_start = start_clap_times.get(ref_view)
            if ref_start:
                # Expected time in other camera for ref start event
                expected_times = [t + end_offset for t in ref_start]
                # Search all clusters in other camera for best match
                other_peaks = clap_info[view]["peaks_samples"]
                other_proms = clap_info[view]["prominences"]
                best_match = None
                best_err = float("inf")
                for i in range(len(other_peaks) - 2):
                    cp = other_peaks[i:i + 3]
                    intervals = np.diff(cp) / SR
                    if not (np.all(intervals >= 0.2) and np.all(intervals <= 1.5)):
                        continue
                    if np.min(intervals) <= 0 or np.max(intervals) / np.min(intervals) >= 3.0:
                        continue
                    times = [float(p) / SR for p in cp]
                    err = np.mean([abs(t - e) for t, e in zip(times, expected_times)])
                    if err < best_err:
                        best_err = err
                        best_match = times

                if best_match and best_err < 1.0:  # within 1s tolerance
                    start_diffs = [c - r for c, r in zip(best_match, ref_start)]
                    start_offset = float(np.mean(start_diffs))
                    start_offset_std = float(np.std(start_diffs))
                    cam["matched_start_clap_times_s"] = [round(t, 4) for t in best_match]
                    cam["matched_start_offset_s"] = round(start_offset, 6)
                    cam["matched_start_offset_std_s"] = round(start_offset_std, 6)
                    print(f"    {view} matched start claps: {[round(t,4) for t in best_match]}")
                    print(f"    {view} start-clap offset: {start_offset:.4f}s (std={start_offset_std:.6f}s)")

                    # Drift = end_offset - start_offset
                    drift = end_offset - start_offset
                    span_s = ref_end[1] - ref_start[1]
                    drift_ppm = round(drift / span_s * 1e6, 2) if span_s > 0 else None
                    cam["drift_s"] = round(drift, 6)
                    cam["drift_ppm"] = drift_ppm
                    print(f"    {view} drift: {drift:.6f}s ({drift_ppm:.1f} ppm over {span_s:.1f}s)")
                else:
                    print(f"    {view} no matching start claps found (best_err={best_err:.2f}s)")
                    cam["drift_s"] = None
                    cam["drift_ppm"] = None

            print(f"    {view} FINAL offset: {cam['offset_s_final']:.4f}s (method={cam['offset_method']})")
    else:
        for view in audio_data:
            if view == ref_view:
                continue
            cam = sync_report["cameras"][view]
            cam["offset_s_final"] = cam.get("offset_s", 0.0)
            cam["offset_method"] = "xcorr_head"

    sync_report["cameras"].setdefault(ref_view, {})
    sync_report["cameras"][ref_view]["offset_s_final"] = 0.0
    sync_report["cameras"][ref_view]["offset_method"] = "reference"
    sync_report["cameras"][ref_view]["residual_error_estimate_s"] = 0.0

    out_path = OUT_DIR / "sync_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sync_report, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    # Cleanup wav files
    for f in tmp_dir.glob("*.wav"):
        f.unlink()
    try:
        tmp_dir.rmdir()
    except:
        pass

    return sync_report


# ═══════════════════════════════════════════════════════════
#  STEP 3: C2 Extrinsic Calibration
# ═══════════════════════════════════════════════════════════
def load_c1_intrinsics():
    """Load approved C1 intrinsics."""
    intrinsics = {}
    for view, cam_id in CAMERA_IDS.items():
        path = C1_DIR / f"intrinsics_{cam_id}_v2.json"
        if not path.exists():
            raise FileNotFoundError(f"C1 intrinsics not found: {path}")
        with open(path) as f:
            data = json.load(f)
        K = np.array(data["camera_matrix"], dtype=np.float64)
        dist = np.array(data["dist_coeffs"], dtype=np.float64)
        intrinsics[view] = {"K": K, "dist": dist, "image_size": tuple(data["image_size"])}
        print(f"  Loaded C1 intrinsics: {cam_id} (RMS={data['rms_reprojection_error']})")
    return intrinsics


def detect_charuco_frame(gray):
    """Detect ChArUco board in a grayscale frame."""
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    if ids is None or len(ids) < 3:
        return None, None, 0

    ret, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, BOARD)
    if ret < 4:
        return None, None, ret if ret else 0
    return cc, ci, ret


def scan_charuco_region(video_path, view, start_s=0, end_s=180, sample_interval=10):
    """
    Scan a time region of the video for ChArUco detections.
    Coarse-only scan (no dense pass) for speed on large/VFR files.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_s * fps)
    end_frame = min(int(end_s * fps), total)

    print(f"  {view}: Scanning frames {start_frame}-{end_frame} (interval={sample_interval})")

    detections = []

    # Sequential read (much faster than random seek on VFR files)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % sample_interval == 0:
            pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cc, ci, n = detect_charuco_frame(gray)

            if cc is not None and n >= 6:
                detections.append((frame_idx, pts_ms, n, cc, ci))

        frame_idx += 1

    cap.release()

    print(f"    Detections: {len(detections)}")
    if detections:
        t0 = detections[0][1] / 1000.0
        t1 = detections[-1][1] / 1000.0
        print(f"    C2 region: {t0:.2f}s - {t1:.2f}s")

    return detections, fps


def step3_c2_calibration(sync_report):
    print("\n" + "=" * 70)
    print("  STEP 3: C2 Extrinsic Calibration")
    print("=" * 70)

    # Load C1 intrinsics
    intrinsics = load_c1_intrinsics()

    # Scan first ~3 min of each video for ChArUco
    all_detections = {}
    all_fps = {}
    for view, path in VIDEOS.items():
        dets, fps = scan_charuco_region(path, view, start_s=0, end_s=180, sample_interval=5)
        all_detections[view] = dets
        all_fps[view] = fps
        if dets:
            t_start = dets[0][1] / 1000.0
            t_end = dets[-1][1] / 1000.0
            print(f"    C2 region: {t_start:.2f}s - {t_end:.2f}s")

    # Convert frame times to synchronized time using sync offsets
    offsets = {}
    for view in VIDEOS:
        cam = sync_report.get("cameras", {}).get(view, {})
        offsets[view] = cam.get("offset_s_final", 0.0)

    def to_sync_time(view, pts_ms):
        """Convert camera-local PTS to synchronized time."""
        return pts_ms / 1000.0 - offsets[view]

    # Find common frames: for each FRONT detection, find closest LEFT and RIGHT
    ref = "FRONT"
    ref_dets = all_detections.get(ref, [])
    other_views = [v for v in VIDEOS if v != ref]

    common_frames = []
    TIME_TOLERANCE_S = 0.1  # 100ms tolerance for "same board pose"

    for r_idx, r_pts_ms, r_n, r_cc, r_ci in ref_dets:
        r_sync = to_sync_time(ref, r_pts_ms)

        matches = {ref: (r_idx, r_pts_ms, r_n, r_cc, r_ci)}
        all_matched = True

        for ov in other_views:
            ov_dets = all_detections.get(ov, [])
            best = None
            best_dt = float("inf")

            for o_idx, o_pts_ms, o_n, o_cc, o_ci in ov_dets:
                o_sync = to_sync_time(ov, o_pts_ms)
                dt = abs(o_sync - r_sync)
                if dt < best_dt:
                    best_dt = dt
                    best = (o_idx, o_pts_ms, o_n, o_cc, o_ci)

            if best is not None and best_dt < TIME_TOLERANCE_S:
                matches[ov] = best
            else:
                all_matched = False
                break

        if all_matched and len(matches) == 3:
            common_frames.append(matches)

    print(f"\n  Common frames (all 3 cameras see board): {len(common_frames)}")

    # Subsample if too many common frames (stereo calibrate is O(n))
    MAX_COMMON = 80
    if len(common_frames) > MAX_COMMON:
        indices = np.linspace(0, len(common_frames) - 1, MAX_COMMON, dtype=int)
        common_frames = [common_frames[i] for i in indices]
        print(f"  Subsampled to {len(common_frames)} common frames for stereo calibration")

    if len(common_frames) < 3:
        print("  WARNING: Very few common frames for extrinsic calibration!")
        c2_report = {
            "status": "FAIL_C2",
            "reason": f"Only {len(common_frames)} common frames (minimum 3 needed)",
            "detections_per_camera": {v: len(d) for v, d in all_detections.items()},
        }
        out_path = OUT_DIR / "C2_CALIBRATION.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(c2_report, f, indent=2, ensure_ascii=False)
        return c2_report

    # Compute extrinsic for each camera using solvePnP per frame, then
    # stereo calibrate pairs (FRONT-LEFT, FRONT-RIGHT)
    obj_points_3d = BOARD.getChessboardCorners()  # (N, 3)

    # For each camera, compute per-frame rvec/tvec using C1 intrinsics
    per_cam_poses = {v: [] for v in VIDEOS}
    for cf in common_frames:
        for view, (f_idx, pts_ms, n_corners, cc, ci) in cf.items():
            K = intrinsics[view]["K"]
            dist = intrinsics[view]["dist"]
            obj = obj_points_3d[ci.flatten()].astype(np.float64)
            img = cc.reshape(-1, 2).astype(np.float64)

            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                per_cam_poses[view].append({
                    "frame_idx": f_idx,
                    "pts_ms": pts_ms,
                    "rvec": rvec.flatten().tolist(),
                    "tvec": tvec.flatten().tolist(),
                    "n_corners": n_corners,
                })

    # Stereo calibration: FRONT as reference
    # For each pair, use common object/image points
    extrinsics = {}
    c2_report = {
        "stage": "C2",
        "session": "S001",
        "reference_camera": ref,
        "board_spec": {
            "dictionary": "DICT_4X4_50",
            "squares_x": 7,
            "squares_y": 5,
            "square_length_m": 0.035,
            "marker_length_m": 0.026,
        },
        "c1_intrinsics_used": {
            v: str(C1_DIR / f"intrinsics_{CAMERA_IDS[v]}_v2.json") for v in VIDEOS
        },
        "detections_per_camera": {v: len(d) for v, d in all_detections.items()},
        "common_frames": len(common_frames),
        "cameras": {},
    }

    # FRONT is identity (reference)
    extrinsics[ref] = {"R": np.eye(3), "t": np.zeros((3, 1))}
    c2_report["cameras"][ref] = {
        "camera_id": CAMERA_IDS[ref],
        "role": "reference",
        "R": np.eye(3).tolist(),
        "t": [0.0, 0.0, 0.0],
        "n_frames_used": len(common_frames),
    }

    for ov in other_views:
        # Collect matching object points and image points for stereo calibration
        obj_list_ref = []
        img_list_ref = []
        obj_list_ov = []
        img_list_ov = []

        for cf in common_frames:
            ref_data = cf[ref]
            ov_data = cf[ov]

            r_ci = ref_data[4].flatten()
            o_ci = ov_data[4].flatten()

            # Find common corner IDs (need >=6 for solvePnP DLT)
            common_ids = np.intersect1d(r_ci, o_ci)
            if len(common_ids) < 6:
                continue

            r_mask = np.isin(r_ci, common_ids)
            o_mask = np.isin(o_ci, common_ids)

            # Reorder to match
            r_order = np.argsort(r_ci[r_mask])
            o_order = np.argsort(o_ci[o_mask])

            obj = obj_points_3d[common_ids[np.argsort(common_ids)]].astype(np.float32)
            r_img = ref_data[3].reshape(-1, 2)[r_mask][r_order].astype(np.float32)
            o_img = ov_data[3].reshape(-1, 2)[o_mask][o_order].astype(np.float32)

            obj_list_ref.append(obj)
            img_list_ref.append(r_img)
            obj_list_ov.append(obj)
            img_list_ov.append(o_img)

        if len(obj_list_ref) < 3:
            print(f"  {ov}: Too few valid frame pairs ({len(obj_list_ref)}) for stereo calibration!")
            c2_report["cameras"][ov] = {
                "camera_id": CAMERA_IDS[ov],
                "status": "FAIL",
                "reason": f"only {len(obj_list_ref)} valid frame pairs",
            }
            continue

        K_ref = intrinsics[ref]["K"]
        d_ref = intrinsics[ref]["dist"]
        K_ov = intrinsics[ov]["K"]
        d_ov = intrinsics[ov]["dist"]
        img_size = intrinsics[ref]["image_size"]

        print(f"\n  Stereo calibrating {ref} ↔ {ov} ({len(obj_list_ref)} frame pairs)...")

        # Use CALIB_FIX_INTRINSIC to keep C1 intrinsics fixed
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

        rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_list_ref, img_list_ref, img_list_ov,
            K_ref, d_ref, K_ov, d_ov,
            img_size, flags=flags, criteria=criteria
        )

        # Per-frame reprojection error
        per_frame_errors = []
        for i in range(len(obj_list_ref)):
            # Project ref object points to ov using R, T
            obj = obj_list_ref[i]
            img_ref = img_list_ref[i]
            img_ov = img_list_ov[i]

            # SolvePnP for ref frame
            ok_r, rv_r, tv_r = cv2.solvePnP(obj, img_ref, K_ref, d_ref)
            if not ok_r:
                continue

            # Transform to ov frame
            R_r, _ = cv2.Rodrigues(rv_r)
            # Object in ref camera: P_ref = R_r * P_obj + tv_r
            # Object in ov camera: P_ov = R * P_ref + T = R * (R_r * P_obj + tv_r) + T
            R_ov = R @ R_r
            T_ov = R @ tv_r + T
            rv_ov, _ = cv2.Rodrigues(R_ov)

            proj, _ = cv2.projectPoints(obj, rv_ov, T_ov, K_ov, d_ov)
            err = np.sqrt(np.mean((img_ov - proj.reshape(-1, 2)) ** 2))
            per_frame_errors.append(round(float(err), 4))

        baseline_m = float(np.linalg.norm(T))

        print(f"    Stereo RMS: {rms:.4f} px")
        print(f"    Baseline: {baseline_m:.4f} m")
        if per_frame_errors:
            print(f"    Per-frame reproj: mean={np.mean(per_frame_errors):.4f}, "
                  f"max={np.max(per_frame_errors):.4f}")

        cam_result = {
            "camera_id": CAMERA_IDS[ov],
            "role": "secondary",
            "R": R.tolist(),
            "t": T.flatten().tolist(),
            "stereo_rms": round(float(rms), 4),
            "baseline_m": round(baseline_m, 4),
            "n_frame_pairs": len(obj_list_ref),
            "per_frame_reproj_errors": per_frame_errors,
            "per_frame_reproj_mean": round(float(np.mean(per_frame_errors)), 4) if per_frame_errors else None,
            "per_frame_reproj_max": round(float(np.max(per_frame_errors)), 4) if per_frame_errors else None,
        }

        extrinsics[ov] = {"R": R, "t": T}
        c2_report["cameras"][ov] = cam_result

    # Board position/distance/angle statistics
    board_stats = {}
    for view in VIDEOS:
        poses = per_cam_poses.get(view, [])
        if not poses:
            continue
        distances = [np.linalg.norm(p["tvec"]) for p in poses]
        board_stats[view] = {
            "n_poses": len(poses),
            "distance_mean_m": round(float(np.mean(distances)), 3),
            "distance_min_m": round(float(np.min(distances)), 3),
            "distance_max_m": round(float(np.max(distances)), 3),
        }
    c2_report["board_statistics"] = board_stats

    # C2 timestamp ranges
    c2_time_ranges = {}
    for view, dets in all_detections.items():
        if dets:
            c2_time_ranges[view] = {
                "start_pts_ms": round(dets[0][1], 1),
                "end_pts_ms": round(dets[-1][1], 1),
                "start_frame": dets[0][0],
                "end_frame": dets[-1][0],
                "duration_s": round((dets[-1][1] - dets[0][1]) / 1000.0, 2),
            }
    c2_report["c2_timestamp_ranges"] = c2_time_ranges

    # Approval criteria suggestion (no existing C2 criteria found)
    rms_values = []
    for view, cam in c2_report["cameras"].items():
        if "stereo_rms" in cam:
            rms_values.append(cam["stereo_rms"])

    c2_report["quality_assessment"] = {
        "note": "No existing C2 approval criteria found in project. Presenting metrics for user review.",
        "suggested_criteria": {
            "stereo_rms_threshold_px": 1.0,
            "min_common_frames": 10,
            "max_per_frame_reproj_px": 3.0,
        },
        "meets_suggested_criteria": {
            "stereo_rms_ok": all(r < 1.0 for r in rms_values) if rms_values else False,
            "common_frames_ok": len(common_frames) >= 10,
        }
    }

    out_path = OUT_DIR / "C2_CALIBRATION.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(c2_report, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    return c2_report


# ═══════════════════════════════════════════════════════════
#  STEP 4: Interim Judgment
# ═══════════════════════════════════════════════════════════
def step4_judgment(manifest, sync_report, c2_report):
    print("\n" + "=" * 70)
    print("  STEP 4: Interim Judgment")
    print("=" * 70)

    issues = []
    warnings = []

    # Check media
    for view in VIDEOS:
        cam = manifest.get("cameras", {}).get(view, {})
        if "error" in cam:
            issues.append(f"MEDIA: {view} — {cam['error']}")
        vfr = cam.get("vfr_check", {})
        if vfr.get("vfr_detected"):
            warnings.append(f"VFR: {view} has variable frame rate (std={vfr.get('interval_std_ms')}ms)")

    # Check sync
    sync_status = sync_report.get("status")
    if sync_status == "FAIL_NO_AUDIO":
        issues.append("SYNC: No audio available for synchronization")
    else:
        for view in ["LEFT", "RIGHT"]:
            cam = sync_report.get("cameras", {}).get(view, {})
            if "error" in cam:
                issues.append(f"SYNC: {view} — {cam['error']}")
            # Large offsets are normal (cameras started at different times)
            # Only warn about drift (clock rate difference)
            drift_ppm = cam.get("drift_ppm")
            if drift_ppm is not None and abs(drift_ppm) > 100:
                warnings.append(f"SYNC: {view} drift={drift_ppm:.1f}ppm (significant)")
            # Warn about high residual
            residual = cam.get("residual_error_estimate_s")
            if residual is not None and residual > 0.033:  # > 1 frame at 30fps
                warnings.append(f"SYNC: {view} residual={residual*1000:.1f}ms (>1 frame)")

    # Check C2
    c2_status = c2_report.get("status")
    if c2_status and "FAIL" in c2_status:
        issues.append(f"C2: {c2_report.get('reason', 'unknown failure')}")
    else:
        n_common = c2_report.get("common_frames", 0)
        if n_common < 5:
            issues.append(f"C2: Only {n_common} common frames")
        elif n_common < 10:
            warnings.append(f"C2: Only {n_common} common frames (10+ preferred)")

        for view in ["LEFT", "RIGHT"]:
            cam = c2_report.get("cameras", {}).get(view, {})
            if cam.get("status") == "FAIL":
                issues.append(f"C2: {view} stereo calibration failed — {cam.get('reason')}")
            stereo_rms = cam.get("stereo_rms")
            if stereo_rms is not None and stereo_rms > 1.0:
                warnings.append(f"C2: {view} stereo RMS={stereo_rms:.3f}px (>1.0)")

    # Judgment
    if any("MEDIA" in i for i in issues):
        judgment = "FAIL_MEDIA"
    elif any("SYNC" in i for i in issues):
        judgment = "FAIL_SYNC"
    elif any("C2" in i for i in issues):
        judgment = "FAIL_C2"
    elif warnings:
        judgment = "CONDITIONAL_PASS"
    else:
        judgment = "PASS"

    verdict = {
        "judgment": judgment,
        "issues": issues,
        "warnings": warnings,
        "summary": {
            "n_cameras": 3,
            "n_actions": 45,
            "sync_method": sync_report.get("cameras", {}).get("LEFT", {}).get("offset_method", "N/A"),
            "c2_common_frames": c2_report.get("common_frames", 0),
        }
    }

    print(f"\n  JUDGMENT: {judgment}")
    if issues:
        print(f"  Issues:")
        for i in issues:
            print(f"    - {i}")
    if warnings:
        print(f"  Warnings:")
        for w in warnings:
            print(f"    - {w}")

    out_path = OUT_DIR / "interim_judgment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")

    return verdict


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-step1", action="store_true",
                        help="Skip step 1 (use existing manifest)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("S001_P001 Session Pipeline -- Steps 1-4")
    print(f"Raw: {RAW_DIR}")
    print(f"Output: {OUT_DIR}")

    # Verify files exist
    for view, path in VIDEOS.items():
        if not path.exists():
            print(f"BLOCKED: {path} not found!")
            sys.exit(1)

    if args.skip_step1:
        manifest_path = OUT_DIR / "session_media_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"  Loaded existing manifest: {manifest_path}")
    else:
        manifest = step1_media_manifest()

    sync_report = step2_synchronization(manifest)
    c2_report = step3_c2_calibration(sync_report)
    verdict = step4_judgment(manifest, sync_report, c2_report)

    print("\n" + "=" * 70)
    print(f"  FINAL JUDGMENT: {verdict['judgment']}")
    print("=" * 70)
