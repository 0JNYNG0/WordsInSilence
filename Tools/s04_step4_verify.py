"""
Tools/s04_step4_verify.py — v3a Verification Gate Step 4: Final Sanity Checks + Summary

Automated checks:
  1. SHA-256 consistency with step1 manifest
  2. Monotonic PTS progression in landmark files
  3. ffmpeg/cv2 decode test on review videos
  4. Zero-filled detection check in landmarks
  5. Generate verification_summary.md
"""

import hashlib
import json
import os
import sys
import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(_THIS_DIR, "s04_out_v3a")

# ── helpers ──────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(name):
    with open(os.path.join(OUT_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)

# ── Check 1: SHA-256 consistency ─────────────────────────────────────────────

def check_sha256(manifest):
    results = []
    for view, info in manifest["videos"].items():
        path = info["absolute_path"]
        expected = info["sha256"]
        if not os.path.exists(path):
            results.append((view, "FAIL", f"file not found: {path}"))
            continue
        actual = sha256_file(path)
        if actual == expected:
            results.append((view, "PASS", expected[:16] + "..."))
        else:
            results.append((view, "FAIL", f"expected {expected[:16]}... got {actual[:16]}..."))
    return results

# ── Check 2: Monotonic PTS ──────────────────────────────────────────────────

def check_monotonic_pts(view):
    fname = f"landmarks_raw_{view}.json"
    path = os.path.join(OUT_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data if isinstance(data, list) else data.get("frames", data)
    violations = 0
    prev_pts = -1.0
    total = 0
    for fr in frames:
        pts = fr.get("source_pts_ms")
        if pts is None:
            continue
        total += 1
        if pts < prev_pts:
            violations += 1
        prev_pts = pts
    return total, violations

# ── Check 3: Video decode test ──────────────────────────────────────────────

def check_video_decode(filename):
    """Decode all frames with cv2, return (frame_count, ok, error_msg)."""
    import cv2
    path = os.path.join(OUT_DIR, filename)
    if not os.path.exists(path):
        return 0, False, "file not found"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, False, "cannot open"
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.size == 0:
            cap.release()
            return count, False, f"empty frame at index {count}"
        count += 1
    cap.release()
    if count == 0:
        return 0, False, "zero frames decoded"
    return count, True, ""

# ── Check 4: Zero-filled detection check ────────────────────────────────────

def check_zero_filled(view):
    """Check for frames where pose/hand landmarks exist but are all zeros."""
    fname = f"landmarks_raw_{view}.json"
    path = os.path.join(OUT_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data if isinstance(data, list) else data.get("frames", data)

    total_frames = len(frames)
    null_pose = 0
    null_hands = 0
    zero_pose = 0
    zero_hands = 0

    for fr in frames:
        # pose check
        pn = fr.get("pose_norm")
        pw = fr.get("pose_world")
        if pn is None and pw is None:
            null_pose += 1
        else:
            # check if all values are exactly 0
            flat = []
            src = pn if pn is not None else pw
            if isinstance(src, list):
                for lm in src:
                    if isinstance(lm, list):
                        flat.extend(lm[:3])
                    elif isinstance(lm, dict):
                        flat.extend([lm.get("x", 0), lm.get("y", 0), lm.get("z", 0)])
            if flat and all(v == 0.0 for v in flat):
                zero_pose += 1

        # hands check
        hands = fr.get("hands")
        if hands is None or len(hands) == 0:
            null_hands += 1
        else:
            for h in hands:
                norm = h.get("norm") or h.get("landmarks")
                if norm and isinstance(norm, list):
                    flat = []
                    for lm in norm:
                        if isinstance(lm, list):
                            flat.extend(lm[:3])
                        elif isinstance(lm, dict):
                            flat.extend([lm.get("x", 0), lm.get("y", 0), lm.get("z", 0)])
                    if flat and all(v == 0.0 for v in flat):
                        zero_hands += 1

    return {
        "total_frames": total_frames,
        "null_pose": null_pose,
        "null_hands": null_hands,
        "zero_pose": zero_pose,
        "zero_hands": zero_hands,
    }

# ── Generate verification_summary.md ────────────────────────────────────────

def generate_summary(manifest, step1, sync, seg, checks):
    lines = []
    a = lines.append

    a("# v3a Verification Gate - Summary Report")
    a("")
    a(f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    a("")

    # ── 1. Raw Video Provenance ──
    a("## 1. Raw Video Provenance")
    a("")
    a("| View | Resolution | Codec | Duration | FPS | VFR/CFR | SHA-256 (first 16) |")
    a("|------|-----------|-------|----------|-----|---------|---------------------|")
    for view in ["front", "left45", "right45"]:
        v = manifest["videos"][view]
        vfr = v["vfr_cfr_analysis"]["determination"]
        sha_short = v["sha256"][:16]
        a(f"| {view} | {v['width']}x{v['height']} | {v['codec_fourcc']} | {v['duration_s']:.1f}s | {v['fps_reported']:.2f} | {vfr} | `{sha_short}...` |")
    a("")

    # SHA-256 check results
    a("### SHA-256 Re-verification")
    a("")
    for view, status, detail in checks["sha256"]:
        a(f"- **{view}**: {status} ({detail})")
    a("")

    # ── 2. Calibration ──
    a("## 2. Calibration Usability")
    a("")
    cal = step1.get("calibration", step1.get("calibration_check", {}))
    usable = cal.get("calibration_usable", cal.get("usable", "unknown"))
    a(f"- **Usable**: {usable}")
    reasons = cal.get("reasons", cal.get("notes", []))
    if isinstance(reasons, list):
        for r in reasons:
            a(f"  - {r}")
    elif isinstance(reasons, str):
        a(f"  - {reasons}")
    a("")

    # ── 3. Sync Candidates ──
    a("## 3. Sync Candidates (Clap Detection)")
    a("")
    corrected = sync.get("automated_analysis", {}).get("corrected_best_candidates", {})
    lags = sync.get("automated_analysis", {}).get("corrected_lags_from_front_ms", {})
    a("| View | Clap Frame | PTS (ms) | Lag from Front (ms) |")
    a("|------|-----------|----------|---------------------|")
    for view in ["front", "left45", "right45"]:
        c = corrected.get(view, {})
        lag = lags.get(view, "?")
        a(f"| {view} | {c.get('frame_index', '?')} | {c.get('source_pts_ms', '?')} | {lag} |")
    a("")

    fp_note = sync.get("automated_analysis", {}).get("explanation", "")
    if fp_note:
        a(f"**Note**: {fp_note}")
        a("")

    consistency = sync.get("automated_analysis", {}).get("cross_view_consistency", "")
    if consistency:
        a(f"**Cross-view**: {consistency}")
        a("")

    # PTS residuals
    a("### PTS vs FPS Residuals")
    a("")
    for view in ["front", "left45", "right45"]:
        vd = sync.get("views", {}).get(view, {})
        res = vd.get("pts_vs_fps_residual_ms", "?")
        a(f"- {view}: {res} ms")
    a("")

    # ── 4. Repetition Segmentation ──
    a("## 4. Repetition Segmentation")
    a("")
    det = seg.get("detected_count", "?")
    nose = seg.get("nose_y_peak_count", "?")
    agree = seg.get("count_agreement", "?")
    a(f"- **Detected repetitions**: {det}")
    a(f"- **Nose-Y head-bob peaks**: {nose}")
    a(f"- **Count agreement**: {agree}")
    a(f"- **Method**: {seg.get('method', '?')}")
    a("")

    params = seg.get("parameters", {})
    a(f"- Threshold: {params.get('threshold_value', '?')} ({params.get('threshold_formula', '?')})")
    a(f"- Smoothing: short={params.get('smoothing_short', '?')}, long={params.get('smoothing_long', '?')}")
    a(f"- Min gap: {params.get('min_gap_frames', '?')} frames, min duration: {params.get('min_duration_frames', '?')} frames")
    a("")

    reps = seg.get("repetitions", [])
    if reps:
        a("| Rep | Start FI | End FI | Duration (ms) | Peak Motion | Mean Motion |")
        a("|-----|----------|--------|---------------|-------------|-------------|")
        for i, r in enumerate(reps):
            a(f"| {i} | {r['start_fi']} | {r['end_fi']} | {r['duration_ms']:.0f} | {r['peak_motion']:.4f} | {r['mean_motion']:.4f} |")
        a("")

    # duration stats
    if reps:
        durs = [r["duration_ms"] for r in reps]
        a(f"- Duration range: {min(durs):.0f} ms - {max(durs):.0f} ms")
        a(f"- Duration mean: {sum(durs)/len(durs):.0f} ms")
        a("")

    # ── 5. Automated Sanity Checks ──
    a("## 5. Automated Sanity Checks")
    a("")

    # 5a. SHA-256
    sha_pass = all(s == "PASS" for _, s, _ in checks["sha256"])
    a(f"### 5a. SHA-256 Consistency: {'PASS' if sha_pass else 'FAIL'}")
    a("")

    # 5b. Monotonic PTS
    a("### 5b. Monotonic PTS Progression")
    a("")
    pts_all_pass = True
    for view, (total, violations) in checks["pts"].items():
        status = "PASS" if violations == 0 else "FAIL"
        if violations > 0:
            pts_all_pass = False
        a(f"- {view}: {status} ({total} frames, {violations} violations)")
    a("")

    # 5c. Video decode
    a("### 5c. Video Decode Test")
    a("")
    vid_all_pass = True
    for fname, (count, ok, err) in checks["video"].items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            vid_all_pass = False
        detail = f"{count} frames" if ok else err
        a(f"- {fname}: {status} ({detail})")
    a("")

    # 5d. Zero-filled
    a("### 5d. Zero-Filled Detection Check")
    a("")
    zero_all_pass = True
    for view, info in checks["zero"].items():
        zp = info["zero_pose"]
        zh = info["zero_hands"]
        total = info["total_frames"]
        null_p = info["null_pose"]
        null_h = info["null_hands"]
        status = "PASS" if zp == 0 and zh == 0 else "FAIL"
        if zp > 0 or zh > 0:
            zero_all_pass = False
        a(f"- {view}: {status} (total={total}, null_pose={null_p}, null_hands={null_h}, zero_pose={zp}, zero_hands={zh})")
    a("")

    all_pass = sha_pass and pts_all_pass and vid_all_pass and zero_all_pass
    a(f"### Overall Automated Checks: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    a("")

    # ── 6. Human Review Checklist ──
    a("## 6. Human Review Checklist")
    a("")
    a("The following items MUST be inspected by a human reviewer before proceeding to Phase 2/3:")
    a("")
    a("- [ ] **Sync verification**: Open `sync_contact_sheet_corrected.png` and confirm the green-bordered frame in each row shows the same physical moment (hands touching in clap).")
    a("- [ ] **Sync video**: Watch `sync_review.mp4` and confirm green CLAP marker aligns across views.")
    a("- [ ] **Segmentation review**: Watch `segmentation_review.mp4` and confirm colored overlays align with actual gesture start/end boundaries.")
    a("- [ ] **Per-rep contact sheets**: Review `rep_contact_00.png` through `rep_contact_12.png`. Each sheet shows 3 views x 3 moments (Start/Peak/End). Verify the performer is executing the same gesture phase across views at each moment.")
    a("- [ ] **Rep count validation**: Confirm 13 detected repetitions matches the actual number of gesture performances in the video. Flag any false positives (non-gesture motion) or missed repetitions.")
    a("- [ ] **Short rep check**: Rep[8] is only 30 frames (1.0s). Verify this is a complete gesture, not a partial or transitional motion.")
    a("- [ ] **Late reps check**: Reps[11,12] are consecutive (fi=2905-3055) near video end. Verify these are distinct repetitions, not a single gesture split by threshold crossing.")
    a("- [ ] **Boundary precision**: For the rep(s) selected for animation, verify start/end frames include the full gesture without clipping. Use `boundaries_override_template.csv` to record corrections if needed.")
    a("- [ ] **Mirror/rotation**: Confirm performer's left/right hand matches expected side in each view (no unexpected mirroring from camera or container rotation).")
    a("")

    # ── 7. Output File Inventory ──
    a("## 7. Output File Inventory")
    a("")
    a("| File | Size | Description |")
    a("|------|------|-------------|")
    inventory = [
        ("manifest.json", "Video provenance + metadata"),
        ("step1_provenance_and_calibration.json", "Step 1 gate report"),
        ("landmarks_raw_front.json", "Front view raw landmarks"),
        ("landmarks_raw_left45.json", "Left45 view raw landmarks"),
        ("landmarks_raw_right45.json", "Right45 view raw landmarks"),
        ("sync_candidates.json", "Clap sync detection results"),
        ("sync_contact_sheet.png", "Original automated sync pick (contains false positive)"),
        ("sync_contact_sheet_corrected.png", "Corrected sync candidates (visually confirmed)"),
        ("sync_review.mp4", "60-frame sync review video"),
        ("segmentation_candidates.json", "Repetition boundary candidates"),
        ("segmentation_review.mp4", "Full video with motion overlay + rep IDs"),
        ("boundaries_override_template.csv", "Editable CSV for manual boundary corrections"),
    ]
    for i in range(13):
        inventory.append((f"rep_contact_{i:02d}.png", f"Rep {i} contact sheet (3 views x Start/Peak/End)"))
    inventory.append(("verification_summary.md", "This file"))

    for fname, desc in inventory:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            if size_kb > 1024:
                size_str = f"{size_kb/1024:.1f} MB"
            else:
                size_str = f"{size_kb:.0f} KB"
        else:
            size_str = "N/A"
        a(f"| `{fname}` | {size_str} | {desc} |")
    a("")

    # ── 8. Status Declaration ──
    a("---")
    a("")
    a("## STATUS: AWAITING_HUMAN_REVIEW")
    a("")
    a("All automated checks have completed. No candidate has been verified, approved, or marked as canonical.")
    a("A human reviewer must complete the checklist in Section 6 before any data from this gate")
    a("may be used in downstream pipeline stages (Phase 2: Fusion, Phase 3: Animation).")
    a("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[Step 4] v3a Verification Gate - Final Sanity Checks")
    print()

    # Load metadata
    manifest = load_json("manifest.json")
    step1 = load_json("step1_provenance_and_calibration.json")
    sync = load_json("sync_candidates.json")
    seg = load_json("segmentation_candidates.json")

    checks = {}

    # ── Check 1: SHA-256 ──
    print("[1/4] SHA-256 re-verification...")
    checks["sha256"] = check_sha256(manifest)
    for view, status, detail in checks["sha256"]:
        print(f"  {view}: {status}")

    # ── Check 2: Monotonic PTS ──
    print("[2/4] Monotonic PTS check...")
    checks["pts"] = {}
    for view in ["front", "left45", "right45"]:
        total, violations = check_monotonic_pts(view)
        checks["pts"][view] = (total, violations)
        status = "PASS" if violations == 0 else f"FAIL ({violations} violations)"
        print(f"  {view}: {status} ({total} frames)")

    # ── Check 3: Video decode ──
    print("[3/4] Video decode test...")
    checks["video"] = {}
    for fname in ["sync_review.mp4", "segmentation_review.mp4"]:
        count, ok, err = check_video_decode(fname)
        checks["video"][fname] = (count, ok, err)
        status = "PASS" if ok else f"FAIL ({err})"
        print(f"  {fname}: {status} ({count} frames)")

    # ── Check 4: Zero-filled ──
    print("[4/4] Zero-filled detection check...")
    checks["zero"] = {}
    for view in ["front", "left45", "right45"]:
        info = check_zero_filled(view)
        checks["zero"][view] = info
        zp, zh = info["zero_pose"], info["zero_hands"]
        status = "PASS" if zp == 0 and zh == 0 else "FAIL"
        print(f"  {view}: {status} (zero_pose={zp}, zero_hands={zh})")

    # ── Generate summary ──
    print()
    print("[summary] Generating verification_summary.md...")
    md = generate_summary(manifest, step1, sync, seg, checks)
    out_path = os.path.join(OUT_DIR, "verification_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  saved: {out_path} ({os.path.getsize(out_path)} bytes)")

    print()
    print("=== Step 4 complete ===")
    print("  STATUS: AWAITING_HUMAN_REVIEW")


if __name__ == "__main__":
    main()
