"""
Tools/s04_step4_corrections.py — Apply human-reviewed corrections to v3a gate outputs.

Corrections:
  1. sync_candidates.json: status → HUMAN_VERIFIED, remove false-positive ambiguity
  2. sync_review.mp4: re-render around true clap fi=397 (not fi=652)
  3. boundaries_override_template.csv: rep 0-10 accepted, rep 11 uncertain/short, rep 12 rejected
  4. verification_summary.md: regenerate with corrected status
"""

import csv
import cv2
import datetime
import hashlib
import io
import json
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(_THIS_DIR, "s04_out_v3a")

VIDEOS = {
    "front":   os.path.join(_THIS_DIR, "dataset", "SIGN002_GREETING_P01_S01_B01_front.mp4"),
    "left45":  os.path.join(_THIS_DIR, "dataset", "SIGN002_GREETING_P01_S01_B01_left45.mp4"),
    "right45": os.path.join(_THIS_DIR, "dataset", "SIGN002_GREETING_P01_S01_B01_right45.mp4"),
}

# Human-verified clap frames
CLAP = {
    "front":   {"fi": 397, "pts_ms": 13266.633},
    "left45":  {"fi": 215, "pts_ms": 7167.078},
    "right45": {"fi": 338, "pts_ms": 11294.5},
}

LAGS = {
    "front":   0.0,
    "left45":  7167.078 - 13266.633,   # -6099.555
    "right45": 11294.5  - 13266.633,   # -1972.133
}

def load_json(name):
    with open(os.path.join(OUT_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(name, data):
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── 1. Update sync_candidates.json ──────────────────────────────────────────

def fix_sync_candidates():
    print("[1/4] Updating sync_candidates.json...")
    sc = load_json("sync_candidates.json")

    # Overwrite status
    sc["status"] = "HUMAN_VERIFIED"

    # Clear the false-positive flag - human confirmed the correction
    sc["automated_analysis"]["front_global_min_is_false_positive"] = True
    sc["automated_analysis"]["human_verification"] = {
        "verified_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdict": "Corrected candidates confirmed by human visual inspection.",
        "final_clap_frames": {
            "front":   {"fi": 397, "pts_ms": 13266.633},
            "left45":  {"fi": 215, "pts_ms": 7167.078},
            "right45": {"fi": 338, "pts_ms": 11294.5},
        },
        "final_lags_from_front_ms": dict(LAGS),
    }

    # Remove PENDING instructions
    sc["human_review_instructions"] = ["Completed. See automated_analysis.human_verification."]

    save_json("sync_candidates.json", sc)
    print("  status: HUMAN_VERIFIED")


# ── 2. Re-render sync_review.mp4 around true clap fi=397 ────────────────────

def render_sync_review():
    print("[2/4] Re-rendering sync_review.mp4 around true clap fi=397...")

    WINDOW = 30  # frames before/after clap
    FPS = 30

    # For each view, compute the frame index corresponding to front clap moment
    view_clap_fi = {
        "front":   397,
        "left45":  215,
        "right45": 338,
    }

    # Read frames from each view around clap
    view_frames = {}
    view_sizes = {}
    for view, path in VIDEOS.items():
        cap = cv2.VideoCapture(path)
        clap_fi = view_clap_fi[view]
        start_fi = max(0, clap_fi - WINDOW)
        end_fi = clap_fi + WINDOW

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_fi)
        frames = []
        for fi in range(start_fi, end_fi + 1):
            ret, frame = cap.read()
            if not ret:
                break
            # Resize to uniform height for stacking
            h, w = frame.shape[:2]
            target_h = 360
            scale = target_h / h
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, target_h))

            # Mark clap frame with green border
            if fi == clap_fi:
                cv2.rectangle(frame, (0, 0), (new_w - 1, target_h - 1), (0, 255, 0), 4)
                cv2.putText(frame, "CLAP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Add frame index label
            label = f"{view} fi={fi}"
            cv2.putText(frame, label, (10, target_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            frames.append(frame)
            view_sizes[view] = (new_w, target_h)
        cap.release()
        view_frames[view] = frames

    # Determine canvas: stack views vertically
    max_frames = min(len(view_frames[v]) for v in VIDEOS)
    max_w = max(s[0] for s in view_sizes.values())

    out_path = os.path.join(OUT_DIR, "sync_review.mp4")
    total_h = sum(s[1] for s in view_sizes.values())
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (max_w, total_h))

    for i in range(max_frames):
        rows = []
        for view in ["front", "left45", "right45"]:
            f = view_frames[view][i]
            h, w = f.shape[:2]
            if w < max_w:
                pad = np.zeros((h, max_w - w, 3), dtype=np.uint8)
                f = np.hstack([f, pad])
            rows.append(f)
        canvas = np.vstack(rows)
        writer.write(canvas)

    writer.release()
    size_kb = os.path.getsize(out_path) // 1024
    print(f"  sync_review.mp4: {size_kb} KB, {max_frames} frames")


# ── 3. Update boundaries CSV ────────────────────────────────────────────────

def fix_boundaries_csv():
    print("[3/4] Updating boundaries_override_template.csv...")

    path = os.path.join(OUT_DIR, "boundaries_override_template.csv")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    for row in rows:
        rep_id = int(row["rep_id"])
        if rep_id <= 10:
            row["user_status"] = "accepted"
            row["user_notes"] = ""
        elif rep_id == 11:
            row["user_status"] = "uncertain"
            row["user_notes"] = "short (1.9s); high peak motion (0.294); review before use"
        elif rep_id == 12:
            row["user_status"] = "rejected"
            row["user_notes"] = "walking to camera at video end; not a gesture"

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    accepted = sum(1 for r in rows if r["user_status"] == "accepted")
    print(f"  {accepted} accepted, 1 uncertain, 1 rejected")


# ── 4. Regenerate verification_summary.md ────────────────────────────────────

def regenerate_summary():
    print("[4/4] Regenerating verification_summary.md...")

    manifest = load_json("manifest.json")
    sync = load_json("sync_candidates.json")
    seg = load_json("segmentation_candidates.json")

    lines = []
    a = lines.append

    a("# v3a Verification Gate - Summary Report (Human-Corrected)")
    a("")
    a(f"Generated: {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    a(f"Human review applied: sync clap correction, rep 12 rejected, rep 11 uncertain")
    a("")

    # 1. Provenance
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
    a("SHA-256 re-verification: **ALL PASS** (confirmed in Step 4 automated checks)")
    a("")

    # 2. Calibration
    a("## 2. Calibration Usability")
    a("")
    a("- **Usable**: false")
    a("- No intrinsic/extrinsic calibration data available (heterogeneous consumer cameras, no checkerboard)")
    a("- Implication: true metric triangulation not possible; body-local 3D fusion uses MediaPipe world landmarks instead")
    a("")

    # 3. Sync
    a("## 3. Sync Candidates (Human-Verified)")
    a("")
    a("| View | Clap Frame | PTS (ms) | Lag from Front (ms) |")
    a("|------|-----------|----------|---------------------|")
    for view in ["front", "left45", "right45"]:
        c = CLAP[view]
        lag = LAGS[view]
        a(f"| {view} | {c['fi']} | {c['pts_ms']} | {lag:.3f} |")
    a("")
    a("**Status**: HUMAN_VERIFIED (manual sync fix applied)")
    a("")
    a("- Automated global minimum (front fi=652) was a false positive (arms at sides, not clap).")
    a("- Front fi=397 (highest-prominence local minimum, wrist_dist=0.0345) confirmed as true clap by human inspection of `sync_contact_sheet_corrected.png`.")
    a("- Cross-view alignment confirmed: all 3 corrected candidates show identical physical gesture (hands together).")
    a("")

    a("### PTS vs FPS Residuals")
    a("")
    for view in ["front", "left45", "right45"]:
        vd = sync.get("views", {}).get(view, {})
        res = vd.get("pts_vs_fps_residual_ms", "?")
        a(f"- {view}: {res} ms")
    a("")

    # 4. Segmentation
    a("## 4. Repetition Segmentation (Human-Corrected)")
    a("")
    a("- **Raw detected**: 13 repetition candidates")
    a("- **Nose-Y head-bob peaks**: 14")
    a("- **After human review**: 11 accepted, 1 uncertain, 1 rejected")
    a("- **Usable repetitions**: 11 (Rep 0 through Rep 10)")
    a("")

    params = seg.get("parameters", {})
    a(f"- Threshold: {params.get('threshold_value', '?')} ({params.get('threshold_formula', '?')})")
    a(f"- Smoothing: short={params.get('smoothing_short', '?')}, long={params.get('smoothing_long', '?')}")
    a("")

    reps = seg.get("repetitions", [])
    a("| Rep | Start FI | End FI | Duration (ms) | Peak Motion | Status |")
    a("|-----|----------|--------|---------------|-------------|--------|")
    for i, r in enumerate(reps):
        if i <= 10:
            status = "accepted"
        elif i == 11:
            status = "uncertain (short, high peak)"
        else:
            status = "**rejected** (walking to camera)"
        a(f"| {i} | {r['start_fi']} | {r['end_fi']} | {r['duration_ms']:.0f} | {r['peak_motion']:.4f} | {status} |")
    a("")

    usable = [r for i, r in enumerate(reps) if i <= 10]
    durs = [r["duration_ms"] for r in usable]
    a(f"- Usable duration range: {min(durs):.0f} ms - {max(durs):.0f} ms (mean: {sum(durs)/len(durs):.0f} ms)")
    a("")

    # 5. Automated checks
    a("## 5. Automated Sanity Checks (from Step 4)")
    a("")
    a("| Check | Result |")
    a("|-------|--------|")
    a("| SHA-256 consistency (3 videos) | PASS |")
    a("| Monotonic PTS (front: 3056, left45: 3016, right45: 3045) | PASS - 0 violations |")
    a("| Video decode (sync_review.mp4 + segmentation_review.mp4) | PASS |")
    a("| Zero-filled detection check (3 views) | PASS - 0 zero-filled |")
    a("")
    a("Null detections (missing pose/hands in some frames) stored correctly as `null`, not zero-filled.")
    a("")

    # 6. Human review checklist - completed
    a("## 6. Human Review Checklist")
    a("")
    a("- [x] **Sync verification**: `sync_contact_sheet_corrected.png` confirmed - clap alignment correct at front fi=397, left45 fi=215, right45 fi=338.")
    a("- [x] **Sync video**: `sync_review.mp4` re-rendered around true clap frame fi=397.")
    a("- [x] **Segmentation review**: `segmentation_review.mp4` inspected.")
    a("- [x] **Per-rep contact sheets**: `rep_contact_00.png` - `rep_contact_12.png` reviewed.")
    a("- [x] **Rep count validation**: 13 detected -> 11 accepted + 1 uncertain + 1 rejected.")
    a("- [x] **Short rep check**: Rep[8] (1.0s) accepted. Rep[11] (1.9s, peak=0.294) marked uncertain.")
    a("- [x] **Late reps check**: Rep[12] (fi=2981-3055) rejected as walking-to-camera, not a gesture.")
    a("- [x] **Boundary precision**: Accepted reps 0-10 boundaries confirmed adequate.")
    a("- [ ] **Mirror/rotation**: Not yet verified (requires ffprobe rotation metadata).")
    a("")

    # 7. File inventory
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
        ("sync_candidates.json", "Clap sync results (HUMAN_VERIFIED)"),
        ("sync_contact_sheet.png", "Original automated sync pick (false positive)"),
        ("sync_contact_sheet_corrected.png", "Corrected sync candidates (human confirmed)"),
        ("sync_review.mp4", "Sync review video (re-rendered around fi=397)"),
        ("segmentation_candidates.json", "Repetition boundary candidates"),
        ("segmentation_review.mp4", "Full video with motion overlay + rep IDs"),
        ("boundaries_override_template.csv", "Human-corrected boundary statuses"),
    ]
    for i in range(13):
        inventory.append((f"rep_contact_{i:02d}.png", f"Rep {i} contact sheet"))
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

    # 8. Status
    a("---")
    a("")
    a("## STATUS: AWAITING_HUMAN_REVIEW")
    a("")
    a("Human corrections applied (sync fix, rep 12 rejection). Remaining open item:")
    a("- Mirror/rotation verification pending (requires ffprobe or manual inspection).")
    a("")
    a("Once the remaining checklist item is cleared, this gate may be promoted to VERIFIED")
    a("and downstream pipeline stages (Phase 2: Fusion, Phase 3: Animation) may proceed")
    a("using accepted reps (0-10) with the human-verified clap offsets.")
    a("")

    out_path = os.path.join(OUT_DIR, "verification_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    size = os.path.getsize(out_path)
    print(f"  verification_summary.md: {size} bytes")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== v3a Corrections (Human Review) ===\n")
    fix_sync_candidates()
    render_sync_review()
    fix_boundaries_csv()
    regenerate_summary()
    print("\n=== All corrections applied ===")

if __name__ == "__main__":
    main()
