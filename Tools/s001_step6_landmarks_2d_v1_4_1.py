"""
Step 6A v1.4.1: Packaging-only correction for v1.4 pilot output.

NO MediaPipe extraction or association is rerun.
Existing v1.4 pilot data are preserved byte-for-byte.

Changes:
1. Generate contact_strong_misassignment.png (QA result image, 0 candidates)
2. Update STEP6A_PILOT_REPORT.md with complete file listing
3. SHA-256 verification of all preserved files
4. Final delivery ZIP
"""

import hashlib, sys, datetime, zipfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed" / "sessions" / "S001_P001"
PILOT_DIR = PROC_DIR / "landmarks_2d_v1_4" / "pilot"
CS_DIR = PILOT_DIR / "low_confidence_contact_sheets"

# ═══════════════════════════════════════════════════════════
#  SHA-256 reference hashes (from v1.4 run)
# ═══════════════════════════════════════════════════════════

PRESERVED_FILES = {
    "landmarks_2d_front.json": "8c3415119085c7e76ef2a1701b7fe6bb7719ad9ff5e100ffecfe9269d998a2e2",
    "landmarks_2d_left.json": "2dd5bf71c604ccce8391b1221ce09f28634c9babace96bffd570d3187b0e9d32",
    "landmarks_2d_right.json": "50bbb4a889600ecb9fc10d2458ac6e249f8b1e7cc2e1a4884086de59b5ffa790",
    "boundary_frame_diagnostics.csv": "91f6eba3fa82d0f083e4671733e817eef877c0259443b74b263f7981a4593491",
    "detection_quality.csv": "309610b6699033751d99184c3406c163c6358f4b29532b8bd69f9e03a4951358",
    "hand_identity_diagnostics.csv": "b2a6d2ace9311b37f6723e0ba280a34c8aa36c56205d79f0d1d450b3e058558c",
    "synchronized_view_coverage.csv": "7d17ae5c1d7ff94d2770f6f7f9c00c720906f4639cb42b366b75b4123137dd64",
    "run_manifest.json": "a381b1371c729e978a36e282e50ef1b121f5238227cad6e86c408f4767994c86",
    "overlays/overlay_pilot_front.mp4": "6799b7be28713b15ad4897f8201c3d4ad16d27a591a51cbe2f8779f75bfd7e84",
    "overlays/overlay_pilot_left.mp4": "c95dd36e5345ba1fdd18a133d2441de3dd7f36e415b7aef173d214670a75bf3b",
    "overlays/overlay_pilot_right.mp4": "ba35578a98107d11a791947cf41a8621e1cbe555387f3fe63483a61793ea2e64",
    "low_confidence_contact_sheets/contact_ambiguous_frames.png": "ae6485bf410d9f17ccf029517af426c480e0b6d0c94081b32870093c6c7b49b2",
    "low_confidence_contact_sheets/contact_duplicate_candidates.png": "03dcb83c4d1defc332212f699fcdc08a6ebd43dad903c1710e390dad253e1778",
    "low_confidence_contact_sheets/contact_fast_front.png": "6a04cf51aed9fe2d77ad3ed3840eb05f7ee14d4e351132ba840c4fb7808844e4",
    "low_confidence_contact_sheets/contact_fast_left.png": "0684e565363e8b979338fe1b8d947d97dfeac7846e4dc787f7ac9f9b143e4ae1",
    "low_confidence_contact_sheets/contact_fast_right.png": "c0a90046fb0d28ad44beec707ccca780106c08ef302ea70b7cbdb226cb785e01",
    "low_confidence_contact_sheets/contact_low_det_greeting_slow_left.png": "a60b96cbf1afa0886a947df1052c1142544bfc08786ca1e4764b12eab587c673",
    "low_confidence_contact_sheets/contact_low_det_love_normal_right.png": "7cd05fa90697730298390e116de17ec23e14df07c0cf60db91a103934c3effb6",
    "low_confidence_contact_sheets/contact_regression_thankyou_fast_left.png": "2a5c23bfb36e2747615f33a00b8903f5e37a5447e88de5e8417e2983139f13de",
    "low_confidence_contact_sheets/contact_regression_thankyou_front.png": "cf27a82561d2a163f336b358414efb304a7bad3bd1a1c2fef8cd876044fac28b",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("Step 6A v1.4.1: Packaging-only correction\n")

    # ══════════════════════════════════════════════════════
    #  1. Verify all preserved files (SHA-256)
    # ══════════════════════════════════════════════════════
    log("  == SHA-256 verification of preserved files ==\n")
    sha_pass = 0
    sha_fail = 0
    for rel, expected in PRESERVED_FILES.items():
        fp = PILOT_DIR / rel.replace("/", "\\")
        if not fp.exists():
            fp = PILOT_DIR / rel  # try forward slash
        if not fp.exists():
            log(f"    FAIL: {rel} — FILE NOT FOUND")
            sha_fail += 1
            continue
        actual = sha256_file(fp)
        if actual == expected:
            sha_pass += 1
        else:
            log(f"    FAIL: {rel}")
            log(f"      expected: {expected}")
            log(f"      actual:   {actual}")
            sha_fail += 1

    log(f"    SHA-256: {sha_pass} PASS, {sha_fail} FAIL out of {len(PRESERVED_FILES)}")
    if sha_fail > 0:
        log("\n  FATAL: Preserved files have been modified. Aborting.")
        sys.exit(1)
    log("")

    # ══════════════════════════════════════════════════════
    #  2. Generate contact_strong_misassignment.png
    # ══════════════════════════════════════════════════════
    log("  == Generating contact_strong_misassignment.png ==\n")

    img_w, img_h = 768, 432
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # dark gray background

    # Draw border
    cv2.rectangle(img, (10, 10), (img_w - 10, img_h - 10), (100, 100, 100), 2)

    # Title
    y = 60
    cv2.putText(img, "Step 6A v1.4.1", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    y += 50
    cv2.putText(img, "Strong anatomical misassignment review", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # Candidate count
    y += 60
    cv2.putText(img, "Candidate frame count: 0", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Criterion
    y += 50
    cv2.putText(img, "Criterion:", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    y += 35
    cv2.putText(img, "assigned_distance > 2 x opposite_distance", (50, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    y += 30
    cv2.putText(img, "and assigned_distance - opposite_distance > 50px", (50, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Result
    y += 55
    cv2.putText(img, "Result: NO CANDIDATE FRAMES", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out_png = CS_DIR / "contact_strong_misassignment.png"
    cv2.imwrite(str(out_png), img)
    log(f"    Created: {out_png.name}")

    # Verify PNG
    assert out_png.exists(), "PNG file not created"
    assert out_png.stat().st_size > 0, "PNG file is empty"
    verify_img = cv2.imread(str(out_png))
    assert verify_img is not None, "PNG cannot be decoded by OpenCV"
    assert verify_img.shape == (img_h, img_w, 3), f"PNG shape mismatch: {verify_img.shape}"
    log(f"    Size: {out_png.stat().st_size / 1024:.1f} KB")
    log(f"    OpenCV decode: OK ({verify_img.shape})")

    # Verify contact sheet count = 10
    all_cs = sorted(CS_DIR.glob("contact_*.png"))
    assert len(all_cs) == 10, f"Expected 10 contact sheets, got {len(all_cs)}: {[p.name for p in all_cs]}"
    log(f"    Total contact sheets: {len(all_cs)}")
    for cs in all_cs:
        log(f"      {cs.name}: {cs.stat().st_size / 1024:.1f} KB")
    log("")

    # ══════════════════════════════════════════════════════
    #  3. Update STEP6A_PILOT_REPORT.md
    # ══════════════════════════════════════════════════════
    log("  == Updating STEP6A_PILOT_REPORT.md ==\n")

    # Collect all files with sizes
    file_list = []
    for p in sorted(PILOT_DIR.rglob("*")):
        if p.is_file():
            sk = p.stat().st_size / 1024
            sz_str = f"{sk / 1024:.1f} MB" if sk > 1024 else f"{sk:.1f} KB"
            file_list.append((str(p.relative_to(PILOT_DIR)).replace("\\", "/"), sz_str))

    report = []
    report.append("# Step 6A v1.4.1 Pilot Report -- MediaPipe 2D Landmark Extraction")
    report.append("")
    report.append("## Status: REQUIRES_EXTERNAL_PACKAGING_CONFIRMATION")
    report.append("")
    report.append(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    report.append("**Session**: S001_P001")
    report.append("**Script**: `Tools/s001_step6_landmarks_2d_v1_4.py` (extraction), `Tools/s001_step6_landmarks_2d_v1_4_1.py` (packaging)")
    report.append("**Pilot**: 9 R03 samples x 3 views")
    report.append("")
    report.append("MediaPipe extraction and anatomical association were not rerun for v1.4.1. Existing v1.4 pilot data were preserved byte-for-byte; this revision only completes QA packaging artifacts.")
    report.append("")

    report.append("## v1.3 Regression Fix Results (from v1.4)")
    report.append("")
    report.append("| Issue | v1.3 | v1.4 |")
    report.append("|-------|------|------|")
    report.append("| bidirectional_single_anchor_correction | 96 | **0** |")
    report.append("| strong_misassignment_count | 42 | **0** |")
    report.append("| Raw vs frame-level AMBIGUOUS mismatch | 111 vs 107 | **consistent** |")
    report.append("| Raw vs frame-level DUPLICATE mismatch | 55 vs 53 | **consistent** |")
    report.append("| Sync coverage matching | greedy nearest | **DP monotonic** |")
    report.append("")

    report.append("## Two-Pass Association (unchanged from v1.3)")
    report.append("")
    report.append("**Observed mapping**: {'Left': 'ANAT_LEFT', 'Right': 'ANAT_RIGHT'}")
    report.append("")

    report.append("## Raw Landmark Extraction Quality")
    report.append("- Pose: 100% usable rate across all 27 view-samples")
    report.append("- 0 PTS-missing frames, 0 timestamp adjustments")
    report.append("- Frame counts: 1001/1177/1178 (identical to v1, v1.1, v1.2, v1.3)")
    report.append("")

    report.append("## Assertion Results")
    report.append("")
    report.append("- **12881 PASS, 0 FAIL** out of 12881")
    report.append("")

    report.append("## Conflict and Quality Counts")
    report.append("")
    report.append("| Metric | Count |")
    report.append("|--------|-------|")
    report.append("| same_nearest_wrist_conflict | 73 |")
    report.append("| close_same_wrist_conflict | 60 |")
    report.append("| accepted_dual_hand_conflict | 0 |")
    report.append("| strong_misassignment_count | 0 |")
    report.append("| bidirectional_single_anchor_correction | 0 |")
    report.append("| DUPLICATE frames | 55 |")
    report.append("| AMBIGUOUS frames | 111 |")
    report.append("| swap comparisons | 2309 |")
    report.append("| swap candidates | 0 |")
    report.append("")

    report.append("## Detection Rates (both hands DETECTED)")
    report.append("")
    report.append("| View | Min | Median | Max | Worst |")
    report.append("|------|-----|--------|-----|-------|")
    report.append("| FRONT | 47% | 77% | 92% | GREETING_FAST_R03 (47%) |")
    report.append("| LEFT | 37% | 56% | 81% | GREETING_SLOW_R03 (37%) |")
    report.append("| RIGHT | 52% | 90% | 98% | LOVE_NORMAL_R03 (52%) |")
    report.append("")

    report.append("## Regression Case: THANK_YOU_NORMAL FRONT 318.7-318.83s")
    report.append("")
    report.append("Losers correctly marked DUPLICATE_DETECTION_CANDIDATE (not DETECTED as ANAT_RIGHT).")
    report.append("```")
    report.append("t=318.6666 hi=0 anat=ANAT_LEFT status=DETECTED dL=29.94 dR=245.52")
    report.append("t=318.7000 hi=0 anat=None status=DUPLICATE_DETECTION_CANDIDATE dL=25.97 dR=250.03")
    report.append("t=318.7000 hi=1 anat=ANAT_LEFT status=DETECTED dL=18.78 dR=255.09")
    report.append("t=318.7333 hi=0 anat=None status=DUPLICATE_DETECTION_CANDIDATE dL=15.22 dR=267.0")
    report.append("t=318.7333 hi=1 anat=ANAT_LEFT status=DETECTED dL=11.56 dR=271.88")
    report.append("t=318.7666 hi=0 anat=None status=DUPLICATE_DETECTION_CANDIDATE dL=3.54 dR=272.97")
    report.append("t=318.7666 hi=1 anat=ANAT_LEFT status=DETECTED dL=10.26 dR=276.7")
    report.append("t=318.8000 hi=0 anat=ANAT_LEFT status=DETECTED dL=5.48 dR=283.65")
    report.append("t=318.8000 hi=1 anat=None status=DUPLICATE_DETECTION_CANDIDATE dL=12.14 dR=284.12")
    report.append("t=318.8333 hi=0 anat=ANAT_LEFT status=DETECTED dL=7.76 dR=288.9")
    report.append("t=318.8333 hi=1 anat=None status=DUPLICATE_DETECTION_CANDIDATE dL=10.86 dR=289.35")
    report.append("t=318.8666 hi=0 anat=ANAT_LEFT status=DETECTED dL=8.21 dR=299.93")
    report.append("t=318.9000 hi=0 anat=ANAT_LEFT status=DETECTED dL=9.59 dR=304.78")
    report.append("t=318.9000 hi=1 anat=ANAT_RIGHT status=DETECTED dL=289.92 dR=14.54")
    report.append("```")
    report.append("")

    report.append("## Regression Case: THANK_YOU_FAST LEFT 391.05-391.4s")
    report.append("")
    report.append("Single-anchor correction REMOVED. Existing global assignment preserved.")
    report.append("```")
    report.append("t=391.055822 hi=0 anat=ANAT_RIGHT status=DETECTED dL=60.11 dR=29.2")
    report.append("t=391.122489 hi=0 anat=ANAT_RIGHT status=DETECTED dL=27.44 dR=16.61")
    report.append("t=391.155822 hi=0 anat=ANAT_RIGHT status=DETECTED dL=25.15 dR=18.51")
    report.append("t=391.189156 hi=0 anat=ANAT_RIGHT status=DETECTED dL=31.65 dR=21.26")
    report.append("t=391.222489 hi=0 anat=None status=DUPLICATE_DETECTION_CANDIDATE dL=45.3 dR=24.45")
    report.append("t=391.222489 hi=1 anat=ANAT_RIGHT status=DETECTED dL=32.38 dR=9.98")
    report.append("t=391.255833 hi=0 anat=ANAT_RIGHT status=DETECTED dL=53.03 dR=34.4")
    report.append("t=391.289167 hi=0 anat=ANAT_RIGHT status=DETECTED dL=59.7 dR=23.68")
    report.append("t=391.322500 hi=0 anat=ANAT_RIGHT status=DETECTED dL=68.7 dR=31.27")
    report.append("t=391.322500 hi=1 anat=ANAT_LEFT status=DETECTED dL=6.38 dR=36.98")
    report.append("t=391.355833 hi=0 anat=ANAT_RIGHT status=DETECTED dL=70.49 dR=39.23")
    report.append("t=391.355833 hi=1 anat=ANAT_LEFT status=DETECTED dL=11.16 dR=41.2")
    report.append("t=391.389167 hi=0 anat=ANAT_RIGHT status=DETECTED dL=68.1 dR=42.87")
    report.append("t=391.389167 hi=1 anat=ANAT_LEFT status=DETECTED dL=9.38 dR=34.86")
    report.append("```")
    report.append("")

    report.append("## Synchronized Multi-View Coverage (DP-based monotonic matching)")
    report.append("")
    report.append("**Note**: Detection availability diagnostic only. Triangulation sufficiency not evaluated.")
    report.append("")
    report.append("| Sample | Hand | ref | fd | 2+v | 3v | cond | reuse |")
    report.append("|--------|------|-----|----|----|----|----|-------|")
    report.append("| GREETING_NORMAL_R03 | LEFT | 131 | 117 | 98% | 87% | 100% | 0 |")
    report.append("| GREETING_NORMAL_R03 | RIGHT | 131 | 114 | 90% | 49% | 96% | 0 |")
    report.append("| GREETING_SLOW_R03 | LEFT | 181 | 139 | 97% | 77% | 100% | 0 |")
    report.append("| GREETING_SLOW_R03 | RIGHT | 181 | 130 | 73% | 34% | 99% | 0 |")
    report.append("| GREETING_FAST_R03 | LEFT | 97 | 93 | 96% | 86% | 99% | 0 |")
    report.append("| GREETING_FAST_R03 | RIGHT | 97 | 46 | 56% | 31% | 93% | 0 |")
    report.append("| THANK_YOU_NORMAL_R03 | LEFT | 144 | 144 | 100% | 97% | 100% | 0 |")
    report.append("| THANK_YOU_NORMAL_R03 | RIGHT | 144 | 111 | 92% | 60% | 99% | 0 |")
    report.append("| THANK_YOU_SLOW_R03 | LEFT | 133 | 132 | 100% | 98% | 100% | 0 |")
    report.append("| THANK_YOU_SLOW_R03 | RIGHT | 133 | 122 | 96% | 47% | 100% | 0 |")
    report.append("| THANK_YOU_FAST_R03 | LEFT | 89 | 86 | 94% | 67% | 98% | 0 |")
    report.append("| THANK_YOU_FAST_R03 | RIGHT | 89 | 75 | 88% | 58% | 99% | 0 |")
    report.append("| LOVE_NORMAL_R03 | LEFT | 75 | 70 | 95% | 43% | 100% | 0 |")
    report.append("| LOVE_NORMAL_R03 | RIGHT | 75 | 57 | 99% | 55% | 100% | 0 |")
    report.append("| LOVE_SLOW_R03 | LEFT | 94 | 90 | 95% | 79% | 98% | 0 |")
    report.append("| LOVE_SLOW_R03 | RIGHT | 94 | 74 | 95% | 69% | 100% | 0 |")
    report.append("| LOVE_FAST_R03 | LEFT | 57 | 52 | 88% | 72% | 94% | 0 |")
    report.append("| LOVE_FAST_R03 | RIGHT | 57 | 34 | 89% | 40% | 100% | 0 |")
    report.append("")

    report.append("## Boundary PTS Errors")
    report.append("")
    report.append("Max start: 71.90ms. Max end: 44.77ms.")
    report.append("Original extraction, PTS, and boundary values are unchanged from v1.2/v1.3.")
    report.append("")

    report.append("## Output Files")
    report.append("")
    report.append("| File | Size |")
    report.append("|------|------|")
    for rel, sz in file_list:
        report.append(f"| {rel} | {sz} |")
    report.append("")

    report.append("## Delivery")
    report.append("")
    report.append("| Item | Path |")
    report.append("|------|------|")
    report.append("| Extraction script | `Tools/s001_step6_landmarks_2d_v1_4.py` |")
    report.append("| Packaging script | `Tools/s001_step6_landmarks_2d_v1_4_1.py` |")
    report.append("| Pilot JSON (3) | `landmarks_2d_{front,left,right}.json` |")
    report.append("| CSV (4) | `boundary_frame_diagnostics.csv`, `detection_quality.csv`, `hand_identity_diagnostics.csv`, `synchronized_view_coverage.csv` |")
    report.append("| Manifest | `run_manifest.json` |")
    report.append("| Overlays (3) | `overlays/overlay_pilot_{front,left,right}.mp4` |")
    report.append("| Contact sheets (10) | `low_confidence_contact_sheets/contact_*.png` |")
    report.append("| Report | `STEP6A_PILOT_REPORT.md` |")
    report.append("| Delivery ZIP | `WordsInSilence_step6a_v1_4_1_delivery.zip` |")
    report.append("| Packaging log | `landmarks_2d_v1_4_1_packaging.log` |")
    report.append("")

    report.append("## Null Policy")
    report.append("- Unassigned detections: AMBIGUOUS or DUPLICATE_DETECTION_CANDIDATE with null landmarks")
    report.append("- No forced assignment, no 0-fill, no interpolation")
    report.append("- hand_wrist_px stored in output")
    report.append("- Frame-level status reconstructed from raw hands after all refinement")
    report.append("")

    report_path = PILOT_DIR / "STEP6A_PILOT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log(f"    Updated: {report_path.name} ({report_path.stat().st_size / 1024:.1f} KB)")
    log("")

    # ══════════════════════════════════════════════════════
    #  4. Final assertions
    # ══════════════════════════════════════════════════════
    log("  == Packaging assertions ==\n")

    assertions = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        assertions.append((name, status, detail))
        if status == "FAIL":
            log(f"    FAIL: {name} — {detail}")
        return cond

    # SHA-256 preservation
    check("sha256_all_preserved", sha_fail == 0, f"{sha_pass}/{len(PRESERVED_FILES)} matched")

    # contact_strong_misassignment.png exists and valid
    sm_png = CS_DIR / "contact_strong_misassignment.png"
    check("strong_misassign_png_exists", sm_png.exists())
    check("strong_misassign_png_size_gt_0", sm_png.stat().st_size > 0 if sm_png.exists() else False,
          f"size={sm_png.stat().st_size}" if sm_png.exists() else "missing")
    if sm_png.exists():
        vimg = cv2.imread(str(sm_png))
        check("strong_misassign_png_decodable", vimg is not None)

    # Exactly 10 contact sheets
    all_sheets = sorted(CS_DIR.glob("contact_*.png"))
    check("contact_sheet_count_10", len(all_sheets) == 10,
          f"got {len(all_sheets)}: {[p.name for p in all_sheets]}")

    # Report exists
    check("report_exists", report_path.exists())
    check("report_size_gt_0", report_path.stat().st_size > 0 if report_path.exists() else False)

    # Report contains required items
    if report_path.exists():
        rtxt = report_path.read_text(encoding="utf-8")
        check("report_has_v1_4_1", "v1.4.1" in rtxt)
        check("report_has_packaging_status", "REQUIRES_EXTERNAL_PACKAGING_CONFIRMATION" in rtxt)
        check("report_has_preservation_note", "preserved byte-for-byte" in rtxt)
        check("report_has_contact_strong", "contact_strong_misassignment.png" in rtxt)
        check("report_has_delivery_zip", "WordsInSilence_step6a_v1_4_1_delivery.zip" in rtxt)
        check("report_has_packaging_log", "packaging.log" in rtxt)
        check("report_has_extraction_script", "s001_step6_landmarks_2d_v1_4.py" in rtxt)
        check("report_has_packaging_script", "s001_step6_landmarks_2d_v1_4_1.py" in rtxt)

    n_pass = sum(1 for _, s, _ in assertions if s == "PASS")
    n_fail = sum(1 for _, s, _ in assertions if s == "FAIL")
    log(f"\n    Assertions: {n_pass} PASS, {n_fail} FAIL out of {len(assertions)}")

    if n_fail > 0:
        log("\n  FATAL: Packaging assertions failed.")
        # Save log before exit
        log_path = PROC_DIR / "landmarks_2d_v1_4_1_packaging.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        sys.exit(1)

    # ══════════════════════════════════════════════════════
    #  5. Create delivery ZIP
    # ══════════════════════════════════════════════════════
    log("\n  == Creating delivery ZIP ==\n")

    zip_path = PROJECT_ROOT / "WordsInSilence_step6a_v1_4_1_delivery.zip"

    # Save packaging log first so it goes into the ZIP
    log_path = PROC_DIR / "landmarks_2d_v1_4_1_packaging.log"
    # We'll write a preliminary log, then update after ZIP
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
        f.write("\n(ZIP creation in progress...)\n")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Scripts
        for script in ["s001_step6_landmarks_2d_v1_4.py", "s001_step6_landmarks_2d_v1_4_1.py"]:
            sp = PROJECT_ROOT / "Tools" / script
            if sp.exists():
                zf.write(sp, f"Tools/{script}")

        # All pilot output
        for p in sorted(PILOT_DIR.rglob("*")):
            if p.is_file():
                arcname = str(p.relative_to(PROC_DIR)).replace("\\", "/")
                zf.write(p, arcname)

        # Packaging log
        zf.write(log_path, "landmarks_2d_v1_4_1_packaging.log")

    log(f"    ZIP: {zip_path.name} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Verify ZIP integrity
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                log(f"    ZIP integrity FAIL: corrupt file {bad}")
                sys.exit(1)
            zip_files = sorted(zf.namelist())
    except Exception as e:
        log(f"    ZIP integrity FAIL: {e}")
        sys.exit(1)

    log(f"    ZIP integrity: OK ({len(zip_files)} files)")
    log(f"    ZIP contents:")
    for zn in zip_files:
        log(f"      {zn}")

    # Verify ZIP contains report
    check_zip_report = any("STEP6A_PILOT_REPORT.md" in zn for zn in zip_files)
    check_zip_strong = any("contact_strong_misassignment.png" in zn for zn in zip_files)
    check_zip_script = any("s001_step6_landmarks_2d_v1_4_1.py" in zn for zn in zip_files)
    log(f"\n    ZIP contains report: {check_zip_report}")
    log(f"    ZIP contains strong_misassignment.png: {check_zip_strong}")
    log(f"    ZIP contains packaging script: {check_zip_script}")

    # Final log update
    log(f"\n  Status: REQUIRES_EXTERNAL_PACKAGING_CONFIRMATION")

    # Write final packaging log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    log(f"\n  Saved: {log_path.name}")


if __name__ == "__main__":
    main()
