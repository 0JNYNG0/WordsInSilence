#!/usr/bin/env python3
"""
Repackage step6b v1.2.4 full delivery into multiple independent ZIPs (each <= 450MB).
Does NOT modify or delete the original output directory or the original ZIP.
"""

import csv
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "data" / "processed" / "sessions" / "S001_P001" / "landmarks_3d_v1" / "full_v1_2_4"
SCRIPT_SRC = PROJECT / "Tools" / "s001_step6b_triangulate_full_v1_2_4.py"
ORIG_ZIP = PROJECT / "WordsInSilence_step6b_v1_2_4_full_delivery.zip"

MAX_ZIP_BYTES = 450 * 1024 * 1024  # 450 MB

# ---- helpers ----

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def classify_file(rel):
    """Return a human-readable artifact classification."""
    if rel.startswith("raw_3d/"):
        return "3d_landmark_json"
    if rel.startswith("overlays/"):
        return "reprojection_overlay_video"
    if rel.startswith("contact_sheets/"):
        return "visual_contact_sheet"
    ext = Path(rel).suffix.lower()
    name = Path(rel).name.lower()
    if name == "delivery_readme.md":
        return "delivery_readme"
    if name.endswith(".py"):
        return "execution_script"
    if ext == ".csv":
        return "qa_csv"
    if ext == ".json":
        return "qa_json"
    if ext == ".md":
        return "report"
    return "other"


# ---- collect all source files ----

MANIFEST_FILES = {"full_delivery_file_manifest.csv", "full_delivery_file_manifest.json"}

def collect_files():
    """Return list of (rel_path, abs_path) for all files in OUT_DIR + the script.
    Excludes manifest files (generated fresh each run)."""
    files = []
    for root, _dirs, fnames in os.walk(OUT_DIR):
        for fn in fnames:
            abspath = Path(root) / fn
            rel = abspath.relative_to(OUT_DIR).as_posix()
            if rel in MANIFEST_FILES:
                continue
            files.append((rel, abspath))
    # Add the execution script
    files.append(("s001_step6b_triangulate_full_v1_2_4.py", SCRIPT_SRC))
    files.sort(key=lambda x: x[0])
    return files


# ---- assign files to ZIPs ----

def assign_zips(files):
    """
    Assign each file to a ZIP name. Returns dict: zip_name -> [(rel, abspath), ...]

    Strategy:
    - core: script, config, raw_3d JSON, all CSVs, JSONs, report, manifest
    - visuals_1: overlays for GREETING + LOVE words
    - visuals_2: overlays for THANK_YOU word + contact sheets
    """
    zips = {
        "core": [],
        "visuals_1": [],
        "visuals_2": [],
    }

    for rel, abspath in files:
        if rel.startswith("overlays/"):
            # Split overlays by word
            if "THANK_YOU" in rel:
                zips["visuals_2"].append((rel, abspath))
            elif "GREETING" in rel:
                zips["visuals_1"].append((rel, abspath))
            else:  # LOVE
                zips["visuals_1"].append((rel, abspath))
        elif rel.startswith("contact_sheets/"):
            zips["visuals_2"].append((rel, abspath))
        else:
            # Everything else goes to core
            zips["core"].append((rel, abspath))

    return zips


# ---- DELIVERY_README.md content ----

def make_readme(zip_assignments):
    lines = [
        "# Step 6B v1.2.4 Full 45-Sample Delivery",
        "",
        "This delivery is split into multiple independent ZIP files due to upload size limits.",
        "Each ZIP can be opened and inspected independently.",
        "",
        "## ZIP Contents",
        "",
    ]

    prefix = "WordsInSilence_step6b_v1_2_4_full_delivery_"

    for zname in sorted(zip_assignments.keys()):
        flist = zip_assignments[zname]
        lines.append(f"### {prefix}{zname}.zip")
        lines.append("")
        if zname == "core":
            lines.append("Numerical/data artifacts for review:")
            lines.append("- Execution script (`s001_step6b_triangulate_full_v1_2_4.py`)")
            lines.append("- Algorithm config (`config.json`)")
            lines.append("- Raw 3D landmark JSONs (45 samples, `raw_3d/`)")
            lines.append("- All QA CSVs (frame_landmark_qa, appearance_records, temporal_anomalies, etc.)")
            lines.append("- All QA JSONs (assertion_results, qa_flag_analysis, reproj_distributions, etc.)")
            lines.append("- Run manifest, sample QA summary, bone data")
            lines.append("- Full report (`STEP6B_FULL_V1_2_4_REPORT.md`)")
        elif zname == "visuals_1":
            lines.append("Visual review artifacts (GREETING + LOVE):")
            lines.append("- Reprojection overlay videos for GREETING samples (15 samples x 3 views)")
            lines.append("- Reprojection overlay videos for LOVE samples (15 samples x 3 views)")
        elif zname == "visuals_2":
            lines.append("Visual review artifacts (THANK_YOU + contact sheets):")
            lines.append("- Reprojection overlay videos for THANK_YOU samples (15 samples x 3 views)")
            lines.append("- Contact sheet images and index CSV")
        lines.append(f"- File count: {len(flist)} (+ this README)")
        lines.append("")

    lines.append("## Integrity Verification")
    lines.append("")
    lines.append("A unified manifest (`full_delivery_file_manifest.csv` / `.json`) is included in the **core** ZIP.")
    lines.append("It lists every file across all ZIPs with SHA-256 hashes for verification.")
    lines.append("")
    lines.append("## Original Execution Results")
    lines.append("")
    lines.append("- Samples: 45")
    lines.append("- Assertions: **452 PASS / 0 FAIL**")
    lines.append("- tri_valid: 329,536 / 350,100")
    lines.append("- qa_valid: 325,391 / 350,100")
    lines.append("- template_eligible: 298,581 / 350,100")
    lines.append("")
    return "\n".join(lines)


# ---- main ----

def main():
    print("Step 6B v1.2.4 Full Delivery - Repackaging")
    print()

    # 1. Collect files
    files = collect_files()
    print(f"  Source files: {len(files)}")

    # 2. Assign to ZIPs
    zip_assignments = assign_zips(files)
    for zname, flist in sorted(zip_assignments.items()):
        total = sum(os.path.getsize(p) for _, p in flist)
        print(f"    {zname}: {len(flist)} files, {total / 1e6:.1f} MB uncompressed")

    # 3. Build README
    readme_content = make_readme(zip_assignments)

    # 4. Compute SHA-256 for all source files
    print("\n  Computing SHA-256 hashes...")
    file_hashes = {}
    for rel, abspath in files:
        file_hashes[rel] = sha256_file(abspath)
    print(f"    Hashed {len(file_hashes)} files")

    # 5. Build manifest data
    manifest_rows = []
    for rel, abspath in files:
        size = os.path.getsize(abspath)
        sha = file_hashes[rel]
        # Find which ZIP it's in
        zip_name = None
        for zname, flist in zip_assignments.items():
            if any(r == rel for r, _ in flist):
                zip_name = f"WordsInSilence_step6b_v1_2_4_full_delivery_{zname}.zip"
                break
        manifest_rows.append({
            "relative_path": rel,
            "size_bytes": size,
            "sha256": sha,
            "zip_file": zip_name,
            "artifact_type": classify_file(rel),
        })

    # Add README entries (in all ZIPs)
    readme_sha = sha256_bytes(readme_content.encode("utf-8"))
    readme_size = len(readme_content.encode("utf-8"))
    for zname in zip_assignments:
        manifest_rows.append({
            "relative_path": "DELIVERY_README.md",
            "size_bytes": readme_size,
            "sha256": readme_sha,
            "zip_file": f"WordsInSilence_step6b_v1_2_4_full_delivery_{zname}.zip",
            "artifact_type": "delivery_readme",
        })

    # Sort manifest
    manifest_rows.sort(key=lambda r: (r["zip_file"], r["relative_path"]))

    # 6. Write manifest CSV and JSON to OUT_DIR (so core ZIP includes them)
    manifest_csv_path = OUT_DIR / "full_delivery_file_manifest.csv"
    manifest_json_path = OUT_DIR / "full_delivery_file_manifest.json"

    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path", "size_bytes", "sha256", "zip_file", "artifact_type"])
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"  Wrote {manifest_csv_path.name} ({len(manifest_rows)} rows)")

    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump({"files": manifest_rows, "total_unique_files": len(files) + 1}, f, indent=2)
    print(f"  Wrote {manifest_json_path.name}")

    # Add manifest files to core assignment (they were just created)
    zip_assignments["core"].append(("full_delivery_file_manifest.csv", manifest_csv_path))
    zip_assignments["core"].append(("full_delivery_file_manifest.json", manifest_json_path))

    # Also add manifest entries for the manifest files themselves
    for mpath, mname in [(manifest_csv_path, "full_delivery_file_manifest.csv"),
                          (manifest_json_path, "full_delivery_file_manifest.json")]:
        msha = sha256_file(mpath)
        msize = os.path.getsize(mpath)
        file_hashes[mname] = msha
        # Update manifest JSON with self-reference
        manifest_rows.append({
            "relative_path": mname,
            "size_bytes": msize,
            "sha256": msha,
            "zip_file": "WordsInSilence_step6b_v1_2_4_full_delivery_core.zip",
            "artifact_type": "delivery_manifest",
        })

    # Re-write manifest files with self-references
    manifest_rows.sort(key=lambda r: (r["zip_file"], r["relative_path"]))
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path", "size_bytes", "sha256", "zip_file", "artifact_type"])
        w.writeheader()
        w.writerows(manifest_rows)
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump({"files": manifest_rows, "total_unique_files": len(files) + 1 + 2}, f, indent=2)
    # Update hashes after rewrite
    file_hashes["full_delivery_file_manifest.csv"] = sha256_file(manifest_csv_path)
    file_hashes["full_delivery_file_manifest.json"] = sha256_file(manifest_json_path)

    # 7. Create ZIPs
    print("\n  Creating ZIPs...")
    zip_paths = {}
    zip_sizes = {}
    zip_file_counts = {}

    for zname in sorted(zip_assignments.keys()):
        flist = zip_assignments[zname]
        zip_filename = f"WordsInSilence_step6b_v1_2_4_full_delivery_{zname}.zip"
        zip_path = PROJECT / zip_filename
        zip_paths[zname] = zip_path

        # Choose compression: deflate for text, stored for mp4/png
        with zipfile.ZipFile(zip_path, "w") as zf:
            # Add README
            zf.writestr("DELIVERY_README.md", readme_content, compress_type=zipfile.ZIP_DEFLATED)
            count = 1

            for rel, abspath in sorted(flist, key=lambda x: x[0]):
                ext = Path(rel).suffix.lower()
                if ext in (".mp4", ".png", ".jpg"):
                    comp = zipfile.ZIP_STORED
                else:
                    comp = zipfile.ZIP_DEFLATED
                zf.write(abspath, rel, compress_type=comp)
                count += 1

            zip_file_counts[zname] = count

        zsize = os.path.getsize(zip_path)
        zip_sizes[zname] = zsize
        print(f"    {zip_filename}: {zsize / 1e6:.1f} MB, {zip_file_counts[zname]} files")

    # 8. Assertions
    print("\n  Running packaging assertions...")
    n_pass = 0
    n_fail = 0
    assertion_results = []

    def chk(cond, name, detail=""):
        nonlocal n_pass, n_fail
        status = "PASS" if cond else "FAIL"
        if not cond:
            n_fail += 1
        else:
            n_pass += 1
        assertion_results.append({"name": name, "status": status, "detail": detail})
        tag = "PASS" if cond else "FAIL"
        if not cond:
            print(f"    [{tag}] {name}: {detail}")

    # A1. Each ZIP <= 450 MB
    for zname, zpath in zip_paths.items():
        zsize = zip_sizes[zname]
        chk(zsize <= MAX_ZIP_BYTES,
            f"zip_size_{zname}",
            f"{zsize / 1e6:.1f} MB <= 450 MB")

    # A2. All ZIPs open and pass CRC
    for zname, zpath in zip_paths.items():
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                bad = zf.testzip()
                chk(bad is None, f"zip_crc_{zname}", f"bad_file={bad}" if bad else "CRC OK")
        except Exception as e:
            chk(False, f"zip_crc_{zname}", str(e))

    # A3. Collect all files from all ZIPs (excluding README duplicates)
    all_zip_files = {}  # rel -> {zip_name, data_hash}
    readme_in_zips = set()

    for zname, zpath in zip_paths.items():
        with zipfile.ZipFile(zpath, "r") as zf:
            for info in zf.infolist():
                rel = info.filename
                if rel == "DELIVERY_README.md":
                    readme_in_zips.add(zname)
                    continue
                data = zf.read(rel)
                h = sha256_bytes(data)
                if rel in all_zip_files:
                    # Duplicate!
                    all_zip_files[rel]["dup_zips"].append(zname)
                else:
                    all_zip_files[rel] = {"zip": zname, "hash": h, "size": len(data), "dup_zips": []}

    # A4. README in all ZIPs (allowed duplicate)
    chk(len(readme_in_zips) == len(zip_paths),
        "readme_in_all_zips",
        f"found_in={len(readme_in_zips)}/{len(zip_paths)}")

    # A5. Build expected file set from source (including manifests)
    expected_rels = set()
    for rel, _ in files:
        expected_rels.add(rel)
    expected_rels.add("full_delivery_file_manifest.csv")
    expected_rels.add("full_delivery_file_manifest.json")

    actual_rels = set(all_zip_files.keys())

    # A6. No missing files
    missing = expected_rels - actual_rels
    chk(len(missing) == 0, "no_missing_files", f"missing={sorted(missing)}" if missing else "0 missing")

    # A7. No unexpected extra files
    extra = actual_rels - expected_rels
    chk(len(extra) == 0, "no_extra_files", f"extra={sorted(extra)}" if extra else "0 extra")

    # A8. No duplicates (excluding README)
    dups = {rel: info for rel, info in all_zip_files.items() if info["dup_zips"]}
    chk(len(dups) == 0, "no_duplicate_files",
        f"duplicates={sorted(dups.keys())}" if dups else "0 duplicates")

    # A9. SHA-256 match for all files
    hash_mismatches = []
    for rel, info in all_zip_files.items():
        if rel in file_hashes:
            if info["hash"] != file_hashes[rel]:
                hash_mismatches.append(rel)
    chk(len(hash_mismatches) == 0, "sha256_all_match",
        f"mismatches={hash_mismatches}" if hash_mismatches else "0 mismatches")

    # A10. Original assertion_results.json still shows 452 PASS / 0 FAIL
    ar_path = OUT_DIR / "assertion_results.json"
    if ar_path.exists():
        with open(ar_path, encoding="utf-8") as f:
            ar = json.load(f)
        orig_pass = sum(1 for r in ar if r.get("status") == "PASS")
        orig_fail = sum(1 for r in ar if r.get("status") == "FAIL")
        chk(orig_pass == 452 and orig_fail == 0,
            "original_assertions_preserved",
            f"{orig_pass} PASS / {orig_fail} FAIL")
    else:
        chk(False, "original_assertions_preserved", "assertion_results.json not found")

    # A11. File counts match
    total_unique = len(actual_rels)
    chk(total_unique == len(expected_rels),
        "file_count_match",
        f"zip_unique={total_unique} expected={len(expected_rels)}")

    # A12. Verify against original ZIP (normalize path prefixes)
    if ORIG_ZIP.exists():
        orig_rels_raw = set()
        with zipfile.ZipFile(ORIG_ZIP, "r") as zf:
            for info in zf.infolist():
                orig_rels_raw.add(info.filename)
        # Strip known prefixes to get comparable basenames
        PREFIX = "landmarks_3d_v1/full_v1_2_4/"
        SCRIPT_PREFIX = "Tools/"
        orig_normalized = set()
        for r in orig_rels_raw:
            if r.startswith(PREFIX):
                orig_normalized.add(r[len(PREFIX):])
            elif r.startswith(SCRIPT_PREFIX):
                orig_normalized.add(r[len(SCRIPT_PREFIX):])
            else:
                orig_normalized.add(r)
        orig_content_rels = orig_normalized - {"DELIVERY_README.md"}
        new_content_rels = actual_rels - {"full_delivery_file_manifest.csv", "full_delivery_file_manifest.json"}
        orig_only = orig_content_rels - new_content_rels
        new_only = new_content_rels - orig_content_rels
        chk(len(orig_only) == 0,
            "orig_zip_coverage",
            f"in_orig_not_split={sorted(orig_only)}" if orig_only else "all original files covered")
        if new_only:
            # These should only be the manifest files
            unexpected_new = new_only - {"full_delivery_file_manifest.csv", "full_delivery_file_manifest.json"}
            chk(len(unexpected_new) == 0,
                "no_unexpected_new_vs_orig",
                f"unexpected={sorted(unexpected_new)}" if unexpected_new else "OK")

    print(f"\n  Packaging assertions: {n_pass} PASS, {n_fail} FAIL")

    # 9. Compute ZIP hashes
    print("\n  ZIP SHA-256:")
    zip_hashes = {}
    for zname in sorted(zip_paths.keys()):
        zpath = zip_paths[zname]
        zh = sha256_file(zpath)
        zip_hashes[zname] = zh
        print(f"    {zpath.name}: {zh}")

    # 10. Summary
    print("\n  ============================================================")
    print("  REPACKAGING SUMMARY")
    print("  ============================================================")
    for zname in sorted(zip_paths.keys()):
        zpath = zip_paths[zname]
        print(f"  {zpath.name}")
        print(f"    Size: {zip_sizes[zname] / 1e6:.1f} MB")
        print(f"    SHA-256: {zip_hashes[zname]}")
        print(f"    Files: {zip_file_counts[zname]}")
    print(f"  Total unique files (excl README): {total_unique}")
    print(f"  Missing: {len(missing)}")
    print(f"  Duplicates: {len(dups)}")
    print(f"  SHA-256 mismatches: {len(hash_mismatches)}")
    print(f"  ZIP CRC: all passed" if n_fail == 0 else f"  ZIP CRC: ISSUES")
    print(f"  Original assertions: 452 PASS / 0 FAIL preserved")
    print(f"  Packaging assertions: {n_pass} PASS / {n_fail} FAIL")
    print()

    if n_fail > 0:
        print("  STATUS: REPACKAGING_BLOCKED")
        sys.exit(1)
    else:
        print("  STATUS: REPACKAGING_COMPLETE")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
