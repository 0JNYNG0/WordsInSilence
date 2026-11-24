"""Find named source videos in EXPLICITLY given local archives. Preparation, not recognition.

This reads a ZIP's central directory only - the member list, sizes and CRCs that the directory
already holds. Nothing is extracted, no member payload is read, and no network, camera, neural
model or extractor is touched. It answers one question: which archive, if any, contains the exact
MP4 basenames functional_12 named.

Why a new tool: the existing zip_probe.selected_directory streams central directories but filters
to `.json` members containing `_F_` and matches a clip-id prefix, so it cannot look for an MP4
basename. The ZIP mechanics themselves are not re-implemented here - stdlib `zipfile` reads the
central directory (ZIP64 included) without touching payloads.

Matching policy, fixed here so the report is readable:
  * comparison is on the BASENAME only, case-insensitively, with `\\` normalised to `/`;
  * a match whose stored spelling differs only by case is still `found`, with `case_mismatch` set;
  * two or more members matching one target is `ambiguous` - every candidate is listed and NONE is
    chosen;
  * an archive numbered 01 is NOT assumed to hold REAL01, or anything else. Only the member list
    decides.

An input whose name marks a partial or numbered multi-part download - `.part`, `.part1`,
`.part01`, `.crdownload` and the like - is reported as `incomplete` and is never opened. An input
that could not be read is never treated as evidence of absence: when any input is unreadable,
incomplete or missing, a target with no hit is `unknown` rather than `missing`, and the report
carries `completeness` and `inputs_unread` saying so.

Nothing is written until the destination is proved safe: an `--out` that resolves to an input path
is refused, and an existing report is only replaced with `--force`.

With no input path given, the tool does nothing: it never searches a disk.

Run:
  py -3.13 tools_source_video_index.py --zip D:/a.zip --zip D:/b.zip [--out evidence/x.json]
  py -3.13 tools_source_video_index.py --list files.txt      # one explicit video path per line
"""
from __future__ import annotations

import argparse
import json
import os
import re
import zipfile

# the three videos functional_12 named, and nothing else by default
TARGETS = ("NIA_SL_SEN0019_REAL01_F.mp4",
           "NIA_SL_SEN0020_REAL01_F.mp4",
           "NIA_SL_SEN0199_REAL01_F.mp4")
# AI Hub hands out multi-part downloads as .zip.part, .zip.part1, .zip.part01, ... so a plain
# endswith(".part") let every numbered part through as if it were a finished archive.
INCOMPLETE_RE = re.compile(r"\.(part\d*|crdownload|partial|tmp|download)$", re.IGNORECASE)


def is_incomplete(path: str) -> bool:
    return bool(INCOMPLETE_RE.search(path))


def same_file(a: str, b: str) -> bool:
    """True when two spellings name one file, before either is written."""
    ra, rb = os.path.realpath(a), os.path.realpath(b)
    return os.path.normcase(ra) == os.path.normcase(rb)


def norm(name: str) -> str:
    return os.path.basename(name.replace("\\", "/")).casefold()


def scan_zip(path: str, targets) -> dict:
    """Central directory only. Never opens a member."""
    if is_incomplete(path):
        return {"path": path, "status": "incomplete",
                "reason": "the name marks a partial or numbered multi-part download; not opened"}
    if not os.path.isfile(path):
        return {"path": path, "status": "not_found"}
    entry = {"path": path, "bytes": os.path.getsize(path)}
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
    except (zipfile.BadZipFile, OSError) as exc:
        entry.update(status="unreadable", reason=f"{type(exc).__name__}: {exc}")
        return entry
    wanted = {norm(t): t for t in targets}
    hits = {t: [] for t in targets}
    for info in infos:
        if info.is_dir():
            continue
        key = norm(info.filename)
        if key in wanted:
            t = wanted[key]
            hits[t].append({"member": info.filename.replace("\\", "/"),
                            "uncompressed_bytes": int(info.file_size),
                            "compressed_bytes": int(info.compress_size),
                            "crc32": f"{info.CRC:08x}",
                            "case_mismatch": os.path.basename(
                                info.filename.replace("\\", "/")) != t})
    entry.update(status="read", members_total=len(infos),
                 hits={t: hits[t] for t in targets if hits[t]})
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", action="append", default=[],
                    help="an explicit local .zip path; repeatable")
    ap.add_argument("--list", help="a text file of explicit video paths, one per line")
    ap.add_argument("--target", action="append", default=[],
                    help="override the MP4 basenames to look for; repeatable")
    ap.add_argument("--out", default=None, help="where to write the JSON report")
    ap.add_argument("--force", action="store_true",
                    help="allow an existing --out file to be replaced")
    args = ap.parse_args()

    targets = tuple(args.target) or TARGETS
    archives, files = list(args.zip), []
    if args.list:
        with open(args.list, encoding="utf-8") as fh:
            files = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]

    report = {
        "task": "source_video_index",
        "purpose": "identify which explicitly given archive holds the named source videos. "
                   "Preparation for a later comparison; NOT a recognition result and not a "
                   "completed feature.",
        "targets": list(targets),
        "matching_policy": "basename only, case-insensitive, backslashes normalised. A case-only "
                           "difference is still found and is flagged. Two or more matches for one "
                           "target is ambiguous and none is chosen. An archive numbered 01 is not "
                           "assumed to be REAL01.",
        "never": ["extract", "read member payload", "network", "camera", "neural model",
                  "extractor", "disk-wide search"],
        "archives": [], "explicit_files": [], "per_target": {},
    }
    # Nothing is written until the destination is proved safe: an --out that resolves to an input
    # would destroy the very material this tool exists to look at, and replacing an earlier report
    # without being asked loses a record someone may still need.
    if args.out:
        for src in archives + files + ([args.list] if args.list else []):
            if same_file(args.out, src):
                print(json.dumps({"status": "REFUSED_OUTPUT_COLLIDES_WITH_INPUT",
                                  "out": args.out, "input": src,
                                  "reason": "writing the report there would overwrite an input"},
                                 ensure_ascii=False, indent=1))
                return 2
        if os.path.exists(args.out) and not args.force:
            print(json.dumps({"status": "REFUSED_OUTPUT_EXISTS", "out": args.out,
                              "reason": "pass --force to replace an existing report"},
                             ensure_ascii=False, indent=1))
            return 3

    no_input = not archives and not files
    if no_input:
        report["note"] = "nothing was searched. Give --zip or --list; this tool never scans a disk."

    for path in archives:
        report["archives"].append(scan_zip(path, targets))

    for path in files:
        if is_incomplete(path):
            report["explicit_files"].append({"path": path, "status": "incomplete"})
        elif os.path.isfile(path):
            report["explicit_files"].append({"path": path, "status": "present",
                                             "bytes": os.path.getsize(path),
                                             "basename": os.path.basename(path)})
        else:
            report["explicit_files"].append({"path": path, "status": "not_found"})

    # An input that could not be read is not evidence of absence. Track those so a target with no
    # hit can say whether everything given was actually searched.
    report["inputs_unread"] = [
        {"path": a["path"], "status": a["status"], "reason": a.get("reason")}
        for a in report["archives"] if a["status"] != "read"
    ] + [
        {"path": f["path"], "status": f["status"], "reason": None}
        for f in report["explicit_files"] if f["status"] != "present"
    ]
    completeness = "all_inputs_read" if not report["inputs_unread"] else "some_inputs_unread"
    report["completeness"] = completeness
    report["completeness_meaning"] = (
        "all_inputs_read: every path given was opened and its directory listed, so a target with "
        "no hit is genuinely absent from what was given. some_inputs_unread: at least one path "
        "was unreadable, incomplete or not present, so a target with no hit is UNKNOWN, not "
        "missing.")

    for t in targets:
        found = []
        for a in report["archives"]:
            for hit in (a.get("hits") or {}).get(t, []):
                found.append(dict(hit, zip_path=a["path"]))
        for f in report["explicit_files"]:
            if f.get("status") == "present" and norm(f["basename"]) == norm(t):
                found.append({"zip_path": None, "member": f["path"],
                              "uncompressed_bytes": f["bytes"], "crc32": None,
                              "case_mismatch": f["basename"] != t})
        if not found:
            report["per_target"][t] = {
                "status": "missing" if completeness == "all_inputs_read" else "unknown",
                "candidates": [], "completeness": completeness,
                "note": None if completeness == "all_inputs_read" else
                       "no match in what was read, and at least one input could not be read"}
        elif len(found) == 1:
            report["per_target"][t] = {"status": "found", "candidates": found,
                                       "completeness": completeness}
        else:
            report["per_target"][t] = {"status": "ambiguous", "candidates": found,
                                       "completeness": completeness,
                                       "note": "more than one member matches; none was chosen"}
    counts = {}
    for v in report["per_target"].values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    report["summary"] = counts
    report["status"] = ("NO_INPUT_PATH_GIVEN" if no_input else
                        "COMPLETE" if counts.get("found", 0) == len(targets) else
                        "INCOMPLETE_INPUTS" if completeness == "some_inputs_unread" else
                        "NOT_ALL_FOUND")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
        print("written", args.out)
    print(json.dumps({"status": report["status"], "summary": counts,
                      "archives": [{"path": a["path"], "status": a["status"],
                                    "members_total": a.get("members_total")}
                                   for a in report["archives"]]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
