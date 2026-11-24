#!/usr/bin/env python3
"""Step 2: extract keypoints for an expanded vocabulary, so the TCN has something to learn from.

The pilot gave a 34,501-parameter model 98 training clips of 5 words - 352 parameters per example.
It scored 32/32 on the source validation split and then predicted GAMSAHAMNIDA for all 20 of the
participant's ALDA attempts (11/40 overall) while DTW carried the same clips through the same
conversion at 40/40. The representation transfers; the model memorised its 98 examples. This builds
a training set roughly forty times larger.

Design choices and why:
  * 80 words, each performed by at least 14 of the 16 signers, so every class has cross-signer
    variation rather than one person's habit.
  * at most 3 instances per (word, signer). Sentence corpora are wildly unbalanced - one word
    appears 4,000 times and another 60 - and an uncapped set would teach frequency, not form. The
    cap also bounds the decode cost.
  * the pilot's signer split is kept exactly: train 01-09, validation 11-13, reserved_test 14-16.
    reserved_test is written to disk and never loaded by anything here, so it stays clean for this
    vocabulary even though the old 163-clip set is already design-exposed
    (see evidence/exposure_record.json).

Read-only against the AIHub archives and the prep folder. No production edit, no web recogniser
change, no training in this script.

Run: py -3.13 expand_build.py
"""
import collections
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PREP = ("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/"
        "WordsInSilence_학습준비_20260912")
sys.path.insert(0, PREP)
from zip_probe import read_member, selected_directory  # noqa: E402

AIHUB = "C:/Users/local-user/Downloads/수어 영상/1.Training"
OUT = os.path.join(ROOT, "expand")
NPZ = os.path.join(OUT, "npz")
FPS = 30.0                      # nominal; original video PTS is NOT verified, same as the pilot
N_WORDS = 80
CAP_PER_WORD_SIGNER = 3
MIN_SIGNERS = 14
MIN_FRAMES = 6
SPLITS = {"train": list(range(1, 10)), "validation": [11, 12, 13],
          "reserved_test": [14, 15, 16]}
FIELDS = [("pose_keypoints_2d", 25), ("hand_left_keypoints_2d", 21),
          ("hand_right_keypoints_2d", 21)]
FRAME_RE = re.compile(r"_F_(\d+)_keypoints\.json$")

# the 19 words this project teaches, in AIHub label form. Only these five exist there with
# enough signers; 인사, 사랑, 자다 and the rest are absent from the sentence corpus.
PROJECT_WORDS = ("알다", "감사합니다", "나", "좋다", "모르다")


def split_for(signer):
    return next(k for k, v in SPLITS.items() if signer in v)


def choose_words(rows):
    """The project's own words first, then the best-covered remainder.

    Ranking purely by how often a label occurs fills the list with sentence connectives - the top
    twenty are 곳, 다음, 내리다, 도착, 가다 and so on - and drops every word this project actually
    teaches. 알다 and 감사합니다 are the only two the participant recorded that AIHub also has, so
    without them there is no way to retest the transfer failure that motivated this build.
    """
    by = collections.defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    stats = [(len({x["signer_id"] for x in v}), len(v), w) for w, v in by.items()]
    stats.sort(key=lambda t: (-t[0], -t[1], t[2]))
    eligible = [w for s, n, w in stats if s >= MIN_SIGNERS]
    forced = [w for w in PROJECT_WORDS if w in eligible]
    rest = [w for w in eligible if w not in forced]
    return forced + rest[:max(0, N_WORDS - len(forced))]


def cap(rows, words):
    keep, seen = [], collections.Counter()
    for r in sorted(rows, key=lambda r: (r["label"], r["signer_id"], r["clip_id"],
                                         r["annotation_index"])):
        if r["label"] not in words:
            continue
        k = (r["label"], r["signer_id"])
        if seen[k] < CAP_PER_WORD_SIGNER:
            seen[k] += 1
            keep.append(r)
    return keep


def decode(handle, member):
    doc = json.loads(read_member(handle, member))
    p = doc["people"]
    if not isinstance(p, dict):
        raise ValueError("unexpected people schema")
    vals = []
    for name, n in FIELDS:
        a = np.asarray(p[name], dtype=np.float32)
        if a.shape != (n * 3,):
            raise ValueError(f"bad {name} shape {a.shape}")
        vals.append(a.reshape(n, 3))
    xyc = np.concatenate(vals, axis=0)
    if not np.isfinite(xyc).all():
        raise ValueError("nonfinite source array")
    if np.any((xyc[:, 2] < 0) | (xyc[:, 2] > 1)):
        raise ValueError("confidence outside [0,1]")
    return xyc


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(NPZ, exist_ok=True)
    rows = json.load(open(os.path.join(OUT, "segments_all.json"), encoding="utf-8"))
    words = choose_words(rows)
    selected = cap(rows, set(words))
    classes = sorted(words)
    index = {w: i for i, w in enumerate(classes)}
    print(f"단어 {len(words)}개, 구간 {len(selected)}개, "
          f"고유 클립 {len({r['clip_id'] for r in selected})}개")

    by_signer = collections.defaultdict(list)
    for r in selected:
        by_signer[r["signer_id"]].append(r)

    manifest, problems, archives = [], [], []
    for signer in sorted(by_signer):
        rs = by_signer[signer]
        arc = os.path.join(AIHUB, f"[라벨]{signer:02d}_real_sen_keypoint.zip")
        if not os.path.exists(arc):
            problems.append({"signer_id": signer, "error": "ARCHIVE_MISSING", "archive": arc})
            continue
        before = os.stat(arc)
        wanted_clips = {r["clip_id"] for r in rs}
        members, dstats = selected_directory(__import__("pathlib").Path(arc), wanted_clips)
        by_clip = collections.defaultdict(dict)
        for m in members:
            mm = FRAME_RE.search(m["name"])
            if mm:
                by_clip[m["clip_id"]][int(mm.group(1))] = m
        made = 0
        with open(arc, "rb") as fh:
            for clip_id, group in itertools.groupby(
                    sorted(rs, key=lambda r: (r["clip_id"], r["annotation_index"])),
                    key=lambda r: r["clip_id"]):
                frame_map = by_clip.get(clip_id, {})
                cache = {}
                for row in group:
                    sid = f"{clip_id}_a{row['annotation_index']:02d}_{index[row['label']]:03d}"
                    if not frame_map:
                        problems.append({"sample_id": sid, "error": "CLIP_MISSING"})
                        continue
                    start = math.ceil(row["start_s"] * FPS - 1e-9)
                    stop = math.ceil(row["end_s"] * FPS - 1e-9)
                    wanted = list(range(start, stop))
                    absent = [i for i in wanted if i not in frame_map]
                    if absent or len(wanted) < MIN_FRAMES:
                        problems.append({"sample_id": sid, "error": "MISSING_OR_TOO_FEW_FRAMES",
                                         "missing": len(absent), "requested": len(wanted)})
                        continue
                    try:
                        arrs = []
                        for i in wanted:
                            if i not in cache:
                                cache[i] = decode(fh, frame_map[i])
                            arrs.append(cache[i])
                    except Exception as exc:                     # noqa: BLE001 - recorded
                        problems.append({"sample_id": sid, "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    xyc = np.stack(arrs)
                    present = (np.isfinite(xyc).all(axis=-1)
                               & ~np.all(xyc[..., :2] == 0, axis=-1))
                    idx = np.asarray(wanted, dtype=np.int64)
                    dest = os.path.join(NPZ, sid + ".npz")
                    np.savez_compressed(
                        dest, xy_native=xyc[:, :, :2], confidence_native=xyc[:, :, 2],
                        coordinate_present=present, frame_index=idx,
                        timestamp_nominal_us=np.rint(idx / FPS * 1e6).astype(np.int64))
                    with np.load(dest, allow_pickle=False) as chk:
                        assert chk["xy_native"].shape == (len(wanted), 67, 2)
                        assert np.array_equal(chk["coordinate_present"], present)
                        assert np.all(np.diff(chk["timestamp_nominal_us"]) > 0)
                    manifest.append({
                        "sample_id": sid, "word_label": row["label"],
                        "class_index": index[row["label"]], "signer_id": signer,
                        "split_role": split_for(signer), "clip_id": clip_id,
                        "annotation_index": row["annotation_index"],
                        "start_s": row["start_s"], "end_s": row["end_s"],
                        "frame_count": len(wanted), "first_frame": start, "last_frame": stop - 1,
                        "nominal_fps": FPS,
                        "npz": os.path.relpath(dest, ROOT).replace("\\", "/"),
                        "npz_sha256": hashlib.sha256(open(dest, "rb").read()).hexdigest(),
                        "hand_present_fraction": {
                            "left": round(float(present[:, 25:46].mean()), 4),
                            "right": round(float(present[:, 46:67].mean()), 4)},
                        "source_schema": "OpenPose BODY25 + left21 + right21, 2D plus confidence",
                        "timebase": "frame_index / 30; original PTS unverified",
                        "crc_verified": True, "semantic_variant_verified": False,
                    })
                    made += 1
                cache.clear()
        after = os.stat(arc)
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), \
            "source archive changed while reading"
        archives.append({"signer_id": signer, "archive": arc, "bytes": before.st_size,
                         "directory": dstats, "selected_members": len(members),
                         "samples_written": made})
        print(f"  수행자 {signer:02d} ({split_for(signer):<13}) 표본 {made:>5}  "
              f"디렉터리 {dstats['seconds']}초  누적 {time.perf_counter() - t0:.0f}초", flush=True)

    counts = collections.Counter((m["word_label"], m["split_role"]) for m in manifest)
    per_split = collections.Counter(m["split_role"] for m in manifest)
    for ident in ("signer_id", "clip_id"):
        sets = {s: {m[ident] for m in manifest if m["split_role"] == s} for s in SPLITS}
        for a in sets:
            for b in sets:
                assert a == b or not sets[a] & sets[b], f"{ident} leakage {a}/{b}"
    covered = {w for w, _ in counts}
    thin = sorted(w for w in covered
                  if min(counts[(w, s)] for s in SPLITS) == 0)

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, ensure_ascii=False)
        fh.write("\n")
    report = {
        "task_id": "expand_build", "date_kst": time.strftime("%Y-%m-%d"),
        "purpose": "enlarge the training set so the TCN has more to learn from than 98 clips",
        "command": "py -3.13 expand_build.py",
        "settings": {"words": N_WORDS, "min_signers": MIN_SIGNERS,
                     "cap_per_word_signer": CAP_PER_WORD_SIGNER, "min_frames": MIN_FRAMES,
                     "nominal_fps": FPS, "splits": SPLITS,
                     "split_policy": "identical to the pilot's signer split, so results stay "
                                     "comparable and no signer crosses a boundary"},
        "classes": classes,
        "written_samples": len(manifest),
        "samples_per_split": dict(per_split),
        "words_missing_a_split": thin,
        "problems": problems[:200], "problem_count": len(problems),
        "archives": archives,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "reserved_test_written_but_never_loaded_here": True,
        "limits": [
            "a label match is not proof of the same lexical variant",
            "intervals are cut from sentences and carry coarticulation",
            "timing uses nominal 30 fps; original video PTS is unverified",
            "OpenPose native confidence is constant 1, so only coordinate presence is used",
            "this is public source data; nothing here is a webcam or new-person performance claim",
        ],
        "no_model_trained_here": True,
    }
    with open(os.path.join(OUT, "build_report.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n표본 {len(manifest)}개  분할 {dict(per_split)}  문제 {len(problems)}건")
    if thin:
        print(f"  분할 하나가 비어 있는 단어 {len(thin)}개: {thin[:8]}")
    print(f"({report['elapsed_s']}초)  -> expand/manifest.json , expand/build_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
