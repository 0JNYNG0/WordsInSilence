#!/usr/bin/env python3
"""Step 1 of the vocabulary expansion: build the segment table, and MEASURE the cost.

The pilot trained a 34,501-parameter TCN on 98 clips of 5 words. It reached 32/32 on the source
validation split and then collapsed to 11/40 on the participant's webcam, predicting GAMSAHAMNIDA
for all 20 ALDA attempts. DTW carried the same 40 clips through the same conversion at 40/40, so
the common representation transfers and what failed is a model with 352 parameters per training
example. More words and more clips is the obvious lever.

This step reads ONLY the 85 MB morpheme annotation archive, which is cheap, and writes a segment
table plus a cost estimate. No keypoint archive is opened and no NPZ is written here, so the
expensive decision is made with measured numbers rather than a guess.

Nothing under the prep folder is modified; it is read-only input.

Run: py -3.13 expand_vocab_scan.py
"""
import collections
import json
import os
import re
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PREP = ("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/"
        "WordsInSilence_학습준비_20260912")
AIHUB = "C:/Users/local-user/Downloads/수어 영상/1.Training"
MORPHEME = os.path.join(AIHUB, "[라벨]01_real_sen_morpheme.zip")
OUT = os.path.join(ROOT, "expand")

# Same filters the pilot used, so the expansion is comparable with what already exists.
MEMBER = re.compile(r"^morpheme/(\d{2})/(NIA_SL_SEN\d+_REAL\d+_F)_morpheme\.json$")
PILOT_WORDS = {"나": "NA", "알다": "ALDA", "좋다": "JOTA", "모르다": "MOREUDA",
               "감사합니다": "GAMSAHAMNIDA"}
MIN_SIGNERS = 14               # of 16; a word must be performed by nearly everyone
MIN_DURATION_S = 0.2           # shorter than this cannot carry a sign at 30 fps
MAX_DURATION_S = 4.0


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    bad = collections.Counter()
    with zipfile.ZipFile(MORPHEME) as z:
        members = [n for n in z.namelist() if MEMBER.match(n)]
        print(f"정면(_F_) 형태소 파일 {len(members)}개 읽는 중...")
        for k, name in enumerate(members):
            m = MEMBER.match(name)
            signer, clip_id = int(m.group(1)), m.group(2)
            try:
                doc = json.loads(z.read(name))
            except Exception:                       # noqa: BLE001 - counted, not hidden
                bad["unreadable"] += 1
                continue
            duration = doc.get("metaData", {}).get("duration")
            for i, a in enumerate(doc.get("data", [])):
                attrs = a.get("attributes") or []
                label = (attrs[0].get("name") if attrs else None)
                s, e = a.get("start"), a.get("end")
                if not label or s is None or e is None:
                    bad["incomplete"] += 1
                    continue
                if not (MIN_DURATION_S <= e - s <= MAX_DURATION_S):
                    bad["duration_out_of_range"] += 1
                    continue
                rows.append({"label": label, "signer_id": signer, "clip_id": clip_id,
                             "annotation_index": i, "start_s": round(float(s), 4),
                             "end_s": round(float(e), 4),
                             "clip_duration_s": duration})
            if (k + 1) % 5000 == 0:
                print(f"  {k + 1}/{len(members)}  구간 {len(rows)}", flush=True)
    print(f"구간 {len(rows)}개, 제외 {dict(bad)}  ({time.perf_counter() - t0:.0f}초)")

    by_label = collections.defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    stats = []
    for label, rs in by_label.items():
        signers = {r["signer_id"] for r in rs}
        stats.append({"label": label, "instances": len(rs), "signers": len(signers),
                      "clips": len({r["clip_id"] for r in rs}),
                      "is_pilot_word": label in PILOT_WORDS})
    stats.sort(key=lambda s: (-s["signers"], -s["instances"]))
    eligible = [s for s in stats if s["signers"] >= MIN_SIGNERS]
    print(f"\n어휘 {len(stats)}개 중 수행자 {MIN_SIGNERS}명 이상 = {len(eligible)}개")
    print(f"파일럿 5단어 포함 여부: "
          f"{[s['label'] for s in eligible if s['is_pilot_word']]}")

    # cost: unique clips is what drives decoding, since one sentence yields several words
    print(f"\n{'단어 수':>8} {'구간':>8} {'고유 클립':>10} {'수행자당 상한 3건 적용시 구간':>26}")
    plans = {}
    for n in (20, 40, 60, 80, 100, 150, len(eligible)):
        if n > len(eligible):
            continue
        pick = {s["label"] for s in eligible[:n]}
        sel = [r for r in rows if r["label"] in pick]
        capped, seen = [], collections.Counter()
        for r in sorted(sel, key=lambda r: (r["label"], r["signer_id"], r["clip_id"],
                                            r["annotation_index"])):
            key = (r["label"], r["signer_id"])
            if seen[key] < 3:
                seen[key] += 1
                capped.append(r)
        plans[n] = {"words": sorted(pick), "intervals": len(sel),
                    "unique_clips": len({r["clip_id"] for r in sel}),
                    "capped_intervals": len(capped),
                    "capped_unique_clips": len({r["clip_id"] for r in capped})}
        p = plans[n]
        print(f"{n:>8} {p['intervals']:>8} {p['unique_clips']:>10} "
              f"{p['capped_intervals']:>12} (클립 {p['capped_unique_clips']})")

    out = {
        "task_id": "expand_vocab_scan", "date_kst": time.strftime("%Y-%m-%d"),
        "purpose": "measure the cost of expanding the training vocabulary before opening any "
                   "keypoint archive",
        "command": "py -3.13 expand_vocab_scan.py",
        "read_only_source": MORPHEME,
        "filters": {"front_view_only": True, "min_signers": MIN_SIGNERS,
                    "min_duration_s": MIN_DURATION_S, "max_duration_s": MAX_DURATION_S},
        "totals": {"annotation_files": len(members), "intervals": len(rows),
                   "excluded": dict(bad), "distinct_labels": len(stats),
                   "labels_meeting_signer_floor": len(eligible)},
        "vocabulary": stats[:400],
        "plans": plans,
        "limits": [
            "a label match is not proof of the same lexical variant; the pilot recorded the same "
            "caveat and it still applies",
            "intervals come from sentences, so they carry coarticulation from neighbouring signs",
            "durations use the annotation's own start and end; original video PTS is unverified",
            "no keypoint archive was opened here, so no clip is confirmed to have keypoints yet",
        ],
        "nothing_extracted_yet": True,
    }
    p = os.path.join(OUT, "vocab_scan.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    pr = os.path.join(OUT, "segments_all.json")
    with open(pr, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rows, fh, ensure_ascii=False)
        fh.write("\n")
    print(f"\n({time.perf_counter() - t0:.0f}초)  -> expand/vocab_scan.json , expand/segments_all.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
