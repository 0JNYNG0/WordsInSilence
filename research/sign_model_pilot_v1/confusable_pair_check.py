#!/usr/bin/env python3
"""How far apart are the five pilot words for a FLUENT signer, measured within that signer?

The ratio check used ALDA vs GAMSAHAMNIDA as its denominator, which was an arbitrary choice. A
tolerance has to survive the NEAREST word, not an average one: the participant points out that NA
(palm held still at the chest) and ALDA (the same palm moving up and down at the chest) differ only
in movement. If fluent signers barely separate that pair, no tolerance borrowed from them can be
both wide enough to accept a learner and tight enough to reject the neighbour.

Every distance here is between two clips of the SAME signer, so signer identity never inflates it.
Source clips only (OpenPose, AIHub). No web recording, no camera, no model.

Run: py -3.13 confusable_pair_check.py
"""
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pilot import data as D  # noqa: E402
from pilot.dtw_baseline import dtw_distance  # noqa: E402

PREP = ("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/"
        "WordsInSilence_학습준비_20260912")


def med(v):
    return float(np.median(v)) if len(v) else None


def main() -> int:
    t0 = time.perf_counter()
    items = [x for x in D.load_manifest(PREP) if x["source_only_pilot_eligible"]]
    by = {}
    for it in items:
        c = D.load_clip(PREP, it)
        by.setdefault(it["word_id"], {}).setdefault(str(c["signer_id"]), []).append(c)

    within = {}
    for w, sg in by.items():
        d = [dtw_distance(a["x"], b["x"])
             for clips in sg.values() for a, b in itertools.combinations(clips, 2)]
        d = [x for x in d if x is not None]
        within[w] = {"n_pairs": len(d), "median": med(d)}

    between = {}
    for w1, w2 in itertools.combinations(sorted(by), 2):
        d = []
        for sid in set(by[w1]) & set(by[w2]):
            for a in by[w1][sid]:
                for b in by[w2][sid]:
                    x = dtw_distance(a["x"], b["x"])
                    if x is not None:
                        d.append(x)
        between[f"{w1}|{w2}"] = {"n_pairs": len(d), "signers": len(set(by[w1]) & set(by[w2])),
                                 "median": med(d)}

    words = sorted(by)
    print("같은 수행자 안에서만 잰 단어 간 거리 (중앙값)")
    print(f"{'':<14}" + "".join(f"{w:>14}" for w in words))
    for w1 in words:
        row = f"{w1:<14}"
        for w2 in words:
            if w1 == w2:
                v = within[w1]["median"]
                row += f"{('[' + format(v, '.3f') + ']') if v else '[  -  ]':>14}"
            else:
                k = "|".join(sorted((w1, w2)))
                row += f"{between[k]['median']:>14.3f}"
        print(row)
    print("  대각선 [ ] = 같은 단어 반복끼리의 흩어짐 (반복이 없는 단어는 -)")

    print("\n단어별: 가장 가까운 다른 단어 = 허용범위가 실제로 상대해야 할 값")
    nearest = {}
    for w in words:
        others = [(between["|".join(sorted((w, o)))]["median"], o) for o in words if o != w]
        dist, who = min(others)
        sp = within[w]["median"]
        nearest[w] = {"nearest_word": who, "distance": round(dist, 5),
                      "own_spread": round(sp, 5) if sp else None,
                      "headroom": round(dist / sp, 3) if sp else None}
        sp_s = f"{sp:.3f}" if sp else "측정불가"
        hr = f"{dist / sp:.2f}배" if sp else "-"
        print(f"  {w:<14} 가장 가까운 단어 {who:<14} 거리 {dist:.3f}"
              f"   자기 흩어짐 {sp_s:>8}   여유 {hr}")

    out = {
        "task_id": "confusable_pair_check", "date_kst": time.strftime("%Y-%m-%d"),
        "question": "for a fluent signer, how close is each pilot word to its nearest neighbour, "
                    "relative to the spread among that signer's own repeats",
        "why_it_matters": "the earlier ratio check normalised by an arbitrary word pair. A "
                          "tolerance must survive the nearest word, and NA vs ALDA differ only in "
                          "movement",
        "command": "py -3.13 confusable_pair_check.py",
        "source": "AIHub OpenPose pilot pack, 163 eligible clips, all distances within one signer",
        "within_word_spread": within,
        "between_word_distance": between,
        "nearest_neighbour": nearest,
        "limits": [
            "GAMSAHAMNIDA and MOREUDA have one clip per signer, so their own spread is not "
            "measurable here and their headroom is unknown",
            "clips are cut from sentences, so a neighbour distance carries coarticulation",
            "fluent signers only; says nothing about how a learner performs these words",
            "not a correctness claim about any attempt",
        ],
        "not_a_correctness_claim": True,
    }
    p = os.path.join(ROOT, "evidence", "result_confusable_pairs.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n({round(time.perf_counter() - t0, 1)}초)  -> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
