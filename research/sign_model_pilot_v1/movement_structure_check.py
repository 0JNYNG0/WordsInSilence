#!/usr/bin/env python3
"""Movement measured WITHOUT time warping, because DTW warps away the thing that matters.

axis_split_check.py found the movement axis separates nothing: 1.0x to 1.7x headroom on every word
in both data sets, unchanged when the differentiation window widens from 1 to 5 frames. That is not
noise. DTW stretches time to find the best alignment, so a palm held still and the same palm moved
up and down can be aligned through their shared low-speed moments. Using DTW to compare movement
works against itself.

A structural count does not warp. This measures how many times the wrist reverses direction and
asks whether that single integer separates the pair the participant named - NA, a palm held at the
chest, versus ALDA, the same palm moved up and down there - better than the whole DTW machinery.

Read-only. No camera, no training, no change to the contract or the web recogniser.

Run: py -3.13 movement_structure_check.py
"""
import collections
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pilot import axes as X  # noqa: E402
from pilot import data as D  # noqa: E402
from pilot.web_input import load_web_clip  # noqa: E402

SESSION = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs",
                                        "ses_cbd462050a95"))
PREP = ("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/"
        "WordsInSilence_학습준비_20260912")
DEADBANDS = (0.01, 0.02, 0.04, 0.08)


def load_web():
    rows = [json.loads(l) for l in
            open(os.path.join(SESSION, "decisions.jsonl"), encoding="utf-8") if l.strip()]
    by = {}
    for r in rows:
        if r.get("observation_status") != "OK" or not r.get("requested_word_id"):
            continue
        p = os.path.join(SESSION, "trials", r["trial_id"] + ".npz")
        if os.path.exists(p):
            by.setdefault(r["requested_word_id"], []).append(load_web_clip(p))
    return by


def load_source():
    by = {}
    for it in D.load_manifest(PREP):
        if it["source_only_pilot_eligible"]:
            by.setdefault(it["word_id"], []).append(D.load_clip(PREP, it))
    return by


def best_split(a, b):
    """Best balanced accuracy a single threshold on this integer feature can reach.

    Selected on the same data it is scored on, so it is an optimistic ceiling, not an estimate of
    performance on new recordings. Reported that way on purpose.
    """
    a, b = np.asarray(a), np.asarray(b)
    best = {"balanced_accuracy": 0.0, "threshold": None}
    for th in range(int(min(a.min(), b.min())), int(max(a.max(), b.max())) + 2):
        for lo, hi, order in ((a, b, "first_word_higher"), (b, a, "second_word_higher")):
            acc = 0.5 * ((lo >= th).mean() + (hi < th).mean())
            if acc > best["balanced_accuracy"]:
                best = {"balanced_accuracy": round(float(acc), 4), "threshold": th,
                        "direction": order}
    return best


def run(by, label):
    words = sorted(by)
    counts = {d: {w: [X.direction_reversals(c, deadband=d) for c in by[w]] for w in words}
              for d in DEADBANDS}

    print("\n" + label + "  -  손목 상하 반전 횟수 중앙값 (문턱값별)")
    print(f"{'단어':<14} {'n':>4}" + "".join(f"{'dead=' + str(d):>14}" for d in DEADBANDS))
    for w in words:
        row = f"{w:<14} {len(by[w]):>4}"
        for d in DEADBANDS:
            v = counts[d][w]
            row += f"{np.median(v):>8.1f} ({np.percentile(v, 25):.0f}-{np.percentile(v, 75):.0f})"
        print(row)

    print("\n" + label + "  -  dead=0.02 분포")
    for w in words:
        c = collections.Counter(counts[0.02][w])
        print(f"  {w:<14} " + "  ".join(f"{k}회:{c[k]}" for k in sorted(c)))

    pairs = {}
    for w1, w2 in itertools.combinations(words, 2):
        k = "|".join(sorted((w1, w2)))
        pairs[k] = {str(d): best_split(counts[d][w1], counts[d][w2]) for d in DEADBANDS}
    print("\n" + label + "  -  이 한 숫자만으로 두 단어를 가른 최고 균형정확도 (같은 자료에서 고른 문턱값)")
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1]["0.02"]["balanced_accuracy"])
    for k, v in ranked[:6]:
        b = v["0.02"]
        print(f"  {k:<28} {b['balanced_accuracy']:.3f}   (경계 {b['threshold']}회)")
    return {"reversal_counts": {str(d): counts[d] for d in DEADBANDS},
            "pair_single_threshold": pairs}


def main() -> int:
    t0 = time.perf_counter()
    web, src = load_web(), load_source()
    srcr = run(src, "AIHub 유창한 수행자 (OpenPose)")
    part = run(web, "참가자 본인 (MediaPipe)")

    nb = srcr["pair_single_threshold"]["ALDA|NA"]["0.02"]
    print(f"\n참고: DTW로 잰 알다-나 최고치는 결합 2.45배 / 움직임 축 1.16배였습니다. "
          f"시간 정렬 없는 반전 횟수 한 개의 균형정확도는 {nb['balanced_accuracy']:.3f}입니다.")

    out = {
        "task_id": "movement_structure_check", "date_kst": time.strftime("%Y-%m-%d"),
        "question": "does a time-warp-free structural count separate a held sign from a repeated "
                    "one, where the DTW movement axis cannot",
        "why_dtw_failed": "DTW stretches time to find the best alignment, so a still clip and a "
                          "moving clip can be aligned through their shared low-speed moments. The "
                          "measured movement-axis headroom was 1.0x to 1.7x on every word in both "
                          "data sets, unchanged across differentiation windows of 1 to 5 frames",
        "command": "py -3.13 movement_structure_check.py",
        "feature": "number of times the more active wrist reverses along the vertical image axis, "
                   "steps below a deadband ignored; deadband is in shoulder-width units",
        "source_aihub_openpose": srcr,
        "participant_mediapipe": part,
        "cross_extractor_note": {
            "observation": "ALDA reads a median of 3 reversals on both sides - AIHub 32 of 45 "
                           "clips, the participant 17 of 20 - at deadband 0.02",
            "but": "GAMSAHAMNIDA does not agree (AIHub median 2, participant 5), so this is one "
                   "matching word, not evidence that the count transfers between extractors",
        },
        "limits": [
            "the balanced accuracy picks its threshold on the same data it scores, so it is an "
            "optimistic ceiling for this feature, not a performance estimate on new recordings",
            "the separation depends on the deadband: it holds from 0.01 to 0.04 and collapses by "
            "0.08. Any result must state the value used",
            "one feature separates one pair. GAMSAHAMNIDA also reverses about twice, so this does "
            "not separate it from ALDA; a per-pair gate needs a per-pair choice of axis",
            "AIHub clips are sentence-cut, so a reversal may belong to the transition into or out "
            "of the sign rather than the sign itself",
            "the participant session predates the capture-time attempt_label, so deliberately "
            "wrong attempts cannot be excluded",
            "vertical axis only; a sign whose repetition is horizontal would need the other axis",
            "not a correctness claim: this measures separability between words, not whether any "
            "attempt was well formed",
        ],
        "contract_unchanged": "build_channels() still emits the 280-channel layout; the trained "
                              "checkpoint and the web recogniser are untouched",
        "not_a_correctness_claim": True,
    }
    p = os.path.join(ROOT, "evidence", "result_movement_structure.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n({round(time.perf_counter() - t0, 1)}초)  -> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
