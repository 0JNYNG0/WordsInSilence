#!/usr/bin/env python3
"""Which axis actually separates each word from its nearest neighbour?

One scalar DTW distance mixes handshape, location and movement into a single number. That is why
angle_variant_check.py could cut the within-word spread by more than half and still buy no
headroom: what it removed was noise on one axis and signal on another, and a single number cannot
tell those apart. NA and ALDA are the sharpest case the participant named - the same palm in the
same place, held still versus moved up and down - where only movement can decide.

This measures each axis on its own:

  headroom(word, axis)  = distance to the nearest other word on that axis
                          / spread among that word's own repeats on that axis

and, per word PAIR, which axis separates the two best. A word whose combined headroom is hopeless
may still have one axis that works; that is what a per-axis gate would use.

Read-only. No camera, no training, no change to the contract or the web recogniser.

Run: py -3.13 axis_split_check.py
"""
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
from pilot.dtw_baseline import dtw_distance  # noqa: E402
from pilot.web_input import load_web_clip  # noqa: E402

SESSION = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs",
                                        "ses_cbd462050a95"))
PREP = ("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/"
        "WordsInSilence_학습준비_20260912")
COLS = ("combined",) + X.AXES


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


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def measure(by, col, per_signer):
    """-> (within {word: median|None}, between {"w1|w2": median|None})"""
    words = sorted(by)
    if col == "combined":
        feat = {w: [(c["x"], c.get("signer_id")) for c in by[w]] for w in words}
        dist = dtw_distance
    else:
        feat = {w: [(X.axis_features(c, col), c.get("signer_id")) for c in by[w]] for w in words}

        def dist(a, b, _c=col):
            return X.distance(a, b, _c)

    within, between = {}, {}
    for w in words:
        pairs = [(a, b) for a, b in itertools.combinations(feat[w], 2)
                 if not per_signer or a[1] == b[1]]
        within[w] = med(dist(a[0], b[0]) for a, b in pairs)
    for w1, w2 in itertools.combinations(words, 2):
        between["|".join(sorted((w1, w2)))] = med(
            dist(a[0], b[0]) for a in feat[w1] for b in feat[w2]
            if not per_signer or a[1] == b[1])
    return within, between


def headroom(within, between, words):
    out = {}
    for w in words:
        cand = [(between["|".join(sorted((w, o)))], o) for o in words if o != w]
        cand = [(d, o) for d, o in cand if d is not None]
        if not cand or not within.get(w):
            out[w] = {"headroom": None, "nearest_word": None,
                      "reason": "no measurable spread (one clip per signer) or no distance"}
            continue
        d, o = min(cand)
        out[w] = {"spread": round(within[w], 5), "nearest_word": o,
                  "nearest_distance": round(d, 5), "headroom": round(d / within[w], 3)}
    return out


def cell(v):
    return (format(v, ".2f") + "배") if v else "-"


def run(by, label, per_signer):
    words = sorted(by)
    res = {c: measure(by, c, per_signer) for c in COLS}
    hr = {c: headroom(res[c][0], res[c][1], words) for c in COLS}

    print("\n" + label + "  -  단어별 여유 (최근접 단어까지 거리 / 자기 흩어짐)")
    print(f"{'단어':<14}" + "".join(f"{c:>14}" for c in COLS) + "   가장 좋은 축")
    for w in words:
        vals = {c: hr[c][w]["headroom"] for c in COLS}
        row = f"{w:<14}" + "".join(f"{cell(vals[c]):>14}" for c in COLS)
        ax = [(vals[c], c) for c in X.AXES if vals[c]]
        print(row + (f"   {max(ax)[1]} {max(ax)[0]:.2f}배" if ax else "   -"))

    print("\n" + label + "  -  단어 쌍을 어느 축이 가르는가 (거리 / 두 단어 흩어짐 중 큰 쪽)")
    pairs = []
    for w1, w2 in itertools.combinations(words, 2):
        k = "|".join(sorted((w1, w2)))
        row = {"pair": k}
        for c in COLS:
            wi, bw = res[c]
            sp = [wi.get(w) for w in (w1, w2) if wi.get(w)]
            row[c] = (round(bw[k] / max(sp), 3) if bw.get(k) and sp else None)
        pairs.append(row)
    pairs.sort(key=lambda r: (r["combined"] is None, r["combined"] or 0))
    for r in pairs[:6]:
        line = f"  {r['pair']:<28}" + "".join(f"{cell(r[c]):>14}" for c in COLS)
        ax = [(r[c], c) for c in X.AXES if r[c]]
        print(line + (f"   {max(ax)[1]}" if ax else ""))
    return {"headroom": hr, "pair_separation": pairs,
            "raw": {c: {"within_word": res[c][0], "between_words": res[c][1]} for c in COLS}}


def main() -> int:
    t0 = time.perf_counter()
    web, src = load_web(), load_source()
    print(f"참가자 {sum(len(v) for v in web.values())}건 / "
          f"AIHub {sum(len(v) for v in src.values())}건")
    part = run(web, "참가자 본인 (MediaPipe)", per_signer=False)
    srcr = run(src, "AIHub 유창한 수행자 (OpenPose, 같은 사람 안에서만)", per_signer=True)

    out = {
        "task_id": "axis_split_check", "date_kst": time.strftime("%Y-%m-%d"),
        "question": "measured one axis at a time, which axis separates each word from its nearest "
                    "neighbour, and does any axis work where the combined distance does not",
        "command": "py -3.13 axis_split_check.py",
        "axis_definition": {
            "handshape": "flexion angles per hand, 38 channels, rotation invariant",
            "location": "hand position in body frame, relative to the face, plus elbows, 12",
            "movement": "wrist and elbow velocity in body frame plus speed, 12, central difference "
                        "over the clip's own timestamps",
            "deviation_from_front_keyphase_v1": "there, movement also carries static arm positions. "
                                                "Here movement is velocity only so the three axes "
                                                "do not overlap and a result can be attributed to "
                                                "one of them",
        },
        "participant_mediapipe": part,
        "source_aihub_openpose_within_signer": srcr,
        "limits": [
            "a pair distance is computed on the channels both clips actually use; a one-handed word "
            "compared with a two-handed one is therefore judged on the shared hand alone, and that "
            "restriction is part of the number",
            "headroom is a separability measure, not a pass line. No threshold is set here and "
            "nothing judges whether an attempt was correct",
            "the participant session predates the capture-time attempt_label, so deliberately "
            "wrong attempts cannot be excluded",
            "all 20 repeats of a word come from one sitting and are correlated, so the spread is "
            "likely optimistic",
            "GAMSAHAMNIDA and MOREUDA have one AIHub clip per signer, so their within-signer "
            "spread and headroom are unmeasurable there",
            "AIHub clips are sentence-cut and carry coarticulation",
        ],
        "contract_unchanged": "build_channels() still emits the 280-channel layout; the trained "
                              "checkpoint and the web recogniser are untouched",
        "not_a_correctness_claim": True,
    }
    p = os.path.join(ROOT, "evidence", "result_axis_split.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n({round(time.perf_counter() - t0, 1)}초)  -> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
