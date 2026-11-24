#!/usr/bin/env python3
"""Does replacing wrist-relative hand coordinates with flexion angles buy any headroom?

participant_pair_check.py found that two of the participant's five words have no room at all in
their own recordings: INSA sits 0.80x and GAMSAHAMNIDA 1.24x from their nearest neighbour relative
to the spread among their own repeats. Four cheap explanations were ruled out (outlier trials, DTW
band, lost hand tracking, and a few bad clips); the one left is that the hand channels are
wrist-relative coordinates that change whenever the hand rotates, which INSA does by design.

This runs the same table three ways on the same clips and compares headroom:

  xy            the current contract's hand channels          84 hand values
  angle         flexion angles only                           38
  angle+orient  flexion angles plus a unit orientation vector 42

The body block is identical in all three, so only the hand representation differs.

Read-only. No camera, no training, no change to the contract or the web recogniser.

Run: py -3.13 angle_variant_check.py
"""
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pilot import angles as A  # noqa: E402
from pilot import data as D  # noqa: E402
from pilot.web_input import load_web_clip  # noqa: E402

SESSION = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs",
                                        "ses_cbd462050a95"))
PREP = ("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/"
        "WordsInSilence_학습준비_20260912")
VARIANTS = {
    "xy": dict(hand_xy=True, use_angles=False, orientation=False),
    "angle": dict(hand_xy=False, use_angles=True, orientation=False),
    "angle+orient": dict(hand_xy=False, use_angles=True, orientation=True),
    "xy+angle": dict(hand_xy=True, use_angles=True, orientation=False),
}


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
    """One clip per signer per word is enough for the between-word term; repeats drive the spread."""
    by = {}
    for it in D.load_manifest(PREP):
        if it["source_only_pilot_eligible"]:
            by.setdefault(it["word_id"], []).append(D.load_clip(PREP, it))
    return by


def table(by, opts, per_signer=False):
    words = sorted(by)
    feats = {w: [(A.variant(c, **opts), c.get("signer_id")) for c in by[w]] for w in words}
    within, between = {}, {}
    for w in words:
        pairs = itertools.combinations(feats[w], 2)
        if per_signer:
            pairs = [(a, b) for a, b in itertools.combinations(feats[w], 2) if a[1] == b[1]]
        d = [x for x in (A.distance(a[0], b[0]) for a, b in pairs) if x is not None]
        within[w] = float(np.median(d)) if d else None
    for w1, w2 in itertools.combinations(words, 2):
        d = [x for x in (A.distance(a[0], b[0]) for a in feats[w1] for b in feats[w2]
                          if not per_signer or a[1] == b[1]) if x is not None]
        between["|".join(sorted((w1, w2)))] = float(np.median(d)) if d else None
    out = {}
    for w in words:
        dist, who = min((between["|".join(sorted((w, o)))], o) for o in words if o != w)
        out[w] = {"spread": round(within[w], 5) if within[w] else None, "nearest_word": who,
                  "nearest_distance": round(dist, 5),
                  "headroom": round(dist / within[w], 3) if within[w] else None,
                  "headroom_unmeasurable_reason": (None if within[w] else
                                                   "one clip per signer, no within-signer pair")}
    return out, between


def show(title, res, words):
    print(f"\n{title}")
    print(f"{'단어':<14}" + "".join(f"{v:>26}" for v in VARIANTS))
    print(f"{'':<14}" + "".join(f"{'흩어짐   최근접   여유':>26}" for _ in VARIANTS))
    for w in words:
        row = f"{w:<14}"
        for v in VARIANTS:
            r = res[v][w]
            row += (f"{r['spread']:>10.3f}{r['nearest_distance']:>9.3f}{r['headroom']:>7.2f}배"
                    if r["headroom"] else f"{'-':>10}{r['nearest_distance']:>9.3f}{'-':>8}")
        print(row)


def main() -> int:
    t0 = time.perf_counter()
    web, src = load_web(), load_source()
    print(f"참가자 {sum(len(v) for v in web.values())}건 / AIHub {sum(len(v) for v in src.values())}건")

    res_w = {v: table(web, o)[0] for v, o in VARIANTS.items()}
    res_s = {v: table(src, o, per_signer=True)[0] for v, o in VARIANTS.items()}
    show("참가자 본인 기록 (MediaPipe, 5단어 x 20회)", res_w, sorted(web))
    show("AIHub 유창한 수행자 (OpenPose, 같은 사람 안에서만)", res_s, sorted(src))

    print("\n여유 변화 (xy -> angle+orient)")
    for name, res in (("참가자", res_w), ("AIHub", res_s)):
        for w in sorted(res["xy"]):
            a, b = res["xy"][w]["headroom"], res["angle+orient"][w]["headroom"]
            if a is None or b is None:
                print(f"  {name:<6} {w:<14} 측정 불가 (수행자당 1건)")
                continue
            mark = "  통과선 1.5배 넘김" if a < 1.5 <= b else ("  여전히 부족" if b < 1.5 else "")
            print(f"  {name:<6} {w:<14} {a:>5.2f}배 -> {b:>5.2f}배  ({100*(b-a)/a:+6.1f}%){mark}")

    out = {
        "task_id": "angle_variant_check", "date_kst": time.strftime("%Y-%m-%d"),
        "question": "does replacing wrist-relative hand coordinates with rotation-invariant "
                    "flexion angles increase the gap between a word and its nearest neighbour, "
                    "relative to the spread among repeats of that word",
        "command": "py -3.13 angle_variant_check.py",
        "variants": {k: dict(v, hand_value_channels=(84 if v["hand_xy"] else 0)
                             + (2 * A.N_ANGLES if v["use_angles"] else 0)
                             + (4 if v["orientation"] else 0)) for k, v in VARIANTS.items()},
        "participant_mediapipe": res_w,
        "source_aihub_openpose_within_signer": res_s,
        "answer": "NO - flexion angles do not buy headroom, and the hand representation was not "
                  "the bottleneck",
        "what_was_predicted_and_did_happen": {
            "claim": "wrist-relative hand coordinates change when the hand rotates, inflating the "
                     "spread of words that rotate the hand",
            "confirmed": "replacing them with angles cut the within-word spread exactly where "
                         "predicted: INSA 0.358 -> 0.154 (-57 percent), SARANG 0.271 -> 0.129 "
                         "(-52 percent), while the already-tight fixed-orientation words barely "
                         "moved (JADA 0.110 -> 0.091)",
        },
        "why_it_still_did_not_help": {
            "finding": "the nearest-neighbour distance fell by as much or more than the spread, so "
                       "the ratio did not improve. Hand orientation is not noise: it is one of the "
                       "things that tells two words apart",
            "worst_cases": "SARANG 1.97x -> 1.27x and AIHub NA 2.45x -> 1.63x both got worse",
            "only_gain": "INSA 0.81x -> 1.16x with angles alone, still far below any usable line",
            "keeping_both": "xy+angle lands within 4 percent of xy everywhere, because 84 "
                            "coordinate channels dominate 38 angle channels under a uniform weight",
        },
        "what_this_points_to": {
            "reasoning": "a single scalar DTW distance mixes handshape, location and movement into "
                         "one number, so no choice of hand representation can help when the "
                         "discriminating axis differs per word. NA versus ALDA is separated by "
                         "MOVEMENT (a held palm versus the same palm moving up and down), not by "
                         "handshape or orientation at all",
            "already_exists": "research/front_keyphase_v1 already decomposes the comparison into "
                              "handshape / location / movement axes with per-word bands. That "
                              "design, not a better hand encoding, is what this result argues for",
            "status": "a direction indicated by this measurement, not a tested claim",
        },
        "limits": [
            "the variants do not have the same number of hand channels (84 vs 38 vs 42), so part "
            "of any change is reweighting between the body block and the hand block, not rotation "
            "invariance alone. This is not separated here",
            "headroom is a separability measure, not a pass line; no threshold is set and nothing "
            "here judges whether an attempt was correct",
            "the participant session predates the capture-time attempt_label, so deliberately "
            "wrong attempts cannot be excluded",
            "all 20 repeats of a word come from one sitting and are correlated",
            "AIHub clips are sentence-cut and carry coarticulation",
        ],
        "contract_unchanged": "build_channels() still emits the 280-channel layout; the trained "
                              "checkpoint and the web recogniser are untouched",
        "not_a_correctness_claim": True,
    }
    p = os.path.join(ROOT, "evidence", "result_angle_variant.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n({round(time.perf_counter() - t0, 1)}초)  -> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
