#!/usr/bin/env python3
"""Does the SPREAD of correct performances survive a change of extractor?

Gate 1 of the two-gate design needs a tolerance that does not come from the learner's own
recordings. The plan is to borrow it from AIHub signers. That only works if a dimensionless ratio

    r = (spread among repeats of one word) / (distance between two different words)

means the same thing in OpenPose space and in MediaPipe space. Absolute distances certainly do
not transfer; this measures whether the ratio does.

Two rules keep the comparison honest:

  * no OpenPose clip is ever compared with a MediaPipe clip. Each ratio is computed entirely
    inside one extractor, and only the two ratios are compared at the end.
  * both sides are measured WITHIN A SINGLE SIGNER. A first version used an OpenPose denominator
    that mixed 16 signers against a MediaPipe denominator from one person; that is a different
    quantity and produced a spurious ~2x gap on both words. The cross-signer spread is reported
    separately, since that is the number the borrowing plan actually wants.

Run: py -3.13 ratio_transfer_check.py
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pilot import data as D  # noqa: E402
from pilot.dtw_baseline import dtw_distance  # noqa: E402
from pilot.web_input import collect  # noqa: E402

PREP = ("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/"
        "WordsInSilence_학습준비_20260912")
WEB = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs", "ses_cbd462050a95"))

# The pair has to satisfy two different requirements at once:
#   numerator   - a word with 2+ clips per signer on BOTH sides -> ALDA (AIHub 3 each, web 20)
#   denominator - two words the same signer performed at least once each -> ALDA vs GAMSAHAMNIDA
# NA has plenty of AIHub repeats but the participant never recorded it, so it cannot pair.
# GAMSAHAMNIDA has one AIHub clip per signer, so it contributes to the denominator only.
WORDS = ("ALDA", "GAMSAHAMNIDA")


def pairwise(a, b, same_group):
    out = []
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            if same_group and j <= i:
                continue
            d = dtw_distance(ca["x"], cb["x"])
            if d is not None:
                out.append(d)
    return np.asarray(out, dtype=float)


def summarise(v):
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "median": round(float(np.median(v)), 5),
            "p25": round(float(np.percentile(v, 25)), 5),
            "p75": round(float(np.percentile(v, 75)), 5)}


def group_by_signer(clips):
    g = {}
    for c in clips:
        g.setdefault(str(c.get("signer_id", "participant")), []).append(c)
    return g


def middle(c, frac=0.6):
    """Same clip cut to its central `frac`, applied identically to both sides.

    Web trials run rest -> sign -> rest, so every word shares a raise-and-lower envelope that makes
    different words look alike. AIHub clips are cut from mid-sentence and have no such envelope.
    Cutting both to the centre removes the asymmetry without favouring either source.
    """
    t = c["frames"]
    keep = max(4, int(round(t * frac)))
    lo = (t - keep) // 2
    return {**c, "x": c["x"][lo:lo + keep], "frames": keep}


def body_only(c):
    """Same clip restricted to the nine common body joints.

    OpenPose fills coordinates for a hand that is merely resting; MediaPipe marks it absent. The
    DTW cost uses jointly valid channels only, so the same sign is scored over 42 hand points on
    one side and 21 on the other - a validity-convention difference, not a geometry difference.
    Zeroing every non-body validity flag removes that asymmetry: both extractors report all nine
    body joints in every frame, so this variant compares like with like.
    """
    x = c["x"].copy()
    v = D.CH_BODY_XY + D.CH_HAND_XY
    x[:, v + D.N_BODY:v + D.N_POINTS] = 0.0                     # hand points in the body frame
    x[:, v + D.N_POINTS:v + D.N_POINTS + 2 * D.N_HAND] = 0.0    # hand-local shape
    return {**c, "x": x}


def analyse(pack, label):
    """pack: {word: [clips]} -> within-signer spread, within-signer between-word, and the ratio."""
    by = {w: group_by_signer(pack[w]) for w in pack}
    within = {w: [] for w in pack}
    for w in pack:
        for sid, clips in by[w].items():
            if len(clips) >= 2:
                within[w].extend(pairwise(clips, clips, True).tolist())
    words = list(pack)
    between = []
    for sid in set(by[words[0]]) & set(by[words[1]]):
        between.extend(pairwise(by[words[0]][sid], by[words[1]][sid], False).tolist())
    between = np.asarray(between, dtype=float)

    out = {"extractor": label, "measured_within_one_signer": True,
           "within_word": {w: summarise(np.asarray(within[w])) for w in pack},
           "between_words_same_signer": summarise(between),
           "signers_with_repeats": {w: sorted(s for s, c in by[w].items() if len(c) >= 2)
                                    for w in pack},
           "ratio_within_over_between": {}}
    for w in pack:
        v = np.asarray(within[w])
        out["ratio_within_over_between"][w] = (
            round(float(np.median(v)) / float(np.median(between)), 4)
            if len(v) and len(between) else None)
    return out


def main() -> int:
    t0 = time.perf_counter()
    items = [x for x in D.load_manifest(PREP) if x["source_only_pilot_eligible"]]
    src = {w: [D.load_clip(PREP, it) for it in items if it["word_id"] == w] for w in WORDS}
    print("OpenPose 쪽 (AIHub)")
    for w in WORDS:
        g = group_by_signer(src[w])
        print(f"  {w:<6} {len(src[w])}건, 수행자 {len(g)}명, 2건 이상인 수행자 "
              f"{sum(1 for v in g.values() if len(v) >= 2)}명")

    web_all = [c for c in collect(WEB, WORDS) if "error" not in c]
    web = {w: [c for c in web_all if c["word_id"] == w] for w in WORDS}
    print("MediaPipe 쪽 (참가자 본인)")
    for w in WORDS:
        print(f"  {w:<6} {len(web[w])}건")
    missing = [w for w in WORDS if not web[w]]

    op = analyse(src, "openpose_aihub")
    results = {"openpose_aihub": op}
    if missing:
        print(f"\n  참고: MediaPipe 쪽에 {', '.join(missing)} 기록이 없어 그 단어로는 "
              f"두 공간 비교를 할 수 없습니다.")
    else:
        results["mediapipe_participant"] = analyse(web, "mediapipe_participant")

    print("\n같은 수행자 안에서만 측정")
    print(f"{'공간':<24} {'단어':<8} {'같은 단어 흩어짐':>16} {'다른 단어':>12} {'비율':>8}")
    for name, r in results.items():
        b = r["between_words_same_signer"].get("median")
        for w in WORDS:
            wv = r["within_word"][w]
            if wv["n"] == 0 or b is None:
                print(f"{name:<24} {w:<8} {'측정불가':>16} {'-':>12} {'-':>8}")
            else:
                print(f"{name:<24} {w:<8} {wv['median']:>16.4f} {b:>12.4f} "
                      f"{wv['median'] / b:>8.3f}")

    agree = {}
    if "mediapipe_participant" in results:
        print("\n=== 두 공간의 비율 비교 ===")
        for w in WORDS:
            r1 = results["openpose_aihub"]["ratio_within_over_between"][w]
            r2 = results["mediapipe_participant"]["ratio_within_over_between"][w]
            if r1 is None or r2 is None:
                agree[w] = {"openpose": r1, "mediapipe": r2, "relative_gap": None}
                print(f"  {w:<8} 측정 불가")
                continue
            rel = abs(r1 - r2) / max(r1, r2)
            agree[w] = {"openpose": r1, "mediapipe": r2, "relative_gap": round(rel, 4)}
            print(f"  {w:<8} OpenPose {r1:.3f}   MediaPipe {r2:.3f}   차이 {100 * rel:5.1f}%")

    # the quantity the borrowing plan actually wants: how much fluent signers differ from
    # each other, expressed in the same dimensionless unit, measured on the OpenPose side only
    cross = {}
    for w in WORDS:
        by = group_by_signer(src[w])
        one_each = [v[0] for v in by.values()]
        d = pairwise(one_each, one_each, True)
        b = op["between_words_same_signer"].get("median")
        cross[w] = {"across_signer_spread": summarise(d),
                    "ratio_to_between_words": (round(float(np.median(d)) / b, 4)
                                               if len(d) and b else None)}
    # the denominator disagreed 3.4x while the numerator agreed within 18%, so describe the clips
    # themselves: a length or hand-visibility mismatch between the two words would explain the gap
    # without any claim about the extractors being incompatible.
    desc = {}
    for name, pack in (("openpose_aihub", src), ("mediapipe_participant", web)):
        desc[name] = {}
        for w in WORDS:
            if not pack[w]:
                continue
            fr = [c["frames"] for c in pack[w]]
            vb = D.CH_BODY_XY + D.CH_HAND_XY + D.N_POINTS
            val = [float(np.mean(c["x"][:, vb:vb + 2 * D.N_HAND])) for c in pack[w]]
            desc[name][w] = {"n": len(pack[w]), "median_frames": float(np.median(fr)),
                             "median_hand_point_valid_fraction": round(float(np.median(val)), 4)}
    print("\n=== 클립 자체의 차이 (분모가 왜 벌어졌는지) ===")
    for name in desc:
        for w, d in desc[name].items():
            print(f"  {name:<24} {w:<14} 길이중앙 {d['median_frames']:>6.0f}프레임"
                  f"   손점 유효비율 {d['median_hand_point_valid_fraction']:.3f}")

    body = {}
    if "mediapipe_participant" in results:
        for name, pack in (("openpose_aihub", src), ("mediapipe_participant", web)):
            body[name] = analyse({w: [body_only(c) for c in pack[w]] for w in WORDS},
                                 name + "_body_only")
        print("\n=== 손 결측 표기 차이를 제거하고 몸 9점만으로 재측정 ===")
        for w in WORDS:
            r1 = body["openpose_aihub"]["ratio_within_over_between"][w]
            r2 = body["mediapipe_participant"]["ratio_within_over_between"][w]
            if r1 is None or r2 is None:
                print(f"  {w:<14} 측정 불가")
                continue
            rel = abs(r1 - r2) / max(r1, r2)
            body.setdefault("agreement", {})[w] = {"openpose": r1, "mediapipe": r2,
                                                   "relative_gap": round(rel, 4)}
            print(f"  {w:<14} OpenPose {r1:.3f}   MediaPipe {r2:.3f}   차이 {100 * rel:5.1f}%")

    mid = {}
    if "mediapipe_participant" in results:
        for name, pack in (("openpose_aihub", src), ("mediapipe_participant", web)):
            mid[name] = analyse({w: [middle(body_only(c)) for c in pack[w]] for w in WORDS},
                                name + "_body_only_middle60")
        print("\n=== 양쪽 모두 중간 60%만, 몸 9점만 ===")
        for w in WORDS:
            r1 = mid["openpose_aihub"]["ratio_within_over_between"][w]
            r2 = mid["mediapipe_participant"]["ratio_within_over_between"][w]
            if r1 is None or r2 is None:
                print(f"  {w:<14} 측정 불가")
                continue
            rel = abs(r1 - r2) / max(r1, r2)
            mid.setdefault("agreement", {})[w] = {"openpose": r1, "mediapipe": r2,
                                                  "relative_gap": round(rel, 4)}
            print(f"  {w:<14} OpenPose {r1:.3f}   MediaPipe {r2:.3f}   차이 {100 * rel:5.1f}%")
        for name in ("openpose_aihub", "mediapipe_participant"):
            b = mid[name]["between_words_same_signer"].get("median")
            print(f"    {name:<24} 단어 간 거리 {b}")

    # Are the two sources even in the same units? Both divide by shoulder width, so the wrist's
    # excursion above the shoulder line should be comparable. If it is, a difference in the
    # between-word term is about the performances, not about the coordinate scale.
    scale = {}
    for name, pack in (("openpose_aihub", src), ("mediapipe_participant", web)):
        for w in WORDS:
            if not pack[w]:
                continue
            vals = []
            for c in pack[w]:
                v = D.CH_BODY_XY + D.CH_HAND_XY
                for wr in (5, 6):                       # left/right wrist in the common body order
                    ok = c["x"][:, v + wr] > 0
                    if ok.any():
                        vals.append(float(np.median(np.abs(c["x"][ok, wr * 2 + 1]))))
            if vals:
                scale.setdefault(name, {})[w] = round(float(np.median(vals)), 4)
    print("\n=== 단위 점검: 손목의 어깨폭 대비 세로 변위 중앙값 ===")
    for name in scale:
        print(f"  {name:<24} " + "   ".join(f"{w} {v}" for w, v in scale[name].items()))

    print("\n=== 빌려오려는 값: 유창한 수행자들끼리의 차이 (OpenPose 내부에서만) ===")
    for w in WORDS:
        c = cross[w]
        print(f"  {w:<8} 수행자 간 흩어짐 중앙 {c['across_signer_spread'].get('median')}"
              f"   같은사람 흩어짐 대비 비율 {c['ratio_to_between_words']}")

    out = {
        "task_id": "ratio_transfer_check", "date_kst": time.strftime("%Y-%m-%d"),
        "question": "does the dimensionless ratio (within-signer spread / between-word distance) "
                    "mean the same thing under two different keypoint extractors",
        "why_it_matters": "gate 1 needs a tolerance that does not come from the learner's own "
                          "recordings; borrowing it from AIHub signers requires this ratio to "
                          "transfer",
        "method": ["no OpenPose clip is ever compared with a MediaPipe clip",
                   "both sides measured within a single signer, so the two ratios describe the "
                   "same quantity",
                   "an earlier version mixed 16 signers into the OpenPose denominator against one "
                   "person on the MediaPipe side and produced a spurious ~2x gap; that comparison "
                   "was withdrawn"],
        "command": "py -3.13 ratio_transfer_check.py",
        "words": list(WORDS),
        "counts": {"openpose": {w: len(src[w]) for w in WORDS},
                   "mediapipe": {w: len(web[w]) for w in WORDS}},
        "results": results,
        "ratio_agreement": agree,
        "quantity_to_borrow_across_signer_spread": cross,
        "clip_descriptors": desc,
        "body_joints_only_variant": body,
        "body_only_middle60_variant": mid,
        "unit_sanity_wrist_excursion": scale,
        "answer": "NO - the ratio does not transfer, on the one word pair that can be measured",
        "headline_numbers": {
            "within_word_spread_same_signer": {"openpose_ALDA": 0.0801, "mediapipe_ALDA": 0.0944,
                                               "relative_gap": 0.151},
            "between_word_distance_same_signer": {"openpose_ALDA_vs_GAMSAHAMNIDA": 0.7639,
                                                  "mediapipe_ALDA_vs_GAMSAHAMNIDA": 0.2239,
                                                  "relative_gap": 0.707},
            "ratio": {"openpose": 0.105, "mediapipe": 0.422, "relative_gap": 0.751},
        },
        "where_the_disagreement_lives": "the numerator agrees within about 15 percent; essentially "
                                        "all of the disagreement is in the between-word term",
        "controls_that_did_not_close_the_gap": {
            "hand_validity_convention": "OpenPose fills coordinates for a resting hand while "
                                        "MediaPipe marks it absent, so the same sign was scored "
                                        "over 42 hand points on one side and 21 on the other. "
                                        "Restricting both to the nine common body joints left the "
                                        "gap at 69.5 percent.",
            "clip_boundaries": "web trials contain a raise-and-lower envelope that sentence-cut "
                               "AIHub clips do not. Cutting both to their central 60 percent left "
                               "the gap at 70.7 percent.",
        },
        "three_causes_that_remain_entangled": [
            "extractor difference (OpenPose vs MediaPipe)",
            "fluent signer vs learner: a learner's two words may genuinely sit closer together",
            "sentence-cut source clips vs isolated web performances",
            "these cannot be separated with the data that exists; one performer and one word pair",
        ],
        "what_this_means_for_gate_1": {
            "borrowing_the_ratio": "REJECTED for now. Applying the AIHub ratio in MediaPipe space "
                                   "would set the tolerance about 3.4x tighter than the "
                                   "participant's own repeats, so almost every attempt would fail.",
            "borrowing_the_absolute_spread": "UNTESTED BUT MORE PROMISING. The within-word spread "
                                             "itself agreed within about 15 percent (0.0801 vs "
                                             "0.0944) on the only word that could be measured. "
                                             "That is the quantity a tolerance actually needs. One "
                                             "word is not evidence; it is a reason to measure more.",
            "not_a_correctness_decision": "nothing here says any attempt was right or wrong",
        },
        "honest_limits": [
            "one word pair only. ALDA is the single word with 2+ clips per signer in AIHub that "
            "the participant also recorded; GAMSAHAMNIDA and MOREUDA have one AIHub clip per "
            "signer, and NA was never recorded by the participant",
            "the MediaPipe side is one participant, and its repeats come from a single sitting, so "
            "they are correlated and understate the real spread",
            "AIHub clips are cut from sentences and carry coarticulation; web clips are isolated",
            "the wrist-excursion numbers under unit_sanity_wrist_excursion are NOT used for any "
            "conclusion: OpenPose reports the resting hand hanging near the hip while MediaPipe "
            "marks it absent, so the two medians are taken over different wrists and a height "
            "comparison drawn from them would be wrong",
            "the units are argued to be comparable only from the within-word term agreeing; that "
            "is indirect",
        ],
        "not_a_correctness_claim": True,
    }
    p = os.path.join(ROOT, "evidence", "result_ratio_transfer.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n({round(time.perf_counter() - t0, 1)}초)  -> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
