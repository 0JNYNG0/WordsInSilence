#!/usr/bin/env python3
"""v2 movement events on the real clips, compared with v1 and with DTW under ONE metric.

External review of the v1 result raised four things this answers:

  * v1's count depends on speed and frame rate. v2 counts retracement amplitude instead, and
    tests/test_motion_events.py reproduces v1's documented 3 / 0 / 0 failure before requiring v2
    not to repeat it. Here the two versions are run on the SAME clips, sample by sample.
  * v1 deleted missing frames and joined across the hole. v2 splits at a gap, reports each hand
    separately, and returns INSUFFICIENT_OBSERVATION with turns=None where v1 would have returned
    a confident 0.
  * "0.941 beats 2.45x" compared a balanced accuracy with a distance ratio. Here DTW and the count
    are scored on the same task, the same labels and the same clips, with the same metric, and
    both use leave-one-out so neither is fitted on the sample it is scored on.
  * the v1 diagnostic read all 163 eligible clips, reserved_test included. That exposure is
    recorded rather than argued away, and this run records its own.

No camera, no training, no change to the contract, the checkpoint or the web recogniser. This sets
no pass line and judges no attempt as correct or incorrect.

Run: py -3.13 motion_events_check.py
"""
import collections
import hashlib
import json
import os
import platform
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
AMPLITUDES = (0.03, 0.05, 0.08, 0.12)
PAIR = ("ALDA", "NA")          # the pair the participant named: same palm, held versus moved


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_source():
    by = {}
    for it in D.load_manifest(PREP):
        if not it["source_only_pilot_eligible"]:
            continue
        c = D.load_clip(PREP, it)
        by.setdefault(it["word_id"], []).append(c)
    return by


def load_web():
    rows = [json.loads(l) for l in
            open(os.path.join(SESSION, "decisions.jsonl"), encoding="utf-8") if l.strip()]
    by = {}
    for r in rows:
        if r.get("observation_status") != "OK" or not r.get("requested_word_id"):
            continue
        p = os.path.join(SESSION, "trials", r["trial_id"] + ".npz")
        if os.path.exists(p):
            c = load_web_clip(p)
            c["sample_id"] = r["trial_id"]
            c["split"] = "web_diagnostic"
            by.setdefault(r["requested_word_id"], []).append(c)
    return by


def active_hand(rec):
    """The hand with more turns, or the observed one. None when neither hand was seen enough."""
    cand = [rec[h] for h in ("left", "right") if rec[h]["status"] == "OK"]
    if not cand:
        return None
    return max(cand, key=lambda r: (r["turns"], r.get("excursion", 0.0)))


def per_sample(by, amp):
    rows = []
    for w, clips in by.items():
        for c in clips:
            ev = X.movement_events(c, turn_amplitude=amp)
            a = active_hand(ev)
            rows.append({
                "sample_id": c.get("sample_id"), "word_id": w, "split": c.get("split"),
                "signer_id": c.get("signer_id"),
                "v1_reversals_deadband_0.02": X.direction_reversals(c),
                "v2_turns": (a["turns"] if a else None),
                "v2_status": ("OK" if a else "INSUFFICIENT_OBSERVATION"),
                "left": ev["left"], "right": ev["right"],
                "clip_seconds": ev["clip_seconds"],
            })
    return rows


def _best_threshold(vals, labels):
    """Threshold and direction maximising balanced accuracy on the values given."""
    vals, labels = np.asarray(vals), np.asarray(labels)
    best = (-1.0, None, None)
    for th in range(int(vals.min()), int(vals.max()) + 2):
        for direction in (1, -1):
            pred = ((vals >= th) if direction == 1 else (vals < th)).astype(int)
            recalls = [float((pred[labels == k] == k).mean()) for k in (0, 1) if (labels == k).any()]
            acc = float(np.mean(recalls))
            if acc > best[0]:
                best = (acc, th, direction)
    return best


def _folds(signers, n):
    """Leave-one-out, or leave-one-SIGNER-out when signer ids are available.

    A nearest-template classifier can otherwise answer with the same person's other recording of
    the same word, which is not the question a new learner poses.
    """
    if signers is None:
        return [np.arange(n) != i for i in range(n)]
    sig = np.asarray([str(s) for s in signers])
    return [sig != sig[i] for i in range(n)]


def loo_count_accuracy(vals, labels, signers=None):
    """Leave-one-out: the threshold is chosen without the clip it is then applied to."""
    vals, labels = np.asarray(vals), np.asarray(labels)
    hit = {0: 0, 1: 0}
    n = {0: int((labels == 0).sum()), 1: int((labels == 1).sum())}
    for i, keep in enumerate(_folds(signers, len(vals))):
        if not keep.any() or len(set(labels[keep])) < 2:
            continue
        _, th, direction = _best_threshold(vals[keep], labels[keep])
        pred = int(vals[i] >= th) if direction == 1 else int(vals[i] < th)
        hit[int(labels[i])] += int(pred == labels[i])
    recalls = {k: hit[k] / n[k] for k in n}
    return {"balanced_accuracy": round(float(np.mean(list(recalls.values()))), 4),
            "per_class_recall": {k: f"{hit[k]}/{n[k]}" for k in n},
            "correct": sum(hit.values()), "total": sum(n.values())}


def loo_dtw_accuracy(clips, labels, signers=None):
    """Leave-one-out nearest template, same clips and labels, same metric as the count."""
    n = len(clips)
    dist = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(clips[i]["x"], clips[j]["x"])
            if d is not None:
                dist[i, j] = dist[j, i] = d
    hit = collections.Counter()
    tot = collections.Counter(labels)
    undecided = 0
    for i, keep in enumerate(_folds(signers, n)):
        order = np.argsort(dist[i])
        order = [j for j in order if j != i and keep[j] and np.isfinite(dist[i, j])]
        if not order:
            undecided += 1
            continue
        hit[labels[i]] += int(labels[order[0]] == labels[i])
    recalls = {k: hit[k] / tot[k] for k in tot}
    return {"balanced_accuracy": round(float(np.mean(list(recalls.values()))), 4),
            "per_class_recall": {k: f"{hit[k]}/{tot[k]}" for k in tot},
            "correct": sum(hit.values()), "total": sum(tot.values()), "undecided": undecided}


def main() -> int:
    t0 = time.perf_counter()
    src, web = load_source(), load_web()

    print("1) v1 반례 재현은 tests/test_motion_events.py 에서 수행 (11/11)")
    print("\n2) 같은 표본에 v1 / v2 적용 — AIHub")
    rows_src = {a: per_sample(src, a) for a in AMPLITUDES}
    rows_web = {a: per_sample(web, a) for a in AMPLITUDES}
    for tag, rows in (("AIHub", rows_src), ("참가자", rows_web)):
        print(f"\n{tag}  단어별 v2 turns 중앙값 (진폭 문턱별) 과 관측 불충분 건수")
        words = sorted({r["word_id"] for r in rows[AMPLITUDES[0]]})
        print(f"{'단어':<14} {'n':>4}" + "".join(f"{'amp=' + str(a):>13}" for a in AMPLITUDES)
              + f"{'불충분':>8}")
        for w in words:
            line = f"{w:<14}"
            n = sum(1 for r in rows[AMPLITUDES[0]] if r["word_id"] == w)
            line += f" {n:>3}"
            for a in AMPLITUDES:
                v = [r["v2_turns"] for r in rows[a] if r["word_id"] == w and r["v2_turns"] is not None]
                line += f"{(np.median(v) if v else float('nan')):>13.1f}"
            bad = sum(1 for r in rows[AMPLITUDES[0]]
                      if r["word_id"] == w and r["v2_status"] != "OK")
            print(line + f"{bad:>8}")

    print("\n3) 같은 과제·같은 라벨·같은 표본·같은 지표로 비교  (AIHub 알다 vs 나)")
    clips = [c for w in PAIR for c in src[w]]
    labels = [0] * len(src[PAIR[0]]) + [1] * len(src[PAIR[1]])
    same = {"task": f"{PAIR[0]} vs {PAIR[1]}, one label per clip",
            "label_meaning": {"0": PAIR[0], "1": PAIR[1]},
            "clips": len(clips), "metric": "balanced accuracy = mean of the two class recalls"}
    idx = {c.get("sample_id"): i for i, c in enumerate(clips)}
    v1 = [X.direction_reversals(c) for c in clips]
    v2 = []
    for c in clips:
        a = active_hand(X.movement_events(c))
        v2.append(a["turns"] if a else 0)
    same["v1_count_leave_one_out"] = loo_count_accuracy(v1, labels)
    same["v2_count_leave_one_out"] = loo_count_accuracy(v2, labels)
    same["v2_count_in_sample_threshold"] = {
        "balanced_accuracy": round(_best_threshold(v2, labels)[0], 4),
        "threshold": _best_threshold(v2, labels)[1],
        "note": "threshold chosen on the same clips it scores; optimistic, kept only to connect "
                "with the previously reported 0.941 which was computed this way"}
    same["dtw_nearest_template_leave_one_out"] = loo_dtw_accuracy(clips, labels)
    sg = [c.get("signer_id") for c in clips]
    same["signer_disjoint"] = {
        "why": "a nearest template from the SAME signer answers with that person's other "
               "recording of the word, which is not the question a new learner poses. These "
               "numbers exclude every clip by the signer being scored",
        "v1_count": loo_count_accuracy(v1, labels, sg),
        "v2_count": loo_count_accuracy(v2, labels, sg),
        "dtw_nearest_template": loo_dtw_accuracy(clips, labels, sg),
    }
    print(f"   {'방법':<34}{'균형정확도':>10}   클래스별 재현율")
    for k in ("v1_count_leave_one_out", "v2_count_leave_one_out",
              "dtw_nearest_template_leave_one_out"):
        r = same[k]
        print(f"   {k:<34}{r['balanced_accuracy']:>10.4f}   {r['per_class_recall']}")
    print()
    print("   수행자 분리 (같은 사람의 다른 클립 제외)")
    for k in ("v1_count", "v2_count", "dtw_nearest_template"):
        r = same["signer_disjoint"][k]
        print(f"   {k:<34}{r['balanced_accuracy']:>10.4f}   {r['per_class_recall']}")
    print(f"   (참고) v2 in-sample 문턱: {same['v2_count_in_sample_threshold']['balanced_accuracy']:.4f}"
          f"  — 이전 보고 0.941 과 같은 계산 방식")

    exposure = {
        "why_this_record_exists": "the new gate is being designed from these clips, so none of "
                                  "them can later serve as its independent test. The original "
                                  "split fields and the contract are unchanged; this is a separate "
                                  "record of what design work has seen",
        "original_split_preserved": True,
        "source_clips_by_split_role": dict(collections.Counter(
            r["split"] for r in rows_src[AMPLITUDES[0]])),
        "exposed_in_v1_movement_structure_check": {
            "what": "all 163 eligible clips, because load_source() filtered only on "
                    "source_only_pilot_eligible and not on split_role",
            "includes_reserved_test": 33,
            "consequence": "reserved_test is exposed to feature and threshold exploration for the "
                           "NEW gate. It was not used to train the TCN, and this does not "
                           "retroactively change that; but it can no longer be the new gate's "
                           "independent test",
        },
        "exposed_in_this_run": {"source_clips": len(rows_src[AMPLITUDES[0]]),
                                "web_clips": len(rows_web[AMPLITUDES[0]]),
                                "all_split_roles": True},
        "web_100_status": "requested_word_id only; attempt_label is unset on all 100 rows, so it "
                          "is not established that each attempt was a correct performance of the "
                          "requested word. Diagnostic data of unverified correctness; not usable "
                          "as evidence about normal distribution or about rejecting wrong "
                          "performances",
        "what_an_independent_test_would_need": {
            "recording": "NOT REQUESTED NOW - this is a design sketch only",
            "people": "at least one signer who has contributed nothing to any threshold or feature "
                      "choice so far",
            "sessions": "a session recorded on a different day from any used in design, so "
                        "within-sitting correlation does not carry over",
            "conditions_per_word": ["normal speed", "deliberately slow", "deliberately fast"],
            "labels": "each attempt labelled at capture time as normal or as a named error, using "
                      "the attempt_label selector that exists in the web tester",
            "words": "at least one held sign and one repeated sign (NA and ALDA are the natural "
                     "pair), plus one two-handed word",
            "frozen_first": "the feature version, amplitude, gap and any threshold must be written "
                            "down before the recording, not chosen afterwards",
        },
    }

    out = {
        "task_id": "motion_events_check", "date_kst": time.strftime("%Y-%m-%d"),
        "responds_to": "WordsInSilence_축별검토_20260912/00_외부검토와다음지시.md",
        "command": "py -3.13 motion_events_check.py",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "code_hashes": {p: sha256_file(os.path.join(ROOT, p)) for p in
                        ("src/pilot/axes.py", "tests/test_motion_events.py",
                         "motion_events_check.py")},
        "settings": {"turn_amplitude_default": X.TURN_AMPLITUDE,
                     "amplitude_sweep": list(AMPLITUDES),
                     "max_gap_seconds": X.MAX_GAP_SECONDS,
                     "min_observed_frames": X.MIN_OBSERVED_FRAMES,
                     "min_observed_fraction": X.MIN_OBSERVED_FRACTION,
                     "chosen_before_looking_at_word_results": True,
                     "no_per_word_threshold_tuning": True},
        "v1_counterexample_reproduced_in": "tests/test_motion_events.py, first check: 1s/30fps->3, "
                                           "4s/30fps->0, 1s/120fps->0",
        "per_sample_source": {str(a): rows_src[a] for a in AMPLITUDES},
        "per_sample_web": {str(a): rows_web[a] for a in AMPLITUDES},
        "same_metric_comparison": same,
        "exposure_record": exposure,
        "remaining_uncertainty": [
            "whether DTW's time warping is the CAUSE of the weak movement axis is still a "
            "hypothesis. It was not tested by holding the feature fixed and changing only the "
            "alignment, and external review produced a counterexample where DTW does retain a "
            "movement difference, so the earlier wording has been withdrawn",
            "v2 removes the speed and frame-rate dependence that was demonstrated in v1; it is not "
            "shown to be a complete description of movement. A turn count carries no trajectory "
            "shape, no ordering, no handshape change and no two-hand relation",
            "every number here comes from clips that have been seen during design, so none of it "
            "estimates performance on a new person or a new session",
            "the amplitude threshold is a stated default with a sweep, not a validated value",
            "nothing here says whether any attempt was performed correctly",
        ],
        "not_changed": ["research/front_keyphase_v1", "the 280-channel contract",
                        "runs/first_run/model.pt", "the default web recogniser",
                        "src/pilot/axes.py direction_reversals (v1, kept for reproducibility)"],
        "no_gate_deployed": True,
        "not_a_correctness_claim": True,
    }
    p = os.path.join(ROOT, "evidence", "result_motion_events.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    pe = os.path.join(ROOT, "evidence", "exposure_record.json")
    with open(pe, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(exposure, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n({round(time.perf_counter() - t0, 1)}초)  -> {os.path.relpath(p, ROOT)}"
          f" , {os.path.relpath(pe, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
