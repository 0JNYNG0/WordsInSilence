#!/usr/bin/env python3
"""Does the OpenPose-trained pilot model recognise MediaPipe input?

This is the transfer question the review brief asks as Q1, run on data that already exists:
two of the five pilot classes (GAMSAHAMNIDA, ALDA) were also recorded through the web path.

Read-only against research/front_keyphase_v1. No web file is modified and no camera is opened.

What a result can and cannot mean
  success  -> evidence that the common input carries across the two extractors for these words
  failure  -> AMBIGUOUS. It could be the extractor gap, or sentence-cut training clips versus
              isolated performances, or this performer differing from the 16 source signers.
              The three cannot be separated with what exists.

Run: py -3.13 cross_extractor_check.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from pilot import data as D  # noqa: E402
from pilot.dtw_baseline import predict as dtw_predict  # noqa: E402
from pilot.model import SignTCN  # noqa: E402
from pilot.web_input import collect  # noqa: E402

PREP = ("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/"
        "WordsInSilence_학습준비_20260912")
WEB_SESSION = os.path.join(ROOT, "..", "front_keyphase_v1", "runs", "ses_cbd462050a95")
OVERLAP = ("GAMSAHAMNIDA", "ALDA")
CKPT = os.path.join(ROOT, "runs", "first_run", "model.pt")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ck = torch.load(CKPT, weights_only=True)
    model = SignTCN(in_channels=ck["in_channels"], width=ck["width"],
                    dilations=tuple(ck["dilations"]), kernel_size=ck["kernel_size"],
                    dropout=ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    clips = collect(os.path.normpath(WEB_SESSION), OVERLAP)
    failed = [c for c in clips if "error" in c]
    clips = [c for c in clips if "error" not in c]
    print(f"MediaPipe 웹 기록 {len(clips)}건 변환 (변환 실패 {len(failed)}건)")
    if not clips:
        print("변환된 클립이 없습니다.")
        return 2

    train = D.load_split(PREP, "train")
    rows = []
    for c in clips:
        x, pad, _ = D.collate([c])
        with torch.no_grad():
            logit = model(torch.from_numpy(x), torch.from_numpy(pad))
            prob = torch.softmax(logit, dim=1).numpy()[0]
        tcn = D.CLASSES[int(np.argmax(prob))]
        d = dtw_predict(train, c)
        rows.append({
            "trial_id": c["trial_id"], "true": c["word_id"], "frames": c["frames"],
            "tcn_predicted": tcn,
            "tcn_probabilities": {cl: round(float(p), 4) for cl, p in zip(D.CLASSES, prob)},
            "dtw_predicted": (D.CLASSES[d["predicted"]] if d.get("predicted") is not None else None),
            "dtw_distance": d.get("distance"),
        })

    print(f"\n{'단어':<14} {'n':>3} {'TCN 맞음':>9} {'DTW 맞음':>9}   TCN 예측 분포")
    from collections import Counter
    summary = {}
    for w in OVERLAP:
        g = [r for r in rows if r["true"] == w]
        t = sum(1 for r in g if r["tcn_predicted"] == w)
        dd = sum(1 for r in g if r["dtw_predicted"] == w)
        dist = dict(Counter(r["tcn_predicted"] for r in g))
        summary[w] = {"n": len(g), "tcn_correct": t, "dtw_correct": dd,
                      "tcn_prediction_distribution": dist,
                      "dtw_prediction_distribution": dict(Counter(r["dtw_predicted"] for r in g))}
        print(f"{w:<14} {len(g):>3} {t:>7}/{len(g):<3} {dd:>7}/{len(g):<3}   {dist}")

    tot = len(rows)
    tcn_ok = sum(1 for r in rows if r["tcn_predicted"] == r["true"])
    dtw_ok = sum(1 for r in rows if r["dtw_predicted"] == r["true"])
    print(f"\n합계  TCN {tcn_ok}/{tot}   DTW {dtw_ok}/{tot}")
    print(f"참고: 클래스 5개 중 2개만 등장하므로 무작위 기대치는 {1/5:.0%}입니다.")

    out = {
        "task_id": "cross_extractor_transfer_check", "date_kst": time.strftime("%Y-%m-%d"),
        "question": "does a model trained on OpenPose source clips recognise MediaPipe web input",
        "command": "py -3.13 cross_extractor_check.py",
        "model_checkpoint_sha256": sha256_file(CKPT),
        "trained_on": "AIHub sentence-cut clips, OpenPose keypoints, signers 01-09",
        "tested_on": {"session": "front_keyphase_v1/runs/ses_cbd462050a95",
                      "extractor": "MediaPipe Tasks", "performer": "project participant",
                      "recording_type": "isolated signs, not sentence-cut",
                      "intended": "all correct performances (100-attempt calibration session)",
                      "classes_present": list(OVERLAP), "clips": tot,
                      "conversion_failures": len(failed)},
        "results": {"tcn_correct": f"{tcn_ok}/{tot}", "dtw_correct": f"{dtw_ok}/{tot}",
                    "per_class": summary, "chance_level": "1/5 classes"},
        "per_clip": rows,
        "interpretation": {
            "a_success_would_mean": "the common 280-channel input carries across the two "
                                    "extractors for these two words",
            "a_failure_is_ambiguous": "extractor gap, sentence-cut training vs isolated testing, "
                                      "and performer difference cannot be separated here",
            "not_a_correctness_claim": "word identity only; says nothing about whether the "
                                       "performances were well formed",
            "only_two_classes": "the model has five outputs but only two true classes appear, so "
                                "per-class counts matter more than the total",
        },
        "web_recogniser_modified": False,
        "self_check_status": "SEE tests/test_pilot.py and tests/test_web_input.py",
        "independent_review_status": "NOT_RUN",
    }
    p = os.path.join(ROOT, "evidence", "result_cross_extractor.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n-> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
