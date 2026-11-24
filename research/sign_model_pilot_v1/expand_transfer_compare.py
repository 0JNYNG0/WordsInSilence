#!/usr/bin/env python3
"""Compare the expanded run with the pilot on the SAME measuring stick.

expand_train.py reported TCN 20/40 and DTW 23/40 on the participant's webcam clips, but that is an
80-way choice while the pilot's TCN 11/40 and DTW 40/40 were a 5-way choice. Putting those numbers
side by side would repeat exactly the error external review caught earlier: comparing quantities
that are not the same quantity. So the 5-way task is reconstructed here - the model's logits are
restricted to the five project classes, and DTW's templates to the same five words' training clips
- and both the 5-way and the 80-way figures are reported with their denominators stated.

Read-only. No training, no camera, no change to any recogniser.

Run: py -3.13 expand_transfer_compare.py
"""
import collections
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from pilot import data as D  # noqa: E402
from pilot.dtw_baseline import dtw_distance  # noqa: E402
from pilot.model import SignTCN  # noqa: E402
from pilot.web_input import load_web_clip  # noqa: E402

RUN = os.path.join(ROOT, "runs", "expanded_v1")
OUT = os.path.join(ROOT, "expand")
SESSION = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs",
                                        "ses_cbd462050a95"))
PROJECT_5 = ("알다", "감사합니다", "나", "좋다", "모르다")
WEB_TO_LABEL = {"ALDA": "알다", "GAMSAHAMNIDA": "감사합니다"}


def load_web():
    rows = [json.loads(l) for l in
            open(os.path.join(SESSION, "decisions.jsonl"), encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        w = r.get("requested_word_id")
        if r.get("observation_status") != "OK" or w not in WEB_TO_LABEL:
            continue
        p = os.path.join(SESSION, "trials", r["trial_id"] + ".npz")
        if os.path.exists(p):
            c = load_web_clip(p)
            c.update(trial_id=r["trial_id"], word_label=WEB_TO_LABEL[w], label=0)
            out.append(c)
    return out


def load_train(manifest, words=None):
    out = []
    for m in manifest:
        if m["split_role"] != "train":
            continue
        if words and m["word_label"] not in words:
            continue
        with np.load(os.path.join(ROOT, m["npz"]), allow_pickle=False) as z:
            c = D.build_channels(z["xy_native"], z["coordinate_present"],
                                 z["timestamp_nominal_us"])
        c["word_label"] = m["word_label"]
        out.append(c)
    return out


def score(rows, key):
    per = collections.defaultdict(lambda: {"n": 0, "hit": 0})
    for r in rows:
        p = per[r["true"]]
        p["n"] += 1
        p["hit"] += int(r[key] == r["true"])
    total = sum(p["n"] for p in per.values())
    hit = sum(p["hit"] for p in per.values())
    recalls = [p["hit"] / p["n"] for p in per.values()]
    return {"correct": f"{hit}/{total}",
            "balanced_accuracy": round(float(np.mean(recalls)), 4),
            "per_word": {w: f"{p['hit']}/{p['n']}" for w, p in per.items()}}


def main() -> int:
    t0 = time.perf_counter()
    ck = torch.load(os.path.join(RUN, "model.pt"), weights_only=True)
    classes = ck["classes"]
    model = SignTCN(in_channels=ck["in_channels"], width=ck["width"],
                    dilations=tuple(ck["dilations"]), kernel_size=ck["kernel_size"],
                    dropout=ck["dropout"], n_classes=len(classes))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    manifest = json.load(open(os.path.join(OUT, "manifest.json"), encoding="utf-8"))
    web = load_web()
    five = [classes.index(w) for w in PROJECT_5]
    print(f"참가자 웹캠 {len(web)}건, 모델 클래스 {len(classes)}개, "
          f"5지선다 복원용 인덱스 {five}")

    rows = []
    for c in web:
        x, pad, _ = D.collate([c])
        with torch.no_grad():
            logit = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
        order = np.argsort(-logit)
        rank = int(order.tolist().index(classes.index(c["word_label"]))) + 1
        rows.append({"trial_id": c["trial_id"], "true": c["word_label"],
                     "tcn80": classes[int(order[0])],
                     "tcn80_top3": [classes[i] for i in order[:3]],
                     "tcn80_rank_of_true": rank,
                     "tcn5": PROJECT_5[int(np.argmax([logit[i] for i in five]))]})

    tmpl5 = load_train(manifest, set(PROJECT_5))
    tmpl80 = None
    print(f"5지선다 DTW 템플릿 {len(tmpl5)}개 대조 중...")
    for r, c in zip(rows, web):
        best = min(((dtw_distance(c["x"], t["x"]), t["word_label"]) for t in tmpl5),
                   key=lambda z: (z[0] is None, z[0]))
        r["dtw5"] = best[1]

    top3 = sum(1 for r in rows if r["true"] in r["tcn80_top3"])
    out = {
        "task_id": "expand_transfer_compare", "date_kst": time.strftime("%Y-%m-%d"),
        "why": "the expanded run is an 80-way choice and the pilot was a 5-way choice; comparing "
               "them directly would repeat the metric mismatch external review caught",
        "command": "py -3.13 expand_transfer_compare.py",
        "checkpoint": {"run": "runs/expanded_v1", "epoch": ck["epoch"],
                       "classes": len(classes)},
        "test_clips": {"source": "front_keyphase_v1/runs/ses_cbd462050a95",
                       "extractor": "MediaPipe", "performer": "project participant",
                       "words": sorted({r["true"] for r in rows}), "clips": len(rows),
                       "correctness_of_each_attempt": "UNVERIFIED - session predates attempt_label"},
        "five_way_same_task_as_pilot": {
            "task": "choose among 알다 / 감사합니다 / 나 / 좋다 / 모르다",
            "chance": "1/5",
            "tcn_expanded": score(rows, "tcn5"),
            "dtw_expanded_templates": score(rows, "dtw5"),
            "pilot_for_reference": {"tcn": "11/40", "dtw": "40/40",
                                    "caveat": "the pilot model was trained on 98 clips of these "
                                              "same 5 words; the expanded model has never been "
                                              "trained on a 5-way task"},
        },
        "eighty_way": {
            "task": "choose among all 80 trained words", "chance": "1/80",
            "tcn": score(rows, "tcn80"),
            "tcn_top3": f"{top3}/{len(rows)}",
            "tcn_median_rank_of_true": float(np.median([r["tcn80_rank_of_true"] for r in rows])),
        },
        "failure_pattern": {
            "observation": "the two methods fail on opposite words",
            "tcn": "감사합니다 perfect, 알다 predicted as 육교 on all 20 clips",
            "dtw": "알다 perfect, 감사합니다 mostly predicted as 버스",
            "caution": "two words and one performer; this is a pattern to check, not a finding "
                       "about complementarity",
        },
        "per_clip": rows,
        "limits": [
            "two words only, one performer, one session; not a performance estimate",
            "the session predates attempt_label, so it is not established that each attempt was a "
            "correct performance of the requested word",
            "a label match between AIHub and the project word list is not proof of the same "
            "lexical variant",
            "the 5-way restriction is applied after training; the model never optimised for it",
            "nothing here judges whether any attempt was performed correctly",
        ],
        "not_a_correctness_claim": True,
    }
    p = os.path.join(RUN, "transfer_compare.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    f = out["five_way_same_task_as_pilot"]
    print(f"\n5지선다 (파일럿과 동일 과제, 무작위 20%)")
    print(f"  {'방법':<28}{'맞은 수':>10}{'균형정확도':>12}   단어별")
    for k, lbl in (("tcn_expanded", "TCN (확장 학습)"),
                   ("dtw_expanded_templates", "DTW (확장 템플릿)")):
        s = f[k]
        print(f"  {lbl:<28}{s['correct']:>10}{s['balanced_accuracy']:>12.4f}   {s['per_word']}")
    print(f"  참고: 파일럿(98건 학습, 5단어) TCN 11/40, DTW 40/40")
    e = out["eighty_way"]
    print(f"\n80지선다 (무작위 1.25%)  TCN {e['tcn']['correct']}  "
          f"top-3 {e['tcn_top3']}  정답 순위 중앙 {e['tcn_median_rank_of_true']:.0f}위")
    print(f"\n({time.perf_counter() - t0:.0f}초)  -> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
