#!/usr/bin/env python3
"""Change ONLY the amount of training data, and see whether the transfer number follows.

expand_train.py improved the participant-webcam transfer from 11/40 to 22/40, but four things
changed at once: training clips 98 -> 2,095, width 32 -> 64, classes 5 -> 80, and the batch and
epoch budget. "More data helped" was therefore not something that run could establish. This one
holds everything fixed except the number of training clips.

Held fixed across every condition:
  the 80-class label set, the signer split, the architecture (width 64, dilations 1/2/4/8,
  kernel 3, dropout 0.1), AdamW lr 1e-3 wd 1e-3, batch 32, inverse-frequency class weights from
  that condition's own training clips, checkpoint by validation macro F1, and - importantly - the
  OPTIMISATION BUDGET: every condition gets the same 3,960 gradient steps, so a small subset simply
  sees more epochs rather than less training.

Varied: the training subset size. Subsets are nested and stratified by class, so the 25 percent set
is contained in the 50 percent set which is contained in the full set, and every class keeps its
share. Three seeds per size, because one run moving is not a trend.

DTW is run at the same three sizes too, since its "training data" is its template pool.

This script only reads existing NPZ files and writes runs/data_ablation_v1/. It does not touch
runs/first_run, runs/expanded_v1, research/front_keyphase_v1, the AIHub archives, the contract, or
the reserved_test split.

Run: py -3.13 data_ablation.py
"""
import collections
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from pilot import data as D  # noqa: E402
from pilot.dtw_baseline import dtw_distance  # noqa: E402
from pilot.model import SignTCN  # noqa: E402
from pilot.web_input import load_web_clip  # noqa: E402

OUT = os.path.join(ROOT, "expand")
RUN = os.path.join(ROOT, "runs", "data_ablation_v1")
SESSION = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs",
                                        "ses_cbd462050a95"))
PROJECT_5 = ("알다", "감사합니다", "나", "좋다", "모르다")
WEB_TO_LABEL = {"ALDA": "알다", "GAMSAHAMNIDA": "감사합니다"}

FRACTIONS = (0.25, 0.5, 1.0)
SEEDS = (20260912, 20260913, 20260914)
STEPS = 3960                # 60 epochs of the full set at batch 32; identical for every condition
EVAL_EVERY = 66             # one full-set epoch
PATIENCE_ROUNDS = 10
FIXED = {"width": 64, "dilations": (1, 2, 4, 8), "kernel_size": 3, "dropout": 0.1,
         "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-3,
         "augmentation": "none", "hyperparameter_search": "none"}


def load_split(manifest, role, classes):
    out = []
    for m in manifest:
        if m["split_role"] != role:
            continue
        with np.load(os.path.join(ROOT, m["npz"]), allow_pickle=False) as z:
            c = D.build_channels(z["xy_native"], z["coordinate_present"],
                                 z["timestamp_nominal_us"])
        c.update(label=classes.index(m["word_label"]), sample_id=m["sample_id"],
                 signer_id=m["signer_id"], word_label=m["word_label"])
        out.append(c)
    return out


def nested_subsets(train, fractions):
    """Stratified nested subsets: each class keeps its share and smaller sets are contained."""
    by = collections.defaultdict(list)
    for i, c in enumerate(train):
        by[c["label"]].append(i)
    rng = np.random.default_rng(11)                 # fixed, independent of the training seed
    order = {k: list(rng.permutation(v)) for k, v in by.items()}
    out = {}
    for f in sorted(fractions):
        keep = []
        for k, idxs in order.items():
            n = max(1, int(round(len(idxs) * f)))
            keep.extend(idxs[:n])
        out[f] = sorted(keep)
    for a, b in zip(sorted(fractions), sorted(fractions)[1:]):
        assert set(out[a]) <= set(out[b]), "subsets are not nested"
    return out


def evaluate(model, clips, n_classes):
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for i in range(0, len(clips), 64):
            x, pad, y = D.collate(clips[i:i + 64])
            pred.extend(model(torch.from_numpy(x), torch.from_numpy(pad)).argmax(1).tolist())
            true.extend(y.tolist())
    pred, true = np.asarray(pred), np.asarray(true)
    f1s = []
    for k in range(n_classes):
        m = true == k
        if not m.any():
            continue
        tp = int((pred[m] == k).sum())
        rec, prec = tp / int(m.sum()), tp / max(int((pred == k).sum()), 1)
        f1s.append(0.0 if tp == 0 else 2 * prec * rec / (prec + rec))
    return {"accuracy": f"{int((pred == true).sum())}/{len(true)}",
            "accuracy_value": round(float((pred == true).mean()), 4),
            "macro_f1": round(float(np.mean(f1s)), 4)}


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


def transfer(model, web, classes):
    five = [classes.index(w) for w in PROJECT_5]
    per5 = collections.defaultdict(lambda: [0, 0])
    per80 = collections.defaultdict(lambda: [0, 0])
    model.eval()
    for c in web:
        x, pad, _ = D.collate([c])
        with torch.no_grad():
            logit = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
        p5 = PROJECT_5[int(np.argmax([logit[i] for i in five]))]
        p80 = classes[int(np.argmax(logit))]
        per5[c["word_label"]][1] += 1
        per5[c["word_label"]][0] += int(p5 == c["word_label"])
        per80[c["word_label"]][1] += 1
        per80[c["word_label"]][0] += int(p80 == c["word_label"])
    def pack(per):
        hit = sum(v[0] for v in per.values())
        tot = sum(v[1] for v in per.values())
        return {"correct": f"{hit}/{tot}", "value": round(hit / tot, 4),
                "per_word": {k: f"{v[0]}/{v[1]}" for k, v in per.items()}}
    return {"five_way": pack(per5), "eighty_way": pack(per80)}


def train_one(train_sub, val, classes, seed, web):
    torch.manual_seed(seed)
    np.random.seed(seed)
    counts = collections.Counter(c["label"] for c in train_sub)
    w = torch.tensor([len(train_sub) / (len(classes) * max(counts[k], 1))
                      for k in range(len(classes))], dtype=torch.float32)
    model = SignTCN(width=FIXED["width"], dilations=FIXED["dilations"],
                    kernel_size=FIXED["kernel_size"], dropout=FIXED["dropout"],
                    n_classes=len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=FIXED["learning_rate"],
                            weight_decay=FIXED["weight_decay"])
    lossf = nn.CrossEntropyLoss(weight=w)
    rng = np.random.default_rng(seed)
    order, cursor = rng.permutation(len(train_sub)), 0
    best = {"macro_f1": -1.0, "step": 0, "state": None}
    rounds_since = 0
    for step in range(1, STEPS + 1):
        if cursor + FIXED["batch_size"] > len(order):
            order, cursor = rng.permutation(len(train_sub)), 0
        idx = order[cursor:cursor + FIXED["batch_size"]]
        cursor += FIXED["batch_size"]
        model.train()
        x, pad, y = D.collate([train_sub[j] for j in idx])
        opt.zero_grad()
        loss = lossf(model(torch.from_numpy(x), torch.from_numpy(pad)), torch.from_numpy(y))
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            vm = evaluate(model, val, len(classes))
            if vm["macro_f1"] > best["macro_f1"]:
                best = {"macro_f1": vm["macro_f1"], "step": step,
                        "state": {k: v.clone() for k, v in model.state_dict().items()},
                        "val": vm}
                rounds_since = 0
            else:
                rounds_since += 1
                if rounds_since >= PATIENCE_ROUNDS:
                    break
    model.load_state_dict(best["state"])
    return {"validation": best["val"], "checkpoint_step": best["step"],
            "steps_run": step, "transfer": transfer(model, web, classes)}


def dtw_transfer(train_sub, web):
    tmpl = [c for c in train_sub if c["word_label"] in PROJECT_5]
    per = collections.defaultdict(lambda: [0, 0])
    for c in web:
        best = min(((dtw_distance(c["x"], t["x"]), t["word_label"]) for t in tmpl),
                   key=lambda z: (z[0] is None, z[0]))
        per[c["word_label"]][1] += 1
        per[c["word_label"]][0] += int(best[1] == c["word_label"])
    hit = sum(v[0] for v in per.values())
    tot = sum(v[1] for v in per.values())
    return {"templates": len(tmpl), "correct": f"{hit}/{tot}", "value": round(hit / tot, 4),
            "per_word": {k: f"{v[0]}/{v[1]}" for k, v in per.items()}}


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RUN, exist_ok=True)
    manifest = json.load(open(os.path.join(OUT, "manifest.json"), encoding="utf-8"))
    classes = sorted({m["word_label"] for m in manifest})
    train = load_split(manifest, "train", classes)
    val = load_split(manifest, "validation", classes)
    held = {m["sample_id"] for m in manifest if m["split_role"] == "reserved_test"}
    loaded = {c["sample_id"] for c in train} | {c["sample_id"] for c in val}
    assert held and not (held & loaded), "a reserved_test sample was loaded"
    web = load_web()
    subsets = nested_subsets(train, FRACTIONS)
    print(f"클래스 {len(classes)}  전체 학습 {len(train)}  검증 {len(val)}  웹 {len(web)}")
    print(f"부분집합: " + ", ".join(f"{int(f*100)}%={len(v)}건" for f, v in sorted(subsets.items()))
          + "  (중첩 확인됨)")
    print(f"모든 조건 동일 예산 {STEPS} 스텝, 구조·optimizer·클래스·분할 고정\n")

    results = []
    for f in FRACTIONS:
        sub = [train[i] for i in subsets[f]]
        dtw = dtw_transfer(sub, web)
        for seed in SEEDS:
            r = train_one(sub, val, classes, seed, web)
            results.append({"fraction": f, "training_clips": len(sub), "seed": seed,
                            "clips_per_class_median": float(np.median(list(
                                collections.Counter(c["label"] for c in sub).values()))),
                            **r, "dtw_same_data": dtw})
            print(f"  {int(f*100):>3}% ({len(sub):>4}건) seed {seed}  "
                  f"검증 macro F1 {r['validation']['macro_f1']:.4f}  "
                  f"전이 5지 {r['transfer']['five_way']['correct']:<6} "
                  f"80지 {r['transfer']['eighty_way']['correct']:<6} "
                  f"({time.perf_counter()-t0:.0f}초)", flush=True)

    print(f"\n{'자료량':>8} {'학습건수':>8} {'검증 macro F1':>18} {'전이 5지선다':>20} {'DTW 5지':>12}")
    summary = []
    for f in FRACTIONS:
        rs = [r for r in results if r["fraction"] == f]
        vf = [r["validation"]["macro_f1"] for r in rs]
        t5 = [r["transfer"]["five_way"]["value"] for r in rs]
        t80 = [r["transfer"]["eighty_way"]["value"] for r in rs]
        row = {"fraction": f, "training_clips": rs[0]["training_clips"],
               "val_macro_f1_mean": round(float(np.mean(vf)), 4),
               "val_macro_f1_spread": [round(min(vf), 4), round(max(vf), 4)],
               "transfer_five_way_mean": round(float(np.mean(t5)), 4),
               "transfer_five_way_spread": [round(min(t5), 4), round(max(t5), 4)],
               "transfer_eighty_way_mean": round(float(np.mean(t80)), 4),
               "dtw_five_way": rs[0]["dtw_same_data"]["value"],
               "per_seed_five_way": {r["seed"]: r["transfer"]["five_way"]["correct"] for r in rs}}
        summary.append(row)
        print(f"{int(f*100):>7}% {row['training_clips']:>8} "
              f"{row['val_macro_f1_mean']:>10.4f} [{row['val_macro_f1_spread'][0]:.3f}-"
              f"{row['val_macro_f1_spread'][1]:.3f}] "
              f"{row['transfer_five_way_mean']:>12.4f} [{row['transfer_five_way_spread'][0]:.2f}-"
              f"{row['transfer_five_way_spread'][1]:.2f}] {row['dtw_five_way']:>10.4f}")

    out = {
        "task_id": "data_ablation", "date_kst": time.strftime("%Y-%m-%d"),
        "question": "with everything else held fixed, does more training data move the "
                    "participant-webcam transfer number",
        "why": "expand_train.py changed data, width, class count and budget together, so it could "
               "not attribute its improvement to data",
        "command": "py -3.13 data_ablation.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "held_fixed": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in FIXED.items()},
                       "classes": len(classes), "label_set": "identical in every condition",
                       "split": "train 01-09 / validation 11-13; reserved_test never loaded",
                       "optimisation_budget_steps": STEPS,
                       "budget_note": "identical step count everywhere, so a smaller subset sees "
                                      "more epochs rather than less training",
                       "early_stopping": f"{PATIENCE_ROUNDS} evaluation rounds without improvement"},
        "varied": {"training_subset_fractions": list(FRACTIONS), "seeds": list(SEEDS),
                   "subsets": "stratified by class and nested; the smaller is contained in the "
                              "larger, verified by assertion"},
        "per_run": results,
        "summary": summary,
        "limits": [
            "the transfer test is two words, one performer, one session, and that session predates "
            "attempt_label so it is not established that each attempt was a correct performance",
            "validation is public sentence-cut clips from 3 signers; it is not a webcam estimate",
            "this varies how many clips, not how many signers; data diversity is a separate axis "
            "that is not tested here",
            "three seeds bound run-to-run noise, they do not make a small difference significant",
            "nothing here judges whether any attempt was performed correctly",
        ],
        "wrote_only": "runs/data_ablation_v1/",
        "untouched": ["runs/first_run", "runs/expanded_v1", "research/front_keyphase_v1",
                      "the 280-channel contract", "the AIHub archives", "reserved_test"],
        "not_a_correctness_claim": True,
    }
    p = os.path.join(RUN, "result.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n({time.perf_counter()-t0:.0f}초)  -> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
