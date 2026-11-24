#!/usr/bin/env python3
"""Step 3: retrain the TCN on the expanded vocabulary and retest the transfer failure.

The question this answers is narrow and was chosen before the run: the pilot TCN scored 32/32 on
its own validation split and then predicted GAMSAHAMNIDA for all 20 of the participant's ALDA
attempts. DTW carried the same 40 clips through the same conversion at 40/40. If 98 training clips
were the cause, more clips should move the transfer number. If it does not move, the cause is
somewhere else and the extra data has told us so.

What is held fixed from the pilot contract, so this is not a hyper-parameter hunt:
  seed 20260912, AdamW lr 1e-3 wd 1e-3, dilations (1,2,4,8), kernel 3, dropout 0.1,
  inverse-frequency class weights computed on train only, checkpoint by validation macro F1,
  signer split train 01-09 / validation 11-13 / reserved_test 14-16.

What changed, and why, decided up front and not searched:
  width 32 -> 64      the pooled vector has to separate 80 classes, not 5
  batch 16 -> 32      about 2,300 training clips instead of 98
  epochs 100 -> 60    with 24x the data an epoch is 24x the gradient steps

reserved_test is never loaded. runs/first_run/ is not touched; this writes runs/expanded_v1/.

Run: py -3.13 expand_train.py
"""
import collections
import glob
import hashlib
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
RUN = os.path.join(ROOT, "runs", "expanded_v1")
SESSION = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs",
                                        "ses_cbd462050a95"))
# the participant's words that also exist in AIHub; the only pair the transfer test can use
WEB_TO_LABEL = {"ALDA": "알다", "GAMSAHAMNIDA": "감사합니다"}
CFG = {"seed": 20260912, "width": 64, "dilations": (1, 2, 4, 8), "kernel_size": 3,
       "dropout": 0.1, "batch_size": 32, "max_epochs": 60, "patience": 10,
       "learning_rate": 1e-3, "weight_decay": 1e-3, "augmentation": "none",
       "hyperparameter_search": "none", "checkpoint_metric": "validation macro F1"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


def evaluate(model, clips, n_classes):
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for i in range(0, len(clips), 64):
            batch = clips[i:i + 64]
            x, pad, y = D.collate(batch)
            logit = model(torch.from_numpy(x), torch.from_numpy(pad))
            pred.extend(logit.argmax(dim=1).tolist())
            true.extend(y.tolist())
    pred, true = np.asarray(pred), np.asarray(true)
    recalls, f1s = {}, []
    for k in range(n_classes):
        m = true == k
        if not m.any():
            continue
        tp = int((pred[m] == k).sum())
        rec = tp / int(m.sum())
        prec = tp / max(int((pred == k).sum()), 1)
        recalls[k] = f"{tp}/{int(m.sum())}"
        f1s.append(0.0 if tp == 0 else 2 * prec * rec / (prec + rec))
    return {"accuracy": f"{int((pred == true).sum())}/{len(true)}",
            "accuracy_value": round(float((pred == true).mean()), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "classes_present": len(recalls), "per_class_recall": recalls}, pred, true


def load_web():
    rows = [json.loads(l) for l in
            open(os.path.join(SESSION, "decisions.jsonl"), encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        w = r.get("requested_word_id")
        if r.get("observation_status") != "OK" or w not in WEB_TO_LABEL:
            continue
        p = os.path.join(SESSION, "trials", r["trial_id"] + ".npz")
        if not os.path.exists(p):
            continue
        c = load_web_clip(p)
        # the requested word is the exercise label, used only to score afterwards; the model
        # never receives it
        c.update(trial_id=r["trial_id"], word_label=WEB_TO_LABEL[w], sample_id=r["trial_id"])
        out.append(c)
    return out


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RUN, exist_ok=True)
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    manifest = json.load(open(os.path.join(OUT, "manifest.json"), encoding="utf-8"))
    classes = sorted({m["word_label"] for m in manifest})
    train = load_split(manifest, "train", classes)
    val = load_split(manifest, "validation", classes)
    # reserved_test must not reach the model by any route, so check the loaded sample ids rather
    # than trusting the filter that produced them
    held = {m["sample_id"] for m in manifest if m["split_role"] == "reserved_test"}
    loaded = {c["sample_id"] for c in train} | {c["sample_id"] for c in val}
    assert held and not (held & loaded), "a reserved_test sample was loaded"
    print(f"클래스 {len(classes)}개  학습 {len(train)}건  검증 {len(val)}건  "
          f"(reserved_test 미로딩)")
    assert train and val
    assert not ({c["signer_id"] for c in train} & {c["signer_id"] for c in val}), "signer leak"

    counts = collections.Counter(c["label"] for c in train)
    w = torch.tensor([len(train) / (len(classes) * max(counts[k], 1))
                      for k in range(len(classes))], dtype=torch.float32)
    model = SignTCN(width=CFG["width"], dilations=CFG["dilations"],
                    kernel_size=CFG["kernel_size"], dropout=CFG["dropout"],
                    n_classes=len(classes))
    print(f"파라미터 {model.parameter_count():,}개  /  학습 예제 {len(train)}건  "
          f"= 예제당 {model.parameter_count() / len(train):.1f}")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"],
                            weight_decay=CFG["weight_decay"])
    lossf = nn.CrossEntropyLoss(weight=w)

    order = np.arange(len(train))
    best = {"macro_f1": -1.0, "epoch": None}
    history = []
    rng = np.random.default_rng(CFG["seed"])
    for epoch in range(1, CFG["max_epochs"] + 1):
        model.train()
        rng.shuffle(order)
        total = 0.0
        for i in range(0, len(order), CFG["batch_size"]):
            batch = [train[j] for j in order[i:i + CFG["batch_size"]]]
            x, pad, y = D.collate(batch)
            opt.zero_grad()
            out = model(torch.from_numpy(x), torch.from_numpy(pad))
            loss = lossf(out, torch.from_numpy(y))
            loss.backward()
            opt.step()
            total += float(loss) * len(batch)
        vm, _, _ = evaluate(model, val, len(classes))
        tm, _, _ = evaluate(model, train, len(classes))
        history.append({"epoch": epoch, "train_loss": round(total / len(train), 5),
                        "train_accuracy": tm["accuracy_value"], "train_macro_f1": tm["macro_f1"],
                        "val_accuracy": vm["accuracy_value"], "val_macro_f1": vm["macro_f1"]})
        if vm["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": vm["macro_f1"], "epoch": epoch}
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "in_channels": D.TOTAL_CHANNELS, "width": CFG["width"],
                        "dilations": list(CFG["dilations"]), "kernel_size": CFG["kernel_size"],
                        "dropout": CFG["dropout"], "epoch": epoch},
                       os.path.join(RUN, "model.pt"))
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  loss {history[-1]['train_loss']:.4f}  "
                  f"train F1 {tm['macro_f1']:.3f}  val F1 {vm['macro_f1']:.3f}  "
                  f"({time.perf_counter() - t0:.0f}초)", flush=True)
        if epoch - best["epoch"] >= CFG["patience"]:
            print(f"  조기 종료: {CFG['patience']} epoch 동안 개선 없음")
            break

    ck = torch.load(os.path.join(RUN, "model.pt"), weights_only=True)
    model.load_state_dict(ck["state_dict"])
    val_metrics, vpred, vtrue = evaluate(model, val, len(classes))
    print(f"\n검증 (수행자 11-13, {len(val)}건, {len(classes)}클래스): "
          f"정확도 {val_metrics['accuracy']}  macro F1 {val_metrics['macro_f1']}  "
          f"(epoch {ck['epoch']} 체크포인트)")

    web = load_web()
    transfer = {"note": "the participant's own webcam recordings, MediaPipe, converted through the "
                        "same common representation. Requested word used only for scoring.",
                "clips": len(web), "classes_in_model": len(classes)}
    if web:
        model.eval()
        rows = []
        for c in web:
            x, pad, _ = D.collate([{**c, "label": 0}])
            with torch.no_grad():
                logit = model(torch.from_numpy(x), torch.from_numpy(pad))
                prob = torch.softmax(logit, dim=1).numpy()[0]
            top = int(np.argmax(prob))
            rows.append({"trial_id": c["trial_id"], "true": c["word_label"],
                         "tcn_predicted": classes[top],
                         "tcn_confidence": round(float(prob[top]), 4),
                         "tcn_rank_of_true": int(
                             np.argsort(-prob).tolist().index(classes.index(c["word_label"]))) + 1})
        for r, c in zip(rows, web):
            best_d, best_lbl = None, None
            for t in train:
                d = dtw_distance(c["x"], t["x"])
                if d is not None and (best_d is None or d < best_d):
                    best_d, best_lbl = d, t["word_label"]
            r["dtw_predicted"] = best_lbl
            r["dtw_distance"] = None if best_d is None else round(best_d, 6)
        per = collections.defaultdict(lambda: {"n": 0, "tcn": 0, "dtw": 0})
        for r in rows:
            p = per[r["true"]]
            p["n"] += 1
            p["tcn"] += int(r["tcn_predicted"] == r["true"])
            p["dtw"] += int(r["dtw_predicted"] == r["true"])
        transfer["per_word"] = dict(per)
        transfer["tcn_correct"] = f"{sum(p['tcn'] for p in per.values())}/{len(rows)}"
        transfer["dtw_correct"] = f"{sum(p['dtw'] for p in per.values())}/{len(rows)}"
        transfer["tcn_median_rank_of_true"] = float(np.median([r["tcn_rank_of_true"]
                                                               for r in rows]))
        transfer["per_clip"] = rows
        transfer["chance_level"] = f"1/{len(classes)}"
        print(f"\n전이 시험 (참가자 웹캠 {len(rows)}건, {len(classes)}개 중 고르기, "
              f"무작위 기대 {1 / len(classes):.1%})")
        print(f"  TCN {transfer['tcn_correct']}   DTW {transfer['dtw_correct']}")
        for word, p in per.items():
            print(f"    {word:<10} n={p['n']:<3} TCN {p['tcn']:<3} DTW {p['dtw']}")
        print(f"  TCN이 정답을 매긴 순위 중앙값: {transfer['tcn_median_rank_of_true']:.0f}위 "
              f"/ {len(classes)}")

    result = {
        "task_id": "expand_train", "date_kst": time.strftime("%Y-%m-%d"),
        "question": "does a much larger training set fix the pilot TCN's transfer failure",
        "command": "py -3.13 expand_train.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()},
        "changed_from_pilot_and_why": {
            "width": "32 -> 64, because the pooled vector must separate 80 classes not 5",
            "batch_size": "16 -> 32, because there are about 2,300 training clips not 98",
            "max_epochs": "100 -> 60, because each epoch now has 24x the gradient steps",
            "no_search": "these were fixed before the run; no alternative was tried",
        },
        "data": {"classes": len(classes), "class_labels": classes,
                 "train_clips": len(train), "validation_clips": len(val),
                 "train_signers": sorted({c["signer_id"] for c in train}),
                 "validation_signers": sorted({c["signer_id"] for c in val}),
                 "reserved_test_loaded": False,
                 "manifest": "expand/manifest.json"},
        "model": {"parameter_count": model.parameter_count(),
                  "parameters_per_training_example": round(
                      model.parameter_count() / len(train), 2),
                  "pilot_comparison": "34,501 parameters over 98 clips = 352 per example"},
        "training_history": history,
        "checkpoint": {"epoch": ck["epoch"], "selected_by": CFG["checkpoint_metric"],
                       "selection_bias": "the checkpoint is chosen on the same validation split "
                                         "it is then reported on; that number is optimistic"},
        "validation": val_metrics,
        "transfer_to_participant_webcam": transfer,
        "code_hashes": {p: sha256_file(os.path.join(ROOT, p)) for p in
                        ("expand_build.py", "expand_train.py", "src/pilot/model.py",
                         "src/pilot/data.py")},
        "limits": [
            "validation is 3 signers of public sentence-cut clips; it is not a new-person webcam "
            "performance estimate",
            "the transfer test covers only the two words the participant recorded that AIHub also "
            "has, and the session predates attempt_label so correctness of each attempt is "
            "unverified",
            "a label match is not proof of the same lexical variant between the two sources",
            "the checkpoint is selected on the validation split it is reported on",
            "80 classes of sentence-cut words is not the project's 19-word task",
            "nothing here judges whether any attempt was performed correctly",
        ],
        "web_recogniser_unchanged": True,
        "pilot_run_untouched": "runs/first_run/ was not modified",
        "not_a_correctness_claim": True,
    }
    p = os.path.join(RUN, "result.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n({time.perf_counter() - t0:.0f}초)  -> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
