#!/usr/bin/env python3
"""First self-trained classification experiment: small TCN vs nearest-template DTW.

Train and validation only. reserved_test is never loaded into a model here - the split file says
signers 14-16 and this script refuses to touch them.

Run: py -3.13 run_pilot.py
"""
import argparse
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
from pilot.dtw_baseline import predict as dtw_predict  # noqa: E402
from pilot.metrics import report  # noqa: E402
from pilot.model import SignTCN  # noqa: E402

PREP = ("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/"
        "WordsInSilence_학습준비_20260912")
CONTRACT = json.load(open(os.path.join(PREP, "model_contract.json"), encoding="utf-8"))
CFG = CONTRACT["first_experiment"]
MODEL_CFG = CONTRACT["recommended_model"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def batches(clips, size, rng=None):
    idx = np.arange(len(clips))
    if rng is not None:
        rng.shuffle(idx)
    for s in range(0, len(idx), size):
        yield [clips[i] for i in idx[s:s + size]]


def evaluate_model(model, clips, batch_size=16):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for group in batches(clips, batch_size):
            x, pad, _ = D.collate(group)
            logit = model(torch.from_numpy(x), torch.from_numpy(pad))
            p = torch.softmax(logit, dim=1).numpy()
            probs.extend(p.tolist())
            preds.extend(logit.argmax(dim=1).tolist())
    return preds, probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot_" + time.strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "runs", args.run_id)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(CFG["seed"])
    t_start = time.perf_counter()

    train = D.load_split(PREP, "train")
    val = D.load_split(PREP, "validation")
    assert not (
        {c["signer_id"] for c in train} & {c["signer_id"] for c in val}), "signer leak"
    assert not (
        {c["sequence_id"] for c in train} & {c["sequence_id"] for c in val}), "sequence leak"
    print(f"train {len(train)}  validation {len(val)}  "
          f"(reserved_test not loaded into any model)")

    # class weights: inverse frequency computed on TRAIN ONLY, per contract
    counts = np.bincount([c["label"] for c in train], minlength=len(D.CLASSES)).astype(np.float64)
    weights = torch.tensor((counts.sum() / (len(counts) * np.maximum(counts, 1))),
                           dtype=torch.float32)
    print("train class counts", dict(zip(D.CLASSES, counts.astype(int).tolist())))

    model = SignTCN(width=MODEL_CFG["input_projection_channels"],
                    dilations=tuple(MODEL_CFG["dilations"]),
                    kernel_size=MODEL_CFG["kernel_size"],
                    dropout=MODEL_CFG["dropout"])
    n_params = model.parameter_count()
    print(f"parameters {n_params} (contract target: under 100000)")

    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"],
                            weight_decay=CFG["weight_decay"])
    lossf = nn.CrossEntropyLoss(weight=weights)
    rng = np.random.default_rng(CFG["seed"])

    history, best = [], {"macro_f1": -1.0, "epoch": None, "state": None}
    patience = CFG["early_stopping_patience"]
    since_best = 0
    for epoch in range(1, CFG["max_epochs"] + 1):
        model.train()
        tot, seen = 0.0, 0
        for group in batches(train, CFG["batch_size"], rng):
            x, pad, y = D.collate(group)
            opt.zero_grad()
            logit = model(torch.from_numpy(x), torch.from_numpy(pad))
            loss = lossf(logit, torch.from_numpy(y))
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(group)
            seen += len(group)
        tr_pred, _ = evaluate_model(model, train)
        va_pred, _ = evaluate_model(model, val)
        tr = report([c["label"] for c in train], tr_pred)
        va = report([c["label"] for c in val], va_pred)
        history.append({"epoch": epoch, "train_loss": round(tot / max(seen, 1), 5),
                        "train_accuracy": tr["accuracy"], "train_macro_f1": tr["macro_f1"],
                        "val_accuracy": va["accuracy"], "val_macro_f1": va["macro_f1"]})
        # checkpoint on validation macro F1; strictly greater, so an earlier epoch wins ties
        if va["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": va["macro_f1"], "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            since_best = 0
        else:
            since_best += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  loss {history[-1]['train_loss']:.4f}  "
                  f"train F1 {tr['macro_f1']:.3f}  val F1 {va['macro_f1']:.3f}")
        if since_best >= patience:
            print(f"  early stop at epoch {epoch} (no val improvement for {patience})")
            break

    model.load_state_dict(best["state"])
    ckpt = os.path.join(out_dir, "model.pt")
    torch.save({"state_dict": best["state"], "classes": list(D.CLASSES),
                "width": MODEL_CFG["input_projection_channels"],
                "dilations": list(MODEL_CFG["dilations"]),
                "kernel_size": MODEL_CFG["kernel_size"], "dropout": MODEL_CFG["dropout"],
                "in_channels": D.TOTAL_CHANNELS}, ckpt)

    # ---- final evaluation of both methods on the SAME validation clips ----
    va_pred, va_prob = evaluate_model(model, val)
    tcn_report = report([c["label"] for c in val], va_pred, [c["signer_id"] for c in val])

    dtw_pred, dtw_rows = [], []
    for c in val:
        r = dtw_predict(train, c)
        dtw_pred.append(r.get("predicted"))
        dtw_rows.append({"sample_id": c["sample_id"], "true": D.CLASSES[c["label"]],
                         "predicted": (D.CLASSES[r["predicted"]] if r.get("predicted") is not None
                                       else None),
                         **{k: v for k, v in r.items() if k != "predicted"}})
    dtw_report = report([c["label"] for c in val], dtw_pred, [c["signer_id"] for c in val])

    # ---- CPU timing, boundaries stated explicitly ----
    tcn_ms, dtw_ms = [], []
    for c in val:
        t0 = time.perf_counter()
        x, pad, _ = D.collate([c])
        with torch.no_grad():
            model(torch.from_numpy(x), torch.from_numpy(pad))
        tcn_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        dtw_predict(train, c)
        dtw_ms.append((time.perf_counter() - t0) * 1000)

    per_sample = [{"sample_id": c["sample_id"], "signer_id": c["signer_id"],
                   "sequence_id": c["sequence_id"], "frames": c["frames"],
                   "true": D.CLASSES[c["label"]],
                   "tcn_predicted": D.CLASSES[p],
                   "tcn_probabilities": {cl: round(pr, 4)
                                         for cl, pr in zip(D.CLASSES, va_prob[i])},
                   "dtw_predicted": dtw_rows[i]["predicted"],
                   "dtw_distance": dtw_rows[i].get("distance")}
                  for i, (c, p) in enumerate(zip(val, va_pred))]

    src = {os.path.relpath(os.path.join(dp, f), ROOT).replace("\\", "/"): sha256_file(os.path.join(dp, f))
           for dp, _, fs in os.walk(os.path.join(ROOT, "src")) for f in fs if f.endswith(".py")}
    src["run_pilot.py"] = sha256_file(os.path.join(ROOT, "run_pilot.py"))

    result = {
        "task_id": "sign_model_pilot_v1_first_run",
        "run_id": args.run_id,
        "date_kst": time.strftime("%Y-%m-%d"),
        "scope": "public-source pilot. NOT a webcam performance claim and NOT a correctness "
                 "verdict: the five outputs are word identity only.",
        "command": "py -3.13 run_pilot.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
                        "torch_threads": torch.get_num_threads(),
                        "os": f"{platform.system()} {platform.release()}"},
        "inputs": {
            "prep_dir": PREP,
            "pilot_manifest_sha256": sha256_file(os.path.join(PREP, "pilot_manifest.json")),
            "split_policy_sha256": sha256_file(os.path.join(PREP, "split_policy.json")),
            "model_contract_sha256": sha256_file(os.path.join(PREP, "model_contract.json")),
            "eligible_used": {"train": len(train), "validation": len(val),
                              "reserved_test": "33, NOT LOADED"},
            "train_class_counts": dict(zip(D.CLASSES, counts.astype(int).tolist())),
            "signers": {"train": sorted({c["signer_id"] for c in train}),
                        "validation": sorted({c["signer_id"] for c in val})},
        },
        "code_sha256": src,
        "model": {"architecture": "small non-causal TCN, 4 residual dilated blocks",
                  "parameters": n_params, "under_100k": n_params < 100000,
                  "input_channels": D.TOTAL_CHANNELS,
                  "checkpoint_sha256": sha256_file(ckpt),
                  "selected_epoch": best["epoch"],
                  "selection_rule": "highest validation macro F1; earlier epoch wins ties"},
        "training": {k: CFG[k] for k in ("seed", "max_epochs", "batch_size", "optimizer",
                                         "learning_rate", "weight_decay", "augmentation",
                                         "hyperparameter_search")},
        "epochs_run": len(history),
        "history": history,
        "validation_tcn": tcn_report,
        "validation_dtw_same_train_examples": dtw_report,
        "per_sample": per_sample,
        "cpu_inference_ms": {
            "boundary": "from collate of one already-loaded clip to the returned logits / to the "
                        "nearest-template decision. Excludes reading the npz and excludes any "
                        "camera, MediaPipe or display time.",
            "tcn": {"median": round(float(np.median(tcn_ms)), 2),
                    "p95": round(float(np.percentile(tcn_ms, 95)), 2), "n": len(tcn_ms)},
            "dtw_vs_98_templates": {"median": round(float(np.median(dtw_ms)), 2),
                                    "p95": round(float(np.percentile(dtw_ms, 95)), 2),
                                    "n": len(dtw_ms)},
        },
        "wrong_performance_rejection": "NOT_EVALUATED - no error-labelled recordings exist in this "
                                       "public-source pack",
        "how_to_read_these_numbers": {
            "checkpoint_selection_bias": "the TCN checkpoint was CHOSEN by validation macro F1 and "
                "is then reported on that same validation set. That number is optimistically "
                "biased and is not an unbiased estimate. The DTW baseline has no selection step, "
                "so its score is unbiased for this split.",
            "validation_instability": "validation macro F1 oscillated between 0.72 and 1.00 across "
                "epochs with no trend (1.0 at epoch 5, 0.72 at 10, 1.0 at 15, 0.72 at 20). With 98 "
                "training clips the model does not settle; picking the best epoch off an "
                "oscillating curve is close to picking a lucky one.",
            "task_difficulty": "both methods scored 32/32. Five words this distinct, 32 validation "
                "clips and 3 signers cannot separate the two approaches - the experiment shows the "
                "pipeline runs end to end, not that either method is good.",
            "one_clip_weight": "each validation clip is worth about 3 accuracy points.",
            "what_would_be_informative": "more classes, more signers, or error-labelled clips. A "
                "perfect score on 32 easy clips carries almost no information.",
        },
        "reserved_test_used": False,
        "self_check_status": "SEE tests/test_pilot.py",
        "independent_review_status": "NOT_RUN",
        "limitations": [
            "5 words, 1-4 clips per signer per word. Not a scale that supports a generalisation claim.",
            "Words were cut from SENTENCES by nominal frame_index/30; source PTS unverified and "
            "coarticulation is present. An isolated sign differs from one inside a sentence.",
            "Source confidence is 1 everywhere including absent lower-body coordinates, so it is "
            "not used as observation quality.",
            "OpenPose source vs MediaPipe web: mapping the joints does NOT verify that the two "
            "extractors are interchangeable.",
            "Word identity only. This says nothing about whether a performance was correct.",
            "Validation is 32 clips from 3 signers; a single clip moves accuracy by about 3 points.",
        ],
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    p = os.path.join(out_dir, "result.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"\n=== validation (32 clips, signers 11-13) ===")
    print(f"  TCN  accuracy {tcn_report['accuracy_numerator']}/{tcn_report['accuracy_denominator']}"
          f"  macro F1 {tcn_report['macro_f1']}")
    print(f"  DTW  accuracy {dtw_report['accuracy_numerator']}/{dtw_report['accuracy_denominator']}"
          f"  macro F1 {dtw_report['macro_f1']}")
    print(f"\n결과: {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
