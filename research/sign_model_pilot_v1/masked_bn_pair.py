#!/usr/bin/env python3
"""One controlled pair: SignTCN versus MaskedSignTCN, everything else held identical.

TASK02 showed nn.BatchNorm1d mixes padded columns into its train-mode statistics, and that the
masked replacement removes that dependence. That is a structural fix with no evidence either way
about recognition. This runs the single comparison that can say something about it.

Controlled by construction, not by hope:
  * one initial state_dict is built once and copied into BOTH models; keys match, so the two start
    from bit-identical weights and buffers (asserted, not assumed).
  * one minibatch schedule is precomputed and shared, so both models see the same samples in the
    same order; its hash is recorded for each model and compared.
  * torch's global RNG is reseeded to SEED + step before every forward, so the dropout masks at a
    given step are identical for the two models.
  * exactly 1980 optimizer steps each, no early stopping. This is a NEW fixed budget for a quick
    diagnostic, about 30 full-set epochs. It is not the same budget as the earlier 3960 max-budget
    ablation and must not be described as such.

Selection: source validation every 66 steps, best by UNROUNDED macro F1, earlier step kept on a
tie. Both the selected checkpoint and the step-1980 checkpoint are saved, so the pair can be read
as a same-step comparison and as a selected-model comparison.

The 40 archived web trials are development diagnostic data. They are evaluated only after both
models have finished training and selection, they never touch training or checkpoint choice, and
requested_word_id is an exercise label - not a verdict that the attempt was performed correctly.
No number here is a real-performance or new-user figure.

reserved_test NPZ are never loaded. The existing model.py, both earlier checkpoints, the contract
and the web recogniser are untouched, and no candidate is promoted anywhere.

Run: py -3.13 masked_bn_pair.py
"""
import collections
import copy
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
from pilot.model import SignTCN  # noqa: E402
from pilot.model_masked_v1 import MaskedSignTCN  # noqa: E402
from pilot.web_input import load_web_clip  # noqa: E402

OUT = os.path.join(ROOT, "expand")
RUN = os.path.join(ROOT, "runs", "masked_bn_pair_v1")
SESSION = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs",
                                        "ses_cbd462050a95"))
PROJECT_5 = ("알다", "감사합니다", "나", "좋다", "모르다")
WEB_TO_LABEL = {"ALDA": "알다", "GAMSAHAMNIDA": "감사합니다"}
SEED = 20260912
STEPS = 1980
EVAL_EVERY = 66
THREADS = 4
CFG = {"width": 64, "dilations": (1, 2, 4, 8), "kernel_size": 3, "dropout": 0.1,
       "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-3,
       "augmentation": "none", "class_weights": "inverse frequency from train only",
       "early_stopping": "none", "seed": SEED, "steps": STEPS}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def state_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes())
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
                 signer_id=m["signer_id"], word_label=m["word_label"],
                 npz_sha256=m["npz_sha256"])
        out.append(c)
    return out


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
            c.update(trial_id=r["trial_id"], requested_word=WEB_TO_LABEL[w], label=0)
            out.append(c)
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
    return {"correct": int((pred == true).sum()), "total": int(len(true)),
            "macro_f1_unrounded": float(np.mean(f1s))}, pred, true


def schedule(n, steps, batch_size, seed):
    rng = np.random.default_rng(seed)
    order, cursor, out = rng.permutation(n), 0, []
    for _ in range(steps):
        if cursor + batch_size > len(order):
            order, cursor = rng.permutation(n), 0
        out.append(order[cursor:cursor + batch_size].tolist())
        cursor += batch_size
    return out


def train_one(cls, init_sd, train, val, classes, plan, weights, tag, log):
    model = cls(width=CFG["width"], dilations=CFG["dilations"], kernel_size=CFG["kernel_size"],
                dropout=CFG["dropout"], n_classes=len(classes))
    model.load_state_dict(copy.deepcopy(init_sd))
    assert state_hash(model.state_dict()) == state_hash(init_sd), "initial state diverged"
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"],
                            weight_decay=CFG["weight_decay"])
    lossf = nn.CrossEntropyLoss(weight=weights)
    history, best = [], None
    used = []
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        idx = plan[step - 1]
        used.append(idx)
        torch.manual_seed(SEED + step)               # identical dropout masks for both models
        model.train()
        x, pad, y = D.collate([train[j] for j in idx])
        opt.zero_grad()
        loss = lossf(model(torch.from_numpy(x), torch.from_numpy(pad)), torch.from_numpy(y))
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            vm, _, _ = evaluate(model, val, len(classes))
            history.append({"step": step, "train_loss": round(float(loss.detach()), 5), **vm})
            if best is None or vm["macro_f1_unrounded"] > best["macro_f1_unrounded"]:
                best = {**vm, "step": step,
                        "state": copy.deepcopy(model.state_dict())}     # ties keep the earlier step
            log.write(f"{tag} step {step:>5} loss {float(loss.detach()):.4f} "
                      f"val {vm['correct']}/{vm['total']} macroF1 {vm['macro_f1_unrounded']:.6f}\n")
            log.flush()
    return {"model": model, "history": history, "best": best,
            "schedule_hash": sha_text(json.dumps(used)),
            "steps_run": STEPS, "seconds": round(time.perf_counter() - t0, 1)}


def save_ckpt(path, sd, classes, step, kind, cls_name):
    torch.save({"state_dict": sd, "classes": classes, "step": step, "kind": kind,
                "model_type": getattr(MaskedSignTCN, "model_type", None)
                if cls_name == "MaskedSignTCN" else "wis_tcn",
                "model_version": getattr(MaskedSignTCN, "model_version", None)
                if cls_name == "MaskedSignTCN" else "pilot_v1",
                "class_name": cls_name, "in_channels": D.TOTAL_CHANNELS,
                "width": CFG["width"], "dilations": list(CFG["dilations"]),
                "kernel_size": CFG["kernel_size"], "dropout": CFG["dropout"],
                "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()}},
               path)
    return sha_file(path)


def web_scores(model, web, classes):
    five = [classes.index(w) for w in PROJECT_5]
    model.eval()
    rows = []
    for c in web:
        x, pad, _ = D.collate([c])
        with torch.no_grad():
            logit = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
        order = np.argsort(-logit)
        ti = classes.index(c["requested_word"])
        rows.append({
            "trial_id": c["trial_id"], "requested_word": c["requested_word"],
            "pred5": PROJECT_5[int(np.argmax([logit[i] for i in five]))],
            "pred80": classes[int(order[0])],
            "rank_of_requested_in_80": int(order.tolist().index(ti)) + 1,
            "logits_80": [round(float(v), 5) for v in logit],
        })
    return rows


def web_summary(rows, key):
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["requested_word"]][1] += 1
        per[r["requested_word"]][0] += int(r[key] == r["requested_word"])
    hit = sum(v[0] for v in per.values())
    tot = sum(v[1] for v in per.values())
    return {"matches_requested_word": f"{hit}/{tot}",
            "per_requested_word": {k: f"{v[0]}/{v[1]}" for k, v in per.items()},
            "label_meaning": "agreement with the word the exercise asked for; NOT a verdict that "
                             "the attempt was performed correctly"}


def main() -> int:
    t0 = time.perf_counter()
    os.makedirs(RUN, exist_ok=True)
    torch.set_num_threads(THREADS)
    log = open(os.path.join(RUN, "train.log"), "w", encoding="utf-8", newline="\n")

    manifest_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    classes = sorted({m["word_label"] for m in manifest})
    train = load_split(manifest, "train", classes)
    val = load_split(manifest, "validation", classes)
    held = {m["sample_id"] for m in manifest if m["split_role"] == "reserved_test"}
    loaded = {c["sample_id"] for c in train} | {c["sample_id"] for c in val}
    assert held and not (held & loaded), "a reserved_test sample was loaded"
    print(f"클래스 {len(classes)}  학습 {len(train)}  검증 {len(val)}  "
          f"(reserved_test {len(held)}건 미로딩)  threads={THREADS}")

    inputs = {"train": [{"sample_id": c["sample_id"], "npz_sha256": c["npz_sha256"]}
                        for c in train],
              "validation": [{"sample_id": c["sample_id"], "npz_sha256": c["npz_sha256"]}
                             for c in val]}
    with open(os.path.join(RUN, "inputs.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(inputs, fh, ensure_ascii=False)
        fh.write("\n")
    inputs_digest = sha_text(json.dumps(
        sorted((r["sample_id"], r["npz_sha256"]) for r in inputs["train"] + inputs["validation"])))

    counts = collections.Counter(c["label"] for c in train)
    weights = torch.tensor([len(train) / (len(classes) * max(counts[k], 1))
                            for k in range(len(classes))], dtype=torch.float32)
    torch.manual_seed(SEED)
    init_sd = copy.deepcopy(SignTCN(width=CFG["width"], dilations=CFG["dilations"],
                                    kernel_size=CFG["kernel_size"], dropout=CFG["dropout"],
                                    n_classes=len(classes)).state_dict())
    init_digest = state_hash(init_sd)
    plan = schedule(len(train), STEPS, CFG["batch_size"], SEED)
    print(f"공통 초기 state {init_digest[:16]}…  배치 일정 {sha_text(json.dumps(plan))[:16]}…")

    results = {}
    for cls, name in ((SignTCN, "SignTCN"), (MaskedSignTCN, "MaskedSignTCN")):
        print(f"  {name} 학습 {STEPS} step …", flush=True)
        r = train_one(cls, init_sd, train, val, classes, plan, weights, name, log)
        results[name] = r
        print(f"    완료 {r['seconds']}초  best step {r['best']['step']}  "
              f"macroF1 {r['best']['macro_f1_unrounded']:.6f}", flush=True)
    assert results["SignTCN"]["schedule_hash"] == results["MaskedSignTCN"]["schedule_hash"], \
        "the two models did not receive the same batch schedule"

    web = load_web()
    report = {}
    for name, r in results.items():
        model = r["model"]
        final_sd = copy.deepcopy(model.state_dict())
        entry = {"class_name": name, "seconds": r["seconds"], "steps_run": r["steps_run"],
                 "schedule_hash": r["schedule_hash"],
                 "parameter_count": model.parameter_count(), "history": r["history"],
                 "selection_rule": "highest unrounded validation macro F1; earlier step kept on a "
                                   "tie; no early stopping",
                 "checkpoints": {}, "validation": {}, "web_diagnostic": {}}
        for kind, sd, step in (("best", r["best"]["state"], r["best"]["step"]),
                               ("final", final_sd, STEPS)):
            path = os.path.join(RUN, f"{name}_{kind}.pt")
            entry["checkpoints"][kind] = {"path": os.path.relpath(path, ROOT).replace("\\", "/"),
                                          "step": step,
                                          "sha256": save_ckpt(path, sd, classes, step, kind, name),
                                          "state_hash": state_hash(sd)}
            model.load_state_dict(sd)
            vm, pred, true = evaluate(model, val, len(classes))
            entry["validation"][kind] = {
                **vm, "accuracy": f"{vm['correct']}/{vm['total']}",
                "per_sample": [{"sample_id": c["sample_id"], "true": classes[int(t)],
                                "pred": classes[int(p)]}
                               for c, t, p in zip(val, true, pred)]}
            rows = web_scores(model, web, classes)
            entry["web_diagnostic"][kind] = {
                "five_candidate_restricted": web_summary(rows, "pred5"),
                "all_eighty_candidates": web_summary(rows, "pred80"),
                "median_rank_of_requested_in_80": float(np.median(
                    [x["rank_of_requested_in_80"] for x in rows])),
                "per_trial": rows}
        model.load_state_dict(final_sd)
        report[name] = entry

    def changed(a, b, kind, field, key):
        return sum(1 for x, y in zip(report[a][field][kind]["per_sample" if field == "validation"
                                                           else "per_trial"],
                                     report[b][field][kind]["per_sample" if field == "validation"
                                                            else "per_trial"])
                   if x[key] != y[key])

    comparison = {}
    for kind in ("best", "final"):
        a, b = report["SignTCN"], report["MaskedSignTCN"]
        comparison[kind] = {
            "validation": {
                "SignTCN": {"accuracy": a["validation"][kind]["accuracy"],
                            "macro_f1_unrounded": a["validation"][kind]["macro_f1_unrounded"],
                            "step": a["checkpoints"][kind]["step"]},
                "MaskedSignTCN": {"accuracy": b["validation"][kind]["accuracy"],
                                  "macro_f1_unrounded": b["validation"][kind]["macro_f1_unrounded"],
                                  "step": b["checkpoints"][kind]["step"]},
                "predictions_changed_on_the_same_sample": changed(
                    "SignTCN", "MaskedSignTCN", kind, "validation", "pred"),
                "samples": a["validation"][kind]["total"]},
            "web_diagnostic": {
                "SignTCN": {
                    "five": a["web_diagnostic"][kind]["five_candidate_restricted"][
                        "matches_requested_word"],
                    "eighty": a["web_diagnostic"][kind]["all_eighty_candidates"][
                        "matches_requested_word"]},
                "MaskedSignTCN": {
                    "five": b["web_diagnostic"][kind]["five_candidate_restricted"][
                        "matches_requested_word"],
                    "eighty": b["web_diagnostic"][kind]["all_eighty_candidates"][
                        "matches_requested_word"]},
                "pred5_changed_on_the_same_trial": changed(
                    "SignTCN", "MaskedSignTCN", kind, "web_diagnostic", "pred5"),
                "pred80_changed_on_the_same_trial": changed(
                    "SignTCN", "MaskedSignTCN", kind, "web_diagnostic", "pred80"),
                "trials": len(web)},
        }

    print("\n비교 (같은 초기가중치·같은 배치일정·같은 dropout RNG)")
    for kind in ("best", "final"):
        c = comparison[kind]["validation"]
        w = comparison[kind]["web_diagnostic"]
        print(f"  [{kind}] 검증  SignTCN {c['SignTCN']['accuracy']} "
              f"F1 {c['SignTCN']['macro_f1_unrounded']:.6f} (step {c['SignTCN']['step']})"
              f"   Masked {c['MaskedSignTCN']['accuracy']} "
              f"F1 {c['MaskedSignTCN']['macro_f1_unrounded']:.6f} (step {c['MaskedSignTCN']['step']})"
              f"   바뀐 예측 {c['predictions_changed_on_the_same_sample']}/{c['samples']}")
        print(f"          웹 진단  5후보 {w['SignTCN']['five']} -> {w['MaskedSignTCN']['five']}"
              f"   80후보 {w['SignTCN']['eighty']} -> {w['MaskedSignTCN']['eighty']}"
              f"   바뀐 예측 5후보 {w['pred5_changed_on_the_same_trial']} / "
              f"80후보 {w['pred80_changed_on_the_same_trial']}")

    out = {
        "task_id": "autonomous_03", "date_kst": time.strftime("%Y-%m-%d"),
        "instruction": "WordsInSilence_자율협업_20260912/CLAUDE_TASK_03.md",
        "question": "with everything except the normalisation held identical, what does the masked "
                    "BatchNorm change",
        "command": "py -3.13 masked_bn_pair.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform(),
                        "torch_threads": THREADS},
        "budget": {"optimizer_steps_each": STEPS, "early_stopping": "none",
                   "evaluation_interval": EVAL_EVERY,
                   "note": "a NEW fixed budget of about 30 full-set epochs for a quick diagnostic. "
                           "This is not the earlier 3960 max-budget ablation and must not be "
                           "reported as the same budget."},
        "controls": {"shared_initial_state_hash": init_digest,
                     "shared_batch_schedule_hash": results["SignTCN"]["schedule_hash"],
                     "schedules_identical": True,
                     "dropout_rng": "torch.manual_seed(SEED + step) before every forward, so the "
                                    "dropout masks at a given step match across the two models",
                     "only_difference": "nn.BatchNorm1d versus MaskedBatchNorm1d"},
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()},
        "data": {"manifest": "expand/manifest.json",
                 "manifest_sha256": sha_file(manifest_path),
                 "classes": len(classes), "class_order": classes,
                 "train_clips": len(train), "validation_clips": len(val),
                 "reserved_test_loaded": False, "reserved_test_clips_on_disk": len(held),
                 "inputs_list": "runs/masked_bn_pair_v1/inputs.json",
                 "inputs_digest": inputs_digest,
                 "known_design_exposure": "evidence/exposure_record.json records that 30 of the "
                                          "700 reserved_test clips cover source intervals seen in "
                                          "earlier feature exploration; that record is unchanged"},
        "code_hashes": {p: sha_file(os.path.join(ROOT, p)) for p in
                        ("masked_bn_pair.py", "src/pilot/model.py", "src/pilot/model_masked_v1.py",
                         "src/pilot/data.py")},
        "per_model": report,
        "comparison": comparison,
        "web_diagnostic_status": {
            "clips": len(web), "words": sorted({c["requested_word"] for c in web}),
            "used_for_training_or_selection": False,
            "evaluated_only_after_both_runs_finished": True,
            "label": "requested_word_id is the exercise prompt. Agreement with it is not a verdict "
                     "that the attempt was performed correctly, not a recognition accuracy, and "
                     "not a new-user performance figure. The session predates attempt_label.",
        },
        "record_correction_from_task02": {
            "what": "tests/test_model_masked.py check 7 recorded runs/first_run/model.pt under "
                    "frozen_unchanged, but only its existence was checked",
            "corrected_to": "existence confirmed; not a hash comparison",
            "actually_hash_verified_in_that_check": ["src/pilot/model.py", "src/pilot/data.py",
                                                     "src/pilot/axes.py",
                                                     "runs/expanded_v1/model.pt"],
            "rerun": False,
        },
        "not_done": ["no extra seed", "no threshold search", "no re-run of the earlier nine runs "
                     "or of DTW", "no backup, Unity build, camera or data re-extraction",
                     "no candidate promoted to the web default"],
        "limits": [
            "one seed and one budget; this pair bounds nothing about variance across seeds",
            "source validation is public sentence-cut clips from 3 signers",
            "the 40 web trials are two words, one performer, one session, correctness unverified",
            "a normalisation fix being correct does not make it better at recognition; whatever "
            "this pair shows, including a worse result, is reported as measured",
        ],
        "not_a_correctness_claim": True,
    }
    p = os.path.join(RUN, "result.json")
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    log.close()
    print(f"\n({time.perf_counter() - t0:.0f}초)  -> {os.path.relpath(p, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
