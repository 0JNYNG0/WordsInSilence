#!/usr/bin/env python3
"""Run the observation-gap experiment exactly as designed, decision rule fixed in advance.

Design: design/observation_gap_experiment.md, recorded before this script existed.

The one thing that varies is how a hand is represented during TRAINING:
  A  as-is. OpenPose fills the resting hand, and the model learns to expect it.
  B  with probability 0.25 one hand of one sample is concealed - its coordinates zeroed and its
     validity mask cleared. The choice never looks at the word label, so this is not a per-word
     rule.

Everything else is held identical per seed: the initial state_dict is built once and copied into
both models, the minibatch schedule is precomputed and shared, and torch's RNG is reseeded to
SEED+step before every forward so the dropout masks match. Concealment draws from its own
generator, so enabling it does not shift the dropout stream.

Measurement, chosen because the 40 web clips moved by 13 within 66 steps of one model in TASK03
and cannot carry a point estimate:
  primary    hand-concealment robustness on the 696 source validation clips - the fraction of
             predictions unchanged when a hand is concealed at test time
  secondary  participant web agreement, reported as a distribution over 30 checkpoints x 3 seeds
  tertiary   ordinary source validation accuracy and macro F1, to show what B costs

Nothing here judges whether any attempt was performed correctly, no candidate is promoted to the
web, and reserved_test is never loaded.

Run: py -3.13 observation_gap_run.py
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
from pilot.model_masked_v1 import MaskedSignTCN  # noqa: E402
from pilot.web_input import load_web_clip  # noqa: E402

OUT = os.path.join(ROOT, "expand")
RUN = os.path.join(ROOT, "runs", "observation_gap_v1")
SESSION = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs",
                                        "ses_cbd462050a95"))
PROJECT_5 = ("알다", "감사합니다", "나", "좋다", "모르다")
WEB_TO_LABEL = {"ALDA": "알다", "GAMSAHAMNIDA": "감사합니다"}
SEEDS = (20260912, 20260913, 20260914)
STEPS = 1980
EVAL_EVERY = 66
CONCEAL_P = 0.25          # fixed in the design; not searched
THREADS = 4
CFG = {"width": 64, "dilations": (1, 2, 4, 8), "kernel_size": 3, "dropout": 0.1,
       "batch_size": 32, "learning_rate": 1e-3, "weight_decay": 1e-3, "augmentation": "none",
       "class_weights": "inverse frequency from train only", "early_stopping": "none",
       "steps": STEPS, "architecture": "MaskedSignTCN"}

# channel spans of one hand, from the 280-channel contract
HB = D.CH_BODY_XY + D.CH_HAND_XY                      # start of the validity blocks


def hand_spans(h: int):
    """(body xy, hand xy, body valid, hand valid) channel slices for hand h (0 left, 1 right)."""
    p0 = D.N_BODY + h * D.N_HAND
    return (slice(p0 * 2, (p0 + D.N_HAND) * 2),
            slice(D.CH_BODY_XY + h * D.N_HAND * 2, D.CH_BODY_XY + (h + 1) * D.N_HAND * 2),
            slice(HB + p0, HB + p0 + D.N_HAND),
            slice(HB + D.N_POINTS + h * D.N_HAND, HB + D.N_POINTS + (h + 1) * D.N_HAND))


def conceal(x: np.ndarray, hand: int) -> np.ndarray:
    """Zero one hand's coordinates and clear its validity, as if the camera never saw it."""
    out = x.copy()
    for sl in hand_spans(hand):
        out[..., sl] = 0.0
    return out


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
                 signer_id=m["signer_id"], word_label=m["word_label"])
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


def predict(model, clips, hand=None):
    model.eval()
    pred = []
    with torch.no_grad():
        for i in range(0, len(clips), 64):
            x, pad, _ = D.collate(clips[i:i + 64])
            if hand is not None:
                x = conceal(x, hand)
            pred.extend(model(torch.from_numpy(x), torch.from_numpy(pad)).argmax(1).tolist())
    return np.asarray(pred)


def macro_f1(pred, true, n_classes):
    f1s = []
    for k in range(n_classes):
        m = true == k
        if not m.any():
            continue
        tp = int((pred[m] == k).sum())
        rec, prec = tp / int(m.sum()), tp / max(int((pred == k).sum()), 1)
        f1s.append(0.0 if tp == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def web_agreement(model, web, classes):
    five = [classes.index(w) for w in PROJECT_5]
    model.eval()
    per5 = collections.defaultdict(lambda: [0, 0])
    per80 = collections.defaultdict(lambda: [0, 0])
    for c in web:
        x, pad, _ = D.collate([c])
        with torch.no_grad():
            logit = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
        p5 = PROJECT_5[int(np.argmax([logit[i] for i in five]))]
        p80 = classes[int(np.argmax(logit))]
        per5[c["requested_word"]][1] += 1
        per5[c["requested_word"]][0] += int(p5 == c["requested_word"])
        per80[c["requested_word"]][1] += 1
        per80[c["requested_word"]][0] += int(p80 == c["requested_word"])
    def pack(per):
        hit = sum(v[0] for v in per.values())
        tot = sum(v[1] for v in per.values())
        return {"value": round(hit / tot, 4), "correct": f"{hit}/{tot}",
                "per_word": {k: f"{v[0]}/{v[1]}" for k, v in per.items()}}
    return {"five": pack(per5), "eighty": pack(per80)}


def schedule(n, steps, batch_size, seed):
    rng = np.random.default_rng(seed)
    order, cursor, out = rng.permutation(n), 0, []
    for _ in range(steps):
        if cursor + batch_size > len(order):
            order, cursor = rng.permutation(n), 0
        out.append(order[cursor:cursor + batch_size].tolist())
        cursor += batch_size
    return out


def train_one(condition, seed, init_sd, train, val, val_true, web, classes, plan, weights, log):
    model = MaskedSignTCN(width=CFG["width"], dilations=CFG["dilations"],
                          kernel_size=CFG["kernel_size"], dropout=CFG["dropout"],
                          n_classes=len(classes))
    model.load_state_dict(copy.deepcopy(init_sd))
    assert state_hash(model.state_dict()) == state_hash(init_sd)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"],
                            weight_decay=CFG["weight_decay"])
    lossf = nn.CrossEntropyLoss(weight=weights)
    # concealment draws from its own generator so enabling it cannot shift the dropout stream
    creal = np.random.default_rng(seed + 777)
    history = []
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        idx = plan[step - 1]
        batch = [train[j] for j in idx]
        x, pad, y = D.collate(batch)
        if condition == "B":
            for i in range(len(batch)):
                if creal.random() < CONCEAL_P:
                    x[i] = conceal(x[i], int(creal.integers(0, 2)))
        torch.manual_seed(seed + step)
        model.train()
        opt.zero_grad()
        loss = lossf(model(torch.from_numpy(x), torch.from_numpy(pad)), torch.from_numpy(y))
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            base = predict(model, val)
            left = predict(model, val, hand=0)
            right = predict(model, val, hand=1)
            wa = web_agreement(model, web, classes)
            history.append({
                "step": step, "train_loss": round(float(loss.detach()), 5),
                "val_accuracy": round(float((base == val_true).mean()), 4),
                "val_macro_f1": round(macro_f1(base, val_true, len(classes)), 6),
                "robust_left_concealed": round(float((left == base).mean()), 4),
                "robust_right_concealed": round(float((right == base).mean()), 4),
                "web_five": wa["five"]["value"], "web_eighty": wa["eighty"]["value"],
                "web_five_per_word": wa["five"]["per_word"],
            })
            log.write(f"{condition} seed {seed} step {step:>5} "
                      f"valF1 {history[-1]['val_macro_f1']:.4f} "
                      f"robustL {history[-1]['robust_left_concealed']:.3f} "
                      f"robustR {history[-1]['robust_right_concealed']:.3f} "
                      f"web5 {history[-1]['web_five']:.3f}\n")
            log.flush()
    return {"history": history, "seconds": round(time.perf_counter() - t0, 1),
            "final_state": copy.deepcopy(model.state_dict())}


def span(values):
    v = [float(x) for x in values]
    return {"n": len(v), "min": round(min(v), 4), "median": round(float(np.median(v)), 4),
            "max": round(max(v), 4), "mean": round(float(np.mean(v)), 4)}


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
    val_true = np.asarray([c["label"] for c in val])
    web = load_web()
    print(f"클래스 {len(classes)}  학습 {len(train)}  검증 {len(val)}  웹 {len(web)}  "
          f"(reserved_test {len(held)}건 미로딩)")
    print(f"조건 A=현행 / B=확률 {CONCEAL_P}로 한 손 가림, seed {len(SEEDS)}개, 각 {STEPS} step\n")

    counts = collections.Counter(c["label"] for c in train)
    weights = torch.tensor([len(train) / (len(classes) * max(counts[k], 1))
                            for k in range(len(classes))], dtype=torch.float32)

    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        init_sd = copy.deepcopy(MaskedSignTCN(
            width=CFG["width"], dilations=CFG["dilations"], kernel_size=CFG["kernel_size"],
            dropout=CFG["dropout"], n_classes=len(classes)).state_dict())
        plan = schedule(len(train), STEPS, CFG["batch_size"], seed)
        for condition in ("A", "B"):
            r = train_one(condition, seed, init_sd, train, val, val_true, web, classes, plan,
                          weights, log)
            h = r["history"]
            runs.append({"condition": condition, "seed": seed,
                         "initial_state_hash": state_hash(init_sd),
                         "batch_schedule_hash": sha_text(json.dumps(plan)),
                         "seconds": r["seconds"], "history": h})
            torch.save({"state_dict": r["final_state"], "classes": classes,
                        "condition": condition, "seed": seed, "step": STEPS,
                        "model_type": MaskedSignTCN.model_type,
                        "model_version": MaskedSignTCN.model_version,
                        "conceal_p": CONCEAL_P if condition == "B" else 0.0,
                        "config": {k: (list(v) if isinstance(v, tuple) else v)
                                   for k, v in CFG.items()}},
                       os.path.join(RUN, f"{condition}_seed{seed}_final.pt"))
            print(f"  {condition} seed {seed}  {r['seconds']}초  "
                  f"valF1 {h[-1]['val_macro_f1']:.4f}  "
                  f"가림내성 L {h[-1]['robust_left_concealed']:.3f} "
                  f"R {h[-1]['robust_right_concealed']:.3f}  "
                  f"웹5 {h[-1]['web_five']:.3f}", flush=True)

    def pull(cond, seed, key):
        return [p[key] for r in runs if r["condition"] == cond and (seed is None or
                                                                    r["seed"] == seed)
                for p in r["history"]]

    per_seed, verdict_rows = {}, []
    for seed in SEEDS:
        row = {}
        for key in ("robust_left_concealed", "robust_right_concealed", "val_macro_f1",
                    "web_five", "web_eighty"):
            row[key] = {"A": span(pull("A", seed, key)), "B": span(pull("B", seed, key))}
        per_seed[str(seed)] = row
        verdict_rows.append(row)

    def non_overlapping_b_higher(key):
        return all(r[key]["B"]["min"] > r[key]["A"]["max"] for r in verdict_rows)

    def b_higher_median(key):
        return all(r[key]["B"]["median"] > r[key]["A"]["median"] for r in verdict_rows)

    primary = {}
    for key in ("robust_left_concealed", "robust_right_concealed"):
        primary[key] = {
            "B_beats_A_in_every_seed_by_median": b_higher_median(key),
            "ranges_non_overlapping_in_every_seed": non_overlapping_b_higher(key),
            "verdict": ("EFFECT" if non_overlapping_b_higher(key)
                        else "NO_EFFECT_DEMONSTRATED"),
        }
    f1_gap = [round(r["val_macro_f1"]["A"]["median"] - r["val_macro_f1"]["B"]["median"], 4)
              for r in verdict_rows]
    web_overlap = {}
    for key in ("web_five", "web_eighty"):
        overlap = any(not (r[key]["B"]["min"] > r[key]["A"]["max"]
                           or r[key]["A"]["min"] > r[key]["B"]["max"])
                      for r in verdict_rows)
        web_overlap[key] = {"distributions_overlap_in_some_seed": overlap,
                            "verdict": "INDISTINGUISHABLE" if overlap else "SEPARATED"}

    print("\n=== 사전에 정한 판정 규칙에 따른 결과 ===")
    for key, v in primary.items():
        print(f"  1차 {key:<26} {v['verdict']}  "
              f"(seed별 중앙값 우세 {v['B_beats_A_in_every_seed_by_median']}, "
              f"범위 분리 {v['ranges_non_overlapping_in_every_seed']})")
    print(f"  대가  val macro F1 (A중앙 - B중앙) seed별 {f1_gap}  "
          f"기준 0.02 이상이면 대가로 기록")
    for key, v in web_overlap.items():
        print(f"  2차 {key:<26} {v['verdict']}")
    print("\n  seed별 1차 지표 범위 (30개 평가시점)")
    for seed in SEEDS:
        for key in ("robust_left_concealed", "robust_right_concealed"):
            a, b = per_seed[str(seed)][key]["A"], per_seed[str(seed)][key]["B"]
            print(f"    {seed} {key:<26} A {a['min']:.3f}~{a['max']:.3f} (중앙 {a['median']:.3f})"
                  f"   B {b['min']:.3f}~{b['max']:.3f} (중앙 {b['median']:.3f})")

    out = {
        "task_id": "observation_gap_v1", "date_kst": time.strftime("%Y-%m-%d"),
        "design": "design/observation_gap_experiment.md",
        "design_sha256": sha_file(os.path.join(ROOT, "design",
                                               "observation_gap_experiment.md")),
        "command": "py -3.13 observation_gap_run.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform(),
                        "torch_threads": THREADS},
        "conditions": {"A": "as-is", "B": f"random single-hand concealment at p={CONCEAL_P}, "
                                          f"label-blind, training only"},
        "held_fixed": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()},
                       "seeds": list(SEEDS), "evaluation_interval": EVAL_EVERY,
                       "shared_per_seed": "initial state_dict and batch schedule, both hashed",
                       "dropout_rng": "torch.manual_seed(seed+step) before every forward",
                       "concealment_rng": "separate generator, so enabling B does not shift the "
                                          "dropout stream"},
        "data": {"manifest_sha256": sha_file(manifest_path), "classes": len(classes),
                 "train_clips": len(train), "validation_clips": len(val),
                 "web_clips": len(web), "reserved_test_loaded": False},
        "metric_definitions": {
            "robust_left_concealed": "fraction of the 696 source validation clips whose predicted "
                                     "word is unchanged when the left hand is concealed at test "
                                     "time; higher means less sensitive to a missing hand",
            "val_macro_f1": "ordinary 80-class macro F1 on the same 696 clips, unmodified input",
            "web_five": "agreement with the requested word on 40 participant clips, restricted to "
                        "the five project words. NOT a recognition accuracy and NOT a verdict "
                        "that any attempt was performed correctly",
        },
        "pre_registered_decision_rule": {
            "effect": "B beats A on concealment robustness in all three seeds with non-overlapping "
                      "ranges over the 30 checkpoints",
            "cost": "reported if A's median val macro F1 exceeds B's by 0.02 or more",
            "web": "if the distributions overlap in any seed, recorded as indistinguishable and "
                   "not used to claim improvement",
        },
        "primary_result": primary,
        "cost_result": {"a_minus_b_median_macro_f1_per_seed": f1_gap,
                        "threshold": 0.02,
                        "exceeds_threshold": [abs(g) >= 0.02 for g in f1_gap]},
        "secondary_result": web_overlap,
        "per_seed_spans": per_seed,
        "per_run": runs,
        "limits": [
            "this measures a remedy, not a cause; it does not establish that the encoding "
            "mismatch produced the 알다 transfer failure",
            "the 40 web clips are two words, one performer, one session, correctness unverified",
            "source validation is public sentence-cut clips from three signers",
            "concealment at test time is a counterfactual, not a real capture",
            "nothing here judges whether any attempt was performed correctly",
        ],
        "web_recogniser_unchanged": True,
        "no_candidate_promoted": True,
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
