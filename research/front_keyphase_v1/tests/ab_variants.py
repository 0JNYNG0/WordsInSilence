"""R2 A/B: detection-dropout separation (1) and a movement group (2), alone and together.

Same 24 recordings, same references, same features, same sampling, same DTW. Each variant
changes exactly one thing, and the combination is reported so the effect can be attributed.

Promotion rule (02_작업순서와통과기준 R3): correct top-1 must not drop, confirmed-error
acceptance must not rise, and at least one existing error acceptance must go away.

Run: py -3.13 tests/ab_variants.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.cli import load_reference_set  # noqa: E402
from wis_lab.core.dtw import dtw  # noqa: E402
from wis_lab.core.observation import Observation  # noqa: E402
from wis_lab.core.recognize import predict  # noqa: E402

SESSION = "ses_cde9d19d75f2"
CONTRACT = json.load(open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8"))
MOTION_ON = {"motion_L": 1.5, "motion_R": 1.5}
LABELS = ({0: "A_one_hand"} | {i: "B_normal" for i in range(1, 16)}
          | {16: "C_fist", 17: "C_fist"} | {18: "C_reversed", 19: "C_reversed"}
          | {20: "C_static", 21: "C_static"} | {22: "C_other", 23: "C_other"})

VARIANTS = {
    "A_baseline": {},
    "1_dropout_split": {"one_sided_mode": "systematic_only"},
    "2_motion": {"group_weights": MOTION_ON},
    "3_resting_hand": {"hand_participation": True},
    "3+2_rest_motion": {"hand_participation": True, "group_weights": MOTION_ON},
}


def rows():
    p = os.path.join(ROOT, "runs", SESSION, "decisions.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def evaluate(refs, cost_cfg):
    cfg = dict(CONTRACT)
    cfg["cost"] = cost_cfg
    out = []
    for i, r in enumerate(rows()):
        obs = Observation.load(
            os.path.join(ROOT, "runs", SESSION, "trials", r["trial_id"] + ".npz"))
        res = predict(obs, refs, cfg)
        w = r["requested_word_id"]
        out.append({
            "label": LABELS[i], "requested": w, "top1": res.get("top1_word_id"),
            "d": next((x["distance"] for x in res["ranking"] if x["word_id"] == w), None),
            "status": res["observation_status"],
        })
    return out


def ref_health(refs, cost_cfg):
    n = len(refs)
    self_bad, off = 0, []
    for i in range(n):
        for j in range(n):
            d = dtw(refs[i], refs[j], **cost_cfg)["distance"]
            if i == j:
                self_bad += int(d != 0.0)
            elif d is not None:
                off.append(d)
    return {"self_nonzero": self_bad, "min_other": round(float(min(off)), 4),
            "median_other": round(float(np.median(off)), 4)}


def main() -> int:
    refs, _ = load_reference_set()
    report, table = {}, {}
    for name, cfg in VARIANTS.items():
        res = evaluate(refs, cfg)
        norm = [r for r in res if r["label"] == "B_normal"]
        errs = [r for r in res if r["label"].startswith("C_")]
        accepted = lambda rs: sum(1 for r in rs if r["top1"] == r["requested"])  # noqa: E731
        nd = [r["d"] for r in norm if r["d"] is not None]
        ed = [r["d"] for r in errs if r["d"] is not None]
        report[name] = {
            "normal_top1": accepted(norm),
            "error_accepted": accepted(errs),
            "by_error": {k: f"{accepted([r for r in errs if r['label'] == k])}/"
                            f"{len([r for r in errs if r['label'] == k])}"
                         for k in ("C_fist", "C_reversed", "C_static", "C_other")},
            "per_word_top1": {w: f"{sum(1 for r in norm if r['requested'] == w and r['top1'] == w)}/3"
                              for w in ("INSA", "GAMSAHAMNIDA", "SARANG", "JADA", "ALDA")},
            "normal_range": [round(min(nd), 4), round(max(nd), 4)],
            "error_range": [round(min(ed), 4), round(max(ed), 4)],
            "threshold_exists": bool(max(nd) < min(ed)),
            "references": ref_health(refs, cfg),
            "config": cfg,
        }
        table[name] = report[name]

    base = table["A_baseline"]
    print(f"{'variant':<18} {'정상top1':>8} {'오류수용':>8} {'주먹':>6} {'순서':>6} {'정지':>6} "
          f"{'다른단어':>8} {'임계값':>7}")
    for name, r in table.items():
        print(f"{name:<18} {r['normal_top1']:>6}/15 {r['error_accepted']:>6}/8 "
              f"{r['by_error']['C_fist']:>6} {r['by_error']['C_reversed']:>6} "
              f"{r['by_error']['C_static']:>6} {r['by_error']['C_other']:>8} "
              f"{str(r['threshold_exists']):>7}")
    print()
    for name, r in table.items():
        print(f"{name:<18} 단어별 {r['per_word_top1']}")
    print()
    for name, r in table.items():
        print(f"{name:<18} 정상 {r['normal_range']}  오류 {r['error_range']}  "
              f"레퍼런스 self!=0 {r['references']['self_nonzero']} "
              f"최근접 {r['references']['min_other']}")

    print()
    for name, r in table.items():
        if name == "A_baseline":
            continue
        keep = r["normal_top1"] >= base["normal_top1"]
        no_worse = r["error_accepted"] <= base["error_accepted"]
        better = r["error_accepted"] < base["error_accepted"]
        r["promote"] = bool(keep and no_worse and better)
        print(f"{name:<18} 승격 {r['promote']}  (정상 {base['normal_top1']}->{r['normal_top1']}, "
              f"오류수용 {base['error_accepted']}->{r['error_accepted']})")

    report["promotion_note"] = (
        "n=15 normal and n=8 error from one participant; error labels come from the agreed "
        "protocol order. This selects which variant to carry forward, NOT a performance claim "
        "and NOT a pass line.")
    out = os.path.join(ROOT, "reports", "r2_ab_variants", "ablation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("\n->", os.path.relpath(out, ROOT).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
