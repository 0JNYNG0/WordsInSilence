"""R2 A/B: flat one-sided penalty (A) vs per-group neutral penalty (B).

Same 24 recordings, same references, same features, same sampling, same DTW. The ONLY change is
what an uncomparable feature group costs. B's numbers are derived from the 18x18 reference matrix
alone - no human trial and neither of the two fist trials touched them.

Promotion rule (02_작업순서와통과기준 R3): correct top-1 count must not drop, confirmed-error
acceptance must not rise, and at least one existing error acceptance must go away.

Run: py -3.13 tests/ab_one_sided_penalty.py
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
PENALTY = json.load(open(os.path.join(ROOT, "evidence", "R2", "penalty_derivation.json"),
                         encoding="utf-8"))
# protocol order agreed with the user
LABELS = ({0: "A_one_hand"} | {i: "B_normal" for i in range(1, 16)}
          | {16: "C_fist", 17: "C_fist"} | {18: "C_reversed", 19: "C_reversed"}
          | {20: "C_static", 21: "C_static"} | {22: "C_other_word", 23: "C_other_word"})

VARIANTS = {
    "A_flat_1.5": None,
    "B_per_group_median": PENALTY["per_group_median"],
}


def rows():
    p = os.path.join(ROOT, "runs", SESSION, "decisions.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def run_variant(refs, penalty):
    cfg = dict(CONTRACT)
    cfg["cost"] = {"one_sided_penalty": penalty}
    out = []
    for i, r in enumerate(rows()):
        obs = Observation.load(
            os.path.join(ROOT, "runs", SESSION, "trials", r["trial_id"] + ".npz"))
        res = predict(obs, refs, cfg)
        w = r["requested_word_id"]
        d = next((x["distance"] for x in res["ranking"] if x["word_id"] == w), None)
        out.append({"i": i, "label": LABELS[i], "requested": w,
                    "top1": res.get("top1_word_id"), "requested_distance": d,
                    "status": res["observation_status"]})
    return out


def reference_matrix(refs, penalty):
    n = len(refs)
    off, self_bad = [], 0
    for i in range(n):
        for j in range(n):
            d = dtw(refs[i], refs[j], one_sided_penalty=penalty)["distance"]
            if i == j:
                self_bad += int(d != 0.0)
            elif d is not None:
                off.append(d)
    return {"self_nonzero": self_bad, "median_other": float(np.median(off)),
            "min_other": float(min(off))}


def summarise(res, label):
    sel = [r for r in res if r["label"] == label]
    hit = sum(1 for r in sel if r["top1"] == r["requested"])
    ds = [r["requested_distance"] for r in sel if r["requested_distance"] is not None]
    return {"n": len(sel), "top1_hit": hit,
            "dist_min": round(min(ds), 4), "dist_med": round(float(np.median(ds)), 4),
            "dist_max": round(max(ds), 4)}


def main() -> int:
    refs, _ = load_reference_set()
    report = {}
    for name, pen in VARIANTS.items():
        res = run_variant(refs, pen)
        norm = [r for r in res if r["label"] == "B_normal"]
        errs = [r for r in res if r["label"].startswith("C_")]
        nd = [r["requested_distance"] for r in norm]
        ed = [r["requested_distance"] for r in errs]
        report[name] = {
            "per_group": {g: summarise(res, g) for g in
                          ("B_normal", "C_fist", "C_reversed", "C_static", "C_other_word")},
            "normal_top1": sum(1 for r in norm if r["top1"] == r["requested"]),
            "fist_accepted": sum(1 for r in res
                                 if r["label"] == "C_fist" and r["top1"] == r["requested"]),
            "error_accepted_total": sum(1 for r in errs if r["top1"] == r["requested"]),
            "normal_range": [round(min(nd), 4), round(max(nd), 4)],
            "error_range": [round(min(ed), 4), round(max(ed), 4)],
            "threshold_exists": bool(max(nd) < min(ed)),
            "overlap_free_margin": round(min(ed) - max(nd), 4),
            "references": reference_matrix(refs, pen),
        }

    a, b = report["A_flat_1.5"], report["B_per_group_median"]
    print(f"{'':<22} {'A flat 1.5':>14} {'B per-group':>14}")
    print(f"{'normal top1 (of 15)':<22} {a['normal_top1']:>14} {b['normal_top1']:>14}")
    print(f"{'fist accepted (of 2)':<22} {a['fist_accepted']:>14} {b['fist_accepted']:>14}")
    print(f"{'any error accepted':<22} {a['error_accepted_total']:>14} {b['error_accepted_total']:>14}")
    print(f"{'normal dist range':<22} {str(a['normal_range']):>14} {str(b['normal_range']):>14}")
    print(f"{'error dist range':<22} {str(a['error_range']):>14} {str(b['error_range']):>14}")
    print(f"{'separating threshold':<22} {str(a['threshold_exists']):>14} {str(b['threshold_exists']):>14}")
    print(f"{'ref self-distance !=0':<22} {a['references']['self_nonzero']:>14} "
          f"{b['references']['self_nonzero']:>14}")
    print(f"{'ref median other':<22} {a['references']['median_other']:>14.4f} "
          f"{b['references']['median_other']:>14.4f}")
    print()
    for g in ("B_normal", "C_fist", "C_reversed", "C_static", "C_other_word"):
        pa, pb = a["per_group"][g], b["per_group"][g]
        print(f"{g:<14} n={pa['n']}  A top1 {pa['top1_hit']}/{pa['n']} "
              f"med {pa['dist_med']:.4f}   |   B top1 {pb['top1_hit']}/{pb['n']} "
              f"med {pb['dist_med']:.4f}")

    promote = (b["normal_top1"] >= a["normal_top1"]
               and b["error_accepted_total"] <= a["error_accepted_total"]
               and b["error_accepted_total"] < a["error_accepted_total"])
    report["promotion_rule"] = {
        "correct_top1_not_reduced": b["normal_top1"] >= a["normal_top1"],
        "error_acceptance_not_increased": b["error_accepted_total"] <= a["error_accepted_total"],
        "at_least_one_error_acceptance_removed":
            b["error_accepted_total"] < a["error_accepted_total"],
        "promote_B": bool(promote),
        "note": "n=15 normal and n=8 error from one participant. This decides which variant to "
                "carry forward, NOT a performance claim and NOT a pass line.",
    }
    print(f"\nB 승격 조건 충족: {promote}  "
          f"(정상 유지 {b['normal_top1']}>={a['normal_top1']}, "
          f"오류수용 {a['error_accepted_total']}->{b['error_accepted_total']})")

    out = os.path.join(ROOT, "reports", "r2_ab_one_sided_penalty", "ablation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("->", os.path.relpath(out, ROOT).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
