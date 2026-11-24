"""Why did a wrong hand shape score well? Split the comparison into feature groups.

Compares the three correct 감사합니다 attempts against the two fist attempts, both scored on the
same 감사합니다 reference, and prints each group's own mean distance next to its weighted share.

Run: py -3.13 tests/diag_group_residuals.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.cli import load_reference_set  # noqa: E402
from wis_lab.core import features as F  # noqa: E402
from wis_lab.core.dtw import group_residuals  # noqa: E402
from wis_lab.core.observation import Observation  # noqa: E402

SESSION = "ses_cde9d19d75f2"
CONTRACT = json.load(open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8"))
# protocol order agreed with the user: rows 5..7 correct 감사합니다, rows 17..18 fist
GROUPS_OF_TRIALS = {"correct": [4, 5, 6], "fist": [16, 17], "static": [20, 21]}
TARGET = "GAMSAHAMNIDA"


def load_rows():
    p = os.path.join(ROOT, "runs", SESSION, "decisions.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sampled(obs):
    seq = F.sequence_features(obs)
    return F.uniform_time_sample(seq, int(CONTRACT["sampling_A"]["budget"]))


def main() -> int:
    refs, _ = load_reference_set()
    ref = next(r for r in refs if r["word_id"] == TARGET)
    rows = load_rows()

    per_group = {}
    for label, idx in GROUPS_OF_TRIALS.items():
        acc = {}
        for i in idx:
            obs = Observation.load(
                os.path.join(ROOT, "runs", SESSION, "trials", rows[i]["trial_id"] + ".npz"))
            res = group_residuals(sampled(obs), ref)
            assert res["status"] == "OK", rows[i]["trial_id"]
            for g, v in res["groups"].items():
                if v["mean_distance"] is not None:
                    acc.setdefault(g, []).append(v["mean_distance"])
        per_group[label] = {g: float(np.mean(v)) for g, v in acc.items()}

    order = sorted(per_group["correct"], key=lambda g: -per_group["correct"][g])
    from wis_lab.core.features import GROUPS, TOTAL_WEIGHT
    print(f"{'group':<12} {'w':>4} {'correct':>9} {'fist':>9} {'static':>9} "
          f"{'fist-corr':>10} {'가중기여차':>11}")
    total_delta = 0.0
    for g in order:
        c = per_group["correct"].get(g)
        f_ = per_group["fist"].get(g)
        s = per_group["static"].get(g)
        if c is None or f_ is None:
            continue
        w = GROUPS[g][1]
        delta = (f_ - c) * w / TOTAL_WEIGHT
        total_delta += delta
        print(f"{g:<12} {w:>4.1f} {c:>9.4f} {f_:>9.4f} "
              f"{(f'{s:.4f}' if s is not None else '-'):>9} {f_ - c:>+10.4f} {delta:>+11.5f}")
    print(f"\n예상 총 거리차(주먹 - 정상) = {total_delta:+.5f}  "
          f"(실측 0.5216 - 0.5728 = {0.5216 - 0.5728:+.5f})")

    shape = [g for g in ("shape_L", "shape_R") if g in per_group["correct"]]
    if shape:
        c = np.mean([per_group["correct"][g] for g in shape])
        f_ = np.mean([per_group["fist"][g] for g in shape])
        print(f"\n수형(shape) 그룹만: 정상 {c:.4f} -> 주먹 {f_:.4f} ({f_ - c:+.4f}), "
              f"가중치 {GROUPS['shape_L'][1]}/{TOTAL_WEIGHT} per hand")
        verdict = ("(가) 수형 거리는 커졌는데 가중치에 묻힘" if f_ > c * 1.15 else
                   "(나) 수형 거리 자체가 거의 안 커짐 - 가중치로는 해결 불가")
        print("판정:", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
