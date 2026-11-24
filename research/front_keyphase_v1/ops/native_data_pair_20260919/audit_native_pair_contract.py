"""Independent synthetic check of the proposed existing-helper integration.

No project dataset, checkpoint, recording, network, or model registration is used.
This is verification code, not the experiment runner or a replacement trainer.
"""
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

OPS = Path(__file__).resolve().parent
WEB = OPS.parents[1]
sys.path.insert(0, str(WEB))
import tools_prep_union93_candidate as u
from pilot import data

spec = importlib.util.spec_from_file_location("unapplied_pair", OPS / "claude_v2_reviewed_not_ready.py")
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(280, 4)
        self.head = torch.nn.Linear(4, 93)

    def forward(self, x, pad):
        pooled = (x * pad.unsqueeze(-1)).sum(1) / pad.sum(1).clamp_min(1).unsqueeze(-1)
        return self.head(torch.tanh(self.encoder(pooled)))


def main():
    torch.set_num_threads(1)
    classes = [f"synthetic_{i:02d}" for i in range(93)]
    changed_class = 86

    def item(k, n):
        marker = 100 * k + n
        x = np.zeros((2 + n % 2, 280), dtype=np.float32)
        x[:, 0] = marker / 10000.0
        x[:, 1] = 0.1
        return {"sample_id": f"fixture_{k:02d}_{n:02d}", "class_index": k,
                "label": classes[k], "x": x}

    control = [item(k, n) for k in range(93) for n in range(1 + (k % 3 if k != changed_class else 0))]
    treatment = control + [item(changed_class, n) for n in range(1, 10)]
    steps, batch = 6, 32
    uniforms = []
    for j in range(steps * batch):
        uniforms += [((j % 93) + 0.5) / 93, ((j * 17) % 97 + 0.5) / 97]
    setting = {"optimizer_steps": steps, "batch": {"total": batch, "source": batch, "nikl": 0},
               "lr": {"encoder": 1e-4, "head": 1e-3}, "weight_decay": 1e-3}
    u.set_seeds(22)
    initial = copy.deepcopy(SyntheticModel().state_dict())
    records, outcomes, checks = {}, {}, []
    original_collate = data.collate

    def forbid(*args, **kwargs):
        raise AssertionError("forbidden augmentation or dataset/checkpoint load")

    with patch.object(u, "augment_nikl", side_effect=forbid), \
         patch.object(torch, "load", side_effect=forbid), \
         patch.object(np, "load", side_effect=forbid):
        for name, rows in (("control", control), ("treatment", treatment)):
            u.set_seeds(22)
            model = SyntheticModel()
            model.load_state_dict(initial)
            records[name] = []

            def record_collate(clips):
                records[name].extend((c["label"], round(float(c["x"][0, 0]) * 10000)) for c in clips)
                return original_collate(clips)

            schedule = pair.UniformSchedule(uniforms)
            with patch.object(data, "collate", side_effect=record_collate):
                stat = u.run_training(model, sorted(rows, key=lambda r: r["sample_id"]), [],
                                      classes, setting, schedule, log=lambda _: None)
            outcomes[name] = {"steps": stat["steps"], "draws": schedule.consumed,
                              "augmentation_examples": stat["augmentation_examples"],
                              "distinct_nikl_clips_seen": stat["distinct_nikl_clips_seen"],
                              "loss_finite": all(math.isfinite(x) for x in stat["losses"]),
                              "weights_updated": any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())}

        a, b = records["control"], records["treatment"]
        checks.append({"name": "class schedule identical with all 93 classes exercised",
                       "passed": [r[0] for r in a] == [r[0] for r in b] and len({r[0] for r in a}) == 93})
        checks.append({"name": "unaffected-class clip choices identical",
                       "passed": all(x == y for x, y in zip(a, b) if x[0] != changed_class)})
        changed = sum(x != y for x, y in zip(a, b) if x[0] == changed_class)
        checks.append({"name": "added clips can affect their own class sampling", "passed": changed > 0,
                       "different_choices": changed})
        checks.append({"name": "both arms consume the full fixed uniform stream",
                       "passed": all(v["draws"] == len(uniforms) for v in outcomes.values())})
        checks.append({"name": "existing trainer works with original inputs and no augmentation",
                       "passed": all(v["steps"] == steps and not v["augmentation_examples"] and
                                     v["distinct_nikl_clips_seen"] == 0 and v["loss_finite"] and
                                     v["weights_updated"] for v in outcomes.values())})

        class FixedLogits(torch.nn.Module):
            def forward(self, x, pad):
                table = torch.tensor([[3., 0., 0.], [0., 1., 2.], [0., 0., 3.]])
                return table[x[:, 0, 0].long()]

        ev_classes = ["fixture_A", "fixture_B", "fixture_C"]
        ev_rows = []
        for i, label in enumerate(ev_classes):
            x = np.zeros((2, 280), dtype=np.float32)
            x[:, 0] = i
            ev_rows.append({"sample_id": f"evaluation_{i}", "label": label, "x": x})
        selected = ev_classes[:2]
        unfiltered = u.evaluate(FixedLogits(), ev_rows, ev_classes, restrict_to=selected)
        filtered = u.evaluate(FixedLogits(), [r for r in ev_rows if r["label"] in selected],
                              ev_classes, restrict_to=selected)
        checks.append({"name": "restricted logits alone do not filter out-of-target ground truth",
                       "passed": unfiltered["n"] == 3 and unfiltered["macro_f1_denominator"] == 3 and
                                 filtered["n"] == 2 and filtered["macro_f1_denominator"] == 2,
                       "unfiltered_n": unfiltered["n"], "correct_target_n": filtered["n"]})

    report = {"status": "PASS_SYNTHETIC_HELPER_CONTRACT" if all(c["passed"] for c in checks) else "FAIL",
              "scope": "proposed helper integration only; full runner remains defective and unapplied",
              "real_data_loaded": 0, "project_checkpoints_loaded": 0,
              "project_training_runs": 0, "synthetic_optimizer_steps_per_arm": steps,
              "all_93_synthetic_classes_exercised": True, "outcomes": outcomes, "checks": checks}
    with (OPS / "synthetic_helper_contract.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"status": report["status"], "passed": sum(c["passed"] for c in checks),
                      "total": len(checks), "project_training_runs": 0}))
    return 0 if all(c["passed"] for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
