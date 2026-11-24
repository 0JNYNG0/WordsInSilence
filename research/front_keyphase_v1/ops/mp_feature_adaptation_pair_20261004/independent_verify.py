"""Independent synthetic audit; no checkpoints, prepared inputs, camera, or full experiment.

Uses its own synthetic model/data and records actual gradients BEFORE AdamW steps.
Writes only a new INDEPENDENT_SYNTHETIC*.json receipt in this new operation folder.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
TRACE = []


class BadGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, g):
        return torch.full_like(g, float("inf"))


class AuditBlock(nn.Module):
    def __init__(self, width, index):
        super().__init__()
        self.index = index
        self.conv = nn.Conv1d(width, width, 3, padding=1)
        self.norm = nn.BatchNorm1d(width)
        self.drop = nn.Dropout(0.35)
        self.bad_gradient = False

    def forward(self, h, mask):
        TRACE.append((self.index, self.training, self.norm.training, self.drop.training))
        out = (torch.tanh(self.drop(self.norm(self.conv(h * mask)))) + 0.2 * h) * mask
        return BadGrad.apply(out) if self.bad_gradient else out


class AuditModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(3, 5, 1)
        self.blocks = nn.ModuleList([AuditBlock(5, index) for index in range(4)])
        self.head = nn.Linear(5, 4)


def fixture(steps=3):
    torch.manual_seed(31337)
    m = AuditModel().train()  # Deliberately starts in train mode; copied learner must use eval.
    with torch.no_grad():
        for block in m.blocks:
            block.norm.running_mean.copy_(torch.linspace(-0.3, 0.2, 5))
            block.norm.running_var.copy_(torch.linspace(0.7, 1.8, 5))
            block.norm.num_batches_tracked.fill_(7 + block.index)
    rng = np.random.default_rng(17191)
    clips = lambda n: [rng.normal(size=((i % 5) + 1, 5)).astype(np.float32) for i in range(n)]
    d = dict(base_prefixes=clips(8), base_y=np.array([0, 1, 2, 3, 1, 2, 0, 3], np.int64),
             context_prefixes=clips(6), context_y=np.array([3, 2, 0, 1, 3, 1], np.int64),
             teacher_base_logits=rng.normal(size=(8, 4)),
             base_indices=np.array([[0, 3, 5, 7], [1, 4, 6, 2], [7, 5, 0, 3]][:steps], np.int64),
             context_indices=np.array([[0, 2, 4, 5], [1, 3, 0, 4], [2, 4, 1, 5]][:steps], np.int64))
    e = dict(extra_prefixes=clips(5), extra_y=np.array([3, 0, 1, 2, 3], np.int64),
             extra_indices=np.array([[0, 2, 4], [1, 3, 0], [4, 2, 1]][:steps], np.int64), extra_weight=0.25)
    return m, d, e


def snapshot(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def state_max(a, b):
    assert set(a) == set(b)
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)


def input_digest(data):
    result = {}
    for key, value in data.items():
        items = value if isinstance(value, list) else [value]
        result[key] = [hashlib.sha256(x.tobytes()).hexdigest() if isinstance(x, np.ndarray) else x for x in items]
    return result


def own_pad(clips):
    h = torch.zeros(len(clips), clips[0].shape[1], max(len(x) for x in clips))
    mask = torch.zeros(len(clips), 1, h.shape[-1])
    for i, x in enumerate(clips):
        h[i, :, :len(x)] = torch.tensor(x.T.copy())
        mask[i, 0, :len(x)] = 1
    return h, mask


def own_pool(h, mask):
    return (h * mask).sum(-1) / mask.sum(-1)


@contextmanager
def capture_steps(names):
    original = torch.optim.AdamW
    recorded = []

    def factory(params, *args, **kwargs):
        params = list(params)
        assert len(params) == len(names)
        opt = original(params, *args, **kwargs)
        original_step = opt.step

        def step(*step_args, **step_kwargs):
            recorded.append({"weights": {n: p.detach().clone() for n, p in zip(names, params)},
                             "grads": {n: None if p.grad is None else p.grad.detach().clone()
                                       for n, p in zip(names, params)}, "weight_decay": kwargs["weight_decay"]})
            return original_step(*step_args, **step_kwargs)
        opt.step = step
        return opt
    torch.optim.AdamW = factory
    try:
        yield recorded
    finally:
        torch.optim.AdamW = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=HERE / "INDEPENDENT_SYNTHETIC.json")
    out = ap.parse_args().output.resolve()
    assert out.parent == HERE and out.name.startswith("INDEPENDENT_SYNTHETIC") and out.suffix == ".json"
    assert not out.exists(), "receipt exists; preserve prior evidence"
    spec = importlib.util.spec_from_file_location("codex_feature_audit_runner", HERE / "run_pair.py")
    RP = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RP)
    R, E, FT = RP.load_modules()  # Source imports only; no load_models/load_inputs/preflight/train/evaluate.
    T, G = E.T, E.G
    torch.set_num_threads(2)
    hashes0 = RP.code_hashes()
    verifier_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records = []

    def check(name, fn):
        try:
            records.append({"name": name, "status": "PASS", "details": fn() or {}})
        except Exception as exc:
            records.append({"name": name, "status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc()})

    def head_clone():
        differences = []
        for extra_mode in ("present", "absent", "zero_weight"):
            m, d, e = fixture()
            e = e if extra_mode == "present" else ({} if extra_mode == "absent" else dict(e, extra_weight=0.))
            before, mode_before = snapshot(m), {n: v.training for n, v in m.named_modules()}
            digest = input_digest(dict(d, **e))
            old = T.train_arm(m, **d, **e, **T.TRAIN_KW)
            new = FT.train_arm(m, **d, **e, **T.TRAIN_KW, G=G, train_last_block=False)
            state_diff = state_max(old["state_dict"], new["state_dict"])
            np.testing.assert_allclose(old["losses"], new["losses"], rtol=0, atol=1e-6, equal_nan=True)
            assert state_diff <= 1e-6 and old["trainable_names"] == new["trainable_names"]
            assert state_max(snapshot(m), before) == 0
            assert mode_before == {n: v.training for n, v in m.named_modules()}
            assert digest == input_digest(dict(d, **e))
            differences.append({"extra": extra_mode, "state_max_abs": state_diff,
                                "loss_bitwise_equal": bool(np.array_equal(old["losses"], new["losses"], equal_nan=True))})
        return {"three_steps_each": differences, "parent_and_inputs_preserved": True}
    check("head_clone_three_steps_with_absent_and_zero_extra", head_clone)

    def gradient_and_buffers():
        m, d, e = fixture()
        before = snapshot(m)
        names = [n for n, _ in m.named_parameters() if n.startswith(("blocks.3.", "head."))]
        TRACE.clear()
        with capture_steps(names) as captured:
            result = FT.train_arm(m, **d, **e, **dict(T.TRAIN_KW, weight_decay=0.), G=G, train_last_block=True)
        assert len(captured) == 3
        assert set(result["trainable_names"]) == set(names)
        assert TRACE and all(index == 3 and not any(modes) for index, *modes in TRACE)
        grads = captured[0]["grads"]
        assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
        norms = {prefix: sum(float(g.double().pow(2).sum()) for n, g in grads.items() if n.startswith(prefix)) ** .5
                 for prefix in ("blocks.3.", "head.")}
        assert all(v > 0 for v in norms.values())
        for n, _ in m.named_parameters():
            if n not in names:
                assert torch.equal(result["state_dict"][n], before[n]), n
        for n, _ in m.named_buffers():
            assert torch.equal(result["state_dict"][n], before[n]), n
        assert state_max(snapshot(m), before) == 0 and m.training
        assert all(p.grad is None for p in m.parameters())
        # Pre-update step 2 weights contain step 1's update; derive anchor independently.
        sq = sum(float((captured[1]["weights"][n].double() - before[n].double()).pow(2).sum()) for n in names)
        expected_anchor = .5 * T.TRAIN_KW["anchor_coefficient"] * sq
        assert expected_anchor > 0
        np.testing.assert_allclose(result["losses"][1, 3], expected_anchor, rtol=1e-5, atol=1e-12)
        return {"first_step_gradient_norm": norms, "weight_decay": 0,
                "anchor_manual": expected_anchor, "anchor_reported": float(result["losses"][1, 3]),
                "frozen_parameters_buffers_parent_inputs_preserved": True,
                "work_forwards_all_eval": True}
    check("actual_preoptimizer_gradient_and_eval_buffer_freeze", gradient_and_buffers)

    def extra_gradient():
        m, d, e = fixture(1)
        names = [n for n, _ in m.named_parameters() if n.startswith(("blocks.3.", "head."))]
        grads = []
        for weight in (0., .25):
            with capture_steps(names) as captured:
                FT.train_arm(m, **d, **dict(e, extra_weight=weight), **dict(T.TRAIN_KW, weight_decay=0.),
                             G=G, train_last_block=True)
            grads.append(captured[0]["grads"])
        work = copy.deepcopy(m).eval()
        params = dict(work.named_parameters())
        ids = e["extra_indices"][0]
        h, mask = own_pad([e["extra_prefixes"][int(i)] for i in ids])
        loss = .25 * F.cross_entropy(work.head(own_pool(work.blocks[3](h, mask), mask)),
                                    torch.tensor(e["extra_y"][ids]))
        expected = torch.autograd.grad(loss, [params[n] for n in names])
        maximum = 0.
        for n, want in zip(names, expected):
            got = grads[1][n] - grads[0][n]
            maximum = max(maximum, float((got - want).abs().max()))
            torch.testing.assert_close(got, want, rtol=0, atol=1e-6)
        block_norm = sum(float(g.double().pow(2).sum()) for n, g in zip(names, expected) if n.startswith("blocks.3.")) ** .5
        assert block_norm > 0
        return {"extra_only_block_gradient_norm": block_norm,
                "observed_gradient_difference_vs_analytic_extra_max_abs": maximum, "tolerance": 1e-6}
    check("extra_CE_reaches_block_independent_of_weight_decay", extra_gradient)

    def candidate_evaluation():
        m, d, e = fixture()
        r = FT.train_arm(m, **d, **e, **T.TRAIN_KW, G=G, train_last_block=True)
        cand = RP.ref_model(m, r["state_dict"])
        # Keep HEAD identical to parent to isolate the last block's effect.
        cand.head.load_state_dict(m.head.state_dict())
        rng = np.random.default_rng(487)
        raw = [rng.normal(size=(n, 3)).astype(np.float32) for n in (1, 2, 4, 7, 9)]
        pre = G.extract_prefix(m, raw)
        got, old = RP.prefix_scores(G, cand, pre), RP.prefix_scores(G, m, pre)
        manual, manual_pool = [], []
        with torch.no_grad():
            for x in raw:
                h, mask = own_pad([x])
                h = cand.proj(h * mask) * mask
                for block in cand.blocks:
                    h = block(h, mask)
                pool = own_pool(h, mask)
                manual_pool.append(pool.numpy()[0])
                manual.append(cand.head(pool).numpy()[0])
        delta = float(np.max(np.abs(got["scores"] - old["scores"])))
        assert delta > 1e-7
        logit_diff = float(np.max(np.abs(got["logits"] - np.stack(manual))))
        pool_diff = float(np.max(np.abs(got["pooled"] - np.stack(manual_pool))))
        assert logit_diff <= 1e-5 and pool_diff <= 1e-5
        assert np.array_equal(got["logits"].argmax(1), np.stack(manual).argmax(1))
        affine = got["pooled"].astype(np.float64) @ cand.head.weight.detach().numpy().astype(np.float64).T
        affine += cand.head.bias.detach().numpy().astype(np.float64)
        assert np.array_equal(affine, got["scores"])
        return {"fixed_raw_lengths": [len(x) for x in raw], "isolated_candidate_block_score_difference": delta,
                "full_forward_vs_prefix_logit_max_abs": logit_diff, "pooled_max_abs": pool_diff,
                "argmax_equal": True, "head_identical_to_parent_for_this_test": True}
    check("actual_candidate_block_full_forward_and_affine", candidate_evaluation)

    def failures_before_step():
        cases = []
        for mode in ("nonfinite_input", "overflow_loss", "nonfinite_gradient"):
            m, d, e = fixture(1)
            if mode == "nonfinite_input":
                e["extra_prefixes"][0][0, 0] = np.nan
            elif mode == "overflow_loss":
                d["teacher_base_logits"][:, 0] = 1e300
                d["teacher_base_logits"][:, 1] = -1e300
            else:
                m.blocks[3].bad_gradient = True
            before = snapshot(m)
            names = [n for n, _ in m.named_parameters() if n.startswith(("blocks.3.", "head."))]
            with capture_steps(names) as captured:
                try:
                    FT.train_arm(m, **d, **e, **T.TRAIN_KW, G=G, train_last_block=True)
                except (ValueError, RuntimeError) as exc:
                    assert not captured, "optimizer stepped before rejection"
                    assert state_max(snapshot(m), before) == 0
                    cases.append({"case": mode, "error": str(exc), "optimizer_steps": 0, "parent_preserved": True})
                else:
                    raise AssertionError(f"accepted {mode}")
        return {"cases": cases}
    check("invalid_input_loss_gradient_rejected_before_optimizer", failures_before_step)

    hashes1 = RP.code_hashes()
    stable = hashes0 == hashes1
    failed = [r["name"] for r in records if r["status"] != "PASS"]
    result = {"status": "PASSED" if stable and not failed else "FAILED", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_hashes_before": hashes0, "source_hashes_after": hashes1, "source_hashes_stable": stable,
              "verifier_sha256": verifier_hash, "passed": len(records) - len(failed), "failed": len(failed),
              "checks": records, "synthetic_only": True, "real_model_data_camera_reads": 0,
              "full_experiment_training_runs": 0,
              "limits": ["Does not certify actual 819 prefix lineage or real six-run checkpoints.",
                         "Does not certify OOF/retention performance; these require later frozen evaluation."]}
    with out.open("x", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    print(json.dumps({"status": result["status"], "passed": result["passed"], "failed_names": failed,
                      "source_hashes_stable": stable, "output": str(out)}))
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
