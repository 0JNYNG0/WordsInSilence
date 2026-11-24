"""Synthetic tests for mask_factorial (tiny seeded model; no data, model file, network or camera).

Run: python -m unittest test_factorial -v
"""
from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn as nn

import mask_factorial as M
import run_factorial as Runner

HERE = os.path.dirname(os.path.abspath(__file__))


# --- tiny structure mirroring the masked TCN forward (test-only, eval path) -------------
class TinyNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(c))
        self.bias = nn.Parameter(torch.randn(c))
        self.register_buffer("running_mean", torch.randn(c))
        self.register_buffer("running_var", torch.rand(c) + 0.5)
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.last_valid_count = None                       # non-persistent diagnostic attribute

    def forward(self, x, mask):
        m = mask.to(x.dtype)
        self.last_valid_count = int(m.sum().item())
        y = (x - self.running_mean.view(1, -1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1) + 1e-5)
        return (y * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)) * m


class TinyBlock(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.conv2 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.norm1, self.norm2, self.act, self.drop = TinyNorm(c), TinyNorm(c), nn.GELU(), nn.Dropout(0.1)

    def forward(self, h, mask):
        y = self.drop(self.act(self.norm1(self.conv1(h * mask), mask))) * mask
        y = self.drop(self.act(self.norm2(self.conv2(y * mask), mask))) * mask
        return (h + y) * mask


class TinyTCN(nn.Module):
    def __init__(self, cin=6, w=8, k=5):
        super().__init__()
        self.proj = nn.Conv1d(cin, w, 1)
        self.blocks = nn.ModuleList([TinyBlock(w, d) for d in (1, 2)])
        self.head = nn.Linear(w, k)

    def forward(self, x, pad_mask):
        h = x.transpose(1, 2)
        m = pad_mask.unsqueeze(1)
        h = self.proj(h * m) * m
        for blk in self.blocks:
            h = blk(h, m)
        denom = pad_mask.sum(dim=1).clamp(min=1.0).unsqueeze(1)
        pooled = (h * m).sum(dim=2) / denom
        return self.head(pooled)


def hidden(model, x, mask):
    with torch.inference_mode():
        h = x.transpose(1, 2)
        m = mask.unsqueeze(1)
        h = model.proj(h * m) * m
        for blk in model.blocks:
            h = blk(h, m)
        return h


def tiny():
    torch.manual_seed(1234)
    return TinyTCN().eval()


def restored(src, present, estimated):
    pr = np.asarray(present, dtype=np.float32)
    est = np.asarray(estimated, dtype=bool)
    src = np.asarray(src, dtype=np.int64)
    obs = pr.astype(bool) & (src >= 0)[:, None]
    return {"exact_source_index": src, "present": pr, "observed": obs, "estimated": est}


# --- support mask -------------------------------------------------------------------------
class SupportMaskTests(unittest.TestCase):
    def setUp(self):
        z, o = [0, 0, 0], [1, 1, 1]
        self.r = restored([0, -1, -1, 1, 2],
                          [z, [1, 0, 0], z, o, [0, 1, 0]],
                          [z, [1, 0, 0], z, z, z])

    def test_rule(self):
        s = M.support_mask(self.r)
        self.assertEqual(s.dtype, np.float32)
        # frame 0: retained source but absent -> kept; frame 1 partly estimated -> kept;
        # frame 2 generated and empty -> excluded
        self.assertEqual(s.tolist(), [1, 1, 0, 1, 1])
        self.assertEqual(M.excluded_indices(s), [2])

    def test_absent_exact_source_never_excluded(self):
        r = restored([0, 1, 2], [[0, 0, 0]] * 3, [[0, 0, 0]] * 3)
        self.assertEqual(M.support_mask(r).tolist(), [1, 1, 1])

    def test_no_mutation(self):
        snap = {k: v.copy() for k, v in self.r.items()}
        for v in self.r.values():
            v.setflags(write=False)
        M.support_mask(self.r)
        for k in snap:
            self.assertTrue(np.array_equal(snap[k], self.r[k]))
            self.assertEqual(snap[k].dtype, self.r[k].dtype)

    def test_rejects(self):
        bad_shape = dict(self.r, present=self.r["present"][:4])
        nonbinary = dict(self.r, present=self.r["present"] * 0.5)
        est_on_exact = dict(self.r, estimated=self.r["estimated"].copy())
        est_on_exact["estimated"][0, 0] = True
        est_on_exact["present"] = self.r["present"].copy()
        est_on_exact["present"][0, 0] = 1
        overlap = dict(self.r, observed=self.r["observed"] | self.r["estimated"])
        uncovered = dict(self.r, present=np.ones_like(self.r["present"]))
        obs_on_generated = dict(self.r, observed=self.r["observed"].copy(), estimated=np.zeros((5, 3), bool))
        obs_on_generated["observed"][1, 0] = True
        float_src = dict(self.r, exact_source_index=self.r["exact_source_index"].astype(np.float64))
        missing = {k: v for k, v in self.r.items() if k != "estimated"}
        all_unsupported = restored([-1, -1], [[0, 0, 0]] * 2, [[0, 0, 0]] * 2)
        for case in (bad_shape, nonbinary, est_on_exact, overlap, uncovered, obs_on_generated, float_src,
                     missing, all_unsupported):
            with self.assertRaises(M.MaskError):
                M.support_mask(case)


class ExcludedInputAndReuseTests(unittest.TestCase):
    def test_excluded_input(self):
        x = np.ones((4, 5), np.float32)
        x[1, :-1] = 0                                    # dt may stay non-zero
        self.assertEqual(M.check_excluded_input(x, np.array([1, 0, 1, 1], np.float32)), 1)
        with self.assertRaises(M.MaskError):
            M.check_excluded_input(x, np.array([1, 1, 0, 1], np.float32))
        with self.assertRaises(M.MaskError):
            M.check_excluded_input(x, np.array([0, 1, 1, 1], np.float32))

    def test_reuse_only_all_support(self):
        self.assertTrue(M.reuse_allowed(np.ones(4, np.float32), "a", "a"))
        self.assertFalse(M.reuse_allowed(np.array([1, 0, 1], np.float32), "a", "a"))
        self.assertFalse(M.reuse_allowed(np.ones(4, np.float32), "a", "b"))
        self.assertFalse(M.reuse_allowed(np.ones(0, np.float32), "a", "a"))


# --- forward ----------------------------------------------------------------------------
class ForwardTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny()
        g = torch.Generator().manual_seed(7)
        self.x = torch.randn(2, 9, 6, generator=g)
        self.sup = torch.ones(2, 9)
        self.sup[0, 3:6] = 0
        self.sup[0, 3:6] = 0
        self.x[0, 3:6, :] = 0
        self.ones = torch.ones(2, 9)

    def run_arm(self, arm, sup=None):
        return M.diagnostic_forward(self.model, self.x, self.sup if sup is None else sup, arm)

    def test_baseline_and_both_match_model(self):
        with torch.inference_mode():
            base = self.model(self.x, self.ones)
            masked = self.model(self.x, self.sup)
        self.assertTrue(torch.equal(self.run_arm("grid_baseline"), base))
        self.assertTrue(torch.equal(self.run_arm("blocks_and_pool"), masked))

    def test_all_one_support_all_arms_identical(self):
        outs = [self.run_arm(a, self.ones) for a in M.ARMS]
        for o in outs[1:]:
            self.assertTrue(torch.equal(o, outs[0]))

    def test_pool_only_same_features_new_denominator(self):
        h = hidden(self.model, self.x, self.ones)
        p = self.sup.unsqueeze(1)
        expect = self.model.head((h * p).sum(dim=2) / self.sup.sum(dim=1).clamp(min=1.0).unsqueeze(1))
        self.assertTrue(torch.allclose(self.run_arm("pool_only"), expect, atol=1e-6))
        # row 1 has full support: identical to baseline
        self.assertTrue(torch.equal(self.run_arm("pool_only")[1], self.run_arm("grid_baseline")[1]))

    def test_blocks_only_and_both_share_hidden_differ_in_pool(self):
        hs = hidden(self.model, self.x, self.sup)
        self.assertTrue(torch.equal(hs[0, :, 3:6], torch.zeros_like(hs[0, :, 3:6])))
        total = hs.sum(dim=2)
        e_blocks = self.model.head(total / torch.full((2, 1), 9.0))
        e_both = self.model.head(total / self.sup.sum(dim=1).unsqueeze(1))
        self.assertTrue(torch.allclose(self.run_arm("blocks_only"), e_blocks, atol=1e-6))
        self.assertTrue(torch.allclose(self.run_arm("blocks_and_pool"), e_both, atol=1e-6))
        self.assertTrue(torch.equal(self.run_arm("blocks_only")[1], self.run_arm("blocks_and_pool")[1]))

    def test_eval_required_including_child(self):
        self.model.train()
        with self.assertRaises(M.MaskError):
            self.run_arm("grid_baseline")
        self.model.eval()
        self.model.blocks[1].norm2.train()
        with self.assertRaises(M.MaskError):
            self.run_arm("grid_baseline")

    def test_state_unchanged(self):
        before = M.state_digest(self.model)
        params = [p.detach().clone() for p in self.model.parameters()]
        for a in M.ARMS:
            self.run_arm(a)
        self.assertEqual(before, M.state_digest(self.model))
        for p, q in zip(params, self.model.parameters()):
            self.assertTrue(torch.equal(p, q))
        self.assertFalse(any(m.training for m in self.model.modules()))

    def test_invalid_inputs(self):
        bad = [(self.x.double(), self.sup), (self.x, self.sup.double()), (self.x, self.sup * 0.5),
               (self.x, torch.zeros(2, 9)), (self.x, torch.ones(2, 8)), (self.x[0], self.sup[0]),
               (self.x.clone().fill_(float("nan")), self.sup), (self.x.numpy(), self.sup)]
        for x, s in bad:
            with self.assertRaises(M.MaskError):
                M.diagnostic_forward(self.model, x, s, "blocks_and_pool")
        with self.assertRaises(M.MaskError):
            M.diagnostic_forward(self.model, self.x, self.sup, "best_arm")


# --- tallies ----------------------------------------------------------------------------
def row(i, signer, true, correct):
    return {"id": f"r{i}", "signer_id": signer, "true": true,
            "arms": {a: {"pred": true if ok else "OTHER", "correct": ok, "rank": 1 if ok else 2, "margin": 0.0}
                     for a, ok in correct.items()}}


class TallyTests(unittest.TestCase):
    def setUp(self):
        A = ("prior_original", "prior_corrupted") + M.ARMS
        spec = [  # prior_original, prior_corrupted, grid, pool, blocks, both
            (1, "a", (1, 1, 0, 0, 1, 1)),
            (1, "b", (1, 1, 1, 1, 1, 0)),
            (2, "a", (1, 0, 0, 0, 0, 0)),
            (2, "b", (0, 1, 0, 1, 0, 1)),
        ]
        self.rows = [row(i, s, t, dict(zip(A, map(bool, c)))) for i, (s, t, c) in enumerate(spec)]

    def test_breakdown(self):
        b = M.breakdown(self.rows, "signer_id", M.ARMS)
        self.assertEqual(b["1"], {"n": 2, "grid_baseline": 1, "pool_only": 1, "blocks_only": 2, "blocks_and_pool": 1})
        self.assertEqual(b["2"], {"n": 2, "grid_baseline": 0, "pool_only": 1, "blocks_only": 0, "blocks_and_pool": 1})

    def test_recoveries(self):
        r = M.recoveries(self.rows)
        self.assertEqual(r["n"], 2)                       # r0, r3
        self.assertEqual((r["correct_in_pool_only"], r["correct_in_blocks_only"], r["correct_in_blocks_and_pool"]),
                         (1, 1, 2))
        self.assertEqual([x["id"] for x in r["rows"]], ["r0", "r3"])

    def test_primary_effect_and_contribution(self):
        e = M.primary_effect(self.rows)
        self.assertEqual((e["base_correct"], e["arm_correct"], e["net"]), (1, 2, 1))
        self.assertEqual(e["newly_correct"], ["r0", "r3"])
        self.assertEqual(e["newly_wrong"], ["r1"])
        self.assertEqual(e["remaining_errors"], ["r1", "r2"])
        self.assertEqual(e["per_signer"]["1"]["net"], 0)
        self.assertEqual(e["per_signer"]["2"]["net"], 1)
        self.assertTrue(M.contribution({"native": e, "mediapipe": e})["diagnostic_contribution"])
        flat = dict(e, net=0)
        c = M.contribution({"native": e, "mediapipe": flat})
        self.assertFalse(c["diagnostic_contribution"])
        self.assertEqual(c["direction_by_group"]["mediapipe"], "neutral")


class NoCameraTests(unittest.TestCase):
    def test_no_camera_imports(self):
        for f in ("mask_factorial.py", "run_factorial.py"):
            tree = ast.parse(Path(HERE, f).read_text(encoding="utf-8"))
            names = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    names |= {a.name.split(".")[0] for a in n.names}
                elif isinstance(n, ast.ImportFrom) and n.module:
                    names.add(n.module.split(".")[0])
            self.assertFalse(names & {"cv2", "mediapipe"}, f)
        self.assertNotIn("cv2", sys.modules)
        self.assertNotIn("mediapipe", sys.modules)


class CompletionGuardTests(unittest.TestCase):
    def test_fresh_ops_allowed_protected_locations_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = Path(tmp) / 'lab'
            helper, prior = lab / 'ops' / 'helper', lab / 'ops' / 'prior'
            sources = lab / 'runs' / 'native'
            paths = [helper / n for n in Runner.HELPER_FILES]
            paths += [prior / n for n in Runner.IRREGULAR_FILES]
            result, plan = prior / 'result' / 'RESULT.json', lab / 'ops' / 'new' / 'PLAN.json'
            scores = {g: str(prior / 'result' / (g + '.npz')) for g in Runner.GROUPS}
            paths += [result, plan, *map(Path, scores.values())]
            cfg = {'helper_dir': str(helper), 'irregular_dir': str(prior), 'lab_root': str(lab),
                   'output_dir': str(lab / 'ops' / 'new' / 'result'), 'prior_result': str(result),
                   'plan_file': str(plan), 'prior_scores': scores,
                   'pins': {str(p): '0' * 64 for p in paths},
                   'repro_indices': Runner.REPRO_INDICES,
                   'native_roots': [str(sources)], 'mp_roots': [str(lab / 'runs' / 'mp')]}
            Runner.precheck(cfg)
            for location in [lab, lab / 'ops', helper / 'result', prior / 'result' / 'new',
                             lab / 'src' / 'new', sources / 'new']:
                with self.assertRaises(Runner.FactorialError):
                    Runner.precheck(dict(cfg, output_dir=str(location)))

    def test_common_manual_error_cannot_pass_cache_identity(self):
        ref = np.array([1., 3., 2.])
        arms = {a: ref + 0.1 for a in M.ARMS}
        checked = Runner.identity_checks(ref, arms, ref, 1e-5)
        self.assertTrue(checked['all_arms_bit_identical'])
        self.assertFalse(checked['ok'])

    def test_existing_tolerance_and_top1_contract(self):
        ref = np.array([1., 3., 2.])
        self.assertTrue(Runner.identity_checks(ref, {a: ref + 1e-6 for a in M.ARMS}, ref, 1e-5)['ok'])
        near = np.array([1., 1. + 1e-7])
        self.assertFalse(Runner.identity_checks(near, {a: near[::-1].copy() for a in M.ARMS}, near, 1e-5)['ok'])


if __name__ == "__main__":
    unittest.main()
