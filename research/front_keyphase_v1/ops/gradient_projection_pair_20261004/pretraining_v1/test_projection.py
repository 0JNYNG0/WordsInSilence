"""Synthetic tests only (no real data, caches or checkpoints).

    python test_projection.py               # runs tests; on success writes TEST_RECEIPT.json (create-exclusive)
    python test_projection.py --no-receipt  # independent re-run without a receipt
"""
from __future__ import annotations

import argparse
import copy
import datetime
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent


def _load_local(name, filename):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / filename)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


P = _load_local("gpp_projected_train", "projected_train.py")
E = P.load_experiment()            # old code modules only; nothing is read from data/model files
T, G = E.T, E.G
torch.set_num_threads(2)
W, K = 6, 5


def t64(*xs):
    return [torch.tensor(x, dtype=torch.float64) for x in xs]


def flat(gs):
    return torch.cat([g.reshape(-1).double() for g in gs])


class Projection(unittest.TestCase):
    def run_proj(self, old, ext, enabled=True):
        oc, ec = [g.clone() for g in old], [g.clone() for g in ext]
        adj, info = P.project_extra_gradients(old, ext, enabled)
        for a, b in zip(old + ext, oc + ec):
            self.assertTrue(torch.equal(a, b), "input mutated")
        for a, e in zip(adj, ext):
            self.assertEqual(a.dtype, e.dtype)
            self.assertEqual(a.shape, e.shape)
            self.assertNotEqual(a.data_ptr(), e.data_ptr())
        return adj, info

    def test_positive_unchanged(self):
        adj, info = self.run_proj(t64([1., 2.]), t64([1., 0.]))
        self.assertTrue(torch.equal(adj[0], torch.tensor([1., 0.], dtype=torch.float64)))
        self.assertEqual(info["projected"], 0.0)
        self.assertEqual(info["removed_norm"], 0.0)
        self.assertEqual(info["dot_before"], info["dot_after"])

    def test_orthogonal_unchanged(self):
        adj, info = self.run_proj(t64([1., 0.]), t64([0., 3.]))
        self.assertTrue(torch.equal(adj[0], torch.tensor([0., 3.], dtype=torch.float64)))
        self.assertEqual(info["projected"], 0.0)

    def test_negative_projected(self):
        o, e = t64([1., 1.]), t64([-2., 0.])
        adj, info = self.run_proj(o, e)
        self.assertEqual(info["projected"], 1.0)
        self.assertTrue(torch.allclose(adj[0], torch.tensor([-1., 1.], dtype=torch.float64), atol=1e-12))
        self.assertAlmostEqual(info["dot_before"], -2.0)
        self.assertAlmostEqual(info["dot_after"], 0.0, places=12)
        self.assertAlmostEqual(info["removed_norm"], 2.0 / np.sqrt(2.0), places=12)
        self.assertAlmostEqual(info["old_norm"], np.sqrt(2.0), places=12)
        self.assertAlmostEqual(info["new_norm"], 2.0, places=12)

    def test_exact_opposite_removed(self):
        o = [torch.tensor([[0.5, -1.0], [2.0, 0.25]]), torch.tensor([1.0, -3.0])]
        e = [-3.0 * g for g in o]
        adj, info = self.run_proj(o, e)
        self.assertEqual(info["projected"], 1.0)
        self.assertLessEqual(float(flat(adj).abs().max()), 1e-6)
        self.assertLessEqual(abs(info["dot_after"]), 1e-5)

    def test_zero_old_skips(self):
        adj, info = self.run_proj(t64([0., 0.]), t64([-1., 2.]))
        self.assertEqual(info["projected"], 0.0)
        self.assertTrue(torch.equal(adj[0], torch.tensor([-1., 2.], dtype=torch.float64)))

    def test_zero_extra_skips(self):
        adj, info = self.run_proj(t64([1., -2.]), t64([0., 0.]))
        self.assertEqual(info["projected"], 0.0)
        self.assertTrue(torch.equal(adj[0], torch.zeros(2, dtype=torch.float64)))

    def test_disabled_never_projects(self):
        adj, info = self.run_proj(t64([1., 1.]), t64([-2., 0.]), enabled=False)
        self.assertEqual(info["projected"], 0.0)
        self.assertLess(info["dot_before"], 0.0)
        self.assertTrue(torch.equal(adj[0], torch.tensor([-2., 0.], dtype=torch.float64)))

    def test_global_weight_bias_not_per_tensor(self):
        # weight dot = -1 but global dot = +1: per-tensor would project the weight; global does not
        o, e = t64([[1., 0.]], [1.]), t64([[-1., 0.]], [2.])
        adj, info = self.run_proj(o, e)
        self.assertEqual(info["projected"], 0.0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(adj, e)))
        # global dot < 0 with bias dot > 0: global projection also changes the bias
        o, e = t64([[1., 0.]], [1.]), t64([[-3., 0.]], [1.])
        adj, info = self.run_proj(o, e)
        self.assertEqual(info["projected"], 1.0)
        self.assertFalse(torch.equal(adj[1], e[1]))
        self.assertAlmostEqual(float(flat(o) @ flat(adj)), 0.0, places=12)   # per-tensor would keep bias

    def test_float32_roundtrip(self):
        g = torch.Generator().manual_seed(0)
        o = [torch.randn(4, 3, generator=g), torch.randn(4, generator=g)]
        e = [-o[0] + 0.1 * torch.randn(4, 3, generator=g), torch.randn(4, generator=g)]
        adj, info = self.run_proj(o, e)
        self.assertEqual(info["projected"], 1.0)
        self.assertTrue(all(a.dtype == torch.float32 for a in adj))
        self.assertLessEqual(abs(info["dot_after"]), 1e-5 * info["old_norm"] * info["new_norm"])

    def test_nonfinite_rejected(self):
        for o, e in ((t64([np.nan, 1.]), t64([1., 1.])), (t64([1., 1.]), t64([np.inf, 1.])),
                     (t64([1., 1.]), t64([-np.inf, 1.]))):
            for enabled in (True, False):
                with self.assertRaises(RuntimeError):
                    P.project_extra_gradients(o, e, enabled)

    def test_bad_arguments(self):
        with self.assertRaises(TypeError):
            P.project_extra_gradients(t64([1.]), t64([1.]), 1)
        with self.assertRaises(ValueError):
            P.project_extra_gradients(t64([1., 2.]), t64([1.]), True)
        with self.assertRaises(ValueError):
            P.project_extra_gradients([], [], True)


# --------------------------------------------------------------------------- #
# tiny model with the proj -> blocks -> masked mean -> head contract
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv = nn.Conv1d(w, w, 3, padding=1)
        self.norm = nn.BatchNorm1d(w)

    def forward(self, h, m):
        return torch.relu(self.norm(self.conv(h * m))) * m


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(4, W, 1)
        self.blocks = nn.ModuleList([Block(W), Block(W)])
        self.head = nn.Linear(W, K)


def fixture(steps=3, seed=0):
    torch.manual_seed(seed)
    model = Tiny().eval()
    with torch.no_grad():
        for b in model.blocks:
            b.norm.running_mean.uniform_(-0.1, 0.1)
            b.norm.running_var.uniform_(0.5, 1.5)
    r = np.random.default_rng(seed)
    clip = lambda: r.normal(size=(int(r.integers(3, 8)), W)).astype(np.float32)
    base, ctx, ext = [clip() for _ in range(10)], [clip() for _ in range(6)], [clip() for _ in range(7)]
    data = dict(base_prefixes=base, base_y=r.integers(0, K, 10).astype(np.int64),
                context_prefixes=ctx, context_y=r.integers(0, K, 6).astype(np.int64),
                teacher_base_logits=r.normal(size=(10, K)),
                base_indices=r.integers(0, 10, (steps, 4)).astype(np.int64),
                context_indices=r.integers(0, 6, (steps, 4)).astype(np.int64))
    extra = dict(extra_prefixes=ext, extra_y=r.integers(0, K, 7).astype(np.int64),
                 extra_indices=r.integers(0, 7, (steps, 3)).astype(np.int64), extra_weight=0.25)
    return model, data, extra


class Training(unittest.TestCase):
    def test_split_gradient_sum_equals_total_backward(self):
        torch.manual_seed(1)
        head = nn.Linear(W, K)
        p0 = [p.detach().clone() for p in head.parameters()]
        with torch.no_grad():
            for p in head.parameters():
                p.add_(0.05 * torch.randn_like(p))
        x, xe = torch.randn(8, W), torch.randn(3, W)
        y, ye = torch.randint(0, K, (8,)), torch.randint(0, K, (3,))
        teacher = torch.randn(4, K, dtype=torch.float64)

        def losses():
            logits = head(x)
            ce = F.cross_entropy(logits, y)
            sq = sum((p - q).pow(2).sum() for p, q in zip(head.parameters(), p0))
            d = logits[:4].double() - teacher
            d = d - d.mean(1, keepdim=True)
            old = ce.double() + (0.5 * 1e-2 * sq).double() + 0.5 * d.pow(2).mean()
            return old, F.cross_entropy(head(xe), ye).double() * 0.25

        params = list(head.parameters())
        old, extra = losses()
        g_old, g_ext = torch.autograd.grad(old, params), torch.autograd.grad(extra, params)
        old, extra = losses()
        (old + extra).backward()
        for go, ge, p in zip(g_old, g_ext, params):
            self.assertLessEqual(float((go + ge - p.grad).abs().max()), 1e-6)

    def test_matched_split_matches_original_train_arm(self):
        model, data, extra = fixture(steps=3)
        kw = dict(T.TRAIN_KW)
        a = T.train_arm(model, **data, **extra, **kw)
        b = P.train_arm(model, **data, **extra, **kw, project_extra=False, G=G)
        self.assertLessEqual(float(np.abs(a["losses"] - b["losses"]).max()), 1e-6)
        for k in a["state_dict"]:
            self.assertLessEqual(float((a["state_dict"][k] - b["state_dict"][k]).abs().max()), 1e-6, k)
        d = b["diagnostics"]
        self.assertEqual(d.shape, (3, len(P.DIAG_COLUMNS)))
        self.assertTrue(np.isfinite(d).all())
        self.assertFalse(d[:, P.DIAG_COLUMNS.index("projected")].any())
        self.assertTrue(np.array_equal(d[:, 2], d[:, 3]))       # no projection -> dot unchanged

    def test_no_extra_matches_original(self):
        model, data, _ = fixture(steps=2)
        a = T.train_arm(model, **data, **T.TRAIN_KW)
        b = P.train_arm(model, **data, **T.TRAIN_KW, project_extra=True, G=G)
        self.assertLessEqual(float(np.abs(np.nan_to_num(a["losses"]) - np.nan_to_num(b["losses"])).max()), 1e-6)
        self.assertTrue(np.isnan(b["losses"][:, 1:3]).all())
        self.assertFalse(b["diagnostics"][:, P.DIAG_COLUMNS.index("projected")].any())

    def test_projected_arm_and_immutability(self):
        model, data, extra = fixture(steps=6, seed=3)
        # extra rows = base rows with shifted labels -> CE gradients conflict with the old objective
        extra = dict(extra_prefixes=[a.copy() for a in data["base_prefixes"]], extra_y=(data["base_y"] + 1) % K,
                     extra_indices=data["base_indices"][:, :3].copy(), extra_weight=2.0)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        r = P.train_arm(model, **data, **extra, **T.TRAIN_KW, project_extra=True, G=G)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        self.assertTrue(r["frozen_parameters_unchanged"] and r["buffers_unchanged"])
        self.assertEqual(r["trainable_names"], ["head.weight", "head.bias"])
        sd = r["state_dict"]
        self.assertTrue(all(torch.equal(sd[k], before[k]) for k in sd if not k.startswith("head.")))
        d = r["diagnostics"]
        c = {n: d[:, i] for i, n in enumerate(P.DIAG_COLUMNS)}
        self.assertTrue(np.isfinite(d).all())
        self.assertTrue(set(np.unique(c["projected"]).tolist()) <= {0.0, 1.0})
        self.assertTrue(np.array_equal(c["projected"] > 0, c["dot_before"] < 0))
        pj = c["projected"] > 0
        self.assertTrue(pj.any())
        self.assertTrue(np.all(np.abs(c["dot_after"][pj]) <= 1e-5 * c["old_norm"][pj] * c["new_norm"][pj] + 1e-12))
        self.assertTrue(np.all(c["removed_norm"][~pj] == 0.0))

    def test_old_dot_actual_delta_recomputed(self):
        model, data, extra = fixture(steps=1, seed=5)
        r = P.train_arm(model, **data, **extra, **T.TRAIN_KW, project_extra=True, G=G)
        # independent pre-update g_old at step 0 (anchor gradient is zero at the start point)
        head = copy.deepcopy(model.head)
        brow, crow = data["base_indices"][0], data["context_indices"][0]
        h, m = G._pad_batch([data["base_prefixes"][int(i)] for i in brow] + [data["context_prefixes"][int(j)] for j in crow])
        with torch.no_grad():
            pooled = G._masked_mean(model.blocks[-1](h, m), m)
        logits = head(pooled)
        y = torch.from_numpy(np.concatenate((data["base_y"][brow], data["context_y"][crow])))
        d = logits[:4].double() - torch.from_numpy(data["teacher_base_logits"][brow])
        d = d - d.mean(1, keepdim=True)
        old = F.cross_entropy(logits, y).double() + 0.5 * T.TRAIN_KW["output_coefficient"] * d.pow(2).mean()
        g = flat(torch.autograd.grad(old, [head.weight, head.bias]))
        delta = torch.cat([(r["state_dict"][k] - model.state_dict()[k]).reshape(-1).double()
                           for k in ("head.weight", "head.bias")])
        expect = float(g @ delta)
        self.assertAlmostEqual(r["diagnostics"][0, 0], float(g.norm()), delta=1e-6 * float(g.norm()))
        self.assertAlmostEqual(r["diagnostics"][0, -1], expect, delta=1e-6 * max(1.0, abs(expect)))

    def test_validations_preserved(self):
        model, data, extra = fixture(steps=2)
        for bad in (dict(extra_weight=float("nan")), dict(extra_weight=-1.0)):
            with self.assertRaises(ValueError):
                P.train_arm(model, **data, **dict(extra, **bad), **T.TRAIN_KW, G=G)
        with self.assertRaises(ValueError):
            P.train_arm(model, **data, **dict(extra, extra_indices=extra["extra_indices"][:1]), **T.TRAIN_KW, G=G)
        with self.assertRaises(TypeError):
            P.train_arm(model, **data, **extra, **T.TRAIN_KW, project_extra=1, G=G)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-receipt", action="store_true")
    a = ap.parse_args(argv)
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (Projection, Training)])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    if not res.wasSuccessful():
        sys.exit(1)
    if not a.no_receipt:
        with (HERE / "TEST_RECEIPT.json").open("x", encoding="utf-8") as fh:
            json.dump({"status": "PASSED", "tests_run": res.testsRun, "code_hashes": P.code_hashes(),
                       "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "torch": torch.__version__, "synthetic_only": True}, fh, indent=2)
            fh.write("\n")


if __name__ == "__main__":
    main()
