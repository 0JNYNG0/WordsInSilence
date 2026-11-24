"""Synthetic tests only (no real data, caches, checkpoints or media).

    python test_feature.py               # runs tests; on success writes TEST_RECEIPT.json (create-exclusive)
    python test_feature.py --no-receipt  # independent re-run without a receipt
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


def _load_runner():
    name = "mfa_run_pair"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "run_pair.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


RP = _load_runner()
R, E, FT = RP.load_modules()       # module source only; nothing is read from data/model files
T, G = E.T, E.G
torch.set_num_threads(2)
W, K = 6, 5
LAST = "blocks.3."


# --------------------------------------------------------------------------- #
# tiny 4-block model with the proj -> blocks -> masked mean -> head contract
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv = nn.Conv1d(w, w, 3, padding=1)
        self.norm = nn.BatchNorm1d(w)

    def forward(self, h, m):
        return torch.relu(self.norm(self.conv(h * m))) * m


class Tiny(nn.Module):
    def __init__(self, n_blocks=4):
        super().__init__()
        self.proj = nn.Conv1d(4, W, 1)
        self.blocks = nn.ModuleList([Block(W) for _ in range(n_blocks)])
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


def first(data, extra, n=1):
    return (dict(data, base_indices=data["base_indices"][:n].copy(), context_indices=data["context_indices"][:n].copy()),
            dict(extra, extra_indices=extra["extra_indices"][:n].copy()))


def state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def maxdiff(a, b):
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)


class FeatureTrain(unittest.TestCase):
    def test_loss_columns_match_old(self):
        self.assertEqual(FT.LOSS_COLUMNS, T.LOSS_COLUMNS)

    def test_head_only_matches_old_train_arm_3_steps(self):
        for with_extra in (True, False):
            model, data, extra = fixture(steps=3)
            ex = extra if with_extra else {}
            a = T.train_arm(model, **data, **ex, **T.TRAIN_KW)
            b = FT.train_arm(model, **data, **ex, **T.TRAIN_KW, G=G)
            d = RP.loss_diff(a["losses"], b["losses"])
            self.assertTrue(d["shape_equal"] and d["nan_pattern_equal"])
            self.assertLessEqual(d["max_abs"], 1e-6)
            self.assertEqual(set(a["state_dict"]), set(b["state_dict"]))
            self.assertLessEqual(maxdiff(a["state_dict"], b["state_dict"]), 1e-6)
            self.assertEqual(a["trainable_names"], b["trainable_names"])
            self.assertEqual(a["frozen_names"], b["frozen_names"])
            self.assertFalse(b["train_last_block"])

    def test_last_block_changes_only_block3_and_head(self):
        model, data, extra = fixture(steps=3, seed=2)
        before = state(model)
        r = FT.train_arm(model, **data, **extra, **T.TRAIN_KW, train_last_block=True, G=G)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()), "source modified")
        pnames = [n for n, _ in model.named_parameters()]
        bnames = [n for n, _ in model.named_buffers()]
        self.assertEqual(sorted(r["trainable_names"]), sorted(n for n in pnames if n.startswith((LAST, "head."))))
        self.assertTrue(r["frozen_parameters_unchanged"] and r["buffers_unchanged"])
        sd = r["state_dict"]
        for k in pnames:
            if not k.startswith((LAST, "head.")):
                self.assertTrue(torch.equal(sd[k], before[k]), k)
        for k in bnames:
            self.assertTrue(torch.equal(sd[k], before[k]), k)
        self.assertTrue(any(not torch.equal(sd[k], before[k]) for k in pnames if k.startswith(LAST)))
        self.assertTrue(any(not torch.equal(sd[k], before[k]) for k in pnames if k.startswith("head.")))
        self.assertTrue(all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point()))

    def test_gradient_reaches_last_block_from_old_and_extra(self):
        # weight_decay=0: Adam moves a parameter only if its gradient is non-zero. Two steps, because a
        # single Adam step is ~lr*sign(g) and could hide a magnitude-only change from the extra term.
        kw = dict(T.TRAIN_KW, weight_decay=0.0)
        model, data, extra = fixture(steps=2, seed=4)
        p0 = state(model)
        no_extra = FT.train_arm(model, **data, **kw, train_last_block=True, G=G)["state_dict"]
        zero_w = FT.train_arm(model, **data, **dict(extra, extra_weight=0.0), **kw, train_last_block=True, G=G)["state_dict"]
        with_w = FT.train_arm(model, **data, **extra, **kw, train_last_block=True, G=G)["state_dict"]
        b3 = [k for k, _ in model.named_parameters() if k.startswith(LAST)]
        self.assertTrue(all(not torch.equal(no_extra[k], p0[k]) for k in b3), "old-batch gradient missed blocks.3")
        self.assertLessEqual(max(float((zero_w[k] - no_extra[k]).abs().max()) for k in b3), 1e-7)
        self.assertTrue(any(not torch.equal(with_w[k], zero_w[k]) for k in b3), "extra gradient missed blocks.3")
        # direct autograd: extra CE has non-zero gradient w.r.t. blocks.3
        work = copy.deepcopy(model)
        eh, em = G._pad_batch([extra["extra_prefixes"][int(i)] for i in extra["extra_indices"][0]])
        loss = F.cross_entropy(work.head(G._masked_mean(work.blocks[3](eh, em), em)),
                               torch.from_numpy(extra["extra_y"][extra["extra_indices"][0]]))
        grads = torch.autograd.grad(loss, list(work.blocks[3].parameters()))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)

    def test_anchor_includes_all_trainable(self):
        model, data, extra = fixture(steps=2, seed=6)
        p0 = state(model)
        d1, e1 = first(data, extra, 1)
        r1 = FT.train_arm(model, **d1, **e1, **T.TRAIN_KW, train_last_block=True, G=G)
        r2 = FT.train_arm(model, **data, **extra, **T.TRAIN_KW, train_last_block=True, G=G)
        sq = {k: float((r1["state_dict"][k].double() - p0[k].double()).pow(2).sum()) for k in r1["trainable_names"]}
        c = T.TRAIN_KW["anchor_coefficient"]
        want = 0.5 * c * sum(sq.values())
        head_only = 0.5 * c * sum(v for k, v in sq.items() if k.startswith("head."))
        self.assertGreater(want, 0.0)
        self.assertAlmostEqual(r2["losses"][1, 3], want, delta=1e-4 * want)     # float32 accumulation
        self.assertGreater(want - head_only, 0.1 * want)                        # blocks.3 term present
        self.assertEqual(r2["losses"][0, 3], 0.0)

    def test_first_step_old_batch_loss_unchanged(self):
        model, data, extra = fixture(steps=3, seed=8)
        a = T.train_arm(model, **data, **extra, **T.TRAIN_KW)
        h = FT.train_arm(model, **data, **extra, **T.TRAIN_KW, G=G)
        b = FT.train_arm(model, **data, **extra, **T.TRAIN_KW, train_last_block=True, G=G)
        self.assertLessEqual(float(np.abs(a["losses"][0] - b["losses"][0]).max()), 1e-6)
        self.assertLessEqual(float(np.abs(h["losses"][0] - b["losses"][0]).max()), 1e-6)
        self.assertGreater(float(np.abs(h["losses"][1:] - b["losses"][1:]).max()), 0.0)

    def test_invalid_and_nonfinite_stop(self):
        model, data, extra = fixture(steps=2)
        bad_prefix = [x.copy() for x in extra["extra_prefixes"]]
        bad_prefix[0][0, 0] = np.nan
        teacher = np.zeros((10, K))
        teacher[:, 0], teacher[:, 1] = 1e300, -1e300          # finite input, output penalty overflows to inf
        for tlb in (False, True):
            run = lambda d=data, e=extra: FT.train_arm(model, **d, **e, **T.TRAIN_KW, train_last_block=tlb, G=G)
            for bad in (dict(extra_weight=float("nan")), dict(extra_weight=-1.0), dict(extra_prefixes=bad_prefix),
                        dict(extra_indices=extra["extra_indices"][:1]), dict(extra_y=extra["extra_y"] + K)):
                with self.assertRaises(ValueError):
                    run(e=dict(extra, **bad))
            with self.assertRaises(ValueError):
                run(e={"extra_weight": 0.25})
            with self.assertRaises(RuntimeError):
                run(d=dict(data, teacher_base_logits=teacher))
        with self.assertRaises(TypeError):
            FT.train_arm(model, **data, **extra, **T.TRAIN_KW, train_last_block=1, G=G)
        with self.assertRaises(TypeError):
            FT.train_arm(model, **data, **extra, **T.TRAIN_KW, train_last_block=True)


class Evaluation(unittest.TestCase):
    def test_candidate_scoring_uses_trained_last_block(self):
        model, data, extra = fixture(steps=5, seed=7)
        r = FT.train_arm(model, **data, **extra, **T.TRAIN_KW, train_last_block=True, G=G)
        cand = RP.ref_model(model, r["state_dict"])
        prefixes = data["base_prefixes"]
        rc, rp = RP.prefix_scores(G, cand, prefixes), RP.prefix_scores(G, model, prefixes)
        self.assertGreater(RP.maxabs(rc["pooled"], rp["pooled"]), 1e-6)       # trained block changes features
        w = cand.head.weight.detach().numpy().astype(np.float64)
        b = cand.head.bias.detach().numpy().astype(np.float64)
        old_pooled_scores = rp["pooled"].astype(np.float64) @ w.T + b
        self.assertGreater(RP.maxabs(old_pooled_scores, rc["scores"]), 1e-6)  # old pooled != candidate scoring
        self.assertLessEqual(rc["logits_vs_scores"], 1e-4)
        # unpadded per-clip forward through the candidate's own last block
        manual = []
        with torch.no_grad():
            for x in prefixes:
                h, m = G._pad_batch([x])
                manual.append(cand.head(G._masked_mean(cand.blocks[3](h, m), m)).numpy()[0])
        self.assertLessEqual(RP.maxabs(np.stack(manual), rc["logits"]), 1e-5)

    def test_head_only_arm_keeps_parent_pooled(self):
        model, data, extra = fixture(steps=3, seed=9)
        r = T.train_arm(model, **data, **extra, **T.TRAIN_KW)
        h = RP.prefix_scores(G, RP.ref_model(model, r["state_dict"]), data["base_prefixes"])
        p = RP.prefix_scores(G, model, data["base_prefixes"])
        self.assertTrue(RP.same(h["pooled"], p["pooled"]))

    def test_full_forward_equals_prefix_forward(self):
        model, _, _ = fixture(seed=10)
        rng = np.random.default_rng(10)
        xs = [rng.normal(size=(int(rng.integers(3, 9)), 4)).astype(np.float32) for _ in range(5)]
        pre = G.extract_prefix(model, xs)
        got = RP.prefix_scores(G, model, pre)["logits"]
        want = []
        with torch.no_grad():
            for x in xs:
                m = torch.ones(1, 1, len(x))
                h = model.proj(torch.from_numpy(x).T.unsqueeze(0) * m) * m
                for blk in model.blocks:
                    h = blk(h, m)
                want.append(model.head(G._masked_mean(h, m)).numpy()[0])
        self.assertLessEqual(RP.maxabs(got, np.stack(want)), 1e-5)

    def test_four_block_guard(self):
        RP.check_four_blocks(G, Tiny(4))
        with self.assertRaises(RuntimeError):
            RP.check_four_blocks(G, Tiny(3))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-receipt", action="store_true")
    a = ap.parse_args(argv)
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (FeatureTrain, Evaluation)])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    if not res.wasSuccessful():
        sys.exit(1)
    if not a.no_receipt:
        with RP.RECEIPT.open("x", encoding="utf-8") as fh:
            json.dump({"status": "PASSED", "tests_run": res.testsRun, "code_hashes": RP.code_hashes(),
                       "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "torch": torch.__version__, "synthetic_only": True}, fh, indent=2)
            fh.write("\n")


if __name__ == "__main__":
    main()
