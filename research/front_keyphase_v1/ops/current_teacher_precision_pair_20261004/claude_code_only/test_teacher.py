"""Synthetic tests only (no checkpoints, caches, media or user data; CPU; temp-dir synthetic npz only).
current_teacher_precision_pair_20261004: the 20 inherited tests unchanged + `Precision` (analytic bounds).

    python test_teacher.py               # runs tests; on success writes TEST_RECEIPT.json (create-exclusive)
    python test_teacher.py --no-receipt  # independent re-run without a receipt
"""
from __future__ import annotations

import argparse
import copy
import datetime
import importlib.util
import inspect
import json
import math
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent


def _load_runner():
    name = "ctp_teacher_pair"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "teacher_pair.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


TP = _load_runner()
R, E = TP.load_modules()           # module source only; nothing is read from data/model files
T, G = E.T, E.G
torch.set_num_threads(2)
W, K, NB, NC, NX, HB = 6, 5, 10, 6, 7, 4
COL = {c: i for i, c in enumerate(T.LOSS_COLUMNS)}
TOL_SHIFT = 1e-6


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


def clips(r, n):
    return [r.normal(size=(int(r.integers(3, 8)), W)).astype(np.float32) for _ in range(n)]


def fixture(steps=3, seed=0, used_base=NB):
    """cache = (base, base_y, ctx, ctx_y, teacher, base_idx, ctx_idx); d = fold extra dict."""
    torch.manual_seed(seed)
    model = Tiny().eval()
    with torch.no_grad():
        for b in model.blocks:
            b.norm.running_mean.uniform_(-0.1, 0.1)
            b.norm.running_var.uniform_(0.5, 1.5)
    r = np.random.default_rng(seed)
    cache = (clips(r, NB), r.integers(0, K, NB).astype(np.int64), clips(r, NC), r.integers(0, K, NC).astype(np.int64),
             r.normal(size=(NB, K)), r.integers(0, used_base, (steps, HB)).astype(np.int64),
             r.integers(0, NC, (steps, HB)).astype(np.int64))
    d = dict(xp=clips(r, NX), xy=r.integers(0, K, NX).astype(np.int64),
             xsch=r.integers(0, NX, (steps, 3)).astype(np.int64))
    return model, cache, d


def state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def maxdiff(a, b):
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)


def bitwise(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def run(model, cache, d, teacher, **kw):
    return TP.run_arm(T, model, cache, teacher, d, "synthetic", **kw)


def first_step_output(model, cache, teacher):
    """Independent re-statement of the frozen output term on step 0, with its head gradient."""
    base, _, ctx, _, _, bidx, cidx = cache
    work = copy.deepcopy(model).eval()
    with torch.enable_grad():
        h, m = G._pad_batch([base[int(i)] for i in bidx[0]] + [ctx[int(j)] for j in cidx[0]])
        logits = work.head(G._masked_mean(work.blocks[len(work.blocks) - 1](h, m), m))
        d = logits[:len(bidx[0])].to(torch.float64) - torch.from_numpy(np.asarray(teacher, np.float64)[bidx[0]])
        d = d - d.mean(dim=1, keepdim=True)
        out = 0.5 * T.TRAIN_KW["output_coefficient"] * d.pow(2).mean()
        grads = torch.autograd.grad(out, [work.head.weight, work.head.bias])
    return float(out), grads


# --------------------------------------------------------------------------- #
class Affine(unittest.TestCase):
    def test_matches_independent_python_sum(self):
        r = np.random.default_rng(1)
        p, w, b = r.normal(size=(7, W)).astype(np.float32), r.normal(size=(K, W)).astype(np.float32), \
            r.normal(size=K).astype(np.float32)
        got = TP.make_teacher(p, w, b)
        self.assertEqual(got.dtype, np.float64)
        self.assertEqual(got.shape, (7, K))
        want = np.array([[math.fsum(float(p[i, j]) * float(w[c, j]) for j in range(W)) + float(b[c])
                          for c in range(K)] for i in range(7)])
        self.assertLessEqual(float(np.abs(got - want).max()), 1e-12)

    def test_float64_and_inputs_untouched(self):
        r = np.random.default_rng(2)
        p, w, b = r.normal(size=(4, W)), r.normal(size=(K, W)), r.normal(size=K)
        keep = (p.copy(), w.copy(), b.copy())
        got = TP.make_teacher(p, w, b)
        self.assertTrue(np.array_equal(got, p @ w.T + b))
        for a, k in zip((p, w, b), keep):
            self.assertTrue(np.array_equal(a, k))

    def test_signature_has_no_labels_or_outputs(self):
        self.assertEqual(list(inspect.signature(TP.make_teacher).parameters), ["pooled", "weight", "bias"])

    def test_rejects_bad_inputs(self):
        r = np.random.default_rng(3)
        p, w, b = r.normal(size=(4, W)), r.normal(size=(K, W)), r.normal(size=K)
        for bad in ((p.tolist(), w, b), (torch.from_numpy(p), w, b), (p.astype(np.int64), w, b),
                    (p, w.astype(np.complex128), b)):
            with self.assertRaises(TypeError):
                TP.make_teacher(*bad)
        nan_p, inf_w, nan_b = p.copy(), w.copy(), b.copy()
        nan_p[0, 0], inf_w[1, 1], nan_b[2] = np.nan, np.inf, np.nan
        for bad in ((p[0], w, b), (p, w[0], b), (p, w, b[None]), (p[:, :-1], w, b), (p, w, b[:-1]),
                    (p[:0], w, b), (nan_p, w, b), (p, inf_w, b), (p, w, nan_b), (p, -inf_w, b)):
            with self.assertRaises(ValueError):
                TP.make_teacher(*bad)
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaises(ValueError):
                TP.make_teacher(np.full((2, W), 1e200), np.full((K, W), 1e200), b)


class TeacherInTrainArm(unittest.TestCase):
    def test_run_arm_is_the_existing_train_arm_call(self):
        model, cache, d = fixture(steps=3, seed=4)
        a = run(model, cache, d, cache[4])
        b = T.train_arm(model, *cache[:5], cache[5], cache[6], extra_prefixes=d["xp"], extra_y=d["xy"],
                        extra_indices=d["xsch"], extra_weight=TP.EXTRA_W, **T.TRAIN_KW)
        self.assertTrue(bitwise(a["state_dict"], b["state_dict"]))
        self.assertTrue(np.array_equal(a["losses"], b["losses"], equal_nan=True))
        self.assertEqual(a["extra_weight"], 0.25)

    def test_same_teacher_identical(self):
        model, cache, d = fixture(steps=4, seed=5)
        a = run(model, cache, d, cache[4])
        b = run(model, cache, d, cache[4].copy())
        self.assertTrue(bitwise(a["state_dict"], b["state_dict"]))
        self.assertTrue(np.array_equal(a["losses"], b["losses"]))

    def test_teacher_change_reaches_output_gradient_and_state(self):
        model, cache, d = fixture(steps=3, seed=6)
        t2 = cache[4] + np.random.default_rng(60).normal(size=cache[4].shape)
        a, b = run(model, cache, d, cache[4]), run(model, cache, d, t2)
        self.assertEqual(a["losses"][0, COL["old_ce"]], b["losses"][0, COL["old_ce"]])
        self.assertEqual(a["losses"][0, COL["extra_ce"]], b["losses"][0, COL["extra_ce"]])
        self.assertNotEqual(a["losses"][0, COL["weighted_output"]], b["losses"][0, COL["weighted_output"]])
        self.assertGreater(maxdiff(a["state_dict"], b["state_dict"]), 1e-7)
        oa, ga = first_step_output(model, cache, cache[4])
        ob, gb = first_step_output(model, cache, t2)
        self.assertLessEqual(abs(oa - a["losses"][0, COL["weighted_output"]]), 1e-12)
        self.assertLessEqual(abs(ob - b["losses"][0, COL["weighted_output"]]), 1e-12)
        self.assertGreater(max(float((x - y).abs().max()) for x, y in zip(ga, gb)), 1e-6)

    def test_class_common_constant_leaves_centered_loss_and_update(self):
        model, cache, d = fixture(steps=4, seed=7)
        r = np.random.default_rng(70)
        a = run(model, cache, d, cache[4])
        ga = first_step_output(model, cache, cache[4])[1]
        for shift in (np.full((NB, 1), 7.0), 3.0 + 5.0 * r.normal(size=(NB, 1))):
            ts = cache[4] + shift                                        # same constant for ALL classes of a row
            b = run(model, cache, d, ts)
            self.assertLessEqual(float(np.abs(a["losses"] - b["losses"]).max()), TOL_SHIFT)
            self.assertLessEqual(maxdiff(a["state_dict"], b["state_dict"]), TOL_SHIFT)
            gb = first_step_output(model, cache, ts)[1]
            self.assertLessEqual(max(float((x - y).abs().max()) for x, y in zip(ga, gb)), TOL_SHIFT)
        one = cache[4].copy()
        one[:, 0] += 7.0                                                 # one class only: must matter
        c = run(model, cache, d, one)
        self.assertGreater(abs(c["losses"][0, COL["weighted_output"]] - a["losses"][0, COL["weighted_output"]]), 1e-6)

    def test_teacher_rows_follow_base_schedule_only(self):
        model, cache, d = fixture(steps=4, seed=8, used_base=5)
        self.assertLess(int(cache[5].max()), 5)
        a = run(model, cache, d, cache[4])
        unused = cache[4].copy()
        unused[5:] = 1e3 * np.random.default_rng(80).normal(size=(NB - 5, K))
        b = run(model, cache, d, unused)
        self.assertTrue(bitwise(a["state_dict"], b["state_dict"]))
        self.assertTrue(np.array_equal(a["losses"], b["losses"]))
        used = cache[4].copy()
        used[int(cache[5][0, 0])] += np.arange(K, dtype=np.float64)
        c = run(model, cache, d, used)
        self.assertNotEqual(a["losses"][0, COL["weighted_output"]], c["losses"][0, COL["weighted_output"]])

    def test_output_term_excludes_context_and_extra_rows(self):
        model, cache, d = fixture(steps=2, seed=9)
        r = np.random.default_rng(90)
        a = run(model, cache, d, cache[4])
        cache2 = cache[:2] + (clips(r, NC),) + cache[3:]
        b = run(model, cache2, dict(d, xp=clips(r, NX)), cache[4])
        self.assertLessEqual(abs(a["losses"][0, COL["weighted_output"]] - b["losses"][0, COL["weighted_output"]]), 1e-6)
        self.assertNotEqual(a["losses"][0, COL["old_ce"]], b["losses"][0, COL["old_ce"]])
        self.assertNotEqual(a["losses"][0, COL["extra_ce"]], b["losses"][0, COL["extra_ce"]])

    def test_invalid_teacher_rejected(self):
        model, cache, d = fixture(steps=2, seed=10)
        nan_t, inf_t = cache[4].copy(), cache[4].copy()
        nan_t[0, 0], inf_t[1, 1] = np.nan, -np.inf
        for bad in (np.zeros((NC, K)), np.zeros((NB, K + 1)), np.zeros(NB * K), nan_t, inf_t):
            with self.assertRaises(ValueError):
                run(model, cache, d, bad)
        for bad in (cache[4].tolist(), cache[4].astype(np.int64)):
            with self.assertRaises(TypeError):
                run(model, cache, d, bad)

    def test_frozen_buffers_and_source_preserved(self):
        model, cache, d = fixture(steps=3, seed=11)
        h0, before = TP.code_hashes(), state(model)
        r = run(model, cache, d, cache[4] + 1.5 * np.random.default_rng(110).normal(size=cache[4].shape))
        self.assertTrue(bitwise(state(model), before), "source model modified")
        sd = r["state_dict"]
        self.assertEqual(sorted(r["trainable_names"]), TP.HEAD_KEYS)
        self.assertTrue(r["frozen_parameters_unchanged"] and r["buffers_unchanged"])
        for k in before:
            if k not in TP.HEAD_KEYS:
                self.assertTrue(torch.equal(sd[k], before[k]), k)
        self.assertTrue(any(not torch.equal(sd[k], before[k]) for k in TP.HEAD_KEYS))
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in sd.values() if v.is_floating_point()))
        self.assertEqual(r["updates"], 3)
        self.assertEqual(TP.code_hashes(), h0)


class EncoderChecks(unittest.TestCase):
    def test_head_only_difference_passes_and_others_refuse(self):
        model, _, _ = fixture(seed=12)
        psd = state(model)
        ref = {k: v.clone() for k, v in psd.items()}
        ref["head.bias"] += 0.5
        info = TP.encoder_checks(G, model, psd, psd, ref, n_classes=K)
        self.assertEqual(info["blocks"], 4)
        bad_buf = {k: v.clone() for k, v in ref.items()}
        bad_buf["blocks.1.norm.running_mean"] += 1e-3
        bad_par = {k: v.clone() for k, v in ref.items()}
        bad_par["blocks.0.conv.weight"][0, 0, 0] += 1e-3
        for r in (bad_buf, bad_par, psd):                                # non-head change, or identical heads
            with self.assertRaises(RuntimeError):
                TP.encoder_checks(G, model, psd, psd, r, n_classes=K)
        with self.assertRaises(RuntimeError):
            TP.encoder_checks(G, model, psd, ref, ref, n_classes=K)       # parent != b8 checkpoint
        with self.assertRaises(RuntimeError):
            TP.encoder_checks(G, model, psd, psd, ref, n_classes=K + 1)
        small = Tiny(3).eval()
        s = state(small)
        r3 = {k: v.clone() for k, v in s.items()}
        r3["head.bias"] += 0.5
        with self.assertRaises(RuntimeError):
            TP.encoder_checks(G, small, s, s, r3, n_classes=K)


def write_ts(dirpath, name, **arrays):
    p = Path(dirpath) / name
    np.savez(p, **arrays)
    return p


class TeacherScores(unittest.TestCase):
    """Synthetic npz in a temp dir mirroring the real key layout (parent/control/treatment/classes/ids)."""

    def setUp(self):
        r = np.random.default_rng(13)
        self.tmp = tempfile.TemporaryDirectory()
        self.classes = [f"c{i}" for i in range(K)]
        self.all_ids = [f"id{i:02d}" for i in range(12)]
        self.parent, self.control, self.treatment = (r.normal(size=(12, K)) for _ in range(3))
        self.sel = [0, 2, 3, 7, 11]                                     # base = order-preserving subset
        self.ids = [self.all_ids[i] for i in self.sel]
        self.cached = self.parent[self.sel].copy()

    def tearDown(self):
        self.tmp.cleanup()

    def full(self, **drop):
        a = dict(parent=self.parent, control=self.control, treatment=self.treatment,
                 classes=np.array(self.classes), ids=np.array(self.all_ids))
        for k in drop:
            a.pop(k)
        return a

    def test_parent_loaded_and_compared(self):
        ts = TP.load_teacher_scores(write_ts(self.tmp.name, "ts.npz", **self.full()))
        self.assertEqual(ts["score_key"], "parent")
        self.assertTrue(np.array_equal(ts["scores"], self.parent))
        c = TP.check_teacher_scores(ts, self.ids, self.classes, self.cached, E.maxabs)
        self.assertTrue(c["parent_compared"])
        self.assertEqual(c["parent_vs_cache_teacher_max_abs"], 0.0)
        self.assertFalse(c["identical_order"])

    def test_parent_mismatch_refuses_even_if_control_matches(self):
        ts = TP.load_teacher_scores(write_ts(self.tmp.name, "ts.npz", **self.full()))
        with self.assertRaises(RuntimeError):
            TP.check_teacher_scores(ts, self.ids, self.classes, self.control[self.sel], E.maxabs)
        bumped = self.cached.copy()
        bumped[1, 2] += 1e-3
        with self.assertRaises(RuntimeError):
            TP.check_teacher_scores(ts, self.ids, self.classes, bumped, E.maxabs)

    def test_missing_parent_refuses_no_fallback(self):
        p = write_ts(self.tmp.name, "ts.npz", **self.full(parent=None))
        with self.assertRaises(RuntimeError):
            TP.load_teacher_scores(p)
        ts = TP.load_teacher_scores(write_ts(self.tmp.name, "ok.npz", **self.full()))
        for bad in (dict(ts, scores=None), dict(ts, score_key="control")):  # never silently skipped
            with self.assertRaises(RuntimeError):
                TP.check_teacher_scores(bad, self.ids, self.classes, self.cached, E.maxabs)
        with self.assertRaises(RuntimeError):
            TP.load_teacher_scores(Path(self.tmp.name) / "ts.json")

    def test_alignment_order_classes_refuse(self):
        ts = TP.load_teacher_scores(write_ts(self.tmp.name, "ts.npz", **self.full()))
        swapped = [self.ids[1], self.ids[0]] + self.ids[2:]
        cached_sw = self.parent[[self.sel[1], self.sel[0]] + self.sel[2:]]
        for ids, classes, cached in ((swapped, self.classes, cached_sw),                # base order broken
                                     (self.ids + ["unknown"], self.classes, self.cached),  # id missing
                                     (self.ids, self.classes[::-1], self.cached),        # class order
                                     (self.ids, self.classes, self.cached[:-1])):        # cached shape
            with self.assertRaises(RuntimeError):
                TP.check_teacher_scores(ts, ids, classes, cached, E.maxabs)
        dup = TP.load_teacher_scores(write_ts(self.tmp.name, "dup.npz", **dict(
            self.full(), ids=np.array(self.all_ids[:-1] + [self.all_ids[0]]))))
        with self.assertRaises(RuntimeError):
            TP.check_teacher_scores(dup, self.ids[1:], self.classes, self.cached[1:], E.maxabs)


class OverlapAudit(unittest.TestCase):
    def setUp(self):
        self.base = [f"b{i}" for i in range(20)]
        self.meta = {"annotation_dev": [{"id": f"a{i}"} for i in range(5)],
                     "whole_dev": [{"id": f"w{i}"} for i in range(3)],
                     "public": [{"id": "b1"}, {"id": "b1", "condition": "mask"}, {"id": "b2"}, {"id": "p0"}],
                     "mp": [{"id": "b3"}, {"id": "m0"}]}
        self.new = [f"n{i}" for i in range(4)]
        self.sup = ["b4", "s0"]

    def test_training_exposed_overlap_allowed_and_reported(self):
        o = TP.id_overlap_audit(self.base, self.meta, self.new, self.sup)
        self.assertEqual(o["heldout_must_be_zero"], {"annotation_dev": 0, "whole_dev": 0, "new36": 0})
        pub, mp = o["training_exposed_regression_reference"]["public"], o["training_exposed_regression_reference"]["mp"]
        self.assertEqual((pub["rows"], pub["unique_ids"], pub["overlap_ids"], pub["overlapping_ids"]),
                         (4, 3, 2, ["b1", "b2"]))
        self.assertEqual((mp["overlap_ids"], mp["overlapping_ids"]), (1, ["b3"]))
        self.assertEqual(o["support113_informational"], 1)
        self.assertIn("not an independent test", o["training_exposed_note"])

    def test_heldout_overlap_rejects(self):
        for g in TP.HELDOUT_GROUPS:
            meta = dict(self.meta, **{g: self.meta[g] + [{"id": "b9"}]})
            with self.assertRaises(RuntimeError):
                TP.id_overlap_audit(self.base, meta, self.new, self.sup)
        with self.assertRaises(RuntimeError):
            TP.id_overlap_audit(self.base, self.meta, self.new + ["b5"], self.sup)

    def test_roles_cover_groups_and_missing_group_refuses(self):
        self.assertEqual(set(TP.HELDOUT_GROUPS) | set(TP.TRAINING_EXPOSED_GROUPS), set(TP.GROUP_N))
        self.assertFalse(set(TP.HELDOUT_GROUPS) & set(TP.TRAINING_EXPOSED_GROUPS))
        meta = {k: v for k, v in self.meta.items() if k != "whole_dev"}
        with self.assertRaises(RuntimeError):
            TP.id_overlap_audit(self.base, meta, self.new, self.sup)


def prec_fixture(seed, n=70, c=11, d=64, old_noise=1e-6):
    """float32 head on float32 features; logits through the same 32-row batches as predict_prefix."""
    r = np.random.default_rng(seed)
    p = r.normal(size=(n, d)).astype(np.float32)
    head = nn.Linear(d, c)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(r.normal(scale=0.3, size=(c, d)).astype(np.float32)))
        head.bias.copy_(torch.from_numpy(r.normal(size=c).astype(np.float32)))
    w, b = head.weight.detach().numpy().copy(), head.bias.detach().numpy().copy()
    old = (p + r.normal(scale=old_noise, size=p.shape)).astype(np.float32) if old_noise else p.copy()
    return head, old, p, w, b, TP.head32_batched(head, p)


class Precision(unittest.TestCase):
    NAMES = ("numpy64_vs_torch64_old_x", "prefix32_vs_numpy64_prefix_x", "prefix32_vs_old_x_float64_teacher")

    def test_constants_exact_rational_rounded_up(self):
        for d in (W, 64):
            c = TP.precision_constants(d)
            k = 2 * d + 1
            self.assertEqual(c["k"], k)
            for key, eps in (("gamma32", np.finfo(np.float32).eps), ("gamma64", np.finfo(np.float64).eps)):
                u = Fraction(float(eps)) / 2
                exact = k * u / (1 - k * u)
                self.assertGreaterEqual(Fraction(c[key]), exact)
                self.assertLess(Fraction(float(np.nextafter(c[key], -np.inf))), exact)
            g64 = k * (Fraction(float(np.finfo(np.float64).eps)) / 2)
            g64 = g64 / (1 - g64)
            self.assertGreaterEqual(Fraction(c["inv_one_minus_gamma64"]), 1 / (1 - g64))
            self.assertGreaterEqual(Fraction(c["k_tiny32"]), k * Fraction(float(np.finfo(np.float32).tiny)))
        with self.assertRaises(ValueError):
            TP._gamma_exact(2 ** 24, np.finfo(np.float32).eps)          # k*u == 1

    def test_usual_float32_float64_rounding_accepted(self):
        for noise in (0.0, 1e-6, 1e-3):
            _, old, p, w, b, lg = prec_fixture(20, old_noise=noise)
            rec, arrays = TP.precision_checks(old, p, w, b, lg)
            self.assertTrue(rec["passed"], rec)
            self.assertTrue(rec["teacher_bitwise_equal_numpy64_affine"])
            for n in self.NAMES:
                self.assertEqual(rec["checks"][n]["violations"], 0)
                self.assertLess(rec["checks"][n]["max_diff_over_bound"], 1.0)
                self.assertEqual(arrays[f"precision__{n}__bound"].shape, (70, 11))
            self.assertGreater(rec["checks"]["prefix32_vs_numpy64_prefix_x"]["max_abs_diff"], 0.0)  # real fp32 rounding
            if noise == 0.0:
                # D = nextafter(0 * inv_one_minus_gamma64, +inf): directed-up zero, smallest positive subnormal
                self.assertEqual(rec["feature_term"]["max"], float(np.nextafter(0.0, np.inf)))
        _, old, p, w, b, lg = prec_fixture(21)
        rec, _ = TP.precision_checks(old.astype(np.float64), p, w, b, lg, teacher=TP.make_teacher(old, w, b))
        self.assertTrue(rec["passed"])                                   # float64-typed float32 values accepted

    def test_intentional_bad_teacher_cannot_pass(self):
        _, old, p, w, b, lg = prec_fixture(22)
        t = TP.make_teacher(old, w, b)
        for bump in (1e-9, 1e-2):
            bad = t.copy()
            bad[3, 4] += bump
            rec, _ = TP.precision_checks(old, p, w, b, lg, teacher=bad)
            self.assertFalse(rec["passed"])
            self.assertFalse(rec["teacher_bitwise_equal_numpy64_affine"])
            self.assertGreaterEqual(rec["checks"]["numpy64_vs_torch64_old_x"]["violations"], 1)
        rec, _ = TP.precision_checks(old, p, w, b, lg, teacher=bad)
        self.assertGreaterEqual(rec["checks"]["prefix32_vs_old_x_float64_teacher"]["violations"], 1)
        other = TP.make_teacher(old, w, (b + 0.5).astype(np.float32))   # another head's teacher
        self.assertFalse(TP.precision_checks(old, p, w, b, lg, teacher=other)[0]["passed"])
        far = (old + 0.05).astype(np.float32)                            # teacher from different features
        self.assertFalse(TP.precision_checks(old, p, w, b, lg, teacher=TP.make_teacher(far, w, b))[0]["passed"])

    def test_large_logit_perturbation_and_misleading_classes_rejected(self):
        _, old, p, w, b, lg = prec_fixture(23)
        bumped = lg.copy()
        bumped[5, 2] += np.float32(1e-2)
        swapped = lg[:, [1, 0] + list(range(2, lg.shape[1]))].copy()
        for bad_lg, bad_w, bad_b in ((bumped, w, b), (swapped, w, b), (lg, w[::-1].copy(), b[::-1].copy())):
            rec, _ = TP.precision_checks(old, p, bad_w, bad_b, bad_lg)
            self.assertFalse(rec["passed"])
            self.assertGreaterEqual(rec["checks"]["prefix32_vs_numpy64_prefix_x"]["violations"], 1)
        t = TP.make_teacher(old, w, b)
        for args in ((old, p[:-1], w, b, lg), (old, p[:, :-1], w, b, lg), (old, p, w[:, :-1], b, lg),
                     (old, p, w, b[:-1], lg), (old, p, w, b, lg[:, :-1]), (old, p, w, b, lg[:-1]),
                     (old[0], p, w, b, lg), (old[:0], p[:0], w, b, lg[:0])):
            with self.assertRaises(ValueError):
                TP.precision_checks(*args)
        with self.assertRaises(ValueError):
            TP.precision_checks(old, p, w, b, lg, teacher=t[:-1])
        for args, kw in (((old, p, w, b, lg.astype(np.float64)), {}), ((old.tolist(), p, w, b, lg), {}),
                         ((old.astype(np.int64), p, w, b, lg), {}), ((old, p, w, b, lg), {"teacher": t.astype(np.float32)}),
                         ((old, p, w.astype(np.float16), b, lg), {})):
            with self.assertRaises(TypeError):
                TP.precision_checks(*args, **kw)

    def test_nonfinite_nonrepresentable_subnormal_overflow_rejected(self):
        _, old, p, w, b, lg = prec_fixture(24)
        cases = []
        for arr_i, val in ((0, np.nan), (1, np.inf), (2, -np.inf), (3, np.nan), (4, np.inf)):
            args = [old.copy(), p.copy(), w.copy(), b.copy(), lg.copy()]
            args[arr_i].flat[0] = val
            cases.append(args)
        nonrep = old.astype(np.float64)
        nonrep[0, 0] = 0.1                                               # not a float32 value
        big64 = old.astype(np.float64)
        big64[0, 0] = 1e300                                              # float32 cast overflow
        sub = p.copy()
        sub[1, 1] = np.float32(1e-40)                                    # float32 subnormal
        cases += [[nonrep, p, w, b, lg], [big64, p, w, b, lg], [old, sub, w, b, lg]]
        huge = np.full((4, 64), 3e38, np.float32)                        # fp32 abs sum overflows float32
        cases.append([huge, huge, np.ones((3, 64), np.float32), np.zeros(3, np.float32), np.zeros((4, 3), np.float32)])
        for args in cases:
            with np.errstate(over="ignore", invalid="ignore"):
                with self.assertRaises(ValueError):
                    TP.precision_checks(*args)

    def test_near_zero_and_underflow_accepted_by_tiny_term(self):
        r = np.random.default_rng(25)
        tiny = np.finfo(np.float32).tiny
        p = (2 * tiny * np.sign(r.normal(size=(40, 64)))).astype(np.float32)
        p[:, ::7] = 0.0                                                  # exact zeros allowed
        head = nn.Linear(64, 5)
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(r.normal(scale=1e-3, size=(5, 64)).astype(np.float32)))
            head.bias.zero_()
        w, b = head.weight.detach().numpy().copy(), head.bias.detach().numpy().copy()
        lg = TP.head32_batched(head, p)                                  # products underflow in float32
        rec, _ = TP.precision_checks(p, p, w, b, lg)
        self.assertTrue(rec["passed"], rec)
        self.assertGreaterEqual(rec["checks"]["prefix32_vs_numpy64_prefix_x"]["bound_min"], rec["constants"]["k_tiny32"])

    def test_head32_same_batches_reproduce_predict_prefix(self):
        model, _, _ = fixture(seed=26)
        pre = clips(np.random.default_rng(26), 70)                        # batches 32, 32, 6 (last partial)
        out = G.predict_prefix(model, pre)
        self.assertEqual(TP.PREDICT_BATCH, 32)
        self.assertTrue(TP.same(TP.head32_batched(model.head, out["pooled"]), out["logits"]))
        moved = copy.deepcopy(model.head)
        with torch.no_grad():
            moved.bias[0] += 1e-3
        self.assertFalse(TP.same(TP.head32_batched(moved, out["pooled"]), out["logits"]))
        w, b = TP.head_np(model.state_dict())
        rec, _ = TP.precision_checks(out["pooled"], out["pooled"], w, b, out["logits"])
        self.assertTrue(rec["passed"], rec)
        with self.assertRaises(ValueError):
            TP.head32_batched(model.head, out["pooled"].astype(np.float64))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-receipt", action="store_true")
    a = ap.parse_args(argv)
    h0 = TP.code_hashes()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c)
                                for c in (Affine, TeacherInTrainArm, EncoderChecks, TeacherScores, OverlapAudit,
                                          Precision)])
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    if not res.wasSuccessful():
        sys.exit(1)
    if TP.code_hashes() != h0:
        print("source changed during tests; no receipt", file=sys.stderr)
        sys.exit(1)
    if not a.no_receipt:
        with TP.RECEIPT.open("x", encoding="utf-8") as fh:
            json.dump({"status": "PASSED", "tests_run": res.testsRun, "code_hashes": h0,
                       "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "torch": torch.__version__, "numpy": np.__version__, "synthetic_only": True,
                       "tolerances": {"class_common_constant": TOL_SHIFT},
                       "precision_bounds": "analytic gamma(2d+1,u) bounds, no empirical tolerance",
                       "unit": str(TP.UNIT), "run_dir": str(TP.RUN)}, fh, indent=2)
            fh.write("\n")


if __name__ == "__main__":
    main()
