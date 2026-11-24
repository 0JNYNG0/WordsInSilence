'''Synthetic tests for paired_finetune helpers (tiny modules; no real model, data, devices, or network).'''
import copy
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paired_finetune as P

DIM, K, T = 6, 5, 7


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, kernel_size=(3, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(4)
        self.register_buffer('A', torch.eye(27) + 0.1)
        self.edge_importance = nn.Parameter(torch.ones(27, 27))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(4, DIM)
        with torch.no_grad():
            self.bn.running_mean.uniform_(-0.5, 0.5)
            self.bn.running_var.uniform_(0.5, 2.0)
        self.eval()

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', x, self.A * self.edge_importance)
        h = torch.relu(self.bn(self.conv(x)))
        return self.fc(self.drop(h.mean(dim=(2, 3))))


def tiny(seed=0):
    torch.manual_seed(seed)
    return Tiny()


def make_head(seed=0):
    rng = np.random.default_rng(seed)
    return {'W': rng.normal(size=(K, DIM)), 'b': rng.normal(size=K), 'mean': rng.normal(size=DIM),
            'std': rng.uniform(0.5, 2.0, size=DIM), 'classes': np.array(['c%d' % i for i in range(K)])}


def make_data(n=10, seed=1):
    rng = np.random.default_rng(seed)
    arrays = [rng.normal(size=(1, 2, T, 27)).astype(np.float32) for _ in range(n)]
    return arrays, rng.integers(0, K, size=n)


def quiet(msg):
    pass


class PairedFinetuneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.arrays, self.labels = make_data()
        self.head = make_head()
        self.classes = [str(c) for c in self.head['classes']]
        self.batches = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [1, 1, 8, 9]])

    def p(self, name):
        return os.path.join(self.tmp.name, name)

    def test_trainable_on_off(self):
        m = tiny()
        keys = P.set_trainable(m, True)
        self.assertEqual(keys, sorted(n for n, _ in m.named_parameters()))
        self.assertTrue(all(p.requires_grad for p in m.parameters()))
        self.assertEqual(P.set_trainable(m, False), [])
        self.assertFalse(any(p.requires_grad for p in m.parameters()))
        self.assertFalse(any(mod.training for mod in m.modules()))

    def test_gradient_probe_candidate_vs_control(self):
        m = tiny()
        before = P.state_snapshot(m)
        c = P.gradient_probe(m, self.head, self.arrays[0], self.labels[0], True)
        self.assertGreater(c['reached_nonzero'], 0)
        self.assertIn('bn.weight', c['reached_keys'])
        self.assertIn('edge_importance', c['reached_keys'])
        self.assertTrue(c['state_unchanged'])
        h = P.gradient_probe(m, self.head, self.arrays[0], self.labels[0], False)
        self.assertEqual(h['reached_nonzero'], 0)
        self.assertTrue(h['head_grad_nonzero'])
        self.assertTrue(all(v == 0.0 for v in P.state_diff(before, m).values()))

    def test_candidate_bn_buffers_fixed_affine_trained(self):
        m = tiny()
        buf = {k: v.clone() for k, v in m.named_buffers()}
        bn_w = m.bn.weight.detach().clone()
        logs = []
        fn = lambda pos: m(torch.from_numpy(self.arrays[pos]))
        W, b, counts, s = P.train_arm('full_encoder', m, self.head, self.batches, self.labels, fn, logs.append, len(self.arrays))
        for k, v in m.named_buffers():
            self.assertTrue(torch.equal(v, buf[k]), k)
        self.assertEqual(int(m.bn.num_batches_tracked), 0)
        self.assertFalse(torch.equal(m.bn.weight.detach(), bn_w))
        self.assertIn('bn.weight', s['changed_keys'])
        self.assertIn('edge_importance', s['changed_keys'])
        self.assertTrue(s['buffers_unchanged'])
        self.assertFalse(s['control_encoder_equal'])
        self.assertEqual(len(s['per_step']), 3)
        np.testing.assert_array_equal(counts, P.exposure_counts(self.batches, len(self.arrays)))
        self.assertEqual(s['exposure_total'], 12)
        x = torch.from_numpy(self.arrays[0])
        with torch.no_grad():
            self.assertTrue(torch.equal(m(x), m(x)))  # dropout disabled in eval
        self.assertFalse(np.array_equal(W.numpy(), self.head['W']))
        self.assertTrue(logs)

    def test_control_encoder_exactly_frozen(self):
        m = tiny()
        before = P.state_snapshot(m)
        with torch.no_grad():
            feats = torch.cat([m(torch.from_numpy(a)) for a in self.arrays])
        fn = lambda pos: feats[pos:pos + 1]
        W, b, counts, s = P.train_arm('head_only', m, self.head, self.batches, self.labels, fn, quiet, len(self.arrays))
        self.assertTrue(s['control_encoder_equal'])
        self.assertEqual(s['trainable_encoder_keys'], [])
        self.assertEqual(s['changed_keys'], {})
        for k, v in m.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)
        self.assertFalse(np.array_equal(W.numpy(), self.head['W']))

    def test_accumulated_mean_ce_l2_matches_reference_batch(self):
        m = tiny()
        ref = copy.deepcopy(m)
        P.set_trainable(m, True)
        P.set_trainable(ref, True)
        rng = np.random.default_rng(3)
        arrays = [rng.normal(size=(1, 2, T, 27)).astype(np.float32) for _ in range(12)]
        labels = rng.integers(0, K, size=12)
        batch = rng.integers(0, 12, size=32)  # same length clips, duplicates allowed
        W, b, mean, std = P.head_params(self.head)
        ce, reg = P.step_gradients(batch, labels, W, b, mean, std, lambda pos: m(torch.from_numpy(arrays[pos])))
        W2, b2, mean2, std2 = P.head_params(self.head)
        x = torch.from_numpy(np.concatenate([arrays[i] for i in batch]))
        logits = P.head_logits(ref(x), W2, b2, mean2, std2)
        ce_ref = F.cross_entropy(logits, torch.from_numpy(labels[batch].astype(np.int64)))
        reg_ref = 0.5 * P.L2 * W2.square().sum()
        (ce_ref + reg_ref).backward()
        self.assertAlmostEqual(ce, float(ce_ref), places=5)
        self.assertEqual(reg, float(reg_ref))
        torch.testing.assert_close(W.grad, W2.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(b.grad, b2.grad, rtol=1e-5, atol=1e-6)
        for (n1, p1), (n2, p2) in zip(m.named_parameters(), ref.named_parameters()):
            self.assertEqual(n1, n2)
            torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-4, atol=1e-5, msg=n1)

    def test_state_and_head_roundtrip_exact(self):
        m = tiny(0)
        with torch.no_grad():
            m.bn.num_batches_tracked.fill_(7)
            m.edge_importance.mul_(1.5)
        sha = P.write_npz_x(self.p('s.npz'), P.state_arrays(m))
        self.assertEqual(sha, P.sha256_file(self.p('s.npz')))
        loaded = P.npz_from_bytes(P.read_verified(self.p('s.npz'), sha))
        m2 = tiny(1)
        P.load_state_arrays(m2, loaded)
        sd2 = m2.state_dict()
        for k, v in m.state_dict().items():
            self.assertEqual(v.dtype, sd2[k].dtype)
            self.assertTrue(torch.equal(v, sd2[k]), k)
        bad = dict(loaded)
        bad['bn.num_batches_tracked'] = bad['bn.num_batches_tracked'].astype(np.int32)
        with self.assertRaises(P.FailClosed):
            P.load_state_arrays(tiny(2), bad)
        missing = dict(loaded)
        missing.pop('fc.bias')
        with self.assertRaises(P.FailClosed):
            P.load_state_arrays(tiny(2), missing)
        hsha = P.write_npz_x(self.p('h.npz'), self.head)
        h = P.validate_head(P.npz_from_bytes(P.read_verified(self.p('h.npz'), hsha)), self.classes, K, DIM)
        for k in ('W', 'b', 'mean', 'std'):
            self.assertEqual(h[k].dtype, np.float64)
            np.testing.assert_array_equal(h[k], self.head[k])
        with self.assertRaises(P.FailClosed):
            P.read_verified(self.p('h.npz'), '0' * 64)

    def test_pin_mismatch_fails(self):
        path = self.p('pinned.bin')
        with open(path, 'wb') as f:
            f.write(b'abc')
        good = P.sha256_file(path)
        self.assertTrue(P.verify_pins({'x': {'path': path, 'sha256': good}}))
        with self.assertRaises(P.FailClosed):
            P.verify_pins({'x': {'path': path, 'sha256': '0' * 64}})
        with self.assertRaises(P.FailClosed):
            P.verify_pins({'x': {'path': self.p('missing'), 'sha256': good}})
        pins = {n: {'path': 'pinned.bin', 'sha256': good} for n in P.PIN_NAMES}
        resolved = P.resolve_pins({'pins': pins}, self.tmp.name)
        self.assertTrue(P.verify_pins(resolved))
        del pins['plan']
        with self.assertRaises(P.FailClosed):
            P.resolve_pins({'pins': pins}, self.tmp.name)

    def test_schedule_mapping_and_bounds(self):
        base_pos = np.arange(10) * 2
        ctx_pos = np.array([100, 101, 102])
        bi = np.array([[0, 9], [3, 3]])
        ci = np.array([[2, 0], [1, 1]])
        out = P.map_schedule(bi, ci, base_pos, ctx_pos, steps=2, per_group=2)
        np.testing.assert_array_equal(out, [[0, 18, 102, 100], [6, 6, 101, 101]])
        cases = ((np.array([[0, 10], [3, 3]]), ci), (bi, np.array([[2, -1], [1, 1]])), (bi[:1], ci),
                 (bi.astype(float), ci), (bi, np.array([[3, 0], [1, 1]])))
        for bad_b, bad_c in cases:
            with self.assertRaises(P.FailClosed):
                P.map_schedule(bad_b, bad_c, base_pos, ctx_pos, steps=2, per_group=2)

    def test_duplicate_exposures_counted(self):
        batches = np.array([[0, 18, 102, 100], [6, 6, 101, 101]])
        counts = P.exposure_counts(batches, 103)
        self.assertEqual(counts[6], 2)
        self.assertEqual(counts[101], 2)
        self.assertEqual(counts[0], 1)
        self.assertEqual(int(counts.sum()), 8)
        self.assertEqual(P.validate_counts(counts, counts.copy()), 8)
        wrong = counts.copy()
        wrong[6] = 1
        with self.assertRaises(P.FailClosed):
            P.validate_counts(counts, wrong)

    def test_refuses_overwrite(self):
        path = self.p('o.json')
        P.write_json_x(path, {'a': 1})
        with self.assertRaises(FileExistsError):
            P.write_json_x(path, {'a': 2})
        with self.assertRaises(FileExistsError):
            P.write_npz_x(path, {'x': np.zeros(1)})
        with self.assertRaises(P.FailClosed):
            P.ensure_absent([path])
        with open(path, 'rb') as f:
            self.assertEqual(json.loads(f.read()), {'a': 1})
        log = P.Logger(self.p('l.txt'))
        log.close()
        with self.assertRaises(FileExistsError):
            P.Logger(self.p('l.txt'))

    def test_nonfinite_rejected(self):
        m = tiny()
        nan_fn = lambda pos: torch.full((1, DIM), float('nan'))
        with self.assertRaises(P.FailClosed):
            P.train_arm('head_only', m, self.head, self.batches, self.labels, nan_fn, quiet, len(self.arrays))
        bad = dict(self.head)
        bad['W'] = bad['W'].copy()
        bad['W'][0, 0] = np.inf
        with self.assertRaises(P.FailClosed):
            P.validate_head(bad, self.classes, K, DIM)
        with self.assertRaises(P.FailClosed):
            P.require_finite([torch.tensor([1.0, float('inf')])], 'x')


if __name__ == '__main__':
    unittest.main()
