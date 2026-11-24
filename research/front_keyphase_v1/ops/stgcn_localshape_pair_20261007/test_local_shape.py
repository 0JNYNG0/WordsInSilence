'''Synthetic unit tests for local_shape (no real data, no checkpoints).

Run with: python -m unittest test_local_shape
'''
import json
import unittest

import numpy as np
import torch
import torch.nn as nn

from local_shape import append_local, augment_paired, expand_input4, local_coordinates

V = 49
BN = 'conv_encoder.data_bn.'
CONV = 'conv_encoder.st_gcn_networks.0.gcn.conv.'


def sptr(t):
    try:
        return t.untyped_storage().data_ptr()
    except AttributeError:
        return t.storage().data_ptr()


class FakeGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 192, kernel_size=1, bias=True)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.shape
        x = x.view(n, 3, kc // 3, t, v)
        return torch.einsum('nkctv,kvw->nctw', x, A)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = FakeGCN()
        self.tcn = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 32, (3, 1), padding=(1, 0)))
        self.residual = lambda x: 0

    def forward(self, x, A):
        return self.tcn(self.gcn(x, A)) + self.residual(x)


class FakeEncoder(nn.Module):
    def __init__(self, bn_kwargs=None):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(98, **(bn_kwargs or {}))
        self.register_buffer('A', torch.rand(3, V, V) / V)
        self.st_gcn_networks = nn.ModuleList([FakeBlock()])

    def forward(self, x):
        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).reshape(n, v * c, t)  # node-major / channel-minor
        x = self.data_bn(x)
        x = x.reshape(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        for blk in self.st_gcn_networks:
            x = blk(x, self.A)
        return x.mean(dim=3)


class FakeModel(nn.Module):
    def __init__(self, bn_kwargs=None):
        super().__init__()
        self.conv_encoder = FakeEncoder(bn_kwargs)
        self.gru = nn.GRU(32, 16, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(16, 5))

    def forward(self, x):
        f = self.conv_encoder(x).permute(0, 2, 1)
        o, _ = self.gru(f)
        return self.head(o[:, -1])


def make_model(bn_kwargs=None, seed=0):
    torch.manual_seed(seed)
    m = FakeModel(bn_kwargs)
    bn = m.conv_encoder.data_bn
    ar = torch.arange(98, dtype=torch.float32)
    with torch.no_grad():
        if bn.affine:
            bn.weight.copy_(1.0 + 0.01 * ar)
            bn.bias.copy_(0.05 * ar - 2.0)
        if bn.track_running_stats:
            bn.running_mean.copy_(0.1 * ar - 1.0)
            bn.running_var.copy_(0.5 + 0.02 * ar)
            bn.num_batches_tracked.fill_(17)
        conv = first_conv(m)
        conv.weight.copy_(torch.arange(384, dtype=torch.float32).reshape(192, 2, 1, 1) * 1e-3 - 0.2)
        conv.bias.copy_(torch.arange(192, dtype=torch.float32) * 1e-2 - 1.0)
    return m


def first_conv(m):
    return m.conv_encoder.st_gcn_networks[0].gcn.conv


def rand_x(n=3, c=2, t=12, seed=5):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, c, t, V, generator=gen)


class TestExpandInput4(unittest.TestCase):
    def test_bn_mapping_defaults_and_attributes(self):
        m = make_model(dict(eps=1e-3, momentum=0.2))
        m.conv_encoder.data_bn.bias.requires_grad_(False)
        m.conv_encoder.data_bn.eval()
        new, _ = expand_input4(m)
        ob, nb = m.conv_encoder.data_bn, new.conv_encoder.data_bn
        self.assertIs(type(nb), nn.BatchNorm1d)
        self.assertEqual(nb.num_features, 196)
        self.assertEqual(ob.num_features, 98)
        self.assertEqual((nb.eps, nb.momentum, nb.affine, nb.track_running_stats), (1e-3, 0.2, True, True))
        self.assertFalse(nb.training)
        self.assertTrue(new.training)
        self.assertTrue(nb.weight.requires_grad)
        self.assertFalse(nb.bias.requires_grad)
        defaults = {'weight': 1.0, 'bias': 0.0, 'running_mean': 0.0, 'running_var': 1.0}
        for name, dv in defaults.items():
            o = getattr(ob, name).detach()
            n = getattr(nb, name).detach()
            self.assertEqual(n.dtype, torch.float32)
            self.assertEqual(n.device, o.device)
            self.assertTrue(torch.equal(n.view(49, 4)[:, :2], o.view(49, 2)))
            self.assertTrue(torch.equal(n.view(49, 4)[:, 2:], torch.full((49, 2), dv)))
            for j in range(49):
                for c in (0, 1):
                    self.assertEqual(n[4 * j + c].item(), o[2 * j + c].item())
        self.assertEqual(nb.num_batches_tracked.item(), 17)
        self.assertEqual(nb.num_batches_tracked.dtype, torch.long)

    def test_first_conv_mapping(self):
        m = make_model()
        new, _ = expand_input4(m)
        oc, nc = first_conv(m), first_conv(new)
        self.assertIs(type(nc), nn.Conv2d)
        self.assertEqual((oc.in_channels, nc.in_channels), (2, 4))
        for k in ('out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups',
                  'padding_mode', 'transposed', 'output_padding', 'training'):
            self.assertEqual(getattr(nc, k), getattr(oc, k), k)
        self.assertEqual(tuple(nc.weight.shape), (192, 4, 1, 1))
        self.assertTrue(torch.equal(nc.weight[:, :2].detach(), oc.weight.detach()))
        self.assertTrue(torch.equal(nc.weight[:, 2:].detach(), torch.zeros(192, 2, 1, 1)))
        self.assertTrue(torch.equal(nc.bias.detach(), oc.bias.detach()))
        self.assertEqual((nc.weight.dtype, nc.weight.device), (oc.weight.dtype, oc.weight.device))
        self.assertTrue(nc.weight.requires_grad and nc.bias.requires_grad)

    def test_state_report_and_source_untouched(self):
        m = make_model()
        before = {k: v.clone() for k, v in m.state_dict().items()}
        new, rep = expand_input4(m)
        src = m.state_dict()
        self.assertEqual(set(src), set(before))
        for k, v in before.items():
            self.assertTrue(torch.equal(src[k], v), k)
        self.assertEqual(first_conv(m).in_channels, 2)
        nsd = new.state_dict()
        self.assertEqual(set(nsd), set(before))
        mapped = {BN + 'weight', BN + 'bias', BN + 'running_mean', BN + 'running_var', CONV + 'weight'}
        self.assertEqual(set(rep['mapped']), mapped)
        self.assertEqual(set(rep['exact']) | mapped, set(before))
        self.assertTrue(set(rep['exact']).isdisjoint(mapped))
        for k in (BN + 'num_batches_tracked', CONV + 'bias', 'conv_encoder.A'):
            self.assertIn(k, rep['exact'])
        for k in rep['exact']:
            self.assertEqual(nsd[k].dtype, before[k].dtype)
            self.assertTrue(torch.equal(nsd[k], before[k]), k)
        self.assertEqual(rep['added_keys'], [])
        self.assertEqual(rep['missing_keys'], [])
        self.assertTrue(rep['all_original_keys_retained'])
        self.assertEqual(rep['shape_changes'][BN + 'weight'], {'old': [98], 'new': [196]})
        self.assertEqual(rep['shape_changes'][CONV + 'weight'], {'old': [192, 2, 1, 1], 'new': [192, 4, 1, 1]})
        self.assertEqual(rep['added_parameter_elements'][CONV + 'weight'], 384)
        json.dumps(rep)

    def test_no_shared_storage(self):
        m = make_model()
        new, _ = expand_input4(m)
        old_ptrs = {sptr(t) for t in list(m.parameters()) + list(m.buffers())}
        for t in list(new.parameters()) + list(new.buffers()):
            self.assertNotIn(sptr(t), old_ptrs)
        ref = m.gru.weight_ih_l0.detach().clone()
        with torch.no_grad():
            new.gru.weight_ih_l0.add_(1.0)
            new.conv_encoder.data_bn.weight.add_(1.0)
        self.assertTrue(torch.equal(m.gru.weight_ih_l0.detach(), ref))
        self.assertEqual(m.conv_encoder.data_bn.weight[0].item(), 1.0)

    def test_global_rng_preserved(self):
        m = make_model()
        torch.manual_seed(1234)
        state = torch.get_rng_state()
        expand_input4(m)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        probe = torch.get_rng_state()
        nn.Conv2d(4, 192, 1)  # plain construction does consume RNG
        self.assertFalse(torch.equal(torch.get_rng_state(), probe))

    def test_rejections_fail_before_mutation(self):
        m = make_model()
        m.head[1].forward = lambda x: x
        with self.assertRaises(ValueError):
            expand_input4(m)
        m = make_model()
        with torch.no_grad():
            m.gru.weight_hh_l0[0, 0] = float('nan')
        before = {k: v.clone() for k, v in m.state_dict().items()}
        with self.assertRaises(ValueError):
            expand_input4(m)
        self.assertEqual(m.conv_encoder.data_bn.num_features, 98)
        self.assertEqual(first_conv(m).in_channels, 2)
        for k, v in m.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0, equal_nan=True)
        m = make_model().double()
        with self.assertRaises(ValueError):
            expand_input4(m)
        m = make_model()
        m.conv_encoder.st_gcn_networks[0].gcn.conv = nn.Conv2d(3, 192, 1)
        with self.assertRaises(ValueError):
            expand_input4(m)
        m = make_model()
        m.conv_encoder.st_gcn_networks[0].residual = nn.Conv2d(2, 192, 1)
        with self.assertRaises(ValueError):
            expand_input4(m)
        m = make_model()
        m.conv_encoder.data_bn = nn.BatchNorm1d(100)
        with self.assertRaises(ValueError):
            expand_input4(m)
        with self.assertRaises(TypeError):
            expand_input4('not a module')

    def test_initial_output_eval_and_train(self):
        x = rand_x()
        x4 = torch.cat([x, rand_x(seed=6)], dim=1)
        m = make_model()
        new, _ = expand_input4(m)
        m.eval()
        new.eval()
        with torch.no_grad():
            torch.testing.assert_close(new(x4), m(x), rtol=1e-5, atol=1e-5)
        m2 = make_model()
        new2, _ = expand_input4(m2)
        m2.train()
        new2.train()
        with torch.no_grad():
            torch.manual_seed(9)
            a = m2(x)
            torch.manual_seed(9)
            b = new2(x4)
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5)
        ob, nb = m2.conv_encoder.data_bn, new2.conv_encoder.data_bn
        torch.testing.assert_close(nb.running_mean.view(49, 4)[:, :2], ob.running_mean.view(49, 2), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(nb.running_var.view(49, 4)[:, :2], ob.running_var.view(49, 2), rtol=1e-5, atol=1e-5)

    def test_state_dict_forward_compatible_and_trainable(self):
        new, _ = expand_input4(make_model())
        other, _ = expand_input4(make_model(seed=3))
        other.load_state_dict(new.state_dict(), strict=True)
        x4 = rand_x(n=2, c=4, t=8)
        new.eval()
        other.eval()
        with torch.no_grad():
            torch.testing.assert_close(other(x4), new(x4))
        new.train()
        opt = torch.optim.SGD(new.parameters(), lr=0.01)
        loss = new(x4).pow(2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            self.assertTrue(bool(torch.isfinite(new(x4)).all()))

    def test_new_conv_local_weights_get_gradient(self):
        new, _ = expand_input4(make_model())
        new.eval()
        x4 = rand_x(n=2, c=4, t=8, seed=11)
        new(x4).pow(2).sum().backward()
        grad = first_conv(new).weight.grad
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad[:, 2:].abs().sum()), 0.0)
        bn_grad = new.conv_encoder.data_bn.weight.grad.view(49, 4)[:, 2:]
        self.assertTrue(bool(torch.isfinite(bn_grad).all()))  # may be zero initially

    def test_non_affine_untracked_bn(self):
        m = make_model(dict(affine=False, track_running_stats=False))
        new, rep = expand_input4(m)
        nb = new.conv_encoder.data_bn
        self.assertFalse(nb.affine or nb.track_running_stats)
        self.assertIsNone(nb.weight)
        self.assertIsNone(nb.running_mean)
        self.assertEqual(set(rep['mapped']), {CONV + 'weight'})
        x = rand_x()
        x4 = torch.cat([x, rand_x(seed=7)], dim=1)
        m.eval()
        new.eval()
        with torch.no_grad():
            torch.testing.assert_close(new(x4), m(x), rtol=1e-5, atol=1e-5)


def rand_clip(T=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, V, 2)), np.ones((T, V), dtype=bool)


class TestLocalCoordinates(unittest.TestCase):
    def test_values_dtype_and_diag(self):
        xy, pr = rand_clip()
        loc, d = local_coordinates(xy, pr)
        self.assertEqual(loc.dtype, np.float32)
        self.assertEqual(loc.shape, (8, V, 2))
        self.assertTrue(np.all(loc[:, :7] == 0))
        self.assertTrue(np.all(loc[:, 7] == 0) and np.all(loc[:, 28] == 0))
        for name, b in (('left', 7), ('right', 28)):
            dists = np.concatenate([np.linalg.norm(xy[:, b + m] - xy[:, b], axis=-1) for m in (5, 9, 13, 17)])
            self.assertAlmostEqual(d[name]['scale'], float(np.median(dists)), places=12)
            self.assertEqual(d[name]['count_valid_pairs'], 8 * 4)
            self.assertEqual(d[name]['valid_local_points'], 8 * 21)
            exp = (xy[:, b:b + 21] - xy[:, b:b + 1]) / d[name]['scale']
            np.testing.assert_allclose(loc[:, b:b + 21], exp, rtol=1e-5, atol=1e-6)
        json.dumps(d)

    def test_translation_and_scale_invariance(self):
        xy, pr = rand_clip(seed=1)
        shift = np.random.default_rng(2).normal(size=(8, 1, 2)) * 10.0
        a, _ = local_coordinates(xy, pr)
        b, _ = local_coordinates(xy * 3.7 + shift, pr)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

    def test_pooled_median_not_frame_average(self):
        xy = np.full((3, V, 2), np.nan)
        pr = np.zeros((3, V), dtype=bool)
        b = 7
        xy[:, b] = 0.0
        pr[:, b] = True
        for m in (5, 9, 13, 17):
            xy[0, b + m] = [1.0, 0.0]
            pr[0, b + m] = True
        xy[1, b + 5] = [10.0, 0.0]
        pr[1, b + 5] = True
        xy[2, b + 9] = [0.0, 10.0]
        pr[2, b + 9] = True
        loc, d = local_coordinates(xy, pr)
        # pooled [1,1,1,1,10,10] -> 1.0 ; per-frame means [1,10,10] -> 10.0
        self.assertEqual(d['left']['scale'], 1.0)
        self.assertEqual(d['left']['count_valid_pairs'], 6)
        np.testing.assert_array_equal(loc[1, b + 5], [10.0, 0.0])
        np.testing.assert_array_equal(loc[2, b + 9], [0.0, 10.0])
        np.testing.assert_array_equal(loc[0, b + 5], [1.0, 0.0])
        self.assertIsNone(d['right']['scale'])
        self.assertEqual(d['right']['status'], 'no_valid_pairs')
        self.assertEqual(d['right']['valid_local_points'], 0)
        self.assertTrue(np.all(loc[:, 28:] == 0))
        self.assertTrue(np.isfinite(loc).all())

    def test_individual_point_missing(self):
        xy, pr = rand_clip(seed=3)
        pr[2, 15] = False
        xy[2, 15] = np.nan
        pr[4, 16] = False  # MCP 9 of left hand
        xy[4, 16] = np.nan
        loc, d = local_coordinates(xy, pr)
        self.assertTrue(np.all(loc[2, 15] == 0))
        self.assertTrue(np.any(loc[2, 14] != 0))
        self.assertTrue(np.all(loc[4, 16] == 0))
        self.assertEqual(d['left']['count_valid_pairs'], 8 * 4 - 1)
        self.assertEqual(d['left']['valid_local_points'], 8 * 21 - 2)

    def test_wrist_missing(self):
        xy, pr = rand_clip(T=6, seed=3)
        pr[2, 28] = False
        xy[2, 28] = np.nan
        loc, d = local_coordinates(xy, pr)
        self.assertTrue(np.all(loc[2, 28:49] == 0))
        self.assertTrue(np.any(loc[3, 29:49] != 0))
        self.assertTrue(np.any(loc[2, 8:28] != 0))
        self.assertEqual(d['right']['count_valid_pairs'], 5 * 4)
        self.assertEqual(d['right']['valid_local_points'], 5 * 21)

    def test_all_missing_and_degenerate(self):
        xy, pr = rand_clip(seed=4)
        xy[:, 28:49] = xy[:, 28:29]
        loc, d = local_coordinates(xy, pr)
        self.assertIsNone(d['right']['scale'])
        self.assertEqual(d['right']['status'], 'degenerate_scale')
        self.assertTrue(np.all(loc[:, 28:] == 0))
        self.assertEqual(d['left']['status'], 'ok')
        xy2, pr2 = rand_clip(seed=4)
        xy2[:, 7:28] = xy2[:, 7:8] + 1e-8
        loc2, d2 = local_coordinates(xy2, pr2)
        self.assertIsNone(d2['left']['scale'])
        self.assertTrue(np.all(loc2[:, 7:28] == 0))
        xy3 = np.full((5, V, 2), np.nan)
        loc3, d3 = local_coordinates(xy3, np.zeros((5, V), dtype=bool))
        self.assertTrue(np.all(loc3 == 0))
        self.assertIsNone(d3['left']['scale'])
        self.assertIsNone(d3['right']['scale'])
        json.dumps(d3)

    def test_observed_nonfinite_rejected_missing_nan_allowed(self):
        xy, pr = rand_clip()
        bad = xy.copy()
        bad[0, 3, 1] = np.nan
        with self.assertRaises(ValueError):
            local_coordinates(bad, pr)
        bad = xy.copy()
        bad[1, 40, 0] = np.inf
        with self.assertRaises(ValueError):
            local_coordinates(bad, pr)
        ok = xy.copy()
        ok[0, 3] = np.nan
        pr2 = pr.copy()
        pr2[0, 3] = False
        loc, _ = local_coordinates(ok, pr2)
        self.assertTrue(np.isfinite(loc).all())

    def test_no_input_mutation_and_validation(self):
        xy, pr = rand_clip()
        xy[0, 20] = np.nan
        pr[0, 20] = False
        xy_c, pr_c = xy.copy(), pr.copy()
        local_coordinates(xy, pr)
        self.assertTrue(np.array_equal(xy, xy_c, equal_nan=True))
        self.assertTrue(np.array_equal(pr, pr_c))
        for bad_xy, bad_pr in ((xy.astype(int), pr), (xy[:, :48], pr[:, :48]), (xy, pr.astype(int)),
                               (xy, pr[:-1]), (xy.tolist(), pr)):
            with self.assertRaises((TypeError, ValueError)):
                local_coordinates(bad_xy, bad_pr)


def make_global(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, 6, 2, 10, V, generator=gen)


class TestAppendLocal(unittest.TestCase):
    def check_local(self, out, local, idx):
        for k, i in enumerate(idx):
            w, t = divmod(k, 10)
            exp = local[int(i)].T if i >= 0 else np.zeros((2, V), np.float32)
            np.testing.assert_array_equal(out[0, w, 2:, t, :].numpy(), exp)

    def test_padding_same_indices_global_bitexact(self):
        xy, pr = rand_clip(T=25, seed=5)
        idx = list(range(25)) + [-1] * 35
        gw = make_global()
        gw_c = gw.clone()
        out, d = append_local(gw, xy, pr, idx)
        self.assertEqual(tuple(out.shape), (1, 6, 4, 10, V))
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(out[:, :, :2], gw))
        self.assertTrue(torch.equal(gw, gw_c))
        local, _ = local_coordinates(xy, pr)
        self.check_local(out, local, idx)
        self.assertEqual((d['num_valid_indices'], d['num_padding']), (25, 35))
        json.dumps(d)

    def test_downsample_uses_full_clip_local(self):
        xy, pr = rand_clip(T=120, seed=6)
        idx = np.arange(0, 120, 2)
        gw = make_global(1)
        out, d = append_local(gw, xy, pr, idx)
        local, _ = local_coordinates(xy, pr)
        self.check_local(out, local, idx)
        self.assertEqual(d['local']['left']['count_valid_pairs'], 120 * 4)
        self.assertTrue(torch.equal(out[:, :, :2], gw))
        out_t, _ = append_local(gw, xy, pr, torch.from_numpy(idx))
        self.assertTrue(torch.equal(out_t, out))

    def test_rejections(self):
        xy, pr = rand_clip(T=25)
        good = list(range(25)) + [-1] * 35
        gw = make_global()
        cases = [
            (gw[..., :48], good), (gw.double(), good), (gw[:, :5], good),
            (gw, good[:59]), (gw, good[:-1] + [25]), (gw, [-2] + good[1:]),
            (gw, [3, 2] + good[2:]), (gw, np.arange(60.0)), (gw, [True] + good[1:]),
        ]
        for g, idx in cases:
            with self.assertRaises((ValueError, TypeError)):
                append_local(g, xy, pr, idx)
        bad = xy.copy()
        bad[3, 30, 0] = np.nan
        with self.assertRaises(ValueError):
            append_local(gw, bad, pr, good)


class ShearRotate:
    '''Stand-in for an official pointwise linear 2D transform on [2, N, V].'''

    def __init__(self):
        self.calls = 0
        self.mats = []

    def __call__(self, x):
        self.calls += 1
        if x.dim() != 3 or x.shape[0] != 2:
            raise AssertionError('bad input shape')
        shear = (torch.rand(2) - 0.5) * 0.6
        angle = (torch.rand(()) - 0.5) * 1.2
        s_mat = torch.eye(2)
        s_mat[0, 1] = shear[0]
        s_mat[1, 0] = shear[1]
        c, s = torch.cos(angle), torch.sin(angle)
        r_mat = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        mat = r_mat @ s_mat
        self.mats.append(mat)
        return torch.einsum('ij,jnv->inv', mat, x)


class TestAugmentPaired(unittest.TestCase):
    def make_sample(self, W=3, T=7):
        gen = torch.Generator().manual_seed(0)
        sample = torch.randn(W, 4, T, V, generator=gen)
        sample[:, 2:, :, :8] = 0.0  # body + left wrist local are zero
        return sample

    def test_shared_single_call_and_global_identical_seed(self):
        W, T = 3, 7
        sample = self.make_sample(W, T)
        orig = sample.clone()
        tf = ShearRotate()
        torch.manual_seed(42)
        fresh = torch.get_rng_state()
        out = augment_paired(sample, tf)
        state_paired = torch.get_rng_state()
        self.assertEqual(tf.calls, 1)
        self.assertEqual(out.shape, sample.shape)
        self.assertEqual(out.dtype, sample.dtype)
        self.assertTrue(torch.equal(sample, orig))
        self.assertFalse(torch.equal(fresh, state_paired))  # no RNG restore inside
        tf2 = ShearRotate()
        torch.manual_seed(42)
        g = sample[:, :2].permute(1, 0, 2, 3).reshape(2, W * T, V)
        g_out = tf2(g).reshape(2, W, T, V).permute(1, 0, 2, 3)
        self.assertTrue(torch.equal(torch.get_rng_state(), state_paired))
        self.assertTrue(torch.equal(tf.mats[0], tf2.mats[0]))
        torch.testing.assert_close(out[:, :2], g_out, rtol=1e-6, atol=1e-6)
        exp_local = torch.einsum('ij,wjtv->witv', tf.mats[0], sample[:, 2:])
        torch.testing.assert_close(out[:, 2:], exp_local, rtol=1e-6, atol=1e-6)
        self.assertTrue(bool((out[:, 2:, :, :8] == 0).all()))

    def test_inplace_transform_does_not_mutate_sample(self):
        sample = self.make_sample()
        orig = sample.clone()
        out = augment_paired(sample, lambda x: x.mul_(2.0))
        self.assertTrue(torch.equal(sample, orig))
        self.assertTrue(torch.equal(out, orig * 2.0))

    def test_rejections(self):
        sample = self.make_sample()
        for fn in (lambda x: x[:, :, :49], lambda x: x * float('nan'), lambda x: x.double(),
                   lambda x: x.numpy()):
            with self.assertRaises((ValueError, TypeError)):
                augment_paired(sample, fn)
        with self.assertRaises(ValueError):
            augment_paired(sample[:, :2], lambda x: x)
        bad = sample.clone()
        bad[0, 0, 0, 0] = float('inf')
        with self.assertRaises(ValueError):
            augment_paired(bad, lambda x: x)


if __name__ == '__main__':
    unittest.main()
