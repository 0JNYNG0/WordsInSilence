'''Unit tests for stgcn_core.

Synthetic inputs and temporary synthetic ZIP/metadata files only; CPU only;
no camera, network or real checkpoints. Run: python -m unittest test_stgcn_core
'''
import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile

import torch

import stgcn_core as sc

PREFIX = sc.ENCODER_PREFIX


def make_adjacency(seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(3, 27, 27, generator=g)
    a = a / a.sum(dim=1, keepdim=True)
    return a / 3.0


def file_sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= max(int(d), 1)
    return list(reversed(stride))


def tensor_bytes(t, byteorder='little'):
    t = t.detach().cpu().contiguous()
    e = '<' if byteorder == 'little' else '>'
    if t.dtype == torch.float32:
        return t.numpy().astype(e + 'f4').tobytes(), 'Float'
    if t.dtype == torch.int64:
        return t.numpy().astype(e + 'i8').tobytes(), 'Long'
    raise TypeError('unsupported dtype for synthetic archive: %s' % t.dtype)


def state_to_spec(state, byteorder='little'):
    storages = {}
    entries = []
    for i, (key, value) in enumerate(state.items()):
        data, dt = tensor_bytes(value, byteorder)
        sk = str(i)
        storages[sk] = data
        shape = list(value.shape)
        entries.append({'key': key, 'shape': shape, 'stride': contiguous_stride(shape),
                        'storage_offset': 0, 'dtype': dt, 'storage_key': sk,
                        'storage_bytes': len(data)})
    return storages, entries


def write_archive(dirpath, tag, storages, entries, byteorder='little', prefix='archive',
                  include_byteorder=True, extra_inner=None, extra_outer=None,
                  corrupt_pattern=None, outer_compression=zipfile.ZIP_STORED):
    inner_buf = io.BytesIO()
    pfx = prefix + '/' if prefix else ''
    with zipfile.ZipFile(inner_buf, 'w', compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(pfx + 'data.pkl', b'placeholder-never-unpickled')
        if include_byteorder:
            zf.writestr(pfx + 'byteorder', byteorder.encode('ascii'))
        for sk, data in storages.items():
            zf.writestr(pfx + 'data/' + sk, data)
        zf.writestr(pfx + 'version', b'3')
        for name, data in (extra_inner or {}).items():
            zf.writestr(name, data)
    inner = inner_buf.getvalue()
    if corrupt_pattern is not None:
        pos = inner.find(corrupt_pattern)
        assert pos >= 0, 'corrupt pattern not found'
        buf = bytearray(inner)
        buf[pos + 1] ^= 0xFF
        inner = bytes(buf)
    outer_path = os.path.join(dirpath, tag + '_outer.zip')
    with zipfile.ZipFile(outer_path, 'w', compression=outer_compression) as zf:
        zf.writestr('synthetic_epoch1.ckpt', inner)
        for name, data in (extra_outer or {}).items():
            zf.writestr(name, data)
    meta_path = os.path.join(dirpath, tag + '_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump({'tensors': entries}, fh)
    return outer_path, meta_path, file_sha256(outer_path)


def randomized_encoder(seed):
    torch.manual_seed(seed)
    enc = sc.STGCNEncoder(make_adjacency(seed))
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(0.01 * torch.randn_like(p))
        enc.train()
        enc(torch.randn(2, 2, 8, 27))
    return enc.eval()


def state_with_heads(enc):
    state = {PREFIX + k: v for k, v in enc.state_dict().items()}
    g = torch.Generator().manual_seed(123)
    state['model.agg.weight_ih_l0'] = torch.randn(12, 8, generator=g)
    state['model.network_pred.0.weight'] = torch.randn(4, 4, generator=g)
    state['model.network_pred.0.bias'] = torch.randn(4, generator=g)
    return state


class ModelTests(unittest.TestCase):
    def test_forward_shape_and_finite(self):
        torch.manual_seed(0)
        enc = sc.STGCNEncoder(make_adjacency()).eval()
        with torch.no_grad():
            y = enc(torch.randn(2, 2, 12, 27))
        self.assertEqual(tuple(y.shape), (2, 256))
        self.assertTrue(bool(torch.isfinite(y).all()))
        clf = sc.STGCNClassifier(make_adjacency(), num_classes=5).eval()
        with torch.no_grad():
            z = clf(torch.randn(3, 2, 9, 27))
        self.assertEqual(tuple(z.shape), (3, 5))

    def test_state_key_layout(self):
        enc = sc.STGCNEncoder(make_adjacency())
        sd = enc.state_dict()
        keys = set(sd)
        self.assertIn('A', keys)
        self.assertEqual(tuple(sd['A'].shape), (3, 27, 27))
        for name in ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'):
            self.assertIn('data_bn.' + name, keys)
        self.assertEqual(enc.data_bn.num_features, 54)
        for i, (cin, cout, _s, _r) in enumerate(sc.BLOCK_SPECS):
            p = f'st_gcn_networks.{i}.'
            self.assertEqual(tuple(sd[p + 'gcn.conv.weight'].shape), (cout * 3, cin, 1, 1))
            self.assertEqual(tuple(sd[p + 'gcn.conv.bias'].shape), (cout * 3,))
            self.assertEqual(tuple(sd[p + 'tcn.2.weight'].shape), (cout, cout, 9, 1))
            self.assertIn(p + 'tcn.0.running_mean', keys)
            self.assertIn(p + 'tcn.3.weight', keys)
            has_res = any(k.startswith(p + 'residual.') for k in keys)
            self.assertEqual(has_res, i in (4, 7))
            self.assertEqual(tuple(sd[f'edge_importance.{i}'].shape), (3, 27, 27))
        self.assertEqual(sum(1 for k in keys if k.startswith('edge_importance.')), 10)
        self.assertEqual(enc.st_gcn_networks[0].residual(torch.ones(1)), 0)
        self.assertEqual(enc.st_gcn_networks[0].residual_kind, 'zero')
        self.assertEqual(enc.st_gcn_networks[1].residual_kind, 'identity')
        self.assertEqual(enc.st_gcn_networks[4].residual_kind, 'conv')

    def test_invalid_adjacency_and_input(self):
        with self.assertRaises(ValueError):
            sc.STGCNEncoder(torch.zeros(3, 27, 26))
        bad = make_adjacency()
        bad[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            sc.STGCNEncoder(bad)
        with self.assertRaises(TypeError):
            sc.STGCNEncoder([[0.0]])
        enc = sc.STGCNEncoder(make_adjacency())
        with self.assertRaises(ValueError):
            enc(torch.randn(1, 3, 8, 27))

    def test_gcn_layer_matmul_matches_einsum(self):
        torch.manual_seed(2)
        g = sc.ConvTemporalGraphical(4, 5).double()
        x = torch.randn(2, 4, 6, 27, dtype=torch.float64)
        a = make_adjacency(2).double()
        with torch.no_grad():
            g.reference_einsum = False
            mm = g(x, a)
            g.reference_einsum = True
            es = g(x, a)
        self.assertLess(float((mm - es).abs().max()), 1e-12)

    def test_encoder_matmul_matches_einsum_eval(self):
        torch.manual_seed(1)
        enc = sc.STGCNEncoder(make_adjacency(1))
        with torch.no_grad():
            for p in enc.edge_importance:
                p.add_(0.1 * torch.randn_like(p))
            enc.train()
            for _ in range(3):
                enc(torch.randn(4, 2, 16, 27))
        enc.eval()
        x = torch.randn(2, 2, 16, 27)
        with torch.no_grad():
            enc.set_reference_einsum(False)
            y_mm = enc(x)
            enc.set_reference_einsum(True)
            y_es = enc(x)
        self.assertTrue(bool(torch.isfinite(y_mm).all()))
        self.assertLessEqual(float((y_mm - y_es).abs().max()), 1e-5)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self._n = 0

    def _write(self, storages, entries, **kw):
        self._n += 1
        return write_archive(self.dir, f'c{self._n}', storages, entries, **kw)

    def _write_state(self, state, **kw):
        storages, entries = state_to_spec(state)
        return self._write(storages, entries, **kw)

    def test_roundtrip_strict_load_and_no_mutation(self):
        src = randomized_encoder(3)
        outer, meta, sha = self._write_state(state_with_heads(src))
        before = (file_sha256(outer), file_sha256(meta))
        dst = sc.STGCNEncoder(make_adjacency(3))
        audit = sc.load_encoder_checkpoint(dst, outer, meta, sha)
        json.dumps(audit)
        dst_sd = dst.state_dict()
        for k, v in src.state_dict().items():
            self.assertTrue(torch.equal(v, dst_sd[k]), k)
        dst.eval()
        x = torch.randn(2, 2, 10, 27)
        with torch.no_grad():
            self.assertTrue(torch.equal(src(x), dst(x)))
        self.assertTrue(audit['strict_load'])
        self.assertEqual(audit['missing_keys'], [])
        self.assertEqual(audit['unexpected_keys'], [])
        self.assertEqual(audit['excluded_keys']['agg'], ['model.agg.weight_ih_l0'])
        self.assertEqual(audit['excluded_keys']['network_pred'],
                         ['model.network_pred.0.bias', 'model.network_pred.0.weight'])
        self.assertEqual(audit['excluded_keys']['other'], [])
        self.assertEqual(audit['archive_prefix'], 'archive')
        self.assertEqual(audit['byteorder'], 'little')
        self.assertFalse(audit['unsafe_deserialization_used'])
        self.assertEqual(before, (file_sha256(outer), file_sha256(meta)))

    def _assert_load_rejected_unchanged(self, state, needle=None):
        outer, meta, sha = self._write_state(state)
        dst = sc.STGCNEncoder(make_adjacency(9))
        snap = {k: v.clone() for k, v in dst.state_dict().items()}
        with self.assertRaises(sc.CheckpointValidationError) as cm:
            sc.load_encoder_checkpoint(dst, outer, meta, sha)
        if needle is not None:
            self.assertIn(needle, str(cm.exception))
        for k, v in dst.state_dict().items():
            self.assertTrue(torch.equal(v, snap[k]), k)

    def test_missing_state_key_raises(self):
        state = state_with_heads(randomized_encoder(4))
        del state[PREFIX + 'data_bn.weight']
        self._assert_load_rejected_unchanged(state, 'data_bn.weight')

    def test_unexpected_state_key_raises(self):
        state = state_with_heads(randomized_encoder(4))
        state[PREFIX + 'extra.weight'] = torch.zeros(3)
        self._assert_load_rejected_unchanged(state, 'extra.weight')

    def test_shape_mismatch_raises(self):
        state = state_with_heads(randomized_encoder(4))
        state[PREFIX + 'data_bn.weight'] = torch.ones(53)
        self._assert_load_rejected_unchanged(state, 'data_bn.weight')

    def test_adjacency_policies(self):
        src = randomized_encoder(5)
        outer, meta, sha = self._write_state(state_with_heads(src))
        dst = sc.STGCNEncoder(make_adjacency(6))
        with self.assertRaises(sc.CheckpointValidationError):
            sc.load_encoder_checkpoint(dst, outer, meta, sha, adjacency_policy='require_match')
        own_a = dst.A.clone()
        audit = sc.load_encoder_checkpoint(dst, outer, meta, sha, adjacency_policy='keep_current')
        self.assertTrue(torch.equal(dst.A, own_a))
        self.assertTrue(torch.equal(dst.data_bn.running_mean, src.data_bn.running_mean))
        self.assertGreater(audit['adjacency']['max_abs_diff_vs_current'], 0.0)

    def test_shared_storage_offsets_and_strides(self):
        base = torch.arange(40, dtype=torch.float32) * 0.5 - 3.0
        lbase = torch.arange(10, dtype=torch.int64) * 7
        fb, _ = tensor_bytes(base)
        lb, _ = tensor_bytes(lbase)
        storages = {'shared': fb, 'longs': lb}
        specs = [
            ('x', [2, 3], [3, 1], 4, 'shared'),
            ('y', [3, 2], [1, 3], 4, 'shared'),
            ('z', [5], [2], 10, 'shared'),
            ('w', [2, 2, 2], [8, 1, 3], 27, 'shared'),
            ('n', [], [], 3, 'longs'),
            ('m', [2, 2], [2, 1], 6, 'longs'),
        ]
        entries = []
        for name, shape, stride, off, sk in specs:
            is_f = sk == 'shared'
            entries.append({'key': PREFIX + name, 'shape': shape, 'stride': stride,
                            'storage_offset': off, 'dtype': 'Float' if is_f else 'Long',
                            'storage_key': sk, 'storage_bytes': len(fb) if is_f else len(lb)})
        outer, meta, sha = self._write(storages, entries, outer_compression=zipfile.ZIP_DEFLATED)
        tensors, audit = sc.read_encoder_tensors(outer, meta, sha)
        json.dumps(audit)
        self.assertEqual(audit['storages_read'], 2)
        for name, shape, stride, off, sk in specs:
            src = base if sk == 'shared' else lbase
            expected = torch.as_strided(src, shape, stride, off)
            got = tensors[name]
            self.assertEqual(got.dtype, src.dtype)
            self.assertTrue(torch.equal(got, expected), name)
            self.assertTrue(got.is_contiguous())
        before_y = tensors['y'].clone()
        tensors['x'][0, 0] += 100.0
        self.assertTrue(torch.equal(tensors['y'], before_y))

    def test_big_endian_roundtrip(self):
        vals = torch.linspace(-2.0, 2.0, 8)
        data, _ = tensor_bytes(vals, 'big')
        entries = [{'key': PREFIX + 't', 'shape': [8], 'stride': [1], 'storage_offset': 0,
                    'dtype': 'Float', 'storage_key': '0', 'storage_bytes': len(data)}]
        outer, meta, sha = self._write({'0': data}, entries, byteorder='big')
        tensors, audit = sc.read_encoder_tensors(outer, meta, sha)
        self.assertEqual(audit['byteorder'], 'big')
        self.assertTrue(torch.equal(tensors['t'], vals))

    def test_missing_byteorder_defaults_little(self):
        storages, entries = self._small()
        outer, meta, sha = self._write(storages, entries, include_byteorder=False)
        tensors, audit = sc.read_encoder_tensors(outer, meta, sha)
        self.assertEqual(audit['byteorder_source'], 'default')
        self.assertTrue(torch.equal(tensors['t'], torch.arange(8, dtype=torch.float32).view(2, 4)))

    def _small(self, values=None):
        base = torch.arange(8, dtype=torch.float32) if values is None else values
        data, _ = tensor_bytes(base)
        entries = [{'key': PREFIX + 't', 'shape': [2, 4], 'stride': [4, 1], 'storage_offset': 0,
                    'dtype': 'Float', 'storage_key': '0', 'storage_bytes': len(data)}]
        return {'0': data}, entries

    def _reject(self, storages, entries, sha=None, **kw):
        outer, meta, real_sha = self._write(storages, entries, **kw)
        with self.assertRaises(sc.CheckpointValidationError):
            sc.read_encoder_tensors(outer, meta, real_sha if sha is None else sha)

    def test_small_baseline_ok(self):
        storages, entries = self._small()
        outer, meta, sha = self._write(storages, entries, outer_compression=zipfile.ZIP_DEFLATED)
        tensors, _audit = sc.read_encoder_tensors(outer, meta, sha)
        self.assertEqual(set(tensors), {'t'})

    def test_rejects_wrong_or_malformed_hash(self):
        storages, entries = self._small()
        self._reject(storages, entries, sha='0' * 64)
        self._reject(storages, entries, sha='abc')

    def test_rejects_out_of_bounds_extent(self):
        storages, entries = self._small()
        entries[0]['storage_offset'] = 1
        self._reject(storages, entries)
        storages, entries = self._small()
        entries[0]['shape'] = [3, 4]
        self._reject(storages, entries)
        storages, entries = self._small()
        entries[0]['stride'] = [5, 1]
        self._reject(storages, entries)

    def test_rejects_bad_strides(self):
        for bad in ([0, 1], [-4, 1], [4]):
            storages, entries = self._small()
            entries[0]['stride'] = bad
            self._reject(storages, entries)

    def test_rejects_disallowed_dtype(self):
        for bad in ('Half', 'Double', 'float32'):
            storages, entries = self._small()
            entries[0]['dtype'] = bad
            self._reject(storages, entries)

    def test_rejects_storage_length_mismatch(self):
        storages, entries = self._small()
        entries[0]['storage_bytes'] = 36
        self._reject(storages, entries)
        storages, entries = self._small()
        entries[0]['storage_bytes'] = 30
        self._reject(storages, entries)
        storages, entries = self._small()
        entries[0]['dtype'] = 'Long'
        self._reject(storages, entries)

    def test_rejects_non_finite(self):
        vals = torch.arange(8, dtype=torch.float32)
        vals[3] = float('nan')
        storages, entries = self._small(vals)
        self._reject(storages, entries)

    def test_rejects_duplicate_keys(self):
        storages, entries = self._small()
        entries.append(dict(entries[0]))
        self._reject(storages, entries)

    def test_rejects_crc_corruption(self):
        vals = torch.full((64,), 1234.5)
        storages, entries = self._small(vals)
        self._reject(storages, entries, corrupt_pattern=storages['0'])

    def test_rejects_bad_archive_structure(self):
        storages, entries = self._small()
        self._reject(storages, entries, extra_outer={'second.ckpt': b'x'})
        storages, entries = self._small()
        entries[0]['storage_key'] = '9'
        self._reject(storages, entries)
        storages, entries = self._small()
        self._reject(storages, entries, extra_inner={'other/data/0': storages['0']})
        storages, entries = self._small()
        entries[0]['key'] = 'model.agg.only'
        self._reject(storages, entries)
        storages, entries = self._small()
        del entries[0]['stride']
        self._reject(storages, entries)
        storages, entries = self._small()
        entries[0]['storage_key'] = '../escape'
        self._reject(storages, entries)


if __name__ == '__main__':
    unittest.main()
