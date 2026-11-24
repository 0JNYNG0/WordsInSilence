'''Synthetic unit-contract tests for reference_adapter (python -m unittest).

The transform/model stand-ins here are NOT official OpenHands code; they only
exercise the adapter contract. Official equivalence with the pinned release
sources and real weights is audited separately.
'''
import builtins
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

import reference_adapter as ra

PRESET = 'shoulder_mediapipe_holistic_minimal_27'
GRAPH_ARGS = {'layout': 'mediapipe-27', 'strategy': 'spatial'}
W, H = 640.0, 480.0
MARK = '_reference_adapter_test_executed'

SPEC_MP75 = (0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54,
             58, 59, 62, 63, 66, 67, 70, 71, 74)
SPEC_NATIVE67 = (0, 16, 15, 5, 2, 6, 3, 25, 29, 30, 33, 34, 37, 38, 41, 42, 45,
                 46, 50, 51, 54, 55, 58, 59, 62, 63, 66)


class StubCenterAndScale:
    '''Stand-in: clip-level shoulder-midpoint center, mean shoulder distance (zeros included).'''

    def __init__(self, reference_points_preset=None, frame_level=False):
        assert reference_points_preset == PRESET and frame_level is False

    def __call__(self, data):
        x = data['frames']
        ref = x[:, :, [3, 4]]
        center = ref.mean(dim=(1, 2), keepdim=True)
        scale = torch.linalg.vector_norm(ref[:, :, 0] - ref[:, :, 1], dim=0).mean()
        data['frames'] = (x - center) / scale
        return data


class IdentityTransform:
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        return data


STUB_TRANSFORMS = types.SimpleNamespace(CenterAndScaleNormalize=StubCenterAndScale)
IDENTITY_TRANSFORMS = types.SimpleNamespace(CenterAndScaleNormalize=IdentityTransform)


def make_clip(t, layout='mp75', seed=0):
    joints = ra.LAYOUTS[layout][0]
    rng = np.random.default_rng(seed)
    return rng.uniform(20.0, 460.0, size=(t, joints, 2)), np.ones((t, joints), dtype=bool)


def expected_normalized(xy, present, layout, width, height):
    '''Independent scale -> select -> stub-normalize, returns C,T,V.'''
    data = np.where(present[..., None], xy, 0.0).astype(np.float32)
    data /= np.array([width, height], dtype=np.float32)
    sel = np.ascontiguousarray(data[:, list(ra.LAYOUTS[layout][1]), :].transpose(2, 0, 1))
    return StubCenterAndScale(PRESET)({'frames': torch.from_numpy(sel)})['frames']


def unwindow(windows):
    '''[1, 6, 2, 10, 27] -> [2, 60, 27].'''
    return windows[0].permute(1, 0, 2, 3).reshape(2, 60, 27)


class PrepareWindowsTests(unittest.TestCase):
    def run_prepare(self, xy, present, layout='mp75', width=W, height=H, transforms=STUB_TRANSFORMS):
        return ra.prepare_windows(xy, present, layout, width, height, transforms)

    def test_layout_indices_match_spec(self):
        self.assertEqual(ra.LAYOUTS['mp75'][1], SPEC_MP75)
        self.assertEqual(ra.LAYOUTS['native67'][1], SPEC_NATIVE67)

    def test_dimension_missing_or_invalid_is_blocker(self):
        xy, present = make_clip(60)
        for bad in (None, 0, -1.0, float('nan'), float('inf'), True, '640', 1e-60):
            for which in ('width', 'height'):
                kwargs = {'width': W, 'height': H, which: bad}
                with self.subTest(bad=bad, which=which):
                    with self.assertRaises(ValueError):
                        self.run_prepare(xy, present, **kwargs)

    def test_present_nonfinite_rejected(self):
        for bad in (np.nan, np.inf):
            xy, present = make_clip(60)
            xy[5, 11, 0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.run_prepare(xy, present)

    def test_present_must_be_bool(self):
        xy, present = make_clip(60)
        with self.assertRaises(ValueError):
            self.run_prepare(xy, present.astype(np.float32))

    def test_bad_shapes_and_empty_rejected(self):
        xy, present = make_clip(60)
        cases = {
            'unknown_layout': (xy, present, 'mp33'),
            'wrong_joints': (xy[:, :67], present[:, :67], 'mp75'),
            'empty': (xy[:0], present[:0], 'mp75'),
        }
        for label, (a, p, layout) in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                self.run_prepare(a, p, layout)

    def test_zero_shoulder_distance_ineligible(self):
        xy, present = make_clip(60)
        with self.assertRaises(ValueError):
            self.run_prepare(xy, np.zeros_like(present))

    def test_inputs_not_mutated(self):
        xy, present = make_clip(40)
        present[:, 33] = False
        xy[:, 33] = np.nan
        xy_before, present_before = xy.copy(), present.copy()
        self.run_prepare(xy, present)
        np.testing.assert_array_equal(xy, xy_before)
        np.testing.assert_array_equal(present, present_before)

    def test_missing_joint_zeroed_before_normalization_not_after(self):
        xy, present = make_clip(60)
        present[:, 33] = False
        xy[:, 33] = np.nan  # absent joints may carry garbage
        out, diag = self.run_prepare(xy, present)
        frames = unwindow(out)
        torch.testing.assert_close(frames, expected_normalized(xy, present, 'mp75', W, H))
        k = SPEC_MP75.index(33)
        self.assertGreater(float(frames[:, :, k].abs().min()), 0.0)
        self.assertEqual(diag['source_missing_count'], 60)
        self.assertEqual(diag['selected_missing_count'], 60)

    def test_length_boundaries_59_60_61(self):
        for t in (59, 60, 61):
            with self.subTest(t=t):
                xy, present = make_clip(t, seed=t)
                if t == 61:  # outlier last frame shifts clip statistics
                    xy[60, 11] = (5.0, 5.0)
                    xy[60, 12] = (630.0, 470.0)
                out, diag = self.run_prepare(xy, present)
                self.assertEqual(tuple(out.shape), (1, 6, 2, 10, 27))
                self.assertEqual(out.dtype, torch.float32)
                self.assertTrue(out.is_contiguous())
                self.assertEqual(diag['original_frames'], t)
                frames = unwindow(out)
                full = expected_normalized(xy, present, 'mp75', W, H)
                if t == 59:
                    self.assertTrue(torch.all(frames[:, 0] == 0))
                    torch.testing.assert_close(frames[:, 1:], full)
                    self.assertEqual((diag['padded_frames'], diag['truncated_frames']), (1, 0))
                elif t == 60:
                    torch.testing.assert_close(frames, full)
                    self.assertEqual((diag['padded_frames'], diag['truncated_frames']), (0, 0))
                else:
                    torch.testing.assert_close(frames, full[:, :60])
                    first60 = expected_normalized(xy[:60], present[:60], 'mp75', W, H)
                    self.assertFalse(torch.allclose(frames, first60))  # stats used all 61 frames
                    self.assertEqual((diag['padded_frames'], diag['truncated_frames']), (0, 1))

    def test_resolution_scaling_invariant(self):
        for layout in ('mp75', 'native67'):
            with self.subTest(layout=layout):
                xy, present = make_clip(45, layout)
                a, _ = self.run_prepare(xy, present, layout, W, H)
                b, _ = self.run_prepare(xy * 3.0, present, layout, W * 3.0, H * 3.0)
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_window_channel_time_joint_order(self):
        for layout in ('mp75', 'native67'):
            with self.subTest(layout=layout):
                joints, index = ra.LAYOUTS[layout]
                code = np.arange(60, dtype=np.float64)[:, None] * 1000.0 + np.arange(joints)[None, :]
                xy = np.stack([code, -code - 0.5], axis=-1)
                out, _ = self.run_prepare(xy, np.ones((60, joints), dtype=bool), layout,
                                          1.0, 1.0, IDENTITY_TRANSFORMS)
                frame = (np.arange(6)[:, None] * 10 + np.arange(10)[None, :]) * 1000.0
                base = frame[:, :, None] + np.array(index, dtype=np.float64)[None, None, :]
                expected = np.stack([base, -base - 0.5], axis=1)[None].astype(np.float32)
                np.testing.assert_array_equal(out.numpy(), expected)


PRELUDE = '''import builtins
builtins.__dict__.setdefault('{mark}', []).append('{tag}')
'''
SOURCES = {
    ra.ST_GCN: PRELUDE.format(mark=MARK, tag='st_gcn') + '''
class STModel:
    pass
''',
    ra.DPC_RNN: PRELUDE.format(mark=MARK, tag='dpc_rnn') + '''
from .st_gcn import STModel


class DPC_RNN_Finetuner:
    encoder_cls = STModel
''',
    ra.POSE_TRANSFORMS: PRELUDE.format(mark=MARK, tag='pose_transforms') + '''
class CenterAndScaleNormalize:
    pass
''',
}
MALICIOUS = PRELUDE.format(mark=MARK, tag='malicious')


def write_reference(root, edit=None, commit=ra.RELEASE_COMMIT):
    entries = []
    for upstream, text in SOURCES.items():
        data = text.encode()
        path = root / upstream
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append({
            'upstream_path': upstream,
            'local_path': upstream,
            'url': f'https://raw.githubusercontent.com/AI4Bharat/OpenHands/{commit}/{upstream}',
            'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data),
        })
    if edit is not None:
        entries = edit(entries)
    pins = json.dumps({'full_commit': commit, 'files': entries}).encode()
    (root / ra.PINS_FILENAME).write_bytes(pins)
    return hashlib.sha256(pins).hexdigest()


class LoadReferenceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / 'ref'
        self.root.mkdir()
        builtins.__dict__.pop(MARK, None)
        self.addCleanup(builtins.__dict__.pop, MARK, None)

    def executed(self):
        return builtins.__dict__.get(MARK, [])

    def test_loads_verified_sources_in_isolated_package(self):
        pins_sha = write_reference(self.root)
        before = {n for n in sys.modules if n.split('.')[0] == 'openhands'}
        dpc, transforms, audit = ra.load_reference(self.root, pins_sha)
        self.assertEqual(self.executed(), ['st_gcn', 'dpc_rnn', 'pose_transforms'])
        st_gcn = sys.modules[dpc.__name__.rsplit('.', 1)[0] + '.st_gcn']
        self.assertIs(dpc.DPC_RNN_Finetuner.encoder_cls, st_gcn.STModel)
        self.assertTrue(hasattr(transforms, 'CenterAndScaleNormalize'))
        self.assertTrue(dpc.__name__.startswith(audit['synthetic_package'] + '.'))
        self.assertEqual(before, {n for n in sys.modules if n.split('.')[0] == 'openhands'})
        self.assertEqual(audit['pins_sha256'], pins_sha)
        self.assertEqual([f['upstream_path'] for f in audit['files']],
                         [ra.ST_GCN, ra.DPC_RNN, ra.POSE_TRANSFORMS])
        _, _, audit2 = ra.load_reference(self.root, pins_sha)
        self.assertNotEqual(audit['synthetic_package'], audit2['synthetic_package'])

    def test_pin_file_hash_required_and_checked(self):
        write_reference(self.root)
        for bad in ('0' * 64, None, 'abc'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ra.load_reference(self.root, bad)
        self.assertEqual(self.executed(), [])

    def test_tampered_source_rejected_before_any_execution(self):
        pins_sha = write_reference(self.root)
        (self.root / ra.POSE_TRANSFORMS).write_text(MALICIOUS)
        with self.assertRaises(ValueError):
            ra.load_reference(self.root, pins_sha)
        # Neither the malicious file nor earlier, correctly pinned files ran.
        self.assertEqual(self.executed(), [])

    def test_wrong_commit_rejected(self):
        pins_sha = write_reference(self.root, commit='0' * 40)
        with self.assertRaises(ValueError):
            ra.load_reference(self.root, pins_sha)
        self.assertEqual(self.executed(), [])

    def test_unsafe_local_paths_rejected(self):
        outside = self.root.parent / 'outside.py'
        outside.write_text(SOURCES[ra.ST_GCN])
        bad_paths = [
            '../outside.py',
            str(outside.resolve()),
            outside.resolve().as_posix(),
            'openhands/./models/ssl/st_gcn.py',
            'openhands//models/ssl/st_gcn.py',
            chr(92).join(ra.ST_GCN.split('/')),
            '',
        ]
        for bad in bad_paths:
            with self.subTest(bad=bad):
                def edit(entries, bad=bad):
                    entries[0]['local_path'] = bad
                    return entries
                pins_sha = write_reference(self.root, edit=edit)
                with self.assertRaises(ValueError):
                    ra.load_reference(self.root, pins_sha)
        self.assertEqual(self.executed(), [])

    def test_duplicate_missing_unexpected_or_inconsistent_entries(self):
        edits = {
            'duplicate_upstream': lambda e: e + [dict(e[0])],
            'duplicate_local': lambda e: [e[0], e[1], dict(e[2], local_path=e[0]['local_path'])],
            'missing': lambda e: e[:2],
            'unexpected': lambda e: e + [dict(e[0], upstream_path='openhands/__init__.py')],
            'size_mismatch': lambda e: [dict(e[0], bytes=e[0]['bytes'] + 1)] + e[1:],
            'url_other_commit': lambda e: [dict(e[0], url=e[0]['url'].replace(ra.RELEASE_COMMIT, '0' * 40))] + e[1:],
        }
        for label, edit in edits.items():
            with self.subTest(label):
                pins_sha = write_reference(self.root, edit=edit)
                with self.assertRaises(ValueError):
                    ra.load_reference(self.root, pins_sha)
        self.assertEqual(self.executed(), [])


GRAPH = torch.arange(3 * 27 * 27, dtype=torch.float32).reshape(3, 27, 27) % 5


class StubEncoder(nn.Module):
    '''Stand-in exposing 170 state keys: graph buffer A + 169 parameters.'''

    def __init__(self):
        super().__init__()
        self.register_buffer('A', GRAPH.clone())
        self.p = nn.ParameterList(nn.Parameter(torch.randn(3)) for _ in range(169))


class StubFinetuner(nn.Module):
    '''Stand-in with the documented DPC_RNN_Finetuner attribute contract.'''

    def __init__(self, num_class=60, pred_steps=2, in_channels=2, hidden_channels=64,
                 hidden_dim=256, dropout=0.5, graph_args=None, edge_importance_weighting=True):
        super().__init__()
        assert graph_args == GRAPH_ARGS
        self.conv_encoder = StubEncoder()
        self.agg = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.final_bn = nn.BatchNorm1d(hidden_dim)
        self.final_fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_class))
        self.conv_encoder.p[0].requires_grad_(False)  # adapter must make it trainable


class StubWithPredictor(StubFinetuner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network_pred = nn.Linear(4, 4)


STUB_DPC = types.SimpleNamespace(DPC_RNN_Finetuner=StubFinetuner)


def checkpoint():
    torch.manual_seed(7)
    enc = {k: v.clone() if k == 'A' else v + 1.0 for k, v in StubEncoder().state_dict().items()}
    gru = {k: torch.full_like(v, 0.01) for k, v in nn.GRU(256, 256, batch_first=True).state_dict().items()}
    return enc, gru


class MakeReferenceModelTests(unittest.TestCase):
    def test_loads_encoder_and_gru_only(self):
        enc, gru = checkpoint()
        model, audit = ra.make_reference_model(STUB_DPC, enc, gru, num_classes=5)
        self.assertFalse(model.training)
        for key, value in enc.items():
            self.assertTrue(torch.equal(model.conv_encoder.state_dict()[key], value), key)
        for key, value in gru.items():
            self.assertTrue(torch.equal(model.agg.state_dict()[key], value), key)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(audit['frozen_by_official_init'], ['conv_encoder.p.0'])
        self.assertEqual(audit['encoder_keys'], 170)
        self.assertEqual(audit['graph_keys_exact_match'], ['A'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261005)
            fresh = StubFinetuner(num_class=5, graph_args=GRAPH_ARGS)
        for key, value in fresh.state_dict().items():
            if key.startswith(('final_fc.', 'final_bn.')):
                self.assertTrue(torch.equal(model.state_dict()[key], value), key)

    def test_caller_rng_and_inputs_untouched(self):
        enc, gru = checkpoint()
        enc_before = {k: v.clone() for k, v in enc.items()}
        torch.manual_seed(1)
        expected = torch.rand(3)
        torch.manual_seed(1)
        ra.make_reference_model(STUB_DPC, enc, gru, num_classes=5)
        self.assertTrue(torch.equal(torch.rand(3), expected))
        for key, value in enc_before.items():
            self.assertTrue(torch.equal(enc[key], value), key)

    def test_train_mode_updates_bn_statistics(self):
        enc, gru = checkpoint()
        model, _ = ra.make_reference_model(STUB_DPC, enc, gru, num_classes=5)
        model.train()
        model.final_bn(torch.randn(8, 256))
        self.assertGreater(int(model.final_bn.num_batches_tracked), 0)

    def test_rejects_bad_states(self):
        cases = {
            'graph_A_differs': lambda e, g: e.update(A=e['A'] + 1),
            'nonfinite_encoder': lambda e, g: e.update({'p.3': torch.tensor([float('nan'), 0.0, 0.0])}),
            'wrong_dtype': lambda e, g: e.update({'p.3': e['p.3'].double()}),
            'wrong_shape': lambda e, g: e.update({'p.3': torch.zeros(4)}),
            'missing_key': lambda e, g: e.pop('p.3'),
            'old_head_key': lambda e, g: e.update({'final_fc.1.weight': torch.zeros(5, 256)}),
            'not_tensor': lambda e, g: e.update({'p.3': [0.0, 0.0, 0.0]}),
            'gru_missing': lambda e, g: g.pop('bias_hh_l0'),
            'gru_nonfinite': lambda e, g: g.update(bias_hh_l0=torch.full_like(g['bias_hh_l0'], float('inf'))),
        }
        for label, corrupt in cases.items():
            with self.subTest(label):
                enc, gru = checkpoint()
                corrupt(enc, gru)
                with self.assertRaises((ValueError, TypeError)):
                    ra.make_reference_model(STUB_DPC, enc, gru, num_classes=5)

    def test_rejects_prediction_network(self):
        enc, gru = checkpoint()
        dpc = types.SimpleNamespace(DPC_RNN_Finetuner=StubWithPredictor)
        with self.assertRaises(ValueError):
            ra.make_reference_model(dpc, enc, gru, num_classes=5)


if __name__ == '__main__':
    unittest.main()
