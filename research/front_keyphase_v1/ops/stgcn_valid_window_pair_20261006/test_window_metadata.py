'''test_window_metadata.py: fast synthetic checks for explicit window_valid metadata.'''
import unittest

import numpy as np
import torch
from torch import nn

import generic_workflow as gw
import training_engine as te
import window_bridge as wb

SHAPE = te.SAMPLE_SHAPE
_MISSING = object()


def make_cfg(**over):
    cfg = dict(num_classes=2, batch_size=2, max_epochs=1, min_epochs=1, patience=1,
               base_lr=0.01, schedule_epochs=1, seed=7, num_samples=4, schema=te.SCHEMA)
    cfg.update(over)
    return cfg


def pattern_mask(offset, n):
    rows = []
    for i in range(n):
        code = (offset + i) % 63 + 1  # nonzero 6-bit code: every row has a True
        rows.append([bool((code >> b) & 1) for b in range(6)])
    return torch.tensor(rows, dtype=torch.bool)


def make_data(n, prefix, mask=True, offset=0):
    gen = torch.Generator().manual_seed(1000 + offset)
    x = torch.randn((n,) + SHAPE, generator=gen)
    for i in range(n):
        x[i, 0, 0, 0, 0] = float(offset + i)  # row marker read back by SpyToy
    data = {'x': x, 'y': torch.tensor([i % 2 for i in range(n)], dtype=torch.int64),
            'weight': torch.ones(n, dtype=torch.float64),
            'ids': [f'{prefix}{i}' for i in range(n)],
            'origins': [f'{prefix}o{i}' for i in range(n)]}
    if mask:
        data['window_valid'] = pattern_mask(offset, n)
    return data


def strip_mask(data):
    return {k: v for k, v in data.items() if k != 'window_valid'}


def augment(sample):
    assert tuple(sample.shape) == SHAPE  # only x samples ever reach augmentation
    out = sample.clone()
    out[1:] += 0.01 * torch.randn(out[1:].shape)  # window 0 keeps the row marker
    return out


class MaskToy(nn.Module):
    '''Compatible model: an all-true mask reproduces the legacy computation exactly.'''

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(int(np.prod(SHAPE)), 2)

    def forward(self, x, window_valid=None):
        if window_valid is not None:
            x = x * window_valid.to(x.dtype)[:, :, None, None, None]
        return self.fc(x.flatten(1))


class SpyToy(MaskToy):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, window_valid=_MISSING):
        rows = [int(v) for v in x[:, 0, 0, 0, 0].tolist()]
        passed = window_valid is not _MISSING
        mask = window_valid.clone() if passed else None
        self.calls.append((self.training, rows, mask, passed))
        return super().forward(x, mask)


class WindowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_encoder = nn.Linear(2 * 10 * 27, 8)
        self.agg = nn.Identity()
        self.final_bn = nn.BatchNorm1d(8)
        self.final_fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.final_fc(self.final_bn(self.agg(self.conv_encoder(x.flatten(2)).mean(1))))


def masked_forward(conv_encoder, agg, final_bn, final_fc, x, window_valid):
    h = conv_encoder(x.flatten(2))  # [B,6,8]
    w = window_valid.to(h.dtype).unsqueeze(-1)
    return final_fc(final_bn(agg((h * w).sum(1) / w.sum(1))))


def snapshot(model):
    return (list(model.state_dict()), [n for n, _ in model.named_modules()],
            [id(p) for p in model.parameters()], [m.training for m in model.modules()])


class MetadataValidationTest(unittest.TestCase):
    def test_dataset_mask_validation(self):
        data = make_data(4, 't')
        self.assertEqual(te.validate_dataset(data, 2), 4)
        self.assertEqual(te.validate_dataset(make_data(3, 'v', offset=10), 2, validation=True), 3)
        self.assertEqual(te.validate_dataset(strip_mask(data), 2), 4)
        empty_row = torch.ones(4, 6, dtype=torch.bool)
        empty_row[2] = False
        cases = [(data['window_valid'].to(torch.uint8), TypeError),
                 (data['window_valid'].tolist(), TypeError),
                 (torch.ones(4, 5, dtype=torch.bool), ValueError),
                 (torch.ones(3, 6, dtype=torch.bool), ValueError),
                 (torch.ones(4, 6, 1, dtype=torch.bool), ValueError),
                 (empty_row, ValueError)]
        for value, exc in cases:
            with self.assertRaises(exc):
                te.validate_dataset(dict(data, window_valid=value), 2)

    def test_select_requires_consistent_presence(self):
        train, val = make_data(4, 't'), make_data(3, 'v', offset=10)
        with self.assertRaisesRegex(ValueError, 'window_valid'):
            te.run_training(MaskToy(), train, strip_mask(val), make_cfg(), mode='select')


class LegacyEquivalenceTest(unittest.TestCase):
    def test_score_all_true_matches_legacy_exactly(self):
        torch.manual_seed(1)
        model = MaskToy()
        model.train()
        x = make_data(5, 's', mask=False)['x']
        legacy = gw.score_batches(model, x, 2)
        masked = gw.score_batches(model, x, 2, window_valid=torch.ones(5, 6, dtype=torch.bool))
        self.assertEqual(legacy.dtype, np.float32)
        self.assertEqual(legacy.shape, (5, 2))
        np.testing.assert_array_equal(legacy, masked)
        self.assertTrue(model.training)

    def test_score_slices_metadata_and_keyword_iff_present(self):
        x = make_data(5, 's', mask=False)['x']
        mask = pattern_mask(0, 5)
        model = SpyToy()
        gw.score_batches(model, x, 2, window_valid=mask)
        self.assertEqual([c[1] for c in model.calls], [[0, 1], [2, 3], [4]])
        for _, rows, got, passed in model.calls:
            self.assertTrue(passed)
            self.assertTrue(torch.equal(got, mask[rows]))
        model.calls.clear()
        gw.score_batches(model, x, 2)
        self.assertTrue(model.calls and all(not c[3] for c in model.calls))

    def test_score_rejects_bad_metadata(self):
        x = make_data(5, 's', mask=False)['x']
        model = MaskToy()
        empty = torch.ones(5, 6, dtype=torch.bool)
        empty[4] = False
        with self.assertRaises(ValueError):
            gw.score_batches(model, x, 2, window_valid=torch.ones(4, 6, dtype=torch.bool))
        with self.assertRaises(ValueError):
            gw.score_batches(model, x, 2, window_valid=empty)
        with self.assertRaises(TypeError):
            gw.score_batches(model, x, 2, window_valid=torch.ones(5, 6, dtype=torch.uint8))

    def test_all_true_training_matches_legacy(self):
        legacy = make_data(4, 't', mask=False)
        masked = dict(legacy, window_valid=torch.ones(4, 6, dtype=torch.bool))
        cfg = make_cfg(max_epochs=2, schedule_epochs=2)
        results = []
        for data in (legacy, masked):
            torch.manual_seed(3)
            model = MaskToy()
            out = te.run_training(model, data, None, cfg, augment=augment, mode='refit', fixed_epochs=2)
            results.append((model.state_dict(), [r['sample_sha256'] for r in out['history']]))
        self.assertEqual(results[0][1], results[1][1])
        for key, value in results[0][0].items():
            self.assertTrue(torch.equal(value, results[1][0][key]))


class RowCorrespondenceTest(unittest.TestCase):
    def test_train_and_eval_rows_match_metadata(self):
        train, val = make_data(4, 't'), make_data(3, 'v', offset=10)
        cfg = make_cfg()
        torch.manual_seed(0)
        model = SpyToy()
        out = te.run_training(model, train, val, cfg, augment=augment, mode='select')
        expected = {i: train['window_valid'][i] for i in range(4)}
        expected.update({10 + i: val['window_valid'][i] for i in range(3)})
        for _, rows, mask, passed in model.calls:
            self.assertTrue(passed)
            self.assertEqual(tuple(mask.shape), (len(rows), 6))
            for row, got in zip(rows, mask):
                self.assertTrue(torch.equal(got, expected[row]))
        sampler = torch.Generator().manual_seed(te.epoch_seed(cfg['seed'], 1))
        weight = train['weight']
        draws = torch.multinomial(weight / weight.sum(), cfg['num_samples'],
                                  replacement=True, generator=sampler).tolist()
        train_calls = [c for c in model.calls if c[0]]
        eval_calls = [c for c in model.calls if not c[0]]
        self.assertEqual([r for c in train_calls for r in c[1]], draws)
        self.assertEqual(out['history'][0]['sample_counts'], [draws.count(i) for i in range(4)])
        self.assertEqual([len(c[1]) for c in eval_calls], [2, 1])  # singleton final eval batch
        self.assertEqual([r for c in eval_calls for r in c[1]], [10, 11, 12])

    def test_legacy_dataset_calls_without_keyword(self):
        model = SpyToy()
        te.run_training(model, make_data(4, 't', mask=False), make_data(3, 'v', mask=False, offset=10),
                        make_cfg(), mode='select')
        self.assertTrue(model.calls and all(not c[3] for c in model.calls))


class SignatureAndSplitTest(unittest.TestCase):
    def setUp(self):
        self.data = make_data(4, 't')
        self.flipped = dict(self.data, window_valid=self.data['window_valid'].clone())
        self.flipped['window_valid'][0] = ~self.flipped['window_valid'][0]

    def test_signature_and_digest_bind_metadata(self):
        sigs = {te._dataset_signature(d) for d in (strip_mask(self.data), self.data, self.flipped)}
        self.assertEqual(len(sigs), 3)
        self.assertNotEqual(gw._dataset_digest(self.data), gw._dataset_digest(self.flipped))

    def test_resume_rejects_changed_metadata(self):
        cfg = make_cfg(max_epochs=2, schedule_epochs=2)
        captured = {}

        def on_epoch(checkpoint, record):
            if checkpoint['completed_epoch'] == 1:
                captured['ck'] = checkpoint

        torch.manual_seed(0)
        te.run_training(MaskToy(), self.data, None, cfg, mode='refit', fixed_epochs=2, on_epoch=on_epoch)
        for other in (self.flipped, strip_mask(self.data)):
            with self.assertRaisesRegex(ValueError, 'train_sha256'):
                te.run_training(MaskToy(), other, None, cfg, mode='refit', fixed_epochs=2,
                                resume=captured['ck'])
        out = te.run_training(MaskToy(), self.data, None, cfg, mode='refit', fixed_epochs=2,
                              resume=captured['ck'])
        self.assertEqual(out['last_checkpoint']['completed_epoch'], 2)

    def test_check_splits(self):
        full = make_data(4, 'r')

        def subset(rows):
            return {'x': full['x'][rows].clone(), 'y': full['y'][rows].clone(),
                    'weight': full['weight'][rows].clone(),
                    'ids': [full['ids'][i] for i in rows],
                    'origins': [full['origins'][i] for i in rows],
                    'window_valid': full['window_valid'][rows].clone()}

        train, val = subset([0, 1]), subset([2, 3])
        gw._check_splits(train, val, full)
        gw._check_splits(strip_mask(train), strip_mask(val), strip_mask(full))
        bad = dict(val, window_valid=val['window_valid'].clone())
        bad['window_valid'][0] = ~bad['window_valid'][0]
        with self.assertRaisesRegex(ValueError, 'differs from all_train'):
            gw._check_splits(train, bad, full)
        with self.assertRaisesRegex(ValueError, 'presence'):
            gw._check_splits(strip_mask(train), val, full)
        with self.assertRaisesRegex(ValueError, 'presence'):
            gw._check_splits(train, val, strip_mask(full))


class BridgeTest(unittest.TestCase):
    def test_indices_masks(self):
        full = list(range(60))
        partial = [-1] * 19 + [5] + list(range(40))
        tail_pad = list(range(30)) + [-1] * 30
        self.assertEqual(wb.window_valid_from_indices(full).tolist(), [True] * 6)
        self.assertEqual(wb.window_valid_from_indices(partial).tolist(), [False] + [True] * 5)
        self.assertEqual(wb.window_valid_from_indices(tail_pad).tolist(), [True] * 3 + [False] * 3)
        self.assertEqual(wb.window_valid_from_indices([0] * 60).tolist(), [True] * 6)  # index 0 is real
        batch = wb.window_valid_from_indices(np.array([full, partial, tail_pad], dtype=np.int64))
        self.assertEqual((tuple(batch.shape), batch.dtype, batch.device.type), ((3, 6), torch.bool, 'cpu'))
        from_tensor = wb.window_valid_from_indices(torch.tensor([partial], dtype=torch.int32))
        self.assertEqual(from_tensor.tolist(), [[False] + [True] * 5])
        with self.assertRaises(ValueError):
            wb.window_valid_from_indices([-1] * 60)
        with self.assertRaises(ValueError):
            wb.window_valid_from_indices([full, [-1] * 60])

    def test_indices_rejections(self):
        for bad in ([True] * 60, [0.0] * 60, [True] + [0] * 59, torch.zeros(60),
                    torch.zeros(60, dtype=torch.bool), np.zeros(60, dtype=bool),
                    np.zeros(60, dtype=np.float32)):
            with self.assertRaises(TypeError):
                wb.window_valid_from_indices(bad)
        for bad in ([-2] + [0] * 59, [0] * 59, [[[0] * 60]], np.zeros((0, 60), dtype=np.int64)):
            with self.assertRaises(ValueError):
                wb.window_valid_from_indices(bad)

    def test_indices_not_mutated(self):
        lst = [-1] * 10 + list(range(50))
        arr = np.array(lst, dtype=np.int64)
        ten = torch.tensor([lst, lst], dtype=torch.int32)
        copies = (list(lst), arr.copy(), ten.clone())
        out_l = wb.window_valid_from_indices(lst)
        out_a = wb.window_valid_from_indices(arr)
        out_t = wb.window_valid_from_indices(ten)
        out_a[:] = False
        self.assertEqual(lst, copies[0])
        np.testing.assert_array_equal(arr, copies[1])
        self.assertTrue(torch.equal(ten, copies[2]))
        self.assertEqual(out_l.tolist(), [False] + [True] * 5)
        self.assertEqual(tuple(out_t.shape), (2, 6))

    def test_bind_preserves_state_and_flags(self):
        torch.manual_seed(0)
        model = WindowNet()
        model.final_bn.eval()  # mixed flags must survive binding
        before = snapshot(model)
        self.assertIs(wb.bind_window_forward(model, masked_forward), model)
        self.assertEqual(snapshot(model), before)
        self.assertIs(wb.bind_window_forward(model, masked_forward), model)  # idempotent
        self.assertEqual(snapshot(model), before)
        with self.assertRaises(RuntimeError):
            wb.bind_window_forward(model, lambda *args: None)
        with self.assertRaises(TypeError):
            model(torch.zeros(2, *SHAPE))  # mask is required once bound
        fresh = WindowNet()
        fresh.load_state_dict(model.state_dict(), strict=True)
        wb.bind_window_forward(fresh, masked_forward)
        model.eval()
        fresh.eval()
        x, mask = make_data(3, 's', mask=False)['x'], pattern_mask(0, 3)
        np.testing.assert_array_equal(gw.score_batches(model, x, 2, window_valid=mask),
                                      gw.score_batches(fresh, x, 2, window_valid=mask))

    def test_bind_masks_and_zero_coordinates(self):
        torch.manual_seed(0)
        model = wb.bind_window_forward(WindowNet(), masked_forward)
        model.eval()
        x = torch.zeros(2, *SHAPE)  # genuine all-zero coordinates, real frames
        moved = x.clone()
        moved[:, 0] = 1.0
        real = wb.window_valid_from_indices([list(range(60))] * 2)
        padded = wb.window_valid_from_indices([[-1] * 10 + list(range(50))] * 2)
        self.assertTrue(bool(real.all()))
        with torch.no_grad():
            self.assertFalse(torch.equal(model(x, window_valid=real), model(moved, window_valid=real)))
            self.assertTrue(torch.equal(model(x, window_valid=padded), model(moved, window_valid=padded)))
            with self.assertRaises(ValueError):
                model(x, window_valid=torch.ones(3, 6, dtype=torch.bool))

    def test_bind_rejects_bad_inputs(self):
        with self.assertRaises(TypeError):
            wb.bind_window_forward(WindowNet(), nn.Linear(1, 1))
        with self.assertRaises(TypeError):
            wb.bind_window_forward(object(), masked_forward)
        with self.assertRaises(ValueError):
            wb.bind_window_forward(nn.Sequential(nn.Linear(1, 1)), masked_forward)
        foreign = WindowNet()
        foreign.forward = lambda x: x
        with self.assertRaises(RuntimeError):
            wb.bind_window_forward(foreign, masked_forward)

    def test_bound_model_with_engine(self):
        torch.manual_seed(0)
        model = wb.bind_window_forward(WindowNet(), masked_forward)
        data = make_data(4, 't')
        out = te.run_training(model, data, None, make_cfg(), mode='refit', fixed_epochs=1)
        self.assertEqual(out['stop_reason'], 'fixed_epochs')
        with self.assertRaises(TypeError):
            te.run_training(model, strip_mask(data), None, make_cfg(), mode='refit', fixed_epochs=1)


if __name__ == '__main__':
    unittest.main()
