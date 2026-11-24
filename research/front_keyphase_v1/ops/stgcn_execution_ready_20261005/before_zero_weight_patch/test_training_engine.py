'''Unit tests for training_engine using tiny synthetic CPU models.'''
import copy
import math
import random
import unittest
from unittest import mock

import torch
from torch import nn

import training_engine as te

C = 3


class TinyNet(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(6 * 2 * 10 * 27, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, C)

    def forward(self, x):
        return self.fc2(self.drop(self.bn(self.fc1(self.flat(x)))))


class ConstantNet(nn.Module):
    '''Always predicts class 0 with fixed logits [2, 0, 0].'''

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([2.0, 0.0, 0.0]))

    def forward(self, x):
        return self.bias.expand(x.shape[0], C) + 0.0 * x.sum()


def make_model(seed=0):
    torch.manual_seed(seed)
    return TinyNet()


def make_data(n=6, prefix='t', seed=0, primary=None):
    gen = torch.Generator().manual_seed(seed)
    data = {
        'x': torch.randn(n, 6, 2, 10, 27, generator=gen),
        'y': torch.arange(n, dtype=torch.int64) % C,
        'weight': torch.linspace(1.0, 2.0, n, dtype=torch.float64),
        'ids': [f'{prefix}{i}' for i in range(n)],
        'origins': [f'{prefix}-orig{i // 2}' for i in range(n)],
    }
    if primary is not None:
        data['primary'] = torch.tensor(primary, dtype=torch.bool)
    return data


def make_cfg(**overrides):
    cfg = {'num_classes': C, 'batch_size': 2, 'max_epochs': 3, 'min_epochs': 1,
           'patience': 3, 'base_lr': 0.01, 'schedule_epochs': 4, 'seed': 7,
           'num_samples': 4, 'schema': 'generic_cpu_v1'}
    cfg.update(overrides)
    return cfg


def clone_data(data):
    return {k: v.clone() if isinstance(v, torch.Tensor) else list(v) for k, v in data.items()}


def metrics(correct, ce, macro=0.5, count=6):
    return {'primary_count': count, 'primary_correct': correct,
            'primary_accuracy': correct / count, 'primary_macro_recall': macro,
            'primary_ce': ce, 'all_count': count, 'all_correct': correct,
            'all_top1': correct / count}


def strip_timing(history):
    return [{k: v for k, v in r.items() if k != 'elapsed_seconds'} for r in history]


class Interrupt(Exception):
    pass


class EngineTestCase(unittest.TestCase):
    def assertSame(self, a, b, path='root'):
        if isinstance(a, torch.Tensor):
            self.assertIsInstance(b, torch.Tensor, path)
            self.assertEqual((a.dtype, tuple(a.shape)), (b.dtype, tuple(b.shape)), path)
            self.assertTrue(torch.equal(a, b), path)
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b), path)
            for key in a:
                self.assertSame(a[key], b[key], f'{path}.{key}')
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b), path)
            for i, (u, v) in enumerate(zip(a, b)):
                self.assertSame(u, v, f'{path}[{i}]')
        else:
            self.assertEqual(a, b, path)


class ValidationTests(EngineTestCase):
    def test_learning_rate_schedule(self):
        cfg = make_cfg(base_lr=0.1, schedule_epochs=4)
        self.assertAlmostEqual(te.learning_rate(1, cfg), 0.1)
        self.assertAlmostEqual(te.learning_rate(3, cfg), 0.05)
        self.assertLess(te.learning_rate(4, cfg), te.learning_rate(3, cfg))
        for bad in (0, 5, 1.0, True):
            with self.subTest(epoch=bad), self.assertRaises(ValueError):
                te.learning_rate(bad, cfg)

    def test_invalid_configs_rejected(self):
        cases = [
            dict(make_cfg(), extra=1),
            {k: v for k, v in make_cfg().items() if k != 'seed'},
            make_cfg(num_classes=1), make_cfg(num_classes=True),
            make_cfg(batch_size=1), make_cfg(num_samples=5), make_cfg(num_samples=0),
            make_cfg(min_epochs=4), make_cfg(min_epochs=0), make_cfg(schedule_epochs=2),
            make_cfg(patience=0), make_cfg(base_lr=0.0), make_cfg(base_lr=float('nan')),
            make_cfg(base_lr='0.1'), make_cfg(seed=1.5), make_cfg(schema='other'),
        ]
        for cfg in cases:
            with self.subTest(cfg=cfg), self.assertRaises((TypeError, ValueError)):
                te.validate_config(cfg)

    def test_invalid_datasets_rejected(self):
        nan = float('nan')
        cases = {
            'float64 x': lambda d: d.update(x=d['x'].double()),
            'bad shape': lambda d: d.update(x=d['x'][:, :, :, :9]),
            'nan x': lambda d: d['x'][0, 0, 0, 0, 0:1].fill_(nan),
            'label too large': lambda d: d['y'][0:1].fill_(C),
            'negative label': lambda d: d['y'][0:1].fill_(-1),
            'int32 y': lambda d: d.update(y=d['y'].int()),
            'zero weight': lambda d: d['weight'][0:1].fill_(0.0),
            'float32 weight': lambda d: d.update(weight=d['weight'].float()),
            'duplicate ids': lambda d: d['ids'].__setitem__(1, d['ids'][0]),
            'short origins': lambda d: d['origins'].pop(),
            'non-str id': lambda d: d['ids'].__setitem__(0, 0),
            'extra key': lambda d: d.update(extra=1),
            'primary on train': lambda d: d.update(primary=torch.ones(6, dtype=torch.bool)),
            'requires grad': lambda d: d.update(x=d['x'].requires_grad_()),
            'empty': lambda d: d.update(x=d['x'][:0], y=d['y'][:0], weight=d['weight'][:0],
                                        ids=[], origins=[]),
        }
        for name, mutate in cases.items():
            data = make_data()
            mutate(data)
            with self.subTest(name), self.assertRaises((TypeError, ValueError)):
                te.validate_dataset(data, C)
        bad_primary = [make_data(primary=[False] * 6), make_data(primary=[True] * 5), make_data()]
        bad_primary[2]['primary'] = torch.ones(6)
        for data in bad_primary:
            with self.subTest('primary'), self.assertRaises((TypeError, ValueError)):
                te.validate_dataset(data, C, validation=True)
        self.assertEqual(te.validate_dataset(make_data(primary=[True] * 6), C, validation=True), 6)

    def test_overlapping_origins_rejected_before_training(self):
        model = make_model()
        before = copy.deepcopy(model.state_dict())
        train, val = make_data(), make_data(prefix='v', seed=1)
        val['origins'][0] = train['origins'][3]
        with self.assertRaises(ValueError):
            te.run_training(model, train, val, make_cfg())
        self.assertSame(before, model.state_dict())

    def test_mode_contracts(self):
        model, train, val, cfg = make_model(), make_data(), make_data(prefix='v', seed=1), make_cfg()
        bad_calls = [
            {'val': None},
            {'fixed_epochs': 2},
            {'mode': 'refit', 'fixed_epochs': 2},
            {'mode': 'refit', 'val': None},
            {'mode': 'refit', 'val': None, 'fixed_epochs': 4},
            {'mode': 'refit', 'val': None, 'fixed_epochs': 0},
            {'mode': 'other'},
        ]
        for kwargs in bad_calls:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                kwargs = dict(kwargs)
                te.run_training(model, train, kwargs.pop('val', val), cfg, **kwargs)


class TrainingBehaviourTests(EngineTestCase):
    def test_parameters_and_batchnorm_update_with_full_batches(self):
        model = make_model()
        before = copy.deepcopy(model.state_dict())
        train_batches = []
        model.register_forward_hook(
            lambda m, inp, out: train_batches.append(inp[0].shape[0]) if m.training else None)
        cfg = make_cfg(max_epochs=2, num_samples=6)
        result = te.run_training(model, make_data(n=5), make_data(n=5, prefix='v', seed=1), cfg)
        after = model.state_dict()
        for key in ('fc1.weight', 'fc2.bias', 'bn.weight', 'bn.running_mean', 'bn.running_var'):
            self.assertFalse(torch.equal(before[key], after[key]), key)
        self.assertEqual(int(after['bn.num_batches_tracked']), 6)
        self.assertEqual(train_batches, [2] * 6)
        for record in result['history']:
            self.assertEqual((record['updates'], record['train_count']), (3, 6))
            self.assertEqual(len(record['sample_counts']), 5)
            self.assertEqual(sum(record['sample_counts']), 6)
            self.assertTrue(math.isfinite(record['train_ce']))
        self.assertSame(result['last_checkpoint']['model_state'], after)

    def test_source_and_validation_tensors_unchanged(self):
        train = make_data()
        val = make_data(prefix='v', seed=1, primary=[True, False] * 3)
        train_copy, val_copy = clone_data(train), clone_data(val)

        def augment(sample):
            return sample.mul_(3.0)  # in place on the engine-supplied clone

        class Mutating(TinyNet):
            def forward(self, x):
                x.add_(1.0)  # scribbles on its input batch
                return super().forward(x)

        torch.manual_seed(0)
        te.run_training(Mutating(), train, val, make_cfg(max_epochs=2), augment)
        self.assertSame(train_copy, train)
        self.assertSame(val_copy, val)

    def test_evaluate_metrics_and_mode_restore(self):
        cfg = make_cfg()
        val = make_data(n=5, prefix='v', primary=[True, True, False, True, False])
        model = ConstantNet().train()
        m = te.evaluate(model, val, cfg)
        self.assertTrue(model.training)
        # primary labels are 0, 1, 0 and the model always predicts 0
        self.assertEqual((m['primary_count'], m['primary_correct']), (3, 2))
        self.assertAlmostEqual(m['primary_accuracy'], 2 / 3)
        self.assertAlmostEqual(m['primary_macro_recall'], 0.5)
        lse = math.log(math.exp(2.0) + 2.0)
        self.assertAlmostEqual(m['primary_ce'], (2 * (lse - 2.0) + lse) / 3, places=6)
        self.assertEqual((m['all_count'], m['all_correct']), (5, 2))

        net = make_model()
        net.train()
        net.drop.eval()
        before = copy.deepcopy(net.state_dict())
        te.evaluate(net, val, cfg)
        self.assertTrue(net.training and net.bn.training)
        self.assertFalse(net.drop.training)
        self.assertSame(before, net.state_dict())
        net.eval()
        te.evaluate(net, make_data(n=5, prefix='v'), cfg)
        self.assertFalse(any(module.training for module in net.modules()))

    def test_weighted_sampler_reproducible_and_independent_of_augmentation(self):
        train = make_data()
        train['weight'] = torch.tensor([1e-12, 1000.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        cfg = make_cfg(max_epochs=2, num_samples=8)

        def noisy(sample):
            random.random()
            torch.rand(3)
            return sample + 0.01 * torch.randn(sample.shape)

        runs = [te.run_training(make_model(), train, make_data(prefix='v', seed=1), cfg, aug)['history']
                for aug in (None, noisy, None)]
        for epoch in range(2):
            keys = [(r[epoch]['sample_sha256'], r[epoch]['sample_counts']) for r in runs]
            self.assertEqual(keys[0], keys[1])
            self.assertEqual(keys[0], keys[2])
            counts = runs[0][epoch]['sample_counts']
            self.assertEqual(counts[0], 0)
            self.assertEqual(max(counts), counts[1])
        self.assertSame(strip_timing(runs[0]), strip_timing(runs[2]))

    def test_augment_called_once_per_training_draw_only(self):
        train = make_data()
        seen = []

        def augment(sample):
            self.assertEqual((tuple(sample.shape), sample.dtype), (te.SAMPLE_SHAPE, torch.float32))
            matches = [i for i in range(6) if torch.equal(sample, train['x'][i])]
            self.assertEqual(len(matches), 1)
            seen.append(matches[0])
            return sample

        cfg = make_cfg(max_epochs=2, num_samples=6)
        result = te.run_training(make_model(), train, make_data(prefix='v', seed=1), cfg, augment)
        self.assertEqual(len(seen), 12)
        for e, record in enumerate(result['history']):
            drawn = seen[6 * e:6 * e + 6]
            self.assertEqual([drawn.count(i) for i in range(6)], record['sample_counts'])

    def test_bad_augment_output_rejected(self):
        bad_augments = (lambda s: s.double(), lambda s: s[:, :, :5], lambda s: s * float('inf'))
        for bad in bad_augments:
            with self.subTest(), self.assertRaises(ValueError):
                te.run_training(make_model(), make_data(), make_data(prefix='v', seed=1), make_cfg(), bad)

    def test_nonfinite_gradient_aborts_before_step(self):
        model = make_model()
        weight_before = model.fc2.weight.detach().clone()
        model.fc2.weight.register_hook(lambda g: g * float('nan'))
        with self.assertRaises(FloatingPointError):
            te.run_training(model, make_data(), make_data(prefix='v', seed=1), make_cfg())
        self.assertTrue(torch.equal(weight_before, model.fc2.weight.detach()))


class SelectionTests(EngineTestCase):
    def run_scripted(self, scripted, **overrides):
        with mock.patch.object(te, '_evaluate', side_effect=scripted) as fake:
            result = te.run_training(make_model(), make_data(), make_data(prefix='v', seed=1),
                                     make_cfg(**overrides))
        self.assertEqual(fake.call_count, len(result['history']))
        return result

    def test_tie_keeps_earliest_and_patience_stops(self):
        result = self.run_scripted([metrics(3, 1.0)] * 3, patience=1)
        self.assertEqual(len(result['history']), 2)
        self.assertEqual((result['best_epoch'], result['stop_reason']), (1, 'patience'))
        self.assertEqual(result['best_checkpoint']['completed_epoch'], 1)

    def test_correct_count_beats_ce_and_ce_breaks_ties(self):
        scripted = [metrics(3, 0.5), metrics(4, 2.0), metrics(4, 1.5), metrics(4, 1.5)]
        result = self.run_scripted(scripted, max_epochs=4, patience=3)
        self.assertEqual([r['is_best'] for r in result['history']], [True, True, True, False])
        self.assertEqual([r['stale'] for r in result['history']], [0, 0, 0, 1])
        self.assertEqual((result['best_epoch'], result['stop_reason']), (3, 'max_epochs'))

    def test_macro_recall_is_not_a_selector(self):
        result = self.run_scripted([metrics(3, 1.0, macro=0.2), metrics(3, 1.0, macro=0.9)],
                                   max_epochs=2, patience=2)
        self.assertEqual(result['best_epoch'], 1)

    def test_min_epochs_respected(self):
        scripted = [metrics(3, 0.5)] + [metrics(2, 0.5)] * 3
        result = self.run_scripted(scripted, max_epochs=4, min_epochs=3, patience=1)
        self.assertEqual(len(result['history']), 3)
        self.assertEqual((result['best_epoch'], result['stop_reason']), (1, 'patience'))

    def test_budget_limited_when_last_epoch_is_best(self):
        result = self.run_scripted([metrics(1, 1.0), metrics(2, 1.0), metrics(3, 1.0)], patience=1)
        self.assertEqual((result['best_epoch'], result['stop_reason']), (3, 'budget_limited_improving'))

    def test_refit_reuses_original_schedule_without_evaluation(self):
        cfg = make_cfg(max_epochs=3, schedule_epochs=6)
        with mock.patch.object(te, '_evaluate', side_effect=AssertionError('refit evaluated')) as fake:
            result = te.run_training(make_model(), make_data(), None, cfg, mode='refit', fixed_epochs=2)
        self.assertEqual(fake.call_count, 0)
        self.assertEqual(result['stop_reason'], 'fixed_epochs')
        self.assertIsNone(result['best_checkpoint'])
        self.assertIsNone(result['best_epoch'])
        lrs = [r['lr'] for r in result['history']]
        self.assertEqual(lrs, [te.learning_rate(1, cfg), te.learning_rate(2, cfg)])
        self.assertAlmostEqual(lrs[1], 0.01 * 0.5 * (1 + math.cos(math.pi / 6)))
        group = result['last_checkpoint']['optimizer_state']['param_groups'][0]
        self.assertEqual((group['lr'], group['weight_decay']), (lrs[1], 0.0))


class ResumeTests(EngineTestCase):
    def setUp(self):
        self.train = make_data()
        self.val = make_data(prefix='v', seed=1, primary=[True, True, True, True, False, True])
        self.cfg = make_cfg(max_epochs=3, patience=3)

    @staticmethod
    def augment(sample):
        return sample + 0.05 * torch.randn(sample.shape)

    def interrupted(self, after_epoch, **kwargs):
        saved = {}

        def stop(ckpt, record):
            if record['epoch'] == after_epoch:
                saved['ckpt'] = ckpt
                raise Interrupt

        with self.assertRaises(Interrupt):
            te.run_training(make_model(), self.train, kwargs.pop('val', self.val), self.cfg,
                            self.augment, on_epoch=stop, **kwargs)
        return saved['ckpt']

    def test_resume_after_two_epochs_matches_uninterrupted(self):
        for kwargs in ({}, {'mode': 'refit', 'val': None, 'fixed_epochs': 3}):
            with self.subTest(mode=kwargs.get('mode', 'select')):
                run_kwargs = dict(kwargs)
                val = run_kwargs.pop('val', self.val)
                full = te.run_training(make_model(), self.train, val, self.cfg, self.augment, **run_kwargs)
                ckpt = self.interrupted(2, **kwargs)
                model = make_model(seed=123)
                resumed = te.run_training(model, self.train, val, self.cfg, self.augment,
                                          resume=ckpt, **run_kwargs)
                self.assertEqual(len(resumed['history']), 3)
                self.assertSame(strip_timing(full['history']), strip_timing(resumed['history']))
                for key in ('model_state', 'optimizer_state', 'completed_epoch', 'stale',
                            'best_epoch', 'best_metrics', 'stop_reason'):
                    self.assertSame(full['last_checkpoint'][key], resumed['last_checkpoint'][key], key)
                self.assertSame(full['last_checkpoint']['model_state'], model.state_dict())
                self.assertEqual((full['best_epoch'], full['stop_reason']),
                                 (resumed['best_epoch'], resumed['stop_reason']))
                if full['best_checkpoint'] is not None:
                    self.assertSame(full['best_checkpoint']['model_state'],
                                    resumed['best_checkpoint']['model_state'])

    def test_resume_contract_violations_rejected_before_mutation(self):
        ckpt = self.interrupted(1)
        changed_x = clone_data(self.train)
        changed_x['x'][0, 0, 0, 0, 0] += 1.0
        changed_ids = clone_data(self.train)
        changed_ids['ids'][0] = 'renamed'
        changed_origins = clone_data(self.train)
        changed_origins['origins'][0] = 't-elsewhere'
        changed_weight = clone_data(self.train)
        changed_weight['weight'][0] = 5.0
        changed_primary = clone_data(self.val)
        changed_primary['primary'][4] = True
        bad_shape = dict(ckpt['model_state'], **{'fc2.bias': torch.zeros(5)})
        cases = {
            'cfg': {'cfg': make_cfg(max_epochs=3, patience=3, base_lr=0.02)},
            'checkpoint cfg': {'resume': dict(ckpt, cfg=dict(ckpt['cfg'], base_lr=0.5))},
            'train x': {'train': changed_x},
            'train ids': {'train': changed_ids},
            'train origins': {'train': changed_origins},
            'train weight': {'train': changed_weight},
            'val primary': {'val': changed_primary},
            'refit mode': {'mode': 'refit', 'val': None, 'fixed_epochs': 3},
            'best snapshot as resume': {'resume': ckpt['best_snapshot']},
            'stale counter': {'resume': dict(ckpt, stale=5)},
            'completed epoch': {'resume': dict(ckpt, completed_epoch=2)},
            'model shape': {'resume': dict(ckpt, model_state=bad_shape)},
        }
        for name, override in cases.items():
            kwargs = {'train': self.train, 'val': self.val, 'cfg': self.cfg, 'resume': ckpt}
            kwargs.update(override)
            model = make_model(seed=5)
            before = copy.deepcopy(model.state_dict())
            with self.subTest(name), self.assertRaises(ValueError):
                te.run_training(model, kwargs.pop('train'), kwargs.pop('val'), kwargs.pop('cfg'), **kwargs)
            self.assertSame(before, model.state_dict(), name)

    def test_snapshots_are_not_modified_by_later_training(self):
        captured = []

        def keep(ckpt, record):
            captured.append((ckpt, copy.deepcopy(ckpt['model_state']), copy.deepcopy(ckpt['optimizer_state'])))

        model = make_model()
        result = te.run_training(model, self.train, self.val, self.cfg, self.augment, on_epoch=keep)
        self.assertEqual(len(captured), 3)
        for ckpt, state, optimizer_state in captured:
            self.assertSame(state, ckpt['model_state'])
            self.assertSame(optimizer_state, ckpt['optimizer_state'])
        self.assertFalse(torch.equal(captured[0][0]['model_state']['fc1.weight'],
                                     model.state_dict()['fc1.weight']))
        best = result['best_checkpoint']
        self.assertIs(result['last_checkpoint']['best_snapshot'], best)
        self.assertIsNone(best['best_snapshot'])
        self.assertSame(captured[best['completed_epoch'] - 1][1], best['model_state'])


if __name__ == '__main__':
    unittest.main()
