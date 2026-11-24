"""Independent synthetic-only crash/recovery verification; no project datasets."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest import mock

import torch
from torch import nn

import generic_workflow as gw
import training_engine as te


class StopForCheck(Exception):
    pass


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 5)
        self.bn = nn.BatchNorm1d(5)
        self.drop = nn.Dropout(0.25)
        self.out = nn.Linear(5, 3)

    def forward(self, x):
        return self.out(self.drop(self.bn(self.fc(x.mean(dim=(1, 3, 4))))))


def factory():
    torch.manual_seed(811)
    return Tiny()


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b, (a, b)


def main():
    torch.set_num_threads(2)
    here = Path(__file__).resolve().parent
    root = Path(tempfile.mkdtemp(prefix='synthetic_recovery_', dir=here))
    config_key = gw._MAX_EPOCHS_KEY
    # The public adapter seam is configured by the production caller, not an
    # engine change. Record this explicit test setting if it is not yet applied.
    cfg = dict(schema='generic_cpu_v1', num_classes=3, batch_size=2,
               max_epochs=3, min_epochs=1, patience=3, base_lr=0.01,
               schedule_epochs=3, seed=991, num_samples=4)
    refit_cfg = dict(cfg, num_samples=8)
    gen = torch.Generator().manual_seed(67)
    all_data = dict(x=torch.randn(8, 6, 2, 10, 27, generator=gen),
                    y=torch.arange(8, dtype=torch.int64) % 3,
                    weight=torch.tensor([1, 0, 2, 1, 1, 0, 2, 1], dtype=torch.float64),
                    ids=[f'synthetic-row-{i}' for i in range(8)],
                    origins=[f'synthetic-origin-{i}' for i in range(8)])

    def subset(start, end):
        return {k: v[start:end].clone() if isinstance(v, torch.Tensor) else v[start:end]
                for k, v in all_data.items()}

    train, val = subset(0, 4), subset(4, 8)
    val['primary'] = torch.ones(4, dtype=torch.bool)
    context = {'synthetic_only': True, 'classes': ['zero', 'one', 'two'],
               'inference_shape': [6, 2, 10, 27]}
    original_train_epoch, original_eval, original_writer = te._train_epoch, te._evaluate, gw._epoch_writer
    epoch_seen = 0
    calls = []

    def tracked_epoch(model, optimizer, data, config, epoch, augment):
        nonlocal epoch_seen
        epoch_seen = epoch
        calls.append(('select' if len(data['ids']) == 4 else 'refit', epoch))
        return original_train_epoch(model, optimizer, data, config, epoch, augment)

    def stable_selector(model, data, config):
        metrics = original_eval(model, data, config)
        # Force E*=3 to exercise unfinished refit and terminal recovery. Real
        # synthetic forward/backward/Adam/BN/dropout still execute unchanged.
        metrics.update(primary_correct=epoch_seen, primary_accuracy=epoch_seen / 4,
                       primary_ce=1.0 / epoch_seen)
        return metrics

    def interrupted_writer(stop_phase, stop_epoch):
        def build(phase, output_dir, last_name, binding):
            normal = original_writer(phase, output_dir, last_name, binding)

            def write(checkpoint, record):
                normal(checkpoint, record)
                if phase == stop_phase and record['epoch'] == stop_epoch:
                    raise StopForCheck(f'{phase}:{stop_epoch}')
            return write
        return build

    def run(path, resume=False):
        return gw.run_experiment(factory, train, val, all_data, cfg, refit_cfg,
                                 str(path), context, resume=resume)

    def read_checkpoint(path, name):
        with open(path / (name + '.meta.json'), encoding='utf-8') as f:
            meta = json.load(f)
        return gw._resume_state(str(path / name), True, meta['context_sha256'], meta['run_sha256'])

    def equal_to_baseline(path, baseline):
        for name in (gw.SELECT_LAST, gw.REFIT_LAST):
            actual, expected = read_checkpoint(path, name), read_checkpoint(baseline, name)
            for field in ('model_state', 'optimizer_state', 'completed_epoch', 'stop_reason',
                          'best_epoch', 'best_metrics', 'stale'):
                same(actual[field], expected[field])
            ah = [{k: v for k, v in r.items() if k != 'elapsed_seconds'} for r in actual['history']]
            eh = [{k: v for k, v in r.items() if k != 'elapsed_seconds'} for r in expected['history']]
            same(ah, eh)
        am = gw.load_final_state(factory(), str(path), context)
        bm = gw.load_final_state(factory(), str(baseline), context)
        same(am.state_dict(), bm.state_dict())

    results = []
    log = io.StringIO()
    with contextlib.redirect_stdout(log), mock.patch.object(gw, '_MAX_EPOCHS_KEY', 'max_epochs'), \
            mock.patch.object(te, '_train_epoch', tracked_epoch), mock.patch.object(te, '_evaluate', stable_selector):
        baseline = root / 'continuous_reference'
        run(baseline)
        cases = [
            ('unfinished_select', 'select', 1, False, [('select', 2), ('select', 3), ('refit', 1), ('refit', 2), ('refit', 3)]),
            ('unfinished_refit', 'refit', 1, False, [('refit', 2), ('refit', 3)]),
            ('terminal_select', 'select', 3, False, [('refit', 1), ('refit', 2), ('refit', 3)]),
            ('terminal_refit', 'refit', 3, False, []),
            ('corrupt_select_current', 'select', 2, True, [('select', 2), ('select', 3), ('refit', 1), ('refit', 2), ('refit', 3)]),
            ('corrupt_refit_current', 'refit', 2, True, [('refit', 2), ('refit', 3)]),
        ]
        for name, phase, epoch, corrupt, expected_calls in cases:
            path = root / name
            try:
                with mock.patch.object(gw, '_epoch_writer', interrupted_writer(phase, epoch)):
                    run(path)
            except StopForCheck:
                pass
            else:
                raise AssertionError(f'{name}: did not interrupt')
            if corrupt:
                last = path / (gw.SELECT_LAST if phase == 'select' else gw.REFIT_LAST)
                last.write_bytes(b'intentionally broken synthetic checkpoint payload')
            calls.clear()
            try:
                run(path, resume=True)
                assert calls == expected_calls, (calls, expected_calls)
                equal_to_baseline(path, baseline)
                results.append(dict(case=name, status='PASS', resumed_epoch_calls=calls.copy(),
                                    state_optimizer_history_exact=True))
            except Exception as exc:
                results.append(dict(case=name, status='FAIL', error=repr(exc),
                                    resumed_epoch_calls=calls.copy()))

        path = root / 'both_generations_corrupt'
        try:
            with mock.patch.object(gw, '_epoch_writer', interrupted_writer('select', 2)):
                run(path)
        except StopForCheck:
            pass
        for name in (gw.SELECT_LAST, gw.SELECT_LAST + '.prev'):
            (path / name).write_bytes(b'intentionally broken synthetic payload')
        calls.clear()
        try:
            run(path, resume=True)
        except ValueError as exc:
            results.append(dict(case='both_generations_corrupt', status='PASS' if not calls else 'FAIL',
                                rejected=str(exc), resumed_epoch_calls=calls.copy()))
        else:
            results.append(dict(case='both_generations_corrupt', status='FAIL', error='corruption accepted'))

    report = {'scope': 'synthetic CPU-only independent interruption validation',
              'torch_threads': torch.get_num_threads(), 'production_key_observed': config_key,
              'test_key_configured': 'max_epochs', 'scripted_selection_only': True,
              'real_project_data_used': False, 'output_dir': str(root), 'cases': results,
              'source_sha256': {n: hashlib.sha256((here / n).read_bytes()).hexdigest()
                                for n in ('training_engine.py', 'generic_workflow.py', Path(__file__).name)}}
    (root / 'CHECK.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (root / 'stdout.txt').write_text(log.getvalue(), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if all(r['status'] == 'PASS' for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
