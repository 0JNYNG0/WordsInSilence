"""Independent pure/mocked execution checks; never run a detector/model forward."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
sys.path.insert(0, str(U))
import training_driver as driver
import training_core as core
import execute_parallel as parallel
import execute_parallel_test as scheduling_tests


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixtures():
    out = []
    for label in range(100):
        for signer in (11, 12, 13):
            row = dict(clip_id=f'fixture_{label}_{signer}', label_index=label, signer_id=signer)
            logits = np.full(100, -2., np.float64); logits[label] = 2.
            out.append(core.prediction(row, logits))
    return sorted(out, key=lambda r: r['clip_id'])


class IndependentExecutionTests(unittest.TestCase):
    def test_ready_gate_rejects_each_missing_evidence_without_source_access(self):
        ready = dict(status='PASS', training_allowed=True, blockers=[],
                     preparation_context_sha256='fixture_context', manifest_sha256=driver.MANIFEST_SHA,
                     independent_input_validation_passed=True, motion_preservation_review_passed=True,
                     fresh_sensor_detector_state_verified=True, roi_hidden_coordinate_invariance_passed=True)
        bad_values = {'status': 'PENDING', 'training_allowed': False, 'blockers': ['unreviewed'],
                      'preparation_context_sha256': 'wrong', 'manifest_sha256': 'wrong',
                      'independent_input_validation_passed': False, 'motion_preservation_review_passed': False,
                      'fresh_sensor_detector_state_verified': False, 'roi_hidden_coordinate_invariance_passed': False}
        args = SimpleNamespace(readiness=str(U/'fixture_ready.json'), prep_context=str(U/'fixture_context.json'),
                               records_dir=str(U/'fixture_records'))
        for key, bad in bad_values.items():
            with self.subTest(gate=key):
                altered = {**ready, key: bad}
                def fake_read(path):
                    if Path(path).name == 'AUTHORIZATION.json': return {'approved': True}
                    if Path(path).name == 'fixture_ready.json': return altered
                    raise RuntimeError('Gate leaked into real source access: ' + str(path))
                with patch.object(driver, 'CONTEXT', U/'never_created_fixture_context.json'), \
                     patch.object(driver, 'read', side_effect=fake_read), \
                     patch.object(driver, 'sha', return_value='fixture_context'), \
                     patch.object(driver, 'atomic_json', side_effect=RuntimeError('Unexpected write')), \
                     patch.object(driver, 'MaskedSignTCN', side_effect=RuntimeError('Unexpected model construction')):
                    with self.assertRaises(AssertionError): driver.freeze(args)

    def test_extreme_logits_and_failed_rows_against_independent_arithmetic(self):
        rng = np.random.default_rng(413)
        results = []
        independent_correct = independent_top5 = 0
        expected_ce = []
        for label in range(100):
            for signer in (11, 12, 13):
                row = dict(clip_id=f'fixture_{label}_{signer}', label_index=label, signer_id=signer)
                if (label + signer) % 17 == 0:
                    result = core.prediction(row, reason='synthetic_unavailable')
                    expected_ce.append(float(np.log(100)))
                    self.assertIsNone(result['logits']); self.assertFalse(result['correct'])
                    self.assertFalse(result['top5_correct'])
                else:
                    logits = rng.normal(size=100) * 1000
                    order = sorted(range(100), key=lambda index: (-float(logits[index]), index))
                    maximum = max(float(x) for x in logits)
                    ce = float(np.log(np.exp(logits - maximum).sum()) - (logits[label] - maximum))
                    result = core.prediction(row, logits)
                    self.assertEqual(result['prediction'], order[0]); self.assertEqual(result['top5'], order[:5])
                    self.assertAlmostEqual(result['cross_entropy'], ce, places=10)
                    independent_correct += order[0] == label
                    independent_top5 += label in order[:5]
                    expected_ce.append(ce)
                results.append(result)
        metric = core.metrics(results)
        self.assertEqual(metric['total'], 300)
        self.assertEqual(metric['top1_correct'], independent_correct)
        self.assertEqual(metric['top5_correct'], independent_top5)
        self.assertAlmostEqual(metric['mean_ce'], np.mean(expected_ce), places=10)

    def test_completed_run_reuses_scores_without_model_construction(self):
        predictions = fixtures()
        groups = {r['clip_id']: 'other' for r in predictions}
        metrics = core.detailed_metrics(predictions, groups)
        result = dict(status='COMPLETE_DEVELOPMENT_ONLY', context_sha256='fixture_hash',
                      checkpoint_sha256='fixture_hash', selected_ce=metrics['mean_ce'],
                      predictions={v: predictions for v in core.VIEWS}, metrics={v: metrics for v in core.VIEWS})
        config = {'missingness_groups': {v: groups for v in core.VIEWS}}
        def fake_exists(path): return str(path).endswith('_RESULT.json')
        with patch.object(Path, 'exists', fake_exists), patch.object(driver, 'read', return_value=result), \
             patch.object(driver, 'sha', return_value='fixture_hash'), patch.object(driver, 'event'), \
             patch.object(driver, 'MaskedSignTCN', side_effect=RuntimeError('Completed cache created model')), \
             patch.object(driver, 'predict', side_effect=RuntimeError('Completed cache re-inferred')):
            self.assertIs(driver.run_arm(config, [], predictions, 'A', 20261013), result)

    def test_terminal_epoch_checkpoint_restores_rng_and_publishes_saved_best_without_forward(self):
        import torch
        seed = 20261013
        driver.set_seed(seed)
        model = driver.MaskedSignTCN(**core.MODEL)
        initial = copy.deepcopy(model.state_dict())
        initial_hash = driver.state_hash(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=core.TRAIN['lr'], weight_decay=core.TRAIN['weight_decay'])
        torch.manual_seed(999); stored_rng = torch.get_rng_state()
        predictions = fixtures(); groups = {r['clip_id']: 'other' for r in predictions}
        metric = core.detailed_metrics(predictions, groups)
        best = dict(epoch=60, step=60*29, ce=metric['mean_ce'], model=initial,
                    predictions={v: predictions for v in core.VIEWS}, metrics={v: metric for v in core.VIEWS})
        last = dict(context_sha256='fixture_hash', arm='B', seed=seed, epoch=80, step=80*29,
                    initial_state_sha256=initial_hash, model=initial, optimizer=optimizer.state_dict(),
                    torch_rng=stored_rng, best=best, history=[], elapsed_seconds=1.)
        config = dict(labels=[f'label{i}' for i in range(100)], word_codes=[f'word{i}' for i in range(100)],
                      missingness_groups={v: groups for v in core.VIEWS})
        writes = []
        def fake_exists(path): return str(path).endswith('_last.pt')
        def save(path, value, *args, **kwargs): writes.append((Path(path).name, value))
        with patch.object(Path, 'exists', fake_exists), \
             patch.object(driver, 'sha', return_value='fixture_hash'), \
             patch.object(torch, 'load', return_value=last), \
             patch.object(driver, 'atomic_torch', side_effect=save), \
             patch.object(driver, 'atomic_json', side_effect=save), \
             patch.object(driver, 'event'), \
             patch.object(driver, 'predict', side_effect=RuntimeError('Unexpected inference')), \
             patch.object(driver.MaskedSignTCN, 'forward', side_effect=RuntimeError('Unexpected model forward')), \
             patch.object(torch.optim.AdamW, 'step', side_effect=RuntimeError('Unexpected optimizer step')):
            result = driver.run_arm(config, [], predictions, 'B', seed)
            self.assertEqual(result['selected_epoch'], 60)
            self.assertEqual(result['epochs_completed'], 80)
            self.assertFalse(result['selected_scores_reinferred'])
            self.assertEqual(result['predictions'], best['predictions'])
            self.assertTrue(torch.equal(torch.get_rng_state(), stored_rng))
        self.assertEqual([name for name, _ in writes], ['training_B_20261013_best.pt', 'training_B_20261013_RESULT.json'])

    def test_full_manifest_four_epoch_schedule_exposure_invariants(self):
        rows = json.loads((U.parent/'word100_training_20261013/MANIFEST.json').read_text(encoding='utf-8'))
        ids = [r['clip_id'] for r in rows if r['role']=='train_candidate']
        self.assertEqual(len(set(ids)), 899)
        for cid in ids:
            for start in range(1, 81, 4):
                values = [core.schedule(cid, epoch, 'C') for epoch in range(start, start+4)]
                self.assertEqual(set(values), {(s,t) for s in ('masked','sensor') for t in ('x_whole','x_context')})
                self.assertEqual(values[0][0], values[1][0])
                self.assertEqual(values[2][0], values[3][0])
                self.assertNotEqual(values[0][0], values[2][0])
                for epoch, (_, temporal) in zip(range(start, start+4), values):
                    self.assertEqual(core.schedule(cid,epoch,'A'), ('native',temporal))
                    self.assertEqual(core.schedule(cid,epoch,'B'), ('masked',temporal))


def main():
    output = U/'EXECUTION_CONTRACT_INDEPENDENT.json'
    assert not output.exists(), 'Preserve previous verification output'
    execution = json.loads((U/'PARALLEL_EXECUTION_CONTEXT.json').read_text(encoding='utf-8'))
    prior = json.loads((U/'training_SELFTEST_RESULT.json').read_text(encoding='utf-8'))
    assert prior['status']=='PASS' and prior['tests']==10
    pins = {**execution['pins'], **prior['pins']}
    for path, digest in pins.items(): assert sha(path)==digest, path
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(scheduling_tests.ParallelDispatchTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentExecutionTests))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    for path, digest in pins.items(): assert sha(path)==digest, path
    report = dict(status='PASS_PURE_AND_MOCKED_CONTRACTS' if result.wasSuccessful() else 'FAILED',
                  tests=result.testsRun, ready_negative_gate_cases=9,
                  errors=[(str(test),err) for test,err in result.errors],
                  failures=[(str(test),err) for test,err in result.failures],
                  log=stream.getvalue(), unchanged_pins=pins, verifier_sha256=sha(__file__),
                  actual_process_pool_executed=False, actual_training=False,
                  optimizer_steps=0, model_forward_calls=0, detector_runs=0, dev_inference=0,
                  runtime_limit='Scheduling tests use ThreadPoolExecutor; actual process/native detector behavior remains a separate execution observation.',
                  checkpoint_test='Terminal committed epoch only; model construction/optimizer state restore in memory, no forward or parameter update.')
    with output.open('x', encoding='utf-8') as file: json.dump(report,file,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('log','unchanged_pins')},ensure_ascii=False))
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__=='__main__': main()
