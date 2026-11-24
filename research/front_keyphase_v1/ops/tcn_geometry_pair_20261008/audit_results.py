"""Independent arithmetic on saved TCN geometry scores; never performs inference.

This auditor imports neither the implementation nor the project's evaluators.
Run --self-test on synthetic scores, or run after the two model scores exist.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUN = ROOT / 'runs/tcn_geometry_pair_v1'
OLD = ROOT / 'runs/stgcn_localshape_v1'
MANIFEST = HERE.parent / 'stgcn_transfer_compare_20261005/INPUT_MANIFEST.json'
OLD_ROWS_SHA = 'e8348698a701ca92d8b09178755f2dc85b8529e86c0e17d9bf9474a3bcb47794'
OLD_SCORES_SHA = '62e5ad031d5ec42506294feb7281d453ac2ed6a71a971e3d06964c2976a13b7e'
MANIFEST_SHA = '84f273120b322bde05e945d4e5ee2d6c7ff2314c8d4b8cc889b85a0c000fd91f'
BASELINE_SHA = '83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91'
TARGET_LABELS = frozenset(['감사합니다', '나', '모르다', '알다', '좋다', '감', '공부', '인사', '자다',
                           '죄송하다', '맥주', '먹다', '물', '너', '빵', '사랑', '싫다', '운동'])
SCORE_KEYS = {'control': 'control_scores', 'treatment': 'treatment_scores', 'baseline': 'baseline_scores'}
SCOPES = ('full93', 'targetGT_full93', 'restricted18_separate_task')


def require(value, message):
    if not bool(value):
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def exact(a, b, description):
    require(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), description)


def assert_tree_equal(a, b, path='root'):
    """Compare semantic JSON trees, using tight tolerance only for finite floats."""
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), path + ': keys')
        for key in a:
            assert_tree_equal(a[key], b[key], path + '/' + str(key))
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), path + ': list length')
        for i, (x, y) in enumerate(zip(a, b)):
            assert_tree_equal(x, y, path + '/' + str(i))
    elif isinstance(a, float):
        require(isinstance(b, (float, int)) and not isinstance(b, bool)
                and np.isfinite(a) and np.isfinite(b)
                and abs(a - b) <= 1e-12, path + ': float')
    else:
        require(type(a) is type(b) and a == b, path + ': value')


def make_subsets(rows):
    """Membership depends only on frozen row identity, never model scores."""
    result = {}
    for group in ('annotation_dev', 'whole_dev', 'public', 'mp', 'ksl36'):
        ids = [i for i, row in enumerate(rows) if row['group'] == group]
        result['group=' + group] = ids
        if group in ('public', 'mp'):
            for condition in sorted({rows[i]['condition'] for i in ids}):
                result['group=' + group + '|condition=' + condition] = [
                    i for i in ids if rows[i]['condition'] == condition]
    ksl = result['group=ksl36']
    result['group=ksl36|primary_unexposed'] = [i for i in ksl if rows[i]['exposed'] is False]
    result['group=ksl36|exposed'] = [i for i in ksl if rows[i]['exposed'] is True]
    for label in ('공부', '먹다'):
        result['group=ksl36|primary_unexposed|label=' + label] = [
            i for i in result['group=ksl36|primary_unexposed'] if rows[i]['label'] == label]
    return result


def measure(prediction, y, rows, indices, classes):
    per_word = {}
    per_signer = {}
    labels = Counter(classes[int(y[i])] for i in indices)
    for label in sorted(labels):
        n = labels[label]
        correct = sum(int(prediction[i]) == int(y[i]) for i in indices if classes[int(y[i])] == label)
        per_word[label] = {'n': n, 'correct': correct, 'recall': correct / n}
    for signer in sorted({rows[i]['signer_id'] for i in indices if rows[i]['signer_id'] is not None}):
        take = [i for i in indices if rows[i]['signer_id'] == signer]
        correct = sum(int(prediction[i]) == int(y[i]) for i in take)
        per_signer[str(signer)] = {'n': len(take), 'correct': correct, 'accuracy': correct / len(take)}
    correct = sum(int(prediction[i]) == int(y[i]) for i in indices)
    macro = (sum((Fraction(v['correct'], v['n']) for v in per_word.values()), Fraction()) / len(per_word)
             if per_word else None)
    return {'n': len(indices), 'correct': correct,
            'accuracy': correct / len(indices) if indices else None,
            'macro_recall': float(macro) if macro is not None else None,
            'macro_recall_exact': {'numerator': macro.numerator, 'denominator': macro.denominator}
            if macro is not None else None,
            'per_word': per_word, 'per_signer': per_signer}


def compare(after, before, y, rows, indices, classes):
    gains, regressions, changed = [], [], []
    for i in indices:
        old, new, truth = int(before[i]), int(after[i]), int(y[i])
        row = {'index': i, 'id': rows[i]['id'], 'group': rows[i]['group'],
               'condition': rows[i]['condition'], 'label': rows[i]['label'],
               'before': classes[old], 'after': classes[new]}
        if old != new:
            changed.append(row)
        if old != truth and new == truth:
            gains.append(row)
        if old == truth and new != truth:
            regressions.append(row)
    return {'delta': len(gains) - len(regressions), 'gain_count': len(gains),
            'regression_count': len(regressions), 'changed_count': len(changed),
            'gains': gains, 'regressions': regressions, 'changed': changed}


def recount(arrays, rows, classes):
    y = arrays['y']
    target = np.array([i for i, label in enumerate(classes) if label in TARGET_LABELS], dtype=np.int64)
    require(len(target) == 18, 'Target label count')
    full = {arm: arrays[key].argmax(axis=1) for arm, key in SCORE_KEYS.items()}
    restricted = {arm: target[arrays[key][:, target].argmax(axis=1)] for arm, key in SCORE_KEYS.items()}
    target_rows = {i for i, c in enumerate(y) if int(c) in set(target.tolist())}
    groups = {}
    for key, indices in make_subsets(rows).items():
        scopes = {}
        for scope in SCOPES:
            selected = indices if scope == 'full93' else [i for i in indices if i in target_rows]
            predictions = restricted if scope == 'restricted18_separate_task' else full
            scopes[scope] = {
                'models': {arm: measure(pred, y, rows, selected, classes) for arm, pred in predictions.items()},
                'comparisons': {
                    'method': compare(predictions['treatment'], predictions['control'], y, rows, selected, classes),
                    'project': compare(predictions['treatment'], predictions['baseline'], y, rows, selected, classes)}}
        groups[key] = scopes
    return groups


def decision_values(groups):
    native = 'group=annotation_dev'
    ksl = 'group=ksl36|primary_unexposed'
    def metric(group, arm='treatment'):
        return groups[group]['full93']['models'][arm]
    def count(group, arm='treatment'):
        return metric(group, arm)['correct']
    method = {'native699_non_decrease': count(native) >= count(native, 'control'),
              'ksl34_non_decrease': count(ksl) >= count(ksl, 'control'),
              'at_least_one_increase': count(native) > count(native, 'control') or count(ksl) > count(ksl, 'control')}
    project = {'native699_non_decrease': count(native) >= count(native, 'baseline'),
               'ksl34_non_decrease': count(ksl) >= count(ksl, 'baseline'),
               'at_least_one_increase': count(native) > count(native, 'baseline') or count(ksl) > count(ksl, 'baseline')}
    fraction = lambda m: Fraction(m['macro_recall_exact']['numerator'], m['macro_recall_exact']['denominator'])
    preservation = {'native_macro_non_decrease': fraction(metric(native)) >= fraction(metric(native, 'baseline'))}
    for signer in ('11', '12', '13'):
        preservation['native_signer_' + signer + '_non_decrease'] = (metric(native)['per_signer'][signer]['correct']
                                                   >= metric(native, 'baseline')['per_signer'][signer]['correct'])
    for key in ('group=whole_dev', 'group=public|condition=intact', 'group=public|condition=reviewed_masked',
                'group=public|condition=additional_train', 'group=mp|condition=whole_clip',
                'group=mp|condition=provided_annotation', ksl + '|label=공부', ksl + '|label=먹다'):
        preservation[key + '_non_decrease'] = count(key) >= count(key, 'baseline')
    result = {name: {'passed': all(criteria.values()), 'criteria': criteria}
              for name, criteria in (('method', method), ('project', project), ('preservation', preservation))}
    result['adoptable'] = all(result[key]['passed'] for key in ('method', 'project', 'preservation'))
    result['automatic_promotion_authorized'] = False
    return result


def check_training(run, config, checks):
    """Read tensor artifacts only; no module construction or forward execution."""
    import torch
    training = read(run / 'training.json')
    ready = read(HERE / 'INPUT_READY.json')
    review = read(HERE / 'PRETRAIN_REVIEW.json')
    require(training['status'] == 'BOTH_MODELS_FROZEN_BEFORE_EVALUATION', 'Models not frozen')
    require(ready['status'] == review['status'] == 'PASS', 'Pretraining checks not PASS')
    require(training['config_sha256'] == sha(HERE / 'CONFIG.json'), 'Training config pin')
    require(training['feature_module_sha256'] == sha(HERE / 'geometry_tcn.py'), 'Training feature source pin')
    require(training['driver_sha256'] == sha(HERE / 'run_pair.py'), 'Training driver source pin')
    require(training['prepared_sha256'] == sha(HERE / 'prepared.npz') == ready['prepared']['sha256'], 'Prepared pin')
    require(training['training_arrays_sha256'] == sha(run / 'training_arrays.npz'), 'Training arrays pin')
    require(training['parent'] == config['parent_checkpoint'] == ready['parent'], 'Parent source agreement')
    require(training['parent']['sha256'] == BASELINE_SHA and sha(training['parent']['path']) == BASELINE_SHA,
            'Initial model must be current832')
    require(training['settings'] == config['settings'], 'Training settings agreement')
    require(training['steps_per_arm'] == 660 and training['training_runs'] == 2 and training['attempts'] == 1,
            'Frozen training budget')
    require(training['teacher_source'] == '832 current baseline only', 'Teacher provenance')
    require(training['source_pins'] == config['source_pins'], 'Original source pins changed')
    for name in ('development_training_rows', 'user_training_rows', 'ksl_training_rows'):
        require(training[name] == 0, name)
    require(training['default_promoted'] is False, 'No automatic model promotion')
    for path, digest in ready['pins'].items():
        require(sha(path) == digest, 'Input preparation code pin: ' + path)
    for path, digest in config['source_pins'].items():
        require(sha(path) == digest, 'Original source pin: ' + path)
    require(sha(config['train_cache']['path']) == config['train_cache']['sha256'], 'Original train cache hash')
    require(sha(HERE / 'SCOPE.md') == config['scope_sha256'], 'Scope changed')
    require(ready['teacher']['model_sha256'] == BASELINE_SHA, 'Prepared teacher source')
    require(ready['scale_rows'] == 2346 and ready['rows'] == {'base': 2122, 'context': 224}, 'RMS training rows')
    require(ready['development_arrays_read'] == ready['user_arrays_read'] == 0, 'RMS/teacher data scope')
    parent = torch.load(config['parent_checkpoint']['path'], map_location='cpu', weights_only=True)
    parent_state = parent['state_dict']
    expected_trainable = {'extra_weight'} | {'base.' + k for k in parent_state
        if k.startswith(('blocks.3.', 'head.')) and not any(t in k for t in ('running_mean', 'running_var', 'num_batches_tracked'))}
    model_summary = {}
    with np.load(HERE / 'prepared.npz', allow_pickle=False) as prepared:
        scale = prepared['scale'].copy()
        require(scale.shape == (18,) and scale.dtype == np.float32 and np.all(scale > 0) and np.isfinite(scale).all(),
                'Prepared scales finite positive')
        require(prepared['teacher832'].shape == (2122, 93) and prepared['teacher832'].dtype == np.float64
                and np.isfinite(prepared['teacher832']).all(), 'Prepared teacher format')
        require(prepared['classes'].tolist() == parent['classes'], 'Prepared class order')
    for arm, spec in training['models'].items():
        require(arm in SCORE_KEYS and arm != 'baseline', 'Unexpected trained arm')
        path = Path(spec['path']).resolve()
        require(path.parent == run and sha(path) == spec['sha256'], 'Final checkpoint identity')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint['state_dict']
        require(checkpoint['config_sha256'] == training['config_sha256'] and checkpoint['steps'] == 660,
                'Checkpoint settings pin')
        require(checkpoint['parent'] == config['parent_checkpoint'] and checkpoint['classes'] == parent['classes'],
                'Checkpoint source/classes')
        require(checkpoint['enabled'] is (arm == 'treatment'), 'Arm enabled identity')
        require(set(state) == {'base.' + k for k in parent_state} | {'extra_weight', 'extra_scale'}, 'Model tensor keys')
        require(set(spec['trainable_names']) == expected_trainable, 'Allowed trainable names')
        require(spec['trainable_count'] == sum(state[k].numel() for k in expected_trainable), 'Trainable count')
        changed = []
        for key, old in parent_state.items():
            new = state['base.' + key]
            require(new.shape == old.shape and new.dtype == old.dtype, 'Parent tensor shape/dtype: ' + key)
            if not torch.equal(old, new):
                changed.append(key)
                require('base.' + key in expected_trainable, 'Frozen parent tensor changed: ' + key)
        require(set(changed) == set(spec['changed_parent_tensors']), 'Changed tensor record')
        require(any(key.startswith('blocks.3.') for key in changed) and any(key.startswith('head.') for key in changed),
                'Expected trained scopes did not change')
        require(all(torch.isfinite(t).all() for t in state.values() if t.is_floating_point()), 'Checkpoint finite values')
        exact(state['extra_scale'].numpy(), scale, 'Fixed RMS changed')
        nonzero = bool(torch.count_nonzero(state['extra_weight']))
        require(nonzero is (arm == 'treatment') and spec['extra_weight_nonzero'] is nonzero,
                'Control zero/candidate active feature path')
        model_summary[arm] = {'sha256': spec['sha256'], 'changed_parent_tensor_count': len(changed),
                              'trainable_parameter_count': spec['trainable_count'], 'extra_weight_nonzero': nonzero}
    require(set(model_summary) == {'control', 'treatment'}, 'Two trained arms required')
    with np.load(config['train_cache']['path'], allow_pickle=False) as original, \
            np.load(run / 'training_arrays.npz', allow_pickle=False) as actual:
        for key in ('base_indices', 'context_indices'):
            exact(actual[key], original[key], 'Frozen sampling schedule: ' + key)
            require(actual[key].shape == (660, 16), 'Sampling shape: ' + key)
        for arm in model_summary:
            loss = actual[arm + '_losses']
            require(loss.shape == (660, 4) and np.isfinite(loss).all(), 'Complete finite loss history')
            require(np.allclose(loss[:, :3].sum(axis=1), loss[:, 3], atol=1e-10, rtol=1e-10), 'Loss term sum')
        require(np.array_equal(actual['control_losses'][0], actual['treatment_losses'][0]), 'First losses differ')
    checks.append('Training budget,832 teacher/source,model tensors,fixed buffers,RMS and sampling verified')
    return training, model_summary


def run_audit(run):
    run = run.resolve()
    require(run == RUN.resolve(), 'Audit restricted to the current run')
    output = run / 'INDEPENDENT_AUDIT.json'
    require(not output.exists(), 'Existing independent result is preserved; no overwrite')
    checks = []
    config = read(HERE / 'CONFIG.json')
    require(Path(config['run_dir']).resolve() == run, 'Configured run identity')
    require(sha(OLD / 'EVALUATION_ROWS.json') == OLD_ROWS_SHA, 'Original evaluation rows pin')
    require(sha(OLD / 'evaluation_scores.npz') == OLD_SCORES_SHA, 'Original baseline score archive pin')
    require(sha(MANIFEST) == MANIFEST_SHA, 'Original manifest pin')
    doc = read(run / 'EVALUATION_ROWS.json')
    require(sha(run / 'EVALUATION_ROWS.json') == OLD_ROWS_SHA, 'Original row bytes not preserved')
    rows, classes = doc['rows'], doc['classes']
    require(len(rows) == 855 and len(classes) == len(set(classes)) == 93, '855 rows and 93 classes')
    manifest = read(MANIFEST)
    require(classes == manifest['classes'], 'Manifest class order')
    expected_rows = [row for row in manifest['rows'] if row['role'] == 'evaluation']
    require(rows == expected_rows, 'Ordered row/source/signer/label identity')
    require(len({(r['group'], r['id'], r['condition']) for r in rows}) == 855, 'Duplicate evaluation key')
    require(dict(Counter(r['group'] for r in rows)) == {
        'annotation_dev': 699, 'whole_dev': 74, 'public': 32, 'mp': 14, 'ksl36': 36}, 'Group denominators')
    require(all(r['role'] == 'evaluation' for r in rows), 'Evaluation role')
    with np.load(run / 'evaluation_scores.npz', allow_pickle=False) as z:
        require(set(z.files) == set(SCORE_KEYS.values()) | {'y'}, 'Score archive keys')
        arrays = {key: z[key].copy() for key in z.files}
    y = np.asarray([r['y'] for r in rows], dtype=np.int64)
    exact(arrays['y'], y, 'Frozen label order')
    require(all(classes[int(y[i])] == row['label'] for i, row in enumerate(rows)), 'Label text/index correspondence')
    for key in SCORE_KEYS.values():
        require(arrays[key].shape == (855, 93) and arrays[key].dtype == np.float64 and np.isfinite(arrays[key]).all(),
                'Score dtype/shape/finite: ' + key)
    with np.load(OLD / 'evaluation_scores.npz', allow_pickle=False) as prior:
        exact(arrays['baseline_scores'], prior['baseline_scores'], 'Stored TCN scores byte identity')
        exact(arrays['y'], prior['y'], 'Stored label identity')
    exact(arrays['baseline_scores'], np.asarray([r['baseline_scores'] for r in rows], dtype=np.float64),
          'Embedded TCN scores identity')
    checks.append('855 row identity,class order and byte-exact stored832 comparator')
    training, model_summary = check_training(run, config, checks)
    evaluation = read(run / 'evaluation.json')
    complete = read(run / 'EVALUATION_COMPLETE.json')
    started = read(run / 'EVALUATION_STARTED.json')
    require(evaluation['status'] == complete['status'] == 'EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT', 'Evaluation completion status')
    for name, digest in complete['files'].items():
        require(Path(name).name == name and sha(run / name) == digest, 'Evaluation artifact pin: ' + name)
    provenance = evaluation['provenance']
    require(provenance['training_record_sha256'] == sha(run / 'training.json'), 'Evaluation/training link')
    require(provenance['evaluation_arrays_sha256'] == sha(run / 'evaluation_scores.npz'), 'Evaluation scores link')
    require(provenance['source_pins'] == started['source_pins'] == complete['source_pins'], 'Evaluation start/end pins')
    for path, digest in provenance['source_pins'].items():
        require(sha(path) == digest, 'Evaluation source preserved: ' + path)
    for record in (provenance, started, complete):
        require(record['baseline_inferences'] == 0 and record['baseline_reused'] is True, 'Baseline reuse')
    require(provenance['new_arm_evaluations'] == 2 and provenance['model_state_unchanged'] is True, 'Two frozen-model evaluations')
    require(evaluation['models'] == complete['models'] == started['models'], 'Evaluation model agreement')
    for arm, spec in evaluation['models'].items():
        require(spec['sha256'] == model_summary[arm]['sha256'] and sha(spec['path']) == spec['sha256'], 'Evaluated model identity')
    require(datetime.fromisoformat(training['at_utc']) <= datetime.fromisoformat(started['started_utc'])
            <= datetime.fromisoformat(provenance['finished_utc']) <= datetime.fromisoformat(complete['finished_utc']),
            'Recorded freeze/evaluation chronology')
    require(evaluation['operating_model_changed'] is False and evaluation['user_data_used'] is False
            and evaluation['camera_access'] is False, 'Scope declarations')
    groups = recount(arrays, rows, classes)
    assert_tree_equal(groups, evaluation['groups'], 'groups')
    decisions = decision_values(groups)
    assert_tree_equal(decisions, evaluation['decisions'], 'decisions')
    assert_tree_equal(decisions, complete['decisions'], 'completion/decisions')
    baseline_counts = {'group=annotation_dev': 691, 'group=whole_dev': 74,
                       'group=public|condition=intact': 18, 'group=public|condition=reviewed_masked': 5,
                       'group=public|condition=additional_train': 9, 'group=mp|condition=whole_clip': 7,
                       'group=mp|condition=provided_annotation': 7, 'group=ksl36|primary_unexposed': 22,
                       'group=ksl36|primary_unexposed|label=공부': 16,
                       'group=ksl36|primary_unexposed|label=먹다': 6}
    for key, expected in baseline_counts.items():
        require(groups[key]['full93']['models']['baseline']['correct'] == expected, 'Baseline fact: ' + key)
    require({key: val['correct'] for key, val in groups['group=annotation_dev']['full93']['models']['baseline']['per_signer'].items()}
            == {'11': 229, '12': 234, '13': 228}, 'Baseline signer facts')
    checks.append('Independent argmax,macro,class/signer metrics,paired changes and frozen decisions exact')
    result = {'status': 'PASS', 'at_utc': datetime.now(timezone.utc).isoformat(),
              'auditor_sha256': sha(__file__), 'checks': checks,
              'model_predictions_recounted': 855 * 3,
              'subset_scope_metrics_recounted': len(groups) * len(SCOPES) * 3,
              'paired_comparisons_recounted': len(groups) * len(SCOPES) * 2,
              'models': model_summary, 'decisions': decisions,
              'evaluation_sha256': sha(run / 'evaluation.json'), 'score_sha256': sha(run / 'evaluation_scores.npz'),
              'training_sha256': sha(run / 'training.json'), 'source_manifest_sha256': MANIFEST_SHA,
              'new_model_inferences': 0, 'new_training': 0, 'source_writes': 0,
              'limits': ['Word-label classification only; actual handshape correctness remains unverified.',
                         'Recorded chronology and hashes are consistency evidence, not exhaustive process surveillance.',
                         'Repeated development data and training references do not establish new-person accuracy.']}
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return result


class SyntheticChecks(unittest.TestCase):
    def test_macro_class_balance(self):
        rows = [{'signer_id': 1}] * 4
        m = measure(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 1]), rows, list(range(4)), ['a', 'b'])
        self.assertEqual(m['accuracy'], .75)
        self.assertEqual(m['macro_recall_exact'], {'numerator': 1, 'denominator': 2})
    def test_gain_regression_wrong_to_wrong(self):
        rows = [{'id': str(i), 'group': 'x', 'condition': 'x', 'label': 'a'} for i in range(4)]
        m = compare(np.array([0, 1, 2, 0]), np.array([1, 0, 1, 0]), np.zeros(4, dtype=int), rows,
                    list(range(4)), ['a', 'b', 'c'])
        self.assertEqual((m['gain_count'], m['regression_count'], m['changed_count'], m['delta']), (1, 1, 3, 0))
    def test_empty(self):
        m = measure(np.array([], dtype=int), np.array([], dtype=int), [], [], ['a'])
        self.assertIsNone(m['macro_recall_exact'])
        self.assertIsNone(m['accuracy'])
    def test_tie_first_class(self):
        a = np.array([[1., 1., 0.]])
        self.assertEqual(int(a.argmax(axis=1)[0]), 0)
    def test_compare_tree_detects_bad_metric(self):
        with self.assertRaises(AssertionError):
            assert_tree_equal({'a': .5}, {'a': .51})
    def test_exact_distinguishes_precision(self):
        with self.assertRaises(AssertionError):
            exact(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float64), 'precision')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--run-dir', type=Path, default=RUN)
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticChecks)
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if outcome.wasSuccessful() else 1)
    result = run_audit(args.run_dir)
    print(json.dumps({key: value for key, value in result.items() if key not in ('models', 'limits')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
