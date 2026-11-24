"""Independent saved-score audit. No learner, model, Torch, or raw-input imports.

Only run after the final model and evaluation artifacts have been frozen:
  python check_final_metrics.py --run-dir <completed run directory>
  python check_final_metrics.py --self-test  # synthetic arithmetic only
Outputs are create-exclusive AUDIT_FINAL_METRICS.json and .md in run-dir.
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
MANIFEST = HERE.parent / 'stgcn_transfer_compare_20261005/INPUT_MANIFEST.json'
MANIFEST_SHA = '84f273120b322bde05e945d4e5ee2d6c7ff2314c8d4b8cc889b85a0c000fd91f'
PLAN_SHA = 'ad84ccbcd1ac6b0a68aba67cdfa8496d2f6fcb544420de0a2dee95cf527db4ad'
TARGET = np.array([2, 11, 23, 46, 63, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92])
TARGET_LABELS = ['감사합니다', '나', '모르다', '알다', '좋다', '감', '공부', '인사', '자다',
                 '죄송하다', '맥주', '먹다', '물', '너', '빵', '사랑', '싫다', '운동']
GROUPS = {'annotation_dev': 699, 'whole_dev': 74, 'public': 32, 'mp': 14, 'ksl36': 36}
SUBCOUNTS = {
    'group=annotation_dev|native_signer=11': 233,
    'group=annotation_dev|native_signer=12': 235,
    'group=annotation_dev|native_signer=13': 231,
    'group=whole_dev|native_signer=11': 25,
    'group=whole_dev|native_signer=12': 25,
    'group=whole_dev|native_signer=13': 24,
    'group=public|condition=intact': 18,
    'group=public|condition=reviewed_masked': 5,
    'group=public|condition=additional_train': 9,
    'group=mp|condition=whole_clip': 7,
    'group=mp|condition=provided_annotation': 7,
    'group=ksl36|primary_unexposed': 34,
    'group=ksl36|exposed': 2,
    'group=ksl36|primary_unexposed|label=공부': 18,
    'group=ksl36|primary_unexposed|label=먹다': 16,
}


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def signed_bytes(path, context_sha=None, run_sha=None):
    meta = read_json(str(path) + '.meta.json')
    data = Path(path).read_bytes()
    require(meta['file'] == Path(path).name, f'companion filename: {path}')
    require(meta['sha256'] == hashlib.sha256(data).hexdigest(), f'companion SHA: {path}')
    if context_sha is not None:
        require(meta['context_sha256'] == context_sha, f'context SHA: {path}')
    if run_sha is not None:
        require(meta['run_sha256'] == run_sha, f'run SHA: {path}')
    return data


def fraction_dict(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def macro_fraction(metric):
    value = metric['macro_recall_exact']
    return Fraction(value['numerator'], value['denominator'])


def metric(pred, y, classes):
    pred, y = np.asarray(pred), np.asarray(y)
    require(pred.shape == y.shape and y.ndim == 1, 'metric shapes')
    hit = pred == y
    labels = {}
    recalls = []
    for c in sorted(set(y.tolist())):
        take = y == c
        n, correct = int(take.sum()), int(hit[take].sum())
        labels[classes[c]] = {'n': n, 'correct': correct, 'recall_pct': correct / n * 100}
        recalls.append(Fraction(correct, n))
    mean = sum(recalls, Fraction()) / len(recalls) if recalls else None
    return {'n': len(y), 'correct': int(hit.sum()),
            'accuracy_pct': int(hit.sum()) / len(y) * 100 if len(y) else None,
            'macro_recall_gt_present_pct': float(mean * 100) if mean is not None else None,
            'macro_recall_exact': fraction_dict(mean) if mean is not None else None,
            'n_gt_present_classes': len(labels), 'per_class': labels}


def comparison(candidate_pred, baseline_pred, y, indices):
    candidate, baseline = candidate_pred == y, baseline_pred == y
    gains = indices[candidate & ~baseline].tolist()
    regressions = indices[baseline & ~candidate].tolist()
    delta = int(candidate.sum()) - int(baseline.sum())
    require(delta == len(gains) - len(regressions), 'gain/regression identity')
    return {'correct_delta': delta, 'accuracy_pp_delta': delta / len(y) * 100 if len(y) else None,
            'gains': len(gains), 'regressions': len(regressions),
            'gain_evaluation_indices': gains, 'regression_evaluation_indices': regressions}


def build_subsets(rows):
    groups = np.array([r['group'] for r in rows])
    conditions = np.array([r['condition'] for r in rows])
    signers = np.array([-1 if r['signer_id'] is None else r['signer_id'] for r in rows])
    exposed = np.array([r['exposed'] for r in rows], dtype=bool)
    native = np.array([r['layout'] == 'native67' for r in rows])
    labels = np.array([r['label'] for r in rows])
    out = {}

    def add(key, mask):
        out[key] = np.flatnonzero(mask)

    for group in GROUPS:
        mask = groups == group
        add('group=' + group, mask)
        if group in ('public', 'mp'):
            for condition in sorted(set(conditions[mask].tolist())):
                add(f'group={group}|condition={condition}', mask & (conditions == condition))
        for signer in sorted(set(signers[mask & native & (signers >= 0)].tolist())):
            add(f'group={group}|native_signer={signer}', mask & native & (signers == signer))
    ksl = groups == 'ksl36'
    add('group=ksl36|primary_unexposed', ksl & ~exposed)
    add('group=ksl36|exposed', ksl & exposed)
    for label in ('공부', '먹다'):
        add(f'group=ksl36|primary_unexposed|label={label}', ksl & ~exposed & (labels == label))
        add(f'group=ksl36|exposed|label={label}', ksl & exposed & (labels == label))
    return out


def summarize(candidate_scores, baseline_scores, y, rows, classes, target_indices=TARGET):
    scores = {'candidate': candidate_scores, 'baseline832': baseline_scores}
    predictions = {name: values.argmax(axis=1) for name, values in scores.items()}
    restricted = {name: target_indices[values[:, target_indices].argmax(axis=1)]
                  for name, values in scores.items()}
    target_gt = np.isin(y, target_indices)
    for name in predictions:
        require(np.all(~(predictions[name][target_gt] == y[target_gt]) |
                       (restricted[name][target_gt] == y[target_gt])), 'restricted18 correctness containment')
    summary = {}
    for key, indices in build_subsets(rows).items():
        partial = indices[target_gt[indices]]
        result = {}
        for scope, ix, pred in (('full93', indices, predictions),
                                ('targetGT_full93', partial, predictions),
                                ('restricted18_separate_task', partial, restricted)):
            result[scope] = {name: metric(value[ix], y[ix], classes) for name, value in pred.items()}
            result[scope]['candidate_minus_baseline832'] = comparison(
                pred['candidate'][ix], pred['baseline832'][ix], y[ix], ix)
        summary[key] = result
    return summary, predictions, restricted


def decisions(summary):
    gates = {}

    def m(key, source='candidate', scope='full93'):
        return summary[key][scope][source]

    def keep(key, scope='full93'):
        gates[f'nonregression/{key}/{scope}'] = m(key, scope=scope)['correct'] >= m(key, 'baseline832', scope)['correct']

    native, ksl = 'group=annotation_dev', 'group=ksl36|primary_unexposed'
    gates['ksl34_at_least31'] = m(ksl)['correct'] >= 31
    gates['study18_at_least17'] = m(ksl + '|label=공부')['correct'] >= 17
    gates['eat16_at_least14'] = m(ksl + '|label=먹다')['correct'] >= 14
    gates['ksl34_improves832'] = m(ksl)['correct'] > m(ksl, 'baseline832')['correct']
    keep(native)
    gates['native_macro_exact_nonregression'] = macro_fraction(m(native)) >= macro_fraction(m(native, 'baseline832'))
    for signer in (11, 12, 13):
        keep(native + f'|native_signer={signer}')
    keep(native, 'targetGT_full93')
    keep(native, 'restricted18_separate_task')
    regression_parts = ['group=whole_dev',
                        'group=public|condition=intact', 'group=public|condition=reviewed_masked',
                        'group=public|condition=additional_train',
                        'group=mp|condition=whole_clip', 'group=mp|condition=provided_annotation']
    for key in regression_parts:
        keep(key)
        keep(key, 'restricted18_separate_task')
    native_delta = m(native)['correct'] - m(native, 'baseline832')['correct']
    ksl_delta = m(ksl)['correct'] - m(ksl, 'baseline832')['correct']
    promising = {'native_non_decrease': native_delta >= 0, 'ksl34_non_decrease': ksl_delta >= 0,
                 'at_least_one_increase': native_delta > 0 or ksl_delta > 0}
    return {'strict_adoption_eligibility': {'passed': all(gates.values()), 'criteria': gates,
                                          'failed_criteria': [k for k, v in gates.items() if not v]},
            'exploratory_promising': {'passed': all(promising.values()), 'criteria': promising},
            'operating_model_replacement_authorized': False}


def markdown(report):
    lines = ['# 최종 저장점수 독립 검산', '',
             '전체93 top1 결과다. 조건별 자료는 중복되므로 전체855 정확도로 합산하지 않는다.', '',
             '| 자료/조건 | N | 새 ST-GCN+GRU | TCN832 | 차이(%p) |',
             '|---|---:|---:|---:|---:|']
    keys = ['group=annotation_dev', 'group=whole_dev',
            'group=public|condition=intact', 'group=public|condition=reviewed_masked',
            'group=public|condition=additional_train', 'group=mp|condition=whole_clip',
            'group=mp|condition=provided_annotation', 'group=ksl36|primary_unexposed',
            'group=ksl36|primary_unexposed|label=공부', 'group=ksl36|primary_unexposed|label=먹다',
            'group=ksl36|exposed']
    for key in keys:
        s = report['summary'][key]['full93']
        a, b, d = s['candidate'], s['baseline832'], s['candidate_minus_baseline832']
        label = key.replace('group=', '').replace('|', ' / ')
        lines.append(f"| {label} | {a['n']} | {a['correct']}/{a['n']} ({a['accuracy_pct']:.2f}%) | "
                     f"{b['correct']}/{b['n']} ({b['accuracy_pct']:.2f}%) | {d['accuracy_pp_delta']:+.2f} |")
    lines += ['', '| 목표부분집합 과제 | N | 새 후보 | TCN832 |', '|---|---:|---:|---:|']
    for key in ('group=annotation_dev', 'group=whole_dev'):
        for scope in ('targetGT_full93', 'restricted18_separate_task'):
            s = report['summary'][key][scope]
            a, b = s['candidate'], s['baseline832']
            lines.append(f"| {key} / {scope} | {a['n']} | {a['correct']} ({a['accuracy_pct']:.2f}%) | {b['correct']} ({b['accuracy_pct']:.2f}%) |")
    native = report['summary']['group=annotation_dev']['full93']
    lines += ['', f"Native GT81 macro recall: 새 후보 {native['candidate']['macro_recall_gt_present_pct']:.8f}%, "
              f"TCN832 {native['baseline832']['macro_recall_gt_present_pct']:.8f}%. 판정은 반올림 전 정확 유리수로 비교했다.", '']
    for name in ('exploratory_promising', 'strict_adoption_eligibility'):
        lines.append(f"- {name}: {'PASS' if report['decisions'][name]['passed'] else 'FAIL'}")
    failures = report['decisions']['strict_adoption_eligibility']['failed_criteria']
    if failures:
        lines += ['', '미통과 채택 조건:'] + ['- ' + name for name in failures]
    lines += ['', '이는 저장 점수의 산술·행정체성 검산이다. 모델 forward, 동결 전후 실행 순서, 최적 수렴을 이 검산으로 인증하지 않는다.',
              'KSL34는 반복 개발자료이며 촬영자 독립 시험이 아니다. 입력과 학습 recipe를 포함한 패키지 비교이며 운영 모델 자동 교체 승인이 아니다.', '']
    return '\n'.join(lines)


def run(run_dir):
    run_dir = Path(run_dir).resolve()
    output = run_dir / 'AUDIT_FINAL_METRICS.json'
    output_md = run_dir / 'AUDIT_FINAL_METRICS.md'
    require(not output.exists() and not output_md.exists(), 'audit outputs already exist; preserve prior audit')
    require(sha(HERE / 'FROZEN_EXPERIMENT.json') == PLAN_SHA, 'preregistered plan changed')
    require(sha(MANIFEST) == MANIFEST_SHA, 'input manifest changed')
    context = read_json(HERE / 'EXECUTION_CONTEXT.json')
    context_sha = canonical_sha(context)
    frozen_path = run_dir / 'FROZEN_MODEL.json'
    frozen = json.loads(signed_bytes(frozen_path, context_sha))
    require(frozen['context'] == context and frozen['context_sha256'] == context_sha, 'frozen context mismatch')
    model_path = run_dir / 'final_model.npz'
    model_bytes = signed_bytes(model_path, context_sha, frozen['run_sha256'])
    require(hashlib.sha256(model_bytes).hexdigest() == frozen['final_model_sha256'], 'frozen model SHA mismatch')
    del model_bytes
    require(context['source_pins']['input_manifest'] == MANIFEST_SHA, 'model input manifest pin mismatch')
    rows_path, scores_path = run_dir / 'EVALUATION_ROWS.json', run_dir / 'EVALUATION_SCORES.npz'
    inputs = [MANIFEST, HERE / 'FROZEN_EXPERIMENT.json', HERE / 'EXECUTION_CONTEXT.json',
              frozen_path, model_path, rows_path, scores_path]
    before = {str(p): sha(p) for p in inputs}
    source = read_json(MANIFEST)
    expected = [(i, r) for i, r in enumerate(source['rows']) if r['role'] == 'evaluation']
    metadata = read_json(rows_path)
    classes, rows = metadata['classes'], metadata['rows']
    require(classes == source['classes'] == context['classes'] == frozen['classes'], '93-class order mismatch')
    require(len(classes) == len(set(classes)) == 93, 'class count/uniqueness')
    require([classes[i] for i in TARGET] == TARGET_LABELS, 'legacy target18 class order mismatch')
    require(len(rows) == len(expected) == 855, '855 condition rows required')
    fields = ('role', 'group', 'id', 'condition', 'label', 'signer_id', 'exposed')
    required = set(fields) | {'baseline_scores'}
    enriched = []
    for j, (row, (global_index, original)) in enumerate(zip(rows, expected)):
        require(required <= set(row), f'missing row metadata at {j}')
        require(all(row[k] == original[k] for k in fields), f'manifest row order/identity at {j}')
        require(type(row['exposed']) is bool and row['role'] == 'evaluation', f'role/exposed type at {j}')
        for key in ('y', 'layout', 'path', 'sha256', 'start', 'end_exclusive', 'mask_hand'):
            if key in row:
                require(row[key] == original[key], f'metadata {key} differs at {j}')
        if 'row_index' in row:
            require(row['row_index'] == global_index, f'manifest row_index differs at {j}')
        require(row['baseline_scores'] == original['baseline_scores'], f'embedded baseline changed at {j}')
        enriched.append(dict(original, manifest_row_index=global_index, evaluation_index=j))
    identities = [(r['group'], r['id'], r['condition']) for r in rows]
    require(len(set(identities)) == 855, 'duplicate condition identity')
    with np.load(scores_path, allow_pickle=False) as archive:
        require({'candidate_scores', 'baseline_scores', 'y'} <= set(archive.files), 'missing score arrays')
        candidate, baseline, y = [archive[k].copy() for k in ('candidate_scores', 'baseline_scores', 'y')]
        if 'classes' in archive.files:
            require(archive['classes'].tolist() == classes, 'NPZ class order mismatch')
        if 'manifest_indices' in archive.files:
            require(np.array_equal(archive['manifest_indices'], [i for i, _ in expected]), 'NPZ manifest indices')
    for name, values in (('candidate', candidate), ('baseline832', baseline)):
        require(values.shape == (855, 93) and values.dtype.kind == 'f' and np.isfinite(values).all(), f'{name} score shape/type/finite')
    require(y.shape == (855,) and y.dtype.kind in 'iu', 'integer y shape/type')
    require(np.array_equal(y, [r['y'] for r in enriched]), 'labels/order mismatch')
    reference_scores = np.asarray([r['baseline_scores'] for r in enriched], dtype=np.float64)
    require(np.array_equal(baseline, reference_scores), 'baseline score values differ from pinned832 manifest')
    subsets = build_subsets(enriched)
    for key, count in {**{'group=' + k: v for k, v in GROUPS.items()}, **SUBCOUNTS}.items():
        require(key in subsets and len(subsets[key]) == count, f'subset denominator {key}')
    require(len(subsets['group=ksl36|exposed|label=공부']) == len(subsets['group=ksl36|exposed|label=먹다']) == 1,
            'two exposed KSL rows must be one per label')
    summary, predictions, restricted = summarize(candidate, baseline, y, enriched, classes)
    for group, count, labels_present in (('annotation_dev', 33, 6), ('whole_dev', 30, 6)):
        target = summary['group=' + group]['targetGT_full93']['candidate']
        require(target['n'] == count and target['n_gt_present_classes'] == labels_present, 'targetGT denominator')
    require(summary['group=annotation_dev']['full93']['candidate']['n_gt_present_classes'] == 81, 'native81 represented labels')
    known = {'group=annotation_dev': 691, 'group=whole_dev': 74, 'group=ksl36|primary_unexposed': 22,
             'group=annotation_dev|native_signer=11': 229, 'group=annotation_dev|native_signer=12': 234,
             'group=annotation_dev|native_signer=13': 228,
             **{k: v for k, v in SUBCOUNTS.items() if '|condition=' in k}}
    for key, correct in known.items():
        require(summary[key]['full93']['baseline832']['correct'] == correct, f'known832 correctness identity: {key}')
    row_predictions = []
    for j, row in enumerate(enriched):
        record = {k: row[k] for k in ('evaluation_index', 'manifest_row_index', 'group', 'id', 'condition',
                                      'label', 'signer_id', 'exposed')}
        record['models'] = {}
        for name, pred in predictions.items():
            record['models'][name] = {'full93_prediction': classes[int(pred[j])], 'full93_correct': bool(pred[j] == y[j]),
                                      'restricted18_prediction': classes[int(restricted[name][j])] if y[j] in TARGET else None}
        row_predictions.append(record)
    require(all(sha(path) == digest for path, digest in before.items()), 'input changed during audit')
    report = {'status': 'INDEPENDENT_SAVED_SCORE_METRICS_VERIFIED', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'verification_passed': True, 'sources': before, 'verifier_sha256': sha(__file__),
              'classes': classes, 'target18_indices': TARGET.tolist(), 'target18_labels': TARGET_LABELS,
              'summary': summary, 'decisions': decisions(summary), 'rows': row_predictions,
              'identities': {'evaluation_condition_rows': 855, 'manifest_order_exact': True,
                             'baseline_values_exact': True, 'gain_minus_regression_equals_correct_delta': True,
                             'restricted18_contains_full93_target_correct': True},
              'limits': ['No pooled 855-row accuracy: groups and conditions overlap.',
                         'KSL34 is repeated development data, not verified unseen-person evaluation.',
                         'This audit verifies saved-score arithmetic and row identity, not forward or execution chronology.',
                         'Package/recipe comparison, not isolated architecture causality; no automatic model replacement.']}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with output_md.open('x', encoding='utf-8') as stream:
        stream.write(markdown(report))
    print(json.dumps({'status': report['status'], 'output': str(output), 'decisions': report['decisions']}, ensure_ascii=False))
    return report


class ArithmeticTests(unittest.TestCase):
    def test_macro_exact_and_absent_classes_excluded(self):
        result = metric(np.array([0, 1, 1]), np.array([0, 0, 1]), ['a', 'b', 'absent'])
        self.assertEqual(result['correct'], 2)
        self.assertEqual(result['n_gt_present_classes'], 2)
        self.assertEqual(macro_fraction(result), Fraction(3, 4))

    def test_changed_row_identity_and_percentage(self):
        result = comparison(np.array([0, 1, 1]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([7, 9, 10]))
        self.assertEqual(result['gain_evaluation_indices'], [7, 9])
        self.assertEqual(result['regression_evaluation_indices'], [10])
        self.assertEqual(result['correct_delta'], 1)
        self.assertAlmostEqual(result['accuracy_pp_delta'], 100 / 3)

    def test_restricted_task_and_stable_ties(self):
        scores = np.array([[2, 0, 1], [0, 1, 1]], dtype=np.float64)
        target = np.array([1, 2])
        truth = np.array([2, 1])
        full = scores.argmax(1)
        restricted = target[scores[:, target].argmax(1)]
        self.assertEqual(full.tolist(), [0, 1])
        self.assertEqual(restricted.tolist(), [2, 1])
        self.assertEqual(int((restricted == truth).sum()), 2)

    def test_ksl_exposure_and_duplicate_clip_conditions(self):
        rows = [dict(group='ksl36', condition='k', signer_id=None, exposed=e, layout='mp75', label=label)
                for e, label in ((False, '공부'), (False, '먹다'), (True, '공부'), (True, '먹다'))]
        rows += [dict(group='mp', condition=c, signer_id=None, exposed=False, layout='mp75', label='공부', id='same')
                 for c in ('whole_clip', 'provided_annotation')]
        parts = build_subsets(rows)
        self.assertEqual(parts['group=ksl36|primary_unexposed'].tolist(), [0, 1])
        self.assertEqual(parts['group=ksl36|exposed'].tolist(), [2, 3])
        self.assertEqual(parts['group=mp|condition=whole_clip'].tolist(), [4])
        self.assertEqual(parts['group=mp|condition=provided_annotation'].tolist(), [5])

    def test_empty_subset_is_not_zero_recall(self):
        result = metric(np.array([], dtype=int), np.array([], dtype=int), ['a'])
        self.assertEqual(result['n'], 0)
        self.assertIsNone(result['accuracy_pct'])
        self.assertIsNone(result['macro_recall_exact'])

    def test_promising_is_separate_from_strict_adoption(self):
        keys = ['group=annotation_dev', 'group=whole_dev',
                'group=ksl36|primary_unexposed', 'group=ksl36|primary_unexposed|label=공부',
                'group=ksl36|primary_unexposed|label=먹다',
                *['group=annotation_dev|native_signer=' + str(i) for i in (11, 12, 13)],
                *['group=public|condition=' + c for c in ('intact', 'reviewed_masked', 'additional_train')],
                *['group=mp|condition=' + c for c in ('whole_clip', 'provided_annotation')]]
        summary = {key: {scope: {source: {'correct': 1, 'macro_recall_exact': fraction_dict(Fraction(71, 72))}
                                  for source in ('candidate', 'baseline832')}
                         for scope in ('full93', 'targetGT_full93', 'restricted18_separate_task')}
                   for key in keys}
        ksl = summary['group=ksl36|primary_unexposed']['full93']
        ksl['baseline832']['correct'], ksl['candidate']['correct'] = 22, 23
        result = decisions(summary)
        self.assertTrue(result['exploratory_promising']['passed'])
        self.assertFalse(result['strict_adoption_eligibility']['passed'])
        ksl['candidate']['correct'] = 31
        summary['group=ksl36|primary_unexposed|label=공부']['full93']['candidate']['correct'] = 17
        summary['group=ksl36|primary_unexposed|label=먹다']['full93']['candidate']['correct'] = 14
        self.assertTrue(decisions(summary)['strict_adoption_eligibility']['passed'])
        # A decrease below ordinary display precision must still fail the exact gate.
        summary['group=annotation_dev']['full93']['candidate']['macro_recall_exact'] = fraction_dict(
            Fraction(71, 72) - Fraction(1, 10**18))
        self.assertFalse(decisions(summary)['strict_adoption_eligibility']['passed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ArithmeticTests)
        raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
    require(args.run_dir is not None, '--run-dir required; no implicit score reads')
    run(args.run_dir)
