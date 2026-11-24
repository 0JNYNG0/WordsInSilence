"""Independent three-way saved-score audit; no model/learner/raw-input imports.

After candidate freeze and one final evaluation:
  python check_comparison.py --run-dir <runs/stgcn_valid_window_v1>
Synthetic arithmetic/contract checks only:
  python check_comparison.py --self-test
Importing this module does not read artifacts or execute checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
OLD_UNIT = OPS / 'stgcn_execution_ready_20261005'
OLD_RUN = OPS.parent / 'runs/stgcn_full_path_v1'
MANIFEST = OPS / 'stgcn_transfer_compare_20261005/INPUT_MANIFEST.json'
METRICS_PATH = OLD_UNIT / 'check_final_metrics.py'
METRICS_SHA = 'f07ac3f820a6bd54b1f7522d31b91a792e07d71bb076124b13f88e3fd8b8d708'
PLAN_SHA = 'e4dc2ad90f70bd337de37890d92d614f260d45338dbf97818547fe37b8bc2389'
OLD_AUDIT_SHA = '723c109c61440c587600ff6f733a8662e8f5e90eb29130b8bfb6fedc80cbd257'
SCOPES = ('full93', 'targetGT_full93', 'restricted18_separate_task')
MODELS = ('candidate_valid_window', 'old_stgcn', 'tcn832')


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_metrics():
    require(sha(METRICS_PATH) == METRICS_SHA, 'reused metric source changed')
    spec = importlib.util.spec_from_file_location('pinned_stgcn_saved_score_metrics', METRICS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def check_arrays(current, previous, expected_y, shape=(855, 93)):
    required = {'candidate_scores', 'old_stgcn_scores', 'baseline_scores', 'y'}
    require(required <= set(current), 'new score archive missing required keys')
    for name in ('candidate_scores', 'old_stgcn_scores', 'baseline_scores'):
        value = current[name]
        require(value.shape == shape and value.dtype.kind == 'f' and np.isfinite(value).all(),
                f'new {name} shape/dtype/finite')
    for name in ('candidate_scores', 'baseline_scores'):
        value = previous[name]
        require(value.shape == shape and value.dtype.kind == 'f' and np.isfinite(value).all(),
                f'old {name} shape/dtype/finite')
    for archive in (current, previous):
        require(archive['y'].shape == (shape[0],) and archive['y'].dtype.kind in 'iu', 'integer y shape/dtype')
        require(np.array_equal(archive['y'], expected_y), 'score y differs from ordered manifest')
    for new_key, old_key in (('old_stgcn_scores', 'candidate_scores'), ('baseline_scores', 'baseline_scores')):
        a, b = current[new_key], previous[old_key]
        require(a.dtype == b.dtype and a.tobytes(order='C') == b.tobytes(order='C'),
                f'{new_key} not byte-exact stored comparator')


def three_way_summary(R, arrays, y, rows, classes):
    """Reuse project arithmetic without assigning old ST-GCN the name baseline832."""
    project, new_preds, new_restricted = R.summarize(
        arrays['candidate_scores'], arrays['baseline_scores'], y, rows, classes)
    old_project, old_preds, old_restricted = R.summarize(
        arrays['old_stgcn_scores'], arrays['baseline_scores'], y, rows, classes)
    predictions = {'candidate_valid_window': new_preds['candidate'],
                   'old_stgcn': old_preds['candidate'], 'tcn832': new_preds['baseline832']}
    restricted = {'candidate_valid_window': new_restricted['candidate'],
                  'old_stgcn': old_restricted['candidate'], 'tcn832': new_restricted['baseline832']}
    target_gt = np.isin(y, R.TARGET)
    summary = {}
    for key, indices in R.build_subsets(rows).items():
        summary[key] = {}
        for scope in SCOPES:
            take = indices if scope == 'full93' else indices[target_gt[indices]]
            pred = restricted if scope == 'restricted18_separate_task' else predictions
            by_model = {'candidate_valid_window': project[key][scope]['candidate'],
                        'old_stgcn': old_project[key][scope]['candidate'],
                        'tcn832': project[key][scope]['baseline832']}
            pairs = {}
            for a, b in (('candidate_valid_window', 'tcn832'),
                         ('candidate_valid_window', 'old_stgcn'), ('old_stgcn', 'tcn832')):
                result = R.comparison(pred[a][take], pred[b][take], y[take], take)
                result['macro_recall_pp_delta'] = (float((R.macro_fraction(by_model[a]) -
                                                         R.macro_fraction(by_model[b])) * 100)
                                                   if len(take) else None)
                pairs[a + '_minus_' + b] = result
            summary[key][scope] = {'n': len(take), 'by_model': by_model, 'comparisons': pairs}
    return summary, project, old_project, predictions, restricted


def method_decision(summary):
    deltas = {name: summary['group=' + key]['full93']['comparisons']
              ['candidate_valid_window_minus_old_stgcn']['correct_delta']
              for name, key in (('native699', 'annotation_dev'), ('ksl34', 'ksl36|primary_unexposed'))}
    criteria = {'native699_non_decrease': deltas['native699'] >= 0,
                'ksl34_non_decrease': deltas['ksl34'] >= 0,
                'at_least_one_increase': any(value > 0 for value in deltas.values())}
    return {'passed': all(criteria.values()), 'criteria': criteria, 'correct_deltas': deltas,
            'meaning': 'prespecified method comparison against old ST-GCN; not project adoption'}


def markdown(report):
    lines = ['# 유효 창 후보 저장점수 독립 비교', '',
             '전체93 top1. 이전 ST-GCN과 TCN832 점수는 저장값을 그대로 재사용했다. 중복 조건855행을 합산하지 않는다.', '',
             '| 자료/조건 | N | 새 후보 | 이전 ST-GCN | TCN832 | 새−이전(%p) | 새−TCN(%p) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    keys = ['group=annotation_dev', 'group=whole_dev',
            'group=public|condition=intact', 'group=public|condition=reviewed_masked',
            'group=public|condition=additional_train', 'group=mp|condition=whole_clip',
            'group=mp|condition=provided_annotation', 'group=ksl36|primary_unexposed',
            'group=ksl36|primary_unexposed|label=공부', 'group=ksl36|primary_unexposed|label=먹다',
            'group=ksl36|exposed']
    for key in keys:
        entry = report['summary'][key]['full93']
        scores = [f"{entry['by_model'][m]['correct']}/{entry['n']} ({entry['by_model'][m]['accuracy_pct']:.2f}%)" for m in MODELS]
        method = entry['comparisons']['candidate_valid_window_minus_old_stgcn']
        project = entry['comparisons']['candidate_valid_window_minus_tcn832']
        label = key.replace('group=', '').replace('|', ' / ')
        lines.append(f"| {label} | {entry['n']} | {' | '.join(scores)} | {method['accuracy_pp_delta']:+.2f} | {project['accuracy_pp_delta']:+.2f} |")
    lines += ['', '| native 세부지표/별도 과제 | 새 후보 | 이전 ST-GCN | TCN832 |', '|---|---:|---:|---:|']
    for signer in (11, 12, 13):
        entry = report['summary'][f'group=annotation_dev|native_signer={signer}']['full93']
        values = [f"{entry['by_model'][m]['correct']}/{entry['n']}" for m in MODELS]
        lines.append(f"| native signer {signer} | {' | '.join(values)} |")
    full = report['summary']['group=annotation_dev']['full93']['by_model']
    lines.append('| native GT81 macro recall | ' + ' | '.join(f"{full[m]['macro_recall_gt_present_pct']:.8f}%" for m in MODELS) + ' |')
    for key in ('group=annotation_dev', 'group=whole_dev'):
        for scope in ('targetGT_full93', 'restricted18_separate_task'):
            entry = report['summary'][key][scope]
            values = [f"{entry['by_model'][m]['correct']}/{entry['n']} ({entry['by_model'][m]['accuracy_pct']:.2f}%)" for m in MODELS]
            lines.append(f"| {key} / {scope} | {' | '.join(values)} |")
    lines += ['', '| 주 집합 비교 | 개선 | 회귀 | 정답수 차이 |', '|---|---:|---:|---:|']
    for key in ('group=annotation_dev', 'group=ksl36|primary_unexposed'):
        for pair in ('candidate_valid_window_minus_old_stgcn', 'candidate_valid_window_minus_tcn832'):
            c = report['summary'][key]['full93']['comparisons'][pair]
            lines.append(f"| {key.replace('|', ' / ')} / {pair} | {c['gains']} | {c['regressions']} | {c['correct_delta']:+d} |")
    lines += ['', f"- 이전 ST-GCN 대비 방법 기준: {'PASS' if report['method_check']['passed'] else 'FAIL'}",
              f"- TCN832 대비 유망성: {'PASS' if report['project_decisions']['exploratory_promising']['passed'] else 'FAIL'}",
              f"- 기존 채택 자격: {'PASS' if report['project_decisions']['strict_adoption_eligibility']['passed'] else 'FAIL'}"]
    failed = report['project_decisions']['strict_adoption_eligibility']['failed_criteria']
    if failed:
        lines += ['', '미통과 채택 조건:'] + ['- ' + key for key in failed]
    lines += ['', '전체31부분집합×3과제의 세 모델 지표, 개선·회귀 행 인덱스와 행별 예측은 JSON에 기록했다.',
              'native 목표GT33은 목표18 중 실제GT가 있는6단어뿐이다. 나머지12단어는native699에서0행이며, 공부·먹다는KSL34에서 별도로 본다.',
              'public18과mask5는 학습 원본 및 파생 조건이다. 사과9·MP7×2도 노출 회귀검사이며, KSL34는 반복 개발자료로 촬영자 독립 시험이 아니다.',
              '방법 효과는 패딩 제외에 따른encoder BN/GRU 처리와 동일 정책으로 새로 선택한학습시점을 포함한다. 원사전학습 입력 정렬·구조만의 우열·동일update수 실험을 뜻하지 않는다.',
              '이 검산은 저장점수 산술·정체성만 확인하며 모델 forward, 학습 추적, 실행 순서 검사를 대신하지 않는다. 운영 자동 교체 승인이 아니다.', '']
    return '\n'.join(lines)


def run(run_dir):
    R = load_metrics()
    run_dir = Path(run_dir).resolve()
    output, output_md = run_dir / 'AUDIT_COMPARISON.json', run_dir / 'AUDIT_COMPARISON.md'
    require(not output.exists() and not output_md.exists(), 'preserve existing comparison audit')
    plan_path, context_path = HERE / 'FROZEN_EXPERIMENT.json', HERE / 'EXECUTION_CONTEXT.json'
    require(sha(plan_path) == PLAN_SHA, 'new preregistered plan changed')
    plan, context = read_json(plan_path), read_json(context_path)
    require(sha(OLD_UNIT / 'FROZEN_EXPERIMENT.json') == plan['source_plan_sha256'], 'old experiment pin')
    require(sha(MANIFEST) == plan['pins']['input_manifest'] == R.MANIFEST_SHA, 'manifest pin')
    require(context['source_pins'] == plan['pins'], 'candidate context source pins differ')
    require(context['input_policy'] == plan['input_policy'], 'candidate input policy differs')
    library_pins = context['library_pins']
    require(str(plan_path) in library_pins and library_pins[str(plan_path)] == PLAN_SHA, 'context must bind new plan')
    for path, digest in library_pins.items():
        require(sha(path) == digest, f'candidate context library pin: {path}')
    context_sha = R.canonical_sha(context)
    frozen_path, model_path = run_dir / 'FROZEN_MODEL.json', run_dir / 'final_model.npz'
    frozen = json.loads(R.signed_bytes(frozen_path, context_sha))
    require(frozen['context'] == context and frozen['context_sha256'] == context_sha, 'candidate frozen context')
    run_sha = frozen['run_sha256']
    R.signed_bytes(frozen_path, context_sha, run_sha)
    model_bytes = R.signed_bytes(model_path, context_sha, run_sha)
    require(hashlib.sha256(model_bytes).hexdigest() == frozen['final_model_sha256'], 'candidate frozen model pin')
    del model_bytes
    old_scores_path, old_rows_path = OLD_RUN / 'EVALUATION_SCORES.npz', OLD_RUN / 'EVALUATION_ROWS.json'
    old_audit_path = OLD_RUN / 'AUDIT_FINAL_METRICS.json'
    source = plan['comparison_source']
    require(sha(old_scores_path) == source['scores_sha256'], 'old score archive pin')
    require(sha(old_rows_path) == source['rows_sha256'], 'old row archive pin')
    require(sha(OLD_RUN / 'final_model.npz') == source['old_model_sha256'], 'old ST-GCN model pin')
    require(sha(old_audit_path) == OLD_AUDIT_SHA, 'old verified metric receipt pin')
    rows_path, scores_path = run_dir / 'EVALUATION_ROWS.json', run_dir / 'EVALUATION_SCORES.npz'
    require(sha(rows_path) == source['rows_sha256'], 'new evaluation rows must be unchanged old rows')
    tracked_paths = [plan_path, context_path, frozen_path, model_path, rows_path, scores_path,
                     MANIFEST, METRICS_PATH, old_scores_path, old_rows_path, old_audit_path,
                     OLD_UNIT / 'FROZEN_EXPERIMENT.json', OLD_RUN / 'final_model.npz']
    tracked_paths += [Path(path) for path in library_pins]
    before = {str(p): sha(p) for p in tracked_paths}
    for p in (frozen_path, model_path):
        before[str(p) + '.meta.json'] = sha(str(p) + '.meta.json')
    manifest, metadata = read_json(MANIFEST), read_json(rows_path)
    expected = [(i, row) for i, row in enumerate(manifest['rows']) if row['role'] == 'evaluation']
    classes, rows = metadata['classes'], metadata['rows']
    require(classes == manifest['classes'] == context['classes'] == frozen['classes'], 'class order')
    require(len(classes) == len(set(classes)) == 93 and [classes[i] for i in R.TARGET] == R.TARGET_LABELS, 'class/target18 contract')
    require(len(rows) == len(expected) == 855, '855 evaluation conditions required')
    fields = ('role', 'group', 'id', 'condition', 'label', 'signer_id', 'exposed', 'baseline_scores')
    enriched = []
    for j, (row, (global_index, original)) in enumerate(zip(rows, expected)):
        require(all(key in row and row[key] == original[key] for key in fields), f'row identity/order {j}')
        require(type(row['exposed']) is bool and row['role'] == 'evaluation', f'evaluation role {j}')
        for key in ('y', 'layout', 'path', 'sha256', 'start', 'end_exclusive', 'mask_hand'):
            if key in row:
                require(row[key] == original[key], f'row {key} {j}')
        if 'row_index' in row:
            require(row['row_index'] == global_index, f'global row index {j}')
        enriched.append(dict(original, manifest_row_index=global_index, evaluation_index=j))
    require(len({(r['group'], r['id'], r['condition']) for r in rows}) == 855, 'duplicate condition identity')
    current, previous = load_arrays(scores_path), load_arrays(old_scores_path)
    y = np.array([row['y'] for row in enriched], dtype=np.int64)
    check_arrays(current, previous, y)
    for arrays in (current, previous):
        if 'classes' in arrays:
            require(arrays['classes'].tolist() == classes, 'NPZ class order')
        if 'manifest_indices' in arrays:
            require(np.array_equal(arrays['manifest_indices'], [i for i, _ in expected]), 'NPZ manifest order')
    require(np.array_equal(current['baseline_scores'], np.array([r['baseline_scores'] for r in enriched], dtype=np.float64)),
            'TCN scores differ from pinned manifest')
    subsets = R.build_subsets(enriched)
    old_audit = read_json(old_audit_path)
    require(len(subsets) == 31 and set(subsets) == set(old_audit['summary']), '31 subset identities')
    for key, count in {**{'group=' + k: v for k, v in R.GROUPS.items()}, **R.SUBCOUNTS}.items():
        require(len(subsets[key]) == count, f'subset denominator {key}')
    summary, project, old_project, predictions, restricted = three_way_summary(R, current, y, enriched, classes)
    require(old_audit['verifier_sha256'] == METRICS_SHA and old_project == old_audit['summary'],
            'reused old metrics do not exactly reproduce old audit')
    for key, count in (('group=annotation_dev', 33), ('group=whole_dev', 30)):
        require(summary[key]['targetGT_full93']['n'] == count, 'native targetGT row count')
        require(summary[key]['targetGT_full93']['by_model']['candidate_valid_window']['n_gt_present_classes'] == 6,
                'native targetGT present labels')
    require(summary['group=annotation_dev']['full93']['by_model']['candidate_valid_window']['n_gt_present_classes'] == 81,
            'native GT81 classes')
    row_predictions = []
    for i, row in enumerate(enriched):
        record = {key: row[key] for key in ('evaluation_index', 'manifest_row_index', 'group', 'id', 'condition',
                                           'label', 'signer_id', 'exposed')}
        record['models'] = {name: {'full93_prediction': classes[int(pred[i])], 'full93_correct': bool(pred[i] == y[i]),
                                   'restricted18_prediction': classes[int(restricted[name][i])] if y[i] in R.TARGET else None}
                            for name, pred in predictions.items()}
        row_predictions.append(record)
    require(all(sha(path) == digest for path, digest in before.items()), 'artifact changed during audit')
    report = {'status': 'INDEPENDENT_THREE_WAY_SAVED_SCORE_AUDIT_PASS', 'verification_passed': True,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'sources': before,
              'verifier_sha256': sha(__file__), 'reused_metric_source_sha256': METRICS_SHA,
              'model_sources': {'candidate_valid_window': frozen['final_model_sha256'],
                               'old_stgcn': source['old_model_sha256'], 'tcn832': plan['pins']['baseline832']},
              'classes': classes, 'target18_indices': R.TARGET.tolist(), 'target18_labels': R.TARGET_LABELS,
              'summary': summary, 'project_decisions': R.decisions(project), 'method_check': method_decision(summary),
              'rows': row_predictions,
              'identities': {'rows': 855, 'subsets': 31, 'scopes_per_subset': 3, 'models': 3,
                             'row_file_byte_exact_previous': True, 'scores_comparators_byte_exact_previous': True,
                             'old_metric_receipt_reproduced_exactly': True, 'candidate_inference_during_audit': 0},
              'limits': plan['interpretation'] + [
                  'No pooled 855-row accuracy; training references and overlapping conditions remain distinct.',
                  'Native targetGT33 covers six target words; twelve target words have no GT in native699.',
                  'This is saved-score verification, not forward/optimization/chronology verification.']}
    rendered = markdown(report)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with output_md.open('x', encoding='utf-8') as stream:
        stream.write(rendered)
    print(json.dumps({'output': str(output), 'verification_passed': True,
                      'method_check': report['method_check'], 'project_decisions': report['project_decisions']}, ensure_ascii=False))
    return report


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = load_metrics()

    def test_comparator_swap_or_precision_loss_rejected(self):
        old = {'candidate_scores': np.array([[0, 2, 1], [2, 0, 1]], dtype=np.float32),
               'baseline_scores': np.array([[2, 0, 1], [0, 2, 1]], dtype=np.float64), 'y': np.array([1, 0])}
        new = {'candidate_scores': old['candidate_scores'].copy(), 'old_stgcn_scores': old['candidate_scores'].copy(),
               'baseline_scores': old['baseline_scores'].copy(), 'y': old['y'].copy()}
        check_arrays(new, old, old['y'], (2, 3))
        for key, value in (('baseline_scores', new['baseline_scores'].astype(np.float32)),
                           ('old_stgcn_scores', new['baseline_scores'].astype(np.float32)),
                           ('y', new['y'][::-1])):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                check_arrays(dict(new, **{key: value}), old, old['y'], (2, 3))

    def test_three_sources_not_misnamed_and_partial_tasks(self):
        classes = ['mock' + str(i) for i in range(93)]
        y = np.array([2, 11, 81, 86])
        rows = [dict(group=g, id=str(i), condition='intact', signer_id=None, exposed=False,
                     layout='mp75', label=classes[int(y[i])])
                for i, g in enumerate(['annotation_dev', 'annotation_dev', 'ksl36', 'ksl36'])]
        def scores(pred, dtype):
            values = np.zeros((4, 93), dtype=dtype)
            values[np.arange(4), pred] = 1
            return values
        arrays = {'candidate_scores': scores([2, 11, 81, 0], np.float32),
                  'old_stgcn_scores': scores([0, 11, 0, 86], np.float32),
                  'baseline_scores': scores([2, 11, 81, 86], np.float64)}
        summary, _, _, _, _ = three_way_summary(self.R, arrays, y, rows, classes)
        native = summary['group=annotation_dev']['full93']
        self.assertEqual(set(native['by_model']), set(MODELS))
        self.assertEqual(native['comparisons']['candidate_valid_window_minus_old_stgcn']['correct_delta'], 1)
        self.assertEqual(native['comparisons']['candidate_valid_window_minus_tcn832']['correct_delta'], 0)
        self.assertTrue(method_decision(summary)['passed'])
        ksl = summary['group=ksl36|primary_unexposed']['full93']
        self.assertEqual(ksl['comparisons']['candidate_valid_window_minus_tcn832']['correct_delta'], -1)
        # Equal counts may conceal one gain and one regression against the old model.
        change = ksl['comparisons']['candidate_valid_window_minus_old_stgcn']
        self.assertEqual((change['gains'], change['regressions']), (1, 1))

    def test_method_gate_requires_both_sets(self):
        summary = {key: {'full93': {'comparisons': {'candidate_valid_window_minus_old_stgcn': {'correct_delta': delta}}}}
                   for key, delta in (('group=annotation_dev', 20), ('group=ksl36|primary_unexposed', -1))}
        self.assertFalse(method_decision(summary)['passed'])
        summary['group=ksl36|primary_unexposed']['full93']['comparisons']['candidate_valid_window_minus_old_stgcn']['correct_delta'] = 0
        self.assertTrue(method_decision(summary)['passed'])


def self_test():
    R = load_metrics()
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(R.ArithmeticTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonTests)])
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        require(args.run_dir is None, '--self-test must not be combined with actual run directory')
        raise SystemExit(0 if self_test() else 1)
    require(args.run_dir is not None, '--run-dir required; no implicit result reads')
    run(args.run_dir)
