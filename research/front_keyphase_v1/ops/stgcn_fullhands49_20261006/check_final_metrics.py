"""Four-model independent final score audit; no model, learner, or Torch import.

Requires completed input, graph-transfer, training-trace and frozen-file audits.
Only saved scores are compared; no candidate or comparator inference is run.
Use --self-test for synthetic arithmetic only, or --run-dir after one evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
PREVIOUS = OPS / 'stgcn_valid_window_pair_20261006'
METHOD_RUN = OPS.parent / 'runs/stgcn_valid_window_v1'
REFERENCE_RUN = OPS.parent / 'runs/stgcn_full_path_v1'
EXPECTED_RUN = OPS.parent / 'runs/stgcn_fullhands49_v1'
MANIFEST = OPS / 'stgcn_transfer_compare_20261005/INPUT_MANIFEST.json'
METRICS_PATH = OPS / 'stgcn_execution_ready_20261005/check_final_metrics.py'
METRICS_SHA = 'f07ac3f820a6bd54b1f7522d31b91a792e07d71bb076124b13f88e3fd8b8d708'
PLAN_SHA = '8a38d09802e55b7cbd2954d2a946b50df65a89bb38ad8906ebc375ed97da8997'
METHOD_AUDIT_SHA = '2f420835646aad6a19fc679bb24b67da3a45e7029d345895a44b1b570f1fa8f3'
REFERENCE_AUDIT_SHA = '723c109c61440c587600ff6f733a8662e8f5e90eb29130b8bfb6fedc80cbd257'
REFERENCE_MODEL_SHA = '4c1554468e48d8a671b2a40be337f87f5bc4ff2935869e6257f3b38569d9625d'
SCOPES = ('full93', 'targetGT_full93', 'restricted18_separate_task')
SCORE_KEYS = {'candidate49': 'candidate_scores', 'valid27': 'old_stgcn_scores',
              'tcn832': 'baseline_scores', 'reference27': 'old_reference_scores'}
MODELS = tuple(SCORE_KEYS)
PAIRS = (('candidate49', 'valid27'), ('candidate49', 'tcn832'), ('candidate49', 'reference27'),
         ('valid27', 'tcn832'), ('valid27', 'reference27'), ('reference27', 'tcn832'))
DISPLAY_KEYS = ('group=annotation_dev', 'group=whole_dev', 'group=public|condition=intact',
                'group=public|condition=reviewed_masked', 'group=public|condition=additional_train',
                'group=mp|condition=whole_clip', 'group=mp|condition=provided_annotation',
                'group=ksl36|primary_unexposed', 'group=ksl36|primary_unexposed|label=공부',
                'group=ksl36|primary_unexposed|label=먹다', 'group=ksl36|exposed')


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_metrics():
    require(sha(METRICS_PATH) == METRICS_SHA, 'Original arithmetic/adoption source changed')
    spec = importlib.util.spec_from_file_location('_fullhands49_pinned_metrics', METRICS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def tensor_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f'{array.dtype.str}|{array.shape}|'.encode('ascii'))
    digest.update(array.tobytes())
    return digest.hexdigest()


def exact_array(a, b, description):
    require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order='C') == b.tobytes(order='C'),
            description)


def check_arrays(current, previous, reference, expected_y, shape=(855, 93)):
    require(set(SCORE_KEYS.values()) | {'y'} <= set(current), 'Four-model archive keys missing')
    for archive, keys in ((current, SCORE_KEYS.values()),
                          (previous, ('candidate_scores', 'baseline_scores', 'old_stgcn_scores')),
                          (reference, ('candidate_scores', 'baseline_scores'))):
        for key in keys:
            values = archive[key]
            require(values.shape == shape and values.dtype.kind == 'f' and np.isfinite(values).all(),
                    f'Score shape/dtype/finite: {key}')
        require(archive['y'].shape == (shape[0],) and archive['y'].dtype.kind in 'iu'
                and np.array_equal(archive['y'], expected_y), 'Labels differ from ordered manifest')
    exact_array(current['old_stgcn_scores'], previous['candidate_scores'], 'valid27 stored scores changed')
    exact_array(current['old_reference_scores'], reference['candidate_scores'], 'reference27 stored scores changed')
    exact_array(previous['old_stgcn_scores'], reference['candidate_scores'], 'Previous reference27 identity changed')
    for archive in (previous, reference):
        exact_array(current['baseline_scores'], archive['baseline_scores'], 'TCN stored scores changed')


def four_way_summary(R, arrays, y, rows, classes):
    project, predictions, restricted = {}, {}, {}
    for model in ('candidate49', 'valid27', 'reference27'):
        result, pred, partial = R.summarize(arrays[SCORE_KEYS[model]], arrays['baseline_scores'],
                                           y, rows, classes)
        project[model] = result
        predictions[model], restricted[model] = pred['candidate'], partial['candidate']
        predictions['tcn832'], restricted['tcn832'] = pred['baseline832'], partial['baseline832']
    target_gt = np.isin(y, R.TARGET)
    summary = {}
    for key, indices in R.build_subsets(rows).items():
        summary[key] = {}
        for scope in SCOPES:
            take = indices if scope == 'full93' else indices[target_gt[indices]]
            pred = restricted if scope == 'restricted18_separate_task' else predictions
            by_model = {model: project[model][key][scope]['candidate']
                        for model in ('candidate49', 'valid27', 'reference27')}
            by_model['tcn832'] = project['candidate49'][key][scope]['baseline832']
            comparisons = {}
            for a, b in PAIRS:
                values = R.comparison(pred[a][take], pred[b][take], y[take], take)
                values['macro_recall_pp_delta'] = (float((R.macro_fraction(by_model[a])
                                                         - R.macro_fraction(by_model[b])) * 100)
                                                     if len(take) else None)
                comparisons[a + '_minus_' + b] = values
            summary[key][scope] = {'n': len(take), 'by_model': by_model, 'comparisons': comparisons}
    return summary, project, predictions, restricted


def method_decision(summary):
    delta = {name: summary[key]['full93']['comparisons']['candidate49_minus_valid27']['correct_delta']
             for name, key in (('native699', 'group=annotation_dev'),
                               ('ksl34', 'group=ksl36|primary_unexposed'))}
    criteria = {'native699_non_decrease': delta['native699'] >= 0,
                'ksl34_non_decrease': delta['ksl34'] >= 0,
                'at_least_one_increase': any(value > 0 for value in delta.values())}
    return {'passed': all(criteria.values()), 'criteria': criteria, 'correct_deltas': delta,
            'comparator': 'valid27_3ca59574', 'meaning': 'Prespecified method check, separate from TCN adoption'}


def context_pin(context, path):
    target = str(Path(path).resolve()).casefold()
    matches = [digest for name, digest in context['library_pins'].items()
               if str(Path(name).resolve()).casefold() == target]
    require(len(matches) == 1 and matches[0] == sha(path), f'Missing/changed execution-context pin: {path}')
    return matches[0]


def hash_field(record, names, expected, description):
    present = [name for name in names if name in record]
    require(present, f'Missing execution hash: {description}')
    require(all(record[name] == expected for name in present), f'Execution hash mismatch: {description}')


def verify_prerequisites(R, run_dir, plan, context):
    """Fail closed before score arithmetic; do not infer a missing PASS."""
    context_sha = R.canonical_sha(context)
    for path, digest in context['library_pins'].items():
        require(sha(path) == digest, f'Execution library/source pin: {path}')
    for path in (HERE / 'FROZEN_EXPERIMENT.json', HERE / 'CONFIG.json', HERE / 'BINDINGS.json',
                 HERE / 'check_final_metrics.py', HERE / 'check_training_trace.py', HERE / 'check_inputs.py',
                 HERE / 'GRAPH_TRANSFER_CHECK.json', HERE / 'INPUT_INDEPENDENT_CHECK.json',
                 HERE / 'CODE_REVIEW.json', HERE / 'prepared/evaluation.npz',
                 HERE / 'prepared/EVALUATION_ROWS.json', HERE / 'prepared/EVALUATION_INPUT_DIAGNOSTICS.json'):
        context_pin(context, path)
    require(context['inference_shape'] == [6, 2, 10, 49], 'Candidate inference shape')
    require(context['input_policy'] == plan['input_policy'], 'Candidate input policy')
    require(read_json(HERE / 'CONFIG.json') == plan['training_config'], 'Frozen training config')
    for path, digest in read_json(HERE / 'BINDINGS.json')['pins'].items():
        require(sha(path) == digest, f'Binding source pin: {path}')
    transfer = read_json(HERE / 'GRAPH_TRANSFER_CHECK.json')
    inputs = read_json(HERE / 'INPUT_INDEPENDENT_CHECK.json')
    review = read_json(HERE / 'CODE_REVIEW.json')
    require(transfer['status'] == inputs['status'] == review['status'] == 'PASS', 'Pre-execution audit incomplete')
    require(inputs['plan_sha256'] == PLAN_SHA and inputs['input_policy'] == plan['input_policy'], 'Input audit plan')
    require(inputs['bindings_sha256'] == sha(HERE / 'BINDINGS.json'), 'Input audit bindings')
    require(inputs['verifier_sha256'] == sha(HERE / 'check_inputs.py'), 'Input verifier changed')
    for name, digest in inputs['artifact_sha256'].items():
        require(sha(HERE / 'prepared' / name) == digest, f'Independently audited input artifact changed: {name}')
    require(inputs['evaluation_rows'] == 855 and inputs['common27_evaluation_digest_exact'] == 855,
            'Input audit coverage')
    require((HERE / 'TRAINING_COMPLETION.json').is_file(), 'Training completion missing')
    trace = read_json(HERE / 'TRAINING_TRACE_AUDIT.json')
    require(trace['status'] == 'PASS' and trace['finalized'] is True, 'Training trace not finalized PASS')
    require(trace['plan_sha256'] == PLAN_SHA and trace['context_sha256'] == context_sha,
            'Training trace does not bind this plan/context')
    require(trace['auditor_sha256'] == sha(HERE / 'check_training_trace.py'), 'Training trace verifier changed')
    frozen_path = run_dir / 'FROZEN_MODEL.json'
    frozen = json.loads(R.signed_bytes(frozen_path, context_sha))
    run_sha = frozen['run_sha256']
    require(trace['run_sha256'] == run_sha and frozen['context'] == context, 'Trace/frozen run identity')
    R.signed_bytes(frozen_path, context_sha, run_sha)
    model_bytes = R.signed_bytes(run_dir / 'final_model.npz', context_sha, run_sha)
    require(hashlib.sha256(model_bytes).hexdigest() == frozen['final_model_sha256'], 'Frozen model bytes')
    require(frozen['inference_shape'] == [6, 2, 10, 49], 'Frozen model input shape')
    del model_bytes
    for filename in ('INIT_SELECT.json', 'INIT_REFIT.json'):
        initial = json.loads(R.signed_bytes(run_dir / filename, context_sha, run_sha))
        require(initial['state_digest'] == transfer['actual_initial_state_digest'], 'Fresh49 initialization differs')
    execution = read_json(run_dir / 'EVALUATION_EXECUTION.json')
    started = read_json(run_dir / 'EVALUATION_STARTED.json')
    hash_field(execution, ('final_model_sha256', 'model_sha256'), frozen['final_model_sha256'], 'model')
    hash_field(execution, ('score_sha256', 'scores_sha256'), sha(run_dir / 'EVALUATION_SCORES.npz'), 'scores')
    hash_field(execution, ('frozen_model_sha256',), sha(frozen_path), 'frozen record')
    hash_field(started, ('frozen_model_sha256',), sha(frozen_path), 'start frozen record')
    require(execution.get('model_state_unchanged') is True, 'Evaluation mutated model or lacked state audit')
    require(execution.get('no_training_or_selection') is True and execution.get('user_data_used') is False,
            'Evaluation scope not attested')
    require(execution.get('comparator_inferences') == 0, 'Stored comparators must not be inferred')
    with np.load(HERE / 'prepared/evaluation.npz', allow_pickle=False) as archive:
        require(set(archive.files) == {'x', 'window_valid', 'y'}, 'Prepared evaluation keys')
        x, mask, y = archive['x'], archive['window_valid'], archive['y']
        require(x.shape == (855, 6, 2, 10, 49) and x.dtype == np.float32, 'Prepared49 shape/dtype')
        require(mask.shape == (855, 6) and mask.dtype == np.bool_, 'Prepared window mask shape/dtype')
        for record in (started, execution):
            require(record['input_sha256'] == tensor_sha(x), 'Scored input differs from audited49 cache')
            require(record['window_valid_sha256'] == tensor_sha(mask), 'Scored temporal mask differs')
        prepared_y = y.copy()
    for filename in ('EVALUATION_ROWS.json', 'EVALUATION_INPUT_DIAGNOSTICS.json'):
        require(sha(run_dir / filename) == sha(HERE / 'prepared' / filename), 'Scored input metadata changed')
    return frozen, prepared_y, {'input': sha(HERE / 'INPUT_INDEPENDENT_CHECK.json'),
                              'transfer': sha(HERE / 'GRAPH_TRANSFER_CHECK.json'),
                              'training_trace': sha(HERE / 'TRAINING_TRACE_AUDIT.json'),
                              'code_review': sha(HERE / 'CODE_REVIEW.json'),
                              'execution': sha(run_dir / 'EVALUATION_EXECUTION.json')}


def markdown(report):
    lines = ['# 49관절 후보 최종 저장점수 독립 검산', '',
             '전체93 top1. valid27은 직전 3ca59574 모델, reference27은 이전 4c155446 참고 모델이다.',
             '855조건은 원본과 파생 조건이 중복되므로 통합 정확도를 만들지 않는다.', '',
             '| 자료/조건 | 후보49 | valid27 | TCN832 | reference27 | 후보−valid27(%p) | 후보−TCN(%p) | 후보−reference27(%p) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in DISPLAY_KEYS:
        entry = report['summary'][key]['full93']
        scores = [f"{entry['by_model'][m]['correct']}/{entry['n']} ({entry['by_model'][m]['accuracy_pct']:.2f}%)"
                  for m in MODELS]
        deltas = [f"{entry['comparisons']['candidate49_minus_' + m]['accuracy_pp_delta']:+.2f}"
                  for m in ('valid27', 'tcn832', 'reference27')]
        lines.append('| ' + key.replace('group=', '').replace('|', ' / ') + ' | '
                     + ' | '.join(scores + deltas) + ' |')
    lines += ['', '| 주 집합의 후보49 비교 | 개선 | 회귀 | 정답수 차이 |', '|---|---:|---:|---:|']
    for key in ('group=annotation_dev', 'group=ksl36|primary_unexposed'):
        for comparator in ('valid27', 'tcn832', 'reference27'):
            pair = report['summary'][key]['full93']['comparisons']['candidate49_minus_' + comparator]
            lines.append(f"| {key.replace('|', ' / ')} / {comparator} | {pair['gains']} | {pair['regressions']} | {pair['correct_delta']:+d} |")
    lines += ['', '| 별도 부분집합/과제 | 후보49 | valid27 | TCN832 | reference27 |', '|---|---:|---:|---:|---:|']
    for key in ('group=annotation_dev', 'group=whole_dev', 'group=ksl36|primary_unexposed|label=공부',
                'group=ksl36|primary_unexposed|label=먹다'):
        for scope in ('targetGT_full93', 'restricted18_separate_task'):
            entry = report['summary'][key][scope]
            values = [f"{entry['by_model'][m]['correct']}/{entry['n']} ({entry['by_model'][m]['accuracy_pct']:.2f}%)" for m in MODELS]
            lines.append('| ' + key.replace('|', ' / ') + ' / ' + scope + ' | ' + ' | '.join(values) + ' |')
    for signer in (11, 12, 13):
        entry = report['summary'][f'group=annotation_dev|native_signer={signer}']['full93']
        values = [f"{entry['by_model'][m]['correct']}/{entry['n']}" for m in MODELS]
        lines.append(f"| native signer {signer} | " + ' | '.join(values) + ' |')
    native = report['summary']['group=annotation_dev']['full93']['by_model']
    lines.append('| native GT81 macro recall | ' + ' | '.join(f"{native[m]['macro_recall_gt_present_pct']:.8f}%" for m in MODELS) + ' |')
    lines += ['', f"- valid27 대비 방법 기준: {'PASS' if report['method_check']['passed'] else 'FAIL'}",
              f"- TCN 대비 유망성: {'PASS' if report['project_decisions']['exploratory_promising']['passed'] else 'FAIL'}",
              f"- 기존23개 채택 조건: {'PASS' if report['project_decisions']['strict_adoption_eligibility']['passed'] else 'FAIL'}"]
    failed = report['project_decisions']['strict_adoption_eligibility']['failed_criteria']
    if failed:
        lines += ['', '미통과 채택 조건:'] + ['- ' + key for key in failed]
    lines += ['', '31개 부분집합×3과제×4모델 및 모든 비교쌍의 개선·회귀 행은 JSON에 보존했다.',
              'native 목표33행은 목표18 중6단어만 포함한다. 제한18은 별도 과제이며 전체93 결과를 대신하지 않는다.',
              '학습참조와 파생 가림 조건은 새 사람 평가가 아니다. KSL34도 반복 개발자료다.',
              '그래프·부분 이전·관절 평균·선택된 학습량이 함께 달라진 비교이며 관절수 단독 원인을 입증하지 않는다.',
              '이 보고서는 단어 인식 결과다. 손모양 의존성은 별도 분석이며 실제 수형 정오나 수행 합격률로 해석하지 않는다.',
              '입력·학습추적·동결해시 감사 PASS를 확인한 뒤 저장점수만 검산했다. 운영 자동 교체 승인은 없다.', '']
    return '\n'.join(lines)


def run(run_dir):
    R = load_metrics()
    run_dir = Path(run_dir).resolve()
    require(run_dir == EXPECTED_RUN.resolve(), 'Only the authorized fullhands49 run may be audited')
    output, output_md = run_dir / 'AUDIT_FINAL_METRICS.json', run_dir / 'AUDIT_FINAL_METRICS.md'
    require(not output.exists() and not output_md.exists(), 'Preserve existing final metrics audit')
    plan_path, context_path = HERE / 'FROZEN_EXPERIMENT.json', HERE / 'EXECUTION_CONTEXT.json'
    require(sha(plan_path) == PLAN_SHA, 'Preregistered49 plan changed')
    plan, context = read_json(plan_path), read_json(context_path)
    require(sha(PREVIOUS / 'FROZEN_EXPERIMENT.json') == plan['source_plan_sha256'], 'Previous plan pin')
    require(sha(MANIFEST) == plan['pins']['input_manifest'] == R.MANIFEST_SHA, 'Manifest pin')
    source = plan['comparison_source']
    require(sha(METHOD_RUN / 'EVALUATION_SCORES.npz') == source['scores_sha256'], 'valid27 score archive pin')
    require(sha(REFERENCE_RUN / 'EVALUATION_SCORES.npz') == source['reference27_scores_sha256'], 'reference27 score pin')
    require(sha(METHOD_RUN / 'final_model.npz') == source['old_model_sha256'], 'valid27 model identity')
    require(sha(REFERENCE_RUN / 'final_model.npz') == REFERENCE_MODEL_SHA, 'reference27 model identity')
    require(sha(METHOD_RUN / 'AUDIT_COMPARISON.json') == METHOD_AUDIT_SHA, 'Previous verified comparison changed')
    require(sha(REFERENCE_RUN / 'AUDIT_FINAL_METRICS.json') == REFERENCE_AUDIT_SHA, 'Older verified comparison changed')
    for directory in (run_dir, METHOD_RUN, REFERENCE_RUN):
        require(sha(directory / 'EVALUATION_ROWS.json') == source['rows_sha256'], 'Evaluation rows/order changed')
    tracked = [plan_path, context_path, MANIFEST, METRICS_PATH, HERE / 'TRAINING_COMPLETION.json',
               HERE / 'TRAINING_TRACE_AUDIT.json', METHOD_RUN / 'AUDIT_COMPARISON.json',
               REFERENCE_RUN / 'AUDIT_FINAL_METRICS.json']
    tracked += [Path(path) for path in context['library_pins']]
    for directory in (run_dir, METHOD_RUN, REFERENCE_RUN):
        tracked += [directory / name for name in ('EVALUATION_SCORES.npz', 'EVALUATION_ROWS.json',
                                                   'FROZEN_MODEL.json', 'final_model.npz')]
    tracked += [run_dir / name for name in ('FROZEN_MODEL.json.meta.json', 'final_model.npz.meta.json',
                'INIT_SELECT.json', 'INIT_SELECT.json.meta.json', 'INIT_REFIT.json', 'INIT_REFIT.json.meta.json',
                'EVALUATION_STARTED.json', 'EVALUATION_EXECUTION.json', 'EVALUATION_INPUT_DIAGNOSTICS.json')]
    before = {str(path): sha(path) for path in tracked}
    frozen, prepared_y, gates = verify_prerequisites(R, run_dir, plan, context)
    manifest, metadata = read_json(MANIFEST), read_json(run_dir / 'EVALUATION_ROWS.json')
    expected = [(i, row) for i, row in enumerate(manifest['rows']) if row['role'] == 'evaluation']
    classes, rows = metadata['classes'], metadata['rows']
    require(classes == manifest['classes'] == context['classes'] == frozen['classes'], 'Class order changed')
    require(len(classes) == len(set(classes)) == 93 and [classes[i] for i in R.TARGET] == R.TARGET_LABELS,
            'Class/target18 contract')
    require(len(rows) == len(expected) == 855, 'Expected855 evaluation conditions')
    enriched = []
    for index, (row, (global_index, original)) in enumerate(zip(rows, expected)):
        require(row == original, f'Manifest row identity/order: {index}')
        require(type(row['exposed']) is bool and row['role'] == 'evaluation', f'Evaluation role: {index}')
        enriched.append(dict(original, manifest_row_index=global_index, evaluation_index=index))
    require(len({(row['group'], row['id'], row['condition']) for row in rows}) == 855, 'Duplicate condition identity')
    y = np.array([row['y'] for row in enriched], dtype=np.int64)
    require(np.array_equal(y, prepared_y), 'Audited input labels differ')
    current = load_arrays(run_dir / 'EVALUATION_SCORES.npz')
    previous = load_arrays(METHOD_RUN / 'EVALUATION_SCORES.npz')
    reference = load_arrays(REFERENCE_RUN / 'EVALUATION_SCORES.npz')
    check_arrays(current, previous, reference, y)
    for arrays in (current, previous, reference):
        if 'classes' in arrays:
            require(arrays['classes'].tolist() == classes, 'NPZ class order')
        if 'manifest_indices' in arrays:
            require(np.array_equal(arrays['manifest_indices'], [i for i, _ in expected]), 'NPZ manifest order')
    require(np.array_equal(current['baseline_scores'], np.array([row['baseline_scores'] for row in enriched], dtype=np.float64)),
            'TCN832 manifest values changed')
    subsets = R.build_subsets(enriched)
    require(len(subsets) == 31, 'Expected31 subset identities')
    for key, count in {**{'group=' + k: v for k, v in R.GROUPS.items()}, **R.SUBCOUNTS}.items():
        require(len(subsets[key]) == count, f'Subset denominator: {key}')
    summary, projects, predictions, restricted = four_way_summary(R, current, y, enriched, classes)
    old_audit = read_json(REFERENCE_RUN / 'AUDIT_FINAL_METRICS.json')
    require(projects['reference27'] == old_audit['summary'], 'Reference27 old audit did not reproduce exactly')
    method_audit = read_json(METHOD_RUN / 'AUDIT_COMPARISON.json')
    require(set(summary) == set(method_audit['summary']), 'Previous subset identities changed')
    for key in summary:
        for scope in SCOPES:
            old = method_audit['summary'][key][scope]
            for name, previous_name in (('valid27', 'candidate_valid_window'), ('tcn832', 'tcn832'), ('reference27', 'old_stgcn')):
                require(summary[key][scope]['by_model'][name] == old['by_model'][previous_name], 'Previous metric identity')
    for key, count in (('group=annotation_dev', 33), ('group=whole_dev', 30)):
        target = summary[key]['targetGT_full93']
        require(target['n'] == count and target['by_model']['candidate49']['n_gt_present_classes'] == 6,
                'Native targetGT coverage changed')
    require(summary['group=annotation_dev']['full93']['by_model']['candidate49']['n_gt_present_classes'] == 81,
            'Native GT81 coverage changed')
    decisions = R.decisions(projects['candidate49'])
    require(len(decisions['strict_adoption_eligibility']['criteria']) == 23, 'Original adoption gates changed')
    row_predictions = []
    for index, row in enumerate(enriched):
        entry = {key: row[key] for key in ('evaluation_index', 'manifest_row_index', 'group', 'id', 'condition',
                                          'label', 'signer_id', 'exposed')}
        entry['models'] = {name: {'full93_prediction': classes[int(pred[index])],
                                  'full93_correct': bool(pred[index] == y[index]),
                                  'restricted18_prediction': classes[int(restricted[name][index])] if y[index] in R.TARGET else None}
                           for name, pred in predictions.items()}
        row_predictions.append(entry)
    require(all(sha(path) == digest for path, digest in before.items()), 'Artifact changed during final audit')
    report = {'status': 'INDEPENDENT_FOUR_MODEL_SAVED_SCORE_AUDIT_PASS', 'verification_passed': True,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'sources': before,
              'verifier_sha256': sha(__file__), 'reused_metric_source_sha256': METRICS_SHA,
              'prerequisite_audits': gates, 'model_sources': {'candidate49': frozen['final_model_sha256'],
                  'valid27': source['old_model_sha256'], 'tcn832': plan['pins']['baseline832'],
                  'reference27': REFERENCE_MODEL_SHA}, 'classes': classes,
              'target18_indices': R.TARGET.tolist(), 'target18_labels': R.TARGET_LABELS,
              'summary': summary, 'project_decisions': decisions, 'method_check': method_decision(summary),
              'rows': row_predictions,
              'identities': {'rows': 855, 'subsets': 31, 'scopes_per_subset': 3, 'models': 4,
                             'all_comparator_scores_byte_exact': True, 'all_old_metrics_reproduced': True,
                             'no_pooled855_accuracy': True, 'original23_adoption_gates_reused': True},
              'limits': ['Input/transfer/finalized training audits and frozen hashes are prerequisites, not newly rerun experiments.',
                         'KSL34 is repeated development data; native targetGT covers six target words only.',
                         'Reference27 is additional context; the method comparator is valid27 and project comparator is TCN832.',
                         'Graph, partial transfer, pooling and selected training duration change together.',
                         'Word accuracy does not establish handshape discrimination or correct sign execution.',
                         'No inference, training, pooled855 accuracy, or operating-model replacement.']}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    with output_md.open('x', encoding='utf-8') as handle:
        handle.write(markdown(report))
    print(json.dumps({'status': report['status'], 'output': str(output), 'method_check': report['method_check'],
                      'project_decisions': decisions}, ensure_ascii=False))
    return report


class FourWayTests(unittest.TestCase):
    def test_four_models_keep_distinct_predictions_and_paired_errors(self):
        R = load_metrics()
        classes = ['class' + str(i) for i in range(93)]
        for index, label in zip(R.TARGET, R.TARGET_LABELS):
            classes[int(index)] = label
        y = np.array([2, 11, 81, 86])
        rows = [dict(group=group, id=str(i), condition='intact', signer_id=None,
                     exposed=False, layout='mp75', label=classes[int(y[i])])
                for i, group in enumerate(('annotation_dev', 'annotation_dev', 'ksl36', 'ksl36'))]
        def scores(predictions):
            array = np.zeros((4, 93), dtype=np.float32)
            array[np.arange(4), predictions] = 1
            return array
        arrays = {'candidate_scores': scores([2, 11, 81, 0]),
                  'old_stgcn_scores': scores([0, 11, 0, 86]),
                  'baseline_scores': scores([2, 11, 81, 86]).astype(np.float64),
                  'old_reference_scores': scores([0, 0, 81, 86])}
        summary, _, predictions, _ = four_way_summary(R, arrays, y, rows, classes)
        self.assertEqual(set(predictions), set(MODELS))
        native = summary['group=annotation_dev']['full93']
        self.assertEqual([native['by_model'][name]['correct'] for name in MODELS], [2, 1, 2, 0])
        ksl = summary['group=ksl36|primary_unexposed']['full93']['comparisons']
        self.assertEqual((ksl['candidate49_minus_valid27']['gains'],
                          ksl['candidate49_minus_valid27']['regressions']), (1, 1))
        self.assertEqual(ksl['candidate49_minus_tcn832']['correct_delta'], -1)
        self.assertEqual(ksl['candidate49_minus_reference27']['correct_delta'], -1)
        self.assertTrue(method_decision(summary)['passed'])
        self.assertNotIn('all', summary)

    def test_comparator_identity_and_dtypes(self):
        a = np.array([[0, 2, 1], [2, 0, 1]], dtype=np.float32)
        b = np.array([[2, 0, 1], [0, 2, 1]], dtype=np.float64)
        y = np.array([1, 0])
        reference = {'candidate_scores': a.copy(), 'baseline_scores': b.copy(), 'y': y}
        previous = dict(reference, old_stgcn_scores=a.copy(), candidate_scores=a[:, ::-1].copy())
        current = {'candidate_scores': a.copy(), 'old_stgcn_scores': previous['candidate_scores'].copy(),
                   'old_reference_scores': a.copy(), 'baseline_scores': b.copy(), 'y': y}
        check_arrays(current, previous, reference, y, (2, 3))
        for key, value in (('old_stgcn_scores', a), ('old_reference_scores', previous['candidate_scores']),
                           ('baseline_scores', b.astype(np.float32)), ('y', y[::-1])):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                check_arrays(dict(current, **{key: value}), previous, reference, y, (2, 3))

    def test_method_cannot_pass_with_ksl_regression(self):
        summary = {key: {'full93': {'comparisons': {'candidate49_minus_valid27': {'correct_delta': value}}}}
                   for key, value in (('group=annotation_dev', 18), ('group=ksl36|primary_unexposed', -1))}
        self.assertFalse(method_decision(summary)['passed'])
        summary['group=ksl36|primary_unexposed']['full93']['comparisons']['candidate49_minus_valid27']['correct_delta'] = 0
        self.assertTrue(method_decision(summary)['passed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        require(args.run_dir is None, 'Synthetic self-test cannot read an actual run')
        R = load_metrics()
        suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(R.ArithmeticTests),
                                   unittest.defaultTestLoader.loadTestsFromTestCase(FourWayTests)])
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    require(args.run_dir is not None, '--run-dir required; no implicit result reads')
    try:
        run(args.run_dir)
    except Exception as exc:
        # Keep the first failure without creating a success artifact.
        failure = HERE / 'FINAL_METRICS_FIRST_FAILURE.json'
        if not failure.exists():
            with failure.open('x', encoding='utf-8') as handle:
                json.dump({'status': 'FAIL', 'error_type': type(exc).__name__, 'reason': str(exc),
                           'traceback': traceback.format_exc(), 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                           'verifier_sha256': sha(__file__)}, handle, ensure_ascii=False, indent=2)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
