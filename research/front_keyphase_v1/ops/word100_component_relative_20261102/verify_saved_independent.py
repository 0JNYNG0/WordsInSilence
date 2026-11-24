"""Independent arithmetic audit. Standard library only; no model calls."""
from pathlib import Path
from collections import defaultdict
from statistics import mean
import hashlib
import json
import math

R = Path(__file__).resolve().parent
B = R.parent / 'word100_component_learning_20261102'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
checks = []

def check(name, value):
    checks.append({'name': name, 'passed': bool(value)})

def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(v, b[k]) for k, v in a.items())
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return b is not None and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
    return a == b

def recalculate(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row['record']['role'] == 'development':
            grouped[row['record']['target'] + '::' + row['variant']].append(row)
    result = {}
    for key, group in grouped.items():
        known = [r for r in group if r['mask']]
        positives = [r for r in known if r['y'] == 1]
        negatives = [r for r in known if r['y'] == 0]
        def rate(items, predicate):
            clips = defaultdict(list)
            for row in items:
                clips[row['record']['clip_id']].append(int(predicate(row)))
            return mean(mean(v) for v in clips.values()) if clips else None
        result[key] = {
            'records': len(group), 'known': len(known), 'unknown': len(group) - len(known),
            'clips': len({r['record']['clip_id'] for r in group}),
            'positive': len(positives), 'negative': len(negatives),
            'TP': sum(r['probability'] >= .5 for r in positives),
            'TN': sum(r['probability'] < .5 for r in negatives),
            'recall': rate(positives, lambda r: r['probability'] >= .5),
            'specificity': rate(negatives, lambda r: r['probability'] < .5),
            'usable_positive_recall': rate(positives, lambda r: r['usable'] and r['probability'] >= .5),
            'usable_negative_specificity': rate(negatives, lambda r: r['usable'] and r['probability'] < .5),
            'confirmed_positive_recall': rate(positives, lambda r: r['usable'] and r['probability'] >= .8),
            'confirmed_negative_specificity': rate(negatives, lambda r: r['usable'] and r['probability'] <= .2),
            'confident_wrong': sum(r['usable'] and r['probability'] <= .2 for r in positives) + sum(r['usable'] and r['probability'] >= .8 for r in negatives),
            'held': sum(not r['usable'] for r in group),
            'uncertain': sum(r['usable'] and .2 < r['probability'] < .8 for r in group),
            'unsupported_not_dropped': True,
        }
    return result

for unit in [B, R]:
    for path, digest in read(unit / 'FROZEN.json')['pins'].items():
        check('pin:' + path, sha(Path(path)) == digest)

manifest = read(B / 'INPUT_MANIFEST.json')['temporal']
fit = [s for s in manifest if s['record']['role'] == 'fit' and s['mask'] and s['variant'] in ['native', 'reverse', 'still']]
check('fit_count_99', len(fit) == 99)
check('fit_ids_01_to_09', {s['record']['person'] for s in fit} == set(range(1, 10)))
check('development_ids_11_to_13', {s['record']['person'] for s in manifest if s['record']['role'] == 'development'} == {11, 12, 13})

results = {}
metrics = {}
for condition in ['output_only', 'relative']:
    data = read(R / (condition.upper() + '_RESULT.json'))
    results[condition] = data
    rows = data['rows']
    check(condition + ':metadata_matches_original_manifest', len(rows) == len(manifest) and all({k: r[k] for k in s} == s for r, s in zip(rows, manifest)))
    check(condition + ':checkpoint_hash', sha(R / 'runs' / (condition + '.pt')) == data['checkpoint_sha256'])
    check(condition + ':history_reaches_80', data['history'][-1]['epoch'] == 80 and all(math.isfinite(x['loss']) for x in data['history']))
    for i, row in enumerate(rows):
        p = row['probability']
        expected_status = ('INSUFFICIENT_OBSERVATION' if not row['usable'] else
                           'COMPONENT_OBSERVED' if p >= .8 else
                           'CONTRARY_COMPONENT_CANDIDATE' if p <= .2 else 'UNCERTAIN')
        check(f'{condition}:row_{i}:classification_and_status', math.isfinite(p) and 0 <= p <= 1 and row['class_prediction'] == int(p >= .5) and row['status'] == expected_status)
    metrics[condition] = recalculate(rows)
    check(condition + ':all_metrics', same(metrics[condition], data['metrics']))

elbow = 'right_elbow_lowers_hand_remains_raised'
traversal = 'right_hand_traverses_left_arm_toward_wrist'
alternation = 'bilateral_height_alternation'
targets = [elbow, traversal, alternation]
point = read(B / 'POINT_RESULT.json')['metrics']
oblique = next(v for k, v in point.items() if 'oblique' in k)
pool = read(B / 'POOLED_RESULT.json')['metrics']
a = metrics['relative']
b = metrics['output_only']
g = {
    'oblique_recall': oblique['usable_positive_recall'] >= .8,
    'oblique_specificity': oblique['usable_negative_specificity'] >= .8,
    'oblique_confirmed_recall': oblique['confirmed_positive_recall'] >= .5,
    'oblique_confirmed_specificity': oblique['confirmed_negative_specificity'] >= .5,
    'oblique_confident_wrong_zero': oblique['confident_wrong'] == 0,
}
for target in targets:
    native, still, gap = [a[target + '::' + v] for v in ['native', 'still', 'image_gap']]
    control = b[target + '::native']
    g.update({
        target + ':native_recall': native['usable_positive_recall'] >= .8,
        target + ':recall_not_below_pool': native['usable_positive_recall'] >= pool[target + '::native']['usable_positive_recall'],
        target + ':still_specificity': still['usable_negative_specificity'] >= .8,
        target + ':gap_all_held': gap['held'] == gap['records'],
        target + ':confirmed_recall': native['confirmed_positive_recall'] >= .5,
        target + ':natural_confident_wrong_zero': native['confident_wrong'] == 0,
        target + ':confirmed_still_specificity': still['confirmed_negative_specificity'] >= .5,
        target + ':paired_recall_retained': native['usable_positive_recall'] >= control['usable_positive_recall'],
        target + ':paired_confirmed_recall_retained': native['confirmed_positive_recall'] >= control['confirmed_positive_recall'],
        target + ':paired_confident_wrong_not_increased': native['confident_wrong'] <= control['confident_wrong'],
    })
    if control['negative']:
        g[target + ':paired_specificity_retained'] = native['usable_negative_specificity'] >= control['usable_negative_specificity']
        g[target + ':paired_confirmed_specificity_retained'] = native['confirmed_negative_specificity'] >= control['confirmed_negative_specificity']
for target in [elbow, traversal]:
    g[target + ':reverse_specificity'] = a[target + '::reverse']['usable_negative_specificity'] >= .8
    g[target + ':confirmed_reverse_specificity'] = a[target + '::reverse']['confirmed_negative_specificity'] >= .5
g['alternation_reverse_recall'] = a[alternation + '::reverse']['usable_positive_recall'] >= .8
g['alternation_confirmed_reverse_recall'] = a[alternation + '::reverse']['confirmed_positive_recall'] >= .5
g['traversal_natural_specificity'] = a[traversal + '::native']['usable_negative_specificity'] >= .8
g['traversal_confirmed_natural_specificity'] = a[traversal + '::native']['confirmed_negative_specificity'] >= .5
g['direction_reverse_better_than_pool'] = sum(a[t + '::reverse']['TN'] for t in [elbow, traversal]) > sum(pool[t + '::reverse']['TN'] for t in [elbow, traversal])
old = read(B / 'COMPARISON.json')['checks']
g['at_least_one_failed_gate_improved'] = any(not v and g[k] for k, v in old.items())
fields = ['usable_positive_recall', 'usable_negative_specificity', 'confirmed_positive_recall', 'confirmed_negative_specificity']
improvements = [dict(group=k, metric=f, output_only=b[k][f], relative=a[k][f]) for k in a if k.endswith(('::native', '::reverse', '::still')) for f in fields if b[k][f] is not None and a[k][f] > b[k][f]]
g['relative_has_added_contribution_over_matched_control'] = bool(improvements)
comparison = read(R / 'COMPARISON.json')
check('comparison_all_gates', g == comparison['checks'])
check('comparison_pass_status', comparison['passed'] == all(g.values()) and comparison['status'] == ('SCOPED_RELATIVE_GATES_PASSED' if all(g.values()) else 'RELATIVE_GATES_NOT_MET'))
complete = read(R / 'COMPLETE.json')
check('completion_scope', complete['status'] == comparison['status'] and complete['new_temporal_runs'] == 2 and complete['whole_goal_complete'] is False)

errors = []
for row in results['relative']['rows']:
    if row['record']['role'] != 'development' or not row['mask']:
        continue
    if row['variant'] not in ['native', 'still'] or row['class_prediction'] == row['y']:
        continue
    errors.append({k: row[k] for k in ['variant', 'y', 'probability', 'status', 'native_extent']} | {k: row['record'][k] for k in ['clip_id', 'target', 'kind']})
report = {
    'reviewer': '/root/retention_direction_review',
    'verification': 'Independent standard-library arithmetic from saved probabilities; no training/inference/import of experiment or model code.',
    'passed': all(c['passed'] for c in checks),
    'check_count': len(checks),
    'failed_checks': [c for c in checks if not c['passed']],
    'condition_rows': {k: len(v['rows']) for k, v in results.items()},
    'comparison_gate_count': len(g),
    'failed_experiment_gates': [k for k, v in g.items() if not v],
    'matched_control_improvements': improvements,
    'remaining_native_or_still_classification_errors': errors,
    'scope': 'Arithmetic verification passing does not mean experiment gates pass. Repeated DEV and geometry controls only; no deployment/learner-error/independent-generalization claim.',
    'frozen_tensor_equality': 'Not independently loaded by this verifier; experiment.py checks old non-out state equality at lines 77-78.',
    'source_hashes': {str(p): sha(p) for p in [R / 'experiment.py', R / 'FROZEN.json', R / 'OUTPUT_ONLY_RESULT.json', R / 'RELATIVE_RESULT.json', R / 'COMPARISON.json', R / 'COMPLETE.json', Path(__file__)]},
    'checks': checks,
}
with (R / 'SAVED_INDEPENDENT_VERIFIED.json').open('x', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps({k: report[k] for k in ['passed', 'check_count', 'failed_checks', 'condition_rows', 'failed_experiment_gates', 'matched_control_improvements', 'remaining_native_or_still_classification_errors']}, ensure_ascii=False, indent=2))
