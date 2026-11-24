"""Independent code/provenance checks. No model import, forward or training.

Only pure array/readout functions are extracted from AST. No production module
top-level imports, model classes, dataset loading or decoder is executed.
"""
from pathlib import Path
from collections import Counter
import ast
import hashlib
import json
import numpy as np

U = Path(__file__).resolve().parent
checks = []


def check(name, value):
    assert bool(value), name
    checks.append(name)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pure_functions(filename, names):
    tree = ast.parse((U / filename).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    ns = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), ns)
    return ns


def main():
    names = ['local_events.py', 'train_local.py', 'evaluate_local.py', 'data.py', 'prepare_data.py']
    for name in names:
        ast.parse((U / name).read_text(encoding='utf-8-sig'))
        check('syntax:' + name, True)
    train = pure_functions('train_local.py', ['class_weights'])
    groups = []
    for t in range(3):
        for person in range(9):
            bags = [dict(train_eligible=True, label=1)]
            if person < 3 + t:
                bags += [dict(train_eligible=True, label=0)] * (person + 1)
            bags += [dict(train_eligible=False, label=None)]
            groups.append(dict(target_index=t, bags=bags))
    weights, counts = train['class_weights'](groups, [0, 1, 2])
    for t in range(3):
        for y in [0, 1]:
            influence = sum(weights[f'{t}:{y}'] for g in groups if g['target_index'] == t
                            and any(b['train_eligible'] and b['label'] == y for b in g['bags'])) / len(groups)
            check(f'source_class_weight:{t}:{y}', np.isclose(influence, 1 / 6))
    f = pure_functions('evaluate_local.py', ['evidence_support', 'score_bag'])
    p = dict(ids=np.arange(12), ts=np.arange(12) * 33333,
             support=dict(row_consumed_support=[list(range(max(0, i-2), min(12, i+3))) for i in range(12)]))
    pred = np.full((12, 5), .1)
    pred[4, 0] = .91
    anchors = np.ones(12, bool)
    anchors[10] = False
    bag = dict(bag_id='synthetic', kind='reviewed_local', label=0,
               frames=[0, 11], indices=[0, 11], evidence_complete=False)
    out = f['score_bag'](pred, anchors, p, bag, 0, True)
    check('observed_positive_survives_unrelated_missing_row', out['high_confirmation'] and out['status'] == 'MOVEMENT_EVIDENCE_WITH_GAPS')
    check('partial_not_full_absence_or_completion', out['absence_assessable'] is False and out['whole_action_complete'] is None)
    check('missing_native_row_retained', out['unavailable_native_frames'] == [10] and np.isclose(out['prediction_coverage'], 11 / 12))
    changed = f['score_bag'](pred, anchors, p, dict(bag, label=None), 0, True)
    check('truth_does_not_select_probability_or_status', all(out[k] == changed[k] for k in out if k != 'label'))
    narrow = f['score_bag'](pred, anchors, p, dict(bag, indices=[2, 6], frames=[2, 6]), 0, True)
    check('same_score_column_narrow_and_broad', narrow['probability'] == out['probability'] and narrow['anchor'] == out['anchor'])
    check('temporal_context_does_not_cross_gap', 10 not in out['context_support']['temporal_native_rows'] and 11 not in out['context_support']['temporal_native_rows'])
    check('rgb_context_outside_temporal_rows_is_disclosed', 10 in out['context_support']['rgb_native_rows'])
    low = f['score_bag'](np.full((12, 5), .1), anchors, p, bag, 0, True)
    check('low_partial_is_not_confirmed_absence', low['status'] == 'INSUFFICIENT_OBSERVATION' and not low['absence_assessable'])
    empty = f['score_bag'](pred, np.zeros(12, bool), p, bag, 0, True)
    check('no_evidence_no_probability', empty['probability'] is None and empty['status'] == 'INSUFFICIENT_OBSERVATION')
    uns = f['score_bag'](pred, anchors, p, bag, 0, False)
    check('unsupported_target_hidden', uns['probability'] is None and uns['status'] == 'UNSUPPORTED_QUERY')

    motion = json.loads((U.parent / 'word100_learned_motion_order_20261102/MOTION_TARGETS.json').read_text(encoding='utf-8-sig'))
    positives = {(r['clip_id'], j): r['frames'] for r in motion['rows'] if r['kind'] == 'directed'
                 for j in range(5) if r['mask'][j] and r['y'][j] == 1}
    zeros = 0
    boundary_touch = []
    for path in sorted(U.glob('WORD*_REVIEW.json')):
        for r in json.loads(path.read_text(encoding='utf-8-sig'))['rows']:
            for q in r['queries']:
                if q['presence'] != 0:
                    continue
                zeros += 1
                a, b = r['native_frames'][0], r['native_frames'][-1]
                c, d = positives[(r['clip_id'], q['target_index'])]
                check('no_whole_positive_inside_known_zero:' + r['clip_id'] + ':' + str(q['target_index']), not a <= c <= d <= b)
                if max(a, c) <= min(b, d):
                    boundary_touch.append(dict(clip_id=r['clip_id'], target=q['target_index'], zero=[a, b], positive=[c, d]))
    result = dict(status='PASS_INDEPENDENT_STATIC_AND_PURE_READOUT_CHECKS', checks=len(checks), names=checks,
                  code_pins={str(U / name): digest(U / name) for name in names},
                  known_zero_bags=zeros, positive_zero_boundary_overlap_count=len(boundary_touch),
                  boundary_overlap_interpretation='Weak positive interval does not label its final frame positive; touching endpoints are not contradictory dense labels.',
                  model_module_imports=0, model_instances=0, model_forwards=0, checkpoint_loads=0,
                  training=0, video_decodes=0, scientific_generalization_proven=False)
    with (U / 'INDEPENDENT_STATIC_CHECKS.json').open('x', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status=result['status'], checks=len(checks), known_zero=zeros,
                          endpoint_overlaps=len(boundary_touch)), ensure_ascii=False))


if __name__ == '__main__':
    main()
