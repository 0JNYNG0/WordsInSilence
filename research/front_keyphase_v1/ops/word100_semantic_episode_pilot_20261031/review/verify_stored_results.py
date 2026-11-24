"""Independent saved-artifact arithmetic. No observer/model forward or training."""
import collections
import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np

U = Path(__file__).resolve().parents[1]
OLD = U.parent / 'word100_stage_feedback_cycle_20261031'
CLASSES = ['up', 'down', 'other']
SEEDS = [42, 43, 44]
failures = []
checks = collections.Counter()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def check(group, name, ok, detail=None):
    checks[group] += 1
    if not bool(ok):
        failures.append({'group': group, 'name': name, 'detail': detail})

def equal(group, name, left, right, tol=1e-9):
    if isinstance(left, dict) and isinstance(right, dict):
        check(group, name + ':keys', set(left) == set(right))
        for k in set(left) & set(right):
            equal(group, name + '.' + k, left[k], right[k], tol)
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        check(group, name + ':length', len(left) == len(right))
        for i, (a, b) in enumerate(zip(left, right)):
            equal(group, f'{name}[{i}]', a, b, tol)
    elif isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        check(group, name, left is not None and right is not None and math.isclose(float(left), float(right), abs_tol=tol, rel_tol=tol), {'computed': left, 'stored': right})
    else:
        check(group, name, left == right, {'computed': left, 'stored': right})

def metrics(truth, prediction):
    cm = [[0 for _ in range(4)] for _ in range(3)]
    for t, p in zip(truth, prediction):
        cm[int(t)][int(p) if p >= 0 else 3] += 1
    correct = sum(cm[i][i] for i in range(3))
    unknown = sum(row[3] for row in cm)
    f1 = []
    for c in range(3):
        actual = sum(cm[c])
        proposed = sum(row[c] for row in cm)
        f1.append(2 * cm[c][c] / (actual + proposed) if actual + proposed else 0.)
    return {'n': len(truth), 'correct': correct, 'accuracy': correct / len(truth),
            'macro_f1': sum(f1) / 3, 'class_f1': f1, 'unknown': unknown,
            'conditional_accuracy': correct / (len(truth) - unknown) if len(truth) > unknown else None,
            'confusion': cm, 'classes': CLASSES, 'columns': CLASSES + ['unknown']}

def real_hand(a):
    return [bool(a['mask'][i, 28] and a['mask'][i, 29] and math.isfinite(float(a['x'][i, 28])) and math.isfinite(float(a['x'][i, 29]))) for i in range(len(a['frame_id']))]

def eligibility(a, ids):
    supported = real_hand(a)
    flags = [supported[i] for i in ids]
    cov = sum(flags) / len(flags)
    continuous = all(int(a['frame_id'][j]) == int(a['frame_id'][i]) + 1 and int(a['timestamp_us'][j]) - int(a['timestamp_us'][i]) <= 150000 for i, j in zip(ids, ids[1:]))
    return len(ids) >= 3 and continuous and cov >= .75 and any(flags[:3]) and any(flags[-3:]), cov

def saved_baselines(a, ids, old, center=None):
    ok, _ = eligibility(a, ids)
    flags = real_hand(a)
    if not ok or (center is not None and not flags[center]):
        return -1, -1
    good = [i for i in ids if flags[i]]
    p_up = math.fsum(float(old[i, 5]) for i in good) / len(good)
    p_down = math.fsum(float(old[i, 6]) for i in good) / len(good)
    old_result = (0 if p_up >= p_down else 1) if max(p_up, p_down) >= .5 else 2
    k = min(3, max(1, len(good) // 3))
    delta = statistics.median(float(a['x'][i, 29]) for i in good[-k:]) - statistics.median(float(a['x'][i, 29]) for i in good[:k])
    geometry = 0 if delta < -.03 else 1 if delta > .03 else 2
    return old_result, geometry

def proposals(pred, a):
    groups = []
    for i, state in enumerate(pred):
        if not groups or int(state) != groups[-1][0] or int(a['frame_id'][i]) != int(a['frame_id'][i - 1]) + 1 or int(a['timestamp_us'][i]) - int(a['timestamp_us'][i - 1]) > 150000:
            groups.append([int(state), [i]])
        else:
            groups[-1][1].append(i)
    out = []
    for state, ids in groups:
        first, last = ids[0], ids[-1]
        out.append({'state': 'unknown' if state < 0 else CLASSES[state],
                    'start_frame': int(a['frame_id'][first]), 'end_frame': int(a['frame_id'][last]),
                    'start_s': int(a['timestamp_us'][first]) / 1e6, 'end_s': int(a['timestamp_us'][last]) / 1e6,
                    'frames': len(ids), 'short_proposal': len(ids) < 3})
    return out

def bracket_distance(value, bracket):
    if bracket[0] <= value <= bracket[1]:
        return 0
    return min(abs(value - endpoint) for endpoint in bracket)

frozen = read(U / 'FROZEN.json')
prepared = read(U / 'PREPARED.json')
evaluation = read(U / 'EVALUATION.json')
events = prepared['events']
auth = read(U / 'AUTHORIZATION.json')
rows = {r['alias']: r for r in auth['rows']}
start_hashes = {str(U / n): sha(U / n) for n in ['FROZEN.json', 'PREPARED.json', 'EVALUATION.json']}
for kind in ['pins', 'protected']:
    for path, expected in frozen[kind].items():
        check('hashes', kind + ':' + path, sha(path) == expected)

equal('protocol', 'seeds', SEEDS, frozen['seeds'])
equal('protocol', 'epochs', 160, frozen['epochs'])
equal('protocol', 'parameters', 24 * 41 * 3 + 24 + 24 * 24 * 3 + 24 + 24 * 3 + 3, frozen['parameters'])
data = {alias: dict(np.load(r['input'], allow_pickle=False)) for alias, r in rows.items()}
oldmeta = read(OLD / 'LEARNING_INPUTS.json')
oldarray = np.load(OLD / 'runs/pose_capacity_matched_42/probabilities.npy', mmap_mode='r')
lookup = {r['alias']: (i, r) for i, r in enumerate(oldmeta['rows'])}
equal('old_alignment', 'up_down_names', ['R_motion_up', 'R_motion_down'], oldmeta['targets'][5:7])
old = {}
for alias, a in data.items():
    i, oldrow = lookup[alias]
    equal('old_alignment', alias + ':input_sha', rows[alias]['input_sha256'], oldrow['input_sha256'])
    equal('old_alignment', alias + ':length', len(a['frame_id']), oldrow['frames'])
    check('original_frames', alias + ':zero_based_full_ids', np.array_equal(a['frame_id'], np.arange(len(a['frame_id']))))
    check('original_frames', alias + ':pts_monotonic', bool((np.diff(a['timestamp_us']) > 0).all()))
    old[alias] = oldarray[i, :len(a['frame_id'])]
    check('old_alignment', alias + ':real_frames_only', len(old[alias]) == len(a['frame_id']) and bool(np.isfinite(old[alias]).all()))

rebuilt = []
for role, file in [('fit', 'LABELS_FIT.json'), ('development', 'LABELS_DEV.json')]:
    for c in read(U / file)['clips']:
        equal('label_roles', c['alias'], role, rows[c['alias']]['role'])
        for j, e in enumerate(c['events']):
            if e['state'] in CLASSES:
                rebuilt.append(dict(e, alias=c['alias'], role=role, event_id=f"{c['alias']}_{j:02}"))
equal('label_roles', 'total_event_count', len(rebuilt), len(events))
core_ids = []
baseline = {'old': [], 'geometry': []}
for i, (label, e) in enumerate(zip(rebuilt, events)):
    for key, value in label.items():
        equal('label_roles', f'{i}:{key}', value, e[key])
    a = data[e['alias']]
    ids = [j for j, fid in enumerate(a['frame_id']) if e['core_start_frame'] <= fid <= e['core_end_frame']]
    core_ids.append(ids)
    ok, cov = eligibility(a, ids)
    equal('eligibility', e['event_id'] + ':eligible', ok, e['eligible'])
    equal('eligibility', e['event_id'] + ':coverage', cov, e['input_coverage'])
    equal('eligibility', e['event_id'] + ':frames', len(ids), e['frames'])
    equal('eligibility', e['event_id'] + ':class', CLASSES.index(e['state']), e['class_index'])
    o, g = saved_baselines(a, ids, old[e['alias']])
    baseline['old'].append(o); baseline['geometry'].append(g)
    equal('baseline_arithmetic', e['event_id'] + ':old', o, evaluation['events'][i]['old'])
    equal('baseline_arithmetic', e['event_id'] + ':geometry', g, evaluation['events'][i]['geometry'])

fit_ids = [e['event_id'] for e in events if e['role'] == 'fit' and e['eligible']]
dev_ids = {e['event_id'] for e in events if e['role'] == 'development'}
equal('training', 'eligible_fit_count', 31, len(fit_ids))
training = []
window_predictions = {}
automatic = {}
artifact_hashes = {}
for seed in SEEDS:
    d = U / 'runs' / str(seed)
    done, trace = read(d / 'COMPLETE.json'), read(d / 'TRAIN_TRACE.json')
    equal('training', f'{seed}:complete_seed', seed, done['seed'])
    equal('training', f'{seed}:trace_seed', seed, trace['seed'])
    equal('training', f'{seed}:final_epoch', 160, done['fixed_epoch'])
    equal('training', f'{seed}:steps', 160, trace['optimizer_steps'])
    equal('training', f'{seed}:trace_length', 160, len(trace['history']))
    equal('training', f'{seed}:fit_count', 31, trace['fit_count'])
    equal('training', f'{seed}:dev_seen', 0, trace['development_examples_seen'])
    for j, epoch in enumerate(trace['history'], 1):
        equal('training', f'{seed}:epoch_number:{j}', j, epoch['epoch'])
        equal('training', f'{seed}:exposure:{j}', fit_ids, epoch['event_ids'])
        check('training', f'{seed}:no_dev:{j}', not dev_ids.intersection(epoch['event_ids']))
        check('training', f'{seed}:finite_loss:{j}', math.isfinite(epoch['loss']))
    equal('training', f'{seed}:final_loss', trace['history'][-1]['loss'], done['final_training_loss'])
    for name, field in [('head.pt', 'checkpoint_sha256'), ('TRAIN_TRACE.json', 'trace_sha256')]:
        actual = sha(d / name)
        equal('hashes', f'{seed}:{field}', actual, done[field])
        artifact_hashes[str(d / name)] = actual
    p = np.load(d / 'window_probabilities.npy', allow_pickle=False)
    saved = np.load(d / 'window_predictions.npy', allow_pickle=False)
    check('window_predictions', f'{seed}:probability_shape', p.shape == (len(events), 3))
    check('window_predictions', f'{seed}:probability_domain', np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and np.allclose(p.sum(1), 1., atol=1e-6))
    recomputed = np.array([int(np.argmax(row)) if e['eligible'] else -1 for row, e in zip(p, events)])
    check('window_predictions', f'{seed}:argmax_mask', np.array_equal(saved, recomputed))
    for i, e in enumerate(events):
        equal('window_predictions', f'{seed}:{e["event_id"]}:json', int(recomputed[i]), evaluation['events'][i]['candidate'][str(seed)])
    window_predictions[f'candidate_{seed}'] = recomputed.tolist()
    for name in ['window_probabilities.npy', 'window_predictions.npy', 'COMPLETE.json']:
        artifact_hashes[str(d / name)] = sha(d / name)
    training.append({'seed': seed, 'epochs': 160, 'fit_windows_per_epoch': 31, 'exposures': 4960,
                     'development_exposures': 0, 'seconds': done['seconds'], 'checkpoint_sha256': done['checkpoint_sha256']})
    for alias, a in data.items():
        path = d / 'auto' / (alias + '.npz')
        z = dict(np.load(path, allow_pickle=False)); automatic[seed, alias] = z
        n = len(a['frame_id'])
        check('automatic_arrays', f'{seed}:{alias}:lengths', all(len(v) == n for v in z.values()))
        check('automatic_arrays', f'{seed}:{alias}:probabilities', z['prob'].shape == (n, 3) and np.isfinite(z['prob']).all() and np.allclose(z['prob'].sum(1), 1., atol=1e-6))
        flags = real_hand(a)
        for center in range(n):
            ids = list(range(max(0, center - 6), min(n, center + 7)))
            ok, cov = eligibility(a, ids); ok = ok and flags[center]
            pred = int(np.argmax(z['prob'][center])) if ok else -1
            equal('automatic_arrays', f'{seed}:{alias}:{center}:eligible', ok, bool(z['eligible'][center]))
            equal('automatic_arrays', f'{seed}:{alias}:{center}:coverage', cov, float(z['coverage'][center]))
            equal('automatic_arrays', f'{seed}:{alias}:{center}:prediction', pred, int(z['pred'][center]))
            o, g = saved_baselines(a, ids, old[alias], center)
            equal('automatic_baselines', f'{seed}:{alias}:{center}:old', o, int(z['old'][center]))
            equal('automatic_baselines', f'{seed}:{alias}:{center}:geometry', g, int(z['geometry'][center]))
        artifact_hashes[str(path)] = sha(path)

all_predictions = dict(baseline, **window_predictions)
known_metrics = []
for condition, pred in all_predictions.items():
    for role in ['fit', 'development']:
        ids = [i for i, e in enumerate(events) if e['role'] == role]
        computed = metrics([events[i]['class_index'] for i in ids], [pred[i] for i in ids])
        reported = next(v for v in evaluation['known_windows'] if v['condition'] == condition and v['role'] == role)
        for k, value in computed.items():
            equal('known_metrics', condition + ':' + role + ':' + k, value, reported[k])
        known_metrics.append(dict(condition=condition, role=role, **computed))

auto_metrics = []
boundary_summary = []
for condition in ['old', 'geometry'] + [f'candidate_{s}' for s in SEEDS]:
    truth, pred, boundary = [], [], []
    seed = int(condition.split('_')[-1]) if condition.startswith('candidate') else 42
    key = 'pred' if condition.startswith('candidate') else condition
    for i, e in enumerate(events):
        if e['role'] != 'development':
            continue
        a = data[e['alias']]; states = automatic[seed, e['alias']][key]
        part = [int(states[j]) for j in core_ids[i]]
        known = [v for v in part if v >= 0]
        counts = [known.count(j) for j in range(3)]
        result = max(range(3), key=lambda j: counts[j]) if len(known) >= .75 * len(part) else -1
        truth.append(e['class_index']); pred.append(result)
        if e['state'] == 'other':
            continue
        possibilities = [r for r in proposals(states, a) if r['state'] == e['state'] and max(r['start_frame'], e['core_start_frame']) <= min(r['end_frame'], e['core_end_frame'])]
        matched = max(possibilities, key=lambda r: min(r['end_frame'], e['core_end_frame']) - max(r['start_frame'], e['core_start_frame']) + 1, default=None)
        boundary.append({'event_id': e['event_id'], 'match': matched,
                         'start_outside_bracket_frames': None if matched is None else bracket_distance(matched['start_frame'], e['start_bracket']),
                         'end_outside_bracket_frames': None if matched is None else bracket_distance(matched['end_frame'], e['end_bracket'])})
    computed = metrics(truth, pred)
    reported = next(v for v in evaluation['automatic_core'] if v['condition'] == condition)
    for k, value in computed.items():
        equal('auto_core_metrics', condition + ':' + k, value, reported[k])
    equal('boundary_arithmetic', condition, boundary, reported['boundary'])
    check('boundary_scope', condition + ':only_8_main', len(boundary) == 8)
    auto_metrics.append(dict(condition=condition, **computed))
    boundary_summary.append({'condition': condition, 'main_event_denominator': 8,
                             'same_state_overlap_matches': sum(b['match'] is not None for b in boundary),
                             'both_boundaries_inside_brackets': sum(b['match'] is not None and b['start_outside_bracket_frames'] == 0 and b['end_outside_bracket_frames'] == 0 for b in boundary),
                             'detail': boundary, 'interpretation': 'Descriptive sampled main-event bracket checks, not full-clip event accuracy or linguistic stage truth.'})

dev = {r['condition']: r for r in known_metrics if r['role'] == 'development'}
auto = {r['condition']: r for r in auto_metrics}
candidate_f1 = sum(dev[f'candidate_{s}']['macro_f1'] for s in SEEDS) / 3
candidate_auto_f1 = sum(auto[f'candidate_{s}']['macro_f1'] for s in SEEDS) / 3
gate_parts = {'known_core_exceeds_old_by_0_05': candidate_f1 >= dev['old']['macro_f1'] + .05,
              'known_core_at_least_geometry': candidate_f1 >= dev['geometry']['macro_f1'],
              'other_count_no_regression_each_seed': all(dev[f'candidate_{s}']['confusion'][2][2] >= dev['old']['confusion'][2][2] for s in SEEDS),
              'auto_core_no_regression_vs_old': candidate_auto_f1 >= auto['old']['macro_f1']}
equal('gate', 'quality_gate', all(gate_parts.values()), evaluation['provisional_quality_gate_pass'])
equal('gate', 'controls_required', True, evaluation['final_gate_requires_controls'])
equal('gate', 'boundary_not_adopted', False, evaluation['boundary_adoption_pass'])
for p, h in start_hashes.items():
    equal('preservation', p, h, sha(p))

identical = all(window_predictions['candidate_42'] == window_predictions[f'candidate_{s}'] for s in [43, 44])
frame_differences = {str(s): sum(int(np.count_nonzero(automatic[42, alias]['pred'] != automatic[s, alias]['pred'])) for alias in rows) for s in [43, 44]}
changed_vs_old = []
for i, e in enumerate(events):
    if e['role'] == 'development' and all_predictions['old'][i] != all_predictions['candidate_42'][i]:
        changed_vs_old.append({'event_id': e['event_id'], 'truth': e['state'], 'old': all_predictions['old'][i], 'candidate': all_predictions['candidate_42'][i], 'geometry': all_predictions['geometry'][i]})

report = {'status': 'PASS' if not failures else 'FAIL', 'reviewer': 'rgb_labels_b',
          'created_at_kst': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
          'checks_total': sum(checks.values()), 'checks_by_group': dict(checks), 'failures': failures,
          'scope': 'Independent arithmetic from saved probabilities/predictions/input masks/frame metadata, frozen labels and logs. No project metric/inference imports, model forward, training, new predictions or label edits.',
          'training': {'runs': training, 'optimizer_steps_total': 480, 'exposures_total': 14880,
                       'fit_windows': 31, 'all_annotated_fit_windows': 42, 'all_development_windows': 16,
                       'new_parameters': 4803, 'gradient_exposure_evidence': 'Pinned fit-only code plus exact31 fit-event IDs in all160 trace entries for each seed; no dev ID appears.'},
          'hashes_verified': {'frozen_pins': len(frozen['pins']), 'protected': len(frozen['protected']), 'new_checkpoints': 3, 'new_training_traces': 3},
          'known_core_metrics': known_metrics, 'automatic_core_metrics': auto_metrics,
          'boundary_checks': boundary_summary, 'gate_components': gate_parts,
          'provisional_quality_gate_pass': all(gate_parts.values()),
          'headline_ko': '새 구간 판정부는 기존 저장 head보다 개발 정답 수가4/16에서7/16으로 늘었지만 기하 기준8/16보다 낮아 사전 gate에 실패했다. 3seed의 구간 분류 결과는 같고, 전체 분모16과 unknown8을 유지했다. 지원8개 조건부 정확도87.5%는 전체 정확도43.75%를 대신하지 않는다.',
          'automatic_caution_ko': '자동 창으로 얻은 core 요약은 후보7/16(unknown9), 기존3/16, 기하4/16이다. 경계/전체 순서 완성은 검증되지 않았고 미주석 구간을 false-positive 정답으로 사용하지 않았다.',
          'seed_consistency': {'all_known_window_labels_identical': identical, 'auto_frame_differences_vs_seed42': frame_differences, 'note': 'Equal aggregate/core labels do not require equal probabilities or every automatic frame label.'},
          'development_changed_vs_old': changed_vs_old,
          'interpretation_limits': ['Sampled same-Codex physical interval drafts; not independent human or educational truth.', 'Previously consumed development contexts; no new generalization claim.', 'Test/check counts are not accuracy.', 'F1 includes unknown as missed truth; supported-only accuracy is separately labelled.', 'Boundary offsets apply only to8 selected movement cores; no complete sequence, required repetition, or unannotated false-positive labels.'],
          'new_artifact_hashes': artifact_hashes,
          'verifier_sha256': sha(__file__)}
out = U / 'review' / 'STORED_RESULT_VERIFICATION.json'
with out.open('x', encoding='utf8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
text = '\n'.join([
    f"저장 결과 독립검산: {report['status']} ({sum(checks.values())}개 검사, 불일치 {len(failures)}개). 검사 수는 정확도가 아니다.",
    '고정3seed×160epoch에서 매번 fit31개 창을 사용: 총480 optimizer step, 14,880 창 노출. dev 노출0; 체크포인트/trace SHA와 보호32개 SHA 검산.',
    report['headline_ko'],
    f"전체 core macroF1: 기존 {dev['old']['macro_f1']*100:.4f}%, 새 후보 {candidate_f1*100:.4f}%, 기하 {dev['geometry']['macro_f1']*100:.4f}%. 새 후보는 기하 기준 이상 조건을 충족하지 못해 미채택이다.",
    report['automatic_caution_ko'],
    '저장값 산술만 수행했다. 모델 재추론·학습·새 품질 예측·주석 수정 없음. 물리 개발 결과이며 교육적 정오/허용폭과 운영 적용의 검증은 아니다.'
]) + '\n'
with (U / 'review' / 'STORED_RESULT_VERIFICATION.txt').open('x', encoding='utf8') as f:
    f.write(text)
print(json.dumps({'status': report['status'], 'checks': sum(checks.values()), 'failures': failures[:8], 'quality_gate': all(gate_parts.values()), 'window_seed_labels_identical': identical, 'auto_frame_differences': frame_differences}, ensure_ascii=False))
