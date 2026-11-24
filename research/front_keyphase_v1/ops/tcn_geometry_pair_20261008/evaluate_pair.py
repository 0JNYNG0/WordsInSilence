"""Score the two frozen geometry arms once; reuse stored TCN832 scores.

No training, camera, extraction, source-data changes, or default-model changes.
All writes are exclusive and confined to this experiment's run directory.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

H = Path(__file__).resolve().parent
L = H.parents[1]
EXPECTED_RUN = L / 'runs/tcn_geometry_pair_v1'
PRIOR = L / 'ops/partial_encoder_pair_20260922'
PROBE = L / 'ops/ksl77_development_probe_20260926'
PROBE_RUN = L / 'runs/ksl77_development_probe_v1'
EXPECTED_COUNTS = {'annotation_dev': 699, 'whole_dev': 74, 'public': 32, 'mp': 14, 'ksl36': 36}
OUTPUTS = ('EVALUATION_STARTED.json', 'evaluation_scores.npz', 'EVALUATION_ROWS.json',
           'evaluation.json', 'EVALUATION_COMPLETE.json')
MODELS = ('control', 'treatment', 'baseline')
TARGET_LABELS = ('감사합니다', '나', '모르다', '알다', '좋다', '감', '공부', '인사', '자다',
                 '죄송하다', '맥주', '먹다', '물', '너', '빵', '사랑', '싫다', '운동')


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def key(row):
    return row['group'], row['id'], row.get('condition', 'intact')


def safe_path(path):
    p = Path(path).resolve()
    require(p.is_relative_to(L.parent), 'input outside research workspace: ' + str(p))
    require('/runs/ses_' not in p.as_posix().lower(), 'user session input is forbidden')
    return p


def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f'{a.dtype.str}|{a.shape}|'.encode('ascii'))
    h.update(a.tobytes())
    return h.hexdigest()


def extra_from_x(x, hand_distances):
    require(x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 280,
            'expected float32 [T,280] channels')
    require(len(x) > 0 and np.isfinite(x).all(), 'invalid model input')
    valid = x[:, 237:279].reshape(-1, 2, 21)
    require(np.all((valid == 0) | (valid == 1)), 'nonbinary local observation mask')
    return hand_distances(x[:, 102:186].reshape(-1, 2, 21, 2), valid.astype(bool))


def load_inputs(rows, classes, pins):
    sys.path[:0] = [str(PRIOR), str(L), str(L / 'src'),
                   str(L.parent / 'sign_model_pilot_v1/src')]
    spec = importlib.util.spec_from_file_location('_geometry_original_io', PRIOR / 'local_io.py')
    io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(io)
    policy = io.policy()
    groups, meta = io.evaluation_inputs(policy)
    pins[str(PRIOR / 'local_io.py')] = sha(PRIOR / 'local_io.py')
    pins[str(PRIOR / 'comparison_policy.json')] = sha(PRIOR / 'comparison_policy.json')
    for p in policy['pins']:
        pins[str(Path(p))] = sha(p)
    lookup = {}
    labels = {}
    for group, xs in groups.items():
        require(len(xs) == len(meta[group]) == EXPECTED_COUNTS[group], 'original group count changed')
        for old, x in zip(meta[group], xs):
            k = (group, old['id'], old.get('condition', 'intact'))
            require(k not in lookup, 'duplicate original input key')
            lookup[k] = x
            labels[k] = old.get('label', old.get('true', old.get('reference_label')))
    plan_path, result_path = PROBE / 'FROZEN_PLAN.json', PROBE / 'RESULT.json'
    plan = {r['id']: r for r in read(plan_path)['rows']}
    prior_result = {r['id']: r for r in read(result_path)['rows']}
    pins[str(plan_path)], pins[str(result_path)] = sha(plan_path), sha(result_path)
    require(len(plan) == len(prior_result) == 36, 'KSL manifest count changed')
    input_rows, xs = [], []
    for index, row in enumerate(rows):
        k = key(row)
        safe_path(row['path'])
        require(sha(row['path']) == row['sha256'], 'row source hash mismatch: ' + row['id'])
        pins[str(Path(row['path']))] = row['sha256']
        if row['group'] == 'ksl36':
            old, result = plan[row['id']], prior_result[row['id']]
            require(old['candidate_label'] == row['label'], 'KSL label mismatch')
            require(old['previous_diagnostic_exposure'] == row['exposed'], 'KSL exposure mismatch')
            require(row['signer_id'] is None and row['role'] == 'evaluation', 'KSL role/signer changed')
            path = safe_path(PROBE_RUN / row['id'] / 'model_input.npz')
            require(path.parent.parent == PROBE_RUN.resolve(), 'unsafe KSL input id')
            digest = sha(path)
            with np.load(path, allow_pickle=False) as z:
                x = z['x'].copy()
            require(digest == result['input_sha256'], 'KSL saved input hash mismatch: ' + row['id'])
            require(len(x) == result['input_frames'], 'KSL frame count mismatch')
            pins[str(path)] = digest
        else:
            require(k in lookup and labels[k] == row['label'], 'row/input identity mismatch: ' + str(k))
            old = next(r for r in meta[row['group']]
                       if r['id'] == row['id'] and r.get('condition', 'intact') == row['condition'])
            if 'signer_id' in old:
                require(old['signer_id'] == row.get('signer_id'), 'row signer mismatch')
            elif row.get('signer_id') is not None:
                match = re.search(r'_REAL(\d+)_', row['id'])
                require(match is not None and int(match.group(1)) == row['signer_id'],
                        'signer lacks metadata or source-id support')
            x = lookup.pop(k)
            path, digest = None, None
        require(classes[row['y']] == row['label'], 'class-label mismatch')
        require(x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 280
                and len(x) > 0 and np.isfinite(x).all(), 'invalid channels for ' + row['id'])
        require(len(x) == row['end_exclusive'] - row['start'], 'frozen frame interval differs: ' + row['id'])
        xs.append(x)
        input_rows.append({'evaluation_index': index, 'id': row['id'], 'group': row['group'],
                           'condition': row['condition'], 'frames': len(x),
                           'input_tensor_sha256': array_sha(x),
                           'saved_model_input_path': str(path) if path else None,
                           'saved_model_input_sha256': digest})
    require(not lookup, 'unused original evaluation input rows')
    return xs, input_rows


def subsets(rows):
    result = {}
    for group in EXPECTED_COUNTS:
        indices = [i for i, r in enumerate(rows) if r['group'] == group]
        result['group=' + group] = np.asarray(indices, dtype=np.int64)
        if group in ('public', 'mp'):
            for condition in sorted({rows[i]['condition'] for i in indices}):
                result[f'group={group}|condition={condition}'] = np.asarray(
                    [i for i in indices if rows[i]['condition'] == condition], dtype=np.int64)
    ksl = result['group=ksl36']
    result['group=ksl36|primary_unexposed'] = np.asarray([i for i in ksl if not rows[i]['exposed']], dtype=np.int64)
    result['group=ksl36|exposed'] = np.asarray([i for i in ksl if rows[i]['exposed']], dtype=np.int64)
    for label in ('공부', '먹다'):
        result['group=ksl36|primary_unexposed|label=' + label] = np.asarray(
            [i for i in ksl if not rows[i]['exposed'] and rows[i]['label'] == label], dtype=np.int64)
    return result


def metric(pred, y, indices, rows, classes):
    correct = pred[indices] == y[indices]
    per_word, recalls = {}, []
    for c in sorted(set(y[indices].tolist())):
        mask = y[indices] == c
        n, count = int(mask.sum()), int(correct[mask].sum())
        per_word[classes[c]] = {'n': n, 'correct': count, 'recall': count / n}
        recalls.append(Fraction(count, n))
    macro = sum(recalls, Fraction()) / len(recalls) if recalls else None
    per_signer = {}
    for signer in sorted({rows[i]['signer_id'] for i in indices if rows[i]['signer_id'] is not None}):
        mask = np.asarray([rows[i]['signer_id'] == signer for i in indices])
        n, count = int(mask.sum()), int(correct[mask].sum())
        per_signer[str(signer)] = {'n': n, 'correct': count, 'accuracy': count / n}
    return {'n': len(indices), 'correct': int(correct.sum()),
            'accuracy': float(correct.mean()) if len(indices) else None,
            'macro_recall': float(macro) if macro is not None else None,
            'macro_recall_exact': {'numerator': macro.numerator, 'denominator': macro.denominator}
            if macro is not None else None, 'per_word': per_word, 'per_signer': per_signer}


def comparison(after, before, y, indices, rows, classes):
    def details(take):
        return [{'index': int(i), 'id': rows[i]['id'], 'group': rows[i]['group'],
                 'condition': rows[i]['condition'], 'label': rows[i]['label'],
                 'before': classes[before[i]], 'after': classes[after[i]]} for i in take]
    good, old_good = after[indices] == y[indices], before[indices] == y[indices]
    gains = details(indices[good & ~old_good])
    regressions = details(indices[old_good & ~good])
    changed = details(indices[after[indices] != before[indices]])
    return {'delta': int(good.sum()) - int(old_good.sum()), 'gain_count': len(gains),
            'regression_count': len(regressions), 'changed_count': len(changed),
            'gains': gains, 'regressions': regressions, 'changed': changed}


def summarize(arrays, rows, classes):
    y = arrays['y']
    target = np.asarray([i for i, label in enumerate(classes) if label in TARGET_LABELS], dtype=np.int64)
    require(len(target) == 18, 'target18 class membership differs')
    predictions = {name: arrays[name + '_scores'].argmax(axis=1) for name in MODELS}
    restricted = {name: target[arrays[name + '_scores'][:, target].argmax(axis=1)] for name in MODELS}
    target_gt = np.isin(y, target)
    out = {}
    for group, indices in subsets(rows).items():
        out[group] = {}
        for scope, take, pred in (('full93', indices, predictions),
                                  ('targetGT_full93', indices[target_gt[indices]], predictions),
                                  ('restricted18_separate_task', indices[target_gt[indices]], restricted)):
            out[group][scope] = {
                'models': {name: metric(pred[name], y, take, rows, classes) for name in MODELS},
                'comparisons': {kind: comparison(pred['treatment'], pred[other], y, take, rows, classes)
                                for kind, other in (('method', 'control'), ('project', 'baseline'))}}
    return out, target.tolist()


def decide(groups):
    def m(group, arm='treatment'):
        return groups[group]['full93']['models'][arm]
    native, ksl = 'group=annotation_dev', 'group=ksl36|primary_unexposed'
    decisions = {}
    for kind in ('method', 'project'):
        changes = {g: groups[g]['full93']['comparisons'][kind]['delta'] for g in (native, ksl)}
        gates = {'native699_non_decrease': changes[native] >= 0,
                 'ksl34_non_decrease': changes[ksl] >= 0,
                 'at_least_one_increase': any(v > 0 for v in changes.values())}
        decisions[kind] = {'passed': all(gates.values()), 'criteria': gates}
    gates = {}
    for signer, count in ((11, 229), (12, 234), (13, 228)):
        require(m(native, 'baseline')['per_signer'][str(signer)]['correct'] == count, 'stored signer baseline changed')
        gates[f'native_signer_{signer}_non_decrease'] = m(native)['per_signer'][str(signer)]['correct'] >= count
    def exact_macro(value):
        v = value['macro_recall_exact']
        return Fraction(v['numerator'], v['denominator'])
    gates['native_macro_non_decrease'] = exact_macro(m(native)) >= exact_macro(m(native, 'baseline'))
    required = {'group=whole_dev': 74, 'group=public|condition=intact': 18,
                'group=public|condition=reviewed_masked': 5, 'group=public|condition=additional_train': 9,
                'group=mp|condition=whole_clip': 7, 'group=mp|condition=provided_annotation': 7,
                'group=ksl36|primary_unexposed|label=공부': 16,
                'group=ksl36|primary_unexposed|label=먹다': 6}
    for group, count in required.items():
        require(m(group, 'baseline')['correct'] == count, 'stored preservation baseline changed: ' + group)
        gates[group + '_non_decrease'] = m(group)['correct'] >= count
    require(m(native, 'baseline')['correct'] == 691 and m(ksl, 'baseline')['correct'] == 22,
            'stored primary baseline changed')
    decisions['preservation'] = {'passed': all(gates.values()), 'criteria': gates}
    decisions['adoptable'] = all(decisions[k]['passed'] for k in ('method', 'project', 'preservation'))
    decisions['automatic_promotion_authorized'] = False
    return decisions


def main():
    config = read(H / 'CONFIG.json')
    require(sha(H / 'SCOPE.md') == config['scope_sha256'], 'frozen scope changed')
    for p, digest in config['source_pins'].items():
        require(sha(p) == digest, 'config source pin mismatch: ' + p)
    run = Path(config['run_dir']).resolve()
    require(run == EXPECTED_RUN.resolve(), 'unexpected output run directory')
    require(run.is_dir(), 'training run is absent')
    require(not any((run / name).exists() for name in OUTPUTS), 'evaluation artifact exists; refusing repeat/overwrite')
    training_path = run / 'training.json'
    training = read(training_path)
    require(training['status'] == 'BOTH_MODELS_FROZEN_BEFORE_EVALUATION', 'training not complete')
    require(training['config_sha256'] == sha(H / 'CONFIG.json'), 'config differs from training')
    require(training['feature_module_sha256'] == sha(H / 'geometry_tcn.py'), 'feature module differs from training')
    require(training['driver_sha256'] == sha(H / 'run_pair.py'), 'training driver differs')
    require(training['prepared_sha256'] == sha(H / 'prepared.npz'), 'prepared training inputs differ')
    require(training['training_arrays_sha256'] == sha(run / 'training_arrays.npz'), 'saved training arrays differ')
    require(set(training['models']) == {'control', 'treatment'}, 'both frozen models are required')
    baseline_run = safe_path(config['baseline_run'])
    rows_path, baseline_path = baseline_run / 'EVALUATION_ROWS.json', baseline_run / 'EVALUATION_SCORES.npz'
    doc = read(rows_path)
    rows, classes = doc['rows'], doc['classes']
    require(len(rows) == 855 and len(classes) == len(set(classes)) == 93, '855 rows / 93 unique classes required')
    require(dict(Counter(r['group'] for r in rows)) == EXPECTED_COUNTS, 'evaluation group counts differ')
    require(len({key(r) for r in rows}) == 855, 'duplicate evaluation row key')
    require(all(r['role'] == 'evaluation' for r in rows), 'non-evaluation role')
    require(sum(r['exposed'] for r in rows if r['group'] == 'ksl36') == 2, 'KSL diagnostic count differs')
    y = np.asarray([r['y'] for r in rows], dtype=np.int64)
    with np.load(baseline_path, allow_pickle=False) as z:
        baseline = z['baseline_scores'].copy()
        require(np.array_equal(z['y'], y), 'stored labels/order differ')
    require(baseline.dtype == np.float64 and baseline.shape == (855, 93) and np.isfinite(baseline).all(),
            'stored TCN score contract differs')
    embedded = np.asarray([r['baseline_scores'] for r in rows], dtype=np.float64)
    require(baseline.tobytes() == embedded.tobytes(), 'stored TCN archive and row scores differ')
    pins = {str(p): sha(p) for p in (H / 'CONFIG.json', H / 'SCOPE.md', H / 'ROLE_AUTHORIZATION.json',
                                    H / 'geometry_tcn.py', H / 'run_pair.py', H / 'prepared.npz',
                                    run / 'training_arrays.npz', Path(__file__), training_path, rows_path, baseline_path)}
    pins.update(config['source_pins'])
    parent_path = safe_path(config['parent_checkpoint']['path'])
    require(sha(parent_path) == config['parent_checkpoint']['sha256'], 'parent checkpoint hash mismatch')
    pins[str(parent_path)] = config['parent_checkpoint']['sha256']
    xs, diagnostics = load_inputs(rows, classes, pins)
    sys.path.insert(0, str(H))
    from geometry_tcn import GeometryTCN, hand_distances, predict
    from tools_prep_union93_candidate import rebuild_from_payload
    extras = [extra_from_x(x, hand_distances) for x in xs]
    for item, extra in zip(diagnostics, extras):
        item['extra_tensor_sha256'] = array_sha(extra)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    parent_payload = torch.load(parent_path, map_location='cpu', weights_only=True)
    require(parent_payload['classes'] == classes, 'parent classes differ')
    models, model_specs = {}, {}
    for arm in ('control', 'treatment'):
        item = training['models'][arm]
        path = Path(item['path']).resolve()
        require(path.is_relative_to(run), 'new checkpoint outside run')
        require(sha(path) == item['sha256'], 'new checkpoint hash mismatch')
        pins[str(path)] = item['sha256']
        pay = torch.load(path, map_location='cpu', weights_only=True)
        require(pay['classes'] == classes and pay['enabled'] is (arm == 'treatment'), 'arm/class identity mismatch')
        require(pay['config_sha256'] == sha(H / 'CONFIG.json'), 'checkpoint config differs')
        require(pay['registered_as_default'] is False, 'candidate default-registration flag changed')
        parent, cl = rebuild_from_payload(parent_payload)
        require(cl == classes, 'rebuilt class order differs')
        model = GeometryTCN(parent, pay['state_dict']['extra_scale'].numpy(), enabled=pay['enabled'])
        model.load_state_dict(pay['state_dict'], strict=True)
        model.eval()
        models[arm], model_specs[arm] = model, {'path': str(path), 'sha256': item['sha256']}
    # Baseline gates validate source identity before any new inference.
    baseline_only = {'y': y, **{name + '_scores': baseline for name in MODELS}}
    baseline_summary, _ = summarize(baseline_only, rows, classes)
    decide(baseline_summary)
    for p, digest in pins.items():
        require(sha(p) == digest, 'pinned input changed before evaluation: ' + p)
    started = {'started_utc': now(), 'models': model_specs, 'source_pins': pins,
               'baseline_reused': True, 'baseline_inferences': 0, 'rows': 855}
    write_json(run / 'EVALUATION_STARTED.json', started)
    arrays = {'baseline_scores': baseline, 'y': y}
    for arm in ('control', 'treatment'):
        model = models[arm]
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        score = predict(model, xs, extras, batch_size=32)
        require(score.dtype == np.float64 and score.shape == (855, 93) and np.isfinite(score).all(),
                'invalid predicted scores: ' + arm)
        require(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()),
                'model changed during evaluation: ' + arm)
        arrays[arm + '_scores'] = score
        print(json.dumps({'evaluated_arm': arm, 'rows': 855}), flush=True)
    groups, target = summarize(arrays, rows, classes)
    decisions = decide(groups)
    for p, digest in pins.items():
        require(sha(p) == digest, 'pinned input changed during evaluation: ' + p)
    with (run / 'evaluation_scores.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    with (run / 'EVALUATION_ROWS.json').open('xb') as f:
        f.write(rows_path.read_bytes())
    result = {'status': 'EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT', 'groups': groups,
              'decisions': decisions, 'target_indices': target, 'models': model_specs,
              'input_diagnostics': diagnostics,
              'provenance': {'baseline_reused': True, 'baseline_inferences': 0,
                             'baseline_scores_path': str(baseline_path), 'baseline_scores_sha256': sha(baseline_path),
                             'baseline_rows_path': str(rows_path), 'baseline_rows_sha256': sha(rows_path),
                             'source_pins': pins, 'new_arm_evaluations': 2, 'model_state_unchanged': True,
                             'evaluation_arrays_sha256': sha(run / 'evaluation_scores.npz'),
                             'training_record_sha256': sha(training_path), 'finished_utc': now()},
              'limits': ['No overall 855-row accuracy: conditions reuse original recordings.',
                         'KSL34 is repeatedly exposed development data; exposed2 is diagnostic only.',
                         'Target18 restricted ranking is a separate task, not full93 recognition.',
                         'Training references and derived hand masks do not establish new-person generalization.',
                         'No actual sign-form correctness or performance-feedback accuracy claim.'],
              'operating_model_changed': False, 'user_data_used': False, 'camera_access': False}
    write_json(run / 'evaluation.json', result)
    write_json(run / 'EVALUATION_COMPLETE.json', {
        'status': result['status'], 'finished_utc': now(), 'models': model_specs,
        'baseline_reused': True, 'baseline_inferences': 0, 'source_pins': pins,
        'files': {name: sha(run / name) for name in OUTPUTS if name != 'EVALUATION_COMPLETE.json'},
        'array_sha256': {name: array_sha(a) for name, a in arrays.items()},
        'decisions': decisions})
    print(json.dumps({'decisions': decisions}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
