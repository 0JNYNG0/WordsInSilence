"""Independent offline validation of two completed checkpoints on fixed development inputs.

Uses the existing web inference service unchanged. No training or production registration.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import copy
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
OUT = Path(__file__).resolve().parent
WEB = OUT.parents[1]
PILOT = WEB.parent / 'sign_model_pilot_v1'
OLD = Path('C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/WordsInSilence_기능완성_20260913')
SESSION = WEB / 'runs/ses_30606a1f9447'
PLAN = OUT / 'plan_LOCAL_ONLY.json'
RESULT = OUT / 'result_LOCAL_ONLY.json'
MODEL_HASHES = {
    'control': 'a9bf191e2c12e883d5f9591ec72d57d6e0f75e3f33338db601cf155f21e12c75',
    'treatment': '65e8372926d83697b5fc0227b9407a3291c55c4928245f6c6812fec3a2599b40',
}

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_new(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()

def prepare():
    if PLAN.exists() or RESULT.exists():
        raise FileExistsError('This fixed evaluation has already been prepared or executed.')
    session = read(SESSION / 'session.json')
    previous = read(OLD / 'new18_independent.json')
    old_plan = read(OLD / 'functional22_development_replay_plan.json')
    assert session['purpose'] == 'development' and session['state'] == 'CLOSED'
    assert session['participant_id'] == 'self'
    assert old_plan['session_id'] == session['session_id'] == previous['session_id']
    assert old_plan['no_training_or_tuning']
    assert len(previous['rows']) == 18
    assert set(session['trial_ids']) == {r['trial_id'] for r in previous['rows']}
    assert len({r['actual'] for r in previous['rows']}) == 18
    assert read(WEB / 'ops/native_data_pair_20260919/native_pair_result_audit.json')['status'] == 'PASS_OUTPUT_INTEGRITY'
    pins = [SESSION / 'session.json', OLD / 'new18_independent.json',
            OLD / 'functional22_development_replay_plan.json',
            WEB / 'src/wis_lab/service/neural.py', WEB / 'src/wis_lab/core/observation.py',
            PILOT / 'src/pilot/data.py', PILOT / 'src/pilot/web_input.py',
            PILOT / 'src/pilot/model_masked_v1.py', Path(__file__)]
    rows = []
    for r in previous['rows']:
        stem = SESSION / 'trials' / r['trial_id']
        obs, saved, side = (stem.with_suffix(s) for s in ('.npz', '.result.json', '.sidecar.json'))
        assert sha(obs) == read(saved)['observation_sha256']
        pins.extend([obs, saved, side])
        rows.append({'trial_id': r['trial_id'], 'actual_word_id': r['actual'],
                     'actual_label': r['label'], 'observation': str(obs),
                     'observation_sha256': sha(obs)})
    models = {}
    for arm, digest in MODEL_HASHES.items():
        path = WEB / ('runs/native_data_pair_v1/checkpoint_%s_final.pt' % arm)
        assert sha(path) == digest
        pins.append(path)
        models[arm] = {'path': str(path), 'sha256': digest, 'fixed_step': 660}
    plan = {'prepared_at_kst': now(), 'fixed_before_inference': True,
            'question': 'Does the completed native-data addition change predictions on existing MediaPipe webcam development observations?',
            'session_id': session['session_id'], 'participant_count': 1,
            'data_role': 'Existing self-participant development set already used in model/input diagnosis. Not an untouched person or final test.',
            'participant_recordings_in_training': False,
            'selection': 'All 18 previously fixed development trials, exactly once per new checkpoint; no exclusions or checkpoint selection.',
            'scopes': ['project18', 'full93'], 'models': models, 'rows': rows,
            'inference': 'Existing neural.recognise(Observation, model_key); truth not passed to model; in-memory registry entries only.',
            'protected_sha256': {str(p): sha(p) for p in pins},
            'no_training_tuning_promotion_or_new_capture': True,
            'no_reserved_test_inputs': True,
            'limits': ['One training-excluded but development-exposed participant',
                       'Self-reported word identity, not expert correctness',
                       'Not a new-person population estimate',
                       'Offline inference timing is not live feedback latency']}
    write_new(PLAN, plan)
    print(json.dumps({'status': 'PREPARED_FIXED_BEFORE_INFERENCE', 'trials': 18,
                      'models': list(models), 'participant_count': 1, 'inference_calls': 0}))

def execute():
    if RESULT.exists():
        raise FileExistsError('Evaluation already executed; do not repeat.')
    plan = read(PLAN)
    assert plan['fixed_before_inference'] and plan['no_reserved_test_inputs']
    assert all(sha(Path(p)) == d for p, d in plan['protected_sha256'].items())
    sys.path[:0] = [str(WEB / 'src'), str(PILOT / 'src')]
    import torch
    torch.set_num_threads(2)
    from wis_lab.service import neural
    from wis_lab.core.observation import Observation
    original_registry = copy.deepcopy(neural.MODELS)
    default = neural.DEFAULT_MODEL
    model_keys = {}
    class_orders = []
    for arm, spec in plan['models'].items():
        key = 'offline_native_pair_' + arm
        assert key not in neural.MODELS
        neural.MODELS[key] = dict(spec, project_words=dict(neural.PROJECT_18),
                                 model_id=key, role='offline development evaluation',
                                 status_label='오프라인 개발 평가')
        loaded = neural._load(key)
        assert loaded['status'] == 'READY'
        assert loaded['checkpoint_step'] == 660
        class_orders.append(loaded['classes'])
        model_keys[arm] = key
    assert class_orders[0] == class_orders[1] and len(class_orders[0]) == 93
    assert set(neural.PROJECT_18) == {r['actual_word_id'] for r in plan['rows']}
    rows = []
    started = time.perf_counter()
    for fixed in plan['rows']:
        obs = Observation.load(fixed['observation'])
        row = dict(fixed, arms={})
        for arm, key in model_keys.items():
            prediction = neural.recognise(obs, key)
            assert prediction['checkpoint_sha256'] == plan['models'][arm]['sha256']
            assert prediction['accepted'] is None and prediction['execution_assessment'] == 'UNCONFIGURED'
            row['arms'][arm] = prediction
        rows.append(row)
    protected_unchanged = all(sha(Path(p)) == d for p, d in plan['protected_sha256'].items())
    assert protected_unchanged and neural.DEFAULT_MODEL == default
    assert all(neural.MODELS[k] == v for k, v in original_registry.items())
    summary = {}
    for scope, scope_key, truth_key in [('project18', 'project18', 'actual_word_id'),
                                        ('full93', 'full', 'actual_label')]:
        def pred(r, arm):
            return r['arms'][arm].get('scopes', {}).get(scope_key, {}).get('top1')
        summary[scope] = {'n': len(rows), 'arms': {}}
        for arm in model_keys:
            summary[scope]['arms'][arm] = {
                'correct': sum(pred(r, arm) == r[truth_key] for r in rows),
                'observation_or_inference_failures': sum(r['arms'][arm]['model_status'] != 'READY' for r in rows),
                'errors': [{'word_id': r['actual_word_id'], 'actual': r[truth_key], 'predicted': pred(r, arm)}
                           for r in rows if pred(r, arm) != r[truth_key]]}
        summary[scope]['gains'] = [r['actual_word_id'] for r in rows
                                  if pred(r, 'control') != r[truth_key] and pred(r, 'treatment') == r[truth_key]]
        summary[scope]['regressions'] = [r['actual_word_id'] for r in rows
                                        if pred(r, 'control') == r[truth_key] and pred(r, 'treatment') != r[truth_key]]
        summary[scope]['prediction_changes'] = [r['actual_word_id'] for r in rows
                                               if pred(r, 'control') != pred(r, 'treatment')]
    result = {'status': 'EXECUTED_PENDING_INDEPENDENT_RECOUNT', 'at_kst': now(),
              'plan_sha256': sha(PLAN), 'data_role': plan['data_role'],
              'participant_count': 1, 'n': len(rows), 'models': plan['models'],
              'summary': summary, 'rows': rows, 'inference_calls': len(rows) * len(model_keys),
              'optimizer_steps': 0, 'protected_files_unchanged': protected_unchanged,
              'protected_file_count': len(plan['protected_sha256']),
              'production_registry_or_default_changed': False, 'model_promoted': False,
              'elapsed_inference_s': round(time.perf_counter() - started, 3),
              'limits': plan['limits']}
    write_new(RESULT, result)
    print(json.dumps({k: result[k] for k in ('status', 'n', 'summary', 'inference_calls',
                                            'optimizer_steps', 'protected_files_unchanged')}, ensure_ascii=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'execute'])
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else execute()
