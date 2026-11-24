"""Run existing independent audits and fixed evaluations on the newly completed pair.

No training implementation, detector change or production registration is made here.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import copy
import hashlib
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
OPS = Path(__file__).resolve().parent
WEB = OPS.parents[1]
RUN = WEB / 'runs/visibility_native_pair_v1'
OLD_WEB_OPS = WEB / 'ops/native_pair_web_development_20260919'
PRIVATE_OLD = Path('C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/WordsInSilence_기능완성_20260913')
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, data):
    with p.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

if (OPS / 'followup_validation_LOCAL_ONLY.json').exists():
    raise FileExistsError('Completed fixed evaluation exists; do not repeat.')
assert (RUN / 'results.json').is_file() and not (RUN / 'error.json').exists()
native_auditor = module('existing_native_auditor', WEB / 'ops/native_data_pair_20260919/audit_native_pair_result.py')
native_auditor.OPS = OPS
native_auditor.RUN = RUN
assert native_auditor.main() == 0
native_audit = read(OPS / 'native_pair_result_audit.json')
policy = read(OPS / 'comparison_policy.json')
assert sha(Path(policy['evaluations_fixed_before_training']['web']['plan'])) == policy['evaluations_fixed_before_training']['web']['sha256']

# Same cohort and web runner as the previous completed comparison. Only model paths change.
plan = copy.deepcopy(read(OLD_WEB_OPS / 'plan_LOCAL_ONLY.json'))
plan['prepared_at_kst'] = datetime.now(timezone(timedelta(hours=9))).isoformat()
plan['question'] = policy['question']
plan['parent_checkpoint'] = read(OPS / 'plan.json')['checkpoint_sha256']
plan['cohort_fixed_before_training'] = True
plan['pretraining_policy_sha256'] = sha(OPS / 'comparison_policy.json')
plan['models'] = {arm: dict(spec, fixed_step=660) for arm, spec in native_audit['checkpoints'].items()}
pins = plan['protected_sha256']
for p in [OPS / 'comparison_policy.json', OPS / 'plan.json', Path(__file__)] + [Path(s['path']) for s in plan['models'].values()]:
    pins[str(p)] = sha(p)
write(OPS / 'web_plan_LOCAL_ONLY.json', plan)
replay = module('existing_web_replay', OLD_WEB_OPS / 'replay_fixed_pair.py')
replay.PLAN = OPS / 'web_plan_LOCAL_ONLY.json'
replay.RESULT = OPS / 'web_result_LOCAL_ONLY.json'
replay.execute()
web = read(replay.RESULT)

# Existing 18 intact reference inputs and exactly the five already reviewed mask conditions.
sys.path[:0] = [str(WEB), str(WEB / 'src'), str(WEB.parent / 'sign_model_pilot_v1/src')]
import tools_prep_union93_candidate as u
import tools_visibility_experiment as v
from wis_lab.service import neural
old_inputs_path = PRIVATE_OLD / 'visibility_experiment_20260914a/inputs.json'
old_inputs = read(old_inputs_path)
nikl, eligible = [], {}
for r in old_inputs['nikl']:
    path = WEB / r['npz']
    nat, present, ts = u._native_from_nikl(str(path), r['npz_sha256'])
    nikl.append({'clip_id': r['id'], 'label': r['label'], 'nat': nat, 'present': present, 'ts': ts})
    if r.get('reviewed_non_sign_hand') in ('left', 'right'):
        eligible[r['id']] = {'side': r['reviewed_non_sign_hand']}
assert len(nikl) == 18 and len(eligible) == 5
clips = v.reference_clips(nikl, eligible)
assert len(clips) == 23 and sum(c['condition'] == 'reviewed_masked' for c in clips) == 5
robustness = {}
for arm in plan['models']:
    loaded = neural._load('offline_native_pair_' + arm)
    assert loaded['checkpoint_sha256'] == plan['models'][arm]['sha256']
    rows = v.reference_rows(loaded['model'], clips, loaded['classes'], sorted({r['label'] for r in nikl}))
    robustness[arm] = {'rows': rows}
    for scope in ('project18', 'full93'):
        robustness[arm][scope] = {
            condition: {'n': sum(r['condition'] == condition for r in rows),
                        'correct': sum(r['condition'] == condition and r[scope]['correct'] for r in rows),
                        'errors': [{'id': r['id'], 'true': r['true'], 'pred': r[scope]['pred']}
                                   for r in rows if r['condition'] == condition and not r[scope]['correct']]}
            for condition in ('intact', 'reviewed_masked')}

# Recount new outputs against original participant reviews, not the reported aggregate alone.
checks = []
def check(name, passed):
    checks.append({'name': name, 'passed': bool(passed)})
session = WEB / 'runs' / plan['session_id']
reviews = {}
for line in (session / 'trial_reviews.jsonl').read_text(encoding='utf-8').splitlines():
    r = json.loads(line)
    if r['trial_id'] not in reviews or r['revision'] > reviews[r['trial_id']]['revision']:
        reviews[r['trial_id']] = r
fixed = {r['trial_id']: r for r in plan['rows']}
actual = {r['trial_id']: r for r in web['rows']}
check('exact 18 predeclared development observations', len(web['rows']) == len(actual) == 18 and set(actual) == set(fixed))
check('original participant self labels', all(r['actual_word_id'] == reviews[tid]['actual_word_id'] == fixed[tid]['actual_word_id'] for tid,r in actual.items()))
check('all protected files unchanged', all(sha(Path(p)) == digest for p,digest in pins.items()))
check('fixed inference count and no optimizer', web['inference_calls'] == 36 and web['optimizer_steps'] == 0)
for name, scope, truth in [('project18','project18','actual_word_id'), ('full93','full','actual_label')]:
    hit = {}
    for arm in plan['models']:
        hit[arm] = {tid for tid,r in actual.items() if r['arms'][arm]['scopes'][scope]['top1'] == r[truth]}
        check(name + ':' + arm + ': hits independently counted', len(hit[arm]) == web['summary'][name]['arms'][arm]['correct'])
        check(name + ':' + arm + ': exact model and output scope', all(r['arms'][arm]['checkpoint_sha256'] == plan['models'][arm]['sha256']
              and r['arms'][arm]['model_status'] == 'READY' and r['arms'][arm]['scopes'][scope]['candidate_count'] == (18 if name == 'project18' else 93)
              and r['arms'][arm]['accepted'] is None and r['arms'][arm]['execution_assessment'] == 'UNCONFIGURED' for r in actual.values()))
    check(name + ': all gains and losses', {actual[t]['actual_word_id'] for t in hit['treatment']-hit['control']} == set(web['summary'][name]['gains'])
          and {actual[t]['actual_word_id'] for t in hit['control']-hit['treatment']} == set(web['summary'][name]['regressions']))
for arm,data in robustness.items():
    rows = data['rows']
    check(arm + ': fixed 23 reference conditions', len(rows) == 23 and len({r['id'] for r in rows}) == 23
          and {r['id'].split('::')[0] for r in rows if r['condition'] == 'reviewed_masked'} == set(eligible))
    check(arm + ': robustness correctness from labels', all(r[s]['correct'] == (r[s]['pred'] == r['true']) for r in rows for s in ('project18','full93')))
report = {'status': 'PASS_FIXED_EVALUATIONS' if all(c['passed'] for c in checks) else 'FAIL',
          'at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(),
          'native_audit': {'passed': native_audit['passed'], 'total': native_audit['total']},
          'checks': checks, 'passed': sum(c['passed'] for c in checks), 'total': len(checks),
          'web_result_sha256': sha(replay.RESULT), 'web_summary': web['summary'],
          'reference_input_metadata_sha256': sha(old_inputs_path),
          'reference_robustness': robustness,
          'reference_scope': 'Reused public training references and the same five reviewed missingness conditions, evaluation only. Not fresh accuracy.',
          'inference_calls_after_training': 82, 'new_training_runs_in_validator': 0,
          'limits': policy['scope_limit'], 'model_promoted': False,
          'participant_scope': 'One training-excluded but development-exposed participant; not a final test.'}
write(OPS / 'followup_validation_LOCAL_ONLY.json', report)
print(json.dumps({'status': report['status'], 'passed': report['passed'], 'total': report['total'],
                  'web': web['summary'], 'robustness': {a:{s:d[s] for s in ('project18','full93')} for a,d in robustness.items()}}, ensure_ascii=True))
if report['status'] != 'PASS_FIXED_EVALUATIONS':
    raise SystemExit(1)
