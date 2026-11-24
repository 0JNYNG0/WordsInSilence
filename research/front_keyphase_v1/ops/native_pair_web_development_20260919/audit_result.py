"""Recount saved predictions against original participant reviews; no model execution."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import re

OUT = Path(__file__).resolve().parent
WEB = OUT.parents[1]
OLD = Path('C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/WordsInSilence_기능완성_20260913')
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

destination = OUT / 'independent_recount_LOCAL_ONLY.json'
if destination.exists():
    raise FileExistsError('Completed recount exists; do not repeat.')
plan = read(OUT / 'plan_LOCAL_ONLY.json')
result = read(OUT / 'result_LOCAL_ONLY.json')
session = WEB / 'runs' / plan['session_id']
reviews_path = session / 'trial_reviews.jsonl'
reviews = {}
for line in reviews_path.read_text(encoding='utf-8').splitlines():
    r = json.loads(line)
    if r['trial_id'] not in reviews or r['revision'] > reviews[r['trial_id']]['revision']:
        reviews[r['trial_id']] = r
checks = []
def check(name, passed):
    checks.append({'name': name, 'passed': bool(passed)})

fixed = {r['trial_id']: r for r in plan['rows']}
actual = {r['trial_id']: r for r in result['rows']}
check('exact 18 trials without omissions or duplicates', len(result['rows']) == len(actual) == 18 and set(actual) == set(fixed))
check('closed development session', read(session / 'session.json')['purpose'] == 'development' and read(session / 'session.json')['state'] == 'CLOSED')
check('plan pin', result['plan_sha256'] == sha(OUT / 'plan_LOCAL_ONLY.json'))
check('36 new inference calls and no optimizer steps', result['inference_calls'] == 36 and result['optimizer_steps'] == 0)
check('all original pins unchanged', all(sha(Path(p)) == d for p, d in plan['protected_sha256'].items()))
for tid, row in actual.items():
    check(tid + ': original participant truth', row['actual_word_id'] == reviews[tid]['actual_word_id'] == fixed[tid]['actual_word_id']
          and row['actual_label'] == fixed[tid]['actual_label'] and reviews[tid]['source'] == 'participant_self_report')
    check(tid + ': original observation pin', sha(Path(row['observation'])) == row['observation_sha256'] == fixed[tid]['observation_sha256'])
    for arm, spec in plan['models'].items():
        p = row['arms'][arm]
        check(tid + ':' + arm + ': exact candidate and valid output', p['model_status'] == 'READY'
              and p['checkpoint_sha256'] == spec['sha256'] and p['checkpoint_step'] == 660
              and p['scopes']['project18']['candidate_count'] == 18 and p['scopes']['full']['candidate_count'] == 93
              and p['accepted'] is None and p['execution_assessment'] == 'UNCONFIGURED')

for name, scope, truth in [('project18', 'project18', 'actual_word_id'), ('full93', 'full', 'actual_label')]:
    successful = {}
    for arm in plan['models']:
        successful[arm] = {tid for tid, r in actual.items() if r['arms'][arm]['scopes'][scope]['top1'] == r[truth]}
        check(name + ':' + arm + ': independent hit count', len(successful[arm]) == result['summary'][name]['arms'][arm]['correct'])
        for r in actual.values():
            s = r['arms'][arm]['scopes'][scope]
            first = s['order'][0]['word_id'] if scope == 'project18' else s['top3'][0]['label']
            check(name + ':' + arm + ':' + r['trial_id'] + ': top rank agrees', first == s['top1'])
    gains = {actual[tid]['actual_word_id'] for tid in successful['treatment'] - successful['control']}
    losses = {actual[tid]['actual_word_id'] for tid in successful['control'] - successful['treatment']}
    check(name + ': gains and regressions', gains == set(result['summary'][name]['gains']) and losses == set(result['summary'][name]['regressions']))

historical_path = OLD / 'functional22_development_replay.json'
historical = read(historical_path)
old_rows = {r['trial_id']: r for r in historical['rows']}
check('historical visibility record uses identical 18 observation hashes', set(old_rows) == set(actual)
      and all(old_rows[tid]['observation_sha256'] == actual[tid]['observation_sha256'] for tid in actual))
historical_summary = {}
for name, scope, truth in [('project18', 'project18', 'actual_word_id'), ('full93', 'full', 'actual_label')]:
    historical_summary[name] = sum(r['arms']['extra_training_plus_reviewed_missingness']['scopes'][scope]['top1'] == r[truth]
                                   for r in old_rows.values())
check('historical visibility record checkpoint', historical['models']['extra_training_plus_reviewed_missingness']['sha256']
      == '258f23e1ad515a25b50e6affac8ac4e284ba8f6ae6fac3372c1e7a31ea6848a8')

base = read(WEB.parent / 'sign_model_pilot_v1/runs/masked_bn_pair_v1/inputs.json')
union = read(WEB / 'runs/union93_candidate_v1/inputs.json')
visibility = read(OLD / 'visibility_experiment_20260914a/inputs.json')
pattern = re.compile(r'^NIA_SL_SEN\d+_REAL0[1-9]_F_')
check('base training IDs all AIHub train signers', len(base['train']) == 2095 and all(pattern.match(r['sample_id']) for r in base['train']))
for name, data in [('union', union), ('visibility', visibility)]:
    check(name + ': source training excludes participant IDs', len(data['source_train']) == 2095 and all(pattern.match(r['id']) for r in data['source_train']))
    check(name + ': other training references all NIKL', len(data['nikl']) == 18 and all(r['id'].startswith('NIKL_') for r in data['nikl']))
check('visibility participant input exclusion record', visibility['participant_captures']['read'] is False)

audit = {'at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(),
         'status': 'PASS_SAVED_OUTPUT_RECOUNT_AND_PROVENANCE' if all(c['passed'] for c in checks) else 'FAIL',
         'passed': sum(c['passed'] for c in checks), 'total': len(checks), 'checks': checks,
         'result_sha256': sha(OUT / 'result_LOCAL_ONLY.json'), 'participant_reviews_sha256': sha(reviews_path),
         'historical_visibility_reference': {'file': str(historical_path), 'sha256': sha(historical_path),
             'same_observation_bytes': True, 'stored_correct': historical_summary,
             'note': 'Reused historical outputs, not a new run or new single-factor comparison against native pair. Training interventions differ.'},
         'participant_interpretation': 'Recordings excluded from TCN training; one training-unseen person but development-exposed. Do not erase this case or claim an untouched multi-person test.',
         'scope': 'Output recount, exact cohort/ground truth/model pins, existing training provenance. No new inference, training, final heldout evaluation or linguistic correctness validation.'}
with destination.open('x', encoding='utf-8') as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': audit['status'], 'passed': audit['passed'], 'total': audit['total'],
                  'historical_visibility_same_inputs': historical_summary}))
if audit['status'] == 'FAIL':
    raise SystemExit(1)
