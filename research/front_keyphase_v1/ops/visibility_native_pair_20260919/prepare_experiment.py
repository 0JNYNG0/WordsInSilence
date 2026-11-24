"""Freeze a new input plan for the unchanged, previously reviewed Claude-authored runner."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json

OPS = Path(__file__).resolve().parent
WEB = OPS.parents[1]
OLD = WEB / 'ops/native_data_pair_20260919'
RUN = WEB / 'runs/visibility_native_pair_v1'
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, data):
    with p.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

assert not RUN.exists() and not (OPS / 'plan.json').exists()
old = read(OLD / 'plan.json')
plan = dict(old)
plan['checkpoint'] = str(WEB / 'runs/visibility_candidate_v1/visibility_final.pt')
plan['checkpoint_sha256'] = '258f23e1ad515a25b50e6affac8ac4e284ba8f6ae6fac3372c1e7a31ea6848a8'
assert {k for k in plan if plan[k] != old[k]} == {'checkpoint', 'checkpoint_sha256'}
assert plan['augmentation'] is False
assert sha(Path(plan['checkpoint'])) == plan['checkpoint_sha256']
runner = WEB / 'tools_native_data_pair.py'
old_ready = read(OLD / 'execution_ready.json')
old_runner_hash = next(d for p, d in old_ready['frozen_execution_inputs'].items() if Path(p).name == runner.name)
assert sha(runner) == old_runner_hash
web_plan_path = WEB / 'ops/native_pair_web_development_20260919/plan_LOCAL_ONLY.json'
web_plan = read(web_plan_path)
assert len(web_plan['rows']) == 18 and web_plan['no_reserved_test_inputs']
assert read(WEB / 'ops/native_pair_web_development_20260919/independent_recount_LOCAL_ONLY.json')['status'] == 'PASS_SAVED_OUTPUT_RECOUNT_AND_PROVENANCE'
policy = {
    'prepared_at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(),
    'question': 'Does the same native source-data addition remain beneficial when both arms start from the existing visibility-trained checkpoint, without further augmentation?',
    'reason': 'Prior native addition corrected apology on existing webcam input but retained NA/JADA failures; the visibility checkpoint has the complementary historical error pattern on the same fixed inputs.',
    'changed_from_prior_native_pair': ['initial checkpoint for BOTH arms'],
    'within_pair_treatment': 'Add the same approved 9 AI Hub SEN0356 train-signer clips; no user data.',
    'unchanged': ['all data IDs', 'signer split', 'label mapping and user semantic confirmation',
                  'seed20260919', '660 steps', 'optimizer configuration', '93 classes',
                  'no new augmentation', 'same preprocessing and evaluation cohorts'],
    'training_executions_allowed': 2,
    'checkpoint_selection': 'Final fixed step660 for each arm only; no epoch/seed/threshold search.',
    'evaluations_fixed_before_training': {
        'native': 'Same 699 dev inputs: source696 plus apology3. Target-restricted33 has only6 GT classes.',
        'web': {'plan': str(web_plan_path), 'sha256': sha(web_plan_path),
                'n': 18, 'scope': 'All fixed one-participant development observations, project18 and full93. Not a final untouched test.'},
        'reference_robustness': 'Existing NIKL18 intact plus exactly the five previously reviewed passive-hand-missing conditions. Evaluation only; no new training augmentation.',
    },
    'report_all_gains_and_regressions': True,
    'no_default_promotion': True,
    'no_user_training': True,
    'no_heldout_reading': True,
    'no_downloads': True,
    'scope_limit': 'Does not fill the remaining target18 multi-person data gaps or prove multi-person webcam generalization.',
}
write(OPS / 'plan.json', plan)
write(OPS / 'comparison_policy.json', policy)
pins = [runner, WEB / 'tools_prep_union93_candidate.py', WEB / 'tools_visibility_experiment.py',
        OPS / 'plan.json', OPS / 'comparison_policy.json', Path(plan['checkpoint']),
        Path(plan['base_inputs']), Path(plan['candidate_report']),
        Path(plan['semantic_review']['evidence']), OLD / 'audit_native_pair_result.py',
        web_plan_path]
write(OPS / 'execution_ready.json', {
    'status': 'NEW_PLAN_FROZEN_UNCHANGED_RUNNER', 'at_kst': policy['prepared_at_kst'],
    'authorship': 'Existing Claude-authored runner reused without modification; Codex prepared experiment inputs and independent validation.',
    'completed_code_checks_reused': 'Previous metadata22/22 and synthetic6/6; no code retest. New plan preflight runs inside the execution command.',
    'frozen_execution_inputs': {str(p): sha(p) for p in pins}, 'out': str(RUN),
})
sizes = [11.04,9.04,11.13,13.62,10.48,10.52,10.42,11.02,11.36,11.49,11.16,10.50,11.12,11.73,10.91,10.48]
keys = [39600] + list(range(39602,39617))
write(OPS / 'official_word_keypoint_catalog.json', {
    'observed_at_kst': policy['prepared_at_kst'],
    'source_url': 'https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103',
    'observed_surface': 'Logged-in existing Chrome tab, official file-list tab; visible DOM text.',
    'path': '004.수어영상/1.Training/라벨링데이터/REAL/WORD',
    'archives': [{'archive_name': '%02d_real_word_keypoint.zip' % (i+1), 'display_GB': size, 'filekey': key}
                 for i, (size,key) in enumerate(zip(sizes,keys))],
    'sum_display_GB': round(sum(sizes),2),
    'exact_bytes': 'NOT_EXPOSED_IN_READ_TABLE',
    'internal_signer_word_membership': 'UNVERIFIED; no inner member list in observed UI. Do not map archive number to signer number.',
    'not_in_training_scope': {'archive_name': '09_real_word_keypoint.zip', 'filekey':39479, 'display_GB':20.70, 'official_split':'Validation'},
    'download_clicked': False, 'bytes_downloaded': 0,
    'decision': 'Do not guess a large archive download. Native WORD source acquisition remains separate; proceed with the supported no-download fixed comparison.',
})
print(json.dumps({'status': 'FROZEN', 'changed_plan_keys': ['checkpoint','checkpoint_sha256'],
                  'training_started': False, 'runner_unchanged': True,
                  'official_word_keypoint_archives': len(sizes), 'sum_display_GB': round(sum(sizes),2)}))
