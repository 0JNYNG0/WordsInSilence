"""Codex-owned synthetic verification, no device/model/user data access."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from wis_lab.service import review_log
import test_r16_candidate_cohorts as legacy

LABELS = ['나', '물', '버스'] + [f'합성분류{i:02d}' for i in range(90)]
MAPPING = {'NA': '나', 'MUL': '물'}
SET_SHA = hashlib.sha256(json.dumps(sorted(LABELS), ensure_ascii=False,
                                   separators=(',', ':')).encode()).hexdigest()
creation = legacy.identity()
creation['trained_class_set'] = {
    'schema': 'wis_trained_class_set_v1', 'status': 'AVAILABLE',
    'model_id': creation['model_id'], 'checkpoint_sha256': creation['checkpoint_sha256'],
    'labels': LABELS, 'count': 93, 'membership_sha256': SET_SHA,
    'project_word_to_model_label': MAPPING,
}


def build_report(folder, old=False):
    (folder / 'trials').mkdir(parents=True)
    cases = [('full_wrong', 'NA', '버스'), ('full_match', 'MUL', '물'),
             ('model_failed', 'NA', None), ('result_missing', 'MUL', None),
             ('unknown', 'UNKNOWN', '나'), ('unconfirmed', None, '나'),
             ('opted_out', 'NA', '나'), ('unsupported', 'EXTRA', '나')]
    ids, meta = [], {}
    for tid, actual, pred in cases:
        ids.append(tid)
        fixed = legacy.identity() if old else copy.deepcopy(creation)
        if tid == 'opted_out':
            fixed['opted_in'] = False
        meta[tid] = {'research_model': fixed}
        result = legacy.result(tid, 'NA', 'NA', 'MUL' if actual == 'MUL' else 'NA')
        result['research_model'] = fixed
        rc = result['research_comparison']
        rc['identity_fixed_at_trial_creation'] = copy.deepcopy(fixed)
        rc['scopes']['full'] = {'scope': 'full', 'candidate_count': 93, 'top1': pred,
                                'top3': [{'label': pred, 'score': 5}],
                                'class_set_sha256': SET_SHA}
        if tid == 'model_failed':
            rc['model_status'] = 'INFERENCE_FAILED'
            rc['scopes'] = None
        if tid != 'result_missing':
            (folder / 'trials' / (tid + '.result.json')).write_text(
                json.dumps(result, ensure_ascii=False), encoding='utf-8')
        if actual is not None:
            review_log.submit(str(folder), 'SYNTHETIC_ONLY', tid, actual, 'unsure')
    return review_log.report(str(folder), ids, trial_meta=meta), ids, meta


def main():
    with tempfile.TemporaryDirectory(prefix='wis_report_verify_') as tmp:
        folder = Path(tmp) / 'synthetic'
        report, ids, meta = build_report(folder)
        full = report['full_scope']
        assert len(full['groups']) == 1
        g = next(iter(full['groups'].values()))
        score = g['per_method']['candidate']
        assert (g['identity']['class_count'], g['n_supported'], score['hits'], score['answered']) == (93, 4, 1, 2)
        rows = {r['trial_id']: r for r in g['rows']}
        assert rows['full_wrong']['hit'] is False and rows['full_wrong']['pick_label'] == '버스'
        assert rows['full_match']['hit'] is True
        assert rows['model_failed']['reason'] == 'CANDIDATE_INFERENCE_FAILED'
        assert rows['result_missing']['reason'] == 'NO_RESULT_FILE'
        assert len(g['excluded']) == 3 and full['counts']['opted_out'] == 1
        restricted = next(g for g in report['cohorts']['groups'].values() if g['scope'] == 'project18')
        assert next(r for r in restricted['rows'] if r['trial_id'] == 'full_wrong')['candidate']['hit'] is True
        for path in (folder / 'trials').glob('*.result.json'):
            obj = json.loads(path.read_text(encoding='utf-8'))
            obj['requested_word_id'] = 'SARANG'
            path.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
        again = review_log.report(str(folder), ids, trial_meta=meta)
        assert again['full_scope'] == full, 'Selected prompt changed full scoring'
        old_report, _, _ = build_report(Path(tmp) / 'legacy', old=True)
        assert old_report['full_scope']['status'] == 'UNAVAILABLE'
        assert len(old_report['cohorts']['groups']) == 2
        (HERE / 'SYNTHETIC_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (HERE / 'LEGACY_REPORT.json').write_text(json.dumps(old_report, ensure_ascii=False, indent=2), encoding='utf-8')
        summary = {'status': 'PASS', 'source': 'independent synthetic report() join',
                   'full_scope': '1/4; two answers, two explicit unanswered',
                   'full_wrong_restricted_right_separated': True,
                   'requested_word_invariance': True, 'legacy_full_unavailable': True,
                   'user_data': 0, 'camera': 0, 'model_inference': 0}
        (HERE / 'INDEPENDENT_REPORT_CHECK.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
