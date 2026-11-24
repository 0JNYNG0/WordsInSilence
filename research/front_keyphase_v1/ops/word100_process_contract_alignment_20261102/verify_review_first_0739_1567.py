"""Structural/hash self-check only; no inference or semantic accuracy claim."""
from pathlib import Path
import hashlib
import json
from collections import Counter

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'REVIEW_FIRST_0739_1567.json'
MANIFEST = HERE / 'FIRST_RUNTIME_INTERVAL_REVIEW_MANIFEST.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    expected = {r['review_id']: r for r in manifest['rows'] if r['word_id'] in ('WORD0739', 'WORD1567')}
    checks = []
    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
    check('manifest_hash', report['manifest']['sha256'] == sha(MANIFEST))
    check('exact_27_reviews', len(report['rows']) == len(expected) == 27)
    check('exact_unique_ids', set(r['review_id'] for r in report['rows']) == set(expected) and len({r['review_id'] for r in report['rows']}) == 27)
    pages = {}
    for row in report['rows']:
        ref = expected[row['review_id']]
        prefix = row['review_id']
        check(prefix + ':FIT_only', ref['role'] == 'fit' and 1 <= int(ref['person']) <= 9)
        for key in ('case_id', 'query_id', 'word_id', 'person', 'physical_query'):
            check(prefix + ':' + key, row[key] == ref[key])
        frames = row['source_frames']
        timestamps = row['source_timestamps_us']
        check(prefix + ':nominal_frames_exact', frames == ref['nominal_source_frames'])
        check(prefix + ':nominal_times_exact', timestamps == ref['nominal_source_timestamps_us'])
        check(prefix + ':axis_length', len(frames) == len(timestamps) and len(frames) > 0)
        check(prefix + ':axis_order', all(b > a for a, b in zip(frames, frames[1:])) and all(b > a for a, b in zip(timestamps, timestamps[1:])))
        expected_pages = [{'path': p['path'], 'sha256': p['sha256']} for p in ref['review_pages']]
        check(prefix + ':all_pages_exact', row['reviewed_pages'] == expected_pages)
        for p in row['reviewed_pages']:
            if p['path'] not in pages:
                actual = sha(p['path'])
                pages[p['path']] = {'expected_sha256': p['sha256'], 'actual_sha256': actual}
                check(p['path'] + ':sha256', actual == p['sha256'])
        check(prefix + ':all_nominal_frames_shown', set(frames).issubset({f for p in ref['review_pages'] for f in p['source_frames']}))
        check(prefix + ':typed_label', row['observed_process_evidence'] is None or type(row['observed_process_evidence']) is int and row['observed_process_evidence'] in (0, 1))
        check(prefix + ':reason_present', bool(row['reason'].strip()))
        witnesses = row['witnesses']
        check(prefix + ':witnesses_nominal_only', all(f in frames for values in witnesses.values() for f in values))
        motion = witnesses['motion_source_frames']
        check(prefix + ':motion_native_contiguous', bool(motion) and all(b == a + 1 for a, b in zip(motion, motion[1:])))
        check(prefix + ':positive_has_three_witness_types', row['observed_process_evidence'] != 1 or all(witnesses.values()))
        check(prefix + ':no_linguistic_or_repetition_target', row['whole_word_correctness'] is None and row['learner_error'] is None and row['required_repetitions'] is None)
    labels = Counter(str(r['observed_process_evidence']) for r in report['rows'])
    counts = {'total': len(report['rows']), 'positive': labels['1'], 'negative': labels['0'], 'unknown': labels['None']}
    check('top_counts', all(report['counts'][k] == v for k, v in counts.items()))
    check('41_pages', len(pages) == report['review_method']['review_pages'] == 41)
    for word in ('WORD0739', 'WORD1567'):
        selected = [r for r in report['rows'] if r['word_id'] == word]
        values = Counter(str(r['observed_process_evidence']) for r in selected)
        actual = {'total': len(selected), 'positive': values['1'], 'negative': values['0'], 'unknown': values['None']}
        check(word + ':counts', actual == report['counts']['by_word'][word])
    failures = [c for c in checks if not c['passed']]
    output = {'status': 'PASS_STRUCTURAL_LINEAGE_ONLY' if not failures else 'FAIL', 'review_sha256': sha(REPORT), 'manifest_sha256': sha(MANIFEST), 'verifier_sha256': sha(__file__), 'check_count': len(checks), 'failure_count': len(failures), 'failures': failures, 'counts': counts, 'unique_pages_hashed': len(pages), 'semantic_validation': 'Not an independent human or expert review, and not model accuracy. Only annotation structure, nominal axes, witness containment and page lineage are verified.', 'model_forward_calls': 0, 'training_calls': 0, 'checks': checks}
    destination = HERE / 'REVIEW_FIRST_0739_1567_VERIFIED.json'
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: output[k] for k in ('status', 'review_sha256', 'check_count', 'failure_count', 'counts', 'unique_pages_hashed')}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
