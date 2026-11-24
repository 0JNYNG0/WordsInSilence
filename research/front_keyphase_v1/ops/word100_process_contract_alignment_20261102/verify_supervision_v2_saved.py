"""Separate saved-value reconstruction; intentionally does not import compiler."""
from pathlib import Path
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import traceback

U = Path(__file__).resolve().parent
A = U.parent / 'word100_action_completion_supervision_20261102'
checks = Counter()


def demand(value, message, group='contract'):
    checks[group] += 1
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def mapping(r, j):
    f = r['ordered_frame_map'][j]
    return dict(source_frame=f['source_native_frame'], output_index=j,
        source_native_timestamp_us=f['source_native_timestamp_us'],
        view_timestamp_us=f['view_timestamp_us'], source_frame_path=f['source_frame_path'],
        source_frame_sha256=f['source_frame_sha256'], image_available=f['image_available'])


def check_split(old_file, new_file, role, n):
    old = read(A / old_file)
    new = read(U / new_file)
    demand(new['original_artifact'] == {'path': str(A / old_file), 'sha256': sha(A / old_file)}, 'Wrong old artifact')
    demand(new['role'] == role and new['training_allowed'] == (role == 'fit'), 'Role/training isolation')
    demand(len(old['rows']) == len(new['rows']) == n, 'Scope row count')
    counts = Counter()
    new_physical_points = set()
    new_physical_intervals = set()
    for before, after in zip(old['rows'], new['rows']):
        demand(before['view_id'] == after['view_id'], 'Row identity/order changed')
        for key, value in before.items():
            if key not in ('sparse_relation_targets', 'fully_retained_motion_interval_targets'):
                demand(after[key] == value, f'Existing field changed: {key}', 'lossless')
        extra = set(after) - set(before)
        demand(extra == {'supervision_v2'}, 'Unexpected new row field')
        for key in ('sparse_relation_targets', 'fully_retained_motion_interval_targets'):
            demand(after[key][:len(before[key])] == before[key], 'Old target/value/order/provenance changed', 'lossless')
        dr = before['direct_review']
        fm = before['ordered_frame_map']
        source_to_output = {f['source_native_frame']: i for i, f in enumerate(fm)}
        demand(len(source_to_output) == len(fm), 'Ambiguous native mapping')
        native_points = {(p['output_index'], field): p[field] for p in before['sparse_relation_targets']
                         for field in ('setup_relation', 'terminal_relation')}
        # Reconstruct expected assertions directly from review text fields,
        # not from the compiler's bound-record ledger or its helper functions.
        asserted = defaultdict(list)
        for i, p in enumerate(dr.get('relations', [])):
            idx = source_to_output[p['source_frame']]
            demand(p.get('output_index', idx) == idx, 'Numeric review point mapping')
            for field in ('setup_relation', 'terminal_relation'):
                if field in p:
                    asserted[(idx, field)].append((p[field], 'relations', i, p))
        for field, target in [('setup_witnesses', 'setup_relation'), ('terminal_witnesses', 'terminal_relation')]:
            for i, p in enumerate(dr[field]):
                idx = source_to_output[p['source_frame']]
                demand(p['output_index'] == idx, 'Typed review point mapping')
                demand(bool(before['predicate_definitions'][target]), 'Typed physical predicate absent')
                asserted[(idx, target)].append((1, field, i, p))
        expected_new = {}
        for key, records in asserted.items():
            labels = {x[0] for x in records}
            demand(len(labels) == 1, 'Conflicting explicit assertions')
            value = records[0][0]
            demand(fm[key[0]]['image_available'], 'Claimed relation in unavailable image')
            if key in native_points:
                demand(native_points[key] == value, 'Direct assertion conflicts with old known/null')
            elif value is not None:
                expected_new[key] = value
        observed_new = {}
        for p in after['sparse_relation_targets'][len(before['sparse_relation_targets']):]:
            idx = p['output_index']
            ident = mapping(before, idx)
            demand(all(p[k] == v for k, v in ident.items()), 'Added point source/time/hash mismatch', 'axis')
            demand(ident['image_available'] is True, 'Added unavailable point')
            demand(p['source_relation_index'] is None, 'New point impersonates native index')
            demand(p['source_annotation'] in [x[3] for key, rs in asserted.items() if key[0] == idx for x in rs], 'Source annotation not original')
            for field in ('setup_relation', 'terminal_relation'):
                key = (idx, field)
                if p[field] is None:
                    demand(key not in expected_new, 'New supervised dimension silently dropped')
                else:
                    demand(key in expected_new and expected_new[key] == p[field], 'Unsupported new point label')
                    demand(key not in observed_new, 'Duplicate new point dimension')
                    observed_new[key] = p[field]
                    new_physical_points.add((before['source_sha256'], before['physical_query'], p['source_frame'], field))
            for entry in p['supervision_v2_provenance']:
                key = (entry['output_index'], entry['relation'])
                demand(entry['value'] == expected_new[key] and entry['status'] == 'added', 'Point provenance target mismatch')
                for pr in entry['provenance']:
                    demand(pr['review_source'] == before['review_source'], 'Point provenance source changed')
                    field, i = pr['review_field'], pr['review_field_index']
                    demand(pr['original_review_record'] == dr[field][i], 'Point annotation not verbatim')
                    demand(pr['predicate_definitions'] == before['predicate_definitions'], 'Point predicate expanded')
                    demand(pr['review_evidence_reason'] == dr['evidence_reason'], 'Point reason changed')
                    demand(pr['source_sha256'] == before['source_sha256'] and pr['view_id'] == before['view_id'], 'Point provenance scope changed')
                    demand(pr['runtime_or_native_propagation_performed'] is False, 'Hidden native propagation')
        demand(observed_new == expected_new, 'Consumed point set incomplete or invented', 'completeness')

        old_motion = {tuple(v['output_indices']): v['label'] for v in before['fully_retained_motion_interval_targets']}
        expected_motion = {}
        numeric_motion_claims = {}
        excluded = []
        for i, p in enumerate(dr.get('motion_witnesses', [])):
            ix = tuple(p['output_indices'])
            demand([source_to_output[k] for k in p['source_frames']] == list(ix), 'Direct motion mapping')
            if 'label' not in p:
                excluded.append(('unlabelled', i))
                continue
            if ix in numeric_motion_claims:
                demand(numeric_motion_claims[ix] == p['label'], 'Direct interval label conflict')
            numeric_motion_claims[ix] = p['label']
            if ix in old_motion:
                demand(old_motion[ix] == p['label'], 'Direct motion conflicts with native/null')
            if p['label'] is None:
                excluded.append(('explicit_unknown', i))
                continue
            demand(p['label'] in (0, 1), 'Invalid direct motion label')
            demand(len(ix) >= 2 and all(b - a == 1 for a, b in zip(ix, ix[1:])), 'Noncontiguous output interval')
            native = [fm[j]['source_native_frame'] for j in ix]
            demand(all(b - a == 1 for a, b in zip(native, native[1:])), 'Motion spans source deletion')
            demand(all(fm[j]['image_available'] for j in ix), 'Motion spans missing RGB')
            if ix not in old_motion:
                expected_motion[ix] = p['label']
        actual_motion = {}
        for p in after['fully_retained_motion_interval_targets'][len(before['fully_retained_motion_interval_targets']):]:
            ix = tuple(p['output_indices'])
            demand(ix not in actual_motion and ix in expected_motion, 'Duplicate/unreviewed new interval')
            actual_motion[ix] = p['label']
            demand(p['source_frames'] == [fm[j]['source_native_frame'] for j in ix], 'Interval native axis changed', 'axis')
            demand(p['native_timestamps_us'] == [fm[j]['source_native_timestamp_us'] for j in ix], 'Interval native PTS changed', 'axis')
            demand(p['view_timestamps_us'] == [fm[j]['view_timestamp_us'] for j in ix], 'Interval view PTS changed', 'axis')
            demand(p['source_motion_interval_index'] is None, 'New interval impersonates old index')
            demand(p['source_annotation'] in dr['motion_witnesses'], 'Interval original annotation absent')
            demand(p['source_annotation']['label'] == p['label'], 'Interval label not explicit')
            for pr in p['supervision_v2_provenance']:
                demand(pr['review_field'] == 'motion_witnesses', 'Wrong motion provenance field')
                demand(pr['original_review_record'] == dr['motion_witnesses'][pr['review_field_index']], 'Motion provenance not original')
                demand(pr['review_source'] == before['review_source'], 'Motion review SHA changed')
                demand(pr['predicate_definitions'] == before['predicate_definitions'], 'Motion predicate changed')
            new_physical_intervals.add((before['source_sha256'], before['physical_query'], tuple(p['source_frames']), p['label']))
        demand(actual_motion == expected_motion, 'Consumed explicit intervals incomplete/invented', 'completeness')
        actual_excluded = [(x['status'], x['provenance']['review_field_index']) for x in after['supervision_v2']['excluded'] if x.get('kind') == 'motion']
        demand(actual_excluded == excluded, 'Unlabelled or unknown interval lost/promoted')
        if before['construction_kind_audit_only'] == 'native':
            demand(after['sparse_relation_targets'] == before['sparse_relation_targets'], 'Edit-only point copied to native')
            demand(after['fully_retained_motion_interval_targets'] == before['fully_retained_motion_interval_targets'], 'Edit-only interval copied to native')
        counts['added_point_dimensions'] += len(expected_new)
        counts['added_point_rows'] += len(after['sparse_relation_targets']) - len(before['sparse_relation_targets'])
        counts['added_exact_intervals'] += len(expected_motion)
        counts['unlabelled_motion_excluded'] += sum(k == 'unlabelled' for k, _ in excluded)
        counts['explicit_unknown_motion_excluded'] += sum(k == 'explicit_unknown' for k, _ in excluded)
    return dict(views=n, role=role, counts=dict(counts),
                added_unique_physical_point_dimensions=len(new_physical_points),
                added_unique_physical_intervals=len(new_physical_intervals),
                output_sha256=sha(U / new_file))


def main():
    summaries = [check_split('PROCESS_VIEW_TARGETS.json', 'FIT_VIEW_TARGETS_V2.json', 'fit', 217),
                 check_split('DEV_PROCESS_VIEW_TARGETS.json', 'DEV_VIEW_TARGETS_V2.json', 'development', 80)]
    pins = {}
    for f in ['FIT_VIEW_TARGETS_V2.json', 'DEV_VIEW_TARGETS_V2.json']:
        for path, expected in read(U / f)['pins'].items():
            demand(path not in pins or pins[path] == expected, 'Conflicting input pin')
            pins[path] = expected
    for path, expected in pins.items():
        demand(sha(path) == expected, f'Input changed: {path}', 'hash')
    contract = read(U / 'SOURCE_EVIDENCE_REUSE_CONTRACT.json')
    demand(contract['automatic_native_or_runtime_propagation'] is False and contract['performed_propagation_count'] == 0, 'Invalid propagation contract')
    demand(contract['dense_label_propagation'] is False, 'Dense labels generated')
    audit = read(A / 'PROCESS_WITNESS_CONSUMPTION_AUDIT.json')
    for s in summaries:
        c = audit['summaries'][s['role']]['counts']
        demand(s['counts']['added_point_dimensions'] == c['union_missing_point_view_dimension_records'], 'Point correction differs from prior audit')
        demand(s['added_unique_physical_point_dimensions'] == c['union_missing_unique_source_query_frame_predicate'], 'Unique point correction differs')
        demand(s['counts']['added_exact_intervals'] == c['motion_explicit_known_unused_exact_interval'], 'Interval correction differs from audit')
    result = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='PASS',
        verifier='predicate_contract_review; separate algorithm reconstructing source annotations without importing compiler',
        same_codex_not_human_expert=True, checks=dict(checks), total_checks=sum(checks.values()),
        summaries=summaries, checked_input_pins=len(pins), compiler_imports=0,
        original_targets_and_completed_results_modified=False, new_inference=0, new_training=0,
        native_and_runtime_propagation=0, performance_improvement_claimed=False,
        exact_old_fields_and_targets_preserved=True, direct_eligible_assertion_consumption_complete=True,
        unlabelled_motion_not_promoted=True, explicit_unknown_preserved=True,
        independent_human_expert=False, whole_goal_complete=False,
        verifier_sha256=sha(__file__))
    out = U / 'SUPERVISION_V2_SAVED_VERIFIED.json'
    demand(not out.exists(), 'Never overwrite completed verifier result')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'checks': sum(checks.values()), 'sha256': sha(out), 'summaries': summaries}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        out = U / ('FIRST_VERIFY_FAILURE_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f') + '.txt')
        out.write_text(traceback.format_exc(), encoding='utf-8')
        raise
