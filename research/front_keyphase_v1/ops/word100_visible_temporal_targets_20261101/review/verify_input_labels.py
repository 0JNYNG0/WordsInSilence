"""Read-only independent arithmetic of prepared inputs and RGB target compilation.

No production/imported compiler functions, inference, training, or label edits.
This verifies data consistency, not RGB interpretation or linguistic correctness.
"""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import sys
import numpy as np

U = Path(__file__).resolve().parents[1]
OUT = U / 'review' / 'INPUT_LABEL_VERIFICATION.json'
checks = collections.Counter()
failures = []
hash_cache = {}


def read(name):
    return json.loads((U / name).read_text(encoding='utf-8-sig'))


def check(category, condition, detail):
    checks[category] += 1
    if not bool(condition):
        failures.append({'category': category, 'detail': detail})


def digest(path):
    p = Path(path).resolve()
    key = str(p)
    if key not in hash_cache:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        hash_cache[key] = h.hexdigest()
    return hash_cache[key]


def hash_check(path, expected, group):
    check(group, Path(path).is_file(), str(path))
    if Path(path).is_file():
        check(group, digest(path) == expected, str(path))


def unique_map(rows, key, group):
    result = {row[key]: row for row in rows}
    check(group, len(result) == len(rows), 'duplicate ' + key)
    return result


def validate_label(row, origin):
    tag = origin + ':' + row['sample_id']
    for h in ['R', 'L']:
        shape = row['center_shape'][h]
        check('label_schema', len(shape) == 3 and all(v is None or type(v) is int and v in [0, 1] for v in shape), tag + ' shape ' + h)
        check('label_schema', shape.count(1) <= 1, tag + ' exclusive shape ' + h)
        check('label_schema', row['motion'][h] in ['up', 'down', 'horizontal', 'still', 'mixed', None], tag + ' motion ' + h)
        for field in ['roi_center', 'roi_window']:
            check('label_schema', row[field][h] is None or type(row[field][h]) is bool, tag + ' ' + field + h)
    check('label_schema', row['center_relation'] in ['R_higher', 'L_higher', 'level', None], tag + ' relation')


def independently_encode(row):
    """Explicit numeric slots from rubric; never reads the saved target arrays."""
    values = [None] * 18
    reasons = [''] * 18
    for hand, shape_start, motion_start in [('R', 0, 8), ('L', 3, 13)]:
        for i in range(3):
            v = row['center_shape'][hand][i]
            eligible = row['roi_center'][hand] is True and v is not None
            values[shape_start + i] = v if eligible else None
            reasons[shape_start + i] = 'visible_center' if eligible else 'unknown_shape_or_outside_roi'
        m = row['motion'][hand]
        eligible = row['roi_window'][hand] is True and m is not None
        for i, category in enumerate(['up', 'down', 'horizontal', 'still', 'mixed']):
            values[motion_start + i] = int(category == m) if eligible else None
            reasons[motion_start + i] = 'visible_whole_13_frame_motion' if eligible else 'unknown_motion_or_outside_roi'
    relation = row['center_relation']
    eligible = row['roi_center']['R'] is True and row['roi_center']['L'] is True and relation is not None
    for i, category in [(6, 'R_higher'), (7, 'L_higher')]:
        values[i] = int(relation == category) if eligible else None
        reasons[i] = 'visible_center_relation' if eligible else 'unknown_bilateral_or_outside_roi'
    return values, reasons


def main():
    rubric = read('RUBRIC.json')
    selection = read('SELECTION.json')
    inputs = read('INPUTS.json')
    visual = read('VISUAL_INDEX.json')
    dataset = read('DATASET.json')
    audit = read('LABEL_AUDIT.json')
    secondary_selection = read('review/SECONDARY_SELECTION.json')
    docs = {name: read('review/LABELS_' + name + '.json') for name in ['A', 'B', 'SECONDARY']}
    targets = ['R_index_only', 'R_open_four', 'R_closed_four', 'L_index_only', 'L_open_four', 'L_closed_four', 'R_higher', 'L_higher', 'R_motion_up', 'R_motion_down', 'R_motion_horizontal', 'R_motion_still', 'R_motion_mixed', 'L_motion_up', 'L_motion_down', 'L_motion_horizontal', 'L_motion_still', 'L_motion_mixed']
    check('targets', dataset['targets'] == audit['targets'] == targets, 'target names/order')
    by_alias = unique_map(inputs['rows'], 'alias', 'coverage')
    selected_alias = unique_map(selection['rows'], 'alias', 'coverage')
    samples = unique_map(visual['samples'], 'sample_id', 'coverage')
    saved = unique_map(dataset['rows'], 'sample_id', 'coverage')
    primary = unique_map(docs['A']['rows'] + docs['B']['rows'], 'sample_id', 'coverage')
    secondary = unique_map(docs['SECONDARY']['rows'], 'sample_id', 'coverage')
    expected_ids = {f'S{i:03d}_W{j}' for i in range(1, 57) for j in range(3)}
    check('coverage', len(by_alias) == 56 and set(by_alias) == set(selected_alias), '56 aliases')
    check('coverage', set(samples) == set(primary) == set(saved) == expected_ids, '168 exact windows')
    check('coverage', set(docs['A']['pending_sample_ids']) == set(), 'A complete')
    check('coverage', len(docs['A']['rows']) == len(docs['B']['rows']) == 84, '84+84 primary')
    check('coverage', set(x['sample_id'] for x in docs['A']['rows']) == {f'S{i:03d}_W{j}' for i in range(1, 29) for j in range(3)}, 'A assignment')
    check('coverage', visual['status'] == 'COMPLETE' and visual['clips_complete'] == 56 and visual['windows_complete'] == 168, 'visual manifest complete')
    check('coverage', len({r['clip_id'] for r in inputs['rows']}) == 56, 'no duplicated clip')
    check('coverage', len({r['source_sha256'] for r in inputs['rows']}) == 56, 'no byte-duplicate source video')
    role_counts = collections.Counter(r['role'] for r in inputs['rows'])
    check('split', dict(role_counts) == {'fit': 36, 'development': 12, 'new_feature_expression_evaluation': 8}, 'clip roles')
    people = {role: sorted({r['person'] for r in inputs['rows'] if r['role'] == role}) for role in role_counts}
    words = {role: sorted({r['word_id'] for r in inputs['rows'] if r['role'] == role}) for role in role_counts}
    check('split', people == {'fit': [1, 2, 3], 'development': [4], 'new_feature_expression_evaluation': [7, 8]}, 'recorded person IDs')
    check('split', words['fit'] == words['development'] and len(words['fit']) == 12, 'same12 expressions fit/dev')
    check('split', len(words['new_feature_expression_evaluation']) == 4 and not set(words['new_feature_expression_evaluation']) & set(selection['excluded_prior16_words']), 'new4 outside prior16')
    check('split', sorted(selection['new_words']) == words['new_feature_expression_evaluation'], 'selected new words fixed')
    for word in words['fit']:
        check('split', sorted(r['person'] for r in inputs['rows'] if r['word_id'] == word) == [1, 2, 3, 4], word)
    for word in words['new_feature_expression_evaluation']:
        check('split', sorted(r['person'] for r in inputs['rows'] if r['word_id'] == word) == [7, 8], word)
    fit_ids = sorted(k for k in samples if by_alias[samples[k]['alias']]['role'] == 'fit')
    fresh_ids = [k for k in samples if by_alias[samples[k]['alias']]['role'] == 'new_feature_expression_evaluation']
    intended_secondary = sorted(set(fit_ids[::4] + fresh_ids))
    check('cross_selection', secondary_selection['sample_ids'] == intended_secondary, '27 every4th fit +24 new evaluation')
    check('cross_selection', set(secondary) == set(intended_secondary) and len(secondary) == 51, '51 unique secondary')

    # Validate recorded pins, without loading any checkpoint/model object.
    hash_check(U / 'SELECTION.json', inputs['selection_sha256'], 'pins')
    for container, field in [(dataset, 'pins'), (selection, 'pins'), (selection, 'protected')]:
        for path, expected in container[field].items():
            hash_check(path, expected, 'protected_hashes' if field == 'protected' else 'pins')
    for name, doc in docs.items():
        hash_check(U / 'RUBRIC.json', doc['rubric_sha256'], 'pins')
        # Top-level shape_order is optional; the pinned rubric defines the order.
        if 'shape_order' in doc:
            check('label_schema', doc['shape_order'] == rubric['shape_order'], name + ' explicit shape order')
        for row in doc['rows']:
            validate_label(row, name)
        image_records = list(doc.get('viewed_images', []))
        for row in doc['rows']:
            image_records.extend(row.get('viewed_images', []))
        known_image_paths = set()
        for entry in image_records:
            hash_check(entry['path'], entry['sha256'], 'image_evidence_hashes')
            known_image_paths.add(str(Path(entry['path']).resolve()))
        for row in doc['rows']:
            sid = row['sample_id']
            own_paths = doc.get('row_evidence', {}).get(sid, [x['path'] for x in row.get('viewed_images', [])])
            check('image_evidence', str(Path(samples[sid]['sheet']).resolve()) in {str(Path(p).resolve()) for p in own_paths}, name + ':' + sid + ' actual window evidence recorded')
            check('image_evidence', all(str(Path(p).resolve()) in known_image_paths for p in own_paths), name + ':' + sid + ' row evidence pinned')
    hash_check(U / 'review/SECONDARY_SELECTION.json', docs['SECONDARY']['selection_sha256'], 'pins')

    # Stored arrays and original observation PTS only: no recomputation of extraction.
    input_frames = 0
    reused = 0
    input_data = {}
    allowed_arrays = {'x', 'mask', 'rgb', 'window_features', 'window_starts', 'window_center_us', 'frame_id', 'timestamp_us'}
    for alias, row in by_alias.items():
        for field in ['input', 'record', 'observation', 'source']:
            hash_check(row[field], row[field + '_sha256'], 'input_hashes')
        original = selected_alias[alias]
        for field in ['clip_id', 'word_id', 'person', 'role', 'record_sha256', 'observation_sha256', 'frames', 'box', 'reused_input']:
            check('selection_preservation', row[field] == original[field], alias + ':' + field)
        check('selection_preservation', row['source_sha256'] == original['archive_source']['sha256'], alias + ' original public bytes')
        if row['reused_input']:
            reused += 1
            check('selection_preservation', row['input'] == original['input'] and row['input_sha256'] == original['input_sha256'], alias + ' old input exact')
        with np.load(row['input'], allow_pickle=False) as f:
            z = {k: f[k] for k in f.files}
        n = row['frames']
        check('input_arrays', set(z) == allowed_arrays, alias + ' no label/score arrays')
        check('input_arrays', z['x'].shape == z['mask'].shape == (n, 69) and z['mask'].dtype == np.bool_, alias + ' pose/mask')
        check('input_arrays', z['rgb'].shape == (n, 768), alias + ' rgb full frames')
        check('input_arrays', np.isfinite(z['x']).all() and np.isfinite(z['rgb']).all() and np.isfinite(z['window_features']).all(), alias + ' finite stored features')
        check('input_arrays', np.array_equal(z['frame_id'], np.arange(n)) and z['timestamp_us'].shape == (n,) and (np.diff(z['timestamp_us']) > 0).all(), alias + ' complete original frames/PTS')
        check('input_arrays', z['window_features'].shape == (row['windows'], 768), alias + ' feature window length')
        with np.load(row['observation'], allow_pickle=False) as obs:
            check('pts', np.array_equal(z['frame_id'], obs['frame_id']) and np.array_equal(z['timestamp_us'], obs['timestamp_us']), alias + ' original observation PTS exact')
        if original['centers'] is not None:
            check('selection_preservation', row['centers'] == original['centers'], alias + ' old fixed centers')
        else:
            # New8 selection pins the algorithm and source intervals, not decoded
            # center indices. Scalar nearest-PTS arithmetic, no preparer call.
            begin = max(0.0, min(v[0] for v in original['intervals']) - .12)
            end = min(float(z['timestamp_us'][-1]) / 1000000.0, max(v[1] for v in original['intervals']) + .12)
            timestamps = [int(t) for t in z['timestamp_us']]
            uniform_indices = []
            for ordinal in range(13):
                wanted_us = (begin + (end - begin) * ordinal / 12.0) * 1000000.0
                uniform_indices.append(min(range(n), key=lambda j: (abs(timestamps[j] - wanted_us), j)))
            check('selection_preservation', len(set(uniform_indices)) == 13, alias + '13 unique uniform PTS indices')
            check('selection_preservation', row['centers'] == [uniform_indices[j] for j in [3, 6, 9]], alias + ' new centers independently computed')
        input_data[alias] = (z['frame_id'], z['timestamp_us'])
        input_frames += n
    check('input_arrays', reused == 48 and len(by_alias) - reused == 8, '48 reused/8 new')
    for sid, sample in samples.items():
        row = by_alias[sample['alias']]
        center = sample['center_frame']
        expected_frames = list(range(center - 6, center + 7))
        check('pts', sample['frame_ids'] == expected_frames, sid + '13 consecutive original frames')
        check('pts', center == row['centers'][sample['window_index']], sid + ' preselected center')
        ids, pts = input_data[sample['alias']]
        check('pts', sample['timestamps_us'] == pts[expected_frames].tolist() and sample['center_timestamp_us'] == int(pts[center]), sid + ' exact PTS')
        check('pts', len(sample['frames']) == 13 and [x['frame_id'] for x in sample['frames']] == expected_frames and [x['timestamp_us'] for x in sample['frames']] == sample['timestamps_us'], sid + ' individual source links')
        check('roi_manifest', sample['roi_box_source_pixels'] == row['box'], sid + ' fixed ROI')
        for f in sample['frames']:
            check('visual_files', Path(f['file']).is_file(), sid + ' frame ' + str(f['frame_id']))
        for field in ['sheet', 'states', 'center_file']:
            check('visual_files', Path(sample[field]).is_file(), sid + ':' + field)
    def forbidden_keys(value):
        if isinstance(value, dict):
            return [k for k in value if k.lower() in {'word', 'word_id', 'label', 'person', 'role', 'clip_id'}] + [k for v in value.values() for k in forbidden_keys(v)]
        if isinstance(value, list):
            return [k for v in value for k in forbidden_keys(v)]
        return []
    check('blind_manifest', not forbidden_keys(visual), 'no word/person/role fields in visual manifest')

    # Independent arithmetic using explicit slots, independent of compiler functions.
    resolved = {}
    disagreements = []
    known_comparisons = known_conflicts = 0
    primary_known = secondary_primary_known = withheld_primary_known = 0
    roi_counts = {name: {field: {h: collections.Counter() for h in ['R', 'L']} for field in ['roi_center', 'roi_window']} for name in docs}
    raw_categories = collections.defaultdict(collections.Counter)
    for name, doc in docs.items():
        for row in doc['rows']:
            for field in ['roi_center', 'roi_window']:
                for h in ['R', 'L']:
                    roi_counts[name][field][h][str(row[field][h]).lower()] += 1
    def raw_compare(group, p, q):
        if p is None and q is None:
            raw_categories[group]['both_unknown'] += 1
        elif p is None or q is None:
            raw_categories[group]['one_unknown'] += 1
        else:
            raw_categories[group]['both_known'] += 1
            raw_categories[group]['disagree'] += int(p != q)
    for sid, sample in samples.items():
        p, reasons = independently_encode(primary[sid])
        vals = list(p)
        primary_known += sum(v is not None for v in p)
        if sid in secondary:
            q, _ = independently_encode(secondary[sid])
            secondary_primary_known += sum(v is not None for v in p)
            for j, (a, b) in enumerate(zip(p, q)):
                if a is not None and b is not None:
                    known_comparisons += 1
                    known_conflicts += int(a != b)
                if a != b:
                    disagreements.append(dict(sample_id=sid, target=targets[j], primary=a, secondary=b))
                    vals[j] = None
                    reasons[j] = 'review_disagreement_or_unknown'
                    withheld_primary_known += int(a is not None)
            for hand in ['R', 'L']:
                for j in range(3):
                    raw_compare('shape_bit', primary[sid]['center_shape'][hand][j], secondary[sid]['center_shape'][hand][j])
                raw_compare('motion_category', primary[sid]['motion'][hand], secondary[sid]['motion'][hand])
                for field in ['roi_center', 'roi_window']:
                    raw_compare(field, primary[sid][field][hand], secondary[sid][field][hand])
            raw_compare('bilateral_category', primary[sid]['center_relation'], secondary[sid]['center_relation'])
        resolved[sid] = vals
        record = saved[sid]
        check('compiled_rows', len(record['y']) == len(record['target_mask']) == 18 and all(type(v) is bool for v in record['target_mask']), sid + ' shape/mask types')
        for j, v in enumerate(vals):
            check('compiled_cells', record['target_mask'][j] == (v is not None) and record['y'][j] == (0.0 if v is None else v), sid + ':' + targets[j])
            check('compiled_reasons', record['reasons'][targets[j]] == reasons[j], sid + ':' + targets[j])
        check('compiled_rows', record['secondary_review'] == (sid in secondary), sid + ' secondary flag')
        check('compiled_rows', record['primary_note'] == primary[sid].get('notes', ''), sid + ' note unchanged')
        for key in ['alias', 'frame_ids', 'center_frame', 'timestamps_us', 'center_timestamp_us']:
            check('compiled_rows', record[key] == sample[key], sid + ':' + key)
        for key in ['clip_id', 'word_id', 'person', 'role', 'input', 'input_sha256']:
            check('compiled_rows', record[key] == by_alias[sample['alias']][key], sid + ':' + key)
        check('compiled_rows', record['image'] == sample['sheet'] and record['center_image'] == sample['center_file'], sid + ' evidence image links')

    support = {}
    by_person = {}
    shortfalls = {}
    thresholds = rubric['class_gate']
    for j, target in enumerate(targets):
        s = {}
        by_person[target] = {}
        failures_for_target = []
        for value, sign in [(1, 'positive'), (0, 'negative')]:
            rows = [by_alias[samples[sid]['alias']] for sid in samples if by_alias[samples[sid]['alias']]['role'] == 'fit' and resolved[sid][j] == value]
            s[sign] = {'windows': len(rows), 'clips': len({r['clip_id'] for r in rows}), 'words': len({r['word_id'] for r in rows}), 'people': len({r['person'] for r in rows})}
            by_person[target][sign] = {str(p): {'windows': sum(r['person'] == p for r in rows), 'clips': len({r['clip_id'] for r in rows if r['person'] == p}), 'words': len({r['word_id'] for r in rows if r['person'] == p})} for p in [1, 2, 3]}
            for metric, count in s[sign].items():
                minimum = thresholds['fit_' + sign + '_' + metric + '_min']
                if count < minimum:
                    failures_for_target.append({'sign': sign, 'metric': metric, 'actual': count, 'minimum': minimum})
        s['passed'] = not failures_for_target
        support[target] = s
        if failures_for_target:
            shortfalls[target] = failures_for_target
    selected = [target for target in targets if support[target]['passed']]
    families = {'shape': any(i < 6 for i, k in enumerate(targets) if k in selected), 'bilateral': any(k in selected for k in targets[6:8]), 'motion': any(k in selected for k in targets[8:])}
    rate = known_conflicts / known_comparisons if known_comparisons else 0.0
    ready = all(families.values()) and rate <= .05
    readiness = 'READY_PROVISIONAL_RGB_FEATURE_STUDY' if ready else 'NOT_READY_LABEL_SUPPORT_OR_REVIEW'
    role_summary = {}
    for role in role_counts:
        keys = [sid for sid in samples if by_alias[samples[sid]['alias']]['role'] == role]
        role_summary[role] = {'windows': len(keys), 'all_target_slots': 18 * len(keys), 'known_target_cells': sum(v is not None for sid in keys for v in resolved[sid]), 'selected_target_slots': len(selected) * len(keys), 'known_selected_cells': sum(resolved[sid][j] is not None for sid in keys for j, target in enumerate(targets) if target in selected), 'per_target': {target: {'positive': sum(resolved[sid][j] == 1 for sid in keys), 'negative': sum(resolved[sid][j] == 0 for sid in keys), 'unknown': sum(resolved[sid][j] is None for sid in keys)} for j, target in enumerate(targets)}}
    known_cells = sum(v is not None for row in resolved.values() for v in row)
    expected_audit = {'status': readiness, 'targets': targets, 'selected': selected, 'support': support, 'families': families, 'known_cross_comparisons': known_comparisons, 'known_cross_conflicts': known_conflicts, 'known_conflict_rate': rate, 'disagreements': disagreements, 'primary_windows': len(primary), 'secondary_windows': len(secondary), 'mask_count': known_cells, 'role_counts': {role: row['windows'] for role, row in role_summary.items()}, 'exclusions_not_silent': True}
    for k, value in expected_audit.items():
        check('audit_summary', audit[k] == value, k)
    check('audit_summary', dataset['selected_targets'] == selected and dataset['selected_indices'] == [targets.index(k) for k in selected], 'selected index/name match')
    check('withholding', primary_known - withheld_primary_known == known_cells, 'known supervision removed exactly once; no unknown promoted')
    if (U / 'review/CATEGORY_AGREEMENT.json').exists():
        raw_audit = read('review/CATEGORY_AGREEMENT.json')
        for group in ['shape_bit', 'motion_category', 'bilateral_category', 'roi_center', 'roi_window']:
            computed = {k: raw_categories[group][k] for k in ['both_known', 'disagree', 'one_unknown', 'both_unknown']}
            check('category_arithmetic', computed == raw_audit['groups'][group], group)

    # Re-hash key review inputs at end to reject an audit over changing labels/data.
    pinned_files = ['AUTHORIZATION.json', 'PLAN_KO.txt', 'RUBRIC.json', 'SELECTION.json', 'INPUTS.json', 'VISUAL_INDEX.json', 'DATASET.json', 'LABEL_AUDIT.json', 'build_targets.py', 'review/LABELS_A.json', 'review/LABELS_B.json', 'review/LABELS_SECONDARY.json', 'review/SECONDARY_SELECTION.json']
    pins = {str(U / name): digest(U / name) for name in pinned_files}
    pins[str(Path(__file__).resolve())] = digest(__file__)
    for p, expected in list(pins.items()):
        hash_cache.pop(str(Path(p).resolve()), None)
        check('stable_at_completion', digest(p) == expected, p)
    result = {
        'status': 'PASS' if not failures else 'FAIL',
        'reviewer': 'Codex B; separate arithmetic from root compiler, same system and original B annotator, not independent human validation',
        'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Saved input identity, schema, original frame/PTS links, exact label encoding/ROI/mask/withholding, fit-only support and readiness; no model calls or RGB relabeling.',
        'build_targets_imported_or_called': False, 'training_or_inference_run': False,
        'verifier_preflight_note': 'First verifier attempt stopped before a report because it assumed optional B.shape_order existed. Next run reported8 center mismatches because new8 SELECTION.centers is intentionally null before preparation. Final verifier independently computes the fixed13-uniform-PTS nearest indices, ordinals3/6/9. The first FAIL report and verifier are preserved in review/*FIRST_FAILURE*. No producer data or labels changed.',
        'checks': dict(checks), 'check_count': sum(checks.values()), 'failures': failures,
        'training_preparation': {'status': readiness, 'code_consistency_pass': not failures, 'provisional_support_ready': ready, 'family_gate': families, 'selected_targets': selected, 'excluded_targets': list(shortfalls), 'class_gate': thresholds, 'additional_root_preparation_condition': {'condition': 'known binary-cell conflict rate <=0.05', 'actual': rate, 'passed': rate <= .05, 'provenance': 'Root implementation additional conservative preparation diagnostic; absent as a numeric threshold from RUBRIC. Parent states written before aggregation/model outcomes. Not annotation accuracy or linguistic reliability.'}},
        'coverage': {'clips': len(by_alias), 'windows': len(samples), 'full_input_frames': input_frames, 'reused_inputs': reused, 'new_inputs': len(by_alias) - reused, 'primary_windows': len(primary), 'secondary_windows': len(secondary), 'clip_roles': dict(role_counts), 'recorded_person_ids': people, 'word_ids': words},
        'target_accounting': {'total_slots': 18 * len(samples), 'known': known_cells, 'unknown': 18 * len(samples) - known_cells, 'primary_roi_eligible_known': primary_known, 'primary_known_lost_by_review': withheld_primary_known, 'all_primary_secondary_unequal_slots': len(disagreements), 'both_known_comparisons': known_comparisons, 'both_known_conflicts': known_conflicts, 'both_known_conflict_rate': rate},
        'raw_category_agreement': {group: {k: raw_categories[group][k] for k in ['both_known', 'disagree', 'one_unknown', 'both_unknown']} for group in raw_categories},
        'roi_annotation_counts': roi_counts,
        'role_support': role_summary, 'fit_support': support, 'fit_person_support': by_person, 'excluded_target_shortfalls': shortfalls,
        'secondary_metadata_exposure': docs['SECONDARY']['source_metadata_exposure'],
        'limitations': [
            'No new visual adjudication in this audit; original RGB drafts remain unchanged. Primary B and this numerical auditor are the same Codex agent.',
            'Secondary previously saw cohort word/person/role metadata, so cross-review is not strictly word-blind and no part is independent expert KSL ground truth.',
            '671 known binary-cell comparisons include correlated one-hot negatives; 25/671 is not 96.27% annotation accuracy. Raw motion-category disagreements are separately reported.',
            'Overlapping 13-frame windows and center-only static labels are not dense all-video labels. Missing/ambiguous/ROI-ineligible labels remain masked, including disagreements.',
            'Split separation is verified by recorded IDs and file identities, not biometric identity verification. New8 are outside listed feature cohorts only; prior TCN TRAIN/native/reference-bank history remains disclosed.',
            'Stored SignRep representations use existing16-frame stride2 context and are not a strictly causal13-frame-only representation.',
            'This check does not repeat root9092 scalar feature checks or evaluate feature-extractor correctness; it independently checks saved arrays, full original frame/PTS identity and hashes.',
            'Support readiness for10 targets is not learning success, model generalization, all18-target support, language correctness, permissible-variation validation, or production adoption.'
        ],
        'pins': pins,
    }
    with OUT.open('w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    text = '\n'.join([
        '입력·주석 집계 독립 산술 검산: ' + result['status'],
        '학습 준비 판정: ' + readiness + ' (연구용10항목 지원만 의미)',
        f'{len(by_alias)}영상/{len(samples)}창, 원본{input_frames}프레임/PTS 보존, 알려진 셀 {known_cells}/{18 * len(samples)}.',
        f'교차 검토 {len(secondary)}창; 알려진 이진 셀 충돌 {known_conflicts}/{known_comparisons} ({rate:.4%}), 전체 불일치/null차이 {len(disagreements)}셀 보류.',
        '범주 차이: motion12/102, 높낮이1/47. 이진 충돌률은 주석 정확도가 아님.',
        '지원 탈락8항목과 fit 인물별 지원을 JSON에 모두 기록. 전체18항목을 통과한 상태가 아님.',
        '같은 Codex RGB 연구 초안이며 사람 수어 전문가 정답 검증 아님. Secondary는 준비 메타데이터를 미리 본 한계가 있음.',
        '원본/주석/집계 코드는 읽기 전용. build_targets 호출, 학습, 추론, 재채점 없음.',
    ]) + '\n'
    (U / 'review/INPUT_LABEL_VERIFICATION_KO.txt').write_text(text, encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'check_count', 'failures', 'target_accounting']}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
