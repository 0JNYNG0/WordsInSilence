"""Connect already-reviewed physical facts to loss targets; never run a model.

AB inputs and completed experiments are immutable. V2 is an additive, per-view
annotation-consumption correction, not a new interpretation of sign correctness.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import datetime
import hashlib
import json
import traceback

U = Path(__file__).resolve().parent
A = U.parent / 'word100_action_completion_supervision_20261102'
SIDES = ('setup_relation', 'terminal_relation')
PINS, CACHE = {}, {}
CHECKS = Counter()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def require(value, message, group='contract'):
    CHECKS[group] += 1
    if not value:
        raise AssertionError(message)


def pin(path, expected=None):
    path = str(Path(path).resolve())
    if path not in PINS:
        PINS[path] = digest(path)
    require(expected is None or expected == PINS[path], f'Hash mismatch: {path}', 'hash')
    return PINS[path]


def read(path):
    path = str(Path(path).resolve())
    pin(path)
    if path not in CACHE:
        CACHE[path] = json.loads(Path(path).read_text(encoding='utf-8'))
    return CACHE[path]


def save(name, value):
    path = U / name
    require(not path.exists(), f'Never overwrite completed output: {path}')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def point_identity(r, idx):
    f = r['ordered_frame_map'][idx]
    require(idx == f['output_index'] and f['image_available'], 'Point requires actual available image', 'point')
    require(idx in r['direct_review']['reviewed_output_indices'], 'Point not directly inspected', 'point')
    pin(f['source_frame_path'], f['source_frame_sha256'])
    return dict(source_frame=f['source_native_frame'], output_index=idx,
                source_native_timestamp_us=f['source_native_timestamp_us'],
                view_timestamp_us=f['view_timestamp_us'], source_frame_path=f['source_frame_path'],
                source_frame_sha256=f['source_frame_sha256'], image_available=True)


def provenance(r, source_key, source_index, annotation, kind, bound=None):
    return {
        'kind': kind, 'review_source': deepcopy(r['review_source']),
        'review_field': source_key, 'review_field_index': source_index,
        'original_review_record': deepcopy(annotation),
        'bound_record': deepcopy(bound),
        'physical_query': r['physical_query'], 'predicate_definitions': deepcopy(r['predicate_definitions']),
        'review_evidence_reason': r['direct_review']['evidence_reason'],
        'review_visible_chain': r['direct_review']['visible_chain'],
        'view_id': r['view_id'], 'query_id': r['query_id'], 'role': r['role'],
        'source_sha256': r['source_sha256'], 'pixel_view_key': r['pixel_view_key'],
        'source_review_whole_word_correctness': None,
        'runtime_or_native_propagation_performed': False,
    }


def validate_row(r, q):
    dr = r['direct_review']
    require(r['query_id'] == q['id'] and r['physical_query'] == q['physical_query'], 'Query identity mismatch')
    require(r['predicate_definitions'] == q['predicate_definitions'], 'Physical predicate was changed')
    require(r['source_path'] == q['source'] and r['source_sha256'] == q['source_sha256'], 'Source identity mismatch')
    require(r['role'] == q['role'] and r['recorded_person_id'] == q['person'], 'Role/person mismatch')
    require(dr['view_id'] == r['view_id'] and dr['physical_query'] == r['physical_query'], 'Review query mismatch')
    require(dr['pixel_view_key'] == r['pixel_view_key'], 'Review pixel identity mismatch')
    require(dr['observed_process_evidence'] == r['observed_process_evidence'], 'Process label mismatch')
    require(all(dr[k] is None for k in ['whole_word_correctness', 'learner_error', 'required_repetitions']), 'Forbidden semantic truth')
    pin(r['source_path'], r['source_sha256'])
    pin(r['review_source']['path'], r['review_source']['sha256'])
    if r['construction_kind_audit_only'] != 'native':
        original = read(r['review_source']['path'])
        require(dr in original['rows'], 'Direct review is not verbatim original', 'provenance')
    fm = r['ordered_frame_map']
    require([f['output_index'] for f in fm] == list(range(len(fm))), 'Output axis malformed')
    require(dr['reviewed_output_indices'] == list(range(len(fm))), 'Review coverage mismatch')
    seams = [i for i in range(1, len(fm)) if fm[i]['source_native_frame'] != fm[i-1]['source_native_frame'] + 1]
    require(seams == r['seam_output_indices'], 'Seam mismatch')
    reviewed_native = set(q.get('reviewed_native_frames', []))
    page_frames = {v for page in q['reviewed_pages'] for v in page.get('frames', [])}
    if reviewed_native and page_frames:
        require(reviewed_native == page_frames, 'Native reviewed-frame list and page catalog disagree')
    reviewed_native = reviewed_native or page_frames
    require(bool(reviewed_native), 'No explicit reviewed-frame inventory')
    for f in fm:
        require(f['source_native_frame'] in reviewed_native, 'Native frame outside reviewed source')
        require(f['image_available'] == (f['pixel_operation'] == 'copy'), 'Availability/pixel mismatch')
    for field in ['setup_witnesses', 'terminal_witnesses']:
        bound = r['bound_' + field]
        require([p['original_witness'] for p in bound] == dr[field], 'Bound point annotations changed')
        for p in bound:
            ident = point_identity(r, p['output_index'])
            require(all(p[k] == v for k, v in ident.items()), 'Bound point source/time/image mismatch')
            require(p['original_witness']['source_frame'] == ident['source_frame'], 'Witness native frame mismatch')


def correct_row(r, q):
    validate_row(r, q)
    out = deepcopy(r)
    old_points = {v['output_index']: v for v in r['sparse_relation_targets']}
    old_intervals = {tuple(v['output_indices']): v for v in r['fully_retained_motion_interval_targets']}
    require(len(old_points) == len(r['sparse_relation_targets']), 'Duplicate original point')
    require(len(old_intervals) == len(r['fully_retained_motion_interval_targets']), 'Duplicate original interval')
    candidate = {}
    ledger, excluded, additions = [], [], []
    fm = r['ordered_frame_map']
    idx_by_frame = {f['source_native_frame']: f['output_index'] for f in fm}

    def offer(idx, side, value, source):
        require(value in (0, 1, None), 'Invalid relation value')
        ident = point_identity(r, idx)
        key = (idx, side)
        if key in candidate:
            require(candidate[key]['value'] == value, 'Direct relation conflict, including explicit unknown')
            candidate[key]['provenance'].append(source)
        else:
            candidate[key] = dict(value=value, identity=ident, provenance=[source])
        if idx in old_points:
            require(old_points[idx][side] == value, 'Native known/unknown target conflicts with new direct review')

    # Explicit numeric annotations first, including explicit unknowns. A typed
    # witness cannot overwrite an explicit0/null merely because it names a role.
    for i, p in enumerate(r['direct_review'].get('relations', [])):
        require(p['source_frame'] in idx_by_frame, 'Direct numeric point absent from view')
        idx = idx_by_frame[p['source_frame']]
        require(p.get('output_index', idx) == idx, 'Numeric point output mismatch')
        for side in SIDES:
            if side in p:
                offer(idx, side, p[side], provenance(r, 'relations', i, p, 'explicit_numeric_relation'))

    for field, side in [('setup_witnesses', SIDES[0]), ('terminal_witnesses', SIDES[1])]:
        require(bool(r['predicate_definitions'][side]), 'Missing physical predicate definition')
        for i, p in enumerate(r['bound_' + field]):
            # Meaning fixed by the audited direct-review rubric, not by source
            # word/class, annotation order, or observed_process_evidence.
            offer(p['output_index'], side, 1,
                  provenance(r, field, i, p['original_witness'], 'typed_direct_visible_relation', p))

    new_point_entries = {}
    for (idx, side), item in sorted(candidate.items()):
        value = item['value']
        state = 'already_consumed' if idx in old_points else 'unknown_excluded' if value is None else 'added'
        record = dict(output_index=idx, source_frame=fm[idx]['source_native_frame'], relation=side,
                      value=value, status=state, provenance=item['provenance'])
        ledger.append(record)
        if state == 'unknown_excluded':
            excluded.append(record)
        if state != 'added':
            continue
        if idx not in new_point_entries:
            new_point_entries[idx] = dict(source_relation_index=None, source_annotation=deepcopy(item['provenance'][0]['original_review_record']),
                                         **item['identity'], setup_relation=None, terminal_relation=None,
                                         supervision_v2_provenance=[])
        new_point_entries[idx][side] = value
        new_point_entries[idx]['supervision_v2_provenance'].append(record)
        additions.append(dict(kind='point', output_index=idx, source_frame=fm[idx]['source_native_frame'], relation=side, value=value))
    # Original list entries retain their values, provenance, order and indices.
    out['sparse_relation_targets'].extend(new_point_entries[k] for k in sorted(new_point_entries))

    direct_motion = r['direct_review'].get('motion_witnesses', [])
    require([p['original_witness'] for p in r['bound_direct_motion_witnesses']] == direct_motion, 'Bound motion annotations changed')
    new_interval_entries = {}
    direct_interval_values = {}
    for i, p in enumerate(r['bound_direct_motion_witnesses']):
        original = p['original_witness']
        explicit = 'label' in original
        require(p['label_was_explicit'] == explicit, 'Motion explicitness changed')
        require(not p['numeric_interval_target_created_from_unlabelled_witness'], 'Implicit motion promotion')
        require(not explicit or original['label'] == p['label'], 'Motion label changed')
        if explicit:
            exact_key = tuple(p['output_indices'])
            if exact_key in direct_interval_values:
                require(direct_interval_values[exact_key] == p['label'], 'Direct interval known/unknown conflict')
            direct_interval_values[exact_key] = p['label']
            if exact_key in old_intervals:
                require(old_intervals[exact_key]['label'] == p['label'], 'Direct interval conflicts with native known/unknown')
        pr = provenance(r, 'motion_witnesses', i, original, 'explicit_motion_interval' if explicit else 'unlabelled_motion_witness', p)
        if not explicit or p['label'] is None:
            excluded.append(dict(kind='motion', status='unlabelled' if not explicit else 'explicit_unknown', provenance=pr))
            continue
        indices = p['output_indices']
        native = [fm[j]['source_native_frame'] for j in indices]
        require(len(indices) >= 2 and indices == list(range(indices[0], indices[-1] + 1)), 'Noncontiguous output motion')
        require(native == p['source_frames'] == list(range(native[0], native[-1] + 1)), 'Motion crosses deletion seam')
        require(not any(indices[0] < s <= indices[-1] for s in r['seam_output_indices']), 'Motion crosses seam')
        require(all(fm[j]['image_available'] for j in indices), 'Motion crosses unavailable image')
        require(p['native_timestamps_us'] == [fm[j]['source_native_timestamp_us'] for j in indices], 'Motion native time mismatch')
        require(p['view_timestamps_us'] == [fm[j]['view_timestamp_us'] for j in indices], 'Motion view time mismatch')
        for j in indices:
            point_identity(r, j)
        key = tuple(indices)
        status = 'already_consumed' if key in old_intervals else 'added'
        if key in old_intervals:
            require(old_intervals[key]['label'] == p['label'], 'Motion target conflicts with native label/null')
        elif key in new_interval_entries:
            require(new_interval_entries[key]['label'] == p['label'], 'Direct motion target conflict')
            new_interval_entries[key]['supervision_v2_provenance'].append(pr)
        else:
            new_interval_entries[key] = dict(source_motion_interval_index=None, source_annotation=deepcopy(original),
                label=p['label'], source_frames=deepcopy(native), output_indices=deepcopy(indices),
                native_timestamps_us=deepcopy(p['native_timestamps_us']), view_timestamps_us=deepcopy(p['view_timestamps_us']),
                supervision_unit='one explicitly reviewed available contiguous interval; never dense frame targets',
                supervision_v2_provenance=[pr])
            additions.append(dict(kind='motion_interval', output_indices=deepcopy(indices), source_frames=deepcopy(native), value=p['label']))
        ledger.append(dict(kind='motion', output_indices=deepcopy(indices), value=p['label'], status=status, provenance=[pr]))
    out['fully_retained_motion_interval_targets'].extend(new_interval_entries[k] for k in sorted(new_interval_entries))
    out['supervision_v2'] = {
        'original_view_id': r['view_id'], 'physical_predicate_changed': False,
        'whole_view_process_label_changed': False, 'consumption_ledger': ledger,
        'additions': additions, 'excluded': excluded,
        'native_or_runtime_propagation_performed': False,
        'dependent_views_are_not_independent_people': True,
        'source_identity_for_multiplicity': ['source_sha256', 'physical_query', 'source_native_frame_or_exact_interval', 'relation_dimension'],
    }
    for key in r:
        if key not in ['sparse_relation_targets', 'fully_retained_motion_interval_targets']:
            require(out[key] == r[key], f'Non-target field changed: {key}', 'lossless')
    return out


def build(input_name, native_name, role, expected_count, output_name):
    original = read(A / input_name)
    native = {q['id']: q for q in read(A / native_name)['rows']}
    rows = original['rows']
    require(len(rows) == expected_count, 'Input scope count differs')
    require(all(r['role'] == role for r in rows), 'Role separation mismatch')
    allowed = range(1, 10) if role == 'fit' else range(11, 14)
    require(all(r['recorded_person_id'] in allowed for r in rows), 'Person outside approved split')
    fixed = [correct_row(r, native[r['query_id']]) for r in rows]
    counts = Counter()
    for r in fixed:
        for x in r['supervision_v2']['additions']:
            counts[x['kind']] += 1
        counts['new_point_rows'] += len(r['sparse_relation_targets']) - len(next(v for v in rows if v['view_id'] == r['view_id'])['sparse_relation_targets'])
    return output_name, {
        'at': now(), 'status': 'DIRECT_SUPERVISION_CONSUMPTION_CORRECTED_NOT_TRAINED',
        'original_artifact': {'path': str(A / input_name), 'sha256': pin(A / input_name)},
        'role': role, 'rows': fixed, 'views': len(fixed), 'additions': dict(counts),
        'training_allowed': role == 'fit', 'repeated_development': role != 'fit',
        'new_images_reviewed': 0, 'new_model_inference': 0, 'new_training': 0,
        'existing_AB_artifacts_modified': False, 'performance_gain_established': False,
        'independent_human_expert': False, 'whole_goal_complete': False,
    }


def main():
    for name in ['PROCESS_LABEL_PROTOCOL.json', 'PHASE_LABEL_ADDENDUM.json',
                 'PARTIAL_CONTROL_CONTRACT.json', 'DEV_PARTIAL_CONTRACT.json',
                 'PROCESS_WITNESS_CONSUMPTION_AUDIT.json']:
        read(A / name)
    for path, expected in read(A / 'PROCESS_WITNESS_CONSUMPTION_AUDIT.json')['pins'].items():
        pin(path, expected)
    pin(__file__)
    results = [build('PROCESS_VIEW_TARGETS.json', 'PROCESS_TARGETS.json', 'fit', 217, 'FIT_VIEW_TARGETS_V2.json'),
               build('DEV_PROCESS_VIEW_TARGETS.json', 'DEV_PROCESS_TARGETS.json', 'development', 80, 'DEV_VIEW_TARGETS_V2.json')]
    require(results[0][1]['additions'] == {'point': 41, 'new_point_rows': 37, 'motion_interval': 7}, 'FIT additions disagree with preimplementation audit')
    require(results[1][1]['additions'] == {'point': 12, 'new_point_rows': 12, 'motion_interval': 12}, 'DEV additions disagree with preimplementation audit')
    # Recheck immutable text contracts; binary media were validated once via the
    # source/review pins and are not read or transformed by any model here.
    for path, expected in PINS.items():
        if Path(path).suffix.lower() in ('.json', '.py'):
            require(digest(path) == expected, f'Input changed during compile: {path}', 'immutable')
    for name, data in results:
        data['pins'] = dict(PINS)
        save(name, data)
    save('SOURCE_EVIDENCE_REUSE_CONTRACT.json', {
        'at': now(), 'automatic_native_or_runtime_propagation': False,
        'performed_propagation_count': 0,
        'current_scope': 'Only a directly reviewed view receives its directly asserted point/interval; native rows receive no copied edit-only annotations.',
        'possible_later_point_reuse_requires_all': [
            'Exact same source video SHA, native frame ID/PTS and visible source pixel SHA; transformed/blackout pixels cannot inherit.',
            'Same physical predicate definition, anatomical side and source-query identity; no sign/guide correctness inference.',
            'Direct positive or numeric point assertion is a visible body relation independent of the edited temporal role, with its reviewed context and ambiguity documented.',
            'No conflict with native explicit0/null or variant uncertainty; conflict must be reviewed, never silently overwritten.',
            'Split preserved; DEV evidence cannot create FIT supervision. New reuse is explicit versioned compilation with deduplication, never a hidden runtime oracle.',
        ],
        'interval_reuse_requires': 'The exact contiguous sequence and visibility/context contract must be retained. Endpoint identity alone or a contained subinterval does not transfer a motion/process label.',
        'runtime_prohibition': 'New user footage never inherits a reference frame label or review boundary. Saved labels are training/evaluation evidence only.',
        'whole_view_process_labels_unchanged': True, 'dense_label_propagation': False,
    })
    save('SUPERVISION_V2_COMPILE_CHECKS.json', {
        'at': now(), 'status': 'PASS', 'checks': dict(CHECKS), 'total_checks': sum(CHECKS.values()),
        'outputs': {name: {'sha256': digest(U / name), 'views': data['views'], 'additions': data['additions']} for name, data in results},
        'new_model_inference': 0, 'new_training': 0, 'original_targets_unchanged': True,
        'independent_verification_pending': True,
    })
    print(json.dumps({'status': 'PASS', 'checks': sum(CHECKS.values()), 'outputs': {n: d['additions'] for n, d in results}}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        path = U / ('FIRST_COMPILE_FAILURE_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f') + '.txt')
        path.write_text(traceback.format_exc(), encoding='utf-8')
        raise
