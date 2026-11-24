"""FIT-only training assembly and shared read-only input adapters.

Only six sensor fields are loaded. No stored predictions, word/person metadata,
labels, frame IDs, PTS, seams or proposal bounds become neural features.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np
from process_head import QUERIES, make_input

U = Path(__file__).resolve().parent
A = U.parent / 'word100_action_completion_supervision_20261102'
FIELDS = ('x', 'mask', 'rgb', 'frame_id', 'timestamp_us', 'image_available')
SIDE = ('setup_relation', 'terminal_relation')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_sensor(path, expected_sha):
    assert sha(path) == expected_sha, path
    with np.load(path, allow_pickle=False) as z:
        assert all(k in z.files for k in FIELDS)
        a = {k: z[k].copy() for k in FIELDS}
    n = len(a['frame_id'])
    assert a['x'].shape == a['mask'].shape == (n, 69)
    assert a['rgb'].shape == (n, 768)
    assert a['timestamp_us'].shape == a['image_available'].shape == (n,)
    assert np.all(np.diff(a['timestamp_us']) > 0)
    return a


def validate_labels(r, frame_ids, availability):
    for p in r['sparse_relation_targets']:
        j = p['output_index']
        assert frame_ids[j] == p['source_frame'] and availability[j]
        assert all(p[k] in (0, 1, None) for k in SIDE)
    for p in r['fully_retained_motion_interval_targets']:
        ii, ff = p['output_indices'], p['source_frames']
        assert len(ii) >= 2 and ii == list(range(ii[0], ii[-1] + 1))
        assert ff == list(range(ff[0], ff[-1] + 1))
        assert ff == [frame_ids[j] for j in ii] and all(availability[j] for j in ii)
        assert p['label'] in (0, 1, None)


def load_case(r, source, sensor=None):
    """Drop-in equivalent of AB evaluator load_case, but never reads predictions."""
    assert source['role'] == r['role'] and source['person'] == r['recorded_person_id']
    assert source['clip_id'] == r['clip_id'] and source['source_sha256'] == r['source_sha256']
    assert r['view_id'] in source['aliases']
    a = sensor if sensor is not None else read_sensor(source['input'], source['input_sha256'])
    fm = r['ordered_frame_map']
    assert a['frame_id'].tolist() == [f['output_index'] for f in fm] == list(range(len(fm)))
    assert a['timestamp_us'].tolist() == [f['view_timestamp_us'] for f in fm]
    assert a['image_available'].tolist() == [f['image_available'] for f in fm]
    assert source['source_frame_indices'] == [f['source_native_frame'] for f in fm]
    assert source['source_timestamps_us'] == [f['source_native_timestamp_us'] for f in fm]
    validate_labels(r, source['source_frame_indices'], a['image_available'])
    return a


def load_runtime_case(binding, sensor=None):
    """Select the already frozen proposal from the original full-native sensor."""
    b = binding
    a = sensor if sensor is not None else read_sensor(b['input'], b['input_sha256'])
    ii = b['input_selection_indices']
    assert len(ii) >= 2 and ii == list(range(ii[0], ii[-1] + 1))
    axis = b['proposal_axis']
    assert len(ii) == len(axis) and [x['output_index'] for x in axis] == list(range(len(ii)))
    assert ii == [x['full_native_index'] for x in axis]
    assert a['frame_id'][ii].tolist() == [x['source_frame'] for x in axis]
    assert a['timestamp_us'][ii].tolist() == [x['source_timestamp_us'] for x in axis]
    assert a['image_available'][ii].tolist() == [x['image_available'] for x in axis]
    assert b['observed_process_evidence'] in (0, 1, None)
    assert b['process_supervision_mask'] == (b['observed_process_evidence'] is not None)
    validate_labels(b, [x['source_frame'] for x in axis], [x['image_available'] for x in axis])
    return {k: v[ii].copy() for k, v in a.items()}


def view_record(r, source):
    return dict(case_id=r['view_id'], query_id=r['query_id'], clip_id=r['clip_id'],
        role=r['role'], person=r['recorded_person_id'], physical_query=r['physical_query'],
        source_sha256=r['source_sha256'], input=source['input'], input_sha256=source['input_sha256'],
        origin='reviewed_view', array_key='view:' + r['pixel_view_key'],
        axis=[dict(output_index=f['output_index'], source_frame=f['source_native_frame'],
                   source_timestamp_us=f['source_native_timestamp_us'], input_timestamp_us=f['view_timestamp_us'],
                   image_available=f['image_available']) for f in r['ordered_frame_map']],
        predicate_definitions=r['predicate_definitions'],
        observed_process_evidence=r['observed_process_evidence'],
        sparse_relation_targets=r['sparse_relation_targets'],
        fully_retained_motion_interval_targets=r['fully_retained_motion_interval_targets'],
        nominal_process_targets=[], original_target=r,
        rgb_context=dict(kind='actual_edited_video_encoder_windows',
                         context_outside_nominal_possible=True, labels_inferred_from_rgb_context=False,
                         metadata=source['rgb_context']))


def runtime_record(b):
    return dict(case_id=b['query_id'] + '__runtime_proposal', query_id=b['query_id'], clip_id=b['clip_id'],
        role=b['role'], person=b['person'], physical_query=b['physical_query'],
        source_sha256=b['source_sha256'], input=b['input'], input_sha256=b['input_sha256'],
        origin='runtime_proposal', array_key='runtime:' + b['query_id'],
        axis=[dict(output_index=f['output_index'], source_frame=f['source_frame'],
                   source_timestamp_us=f['source_timestamp_us'], input_timestamp_us=f['source_timestamp_us'],
                   image_available=f['image_available']) for f in b['proposal_axis']],
        predicate_definitions=b['predicate_definitions'],
        observed_process_evidence=b['observed_process_evidence'],
        sparse_relation_targets=b['sparse_relation_targets'],
        fully_retained_motion_interval_targets=b['fully_retained_motion_interval_targets'],
        nominal_process_targets=[], original_target=b,
        rgb_context=dict(kind='frozen_full_native_encoder_windows_then_fixed_proposal',
                         context_outside_nominal_possible=True, labels_inferred_from_rgb_context=False))


def bind_nominal_targets(records, manifest_path, pins, required):
    if manifest_path is None:
        assert not required, 'Exact-span supervision manifest is required by this training contract'
        return dict(status='PENDING_DIRECT_NOMINAL_SPAN_SUPERVISION', records=0, known=0)
    p = Path(manifest_path).resolve()
    pins[str(p)] = sha(p)
    d = read(p)
    assert d['status'] == 'DIRECT_NOMINAL_PROCESS_TARGETS_READY'
    by_case = {r['case_id']: r for r in records}
    seen = set()
    labels = []
    for t in d['rows']:
        # A mixed audit file may contain DEV metadata. DEV rows are excluded
        # before reading any target, source array or external evidence file.
        if t['role'] != 'fit':
            continue
        assert t['span_id'] not in seen
        seen.add(t['span_id'])
        r = by_case[t['case_id']]
        assert r['role'] == t['role'] == 'fit' and t['person'] == r['person'] in range(1, 10)
        assert t['query_id'] == r['query_id'] and t['physical_query'] == r['physical_query']
        assert t['source_sha256'] == r['source_sha256'] and t['input_sha256'] == r['input_sha256']
        assert t['predicate_definitions'] == r['predicate_definitions']
        ii = t['output_indices']
        assert len(ii) >= 2 and ii == list(range(ii[0], ii[-1] + 1))
        assert 0 <= ii[0] <= ii[-1] < len(r['axis'])
        axis = [r['axis'][j] for j in ii]
        assert t['source_frames'] == [x['source_frame'] for x in axis]
        assert t['source_timestamps_us'] == [x['source_timestamp_us'] for x in axis]
        assert t['image_available'] == [x['image_available'] for x in axis]
        assert t['label_scope'] == 'nominal_visible_process_only'
        assert t['observed_process_evidence'] in (0, 1, None)
        assert t['whole_word_correctness'] is None and t['learner_error'] is None
        binding_kind = t['binding_kind']
        assert binding_kind in ('direct_nominal_review', 'explicit_direct_chain', 'contained_reviewed_absence')
        if binding_kind == 'contained_reviewed_absence':
            assert t['observed_process_evidence'] == 0
            assert t['label_inferred_from_whole_view'] is True
            assert t['parent_whole_process_truth'] == r['observed_process_evidence'] == 0
            assert t['support_containment_verified'] is True
            support = t['support_output_indices']
            reviewed = t['parent_reviewed_output_indices']
            assert set(ii) <= set(support) <= set(reviewed) <= set(range(len(r['axis'])))
        else:
            assert t['label_inferred_from_whole_view'] is False
            assert t['nominal_scope_reviewed'] is True
            if binding_kind == 'explicit_direct_chain':
                assert t['observed_process_evidence'] == 1
                witnesses = t['explicit_chain_witness_output_indices']
                assert witnesses and set(witnesses) <= set(ii)
                assert all(r['axis'][j]['image_available'] for j in witnesses)
        assert t['rgb_context_outside_nominal_possible'] is True
        review = t['direct_review']
        assert review['reason'] and review['path'] and review['sha256']
        assert sha(review['path']) == review['sha256']
        pins[str(Path(review['path']).resolve())] = review['sha256']
        # Nominal span may contain an edit seam or missing images. Keep direct
        # review, explicit positive-chain containment and logically contained
        # reviewed absence distinguishable; never copy a whole-view positive.
        r['nominal_process_targets'].append(t)
        labels.append(t['observed_process_evidence'])
    assert not required or labels, 'Required FIT nominal labels not provided'
    return dict(status='BOUND_DIRECT_NOMINAL_PROCESS_LABELS', records=len(labels),
                known=sum(v is not None for v in labels), positive=sum(v == 1 for v in labels),
                negative=sum(v == 0 for v in labels), unknown=sum(v is None for v in labels),
                manifest=str(p), manifest_sha256=pins[str(p)])


def build_fit_dataset(span_manifest=None, require_spans=False):
    target_path = U / 'FIT_VIEW_TARGETS_V2.json'
    input_path = A / 'PROCESS_INPUT_MANIFEST.json'
    runtime_path = A / 'PROCESS_RUNTIME_SUPERVISION_BINDING.json'
    pins = {str(p): sha(p) for p in [target_path, input_path, runtime_path]}
    targets = read(target_path)['rows']
    inputs = {r['pixel_view_key']: r for r in read(input_path)['rows']}
    assert len(targets) == 217 and len(inputs) == 216
    arrays, records = {}, []
    for k, src in inputs.items():
        assert src['role'] == 'fit' and src['person'] in range(1, 10)
        pins[str(Path(src['input']).resolve())] = src['input_sha256']
        arrays['view:' + k] = read_sensor(src['input'], src['input_sha256'])
    # Keep exactly the old FIT216-only normalization algorithm. Additional
    # contexts, annotation copies and all DEV data cannot alter this scaler.
    valid = np.concatenate([arrays['view:' + k]['rgb'][arrays['view:' + k]['image_available']]
                            for k in inputs], axis=0)
    stats = dict(mean=valid.mean(0, dtype=np.float64).tolist(),
                 std=np.maximum(valid.std(0, dtype=np.float64), .001).tolist(),
                 source='Original FIT216 unique actual pixel/time inputs only; same algorithm as AB; no runtime/DEV duplication',
                 frames=len(valid))
    for r in targets:
        assert r['role'] == 'fit' and r['recorded_person_id'] in range(1, 10)
        src = inputs[r['pixel_view_key']]
        load_case(r, src, arrays['view:' + r['pixel_view_key']])
        records.append(view_record(r, src))
    runtime_rows = [b for b in read(runtime_path)['rows'] if b['role'] == 'fit']
    assert len(runtime_rows) == 45
    native_cache = {}
    for b in runtime_rows:
        assert b['person'] in range(1, 10) and b['training_role_eligible']
        assert b['development_training_allowed'] is False
        path = str(Path(b['input']).resolve())
        if path not in native_cache:
            native_cache[path] = read_sensor(path, b['input_sha256'])
        pins[path] = b['input_sha256']
        r = runtime_record(b)
        arrays[r['array_key']] = load_runtime_case(b, native_cache[path])
        records.append(r)
    assert len(native_cache) == 36 and len(records) == 262
    assert len({r['case_id'] for r in records}) == len(records)
    groups = defaultdict(list)
    for r in records:
        assert r['physical_query'] in QUERIES
        g = (r['person'], r['source_sha256'], r['physical_query'])
        r['group_key'] = g
        groups[g].append(r)
    assert len(groups) == 45
    spans = bind_nominal_targets(records, span_manifest, pins, require_spans)
    sequences = {k: make_input(a, stats) for k, a in arrays.items()}
    return dict(records=records, arrays=arrays, sequences=sequences, stats=stats,
                groups=dict(groups), pins=pins, nominal_span_status=spans,
                sensor_fields_loaded=list(FIELDS), prediction_fields_loaded=[],
                counts=dict(reviewed_views=217, runtime_proposals=45, groups=45,
                            source_clips=36, unique_edited_inputs=216, unique_native_inputs=36),
                development_arrays_opened=0, model_forwards=0, training_steps=0)


def process_identity(r, indices=None):
    axis = r['axis'] if indices is None else [r['axis'][j] for j in indices]
    return (tuple(x['source_frame'] for x in axis), tuple(x['image_available'] for x in axis))


def load_eval_dataset(stats, include_runtime=True):
    """Evaluation-only loading. Never fit a scaler or select a candidate here.

    This function is intentionally not called by training or preparation checks.
    Both FIT and DEV use the same schema/feature transform and corrected targets.
    """
    records, arrays, pins = [], {}, {}
    for role, target_file, input_file, expected, allowed in [
        ('fit', 'FIT_VIEW_TARGETS_V2.json', 'PROCESS_INPUT_MANIFEST.json', 217, set(range(1, 10))),
        ('development', 'DEV_VIEW_TARGETS_V2.json', 'DEV_PROCESS_INPUT_MANIFEST.json', 80, {11, 12, 13}),
    ]:
        tp, ip = U / target_file, A / input_file
        pins[str(tp)] = sha(tp)
        pins[str(ip)] = sha(ip)
        targets = read(tp)['rows']
        inputs = {r['pixel_view_key']: r for r in read(ip)['rows']}
        assert len(targets) == expected
        for r in targets:
            assert r['role'] == role and r['recorded_person_id'] in allowed
            src = inputs[r['pixel_view_key']]
            arkey = 'view:' + r['pixel_view_key']
            if arkey not in arrays:
                arrays[arkey] = read_sensor(src['input'], src['input_sha256'])
            pins[str(Path(src['input']).resolve())] = src['input_sha256']
            load_case(r, src, arrays[arkey])
            records.append(view_record(r, src))
    if include_runtime:
        rp = A / 'PROCESS_RUNTIME_SUPERVISION_BINDING.json'
        pins[str(rp)] = sha(rp)
        raw_cache = {}
        runtime_rows = read(rp)['rows']
        assert len(runtime_rows) == 60
        for b in runtime_rows:
            assert (b['role'] == 'fit' and b['person'] in range(1, 10)) or (b['role'] == 'development' and b['person'] in (11, 12, 13))
            path = str(Path(b['input']).resolve())
            if path not in raw_cache:
                raw_cache[path] = read_sensor(path, b['input_sha256'])
            pins[path] = b['input_sha256']
            r = runtime_record(b)
            arrays[r['array_key']] = load_runtime_case(b, raw_cache[path])
            records.append(r)
    assert len(records) == (357 if include_runtime else 297)
    assert len({r['case_id'] for r in records}) == len(records)
    return dict(records=records, arrays=arrays,
                sequences={k: make_input(a, stats) for k, a in arrays.items()},
                stats=stats, pins=pins, normalization_refit=False,
                training_allowed=False, model_forwards=0)


def multiplicities(records):
    """Each actual source annotation has total weight1 across copied contexts."""
    out = {}
    by_group = defaultdict(list)
    for r in records:
        by_group[r['group_key']].append(r)
    for g, rows in by_group.items():
        counts = {k: defaultdict(int) for k in ('whole', 'nominal', 'point', 'netmotion')}
        truth = {k: {} for k in counts}
        def add(kind, key, value):
            if value is None:
                return
            if key in truth[kind]:
                assert truth[kind][key] == value, 'Conflicting duplicate physical supervision'
            truth[kind][key] = value
            counts[kind][key] += 1
        for r in rows:
            add('whole', process_identity(r), r['observed_process_evidence'])
            for t in r['nominal_process_targets']:
                add('nominal', process_identity(r, t['output_indices']), t['observed_process_evidence'])
            for t in r['sparse_relation_targets']:
                for dim, field in enumerate(SIDE):
                    add('point', (t['source_frame'], dim), t[field])
            for t in r['fully_retained_motion_interval_targets']:
                add('netmotion', tuple(t['source_frames']), t['label'])
        out[g] = counts
    return out
