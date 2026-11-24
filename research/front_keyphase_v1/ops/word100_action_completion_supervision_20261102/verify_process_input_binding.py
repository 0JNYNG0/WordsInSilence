"""Independent saved-value checks: raw reviews -> targets -> read-only input join.

Does not import the binder or root compiler/verifier. No torch/OpenVINO/model code.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import traceback
import numpy as np

U = Path(__file__).resolve().parent
FIELDS = {'x', 'mask', 'rgb', 'prelinear', 'frame_id', 'timestamp_us', 'image_available'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_values_preserved(before, after, where='root'):
    """Added provenance is allowed; no original value, list order or type can change."""
    if isinstance(before, dict):
        assert isinstance(after, dict), where
        for key, value in before.items():
            assert key in after, (where, key)
            original_values_preserved(value, after[key], where + '/' + key)
    elif isinstance(before, list):
        assert isinstance(after, list) and len(before) == len(after), where
        for i, (a, b) in enumerate(zip(before, after)):
            original_values_preserved(a, b, where + '/' + str(i))
    else:
        assert type(before) is type(after) and before == after, (where, before, after)


def main():
    out_path = U / 'PROCESS_INPUT_BINDING_VERIFIED.json'
    assert not out_path.exists(), 'Preserve completed verifier output'
    path = U / 'PROCESS_INPUT_BINDING.json'
    bound = load(path)
    targets = load(U / 'PROCESS_TARGETS.json')
    manifest = {row['id']: row for row in load(U / 'PROCESS_REVIEW_MANIFEST.json')['rows']}
    assert bound['source_clips'] == 36 and bound['queries'] == 45
    assert set(bound['allowed_input_fields']) == FIELDS
    assert len(bound['sources']) == 36 and len(bound['rows']) == 45
    assert len({r['id'] for r in bound['rows']}) == 45
    assert {r['person'] for r in bound['sources']} == set(range(1, 10))
    for name, digest in bound['pins'].items():
        assert hash_file(name) == digest, ('pin changed', name)
    source_data = {}
    source_counts = Counter()
    for source in bound['sources']:
        assert source['role'] == 'fit' and source['kind'] == 'original'
        assert source['input_is_read_only'] and not source['saved_prediction_fields_read']
        assert set(source['read_fields']) == FIELDS and set(source['field_contract']) == FIELDS
        with np.load(source['input'], allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in FIELDS}
        for key, value in arrays.items():
            spec = source['field_contract'][key]
            assert list(value.shape) == spec['shape'] and str(value.dtype) == spec['dtype']
            canonical = str(value.dtype).encode() + json.dumps(list(value.shape)).encode() + value.tobytes(order='C')
            assert hashlib.sha256(canonical).hexdigest() == spec['array_sha256']
        ids = arrays['frame_id']
        times = arrays['timestamp_us']
        assert np.array_equal(ids, np.arange(len(ids))) and np.all(np.diff(times) > 0)
        assert arrays['image_available'].dtype == bool
        source_data[source['clip_id']] = arrays
        source_counts[source['word_id']] += 1
    assert len(source_counts) == 4 and set(source_counts.values()) == {9}

    raw_files = {}
    seen_lineages = set()
    counts = Counter()
    unique_review_frames = set()
    variant_count = 0
    comparisons = 0
    for index, item in enumerate(bound['rows']):
        target = targets['rows'][index]
        assert item['target_pointer'] == '/rows/' + str(index)
        assert item['annotation'] == target and item['id'] == target['id']
        assert item['target_sha256'] == hash_file(U / 'PROCESS_TARGETS.json')
        assert target['person'] in range(1, 10) and target['role'] == 'fit'
        raw_path = target['source_annotation']
        assert hash_file(raw_path) == target['source_annotation_sha256']
        if raw_path not in raw_files:
            raw_files[raw_path] = load(raw_path)
        raw_document = raw_files[raw_path]
        raw_rows = raw_document.get('rows', raw_document.get('records'))
        raw = next(row for row in raw_rows if row['id'] == target['id'])
        original_values_preserved(raw, target, target['id'])
        if 'annotation_lineage' in raw:
            lineage = raw['annotation_lineage']
            assert hash_file(lineage['path']) == lineage['sha256']
            original = load(lineage['path'])[lineage['source_records_key']][lineage['source_record_index']]
            assert original['id'] == lineage['source_record_id'] == raw['id']
            original_values_preserved(original, raw, raw['id'] + '/compat')
            seen_lineages.add(raw['id'])
        arrays = source_data[item['clip_id']]
        ids, ts = arrays['frame_id'], arrays['timestamp_us']
        review = manifest[item['id']]
        times = {entry['frame']: entry['timestamp_us'] for entry in review['frames']}
        assert target['source_sha256'] == review['source_sha256']
        source = next(s for s in bound['sources'] if s['clip_id'] == item['clip_id'])
        assert source['source_sha256'] == target['source_sha256']
        unique_review_frames.update((item['clip_id'], f) for f in times)

        def check_point(point, frame):
            nonlocal comparisons
            match = np.flatnonzero(ids == frame)
            assert len(match) == 1
            i = int(match[0])
            assert point['native_frame'] == frame and point['input_index'] == i
            assert point['input_timestamp_us'] == int(ts[i])
            assert point['annotation_timestamp_us'] == int(times[frame])
            assert abs(int(ts[i]) - int(times[frame])) <= 2
            assert type(point['image_available']) is bool
            assert point['image_available'] == bool(arrays['image_available'][i])
            comparisons += 1

        def check_context(context, span):
            assert context['native_interval'] == span and span[0] < span[1]
            assert context['context_is_not_dense_supervision'] is True
            frame_list = list(range(span[0], span[1] + 1))
            assert len(context['context_bindings']) == len(frame_list)
            for point, frame in zip(context['context_bindings'], frame_list):
                check_point(point, frame)
                assert 'label' not in point and 'y' not in point

        def check_sequence(binding, seq):
            assert len(binding['setup']) == len(seq['setup_frames'])
            assert len(binding['terminal']) == len(seq['terminal_frames'])
            for point, frame in zip(binding['setup'], seq['setup_frames']):
                check_point(point, frame)
            for point, frame in zip(binding['terminal'], seq['terminal_frames']):
                check_point(point, frame)
            check_context(binding['motion_context'], seq['motion_interval'])
            assert binding['visible_chain'] == seq['visible_chain']
            assert binding['completion_target'] is None
            if 'observed_motion_witness_intervals' in seq:
                assert len(binding['observed_motion_witness_contexts']) == len(seq['observed_motion_witness_intervals'])
                for context, span in zip(binding['observed_motion_witness_contexts'], seq['observed_motion_witness_intervals']):
                    check_context(context, span)

        assert len(item['relation_bindings']) == len(target['relations'])
        for ai, (binding, relation) in enumerate(zip(item['relation_bindings'], target['relations'])):
            assert binding['annotation_index'] == ai
            check_point(binding, relation['frame'])
            for key in ['setup_relation', 'terminal_relation']:
                assert binding[key + '_known'] is (relation[key] is not None)
                counts[key + ':' + str(relation[key])] += 1
        assert len(item['motion_interval_bindings']) == len(target['motion_intervals'])
        for ai, (binding, interval) in enumerate(zip(item['motion_interval_bindings'], target['motion_intervals'])):
            assert binding['annotation_index'] == ai
            check_context(binding, [interval['start'], interval['end']])
            assert binding['label_known'] is (interval['label'] is not None)
            assert binding['label_broadcast_to_context'] is False
            assert binding['supervision_unit'] == 'one_reviewed_interval'
            counts['motion:' + str(interval['label'])] += 1
        assert len(item['sequence_bindings']) == len(target['sequence_links'])
        for ai, (binding, seq) in enumerate(zip(item['sequence_bindings'], target['sequence_links'])):
            assert binding['annotation_index'] == ai
            check_sequence(binding, seq)
            counts['chain:' + seq['visible_chain']] += 1
        if 'actual_terminal_variant' in target:
            variant_count += 1
            variant = target['actual_terminal_variant']
            vb = item['actual_terminal_variant_bindings']
            assert len(vb['witnesses']) == len(variant['witness_frames'])
            for point, frame in zip(vb['witnesses'], variant['witness_frames']):
                check_point(point, frame)
            check_sequence(vb['sequence'], variant['sequence_link'])
            assert vb['canonical_relation_override'] is False and vb['correctness_target'] is None
        else:
            assert item['actual_terminal_variant_bindings'] is None
        assert item['unlisted_targets'] == 'unknown' and item['dense_targets_generated'] is False
    assert len(seen_lineages) == 18, 'Original 1567 record lineage must be preserved'
    assert len(unique_review_frames) == 1354
    assert dict(counts) == targets['counts'] == bound['counts']
    result = dict(status='PASS_SAVED_SOURCE_REVIEW_TARGET_AND_INPUT_BINDING',
        at=datetime.now(timezone.utc).isoformat(), binding_path=str(path), binding_sha256=hash_file(path),
        verifier_sha256=hash_file(__file__), source_clips=36, queries=45,
        original_1567_records_preserved_through_compat=18, unique_reviewed_source_frames=1354,
        native_frame_time_availability_links_checked=comparisons, counts=dict(counts),
        direct_review_values_preserved=True, compiled_annotation_exact=True,
        only_allowed_numpy_fields_read=True, unknown_not_negative=True,
        interval_labels_not_broadcast=True, actual_terminal_variants_preserved=variant_count,
        pinned_files_checked=len(bound['pins']),
        note='Independent saved-value algorithm from the root compiler/verifier and the binder; same Codex team, not independent human semantic adjudication.',
        new_inference=0, new_feature_extraction=0, new_training=0, dev_accessed=False,
        whole_goal_complete=False)
    with out_path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        failure = U / 'PROCESS_INPUT_BINDING_VERIFY_FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(), encoding='utf-8')
        raise
