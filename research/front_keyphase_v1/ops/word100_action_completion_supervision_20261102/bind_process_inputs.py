"""Join saved direct physical labels to existing public FIT input axes only.

No model imports, feature extraction, dense target construction, or predictions.
Run once. The separate verifier checks original review -> compiled target -> join.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import copy
import hashlib
import json
import traceback
import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
N = O / 'word100_learned_motion_invariance_20261102'
ALLOWED = ('x', 'mask', 'rgb', 'prelinear', 'frame_id', 'timestamp_us', 'image_available')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def save_once(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    destination = U / 'PROCESS_INPUT_BINDING.json'
    assert not destination.exists(), 'Existing completed binding must not be overwritten'
    target_path = U / 'PROCESS_TARGETS.json'
    manifest_path = U / 'PROCESS_REVIEW_MANIFEST.json'
    targets = read(target_path)
    reviews = {r['id']: r for r in read(manifest_path)['rows']}
    inputs = [r for r in read(N / 'INPUT_MANIFEST.json')['rows']
              if r['role'] == 'fit' and r['kind'] == 'original']
    assert len(inputs) == 36 and {r['person'] for r in inputs} == set(range(1, 10))
    assert len(targets['rows']) == 45 and len(reviews) == 45
    pins = {str(p): sha(p) for p in [target_path, manifest_path,
        U / 'PROCESS_LABEL_PROTOCOL.json', U / 'PHASE_LABEL_ADDENDUM.json',
        U / 'PROCESS_INPUT_REVIEW.json', N / 'INPUT_MANIFEST.json',
        Path(__file__), U / 'verify_process_input_binding.py']}
    for item in targets['review_sources']:
        assert sha(item['path']) == item['sha256']
        pins[item['path']] = item['sha256']
    sources = {}
    axes = {}
    for row in inputs:
        cid = row['clip_id']
        assert cid not in sources
        assert sha(row['source']) == row['source_sha256'] == row['video_sha256']
        assert Path(row['source']).resolve() == Path(row['path']).resolve()
        assert sha(row['input']) == row['analysis_sha256']
        pins[row['source']] = row['source_sha256']
        pins[row['input']] = row['analysis_sha256']
        # np.load is lazy: only the named observation/feature fields are accessed.
        with np.load(row['input'], allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in ALLOWED}
        for value in arrays.values():
            value.setflags(write=False)
        ids = arrays['frame_id']
        times = arrays['timestamp_us']
        available = arrays['image_available']
        assert len(ids) == row['frames'] and np.array_equal(ids, row['source_frame_indices'])
        assert len(np.unique(ids)) == len(ids) and np.all(np.diff(times) > 0)
        assert arrays['x'].shape == arrays['mask'].shape == (len(ids), 69)
        assert arrays['rgb'].shape == arrays['prelinear'].shape == (len(ids), 768)
        assert arrays['mask'].dtype == bool and available.dtype == bool
        assert available.shape == (len(ids),)
        assert np.isfinite(arrays['x'][arrays['mask']]).all()
        assert np.isfinite(arrays['rgb']).all() and np.isfinite(arrays['prelinear']).all()
        axes[cid] = (ids, times, available)
        sources[cid] = dict(clip_id=cid, word_id=row['word_id'], person=row['person'],
            role='fit', kind='original', source=row['source'], source_sha256=row['source_sha256'],
            input=row['input'], input_sha256=row['analysis_sha256'], frames=len(ids),
            read_fields=list(ALLOWED), input_is_read_only=True,
            field_contract={key: dict(shape=list(a.shape), dtype=str(a.dtype),
                array_sha256=array_sha(a)) for key, a in arrays.items()},
            image_available_frames=int(available.sum()),
            saved_prediction_fields_read=False, actual_sensor_extent=[0, 0, 1920, 1080])

    rows = []
    counts = Counter()
    for ti, annotation in enumerate(targets['rows']):
        qid = annotation['id']
        review = reviews[qid]
        cid = annotation['clip_id']
        source = sources[cid]
        assert annotation['source_sha256'] == source['source_sha256'] == review['source_sha256']
        assert annotation['person'] == source['person'] == review['person']
        assert annotation['word_id'] == source['word_id'] == review['word_id']
        assert annotation['role'] == 'fit'
        ids, times, available = axes[cid]
        lookup = {int(frame): i for i, frame in enumerate(ids)}
        image_by_frame = {f['frame']: f for f in review['frames']}

        def point(frame):
            frame = int(frame)
            index = lookup[frame]
            image = image_by_frame[frame]
            assert abs(int(times[index]) - int(image['timestamp_us'])) <= 2
            return dict(native_frame=frame, input_index=index,
                input_timestamp_us=int(times[index]), annotation_timestamp_us=int(image['timestamp_us']),
                image_available=bool(available[index]))

        def interval(bounds):
            start, end = map(int, bounds)
            assert start < end
            context = [point(f) for f in range(start, end + 1)]
            return dict(native_interval=[start, end], context_bindings=context,
                context_is_not_dense_supervision=True)

        def sequence(seq):
            result = dict(setup=[point(f) for f in seq['setup_frames']],
                motion_context=interval(seq['motion_interval']),
                terminal=[point(f) for f in seq['terminal_frames']],
                visible_chain=seq['visible_chain'], completion_target=None)
            if 'observed_motion_witness_intervals' in seq:
                result['observed_motion_witness_contexts'] = [interval(span)
                    for span in seq['observed_motion_witness_intervals']]
            return result

        relation_bindings = []
        for ai, label in enumerate(annotation['relations']):
            bound = point(label['frame'])
            assert abs(bound['input_timestamp_us'] - int(label['timestamp_us'])) <= 2
            bound['annotation_index'] = ai
            for key in ['setup_relation', 'terminal_relation']:
                value = label[key]
                assert value is None or type(value) is int and value in (0, 1)
                bound[key + '_known'] = value is not None
                counts[key + ':' + str(value)] += 1
            relation_bindings.append(bound)
        motion_bindings = []
        for ai, label in enumerate(annotation['motion_intervals']):
            value = label['label']
            assert value is None or type(value) is int and value in (0, 1)
            bound = interval([label['start'], label['end']])
            bound.update(annotation_index=ai, label_known=value is not None,
                         supervision_unit='one_reviewed_interval', label_broadcast_to_context=False)
            motion_bindings.append(bound)
            counts['motion:' + str(value)] += 1
        sequence_bindings = []
        for ai, seq in enumerate(annotation['sequence_links']):
            sequence_bindings.append(dict(annotation_index=ai, **sequence(seq)))
            counts['chain:' + seq['visible_chain']] += 1
        variant_bindings = None
        if 'actual_terminal_variant' in annotation:
            variant = annotation['actual_terminal_variant']
            variant_bindings = dict(witnesses=[point(f) for f in variant['witness_frames']],
                sequence=sequence(variant['sequence_link']),
                canonical_relation_override=False, correctness_target=None)
        rows.append(dict(id=qid, clip_id=cid, source_input_key=cid,
            target_pointer=f'/rows/{ti}', target_sha256=pins[str(target_path)],
            annotation=copy.deepcopy(annotation), relation_bindings=relation_bindings,
            motion_interval_bindings=motion_bindings, sequence_bindings=sequence_bindings,
            actual_terminal_variant_bindings=variant_bindings,
            unlisted_targets='unknown', dense_targets_generated=False))
    assert len(rows) == 45 and len({r['clip_id'] for r in rows}) == 36
    assert dict(counts) == targets['counts'], 'Source target counts changed while binding'
    output = dict(status='SAVED_FIT_INPUTS_BOUND_WITHOUT_INFERENCE_OR_DENSE_LABELS',
        at=datetime.now(timezone.utc).isoformat(), sources=list(sources.values()), rows=rows,
        counts=dict(counts), source_clips=36, queries=45, registered_fit_ids=list(range(1, 10)),
        allowed_input_fields=list(ALLOWED), excluded_inputs='latent, probabilities, predicted stages and every other stored prediction field',
        frame_time_tolerance_us=2, source_variants_preserved=True,
        supervision='Sparse relation values and one label per reviewed motion interval; physical chains are not completion/correctness labels.',
        dependency_unit='Original public clip; overlapping queries and frames are dependent.',
        pins=pins, new_model_inference=0, new_feature_extraction=0, new_training=0,
        dev_accessed=False, private_accessed=False, whole_goal_complete=False)
    save_once(destination, output)
    print(json.dumps(dict(output=str(destination), sources=36, queries=45,
        relation_samples=sum(len(r['relation_bindings']) for r in rows),
        interval_samples=sum(len(r['motion_interval_bindings']) for r in rows),
        counts=dict(counts)), ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        failure = U / 'PROCESS_INPUT_BINDING_FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(), encoding='utf-8')
        raise
