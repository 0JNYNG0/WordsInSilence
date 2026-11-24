"""Inventory saved FIT paths; stop on query-input mismatch, without new DTW.

This is a source-contract audit, not an extent/accuracy/semantic-completion test.
Only the predeclared FIT01 example is opened after inventorying FIT metadata.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
P = O / 'word100_learned_guide_feedback_20261102'
F = O / 'word100_learned_flow_role_20261102'
B = O / 'word100_learned_action_process_20261102'
S = O / 'word100_source_component_connector_20261102'
CID = 'NIA_SL_WORD0739_REAL01_F'
TARGETS = {
    'right_elbow_lowers_hand_remains_raised',
    'right_hand_traverses_left_arm_toward_wrist',
    'bilateral_height_alternation',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence(path, pointer=None):
    out = {'path': str(Path(path).resolve()), 'sha256': sha(path)}
    if pointer is not None:
        out['pointer'] = pointer
    return out


def main():
    dest = U / 'FIT_PATH_AUDIT.json'
    assert not dest.exists(), 'Completed audit is immutable; do not rerun.'
    direct_path = S / 'DIRECT_TARGETS.json'
    direct = read(direct_path)
    fit = [r for r in direct['rows'] if r['role'] == 'fit'
           and r['person'] in range(1, 10) and r['target'] in TARGETS
           and r['kind'] == 'sequence' and r['mask'] and r['y'] == 1]
    assert len(fit) == 27
    assert all(sum(r['target'] == target for r in fit) == 9 for target in TARGETS)

    flow_path = F / 'FLOW_INPUTS.json'
    flow = read(flow_path)
    original = {v['clip_id']: (k, v) for k, v in flow['items'].items()
                if k.startswith('fit_') and v['kind'] == 'original'
                and v['person'] in range(1, 10)}
    assert len(original) == 36
    assert all(r['clip_id'] in original for r in fit)
    flow_key, saved = original[CID]
    assert saved['person'] == 1
    path_file = Path(saved['path'])
    assert sha(path_file) == saved['sha256']

    prepared = read(B / 'PREPARED.json')
    source_index, source = next((i, r) for i, r in enumerate(prepared['rows'])
        if r['clip_id'] == CID and r['person'] == 1 and r['kind'] == 'original')
    assert sha(source['cache']) == source['cache_sha256']
    assert sha(source['input']) == source['input_sha256']
    assert flow['pins'][source['cache']] == source['cache_sha256']
    pm = read(P / 'INPUT_MANIFEST.json')
    p_index, p_source = next((i, r) for i, r in enumerate(pm['rows'])
        if r['clip_id'] == CID and r['role'] == 'fit' and r['person'] == 1)
    assert sha(p_source['input']) == p_source['input_sha256']
    prediction_path = P / 'runs/projection_finetune_42/predictions' / (CID + '.npz')
    with np.load(source['cache'], allow_pickle=False) as a:
        saved_query = a['x'].copy()
    with np.load(source['input'], allow_pickle=False) as a:
        source_latent = a['latent'].copy()
        source_frames = a['frame_id'].copy()
        source_time = a['timestamp_us'].copy()
    with np.load(prediction_path, allow_pickle=False) as a:
        p_query = a['latent'].copy()
        p_frames = a['frame_id'].copy()
        p_time = a['timestamp_us'].copy()
    with np.load(p_source['input'], allow_pickle=False) as a:
        native_frames = a['frame_id'].copy()
        native_time = a['timestamp_us'].copy()
    with np.load(path_file, allow_pickle=False) as a:
        path = a['path'].copy()
        mapped_frames = a['source_frame_id'].copy()
        query_indices = a['query_frame_id'].copy()
    assert np.array_equal(saved_query, source_latent)
    assert saved_query.shape == p_query.shape == (120, 64)
    same = bool(np.array_equal(saved_query, p_query))
    assert not same, 'Mismatch no longer reproduces; do not silently change conclusion.'
    assert np.array_equal(mapped_frames, native_frames)
    assert np.array_equal(query_indices, np.arange(len(native_frames)))
    assert np.array_equal(source_frames, p_frames) and np.array_equal(p_frames, native_frames)
    assert np.array_equal(source_time, p_time) and np.array_equal(p_time, native_time)
    assert np.array_equal(np.unique(path[:, 0]), query_indices)
    model = P / 'runs/projection_finetune_42/model.pt'
    assert flow['pins'][str(model)] == sha(model)
    delta = np.abs(saved_query.astype(np.float64) - p_query.astype(np.float64))
    sources = [direct_path, flow_path, B / 'PREPARED.json',
               P / 'INPUT_MANIFEST.json', F / 'flow_model.py', F / 'prepare_flow.py',
               P / 'calibrate.py', P / 'engine.py',
               O / 'word100_learned_feature_retention_20261102/build_and_evaluate.py',
               Path(__file__)]
    out = {
        'status': 'SAVED_FIT_PATH_FOUND_BUT_QUERY_CONTRACT_DIFFERS_STOPPED_BEFORE_EXTENT_AUDIT',
        'at_utc': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'Codex child predicate_contract_review; separate code/data audit, not independent human KSL gold',
        'scope': 'FIT metadata inventory and one predeclared public FIT01 input-identity check only',
        'inventory': {
            'requested_direct_positive_sequences': 27,
            'targets': {t: 9 for t in sorted(TARGETS)},
            'saved_original_fit_paths': 36,
            'requested_sequences_with_saved_flow_path_metadata': 27,
            'saved_path_index': evidence(flow_path, '/items'),
            'exact_P_calibration_fit_paths_saved': False,
            'exact_P_calibration_path_evidence': [
                {'path': str(P / 'calibrate.py'), 'observation': 'Computes FIT engine.compare paths, but serializes stage aggregate calibration only.'},
                {'path': str(O / 'word100_learned_feature_retention_20261102/build_and_evaluate.py'), 'observation': 'FIT compare output only feeds calibration; individual serialized cases are development only.'},
            ],
            'scope_of_absence_claim': 'Inspected P/RET calibration writers and relevant public saved path sources; not a whole-disk proof of absence.',
        },
        'checked_example': {
            'clip_id': CID, 'person': 1, 'role': 'fit', 'word_id': 'WORD0739',
            'flow_key': flow_key,
            'saved_path': evidence(path_file),
            'saved_path_axes': ['query_native_index', 'learned_canonical_index'],
            'saved_path_shape': list(path.shape),
            'same_prototype_checkpoint': evidence(model),
            'flow_query_cache': evidence(source['cache']),
            'flow_query_origin': evidence(source['input']),
            'flow_source_record': evidence(B / 'PREPARED.json', '/rows/' + str(source_index)),
            'P_query_prediction': evidence(prediction_path),
            'P_input_record': evidence(P / 'INPUT_MANIFEST.json', '/rows/' + str(p_index)),
            'query_shape': list(saved_query.shape),
            'flow_cache_exactly_matches_its_source_latent': True,
            'flow_query_equals_original_P_query': same,
            'maximum_absolute_latent_difference': float(delta.max()),
            'mean_absolute_latent_difference': float(delta.mean()),
            'native_frame_ids_exactly_equal': True,
            'native_timestamp_us_exactly_equal': True,
            'first_last_native_frame': [int(native_frames[0]), int(native_frames[-1])],
            'first_last_native_timestamp_us': [int(native_time[0]), int(native_time[-1])],
            'interpretation': 'The saved flow path is real, but this query array is from a different already-extracted input chain. It cannot be called the exact P-query counterpart merely because the prototype and native times match.',
            'cause_of_latent_difference': 'Not diagnosed or inferred in this narrow audit.',
        },
        'decision': 'Stop the proposed27-path extent projection, per parent revision. Preserve current findings and proceed with existing fixed automatic window -> retained same-target observer -> source-linked demo controls.',
        'not_performed': {
            'witness_projection': True, 'coverage_or_spread_aggregation': True,
            'new_DTW_calculation': True, 'model_forward_inference': True,
            'training': True, 'new_reference_boundary_selection': True,
            'development_H_user_mentor_array_or_video_reads': True,
        },
        'limits': [
            'One actual query-identity mismatch suffices to reject the assumption of exact input equivalence; it does not establish the count of mismatching clips.',
            'Sparse first/last reviewed witnesses are not complete event boundaries or proof of completion.',
            'Same-target witnesses across people are not verified identical semantic subphase anchors.',
            'Global DTW covering the full canonical timeline is not evidence that a truncated action is complete.',
            'No improvement, new accuracy, or educational correctness claim follows from this inventory.',
        ],
        'pins': {str(p.resolve()): sha(p) for p in sources},
        'whole_goal_complete': False,
    }
    with dest.open('x', encoding='utf-8') as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
    print(out['status'])
    print('27 requested FIT sequences; saved paths found; only FIT01 identity audited; no projection or new DTW.')


if __name__ == '__main__':
    main()
