"""AO pinned Y rows plus frozen AH evidence. No model construction or forward.

Input identity includes the actual observer snapshot, not only a video name.
Old Y flow keys are assigned before filtering. Metadata is never concatenated
to a trainable tensor. Missing AH predictions fail closed by default.
"""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import sys
import numpy as np

AO = Path(__file__).resolve().parent
O = AO.parent
Y = O / 'word100_direct_occurrence_order_20261102'
AH = O / 'word100_local_event_learning_20261103'
AJ = O / 'word100_local_event_controls_20261103'
AN = O / 'word100_feedback_discrimination_20261103'
M = O / 'word100_learned_motion_order_20261102'
R = O / 'word100_learned_rgb_motion_20261102'
MODEL_SHA = '27aced2983a25de0a980d511e912c6a6bfaa3c9383ecf75fecf06bf316b5d9c6'
FIT_KINDS = {'original', 'actual_reversed_pixels'}
DEV_KINDS = {'native', 'actual_reverse_main', 'reencoded_normal', 'speed_0.8',
             'speed_1.25', 'trim_initial_rest', 'action_cut', 'main_blackout'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def verify_pins(pins):
    for path, digest in pins.items():
        require(sha(path) == digest, 'Frozen file changed: ' + path)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def y_module():
    """Old cache-only loader was independently checked to load no weights."""
    if 'readout_model' in sys.modules:
        require(Path(sys.modules['readout_model'].__file__).resolve() ==
                (Y / 'readout_model.py').resolve(), 'readout_model name collision')
    sys.path.insert(0, str(Y))
    import readout_model
    return readout_model


def selected_y_rows(split):
    require(split in {'fit', 'dev'}, 'Expected fit or dev')
    model = y_module()
    rows = model.get_rows(split)
    require(len(rows) == (252 if split == 'fit' else 88), 'Old Y row inventory changed')
    if split == 'fit':
        # The old attach helper deliberately verifies all576 known/54 held first.
        rows = model.attach_queries(rows)
        rows = [r for r in rows if r['kind'] in FIT_KINDS]
    else:
        rows = [r for r in rows if r['kind'] in DEV_KINDS or
                r['kind'].startswith(('pose_only_', 'pose_and_image_'))]
    require(len(rows) == (72 if split == 'fit' else 76), 'AO selected inventory changed')
    return rows


def npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def materialize_raw(binding):
    """Reconstruct only an explicitly frozen sensor-gap recipe, never pixels."""
    a = npz(binding['analysis']['path'])
    if binding['route'] == 'same_analysis':
        return a
    require(binding['route'] == 'sensor_gap_reconstruction', 'Unknown raw route')
    p = npz(binding['gap_prediction']['path'])
    ix = np.asarray(binding['gap_frames'], dtype=np.int64)
    require(len(ix) in {3, 7} and np.array_equal(ix, np.arange(ix[0], ix[-1] + 1)),
            'Invalid fixed gap coordinates')
    a['x'][ix] = 0
    a['mask'][ix] = False
    if binding['gap_kind'] == 'pose_and_image_7':
        a['rgb'][ix] = 0
        a['prelinear'][ix] = 0
        a['image_available'][ix] = False
    else:
        require(binding['gap_kind'] in {'pose_only_3', 'pose_only_7'}, 'Unknown gap kind')
    # These are actual historical gap observations, never remapped originals.
    for key in ['latent', 'probability', 'stage_probability',
                'retained_probability', 'relation_probability']:
        require(key in p and len(p[key]) == len(a['frame_id']), 'Gap observation shape: ' + key)
        a[key] = p[key]
    return a


def prepare_915(binding, geometry_valid):
    """Exact frozen AH arithmetic using actual matching raw/shape/relation cache."""
    a = materialize_raw(binding)
    return prepare_915_from_arrays(a, geometry_valid)


def prepare_915_from_arrays(a, geometry_valid):
    """Pure arithmetic shared by matching-input preparation and snapshot comparison."""
    for key in ['frame_id', 'timestamp_us', 'x', 'mask', 'rgb', 'image_available',
                'retained_probability', 'relation_probability']:
        require(key in a, 'Missing exact transformed observation: ' + key)
    source = io.BytesIO()
    np.savez(source, **a)
    source.seek(0)
    relations = io.BytesIO()
    np.savez(relations, probability=a['relation_probability'], frame_id=a['frame_id'],
             timestamp_us=a['timestamp_us'])
    relations.seek(0)
    helper = module_at('_ao_aj_arithmetic', AJ / 'aj_data.py')
    p = helper.prepare_arrays(source, relations, read(M / 'NORMALIZATION.json'),
                              read(R / 'RGB_NORMALIZATION.json'), geometry_valid)
    p['raw'] = a
    return p


def aux_from_prediction(saved, expected):
    """Soft frozen features only; no thresholds, labels or bag rescoring."""
    required = ['probabilities', 'motion_anchor', 'evidence_available',
                'rgb_available', 'frame_id', 'timestamp_us']
    require(all(k in saved for k in required), 'Incomplete frozen AH output')
    n = len(expected['native_frame_ids'])
    require(np.array_equal(saved['frame_id'], expected['native_frame_ids']) and
            np.array_equal(saved['timestamp_us'], expected['timestamp_us']), 'AH/Y native axis mismatch')
    p = saved['probabilities']
    require(p.shape == (n, 5) and np.isfinite(p).all() and
            bool(((p >= 0) & (p <= 1)).all()), 'AH probability shape/range')
    anchor = saved['motion_anchor']
    evidence = saved['evidence_available']
    rgb = saved['rgb_available']
    for name, value in [('motion_anchor', anchor), ('evidence_available', evidence), ('rgb_available', rgb)]:
        require(value.shape == (n,) and value.dtype.kind == 'b', 'AH availability dtype/shape')
        require(np.array_equal(value, expected['aux_masks'][name]), 'AH snapshot availability changed: ' + name)
    require(not bool((anchor & ~evidence).any()), 'Anchor outside evidence')
    require(not bool((rgb & ~evidence).any()), 'RGB outside evidence')
    aux = np.column_stack([(2 * p - 1) * anchor[:, None], anchor, evidence, rgb]).astype(np.float32)
    require(aux.shape == (n, 8) and np.isfinite(aux).all(), 'Auxiliary shape')
    require(np.all(aux[~anchor, :5] == 0), 'Missing evidence interpreted as motion')
    return aux, anchor.copy()


def additional_predictions(plan):
    path = AO / 'AUXILIARY_PREDICTIONS.json'
    if not path.exists():
        return {}
    doc = read(path)
    require(doc['input_plan_sha256'] == sha(AO / 'INPUT_PLAN.json'), 'AH output belongs to another input plan')
    require(doc['model_sha256'] == MODEL_SHA, 'Wrong AH model')
    rows = doc['rows']
    expected = {r['flow_key'] for rr in plan['rows'].values() for r in rr
                if r['aux_binding']['status'] == 'NEEDS_FROZEN_AH_FORWARD'}
    require(len(rows) == len(expected) and {r['flow_key'] for r in rows} == expected,
            'New AH output inventory must equal the frozen missing inventory')
    return {r['flow_key']: r for r in rows}


def load_rows(split, require_aux=True):
    plan = read(AO / 'INPUT_PLAN.json')
    require(plan['status'] == 'FROZEN_CACHE_INPUT_PLAN_PENDING_MISSING_AH', 'Input plan not ready')
    verify_pins(plan['pins'])
    metadata = {r['flow_key']: r for r in plan['rows'][split]}
    rows = selected_y_rows(split)
    require({r['flow_key'] for r in rows} == set(metadata), 'Y source keys changed')
    added = additional_predictions(plan)
    for r in rows:
        m = metadata[r['flow_key']]
        for key in ['clip_id', 'word_id', 'person', 'kind', 'input', 'input_sha256']:
            require(r[key] == m['y_row'][key], 'Y identity changed: ' + key)
        require(np.array_equal(r['mapping'], m['source_frame_mapping']), 'Y source mapping changed')
        require(np.array_equal(r['available'], m['y_available']), 'Y image availability changed')
        require(len(r['x']) == len(m['native_frame_ids']), 'Y native rows changed')
        for key, digest in m['y_array_hashes'].items():
            require(array_sha(r[key]) == digest, 'Y array changed: ' + key)
        r['times'] = np.asarray(m['timestamp_us'], dtype=np.int64) / 1e6
        r['native_frame_ids'] = np.asarray(m['native_frame_ids'], dtype=np.int64)
        r['timestamp_us'] = np.asarray(m['timestamp_us'], dtype=np.int64)
        r['source_group_id'] = r['clip_id']
        r['split'] = split
        r['eval_queries'] = m['eval_queries']
        if split == 'fit':
            require(r['order_queries'] == m['training_queries'], 'FIT labels changed')
        r['aux_binding'] = m['aux_binding']
        binding = m['aux_binding']
        pred = binding.get('prediction')
        if pred is None and r['flow_key'] in added:
            entry = added[r['flow_key']]
            require(entry['normalized_915_sha256'] == binding['normalized_915_sha256'],
                    'New AH output input snapshot mismatch')
            require(entry.get('model_sha256', MODEL_SHA) == MODEL_SHA, 'AH row model mismatch')
            pred = entry['prediction']
        if pred is None:
            require(not require_aux, 'Missing frozen AH output: ' + r['flow_key'])
            r.update(aux=None, aux_available=None, needs_frozen_AH_forward=True)
        else:
            require(sha(pred['path']) == pred['sha256'], 'AH prediction changed')
            aux, available = aux_from_prediction(npz(pred['path']), m)
            r.update(aux=aux, aux_available=available, needs_frozen_AH_forward=False)
    return rows


def load_missing_inputs():
    """Root runner uses these frozen inputs exactly once; this function does not score."""
    plan = read(AO / 'INPUT_PLAN.json')
    verify_pins(plan['pins'])
    result = {}
    for split, rows in plan['rows'].items():
        for row in rows:
            binding = row['aux_binding']
            if binding['status'] != 'NEEDS_FROZEN_AH_FORWARD':
                continue
            p = prepare_915(binding['input_binding'], plan['geometry_valid'])
            require(array_sha(p['x']) == binding['normalized_915_sha256'], 'Missing AH915 changed')
            require(p['ids'].tolist() == row['native_frame_ids'] and p['ts'].tolist() == row['timestamp_us'],
                    'Missing AH axis changed')
            for key, stored in [('motion_anchor', 'motion_anchor'), ('evidence_available', 'evidence_available'),
                                ('available', 'rgb_available')]:
                require(p[key].tolist() == row['aux_masks'][stored], 'Missing AH mask changed')
            p.pop('raw')
            p.update(flow_key=row['flow_key'], split=split, source_group_id=row['y_row']['clip_id'],
                     normalized_915_sha256=binding['normalized_915_sha256'])
            result[row['flow_key']] = p
    require(len(result) == plan['new_AH_forward_request']['total'], 'Missing AH budget changed')
    return result
