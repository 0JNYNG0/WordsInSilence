# Local C4 bindings: verified global49 input plus wrist-relative hand coordinates.
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
U = Path(__file__).resolve().parent
sys.path.insert(0, str(U))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def read(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def stamp():
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


BINDINGS = read(U / 'BINDINGS.json')
P = BINDINGS['paths']
RUN = Path(P['run_dir']).resolve()
PREP = U / 'prepared'
INFERENCE_SHAPE = (6, 4, 10, 49)
FORBIDDEN_PATH = '/runs/ses_'
EXPECTED = {'rows': 2346, 'fit': 2089, 'internal_val': 257, 'primary': 233, 'zero_weight': 19}
CHECKS = ('INPUT_EXPANSION_CHECK.json', 'INPUT_INDEPENDENT_CHECK.json', 'CODE_REVIEW.json')
LOCAL_FILES = ('FROZEN_EXPERIMENT.json', 'CONFIG.json', 'BINDINGS.json', 'ROLE_EXCEPTION.json',
               'fullhands49.py', 'local_shape.py', 'training_engine.py', 'generic_pose_io.py', 'generic_workflow.py',
               'window_bridge.py', 'valid_window_forward.py', 'check_training_trace.py', 'runtime_binding.py',
               'prepare_inputs.py', 'run_comparison.py', 'evaluate_frozen_candidate.py',
               'check_final_metrics.py', 'check_inputs.py') + CHECKS
PREPARED_FILES = ('TRAINING_META.json', 'training.npz', 'ROW_DIAGNOSTICS.json', 'evaluation.npz',
                  'EVALUATION_INPUT_DIAGNOSTICS.json', 'EVALUATION_ROWS.json', 'INPUT_PREPARATION.json')


def verify_pins():
    if read(U / 'BINDINGS.json') != BINDINGS:
        raise SystemExit('BINDINGS.json changed after imports')
    if not BINDINGS['pins'] or any(not Path(p).is_absolute() for p in BINDINGS['pins']):
        raise SystemExit('binding pins must be a non-empty map of absolute paths')
    bad = [p for p, h in sorted(BINDINGS['pins'].items()) if not Path(p).is_file() or sha(p) != h]
    if bad:
        raise SystemExit('pinned file mismatch (%d): %s' % (len(bad), bad))


verify_pins()
sys.path.append(str(Path(P['reference_adapter_dir'])))
sys.path.append(str(Path(P['tensor_reader_dir'])))

import training_engine  # noqa: E402
import generic_pose_io as gp  # noqa: E402
import generic_workflow as gw  # noqa: E402
import window_bridge  # noqa: E402
import fullhands49  # noqa: E402
import local_shape  # noqa: E402
import valid_window_forward  # noqa: E402
import reference_adapter as ra  # noqa: E402
import stgcn_core  # noqa: E402

_BOUND = [(m, U) for m in (training_engine, gp, gw, window_bridge, fullhands49, local_shape, valid_window_forward)]
_BOUND += [(ra, Path(P['reference_adapter_dir'])), (stgcn_core, Path(P['tensor_reader_dir']))]
for _mod, _dir in _BOUND:
    if Path(_mod.__file__).resolve().parent != _dir.resolve():
        raise SystemExit('module %s was not imported from its bound directory' % _mod.__name__)
if tuple(training_engine.SAMPLE_SHAPE) != INFERENCE_SHAPE:
    raise SystemExit('training_engine.SAMPLE_SHAPE is not the C4 graph49 shape')

_CPU_CONFIGURED = False


def configure_cpu():
    global _CPU_CONFIGURED
    torch.set_num_threads(2)
    if not _CPU_CONFIGURED:
        torch.set_num_interop_threads(1)
        _CPU_CONFIGURED = True
    torch.use_deterministic_algorithms(True)


def _target(path):
    p = Path(path).resolve()
    if not (p.is_relative_to(U) or p.is_relative_to(RUN)):
        raise SystemExit('refusing to write outside U or run_dir: %s' % p)
    if p.exists():
        raise SystemExit('refusing to overwrite existing artifact: %s' % p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_bytes(path, data):
    with open(_target(path), 'xb') as fh:
        fh.write(data)


def write_json(path, obj):
    write_bytes(path, json.dumps(obj, indent=2, sort_keys=True, allow_nan=False).encode('utf-8'))


def write_npz(path, **arrays):
    with open(_target(path), 'xb') as fh:
        np.savez(fh, **arrays)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_row_path(row):
    if FORBIDDEN_PATH in str(row['path']).replace('\\', '/').lower():
        raise SystemExit('user session row is not allowed: %s' % row.get('id'))


def layout_specs49(source_specs):
    # These describe raw files. Selection belongs to make_prepare, not the NPZ schema.
    for name, spec in source_specs.items():
        if (name not in fullhands49.LAYOUTS49 or not isinstance(spec, dict)
                or spec['vertices'] != fullhands49.LAYOUTS49[name][0]):
            raise SystemExit('layout %s does not match LAYOUTS49' % name)
    return source_specs


def make_prepare(tf):
    norm = tf.CenterAndScaleNormalize(reference_points_preset=ra.NORMALIZE_PRESET, frame_level=False)

    def prepare(xy, mask, layout):
        indices = fullhands49.LAYOUTS49[layout][1]
        return gp.prepare_windows(xy, mask, indices, [3, 4], 60, 10,
                                  lambda t: norm({'frames': t})['frames'])
    return prepare


def load_local_row(row, specs, tf):
    """Keep generic_pose_io's raw validation and C2 contract, capture C4 once."""
    check_row_path(row)
    layout_specs49(specs)
    prepare_global = make_prepare(tf)
    captured = []

    def prepare(xy, mask, layout):
        if captured:
            raise SystemExit('row preparation callback was invoked more than once')
        global_tensor, diag = prepare_global(xy, mask, layout)
        indices = list(fullhands49.LAYOUTS49[layout][1])
        combined, local_diag = local_shape.append_local(
            global_tensor, xy[:, indices], mask[:, indices], diag['indices'])
        captured.append((combined, local_diag))
        return global_tensor, diag

    global_sample, diag = gp.load_row(row, specs, prepare)
    if len(captured) != 1:
        raise SystemExit('row preparation callback must run exactly once')
    combined, local_diag = captured[0]
    if (not isinstance(combined, torch.Tensor) or combined.device.type != 'cpu'
            or combined.dtype != torch.float32 or tuple(combined.shape) != (1,) + INFERENCE_SHAPE
            or not bool(torch.isfinite(combined).all())):
        raise SystemExit('local row must be finite CPU float32 [1,6,4,10,49]')
    sample = combined[0].detach().clone().contiguous()
    if sample[:, :2].contiguous().numpy().tobytes() != global_sample.numpy().tobytes():
        raise SystemExit('local row changed global channels')
    result_diag = dict(diag)
    result_diag['global_input_tensor_sha256'] = diag['input_tensor_sha256']
    result_diag['input_tensor_sha256'] = gp.tensor_sha256(sample)
    result_diag['local_diag'] = local_diag
    return sample, result_diag


def make_augment(tf):
    shear, rotate = tf.ShearTransform(.1), tf.RotatationTransform(.1)

    def transform(sample):
        return rotate(shear({'frames': sample}))['frames']

    def augment(x):
        return local_shape.augment_paired(x, transform)
    return augment


def make_factory(dpc, cfg, encoder_dict, gru_dict):
    def factory4():
        # Always rebuild from the verified original27 tensors, then expand and bind.
        enc = {k: v.clone() for k, v in encoder_dict.items()}
        gru = {k: v.clone() for k, v in gru_dict.items()}
        model27, _ = ra.make_reference_model(dpc, enc, gru, num_classes=93, seed=cfg['seed'])
        model49, _ = fullhands49.expand_model49(model27)
        model4, _ = local_shape.expand_input4(model49)
        return window_bridge.bind_window_forward(model4, valid_window_forward.forward_valid_windows)
    return factory4


def setup():
    verify_pins()
    plan, cfg = read(U / 'FROZEN_EXPERIMENT.json'), read(U / 'CONFIG.json')
    if plan['training_config'] != cfg:
        raise SystemExit('CONFIG.json differs from the frozen training_config')
    if sha(P['baseline_checkpoint']) != plan['pins']['baseline832']:
        raise SystemExit('baseline checkpoint hash differs from the frozen pin')
    source_meta = read(Path(P['source_unit']) / 'prepared' / 'TRAINING_META.json')
    if len(source_meta['rows']) != EXPECTED['rows'] or len(source_meta['classes']) != 93:
        raise SystemExit('source metadata row/class count differs from the frozen contract')
    layout_specs49(source_meta['layout_specs'])
    for row in source_meta['rows']:
        check_row_path(row)
    dpc, tf, _ = ra.load_reference(P['reference_sources'], plan['pins']['source_pins'])
    art = Path(P['artifact_dir'])
    tensors, _ = stgcn_core.read_encoder_tensors(art / 'raw_dpc.zip', art / 'TENSOR_METADATA.json',
                                                 plan['pins']['pretrained_zip'], encoder_prefix='model.')
    enc = {k[len('conv_encoder.'):]: v for k, v in tensors.items() if k.startswith('conv_encoder.')}
    gru = {k[len('agg.'):]: v for k, v in tensors.items() if k.startswith('agg.')}
    if len(enc) != 170 or len(gru) != 4:
        raise SystemExit('Required 170 encoder and 4 GRU tensors are missing')
    return plan, cfg, source_meta, make_factory(dpc, cfg, enc, gru), make_augment(tf), tf


def datasets(meta=None):
    # Only training inputs and source metadata are read; no evaluation scores.
    saved_meta = read(PREP / 'TRAINING_META.json')
    meta = saved_meta if meta is None else meta
    if meta != saved_meta:
        raise SystemExit('supplied training metadata differs from the saved metadata')
    source_prep = Path(P['source_unit']) / 'prepared'
    source_meta = read(source_prep / 'TRAINING_META.json')
    for key in ('rows', 'ids', 'origins', 'classes', 'layout_specs', 'source_split_sha256'):
        if meta[key] != source_meta[key]:
            raise SystemExit('%s differs from the source global49 metadata' % key)
    if meta['input_policy'] != read(U / 'FROZEN_EXPERIMENT.json')['input_policy']:
        raise SystemExit('prepared input policy differs from the frozen plan')
    for row in meta['rows']:
        check_row_path(row)
    layout_specs49(meta['layout_specs'])
    path = PREP / 'training.npz'
    if sha(path) != meta['cache_sha256']:
        raise SystemExit('prepared/training.npz does not match TRAINING_META cache_sha256')
    a = load_npz(path)
    rows, n = meta['rows'], len(meta['rows'])
    if sorted(a) != sorted(['x', 'y', 'weight', 'fit', 'internal_val', 'primary', 'window_valid']):
        raise SystemExit('unexpected arrays in prepared/training.npz')
    if a['x'].shape != (n,) + INFERENCE_SHAPE or a['x'].dtype != np.float32:
        raise SystemExit('xC4 shape/dtype mismatch')
    if a['window_valid'].shape != (n, 6) or a['window_valid'].dtype != np.bool_:
        raise SystemExit('window_valid shape/dtype mismatch')
    for k in ('fit', 'internal_val', 'primary'):
        if a[k].shape != (n,) or a[k].dtype != np.bool_:
            raise SystemExit('%s shape/dtype mismatch' % k)
    if a['y'].shape != (n,) or a['y'].dtype != np.int64 or a['weight'].shape != (n,) or a['weight'].dtype != np.float64:
        raise SystemExit('y/weight shape or dtype mismatch')
    if len(meta['classes']) != 93 or a['y'].min() < 0 or a['y'].max() >= 93:
        raise SystemExit('labels outside the 93 training classes')
    if not np.isfinite(a['weight']).all() or (a['weight'] < 0).any() or not a['window_valid'].any(axis=1).all():
        raise SystemExit('invalid weight or empty valid-window row')
    source_path = source_prep / 'training.npz'
    if sha(source_path) != source_meta['cache_sha256']:
        raise SystemExit('source training cache hash differs from source metadata')
    with np.load(source_path, allow_pickle=False) as source:
        for key in ('y', 'weight', 'fit', 'internal_val', 'primary', 'window_valid'):
            old = source[key]
            if (a[key].shape != old.shape or a[key].dtype != old.dtype
                    or a[key].tobytes() != old.tobytes()):
                raise SystemExit('%s differs from the source global49 cache' % key)
    expect = {'y': [int(r['y']) for r in rows], 'weight': [r['weight_count'] for r in rows],
              'fit': [r['partition'] == 'fit' for r in rows],
              'internal_val': [r['partition'] == 'internal_val' for r in rows],
              'primary': [bool(r['primary_validation']) for r in rows]}
    for k, v in expect.items():
        if a[k].tolist() != v:
            raise SystemExit('%s disagrees with TRAINING_META rows' % k)
    fit, val, primary = a['fit'], a['internal_val'], a['primary']
    counts = {'rows': n, 'fit': int(fit.sum()), 'internal_val': int(val.sum()), 'primary': int(primary.sum()),
              'zero_weight': int((a['weight'] == 0).sum())}
    if counts != EXPECTED or (fit & val).any() or not (fit | val).all() or (primary & ~val).any():
        raise SystemExit('split counts/partition mismatch: %s' % counts)
    ids, origins = [r['id'] for r in rows], [r['origin'] for r in rows]
    if (ids != meta['ids'] or origins != meta['origins'] or len(set(ids)) != n
            or not np.isfinite(a['x']).all()):
        raise SystemExit('prepared row identity or finite input check failed')

    def part(mask, with_primary=False):
        idx = np.flatnonzero(mask)
        d = {'x': torch.from_numpy(a['x'][idx]), 'y': torch.from_numpy(a['y'][idx]),
             'weight': torch.from_numpy(a['weight'][idx]), 'window_valid': torch.from_numpy(a['window_valid'][idx]),
             'ids': [ids[i] for i in idx], 'origins': [origins[i] for i in idx]}
        if with_primary:
            d['primary'] = torch.from_numpy(primary[idx])
        return d
    train, valid, all_data = part(fit), part(val, True), part(np.ones(n, dtype=bool))
    gw._check_splits(train, valid, all_data)
    return train, valid, all_data


def passed(name):
    doc = read(U / name)
    if doc.get('status') != 'PASS':
        raise SystemExit('%s is not PASS' % name)
    return doc


def context(plan, meta):
    verify_pins()
    if plan != read(U / 'FROZEN_EXPERIMENT.json') or plan['training_config'] != read(U / 'CONFIG.json'):
        raise SystemExit('supplied plan/config differs from the current frozen files')
    if meta != read(PREP / 'TRAINING_META.json') or meta['input_policy'] != plan['input_policy']:
        raise SystemExit('supplied metadata/input policy differs from the prepared files')
    expansion = passed(CHECKS[0])
    init_digest = expansion['actual_initial_state_digest']
    if (expansion['trainable_tensor_count'] != 108 or not isinstance(init_digest, str)
            or len(init_digest) != 64 or not set(init_digest) <= set('0123456789abcdef')):
        raise SystemExit('input expansion audit has an invalid initial digest or trainable tensor count')
    if expansion['source_component_sha256'] != sha(U / 'local_shape.py'):
        raise SystemExit('input expansion audit does not match the current local_shape component')
    inputs = passed('INPUT_INDEPENDENT_CHECK.json')
    expected_input_pins = {'plan_sha256': sha(U / 'FROZEN_EXPERIMENT.json'),
                           'bindings_sha256': sha(U / 'BINDINGS.json'),
                           'verifier_sha256': sha(U / 'check_inputs.py')}
    if any(inputs.get(key) != value for key, value in expected_input_pins.items()):
        raise SystemExit('independent input audit has stale plan, binding, or verifier pins')
    prepared_pins = {name: sha(PREP / name) for name in PREPARED_FILES}
    if inputs.get('artifact_sha256') != prepared_pins or inputs.get('input_policy') != plan['input_policy']:
        raise SystemExit('independent input audit does not cover the current prepared artifacts/policy')
    code = passed('CODE_REVIEW.json')
    reviewed = code.get('reviewed_source_sha256')
    required_sources = {name for name in LOCAL_FILES if name.endswith('.py')}
    if not isinstance(reviewed, dict) or not required_sources <= set(reviewed):
        raise SystemExit('code review must cover all local runtime, model, and checker sources')
    for name, expected in reviewed.items():
        if (not isinstance(name, str) or Path(name).name != name or not (U / name).is_file()
                or sha(U / name) != expected):
            raise SystemExit('code review source pin mismatch: %s' % name)
    if sha(PREP / 'training.npz') != meta['cache_sha256']:
        raise SystemExit('prepared cache hash differs from TRAINING_META')
    library_pins = dict(BINDINGS['pins'])
    library_pins.update({str(U / n): sha(U / n) for n in LOCAL_FILES})
    library_pins.update({str(PREP / n): h for n, h in prepared_pins.items()})
    return {
        'unit': str(U), 'run_dir': str(RUN),
        'plan_sha256': sha(U / 'FROZEN_EXPERIMENT.json'),
        'training_config': plan['training_config'],
        'refit_num_samples': plan['refit_num_samples'],
        'local_sha256': {n: sha(U / n) for n in LOCAL_FILES},
        'prepared_sha256': prepared_pins,
        'library_pins': library_pins,
        'binding_paths': P, 'binding_pins': BINDINGS['pins'],
        'baseline_checkpoint_sha256': sha(P['baseline_checkpoint']),
        'classes': meta['classes'],
        'input_policy': plan['input_policy'],
        'inference_shape': list(INFERENCE_SHAPE),
        'prepared_cache_sha256': meta['cache_sha256'],
        'new_init_digest': init_digest,
        'cpu_threads': 2, 'interop_threads': 1, 'deterministic': True,
    }


def trace_pass(execute=True):
    if execute:
        proc = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(U / 'check_training_trace.py')],
                              cwd=str(U), capture_output=True, text=True, encoding='utf-8')
        if proc.returncode != 0:
            print(proc.stdout[-3000:], proc.stderr[-3000:])
            raise SystemExit('check_training_trace.py failed (rc=%d)' % proc.returncode)
    doc = read(U / 'TRAINING_TRACE_AUDIT.json')
    ctx = read(U / 'EXECUTION_CONTEXT.json')
    if (doc['status'] != 'PASS' or doc['finalized'] is not True
            or doc['context_sha256'] != gw.context_sha256(ctx)
            or Path(doc['run_dir']).resolve() != RUN):
        raise SystemExit('finalized training audit does not match this execution')
    return {'status': 'PASS', 'finalized': True,
            'audit_sha256': sha(U / 'TRAINING_TRACE_AUDIT.json')}
