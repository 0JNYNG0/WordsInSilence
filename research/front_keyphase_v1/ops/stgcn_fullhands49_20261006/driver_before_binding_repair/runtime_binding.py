# Shared local bindings for unit U: pin checks, bound imports, factory49, datasets and context.
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
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def stamp():
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


BINDINGS = read(U / 'BINDINGS.json')
P = BINDINGS['paths']
RUN = Path(P['run_dir']).resolve()
PREP = U / 'prepared'
INFERENCE_SHAPE = (6, 2, 10, 49)
FORBIDDEN_PATH = '/runs/ses_'
EXPECTED = {'rows': 2346, 'fit': 2089, 'internal_val': 257, 'primary': 233, 'zero_weight': 19}
CHECKS = ('GRAPH_TRANSFER_CHECK.json', 'INPUT_INDEPENDENT_CHECK.json', 'CODE_REVIEW.json')
LOCAL_FILES = ('FROZEN_EXPERIMENT.json', 'CONFIG.json', 'BINDINGS.json', 'HANDSHAPE_EVALUATION_PLAN.md',
               'fullhands49.py', 'training_engine.py', 'generic_pose_io.py', 'generic_workflow.py',
               'window_bridge.py', 'valid_window_forward.py', 'check_training_trace.py', 'runtime_binding.py',
               'prepare_inputs.py', 'run_comparison.py', 'evaluate_frozen_candidate.py') + CHECKS
PREPARED_FILES = ('TRAINING_META.json', 'training.npz', 'ROW_DIAGNOSTICS.json', 'evaluation.npz',
                  'EVALUATION_INPUT_DIAGNOSTICS.json', 'EVALUATION_ROWS.json', 'INPUT_PREPARATION.json')


def verify_pins():
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
import valid_window_forward  # noqa: E402
import reference_adapter as ra  # noqa: E402
import stgcn_core  # noqa: E402

_BOUND = [(m, U) for m in (training_engine, gp, gw, window_bridge, fullhands49, valid_window_forward)]
_BOUND += [(ra, Path(P['reference_adapter_dir'])), (stgcn_core, Path(P['tensor_reader_dir']))]
for _mod, _dir in _BOUND:
    if Path(_mod.__file__).resolve().parent != _dir.resolve():
        raise SystemExit('module %s was not imported from its bound directory' % _mod.__name__)
if tuple(training_engine.SAMPLE_SHAPE) != INFERENCE_SHAPE:
    raise SystemExit('training_engine.SAMPLE_SHAPE is not the graph49 shape')


def configure_cpu():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
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
    write_bytes(path, json.dumps(obj, indent=2, sort_keys=True).encode('utf-8'))


def write_npz(path, **arrays):
    with open(_target(path), 'xb') as fh:
        np.savez(fh, **arrays)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_row_path(row):
    if FORBIDDEN_PATH in Path(str(row['path'])).as_posix():
        raise SystemExit('user session row is not allowed: %s' % row.get('id'))


def layout_specs49(source_specs):
    # Swap each layout's 27 raw indices for the 49 selection; the 27 must be the OLD_TO_NEW subset.
    out = {}
    for name, spec in source_specs.items():
        raw_v, idx49 = fullhands49.LAYOUTS49[name]
        idx27 = [idx49[i] for i in fullhands49.OLD_TO_NEW]
        if isinstance(spec, dict):
            keys = [k for k, v in spec.items() if isinstance(v, (list, tuple)) and list(v) == idx27]
            if len(keys) != 1:
                raise SystemExit('layout %s: 27-index field not found exactly once' % name)
            out[name] = dict(spec, **{keys[0]: list(idx49)})
        elif len(spec) == 2 and spec[0] == raw_v and list(spec[1]) == idx27:
            out[name] = [raw_v, list(idx49)]
        else:
            raise SystemExit('layout %s does not match LAYOUTS49' % name)
    return out


def make_prepare(tf):
    norm = tf.CenterAndScaleNormalize(reference_points_preset=ra.NORMALIZE_PRESET, frame_level=False)

    def prepare(xy, mask, indices):
        return gp.prepare_windows(xy, mask, indices, [3, 4], 60, 10, norm)
    return prepare


def make_augment(tf):
    shear, rotate = tf.ShearTransform(.1), tf.RotatationTransform(.1)

    def transform(sample):
        return rotate(shear(sample))

    def augment(x):
        return gp.augment_windows(x, transform)
    return augment


def make_factory(dpc, cfg, encoder_dict, gru_dict):
    def factory49():
        # Always rebuild from the verified original27 tensors, then expand and bind.
        enc = {k: v.clone() for k, v in encoder_dict.items()}
        gru = {k: v.clone() for k, v in gru_dict.items()}
        model27, _ = ra.make_reference_model(dpc, enc, gru, num_classes=93, seed=cfg['seed'])
        model49, _ = fullhands49.expand_model49(model27)
        return window_bridge.bind_window_forward(model49, valid_window_forward.forward_valid_windows)
    return factory49


def setup():
    plan, cfg = read(U / 'FROZEN_EXPERIMENT.json'), read(U / 'CONFIG.json')
    if plan['training_config'] != cfg:
        raise SystemExit('CONFIG.json differs from the frozen training_config')
    if sha(P['baseline_checkpoint']) != plan['pins']['baseline832']:
        raise SystemExit('baseline checkpoint hash differs from the frozen pin')
    source_meta = read(Path(P['source_unit']) / 'prepared' / 'TRAINING_META.json')
    dpc, tf, _ = ra.load_reference(P['reference_sources'], plan['pins']['source_pins'])
    art = Path(P['artifact_dir'])
    tensors, _ = stgcn_core.read_encoder_tensors(art / 'raw_dpc.zip', art / 'TENSOR_METADATA.json',
                                                 plan['pins']['pretrained_zip'], encoder_prefix='model.')
    enc = {k[len('conv_encoder.'):]: v for k, v in tensors.items() if k.startswith('conv_encoder.')}
    gru = {k[len('agg.'):]: v for k, v in tensors.items() if k.startswith('agg.')}
    if not enc or not gru or len(enc) + len(gru) != len(tensors):
        raise SystemExit('encoder tensors are not exactly conv_encoder.* and agg.*')
    return plan, cfg, source_meta, make_factory(dpc, cfg, enc, gru), make_augment(tf), tf


def datasets(meta=None):
    # Reads only the new prepared training cache; evaluation scores are never touched here.
    meta = read(PREP / 'TRAINING_META.json') if meta is None else meta
    path = PREP / 'training.npz'
    if sha(path) != meta['cache_sha256']:
        raise SystemExit('prepared/training.npz does not match TRAINING_META cache_sha256')
    a = load_npz(path)
    rows, n = meta['rows'], len(meta['rows'])
    if sorted(a) != sorted(['x49', 'y', 'weight', 'fit', 'internal_val', 'primary', 'window_valid']):
        raise SystemExit('unexpected arrays in prepared/training.npz')
    if a['x49'].shape != (n,) + INFERENCE_SHAPE or a['x49'].dtype != np.float32:
        raise SystemExit('x49 shape/dtype mismatch')
    if a['window_valid'].shape != (n, 6) or a['window_valid'].dtype != np.bool_:
        raise SystemExit('window_valid shape/dtype mismatch')
    for k in ('fit', 'internal_val', 'primary'):
        if a[k].shape != (n,) or a[k].dtype != np.bool_:
            raise SystemExit('%s shape/dtype mismatch' % k)
    if a['y'].shape != (n,) or a['y'].dtype.kind not in 'iu' or a['weight'].shape != (n,) or a['weight'].dtype.kind != 'f':
        raise SystemExit('y/weight shape or dtype mismatch')
    if len(meta['classes']) != 93 or a['y'].min() < 0 or a['y'].max() >= 93:
        raise SystemExit('labels outside the 93 training classes')
    expect = {'y': [int(r['y']) for r in rows], 'fit': [r['partition'] == 'fit' for r in rows],
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

    def part(mask, with_primary=False):
        idx = np.flatnonzero(mask)
        d = {'x': torch.from_numpy(a['x49'][idx]), 'y': torch.from_numpy(a['y'][idx]),
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
    if str(doc.get('status', doc.get('verdict'))).upper() != 'PASS':
        raise SystemExit('%s is not PASS' % name)
    return doc


def context(plan, meta):
    graph = passed(CHECKS[0])
    for name in CHECKS[1:]:
        passed(name)
    if sha(PREP / 'training.npz') != meta['cache_sha256']:
        raise SystemExit('prepared cache hash differs from TRAINING_META')
    return {
        'unit': str(U), 'run_dir': str(RUN),
        'plan_sha256': sha(U / 'FROZEN_EXPERIMENT.json'),
        'training_config': plan['training_config'],
        'refit_num_samples': plan['refit_num_samples'],
        'local_sha256': {n: sha(U / n) for n in LOCAL_FILES},
        'prepared_sha256': {n: sha(PREP / n) for n in PREPARED_FILES},
        'binding_paths': P, 'binding_pins': BINDINGS['pins'],
        'baseline_checkpoint_sha256': sha(P['baseline_checkpoint']),
        'classes': meta['classes'],
        'inference_shape': list(INFERENCE_SHAPE),
        'prepared_cache_sha256': meta['cache_sha256'],
        'new_init_digest': graph['actual_initial_state_digest'],
        'cpu_threads': 2, 'interop_threads': 1, 'deterministic': True,
    }


def trace_pass():
    proc = subprocess.run([sys.executable, str(U / 'check_training_trace.py'), str(RUN)], cwd=str(U),
                          capture_output=True, text=True)
    if proc.returncode != 0 or 'FINALIZED_PASS' not in proc.stdout:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        raise SystemExit('check_training_trace.py did not report FINALIZED_PASS (rc=%d)' % proc.returncode)
    return {'returncode': 0, 'checked_utc': now(),
            'stdout_sha256': hashlib.sha256(proc.stdout.encode('utf-8')).hexdigest()}
