"""Local routing for the one predeclared geometry TCN pair. No camera or service writes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

H = Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(H), str(L), str(L / 'src'), str(L.parent / 'sign_model_pilot_v1/src')]
import geometry_tcn as G
import tools_prep_union93_candidate as U


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_config():
    c = read(H / 'CONFIG.json')
    for item in ('parent_checkpoint', 'train_cache'):
        assert sha(c[item]['path']) == c[item]['sha256'], item
    for path, digest in c['source_pins'].items():
        assert sha(path) == digest, path
    assert sha(H / 'SCOPE.md') == c['scope_sha256']
    return c


def unpack(values, offsets):
    assert offsets[0] == 0 and offsets[-1] == len(values) and np.all(np.diff(offsets) > 0)
    return [values[a:b].copy() for a, b in zip(offsets[:-1], offsets[1:])]


def extra_from_x(x):
    assert x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 280
    q = x[:, 102:186].reshape(len(x), 2, 21, 2)
    valid = x[:, 237:279].reshape(len(x), 2, 21)
    assert np.all((valid == 0) | (valid == 1))
    return G.hand_distances(q, valid.astype(bool))


def source(c):
    with np.load(c['train_cache']['path'], allow_pickle=False) as z:
        d = {k: z[k].copy() for k in ('classes', 'base_x', 'base_offsets', 'base_y',
             'pair_context_x', 'pair_context_offsets', 'pair_context_y',
             'base_indices', 'context_indices')}
    assert d['base_indices'].shape == d['context_indices'].shape == (660, 16)
    assert len(d['base_y']) == 2122 and len(d['pair_context_y']) == 224
    bx = unpack(d['base_x'], d['base_offsets'])
    cx = unpack(d['pair_context_x'], d['pair_context_offsets'])
    for xs in (bx, cx):
        assert all(x.shape[1] == 280 and x.dtype == np.float32 and np.isfinite(x).all() for x in xs)
    assert 0 <= d['base_indices'].min() <= d['base_indices'].max() < len(bx)
    assert 0 <= d['context_indices'].min() <= d['context_indices'].max() < len(cx)
    meta = read(c['metadata_path'])
    classes = d['classes'].tolist()
    for key, yy in (('base', d['base_y']), ('pair_context', d['pair_context_y'])):
        assert len(meta[key]) == len(yy)
        for r, y in zip(meta[key], yy):
            assert r['role'] == 'train'
            assert r.get('label', r.get('true')) == classes[int(y)]
            assert r['id'].startswith('NIKL_') or r.get('signer_id') in range(1, 10)
            assert not r['id'].startswith(('KSL', 'tri_', 'ses_'))
    return d, bx, cx


def parent_model(c):
    ck = torch.load(c['parent_checkpoint']['path'], map_location='cpu', weights_only=True)
    m, classes = U.rebuild_from_payload(ck)
    m.eval()
    return m, classes


def parent_predict(parent, xs, batch_size=32):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(xs), batch_size):
            batch = xs[start:start + batch_size]
            x = np.zeros((len(batch), max(map(len, batch)), 280), np.float32)
            p = np.zeros(x.shape[:2], np.float32)
            for i, a in enumerate(batch):
                x[i, :len(a)] = a
                p[i, :len(a)] = 1
            outputs.append(parent(torch.from_numpy(x), torch.from_numpy(p)).numpy().astype(np.float64))
    return np.concatenate(outputs)


def independent_geometry_check(xs, extras):
    pairs = ((4, 8), (4, 12), (4, 16), (4, 20), (0, 4), (0, 8), (0, 12), (0, 16), (0, 20))
    worst, n = 0.0, 0
    observed_counts = np.zeros(18, np.int64)
    for x, e in zip(xs, extras):
        q = x[:, 102:186].astype(np.float64).reshape(len(x), 2, 21, 2)
        present = x[:, 237:279].reshape(len(x), 2, 21) == 1
        for h in range(2):
            for j, (a, b) in enumerate(pairs):
                k = 9 * h + j
                mask = present[:, h, a] & present[:, h, b]
                expected = np.hypot(q[:, h, a, 0] - q[:, h, b, 0],
                                    q[:, h, a, 1] - q[:, h, b, 1])
                expected[~mask] = 0
                assert np.array_equal(e[:, 18 + k], mask.astype(np.float32))
                assert np.allclose(e[:, k], expected, atol=1e-6, rtol=1e-6)
                assert np.all(e[~mask, k] == 0)
                worst = max(worst, float(np.max(np.abs(e[:, k] - expected))))
                observed_counts[k] += int(mask.sum())
                n += len(x)
    return {'values_checked': n, 'maximum_absolute_difference': worst,
            'observed_per_feature': observed_counts.tolist()}


def prepare():
    c = load_config()
    assert not (H / 'INPUT_READY.json').exists(), 'reuse existing preparation, never overwrite'
    torch.set_num_threads(c['cpu_threads'])
    d, bx, cx = source(c)
    p, classes = parent_model(c)
    assert classes == d['classes'].tolist()
    be, ce = [extra_from_x(x) for x in bx], [extra_from_x(x) for x in cx]
    geometry = independent_geometry_check(bx + cx, be + ce)
    scale = G.fit_scale(be + ce, np.concatenate([d['base_y'], d['pair_context_y']]))
    assert scale.shape == (18,) and np.isfinite(scale).all() and (scale > 0).all()
    # Independent RMS, using an explicit equal-class / equal-clip accumulation.
    yall = np.concatenate([d['base_y'], d['pair_context_y']])
    num, den = np.zeros(18), np.zeros(18)
    unique = np.unique(yall)
    for label in unique:
        indices = np.flatnonzero(yall == label)
        w = 1.0 / (len(unique) * len(indices))
        for i in indices:
            e = (be + ce)[int(i)]
            for k in range(18):
                v = e[e[:, k + 18] == 1, k].astype(np.float64)
                if len(v):
                    num[k] += w * float(np.dot(v, v) / len(v))
                    den[k] += w
    expected = np.ones(18)
    valid = (num > 0) & (den > 0)
    expected[valid] = np.sqrt(num[valid] / den[valid])
    assert np.allclose(scale, expected, atol=1e-6, rtol=1e-6)
    teacher = parent_predict(p, bx)
    zero = G.GeometryTCN(p, scale, enabled=True).eval()
    zero_scores = G.predict(zero, bx, be)
    difference = float(np.max(np.abs(zero_scores - teacher)))
    assert difference <= 1e-5
    path = H / 'prepared.npz'
    with path.open('xb') as f:
        np.savez_compressed(f, base_extra=np.concatenate(be), context_extra=np.concatenate(ce),
                            scale=scale, teacher832=teacher, classes=d['classes'])
    pins = {str(H / n): sha(H / n) for n in ('CONFIG.json', 'SCOPE.md', 'geometry_tcn.py', 'run_pair.py')}
    write(H / 'INPUT_READY.json', {
        'status': 'PASS', 'at_utc': utc(), 'pins': pins,
        'prepared': {'path': str(path), 'sha256': sha(path)},
        'parent': c['parent_checkpoint'], 'source': c['train_cache'],
        'rows': {'base': len(bx), 'context': len(cx)},
        'scale_rows': 2346, 'class_balanced_per_clip_not_unique_original': True,
        'independent_geometry': geometry, 'independent_rms': True,
        'initial_logits_max_difference': difference,
        'teacher': {'model_sha256': c['parent_checkpoint']['sha256'],
                    'dtype': str(teacher.dtype), 'computation': 'parent float32 forward cast to float64'},
        'scales': scale.tolist(), 'development_arrays_read': 0, 'user_arrays_read': 0,
        'new_extraction': 0, 'training_runs': 0})
    print(json.dumps({'prepared': True, 'initial_logits_max_difference': difference, **geometry}))


def train():
    c = load_config()
    ready = read(H / 'INPUT_READY.json')
    assert ready['status'] == 'PASS'
    for path, digest in ready['pins'].items():
        assert sha(path) == digest, path
    assert sha(ready['prepared']['path']) == ready['prepared']['sha256']
    assert read(H / 'PRETRAIN_REVIEW.json')['status'] == 'PASS'
    r = Path(c['run_dir'])
    assert not r.exists(), 'training run already exists; preserve it and investigate, do not rerun'
    torch.set_num_threads(c['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    d, bx, cx = source(c)
    p, classes = parent_model(c)
    assert classes == d['classes'].tolist()
    with np.load(ready['prepared']['path'], allow_pickle=False) as z:
        assert z['classes'].tolist() == classes
        be = unpack(z['base_extra'], d['base_offsets'])
        ce = unpack(z['context_extra'], d['pair_context_offsets'])
        scale, teacher = z['scale'].copy(), z['teacher832'].copy()
    r.mkdir(exist_ok=False)
    write(r / 'TRAINING_STARTED.json', {'at_utc': utc(), 'config_sha256': sha(H / 'CONFIG.json'),
                                       'parent_sha256': c['parent_checkpoint']['sha256']})
    started = time.monotonic()

    def progress(arm, done, total, lossrecord):
        print(json.dumps({'arm': arm, 'done': done, 'total': total,
                          'elapsed_seconds': round(time.monotonic() - started, 2)}), flush=True)

    out = G.train_pair(p, bx, d['base_y'], cx, d['pair_context_y'], teacher,
                       d['base_indices'], d['context_indices'], be, ce, scale,
                       c['settings'], progress=progress)
    models, arrays = {}, {'base_indices': d['base_indices'], 'context_indices': d['context_indices']}
    initial = p.state_dict()
    for arm in ('control', 'treatment'):
        result = out[arm]
        st = result['state_dict']
        assert result['updates'] == 660
        assert result['frozen_parameters_unchanged'] and result['buffers_unchanged']
        assert result['extra_weight_nonzero'] == (arm == 'treatment')
        assert all(torch.isfinite(t).all() for t in st.values() if t.is_floating_point())
        changed = [k for k in initial if not torch.equal(initial[k], st['base.' + k])]
        assert changed and all(k.startswith(('blocks.3.', 'head.')) for k in changed)
        names = result['trainable_names']
        assert set(names) == {'extra_weight'} | {
            'base.' + k for k, _ in p.named_parameters() if k.startswith(('blocks.3.', 'head.'))}
        payload = {'kind': 'tcn_geometry_pair_v1', 'classes': classes, 'state_dict': st,
                   'enabled': arm == 'treatment', 'parent': c['parent_checkpoint'],
                   'feature_pairs': c['feature_pairs'], 'steps': 660,
                   'config_sha256': sha(H / 'CONFIG.json'), 'registered_as_default': False}
        path = r / f'checkpoint_{arm}_final.pt'
        with path.open('xb') as f:
            torch.save(payload, f)
        models[arm] = {'path': str(path), 'sha256': sha(path), 'trainable_names': names,
                       'trainable_count': sum(st[n].numel() for n in names),
                       'changed_parent_tensors': changed, 'extra_weight_nonzero': result['extra_weight_nonzero']}
        arrays[arm + '_losses'] = result['losses']
    with (r / 'training_arrays.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    load_config()
    write(r / 'training.json', {
        'status': 'BOTH_MODELS_FROZEN_BEFORE_EVALUATION', 'at_utc': utc(),
        'models': models, 'parent': c['parent_checkpoint'], 'settings': c['settings'],
        'elapsed_training_seconds': round(time.monotonic() - started, 3),
        'steps_per_arm': 660, 'training_runs': 2, 'attempts': 1,
        'source_pins': c['source_pins'], 'config_sha256': sha(H / 'CONFIG.json'),
        'prepared_sha256': sha(H / 'prepared.npz'),
        'training_arrays_sha256': sha(r / 'training_arrays.npz'),
        'feature_module_sha256': sha(H / 'geometry_tcn.py'),
        'driver_sha256': sha(H / 'run_pair.py'),
        'teacher_source': '832 current baseline only', 'evaluation_started': False,
        'development_training_rows': 0, 'user_training_rows': 0, 'ksl_training_rows': 0,
        'same_declared_parameters_but_control_zero_branch_inactive': True,
        'default_promoted': False})
    print(json.dumps({'status': 'FROZEN', 'seconds': time.monotonic() - started,
                      'models': {k: v['sha256'] for k, v in models.items()}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'train'))
    args = parser.parse_args()
    {'prepare': prepare, 'train': train}[args.action]()
