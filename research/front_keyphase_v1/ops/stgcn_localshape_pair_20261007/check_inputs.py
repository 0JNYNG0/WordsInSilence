"""Independently audit every C4 input against frozen C2-49 and raw NPZ data.

No candidate preparation helpers, model, Torch, training, or score archives are
imported/read. Only the selected audit JSON is written, without overwriting.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import datetime
import hashlib
import io
import json
import math
import statistics

import numpy as np


U = Path(__file__).resolve().parent
PREP = U / 'prepared'
POLICY = 'provided_xy_global49_plus_wrist_local_xy_v1'
PLAN_SHA = '1ac4e015b1a2caefa495624b004eaad514e9e676b84db67bd8e988935c8e32f3'
COMPONENT_SHA = 'c1f2c8697757117f12b7868ca10ee5718c17c998c118b053fd7d4c7d93d25983'
LOCAL_RTOL, LOCAL_ATOL = 1e-7, 1e-7
SPECS = {
    'native67': {'xy_key': 'xy_native', 'mask_key': 'coordinate_present',
                 'time_key': 'timestamp_nominal_us', 'vertices': 67,
                 'hand_slices': {'left': [25, 46], 'right': [46, 67]}},
    'mp75': {'xy_key': 'xy_px', 'mask_key': 'observed', 'time_key': 'timestamp_us',
             'vertices': 75, 'hand_slices': {'left': [33, 54], 'right': [54, 75]}},
}
JOINTS = {'native67': [0, 16, 15, 5, 2, 6, 3] + list(range(25, 67)),
          'mp75': [0, 2, 5, 11, 12, 13, 14] + list(range(33, 75))}
FILES = ('training.npz', 'TRAINING_META.json', 'ROW_DIAGNOSTICS.json',
         'evaluation.npz', 'EVALUATION_ROWS.json', 'EVALUATION_INPUT_DIAGNOSTICS.json',
         'INPUT_PREPARATION.json')
GROUPS = {'annotation_dev': 699, 'whole_dev': 74, 'public': 32, 'mp': 14, 'ksl36': 36}


def need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def tensor_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(f'{value.dtype.str}|{value.shape}|'.encode('ascii'))
    h.update(value.tobytes())
    return h.hexdigest()


def exact(a, b, message):
    need(a.dtype == b.dtype and a.shape == b.shape
         and a.tobytes(order='C') == b.tobytes(order='C'), message)


def stat(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns


def indices_for(length):
    need(type(length) is int and length > 0, 'Invalid clip length')
    return ([-1] * (60 - length) + list(range(length)) if length <= 60
            else [i * (length - 1) // 59 for i in range(60)])


def pin(pins, path):
    matches = [h for p, h in pins.items() if Path(p).resolve() == Path(path).resolve()]
    need(len(matches) == 1 and file_sha(path) == matches[0], f'Input pin mismatch: {path}')


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def independent_local(xy, observed):
    """Scalar distance/median arithmetic; never calls the implementation oracle."""
    out = np.zeros(xy.shape, dtype=np.float32)
    zero_points = np.ones(observed.shape, dtype=bool)
    diagnostics = {}
    for name, base in (('left', 7), ('right', 28)):
        distances = [math.dist([float(v) for v in xy[t, base]],
                               [float(v) for v in xy[t, base + off]])
                     for t in range(len(xy)) for off in (5, 9, 13, 17)
                     if observed[t, base] and observed[t, base + off]]
        scale = statistics.median(distances) if distances else None
        status = ('no_valid_pairs' if scale is None else
                  'nonfinite_scale' if not math.isfinite(scale) else
                  'degenerate_scale' if scale <= 1e-6 else 'ok')
        if status != 'ok':
            scale = None
        count = 0
        if scale is not None:
            for t in range(len(xy)):
                if observed[t, base]:
                    for off in range(21):
                        if observed[t, base + off]:
                            for c in (0, 1):
                                out[t, base + off, c] = (float(xy[t, base + off, c])
                                                         - float(xy[t, base, c])) / scale
                            zero_points[t, base + off] = off == 0
                            count += 1
        diagnostics[name] = {'scale': scale, 'count_valid_pairs': len(distances),
                             'valid_local_points': count, 'status': status}
    need(np.isfinite(out).all(), 'Independent local reconstruction overflow')
    return out, zero_points, diagnostics


def check_local_diag(actual, expected, frames, label):
    # append_local() returns a container whose "local" entry is the scale audit.
    d = actual['local'] if 'local' in actual else actual
    need(d['num_frames'] == frames and d['min_scale'] == 1e-6, f'{label}: local diagnostic policy')
    for name, hand in expected.items():
        for key in ('count_valid_pairs', 'valid_local_points', 'status'):
            need(d[name][key] == hand[key], f'{label}: local {name}/{key}')
        a, b = d[name]['scale'], hand['scale']
        need((a is None and b is None) or (a is not None and b is not None
             and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)), f'{label}: local {name}/scale')


def check_diag(row, diag, old, x, valid, label):
    for key in ('source_sha256', 'layout', 'slice', 'mask_hand', 'hand_slice'):
        need(diag[key] == old[key], f'{label}: original diagnostic {key}')
    need(diag['id'] == row['id'] and diag['source_sha256'] == row['sha256'], f'{label}: identity/hash')
    need(diag['layout'] == row['layout'] and diag['slice'] == [row['start'], row['end_exclusive']],
         f'{label}: original layout/slice')
    hand = row.get('mask_hand')
    need(hand is None or hand in SPECS[row['layout']]['hand_slices'], f'{label}: masked hand')
    need(diag['mask_hand'] == hand and diag['hand_slice'] ==
         (SPECS[row['layout']]['hand_slices'][hand] if hand else None), f'{label}: hand-mask extent')
    frames = row['end_exclusive'] - row['start']
    prepare = diag['prepare_diag']
    need(prepare == old['prepare_diag'], f'{label}: original global preparation diagnostics')
    need(prepare['T'] == frames and prepare['indices'] == indices_for(frames), f'{label}: sampling/time')
    need(prepare['shape'] == [1, 6, 2, 10, 49], f'{label}: global diagnostic shape')
    expected_valid = (np.asarray(indices_for(frames)).reshape(6, 10) >= 0).any(1)
    exact(valid, expected_valid, f'{label}: independent temporal validity')
    need(diag['input_tensor_sha256'] == tensor_sha(x), f'{label}: actual C4 digest')
    need('local_diag' in diag, f'{label}: local diagnostic missing')
    if 'global_input_tensor_sha256' in diag:
        need(diag['global_input_tensor_sha256'] == tensor_sha(x[:, :2]), f'{label}: global digest')
    if 'valid_windows' in diag:
        need(diag['valid_windows'] == int(valid.sum()), f'{label}: valid-window count')


def audit():
    need(all((PREP / name).is_file() for name in FILES), 'Prepared inputs incomplete; no raw input read')
    need(file_sha(U / 'local_shape.py') == COMPONENT_SHA, 'Claude component changed')
    need(file_sha(U / 'FROZEN_EXPERIMENT.json') == PLAN_SHA, 'Frozen plan changed')
    plan, bindings = read(U / 'FROZEN_EXPERIMENT.json'), read(U / 'BINDINGS.json')
    source = Path(bindings['paths']['source_unit'])
    old_prep = source / 'prepared'
    frozen_path = Path(bindings['paths']['method_run']) / 'FROZEN_MODEL.json'
    pin(bindings['pins'], frozen_path)
    frozen, context = read(frozen_path), read(source / 'EXECUTION_CONTEXT.json')
    need(frozen['context'] == context and frozen['context_sha256'] ==
         hashlib.sha256(canonical(context).encode()).hexdigest(), 'Prior frozen context binding')
    need(frozen['final_model_sha256'] == plan['comparison_source']['method_model_sha256'], 'Prior49 identity')
    for name in FILES:
        pin(context['library_pins'], old_prep / name)
    watched = [U / 'local_shape.py', U / 'FROZEN_EXPERIMENT.json', U / 'BINDINGS.json',
               Path(__file__), frozen_path, source / 'EXECUTION_CONTEXT.json']
    watched += [base / name for base in (PREP, old_prep) for name in FILES]
    before = {str(p): file_sha(p) for p in watched}
    meta, old_meta = read(PREP / 'TRAINING_META.json'), read(old_prep / 'TRAINING_META.json')
    need(meta['input_policy'] == plan['input_policy'] == POLICY, 'Declared input policy')
    need(meta['shape'] == [2346, 6, 4, 10, 49], 'Training metadata shape')
    need(meta['cache_sha256'] == before[str(PREP / 'training.npz')], 'Training cache metadata hash')
    for key in ('ids', 'origins', 'classes', 'rows', 'layout_specs', 'source_split_sha256'):
        need(meta[key] == old_meta[key], f'Training provenance changed: {key}')
    need(meta['layout_specs'] == SPECS and meta['source_split_sha256'] == plan['pins']['train_split'], 'Raw/split policy')
    need(len(meta['rows']) == len(meta['ids']) == len(set(meta['ids'])) == 2346, 'Training row identity count')
    need(len(meta['classes']) == len(set(meta['classes'])) == 93, 'Class order/count')
    eval_doc, old_eval = read(PREP / 'EVALUATION_ROWS.json'), read(old_prep / 'EVALUATION_ROWS.json')
    need(before[str(PREP / 'EVALUATION_ROWS.json')] == before[str(old_prep / 'EVALUATION_ROWS.json')]
         and eval_doc == old_eval, 'Evaluation rows/order not byte-identical')
    need(eval_doc['classes'] == meta['classes'] and len(eval_doc['rows']) == 855, 'Evaluation class/count')
    need(dict(Counter(r['group'] for r in eval_doc['rows'])) == GROUPS, 'Evaluation group coverage')
    jobs, source_hashes, source_stats = defaultdict(list), {}, {}
    datasets = {}
    for split, filename, rows, diag_name, n in (
            ('training', 'training.npz', meta['rows'], 'ROW_DIAGNOSTICS.json', 2346),
            ('evaluation', 'evaluation.npz', eval_doc['rows'], 'EVALUATION_INPUT_DIAGNOSTICS.json', 855)):
        new, old = arrays(PREP / filename), arrays(old_prep / filename)
        keys = {'x', 'window_valid', 'y'} | ({'weight', 'fit', 'internal_val', 'primary'} if split == 'training' else set())
        need(set(new) == set(old) == keys, f'{split}: cache keys')
        x, old_x = new['x'], old['x']
        need(x.dtype == old_x.dtype == np.float32 and x.shape == (n, 6, 4, 10, 49)
             and old_x.shape == (n, 6, 2, 10, 49), f'{split}: coordinate shape/dtype')
        need(np.isfinite(x).all(), f'{split}: finite C4 inputs')
        exact(x[:, :, :2], old_x, f'{split}: all global channels byte preservation')
        for key in keys - {'x'}:
            exact(new[key], old[key], f'{split}: non-x array {key}')
        need(new['window_valid'].shape == (n, 6) and new['window_valid'].dtype == np.bool_, f'{split}: window mask')
        need(new['y'].shape == (n,) and new['y'].dtype == np.int64, f'{split}: label dtype/shape')
        diagnostics, old_diags = read(PREP / diag_name), read(old_prep / diag_name)
        need(len(diagnostics) == len(old_diags) == n, f'{split}: diagnostic count')
        if split == 'training':
            fit, val, primary, weight = (new[k] for k in ('fit', 'internal_val', 'primary', 'weight'))
            need(all(a.shape == (n,) and a.dtype == np.bool_ for a in (fit, val, primary)), 'Partition shape/dtype')
            need((fit ^ val).all() and fit.sum() == 2089 and val.sum() == 257 and primary.sum() == 233
                 and not (primary & ~val).any(), 'Partition integrity')
            need(weight.shape == (n,) and weight.dtype == np.float64 and np.isfinite(weight).all()
                 and (weight >= 0).all() and (weight == 0).sum() == 19, 'Weight integrity')
            need(not (set(np.asarray(meta['origins'])[fit]) & set(np.asarray(meta['origins'])[val])), 'Origin leakage')
        for i, (row, d, od) in enumerate(zip(rows, diagnostics, old_diags)):
            label = f'{split}/{i}'
            need(row['layout'] in SPECS and '/runs/ses_' not in row['path'].replace('\\', '/').lower(), f'{label}: source scope')
            need(new['y'][i] == row['y'] and meta['classes'][row['y']] == row['label'], f'{label}: ordered label')
            if split == 'training':
                need(meta['ids'][i] == row['id'] and meta['origins'][i] == row['origin'], f'{label}: identity')
                need(fit[i] == (row['partition'] == 'fit') and val[i] == (row['partition'] == 'internal_val')
                     and primary[i] == row['primary_validation'] and weight[i] == row['weight_count'], f'{label}: partition/weight')
            else:
                need(row['role'] == 'evaluation', f'{label}: evaluation role')
            check_diag(row, d, od, x[i], new['window_valid'][i], label)
            path = str(Path(row['path']).resolve())
            need(source_hashes.setdefault(path, row['sha256']) == row['sha256'], f'{label}: inconsistent raw hash')
            jobs[path].append((split, i, row, d))
        datasets[split] = x
    local_counts, exact_counts, statuses = Counter(), Counter(), Counter()
    max_error, masked_conditions = 0.0, []
    for number, (path, conditions) in enumerate(jobs.items(), 1):
        source_stats[path] = stat(path)
        raw = Path(path).read_bytes()
        need(hashlib.sha256(raw).hexdigest() == source_hashes[path], f'Raw source hash: {path}')
        layouts = {row['layout'] for _, _, row, _ in conditions}
        need(len(layouts) == 1, f'Conflicting layouts: {path}')
        layout = next(iter(layouts)); spec = SPECS[layout]
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            xy, mask, stamps = (np.array(z[spec[k]]) for k in ('xy_key', 'mask_key', 'time_key'))
        need(xy.dtype.kind == 'f' and xy.ndim == 3 and xy.shape[1:] == (spec['vertices'], 2), f'Raw XY schema: {path}')
        need(mask.shape == xy.shape[:2] and (mask.dtype == np.bool_ or
             (mask.dtype.kind in 'iu' and np.isin(mask, (0, 1)).all())), f'Raw observation schema: {path}')
        need(stamps.dtype.kind in 'iuf' and stamps.shape == (len(xy),)
             and np.isfinite(stamps).all() and (np.diff(stamps.astype(np.float64)) > 0).all(), f'Raw timestamps: {path}')
        for split, i, row, d in conditions:
            label = f'{split}/{i}'
            lo, hi = row['start'], row['end_exclusive']
            need(type(lo) is int and type(hi) is int and 0 <= lo < hi <= len(xy), f'{label}: raw slice bounds')
            present = mask[lo:hi].astype(bool, copy=True)
            if row.get('mask_hand') is not None:
                a, b = spec['hand_slices'][row['mask_hand']]; present[:, a:b] = False
            coords = xy[lo:hi]
            need(np.isfinite(coords[present]).all() and np.isfinite(coords[present].astype(np.float32)).all(), f'{label}: observed finite XY')
            selected, observed = coords[:, JOINTS[layout]], present[:, JOINTS[layout]]
            local, zeros, hand_diag = independent_local(selected, observed)
            check_local_diag(d['local_diag'], hand_diag, hi - lo, label)
            idx = np.asarray(indices_for(hi - lo)); gathered = np.zeros((60, 49, 2), np.float32)
            zero_mask = np.ones((60, 49), bool)
            real = idx >= 0; gathered[real] = local[idx[real]]; zero_mask[real] = zeros[idx[real]]
            expected = gathered.reshape(6, 10, 49, 2).transpose(0, 3, 1, 2)
            actual = datasets[split][i, :, 2:]
            need(np.allclose(actual, expected, rtol=LOCAL_RTOL, atol=LOCAL_ATOL), f'{label}: independently reconstructed local channels')
            need((actual.transpose(0, 2, 3, 1)[zero_mask.reshape(6, 10, 49)] == 0).all(), f'{label}: missing/body/wrist/padding local zeros')
            max_error = max(max_error, float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))))
            exact_counts[split] += int(actual.tobytes(order='C') == expected.tobytes(order='C'))
            local_counts[split] += 1
            for hand, values in hand_diag.items():
                statuses[f'{split}/{hand}/{values["status"]}'] += 1
            if row.get('mask_hand') is not None:
                base = 7 if row['mask_hand'] == 'left' else 28
                need((actual[..., base:base + 21] == 0).all(), f'{label}: masked hand has local information')
                masked_conditions.append({'split': split, 'index': i, 'id': row['id'], 'hand': row['mask_hand']})
        need(stat(path) == source_stats[path], f'Raw file changed while reconstructing: {path}')
        if number % 500 == 0:
            print(json.dumps({'checked_raw_files': number, 'total_raw_files': len(jobs),
                              'local_rows_checked': dict(local_counts)}), flush=True)
    need(dict(local_counts) == {'training': 2346, 'evaluation': 855}, 'Incomplete reconstruction coverage')
    need(len(masked_conditions) == 5 and all(r['split'] == 'evaluation' for r in masked_conditions), 'Masked-condition coverage')
    for path, digest in source_hashes.items():
        need(stat(path) == source_stats[path] and file_sha(path) == digest, f'Raw source changed during full audit: {path}')
    need(all(file_sha(path) == digest for path, digest in before.items()), 'Prepared/source/component file changed during audit')
    preparation = read(PREP / 'INPUT_PREPARATION.json')
    need(preparation['plan_sha256'] == PLAN_SHA, 'Preparation plan binding')
    need(set(preparation['outputs']) == set(FILES) - {'INPUT_PREPARATION.json'}, 'Preparation output coverage')
    for name, digest in preparation['outputs'].items():
        need(name in FILES and name != 'INPUT_PREPARATION.json' and file_sha(PREP / name) == digest, 'Preparation output hash')
    return {'status': 'PASS', 'plan_sha256': PLAN_SHA, 'bindings_sha256': before[str(U / 'BINDINGS.json')],
            'input_policy': POLICY, 'artifact_sha256': {name: before[str(PREP / name)] for name in FILES},
            'training_rows': 2346, 'evaluation_rows': 855, 'global2_training_exact': 2346,
            'global2_evaluation_exact': 855, 'local_formula_reconstructed_rows': dict(local_counts),
            'local_formula_byte_exact_rows': dict(exact_counts), 'local_max_abs_error': max_error,
            'local_tolerance': {'rtol': LOCAL_RTOL, 'atol': LOCAL_ATOL}, 'local_scale_status_counts': dict(statuses),
            'independent_temporal_mask_rows': 3201, 'masked_conditions': masked_conditions,
            'nonx_arrays_byte_exact': True, 'original_rows_classes_splits_weights_preserved': True,
            'original_raw_masks_timestamps_hashes_preserved': True, 'unique_source_files_hashed': len(source_hashes),
            'raw_sources_digest': hashlib.sha256(canonical(source_hashes).encode()).hexdigest(),
            'source_files_unchanged': True, 'component_sha256': COMPONENT_SHA,
            'audited_source_artifacts': before, 'model_forward': 0, 'training': 0, 'score_arrays_read': 0,
            'limitations': ['Global XY verified byte-exact against the accepted frozen49 cache; local XY independently reconstructed for every condition.',
                           'This verifies the declared input representation, not word accuracy or correct sign form.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=U / 'INPUT_INDEPENDENT_CHECK.json')
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.parent == U and not output.exists(), 'Output must be a new file directly inside this unit')
    try:
        result, code = audit(), 0
    except Exception as exc:
        result, code = {'status': 'FAIL', 'error_type': type(exc).__name__, 'reason': str(exc),
                        'model_forward': 0, 'training': 0, 'score_arrays_read': 0}, 1
    result['checked_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['verifier_sha256'] = file_sha(Path(__file__))
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'training_rows', 'evaluation_rows', 'reason',
                      'local_max_abs_error', 'local_formula_byte_exact_rows') if k in result}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
