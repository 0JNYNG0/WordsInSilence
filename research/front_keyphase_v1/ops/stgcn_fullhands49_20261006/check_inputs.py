"""Independent 49-point cache audit; no production helpers, model or scores.

Reconstructs six preselected training conditions and all five masked evaluation
conditions from raw coordinates with a direct tensor expression. Other added
nodes receive metadata, finite-value and digest checks, not full reconstruction.
Only a new report is written; use --output to preserve an earlier failed report.
"""
from pathlib import Path
from collections import Counter
import argparse
import datetime
import hashlib
import io
import json

import numpy as np
import torch


U = Path(__file__).resolve().parent
PREP = U / 'prepared'
POLICY = 'provided_xy_isotropic_fullhands49_v1'
OLD_EVALUATION_DIAGNOSTICS_SHA = 'c00ba8a65889a603888dc859f4b1b39429a202d6192cbee7e488bab11ecc61f6'
COMMON = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 15, 16, 19, 20, 23, 24, 27,
          28, 32, 33, 36, 37, 40, 41, 44, 45, 48]
ADDED = sorted(set(range(49)) - set(COMMON))
GROUPS = {'annotation_dev': 699, 'whole_dev': 74, 'public': 32, 'mp': 14, 'ksl36': 36}
REPRESENTATIVES = [('native_short', 0), ('native_long', 910),
                   ('native_apology_base', 2113), ('native_same_origin_context', 2140),
                   ('NIKL_nonzero_slice', 2095), ('NIKL_study', 2098)]
SPECS = {
    'native67': {'xy_key': 'xy_native', 'mask_key': 'coordinate_present',
                 'time_key': 'timestamp_nominal_us', 'vertices': 67,
                 'hand_slices': {'left': [25, 46], 'right': [46, 67]}},
    'mp75': {'xy_key': 'xy_px', 'mask_key': 'observed', 'time_key': 'timestamp_us',
             'vertices': 75, 'hand_slices': {'left': [33, 54], 'right': [54, 75]}},
}
JOINTS = {'native67': [0, 16, 15, 5, 2, 6, 3] + list(range(25, 67)),
          'mp75': [0, 2, 5, 11, 12, 13, 14] + list(range(33, 75))}


def need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def tensor_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f'{array.dtype.str}|{array.shape}|'.encode('ascii'))
    digest.update(array.tobytes())
    return digest.hexdigest()


def stat_key(path):
    st = path.stat()
    return st.st_size, st.st_mtime_ns


def frame_indices(frames):
    need(type(frames) is int and frames > 0, 'Invalid source-frame count')
    return ([-1] * (60 - frames) + list(range(frames)) if frames <= 60
            else [j * (frames - 1) // 59 for j in range(60)])


def window_mask(frames):
    # A window is valid iff at least one original frame occurs in it.
    return np.array([10 * (j + 1) > max(0, 60 - frames) for j in range(6)], dtype=bool)


def check_diag(row, diag, sample, mask, prefix):
    need(row['layout'] in SPECS, f'{prefix}: unknown layout')
    need(diag['source_sha256'] == row['sha256'], f'{prefix}: source digest')
    need(diag['layout'] == row['layout'], f'{prefix}: layout')
    need(diag['slice'] == [row['start'], row['end_exclusive']], f'{prefix}: slice')
    hand = row.get('mask_hand')
    need(diag['mask_hand'] == hand, f'{prefix}: source mask policy')
    expected_hand = SPECS[row['layout']]['hand_slices'][hand] if hand is not None else None
    need(diag['hand_slice'] == expected_hand, f'{prefix}: masked hand extent')
    frames = row['end_exclusive'] - row['start']
    d = diag['prepare_diag']
    need(d['T'] == frames and d['indices'] == frame_indices(frames), f'{prefix}: temporal policy')
    need(d['shape'] == [1, 6, 2, 10, 49], f'{prefix}: prepared diagnostic shape')
    need(np.isfinite(d['shoulder_distance']) and d['shoulder_distance'] > 0, f'{prefix}: scale')
    need(np.array_equal(mask, window_mask(frames)), f'{prefix}: independent window mask')
    need(tensor_sha(sample) == diag['input_tensor_sha256'], f'{prefix}: cache tensor digest')
    need(np.isfinite(sample).all(), f'{prefix}: nonfinite coordinates')


def reconstruct(row):
    """Direct CTV arithmetic, independent of all runtime/input/graph modules."""
    path = Path(row['path'])
    before = stat_key(path)
    raw = path.read_bytes()
    need(sha(raw) == row['sha256'], f'Raw source digest: {row["id"]}')
    spec = SPECS[row['layout']]
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        xy = np.array(z[spec['xy_key']])
        present = np.array(z[spec['mask_key']])
        stamps = np.array(z[spec['time_key']])
        usable = np.array(z['usable']) if row['layout'] == 'mp75' and 'usable' in z.files else None
    need(xy.dtype.kind == 'f' and xy.ndim == 3 and xy.shape[1:] == (spec['vertices'], 2),
         f'Raw coordinate shape/dtype: {row["id"]}')
    need(present.shape == xy.shape[:2] and (present.dtype == np.bool_ or
         (present.dtype.kind in 'iu' and np.isin(present, (0, 1)).all())),
         f'Raw observation mask: {row["id"]}')
    need(stamps.dtype.kind in 'iuf' and stamps.shape == (len(xy),), f'Raw time dtype/shape: {row["id"]}')
    stamps = stamps.astype(np.float64)
    need(np.isfinite(stamps).all() and (np.diff(stamps) > 0).all(), f'Raw time order: {row["id"]}')
    start, end = row['start'], row['end_exclusive']
    need(type(start) is int and type(end) is int and 0 <= start < end <= len(xy), f'Raw bounds: {row["id"]}')
    xy = xy[start:end].astype(np.float32, copy=True)
    present = present[start:end].astype(bool, copy=True)
    observed_not_usable = (int((present & ~usable[start:end].astype(bool)).sum())
                           if usable is not None else None)
    if row.get('mask_hand') is not None:
        lo, hi = spec['hand_slices'][row['mask_hand']]
        present[:, lo:hi] = False
    need(np.isfinite(xy[present]).all(), f'Observed nonfinite raw coordinate: {row["id"]}')
    xy[~present] = 0
    selected = xy[:, JOINTS[row['layout']]]
    ctv = torch.from_numpy(np.ascontiguousarray(selected.transpose(2, 0, 1)))
    tvc = ctv.permute(1, 2, 0)
    vtc = tvc.permute(1, 0, 2)
    left, right = vtc[3].reshape(-1, 2), vtc[4].reshape(-1, 2)
    center = torch.mean((left + right) / 2, dim=0)
    scale = torch.mean(torch.sqrt(((left - right) ** 2).sum(-1)))
    need(torch.isfinite(scale) and scale > 0, f'Raw shoulder scale: {row["id"]}')
    normalized = ((tvc - center) * (1.0 / scale)).permute(2, 0, 1)
    frames = len(xy)
    if frames <= 60:
        packed = torch.cat((normalized.new_zeros((2, 60 - frames, 49)), normalized), dim=1)
    else:
        packed = normalized[:, frame_indices(frames)]
    sample = packed.reshape(2, 6, 10, 49).permute(1, 0, 2, 3).contiguous().numpy()
    distance64 = float(np.linalg.norm(selected[:, 3].astype(np.float64) - selected[:, 4].astype(np.float64), axis=1).mean())
    need(before == stat_key(path), f'Raw source changed during reconstruction: {row["id"]}')
    return sample, distance64, {'id': row['id'], 'layout': row['layout'], 'source_sha256': row['sha256'],
             'slice': [start, end], 'mask_hand': row.get('mask_hand'), 'frames': frames,
             'selected_missing_points': int((~present[:, JOINTS[row['layout']]]).sum()),
             'observed_not_usable_points': observed_not_usable, 'source_stat_unchanged': True}


def audit():
    required = ['training.npz', 'TRAINING_META.json', 'ROW_DIAGNOSTICS.json',
                'evaluation.npz', 'EVALUATION_ROWS.json', 'EVALUATION_INPUT_DIAGNOSTICS.json']
    need(all((PREP / name).is_file() for name in required), 'Prepared files incomplete; no raw sources read')
    prepared_stats = {name: stat_key(PREP / name) for name in required}
    bindings = read(U / 'BINDINGS.json')
    for name, digest in bindings['pins'].items():
        need(file_sha(Path(name)) == digest, f'Immutable binding pin mismatch: {name}')
    paths = {key: Path(value) for key, value in bindings['paths'].items()}
    source = paths['source_unit']
    reference = paths['reference_run']
    need(file_sha(reference / 'EVALUATION_INPUT_DIAGNOSTICS.json') == OLD_EVALUATION_DIAGNOSTICS_SHA,
         'Original evaluation diagnostic digest changed')
    plan = read(U / 'FROZEN_EXPERIMENT.json')
    split_path = source / 'TRAIN_SPLIT.json'
    manifest_path = source.parent / 'stgcn_transfer_compare_20261005/INPUT_MANIFEST.json'
    need(file_sha(split_path) == plan['pins']['train_split'], 'Split pin')
    need(file_sha(manifest_path) == plan['pins']['input_manifest'], 'Manifest pin')
    split, manifest = read(split_path), read(manifest_path)
    old_meta = read(source / 'prepared/TRAINING_META.json')
    meta = read(PREP / 'TRAINING_META.json')
    diags = read(PREP / 'ROW_DIAGNOSTICS.json')
    need(meta['input_policy'] == plan['input_policy'] == POLICY, 'New declared input policy')
    need(meta['shape'] == [2346, 6, 2, 10, 49], 'New metadata shape missing/incorrect')
    need(meta['cache_sha256'] == file_sha(PREP / 'training.npz'), 'Training cache hash')
    for key in ('ids', 'origins', 'classes', 'rows', 'layout_specs', 'source_split_sha256'):
        need(meta[key] == old_meta[key], f'Training metadata changed: {key}')
    need(meta['source_split_sha256'] == plan['pins']['train_split'], 'Prepared split reference')
    need(meta['layout_specs'] == SPECS, 'Observation source/mask layout changed')
    need(meta['classes'] == split['classes'] == manifest['classes'] and len(meta['classes']) == 93, 'Class order')
    need(len(diags) == len(meta['rows']) == len(meta['ids']) == len(meta['origins']) == len(split['rows']) == 2346,
         'Training row count')
    all_source_refs = {}
    representatives = []
    with np.load(PREP / 'training.npz', allow_pickle=False) as new, np.load(source / 'prepared/training.npz', allow_pickle=False) as old:
        need(set(new.files) == {'x', 'y', 'weight', 'fit', 'internal_val', 'primary', 'window_valid'}, 'Training cache keys')
        x = new['x']
        old_x = old['x']
        need(x.shape == (2346, 6, 2, 10, 49) and x.dtype == np.float32, 'Training x shape/dtype')
        need(old_x.shape == (2346, 6, 2, 10, 27) and old_x.dtype == np.float32, 'Old x shape/dtype')
        for key in ('y', 'weight', 'fit', 'internal_val', 'primary'):
            value, expected = new[key], old[key]
            need(value.dtype == expected.dtype and np.array_equal(value, expected), f'Copied training array: {key}')
        need(new['y'].dtype == np.int64 and new['weight'].dtype == np.float64, 'Label/weight dtype')
        fit, val, primary, weights, mask, labels = (new[key] for key in
             ('fit', 'internal_val', 'primary', 'weight', 'window_valid', 'y'))
        need(all(a.dtype == np.bool_ and a.shape == (2346,) for a in (fit, val, primary)), 'Partition dtype/shape')
        need(mask.dtype == np.bool_ and mask.shape == (2346, 6), 'Training window-mask dtype/shape')
        need((fit ^ val).all() and fit.sum() == 2089 and val.sum() == 257 and primary.sum() == 233, 'Partition counts')
        need((weights == 0).sum() == 19 and (weights[fit] == 0).sum() == 16 and (weights[val] == 0).sum() == 3, 'Zero weights changed')
        need(np.isfinite(weights).all() and (weights >= 0).all(), 'Invalid sample weights')
        need(not (set(np.array(meta['origins'])[fit]) & set(np.array(meta['origins'])[val])), 'Original-source leakage')
        need(not ({r['sha256'] for r in split['rows'] if r['partition'] == 'fit'} &
                  {r['sha256'] for r in split['rows'] if r['partition'] == 'internal_val'}), 'Source-hash leakage')
        expected_indices = {i for i, row in enumerate(manifest['rows']) if row['role'] == 'train'}
        need({r['manifest_index'] for r in split['rows']} == expected_indices, 'Training source membership')
        need(len(set(meta['ids'])) == 2346, 'Duplicate prepared row IDs')
        for i, (source_row, row, diag) in enumerate(zip(split['rows'], meta['rows'], diags)):
            original = manifest['rows'][source_row['manifest_index']]
            expected_id = f"{source_row['group']}:{source_row['manifest_index']}:{source_row['id']}"
            need(row == {**source_row, 'id': expected_id} and meta['ids'][i] == expected_id
                 and meta['origins'][i] == source_row['origin'], f'Train/{i}: row provenance')
            fields = ('group', 'id', 'label', 'y', 'signer_id', 'path', 'sha256', 'layout', 'start', 'end_exclusive', 'weight_count')
            need(all(source_row[k] == original[k] for k in fields), f'Train/{i}: manifest membership')
            need(original['role'] == 'train' and original['mask_hand'] is None, f'Train/{i}: training-only role')
            need('/runs/ses_' not in row['path'].replace('\\', '/').lower(), f'Train/{i}: user-session source')
            need(labels[i] == row['y'] and weights[i] == row['weight_count'], f'Train/{i}: labels/weights')
            need(fit[i] == (row['partition'] == 'fit') and val[i] == (row['partition'] == 'internal_val')
                 and primary[i] == row['primary_validation'], f'Train/{i}: split flags')
            need(diag.get('id') == expected_id, f'Train/{i}: diagnostic identity')
            check_diag(row, diag, x[i], mask[i], f'Train/{i}')
            need(np.array_equal(np.take(x[i], COMMON, axis=-1), old_x[i]), f'Train/{i}: common27 changed')
            previous = all_source_refs.setdefault(row['path'], row['sha256'])
            need(previous == row['sha256'], f'Train/{i}: inconsistent source hash')
        for kind, position in REPRESENTATIVES:
            expected, distance, detail = reconstruct(meta['rows'][position])
            need(np.array_equal(expected, x[position]), f'Train/{position}: independent full49 reconstruction')
            need(distance == diags[position]['prepare_diag']['shoulder_distance'], f'Train/{position}: independent scale')
            detail.update(kind=kind, row=position, split='train', exact_full49=True, exact_added22=True)
            representatives.append(detail)
        del x, old_x
    evaluation = read(PREP / 'EVALUATION_ROWS.json')
    old_evaluation = read(reference / 'EVALUATION_ROWS.json')
    old_diags = read(reference / 'EVALUATION_INPUT_DIAGNOSTICS.json')
    eval_diags = read(PREP / 'EVALUATION_INPUT_DIAGNOSTICS.json')
    rows = evaluation['rows']
    need(evaluation == old_evaluation, 'Evaluation metadata changed')
    need(rows == [r for r in manifest['rows'] if r['group'] in GROUPS], 'Evaluation manifest membership')
    need(len(rows) == len(old_diags) == len(eval_diags) == 855, 'Evaluation row count')
    need(dict(Counter(r['group'] for r in rows)) == GROUPS, 'Evaluation groups')
    need(evaluation['classes'] == meta['classes'], 'Evaluation class order')
    masked = []
    with np.load(PREP / 'evaluation.npz', allow_pickle=False) as archive:
        need(set(archive.files) == {'x', 'window_valid', 'y'}, 'Evaluation cache keys')
        x, mask, y = archive['x'], archive['window_valid'], archive['y']
        need(x.shape == (855, 6, 2, 10, 49) and x.dtype == np.float32, 'Evaluation x shape/dtype')
        need(mask.shape == (855, 6) and mask.dtype == np.bool_, 'Evaluation window-mask shape/dtype')
        need(y.shape == (855,) and y.dtype == np.int64, 'Evaluation label shape/dtype')
        for i, (row, diag, previous) in enumerate(zip(rows, eval_diags, old_diags)):
            need(row['role'] != 'train' and '/runs/ses_' not in row['path'].replace('\\', '/').lower(), f'Eval/{i}: forbidden source')
            need(y[i] == row['y'] and meta['classes'][row['y']] == row['label'], f'Eval/{i}: labels')
            check_diag(row, diag, x[i], mask[i], f'Eval/{i}')
            need(tensor_sha(np.take(x[i], COMMON, axis=-1)) == previous['input_tensor_sha256'], f'Eval/{i}: common27 digest changed')
            need(diag['prepare_diag']['indices'] == previous['prepare_diag']['indices']
                 and diag['prepare_diag']['shoulder_distance'] == previous['prepare_diag']['shoulder_distance'], f'Eval/{i}: geometry/time changed')
            expected_source = all_source_refs.setdefault(row['path'], row['sha256'])
            need(expected_source == row['sha256'], f'Eval/{i}: inconsistent source hash')
            if row['mask_hand'] is not None:
                expected, distance, detail = reconstruct(row)
                need(np.array_equal(expected, x[i]), f'Eval/{i}: independent masked full49 reconstruction')
                need(distance == diag['prepare_diag']['shoulder_distance'], f'Eval/{i}: independent scale')
                detail.update(kind='masked_hand', row=i, split='evaluation', exact_full49=True, exact_added22=True)
                masked.append(detail)
    need(len(masked) == 5, 'Expected all five masked conditions')
    for name, digest in all_source_refs.items():
        path = Path(name)
        before = stat_key(path)
        need(file_sha(path) == digest, f'Raw source hash changed: {name}')
        need(before == stat_key(path), f'Raw source changed while hashing: {name}')
    artifacts = {name: file_sha(PREP / name) for name in required}
    need(all(stat_key(PREP / name) == prepared_stats[name] for name in required), 'Prepared artifact changed during audit')
    need(artifacts['training.npz'] == meta['cache_sha256'], 'Training cache changed during audit')
    return {'status': 'PASS', 'input_policy': POLICY, 'training_rows': 2346, 'evaluation_rows': 855,
            'common27_training_exact': 2346, 'common27_evaluation_digest_exact': 855,
            'independent_temporal_mask_rows': 3201, 'source_membership_and_hashes': 'PASS',
            'unique_source_files_hashed': len(all_source_refs), 'added_node_count': len(ADDED),
            'training_reconstruction': representatives, 'masked_evaluation_reconstruction': masked,
            'original_split_labels_weights_ids_preserved': True, 'source_stat_unchanged': True,
            'artifact_sha256': artifacts, 'bindings_sha256': file_sha(U / 'BINDINGS.json'),
            'old_evaluation_diagnostics_sha256': OLD_EVALUATION_DIAGNOSTICS_SHA,
            'plan_sha256': file_sha(U / 'FROZEN_EXPERIMENT.json'),
            'model_forward': 0, 'training': 0, 'score_arrays_read': 0,
            'limitations': ['Added22 coordinates reconstructed independently for six fixed training rows and five masked evaluation conditions only; other rows are not a full independent raw-coordinate reconstruction.',
                           'Native original width/height and exact original pretraining input equivalence remain unverified; this checks the declared isotropic input policy.',
                           'NIKL signer identity is unknown; source disjointness is not person independence across sources.',
                           'Input preservation is not a word-accuracy or handshape-correctness result.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=U / 'INPUT_INDEPENDENT_CHECK.json')
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.parent == U.resolve(), 'Audit output must stay directly within this unit')
    if output.exists():
        raise SystemExit(f'Refusing to overwrite audit: {output}; use a new --output path for a repair check')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    try:
        result, exit_code = audit(), 0
    except Exception as exc:
        result = {'status': 'FAIL', 'error_type': type(exc).__name__, 'reason': str(exc),
                  'model_forward': 0, 'training': 0, 'score_arrays_read': 0}
        exit_code = 1
    result['checked_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['verifier_sha256'] = file_sha(Path(__file__))
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ('training_reconstruction', 'masked_evaluation_reconstruction', 'artifact_sha256')}, ensure_ascii=False))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
