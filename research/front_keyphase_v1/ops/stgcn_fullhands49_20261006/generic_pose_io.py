"""Generic pose-window preparation and verified NPZ row loading.

Joint choices, normalization, augmentation and NPZ layouts are supplied by the
caller; nothing here knows about a particular dataset, skeleton or label set.
"""
import hashlib
import io
import math

import numpy as np
import torch

_HEX = frozenset('0123456789abcdef')


def _int(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer')
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return value


def _indices(values, name, size):
    if isinstance(values, (str, bytes)) or not hasattr(values, '__iter__'):
        raise TypeError(f'{name} must be a sequence of integers')
    out = [_int(v, name) for v in values]
    if not out or len(set(out)) != len(out) or not all(0 <= v < size for v in out):
        raise ValueError(f'{name} must be unique indices in [0, {size})')
    return out


def _numpy(value, name):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    raise TypeError(f'{name} must be a numpy array or torch tensor')


def _finite_float32(tensor, shape, name):
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
        raise TypeError(f'{name} must be a float32 tensor')
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f'{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}')
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f'{name} must be finite')
    return tensor


def tensor_sha256(tensor):
    """SHA-256 over dtype, shape and contiguous bytes of a CPU copy."""
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    digest = hashlib.sha256(f'{array.dtype.str}|{array.shape}|'.encode('ascii'))
    digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_windows(xy, present, joint_indices, shoulder_indices, target_frames,
                    window_frames, normalize_fn):
    """Turn one variable-length 2-D keypoint clip into fixed-size windows.

    xy: real numeric [T, V, 2]; entries where present is False may be NaN.
    present: bool [T, V].
    joint_indices: unique vertex indices to keep, in output order.
    shoulder_indices: two positions into joint_indices used for the scale check
        (mean distance over all original frames, with missing joints set to 0,
        must be finite and > 0).
    normalize_fn: receives a float32 [2, T, Vselected] clone with missing joints
        set to 0 and must return a finite float32 tensor of the same shape.

    Clips with T <= target_frames are front-padded with zero frames; longer clips
    keep frames floor(j * (T - 1) / (target_frames - 1)).
    Returns (float32 [1, target_frames // window_frames, 2, window_frames,
    Vselected], diag).
    """
    xy = _numpy(xy, 'xy')
    present = _numpy(present, 'present')
    if xy.dtype.kind not in 'iuf' or xy.ndim != 3 or xy.shape[2] != 2 or min(xy.shape[:2]) < 1:
        raise ValueError('xy must be real numeric [T, V, 2] with T, V >= 1')
    if present.dtype != np.bool_ or present.shape != xy.shape[:2]:
        raise ValueError('present must be bool [T, V]')
    n_frames, n_vertices = xy.shape[:2]
    joints = _indices(joint_indices, 'joint_indices', n_vertices)
    shoulders = _indices(shoulder_indices, 'shoulder_indices', len(joints))
    if len(shoulders) != 2:
        raise ValueError('shoulder_indices must contain exactly two indices')
    target_frames = _int(target_frames, 'target_frames', 2)
    window_frames = _int(window_frames, 'window_frames', 1)
    if target_frames % window_frames:
        raise ValueError('target_frames must be divisible by window_frames')
    if not callable(normalize_fn):
        raise TypeError('normalize_fn must be callable')

    coords = xy.astype(np.float32, copy=True)
    mask = present.copy()
    if not np.isfinite(coords[mask]).all():
        raise ValueError('observed coordinates must be finite in float32')
    coords[~mask] = 0.0
    coords = coords[:, joints, :]

    pair = coords[:, shoulders, :].astype(np.float64)
    distance = float(np.linalg.norm(pair[:, 0] - pair[:, 1], axis=1).mean())
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('mean shoulder distance must be finite and positive')

    clip = torch.from_numpy(np.ascontiguousarray(coords.transpose(2, 0, 1)))
    normalized = _finite_float32(normalize_fn(clip.clone()), clip.shape, 'normalize_fn output')
    normalized = normalized.detach().cpu()
    if n_frames <= target_frames:
        pad = target_frames - n_frames
        frames = torch.cat([normalized.new_zeros((2, pad, len(joints))), normalized], dim=1)
        indices = [-1] * pad + list(range(n_frames))
    else:
        indices = [j * (n_frames - 1) // (target_frames - 1) for j in range(target_frames)]
        frames = normalized[:, indices, :]
    n_windows = target_frames // window_frames
    out = frames.reshape(2, n_windows, window_frames, len(joints)).permute(1, 0, 2, 3)
    out = out.unsqueeze(0).contiguous()
    diag = {'T': n_frames, 'indices': indices, 'shoulder_distance': distance,
            'shape': list(out.shape)}
    return out, diag


def augment_windows(sample, transform_fn):
    """Apply a caller-owned transform once to a whole windowed clip.

    sample: floating [W, 2, T, V]; never mutated. transform_fn receives a fresh
    [2, W * T, V] tensor, so per-call random parameters are shared across all
    windows and frames, and must return a finite tensor of the same shape/dtype.
    """
    if not isinstance(sample, torch.Tensor) or sample.ndim != 4 or sample.shape[1] != 2 \
            or not sample.is_floating_point():
        raise ValueError('sample must be a floating tensor [W, 2, T, V]')
    if not callable(transform_fn):
        raise TypeError('transform_fn must be callable')
    n_windows, _, n_frames, n_vertices = sample.shape
    clip = sample.detach().clone().permute(1, 0, 2, 3).reshape(2, n_windows * n_frames, n_vertices)
    out = transform_fn(clip)
    if not isinstance(out, torch.Tensor) or out.shape != clip.shape or out.dtype != sample.dtype:
        raise ValueError('transform_fn must return a tensor with the input shape and dtype')
    if not bool(torch.isfinite(out).all()):
        raise ValueError('transform_fn output must be finite')
    return out.reshape(2, n_windows, n_frames, n_vertices).permute(1, 0, 2, 3).contiguous()


def _check_spec(spec, layout):
    if not isinstance(spec, dict):
        raise TypeError(f'layout {layout!r} spec must be a dict')
    for key in ('xy_key', 'mask_key', 'time_key'):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise ValueError(f'layout {layout!r} needs a non-empty string {key}')
    vertices = _int(spec.get('vertices'), f'{layout}.vertices', 1)
    slices = spec.get('hand_slices')
    if not isinstance(slices, dict):
        raise TypeError(f'layout {layout!r} hand_slices must be a dict')
    for name, bounds in slices.items():
        if not isinstance(name, str) or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f'layout {layout!r} hand_slices entries must be name -> [lo, hi]')
        lo, hi = (_int(b, f'{layout}.hand_slices.{name}') for b in bounds)
        if not 0 <= lo < hi <= vertices:
            raise ValueError(f'layout {layout!r} hand slice {name!r} out of range')
    return vertices


def load_row(row, layout_specs, prepare_fn):
    """Load one hash-verified NPZ clip slice and prepare it with a caller function.

    row: dict with path, sha256 (lowercase hex of the exact file bytes), layout,
        start, end_exclusive and optional mask_hand (None or a hand_slices name).
    layout_specs[layout]: dict with xy_key, mask_key, time_key, vertices and
        hand_slices {name: [lo, hi]}.
    prepare_fn(xy float [T, V, 2], mask bool [T, V], layout) must return
        (float32 [1, W, 2, T', V'], diag), e.g. a partial of prepare_windows.
    Returns (float32 [W, 2, T', V'], diag).
    """
    if not isinstance(row, dict):
        raise TypeError('row must be a dict')
    path, expected, layout = row.get('path'), row.get('sha256'), row.get('layout')
    if not isinstance(path, str) or not path:
        raise ValueError('row path must be a non-empty string')
    if not isinstance(expected, str) or len(expected) != 64 or not set(expected) <= _HEX:
        raise ValueError('row sha256 must be 64 lowercase hex characters')
    if not isinstance(layout_specs, dict) or not isinstance(layout, str) or layout not in layout_specs:
        raise ValueError(f'unknown layout {layout!r}')
    spec = layout_specs[layout]
    vertices = _check_spec(spec, layout)
    start = _int(row.get('start'), 'start', 0)
    end = _int(row.get('end_exclusive'), 'end_exclusive', 1)
    mask_hand = row.get('mask_hand')
    if mask_hand is not None and (not isinstance(mask_hand, str) or mask_hand not in spec['hand_slices']):
        raise ValueError(f'unknown mask_hand {mask_hand!r} for layout {layout!r}')
    if not callable(prepare_fn):
        raise TypeError('prepare_fn must be callable')

    with open(path, 'rb') as handle:
        raw = handle.read()
    source_sha = hashlib.sha256(raw).hexdigest()
    if source_sha != expected:
        raise ValueError(f'sha256 mismatch for {path}')
    keys = (spec['xy_key'], spec['mask_key'], spec['time_key'])
    loaded = np.load(io.BytesIO(raw), allow_pickle=False)
    if not hasattr(loaded, 'files'):
        raise ValueError(f'{path} is not an NPZ archive')
    with loaded as npz:
        missing = [key for key in keys if key not in npz.files]
        if missing:
            raise KeyError(f'{path} lacks arrays {missing}')
        xy, mask, times = (np.array(npz[key]) for key in keys)

    if xy.dtype.kind != 'f' or xy.ndim != 3 or xy.shape[1:] != (vertices, 2):
        raise ValueError(f'{keys[0]} must be float [T, {vertices}, 2]')
    n_frames = xy.shape[0]
    if mask.shape != (n_frames, vertices):
        raise ValueError(f'{keys[1]} must have shape [T, {vertices}]')
    if mask.dtype == np.bool_:
        mask = mask.copy()
    elif mask.dtype.kind in 'iu' and np.isin(mask, (0, 1)).all():
        mask = mask.astype(bool)
    else:
        raise ValueError(f'{keys[1]} must be bool or 0/1 integers')
    stamps = times.astype(np.float64) if times.dtype.kind in 'iuf' else None
    if stamps is None or stamps.shape != (n_frames,) or not np.isfinite(stamps).all() \
            or not (np.diff(stamps) > 0).all():
        raise ValueError(f'{keys[2]} must be a finite, strictly increasing 1-D array of length T')
    if not start < end <= n_frames:
        raise ValueError(f'slice [{start}, {end}) out of bounds for T={n_frames}')

    xy = xy[start:end].copy()
    mask = mask[start:end].copy()
    hand_slice = None
    if mask_hand is not None:
        lo, hi = (int(b) for b in spec['hand_slices'][mask_hand])
        mask[:, lo:hi] = False
        hand_slice = [lo, hi]
    result = prepare_fn(xy, mask, layout)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError('prepare_fn must return (tensor, diag)')
    tensor, prepare_diag = result
    _finite_float32(tensor, None, 'prepare_fn tensor')
    if tensor.ndim != 5 or tensor.shape[0] != 1 or tensor.shape[2] != 2:
        raise ValueError('prepare_fn tensor must have shape [1, W, 2, T, V]')
    sample = tensor[0].detach().cpu().clone().contiguous()
    diag = {'source_sha256': source_sha, 'input_tensor_sha256': tensor_sha256(sample),
            'layout': layout, 'slice': [start, end], 'mask_hand': mask_hand,
            'hand_slice': hand_slice, 'prepare_diag': prepare_diag}
    return sample, diag


def prepare_dataset(rows, layout_specs, prepare_fn):
    """Load exactly the given rows into the engine's data dict.

    Each row adds id (unique non-empty string), origin (non-empty string),
    y (integer >= 0) and weight_count (finite, >= 0; zero is allowed) to the
    load_row fields. Returns ({x, y, weight, ids, origins}, row_diagnostics).
    Validation-only fields such as primary are left for the caller to add.
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('rows must be a non-empty list')
    samples, labels, weights, ids, origins, diagnostics = [], [], [], [], [], []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f'row {position} must be a dict')
        row_id, origin = row.get('id'), row.get('origin')
        if not isinstance(row_id, str) or not row_id or row_id in ids:
            raise ValueError(f'row {position}: id must be a unique non-empty string')
        if not isinstance(origin, str) or not origin:
            raise ValueError(f'row {row_id!r}: origin must be a non-empty string')
        label = _int(row.get('y'), f'row {row_id!r} y', 0)
        weight = row.get('weight_count')
        if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, (int, float, np.integer, np.floating)):
            raise TypeError(f'row {row_id!r}: weight_count must be a number')
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f'row {row_id!r}: weight_count must be finite and >= 0')
        sample, diag = load_row(row, layout_specs, prepare_fn)
        if samples and sample.shape != samples[0].shape:
            raise ValueError(f'row {row_id!r}: sample shape {tuple(sample.shape)} differs from '
                             f'{tuple(samples[0].shape)}')
        samples.append(sample)
        labels.append(label)
        weights.append(weight)
        ids.append(row_id)
        origins.append(origin)
        diagnostics.append({'id': row_id, **diag})
    dataset = {'x': torch.stack(samples), 'y': torch.tensor(labels, dtype=torch.int64),
               'weight': torch.tensor(weights, dtype=torch.float64), 'ids': ids, 'origins': origins}
    return dataset, diagnostics
