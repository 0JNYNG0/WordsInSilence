"""Full-span temporal adapter: normalize every observed frame, then sample 60 frame slots."""
import math
import numbers

import numpy as np
import torch

from reference_adapter import LAYOUTS

NUM_FRAMES, NUM_WINDOWS, WINDOW, NUM_JOINTS = 60, 6, 10, 27
POLICY = 'full_span_uniform_frames_v1'
PRESET = 'shoulder_mediapipe_holistic_minimal_27'


def _dimension(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f'{name} must be a real number supplied by the caller')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def sample_indices(num_frames):
    """Source frame per output slot; -1 marks leading zero padding."""
    if isinstance(num_frames, (bool, np.bool_)) or not isinstance(num_frames, numbers.Integral):
        raise TypeError('num_frames must be an integer')
    num_frames = int(num_frames)
    if num_frames < 1:
        raise ValueError('num_frames must be >= 1')
    if num_frames <= NUM_FRAMES:
        return [-1] * (NUM_FRAMES - num_frames) + list(range(num_frames))
    steps = torch.arange(NUM_FRAMES, dtype=torch.long) * (num_frames - 1)
    return (steps // (NUM_FRAMES - 1)).tolist()


def prepare_full_span_windows(xy_pixels, present, layout, width, height, transforms_module):
    if layout not in LAYOUTS:
        raise ValueError(f'unknown layout {layout!r}')
    num_vertices, joint_indices = LAYOUTS[layout]
    joint_indices = [int(i) for i in joint_indices]
    if len(joint_indices) != NUM_JOINTS:
        raise ValueError(f'layout {layout!r} must select {NUM_JOINTS} joints')
    width, height = _dimension('width', width), _dimension('height', height)

    xy = np.array(xy_pixels, copy=True)
    mask = np.array(present, copy=True)
    if xy.dtype.kind not in 'iuf':
        raise TypeError('xy_pixels must be a real numeric array')
    if mask.dtype != np.bool_:
        raise TypeError('present must be a strict bool array')
    if xy.ndim != 3 or xy.shape[0] < 1 or xy.shape[1:] != (num_vertices, 2):
        raise ValueError(f'xy_pixels must have shape (T>=1, {num_vertices}, 2), got {xy.shape}')
    if mask.shape != xy.shape[:2]:
        raise ValueError(f'present must have shape {xy.shape[:2]}, got {mask.shape}')
    xy = xy.astype(np.float32)
    if not np.isfinite(xy[mask]).all():
        raise ValueError('present joints must have finite coordinates')
    xy[~mask] = 0.0  # missing -> 0 before normalization only
    xy /= np.array([width, height], dtype=np.float32)
    num_frames = xy.shape[0]
    frames = torch.from_numpy(np.ascontiguousarray(xy[:, joint_indices].transpose(2, 0, 1)))

    shoulder = torch.linalg.vector_norm(frames[:, :, 3] - frames[:, :, 4], dim=0).mean().item()
    if not math.isfinite(shoulder) or shoulder <= 0:
        raise ValueError(f'ineligible: mean shoulder distance {shoulder} is not finite and positive')

    normalizer = transforms_module.CenterAndScaleNormalize(reference_points_preset=PRESET, frame_level=False)
    result = normalizer({'frames': frames})
    norm = result.get('frames') if isinstance(result, dict) else None
    if (not isinstance(norm, torch.Tensor) or norm.dtype != torch.float32
            or tuple(norm.shape) != (2, num_frames, NUM_JOINTS)):
        raise ValueError('normalizer must return float32 frames of shape (2, T, 27)')
    if not torch.isfinite(norm).all():
        raise ValueError('normalized frames contain non-finite values')

    indices = sample_indices(num_frames)
    padded = max(0, NUM_FRAMES - num_frames)
    observed = norm[:, torch.tensor(indices[padded:], dtype=torch.long)]
    seq = torch.cat([norm.new_zeros(2, padded, NUM_JOINTS), observed], dim=1)
    out = seq.reshape(2, NUM_WINDOWS, WINDOW, NUM_JOINTS).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    diagnostics = {
        'original_frames': num_frames,
        'selected_source_indices': indices,
        'padded_frames': padded,
        'dropped_frames': max(0, num_frames - NUM_FRAMES),
        'policy': POLICY,
        'width': width,
        'height': height,
        'input_shape': tuple(xy.shape),
        'output_shape': tuple(out.shape),
    }
    return out, diagnostics
