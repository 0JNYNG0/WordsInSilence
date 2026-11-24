"""Common fixed lower boundary; preserve stable-rest results, body fallback.

Offline source framing only. The actual cropped detector never sees hidden
source coordinates or source tracking history. No label/core/model score is an
input. The fallback ratio is fitted once on TRAIN stable-rest successes.
"""
from __future__ import annotations
import copy
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
UNIT = Path(__file__).resolve().parent
OLD = UNIT.parent / 'word100_stable_rest_roi_20261017' / 'stable_rest.py'
spec = importlib.util.spec_from_file_location('body_boundary_preserved_rest', OLD)
stable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stable)
SCHEMA = 'stable_rest_else_train_median_shoulder_ratio_v1'


def shoulder_anchor(obs):
    """First stable shoulders-only window under unchanged temporal thresholds."""
    obs.validate()
    if 'roi_mask' in obs.sidecar:
        raise ValueError('Full-source unmasked observation required')
    if not obs.n_frames:
        return {'status': 'NOT_CALIBRATED', 'reason': 'EMPTY_OBSERVATION'}
    n = int(np.searchsorted(obs.timestamp_us, int(obs.timestamp_us[0]) + 1_000_000, side='right'))
    xy = obs.xy_px[:n, [11, 12]].astype(np.float64)
    valid = (obs.observed[:n, [11, 12]] & obs.usable[:n, [11, 12]]
             & np.isfinite(xy).all(axis=-1)).all(axis=1)
    gaps = np.diff(obs.timestamp_us[:n])
    for end in range(7, n):
        for start in range(end - 6):
            if int(obs.timestamp_us[end]) - int(obs.timestamp_us[start]) < 300_000:
                continue
            if not valid[start:end + 1].all() or (gaps[start:end] > 100_000).any():
                continue
            win = xy[start:end + 1]
            scale = float(np.median(np.linalg.norm(win[:, 0] - win[:, 1], axis=1)))
            if not math.isfinite(scale) or scale <= 0:
                continue
            dev = np.linalg.norm(win - np.median(win, axis=0), axis=-1)
            if float(dev.max()) > .05 * scale:
                continue
            center = np.median((win[:, 0] + win[:, 1]) * .5, axis=0)
            return {'status': 'CALIBRATED', 'reason': 'STABLE_SHOULDERS_ONLY',
                    'selected_frame_indices': list(range(start, end + 1)),
                    'selected_frame_ids': obs.frame_id[start:end + 1].tolist(),
                    'selected_timestamp_us': obs.timestamp_us[start:end + 1].tolist(),
                    'shoulder_width': scale, 'median_shoulder_center': center.tolist(),
                    'maximum_stability_distance_pixels': float(dev.max())}
    return {'status': 'NOT_CALIBRATED', 'reason': 'NO_STABLE_INITIAL_SHOULDERS_WINDOW'}


def calibrate(obs, width, height, coefficient):
    """Do not silently clamp or invent a boundary if geometry is unusable."""
    original = stable.calibrate_rest(obs, width, height)
    if original['status'] == 'CALIBRATED':
        result = copy.deepcopy(original)
        result.update(schema=SCHEMA, method='preserved_stable_rest', original_schema=original['schema'])
        return result
    if coefficient.get('status') != 'FROZEN_TRAIN_MEDIAN' or coefficient.get('schema') != SCHEMA:
        raise ValueError('Frozen TRAIN-only common coefficient required')
    k = float(coefficient['ratio'])
    if not math.isfinite(k) or k <= 0:
        raise ValueError('Invalid common ratio')
    anchor = shoulder_anchor(obs)
    result = {**anchor, 'schema': SCHEMA, 'method': 'train_median_body_fallback',
              'old_failure_reason': original['reason'], 'source_width': int(width),
              'source_height': int(height), 'ratio': k,
              'input_scope': 'offline_full_source_first_second_shoulders_only',
              'time_parameters': {'horizon_us': 1_000_000, 'min_duration_us': 300_000,
                                  'min_frames': 8, 'max_gap_us': 100_000,
                                  'max_shoulder_motion_scale': .05},
              'label_or_hand_motion_in_boundary': False}
    if anchor['status'] != 'CALIBRATED':
        return result
    bottom = math.floor(anchor['median_shoulder_center'][1] + k * anchor['shoulder_width'])
    if not 0 < bottom < height:
        result.update(status='NOT_CALIBRATED', reason='BODY_BOUNDARY_OUTSIDE_SOURCE', proposed_bottom=bottom)
        return result
    result.update(reason='COMMON_TRAIN_MEDIAN_BODY_RATIO', bottom=bottom,
                  box_source_pixels=[0, 0, int(width), int(bottom)])
    return result
