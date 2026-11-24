"""Label-free model input adapter. No inference, extraction or training.

Packed audit arrays retain original values, including unusable coordinates.
Models must use this adapter, which masks those values and never returns the
audit-only raw/native streams, filenames, Gloss, core intervals or targets.
"""
from pathlib import Path
import json
import numpy as np

UNIT = Path(__file__).resolve().parent
PREFIX = {'B0': 'baseline_roi', 'B1': 'redetected_roi'}


def _row(index):
    rows = json.loads((UNIT / 'MANIFEST.json').read_text(encoding='utf-8'))
    if not isinstance(index, int) or not 0 <= index < len(rows):
        raise ValueError('Expected an in-range controller ordinal, not a word label')
    return rows[index]


def from_arrays(arrays, condition='B0', dropout=None):
    """Build only permitted observations; dropout has side/start/stop indices.

    A synthetic dropout removes both the coordinates and their evidence fields.
    Uncorrupted audit arrays remain outside the returned model payload.
    """
    if condition not in PREFIX:
        raise ValueError('Only saved B0/B1 inputs exist; B2/B3 are not implemented')
    p = PREFIX[condition]
    valid = np.array(arrays[p + '_usable'], copy=True)
    observed = np.array(arrays[p + '_observed'], copy=True) & valid
    xy = np.array(arrays[p + '_xy_px'], copy=True)
    conf = np.array(arrays[p + '_conf'], copy=True)
    scores = np.array(arrays[p + '_hand_score'], copy=True)
    # Stored hand landmarks have no per-joint confidence in some sources.
    # Unknown confidence is not evidence of a bad coordinate: retain both
    # the valid coordinate and an explicit confidence-availability mask.
    conf_available = valid & np.isfinite(conf)
    score_available = np.isfinite(scores)
    if condition == 'B0':
        origin = valid.astype(np.uint8)
    else:
        origin = np.array(arrays['point_origin_roi'], copy=True)
    if dropout is not None:
        side, start, stop = (dropout[k] for k in ('side', 'start_index', 'end_index_exclusive'))
        if side not in (0, 1) or not 0 <= start < stop <= len(xy):
            raise ValueError('Invalid synthetic dropout interval')
        points = slice(33 + side * 21, 54 + side * 21)
        valid[start:stop, points] = False
        observed[start:stop, points] = False
        conf_available[start:stop, points] = False
        scores[start:stop, side] = 0
        score_available[start:stop, side] = False
    # Zero every unusable value, including coordinates outside the final ROI.
    xy[~valid] = 0
    conf[~conf_available] = 0
    origin[~valid] = 0
    for side in (0, 1):
        has_hand = valid[:, 33 + side * 21:54 + side * 21].any(axis=1)
        scores[~has_hand, side] = 0
        score_available[~has_hand, side] = False
    scores[~score_available] = 0
    result = dict(xy_px=xy, usable=valid, observed=observed, conf=conf,
                  conf_available=conf_available, hand_score=scores,
                  hand_score_available=score_available, point_origin=origin)
    for key in ('frame_id', 'timestamp_us', 'width', 'height', 'roi_box'):
        result[key] = np.array(arrays[key], copy=True)
    return result


def load_observations(index, condition='B0', dropout=None):
    row = _row(index)
    with np.load(row['arrays'], allow_pickle=False) as arrays:
        return from_arrays(arrays, condition, dropout)


def iter_rgb_frames(index):
    """Original full RGB frames and original times, no preview overlays/labels.

    Applies to full-source replay only. This is not the physical-camera-crop
    experiment, which must separately withhold unavailable outside pixels.
    """
    import cv2
    row = _row(index)
    with np.load(row['arrays'], allow_pickle=False) as arrays:
        times = arrays['timestamp_us'].copy()
    cap = cv2.VideoCapture(row['rgb_path'])
    if not cap.isOpened():
        raise OSError('Public source video unavailable')
    try:
        for i, timestamp in enumerate(times):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Unexpected end of original video')
            yield i, int(timestamp), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if cap.read()[0]:
            raise ValueError('Unexpected extra source frame')
    finally:
        cap.release()
