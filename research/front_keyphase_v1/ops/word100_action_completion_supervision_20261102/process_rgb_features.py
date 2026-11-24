"""Full-frame RGB bridge for reviewed process inputs, including short views.

The frozen 16-frame encoder receives edge padding only internally. A padded
slot is never a new observation, timestamp, label, or proof of motion/holding.
No models are compiled or downloaded by this module.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

WINDOW = 16
DIM = 768


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _pins(shared):
    paths = [Path(__file__).resolve(), Path(shared.__file__).resolve(),
             Path(shared.base.__file__).resolve(),
             Path(shared.base.SIGNREP) / 'common.py']
    return {str(p): _sha(p) for p in paths}


def _axis(ids, ts, width, height):
    ids, ts = np.asarray(ids), np.asarray(ts)
    for name, values in [('frame_id', ids), ('timestamp_us', ts)]:
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f'{name} must be a one-dimensional integer array')
        if len(values) and ((values < 0).any() or (np.diff(values) <= 0).any()):
            raise ValueError(f'{name} must be nonnegative and strictly increasing')
    if len(ids) != len(ts):
        raise ValueError('frame/time length mismatch')
    if int(width) != width or int(height) != height or width <= 0 or height <= 0:
        raise ValueError('Positive integer full-frame width and height required')
    return ids, ts, int(width), int(height)


def _validate_features(rgb, prelinear, n):
    for name, value in [('rgb', rgb), ('prelinear', prelinear)]:
        if np.asarray(value).shape != (n, DIM) or not np.isfinite(value).all():
            raise ValueError(f'{name} must be finite [{n},{DIM}]')


def extract_rgb(shared, source, ids, ts, width, height, compiled, progress=None):
    """Return (rgb[N,768], prelinear[N,768], elapsed_seconds, metadata).

    shared is the existing learned_guide_feedback live_adapter module.
    ids/ts address the *actual supplied video*, not the unedited source. The
    caller keeps original-to-edited mappings and image-availability masks.
    For N<2 no encoder is called: NaN arrays and supported=False mean unknown.
    For 2<=N<16 one padded-window estimate is broadcast to the N real times;
    those N rows are not independent frame-level motion estimates.
    """
    started = time.perf_counter()
    ids, ts, width, height = _axis(ids, ts, width, height)
    source = Path(source).resolve()
    source_hash = _sha(source)
    n = len(ids)
    meta = {
        'schema': 'process_rgb_short_input_v1',
        'source_path': str(source), 'source_sha256': source_hash,
        'code_pins': _pins(shared),
        'real_frames': n, 'real_frame_id': ids.tolist(),
        'real_timestamp_us': ts.tolist(), 'returned_rows': n,
        'actual_sensor_extent': [0, 0, width, height],
        'resize': [224, 224], 'resize_interpolation': 'cv2.INTER_LINEAR',
        'normalization': 'unchanged SignRep common.norm RGB',
        'output_contract': {'0': 'rgb768', '1': 'prelinear768'},
        'source_mask_meaning': 'True=real encoder slot; False=internal repeated final-frame slot. This is NOT hand visibility or image-availability evidence.',
        'padding_is_observation': False, 'padding_is_motion_label': False,
        'labels_modified': False, 'timestamps_extended': False,
        'frame_ids_extended': False,
        'caller_image_available_mask_unchanged': True,
        'short_motion_accuracy_validated': False,
    }
    if n < 2:
        meta.update(supported=False, status='UNSUPPORTED_TOO_FEW_REAL_FRAMES_UNKNOWN',
                    model_window_indices=[], source_mask=[], padding_count=0,
                    encoder_calls=0, exact_decode_pts_verified=False,
                    feature_time_mapping='none; unknown NaN features')
        unknown = np.full((n, DIM), np.nan, np.float32)
        return unknown, unknown.copy(), time.perf_counter() - started, meta

    if n > shared.base.MAX_DECODED_FRAMES or int(ts[-1]) > shared.base.MAX_ELAPSED_US:
        raise ValueError('Input exceeds existing bounded video limits')
    common = shared.base._module('_live_signrep_common', Path(shared.base.SIGNREP) / 'common.py')
    if n >= WINDOW:
        # Preserve the original encoder/windowing/interpolation implementation.
        rgb, prelinear, delegated_seconds = shared.extract_rgb(
            source, ids, ts, width, height, compiled, progress)
        _validate_features(rgb, prelinear, n)
        starts = list(map(int, common.starts(n)))
        meta.update(supported=True, status='UNCHANGED_SHARED_LONG_INPUT',
                    model_window_indices=[list(range(s, s + WINDOW)) for s in starts],
                    source_mask=[[True] * WINDOW for _ in starts], padding_count=0,
                    encoder_calls=len(starts), exact_decode_pts_verified=True,
                    delegated_extract_seconds=float(delegated_seconds),
                    window_centers_us=[float((int(ts[s]) + int(ts[s + WINDOW - 1])) / 2) for s in starts],
                    feature_time_mapping='unchanged shared window-center interpolation at real timestamps')
    else:
        import av
        import cv2

        frames = []
        got = []
        wanted = {int(fid): i for i, fid in enumerate(ids)}
        origin = None
        last = -1
        kept = dropped = decoded = 0
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            if (int(stream.codec_context.width), int(stream.codec_context.height)) != (width, height):
                raise ValueError('Full-frame dimensions differ from declared source')
            if stream.time_base is None:
                raise ValueError('Exact decoder PTS needs video time_base')
            for packet in container.demux(stream):
                for frame in packet.decode():
                    decoded += 1
                    if decoded > shared.base.MAX_DECODED_FRAMES:
                        raise ValueError('Decoded frame limit exceeded')
                    if frame.pts is None:
                        continue
                    if origin is None:
                        origin = frame.pts
                    stamp = int(round(float((frame.pts - origin) * stream.time_base) * 1e6))
                    if stamp > shared.base.MAX_ELAPSED_US:
                        raise ValueError('Decoded elapsed-time limit exceeded')
                    if stamp <= last:
                        dropped += 1
                        continue
                    last = stamp
                    fid = kept + dropped
                    kept += 1
                    if fid not in wanted:
                        raise ValueError('RGB/body source-frame mismatch')
                    i = wanted[fid]
                    if i != len(got) or stamp != int(ts[i]):
                        raise ValueError('RGB/body exact PTS mismatch')
                    got.append(fid)
                    image = frame.to_ndarray(format='rgb24')
                    if image.shape != (height, width, 3):
                        raise ValueError('Decoded full-frame dimensions changed')
                    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
                    frames.append(common.norm(image[None])[0])
        if got != list(map(int, ids)) or len(frames) != n:
            raise ValueError('RGB output source coverage changed')
        # Every real frame once, then only the final real frame as an internal
        # encoder-size extension. No retiming/resampling or model interpolation.
        indices = list(range(n)) + [n - 1] * (WINDOW - n)
        tensor = np.ascontiguousarray(np.stack([frames[i] for i in indices]).transpose(1, 0, 2, 3)[None])
        if tensor.shape != (1, 3, WINDOW, 224, 224):
            raise ValueError('Invalid 16-frame encoder input')
        output = compiled([tensor])
        one_rgb = np.array(output[0], dtype=np.float32, copy=True).reshape(1, DIM)
        one_pre = np.array(output[1], dtype=np.float32, copy=True).reshape(1, DIM)
        rgb = np.repeat(one_rgb, n, axis=0)
        prelinear = np.repeat(one_pre, n, axis=0)
        _validate_features(rgb, prelinear, n)
        meta.update(supported=True, status='SHORT_INTERNAL_FINAL_FRAME_EDGE_PADDING',
                    model_window_indices=[indices],
                    source_mask=[[True] * n + [False] * (WINDOW - n)],
                    padding_count=WINDOW - n, encoder_calls=1,
                    exact_decode_pts_verified=True, decoded_frames=decoded,
                    discarded_nonmonotone_pts=dropped,
                    feature_time_mapping='one padded-window estimate broadcast only to N actual frame times; not N independent temporal estimates',
                    artificial_context='repeated last actual decoded frame; not evidence that the signer held or completed any action',
                    potential_bias='Padding changes encoder context; actual short-motion quality and speed robustness need separate validation.')
        if progress:
            progress({'stage': 'rgb_features', 'windows': 1, 'total_windows': 1,
                      'real_frames': n, 'internal_padding_count': WINDOW - n})

    if _sha(source) != source_hash:
        raise ValueError('Source changed during RGB extraction')
    return rgb, prelinear, time.perf_counter() - started, meta
