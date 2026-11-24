"""Offline RGB/pose pairing. No model, extraction, network or camera access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

SIZE = 128
MULTIPLIER = 1.5
HANDS = (slice(33, 54), slice(54, 75))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(jsonable(data), f, ensure_ascii=False, indent=2, allow_nan=False)


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    return {'n': int(a.size), 'min': float(a.min()) if a.size else None,
            'median': float(np.median(a)) if a.size else None,
            'max': float(a.max()) if a.size else None}


def make_crop(rgb, xy, usable, shoulders):
    """Crop without changing inputs; pixel_valid describes geometric padding."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('RGB must be uint8 H,W,3')
    xy = np.asarray(xy, dtype=np.float64)
    usable = np.asarray(usable)
    if xy.shape != (21, 2) or usable.shape != (21,) or usable.dtype != bool:
        raise ValueError('Expected21 xy points and boolean usable mask')
    height, width = rgb.shape[:2]
    out = {'rgb': np.zeros((SIZE, SIZE, 3), np.uint8),
           'pixel_valid': np.zeros((SIZE, SIZE), bool), 'crop_valid': False,
           'crop_reason': 'missing', 'bbox_xyxy': np.full(4, -1, np.int32),
           'padding_fraction': float('nan'), 'shoulder_relative_size': float('nan'),
           'frame_relative_size': float('nan')}
    if not usable.any():
        return out
    if not usable.all():
        out['crop_reason'] = 'partial'
        return out
    if not np.isfinite(xy).all():
        out['crop_reason'] = 'nonfinite'
        return out
    if ((xy[:, 0] < 0) | (xy[:, 0] >= width) |
            (xy[:, 1] < 0) | (xy[:, 1] >= height)).any():
        out['crop_reason'] = 'outside'
        return out
    lo, hi = xy.min(0), xy.max(0)
    extent = float((hi - lo).max())
    if extent <= 0:
        out['crop_reason'] = 'degenerate'
        return out
    side = max(2, math.ceil(MULTIPLIER * extent))
    if side > 3 * max(width, height):
        raise ValueError('Unsafe crop allocation')
    center = (lo + hi) / 2
    x0, y0 = map(int, np.floor(center - side / 2))
    x1, y1 = x0 + side, y0 + side
    ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    canvas = np.zeros((side, side, 3), np.uint8)
    mask = np.zeros((side, side), np.uint8)
    canvas[iy0-y0:iy1-y0, ix0-x0:ix1-x0] = rgb[iy0:iy1, ix0:ix1]
    mask[iy0-y0:iy1-y0, ix0-x0:ix1-x0] = 255
    out.update(rgb=np.asarray(Image.fromarray(canvas).resize((SIZE, SIZE), Image.Resampling.BILINEAR)),
               pixel_valid=np.asarray(Image.fromarray(mask).resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 0,
               crop_valid=True, crop_reason='valid', bbox_xyxy=np.array([x0, y0, x1, y1], np.int32),
               padding_fraction=1 - (ix1-ix0)*(iy1-iy0)/(side*side),
               frame_relative_size=side/width)
    shoulders = np.asarray(shoulders, dtype=np.float64)
    if shoulders.shape == (2, 2) and np.isfinite(shoulders).all():
        shoulder_span = float(np.linalg.norm(shoulders[1] - shoulders[0]))
        if shoulder_span > 0:
            out['shoulder_relative_size'] = side / shoulder_span
    return out


def allocate(n):
    return dict(rgb=np.zeros((n, 2, SIZE, SIZE, 3), np.uint8),
                pixel_valid=np.zeros((n, 2, SIZE, SIZE), bool),
                crop_valid=np.zeros((n, 2), bool), crop_reason=np.full((n, 2), '', dtype='U16'),
                bbox_xyxy=np.full((n, 2, 4), -1, np.int32),
                padding_fraction=np.full((n, 2), np.nan, np.float64),
                shoulder_relative_size=np.full((n, 2), np.nan, np.float64),
                frame_relative_size=np.full((n, 2), np.nan, np.float64),
                source_width_height=np.zeros((n, 2), np.int32))


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sheet(path, sample_id, panels):
    # Three times per sample: unmodified raw PNGs are saved separately.
    canvas = Image.new('RGB', (600 * len(panels), 640), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, (fid, ts, rgb, crops) in enumerate(panels):
        thumb = Image.fromarray(rgb)
        thumb.thumbnail((590, 332))
        sx, sy = thumb.width/rgb.shape[1], thumb.height/rgb.shape[0]
        td = ImageDraw.Draw(thumb)
        for side, cr in enumerate(crops):
            if cr['crop_valid']:
                x0, y0, x1, y1 = cr['bbox_xyxy']
                td.rectangle((x0*sx, y0*sy, x1*sx, y1*sy),
                             outline=('red' if side == 0 else 'green'), width=3)
        x = col * 600
        draw.text((x+5, 4), sample_id, fill='black')
        draw.text((x+5, 21), f'frame={fid} time_us={ts}', fill='black')
        canvas.paste(thumb, (x+5, 44))
        for side, cr in enumerate(crops):
            xx = x+12+side*288
            draw.text((xx, 388), f'{("LEFT", "RIGHT")[side]} {cr["crop_reason"]}', fill='black')
            canvas.paste(Image.fromarray(cr['rgb']).resize((224, 224)), (xx, 411))
    canvas.save(path)


def run(config, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        return _run(config, out)
    except Exception as exc:
        write_json(out/'ERROR.json', {'type': type(exc).__name__, 'message': str(exc),
                                     'first_failure_preserved': True})
        raise


def _run(config, out):
    write_json(out/'CONFIG.json', config)
    if config.get('crop_size', SIZE) != SIZE or config.get('bbox_multiplier', MULTIPLIER) != MULTIPLIER:
        raise ValueError('Frozen crop contract differs')
    rows = config['rows']
    ids = [r['sample_id'] for r in rows]
    if not rows or len(ids) != len(set(ids)) or not all(re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in ids):
        raise ValueError('Invalid or duplicate sample IDs')
    pins = {}
    groups = defaultdict(list)
    for r in rows:
        for key in ('raw', 'observation', 'input'):
            path = Path(r[f'{key}_path'])
            if not path.is_file() or (key == 'raw' and path.suffix.lower() != '.mp4'):
                raise ValueError(f'Existing regular file required: {path}')
            expected = r[f'{key}_sha256']
            if str(path) not in pins:
                pins[str(path)] = sha(path)
            if pins[str(path)] != expected:
                raise ValueError(f'Hash mismatch: {path}')
        groups[(r['raw_path'], r['observation_path'])].append(r)
    write_json(out/'SOURCE_PINS.json', pins)
    visual_ids = set(sorted(ids)[:6])
    reports = []
    total_decode = 0
    for (video_path, obs_path), clip_rows in sorted(groups.items()):
        obs = load_npz(obs_path)
        xy, usable = obs['xy_px'], obs['usable']
        t = len(xy)
        if xy.shape != (t, 75, 2) or usable.shape != (t, 75) or usable.dtype != bool:
            raise ValueError('Observation shape/dtype mismatch')
        if not np.array_equal(obs['frame_id'], np.arange(t)) or not (np.diff(obs['timestamp_us']) > 0).all():
            raise ValueError('Observation frame/time ordering invalid')
        buffers, inputs, needed, frames_sha, panels, audit_positions = {}, {}, defaultdict(list), {}, {}, {}
        for r in clip_rows:
            sid = r['sample_id']
            inp = load_npz(r['input_path'])
            fid, ts = inp['frame_id'], inp['timestamp_us']
            if fid.dtype.kind not in 'iu' or fid.ndim != 1 or not len(fid) or fid.min() < 0 or fid.max() >= t:
                raise ValueError('Prepared frame IDs invalid')
            if not (np.diff(fid) > 0).all() or not np.array_equal(ts, obs['timestamp_us'][fid]):
                raise ValueError('Prepared/observed frame-time mismatch')
            if inp['x'].shape != (len(fid), 280) or not np.isfinite(inp['x']).all():
                raise ValueError('Prepared x invalid')
            arrays = allocate(len(fid))
            arrays.update(frame_id=fid.copy(), timestamp_us=ts.copy(),
                          hand_xy_px=np.stack([xy[fid, s] for s in HANDS], axis=1),
                          hand_usable=np.stack([usable[fid, s] for s in HANDS], axis=1))
            buffers[sid], inputs[sid], frames_sha[sid], panels[sid] = arrays, inp, [], []
            audit_positions[sid] = {0, len(fid)//2, len(fid)-1} if sid in visual_ids else set()
            for j, frame_id in enumerate(fid):
                needed[int(frame_id)].append((sid, j))
        got, timing_max, previous_us = set(), 0, -1
        with av.open(str(video_path), mode='r') as con:
            stream = con.streams.video[0]
            if float(stream.metadata.get('rotate', '0')) % 360 != 0:
                raise ValueError('Rotated stream requires separate reviewed mapping')
            if stream.sample_aspect_ratio is not None and stream.sample_aspect_ratio != 1:
                raise ValueError('Non-unit sample aspect ratio')
            dims = (stream.width, stream.height)
            for fid, frame in enumerate(con.decode(stream)):
                total_decode += 1
                if frame.pts is None or stream.time_base is None:
                    raise ValueError('Missing PTS/timebase')
                us = int(round(frame.pts * stream.time_base * 1_000_000))
                if fid == 0 and us != 0:
                    raise ValueError('Nonzero origin is outside eligible prepared113 contract')
                if us <= previous_us:
                    raise ValueError('Non-monotonic PTS')
                previous_us = us
                if (frame.width, frame.height) != dims or getattr(frame, 'rotation', 0) % 360 != 0:
                    raise ValueError('Changed dimensions/rotation')
                if fid in needed:
                    delta = abs(us-int(obs['timestamp_us'][fid]))
                    if delta > 1:
                        raise ValueError(f'PTS mismatch at frame {fid}: {delta}us')
                    timing_max = max(timing_max, delta)
                    rgb = frame.to_ndarray(format='rgb24')
                    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                    shoulder = xy[fid, [11, 12]].copy()
                    shoulder[~usable[fid, [11, 12]]] = np.nan
                    crops = [make_crop(rgb, xy[fid, s], usable[fid, s], shoulder) for s in HANDS]
                    for sid, j in needed[fid]:
                        arrays = buffers[sid]
                        arrays['source_width_height'][j] = dims
                        for side, cr in enumerate(crops):
                            for key, value in cr.items():
                                arrays[key][j, side] = value
                        frames_sha[sid].append({'frame_id': fid, 'pts_us': us, 'rgb_sha256': digest})
                        if j in audit_positions[sid]:
                            Image.fromarray(rgb).save(out/f'{sid}_f{fid:06d}_raw.png')
                            panels[sid].append((fid, us, rgb.copy(), crops))
                    got.add(fid)
                if fid >= max(needed):
                    break
        if got != set(needed):
            raise ValueError('Video ended before requested frames')
        for r in clip_rows:
            sid = r['sample_id']
            arrays = buffers[sid]
            npz_path = out/f'{sid}.npz'
            with npz_path.open('xb') as f:
                np.savez_compressed(f, **arrays)
            rec = {'sample_id': sid, 'clip_id': r['clip_id'], 'label': r['label'],
                   'signer_id': r['signer_id'], 'frames': len(arrays['frame_id']),
                   'source': {k: v for k, v in r.items() if k.endswith(('_path', '_sha256'))},
                   'output_path': str(npz_path.resolve()), 'output_sha256': sha(npz_path),
                   'timing_max_error_us': timing_max,
                   'per_hand': [{ 'side': side,
                       'reasons': dict(Counter(arrays['crop_reason'][:, h].tolist())),
                       'padding': stats(arrays['padding_fraction'][:, h]),
                       'shoulder_relative_size': stats(arrays['shoulder_relative_size'][:, h]),
                       'frame_relative_size': stats(arrays['frame_relative_size'][:, h])}
                       for h, side in enumerate(('left', 'right'))],
                   'frames_rgb': frames_sha[sid], 'semantic_alignment_verified': False,
                   'landmark_anatomical_accuracy_verified': False,
                   'notes': 'Crops are stored observations, not anatomy/sign truth. Black padding/missing are separately masked. No classification performed.'}
            if panels[sid]:
                sheet(out/f'{sid}_contact.png', sid, panels[sid])
                rec['contact_sheet'] = f'{sid}_contact.png'
            write_json(out/f'{sid}.json', rec)
            reports.append(rec)
    # Source preservation check on a fresh read, including all processed video bytes.
    if any(sha(path) != digest for path, digest in pins.items()):
        raise ValueError('Input source changed during preparation')
    counters = [Counter(), Counter()]
    for r in reports:
        for side in (0, 1):
            counters[side].update(r['per_hand'][side]['reasons'])
    summary = {'status': 'PREPARED_NOT_A_MODEL_OR_SEMANTIC_VALIDATION', 'rows': len(reports),
               'clips': len({r['clip_id'] for r in reports}), 'classes': len({r['label'] for r in reports}),
               'signers': sorted({r['signer_id'] for r in reports}),
               'frames': sum(r['frames'] for r in reports), 'per_hand_reasons': [dict(c) for c in counters],
               'timing_max_error_us': max(r['timing_max_error_us'] for r in reports),
               'decoded_frames': total_decode, 'source_hashes_unchanged': True,
               'new_inference': 0, 'new_training': 0, 'new_pose_extraction': 0,
               'rows_detail': reports}
    write_json(out/'SUMMARY.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    result = run(config, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows_detail'}, ensure_ascii=True))
