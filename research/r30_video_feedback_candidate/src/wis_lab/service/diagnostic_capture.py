"""Opt-in, bounded RGB samples linked to the exact observation frames.

No camera, model, audio or training access. JPEG samples preserve their actual read
timestamps; this is not a constant-frame-rate video or sensor-exposure timing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np

SCHEMA = 'wis_local_capture_review_v1'
MAX_BYTES = 16 * 1024 * 1024
MAX_FRAMES = 128
INTERVAL_US = 100_000       # at most 10 RGB samples/sec; keypoints remain unchanged
MAX_EDGE = 1280


class CaptureBuffer:
    def __init__(self):
        self.frames = []
        self.bytes = 0
        self.last_us = None
        self.failures = 0
        self.limit_reached = False
        self.encode_ms = 0.0

    def add(self, rgb, frame_id: int, timestamp_us: int):
        if self.limit_reached or (self.last_us is not None
                                 and timestamp_us - self.last_us < INTERVAL_US):
            return
        self.last_us = timestamp_us
        if len(self.frames) >= MAX_FRAMES:
            self.limit_reached = True
            return
        start = time.perf_counter_ns()
        try:
            h, w = rgb.shape[:2]
            ratio = min(1.0, MAX_EDGE / max(w, h))
            if ratio < 1:
                rgb = cv2.resize(rgb, (round(w * ratio), round(h * ratio)))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, data = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                self.failures += 1
                return
            blob = data.tobytes()
            if self.bytes + len(blob) > MAX_BYTES:
                self.limit_reached = True
                return
            self.frames.append({'frame_id': int(frame_id), 'timestamp_us': int(timestamp_us),
                                'width': int(bgr.shape[1]), 'height': int(bgr.shape[0]),
                                'source_width': int(w), 'source_height': int(h), 'jpeg': blob})
            self.bytes += len(blob)
        except (cv2.error, ValueError, TypeError):
            self.failures += 1
        finally:
            self.encode_ms += (time.perf_counter_ns() - start) / 1e6


def persist_capture(buffer, obs, path, *, session_id, trial_id):
    """Write only an explicitly captured buffer. Failure never supplies fabricated frames."""
    if buffer is None:
        return {'status': 'DISABLED', 'opted_in': False}
    base = {'schema': SCHEMA, 'opted_in': True, 'frame_count': len(buffer.frames),
            'limit_reached': buffer.limit_reached, 'encode_failures': buffer.failures,
            'encode_total_ms': round(buffer.encode_ms, 3), 'max_sample_fps': 10,
            'audio': False, 'training_use': False}
    if not buffer.frames:
        return {**base, 'status': 'UNAVAILABLE', 'reason': 'NO_RGB_SAMPLES'}
    rows = {int(fid): i for i, fid in enumerate(obs.frame_id)}
    if len(rows) != obs.n_frames:
        raise ValueError('duplicate observation frame ids')
    manifest = {**base, 'status': 'PARTIAL' if buffer.limit_reached or buffer.failures else 'AVAILABLE',
                'session_id': session_id, 'trial_id': trial_id,
                'observation_sha256': obs.sidecar['observation_sha256'],
                'clock_id': obs.sidecar.get('clock_id'),
                'clock': 'same backend frame-read monotonic timestamps as the keypoints; not exposure time',
                'mirrored': False, 'frames': []}
    for frame in buffer.frames:
        i = rows.get(frame['frame_id'])
        if i is None or int(obs.timestamp_us[i]) != frame['timestamp_us']:
            raise ValueError('RGB/keypoint frame identity mismatch')
        if (frame['source_width'], frame['source_height']) != (obs.sidecar['width'], obs.sidecar['height']):
            raise ValueError('RGB/keypoint resolution mismatch')
        points = []
        for point, usable in zip(obs.xy_px[i], obs.usable[i]):
            points.append([float(point[0] / frame['source_width']),
                           float(point[1] / frame['source_height'])]
                          if usable and np.isfinite(point).all() else None)
        entry = {k: v for k, v in frame.items() if k != 'jpeg'}
        entry.update(observation_row=i, image=f'frames/{len(manifest["frames"]):04d}.jpg',
                     sha256=hashlib.sha256(frame['jpeg']).hexdigest(), points_xy_normalized=points)
        manifest['frames'].append(entry)
    path = Path(path)
    # Exclusive creation: a diagnostic file may never replace an existing take.
    with path.open('xb') as output:
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
            for entry, frame in zip(manifest['frames'], buffer.frames):
                archive.writestr(entry['image'], frame['jpeg'])
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False,
                                                        allow_nan=False).encode('utf-8'))
    return {**base, 'status': manifest['status'],
            'observation_sha256': manifest['observation_sha256'], 'bytes': path.stat().st_size}


def read_manifest(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read('manifest.json'))


def read_frame(path, index):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if index < 0 or index >= len(manifest['frames']):
            raise IndexError(index)
        entry = manifest['frames'][index]
        data = archive.read(entry['image'])
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('RGB frame hash mismatch')
        return data
