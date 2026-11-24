"""Isolated output-preserving latency candidate; no cached video predictions.

Changes from the preserved baseline: preflight metadata is consumed directly by
an exact local extraction loop, and RGB preparation overlaps one sequential GPU
consumer through a bounded queue. All frames/windows/face/precision stay intact.

Use the existing SignRep runtime/Scripts/python.exe. One signer per video;
multiple signers are separate submissions. Fixed seed42 is a route choice, not
a claim that it is best. All outputs are estimates, never a KSL grade.

analyze_video(path, workdir, progress=None) writes workdir/adapter exactly once.
It returns numpy arrays. NaN posterior means unavailable (new4 at six boundary
frames on either side); geometry mask and posterior availability are separate.
Face observations are a separate sidecar, never inputs to the trained69/7 model.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import queue
import threading
from collections import deque
from functools import lru_cache

import numpy as np

UNIT = Path(__file__).resolve().parent
OPS = UNIT.parent
LAB = OPS.parent
ROOT = LAB.parents[1]
BASELINE_UNIT = OPS / 'word100_multipeople_live_feedback_20261101'
BASELINE_ADAPTER_SHA256 = '93d28ec8ccd787c4d42f9fd6c3b9425f1470da1ef8449f428f7bce4c103f1f48'
SIGNREP = OPS / 'word100_signrep_feasibility_20261029'
OLD = OPS / 'word100_stage_feedback_cycle_20261031'
SEMANTIC = OPS / 'word100_semantic_feature_cycle_20261101'
SEED = 42
TARGETS = ('R_index_only', 'R_open_four', 'L_open_four', 'R_closed_four',
           'L_closed_four', 'R_higher', 'L_higher')
RUNTIME_PYTHON = SIGNREP / 'runtime/Scripts/python.exe'
MAX_DECODED_FRAMES = 1200
MAX_ELAPSED_US = 20_000_000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.dont_write_bytecode = previous
    return mod


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def _save(path, **arrays):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **arrays)


def _notify(progress, stage, **fields):
    if progress is not None:
        progress(dict(stage=stage, **fields))


def normalize(a, stats):
    """Exactly the saved FIT normalization; never fit to a submitted video."""
    x, mask, rgb = np.asarray(a['x']), np.asarray(a['mask']), np.asarray(a['rgb'])
    if x.ndim != 2 or x.shape[1] != 69 or mask.shape != x.shape or mask.dtype != bool:
        raise ValueError('Expected geometry[N,69] and boolean mask')
    if rgb.shape != (len(x), 768) or not np.isfinite(rgb).all():
        raise ValueError('Finite aligned SignRep[N,768] required')
    if not np.isfinite(x[mask]).all():
        raise ValueError('Supported geometry must be finite')
    pm, ps, zm, zs = (np.asarray(stats[k], dtype=np.float64) for k in
                      ('pose_mean', 'pose_std', 'rgb_mean', 'rgb_std'))
    if (pm.shape != (69,) or ps.shape != (69,) or zm.shape != (768,) or zs.shape != (768,)
            or not all(np.isfinite(v).all() for v in (pm, ps, zm, zs))
            or (ps <= 0).any() or (zs <= 0).any()):
        raise ValueError('Invalid saved normalization')
    values = np.where(mask, (x - pm) / ps, 0)
    return (np.concatenate((values, mask.astype(np.float32)), axis=1).astype(np.float32),
            ((rgb - zm) / zs).astype(np.float32))


def input_support(mask, names):
    """Detector support proxy only: it must not hide a finite model estimate."""
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape[1:] != (69,) or len(set(names)) != 69:
        raise ValueError('Invalid69 support schema')
    out = np.zeros((len(mask), 7), bool)
    for j, target in enumerate(TARGETS):
        if target.endswith('higher'):
            cols = [names.index(s + '_palm_center_' + d) for s in ('L', 'R') for d in ('x', 'y')]
            out[:, j] = mask[:, cols].all(1)
        else:
            side = target[0]
            wrist = [names.index(side + '_hand_wrist_' + d) for d in ('x', 'y')]
            fingers = [k for k, name in enumerate(names)
                       if name.startswith(side + '_') and '_angle' in name and 'elbow' not in name]
            if len(fingers) != 14:
                raise ValueError('Expected14 finger angles per hand')
            out[:, j] = mask[:, wrist].all(1) & (mask[:, fingers].sum(1) >= 10)
    return out


def window_centers(frame_id):
    ids = np.asarray(frame_id)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or (np.diff(ids) <= 0).any():
        raise ValueError('Strict original integer frame IDs required')
    return np.asarray([i for i in range(6, len(ids) - 6)
                       if np.array_equal(ids[i-6:i+7], np.arange(ids[i]-6, ids[i]+7))], dtype=np.int64)


def roi_arrays(observation, box):
    """Preserve full-source observations; limit only the69 feature usable mask."""
    a = {k: np.array(observation[k], copy=True)
         for k in ('xy_px', 'observed', 'usable', 'timestamp_us', 'frame_id')}
    xy = a['xy_px']; x0, y0, x1, y1 = box
    inside = (np.isfinite(xy).all(-1) & (xy[..., 0] >= x0) & (xy[..., 0] < x1)
              & (xy[..., 1] >= y0) & (xy[..., 1] < y1))
    a['usable'] &= inside
    return a


def runtime_info():
    """Inventory only; no image, checkpoint or inference execution."""
    packages = {n: importlib.util.find_spec(n) is not None
                for n in ('numpy', 'torch', 'mediapipe', 'av', 'cv2', 'openvino')}
    return dict(python=sys.executable, recommended_python=str(RUNTIME_PYTHON), packages=packages,
                models_present={name: (ROOT/'Tools/models'/name).is_file() for name in
                                ('pose_landmarker_lite.task', 'hand_landmarker.task', 'face_landmarker.task')})


def _compile_signrep_uncached():
    os.environ['INTEL_OPENVINO_TELEMETRY_CONSENT'] = '0'
    os.environ['OPENVINO_TELEMETRY_DISABLED'] = '1'
    sys.modules.setdefault('openvino_telemetry', None)
    import openvino as ov
    exported = read(SIGNREP/'EXPORT.json')
    xml, binary = (SIGNREP/'converted'/n for n in ('signrep_features_fp32.xml', 'signrep_features_fp32.bin'))
    for p, k in ((xml, 'xml_sha256'), (binary, 'bin_sha256')):
        if sha(p) != exported[k]:
            raise ValueError('Frozen SignRep IR changed')
    core = ov.Core(); devices = list(core.available_devices); attempts = []
    requested = ['GPU', 'CPU'] if any(d.startswith('GPU') for d in devices) else ['CPU']
    for device in requested:
        try:
            ir = core.read_model(xml); ir.reshape([1, 3, 16, 224, 224])
            config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1',
                      'INFERENCE_PRECISION_HINT': 'f16' if device == 'GPU' else 'f32'}
            compiled = core.compile_model(ir, device, config)
            actual = [str(d) for d in compiled.get_property('EXECUTION_DEVICES')]
            if not all(device in d for d in actual):
                raise RuntimeError('Unexpected actual inference device')
            return compiled, dict(requested=device, actual=actual, available=devices,
                                  precision=config['INFERENCE_PRECISION_HINT'], failed_attempts=attempts)
        except Exception as exc:
            attempts.append(dict(device=device, error=repr(exc)))
    raise RuntimeError('No existing OpenVINO device could compile SignRep: ' + repr(attempts))


@lru_cache(maxsize=1)
def _compile_signrep():
    """One immutable frozen engine per process; caller serializes inference."""
    return _compile_signrep_uncached()


def _bounded_video_preflight(source):
    """Bound actual decoding before allocating observations or creating detectors."""
    import av
    from fractions import Fraction
    decoded = 0; timestamped = 0; origin = None; maximum = 0; last_pts = None
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        tb = stream.time_base or Fraction(1, 1000)
        for packet in container.demux(stream):
            for frame in packet.decode():
                decoded += 1
                if decoded > MAX_DECODED_FRAMES:
                    raise ValueError('VIDEO_FRAME_LIMIT_EXCEEDED: at most1200 decoded frames')
                if frame.pts is not None:
                    timestamped += 1; last_pts = frame.pts
                    if origin is None:
                        origin = frame.pts
                    elapsed = int(round(float((frame.pts-origin)*tb)*1e6))
                    maximum = max(maximum, elapsed)
                    if elapsed > MAX_ELAPSED_US:
                        raise ValueError('VIDEO_TIME_LIMIT_EXCEEDED: at most20s from first actual PTS')
        metadata = dict(frame_count=timestamped, width=int(stream.codec_context.width),
            height=int(stream.codec_context.height), time_base=f'{tb.numerator}/{tb.denominator}',
            duration_ms=round(float((last_pts-origin)*tb)*1000, 3) if timestamped > 1 else 0.,
            avg_rate=(f'{stream.average_rate.numerator}/{stream.average_rate.denominator}'
                      if stream.average_rate else None),
            codec=stream.codec_context.name, pix_fmt=stream.codec_context.pix_fmt)
    if origin is None:
        raise ValueError('No timestamped video frame')
    return dict(decoded_frames=decoded, maximum_elapsed_us=maximum, first_source_pts=int(origin),
                max_decoded_frames=MAX_DECODED_FRAMES, max_elapsed_us=MAX_ELAPSED_US,
                probe_metadata=metadata)


def _extract_with_metadata(path, lm, clip_id, meta):
    """Local copy of frozen video_source.extract, omitting only its probe call.

    `meta` must come from the completed bounded preflight, never header counts.
    Timestamped preflight frames allocate the same original upper bound; the
    extraction loop retains original PTS/drop/frame-ID and pixel operations.
    """
    import av
    from fractions import Fraction
    from wis_lab.core.observation import Observation, empty_frames, sha256_file
    if not 0 < meta['frame_count'] <= MAX_DECODED_FRAMES:
        raise ValueError('Bounded timestamped preflight metadata required')
    out = empty_frames(meta['frame_count']); dropped_non_monotonic = 0; row = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        tb = stream.time_base or Fraction(1, 1000)
        origin_pts = None; last_us = -1
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                if origin_pts is None:
                    origin_pts = frame.pts
                ts_us = int(round(float((frame.pts-origin_pts)*tb)*1_000_000))
                if ts_us <= last_us:
                    dropped_non_monotonic += 1; continue
                last_us = ts_us
                if row >= len(out['frame_id']):
                    raise ValueError('Source grew after bounded preflight')
                out['timestamp_us'][row] = ts_us
                out['frame_id'][row] = row+dropped_non_monotonic
                lm.process(frame.to_ndarray(format='rgb24'), ts_us, out, row)
                row += 1
    for key in out:
        out[key] = out[key][:row]
    obs = Observation(**out)
    obs.sidecar = dict(schema='wis_observation_2d_v1', origin='clip', clip_id=clip_id,
        source_path=str(path).replace('\\', '/'), source_sha256=sha256_file(str(path)),
        width=meta['width'], height=meta['height'], time_base=meta['time_base'],
        decoded_frames=row, dropped_non_monotonic_pts=dropped_non_monotonic,
        mirrored=False, rotation_deg=0, extractor_id=lm.extractor_id(),
        hand_assignment_policy=lm.contract['hand_assignment']['policy'],
        handedness_label_disagreements=lm.handedness_disagreements,
        handedness_label_comparisons=lm.handedness_comparisons,
        unassigned_hand_detections=lm.unassigned_hands,
        clock='video PTS * time_base, microseconds from first decoded frame')
    obs.sidecar['quality'] = obs.quality_summary()
    return obs


def _capture(source, directory, progress):
    """Original raw detector + original joint_v3, with separate same-frame face."""
    sys.path.insert(0, str(LAB/'src')) if str(LAB/'src') not in sys.path else None
    from wis_lab.capture.face_landmarks import FaceLandmarks
    from wis_lab.core.observation import Observation, empty_frames
    cap = _module('_live_raw_capture', OPS/'word100_hand_assignment_20261016/capture_raw.py')
    assign = _module('_live_joint_assignment', OPS/'word100_hand_assignment_joint_20261016/assignment.py')
    contract = read(LAB/'config/contract.json')
    capture_profile = {}; t = time.perf_counter()
    limits = _bounded_video_preflight(source)
    capture_profile['preflight_decode_seconds'] = time.perf_counter()-t

    class Capture(cap.RawCaptureLandmarker):
        def __init__(self):
            super().__init__(contract)
            try:
                self.face = FaceLandmarks(ROOT/'Tools/models/face_landmarker.task')
            except BaseException:
                super().close(); raise
            self.faces = []
            self.body_hands_seconds = 0.; self.face_seconds = 0.

        def process(self, rgb, timestamp_us, out, row):
            if row >= MAX_DECODED_FRAMES or timestamp_us > MAX_ELAPSED_US:
                raise ValueError('Actual decoded frame/time limit exceeded before detector call')
            t = time.perf_counter(); super().process(rgb, timestamp_us, out, row)
            self.body_hands_seconds += time.perf_counter()-t
            # Same strictly increasing detector ms as body; original us stored independently.
            t = time.perf_counter(); self.faces.append(self.face.process(rgb, self._last_ms * 1000))
            self.face_seconds += time.perf_counter()-t
            if row % 30 == 0:
                _notify(progress, 'detecting', frames=row+1)

        def close(self):
            super().close()
            if hasattr(self, 'face'):
                self.face.close()

    t = time.perf_counter(); lm = Capture()
    capture_profile['detector_initialization_seconds'] = time.perf_counter()-t
    t = time.perf_counter()
    try:
        baseline = _extract_with_metadata(str(source), lm, 'submitted_' + sha(source)[:16], limits['probe_metadata'])
    finally:
        lm.close()
    capture_profile['extract_and_detector_close_seconds'] = time.perf_counter()-t
    capture_profile['body_hands_raw_record_seconds'] = lm.body_hands_seconds
    capture_profile['face_seconds'] = lm.face_seconds
    t = time.perf_counter()
    if not lm.raw_rows:
        raise ValueError('No decodable timestamped frames')
    raw = {k: np.stack([r[k] for r in lm.raw_rows]) for k in lm.raw_rows[0]}
    cap.verify_capture(raw, baseline, contract)
    # Same frozen associate implementation, without invoking its old runner.
    out = empty_frames(baseline.n_frames)
    for k in ('xy_px', 'observed', 'usable', 'conf'):
        out[k][:, :33] = getattr(baseline, k)[:, :33]
    out['frame_id'][:] = baseline.frame_id; out['timestamp_us'][:] = baseline.timestamp_us
    ass = assign.HandAssigner(); sides = np.full((baseline.n_frames, 2), -1, np.int8); trace = []
    margin = contract['usable_rule']['bounds_margin_ratio']; threshold = contract['usable_rule']['hand_score_threshold']
    for f in range(baseline.n_frames):
        n = int(raw['raw_count'][f]); w, h = int(raw['width'][f]), int(raw['height'][f])
        chosen, diag = ass.assign(raw['hand_xy_px'][f, :n], raw['raw_label'][f, :n], raw['raw_score'][f, :n],
                                  raw['pose_xy_px'][f], raw['pose_usable'][f], int(baseline.timestamp_us[f]), w, h)
        diag.update(frame=int(baseline.frame_id[f]), timestamp_us=int(baseline.timestamp_us[f])); trace.append(diag)
        for j in range(n):
            side = chosen[j]
            if side is None:
                continue
            sides[f, j] = side; sl = slice(33+21*side, 54+21*side)
            pts = raw['hand_xy_px'][f, j].astype(np.float32); score = raw['raw_score'][f, j]
            out['xy_px'][f, sl] = pts; out['observed'][f, sl] = True; out['hand_score'][f, side] = score
            out['usable'][f, sl] = (np.isfinite(pts).all(1) & (pts[:, 0] >= -margin*w)
                & (pts[:, 0] <= w*(1+margin)) & (pts[:, 1] >= -margin*h)
                & (pts[:, 1] <= h*(1+margin)) & (score >= threshold))
    ob = Observation(**out)
    ob.sidecar = dict(baseline.sidecar, hand_assignment_policy='frozen_joint_v3',
                      assignment_sha256=assign.HandAssigner.identity(), video_limits=limits)
    ob.validate()
    capture_profile['assignment_and_validation_seconds'] = time.perf_counter()-t
    t = time.perf_counter()
    baseline.save(str(directory/'baseline.npz')); ob.save(str(directory/'observations.npz'))
    _save(directory/'raw.npz', **raw); _save(directory/'assignment.npz', sides=sides)
    _write(directory/'assignment_trace.json', trace)
    face = np.asarray(lm.faces, np.float32)
    if face.shape != (ob.n_frames, 478, 3):
        raise ValueError('Face/body time axes differ')
    _save(directory/'face.npz', xyz_normalized=face, found=np.isfinite(face).all((1, 2)),
          frame_id=ob.frame_id, timestamp_us=ob.timestamp_us,
          detector_timestamp_ms=raw['detector_timestamp_ms'], image_size=np.array([w, h]))
    capture_profile['capture_artifact_save_seconds'] = time.perf_counter()-t
    _write(directory/'CAPTURE_TIMING.json', capture_profile)
    return ob, face


class EncodingCancelled(RuntimeError):
    pass


def _window_pipeline(items, compiled, progress=None, *, total_windows=None,
                     cancel_event=None, profile=None):
    """Overlap preparation with ONE sequential batch1 inference owner.

    Every queued input owns its memory. Cancellation stops future calls and waits
    for the current synchronous call before returning; it never leaves a live
    worker using a cached engine. An already executing device call cannot be
    forcibly interrupted by this pure scheduling change. No GPU batching,
    additional streams, multiprocessing or model-state sharing is introduced.
    """
    work = queue.Queue(maxsize=4); stop = threading.Event(); done = threading.Event()
    sentinel = object(); errors = []; outputs = []; completed_starts = []
    stats = dict(producer_prepare_seconds=0., enqueue_wait_seconds=0.,
                 inference_call_seconds=0., consumer_seconds=0.,
                 pipeline_wall_seconds=0., worker_join_seconds=0.,
                 maximum_queue_depth=0, windows=0)
    began = time.perf_counter()

    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    def raise_worker_error():
        if errors:
            exc, tb = errors[0]
            raise exc.with_traceback(tb)
        if cancelled():
            raise EncodingCancelled('RGB encoding cancelled; no partial output is accepted')
        if done.is_set():
            raise RuntimeError('GPU worker exited before producer completion')

    def consume():
        t0 = time.perf_counter()
        try:
            while not stop.is_set():
                if cancelled():
                    raise EncodingCancelled('RGB encoding cancelled; no partial output is accepted')
                try:
                    item = work.get(timeout=.1)
                except queue.Empty:
                    continue
                if item is sentinel:
                    break
                start, inp = item
                if stop.is_set() or cancelled():
                    raise EncodingCancelled('RGB encoding cancelled before next GPU call')
                t = time.perf_counter()
                value = np.array(compiled([inp])[0], dtype=np.float32, copy=True).reshape(1, 768)
                stats['inference_call_seconds'] += time.perf_counter()-t
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite SignRep output')
                completed_starts.append(int(start)); outputs.append(value)
        except BaseException as exc:
            errors.append((exc, exc.__traceback__)); stop.set()
        finally:
            stats['consumer_seconds'] = time.perf_counter()-t0
            done.set()

    iterator = iter(items); expected_starts = []
    worker = threading.Thread(target=consume, name='candidate-signrep-single-consumer', daemon=False)
    worker.start(); producer_error = None

    def enqueue(value):
        t = time.perf_counter()
        try:
            while True:
                raise_worker_error()
                try:
                    work.put(value, timeout=.1)
                    stats['maximum_queue_depth'] = max(stats['maximum_queue_depth'], work.qsize())
                    return
                except queue.Full:
                    continue
        finally:
            stats['enqueue_wait_seconds'] += time.perf_counter()-t

    try:
        while True:
            raise_worker_error()
            t = time.perf_counter()
            try:
                start, inp = next(iterator)
            except StopIteration:
                stats['producer_prepare_seconds'] += time.perf_counter()-t
                break
            stats['producer_prepare_seconds'] += time.perf_counter()-t
            if inp.shape != (1, 3, 16, 224, 224) or inp.dtype != np.float32:
                raise ValueError('Exact batch1 float32 SignRep tensor required')
            # The generator's input is not reused after yield. Enforce owned
            # contiguous storage so a decoder/ring buffer cannot mutate it.
            inp = np.array(inp, dtype=np.float32, copy=True, order='C')
            enqueue((int(start), inp)); expected_starts.append(int(start))
            if len(expected_starts) % 20 == 0:
                _notify(progress, 'rgb_features', windows=len(expected_starts), total_windows=total_windows,
                        meaning='windows prepared; completion waits for the sole GPU consumer')
        enqueue(sentinel)
    except BaseException as exc:
        producer_error = (exc, exc.__traceback__); stop.set()
    finally:
        try:
            if hasattr(iterator, 'close'):
                iterator.close()
        except BaseException as exc:
            if producer_error is None:
                producer_error = (exc, exc.__traceback__); stop.set()
        t = time.perf_counter()
        worker.join()  # do not return a cached engine with an in-flight consumer
        stats['worker_join_seconds'] = time.perf_counter()-t
        stats['pipeline_wall_seconds'] = time.perf_counter()-began
        stats['windows'] = len(outputs)
        if profile is not None:
            profile.update(stats)
    if producer_error is not None:
        exc, tb = producer_error
        raise exc.with_traceback(tb)
    if errors:
        exc, tb = errors[0]
        raise exc.with_traceback(tb)
    if cancelled():
        raise EncodingCancelled('RGB encoding cancelled; no partial output is accepted')
    if not expected_starts or completed_starts != expected_starts:
        raise ValueError('RGB worker changed full window order or coverage')
    if total_windows is not None and len(outputs) != total_windows:
        raise ValueError('RGB window count differs from fixed starts')
    return np.concatenate(outputs), np.asarray(completed_starts, dtype=np.int64)


def _encode(source, box, frame_id, timestamps, compiled, progress, *, cancel_event=None, profile=None):
    """Original tensor math/PTS join; bounded preparation overlaps one GPU worker."""
    import av
    import cv2
    common = _module('_live_signrep_common', SIGNREP/'common.py')
    seq = common.starts(len(timestamps)); ends = {s+15: s for s in seq}
    wanted = {int(f): i for i, f in enumerate(frame_id)}
    got = []
    def prepared_windows():
        buf = deque(maxlen=16); origin = None; last = -1; kept = 0; dropped = 0
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            for packet in container.demux(stream):
                for frame in packet.decode():
                    if cancel_event is not None and cancel_event.is_set():
                        raise EncodingCancelled('RGB decoding cancelled')
                    if frame.pts is None:
                        continue
                    if origin is None:
                        origin = frame.pts
                    ts = int(round(float((frame.pts-origin)*stream.time_base)*1e6))
                    if ts <= last:
                        dropped += 1; continue
                    last = ts; fid = kept+dropped; kept += 1
                    if fid not in wanted:
                        raise ValueError('RGB/body source-frame mismatch')
                    i = wanted[fid]
                    if i != len(got) or ts != int(timestamps[i]):
                        raise ValueError('RGB/body exact PTS mismatch')
                    got.append(fid); im = frame.to_ndarray(format='rgb24'); x0, y0, x1, y1 = box
                    im[:y0] = [124, 116, 104]; im[y1:] = [124, 116, 104]
                    im[:, :x0] = [124, 116, 104]; im[:, x1:] = [124, 116, 104]
                    im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
                    buf.append(common.norm(im[None])[0])
                    if i in ends:
                        inp = np.ascontiguousarray(np.stack(buf).transpose(1, 0, 2, 3)[None])
                        yield ends[i], inp
    features, starts = _window_pipeline(prepared_windows(), compiled, progress,
        total_windows=len(seq), cancel_event=cancel_event, profile=profile)
    if got != list(map(int, frame_id)) or starts.tolist() != seq:
        raise ValueError('RGB output source coverage changed')
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite SignRep output')
    t = time.perf_counter()
    centers = (timestamps[starts]+timestamps[starts+15])/2
    rgb = np.stack([np.interp(timestamps, centers, features[:, j]) for j in range(768)], 1).astype(np.float32)
    if profile is not None:
        profile['interpolation_seconds'] = time.perf_counter()-t
    return rgb, features, starts, centers


def _load_head_models():
    import torch
    torch.set_num_threads(4)
    comparison = _module('_live_semantic_comparison', SEMANTIC/'comparison.py')
    old_head = comparison.load_frozen(SEED)
    complete = read(SEMANTIC/f'runs/protected_shape_split_features_{SEED}/COMPLETE.json')
    checkpoint = SEMANTIC/f'runs/protected_shape_split_features_{SEED}/head.pt'
    if sha(checkpoint) != complete['checkpoint_sha256'] or sha(SEMANTIC/'NORMALIZATION.json') != complete['normalization_sha256']:
        raise ValueError('Frozen new4 checkpoint/normalization changed')
    if tuple(read(SEMANTIC/'DATASET.json')['selected_targets']) != TARGETS:
        raise ValueError('Frozen target order changed')
    head = comparison.WindowHead(4)
    head.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    head.eval(); head.requires_grad_(False)
    old_names = read(OLD/'LEARNING_INPUTS.json')['targets']
    cols = [old_names.index(t) for t in TARGETS[:3]]
    return dict(old=old_head, new=head, old_stats=read(OLD/'NORMALIZATION.json'),
                new_stats=read(SEMANTIC/'NORMALIZATION.json'), old_columns=cols)


@lru_cache(maxsize=1)
def _head_runtime():
    """Checkpoint hashes are verified once at load, with unchanged FIT stats."""
    return _load_head_models()


def _frozen_heads(runtime):
    for model in (runtime['old'], runtime['new']):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Shared observer must remain eval and gradient-disabled')


def _heads(a):
    import torch
    runtime = _head_runtime(); _frozen_heads(runtime)
    old_head, head, cols = runtime['old'], runtime['new'], runtime['old_columns']
    ox, oz = normalize(a, runtime['old_stats'])
    nx, nz = normalize(a, runtime['new_stats'])
    centers = window_centers(a['frame_id']); n = len(ox)
    p = np.full((n, 7), np.nan, np.float32)
    with torch.inference_mode():
        old_p = old_head(torch.from_numpy(ox[None]), torch.from_numpy(oz[None]), torch.ones((1, n))).sigmoid().numpy()[0]
        p[:, :3] = old_p[:, cols]
        for chunk in range(0, len(centers), 64):
            ids = centers[chunk:chunk+64]
            xx = torch.from_numpy(np.stack([nx[i-6:i+7] for i in ids]))
            zz = torch.from_numpy(np.stack([nz[i-6:i+7] for i in ids]))
            p[ids, 3:] = head(xx, zz).sigmoid().numpy()
    if not np.isfinite(p[:, :3]).all() or not np.isfinite(p[centers, 3:]).all():
        raise ValueError('Nonfinite trained-head prediction')
    return p, centers


@lru_cache(maxsize=1)
def warmup():
    """Optional startup-only initialization using synthetic zeros, not a video.

    The server calls this before declaring ready and owns a single analysis
    worker. No video predictions are cached; MediaPipe state is still fresh for
    every submission. No training, precision change or frame omission occurs.
    """
    import torch
    started = time.perf_counter()
    t = time.perf_counter(); compiled, device = _compile_signrep()
    compile_seconds = time.perf_counter()-t
    t = time.perf_counter()
    compiled([np.zeros((1, 3, 16, 224, 224), np.float32)])
    rgb_warm_seconds = time.perf_counter()-t
    t = time.perf_counter(); runtime = _head_runtime(); _frozen_heads(runtime)
    with torch.inference_mode():
        x = torch.zeros(1, 13, 138); z = torch.zeros(1, 13, 768)
        runtime['old'](x, z, torch.ones(1, 13))
        runtime['new'](x, z)
    return dict(status='WARMED_FROZEN_OBSERVERS', device=dict(device),
                compile_seconds=compile_seconds, synthetic_rgb_warm_seconds=rgb_warm_seconds,
                head_load_and_synthetic_warm_seconds=time.perf_counter()-t,
                total_seconds=time.perf_counter()-started, real_video_inference=0,
                synthetic_rgb_windows=1, synthetic_head_windows=1, single_analysis_worker_required=True)


def analyze_video(path, workdir, progress=None):
    """Analyze this exact submitted video once; caller owns upload authorization.

    workdir must be inside this research unit. A prior adapter directory causes
    refusal, including partial failures; completed evidence is never overwritten.
    Uncalibrated/too-short inputs return explicit unsupported status, not guesses.
    """
    source = Path(path).resolve(); destination = Path(workdir).resolve()/'adapter'
    if not source.is_file():
        raise FileNotFoundError(source)
    if not destination.is_relative_to(UNIT.resolve()):
        raise ValueError('Adapter output must remain inside its research unit')
    if sha(BASELINE_UNIT/'adapter.py') != BASELINE_ADAPTER_SHA256:
        raise ValueError('Preserved baseline adapter differs from reviewed candidate source')
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); timings = {}; source_hash = sha(source)
    _write(destination/'STARTED.json', dict(source_sha256=source_hash, seed=SEED,
                                          fresh_video=True, training=False, cached_predictions_used=False,
                                          candidate='preflight_metadata_and_queue4_single_GPU',
                                          baseline_adapter_sha256=BASELINE_ADAPTER_SHA256))
    try:
        _notify(progress, 'starting', runtime=runtime_info())
        t = time.perf_counter(); ob, face = _capture(source, destination, progress)
        timings['decode_mediapipe_assignment_save_seconds'] = time.perf_counter()-t
        n = ob.n_frames; width, height = int(ob.sidecar['width']), int(ob.sidecar['height'])
        boundary = _module('_live_body_boundary', OPS/'word100_body_boundary_20261017/boundary.py')
        t = time.perf_counter()
        calibration = boundary.calibrate(ob, width, height, read(OPS/'word100_body_boundary_20261017/design_data_COEFFICIENT.json'))
        _write(destination/'CALIBRATION.json', calibration)
        full = {k: getattr(ob, k).copy() for k in ('xy_px', 'observed', 'usable', 'timestamp_us', 'frame_id', 'conf', 'hand_score')}
        observer = _module('_live_motion_observer', OPS/'word100_guide_video_feedback_20261028/motion_observer.py')
        unsupported = []
        if calibration['status'] != 'CALIBRATED':
            unsupported.append('UPPER_BODY_BOUNDARY_UNAVAILABLE')
            masked = {k: v.copy() for k, v in full.items()}; masked['usable'][:] = False
        else:
            masked = roi_arrays(full, calibration['box_source_pixels'])
        feat = observer.extract_features(masked)
        a = dict(x=np.nan_to_num(feat['x']).astype(np.float32), mask=feat['mask'], frame_id=ob.frame_id,
                 timestamp_us=ob.timestamp_us, rgb=np.full((n, 768), np.nan, np.float32))
        timings['boundary_and_geometry_seconds'] = time.perf_counter()-t
        if n < 16:
            unsupported.append('FEWER_THAN16_REAL_FRAMES_NO_PADDING')
        probabilities = np.full((n, 7), np.nan, np.float32); centers = np.empty(0, np.int64); device = None
        runtime_reuse = dict(signrep_initialized_before_video=_compile_signrep.cache_info().currsize > 0,
                             heads_initialized_before_video=_head_runtime.cache_info().currsize > 0,
                             single_analysis_worker_required=True, predictions_cached=False)
        if not unsupported:
            _notify(progress, 'initializing_rgb_engine')
            t = time.perf_counter(); compiled, device = _compile_signrep()
            timings['signrep_compile_seconds'] = time.perf_counter()-t
            t = time.perf_counter()
            encode_profile = {}
            a['rgb'], windows, starts, mid = _encode(source, calibration['box_source_pixels'], ob.frame_id, ob.timestamp_us, compiled, progress, profile=encode_profile)
            timings['rgb_decode_and_signrep_seconds'] = time.perf_counter()-t
            _write(destination/'ENCODE_TIMING.json', encode_profile)
            _save(destination/'rgb_windows.npz', features=windows, starts=starts, center_timestamp_us=mid)
            _notify(progress, 'predicting_attributes')
            t = time.perf_counter(); probabilities, centers = _heads(a)
            timings['head_load_and_inference_seconds'] = time.perf_counter()-t
        names = list(feat['feature_names']); support = input_support(a['mask'], names)
        available = np.isfinite(probabilities)
        _save(destination/'analysis.npz', **a, probabilities=probabilities, posterior_available=available,
              input_support=support, new_four_center_indices=centers)
        if sha(source) != source_hash:
            raise ValueError('Source changed during analysis')
        timings['total_adapter_seconds'] = time.perf_counter()-started
        summary = dict(status='UNSUPPORTED_INPUT' if unsupported else 'COMPLETE_RESEARCH_ESTIMATES',
            unavailable_reasons=unsupported, source_sha256=source_hash, frames=n, targets=list(TARGETS),
            feature_names=names, seed=SEED, device=device, calibration=calibration, latency=timings,
            runtime_reuse=runtime_reuse,
            probability_route='fresh old3 fullclip + fresh new4 exact13frame centers; separate saved FIT normalizations',
            new_four_predicted_frames=len(centers), cached_predictions_used=False, training=False,
            postcapture_future_context=True, face_used_by_attribute_head=False, input_support_is_confidence=False,
            no_educational_or_sign_correctness_verdict=True, one_signer_per_video=True,
            face_z_is_metric_depth=False, total_excludes_upload_dtw_and_render=True,
            performance_candidate=dict(baseline_unit=str(BASELINE_UNIT), baseline_adapter_sha256=BASELINE_ADAPTER_SHA256,
                full_decode_passes=3, queue_limit=4, sequential_GPU_consumers=1, face_preserved=True,
                no_frame_or_window_omission=True, precision_and_batch_unchanged=True,
                cancellation='Stop new work and join current synchronous inference before returning; no forced in-flight device cancellation'),
            output=str(destination/'analysis.npz'),
            artifacts={p.name: sha(p) for p in destination.iterdir() if p.is_file()})
        _write(destination/'COMPLETE.json', summary)
        _notify(progress, 'complete', status=summary['status'], frames=n)
        return dict(**a, probabilities=probabilities, posterior_available=available, input_support=support,
                    timestamps=ob.timestamp_us, observations=full, face_xyz_normalized=face,
                    face_found=np.isfinite(face).all((1, 2)), feature_names=names, targets=list(TARGETS),
                    new_four_center_indices=centers, latency=timings, metadata=summary,
                    calibration=calibration, status=summary['status'])
    except BaseException:
        failure = destination/'FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video'); parser.add_argument('--workdir'); parser.add_argument('--runtime-info', action='store_true')
    args = parser.parse_args()
    if args.runtime_info:
        print(json.dumps(runtime_info(), indent=2))
    elif args.video and args.workdir:
        result = analyze_video(args.video, args.workdir,
                               lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
        print(json.dumps(result['metadata'], ensure_ascii=False, indent=2))
    else:
        parser.error('--video and --workdir are required')
