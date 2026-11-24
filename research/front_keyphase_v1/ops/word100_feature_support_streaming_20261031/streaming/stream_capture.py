"""Isolated single-decode capture. Production decoder/detector/assigner are untouched."""
from __future__ import annotations
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import queue
import sys
import threading
import time

import av
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
UNIT = HERE.parent
OPS = UNIT.parent
LAB = OPS.parent
ROOT = LAB.parent.parent
sys.path.insert(0, str(LAB / 'src'))
from wis_lab.core.observation import Observation, empty_frames, sha256_file
from wis_lab.capture import video_source

ARRAY_NAMES = ['xy_px', 'timestamp_us', 'frame_id', 'observed', 'usable', 'conf', 'hand_score']


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dependencies():
    capture = load_module('streaming_frozen_capture', OPS / 'word100_hand_assignment_20261016/capture_raw.py')
    assignment = load_module('streaming_frozen_assignment', OPS / 'word100_hand_assignment_joint_20261016/assignment.py')
    base = load_module('streaming_frozen_associate', OPS / 'word100_fullframe_roi_pair_20261016/run_pair.py')
    return capture, assignment, base


def equal_array(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b, equal_nan=True) if np.issubdtype(a.dtype, np.floating) else np.array_equal(a, b))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def concatenate_rows(rows):
    assert rows
    return {key: np.concatenate([row[key] for row in rows], axis=0) for key in ARRAY_NAMES}


def associate_one(raw, baseline, assigner, contract, frame):
    """Same frozen associate() arithmetic, causal one-frame application."""
    out = empty_frames(1)
    for key in ['xy_px', 'observed', 'usable', 'conf']:
        out[key][0, :33] = baseline[key][0, :33]
    out['frame_id'][0] = baseline['frame_id'][0]
    out['timestamp_us'][0] = baseline['timestamp_us'][0]
    n, w, h = int(raw['raw_count']), int(raw['width']), int(raw['height'])
    chosen, diagnostic = assigner.assign(raw['hand_xy_px'][:n], raw['raw_label'][:n], raw['raw_score'][:n], raw['pose_xy_px'], raw['pose_usable'], int(out['timestamp_us'][0]), w, h)
    diagnostic.update(frame=frame, timestamp_us=int(out['timestamp_us'][0]))
    sides = np.full(2, -1, np.int8)
    margin = contract['usable_rule']['bounds_margin_ratio']
    threshold = contract['usable_rule']['hand_score_threshold']
    for i in range(n):
        side = chosen[i]
        if side is None:
            continue
        sides[i] = side
        sl = slice(33+21*side, 54+21*side)
        points = raw['hand_xy_px'][i].astype(np.float32)
        score = raw['raw_score'][i]
        out['xy_px'][0, sl] = points
        out['observed'][0, sl] = True
        out['hand_score'][0, side] = score
        out['usable'][0, sl] = np.isfinite(points).all(1) & (points[:, 0]>=-margin*w) & (points[:, 0]<=w*(1+margin)) & (points[:, 1]>=-margin*h) & (points[:, 1]<=h*(1+margin)) & (score>=threshold)
    return out, sides, diagnostic


class FrameProcessor:
    """One detector owner; each source frame goes through unchanged model calls once."""
    def __init__(self, landmarker, assigner, contract):
        self.landmarker = landmarker
        self.assigner = assigner
        self.contract = contract
        self.cached_extractor_id = landmarker.extractor_id()
        self.baseline_rows, self.assigned_rows, self.sides, self.diagnostics, self.timing = [], [], [], [], []
        self.epoch_ns = 0
        self.legacy = False

    def __getattr__(self, name):
        return getattr(self.landmarker, name)

    def extractor_id(self):
        return self.cached_extractor_id

    def process(self, rgb, timestamp_us, out, row):
        # The unchanged production extract() calls this synchronous compatibility path.
        self._process(rgb, timestamp_us, out, row, time.perf_counter_ns(), 0, None)

    def process_frame(self, rgb, timestamp_us, frame_id, available_ns, due_ns, source_pts):
        out = empty_frames(1)
        out['timestamp_us'][0] = timestamp_us
        out['frame_id'][0] = frame_id
        self._process(rgb, timestamp_us, out, 0, available_ns, due_ns, source_pts)

    def _process(self, rgb, timestamp_us, out, row, available_ns, due_ns, source_pts):
        started = time.perf_counter_ns()
        self.landmarker.process(rgb, timestamp_us, out, row)
        detector_done = time.perf_counter_ns()
        baseline = {key: out[key][row:row+1].copy() for key in ARRAY_NAMES}
        assigned, sides, diagnostic = associate_one(self.landmarker.raw_rows[-1], baseline, self.assigner, self.contract, len(self.timing))
        assigned_done = time.perf_counter_ns()
        self.baseline_rows.append(baseline)
        self.assigned_rows.append(assigned)
        self.sides.append(sides)
        self.diagnostics.append(diagnostic)
        self.timing.append(dict(frame_id=int(out['frame_id'][row]), timestamp_us=int(timestamp_us), source_pts=None if source_pts is None else int(source_pts), available_ns=int(available_ns), due_ns=int(due_ns), process_start_ns=started, detector_end_ns=detector_done, assignment_end_ns=assigned_done, queue_wait_seconds=(started-available_ns)/1e9, detector_seconds=(detector_done-started)/1e9, assignment_seconds=(assigned_done-detector_done)/1e9))


def decode_once(path):
    """Yield every decoded source frame and its original PTS; never probe or resample."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        tb = stream.time_base or Fraction(1, 1000)
        origin_pts, last_us = None, -1
        for ordinal, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f'Missing PTS at decoded ordinal {ordinal}; no frame may silently disappear')
            if origin_pts is None:
                origin_pts = frame.pts
            timestamp_us = int(round(float((frame.pts-origin_pts)*tb)*1_000_000))
            if timestamp_us <= last_us:
                raise ValueError(f'Nonmonotonic PTS at ordinal {ordinal}; source retained, no frame dropping')
            last_us = timestamp_us
            metadata = dict(width=frame.width, height=frame.height, time_base=f'{tb.numerator}/{tb.denominator}', origin_pts=int(origin_pts))
            yield ordinal, int(frame.pts), timestamp_us, frame.to_ndarray(format='rgb24'), metadata


def feed_single_decode(path, processor, mode, epoch_ns):
    assert mode in ['single_decode_batch', 'single_decode_paced']
    processor.epoch_ns = epoch_ns
    arrivals, metadata = [], None
    max_queue_frames = 0
    if mode == 'single_decode_batch':
        for ordinal, pts, us, rgb, metadata in decode_once(path):
            arrival = time.perf_counter_ns()
            arrivals.append(dict(frame_id=ordinal, source_pts=pts, timestamp_us=us, available_ns=arrival, due_ns=epoch_ns))
            processor.process_frame(rgb, us, ordinal, arrival, epoch_ns, pts)
    else:
        work = queue.Queue()  # Unbounded for this frozen small clip set: never drop/block arrivals.
        errors = []
        def consume():
            try:
                while True:
                    item = work.get()
                    if item is None:
                        break
                    processor.process_frame(*item)
            except BaseException as error:
                errors.append(error)
        consumer = threading.Thread(target=consume, name='single-owner-mediapipe-consumer')
        consumer.start()
        try:
            for ordinal, pts, us, rgb, metadata in decode_once(path):
                if errors:
                    raise errors[0]
                due = epoch_ns + us*1000
                delay = (due-time.perf_counter_ns())/1e9
                if delay > 0:
                    time.sleep(delay)
                arrival = time.perf_counter_ns()
                arrivals.append(dict(frame_id=ordinal, source_pts=pts, timestamp_us=us, available_ns=arrival, due_ns=due))
                work.put((rgb, us, ordinal, arrival, due, pts))
                max_queue_frames = max(max_queue_frames, work.qsize())
                if work.qsize()*rgb.nbytes > 2*1024**3:
                    raise MemoryError('Frozen stream exceeded 2GiB queued RGB budget; fail without dropping frames')
        finally:
            work.put(None)
            consumer.join()
        if errors:
            raise errors[0]
    assert arrivals and len(arrivals)==len(processor.timing)
    return dict(metadata=metadata, arrivals=arrivals, maximum_queued_frames=max_queue_frames, last_frame_available_ns=arrivals[-1]['available_ns'], decode_passes=1)


def make_observations(processor, row, metadata, extractor_id):
    baseline = Observation(**concatenate_rows(processor.baseline_rows))
    baseline.sidecar = dict(schema='wis_observation_2d_v1', origin='clip', clip_id=row['clip_id'], source_path=row['source'].replace('\\','/'), source_sha256=row['source_sha256'], width=metadata['width'], height=metadata['height'], time_base=metadata['time_base'], decoded_frames=baseline.n_frames, dropped_non_monotonic_pts=0, mirrored=False, rotation_deg=0, extractor_id=extractor_id, hand_assignment_policy=processor.contract['hand_assignment']['policy'], handedness_label_disagreements=processor.landmarker.handedness_disagreements, handedness_label_comparisons=processor.landmarker.handedness_comparisons, unassigned_hand_detections=processor.landmarker.unassigned_hands, clock='video PTS * time_base, microseconds from first decoded frame')
    baseline.sidecar['quality']=baseline.quality_summary()
    assigned=Observation(**concatenate_rows(processor.assigned_rows))
    assigned.sidecar=dict(baseline.sidecar, hand_assignment_policy='frozen_joint_v3', assignment_sha256=processor.assigner.identity())
    baseline.validate(); assigned.validate()
    raw={key:np.stack([r[key] for r in processor.landmarker.raw_rows]) for key in processor.landmarker.raw_rows[0]}
    return baseline,assigned,raw,np.stack(processor.sides)


def trace_summary(records, epoch_ns, capture_end_ns, saved_ns, paced):
    last_done=max(r['assignment_end_ns'] for r in records)
    last_us=records[-1]['timestamp_us']
    wait=[r['queue_wait_seconds'] for r in records]
    lateness=[max(0,(r['available_ns']-r['due_ns'])/1e9) for r in records] if paced else []
    return dict(frames=len(records), source_pts_span_seconds=last_us/1e6, entire_wall_seconds=(saved_ns-epoch_ns)/1e9,
                capture_end_seconds=(capture_end_ns-epoch_ns)/1e9 if paced else 0.,
                source_end_definition='Last original frame available at its PTS; no extra terminal exposure period assumed' if paced else 'Entire recorded file already available before batch processing begins',
                after_capture_detector_assignment_tail_seconds=max(0,(last_done-capture_end_ns)/1e9),
                after_capture_saved_tail_seconds=max(0,(saved_ns-capture_end_ns)/1e9),
                processed_frames_at_capture_end=sum(r['assignment_end_ns']<=capture_end_ns for r in records),
                unfinished_frames_at_capture_end=sum(r['assignment_end_ns']>capture_end_ns for r in records),
                queue_wait_mean_seconds=float(np.mean(wait)), queue_wait_p95_seconds=float(np.quantile(wait,.95)), queue_wait_max_seconds=max(wait),
                arrival_due_lateness_mean_seconds=float(np.mean(lateness)) if lateness else None,
                arrival_due_lateness_max_seconds=max(lateness) if lateness else None,
                detector_service_seconds=sum(r['detector_seconds'] for r in records),
                assignment_service_seconds=sum(r['assignment_seconds'] for r in records))
