"""Same-frame detector concurrency around the immutable capture implementation.

capture_parallel(source, directory, progress=None) -> (Observation, face array).
Each VIDEO detector has one persistent worker. All three calls finish before
the next frame starts. Assignment and raw writes stay on the original thread.
The original _capture/extract bytecode is reused with private namespace proxies;
no original module attributes or files are patched.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
import builtins
import hashlib
import importlib.util
import json
import sys
import threading
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
ROOT = LAB.parents[1]
OPS = HERE.parent
SEATED = OPS / 'word100_seated_web_framing_20261102/adapter.py'
RAW = OPS / 'word100_hand_assignment_20261016/capture_raw.py'
ASSIGN = OPS / 'word100_hand_assignment_joint_20261016/assignment.py'
PINS = {
    'research/front_keyphase_v1/ops/word100_seated_web_framing_20261102/adapter.py': '913e58e8711f166c3e542e4d18fb97feb6301edfc0e6666510d0c95cc4a749df',
    'research/front_keyphase_v1/ops/word100_hand_assignment_20261016/capture_raw.py': '16179df03a1d17cd114689a39bc9376c1f8b42b2c5731097c03e809413adc3af',
    'research/front_keyphase_v1/ops/word100_hand_assignment_joint_20261016/assignment.py': '400491fa1a360d30e5d85c0cc214ada9258055da2bb2c203a90739696884a5c6',
    'research/front_keyphase_v1/src/wis_lab/capture/landmarker.py': '138f12c6cbe12c94e07abf1751409f1411e1c98b812c4d6cea5c33df9c9158e8',
    'research/front_keyphase_v1/src/wis_lab/capture/face_landmarks.py': 'e8a43ad4111d717f36ac1cc67573a53b5b0a4161eec6d5b2d15fa63fd43dad77',
    'research/front_keyphase_v1/src/wis_lab/capture/video_source.py': '8217065015b7f01355bb782d6919296b3eb3907bf46921ca312dd1d4822a3755',
    'research/front_keyphase_v1/src/wis_lab/core/observation.py': '88adc956aa2b9e2b34817ecbafe3fc7fe9ce70cd464d15804edd9cd5b89defff',
    'research/front_keyphase_v1/config/contract.json': 'bf64d3f9549f4ac53c41e899d90fbe08ca9eda1d2df2c857cd9472b666119e7e',
    'Tools/models/hand_landmarker.task': 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1',
    'Tools/models/pose_landmarker_lite.task': '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a',
    'Tools/models/face_landmarker.task': '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


class Timings:
    def __init__(self):
        self.totals = {}; self.calls = {}; self.lock = threading.Lock()

    def add(self, name, seconds):
        with self.lock:
            self.totals[name] = self.totals.get(name, 0.) + float(seconds)
            self.calls[name] = self.calls.get(name, 0) + 1

    def call(self, name, function, *args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self.add(name, time.perf_counter() - start)


class FrameScheduler:
    """Three owned single-worker queues; original caller consumes pose/hand/face."""
    def __init__(self, pose, hand, face, timings=None):
        self.engines = dict(pose=pose, hand=hand, face=face)
        self.timings = timings or Timings()
        self.pools = {name: ThreadPoolExecutor(max_workers=1, thread_name_prefix='AL_' + name)
                      for name in ['pose', 'hand', 'face']}
        self.active = None; self.last_ms = -1; self.last_row = -1
        self.failed = False; self.closed = False; self.frames = []

    def begin(self, rgb, row):
        require(not self.closed and not self.failed, 'Detector scheduler closed or previously failed')
        require(self.active is None and row == self.last_row + 1, 'Previous frame barrier incomplete or row repeated')
        self.active = dict(rgb=rgb, row=int(row), consumed=[], futures={}, tasks={}, image=None, ms=None)

    def _call(self, name, image, ms, state):
        start = time.perf_counter(); error = None
        try:
            if name == 'face':
                return self.engines[name].process(state['rgb'], ms * 1000)
            return self.engines[name].detect_for_video(image, ms)
        except BaseException as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            end = time.perf_counter()
            state['tasks'][name] = dict(start_seconds=start, end_seconds=end,
                duration_seconds=end - start, thread_id=threading.get_ident(), error=error)
            self.timings.add(name + ('_process_seconds' if name == 'face' else '_detect_seconds'), end - start)

    def detector(self, name, image, ms):
        require(name in ['pose', 'hand'] and self.active is not None, 'No active detector frame')
        state = self.active
        if name == 'pose':
            require(not state['futures'] and state['ms'] is None, 'Pose called twice on one frame')
            require(type(ms) is int and ms > self.last_ms, 'VIDEO detector milliseconds must strictly increase')
            state['image'] = image; state['ms'] = ms
            try:
                for target in ['pose', 'hand', 'face']:
                    state['futures'][target] = self.pools[target].submit(self._call, target, image, ms, state)
            except BaseException:
                self.abort()
                raise
        else:
            require(state['image'] is image and state['ms'] == ms and state['consumed'] == ['pose'], 'Hand input/order differs from original pose call')
        return self._result(name)

    def face(self, rgb, timestamp_us):
        require(self.active is not None, 'No active face frame')
        state = self.active
        require(rgb is state['rgb'] and timestamp_us == state['ms'] * 1000
                and state['consumed'] == ['pose', 'hand'], 'Face RGB/timestamp/order differs from original capture')
        value = self._result('face')
        wait(list(state['futures'].values()))
        require(all(f.done() for f in state['futures'].values()), 'Frame barrier not joined')
        self.last_row = state['row']; self.last_ms = state['ms']
        self.frames.append(dict(row=state['row'], detector_timestamp_ms=state['ms'],
            rgb_shape=list(state['rgb'].shape), calls={k: 1 for k in state['futures']},
            tasks=state['tasks'], consumed=list(state['consumed']), frame_barrier_complete=True))
        self.active = None
        return value

    def _result(self, name):
        state = self.active
        require(name not in state['consumed'] and name in state['futures'], 'Duplicate or unscheduled result request')
        try:
            result = state['futures'][name].result()
            state['consumed'].append(name)
            return result
        except BaseException:
            self.abort()
            raise

    def abort(self):
        # Native calls cannot safely be cancelled. Join them before caller closes
        # models. The frame is never retried and another frame cannot enter.
        self.failed = True
        if self.active is not None:
            wait(list(self.active['futures'].values()))

    def close(self):
        if self.closed:
            return
        if self.active is not None:
            wait(list(self.active['futures'].values()))
        for pool in self.pools.values():
            pool.shutdown(wait=True, cancel_futures=False)
        self.closed = True


class DetectorProxy:
    def __init__(self, name, engine, scheduler):
        self.name = name; self.engine = engine; self.scheduler = scheduler; self.closed = False
    def __getattr__(self, name):
        return getattr(self.engine, name)
    def detect_for_video(self, image, ms):
        return self.scheduler.detector(self.name, image, ms)
    def close(self):
        if not self.closed:
            require(self.scheduler.closed, 'Never close an engine before worker join')
            self.closed = True
            return self.engine.close()


class FaceProxy:
    def __init__(self, face, scheduler):
        self.real = face; self.scheduler = scheduler; self.closed = False
    def __getattr__(self, name):
        return getattr(self.real, name)
    def process(self, rgb, timestamp_us):
        return self.scheduler.face(rgb, timestamp_us)
    def close(self):
        if not self.closed:
            require(self.scheduler.closed, 'Never close face before worker join')
            self.closed = True
            return self.real.close()


def parallel_raw_class(original, timings, owners):
    class ParallelRaw(original):
        def __init__(self, contract):
            self.parallel_scheduler = None; self.parallel_closed = False
            try:
                timings.call('pose_hand_construction_seconds', super().__init__, contract)
            except BaseException:
                # The unchanged Landmarker constructor may create hand before
                # pose construction fails; explicitly release that partial pair.
                for name in ['hand', 'pose']:
                    engine = getattr(self, name, None)
                    if engine is not None:
                        try: engine.close()
                        except Exception: pass
                raise
            owners.append(self)

        def process(self, rgb, timestamp_us, out, row):
            if self.parallel_scheduler is None:
                require(hasattr(self, 'face'), 'Original Capture must create its separate face model')
                self.parallel_scheduler = FrameScheduler(self.pose, self.hand, self.face, timings)
                self.pose = DetectorProxy('pose', self.pose, self.parallel_scheduler)
                self.hand = DetectorProxy('hand', self.hand, self.parallel_scheduler)
                self.face = FaceProxy(self.face, self.parallel_scheduler)
            self.parallel_scheduler.begin(rgb, row)
            try:
                return timings.call('raw_process_with_pose_hand_wait_seconds', super().process, rgb, timestamp_us, out, row)
            except BaseException:
                self.parallel_scheduler.abort()
                raise

        def _assign_sides(self, *args, **kwargs):
            return timings.call('baseline_assignment_and_raw_record_seconds', super()._assign_sides, *args, **kwargs)

        def close(self):
            if self.parallel_closed:
                return
            self.parallel_closed = True
            if self.parallel_scheduler is not None:
                self.parallel_scheduler.close()
            timings.call('pose_hand_close_seconds', super().close)
    return ParallelRaw


def cloned_function(function, replacements):
    namespace = dict(function.__globals__); namespace.update(replacements)
    clone = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    require(clone.__code__ is function.__code__, 'Original body bytecode must be reused')
    return clone


def module_view(original, **replacements):
    view = types.ModuleType(original.__name__ + '_AL_private_view')
    view.__dict__.update(original.__dict__); view.__dict__.update(replacements)
    return view


def timed_iterator(iterable, metric, timings, wrap=lambda x: x):
    iterator = iter(iterable)
    while True:
        start = time.perf_counter()
        try:
            item = next(iterator)
        except StopIteration:
            timings.add(metric, time.perf_counter() - start)
            return
        timings.add(metric, time.perf_counter() - start)
        yield wrap(item)


class FrameView:
    def __init__(self, frame, timings): self.real = frame; self.timings = timings
    def __getattr__(self, name): return getattr(self.real, name)
    def to_ndarray(self, *args, **kwargs):
        return self.timings.call('RGB_ndarray_conversion_seconds', self.real.to_ndarray, *args, **kwargs)


class PacketView:
    def __init__(self, packet, timings): self.real = packet; self.timings = timings
    def __getattr__(self, name): return getattr(self.real, name)
    def decode(self, *args, **kwargs):
        frames = self.timings.call('native_decode_call_seconds', self.real.decode, *args, **kwargs)
        return timed_iterator(frames, 'native_decode_iteration_seconds', self.timings,
                              lambda frame: FrameView(frame, self.timings))


class ContainerView:
    def __init__(self, container, timings): self.real = container; self.timings = timings
    def __getattr__(self, name): return getattr(self.real, name)
    def __enter__(self): self.real.__enter__(); return self
    def __exit__(self, *args): return self.timings.call('container_close_seconds', self.real.__exit__, *args)
    def demux(self, *args, **kwargs):
        packets = self.real.demux(*args, **kwargs)
        return timed_iterator(packets, 'demux_iteration_seconds', self.timings,
                              lambda packet: PacketView(packet, self.timings))


def build_capture(seated, cap, assignment, video_source, face_module, observation, timings, owners):
    """Build private import/views, leaving every original module untouched."""
    ParallelRaw = parallel_raw_class(cap.RawCaptureLandmarker, timings, owners)
    class TimedFace(face_module.FaceLandmarks):
        def __init__(self, *args, **kwargs):
            timings.call('face_construction_seconds', super().__init__, *args, **kwargs)
            engine = self.engine
            class FaceEngine:
                def detect_for_video(_self, *a, **kw):
                    return timings.call('face_detect_seconds', engine.detect_for_video, *a, **kw)
                def close(_self):
                    return timings.call('face_close_seconds', engine.close)
            self.engine = FaceEngine()
    class TimedObservation(observation.Observation):
        def save(self, *args, **kwargs):
            return timings.call('observation_save_seconds', super().save, *args, **kwargs)
    class TimedAssignment(assignment.HandAssigner):
        def assign(self, *args, **kwargs):
            return timings.call('joint_v3_assignment_seconds', super().assign, *args, **kwargs)
    class TimedLM:
        def __init__(self, original): self.real = original
        def __getattr__(self, name): return getattr(self.real, name)
        def process(self, *args, **kwargs):
            return timings.call('complete_frame_process_seconds', self.real.process, *args, **kwargs)
    av_view = module_view(video_source.av,
        open=lambda *a, **kw: ContainerView(timings.call('container_open_seconds', video_source.av.open, *a, **kw), timings))
    extract_impl = cloned_function(video_source.extract, dict(av=av_view, Observation=TimedObservation,
        probe=lambda *a, **kw: timings.call('original_probe_decode_seconds', video_source.probe, *a, **kw)))
    def extract(path, lm, clip_id):
        return timings.call('extraction_wall_seconds', extract_impl, path, TimedLM(lm), clip_id)
    imports = {
        'wis_lab.capture.video_source': module_view(video_source, extract=extract),
        'wis_lab.capture.face_landmarks': module_view(face_module, FaceLandmarks=TimedFace),
        'wis_lab.core.observation': module_view(observation, Observation=TimedObservation),
    }
    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in imports and fromlist:
            return imports[name]
        return builtins.__import__(name, globals, locals, fromlist, level)
    raw_view = module_view(cap, RawCaptureLandmarker=ParallelRaw)
    assignment_view = module_view(assignment, HandAssigner=TimedAssignment)
    def private_module(name, path):
        resolved = Path(path).resolve()
        if resolved == RAW.resolve(): return raw_view
        if resolved == ASSIGN.resolve(): return assignment_view
        raise ValueError('Unexpected module requested by pinned capture body: ' + str(path))
    private_builtins = dict(vars(builtins)); private_builtins['__import__'] = private_import
    return cloned_function(seated._capture, dict(__builtins__=private_builtins, _module=private_module,
        _bounded_video_preflight=lambda *a, **kw: timings.call('original_preflight_decode_seconds', seated._bounded_video_preflight, *a, **kw),
        _save=lambda *a, **kw: timings.call('other_array_save_seconds', seated._save, *a, **kw),
        _write=lambda *a, **kw: timings.call('assignment_trace_save_seconds', seated._write, *a, **kw)))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def capture_parallel(source, directory, progress=None):
    """Only called by the reviewed AL runner; not executed by synthetic tests."""
    directory = Path(directory).resolve(); source = Path(source).resolve()
    require(directory.is_relative_to(HERE.resolve()) and directory != HERE.resolve() and directory.is_dir(), 'Candidate output must be an existing isolated AL subdirectory')
    names = ['baseline.npz', 'observations.npz', 'raw.npz', 'assignment.npz', 'assignment_trace.json', 'face.npz', 'CAPTURE_STARTED.json', 'CAPTURE_TIMING.json', 'CAPTURE_FIRST_FAILURE.txt']
    require(not any((directory / name).exists() for name in names), 'Capture already attempted; do not overwrite or rerun')
    for rel, digest in PINS.items():
        require(sha(ROOT / rel) == digest, 'Frozen source/model/contract changed: ' + rel)
    source_digest = sha(source)
    write_new(directory / 'CAPTURE_STARTED.json', dict(source=str(source), source_sha256=source_digest,
        implementation_sha256=sha(__file__), at=time.time(), model_scheduling='one persistent worker per detector; same-frame barrier',
        new_training=0, new_classifier_inferences=0))
    timings = Timings(); owners = []; old_path = sys.path.copy(); old_bytecode = sys.dont_write_bytecode
    started = time.perf_counter(); status = 'FAILED'; error = None
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(LAB / 'src'))
        from wis_lab.capture import video_source, face_landmarks
        from wis_lab.core import observation
        seated = load_module('_al_original_seated', SEATED)
        cap = load_module('_al_original_raw', RAW)
        assignment = load_module('_al_original_joint', ASSIGN)
        capture = build_capture(seated, cap, assignment, video_source, face_landmarks, observation, timings, owners)
        ob, face = capture(source, directory, progress)
        require(sha(source) == source_digest, 'Source changed during capture')
        for rel, digest in PINS.items(): require(sha(ROOT / rel) == digest, 'Frozen resource changed during capture')
        status = 'CAPTURE_COMPLETE_NOT_YET_BASELINE_VERIFIED'
        return ob, face
    except BaseException:
        error = traceback.format_exc()
        with (directory / 'CAPTURE_FIRST_FAILURE.txt').open('x', encoding='utf-8') as f: f.write(error)
        raise
    finally:
        # Original _capture closes normally; this also covers construction and
        # processing failures before its local lm/finally block is established.
        for owner in owners:
            try:
                if not owner.parallel_closed:
                    owner.close()
            except Exception:
                if error is None: error = traceback.format_exc()
        sys.path[:] = old_path; sys.dont_write_bytecode = old_bytecode
        frames = [r for owner in owners if owner.parallel_scheduler is not None for r in owner.parallel_scheduler.frames]
        write_new(directory / 'CAPTURE_TIMING.json', dict(status=status, wall_seconds=time.perf_counter() - started,
            totals=timings.totals, calls=timings.calls, frames=frames,
            schedulers_closed=all(o.parallel_scheduler is None or o.parallel_scheduler.closed for o in owners),
            cleanup_error=error, original_point_layout=75, face_points=478,
            original_capture_and_extract_bytecode_reused=True, original_assignment_unchanged=True,
            original_modules_patched=False, frames_parallel=False, detectors_same_frame_parallel=True,
            timing_scope='Detector durations overlap and must not be summed as wall latency. Probe/preflight decode retained. Native decode and RGB conversion are measured separately; frame processing includes conversion to MediaPipe images and original assembly.',
            source_sha256=source_digest, pins={str(ROOT / k): v for k, v in PINS.items()},
            performance_improvement_claimed=False, exact_baseline_equality_verified=False))
