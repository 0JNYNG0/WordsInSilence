"""Fake detector/concurrency tests only: no MediaPipe, models or real video."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import threading
import types
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_al_synthetic_capture', HERE / 'parallel_capture.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class State:
    def __init__(self, failure=None, hold=False):
        self.barrier = threading.Barrier(3, timeout=3)
        self.lock = threading.Lock(); self.active = {}; self.calls = []; self.closes = []
        self.threads = {}; self.failure = failure; self.max_active = 0
        self.hold = hold; self.entered = threading.Event(); self.release = threading.Event()
    def run(self, name, image, rgb, ms):
        tid = threading.get_ident()
        with self.lock:
            assert name not in self.active, 'Same VIDEO model processed two frames'
            assert name not in self.threads or self.threads[name] == tid, 'Engine thread owner changed'
            self.threads[name] = tid; self.active[name] = ms
            self.max_active = max(self.max_active, len(self.active))
            self.calls.append((name, ms, id(rgb), id(image) if image is not None else None, tid))
        try:
            self.barrier.wait()
            if self.failure == name:
                self.entered.set()
                raise RuntimeError('fake failure: ' + name)
            if self.hold:
                assert self.release.wait(3), 'Test release timeout'
            return dict(name=name, ms=ms, pixels=int(rgb.sum()), missing=bool(ms % 2))
        finally:
            with self.lock:
                self.active.pop(name)
    def close(self, name):
        with self.lock:
            assert not self.active, 'Closed a detector while another native call remained'
            assert name not in self.closes, 'Detector closed more than once'
            self.closes.append(name)


class FakeImage:
    def __init__(self, rgb): self.rgb = rgb


class FakeDetector:
    def __init__(self, state, name): self.state = state; self.name = name
    def detect_for_video(self, image, ms): return self.state.run(self.name, image, image.rgb, ms)
    def close(self): self.state.close(self.name)


class FakeFace:
    def __init__(self, state): self.state = state; self.last_ms = -1
    def process(self, rgb, timestamp_us):
        ms = timestamp_us // 1000
        assert ms > self.last_ms
        self.last_ms = ms
        return self.state.run('face', None, rgb, ms)
    def close(self): self.state.close('face')


def make_capture(state, timings, owners):
    class FakeRaw:
        def __init__(self, contract):
            self.contract = contract; self.pose = FakeDetector(state, 'pose')
            self.hand = FakeDetector(state, 'hand'); self._last_ms = -1
            self.raw_rows = []; self.assignment_threads = []
        def process(self, rgb, timestamp_us, out, row):
            # Exact relevant control order of original Landmarker+RawCapture:
            # advance clock once, shared image, pose,hand, assignment,raw append.
            ms = max(int(timestamp_us // 1000), self._last_ms + 1); self._last_ms = ms
            image = FakeImage(rgb)
            pose = self.pose.detect_for_video(image, ms)
            hand = self.hand.detect_for_video(image, ms)
            value = self._assign_sides(pose, hand)
            out[row] = dict(value=value, source_timestamp_us=timestamp_us, row=row)
            self.raw_rows.append(out[row])
        def _assign_sides(self, pose, hand):
            self.assignment_threads.append(threading.get_ident())
            return dict(pose=pose, hand=hand)
        def close(self):
            self.hand.close(); self.pose.close()
    Parallel = m.parallel_raw_class(FakeRaw, timings, owners)
    class Capture(Parallel):
        def __init__(self):
            super().__init__({'same_options': True})
            self.face = FakeFace(state); self.faces = []
        def process(self, rgb, us, out, row):
            super().process(rgb, us, out, row)
            self.faces.append(self.face.process(rgb, self._last_ms * 1000))
        def close(self):
            super().close()
            self.face.close()
    return Capture()


def main():
    checks = []
    def check(name, value):
        assert bool(value), name
        checks.append(name)
    def rejects(name, call):
        try: call()
        except ValueError: checks.append(name)
        else: raise AssertionError('Expected rejection: ' + name)
    imported_before = set(sys.modules)
    state = State(); timings = m.Timings(); owners = []
    cap = make_capture(state, timings, owners); output = {}; caller = threading.get_ident()
    images = [np.full((6, 7, 3), i + 1, dtype=np.uint8) for i in range(3)]
    timestamps = [0, 400, 33333]  # collapse two original us onto0ms -> bumped1ms
    try:
        for i, (rgb, us) in enumerate(zip(images, timestamps)):
            cap.process(rgb, us, output, i)
            check('frame' + str(i) + ' all detector tasks joined', cap.parallel_scheduler.active is None and not state.active)
            expected_ms = [0, 1, 33][i]
            check('frame' + str(i) + ' source us and adjusted ms retained', output[i]['source_timestamp_us'] == us and cap.faces[i]['ms'] == expected_ms)
            check('frame' + str(i) + ' exact returned values including missing marker', output[i]['value'] == dict(
                pose=dict(name='pose', ms=expected_ms, pixels=int(rgb.sum()), missing=bool(expected_ms % 2)),
                hand=dict(name='hand', ms=expected_ms, pixels=int(rgb.sum()), missing=bool(expected_ms % 2))))
        check('three branches actually concurrent', state.max_active == 3)
        check('exactly one call each perframe', len(state.calls) == 9 and all(sum(c[0] == name for c in state.calls) == 3 for name in ['pose', 'hand', 'face']))
        check('same native pixels each branch', all(c[2] == id(images[[0, 1, 33].index(c[1])]) for c in state.calls))
        check('pose hand share exact Image object', all(next(c for c in state.calls if c[0] == 'pose' and c[1] == ms)[3] == next(c for c in state.calls if c[0] == 'hand' and c[1] == ms)[3] for ms in [0, 1, 33]))
        check('one stable worker per model', len(set(state.threads.values())) == 3)
        check('raw assignment on original caller only', cap.assignment_threads == [caller] * 3 and all(t != caller for t in state.threads.values()))
        check('all native raw rows preserved', cap.raw_rows == [output[i] for i in range(3)])
        check('frame log barriers and original order', all(f['consumed'] == ['pose', 'hand', 'face'] and f['frame_barrier_complete'] for f in cap.parallel_scheduler.frames))
        check('recorded detector task timing', all(t['duration_seconds'] >= 0 and t['error'] is None for f in cap.parallel_scheduler.frames for t in f['tasks'].values()))
        rejects('repeated frame rejected', lambda: cap.parallel_scheduler.begin(images[0], 2))
    finally:
        cap.close()
    check('all owned workers joined and engines closed', cap.parallel_scheduler.closed and sorted(state.closes) == ['face', 'hand', 'pose'])
    cap.close()
    check('close idempotent after normal frames', len(state.closes) == 3)
    rejects('closed scheduler cannot accept frame', lambda: cap.parallel_scheduler.begin(images[0], 3))

    # Check each failure position, including early pose failure while two other
    # futures remain active. The exception must not escape before their join.
    for failure in ['pose', 'hand', 'face']:
        state = State(failure=failure, hold=True); owners = []
        cap = make_capture(state, m.Timings(), owners)
        done = threading.Event(); errors = []
        def run():
            try:
                cap.process(images[0], 0, {}, 0)
            except BaseException as exc:
                errors.append(str(exc))
            finally:
                cap.close(); done.set()
        thread = threading.Thread(target=run, name='AL_fake_original_caller')
        thread.start()
        try:
            check(failure + ' failure reached fake detector', state.entered.wait(3))
            check(failure + ' exception waits outstanding native calls', not done.is_set())
            state.release.set(); thread.join(3)
            check(failure + ' all futures drained before close', done.is_set() and not state.active and len(state.closes) == 3)
            check(failure + ' original exception preserved', errors == ['fake failure: ' + failure])
            check(failure + ' failure poisons frame scheduler', cap.parallel_scheduler.failed and cap.parallel_scheduler.closed)
        finally:
            state.release.set(); thread.join(3)

    # Active frame identity and clock guards without scheduling native work.
    state = State(); sched = m.FrameScheduler(FakeDetector(state, 'pose'), FakeDetector(state, 'hand'), FakeFace(state))
    try:
        sched.begin(images[0], 0)
        rejects('no second frame before barrier', lambda: sched.begin(images[1], 1))
        rejects('hand cannot run before pose', lambda: sched.detector('hand', FakeImage(images[0]), 0))
    finally:
        sched.close()
    check('empty or invalid submission cleans all pools', sched.closed)

    # Decode/conversion proxies forward actual values; they do not resize,
    # suppress, reorder, or synthesize source frames.
    class Frame:
        def __init__(self, i): self.pts = i; self.rgb = images[i]
        def to_ndarray(self, format):
            assert format == 'rgb24'
            return self.rgb
    class Packet:
        def decode(self): return [Frame(0), Frame(1), Frame(2)]
    t = m.Timings(); frames = list(m.PacketView(Packet(), t).decode())
    decoded = [f.to_ndarray(format='rgb24') for f in frames]
    check('decode proxy preserves ordered PTS and same arrays', [f.pts for f in frames] == [0, 1, 2] and all(x is y for x, y in zip(decoded, images)))
    check('decode and conversion measured separately', t.calls['native_decode_call_seconds'] == 1 and t.calls['RGB_ndarray_conversion_seconds'] == 3)
    namespace = {'VALUE': 'original'}
    exec('def sample():\n return VALUE\n', namespace)
    original = namespace['sample']; clone = m.cloned_function(original, {'VALUE': 'private'})
    check('function bytecode reused without altering original globals', clone.__code__ is original.__code__ and clone() == 'private' and original() == 'original')
    module = types.ModuleType('fake_original'); module.kept = object(); module.changed = 'old'
    view = m.module_view(module, changed='new')
    check('private module view does not monkeypatch original', module.changed == 'old' and view.changed == 'new' and view.kept is module.kept)
    check('no native detector/video imports', not any(n.startswith(('mediapipe', 'av', 'wis_lab')) for n in set(sys.modules) - imported_before))
    report = dict(status='PASS_FAKE_FRAME_PARALLEL_CONTRACT', checks_count=len(checks), checks=checks,
        actual_model_loads=0, actual_detector_calls=0, source_video_reads=0, synthetic_detector_calls=18,
        original_capture_reruns=0, performance_improvement_claimed=False,
        pins={str(HERE / n): hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ['parallel_capture.py', 'check_parallel_capture.py']})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
