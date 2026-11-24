"""Isolated new-input pipeline: shared observation, one prototype DTW, local events."""
from pathlib import Path
import sys, importlib.util, json, hashlib, time, subprocess, threading, queue, uuid
from functools import wraps
import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
X = O / 'word100_learned_component_extent_20261102'
PY = O / 'word100_signrep_feasibility_20261029/runtime/Scripts/python.exe'
sys.path.insert(0, str(X))
spec = importlib.util.spec_from_file_location('_local_event_existing_source', X/'learner_adapter.py')
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)
WORDS = source.WORDS
ANALYZE_LOCK = threading.RLock()

def serialized(function):
    @wraps(function)
    def call(*args, **kwargs):
        with ANALYZE_LOCK:
            return function(*args, **kwargs)
    return call

def validate_request(video, directory, word, provenance):
    d=Path(directory).resolve()
    parts=d.relative_to(U.resolve()).parts
    assert len(parts)==3 and parts[2]=='analysis'
    assert len(parts[1])==32 and all(c in '0123456789abcdef' for c in parts[1])
    roles={'public_runtime_runs':'public_runtime_inference','private_sessions':'user_inference_only'}
    assert parts[0] in roles and provenance['source_role']==roles[parts[0]]
    assert word in WORDS and word==provenance['word_id']
    assert all(provenance[k] is False for k in ('training','tuning','candidate_selection'))
    if parts[0]=='private_sessions':
        assert Path(video).resolve()==d.parent/'input.video'
    assert not d.exists(), 'Job analysis already exists; inspect the first attempt'
    source_hash=sha(video)
    assert source_hash==provenance['source_sha256']
    return d,source_hash

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

class Worker:
    def __init__(self):
        self.lock = threading.Lock()
        self.messages = queue.Queue()
        self.broken = False
        self.log = (U/'WORKER.stderr.log').open('a', encoding='utf-8')
        self.proc = subprocess.Popen([str(PY), '-B', '-u', str(U/'order_event_worker.py')],
            cwd=str(U), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        threading.Thread(target=self._read, daemon=True).start()
        reply = self._next()
        if reply.get('status') != 'ready':
            self.close()
            raise RuntimeError(str(reply))
        self.ready = reply

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.messages.put(json.loads(line))
        except Exception as exc:
            self.messages.put(dict(status='failed', error=str(exc)))
        finally:
            self.messages.put(dict(status='terminated', exit_code=self.proc.poll()))

    def _next(self):
        try:
            return self.messages.get(timeout=60)
        except queue.Empty as exc:
            self.broken = True
            raise RuntimeError('분석기의 응답을 확인하지 못했습니다. 자동 중복 실행은 하지 않습니다.') from exc

    def infer(self, directory, word):
        with self.lock:
            if self.broken or self.proc.poll() is not None:
                raise RuntimeError('분석기가 준비되지 않았습니다.')
            rid = uuid.uuid4().hex
            job = directory.parent.name
            self.proc.stdin.write(json.dumps(dict(command='infer', id=rid, directory=str(directory),
                word_id=word, job_id=job))+'\n')
            self.proc.stdin.flush()
            reply = self._next()
            if reply.get('id') != rid or reply.get('job_id') != job:
                self.broken = True
                raise RuntimeError('요청과 분석 결과가 일치하지 않습니다.')
            if reply.get('status') != 'complete':
                raise RuntimeError(reply.get('error', '분석 실패'))
            path = directory/'LOCAL_EVENT_FEEDBACK.json'
            assert Path(reply['path']).resolve() == path.resolve()
            assert sha(path) == reply['result_sha256']
            return read(path)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=10)
        self.log.close()

worker = None

@serialized
def warmup():
    global worker
    source.U = U
    start = time.perf_counter()
    # Avoid historical warmup's two unused baseline engine copies. The actual
    # inference engines and exported backbone remain exactly the same.
    shared_capture = source.old.shared_capture
    shared = shared_capture.shared
    compiled, device = shared.compile_rgb()
    compiled([np.zeros((1, 3, 16, 224, 224), np.float32)])
    shared_capture.engine('rgb_geometry')
    source.old.motion()
    out = dict(device=device, warmup_source='one synthetic zero RGB window',
        unused_baseline_engines_loaded=0)
    if worker is None:
        worker = Worker()
    return dict(base=out, worker=worker.ready, total_warmup_seconds=time.perf_counter()-start)

@serialized
def close():
    global worker
    if worker is not None:
        worker.close()
        worker = None

def extract_with_support(directory, shared):
    """Identical extraction, additionally retain actual window lineage."""
    def extract(video, ids, ts, width, height, compiled, progress):
        start = time.perf_counter()
        class Tap:
            def __init__(self):
                self.pre = []
            def __call__(self, inputs):
                out = compiled(inputs)
                self.pre.append(np.array(out[1], np.float32, copy=True).reshape(1,768))
                return {0:np.array(out[0], np.float32, copy=True)}
        tap = Tap()
        rgb, features, starts, centers = shared.base._encode(video, [0,0,width,height], ids, ts, tap, progress)
        pre = np.concatenate(tap.pre)
        prelinear = np.stack([np.interp(ts, centers, pre[:,j]) for j in range(768)], 1).astype(np.float32)
        with (directory/'RGB_NATIVE_SUPPORT.npz').open('xb') as f:
            np.savez_compressed(f, features=features, starts=starts, center_timestamp_us=centers,
                prelinear_windows=pre, frame_id=ids, timestamp_us=ts)
        return rgb, prelinear, time.perf_counter()-start
    return extract

@serialized
def analyze(video, directory, word, condition='learned_motion', progress=None, *, provenance):
    assert worker is not None and condition == 'learned_motion' and word in WORDS
    d,source_hash=validate_request(video,directory,word,provenance)
    start = time.perf_counter()
    source.U = U
    shared = source.old.shared_capture.shared
    original_extract = shared.extract_rgb
    original_capture = shared.base._capture
    from parallel_capture import capture_parallel
    shared.extract_rgb = extract_with_support(d, shared)
    shared.base._capture = capture_parallel
    try:
        result = source.analyze(video, d, word, condition, progress)
    finally:
        shared.extract_rgb = original_extract
        shared.base._capture = original_capture
    a = dict(np.load(d/'analysis.npz', allow_pickle=False))
    assert sha(video) == result['source_sha256'] == source_hash
    contract = dict(provenance, job_id=d.parent.name, input=str(d/'analysis.npz'),
        input_sha256=sha(d/'analysis.npz'), base=str(d/'SOURCE_FEEDBACK.json'),
        base_sha256=sha(d/'SOURCE_FEEDBACK.json'),
        rgb_support_path=str(d/'RGB_NATIVE_SUPPORT.npz'), rgb_support_sha256=sha(d/'RGB_NATIVE_SUPPORT.npz'),
        native_axis_sha256=hashlib.sha256(a['frame_id'].tobytes()+a['timestamp_us'].tobytes()).hexdigest())
    write(d/'RUNTIME_INPUT_CONTRACT.json', contract)
    if progress:
        progress(dict(stage='checking_order_and_local_movement'))
    out = worker.infer(d, word)
    out['local_combined_feedback_latency_seconds'] = time.perf_counter()-start
    out['local_combined_latency_scope'] = 'Server analysis including observation and worker roundtrip; excludes upload and cold model boot.'
    out['inference_privacy'] = dict(source_role=provenance['source_role'],training=False,tuning=False,candidate_selection=False)
    out['capture_execution'] = dict(scheduling='same_frame_pose_hand_face_parallel',
        equivalent_public_sensor_test='word100_capture_parallel_20261103',
        frames_skipped=0,detector_options_changed=False,full_source_extent=True)
    write(d/'LIVE_FEEDBACK.json', out)
    if progress:
        progress(dict(stage='complete'))
    return out
