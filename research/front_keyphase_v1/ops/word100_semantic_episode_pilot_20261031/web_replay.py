"""Isolated four-public-clip replay; explicit freeze/run, no camera or deployment."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import ast
import copy
import hashlib
import importlib.util
import json
import re
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
OPS = U.parent
STREAM = OPS / 'word100_feature_support_streaming_20261031/streaming'
ALIASES = ['P036', 'P048', 'P066', 'P072']
RUNS = U / 'web_runs'
MODE = 'during_recording'
SCOPE = '새 방향 구간 관측기와 참고 3개를 연결한 연구 비교입니다. 전체 수어의 정답·오답 판정이 아닙니다.'
CAUTION = '관측이 부족한 구간은 비교 근거 부족입니다. ROI 밖의 정상적인 손 내림도 동작 실패라는 뜻이 아닙니다. 시간 정렬과 차이 수치만으로 올바른 수어를 판정하지 않습니다.'
TIMING_SCOPE = '측정 범위: 공개 파일을 원본 PTS로 재생하며 관절 처리 → 방향 구간 → 참고 3개 비교 → 웹 표시. 손 모양 head·SignRep·TCN을 포함한 전체 게임 총지연이 아닙니다.'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def first_failure(p, error):
    if not Path(p).exists():
        write_new(p, {'type': type(error).__name__, 'error': str(error),
                      'traceback': traceback.format_exc(), 'at_ns': time.perf_counter_ns()})


def load_module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def freeze():
    """Root runs this after protocol review and seed42/fit-auto artifacts exist."""
    assert not (U / 'WEB_FROZEN.json').exists(), 'Web protocol already frozen'
    assert not RUNS.exists(), 'Existing partial/completed runs are immutable'
    auth = read(U / 'AUTHORIZATION.json')
    rows = auth['rows']
    dev = [next(r for r in rows if r['alias'] == a) for a in ALIASES]
    assert all(r['role'] == 'development' for r in dev)
    refs = [r for r in rows if r['role'] == 'fit']
    assert len(refs) == 12
    for q in dev:
        assert len([r for r in refs if r['word_id'] == q['word_id']]) == 3
    scientific = read(U / 'FROZEN.json')
    # Hash-only preservation of approved operational assets; no scores are read.
    protected = dict(scientific['protected'])
    for p, h in protected.items():
        assert sha(p) == h, p
    # Old frozen physical/shape heads are preserved, but are not executed here.
    old_heads = OPS / 'word100_stage_feedback_cycle_20261031/runs'
    for p in old_heads.glob('*/head.pt'):
        protected[str(p.resolve())] = sha(p)
    files = [U / n for n in ['AUTHORIZATION.json', 'FROZEN.json', 'episode.py',
                            'web_replay.py', 'web_index.html', 'web_browser_test.py']]
    files += [U / 'runs/42/head.pt', STREAM / 'stream_capture.py',
              OPS.parent / 'config/contract.json',
              OPS / 'word100_hand_assignment_20261016/capture_raw.py',
              OPS / 'word100_hand_assignment_joint_20261016/assignment.py',
              OPS / 'word100_fullframe_roi_pair_20261016/run_pair.py',
              OPS / 'word100_guide_video_feedback_20261028/motion_observer.py']
    prior = read(STREAM / 'FROZEN.json')
    # Detector assets are pinned without reading prior benchmark/accuracy outputs.
    detector_dependencies = [Path(p) for p in prior['pins']
        if Path(p).suffix.lower() in ('.task', '.tflite') or '/src/wis_lab/' in Path(p).as_posix()]
    for p in detector_dependencies:
        assert sha(p) == prior['pins'][str(p)], p
    files += detector_dependencies
    for r in dev + refs:
        for name in ['source', 'input', 'record']:
            assert sha(r[name]) == r[name + '_sha256'], r[name]
        files += [Path(r[k]) for k in ['source', 'input', 'record', 'observation']]
    files += [U / 'runs/42/auto' / (r['alias'] + '.npz') for r in refs]
    pins = {str(p.resolve()): sha(p) for p in files}
    result = dict(status='FROZEN_BEFORE_FOUR_WEB_REPLAYS', aliases=ALIASES,
        mode=MODE, repetitions=1, seed=42, rows=dev, references=refs,
        selection='Fixed seed42, fixed public development aliases, no UI candidate selection or quality tuning',
        route='Single decode original PTS -> CPU full-frame MediaPipe -> frozen joint assignment -> frozen ROI -> physical episode observer -> 3 whole references -> HTTP/browser',
        timing='Warm application/reference cache; fresh detector per clip before capture epoch. End is last original source frame arrival. Two-RAF and media-loaded acknowledgements are separate conservative server receipt bounds, not physical display latency.',
        timing_scope=TIMING_SCOPE, scope=SCOPE,
        protected=protected, pins=pins,
        inference_contract='Calls episode.load_model(42), automatic, runs and compare. Reference x/mask and saved seed42 auto pred only. No old shape, SignRep or TCN inference in this replay.',
        forbidden=['camera', 'personal input', 'production port8765 changes', 'uploads', 'training', 'old12 benchmark rerun', 'whole-sign correctness verdict'])
    write_new(U / 'WEB_FROZEN.json', result)
    return result


def byte_range(header, size):
    if not header:
        return 0, size - 1, False
    match = re.fullmatch(r'bytes=(\d+)-(\d*)', header)
    if not match:
        raise ValueError('Unsupported byte range')
    start = int(match[1])
    end = min(int(match[2]) if match[2] else size - 1, size - 1)
    if start > end or start >= size:
        raise ValueError('Invalid byte range')
    return start, end, True


class Engine:
    def __init__(self):
        started = time.perf_counter()
        self.frozen = read(U / 'WEB_FROZEN.json')
        self.check_pins()
        import numpy as np
        import torch
        self.np = np
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        self.ep = load_module('semantic_episode_web_observer', U / 'episode.py')
        self.sc = load_module('semantic_episode_web_capture', STREAM / 'stream_capture.py')
        self.model = self.ep.load_model(42)
        self.rows = {r['alias']: r for r in self.frozen['rows']}
        self.refrows = {r['alias']: r for r in self.frozen['references']}
        self.refs = {}
        self.refstates = {}
        for alias, r in self.refrows.items():
            with np.load(r['input'], allow_pickle=False) as z:
                self.refs[alias] = {k: np.array(z[k]) for k in ['x', 'mask', 'frame_id', 'timestamp_us']}
            with np.load(U / 'runs/42/auto' / (alias + '.npz'), allow_pickle=False) as z:
                self.refstates[alias] = np.array(z['pred'])
            assert len(self.refstates[alias]) == len(self.refs[alias]['frame_id'])
        self.contract = read(self.sc.LAB / 'config/contract.json')
        self.capture, self.assignment, _ = self.sc.dependencies()
        self.jobs = {}
        self.lock = threading.Lock()
        self.active = None
        self.worker = None
        self.setup_seconds = time.perf_counter() - started

    def check_pins(self):
        for group in ['pins', 'protected']:
            for p, h in self.frozen[group].items():
                assert sha(p) == h, p

    def start(self, alias):
        if alias not in ALIASES:
            raise ValueError('Alias is not one of the four fixed public development clips')
        key = alias + '_1_during_recording'
        with self.lock:
            if self.active is not None:
                raise ValueError('One replay is already active or waiting for browser acknowledgement')
            d = RUNS / key
            if key in self.jobs or d.exists():
                raise ValueError('Partial/completed replay is immutable; no silent retry')
            d.mkdir(parents=True, exist_ok=False)
            job = dict(key=key, alias=alias, mode=MODE, repeat=1, status='preparing',
                       processed=0, expected=self.rows[alias]['frames'])
            write_new(d / 'RUN_REQUEST.json', dict(job, requested_ns=time.perf_counter_ns()))
            self.jobs[key] = job
            self.active = key
            self.worker = threading.Thread(target=self.run, args=(job,), daemon=True)
            self.worker.start()
        return key

    def public(self, key):
        with self.lock:
            return copy.deepcopy(self.jobs[key])

    def update(self, job, **kw):
        with self.lock:
            job.update(kw)

    def run(self, job):
        np = self.np
        d = RUNS / job['key']
        row = self.rows[job['alias']]
        lm = None
        try:
            self.check_pins()
            refrows = [r for r in self.frozen['references'] if r['word_id'] == row['word_id']]
            prefiles = [Path(row[k]) for k in ['source', 'input', 'observation', 'record']]
            prefiles += [Path(r['input']) for r in refrows]
            prefiles += [U / 'runs/42/auto' / (r['alias'] + '.npz') for r in refrows]
            prefiles += [U / 'runs/42/head.pt', U / 'episode.py']
            write_new(d / 'PRE_RUN_INPUTS.json', dict(at_ns=time.perf_counter_ns(),
                sha256={str(p): sha(p) for p in prefiles}, frozen_protocol_sha256=sha(U / 'WEB_FROZEN.json')))
            setup = time.perf_counter_ns()
            lm = self.capture.RawCaptureLandmarker(self.contract)
            processor = self.sc.FrameProcessor(lm, self.assignment.HandAssigner(), self.contract)
            detector_setup = (time.perf_counter_ns() - setup) / 1e9
            original = processor.process_frame
            def process(*args):
                original(*args)
                self.update(job, processed=len(processor.timing))
            processor.process_frame = process
            epoch = time.perf_counter_ns()
            self.update(job, status='recording')
            feed = self.sc.feed_single_decode(row['source'], processor, 'single_decode_paced', epoch)
            capture_end = feed['last_frame_available_ns']
            self.update(job, status='processing')
            baseline, obs, raw, sides = self.sc.make_observations(processor, row, feed['metadata'], processor.extractor_id())
            with np.load(row['input'], allow_pickle=False) as old:
                assert np.array_equal(obs.frame_id, old['frame_id'])
                assert np.array_equal(obs.timestamp_us, old['timestamp_us'])
            assert obs.n_frames == row['frames'] and feed['decode_passes'] == 1
            baseline.save(str(d / 'baseline.npz'))
            obs.save(str(d / 'candidate.npz'))
            np.savez_compressed(d / 'raw.npz', **raw)
            np.savez_compressed(d / 'assignment.npz', sides=sides)
            write_new(d / 'ASSIGNMENT_DIAGNOSTICS.json', processor.diagnostics)
            lm.close()
            lm = None
            saved = time.perf_counter_ns()
            feature_start = saved
            raw_ob = {k: getattr(obs, k).copy() for k in ['xy_px', 'observed', 'usable', 'frame_id', 'timestamp_us']}
            xy, b = raw_ob['xy_px'], row['box']
            raw_ob['usable'] &= ((xy[:, :, 0] >= b[0]) & (xy[:, :, 0] < b[2]) &
                                 (xy[:, :, 1] >= b[1]) & (xy[:, :, 1] < b[3]))
            feat = self.ep.extract_features(raw_ob)
            arr = dict(x=np.nan_to_num(feat['x']).astype('float32'), mask=feat['mask'],
                       frame_id=obs.frame_id.copy(), timestamp_us=obs.timestamp_us.copy())
            np.savez_compressed(d / 'input.npz', **arr)
            feature_end = time.perf_counter_ns()
            auto = self.ep.automatic(arr, self.model)
            np.savez_compressed(d / 'automatic.npz', **auto)
            episodes = self.ep.runs(auto['pred'], arr['frame_id'], arr['timestamp_us'])
            head_end = time.perf_counter_ns()
            self.update(job, status='comparing')
            pairs = []
            for r in refrows:
                ref = self.refs[r['alias']]
                pair = dict(reference=r['alias'], **self.ep.compare(arr, ref, auto['pred'], self.refstates[r['alias']]))
                pair['reference_seconds'] = (ref['timestamp_us'] / 1e6).tolist()
                pair['reference_frame_ids'] = ref['frame_id'].tolist()
                pair['full_scope_reference_frames'] = len(ref['frame_id'])
                pair['full_scope_query_frames'] = len(arr['frame_id'])
                if pair['status'] != 'no_path':
                    assert pair['reference_frames'] == len(ref['frame_id'])
                    assert pair['reference_scope_retained']
                    assert pair['all_query_frames_accounted'] == len(arr['frame_id'])
                pairs.append(pair)
            compare_end = time.perf_counter_ns()
            payload = dict(label=row['label'], alias=row['alias'], scope=SCOPE, caution=CAUTION,
                timing_scope=TIMING_SCOPE, educational_verdict=None, full_clip_correctness=None,
                source_frames=obs.n_frames, frame_id=arr['frame_id'].tolist(),
                seconds=(arr['timestamp_us'] / 1e6).tolist(),
                episodes=episodes, pairs=pairs,
                references=[dict(alias=r['alias'], label=r['label'], full_scope_frames=r['frames']) for r in refrows],
                support=dict(actual_right_hand_frames=int(self.ep.support(arr).sum()), total_frames=obs.n_frames,
                             unknown_episode_frames=int((auto['pred'] < 0).sum())),
                alignment_note='Reference-to-query median mapping is for viewing; comparison keeps every original DTW path point and full reference scope.',
                episode_note='Automatic centered-window proposals, including short runs; no exhaustive stage or boundary ground truth.')
            write_new(d / 'feedback.json', payload)
            ready = time.perf_counter_ns()
            write_new(d / 'TRACE.json', dict(epoch_ns=epoch, capture_end_ns=capture_end,
                saved_ns=saved, feature_end_ns=feature_end, head_end_ns=head_end,
                compare_end_ns=compare_end, ready_ns=ready, rows=processor.timing, arrivals=feed['arrivals']))
            metrics = dict(key=job['key'], alias=row['alias'], mode=MODE, repeat=1, frames=obs.n_frames,
                source_pts_seconds=float(obs.timestamp_us[-1] / 1e6), detector_setup_seconds=detector_setup,
                capture_end_ns=capture_end, ready_ns=ready,
                backend_after_end_seconds=(ready-capture_end)/1e9,
                capture_saved_tail_seconds=(saved-capture_end)/1e9,
                feature_seconds=(feature_end-feature_start)/1e9,
                episode_seconds=(head_end-feature_end)/1e9,
                dtw_three_refs_seconds=(compare_end-head_end)/1e9,
                maximum_queued_frames=feed['maximum_queued_frames'],
                remaining_at_capture_end=sum(r['assignment_end_ns'] > capture_end for r in processor.timing),
                queue_wait_max_seconds=max(r['queue_wait_seconds'] for r in processor.timing),
                arrival_lateness_max_seconds=max(max(0, (a['available_ns']-a['due_ns'])/1e9) for a in feed['arrivals']),
                decode_passes=1, no_dropped_frames=True, timing_scope=TIMING_SCOPE,
                backend_ready_definition='feedback.json persisted; following trace/metrics persistence and polling are included in ready-to-browser acknowledgement',
                source_sha256=sha(row['source']), input_before_sha256=sha(row['input']),
                generated_input_sha256=sha(d / 'input.npz'),
                output_sha256={p.name: sha(p) for p in d.iterdir() if p.is_file()})
            write_new(d / 'BACKEND_COMPLETE.json', metrics)
            self.update(job, status='ready', payload=payload, metrics=metrics)
        except BaseException as error:
            first_failure(d / 'FIRST_FAILURE.json', error)
            self.update(job, status='failed', error=str(error))
            with self.lock:
                self.active = None
        finally:
            if lm is not None:
                lm.close()

    def paint_ack(self, key, body):
        received = time.perf_counter_ns()
        with self.lock:
            job = self.jobs[key]
            assert job['status'] == 'ready' and self.active == key
            assert body['rendered_key'] == key and body['paint_callbacks'] == 2
            assert body['visible_scope'] and body['visible_caution'] and body['reference_rows'] == 3
            result = dict(render_ack_received_ns=received,
                backend_ready_to_2raf_ack_seconds=(received-job['metrics']['ready_ns'])/1e9,
                capture_end_to_2raf_ack_seconds=(received-job['metrics']['capture_end_ns'])/1e9,
                browser=body)
            write_new(RUNS / key / 'PAINT_ACK.json', result)
            job['paint_ack'] = result
            return result

    def acknowledge(self, key, body):
        received = time.perf_counter_ns()
        with self.lock:
            job = self.jobs[key]
            assert job['status'] == 'ready' and self.active == key and 'paint_ack' in job
            assert body['rendered_key'] == key
            for kind in ['query', 'reference']:
                assert body[kind]['ready_state'] >= 2 and body[kind]['video_width'] > 0
                assert body[kind]['loadeddata_or_already_ready'] is True
            assert body['query']['alias'] == job['alias']
            assert body['reference']['alias'] in [r['alias'] for r in job['payload']['references']]
            metrics = dict(job['metrics'], **{k:v for k,v in job['paint_ack'].items() if k != 'browser'},
                media_and_paint_ack_received_ns=received,
                backend_ready_to_media_and_paint_ack_seconds=(received-job['metrics']['ready_ns'])/1e9,
                capture_end_to_media_and_paint_ack_seconds=(received-job['metrics']['capture_end_ns'])/1e9,
                paint_ack_to_final_ack_seconds=(received-job['paint_ack']['render_ack_received_ns'])/1e9,
                browser_paint=job['paint_ack']['browser'], browser_media=body)
            write_new(RUNS / key / 'COMPLETE.json', metrics)
            job['status'] = 'complete'
            job['metrics'] = metrics
            self.active = None
            return metrics


def make_server(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def handle(self):
            try:
                super().handle()
            except ConnectionAbortedError as error:
                if getattr(error, 'winerror', None) != 10053:
                    raise
                with (U / 'WEB_TRANSPORT_EVENTS.jsonl').open('a', encoding='utf-8') as out:
                    out.write(json.dumps(dict(event='client_video_request_cancelled', winerror=10053,
                                              at_ns=time.perf_counter_ns())) + '\n')

        def reply(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            p = urlparse(self.path)
            if p.path == '/':
                data = (U / 'web_index.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif p.path == '/status':
                try:
                    self.reply(200, engine.public(parse_qs(p.query)['key'][0]))
                except (KeyError, ValueError):
                    self.reply(404, {'error': 'Unknown replay'})
            elif p.path.startswith('/video/'):
                alias = p.path.split('/')[-1]
                row = engine.rows.get(alias) or engine.refrows.get(alias)
                if row is None:
                    self.reply(404, {'error': 'Not whitelisted'})
                    return
                source = Path(row['source'])
                size = source.stat().st_size
                try:
                    start, end, partial = byte_range(self.headers.get('Range'), size)
                except ValueError as error:
                    self.reply(416, {'error': str(error)})
                    return
                self.send_response(206 if partial else 200)
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(end-start+1))
                if partial:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                try:
                    with source.open('rb') as f:
                        f.seek(start)
                        left = end-start+1
                        while left:
                            chunk = f.read(min(left, 65536))
                            if not chunk:
                                raise IOError('Unexpected video EOF')
                            self.wfile.write(chunk)
                            left -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            try:
                origin = self.headers.get('Origin')
                assert origin is None or origin == f'http://127.0.0.1:{self.server.server_port}'
                length = int(self.headers.get('Content-Length', '0'))
                assert 0 < length < 16384
                body = json.loads(self.rfile.read(length))
                if self.path == '/start':
                    self.reply(200, {'key': engine.start(body['alias'])})
                elif self.path == '/paint_ack':
                    self.reply(200, engine.paint_ack(body.pop('key'), body))
                elif self.path == '/ack':
                    self.reply(200, engine.acknowledge(body.pop('key'), body))
                else:
                    self.reply(404, {'error': 'Not found'})
            except Exception as error:
                self.reply(400, {'error': str(error)})
    return ThreadingHTTPServer(('127.0.0.1', 0), Handler)


def check():
    """Static/transport contract checks only; no model, detector or server starts."""
    for name in ['web_replay.py', 'web_browser_test.py']:
        ast.parse((U / name).read_text(encoding='utf-8'), filename=name)
    assert byte_range(None, 100) == (0, 99, False)
    assert byte_range('bytes=20-39', 100) == (20, 39, True)
    assert byte_range('bytes=20-', 100) == (20, 99, True)
    for bad in ['bytes=100-', 'bytes=40-20', 'bytes=-10', 'bytes=0-2,5-8']:
        try:
            byte_range(bad, 100)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    html = (U / 'web_index.html').read_text(encoding='utf-8')
    assert 'loadeddata' in html and '/paint_ack' in html and '/ack' in html
    assert 'getUserMedia' not in html and '8765' not in html
    return dict(status='PASS_STATIC_CONTRACT_ONLY', inference=0, server_started=False,
                media_playback_verified=False, timings_measured=False)


if __name__ == '__main__':
    if sys.argv[1:] == ['check']:
        print(json.dumps(check()))
    elif sys.argv[1:] == ['freeze']:
        freeze()
        print('WEB_FROZEN; replay has not run')
    else:
        raise SystemExit('Use check or freeze. Actual reviewed run: web_browser_test.py run')
