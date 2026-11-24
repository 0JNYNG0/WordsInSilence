"""Loopback-only, isolated upload/research-demo server. No model work at import.

POST /api/analyze?word=WORD#### accepts a raw video body, <=100 MiB.
The engine is injected by create_server for synthetic HTTP tests, or lazily
imported by main. No uploaded file can become a reference/catalog entry here.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
import argparse
import hashlib
import hmac
import importlib
import json
import mimetypes
import re
import secrets
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 100 * 1024 * 1024
MIME_EXT = {'video/mp4': '.mp4', 'video/webm': '.webm', 'video/quicktime': '.mov',
            'video/x-matroska': '.mkv', 'video/x-msvideo': '.avi', 'application/octet-stream': '.video'}
JOB_ID = re.compile(r'^[a-f0-9]{32}$')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def byte_range(value, size):
    if value is None:
        return 0, size - 1, False
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', value.strip())
    if match is None or size <= 0 or not any(match.groups()):
        raise ValueError('invalid range')
    left, right = match.groups()
    if not left:
        length = int(right)
        if length <= 0:
            raise ValueError('empty suffix')
        start, end = max(0, size - length), size - 1
    else:
        start = int(left)
        end = min(int(right), size - 1) if right else size - 1
        if start >= size or end < start:
            raise ValueError('outside file')
    return start, end, True


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, engine, storage_root=None, warmup=None):
        if address[0] != '127.0.0.1':
            raise ValueError('Only 127.0.0.1 binding is permitted')
        self.engine = engine
        self.ready = False
        self.started_job_count = 0
        startup_begin = time.monotonic()
        self.startup = dict(status='PREPARING', started_at=now(),
                            accepting_analysis_requests=False,
                            warmup_mode='synthetic_frozen_observers' if warmup else 'injected_engine_no_model_warmup')
        self.token = secrets.token_hex(32)
        self.lock = threading.RLock()
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='isolated-demo-job')
        parent = Path(storage_root or ROOT / 'sessions').resolve()
        self.session_dir = parent / secrets.token_hex(16)
        self.session_dir.mkdir(parents=True, exist_ok=False)
        raw = engine.catalog()
        self.catalog_data = json.loads(json.dumps(raw, allow_nan=False))
        if not isinstance(self.catalog_data.get('words'), list):
            raise ValueError('catalog requires words list')
        self.words, self.media = {}, {}
        for word in self.catalog_data['words']:
            word_id = word['id']
            if not isinstance(word_id, str) or not re.fullmatch(r'WORD\d+', word_id) or word_id in self.words:
                raise ValueError('Invalid/duplicate catalog word')
            self.words[word_id] = word
            for reference in word['references']:
                clip = str(reference['id'])
                path = Path(reference.pop('media_path')).resolve()
                if not path.is_file():
                    raise ValueError('Missing fixed reference media: ' + clip)
                if clip in self.media and self.media[clip] != path:
                    raise ValueError('Conflicting reference media ID')
                self.media[clip] = path
                reference['media_url'] = '/media?clip=' + __import__('urllib.parse', fromlist=['quote']).quote(clip, safe='')
        self.catalog_data['upload_limit_bytes'] = MAX_UPLOAD
        self.catalog_data['new_submission_training_allowed'] = False
        self.startup['catalog_prepare_seconds'] = time.monotonic()-startup_begin
        try:
            before_warmup = time.monotonic()
            # This precedes socket binding. No analysis upload is accepted while
            # models compile/load or execute their one synthetic startup forward.
            warm_result = warmup() if warmup is not None else None
            self.startup['warmup_seconds'] = time.monotonic()-before_warmup
            self.startup['warmup'] = json.loads(json.dumps(warm_result, allow_nan=False))
            super().__init__(address, Handler)
            self.startup.update(status='READY', ready_at=now(),
                                total_startup_seconds=time.monotonic()-startup_begin,
                                accepting_analysis_requests=True,
                                first_actual_request_included=False)
            write_new_json(self.session_dir/'STARTUP.json', self.startup)
            self.ready = True
        except BaseException:
            self.startup.update(status='STARTUP_FAILED', failed_at=now(),
                                accepting_analysis_requests=False,
                                total_startup_seconds=time.monotonic()-startup_begin)
            write_new_json(self.session_dir/'STARTUP_FAILED.json', self.startup)
            (self.session_dir/'STARTUP_FIRST_FAILURE.txt').write_text(traceback.format_exc(), encoding='utf8')
            self.executor.shutdown(wait=False, cancel_futures=False)
            raise

    def origins(self):
        port = self.server_port
        return {f'http://127.0.0.1:{port}', f'http://localhost:{port}'}

    def update(self, job_id, stage=None, percent=None, detail=None, **extra):
        if isinstance(stage, dict):
            data = dict(stage)
        else:
            data = dict(stage=stage, percent=percent, detail=detail, **extra)
        # A progress callback is display information, never an arbitrary job mutation.
        safe = {k: data[k] for k in ['stage', 'percent', 'detail'] if k in data and data[k] is not None}
        if 'percent' in safe:
            try:
                safe['percent'] = min(100.0, max(0.0, float(safe['percent'])))
            except (TypeError, ValueError):
                safe.pop('percent')
        with self.lock:
            self.jobs[job_id]['progress'] = safe
            self.jobs[job_id]['updated_at'] = now()

    def process(self, job_id):
        with self.lock:
            if not self.ready:
                raise RuntimeError('Server observers are not ready')
            job = self.jobs[job_id]
            self.started_job_count += 1
            job['status'] = 'RUNNING'
            job['started_at'] = now()
            job['started_monotonic'] = time.monotonic()
            job['is_first_actual_analysis_request'] = self.started_job_count == 1
            path, directory = job['_path'], job['_dir']
        try:
            result = self.engine.analyze_submission(path, directory, lambda *a, **kw: self.update(job_id, *a, **kw))
            # Reject unserializable/NaN results instead of inventing usable feedback.
            serialized = json.dumps(result, ensure_ascii=False, allow_nan=False)
            with (directory / 'server_result.json').open('x', encoding='utf-8') as f:
                f.write(serialized)
            with self.lock:
                job.update(status='COMPLETE', result=result, progress={'stage': '완료', 'percent': 100})
        except BaseException as exc:
            with (directory / 'SERVER_FIRST_FAILURE.txt').open('x', encoding='utf-8') as f:
                f.write(traceback.format_exc())
            with self.lock:
                job.update(status='FAILED', error={'type': type(exc).__name__, 'message': str(exc)[:1200]})
        finally:
            with self.lock:
                job['updated_at'] = now()
                job['elapsed_seconds'] = time.monotonic() - job['started_monotonic']
                timing = dict(job_id=job_id, status=job['status'],
                    is_first_actual_analysis_request=job['is_first_actual_analysis_request'],
                    analysis_seconds=job['elapsed_seconds'], startup_seconds_included=False,
                    upload_seconds=max(0, job.get('_queued_monotonic', job['started_monotonic'])
                                       - job.get('_received_monotonic', job['started_monotonic'])),
                    queue_seconds=max(0, job['started_monotonic']-job.get('_queued_monotonic', job['started_monotonic'])),
                    separately_recorded_startup_seconds=self.startup['total_startup_seconds'],
                    separately_recorded_warmup_seconds=self.startup['warmup_seconds'])
                job['request_timing'] = timing
                write_new_json(directory/'REQUEST_TIMING.json', timing)
                if job['is_first_actual_analysis_request']:
                    write_new_json(self.session_dir/'FIRST_REQUEST_TIMING.json', timing)

    def public_job(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                return None
            job = self.jobs[job_id]
            public = {k: v for k, v in job.items() if not k.startswith('_') and k != 'started_monotonic'}
            if job['status'] == 'RUNNING':
                public['elapsed_seconds'] = time.monotonic() - job['started_monotonic']
            return json.loads(json.dumps(public, allow_nan=False))

    def server_close(self):
        super().server_close()
        self.executor.shutdown(wait=False, cancel_futures=False)


class Handler(BaseHTTPRequestHandler):
    def handle(self):
        # Browsers cancel old video/range requests when seeking or switching refs.
        # Model/job exceptions are handled and recorded by DemoServer.process.
        try:
            return super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return

    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        # No request query, filename, token or submitted content in access logs.
        return

    def security_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Permissions-Policy', 'camera=(self), microphone=()')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'")

    def send_bytes(self, code, body, mime='application/json; charset=utf-8', headers=None):
        self.send_response(code)
        self.security_headers()
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def json(self, code, value, close=False):
        if close:
            self.close_connection = True
        self.send_bytes(code, json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                        headers={'Connection': 'close'} if close else None)

    def reject(self, code, message):
        # Drain a small already-declared body before closing: otherwise Windows
        # may reset the connection and hide the useful 4xx response from fetch.
        # Never consume an oversized upload merely to report its rejection.
        if self.command == 'POST' and not getattr(self, '_body_read_started', False):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if 0 < length <= 1024 * 1024 and not self.headers.get('Transfer-Encoding'):
                    self.connection.settimeout(2)
                    self.rfile.read(length)
            except (ValueError, OSError):
                pass
        self.json(code, {'error': message}, close=self.command == 'POST')

    def base_guard(self):
        host = self.headers.get('Host', '')
        expected = {origin.removeprefix('http://') for origin in self.server.origins()}
        origin = self.headers.get('Origin')
        if host not in expected:
            self.reject(403, '허용되지 않은 주소입니다. localhost 주소로 접속해 주세요.'); return False
        if origin is not None and origin not in self.server.origins():
            self.reject(403, '외부 사이트 요청은 허용하지 않습니다.'); return False
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            self.reject(403, '외부 사이트 요청은 허용하지 않습니다.'); return False
        return True

    def token_guard(self):
        candidate = self.headers.get('X-Session-Token', '')
        if not hmac.compare_digest(candidate.encode('utf-8'), self.server.token.encode('ascii')):
            self.reject(403, '세션을 확인할 수 없습니다. 이 화면을 새로 열어 주세요.'); return False
        return True

    def cookie_guard(self):
        try:
            cookie = SimpleCookie(self.headers.get('Cookie', ''))
            candidate = cookie['wis_demo_session'].value if 'wis_demo_session' in cookie else ''
        except Exception:
            candidate = ''
        if not hmac.compare_digest(candidate.encode('utf-8'), self.server.token.encode('ascii')):
            self.reject(403, '현재 로컬 세션에서만 제출 영상을 볼 수 있습니다.'); return False
        return True

    def file(self, path, mime=None):
        size = path.stat().st_size
        try:
            start, end, partial = byte_range(self.headers.get('Range'), size)
        except ValueError:
            self.send_bytes(416, b'', headers={'Content-Range': f'bytes */{size}'}); return
        length = max(0, end - start + 1)
        self.send_response(206 if partial else 200)
        self.security_headers()
        self.send_header('Content-Type', mime or mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(length))
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if self.command != 'HEAD':
            try:
                with path.open('rb') as f:
                    f.seek(start)
                    remaining = length
                    while remaining:
                        block = f.read(min(256 * 1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.base_guard():
            return
        parsed = urlsplit(self.path); path = unquote(parsed.path)
        if path == '/':
            page = (ROOT / 'index.html').read_text(encoding='utf-8').replace('__SESSION_TOKEN__', self.server.token)
            self.send_bytes(200, page.encode(), 'text/html; charset=utf-8',
                            {'Set-Cookie': f'wis_demo_session={self.server.token}; HttpOnly; SameSite=Strict; Path=/'})
        elif path == '/app.js':
            self.file(ROOT / 'app.js', 'text/javascript; charset=utf-8')
        elif path == '/health':
            self.json(200 if self.server.ready else 503,
                      {'status': 'ok' if self.server.ready else 'preparing',
                       'ready': self.server.ready, 'mode': 'isolated_local_research_demo',
                       'port': self.server.server_port, 'startup': self.server.startup})
        elif path == '/api/catalog':
            if self.token_guard():
                self.json(200, self.server.catalog_data)
        elif path.startswith('/api/jobs/'):
            if not self.token_guard():
                return
            job_id = path.removeprefix('/api/jobs/')
            job = self.server.public_job(job_id) if JOB_ID.fullmatch(job_id) else None
            if job is None:
                self.reject(404, '현재 세션의 작업을 찾을 수 없습니다.')
            else:
                self.json(200, job)
        elif path == '/media':
            values = parse_qs(parsed.query).get('clip', [])
            media = self.server.media.get(values[0]) if len(values) == 1 else None
            if media is None:
                self.reject(404, '허용된 공개 참고 영상이 아닙니다.')
            else:
                self.file(media)
        elif path.startswith('/submission/'):
            if not self.cookie_guard():
                return
            job_id = path.removeprefix('/submission/')
            with self.server.lock:
                job = self.server.jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None
                source = job['_path'] if job and job['status'] != 'RECEIVING' else None
            if source is None:
                self.reject(404, '현재 세션의 제출 영상을 찾을 수 없습니다.')
            else:
                self.file(source, job['_mime'])
        else:
            self.reject(404, '요청한 항목이 없습니다.')

    def do_POST(self):
        if not self.base_guard() or not self.token_guard():
            return
        if not self.server.ready:
            self.reject(503, '분석 모델을 준비하고 있습니다. 준비가 끝난 뒤 다시 전송해 주세요.'); return
        if self.headers.get('Origin') not in self.server.origins():
            self.reject(403, '로컬 화면에서 전송해 주세요.'); return
        parsed = urlsplit(self.path)
        if parsed.path != '/api/analyze':
            self.reject(404, '알 수 없는 요청입니다.'); return
        parameters = parse_qs(parsed.query)
        words = parameters.get('word', [])
        if len(words) != 1 or words[0] not in self.server.words:
            self.reject(400, '비교할 목표 표현을 선택해 주세요.'); return
        if self.headers.get('Transfer-Encoding'):
            self.reject(411, '영상 크기를 확인할 수 없습니다.'); return
        try:
            length = int(self.headers.get('Content-Length', ''))
        except ValueError:
            self.reject(411, '영상 크기를 확인할 수 없습니다.'); return
        if length <= 0 or length > MAX_UPLOAD:
            self.reject(413, '영상은 0바이트 초과, 100MB 이하여야 합니다.'); return
        mime = self.headers.get('Content-Type', '').split(';')[0].strip().lower()
        if mime not in MIME_EXT:
            self.reject(415, 'MP4, WebM, MOV 등의 영상 파일을 선택해 주세요.'); return
        with self.server.lock:
            if any(j['status'] in {'RECEIVING', 'QUEUED', 'RUNNING'} for j in self.server.jobs.values()):
                self.reject(409, '현재 영상을 분석 중입니다. 완료 후 다시 전송해 주세요.'); return
            if len(self.server.jobs) >= 20:
                self.reject(429, '이 세션의 작업 한도에 도달했습니다. 서버를 다시 시작해 주세요.'); return
            job_id = secrets.token_hex(16)
            directory = self.server.session_dir / job_id
            directory.mkdir(exist_ok=False)
            source = directory / ('submission' + MIME_EXT[mime])
            request = dict(word=words[0], content_type=mime, bytes=length, created_at=now(),
                           original_filename=self.headers.get('X-Filename', '')[:240],
                           source='browser_recording' if self.headers.get('X-Submission-Source') == 'browser_recording' else 'file_upload',
                           isolated_user_submission=True, training_allowed=False, reference_bank_addition_allowed=False)
            (directory / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf-8')
            self.server.jobs[job_id] = dict(id=job_id, word=words[0], status='RECEIVING', created_at=now(),
                                             progress={'stage': '영상 받는 중', 'percent': 0}, elapsed_seconds=0,
                                             submission_url='/submission/' + job_id,
                                             _path=source, _dir=directory, _mime=mime,
                                             _received_monotonic=time.monotonic())
        try:
            self.connection.settimeout(60)
            self._body_read_started = True
            h = hashlib.sha256(); remaining = length
            with source.open('xb') as f:
                while remaining:
                    block = self.rfile.read(min(256 * 1024, remaining))
                    if not block:
                        raise ValueError('영상 전송이 끝나기 전에 연결이 끊어졌습니다.')
                    f.write(block); h.update(block); remaining -= len(block)
            with self.server.lock:
                self.server.jobs[job_id].update(status='QUEUED', submission_sha256=h.hexdigest(), updated_at=now(),
                                               _queued_monotonic=time.monotonic())
            self.server.executor.submit(self.server.process, job_id)
            self.json(202, {'id': job_id, 'status': 'QUEUED', 'status_url': '/api/jobs/' + job_id})
        except Exception as exc:
            with self.server.lock:
                self.server.jobs[job_id].update(status='FAILED', error={'type': type(exc).__name__, 'message': '영상 전송을 완료하지 못했습니다.'})
            self.reject(400, '영상 전송을 완료하지 못했습니다. 파일을 다시 선택해 주세요.')

    def do_OPTIONS(self):
        self.reject(403, '외부 사이트 연결은 허용하지 않습니다.')


def create_server(engine, port=8802, storage_root=None, warmup=None):
    """Tests may inject a model-free engine; real main always supplies warmup."""
    return DemoServer(('127.0.0.1', port), engine, storage_root, warmup)


def startup_selftest():
    """No socket, model, video or camera: mock startup and two fake requests."""
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch
    checks = []
    def check(name, good):
        if not good:
            raise AssertionError(name)
        checks.append(name)
    events = []
    class Engine:
        def catalog(self):
            events.append('catalog'); return {'words': []}
        def analyze_submission(self, path, directory, progress):
            events.append('fake_analysis'); return {'synthetic': True, 'latency': {'test_seconds': 0}}
    def warm():
        events.append('warmup')
        return dict(status='WARMED_FROZEN_OBSERVERS', device=dict(requested='CPU', actual=['CPU'],
                    failed_attempts=[dict(device='GPU', error='synthetic fallback')]),
                    synthetic_rgb_windows=1, real_video_inference=0)
    def fake_bind(server, address, handler):
        events.append('bind'); server.server_port=0
    (ROOT/'review').mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='wis_server_startup_', dir=ROOT/'review') as root:
        if not Path(root).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Synthetic temporary cleanup must remain inside research unit')
        with patch.object(ThreadingHTTPServer, '__init__', fake_bind), patch.object(ThreadingHTTPServer, 'server_close'):
            server = create_server(Engine(), 0, root, warm)
            check('warmup completes before any bind/listen', events == ['catalog','warmup','bind'])
            check('ready only after startup completion', server.ready and server.startup['status'] == 'READY')
            saved = json.loads((server.session_dir/'STARTUP.json').read_text())
            check('startup preserves actual CPU fallback disclosure', saved['warmup']['device']['actual'] == ['CPU']
                  and len(saved['warmup']['device']['failed_attempts']) == 1)
            check('startup excludes real video calls', saved['warmup']['real_video_inference'] == 0
                  and saved['first_actual_request_included'] is False)
            handler = object.__new__(Handler); handler.server=server
            handler.base_guard=lambda: True; handler.token_guard=lambda: True
            rejected=[]; handler.reject=lambda *args: rejected.append(args)
            server.ready=False; handler.do_POST()
            check('preparing POST rejects503 without job creation', rejected[0][0] == 503 and not server.jobs)
            server.ready=True
            for i in range(2):
                key=str(i); directory=server.session_dir/key; directory.mkdir()
                server.jobs[key]=dict(_dir=directory, _path=directory/'fake.video', status='QUEUED',
                                     _received_monotonic=time.monotonic(), _queued_monotonic=time.monotonic())
                server.process(key)
            first=json.loads((server.session_dir/'FIRST_REQUEST_TIMING.json').read_text())
            check('first actual request timing stored separately', first['job_id'] == '0'
                  and first['startup_seconds_included'] is False and first['analysis_seconds'] >= 0)
            check('second request cannot overwrite first timing', server.jobs['1']['is_first_actual_analysis_request'] is False
                  and (server.session_dir/'1/REQUEST_TIMING.json').is_file())
            check('startup warmup not repeated for requests', events.count('warmup') == 1 and events.count('fake_analysis') == 2)
            server.server_close()
            before=events.count('bind')
            def broken_warm():
                raise RuntimeError('synthetic initialization failure')
            try:
                create_server(Engine(),0,root,broken_warm)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Failed warmup accepted')
            check('warmup failure never binds socket', events.count('bind') == before)
            failures=list(Path(root).glob('*/STARTUP_FAILED.json'))
            check('first startup failure preserved', len(failures) == 1
                  and json.loads(failures[0].read_text())['accepting_analysis_requests'] is False)
    return dict(status='PASS', checks=len(checks), results=checks,
                server_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                actual_server_started=False, sockets_opened=0, model_calls=0, video_reads=0, camera_calls=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8802)
    parser.add_argument('--engine-module', choices=['engine','live_engine'], default='engine')
    parser.add_argument('--startup-selftest', action='store_true')
    args = parser.parse_args()
    if args.startup_selftest:
        print(json.dumps(startup_selftest(), ensure_ascii=False, indent=2)); return
    print('Preparing frozen observers; analysis requests are not accepted yet.', flush=True)
    server = create_server(importlib.import_module(args.engine_module), args.port,
                           warmup=importlib.import_module('adapter').warmup)
    print(f'Local isolated feedback demo: http://127.0.0.1:{server.server_port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
