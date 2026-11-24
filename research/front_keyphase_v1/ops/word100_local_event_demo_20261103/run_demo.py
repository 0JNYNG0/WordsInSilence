"""Loopback research demo. Importing this module never imports a model runtime."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import hashlib
import importlib.util
import json
import os
import re
import secrets
import socket
import threading
import traceback
import uuid

U = Path(__file__).resolve().parent
PORT = 8810
MAX_UPLOAD = 64 * 1024 * 1024
JOB_RE = re.compile(r'[0-9a-f]{32}')
LABELS = {'WORD0739': '순종', 'WORD1133': '잘하다', 'WORD1567': '조상', 'WORD2523': '이집트'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_public_assets(unit, words):
    """Only named public manifests and their exact allowlisted files are read."""
    ops = Path(unit).parent
    previous = ops / 'word100_local_event_runtime_20261103'
    run_path = previous / 'PUBLIC_RUN_RESULT.json'
    input_path = previous / 'PUBLIC_INPUT_FROZEN.json'
    run, source = read(run_path), read(input_path)
    jid = run['job_id']
    if not JOB_RE.fullmatch(jid):
        raise ValueError('Invalid saved public job identity')
    result_path = Path(run['result']).resolve()
    expected = (previous / 'public_runtime_runs' / jid / 'analysis' / 'LIVE_FEEDBACK.json').resolve()
    video_path = Path(source['source']).resolve()
    if result_path != expected or video_path.parent != (previous / 'public_input').resolve():
        raise ValueError('Saved example must be the named public job/input')
    if run['word_id'] != 'WORD1133' or source['word_id'] != 'WORD1133':
        raise ValueError('Saved example word mismatch')
    if source['source_role'] != 'public_runtime_inference':
        raise ValueError('Saved example is not a public source')
    if any(source[k] is not False for k in ('training', 'tuning', 'candidate_selection')):
        raise ValueError('Invalid example inference-only provenance')
    if sha(result_path) != run['result_sha256'] or sha(video_path) != source['source_sha256']:
        raise ValueError('Saved public file changed')
    if run['source_sha256'] != source['source_sha256']:
        raise ValueError('Saved example source mismatch')
    result = read(result_path)
    if result['word_id'] != 'WORD1133' or result['source_sha256'] != source['source_sha256']:
        raise ValueError('Saved result word/source mismatch')
    manifest_path = ops / 'word100_learned_guide_feedback_20261102' / 'INPUT_MANIFEST.json'
    manifest = read(manifest_path)['rows']
    references = {}
    for word in words:
        rows = [r for r in manifest if r['word_id'] == word and r['person'] == 1]
        if len(rows) != 1:
            raise ValueError('Exactly one public ID01 reference is required')
        row = rows[0]
        path = Path(row['source']).resolve()
        if row['role'] != 'fit' or sha(path) != row['source_sha256']:
            raise ValueError('Public reference provenance mismatch')
        references[word] = {'path': str(path), 'sha256': row['source_sha256'], 'person': 1}
    return {
        'example': {'word_id': 'WORD1133', 'path': str(video_path),
                    'sha256': source['source_sha256'], 'result_path': str(result_path),
                    'result_sha256': run['result_sha256'], 'result': result},
        'references': references,
        'manifest_pins': {str(p.resolve()): sha(p) for p in (run_path, input_path, manifest_path)},
    }


class DemoApplication:
    def __init__(self, adapter, unit=None, public_assets=None, port=PORT):
        self.adapter = adapter
        self.unit = Path(unit if unit is not None else adapter.U).resolve()
        self.words = tuple(adapter.WORDS)
        if set(self.words) != set(LABELS):
            raise ValueError('Unsupported research word set')
        self.port = port
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.request_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='inference-only')
        self.jobs = {}  # Current process only. Never restore old private jobs.
        self.future = None
        self.receiving = False
        self.closed = False
        self.initialization_started = False
        self.init_thread = None
        self.state = {'ready': False, 'message': '연구용·추론 전용: 분석기를 준비하고 있습니다.'}
        self.assets = public_assets if public_assets is not None else load_public_assets(self.unit, self.words)
        self.validate_assets()

    def validate_assets(self):
        example = self.assets['example']
        if example['word_id'] != 'WORD1133' or example['result']['word_id'] != 'WORD1133':
            raise ValueError('Example word mismatch')
        if example['result']['source_sha256'] != example['sha256']:
            raise ValueError('Example source identity mismatch')
        if sha(example['path']) != example['sha256'] or sha(example['result_path']) != example['result_sha256']:
            raise ValueError('Example bytes do not match the pinned result/media')
        if read(example['result_path']) != example['result']:
            raise ValueError('Example object differs from its pinned file')
        if set(self.assets['references']) != set(self.words):
            raise ValueError('Reference allowlist mismatch')
        for row in self.assets['references'].values():
            if row['person'] != 1 or sha(row['path']) != row['sha256']:
                raise ValueError('Reference identity mismatch')

    def start_initialization(self):
        with self.lock:
            if self.initialization_started or self.closed:
                return False
            self.initialization_started = True
            self.init_thread = threading.Thread(target=self._initialize, daemon=True)
            self.init_thread.start()
            return True

    def _initialize(self):
        try:
            warm = self.adapter.warmup()
            # Verify serializability before marking the service ready.
            json.dumps(warm, allow_nan=False)
            with self.lock:
                self.state.update(ready=True, message='연구용·추론 전용: 새 촬영을 분석할 수 있습니다.', warmup=warm)
        except Exception:
            self.first_failure('MODEL_INITIALIZATION_FIRST_FAILURE.txt')
            with self.lock:
                self.state.update(ready=False, message='연구용·추론 전용: 분석기 준비에 실패했습니다.')

    def first_failure(self, name, directory=None):
        path = Path(directory or self.unit) / name
        try:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(traceback.format_exc())
        except FileExistsError:
            pass

    def busy(self):
        with self.lock:
            return self.receiving or (self.future is not None and not self.future.done())

    def snapshot(self):
        with self.lock:
            return deepcopy(dict(self.state, token=self.token, words=LABELS, busy=self.busy(),
                                 research_inference_only=True, training=False))

    def example_result(self):
        record = self.assets['example']
        if sha(record['result_path']) != record['result_sha256'] or sha(record['path']) != record['sha256']:
            raise ValueError('Pinned saved example changed')
        result = deepcopy(record['result'])
        identity = {'word_id': 'WORD1133', 'case_kind': 'saved_public_replay',
                    'media_url': '/media/example/WORD1133', 'source_sha256': record['sha256']}
        for key, value in (('ui_source_identity', identity), ('ui_result_provenance', 'saved_public_replay')):
            if key in result and result[key] != value:
                raise ValueError('Do not rewrite a mismatched saved UI identity')
            result[key] = value
        return result

    def perform(self, jid, source, word, condition, provenance):
        directory = self.unit / 'private_sessions' / jid
        try:
            def progress(value):
                json.dumps(value, allow_nan=False)
                with self.lock:
                    self.jobs[jid]['progress'] = deepcopy(value)
            result = self.adapter.analyze(source, directory / 'analysis', word, condition,
                                          progress, provenance=deepcopy(provenance))
            if result['source_sha256'] != provenance['source_sha256'] or result['word_id'] != word:
                raise ValueError('The result does not belong to the submitted video/word')
            if result.get('job_id', jid) != jid:
                raise ValueError('Result job mismatch')
            identity = {'word_id': word, 'case_kind': 'user_upload',
                        'media_url': '/media/session/' + jid,
                        'source_sha256': provenance['source_sha256'], 'job_id': jid}
            if 'ui_source_identity' in result and result['ui_source_identity'] != identity:
                raise ValueError('Unexpected existing UI identity')
            result = deepcopy(result)
            result['ui_source_identity'] = identity
            result['ui_result_provenance'] = 'user_upload'
            json.dumps(result, allow_nan=False)
            with self.lock:
                self.jobs[jid].update(status='complete', result=result)
        except Exception:
            self.first_failure('ANALYSIS_FIRST_FAILURE.txt', directory)
            with self.lock:
                self.jobs[jid].update(status='failed', error='분석이 완료되지 않았습니다. 같은 요청을 자동 재실행하지 않습니다.')

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.state['ready'] = False
        if self.init_thread is not None:
            self.init_thread.join()
        self.pool.shutdown(wait=True, cancel_futures=False)
        self.adapter.close()

    def create_server(self):
        app = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def safe_request(self):
                hosts = {f'127.0.0.1:{app.port}', f'localhost:{app.port}'}
                if len(self.headers.get_all('Host', [])) != 1 or self.headers.get('Host') not in hosts:
                    self.send_json({'error': 'Loopback host required'}, 403)
                    return False
                origin = self.headers.get('Origin')
                if origin and origin not in {'http://' + host for host in hosts}:
                    self.send_json({'error': 'Origin rejected'}, 403)
                    return False
                if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                    self.send_json({'error': 'Cross-site request rejected'}, 403)
                    return False
                if '%' in self.path or '\\' in self.path or '..' in self.path:
                    self.send_json({'error': 'Invalid path'}, 400)
                    return False
                return True

            def send_json(self, value, status=200):
                body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def send_file(self, path, content_type, expected_sha=None):
                path = Path(path)
                if not path.is_file():
                    return self.send_json({'error': 'File not found'}, 404)
                if expected_sha is not None and sha(path) != expected_sha:
                    return self.send_json({'error': 'Pinned media changed'}, 409)
                size = path.stat().st_size
                lo, hi, partial = 0, size - 1, False
                header = self.headers.get('Range')
                if header:
                    match = re.fullmatch(r'bytes=(\d+)-(\d*)', header)
                    if not match:
                        return self.send_json({'error': 'Unsupported range'}, 416)
                    lo = int(match[1]); hi = min(int(match[2]) if match[2] else hi, hi)
                    if lo > hi:
                        return self.send_json({'error': 'Invalid range'}, 416)
                    partial = True
                self.send_response(206 if partial else 200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(max(0, hi - lo + 1)))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                if partial:
                    self.send_header('Content-Range', f'bytes {lo}-{hi}/{size}')
                self.end_headers()
                try:
                    with path.open('rb') as stream:
                        stream.seek(lo)
                        remaining = max(0, hi - lo + 1)
                        while remaining:
                            chunk = stream.read(min(1048576, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def do_GET(self):
                if not self.safe_request():
                    return
                parsed = urlsplit(self.path)
                route = parsed.path
                if parsed.query:
                    return self.send_json({'error': 'Unexpected query'}, 400)
                if route == '/':
                    return self.send_file(app.unit / 'demo.html', 'text/html; charset=utf-8')
                if route == '/app.js':
                    return self.send_file(app.unit / 'demo.js', 'text/javascript; charset=utf-8')
                if route == '/api/state':
                    return self.send_json(app.snapshot())
                if route == '/api/example/WORD1133':
                    try:
                        return self.send_json(app.example_result())
                    except Exception:
                        app.first_failure('EXAMPLE_FIRST_FAILURE.txt')
                        return self.send_json({'error': 'Saved example identity verification failed'}, 409)
                if route == '/media/example/WORD1133':
                    record = app.assets['example']
                    return self.send_file(record['path'], 'video/mp4', record['sha256'])
                match = re.fullmatch(r'/media/reference/(WORD\d{4})', route)
                if match and match[1] in app.words:
                    record = app.assets['references'][match[1]]
                    return self.send_file(record['path'], 'video/mp4', record['sha256'])
                match = re.fullmatch(r'/(api/job|media/session)/([0-9a-f]{32})', route)
                if match:
                    jid = match[2]
                    with app.lock:
                        job = deepcopy(app.jobs.get(jid))
                    if job is None:
                        return self.send_json({'error': 'Unknown current-session job'}, 404)
                    if match[1] == 'api/job':
                        return self.send_json(job)
                    path = app.unit / 'private_sessions' / jid / 'input.video'
                    return self.send_file(path, job['mime'], job['source_sha256'])
                return self.send_json({'error': 'Not found'}, 404)

            def do_POST(self):
                if not self.safe_request():
                    return
                parsed = urlsplit(self.path)
                if parsed.path != '/api/analyze':
                    return self.send_json({'error': 'Not found'}, 404)
                tokens = self.headers.get_all('X-Session-Token', [])
                if len(tokens) != 1 or not secrets.compare_digest(tokens[0], app.token):
                    return self.send_json({'error': 'Invalid session token'}, 403)
                try:
                    query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
                except ValueError:
                    return self.send_json({'error': 'Invalid query'}, 400)
                if set(query) != {'word', 'condition'} or any(len(v) != 1 for v in query.values()):
                    return self.send_json({'error': 'Exactly one word and condition are required'}, 400)
                word, condition = query['word'][0], query['condition'][0]
                if word not in app.words or condition != 'learned_motion':
                    return self.send_json({'error': 'Unsupported condition'}, 400)
                lengths = self.headers.get_all('Content-Length', [])
                if self.headers.get('Transfer-Encoding') or len(lengths) != 1 or not re.fullmatch(r'[0-9]+', lengths[0]):
                    return self.send_json({'error': 'A single Content-Length is required'}, 400)
                length = int(lengths[0])
                if not 0 < length <= MAX_UPLOAD:
                    return self.send_json({'error': '영상은 0바이트보다 크고 64MB 이하여야 합니다.'}, 413)
                if not app.request_lock.acquire(blocking=False):
                    return self.send_json({'error': 'Another upload is in progress'}, 409)
                try:
                    with app.lock:
                        if not app.state['ready'] or app.closed or app.busy():
                            return self.send_json({'error': '분석이 끝난 뒤 다시 시도해 주세요.'}, 409)
                        app.receiving = True
                    previous_timeout = self.connection.gettimeout()
                    try:
                        self.connection.settimeout(20)
                        body = self.rfile.read(length)
                    except (TimeoutError, socket.timeout, OSError):
                        return self.send_json({'error': 'Upload interrupted'}, 400)
                    finally:
                        self.connection.settimeout(previous_timeout)
                    if len(body) != length:
                        return self.send_json({'error': 'Upload interrupted'}, 400)
                    jid = uuid.uuid4().hex
                    directory = app.unit / 'private_sessions' / jid
                    directory.mkdir(parents=True, exist_ok=False)
                    source = directory / 'input.video'
                    with source.open('xb') as stream:
                        stream.write(body)
                    source_hash = hashlib.sha256(body).hexdigest()
                    provenance = dict(source_role='user_inference_only', source_sha256=source_hash,
                                      word_id=word, training=False, tuning=False, candidate_selection=False)
                    write_new(directory / 'INFERENCE_ONLY.json', dict(provenance, condition=condition,
                              source='Explicit upload to local research demo'))
                    mime = self.headers.get('X-Video-Type', 'video/webm')
                    if mime not in ('video/webm', 'video/mp4'):
                        mime = 'application/octet-stream'
                    with app.lock:
                        app.jobs[jid] = dict(id=jid, word_id=word, status='running',
                                            progress={'stage': 'starting'}, mime=mime,
                                            source_sha256=source_hash, training=False,
                                            tuning=False, candidate_selection=False)
                        app.future = app.pool.submit(app.perform, jid, source, word, condition, provenance)
                    return self.send_json({'id': jid}, 202)
                except Exception:
                    app.first_failure('UPLOAD_FIRST_FAILURE.txt')
                    return self.send_json({'error': 'Upload could not be accepted'}, 500)
                finally:
                    with app.lock:
                        app.receiving = False
                    app.request_lock.release()

        server = ThreadingHTTPServer(('127.0.0.1', self.port), Handler)
        self.port = server.server_port
        return server


def main():
    # Never import native inference modules before the root-owned execution freeze passes.
    frozen = read(U / 'EXECUTION_FROZEN.json')
    for path, expected in frozen['pins'].items():
        if sha(path) != expected:
            raise RuntimeError('Execution pin mismatch: ' + path)
    spec = importlib.util.spec_from_file_location('_am_live_adapter', U / 'learner_adapter.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    app = DemoApplication(adapter)
    server = app.create_server()
    try:
        write_new(U / 'DEMO_LAUNCH.json', dict(pid=os.getpid(), url=f'http://127.0.0.1:{PORT}/',
                  research_inference_only=True, camera_opened=False, training=False,
                  new_public_analysis_route=False, saved_example_only='WORD1133'))
        app.start_initialization()
        print(f'http://127.0.0.1:{PORT}/', flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        app.close()


if __name__ == '__main__':
    main()
