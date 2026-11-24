"""Synthetic HTTP contract tests. Fake engine only; never opens camera/model/video."""
from pathlib import Path
from http.client import HTTPConnection
from html.parser import HTMLParser
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
import threading
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('isolated_server_under_test', ROOT / 'server.py')
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)
checks = []


def check(name, value):
    checks.append({'name': name, 'pass': bool(value)})
    if not value:
        raise AssertionError(name)


class FakeEngine:
    def __init__(self, media):
        self.media = media
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls = 0

    def catalog(self):
        return {'words': [{'id': 'WORD0001', 'label': '합성 참고', 'references': [
            {'id': 'SYNTHETIC_REF', 'label': '합성 참고1', 'media_path': str(self.media)}]}],
            'limits': {'max_duration_seconds': 30}}

    def analyze_submission(self, path, jobdir, progress):
        self.calls += 1
        request = json.loads((jobdir / 'request.json').read_text(encoding='utf-8'))
        check('word passed through isolated request', request['word'] == 'WORD0001')
        check('user data never training/reference', request['training_allowed'] is False and request['reference_bank_addition_allowed'] is False)
        check('raw upload preserved', path.read_bytes() == b'SYNTHETIC_VIDEO_BYTES_NOT_A_REAL_VIDEO')
        progress({'stage': '합성 진행 검사', 'percent': 40, 'detail': '실제 추론 없음'})
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError('selftest did not release fake engine')
        return {'stages': [{'id': 'synthetic', 'status': 'INSUFFICIENT_EVIDENCE', 'message': '합성 응답'}],
                'references': [{'id': 'SYNTHETIC_REF', 'stages': []}], 'educational_verdict': None}


def run():
    class IdParser(HTMLParser):
        def __init__(self):
            super().__init__(); self.ids = []

        def handle_starttag(self, tag, attrs):
            self.ids.extend(v for k, v in attrs if k == 'id')

    html = (ROOT / 'index.html').read_text(encoding='utf-8')
    js = (ROOT / 'app.js').read_text(encoding='utf-8')
    parser = IdParser(); parser.feed(html)
    check('unique HTML IDs', len(parser.ids) == len(set(parser.ids)))
    required = set(re.findall(r"\$\('([^']+)'\)", js))
    check('all JS selectors exist', required <= set(parser.ids))
    check('no third-party script or network URL', not re.search(r'https?://', html + js))
    check('camera explicit click only', "$('camera').addEventListener('click', openCamera)" in js and js.count('getUserMedia({') == 1)
    check('audio acquisition disabled', 'audio: false' in js)
    check('no dangerous HTML rendering of engine output', 'innerHTML' not in js and 'insertAdjacentHTML' not in js)
    check('three status labels explicit', all(s in js for s in ['SIMILAR_REFERENCE_RANGE', 'DIFFERENCE_CANDIDATE', 'INSUFFICIENT_EVIDENCE']))
    check('selftest bind guard', server_module.byte_range('bytes=-3', 10) == (7, 9, True))
    with tempfile.TemporaryDirectory(prefix='_http_selftest_', dir=ROOT) as temporary:
        base = Path(temporary); media = base / 'synthetic.mp4'; media.write_bytes(b'0123456789')
        fake = FakeEngine(media)
        server = server_module.create_server(fake, port=0, storage_root=base / 'sessions')
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        origin = f'http://127.0.0.1:{server.server_port}'

        def request(method, path, body=None, headers=None, auth=True):
            values = {'Origin': origin}
            if auth:
                values['X-Session-Token'] = server.token
            values.update(headers or {})
            connection = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
            try:
                connection.request(method, path, body=body, headers=values)
                response = connection.getresponse()
                return response.status, dict(response.getheaders()), response.read()
            finally:
                connection.close()

        try:
            status, headers, body = request('GET', '/', auth=False)
            check('root served and token injected', status == 200 and server.token.encode() in body and b'__SESSION_TOKEN__' not in body)
            cookie = headers['Set-Cookie'].split(';', 1)[0]
            check('session cookie constrained', 'HttpOnly' in headers['Set-Cookie'] and 'SameSite=Strict' in headers['Set-Cookie'])
            check('CSP blocks external connections', "connect-src 'self'" in headers['Content-Security-Policy'])
            check('anti iframe header', headers['X-Frame-Options'] == 'DENY')
            check('untrusted Host blocked', request('GET', '/', headers={'Host': 'attacker.invalid'})[0] == 403)
            check('cross Origin blocked', request('GET', '/', headers={'Origin': 'https://attacker.invalid'})[0] == 403)
            check('cross-site fetch blocked', request('GET', '/', headers={'Sec-Fetch-Site': 'cross-site'})[0] == 403)
            check('API token required', request('GET', '/api/catalog', auth=False)[0] == 403)
            status, _, body = request('GET', '/api/catalog')
            catalog = json.loads(body)
            check('catalog strips server filesystem', status == 200 and 'media_path' not in catalog['words'][0]['references'][0])
            check('catalog allowlisted media URL', catalog['words'][0]['references'][0]['media_url'] == '/media?clip=SYNTHETIC_REF')
            check('path traversal rejected', request('GET', '/%2e%2e/server.py')[0] == 404)
            check('arbitrary media path rejected', request('GET', '/media?clip=../../server.py')[0] == 404)
            status, headers, body = request('GET', '/media?clip=SYNTHETIC_REF', headers={'Range': 'bytes=2-5'})
            check('range bytes exact', status == 206 and body == b'2345' and headers['Content-Range'] == 'bytes 2-5/10')
            check('suffix range exact', request('GET', '/media?clip=SYNTHETIC_REF', headers={'Range': 'bytes=-2'})[2] == b'89')
            check('out-of-bounds range rejected', request('GET', '/media?clip=SYNTHETIC_REF', headers={'Range': 'bytes=99-'})[0] == 416)
            check('multi-range rejected', request('GET', '/media?clip=SYNTHETIC_REF', headers={'Range': 'bytes=0-1,4-5'})[0] == 416)
            check('HEAD length no body', request('HEAD', '/media?clip=SYNTHETIC_REF')[2] == b'')
            check('upload no token rejected', request('POST', '/api/analyze?word=WORD0001', b'x', {'Content-Type': 'video/webm'}, auth=False)[0] == 403)
            check('upload unknown word rejected', request('POST', '/api/analyze?word=WORD9999', b'x', {'Content-Type': 'video/webm'})[0] == 400)
            check('upload invalid MIME rejected', request('POST', '/api/analyze?word=WORD0001', b'x', {'Content-Type': 'text/html'})[0] == 415)
            check('oversize before write rejected', request('POST', '/api/analyze?word=WORD0001', None, {'Content-Type': 'video/webm', 'Content-Length': str(server_module.MAX_UPLOAD + 1)})[0] == 413)
            check('empty upload rejected', request('POST', '/api/analyze?word=WORD0001', b'', {'Content-Type': 'video/webm'})[0] == 413)
            check('zero inference before valid request', fake.calls == 0 and len(server.jobs) == 0)
            payload = b'SYNTHETIC_VIDEO_BYTES_NOT_A_REAL_VIDEO'
            status, _, body = request('POST', '/api/analyze?word=WORD0001', payload,
                                      {'Content-Type': 'video/webm', 'X-Filename': '../../escape.webm', 'X-Submission-Source': 'browser_recording'})
            job_id = json.loads(body)['id']
            check('upload accepted async', status == 202 and fake.started.wait(2))
            check('one active job enforced', request('POST', '/api/analyze?word=WORD0001', payload, {'Content-Type': 'video/webm'})[0] == 409)
            status, _, body = request('GET', '/api/jobs/' + job_id)
            job = json.loads(body)
            check('live progress preserved', status == 200 and job['status'] == 'RUNNING' and job['progress']['percent'] == 40)
            check('no server paths in job response', '_path' not in job and '_dir' not in job)
            check('submission cookie required', request('GET', '/submission/' + job_id)[0] == 403)
            check('own submission exact bytes', request('GET', '/submission/' + job_id, headers={'Cookie': cookie})[2] == payload)
            check('foreign job unavailable', request('GET', '/api/jobs/' + 'a' * 32)[0] == 404)
            check('filename never used as path', not (base / 'escape.webm').exists() and server.jobs[job_id]['_path'].name == 'submission.webm')
            fake.release.set()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                job = json.loads(request('GET', '/api/jobs/' + job_id)[2])
                if job['status'] in {'COMPLETE', 'FAILED'}:
                    break
                time.sleep(.02)
            check('completed fake engine once', job['status'] == 'COMPLETE' and fake.calls == 1)
            check('no educational verdict invented', job['result']['educational_verdict'] is None)
            check('source hash preserved', job['submission_sha256'] == hashlib.sha256(payload).hexdigest())
            check('session confinement', server.jobs[job_id]['_dir'].resolve().parent == server.session_dir.resolve())
        finally:
            fake.release.set()
            server.shutdown(); server.server_close(); thread.join(timeout=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='SERVER_SELFTEST.json')
    arguments = parser.parse_args()
    output = ROOT / arguments.output
    if output.resolve().parent != ROOT.resolve() or output.suffix != '.json':
        raise SystemExit('Selftest output must be a JSON file in this unit.')
    if output.exists():
        raise SystemExit('Preserve existing SELFTEST; explicit new output needed for a new revision.')
    error = None
    try:
        run()
    except BaseException:
        error = traceback.format_exc()
    files = ['server.py', 'index.html', 'app.js', 'server_selftest.py']
    report = {'status': 'PASS' if error is None and all(c['pass'] for c in checks) else 'FAIL',
              'checks': len(checks), 'results': checks, 'exception': error,
              'source_pins': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
              'scope': 'Synthetic HTTP contracts and static UI checks. Fake engine only; not browser rendering or real video/model integration.',
              'camera_access': 0, 'model_inference': 0, 'training': 0, 'user_data': 0}
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': report['status'], 'checks': report['checks'], 'output': str(output)}))
    if error:
        print(error); raise SystemExit(1)
