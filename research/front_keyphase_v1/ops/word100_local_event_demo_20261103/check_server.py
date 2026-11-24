"""Stdlib-only HTTP contract tests with synthetic bytes and an injected fake adapter."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import http.client
import importlib.util
import json
import sys
import tempfile
import threading

U = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_am_server_contract_test', U / 'run_demo.py')
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)


def fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / 'synthetic_public.mp4'
    video.write_bytes(b'synthetic bytes; never decoded as a real video')
    source_hash = server_module.sha(video)
    result = {'word_id': 'WORD1133', 'source_sha256': source_hash,
              'old_shape': {'value': [1, None]}, 'sequence_verdict': 'partial',
              'whole_action_complete': None, 'training': False}
    result_path = directory / 'public_result.json'
    server_module.write_new(result_path, result)
    (directory / 'demo.html').write_text('Research inference only', encoding='utf-8')
    (directory / 'demo.js').write_text('// fake UI', encoding='utf-8')
    assets = {'example': {'word_id': 'WORD1133', 'path': str(video), 'sha256': source_hash,
                         'result_path': str(result_path), 'result_sha256': server_module.sha(result_path),
                         'result': result},
              'references': {word: {'path': str(video), 'sha256': source_hash, 'person': 1}
                             for word in server_module.LABELS}}
    return assets


class FakeAdapter:
    WORDS = tuple(server_module.LABELS)
    def __init__(self, directory):
        self.U = directory
        self.warmups = 0
        self.closes = 0
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.mode = 'success'

    def warmup(self):
        self.warmups += 1
        if self.mode == 'warmup_fail':
            raise RuntimeError('Synthetic warmup failure')
        return {'synthetic_only': True, 'actual_model_loads': 0}

    def analyze(self, source, directory, word, condition, progress, *, provenance):
        self.calls.append({'source': str(source), 'directory': str(directory), 'word': word,
                           'condition': condition, 'provenance': deepcopy(provenance)})
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError('Test did not release fake analysis')
        directory.mkdir()
        progress({'stage': 'fake_analysis'})
        result = {'word_id': word, 'source_sha256': server_module.sha(source),
                  'sequence_verdict': 'unchanged', 'old_fields': [None, False, 1],
                  'whole_action_complete': None}
        if self.mode == 'word_mismatch':
            result['word_id'] = 'WORD2523' if word != 'WORD2523' else 'WORD1133'
        elif self.mode == 'source_mismatch':
            result['source_sha256'] = '0' * 64
        elif self.mode == 'job_mismatch':
            result['job_id'] = '0' * 32
        return result

    def close(self):
        self.closes += 1


def main():
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    native_names = ('numpy', 'torch', 'mediapipe', 'av', 'learner_adapter')
    check('server import did not load native models or decoder modules',
          not any(name in sys.modules for name in native_names))
    temporary_root = (U / 'server_test_tmp').resolve()
    if temporary_root.parent != U.resolve():
        raise ValueError('Temporary files must stay inside the assigned workspace unit')
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='am_fake_server_', dir=temporary_root) as temp:
        directory = Path(temp)
        assets = fixture(directory)
        adapter = FakeAdapter(directory)
        app = server_module.DemoApplication(adapter, public_assets=assets, port=0)
        server = app.create_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_headers = {'X-Session-Token': app.token, 'Origin': f'http://127.0.0.1:{app.port}',
                        'X-Video-Type': 'video/webm'}
        endpoint = '/api/analyze?word=WORD1133&condition=learned_motion'

        def request(method, path, body=None, headers=None, raw=False):
            connection = http.client.HTTPConnection('127.0.0.1', app.port, timeout=5)
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            payload = response.read()
            status, response_headers = response.status, dict(response.getheaders())
            connection.close()
            if not raw:
                payload = json.loads(payload)
            return status, payload, response_headers

        def post(path=endpoint, body=b'new synthetic upload', headers=None):
            return request('POST', path, body, base_headers if headers is None else headers)

        try:
            check('server binds loopback only', server.server_address[0] == '127.0.0.1')
            check('no warmup or analysis on construction', adapter.warmups == 0 and not adapter.calls)
            check('upload before ready rejected', post()[0] == 409)
            check('first initialization accepted', app.start_initialization())
            check('second initialization ignored', not app.start_initialization())
            app.init_thread.join(timeout=5)
            status, state, _ = request('GET', '/api/state')
            check('warmup exactly once', adapter.warmups == 1)
            check('state ready and research only', status == 200 and state['ready'] and state['research_inference_only'])
            check('state exposes current token and four labels', state['token'] == app.token and state['words'] == server_module.LABELS)
            check('cross-site state token read rejected', request('GET', '/api/state', headers={'Origin': 'https://elsewhere.invalid'})[0] == 403)
            check('foreign Host rejected', request('GET', '/api/state', headers={'Host': 'evil.invalid'})[0] == 403)
            check('cross-site fetch rejected', request('GET', '/api/state', headers={'Sec-Fetch-Site': 'cross-site'})[0] == 403)
            check('missing auth rejected', post(headers={})[0] == 403)
            check('wrong auth rejected', post(headers=dict(base_headers, **{'X-Session-Token': 'wrong'}))[0] == 403)
            check('foreign POST origin rejected', post(headers=dict(base_headers, Origin='https://elsewhere.invalid'))[0] == 403)
            check('unsupported word rejected', post('/api/analyze?word=WORD9999&condition=learned_motion')[0] == 400)
            check('unsupported condition rejected', post('/api/analyze?word=WORD1133&condition=train')[0] == 400)
            check('duplicate word rejected', post(endpoint + '&word=WORD0739')[0] == 400)
            check('extra target parameter rejected', post(endpoint + '&output=C:/x')[0] == 400)
            check('empty query rejected', post('/api/analyze')[0] == 400)
            check('new public analysis route absent', post('/api/analyze-public?word=WORD1133&condition=learned_motion')[0] == 404)
            check('empty upload rejected', post(body=b'')[0] == 413)
            check('over-limit request rejected before body read', post(body=b'', headers=dict(base_headers, **{'Content-Length': str(server_module.MAX_UPLOAD + 1)}))[0] == 413)
            check('negative length rejected', post(body=b'', headers=dict(base_headers, **{'Content-Length': '-1'}))[0] == 400)
            check('chunked upload rejected', post(body=b'', headers=dict(base_headers, **{'Transfer-Encoding': 'chunked'}))[0] == 400)
            for path in ('/media/session/../input.video', '/media/session/%2e%2e', '/media/session/a/b', '/api/job/short', '/api/public/WORD1133', '/api/example/WORD0739'):
                check('unsafe or nonexistent route: ' + path, request('GET', path)[0] in (400, 404))
            previous_jid = 'a' * 32
            previous = directory / 'private_sessions' / previous_jid
            previous.mkdir(parents=True)
            (previous / 'input.video').write_bytes(b'old synthetic private fixture must never be returned')
            check('old private job not discovered', request('GET', '/api/job/' + previous_jid)[0] == 404)
            check('old private media inaccessible', request('GET', '/media/session/' + previous_jid)[0] == 404)
            status, example, _ = request('GET', '/api/example/WORD1133')
            check('saved example returned', status == 200)
            check('saved example old fields unchanged', all(example[k] == v for k, v in assets['example']['result'].items()))
            check('saved example identity is explicit replay', example['ui_source_identity'] == {
                'word_id': 'WORD1133', 'case_kind': 'saved_public_replay',
                'media_url': '/media/example/WORD1133', 'source_sha256': assets['example']['sha256']})
            check('saved example does not invoke analysis', not adapter.calls)
            status, media, _ = request('GET', '/media/example/WORD1133', raw=True)
            check('saved media exactly matches source SHA', status == 200 and hashlib.sha256(media).hexdigest() == example['source_sha256'])
            for word in app.words:
                check('public ID01 reference allowlist: ' + word, request('GET', '/media/reference/' + word, raw=True)[0] == 200)
            status, piece, headers = request('GET', '/media/example/WORD1133', headers={'Range': 'bytes=2-7'}, raw=True)
            check('range returns original byte interval', status == 206 and piece == media[2:8] and headers['Content-Range'].startswith('bytes 2-7/'))
            check('bad range rejected', request('GET', '/media/example/WORD1133', headers={'Range': 'bytes=99-100'})[0] == 416)
            check('all invalid requests created no jobs', app.jobs == {} and not adapter.calls)

            # Block fake analysis until concurrency and pending-source identity are checked.
            upload = b'new explicit synthetic upload A'
            status, created, _ = post(body=upload)
            jid = created['id']
            check('accepted upload has 32hex job ID', status == 202 and server_module.JOB_RE.fullmatch(jid) is not None)
            check('fake analysis entered', adapter.started.wait(timeout=5))
            check('second simultaneous job rejected', post(body=b'B')[0] == 409)
            check('busy state while analysis pending', request('GET', '/api/state')[1]['busy'])
            pending = request('GET', '/api/job/' + jid)[1]
            check('pending identity matches selected word/hash', pending['word_id'] == 'WORD1133' and pending['source_sha256'] == hashlib.sha256(upload).hexdigest())
            check('current submitted media returned only by own job', request('GET', '/media/session/' + jid, raw=True)[1] == upload)
            check('one queued analysis only', len(adapter.calls) == 1 and len(app.jobs) == 1)
            call = adapter.calls[0]
            check('analysis path is exact current job', Path(call['directory']) == directory / 'private_sessions' / jid / 'analysis')
            check('input path is exact current job', Path(call['source']) == directory / 'private_sessions' / jid / 'input.video')
            check('inference-only provenance propagated', call['provenance'] == {
                'source_role': 'user_inference_only', 'source_sha256': hashlib.sha256(upload).hexdigest(),
                'word_id': 'WORD1133', 'training': False, 'tuning': False, 'candidate_selection': False})
            intent = server_module.read(directory / 'private_sessions' / jid / 'INFERENCE_ONLY.json')
            check('inference-only receipt saved', all(intent[k] is False for k in ('training', 'tuning', 'candidate_selection')))
            adapter.release.set()
            app.future.result(timeout=5)
            status, finished, _ = request('GET', '/api/job/' + jid)
            result = finished['result']
            check('successful job complete', status == 200 and finished['status'] == 'complete')
            check('result/job/source identity bound', result['ui_source_identity'] == {
                'word_id': 'WORD1133', 'case_kind': 'user_upload', 'media_url': '/media/session/' + jid,
                'source_sha256': hashlib.sha256(upload).hexdigest(), 'job_id': jid})
            check('old result fields preserved', result['old_fields'] == [None, False, 1] and result['sequence_verdict'] == 'unchanged')
            check('no action-completion promotion', result['whole_action_complete'] is None)
            check('busy clears after completion', not request('GET', '/api/state')[1]['busy'])
            for mode in ('word_mismatch', 'source_mismatch', 'job_mismatch'):
                adapter.mode = mode
                status, created, _ = post(body=('synthetic ' + mode).encode())
                app.future.result(timeout=5)
                failed = request('GET', '/api/job/' + created['id'])[1]
                check(mode + ' fails closed', status == 202 and failed['status'] == 'failed' and 'result' not in failed)
                check(mode + ' first failure preserved', (directory / 'private_sessions' / created['id'] / 'ANALYSIS_FIRST_FAILURE.txt').is_file())
            check('failed results not retried', len(adapter.calls) == 4)
            original = Path(assets['example']['path']).read_bytes()
            Path(assets['example']['path']).write_bytes(original + b'changed')
            check('changed example media rejected', request('GET', '/media/example/WORD1133')[0] == 409)
            check('changed example result pairing rejected', request('GET', '/api/example/WORD1133')[0] == 409)
            Path(assets['example']['path']).write_bytes(original)
        finally:
            adapter.release.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            app.close()
            app.close()
        check('adapter closed exactly once', adapter.closes == 1)
        check('closed app does not initialize again', not app.start_initialization())
        for label, change in (
            ('wrong example word', lambda a: a['example'].update(word_id='WORD0739')),
            ('wrong example source', lambda a: a['example'].update(sha256='0' * 64)),
            ('non-ID01 reference', lambda a: a['references']['WORD1133'].update(person=2)),
        ):
            bad = deepcopy(assets)
            change(bad)
            rejected = False
            try:
                server_module.DemoApplication(FakeAdapter(directory), public_assets=bad, port=0)
            except ValueError:
                rejected = True
            check(label + ' rejected during initialization', rejected)
        failed_dir = directory / 'warmup_failure'
        failed_assets = fixture(failed_dir)
        failed_adapter = FakeAdapter(failed_dir)
        failed_adapter.mode = 'warmup_fail'
        failed_app = server_module.DemoApplication(failed_adapter, public_assets=failed_assets, port=0)
        try:
            failed_app.start_initialization()
            failed_app.init_thread.join(timeout=5)
            check('warmup failure stays unready', not failed_app.state['ready'])
            check('warmup failure preserved', (failed_dir / 'MODEL_INITIALIZATION_FIRST_FAILURE.txt').is_file())
            check('warmup failure not automatically retried', not failed_app.start_initialization() and failed_adapter.warmups == 1)
        finally:
            failed_app.close()
    check('no native inference/decoder imported by tests', not any(name in sys.modules for name in native_names))
    report = {'status': 'PASS_FAKE_SERVER_CONTRACT', 'check_count': len(checks), 'checks': checks,
              'actual_model_loads': 0, 'actual_model_forwards': 0, 'actual_video_decodes': 0,
              'actual_camera_access': 0, 'actual_user_submissions_read': 0,
              'old_private_paths_read': 0, 'old_baseline_reruns': 0,
              'test_scope': 'Synthetic files and injected fake adapter; temporary loopback HTTP only',
              'not_tested': ['real browser/video decoding', 'actual inference latency or accuracy', 'actual model startup'],
              'code_pins': {str(path.resolve()): server_module.sha(path) for path in (U / 'run_demo.py', Path(__file__))}}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
