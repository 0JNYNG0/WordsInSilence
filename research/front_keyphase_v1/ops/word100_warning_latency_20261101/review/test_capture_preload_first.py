"""Startup routing and import-only verification; no models, video or server run."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast, hashlib, importlib.util, json, sys

U = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def run():
    m = load('_preload_test_wrapper', U/'run_server.py')
    candidate = load('_preload_test_candidate', U/'candidate_adapter.py')
    checks = []
    def check(name, value):
        if not value: raise AssertionError(name)
        checks.append(name)
    adapter_sha = sha(U/'candidate_adapter.py')
    check('candidate adapter remains exact frozen implementation', adapter_sha == 'c5891dd0dd19a02724b7084ee7b3015337568d90ccb2d2ee17d3581cccbe4098')
    expected = {
        'wis_lab.capture.face_landmarks': candidate.LAB/'src/wis_lab/capture/face_landmarks.py',
        'wis_lab.core.observation': candidate.LAB/'src/wis_lab/core/observation.py',
        '_live_raw_capture': candidate.OPS/'word100_hand_assignment_20261016/capture_raw.py',
        '_live_joint_assignment': candidate.OPS/'word100_hand_assignment_joint_20261016/assignment.py'}
    calls = []
    def fake_import(name):
        calls.append(name)
        return SimpleNamespace(__file__=str(expected[name]))
    def fake_local(name, path):
        check('exact local dependency path '+name, path == expected[name])
        return fake_import(name)
    original_warm = {'status':'WARM', 'total_seconds':2.5, 'fake':True}
    def fake_warm(): calls.append('observer_warmup'); return dict(original_warm)
    fake = SimpleNamespace(LAB=candidate.LAB, OPS=candidate.OPS, _module=fake_local, warmup=fake_warm)
    with patch.object(m.importlib, 'import_module', fake_import):
        result = m.startup_warmup(fake)
    check('observer warmup precedes four capture imports', calls == ['observer_warmup']+list(expected))
    check('observer warmup original fields preserved', all(result[k] == v for k,v in original_warm.items()))
    check('capture import timing separate', result['capture_import_preload']['seconds'] >= 0 and result['startup_warmup_wrapper_seconds'] >= result['capture_import_preload']['seconds'])
    check('historical gap explicitly not measured import duration', 'not an exact import' in result['historical_gap_note'])
    check('all exact source hashes recorded', all(result['capture_import_preload']['modules'][k]['sha256'] == sha(p) for k,p in expected.items()))
    with patch.object(m.importlib, 'import_module', lambda _:SimpleNamespace(__file__=str(U/'run_server.py'))):
        try: m.preload_capture_modules(fake)
        except ValueError: check('wrong import route rejected', True)
        else: raise AssertionError('wrong import accepted')
    with patch.object(m, 'preload_capture_modules', side_effect=RuntimeError('synthetic import failure')):
        try: m.startup_warmup(fake)
        except RuntimeError: check('preload failure blocks server readiness', True)
        else: raise AssertionError('preload failure swallowed')
    tree = ast.parse((U/'run_server.py').read_text(encoding='utf8'))
    main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    server_call = next(n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='create_server')
    check('server receives startup wrapper callback', ast.unparse(next(k.value for k in server_call.keywords if k.arg=='warmup')) == 'lambda: startup_warmup(candidate)')
    for name in ['preload_capture_modules','startup_warmup']:
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        text = ast.unparse(node)
        check(name+' has no detector/video/camera invocation', all(s not in text for s in ['.process(', 'VideoCapture(', '.analyze_video(', '.extract(', '.serve_forever(']))
    # Real module import smoke. Python call tracer rejects all request-time
    # capture/model/assignment constructors and decode functions before execution.
    blocked_names = {'__init__','process','extract','extract_face_video','assign','run','prepare'}
    protected = {str(p.resolve()).lower() for p in expected.values()}
    protected.add(str((candidate.LAB/'src/wis_lab/capture/landmarker.py').resolve()).lower())
    protected.add(str((candidate.LAB/'src/wis_lab/capture/video_source.py').resolve()).lower())
    forbidden = []
    def trace(frame,event,arg):
        if event=='call' and str(Path(frame.f_code.co_filename).resolve()).lower() in protected and frame.f_code.co_name in blocked_names:
            forbidden.append(frame.f_code.co_name)
            raise AssertionError('Import attempted request operation: '+frame.f_code.co_name)
    previous = sys.getprofile()
    sys.setprofile(trace)
    try: real = m.preload_capture_modules(candidate)
    finally: sys.setprofile(previous)
    check('real imports execute no guarded request operations', not forbidden)
    check('real imports route all four dependencies exactly', set(real['modules'])==set(expected))
    identities = {k:id(sys.modules[k]) for k in expected}
    again = m.preload_capture_modules(candidate)
    check('repeated import reuses exact module objects', identities == {k:id(sys.modules[k]) for k in expected})
    check('module reimport never shares VIDEO request state', again['request_VIDEO_state_shared'] is False)
    check('candidate source unchanged after imports', sha(U/'candidate_adapter.py') == adapter_sha)
    preserved = json.loads((U/'before_capture_preload/PRESERVED.json').read_text(encoding='utf8'))
    project = candidate.ROOT
    for path,pin in preserved['pins'].items():
        original = project/path
        if original.resolve() == (U/'run_server.py').resolve():
            check('old wrapper preserved exact', sha(U/'before_capture_preload/run_server.py')==pin)
        else: check('first measurement preserved '+original.name, sha(original)==pin)
    return dict(status='PASS_IMPORT_ONLY', checks=len(checks), results=checks,
        wrapper_sha256=sha(U/'run_server.py'),test_sha256=sha(__file__),adapter_sha256=adapter_sha,
        real_import_smoke=real, repeat_import_seconds=again['seconds'],
        real_detector_instances=0,real_model_forwards=0,real_video_reads=0,server_starts=0,
        limitation='Import smoke timings are not request latency or a matched performance benchmark; root owns the later fresh request comparison.')

if __name__=='__main__':
    result=run()
    with (U/'review/CAPTURE_PRELOAD_SELFTEST.json').open('x',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':result['status'],'checks':result['checks'],'wrapper_sha256':result['wrapper_sha256']}))
