"""Owned localhost demo browser verification; public inputs only, no camera.

Reuses the established headless Chrome/CDP test pattern. One real public
analysis is permitted. The file-upload branch is intercepted in-browser and
never POSTs footage or writes private_sessions. No old result is re-inferred.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, time, subprocess, urllib.request, websocket, base64, uuid, traceback, hashlib, sys
import numpy as np

U = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:8808'
WORDS = ['WORD0739', 'WORD1133', 'WORD1567', 'WORD2523']
IDS = ['word', 'file', 'camera', 'public', 'publicCut', 'publicOrder', 'publicFresh']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    out = U / 'SOURCE_DEMO_BROWSER_VERIFIED.json'
    intent = U / 'FRESH_PUBLIC_BROWSER_INTENT.json'
    assert not out.exists()
    resume = '--resume-existing-public-job' in sys.argv
    prior_intent = read(intent) if intent.exists() else None
    if resume:
        assert prior_intent and prior_intent.get('job_id'), 'Explicit resume requires a submitted public job.'
    else:
        assert not intent.exists(), 'A fresh attempt already exists: inspect its job, never repeat automatically.'
    launch = read(U / 'DEMO_LAUNCH.json')
    assert launch['url'] == BASE + '/'
    assert all((U / name).stat().st_mtime <= (U / 'DEMO_LAUNCH.json').stat().st_mtime
               for name in ['run_demo.py', 'learner_adapter.py', 'source_bridge.py'])
    state = json.load(urllib.request.urlopen(BASE + '/api/state', timeout=10))
    assert state['ready'] and not state['busy']
    profile = U / ('source_ui_check_' + uuid.uuid4().hex[:8])
    profile.mkdir()
    log = (profile / 'chrome.log').open('w', encoding='utf-8')
    proc = subprocess.Popen([
        'C:/Program Files/Google/Chrome/Application/chrome.exe', '--headless=new',
        '--remote-debugging-port=0', '--remote-allow-origins=http://127.0.0.1',
        '--disable-background-networking', '--disable-component-update', '--no-first-run',
        '--autoplay-policy=no-user-gesture-required', '--user-data-dir=' + str(profile),
        '--window-size=1400,1100', 'about:blank'], stdout=subprocess.DEVNULL,
        stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    ws = None
    errors, records, checks = [], [], []
    fresh_job = None
    try:
        tab = None
        for _ in range(150):
            try:
                port = int((profile / 'DevToolsActivePort').read_text().splitlines()[0])
                tab = next(t for t in json.load(urllib.request.urlopen(
                    f'http://127.0.0.1:{port}/json/list', timeout=2)) if t['type'] == 'page')
                break
            except (OSError, ValueError, StopIteration):
                time.sleep(.1)
        assert tab
        ws = websocket.create_connection(tab['webSocketDebuggerUrl'], origin='http://127.0.0.1', timeout=30)
        seq = 0

        def call(method, params=None):
            nonlocal seq
            seq += 1
            ws.send(json.dumps(dict(id=seq, method=method, params=params or {})))
            while True:
                d = json.loads(ws.recv())
                if d.get('method') == 'Runtime.exceptionThrown':
                    errors.append(d)
                if d.get('id') == seq:
                    assert not d.get('error'), d
                    return d.get('result', {})

        def js(code):
            d = call('Runtime.evaluate', dict(expression=code, returnByValue=True, awaitPromise=True))
            assert not d.get('exceptionDetails'), d
            return d.get('result', {}).get('value')

        def wait(code, seconds=20):
            start = time.monotonic()
            while time.monotonic() - start < seconds:
                if js(code):
                    return
                time.sleep(.1)
            raise AssertionError(('timeout', code, js('$(' + json.dumps('progress') + ').textContent')))

        def locked():
            return js('busy && ' + json.dumps(IDS) + '.every(id=>$(id).disabled)')

        call('Runtime.enable')
        call('Page.enable')
        call('Page.addScriptToEvaluateOnNewDocument', {'source': r"""
window.__audit={network:[],stubbedUploads:0,cameraCalls:0,delaySaved:false,mismatchSaved:false,freshSubmission:null};
const auditFetch=window.fetch.bind(window);
window.fetch=async function(input,options={}){
 const u=new URL(typeof input==='string'?input:input.url,location.href);const method=(options.method||'GET').toUpperCase();
 if(u.pathname==='/api/analyze'){
  __audit.stubbedUploads++;__audit.stubbedUploadType=options.body?.type;
  return new Response(JSON.stringify({error:'PUBLIC_BLOB_PREVIEW_ONLY_NO_UPLOAD'}),{status:409,headers:{'Content-Type':'application/json'}});
 }
 __audit.network.push({path:u.pathname,method});
 if(__audit.delaySaved&&/^\/api\/(public|control|order)\//.test(u.pathname))await new Promise(r=>setTimeout(r,350));
 const response=await auditFetch(input,options);
 if(u.pathname==='/api/analyze-public'&&method==='POST'){
  __audit.freshSubmission=await response.clone().json();
 }
 if(__audit.mismatchSaved&&u.pathname.startsWith('/api/public/')){
  const v=await response.json();v.word_id='WORD2523';
  return new Response(JSON.stringify(v),{status:200,headers:{'Content-Type':'application/json'}});
 }
 return response;
};
if(navigator.mediaDevices)navigator.mediaDevices.getUserMedia=async()=>{__audit.cameraCalls++;throw Error('CAMERA_FORBIDDEN_IN_PUBLIC_TEST');};
"""})
        call('Page.navigate', dict(url=BASE))
        wait('typeof ready!=="undefined"&&ready', 30)
        assert js('stream===null && __audit.cameraCalls===0')
        js("document.querySelectorAll('video').forEach(v=>v.muted=true)")
        checks.append('Server ready, declared GPU warmup; camera API never used.')

        for word in WORDS:
            js("$('word').value=" + json.dumps(word) + ";setWord();__audit.delaySaved=true")
            for kind, button, route, folder in [
                ('normal', 'public', 'public', 'demo_public'),
                ('cut', 'publicCut', 'control', 'demo_controls'),
                ('order', 'publicOrder', 'order', 'demo_order')]:
                js("$('" + button + "').click()")
                assert locked(), (word, kind, 'controls not locked')
                wait("!busy&&result!==null&&result.word_id===" + json.dumps(word))
                js("$('query').load();$('reference').load()")
                wait("$('query').readyState>=2&&$('reference').readyState>=2&&$('query').currentSrc.endsWith('/media/" + route + "/" + word + "')&&$('reference').currentSrc.endsWith('/media/reference/" + word + "')")
                r = js('result')
                saved = read(U / folder / word / 'RESULT_v2.json')
                assert r['source_components'] == saved['source_components']
                assert r['path'] == saved['path']
                assert r['ui_result_provenance'] == 'saved_public_replay'
                assert r['latency'].get('cached_example_not_new_end_to_end_measurement', True) is True
                assert js("$('result').innerText.includes('저장된 공개 결과 · 이번에 새로 측정한 대기시간은 아닙니다.')")
                assert r['individual_reference_comparisons'] == 0 and r['learned_prototype_comparisons'] == 1
                assert len(r['overlay']['points']) == len(r['overlay']['times']) == r['frames']
                displayed = [s for s in r['source_components'] if s['status'] not in
                             ['COMPONENT_NOT_RUNTIME_VALIDATED', 'EXACT_SCOPE_OBSERVER_UNAVAILABLE']]
                assert js('document.querySelectorAll(".source-cards > .stage").length') == len(displayed)
                assert all(js("$('result').innerText.includes(" + json.dumps(c['text']) + ")")
                           for s in displayed for c in s['source_clauses'])
                if not displayed:
                    assert js("[...document.querySelectorAll('#result > details')].some(d=>d.open&&d.querySelector('summary').textContent==='기존 손모양·관계·이동 분석 전체 보기')")
                expected = displayed[0]['start_seconds'] if displayed else r['stages'][0]['start_seconds']
                js("(document.querySelector('.source-cards .stage button')||document.querySelector('#result details .stage button')).click()")
                wait("!$('query').seeking&&$('query').readyState>=2")
                time.sleep(.06)
                js("$('query').pause()")
                actual = js("$('query').currentTime")
                assert abs(actual - expected) < .3, (word, kind, actual, expected)
                blue = js("[...document.querySelectorAll('.source-estimated')].map(e=>({color:getComputedStyle(e).backgroundColor,label:e.getAttribute('aria-label')}))")
                assert all(x['color'] == 'rgb(84, 127, 177)' and '추정' in x['label'] for x in blue)
                records.append(dict(word=word, kind=kind, source_cards=len(displayed),
                    source_components=len(r['source_components']), decoded_query=True, decoded_reference=True,
                    replay_expected_seconds=expected, replay_actual_seconds=actual,
                    latency=r['latency'], saved_latency_label=True, source_probability_and_path_unchanged=True,
                    image_time_estimate_runs=len(blue), source_links=js("[...document.querySelectorAll('.source-cards a')].map(a=>a.href)")))
                print('saved', word, kind, 'verified', flush=True)
        js('__audit.delaySaved=false')
        checks.append('12 saved normal/cut/order displays preserve stored outputs; both videos decode; native-time replay and saved-latency labels checked.')

        # Verify word identity rejection through a local fetch fixture, not model inference.
        js("$('word').value='WORD1133';setWord();$('public').click()")
        wait("!busy&&result?.word_id==='WORD1133'")
        previous = js("({word:result.word_id,src:$('query').src})")
        js("__audit.mismatchSaved=true;$('public').click()")
        wait('!busy')
        assert js("$('progress').textContent.includes('표현이 일치하지 않아')")
        assert js("({word:result.word_id,src:$('query').src})") == previous
        js('__audit.mismatchSaved=false')
        checks.append('Mismatched-word response rejected; previous result/video retained.')

        # Public Blob preview restores upload preview after a saved video replaces it.
        js("(async()=>{const r=await fetch('/media/public/WORD1133');setBlob(await r.blob());})()")
        assert js('blob!==null && objectUrl!==null')
        js("$('public').click()")
        wait("!busy&&$('query').src.endsWith('/media/public/WORD1133')")
        js("$('analyze').click()")
        wait('!busy')
        assert js("$('query').src.startsWith('blob:')&&$('progress').textContent==='PUBLIC_BLOB_PREVIEW_ONLY_NO_UPLOAD'")
        assert js("__audit.stubbedUploads===1&&__audit.network.filter(r=>r.path==='/api/analyze'&&r.method==='POST').length===0")
        wait("$('query').readyState>=2")
        checks.append('Public Blob preview restored after saved playback; upload branch stubbed locally, zero actual upload POSTs/private writes.')

        # Ensure estimate colour can be tested even if no estimate occurs in the 12 saved cases.
        estimate_fixture = False
        if not sum(r['image_time_estimate_runs'] for r in records):
            js("window.__fixture=JSON.parse(JSON.stringify(result||{}))")
            # Reload only a stored public result, then change its UI copy; no probability claim.
            js("$('public').click()")
            wait("!busy&&result?.word_id==='WORD1133'")
            js("const f=JSON.parse(JSON.stringify(result));const c=f.source_components.find(c=>c.point_evidence?.length);c.point_evidence[0].status='IMAGE_TIME_ESTIMATE';c.point_evidence[0].evidence='RGB_TIME_ESTIMATE';render(f,'색상 표시만 확인하는 시험용 복사본');")
            assert js("[...document.querySelectorAll('.source-estimated')].some(e=>getComputedStyle(e).backgroundColor==='rgb(84, 127, 177)'&&e.title.includes('추정'))")
            estimate_fixture = True
        checks.append('RGB/time estimates shown in blue with separate estimate text, not green direct confirmation.')

        # Single real fresh request, durably marked before clicking. Never retry this POST.
        js("$('word').value='WORD1133';setWord();__audit.delaySaved=false")
        if resume:
            fresh_job = prior_intent['job_id']
            job = json.load(urllib.request.urlopen(BASE + '/api/job/' + fresh_job, timeout=10))
            assert job['status'] == 'complete' and job['public_only'] is True
            js("clearStream();$('query').src='/media/public/WORD1133';render(" + json.dumps(job['result']) + ",'완료된 공개 새 분석 결과 · UI 재검증, 재추론 없음')")
            fresh_elapsed = None  # Earlier button-wall value was not serialized; never invent it.
        else:
            started = time.monotonic()
            write(intent, dict(status='ONE_FRESH_BUTTON_CLICK_AUTHORIZED', word='WORD1133',
                public_only=True, camera=False, training=False, at_utc=datetime.now(timezone.utc).isoformat()))
            js("$('publicFresh').click()")
            assert locked()
            for _ in range(2400):
                submitted = js('__audit.freshSubmission')
                if submitted and submitted.get('id') and fresh_job is None:
                    fresh_job = submitted['id']
                    write(intent, dict(status='SUBMITTED_DO_NOT_REPEAT', job_id=fresh_job,
                        word='WORD1133', public_only=True, training=False,
                        analysis_directory=str(U / 'public_runtime_runs' / fresh_job / 'analysis')))
                    print('fresh public job', fresh_job, flush=True)
                if js('!busy'):
                    break
                time.sleep(.1)
            fresh_elapsed = time.monotonic() - started
        assert fresh_job and js("result!==null&&result.word_id==='WORD1133'&&!busy"), js("$('progress').textContent")
        r = js('result')
        job = json.load(urllib.request.urlopen(BASE + '/api/job/' + fresh_job, timeout=10))
        assert job['status'] == 'complete' and job['public_only'] is True
        assert r == job['result']
        assert r['training'] is False and r['tuning'] is False
        assert r['individual_reference_comparisons'] == r['runtime_individual_reference_video_comparisons'] == 0
        assert r['learned_prototype_comparisons'] == 1
        assert r['source_component_contract']['new_inference'] == 0
        assert r['source_component_contract']['new_training'] == 0
        assert not r['latency'].get('cached_example_not_new_end_to_end_measurement', False)
        assert js("$('result').innerText.includes('영상 분석부터 결과까지')")
        assert js("__audit.network.filter(r=>r.path==='/api/analyze-public'&&r.method==='POST').length") == (0 if resume else 1)
        assert js('stream===null&&__audit.cameraCalls===0')
        analysis = Path(job['analysis_directory'])
        with np.load(analysis / 'analysis.npz', allow_pickle=False) as a:
            frames, times = a['frame_id'], a['timestamp_us']
            assert len(frames) == r['frames']
            assert np.all(np.diff(frames) > 0) and np.all(np.diff(times) > 0)
            assert np.allclose(np.asarray(r['overlay']['times']), times / 1e6, atol=1e-8, rtol=0)
            frame_time = dict(zip(frames.tolist(), (times / 1e6).tolist()))
            for component in r['source_components']:
                assert abs(component['start_seconds'] - frame_time[component['frame_ids'][0]]) < 1e-8
                assert abs(component['end_seconds'] - frame_time[component['frame_ids'][-1]]) < 1e-8
        source = U.parent / 'word100_learned_feature_retention_20261102/video_controls/WORD1133_reencoded_normal.mp4'
        assert r['source_sha256'] == sha(source)
        wait("$('query').readyState>=2&&$('reference').readyState>=2")
        js('scrollTo(0,0)')
        screen_name = 'SOURCE_DEMO_SCREEN_UI_RECHECK.png' if resume else 'SOURCE_DEMO_SCREEN.png'
        result_name = 'SOURCE_DEMO_RESULT_UI_RECHECK.png' if resume else 'SOURCE_DEMO_RESULT.png'
        (U / screen_name).write_bytes(base64.b64decode(call('Page.captureScreenshot', {'format': 'png'})['data']))
        js("$('result').scrollIntoView()")
        (U / result_name).write_bytes(base64.b64decode(call('Page.captureScreenshot', {'format': 'png'})['data']))
        colors = js("[...document.querySelectorAll('h1,h2,h3,p,button,select')].filter(e=>e.getClientRects().length).map(e=>[getComputedStyle(e).color,getComputedStyle(e).backgroundColor])")
        assert all(a != b for a, b in colors)
        call('Emulation.setDeviceMetricsOverride', dict(width=700, height=1000, deviceScaleFactor=1, mobile=False))
        assert js('document.documentElement.scrollWidth<=innerWidth+2')
        assert not errors, errors
        checks.append('One real public fresh video completed; source hash/native times/GPU/zero individual references/training false verified.')
        checks.append('No JS exceptions; visible text contrast; 700px layout without horizontal overflow.')
        write(out, dict(status='PASS', checks=checks, saved_cases=records, fresh=dict(
            job_id=fresh_job, analysis_directory=str(analysis), source=str(source), source_sha256=r['source_sha256'],
            frames=r['frames'], wall_seconds_from_button=fresh_elapsed, latency=r['latency'], device=r['device'],
            runtime_individual_reference_video_comparisons=0, learned_prototype_comparisons=1,
            source_components=[dict(target=c['component_id'],status=c['status'],native_frames=c['frame_ids']) for c in r['source_components']],
            training=False, tuning=False, native_times_preserved=True,
            source_feedback_sha256=sha(analysis / 'SOURCE_FEEDBACK.json')),
            actual_fresh_public_posts_across_attempts=1, current_browser_fresh_posts=0 if resume else 1,
            completed_public_job_reused_for_ui_only=resume,
            actual_upload_posts=0, local_stubbed_upload_calls=1,
            estimate_ui_fixture_used=estimate_fixture, camera_calls=0, stream_null=True,
            private_footage_access=False, private_upload_write=False, old_models_changed=False,
            test_code_sha256=sha(__file__), server_launch=launch,
            screenshots=[str(U/screen_name),str(U/result_name)],
            prior_harness_corrections=read(U/'BROWSER_HARNESS_CORRECTIONS.json') if resume else None,
            limits=['Browser/integration evidence, not independent KSL educational correctness.',
                    'Saved12 controls are displayed without re-inference; their label does not turn every transformed clip into a linguistic error.',
                    'Source-linked partial observations do not validate completion, exact contact, full order, or mandatory repeats.',
                    'Fresh timing is one GPU-assisted public run on this laptop, not deployment latency distribution.'],
            whole_goal_complete=False))
        write(intent, dict(status='COMPLETE_DO_NOT_REPEAT', job_id=fresh_job, verification=str(out)))
        print('PASS', 'fresh_total',r['latency']['total_analysis_seconds'],'device',r['device'],flush=True)
    finally:
        if ws:
            try:
                call('Browser.close')
            except Exception:
                pass
            ws.close()
        if proc.poll() is None:
            proc.terminate()
        log.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        first = U / 'SOURCE_DEMO_BROWSER_FIRST_FAILURE.txt'
        if not first.exists():
            first.write_text(traceback.format_exc(), encoding='utf-8')
        raise
