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
BASE = 'http://127.0.0.1:8809'
WORDS = ['WORD0739', 'WORD1133', 'WORD1567', 'WORD2523']
IDS = ['word', 'condition', 'file', 'camera', 'public', 'publicCut', 'publicOrder', 'publicGap', 'publicFresh', 'analyze', 'record', 'stop']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    out = U / 'LIVE_ORDER_BROWSER_VERIFIED.json'
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
               for name in ['run_demo.py', 'learner_adapter.py', 'order_worker.py', 'demo.html', 'demo.js'])
    state = json.load(urllib.request.urlopen(BASE + '/api/state', timeout=10))
    assert state['ready'] and not state['busy']
    profile = U / ('order_ui_check_' + uuid.uuid4().hex[:8])
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
window.__audit={network:[],stubbedUploads:0,cameraCalls:0,delaySaved:false,mismatchSaved:null,freshSubmission:null,releaseFreshResponse:false};
const auditFetch=window.fetch.bind(window);
window.fetch=async function(input,options={}){
 const u=new URL(typeof input==='string'?input:input.url,location.href);const method=(options.method||'GET').toUpperCase();
 if(u.pathname==='/api/analyze'){
  __audit.stubbedUploads++;__audit.stubbedUploadType=options.body?.type;
  return new Response(JSON.stringify({error:'PUBLIC_BLOB_PREVIEW_ONLY_NO_UPLOAD'}),{status:409,headers:{'Content-Type':'application/json'}});
 }
 __audit.network.push({path:u.pathname,method});
 if(__audit.delaySaved&&/^\/api\/(public|control|order|gap)\//.test(u.pathname))await new Promise(r=>setTimeout(r,350));
 const response=await auditFetch(input,options);
 if(u.pathname==='/api/analyze-public'&&method==='POST'){
  __audit.freshSubmission=await response.clone().json();
  if(__audit.freshSubmission.id)while(!__audit.releaseFreshResponse)await new Promise(r=>setTimeout(r,50));
 }
 if(__audit.mismatchSaved&&u.pathname.startsWith('/api/public/')){
  const v=await response.json();
  if(__audit.mismatchSaved==='word')v.word_id='WORD2523';
  if(__audit.mismatchSaved==='hash')v.ui_source_identity.source_sha256='0'.repeat(64);
  if(__audit.mismatchSaved==='kind')v.ui_source_identity.case_kind='gap';
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
        checks.append('Server ready; camera API never used.')
        manifest=read(U/'SAVED_CASES_MANIFEST.json')

        def load_saved(word,kind,button,route):
            js("$('word').value="+json.dumps(word)+";setWord();__audit.delaySaved=true")
            js("$('"+button+"').click()")
            assert locked(),(word,kind,'controls not locked')
            wait("!busy&&result?.word_id==="+json.dumps(word)+"&&result?.ui_source_identity?.case_kind==="+json.dumps(kind))
            wait("$('query').readyState>=2&&$('reference').readyState>=2&&$('query').currentSrc.endsWith('/media/"+route+"/"+word+"')")
            return js('result')

        for word in WORDS:
            for kind,button,route in [('normal','public','public'),('cut','publicCut','control'),('order','publicOrder','order'),('gap','publicGap','gap')]:
                r=load_saved(word,kind,button,route)
                entry=manifest['cases'][word][kind]; saved=read(entry['result'])
                assert sha(entry['result'])==entry['result_sha256']
                assert {k:r[k] for k in saved}==saved
                assert r['source_sha256']==r['ui_source_identity']['source_sha256']==entry['video_sha256']
                assert r['ui_source_identity']['media_url']==entry['media_url']
                assert r['individual_reference_comparisons']==0 and r['learned_prototype_comparisons']==1
                partial=r['partial_order_feedback']
                assert js("document.querySelectorAll('#partialOrder .order-card').length")==len(partial['items'])
                assert js("document.querySelector('.total-latency').textContent.includes('이번에 새로 측정한 대기시간은 아닙니다.')")
                assert js("$('partialOrder').innerText.includes('전체 행동 완료 · 동작 생략 · 필수 반복 횟수: 미확인')")
                buttons=js("[...document.querySelectorAll('#partialOrder .order-replay')].map(b=>({start:Number(b.dataset.start),end:Number(b.dataset.end),side:b.dataset.side}))")
                expected=[]
                for item in partial['items']:
                    assert item['completion_unknown'] and item['education_error'] is None
                    for side in ('first','second'):
                        evidence=item[side+'_evidence'];times=evidence['native_seconds']
                        if times:expected.append(dict(start=times[0],end=times[-1],side='A' if side=='first' else 'B'))
                assert buttons==expected
                replay=[]
                for i,button_info in enumerate(buttons):
                    js("document.querySelectorAll('#partialOrder .order-replay')["+str(i)+"].click()")
                    wait("!$('query').seeking&&$('query').readyState>=2")
                    time.sleep(.06);js("$('query').pause()")
                    actual=js("$('query').currentTime")
                    assert abs(actual-button_info['start'])<.3,(word,kind,i,actual,button_info)
                    replay.append(dict(**button_info,actual=actual))
                missing=partial['global_missing_image_frames']
                if missing:assert js("document.querySelectorAll('#partialOrder .order-gap').length")>0
                if not partial['query_count']:
                    assert partial['status']=='NO_DISTINCT_ORDER_QUERY'
                    assert js("$('partialOrder').innerText.includes('반복 과정·반복 횟수: 아직 검증하지 않았습니다.')")
                if word=='WORD1133' and kind=='cut':
                    assert partial['items'][0]['observation_status']=='OBSERVED_A_BEFORE_B'
                    assert partial['items'][0]['completion_unknown']
                if word=='WORD0739' and kind=='order':
                    assert [x['observation_status'] for x in partial['items']].count('OBSERVED_B_BEFORE_A')==2
                    assert js("$('partialOrder').innerText.includes('차이 후보')")
                displayed=[x for x in r.get('source_components',[]) if x['status'] not in ['COMPONENT_NOT_RUNTIME_VALIDATED','EXACT_SCOPE_OBSERVER_UNAVAILABLE']]
                assert js("document.querySelectorAll('.source-cards > .stage').length")==len(displayed)
                if not displayed:
                    assert js("[...document.querySelectorAll('#result > details')].some(d=>d.open&&d.querySelector('summary').textContent==='기존 손모양·관계·이동 분석 전체 보기')")
                if r.get('overlay'):
                    assert len(r['overlay']['points'])==len(r['overlay']['times'])==r['frames']
                records.append(dict(word=word,kind=kind,decoded_query=True,decoded_reference=True,
                    source_sha256=r['source_sha256'],saved_result_exact=True,queries=len(partial['items']),
                    statuses=[x['observation_status'] for x in partial['items']],native_replay=replay,
                    missing_frames=missing,overlay_available=bool(r.get('overlay')),source_cards=len(displayed),
                    saved_latency_label=True))
                write(U/'LIVE_BROWSER_PROGRESS.json',dict(saved_cases_checked=len(records),fresh_job_id=fresh_job,records=records))
                print('saved verified',word,kind,flush=True)
        js('__audit.delaySaved=false')
        checks.append('16 cached cases preserve saved fields and source identity; both videos decode, partial-order native replay checked.')

        load_saved('WORD1133','normal','public','public');js('__audit.delaySaved=false')
        previous=js("({word:result.word_id,src:$('query').src,hash:result.source_sha256})")
        for mismatch in ['word','hash','kind']:
            js('__audit.mismatchSaved='+json.dumps(mismatch)+";$('public').click()")
            wait('!busy')
            assert js("$('progress').textContent.includes('표시를 중단했습니다.')")
            assert js("({word:result.word_id,src:$('query').src,hash:result.source_sha256})")==previous
            js('__audit.mismatchSaved=null')
        assert js("(()=>{try{guardResultIdentity(result,{word:'WORD1133',kind:'normal',serial:requestSerial,job:'WRONG_JOB'});return false}catch(e){return true}})()")
        checks.append('Word, source SHA, case kind and job identity mismatches rejected without replacing the previous video/result.')

        js("(async()=>{const r=await fetch('/media/public/WORD1133');setBlob(await r.blob());})()")
        assert js('blob!==null&&objectUrl!==null')
        js("$('public').click()");wait("!busy&&result?.word_id==='WORD1133'")
        js("$('analyze').click()");wait('!busy')
        assert js("$('query').src.startsWith('blob:')&&$('progress').textContent==='PUBLIC_BLOB_PREVIEW_ONLY_NO_UPLOAD'")
        assert js("__audit.stubbedUploads===1&&__audit.network.filter(x=>x.path==='/api/analyze'&&x.method==='POST').length===0")
        wait("$('query').readyState>=2")
        checks.append('Public video Blob restores upload preview and hashes it locally; upload POST intercepted, private writes zero.')

        load_saved('WORD1133','normal','public','public');js('__audit.delaySaved=false')
        js("const fixture=JSON.parse(JSON.stringify(result));const c=fixture.source_components.find(c=>c.point_evidence?.length);c.point_evidence[0].status='FEATURE_OBSERVED';c.point_evidence[0].evidence='RGB_TIME_ESTIMATE';render(fixture,'추정 색상 표시만 확인하는 복사본');")
        assert js("[...document.querySelectorAll('.source-estimated')].some(e=>getComputedStyle(e).backgroundColor==='rgb(84, 127, 177)'&&e.title.includes('추정'))")
        checks.append('UI-only uncertainty fixture displays RGB_TIME_ESTIMATE in blue, separate from green observed features.')

        # Exactly one public POST. Its response is held in the browser until
        # Python saves the job ID, so client polling cannot precede the marker.
        word='WORD0739';js("$('word').value='WORD0739';setWord();__audit.delaySaved=false")
        if resume:
            fresh_job=prior_intent['job_id']
            job=json.load(urllib.request.urlopen(BASE+'/api/job/'+fresh_job,timeout=10))
            assert job['status']=='complete' and job['public_only'] is True
            js("clearStream();$('query').src='/media/public/WORD0739';render("+json.dumps(job['result'])+",'완료된 공개 분석 결과 · UI 재확인, 재추론 없음')")
            fresh_elapsed=prior_intent.get('wall_seconds_from_button')
        else:
            write(intent,dict(status='ONE_FRESH_BUTTON_CLICK_AUTHORIZED',word=word,public_only=True,camera=False,training=False,at_utc=datetime.now(timezone.utc).isoformat()))
            started=time.monotonic();js("$('publicFresh').click()");assert locked()
            wait('__audit.freshSubmission!==null',30)
            submitted=js('__audit.freshSubmission');assert submitted.get('id'),submitted
            fresh_job=submitted['id']
            write(intent,dict(status='SUBMITTED_DO_NOT_REPEAT',job_id=fresh_job,word=word,public_only=True,training=False))
            print('fresh public job persisted before polling',fresh_job,flush=True)
            js('__audit.releaseFreshResponse=true')
            wait('!busy',180)
            fresh_elapsed=time.monotonic()-started
            write(intent,dict(status='CLIENT_FINISHED_DO_NOT_REPEAT',job_id=fresh_job,word=word,wall_seconds_from_button=fresh_elapsed))
        assert fresh_job and js("result!==null&&result.word_id==='WORD0739'&&!busy"),js("$('progress').textContent")
        r=js('result');job=json.load(urllib.request.urlopen(BASE+'/api/job/'+fresh_job,timeout=10))
        assert job['status']=='complete' and job['public_only'] is True
        assert r==job['result']
        assert r['ui_source_identity']['job_id']==fresh_job and r['ui_source_identity']['case_kind']=='fresh_public'
        assert r['training'] is False and r['tuning'] is False
        assert r['individual_reference_comparisons']==r['runtime_individual_reference_video_comparisons']==0
        assert r['learned_prototype_comparisons']==1
        assert r['partial_order_feedback']['completion_unknown']
        assert r['combined_feedback_latency_seconds']>=r['latency']['total_analysis_seconds']
        assert js("document.querySelector('.total-latency').textContent.includes('영상 분석부터 부분 순서 결과까지')")
        assert js("__audit.network.filter(x=>x.path==='/api/analyze-public'&&x.method==='POST').length")==(0 if resume else 1)
        assert js('stream===null&&__audit.cameraCalls===0')
        analysis=Path(job['analysis_directory']);source=manifest['cases'][word]['normal']['video']
        assert r['source_sha256']==sha(source)
        with np.load(analysis/'analysis.npz',allow_pickle=False) as a:
            assert len(a['frame_id'])==r['frames']
            assert np.allclose(np.asarray(r['overlay']['times']),a['timestamp_us']/1e6,rtol=0,atol=1e-9)
            for item in r['partial_order_feedback']['items']:
                for side in ['first','second']:
                    ev=item[side+'_evidence'];assert ev['native_seconds']==(a['timestamp_us'][ev['input_frames']]/1e6).tolist()
        wait("$('query').readyState>=2&&$('reference').readyState>=2")
        screens=[]
        for name,scroll in [('LIVE_ORDER_SCREEN.png','scrollTo(0,0)'),('LIVE_ORDER_RESULT.png',"$('partialOrder').scrollIntoView()")]:
            js(scroll);p=U/name;p.write_bytes(base64.b64decode(call('Page.captureScreenshot',{'format':'png'})['data']));screens.append(str(p))
        colors=js("[...document.querySelectorAll('h1,h2,h3,p,button,select')].filter(e=>e.getClientRects().length).map(e=>[getComputedStyle(e).color,getComputedStyle(e).backgroundColor])")
        assert all(a!=b for a,b in colors)
        call('Emulation.setDeviceMetricsOverride',dict(width=700,height=1000,deviceScaleFactor=1,mobile=False))
        assert js('document.documentElement.scrollWidth<=innerWidth+2')
        assert not errors,errors
        checks.append('One fresh public WORD0739 job: source/native times, combined latency, no camera, zero per-person references verified.')
        write(out,dict(status='PASS',checks=checks,saved_cases=records,fresh=dict(job_id=fresh_job,word=word,
            analysis_directory=str(analysis),source_sha256=r['source_sha256'],frames=r['frames'],
            wall_seconds_from_button=fresh_elapsed,combined_feedback_latency_seconds=r['combined_feedback_latency_seconds'],
            base_latency=r['latency'],partial_order_runtime=r['partial_order_runtime'],device=r['device'],
            native_times_preserved=True,training=False,tuning=False,individual_reference_comparisons=0),
            actual_fresh_public_posts_across_attempts=1,current_browser_fresh_posts=0 if resume else 1,
            completed_public_job_reused_for_ui_only=resume,actual_upload_posts=0,local_stubbed_upload_calls=1,
            estimate_ui_fixture_used=True,camera_calls=0,stream_null=True,private_footage_access=False,
            screenshots=screens,test_code_sha256=sha(__file__),server_launch=launch,
            limits=['Browser integration tests, not independent educational accuracy.',
                    '16 cached cases are not re-inferred or re-scored.',
                    'Cut/blackout/reversal controls are not automatically linguistic errors.',
                    'One fresh laptop run is not a latency distribution.'],whole_goal_complete=False))
        write(intent,dict(status='COMPLETE_DO_NOT_REPEAT',job_id=fresh_job,word=word,verification=str(out),wall_seconds_from_button=fresh_elapsed))
        print('PASS combined seconds',r['combined_feedback_latency_seconds'],flush=True)
    finally:
        if ws:
            try:call('Browser.close')
            except Exception:pass
            ws.close()
        if proc.poll() is None:proc.terminate()
        log.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        first=U/'LIVE_ORDER_BROWSER_FIRST_FAILURE.txt'
        if not first.exists():first.write_text(traceback.format_exc(),encoding='utf8')
        raise

