"""Own software integration browser; four public cropped videos, camera never used."""
from pathlib import Path
import json,time,subprocess,urllib.request,websocket,base64,hashlib,uuid
U=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8804'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    rows=read(U/'INPUT_VALIDATION.json')['public_crops'];checks=[];errors=[];results=[];ws=None
    profile=U/('browser_test_'+uuid.uuid4().hex[:8]);profile.mkdir()
    log=(profile/'chrome_stderr.txt').open('w',encoding='utf8')
    proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0',
        '--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run',
        '--autoplay-policy=no-user-gesture-required','--user-data-dir='+str(profile),'--window-size=1440,1100','about:blank'],
        stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        tab=None
        for _ in range(100):
            try:
                port=int((profile/'DevToolsActivePort').read_text().splitlines()[0])
                tab=next(t for t in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=2)) if t['type']=='page');break
            except (OSError,ValueError,StopIteration):time.sleep(.1)
        assert tab
        ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=30);seq=0
        def call(method,params=None):
            nonlocal seq
            seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
            while True:
                d=json.loads(ws.recv())
                if d.get('method')=='Runtime.exceptionThrown':errors.append(d)
                if d.get('id')==seq:
                    assert not d.get('error'),d
                    return d.get('result',{})
        def js(code):
            d=call('Runtime.evaluate',dict(expression=code,returnByValue=True,awaitPromise=True));assert not d.get('exceptionDetails'),d
            return d.get('result',{}).get('value')
        call('Runtime.enable');call('Page.enable');call('Page.navigate',dict(url=BASE))
        for _ in range(100):
            if js('typeof state!=="undefined" && state.catalog!==null'):break
            time.sleep(.1)
        assert js('state.catalog.words.length')==4
        assert js('state.catalog.words.every(w=>w.references.length===9)')
        assert js('state.catalog.input_policy')=='live_sensor_extent_v1'
        assert js("document.body.textContent.includes('별도 가로 경계를 적용하지 않습니다')")
        checks.append('correct visible sensor policy, four expressions x nine references')
        for row in rows:
            js("$('word').value="+json.dumps(row['word_id'])+";$('word').dispatchEvent(new Event('change'))")
            node=call('DOM.querySelector',dict(nodeId=call('DOM.getDocument')['root']['nodeId'],selector='#file'))['nodeId']
            call('DOM.setFileInputFiles',dict(nodeId=node,files=[row['video']]))
            assert js('state.file!==null && !$("analyze").disabled')
            start=U/(row['clip_id']+'_UPLOAD.json')
            if start.exists():
                prior=read(start);assert prior['source_sha256']==row['video_sha256'];jobid=prior['job_id']
            else:
                intent=U/(row['clip_id']+'_INTENT.json')
                assert not intent.exists(),'Unresolved upload intent; inspect before retry'
                save(intent,dict(public=True,source=row['video'],source_sha256=row['video_sha256']))
                previous_job=js('state.currentJob')
                js("$('analyze').click()")
                for _ in range(100):
                    if js('state.currentJob') and js('state.currentJob')!=previous_job:break
                    time.sleep(.1)
                jobid=js('state.currentJob');assert jobid and jobid!=previous_job
                save(start,dict(job_id=jobid,source_sha256=row['video_sha256']))
            for i in range(120):
                job=js('api("/api/jobs/'+jobid+'")')
                if job['status'] in ['COMPLETE','FAILED']:break
                if i%20==0:print(json.dumps(dict(clip=row['clip_id'],status=job['status'],progress=job.get('progress')),ensure_ascii=True),flush=True)
                time.sleep(1)
            assert job['status']=='COMPLETE',{'clip':row['clip_id'],'status':job['status'],'error':job.get('error')}
            assert job['submission_sha256']==row['video_sha256']
            result=job['result'];assert len(result['references'])==9 and result['training_used_submission'] is False
            assert result['input_policy']['box_source_pixels']==row['crop']
            assert not result['input_policy']['inferred_body_boundary_used']
            assert result['evidence_guard']['applied']
            js('state.currentJob='+json.dumps(jobid)+';state.busy=false;controls();api("/api/jobs/'+jobid+'").then(j=>renderResult(j.result,j))')
            ready="""new Promise((ok,no)=>{let k=0,t=setInterval(()=>{if([...document.querySelectorAll('video')].every(v=>v.readyState>=2&&!v.seeking&&v.videoWidth)){clearInterval(t);ok(true)}else if(++k>250){clearInterval(t);no('video decode timeout')}},40)})"""
            js(ready)
            assert js("$('stages').children.length")==len(result['stages'])
            assert js("$('referenceResults').children.length")==9
            compared=sum(f['status'] in ['SIMILAR_REFERENCE_RANGE','DIFFERENCE_CANDIDATE'] for s in result['stages'] for f in s['features'])
            assert js("$('resultSummary').textContent.includes("+json.dumps('항목 '+str(compared)+'개')+")")
            assert js("$('resultSummary').textContent.includes('단어 정답률이 아닙니다')")
            assert js('state.stream===null')
            first=result['selected_whole_references'][0]
            js('startSync('+json.dumps(first)+',[0,1.5])');time.sleep(.4)
            timing=js('({q:$("submissionVideo").currentTime,r:$("referenceVideo").currentTime,active:!!state.sync})')
            assert timing['q']>0 and timing['active'],timing
            js("$('syncStop').click()")
            assert js("$('submissionVideo').paused && $('referenceVideo').paused")
            checks.append(row['clip_id']+' full fresh pipeline nine references, stages rendered, paired playback, camera off')
            summary=dict(clip=row['clip_id'],word=row['word_id'],job_id=jobid,request_timing=job.get('request_timing'),
                         latency=result['latency'],stage_statuses=[s['status'] for s in result['stages']],
                         feature_statuses=[f['status'] for s in result['stages'] for f in s['features']],
                         references=len(result['references']),input_policy=result['input_policy'])
            results.append(summary);save(U/'PUBLIC_WEB_PROGRESS.json',results)
            print(json.dumps(summary,ensure_ascii=True),flush=True)
            js("$('stages').scrollIntoView({block:'start'})")
            (U/(row['clip_id']+'_RESULT.png')).write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
        js('scrollTo(0,0)');time.sleep(.3)
        (U/'LIVE_DEMO.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
        colors=js("[...document.querySelectorAll('h1,h2,p,button,select')].filter(e=>e.getClientRects().length).map(e=>[getComputedStyle(e).color,getComputedStyle(e).backgroundColor])")
        assert all(a!=b for a,b in colors)
        call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False))
        assert js('document.documentElement.scrollWidth<=innerWidth+2')
        assert not errors,errors
        checks+=['text contrast','760px no overflow','no JS exceptions','no camera access']
        save(U/'LIVE_UI_VERIFICATION.json',dict(status='PASS',checks=checks,results=results,private_inputs=0,camera_access=False))
        print('LIVE_UI_VERIFICATION PASS',flush=True)
        call('Browser.close')
    except BaseException as e:
        save(U/('UI_FIRST_FAILURE_'+uuid.uuid4().hex[:8]+'.json'),dict(error=str(e),checks=checks,results=results,js_errors=errors));raise
    finally:
        if ws:ws.close()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
        log.close()
if __name__=='__main__':main()
