"""Owned local application test in a new headless profile, saved public media only."""
from pathlib import Path
import json,time,subprocess,urllib.request,websocket,base64,uuid,traceback,hashlib
U=Path(__file__).resolve().parent; BASE='http://127.0.0.1:8810'
def main():
    assert not (U/'BROWSER_VERIFIED.json').exists()
    profile=U/('browser_check_'+uuid.uuid4().hex[:8]);profile.mkdir()
    checks=[];errors=[];requests=[];ws=None
    def check(name,value):
        assert value,name
        checks.append(name)
    log=(profile/'chrome.stderr.log').open('w',encoding='utf-8')
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
                value=json.loads(ws.recv())
                if value.get('method')=='Runtime.exceptionThrown':errors.append(value)
                if value.get('method')=='Network.requestWillBeSent':
                    request=value['params']['request'];requests.append(dict(url=request['url'],method=request['method']))
                if value.get('id')==seq:
                    assert not value.get('error'),value
                    return value.get('result',{})
        def js(expression):
            value=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
            assert not value.get('exceptionDetails'),value
            return value.get('result',{}).get('value')
        def until(expression,n=150):
            for _ in range(n):
                if js(expression):return
                time.sleep(.1)
            raise AssertionError('Condition not reached: '+expression)
        def screenshot(name):
            image=call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))
            (U/name).write_bytes(base64.b64decode(image['data']))
        call('Runtime.enable');call('Network.enable');call('Page.enable')
        call('Page.addScriptToEvaluateOnNewDocument',dict(source="window.__qaCameraCalls=0; if(navigator.mediaDevices){navigator.mediaDevices.getUserMedia=async()=>{window.__qaCameraCalls++;throw Error('QA never opens camera');};}"))
        call('Page.navigate',dict(url=BASE))
        until("document.querySelector('#example') && document.querySelector('#reference').readyState>=1")
        js("(async()=>{let v=document.querySelector('#reference');v.muted=true;await v.play();})()")
        until("document.querySelector('#reference').readyState>=2")
        js("document.querySelector('#reference').pause()")
        until("document.querySelector('#progress').textContent.includes('분석할 수')",300)
        check('new demo title visible',js("document.querySelector('h1').textContent.includes('설명')"))
        check('four supported words',js("document.querySelector('#word').options.length===4"))
        check('camera never opened on load',js("window.__qaCameraCalls===0 && document.querySelector('#query').srcObject===null"))
        screenshot('DEMO_TOP.png')
        js("document.querySelector('#example').click()")
        until("document.querySelector('#localMovement .result-card') && document.querySelector('#query').readyState>=1 && document.querySelector('#reference').readyState>=1")
        js("Promise.all(['query','reference'].map(async id=>{let v=document.getElementById(id);v.muted=true;await v.play();}))")
        until("document.querySelector('#query').readyState>=2 && document.querySelector('#reference').readyState>=2")
        js("['query','reference'].forEach(id=>document.getElementById(id).pause())")
        check('saved example target matched',js("document.querySelector('#word').value==='WORD1133' && document.querySelector('#query').currentSrc.endsWith('/media/example/WORD1133')"))
        check('both public videos decoded',js("['query','reference'].every(id=>document.getElementById(id).videoWidth>0&&document.getElementById(id).duration>0)"))
        check('new local event displayed',js("document.querySelector('#localMovement').textContent.includes('움직임 근거')"))
        check('source phrase displayed',js("document.querySelector('#localMovement .quote').textContent.length>5"))
        check('saved analysis distinguished',js("document.querySelector('#result').textContent.includes('지금 다시 분석하지 않았습니다')"))
        check('partial evidence not full completion',js("document.querySelector('#result').textContent.includes('끝까지 수행했는지는 별도 확인')"))
        observed=js("Array.from(document.querySelectorAll('#localMovement .local-timeline button')).map(b=>({time:Number(b.dataset.nativeSeconds),frame:Number(b.dataset.nativeFrame),class:b.className}))")
        check('all17 original timestamps rendered',len(observed)==17 and abs(observed[0]['time']-1.6)<1e-8 and observed[0]['frame']==48)
        check('five high evidence points preserved',sum(x['class']=='point-observed' for x in observed)==5)
        js("document.querySelector('#localMovement').scrollIntoView({block:'start'})")
        screenshot('DEMO_LOCAL_MOVEMENT.png')
        js("document.querySelector('#localMovement .result-card .time-row button').click()")
        until("document.querySelector('#query').currentTime>=1.6")
        js("document.querySelector('#query').pause()")
        check('original-time replay button',js("document.querySelector('#query').currentTime>=1.6 && document.querySelector('#query').currentTime<2.5"))
        js("document.querySelector('#wholeDirection').open=true; document.querySelector('#wholeDirection').scrollIntoView({block:'center'})")
        check('old whole-direction unconfirmed remains separate',js("document.querySelector('#wholeDirection').textContent.includes('확인 보류')"))
        screenshot('DEMO_SEPARATE_DIRECTION.png')
        check('no white on white main text',js("Array.from(document.querySelectorAll('#result p,#result h2,#result h3')).every(e=>getComputedStyle(e).color!=='rgb(255, 255, 255)')"))
        call('Emulation.setDeviceMetricsOverride',dict(width=720,height=1000,deviceScaleFactor=1,mobile=False))
        js("document.querySelector('#localMovement').scrollIntoView({block:'start'})")
        check('narrow page no horizontal overflow',js("document.documentElement.scrollWidth<=window.innerWidth+2"))
        screenshot('DEMO_NARROW.png')
        check('no camera request',js('window.__qaCameraCalls===0'))
        check('no upload or inference request',not any(v['method']=='POST' for v in requests))
        check('only local app network requests',all(v['url'].startswith(BASE) or v['url'].startswith('data:') for v in requests))
        check('no JavaScript runtime exception',not errors)
        report=dict(status='PASS_HEADLESS_PUBLIC_BROWSER',checks=checks,check_count=len(checks),
            camera_calls=0,uploads=0,new_model_forwards=0,saved_public_only=True,
            displayed_points=observed,requests=[v for v in requests if not v['url'].startswith('data:')],
            inline_data_resource_count=sum(v['url'].startswith('data:') for v in requests),exceptions=errors,
            tested_browser='Isolated headless Chrome; browser rendered/decoded public videos; not the user desktop browser',
            images={n:hashlib.sha256((U/n).read_bytes()).hexdigest() for n in ['DEMO_TOP.png','DEMO_LOCAL_MOVEMENT.png','DEMO_SEPARATE_DIRECTION.png','DEMO_NARROW.png']})
        with (U/'BROWSER_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        print(report['status'],len(checks))
    except BaseException:
        (U/('BROWSER_FIRST_FAILURE_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8')
        if ws:
            try:
                detail=js("({url:location.href,text:document.body.innerText,videos:Array.from(document.querySelectorAll('video')).map(v=>({id:v.id,src:v.currentSrc,state:v.readyState,error:v.error?.message}))})")
                (U/('BROWSER_DIAGNOSTIC_'+str(time.time_ns())+'.json')).write_text(json.dumps(dict(page=detail,requests=requests,errors=errors),ensure_ascii=False,indent=2),encoding='utf-8')
            except Exception: pass
        raise
    finally:
        if ws:ws.close()
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
        log.close()
if __name__=='__main__':main()
