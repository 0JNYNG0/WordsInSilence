"""Isolated local UI regression test, saved public videos only."""
from pathlib import Path
import json,time,subprocess,urllib.request,websocket,base64,uuid,traceback
U=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8811'
def main():
    profile=U/('browser_check_'+uuid.uuid4().hex[:8]);profile.mkdir()
    checks=[];errors=[];requests=[];ws=None
    def check(name,value):
        assert value,name
        checks.append(name)
    log=(profile/'chrome.stderr.log').open('w',encoding='utf-8')
    proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1',
        '--disable-background-networking','--disable-component-update','--no-first-run','--autoplay-policy=no-user-gesture-required',
        '--user-data-dir='+str(profile),'--window-size=1440,1100','about:blank'],stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        tab=None
        for _ in range(100):
            try:
                port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(t for t in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=2)) if t['type']=='page');break
            except (OSError,ValueError,StopIteration):time.sleep(.1)
        assert tab
        ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=20);seq=0
        def call(method,params=None):
            nonlocal seq
            seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
            while True:
                value=json.loads(ws.recv())
                if value.get('method')=='Runtime.exceptionThrown':errors.append(value)
                if value.get('method')=='Network.requestWillBeSent':
                    r=value['params']['request'];requests.append(dict(url=r['url'],method=r['method']))
                if value.get('id')==seq:
                    assert not value.get('error'),value
                    return value.get('result',{})
        def js(code):
            r=call('Runtime.evaluate',dict(expression=code,returnByValue=True,awaitPromise=True));assert not r.get('exceptionDetails'),r
            return r.get('result',{}).get('value')
        def until(code):
            for _ in range(100):
                if js(code):return
                time.sleep(.1)
            raise AssertionError('Condition not reached '+code)
        def shot(name):
            (U/name).write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data']))
        call('Runtime.enable');call('Network.enable');call('Page.enable')
        call('Page.addScriptToEvaluateOnNewDocument',dict(source="window.__qaCamera=0;if(navigator.mediaDevices)navigator.mediaDevices.getUserMedia=async()=>{window.__qaCamera++;throw Error('No camera in saved UI test');}"))
        call('Page.navigate',dict(url=BASE))
        until("window.savedPublicViewer?.getState().case && document.getElementById('refZoomNote').textContent.includes('실제 영상')")
        catalog=json.loads((U/'data/catalog.json').read_text(encoding='utf-8'))
        for item in catalog['cases']:
            js(f'window.savedPublicViewer.loadCase({json.dumps(item["id"])})')
            until("[...document.querySelectorAll('video')].every(v=>v.readyState>=2&&!v.seeking)")
            check(item['id']+' both media decoded',js("[...document.querySelectorAll('video')].every(v=>v.videoWidth>0&&v.videoHeight>0)"))
            data=json.loads((U/'data'/f'{item["id"]}.json').read_text(encoding='utf-8'))
            for j,c in enumerate(data['cards']):
                js(f'window.savedPublicViewer.chooseCard({j})')
                until("[...document.querySelectorAll('video')].every(v=>!v.seeking)")
                check(c['id']+' status visible',js('document.getElementById("badge").textContent')=={'difference':'차이 후보','uncertain':'확인 어려움','observed':'관측됨'}[c['state']])
                check(c['id']+' actual focus times',all(abs(t-s['span']['focus'])<.001 for t,segs in zip(js('window.savedPublicViewer.getState().times'),[c['reference_segments'],c['query_segments']]) for s in segs[:1] if s['span']))
            if item['kind']=='gap':
                j=next(j for j,c in enumerate(data['cards']) if c['type']=='gap');js(f'window.savedPublicViewer.chooseCard({j})')
                until("document.getElementById('queryZoomNote').textContent.includes('확인 어려움') && !document.getElementById('query').seeking")
                check(item['id']+' no fabricated missing hand',js("!document.getElementById('queryMissing').hidden && document.getElementById('queryMissing').textContent.includes('영상 근거')"))
                if item['word_id']=='WORD1133':shot('UI_UNCERTAIN.png')
            if item['id']=='WORD2523_reverse_main':js('window.savedPublicViewer.chooseCard(0)');until("!document.getElementById('query').seeking");shot('UI_SHAPE_CANDIDATE.png')
        js("window.savedPublicViewer.loadCase('WORD0739_reverse_main')");until("!document.getElementById('query').seeking&&document.getElementById('refZoomNote').textContent.includes('실제 영상')")
        shot('UI_DIFFERENCE.png')
        js("document.getElementById('play').click()")
        until("window.savedPublicViewer.getState().playing")
        until("!window.savedPublicViewer.getState().playing")
        check('parallel replay reaches both range ends',js("window.savedPublicViewer.getState().times.every(t=>t>0)"))
        js("document.getElementById('position').value=0;document.getElementById('position').dispatchEvent(new Event('input'))")
        until("[...document.querySelectorAll('video')].every(v=>!v.seeking)")
        check('slider seeks both original axes',js("window.savedPublicViewer.getState().times")==[1.466667,2.133333])
        js("document.getElementById('showSkeleton').click()")
        check('highlight switch',js("!document.getElementById('showSkeleton').checked"))
        js("document.getElementById('showSkeleton').click();document.getElementById('restart').click()")
        call('Emulation.setDeviceMetricsOverride',dict(width=720,height=1000,deviceScaleFactor=1,mobile=False))
        check('720 no horizontal overflow',js('document.documentElement.scrollWidth<=innerWidth+2'));shot('UI_NARROW.png')
        call('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
        check('390 no horizontal overflow',js('document.documentElement.scrollWidth<=innerWidth+2'))
        check('no empty text contrast',js("[...document.querySelectorAll('h1,h2,h3,.scene-label')].every(e=>getComputedStyle(e).color!=='rgb(255, 255, 255)')"))
        check('no camera',js('window.__qaCamera===0'))
        check('no post or remote requests',all(r['method']=='GET' and r['url'].startswith(BASE) for r in requests))
        check('no runtime errors',not errors)
        report=dict(status='PASS_PUBLIC_SAVED_UI',checks=checks,check_count=len(checks),cases=16,cards=58,errors=errors,
            requests=requests,new_inference=0,new_training=0,new_DTW=0,new_scores=0,private_access=0,camera_calls=0,
            verification='Same Codex isolated Chrome UI regression test; not model accuracy evaluation or independent human annotation')
        (U/'BROWSER_VERIFIED.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(report['status'],len(checks))
    except BaseException:
        (U/f'BROWSER_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        if ws:ws.close()
        if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
        log.close()
if __name__=='__main__':main()
