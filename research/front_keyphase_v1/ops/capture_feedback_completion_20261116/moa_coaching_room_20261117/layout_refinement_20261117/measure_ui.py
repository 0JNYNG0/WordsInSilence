"""Local browser measurement with intercepted camera APIs and public fixtures."""
from pathlib import Path
import subprocess,urllib.request,websocket,json,time,base64,uuid,sys
from check_browser import MOCK
D=Path(__file__).resolve().parent
def main():
    tag=sys.argv[1] if len(sys.argv)>1 else 'after'
    profile=D/'qa'/('measure_'+uuid.uuid4().hex[:8]);profile.mkdir()
    log=(profile/'chrome.log').open('w',encoding='utf-8')
    proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--autoplay-policy=no-user-gesture-required','--user-data-dir='+str(profile),'about:blank'],stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
    ws=None;seq=0;errors=[];checks=[]
    try:
        for _ in range(100):
            try:
                port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(x for x in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list')) if x['type']=='page');break
            except (OSError,ValueError,StopIteration):time.sleep(.1)
        ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=30)
        def call(method,params=None):
            nonlocal seq
            seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
            while True:
                r=json.loads(ws.recv())
                if r.get('method')=='Runtime.exceptionThrown':errors.append(r)
                if r.get('id')==seq:
                    assert not r.get('error'),r;return r.get('result',{})
        def js(expr):
            r=call('Runtime.evaluate',dict(expression=expr,returnByValue=True,awaitPromise=True));assert not r.get('exceptionDetails'),r;return r.get('result',{}).get('value')
        def until(expr):
            for _ in range(160):
                if js(expr):return
                time.sleep(.1)
            raise AssertionError(expr)
        def check(name,ok):checks.append(dict(name=name,passed=bool(ok)));assert ok,name
        def shot(name):(D/'qa'/name).write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data']))
        call('Runtime.enable');call('Page.enable');call('Page.addScriptToEvaluateOnNewDocument',dict(source=MOCK))
        call('Emulation.setDeviceMetricsOverride',dict(width=1366,height=768,deviceScaleFactor=1,mobile=False))
        call('Page.navigate',dict(url='http://127.0.0.1:8822/'))
        until("document.body?.classList.contains('ready')&&document.querySelector('#word')?.options.length>0")
        js("document.querySelector('[data-scene=practice]').click()")
        until("document.body.dataset.scene==='practice'&&document.querySelector('#avatar-controls').dataset.word==='WORD0099'")
        js("document.querySelector('#gesture').click()")
        js("import('./renderer.js').then(({StageRenderer})=>{const old=StageRenderer.prototype.begin;window.__frames=[];StageRenderer.prototype.begin=function(...args){__frames.push(performance.now());return old.apply(this,args)}})")
        time.sleep(.5);js('__frames=[]');time.sleep(3)
        perf=js("(()=>{const a=__frames;return {frames:a.length,spanMs:a.at(-1)-a[0],fps:(a.length-1)*1000/(a.at(-1)-a[0])}})()")
        sizes=[]
        for w,h in [(1366,768),(1280,720),(1920,1080),(720,1000)]:
            call('Emulation.setDeviceMetricsOverride',dict(width=w,height=h,deviceScaleFactor=1,mobile=False));time.sleep(.45)
            # Long, real fixture-style coaching plus a result button exercises bubble expansion.
            js("window.dispatchEvent(new CustomEvent('coach:say',{detail:{text:'손가락을 편 채로 손을 움직여 봐. 겹쳐서 보이지 않는 부분은 조금 더 확인이 필요해.'}}));document.querySelector('#show-feedback').hidden=false;document.querySelector('#next').click()")
            time.sleep(.2)
            q=js("(()=>{const rect=id=>{const r=document.getElementById(id).getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height,right:r.right,bottom:r.bottom}};return {width:innerWidth,height:innerHeight,dialogue:rect('dialogue'),left:rect('angle-left'),right:rect('angle-right'),controls:rect('avatar-controls'),anchor:JSON.parse(document.querySelector('#dialogue').dataset.anchor||'null'),overflow:document.documentElement.scrollWidth>innerWidth+2}})()")
            sizes.append(q);shot(f'{tag}_{w}x{h}.png')
            if tag!='before':
                a=q['anchor'];d=q['dialogue']
                check(f'{w} bubble below MOA',a and d['y']>=a['bottom']+10)
                check(f'{w} bubble inside screen',d['x']>=6 and d['right']<=w-6 and d['bottom']<=h-6)
                check(f'{w} tail points to MOA',abs((d['x']+a['tail'])-a['x'])<=2)
                check(f'{w} controls below examples',q['controls']['y']>=q['left']['bottom']+8 and q['controls']['y']>=q['right']['bottom']+8)
                check(f'{w} no horizontal overflow',not q['overflow'])
        if tag!='before':
            before=json.loads((D/'qa/MEASURE_BEFORE.json').read_text())
            check('larger left/right examples at desktop',sizes[0]['left']['w']>=before['sizes'][0]['left']['w']*1.12 and sizes[0]['left']['h']>before['sizes'][0]['left']['h'])
            check('smoother measured render cadence',perf['fps']>before['performance']['fps']*1.5 and perf['fps']>=22)
            js("__qa.state='recording'");until("document.body.dataset.renderMode==='observation-priority'")
            js("__frames=[]");time.sleep(.55)
            check('capture still pauses 3D',js('__frames.length===0'))
            js("__qa.state='preview'")
            check('no JS errors',not errors)
        out=dict(passed=True,tag=tag,performance=perf,sizes=sizes,checks=checks,errors=errors,physical_camera=False,scope='Mocked preview/browser cadence, not actual webcam or total inference latency')
        (D/'qa'/('MEASURE_'+tag.upper()+'.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(tag=tag,performance=perf,checks=len(checks)),ensure_ascii=False))
    except BaseException:
        if ws:
            try:shot(tag+'_FAILURE.png')
            except Exception:pass
        raise
    finally:
        if ws:ws.close()
        if proc.poll() is None:proc.terminate();proc.wait(10)
        log.close()
if __name__=='__main__':main()
