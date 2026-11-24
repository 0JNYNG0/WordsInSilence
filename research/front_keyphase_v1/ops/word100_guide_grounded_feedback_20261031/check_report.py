"""Headless local HTML/media test; never imports a model or production app."""
from pathlib import Path
import subprocess,json,time,urllib.request,websocket,base64
U=Path(__file__).resolve().parent

def main():
    profile=U/'report_profile';profile.mkdir(exist_ok=False)
    proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--user-data-dir='+str(profile),'--window-size=1440,1100','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    ws=None
    try:
        tabs=[]
        for _ in range(100):
            try:
                assert proc.poll() is None
                port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);endpoint=f'http://127.0.0.1:{port}'
                tabs=json.load(urllib.request.urlopen(endpoint+'/json/list'))
                if any(t['type']=='page' for t in tabs):break
            except Exception:pass
            time.sleep(.1)
        ws=websocket.create_connection(next(t for t in tabs if t['type']=='page')['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=60);seq=0;errors=[]
        def call(method,params=None):
            nonlocal seq
            seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
            while True:
                m=json.loads(ws.recv())
                if m.get('method')=='Runtime.exceptionThrown':errors.append(m)
                if m.get('id')==seq:return m.get('result',{})
        def js(s):
            x=call('Runtime.evaluate',dict(expression=s,returnByValue=True,awaitPromise=True))
            if x.get('exceptionDetails'):raise RuntimeError(x)
            return x.get('result',{}).get('value')
        call('Page.enable');call('Runtime.enable');call('Page.navigate',dict(url=(U/'RESULT_KO.html').as_uri()));time.sleep(.3)
        aliases=js('Object.keys(D)');results=[]
        assert len(aliases)==12
        for alias in aliases:
            js(f"$('#choose').value='{alias}';choose()")
            result=js("new Promise((ok,no)=>{let n=0,h=setInterval(()=>{if([...document.querySelectorAll('video')].every(v=>v.readyState>=2&&v.videoWidth)){clearInterval(h);let i=current.anchors.findIndex(a=>a.state==='설명과 반대 특징 예측');if(i<0)i=0;if(current.anchors.length)jump(i);setTimeout(()=>ok({alias:$('#choose').value,videos:[...document.querySelectorAll('video')].map(v=>({width:v.videoWidth,time:v.currentTime,error:v.error?.code||null})),clauses:document.querySelectorAll('.clause').length,anchors:current.anchors.length,plot:!$('#plots').hidden,scope:$('#scope').textContent}),250)}else if(++n>150){clearInterval(h);no('media timeout')}},50)})")
            assert result['alias']==alias and result['clauses']>0
            assert all(v['width']>0 and not v['error'] for v in result['videos'])
            if result['anchors']:assert all(v['time']>0 for v in result['videos'])
            results.append(result)
        js("$('#choose').value='P048';choose();window.scrollTo(0,0)");time.sleep(.2)
        (U/'REPORT_SCREENSHOT.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
        js("$('#plots').scrollIntoView()");time.sleep(.1)
        (U/'REPORT_PLOT_SCREENSHOT.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
        call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False));time.sleep(.1)
        assert js('document.documentElement.scrollWidth<=window.innerWidth+2')
        assert js("document.querySelectorAll('svg').length===8")
        assert not errors
        (U/'REPORT_UI_VERIFICATION.json').write_text(json.dumps(dict(status='PASS',public_pairs=results,javascript_errors=errors,narrow_no_overflow=True,graphs=8,no_model_inference=True),ensure_ascii=False,indent=2),encoding='utf8')
        call('Browser.close')
    finally:
        if ws:ws.close()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
if __name__=='__main__':main()
