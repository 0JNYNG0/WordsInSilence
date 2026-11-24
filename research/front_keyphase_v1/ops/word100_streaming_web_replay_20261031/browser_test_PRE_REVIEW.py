"""One local server and isolated installed Chrome; no external service or camera."""
from server import *
import subprocess,urllib.request,websocket,base64,os

def main():
    assert not (U/'RESULT.json').exists() and not (U/'RUNNING.json').exists()
    write(U/'RUNNING.json',dict(pid=os.getpid(),status='STARTING'))
    engine=Engine(); server=make_server(engine)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    endpoint='http://127.0.0.1:9233'
    proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=9233','--remote-allow-origins='+endpoint,'--disable-background-networking','--disable-component-update','--no-first-run','--user-data-dir='+str(U/'chrome_profile'),'--window-size=1440,1050','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    ws=None
    try:
        tabs=[]
        for _ in range(100):
            try:
                tabs=json.load(urllib.request.urlopen(endpoint+'/json/list'))
                if any(t['type']=='page' for t in tabs):break
            except Exception:pass
            time.sleep(.1)
        tab=next(t for t in tabs if t['type']=='page');ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin=endpoint,timeout=600);seq=0;errors=[]
        def call(method,params=None):
            nonlocal seq
            seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
            while True:
                msg=json.loads(ws.recv())
                if msg.get('method')=='Runtime.exceptionThrown':errors.append(msg)
                if msg.get('id')==seq:
                    if 'error' in msg:raise RuntimeError(msg)
                    return msg.get('result',{})
        def js(expression):
            res=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
            if res.get('exceptionDetails'):raise RuntimeError(res)
            return res.get('result',{}).get('value')
        call('Page.enable');call('Runtime.enable');call('Page.navigate',dict(url=f'http://127.0.0.1:{server.server_port}/'));time.sleep(.2)
        output=[]
        for repeat in [1,2]:
            for alias in ALIASES:
                for mode in MODES:
                    r=js(f'runOne({json.dumps(alias)},{json.dumps(mode)},{repeat})');output.append(r)
                    print(json.dumps({k:r[k] for k in ['key','frames','backend_after_end_seconds','end_to_render_ack_seconds']},ensure_ascii=False),flush=True)
        media=js("new Promise((ok,no)=>{let n=0,h=setInterval(()=>{if([...document.querySelectorAll('video')].every(v=>v.readyState>=2&&v.videoWidth)){clearInterval(h);document.querySelector('#seek').value=45;seekPair();setTimeout(()=>ok([...document.querySelectorAll('video')].map(v=>({width:v.videoWidth,time:v.currentTime}))),200)}else if(++n>100){clearInterval(h);no('media timeout')}},50)})")
        assert len(media)==2 and all(m['width']>0 and m['time']>0 for m in media)
        js("document.querySelector('#result').scrollIntoView()")
        (U/'UI_SCREENSHOT.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
        call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False));time.sleep(.1)
        assert js('document.documentElement.scrollWidth<=window.innerWidth+2')
        (U/'UI_NARROW_SCREENSHOT.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
        assert not errors
        engine.check_pins()
        result=dict(status='COMPLETE_PENDING_INDEPENDENT_VERIFICATION',rows=output,application_setup_seconds=engine.setup_seconds,
          media_seek=media,js_errors=errors,narrow_no_overflow=True,device='MediaPipe / fixed pose head / DTW on CPU; browser local. No SignRep inference in this route.',
          browser=call('Browser.getVersion'),scope=engine.frozen['scope'],timing_definition=engine.frozen['timing'])
        write(U/'RESULT.json',result)
        call('Browser.close')
    finally:
        if ws:ws.close()
        server.shutdown();server.server_close();thread.join()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
        # Only our own completed lock; no PID-only stale-lock inference.
        write(U/'EXECUTION_COMPLETE.json',dict(pid=os.getpid(),active=None,status='COMPLETE' if (U/'RESULT.json').exists() else 'FAILED',server_closed=True,chrome_closed=True))
        (U/'RUNNING.json').unlink()

if __name__=='__main__':
    try:main()
    except BaseException:
        p=U/'FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
        raise
