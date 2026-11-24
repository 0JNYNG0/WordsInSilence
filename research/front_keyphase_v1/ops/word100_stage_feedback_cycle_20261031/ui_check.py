"""Isolated installed Chrome rendering fallback after browser/desktop kernel failures."""
from pathlib import Path
import subprocess,json,time,urllib.request,websocket,base64,threading,sys
from serve_ui import Handler,ThreadingHTTPServer
U=Path(__file__).resolve().parent
def main():
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start();report_url=f'http://127.0.0.1:{server.server_port}/RESULT_KO.html'
 file_mode='--file' in sys.argv
 if file_mode:report_url=(U/'RESULT_KO.html').as_uri()
 exe='C:/Program Files/Google/Chrome/Application/chrome.exe';profile=U/'ui_profile'
 proc=subprocess.Popen([exe,'--headless=new','--remote-debugging-port=9228','--remote-allow-origins=http://127.0.0.1:9228','--disable-background-networking','--disable-component-update','--no-first-run','--user-data-dir='+str(profile),'--window-size=1440,1000','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  tabs=[]
  for _ in range(80):
   try:
    tabs=json.load(urllib.request.urlopen('http://127.0.0.1:9228/json/list'))
    if any(t['type']=='page' for t in tabs):break
    time.sleep(.1)
   except Exception:time.sleep(.1)
  if not any(t['type']=='page' for t in tabs):
   request=urllib.request.Request('http://127.0.0.1:9228/json/new?about:blank',method='PUT')
   tabs.append(json.load(urllib.request.urlopen(request)))
  tab=next(t for t in tabs if t['type']=='page');ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1:9228');seq=0;errors=[]
  def call(method,params=None):
   nonlocal seq;seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
   while True:
    msg=json.loads(ws.recv())
    if msg.get('method')=='Runtime.exceptionThrown':errors.append(msg)
    if msg.get('id')==seq:
     if 'error' in msg:raise RuntimeError(msg)
     return msg.get('result',{})
  def js(expression):
   result=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
   if result.get('exceptionDetails'):raise RuntimeError(result)
   return result.get('result',{}).get('value')
  call('Page.enable');call('Runtime.enable');call('Page.navigate',dict(url=report_url))
  ready=js("new Promise((resolve,reject)=>{let n=0;let h=setInterval(()=>{if(typeof D!=='undefined'&&D&&typeof active!=='undefined'&&active&&document.getElementById('qvideo').readyState>=2&&document.getElementById('rvideo').readyState>=2){clearInterval(h);resolve(true)}else if(++n>200){clearInterval(h);reject('viewer timeout')}},100)})");assert ready
  tests=[]
  for c in ['legacy_dtw','pose_capacity_matched','pose_plus_signrep']:
   for a in ['P005','P079']:
    js(f"$('condition').value='{c}';$('query').value='{a}';updateQuery();$('seek').value=40;$('seek').oninput();")
    state=js("new Promise(resolve=>setTimeout(()=>resolve({condition:$('condition').value,alias:$('query').value,queryTime:$('qvideo').currentTime,referenceTime:$('rvideo').currentTime,expectedQ:active.q.time[40],expectedR:active.r.time[active.p.query_to_reference[40]],queryReady:$('qvideo').readyState,referenceReady:$('rvideo').readyState,queryWidth:$('qvideo').videoWidth,referenceWidth:$('rvideo').videoWidth,graphLines:$('graph').querySelectorAll('polyline').length,current:$('current').textContent,evidence:$('evidence').textContent,events:$('events').textContent}),400))")
    state['pass']=abs(state['queryTime']-state['expectedQ'])<.05 and abs(state['referenceTime']-state['expectedR'])<.05 and state['queryReady']>=2 and state['referenceReady']>=2 and state['queryWidth']>0 and state['referenceWidth']>0
    tests.append(state)
  js("document.getElementById('query').scrollIntoView({block:'start'});")
  time.sleep(.2);png=call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data'];(U/'UI_SCREENSHOT.png').write_bytes(base64.b64decode(png))
  call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1000,deviceScaleFactor=1,mobile=False));time.sleep(.2);shot=call('Page.captureScreenshot',dict(format='png'))['data'];(U/'UI_NARROW_SCREENSHOT.png').write_bytes(base64.b64decode(shot))
  result=dict(status='PASS' if all(t['pass'] for t in tests) and not errors else 'FAIL',tests=tests,errors=errors,method='Own local research HTML in isolated preinstalled Chrome headless via CDP; browser/desktop plugin kernel initialization failed. No production browser/user data touched.',browser=call('Browser.getVersion'))
  result['transport']='file' if file_mode else 'loopback_http'
  (U/('UI_FILE_VERIFICATION.json' if file_mode else 'UI_VERIFICATION.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(dict(status=result['status'],cases=len(tests),errors=len(errors))))
  call('Browser.close');ws.close()
 finally:
  server.shutdown();server.server_close()
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
if __name__=='__main__':main()
