"""Actual Chromium recording of allowlisted public clips; never opens a camera."""
from pathlib import Path
import json,time,subprocess,urllib.request,websocket,uuid,traceback,sys,base64,os,psutil
U=Path(__file__).resolve().parent
PY=U.parent/'word100_signrep_feasibility_20261029/runtime/Scripts/python.exe'
def main():
 trial=U/('browser_'+uuid.uuid4().hex[:8]);trial.mkdir();ws=None;chrome=None;server=None
 log=(trial/'server.log').open('w',encoding='utf-8');clog=(trial/'chrome.log').open('w',encoding='utf-8')
 results=[];errors=[];requests=[]
 try:
  server=subprocess.Popen([str(PY),'-X','utf8','-u',str(U/'serve_test.py')],env={**os.environ,'WIS_BROWSER_RUN_ID':trial.name},stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
  deadline=time.monotonic()+120;state=None
  while time.monotonic()<deadline:
   if server.poll() is not None:raise RuntimeError('Server failed: '+str(trial/'server.log'))
   try:
    candidate=json.loads((U/'SERVER.json').read_text())
    if candidate.get('run_id')==trial.name:state=candidate;break
   except (OSError,ValueError):pass
   time.sleep(.25)
  if not state:raise TimeoutError('Server startup')
  profile=trial/'profile';profile.mkdir()
  chrome=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--autoplay-policy=no-user-gesture-required','--user-data-dir='+str(profile),'--window-size=1280,800','about:blank'],stdout=subprocess.DEVNULL,stderr=clog,creationflags=subprocess.CREATE_NO_WINDOW)
  tab=None
  for _ in range(100):
   try:
    port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(x for x in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=2)) if x['type']=='page');break
   except (OSError,ValueError,StopIteration):time.sleep(.1)
  if not tab:raise RuntimeError('Chrome startup')
  ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=120);seq=0
  def call(method,params=None):
   nonlocal seq
   seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
   while True:
    r=json.loads(ws.recv())
    if r.get('method')=='Runtime.exceptionThrown':errors.append(r)
    if r.get('method')=='Network.requestWillBeSent':requests.append(r['params']['request'])
    if r.get('id')==seq:
     if r.get('error'):raise RuntimeError(r)
     return r.get('result',{})
  def js(code):
   r=call('Runtime.evaluate',dict(expression=code,returnByValue=True,awaitPromise=True))
   if r.get('exceptionDetails'):raise RuntimeError(r)
   return r.get('result',{}).get('value')
  # CDP Network events may copy every 8MB RGBA body and stall the browser.
  # Audit URLs only inside the test page; CSP prohibits external requests.
  call('Runtime.enable');call('Page.enable');call('Page.navigate',dict(url=state['url']))
  for _ in range(100):
   if js('window.testReady===true'):break
   time.sleep(.1)
  assert js('window.testReady===true')
  specs=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'))['rows']
  # Cancellation first; next full take demonstrates state is fresh.
  jobs=[(specs[0],True)]+[(s,False) for s in specs]
  for spec,cancel in jobs:
   result=js('runPublic('+json.dumps(spec['clip_id'])+','+str(cancel).lower()+')');results.append(result)
   (trial/'RUNS.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
   print(json.dumps(dict(label=spec['label'],cancel=cancel,frames=result['result']['rtmw_frames'],complete=result['result']['client_input_complete'],finish_ms=result['result']['client_finish_ms']),ensure_ascii=False),flush=True)
  (trial/'UI.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',{'format':'png'})['data']))
  summary=dict(status='EXECUTED_NEEDS_INDEPENDENT_VERIFICATION',trial=str(trial),results=results,errors=errors,requests=js('window.requestAudit'),hardware_camera_calls=js('window.hardwareCalls'),browser=call('Browser.getVersion'))
  (trial/'COMPLETE.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
  (U/'LATEST_TRIAL.json').write_text(json.dumps(dict(trial=str(trial))),encoding='utf-8')
  print('COMPLETE',trial,flush=True)
 except BaseException:
  (trial/'FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
 finally:
  if ws:ws.close()
  for p in [chrome,server]:
   if p and p.poll() is None:
    children=psutil.Process(p.pid).children(recursive=True)
    for child in reversed(children):
     try:child.terminate()
     except psutil.NoSuchProcess:pass
    p.terminate();p.wait(timeout=20)
  log.close();clog.close()
if __name__=='__main__':main()
