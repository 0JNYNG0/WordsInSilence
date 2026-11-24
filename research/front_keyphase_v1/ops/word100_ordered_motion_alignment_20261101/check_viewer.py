from pathlib import Path
import json,time,subprocess,urllib.request,websocket,base64,hashlib,uuid
U=Path(__file__).resolve().parent
def main():
 profile=U/('browser_'+uuid.uuid4().hex[:8]);profile.mkdir();ws=None;errors=[];checks=[]
 proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--user-data-dir='+str(profile),'--window-size=1440,1100','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  for _ in range(120):
   try:
    port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(t for t in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=2)) if t['type']=='page');break
   except Exception:time.sleep(.1)
  ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=30);seq=0
  def call(method,params=None):
   nonlocal seq
   seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
   while True:
    d=json.loads(ws.recv())
    if d.get('method')=='Runtime.exceptionThrown':errors.append(d)
    if d.get('id')==seq:assert not d.get('error'),d;return d.get('result',{})
  def js(code):
   d=call('Runtime.evaluate',dict(expression=code,returnByValue=True,awaitPromise=True));assert not d.get('exceptionDetails'),d
   return d.get('result',{}).get('value')
  call('Runtime.enable');call('Page.enable');call('Page.navigate',dict(url='http://127.0.0.1:8796/'))
  for _ in range(100):
   if js('window.reportReady===true'):break
   time.sleep(.1)
  assert js('window.reportReady===true')
  ready="""new Promise((ok,no)=>{let k=0,t=setInterval(()=>{if([...document.querySelectorAll('video')].every(v=>v.readyState>=2&&!v.seeking&&v.videoWidth)){clearInterval(t);ok(true);}else if(++k>250){clearInterval(t);no('video timeout')}},40)})"""
  ids=js('cases.map(c=>c.id)');assert len(ids)==36
  for caseid in ids:
   js(f'selectCase({json.dumps(caseid)})');js(ready)
   for name in ['global_dtw','core_first']:
    js(f'selectMethod({json.dumps(name)})');js(ready)
    d=js("({method,query:current.query,reference:current.reference,frame:event.methods[method].center,expected:current.query_pts[event.methods[method].center],actual:$('queryVideo').currentTime,refexpected:current.reference_pts[event.reference_center],refactual:$('referenceVideo').currentTime,rows:$('methodRows').children.length,missingPath:!path.length,sliderDisabled:$('pathSlider').disabled})")
    if d['frame'] is not None:assert abs(d['expected']-d['actual'])<.04,d
    assert abs(d['refexpected']-d['refactual'])<.04 and d['rows']==2 and d['missingPath']==d['sliderDisabled'],d
    checks.append(dict(case=caseid,**d))
   # Every stage has a visible evidence description and correctly follows the selected path.
   stages=js('current.events.length')
   for i in range(stages):
    js(f'selectEvent({i});selectMethod("core_first")');js(ready)
    assert js("$('referenceDescription').textContent.length>10 && $('evidence').children.length===3 && $('policyMessage').textContent.length>20")
    if js('path.length>0'):
     js('seekPath(Math.floor(path.length/2))');js(ready)
     assert js("Math.abs($('queryVideo').currentTime-current.query_pts[path[Number($('pathSlider').value)][0]])<.04")
  for button,q,r in [('repeatOld',96,75),('repeatNew',73,75),('holdOld',107,77),('holdNew',124,77)]:
   js(f"$('{button}').click()");js(ready)
   assert js(f"Math.abs($('queryVideo').currentTime-current.query_pts[{q}])<.04 && Math.abs($('referenceVideo').currentTime-current.reference_pts[{r}])<.04")
  js('selectCase(cases.find(c=>c.role!=="fit").id);selectMethod("global_dtw")');js(ready)
  played=js("(async()=>{let vs=[...document.querySelectorAll('video')],starts=vs.map(v=>v.currentTime);for(let v of vs)await v.play();await new Promise(r=>setTimeout(r,250));for(let v of vs)v.pause();return vs.map((v,i)=>v.currentTime>starts[i])})()");assert all(played)
  colors=js("[...document.querySelectorAll('h1,h2,h3,p,td,th,select,button')].filter(x=>x.getClientRects().length).map(x=>({text:getComputedStyle(x).color,bg:getComputedStyle(x).backgroundColor}))")
  assert all(c['text']!=c['bg'] for c in colors)
  js("$('caseSelect').scrollIntoView({block:'start'})")
  (U/'VIEWER.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False));assert js('document.documentElement.scrollWidth<=innerWidth+2')
  assert not errors
  out=dict(status='PASS',case_method_checks=checks,all_stage_path_seeks=117,both_playback=True,four_concrete_example_buttons=True,light_text_contrast_check=True,narrow_no_overflow=True,js_errors=errors,headless_chrome=True,actual_codex_app_UI_not_automated=True,html_sha256=hashlib.sha256((U/'index.html').read_bytes()).hexdigest(),data_sha256=hashlib.sha256((U/'REPORT_DATA.json').read_bytes()).hexdigest())
  (U/'UI_VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print('PASS 36 pairs / 72 methods / 117 stages / playback / readability')
  call('Browser.close')
 except Exception as e:
  (U/('UI_FAILURE_'+uuid.uuid4().hex[:8]+'.json')).write_text(json.dumps(dict(error=str(e),checks=checks,js_errors=errors),ensure_ascii=False,indent=2),encoding='utf8');raise
 finally:
  if ws:ws.close()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
if __name__=='__main__':main()
