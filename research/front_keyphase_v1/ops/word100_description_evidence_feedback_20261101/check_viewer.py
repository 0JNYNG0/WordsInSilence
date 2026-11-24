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
  call('Runtime.enable');call('Page.enable');call('Page.navigate',dict(url='http://127.0.0.1:8797/'))
  for _ in range(200):
   if js('window.reportReady===true'):break
   time.sleep(.1)
  assert js('window.reportReady===true')
  ready="""new Promise((ok,no)=>{let k=0,t=setInterval(()=>{if([...document.querySelectorAll('video')].every(v=>v.readyState>=2&&!v.seeking&&v.videoWidth)){clearInterval(t);ok(true);}else if(++k>250){clearInterval(t);no('video timeout')}},40)})"""
  ids=js('cases.map(c=>c.id)');assert len(ids)==36
  for caseid in ids:
   js(f'selectCase({json.dumps(caseid)})');js(ready)
   count=js('current.stages.length')
   for i in range(count):
    js(f'selectStage({i});seek(Math.floor(path.length/2))');js(ready)
    z=js("({pair:current.id,stage:stage.id,n:path.length,q:$('queryVideo').currentTime,r:$('referenceVideo').currentTime,eq:current.query_pts[path[Number($('slider').value)][0]],er:current.reference_pts[path[Number($('slider').value)][1]],sourceText:$('sources').textContent.length})")
    assert abs(z['q']-z['eq'])<.04 and abs(z['r']-z['er'])<.04 and z['sourceText']>10,z
    # Every generic feature and component renders, including all-missing groups.
    js("$('sourceSelect').value='all';renderGroups();for(let g of Object.keys(stage.groups)){chooseGroup(g);for(let i=0;i<$('componentSelect').options.length;i++){$('componentSelect').value=i;draw()}}")
    assert js("$('detail').textContent.includes('현재 선택한 세부 값')")
    checks.append(z)
  assert len(checks)==117
  js("selectCase(cases.find(c=>c.query==='P048'&&c.reference==='P045').id);selectStage(0)");js(ready)
  js("chooseGroup('R_to_L_forearm');$('componentSelect').value='0';draw();seek(0);$('play').click()")
  time.sleep(.5);assert js("Number($('slider').value)>0");js('stop()');js(ready)
  played=js("(async()=>{let vs=[...document.querySelectorAll('video')],starts=vs.map(v=>v.currentTime);for(let v of vs)await v.play();await new Promise(r=>setTimeout(r,250));for(let v of vs)v.pause();return vs.map((v,i)=>v.currentTime>starts[i])})()");assert all(played)
  colors=js("[...document.querySelectorAll('h1,h2,h3,p,td,th,select,button')].filter(x=>x.getClientRects().length).map(x=>({text:getComputedStyle(x).color,bg:getComputedStyle(x).backgroundColor}))")
  assert all(c['text']!=c['bg'] for c in colors)
  js("$('caseSelect').scrollIntoView({block:'start'})")
  (U/'VIEWER.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  js("$('groupSelect').scrollIntoView({block:'start'})")
  (U/'GRAPH.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False));assert js('document.documentElement.scrollWidth<=innerWidth+2')
  assert not errors
  out=dict(status='PASS',pairs=len(ids),stage_checks=checks,all_group_components_rendered=True,both_video_playback=True,aligned_player=True,readability=True,narrow_no_overflow=True,js_errors=errors,headless_chrome=True,actual_codex_app_UI_not_automated=True,html_sha256=hashlib.sha256((U/'index.html').read_bytes()).hexdigest(),data_sha256=hashlib.sha256((U/'REPORT_DATA.json').read_bytes()).hexdigest())
  (U/'UI_VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print('PASS 36 pairs, 117 stages, all feature components, videos, aligned player, readability')
  call('Browser.close')
 except Exception as e:
  (U/('UI_FAILURE_'+uuid.uuid4().hex[:8]+'.json')).write_text(json.dumps(dict(error=str(e),checks=checks,js_errors=errors),ensure_ascii=False,indent=2),encoding='utf8');raise
 finally:
  if ws:ws.close()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
if __name__=='__main__':main()
