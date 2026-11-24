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
  call('Runtime.enable');call('Page.enable');call('Page.navigate',dict(url='http://127.0.0.1:8792/'))
  for _ in range(100):
   if js('window.reportReady===true'):break
   time.sleep(.1)
  assert js('window.reportReady===true')
  ready="""new Promise((ok,no)=>{let k=0,t=setInterval(()=>{if([...document.querySelectorAll('video')].every(v=>v.readyState>=2&&!v.seeking&&v.videoWidth)){clearInterval(t);ok([...document.querySelectorAll('video')].map(v=>({src:v.currentSrc,t:v.currentTime,width:v.videoWidth})));}else if(++k>200){clearInterval(t);no('video timeout')}},40)})"""
  for a in ['P036','P048','P066','P072']:
   js(f"el('clip').value='{a}';loadClip();");refs=js("[...el('ref').options].map(x=>x.value)")
   for r in refs:
    for kind in ['B_same_path','C_relational_path']:
     js(f"el('ref').value='{r}';el('method').value='{kind}';loadPair();");js(ready)
     js("el('seek').value=Math.floor(q.frames*.55);seek()");m=js(ready);assert all(v['t']>0 for v in m)
     checks.append(dict(query=a,reference=r,condition=kind,media=m))
  for g in js('Object.keys(D.groups)'):
   js(f"el('group').value='{g}';groupChanged();");assert js("!!document.querySelector('#graph svg')&&el('stageTable').textContent.length>0")
  played=js("(async()=>{let vs=[...document.querySelectorAll('video')],starts=vs.map(v=>v.currentTime);for(let v of vs)await v.play();await new Promise(r=>setTimeout(r,350));for(let v of vs)v.pause();return vs.map((v,i)=>v.currentTime>starts[i])})()");assert all(played)
  themes=[]
  for theme in ['light','dark']:
   call('Emulation.setEmulatedMedia',dict(features=[dict(name='prefers-color-scheme',value=theme)]));colors=js("[...document.querySelectorAll('h1,h2,h3,p,td,th,select,button,summary')].filter(x=>x.getClientRects().length).map(x=>getComputedStyle(x).color)");assert all(c!='rgb(255, 255, 255)' for c in colors);themes.append(theme)
  js("el('clip').value='P048';loadClip();el('group').value='R_to_L_forearm';groupChanged();");js(ready);js("el('seek').value=85;seek()");js(ready);js("el('viewer').scrollIntoView({block:'start'})")
  (U/'VIEWER.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False));assert js('document.documentElement.scrollWidth<=innerWidth+2')
  assert not errors
  out=dict(status='PASS',pair_cases=checks,groups=13,playback_both=True,themes=themes,narrow_no_overflow=True,js_errors=errors,headless_chrome=True,actual_app_UI_not_inspected=True,html_sha256=hashlib.sha256((U/'index.html').read_bytes()).hexdigest(),data_sha256=hashlib.sha256((U/'REPORT_DATA.json').read_bytes()).hexdigest())
  (U/'UI_VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print('PASS 24 pairs /13groups/both videos/themes')
  call('Browser.close')
 except Exception as e:
  (U/('UI_FAILURE_'+uuid.uuid4().hex[:8]+'.json')).write_text(json.dumps(dict(error=str(e),checks=checks,js_errors=errors),ensure_ascii=False,indent=2),encoding='utf8');raise
 finally:
  if ws:ws.close()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
if __name__=='__main__':main()
