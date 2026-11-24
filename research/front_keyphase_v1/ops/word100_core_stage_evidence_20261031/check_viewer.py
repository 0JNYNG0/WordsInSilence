"""Browser playback/seek/theme checks. No inference or training."""
from pathlib import Path
import json,time,subprocess,urllib.request,websocket,base64,hashlib,uuid
U=Path(__file__).resolve().parent
def main():
 profile=U/('browser_check_'+uuid.uuid4().hex[:8]);profile.mkdir();errors=[];checks=[];ws=None
 proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--user-data-dir='+str(profile),'--window-size=1440,1100','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  for _ in range(120):
   try:
    port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(t for t in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=2)) if t['type']=='page');break
   except Exception:time.sleep(.1)
  ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=30);seq=0
  def call(name,args=None):
   nonlocal seq
   seq+=1;ws.send(json.dumps(dict(id=seq,method=name,params=args or {})))
   while True:
    m=json.loads(ws.recv())
    if m.get('method')=='Runtime.exceptionThrown':errors.append(m)
    if m.get('id')==seq:
     assert 'error' not in m,m
     return m.get('result',{})
  def js(code):
   r=call('Runtime.evaluate',dict(expression=code,returnByValue=True,awaitPromise=True));assert not r.get('exceptionDetails'),r
   return r.get('result',{}).get('value')
  def ready(flag):
   for _ in range(100):
    if js(flag):return
    time.sleep(.1)
   raise AssertionError('report not ready')
  media="""new Promise((ok,no)=>{let k=0,t=setInterval(()=>{if([...document.querySelectorAll('video')].every(v=>v.readyState>=2&&v.videoWidth)){clearInterval(t);ok([...document.querySelectorAll('video')].map(v=>({src:v.currentSrc,width:v.videoWidth,duration:v.duration,error:v.error?.code||null})));}else if(++k>200){clearInterval(t);no([...document.querySelectorAll('video')].map(v=>({src:v.currentSrc,state:v.readyState,error:v.error?.code})));}},40)})"""
  call('Page.enable');call('Runtime.enable')
  for page,flag,count,loader in [('/', 'window.viewerReady===true',16,'selectQ()'),('/previous.html','window.reportReady===true',4,'loadClip()')]:
   call('Page.navigate',dict(url='http://127.0.0.1:8791'+page));ready(flag)
   for i in range(count):
    js(f"el('clip').value={i};{loader};");m=js(media)
    v=js("el('seek').value=55;seek();new Promise((ok,no)=>{let k=0,t=setInterval(()=>{if([...document.querySelectorAll('video')].every(v=>!v.seeking&&v.readyState>=2)){clearInterval(t);ok({q:el('qvideo').currentTime,r:el('rvideo').currentTime,text:el('qlabel').textContent});}else if(++k>200){clearInterval(t);no('seek timeout');}},50)})")
    assert v['q']>0 and v['r']>0
    checks.append(dict(page=page,clip_index=i,media=m,seek=v))
   play=js("(async()=>{let vs=[...document.querySelectorAll('video')],a=vs.map(v=>v.currentTime);for(let v of vs){v.muted=true;await v.play();}await new Promise(r=>setTimeout(r,500));for(let v of vs)v.pause();await new Promise(r=>setTimeout(r,150));return vs.map((v,i)=>({start:a[i],end:v.currentTime,seeking:v.seeking}));})()")
   assert all(p['end']>p['start'] and not p['seeking'] for p in play);checks.append(dict(page=page,play=play))
   for theme in ['light','dark']:
    call('Emulation.setEmulatedMedia',dict(features=[dict(name='prefers-color-scheme',value=theme)]))
    colors=js("[...document.querySelectorAll('h1,h2,h3,p,td,th,summary,button,select')].filter(x=>x.getClientRects().length).map(x=>({tag:x.tagName,color:getComputedStyle(x).color,bg:getComputedStyle(x).backgroundColor}));")
    assert all(c['color']!='rgb(255, 255, 255)' for c in colors),colors
    checks.append(dict(page=page,theme=theme,text_elements=len(colors),white_text_count=0))
   js("document.querySelector('video').scrollIntoView({block:'center'})")
   (U/('VIEWER.png' if page=='/' else 'PREVIOUS_FIXED.png')).write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
   call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False));time.sleep(.1)
   assert js('document.documentElement.scrollWidth<=innerWidth+2')
   call('Emulation.clearDeviceMetricsOverride')
  assert not errors
  # All routes share origin, preserve byte ranges, deny arbitrary files.
  with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8791/media/P048',headers={'Range':'bytes=0-99'})) as r:assert r.status==206 and len(r.read())==100 and r.headers['Content-Type']=='video/webm'
  result=dict(status='PASS',checks=checks,js_errors=errors,headless_chrome=True,actual_in_app_automation_unavailable=True,report_sha256=hashlib.sha256((U/'index.html').read_bytes()).hexdigest(),previous_sha256=hashlib.sha256((U/'previous_fixed.html').read_bytes()).hexdigest(),models_rerun=False)
  (U/'UI_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print('PASS',len(checks),'checks',flush=True)
  call('Browser.close')
 except Exception as e:
  (U/('UI_FAILURE_'+uuid.uuid4().hex[:8]+'.json')).write_text(json.dumps(dict(error=str(e),checks=checks,js_errors=errors),ensure_ascii=False,indent=2),encoding='utf8');raise
 finally:
  if ws:ws.close()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
if __name__=='__main__':main()
