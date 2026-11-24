from pathlib import Path
import json,time,subprocess,urllib.request,websocket,base64,hashlib,uuid
U=Path(__file__).resolve().parent
def main():
 profile=U/('browser_'+uuid.uuid4().hex[:8]);profile.mkdir();ws=None;errors=[];checks=[]
 proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--user-data-dir='+str(profile),'--window-size=1440,1100','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  for _ in range(100):
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
  call('Runtime.enable');call('Page.enable');call('Page.navigate',dict(url='http://127.0.0.1:8798/'))
  for _ in range(100):
   if js("typeof D!='undefined' && document.querySelectorAll('#sample option').length===168"):break
   time.sleep(.1)
  assert js("D.items.length===168")
  for i in range(168):
   js(f'select({i})');assert js('Promise.all(images.map(im=>im.decode().then(()=>im.naturalWidth>0))).then(x=>x.every(Boolean))')
   assert js("document.querySelectorAll('#truth tr').length===18")
   assert js('images.length===13 && D.items[current].frames.every((f,j,a)=>j===0||f.id===a[j-1].id+1)')
   js("el('center').click();el('forward').click()");assert js("+el('frame').value===7")
   js("el('back').click()");assert js("+el('frame').value===6")
   assert js("el('time').textContent.includes('중앙정답')")
   checks.append(i)
  js('select(0)');js('Promise.all(images.map(im=>im.decode()))');js("el('play').click()")
  time.sleep(.7);assert js("+el('frame').value===12")
  js("el('center').click();el('canvas').scrollIntoView({block:'start'})")
  colors=js("[...document.querySelectorAll('h1,h2,p,td,th,select,button')].filter(x=>x.getClientRects().length).map(x=>({fg:getComputedStyle(x).color,bg:getComputedStyle(x).backgroundColor}))")
  assert all(c['fg']!=c['bg'] for c in colors)
  (U/'VIEWER.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  js('scrollTo(0,0)');(U/'SUMMARY.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False));assert js('document.documentElement.scrollWidth<=innerWidth+2')
  assert not errors
  out=dict(status='PASS',samples=len(checks),original_frames_per_window=13,all_images_decode=True,frame_controls_native_step=True,playback=True,contrast=True,narrow_no_overflow=True,js_errors=errors,headless_chrome=True,html_sha256=hashlib.sha256((U/'index.html').read_bytes()).hexdigest())
  with (U/'UI_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
  print(json.dumps(out));call('Browser.close')
 except Exception as e:
  with (U/('UI_FAILURE_'+uuid.uuid4().hex[:8]+'.json')).open('x',encoding='utf8') as f:json.dump(dict(error=str(e),completed=len(checks),js_errors=errors),f,indent=2)
  raise
 finally:
  if ws:ws.close()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
if __name__=='__main__':main()
