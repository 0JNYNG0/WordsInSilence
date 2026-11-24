"""Local static report check. No model inference/training/replay jobs."""
from pathlib import Path
import json,time,subprocess,urllib.request,websocket,base64,hashlib
U=Path(__file__).resolve().parent
def main():
 profile=U/'report_check_profile';profile.mkdir(exist_ok=False)
 proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--user-data-dir='+str(profile),'--window-size=1440,1100','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
 ws=None;errors=[];checks=[]
 try:
  for _ in range(120):
   try:
    port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tabs=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=2));tab=next(t for t in tabs if t['type']=='page');break
   except Exception:time.sleep(.1)
  ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=40);seq=0
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
  call('Page.enable');call('Runtime.enable');call('Page.navigate',dict(url=(U/'RESULT_KO.html').as_uri()))
  for _ in range(100):
   if js('window.reportReady===true'):break
   time.sleep(.1)
  assert js('window.reportReady===true')
  for i in range(4):
   js(f"el('clip').value={i};loadClip();el('ref').value=1;loadRef();")
   value=js("new Promise((ok,no)=>{let n=0;let t=setInterval(()=>{if([...document.querySelectorAll('video')].every(v=>v.readyState>=2&&v.videoWidth)){clearInterval(t);el('seek').value=55;seek();setTimeout(()=>ok({q:el('qvideo').currentTime,r:el('rvideo').currentTime,label:el('qlabel').textContent,scope:el('coverage').textContent,svg:!!document.querySelector('svg path')}),200)}else if(++n>150){clearInterval(t);no('media not ready')}},50)})")
   assert value['q']>0 and value['r']>0 and value['svg'];checks.append(value)
  js("el('graph').scrollIntoView({block:'center'})")
  (U/'REPORT_UI.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False));time.sleep(.2)
  assert js('document.documentElement.scrollWidth<=innerWidth+2')
  assert not errors,errors
  with (U/'REPORT_UI_CHECK.json').open('x',encoding='utf8') as f:json.dump(dict(status='PASS',report_sha256=hashlib.sha256((U/'RESULT_KO.html').read_bytes()).hexdigest(),cases=checks,js_errors=errors,narrow_no_overflow=True,models_rerun=False),f,ensure_ascii=False,indent=2)
  call('Browser.close')
 finally:
  if ws:ws.close()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
if __name__=='__main__':main()
