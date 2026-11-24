"""Test only the new local report using an isolated installed Chrome profile."""
from pathlib import Path
import subprocess,json,time,urllib.request,websocket,base64
U=Path(__file__).resolve().parent
def main():
 port=9232;endpoint=f'http://127.0.0.1:{port}';exe='C:/Program Files/Google/Chrome/Application/chrome.exe'
 proc=subprocess.Popen([exe,'--headless=new',f'--remote-debugging-port={port}',f'--remote-allow-origins={endpoint}','--disable-background-networking','--disable-component-update','--no-first-run','--user-data-dir='+str(U/'ui_profile'),'--window-size=1440,1000','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  tabs=[]
  for _ in range(80):
   try:
    tabs=json.load(urllib.request.urlopen(endpoint+'/json/list'))
    if any(t['type']=='page' for t in tabs):break
   except Exception:pass
   time.sleep(.1)
  tab=next(t for t in tabs if t['type']=='page');ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin=endpoint);seq=0;errors=[]
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
  call('Page.enable');call('Runtime.enable');call('Page.navigate',dict(url=(U/'RESULT_KO.html').as_uri()));time.sleep(.5)
  tests=[]
  for a in ['F001','F017','F025','F029']:
   js(f"document.querySelector('#clip').value='{a}';change();")
   state=js("new Promise((ok,no)=>{let tries=0;let h=setInterval(()=>{let v=document.querySelector('video');if(v.readyState>=2&&v.videoWidth){clearInterval(h);v.currentTime=Math.min(1,v.duration/2);setTimeout(()=>ok({width:v.videoWidth,duration:v.duration,current:v.currentTime,alias:document.querySelector('#clip').value,images:[...document.images].every(i=>i.complete&&i.naturalWidth>0),title:document.title}),150)}else if(++tries>100){clearInterval(h);no('video timeout')}},50)})")
   assert state['alias']==a and state['width']>0 and state['images'];tests.append(state)
  js('window.scrollTo(0,0)');shot=call('Page.captureScreenshot',dict(format='png'))['data'];(U/'UI_SCREENSHOT.png').write_bytes(base64.b64decode(shot))
  call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1000,deviceScaleFactor=1,mobile=False));time.sleep(.2)
  assert js('document.documentElement.scrollWidth<=window.innerWidth+2')
  shot=call('Page.captureScreenshot',dict(format='png'))['data'];(U/'UI_NARROW_SCREENSHOT.png').write_bytes(base64.b64decode(shot));assert not errors
  result=dict(status='PASS_LOCAL_FILE_REPORT',tests=tests,javascript_errors=errors,narrow_no_page_overflow=True,scope='Isolated Chrome profile, file report and public videos only; no camera/production browser used. No model inference.',browser=call('Browser.getVersion'));p=U/'UI_VERIFICATION.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print('4 video selections + images + responsive width PASS')
  call('Browser.close');ws.close()
 finally:
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
if __name__=='__main__':main()
