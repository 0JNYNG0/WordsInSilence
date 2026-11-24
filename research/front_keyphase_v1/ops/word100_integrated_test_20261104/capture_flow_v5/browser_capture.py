"""Actual browser/MediaRecorder tests, with synthetic camera and intercepted POSTs only."""
from pathlib import Path
import json,time,subprocess,urllib.request,websocket,uuid,traceback,sys
A=Path(__file__).resolve().parent;U=A.parent;BASE='http://127.0.0.1:8812'
before='--before' in sys.argv
def main():
 profile=A/('browser_'+uuid.uuid4().hex[:8]);profile.mkdir();checks=[];errors=[];requests=[];cases=[];ws=None
 def check(n,v):assert v,n;checks.append(n)
 log=(profile/'stderr.log').open('w',encoding='utf-8')
 proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--autoplay-policy=no-user-gesture-required','--user-data-dir='+str(profile),'--window-size=1280,850','about:blank'],stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  tab=None
  for _ in range(120):
   try:
    port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(t for t in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=2)) if t['type']=='page');break
   except (OSError,ValueError,StopIteration):time.sleep(.1)
  assert tab
  ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=20);seq=0
  def call(method,params=None):
   nonlocal seq
   seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
   while True:
    v=json.loads(ws.recv())
    if v.get('method')=='Runtime.exceptionThrown':errors.append(v)
    if v.get('method')=='Network.requestWillBeSent':requests.append(v['params']['request'])
    if v.get('id')==seq:assert not v.get('error'),v;return v.get('result',{})
  def js(code):
   r=call('Runtime.evaluate',dict(expression=code,returnByValue=True,awaitPromise=True));assert not r.get('exceptionDetails'),r
   return r.get('result',{}).get('value')
  def until(code,seconds=15):
   end=time.monotonic()+seconds
   while time.monotonic()<end:
    if js(code):return
    time.sleep(.05)
   raise AssertionError('Not reached '+code)
  call('Runtime.enable');call('Network.enable');call('Page.enable')
  call('Page.addScriptToEvaluateOnNewDocument',dict(source="window.__hardwareCameraCalls=0;navigator.mediaDevices.getUserMedia=async()=>{__hardwareCameraCalls++;throw Error('Hardware forbidden');};"))
  call('Page.navigate',dict(url=BASE));until("typeof state!=='undefined' && state?.ready",40)
  js("""window.__canvas=document.createElement('canvas');__canvas.width=640;__canvas.height=360;
  window.__timer=setInterval(()=>{const c=__canvas.getContext('2d');c.fillStyle='#426cd7';c.fillRect(0,0,640,360);c.fillStyle='white';c.fillText(String(performance.now()),20,40)},30);
  navigator.mediaDevices.getUserMedia=async()=>__canvas.captureStream(30);
  const Recorder=MediaRecorder;window.MediaRecorder=class extends Recorder{start(...args){window.__recordStart=performance.now();super.start(...args)}};
  window.__realFetch=fetch;window.__scenario='one';window.__previews=0;window.__uploads=0;window.__hung=false;window.__aborted=false;window.__didHang=false;
  window.fetch=async(url,opts)=>{
   if(String(url)==='/api/preview'){
    __previews++;const t=(performance.now()-__recordStart)/1000;
    if(__scenario==='hung'&&!__didHang&&t>.35){__didHang=true;__hung=true;await new Promise((resolve,reject)=>{opts.signal?.addEventListener('abort',()=>{__aborted=true;reject(new DOMException('Aborted','AbortError'))},{once:true})});}
    let hands=[];
    if(__scenario==='one'||__scenario==='two'||__scenario==='hung')hands=t<1.2?(__scenario==='two'?[[.4,.4],[.6,.4]]:[[.4,.4]]):[];
    if(__scenario==='brief')hands=t<2.4&&!(t>=.8&&t<1.4)?[[.4,.4],[.6,.4]]:[];
    if(__scenario==='one_remains')hands=t<.8?[[.4,.4],[.6,.4]]:t<2.4?[[.4,.4]]:[];
    if(__scenario==='initial_wait')hands=t>=1.2&&t<2.4?[[.4,.4]]:[];
    if(__scenario==='slow'){hands=t<1.8?[[.4,.4]]:[];await new Promise(r=>setTimeout(r,800));}
    if(__scenario==='visible_hold')hands=[[.4,.4]];
    return new Response(JSON.stringify({hands,wrists:[[.4,.4],[.6,.4]],elbows:[[.3,.65],[.7,.65]]}),{status:200});
   }
   if(String(url).startsWith('/api/analyze')){__uploads++;window.__uploadAt=performance.now();window.__uploadBytes=opts.body.size;return new Response(JSON.stringify({id:'0'.repeat(32)}),{status:202})}
   if(String(url).includes('/api/job/'+'0'.repeat(32))){await new Promise(r=>setTimeout(r,400));return new Response(JSON.stringify({status:'failed',error:'가상 촬영 흐름 확인 — 모델 실행 없음'}),{status:200})}
   return __realFetch(url,opts);
  };""")
  js('enableCamera()');until('stream && q.srcObject && q.readyState>=2')
  # Reproduce/check an old in-flight preview blocking every later capture.
  js("__scenario='hung';startCountdown()");until("mode==='countdown'");check('countdown begins at 3',js("$('countdown').textContent==='3'"))
  time.sleep(.4);check('no recording during countdown',js("mode==='countdown'&&recorder===null"))
  until('__hung',8);js("finishRecording('가상 수동 종료')");until("mode==='idle'",5)
  count=js('__previews');js("__scenario='one';enableCamera().then(()=>startCountdown())");until("mode==='recording'",5);time.sleep(.8)
  stuck=js('__previews')==count
  if before:
   check('reproduced pending preview permanently blocks next take',stuck)
   (A/'BEFORE_BROWSER_REPRODUCTION.json').write_text(json.dumps(dict(status='REPRODUCED',checks=checks,stuck_preview=stuck,hardware_camera=0,real_POSTs=0),ensure_ascii=False,indent=2),encoding='utf-8');print('REPRODUCED stuck preview',flush=True);return
  check('stop cancels pending preview',js('__aborted'))
  check('retake continues receiving previews',not stuck)
  until("mode==='idle'",8)
  for kind,minimum in [('one',1.9),('two',1.9),('brief',3.1),('one_remains',3.1),('initial_wait',3.1),('slow',2.5)]:
   uploads=js('__uploads');js(f"__scenario='{kind}';enableCamera().then(()=>startCountdown())");until("mode==='recording'",5)
   until(f'__uploads==={uploads+1}',10)
   check(kind+' loading immediately after stop',js("!$('analysisLoading').hidden"))
   elapsed=js('(__uploadAt-__recordStart)/1000');check(kind+' no premature stop',elapsed>=minimum)
   check(kind+' encoded recording submitted',js('__uploadBytes>100'))
   until("mode==='idle'",4);check(kind+' exactly one submission',js('__uploads')==uploads+1)
   check(kind+' restored after result/error',js("$('analysisLoading').hidden&&!$('record').disabled"))
   cases.append(dict(kind=kind,recording_seconds=elapsed));print(kind,round(elapsed,2),flush=True)
  js("__scenario='hung';__hung=false;__didHang=false;__aborted=false;enableCamera().then(()=>startCountdown())")
  until('__hung',8);until('__aborted',5);check('unresponsive request times out',True);until("mode==='idle'",8)
  check('no hardware camera calls',js('__hardwareCameraCalls===0'))
  check('no actual analysis or preview POST',not any(x['method']=='POST' for x in requests))
  check('no JS exceptions',not errors)
  check('local requests only',all(x['url'].startswith((BASE,'blob:'+BASE,'data:')) for x in requests))
  js('clearInterval(__timer);cameraOff()')
  (A/'BROWSER_VERIFIED.json').write_text(json.dumps(dict(status='PASS',checks=checks,count=len(checks),cases=cases,hardware_camera=0,feedback_inference=0,training=0,synthetic_MediaRecorder=True),ensure_ascii=False,indent=2),encoding='utf-8')
  print('PASS',len(checks),flush=True)
 except BaseException:
  (A/('BROWSER_FAILURE_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8');raise
 finally:
  if ws:ws.close()
  if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
  log.close()
if __name__=='__main__':main()
