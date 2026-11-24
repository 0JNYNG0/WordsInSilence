from pathlib import Path
import subprocess,urllib.request,websocket,base64,uuid,traceback,json,time
D=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8822'
MOCK=r'''
window.__qa={state:'ready',posts:[],attempt:'aaaaaaaaaaaaaaaaaaaaaaaa',resultSeen:false};
const realFetch=window.fetch.bind(window),response=x=>new Response(JSON.stringify(x),{status:200,headers:{'Content-Type':'application/json'}});
const imageSrc=Object.getOwnPropertyDescriptor(HTMLImageElement.prototype,'src');Object.defineProperty(HTMLImageElement.prototype,'src',{...imageSrc,set(v){imageSrc.set.call(this,String(v).includes('/preview.mjpeg')?'/qa/public.jpg':v)}});
const videoSrc=Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype,'src');Object.defineProperty(HTMLMediaElement.prototype,'src',{...videoSrc,set(v){videoSrc.set.call(this,String(v).includes('/session-recording/')?'/public-replay/WORD0099.mp4':v)}});
window.fetch=async(url,opt)=>{const p=String(url);if(p==='/api/status')return response({state:__qa.state,auto_end_ready:true,elapsed:1,countdown:3,has_result:__qa.state==='done',attempt_id:__qa.attempt,preparation_message:'준비됐어요. 촬영을 시작해 주세요.'});
if(p.startsWith('/api/result?')){__qa.resultSeen=true;return realFetch('/qa/result.json')}
if(opt?.method==='POST'){__qa.posts.push(p);if(p==='/api/open')__qa.state='preview';if(p==='/api/start')__qa.state='countdown';if(p==='/api/finish')__qa.state='done';if(p==='/api/close')__qa.state='ready';return response({ok:true,attempt_id:__qa.attempt,stop_to_browser_ack_seconds:.7,stop_to_result_seconds:.4});}return realFetch(url,opt)};
'''
def main():
 profile=D/'qa'/('chrome_'+uuid.uuid4().hex[:7]);profile.mkdir();checks=[];errors=[];requests=[];ws=None
 log=(profile/'chrome.log').open('w',encoding='utf-8')
 proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--autoplay-policy=no-user-gesture-required','--user-data-dir='+str(profile),'--window-size=1440,960','about:blank'],stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
 def check(name,v):
  checks.append(dict(name=name,passed=bool(v)));assert v,name
 try:
  for _ in range(100):
   try:
    port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(x for x in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list'))if x['type']=='page');break
   except (OSError,ValueError,StopIteration):time.sleep(.1)
  ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=30);seq=0
  def call(method,params=None):
   nonlocal seq
   seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
   while True:
    r=json.loads(ws.recv())
    if r.get('method')=='Runtime.exceptionThrown':errors.append(r)
    if r.get('method')=='Network.requestWillBeSent':requests.append(r['params']['request'])
    if r.get('id')==seq:
     assert not r.get('error'),r;return r.get('result',{})
  def js(s):
   r=call('Runtime.evaluate',dict(expression=s,returnByValue=True,awaitPromise=True));assert not r.get('exceptionDetails'),r;return r.get('result',{}).get('value')
  def until(s,sec=25):
   deadline=time.monotonic()+sec
   while time.monotonic()<deadline:
    if js(s):return
    time.sleep(.15)
   raise AssertionError(s)
  def shot(name):(D/'qa'/name).write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data']))
  def space():
   call('Input.dispatchKeyEvent',dict(type='keyDown',code='Space',key=' ',windowsVirtualKeyCode=32));call('Input.dispatchKeyEvent',dict(type='keyUp',code='Space',key=' ',windowsVirtualKeyCode=32))
  call('Runtime.enable');call('Page.enable');call('Network.enable');call('Page.addScriptToEvaluateOnNewDocument',dict(source=MOCK));call('Page.navigate',dict(url=BASE))
  until("document.body?.classList.contains('ready')&&document.querySelector('#word')?.options.length>0")
  check('intro does not request camera',js('__qa.posts.length===0'))
  js("document.querySelector('[data-scene=practice]').click()")
  start=time.monotonic()
  for at,name in [(1.15,'story_1.png'),(1.85,'story_2.png'),(2.50,'story_3.png'),(4.25,'story_4.png')]:
   while time.monotonic()-start<at:time.sleep(.02)
   shot(name)
  check('no physical camera POST',not any(r['method']=='POST'for r in requests))
  (D/'qa/STORY_CAPTURE.json').write_text(json.dumps(dict(passed=True,physical_camera=False,rendered_actual_UI=True,checks=checks)),encoding='utf-8')
  print('STORY CAPTURED')

 except BaseException:
  (D/'qa/BROWSER_FAILURE.txt').write_text(traceback.format_exc()+json.dumps(errors,ensure_ascii=False),encoding='utf-8')
  if ws:
   try:shot('FAILURE.png')
   except Exception:pass
  raise
 finally:
  if ws:ws.close()
  if proc.poll()is None:proc.terminate();proc.wait(10)
  log.close()
if __name__=='__main__':main()
