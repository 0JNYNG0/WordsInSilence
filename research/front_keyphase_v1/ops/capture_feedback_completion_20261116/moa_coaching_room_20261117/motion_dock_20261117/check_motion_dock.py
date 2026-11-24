"""Observe actual rendered transforms and layout; no physical camera or inference."""
from pathlib import Path
import subprocess,urllib.request,websocket,json,time,base64,uuid,importlib.util
from check_browser import MOCK
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('previous_visual_clock',D.parent/'capture_sweep.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
HOOK="""import('./renderer.js').then(({StageRenderer})=>{
 const begin=StageRenderer.prototype.begin,asset=StageRenderer.prototype.asset;
 StageRenderer.prototype.begin=function(...args){this.__i=0;return begin.apply(this,args)};
 StageRenderer.prototype.asset=function(a,m,...args){
  if(this.__i===0)window.__robot={x:m[12],y:m[13],z:m[14],yaw:Math.atan2(-m[2],m[0]),scene:document.body.dataset.scene};
  if(this.__i===1){const hip=a.nodes.find(n=>/Hips$/i.test(n.name||''));if(hip){const p=[hip.world[12],hip.world[13],hip.world[14]];window.__hip=this.project([0,1,2].map(r=>m[12+r]+p[0]*m[r]+p[1]*m[4+r]+p[2]*m[8+r]));}}
  this.__i++;return asset.call(this,a,m,...args);
 };})"""
def main():
 profile=D/'qa'/('motion_'+uuid.uuid4().hex[:8]);profile.mkdir();log=(profile/'chrome.log').open('w',encoding='utf-8')
 proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--autoplay-policy=no-user-gesture-required','--user-data-dir='+str(profile),'about:blank'],stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
 ws=None;seq=0;errors=[];checks=[];traces={};layouts=[]
 try:
  for _ in range(100):
   try:
    port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(x for x in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list')) if x['type']=='page');break
   except (OSError,ValueError,StopIteration):time.sleep(.1)
  ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=30)
  def call(method,params=None):
   nonlocal seq
   seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
   while True:
    r=json.loads(ws.recv())
    if r.get('method')=='Runtime.exceptionThrown':errors.append(r)
    if r.get('id')==seq:assert not r.get('error'),r;return r.get('result',{})
  def js(expr):
   r=call('Runtime.evaluate',dict(expression=expr,returnByValue=True,awaitPromise=True));assert not r.get('exceptionDetails'),r;return r.get('result',{}).get('value')
  def until(expr):
   for _ in range(160):
    if js(expr):return
    time.sleep(.1)
   raise AssertionError(expr)
  def check(name,ok):checks.append(dict(name=name,passed=bool(ok)));assert ok,name
  def shot(name):(D/'qa'/name).write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data']))
  def trace(name,total):
   rows=[]
   for ms in range(100,total+1,100):
    js('__advanceVisual(100)');q=js('__robot');q['ms']=ms;rows.append(q)
   traces[name]=rows;return rows
  call('Runtime.enable');call('Page.enable');call('Page.addScriptToEvaluateOnNewDocument',dict(source=MOCK+old.CLOCK))
  call('Emulation.setDeviceMetricsOverride',dict(width=1366,height=768,deviceScaleFactor=1,mobile=False));call('Page.navigate',dict(url='http://127.0.0.1:8822/'))
  until("document.body?.classList.contains('ready')&&document.querySelector('#word')?.options.length>0")
  js(HOOK);time.sleep(.2);js('__holdVisual()');time.sleep(.05)
  js("document.querySelector('[data-scene=practice]').click()")
  enter=trace('enter',4400)
  check('departure faster than previous 2365ms',next(r['ms'] for r in enter if r['y']>5.15)<=2100)
  entry=[r for r in enter if 3200<r['ms']<4200]
  check('right reentry spreads over almost one second',entry[0]['x']>2.8 and entry[-1]['x']<1.16)
  check('practice finishes at inward gaze',abs(enter[-1]['yaw']+.55)<.04)
  js("document.querySelector('#next').click();document.querySelector('#next').click()")
  leave=trace('leave',1400)
  check('return turns toward front during movement',all(abs(r['yaw']+.55*r['x']/1.13)<.04 for r in leave))
  check('return reaches front without final snap',abs(leave[-1]['x'])<.01 and abs(leave[-1]['yaw'])<.04 and abs(leave[-2]['yaw'])<.04)
  js("document.querySelector('#next').click();document.querySelector('#next').click()")
  quest=trace('quest',2500)
  check('quest turns inward while moving right',all(abs(r['yaw']+.55*r['x']/1.13)<.04 for r in quest))
  check('quest gaze settled before NPC reveal completes',abs(quest[10]['yaw']+.55)<.04 and quest[-1]['scene']=='quest')
  js("document.querySelector('[data-scene=practice]').click()")
  back=trace('quest_to_practice',5800)
  check('quest to practice returns through center',back[0]['scene']=='deprojecting' and back[0]['x']>.9 and back[-1]['scene']=='practice')
  for w,h in [(1366,768),(1280,720),(1920,1080),(720,1000)]:
   call('Emulation.setDeviceMetricsOverride',dict(width=w,height=h,deviceScaleFactor=1,mobile=False));time.sleep(.25);js('__advanceVisual(200)')
   q=js("(()=>{const r=id=>{const x=document.getElementById(id).getBoundingClientRect();return {left:x.left,top:x.top,right:x.right,bottom:x.bottom,width:x.width,height:x.height}};return {viewport:[innerWidth,innerHeight],camera:r('self-view'),image:r('live-camera'),controls:r('live-controls'),hip:__hip,fit:getComputedStyle(document.querySelector('#live-camera')).objectFit}})()")
   layouts.append(q);shot(f'dock_{w}x{h}.png')
   check(f'{w} webcam near bottom',q['camera']['bottom']>=h-15 and q['camera']['bottom']<=h)
   check(f'{w} controls beside camera',q['controls']['right']+8<=q['camera']['left'])
   check(f'{w} torso unobstructed',q['camera']['top']>q['hip'][1])
   check(f'{w} preview uncropped',q['fit']=='contain')
  for width,height in [(640,480),(1600,900),(480,640)]:
   js(f"document.querySelector('#live-camera').src='data:image/svg+xml,'+encodeURIComponent('<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\"><rect width=\"100%\" height=\"100%\" fill=\"gray\"/></svg>')")
   until(f"document.querySelector('#live-camera').naturalWidth==={width}");time.sleep(.1)
   ratio=js("(()=>{const r=document.querySelector('#live-camera').getBoundingClientRect();return r.width/r.height})()")
   check(f'native {width}x{height} aspect preserved',abs(ratio-width/height)<.003)
  check('no JavaScript errors',not errors)
  out=dict(passed=True,checks=checks,traces=traces,layouts=layouts,physical_camera=False,private_data_read=False,scope='Actual WebGL transforms under controlled animation clock; mock camera and public fixture')
  (D/'qa/MOTION_DOCK_CHECK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print('MOTION/DOCK PASS',len(checks))
 except BaseException:
  (D/'qa/MOTION_FAILURE.json').write_text(json.dumps(dict(checks=checks,traces=traces,layouts=layouts,errors=errors),ensure_ascii=False,indent=2),encoding='utf-8')
  if ws:
   try:shot('MOTION_FAILURE.png')
   except Exception:pass
  raise
 finally:
  if ws:ws.close()
  if proc.poll() is None:proc.terminate();proc.wait(10)
  log.close()
if __name__=='__main__':main()
