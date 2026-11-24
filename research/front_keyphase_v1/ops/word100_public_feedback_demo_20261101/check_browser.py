"""Isolated headless browser test of our local demo; never controls user's tabs."""
from pathlib import Path
import json,time,subprocess,urllib.request,urllib.error,websocket,base64,hashlib,uuid
U=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8801'
def main():
 profile=U/('browser_test_'+uuid.uuid4().hex[:8]);profile.mkdir();ws=None;errors=[];checks=[]
 exe=Path('C:/Program Files/Google/Chrome/Application/chrome.exe');assert exe.is_file()
 log=(profile/'chrome_stderr.txt').open('w',encoding='utf8')
 proc=subprocess.Popen([str(exe),'--headless=new','--enable-logging=stderr','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1',
 '--disable-background-networking','--disable-component-update','--no-first-run','--user-data-dir='+str(profile),
 '--window-size=1440,1100','about:blank'],stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  tab=None
  for _ in range(100):
   try:
    port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(t for t in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=2)) if t['type']=='page');break
   except (OSError,ValueError,StopIteration):
    if proc.poll() is not None:break
    time.sleep(.1)
  if tab is None:
   log.flush()
   raise RuntimeError('Browser did not expose a test page; exit='+str(proc.poll())+'; log='+str(profile/'chrome_stderr.txt'))
  ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=30);seq=0
  def call(method,params=None):
   nonlocal seq
   seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
   while True:
    d=json.loads(ws.recv())
    if d.get('method')=='Runtime.exceptionThrown':errors.append(d)
    if d.get('id')==seq:
     assert not d.get('error'),d
     return d.get('result',{})
  def js(code):
   d=call('Runtime.evaluate',dict(expression=code,returnByValue=True,awaitPromise=True));assert not d.get('exceptionDetails'),d
   return d.get('result',{}).get('value')
  call('Runtime.enable');call('Page.enable');call('Page.navigate',dict(url=BASE))
  for _ in range(100):
   if js('window.demoState?.ready===true'):break
   time.sleep(.1)
  assert js('window.demoState?.ready===true'),js('document.body.innerText')
  ready="""new Promise((ok,no)=>{let k=0,t=setInterval(()=>{if([...document.querySelectorAll('video')].every(v=>v.readyState>=2&&!v.seeking&&v.videoWidth)){clearInterval(t);ok(true)}else if(++k>200){clearInterval(t);no('video timeout')}},40)})"""
  pairs=js('data.pairs.map(p=>[p.query,p.reference])');assert len(pairs)==12
  for q,r in pairs:
   js(f"$('word').value={json.dumps(q)};$('word').dispatchEvent(new Event('change'));$('reference').value={json.dumps(r)};$('reference').dispatchEvent(new Event('change'))")
   js(ready)
   stages=js("$('stages').children.length")
   for j in range(stages):
    js(f"$('stages').children[{j}].click()");js(ready)
    state=js("({...window.demoState,qtime:$('qvideo').currentTime,rtime:$('rvideo').currentTime,expectedQ:qclip.times[frame],expectedR:rclip.times[pair.query_to_reference[frame]],rows:$('features').children.length,description:$('description').textContent,feedback:$('feedback').textContent})")
    assert abs(state['qtime']-state['expectedQ'])<.04 and abs(state['rtime']-state['expectedR'])<.04,state
    assert state['rows']==7 and len(state['description'])>5 and len(state['feedback'])>10
    checks.append(dict(query=q,reference=r,stage=j,frame=state['frame'],reference_frame=state['referenceFrame']))
   prev=js('frame');js("$('next').click()");js(ready);assert js('frame')==min(prev+1,js('qclip.frames-1'))
  js("$('jumpgap').click()");js(ready)
  assert js('window.demoState.referenceFrame')==48
  assert '왼손 0/21' in js("$('rnote').textContent")
  assert js("$('features').textContent.includes('관절 근거 부족')")
  # Replayed videos actually decode, and timeline advances through native frames.
  played=js("(async()=>{let before=frame;$('play').click();await new Promise(r=>setTimeout(r,400));$('play').click();return {before,after:frame,paused:$('qvideo').paused}})()")
  assert played['after']>played['before'] and played['paused'],played
  js("$('jumpgap').click()");js(ready)
  colors=js("[...document.querySelectorAll('h1,h2,p,td,th,button,select')].filter(e=>e.getClientRects().length).map(e=>({color:getComputedStyle(e).color,bg:getComputedStyle(e).backgroundColor}))")
  assert all(c['color']!=c['bg'] for c in colors)
  (U/'DEMO_VIEW.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False))
  assert js('document.documentElement.scrollWidth<=innerWidth+2')
  js("$('features').scrollIntoView({block:'start'})")
  (U/'DEMO_FEATURES.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  assert not errors,errors
  with urllib.request.urlopen(urllib.request.Request(BASE+'/media/P065',headers={'Range':'bytes=0-31'}),timeout=3) as r:
   assert r.status==206 and len(r.read())==32
  for url,method in [('/media/UNKNOWN','GET'),('/../../AGENTS.md','GET'),('/data.json','POST')]:
   try:urllib.request.urlopen(urllib.request.Request(BASE+url,method=method),timeout=3);raise AssertionError('Endpoint allowed '+url)
   except urllib.error.HTTPError as e:assert e.code==(405 if method=='POST' else 404)
  out=dict(status='PASS',pair_count=len(pairs),stage_seek_checks=checks,native_frame_buttons=True,missing_hand_example=True,
    playback=played,video_decode=True,text_color_checked=True,narrow_layout_no_overflow=True,range_requests=True,
    arbitrary_files_and_post_blocked=True,js_errors=errors,method='isolated headless Chrome; computer-use kernels unavailable',
    actual_user_browser_not_automated=True,html_sha256=hashlib.sha256((U/'index.html').read_bytes()).hexdigest(),
    js_sha256=hashlib.sha256((U/'app.js').read_bytes()).hexdigest())
  with (U/'UI_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
  print('PASS 12 pairs, 39 stage seeks, native-frame steps, video playback, readability, HTTP access scope')
  call('Browser.close')
 except BaseException as e:
  p=U/('UI_FIRST_FAILURE_'+uuid.uuid4().hex[:8]+'.json');p.write_text(json.dumps(dict(error=str(e),checks=checks,js_errors=errors),ensure_ascii=False,indent=2),encoding='utf8');raise
 finally:
  if ws:ws.close()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
  log.close()
if __name__=='__main__':main()
