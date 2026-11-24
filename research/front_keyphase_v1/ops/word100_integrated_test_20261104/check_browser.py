"""Isolated browser integration tests. Camera is always a synthetic canvas."""
from pathlib import Path
import json,time,subprocess,urllib.request,websocket,base64,uuid,traceback
U=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8812'
def main():
 profile=U/('browser_check_'+uuid.uuid4().hex[:8]);profile.mkdir();checks=[];errors=[];requests=[];ws=None
 def check(n,v):assert v,n;checks.append(n)
 log=(profile/'chrome.stderr.log').open('w',encoding='utf8')
 proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--autoplay-policy=no-user-gesture-required','--user-data-dir='+str(profile),'--window-size=1440,1080','about:blank'],stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  tab=None
  for _ in range(100):
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
   deadline=time.monotonic()+seconds
   while time.monotonic()<deadline:
    if js(code):return
    time.sleep(.1)
   raise AssertionError('Not reached '+code)
  def shot(name):(U/name).write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data']))
  call('Runtime.enable');call('Network.enable');call('Page.enable')
  call('Page.addScriptToEvaluateOnNewDocument',dict(source="window.__hardwareCameraCalls=0;navigator.mediaDevices.getUserMedia=async()=>{window.__hardwareCameraCalls++;throw Error('Hardware camera forbidden in UI test');};"))
  call('Page.navigate',dict(url=BASE));until("typeof state!=='undefined' && state?.ready===true",40)
  js("document.getElementById('example').click()");until("result?.word_id==='WORD1133' && mode==='idle'")
  until("[q,rv].every(v=>v.readyState>=2&&!v.seeking)")
  check('two public videos decoded',js('[q,rv].every(v=>v.videoWidth>0)'))
  check('exactly three TCN labels',js("document.querySelectorAll('.rank').length===3"))
  check('target T3F first',js("document.querySelector('.rank.target').textContent.includes('1위')"))
  check('horizontal ranking',js("new Set([...document.querySelectorAll('.rank')].map(x=>Math.round(x.getBoundingClientRect().top))).size===1"))
  check('no score bars',js("document.querySelectorAll('.top3 canvas,.top3 progress').length===0"))
  check('distinct phase and evidence strips',js("phase.length===result.overlay.times.length && document.getElementById('phaseBar')!==document.getElementById('evidenceBar')"))
  check('whole query mapped',js('alignment.length===result.overlay.times.length'))
  check('single learned DTW',js('result.learned_prototype_comparisons===1 && result.runtime_individual_reference_video_comparisons===0'))
  check('no whole-sign correctness claim',js('result.whole_sign_correctness===null'))
  shot('UI_DESKTOP.png')
  js("document.getElementById('model').value='C';document.getElementById('model').dispatchEvent(new Event('change'))")
  check('preserved C ranks target second',js("document.querySelector('.rank.target').textContent.includes('2위')"))
  js("document.getElementById('model').value='T3F';renderTop();seek(.1)")
  until('!q.seeking&&!rv.seeking');check('preparation labelled candidate',js("document.getElementById('feedbackCard').textContent.includes('추정')"))
  js("seek(result.stages[0].start_seconds);document.getElementById('play').click()")
  until('playing');initial=js('q.currentTime');time.sleep(.5);check('query replay advances',js('q.currentTime')>initial)
  until('!playing',12);check('whole clip reaches end',js('q.currentTime>=result.overlay.times.at(-1)-.1'))
  js("seek(result.stages[0].start_seconds)");until('!q.seeking&&!rv.seeking')
  check('reference from canonical DTW',js('Math.abs(rv.currentTime-alignment[IntegratedLogic.nearest(result.overlay.times,q.currentTime)])<.04'))
  # Synthetic evidence fixture checks UI color/uncertainty, never saved as a model result.
  js("window.__saved=structuredClone(result);result.stages[0].features[0].status='DIFFERENCE_CANDIDATE';seek(result.stages[0].start_seconds)")
  check('difference is candidate not correction verdict',js("document.getElementById('feedbackCard').textContent.includes('차이 후보')"));shot('UI_SYNTHETIC_DIFFERENCE.png')
  js('result=window.__saved;seek(result.stages[0].start_seconds)')
  for word in ['WORD0739','WORD1567','WORD2523','WORD1133']:
   js(f"document.getElementById('word').value='{word}';changeWord()")
   until(f"ref?.word_id==='{word}' && rv.readyState>=2")
   check(word+' public reference decoded',js('rv.videoWidth>0'))
  js("document.getElementById('example').click()");until('result && mode===\'idle\'');until('!q.seeking&&!rv.seeking')
  for width,height in [(1366,768),(720,1000),(390,844)]:
   call('Emulation.setDeviceMetricsOverride',dict(width=width,height=height,deviceScaleFactor=1,mobile=width<500))
   check(str(width)+' no overflow',js('document.documentElement.scrollWidth<=innerWidth+2'))
   check(str(width)+' videos remain side by side',js('Math.abs(q.getBoundingClientRect().top-rv.getBoundingClientRect().top)<3'))
   if width==1366:check('1366 timeline visible',js("document.getElementById('evidenceBar').getBoundingClientRect().bottom<innerHeight"))
   shot('UI_'+str(width)+'.png')
  call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1080,deviceScaleFactor=1,mobile=False))
  # Capture UX uses fake webcam canvas; POSTs intercepted before network.
  js("""window.__fakeCanvas=document.createElement('canvas');__fakeCanvas.width=640;__fakeCanvas.height=360;
  window.__drawTimer=setInterval(()=>{let c=__fakeCanvas.getContext('2d');c.fillStyle='#247b90';c.fillRect(0,0,640,360);c.fillStyle='white';c.fillText(String(performance.now()),10,40)},30);
  navigator.mediaDevices.getUserMedia=async()=>__fakeCanvas.captureStream(30);
  window.__realFetch=fetch;window.__uploads=0;window.__previews=0;window.__startAt=0;
  window.fetch=async(url,opts)=>{if(String(url)==='/api/preview'){window.__previews++;let h=__previews<=5?[[.4,.4]]:__previews<=8?[[.4,.88]]:[];let w=__previews<=5?[[.4,.4],[.7,.85]]:[[.4,.9],[.7,.92]];return new Response(JSON.stringify({hands:h,wrists:w,elbows:[[.3,.65],[.7,.65]]}),{status:200})}
  if(String(url).startsWith('/api/analyze')){window.__uploads++;window.__submittedAt=performance.now();return new Response(JSON.stringify({id:'00000000000000000000000000000001'}),{status:202})}
  if(String(url).includes('/api/job/00000000000000000000000000000001'))return new Response(JSON.stringify({status:'failed',error:'합성 촬영 흐름 시험 완료 (실제 분석 호출 없음)'}),{status:200});return __realFetch(url,opts)};""")
  js("document.getElementById('camera').click()");until('stream && q.srcObject && q.readyState>=2')
  js("document.getElementById('record').click()");until("mode==='countdown'");check('countdown begins at three',js("document.getElementById('countdown').textContent==='3'"))
  js("document.getElementById('stop').click()");check('countdown cancel no recorder/upload',js("mode==='idle'&&!recorder&&__uploads===0"))
  js("window.__startAt=performance.now();document.getElementById('record').click()")
  time.sleep(1);check('no capture before countdown',js("mode==='countdown'&&recorder===null&&__uploads===0"))
  until("mode==='recording'",5);check('three seconds before actual record',js('performance.now()-__startAt>=2950'))
  until('__uploads===1',10);until("mode==='idle'",4)
  check('automatic stop triggers one automatic analyze',js('__uploads===1&&__previews>=12'))
  check('record stop controls reset',js("!document.getElementById('record').disabled&&document.getElementById('stop').disabled"))
  js("clearInterval(__drawTimer);window.fetch=__realFetch;cameraOff()")
  check('tracks released',js('stream===null'))
  check('hardware camera never requested',js('__hardwareCameraCalls===0'))
  check('no real analyze POST',not any(r['method']=='POST' for r in requests))
  check('local requests only',all(r['url'].startswith((BASE,'blob:'+BASE,'data:')) for r in requests))
  check('no browser runtime exceptions',not errors)
  (U/'BROWSER_VERIFIED.json').write_text(json.dumps(dict(status='PASS',checks=checks,count=len(checks),hardware_camera=0,real_user_video=0,mock_capture_analysis_calls=1,requests=[dict(url=r['url'],method=r['method']) for r in requests],errors=errors,verification='Same Codex isolated UI regression; synthetic capture and existing public playback, not real webcam or accuracy test'),ensure_ascii=False,indent=2),encoding='utf8');print('PASS',len(checks))
 except BaseException:
  (U/('BROWSER_FAILURE_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf8');raise
 finally:
  if ws:ws.close()
  if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
  log.close()
if __name__=='__main__':main()
