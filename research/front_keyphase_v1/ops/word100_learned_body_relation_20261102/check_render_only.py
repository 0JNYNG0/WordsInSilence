"""Owned local web app integration test, public-only inputs, no camera."""
from pathlib import Path
import json,time,subprocess,urllib.request,websocket,base64,uuid,traceback
U=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8806'
def main():
 assert not (U/'DEMO_RENDER_VERIFICATION.json').exists()
 profile=U/('ui_check_'+uuid.uuid4().hex[:8]);profile.mkdir();errors=[];checks=[];ws=None
 log=(profile/'chrome.log').open('w',encoding='utf8')
 proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0',
  '--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run',
  '--autoplay-policy=no-user-gesture-required','--user-data-dir='+str(profile),'--window-size=1400,1050','about:blank'],stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  tab=None
  for _ in range(100):
   try:
    port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(t for t in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=2)) if t['type']=='page');break
   except (OSError,ValueError,StopIteration):time.sleep(.1)
  assert tab;ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=30);seq=0
  def call(method,params=None):
   nonlocal seq
   seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
   while True:
    d=json.loads(ws.recv())
    if d.get('method')=='Runtime.exceptionThrown':errors.append(d)
    if d.get('id')==seq:
     assert not d.get('error'),d;return d.get('result',{})
  def js(code):
   d=call('Runtime.evaluate',dict(expression=code,returnByValue=True,awaitPromise=True));assert not d.get('exceptionDetails'),d
   return d.get('result',{}).get('value')
  call('Runtime.enable');call('Page.enable');call('Page.navigate',dict(url=BASE))
  for _ in range(200):
   if js('typeof ready!=="undefined"&&ready'):break
   time.sleep(.1)
  assert js('ready');assert js('stream===null');checks.append('model ready; camera never opened')
  for word in ['WORD0739','WORD1133','WORD1567','WORD2523']:
   js("$('word').value="+json.dumps(word)+";setWord();$('public').click()")
   for _ in range(100):
    if js('result!==null&&$("query").readyState>=2&&$("reference").readyState>=2'):break
    time.sleep(.1)
   assert js('result.word_id')==word
   assert js('result.individual_reference_comparisons')==0 and js('result.learned_prototype_comparisons')==1
   assert js('result.overlay.points.length===result.frames')
   assert js('document.querySelectorAll(".stage").length===result.stages.length')
   js('document.querySelector(".stage button").click()');time.sleep(.25)
   assert js('$("query").currentTime>=result.stages[0].start_seconds')
   js('$("query").pause()');checks.append(word+' public result, decoded both videos, full skeleton, timed replay')
  for word in ['WORD0739','WORD1133','WORD1567','WORD2523']:
   js("$('word').value="+json.dumps(word)+";setWord();$('publicCut').click()")
   for _ in range(100):
    if js('result!==null&&$("query").readyState>=2'):break
    time.sleep(.1)
   assert js('result.word_id')==word
   assert js('result.overlay.points.length===result.frames')
   assert js('$("query").currentSrc.includes("/media/control/")')
   checks.append(word+' edited public comparison visible without any new inference')
  assert js('result.stages[0].status')=='IMPORTANT_RELATION_UNCONFIRMED'
  assert js('$("result").textContent.includes("양손의 긴 손가락 끝이 한곳으로 모임")')
  assert js('$("result").textContent.includes("이 관계를 확인하지 못함")')
  assert not js('$("result").textContent.includes("IMAGE_ESTIMATE_NEEDS_CONFIRMATION")')
  assert js('document.querySelector(".feature").textContent.includes("배운 기준:")')
  assert js('document.querySelector(".feature").textContent.includes("영상 추정:")')
  checks.append('positive/negative feature presence shown explicitly, avoiding contradictory all-similar labels')
  js("scrollTo(0,0)");(U/'DEMO_CONTROL_SCREEN.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',{'format':'png'})['data']))
  js("$('result').scrollIntoView()");(U/'DEMO_CONTROL_RESULT.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',{'format':'png'})['data']))
  colors=js("[...document.querySelectorAll('h1,h2,p,button,select')].filter(e=>e.getClientRects().length).map(e=>[getComputedStyle(e).color,getComputedStyle(e).backgroundColor])")
  assert all(a!=b for a,b in colors)
  call('Emulation.setDeviceMetricsOverride',dict(width=700,height=1000,deviceScaleFactor=1,mobile=False))
  assert js('document.documentElement.scrollWidth<=innerWidth+2')
  assert not errors,errors
  result=dict(status='PASS',checks=checks+['no JS exceptions','visible contrast','700px no overflow'],camera_access=False,private_footage_access=False,
   saved_example_seconds=js('result.latency.total_analysis_seconds'),individual_reference_comparisons=0)
  (U/'DEMO_RENDER_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False),flush=True)
 finally:
  if ws:
   try:call('Browser.close')
   except Exception:pass
   ws.close()
  if proc.poll() is None:proc.terminate()
  log.close()
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'DEMO_RENDER_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
