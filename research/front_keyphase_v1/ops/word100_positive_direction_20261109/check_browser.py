from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'word100_action_feedback_demo_20261108'))
from task import *
U=Path(__file__).resolve().parent
import subprocess,urllib.request,websocket,base64,uuid,traceback
BASE='http://127.0.0.1:8814'
def main():
 profile=U/('browser_check_'+uuid.uuid4().hex[:8]);profile.mkdir();checks=[];errors=[];requests=[];ws=None
 def check(name,value):
  assert value,name
  checks.append(name)
 log=(profile/'chrome.log').open('w',encoding='utf-8');proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--autoplay-policy=no-user-gesture-required','--user-data-dir='+str(profile),'--window-size=1366,900','about:blank'],stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  tab=None
  for _ in range(100):
   try:
    port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(x for x in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list')) if x['type']=='page');break
   except (OSError,ValueError,StopIteration):time.sleep(.1)
  if tab is None:raise RuntimeError('Headless Chrome debugging connection unavailable; inspect chrome.log and local socket permissions')
  ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=20);seq=0
  def call(method,params=None):
   nonlocal seq
   seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
   while True:
    r=json.loads(ws.recv())
    if r.get('method')=='Runtime.exceptionThrown':errors.append(r)
    if r.get('method')=='Network.requestWillBeSent':requests.append(r['params']['request'])
    if r.get('id')==seq:
     assert not r.get('error'),r
     return r.get('result',{})
  def js(s):
   r=call('Runtime.evaluate',dict(expression=('(async()=>{'+s+'})()' if 'await loadResult' in s else s),returnByValue=True,awaitPromise=True));assert not r.get('exceptionDetails'),r;return r.get('result',{}).get('value')
  def until(s,seconds=20):
   deadline=time.monotonic()+seconds
   while time.monotonic()<deadline:
    if js(s):return
    time.sleep(.1)
   raise AssertionError(s)
  def shot(name):(U/name).write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=False))['data']))
  call('Runtime.enable');call('Page.enable');call('Network.enable');call('Page.navigate',dict(url=BASE));until("typeof meta!=='undefined' && meta && $('ref').readyState>=2")
  check('camera off initially',js("$('live').hidden && !$('placeholder').hidden && $('query').hidden"));check('no skeleton before result',js('result===null'));shot('UI_START.png')
  js("document.querySelectorAll('#examples button')[0].click()");until("result?.word==='WORD0739' && $('query').readyState>=2 && $('ref').readyState>=2");check('two public videos decoded',js("$('query').videoWidth>0 && $('ref').videoWidth>0"));check('horizontal top3',js("document.querySelectorAll('.rank').length===3 && new Set([...document.querySelectorAll('.rank')].map(x=>Math.round(x.getBoundingClientRect().top))).size===1"));check('no score bars',js("document.querySelectorAll('.ranks canvas,.ranks progress').length===0"));check('no whole word correctness',js('result.normal_or_wrong===null && result.overall_pass===false'))
  js('focusCard(0)');until("!$('ref').seeking&&!$('query').seeking");shot('UI_NORMAL.png');js("$('play').click()");until('playing');start=js("$('query').currentTime");time.sleep(.5);check('playback advances',js("$('query').currentTime")>start);js('pause()')
  js("document.querySelectorAll('#examples button')[1].click()");until('result?.synthetic===true');js('focusCard(2)');until("!$('ref').seeking");check('missing stage reference only',js("result.cards[2].query_indices===null && referenceOnly && $('query').hidden && !$('placeholder').hidden"));check('synthetic disclosed',js("$('notice').textContent.includes('일부러')"));check('nonassertive missing instruction',js("$('message').textContent.includes('확인') && !$('message').textContent.includes('틀렸')"));shot('UI_MISSING.png')
  # Pure UI fixture: this is not an AI performance result or a real learner clip.
  js("document.querySelectorAll('#examples button')[2].click()");until("result?.word==='WORD1133'")
  check('current positive evidence version',js("result.evidence_version==='positive_shape_and_signed_direction_v1'"))
  check('normal cue names checked features',js("result.cards[0].message.includes('오른손 펴기')&&result.cards[0].message.includes('이동 방향')"))
  js("window.__normal=JSON.parse(JSON.stringify(result));window.__fixture=JSON.parse(JSON.stringify(result));__fixture.synthetic=true;__fixture.edit_note='UI 표시 검사: 합성 응답이며 모델 평가 결과가 아닙니다.';__fixture.cards[0].display_status='difference';__fixture.cards[0].message='오른손이 닫힌 모양으로 보여요. 펴는 부분을 확인해 보세요.';__fixture.cards[0].difference_spans=[[__fixture.cards[0].query_indices[0]+2,__fixture.cards[0].query_indices[0]+4]];await loadResult(__fixture);focusCard(0)")
  check('orange cue separate from uncertainty',js("document.querySelectorAll('.card.difference').length===1&&document.querySelectorAll('.band.difference').length===1"))
  check('only contradictory frames orange',js("evidenceColor(result.cards[0],result.cards[0].difference_spans[0][0])==='#d87911'&&evidenceColor(result.cards[0],result.cards[0].query_indices[0])==='#8b95a5'"))
  until("$('query').readyState>=2&&!$('query').seeking&&!$('ref').seeking");shot('UI_ORANGE_FIXTURE.png');js('await loadResult(__normal);focusCard(0)');until("$('query').readyState>=2&&!$('query').seeking&&!$('ref').seeking");shot('UI_UPDATED_NORMAL.png')
  for width,height in [(1366,768),(720,1000),(390,844)]:
   call('Emulation.setDeviceMetricsOverride',dict(width=width,height=height,deviceScaleFactor=1,mobile=width<500));check(f'no overflow {width}',js('document.documentElement.scrollWidth<=innerWidth+2'));check(f'side by side {width}',js("Math.abs($('ref').getBoundingClientRect().top-$('query').parentNode.getBoundingClientRect().top)<3"));shot(f'UI_{width}.png')
  call('Emulation.setDeviceMetricsOverride',dict(width=1366,height=900,deviceScaleFactor=1,mobile=False))
  # Mock HTTP boundary only; never invoke the hardware camera route.
  js("window.__realFetch=fetch;window.__state='ready';window.__posts=[];window.fetch=async(url,opt)=>{if(String(url)==='/api/status')return new Response(JSON.stringify({state:__state,countdown:3,elapsed:1,has_result:false}),{status:200});if(opt?.method==='POST'){__posts.push(String(url));if(String(url)==='/api/open')__state='preview';if(String(url)==='/api/start')__state='countdown';if(String(url)==='/api/close')__state='ready';return new Response(JSON.stringify({ok:true}),{status:200})}return __realFetch(url,opt)}")
  until("!$('open').disabled");js("$('open').click()");until("previousState==='preview'");js("$('start').click()");until("previousState==='countdown'");check('countdown visible3',js("!$('count').hidden&&$('count').textContent==='3'"));js("__state='recording'");until("previousState==='recording'");check('manual finish available',js("!$('finish').disabled"));js("__state='analysing'");until("!$('busy').hidden");check('loading overlay visible',js("getComputedStyle($('busy')).position==='fixed'"));shot('UI_LOADING.png');js("window.fetch=__realFetch")
  check('no real camera or start POST',not any(r['method']=='POST' for r in requests));check('only local app requests',all(r['url'].startswith((BASE,'data:','blob:'+BASE)) for r in requests));check('no browser JS exceptions',not errors)
  write(U/'BROWSER_VERIFIED.json',dict(status='PASS',count=len(checks),checks=checks,errors=errors,camera=False,actual_capture_test=False,ui_mock_not_model_performance=True));print('PASS',len(checks))
 except BaseException:
  (U/f'BROWSER_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
 finally:
  if ws:ws.close()
  if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
  log.close()
if __name__=='__main__':main()
