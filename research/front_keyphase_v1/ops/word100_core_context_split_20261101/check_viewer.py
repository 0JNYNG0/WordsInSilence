from pathlib import Path
import json,time,subprocess,urllib.request,websocket,base64,hashlib,uuid
U=Path(__file__).resolve().parent
def main():
 profile=U/('browser_'+uuid.uuid4().hex[:8]);profile.mkdir();ws=None;errors=[];checks=[]
 proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--user-data-dir='+str(profile),'--window-size=1440,1100','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  for _ in range(120):
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
  call('Runtime.enable');call('Page.enable');call('Page.navigate',dict(url='http://127.0.0.1:8794/'))
  for _ in range(100):
   if js('window.reportReady===true'):break
   time.sleep(.1)
  assert js('window.reportReady===true')
  ready="""new Promise((ok,no)=>{let k=0,t=setInterval(()=>{if([...document.querySelectorAll('video')].every(v=>v.readyState>=2&&!v.seeking&&v.videoWidth)){clearInterval(t);ok([...document.querySelectorAll('video')].map(v=>({src:v.currentSrc,t:v.currentTime,width:v.videoWidth})));}else if(++k>250){clearInterval(t);no('video timeout')}},40)})"""
  ids=js("cases.filter(c=>c.control==='identity'||c.kind==='person_pair'&&c.role!=='fit').map(c=>c._key)")
  assert len(ids)==28
  for caseid in ids:
   js(f"$('caseSelect').value={json.dumps(caseid)};selectCase({json.dumps(caseid)})")
   js(ready)
   for phase in ['entry','core','exit']:js(f"phase='{phase}';renderGroups()")
   js("seekCoreMiddle()");media=js(ready)
   assert len(media)==2 and all(v['width']>0 for v in media)
   checks.append(dict(caseid=caseid,media=media))
  js("selectCase(cases.find(c=>c.id==='P063_identity')._key);selectAnchor(current.anchors.find(a=>a.segment_id==='P063_s05'));seekCoreMiddle()")
  media=js(ready)
  assert js("$('syntheticWarning').hidden")
  captions=js("[$('queryCaption').textContent,$('referenceCaption').textContent]")
  assert all('75' in c for c in captions),captions
  q0=js("$('queryVideo').currentTime");js("$('oldPoint').click()");js(ready);assert abs(js("$('queryVideo').currentTime")-q0)<.04
  assert js("$('semanticVerdict').textContent==='판정하지 않음' && $('feedbackMessages').children.length>0")
  # A genuine saved location conflict: old combined context vs new core-only minimum.
  js("selectCase(cases.find(c=>c.id==='P070_swap_adjacent_core_blocks')._key);selectAnchor(current.anchors.find(a=>a.segment_id==='P070_s03'));seekCoreMiddle()")
  js(ready);new_t=js("$('queryVideo').currentTime")
  js("$('oldPoint').click()");js(ready);old_t=js("$('queryVideo').currentTime")
  assert abs(new_t-old_t)>.04,(new_t,old_t)
  assert js("anchor.feedback.messages.some(x=>x.includes('위치가 다릅니다')) && JSON.stringify([...$('feedbackMessages').children].map(x=>x.textContent))===JSON.stringify(anchor.feedback.messages)")
  js("selectCase(cases.find(c=>c.id==='P063_right_hand_unobserved')._key);seekCoreMiddle()")
  js(ready);assert js("anchor.state==='observation_insufficient' && $('semanticVerdict').textContent==='판정하지 않음'")
  # Synthetic mapped video remains explicitly original and follows original index.
  js("selectCase(cases.find(c=>c.id==='P063_swap_adjacent_core_blocks')._key);seekCoreMiddle()")
  js(ready);assert js("!$('syntheticWarning').hidden")
  mapped=js("(()=>{let z=path[Number($('pathSlider').value)],oi=current.original_indices[z[0]];return {expected:current.original_query_pts[oi],actual:$('queryVideo').currentTime}})()")
  assert abs(mapped['expected']-mapped['actual'])<.04,mapped
  played=js("(async()=>{let vs=[...document.querySelectorAll('video')],starts=vs.map(v=>v.currentTime);for(let v of vs)await v.play();await new Promise(r=>setTimeout(r,250));for(let v of vs)v.pause();return vs.map((v,i)=>v.currentTime>starts[i])})()");assert all(played)
  themechecks=[]
  for mode in ['light','dark']:
   js(f"theme('{mode}')")
   colors=js("[...document.querySelectorAll('h1,h2,h3,p,td,th,select,button,summary')].filter(x=>x.getClientRects().length).map(x=>({tag:x.tagName,text:getComputedStyle(x).color,bg:getComputedStyle(x).backgroundColor}))")
   # Verify no foreground equals its opaque own background; screenshot checks overall layout.
   assert all(c['text']!=c['bg'] for c in colors)
   themechecks.append(dict(theme=mode,elements=len(colors)))
  js("theme('light');selectCase(cases.find(c=>c.id==='P063_identity')._key);selectAnchor(current.anchors.find(a=>a.segment_id==='P063_s05'));seekCoreMiddle()")
  js(ready);js("$('casePanel').scrollIntoView({block:'start'})")
  (U/'VIEWER.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False));assert js('document.documentElement.scrollWidth<=innerWidth+2')
  assert not errors
  out=dict(status='PASS',cases=checks,stage='P063_s05',new_and_old_seek=True,conflict_seek=dict(case='P070_swap_adjacent_core_blocks',stage='P070_s03',new_time=new_t,old_time=old_t),missing_hand_retained=True,semantic_verdict_null=True,synthetic_original_mapping=mapped,both_playback=True,themes=themechecks,narrow_no_overflow=True,js_errors=errors,headless_chrome=True,actual_codex_app_UI_not_automated=True,html_sha256=hashlib.sha256((U/'index.html').read_bytes()).hexdigest(),data_sha256=hashlib.sha256((U/'REPORT_DATA.json').read_bytes()).hexdigest())
  (U/'UI_VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print('PASS 28cases/entry-core-exit/videos/old-new/syntheticmapping/themes')
  call('Browser.close')
 except Exception as e:
  (U/('UI_FAILURE_'+uuid.uuid4().hex[:8]+'.json')).write_text(json.dumps(dict(error=str(e),checks=checks,js_errors=errors),ensure_ascii=False,indent=2),encoding='utf8');raise
 finally:
  if ws:ws.close()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
if __name__=='__main__':main()
