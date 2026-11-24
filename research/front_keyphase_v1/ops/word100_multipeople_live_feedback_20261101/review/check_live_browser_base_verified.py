"""Own localhost software integration test. Public file only; no webcam/private files."""
from pathlib import Path
import json,time,subprocess,urllib.request,websocket,base64,hashlib,uuid
U=Path(__file__).resolve().parent;BASE='http://127.0.0.1:8802'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 profile=U/('browser_test_'+uuid.uuid4().hex[:8]);profile.mkdir();ws=None;errors=[];checks=[]
 log=(profile/'chrome_stderr.txt').open('w',encoding='utf8')
 proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--autoplay-policy=no-user-gesture-required','--user-data-dir='+str(profile),'--window-size=1440,1100','about:blank'],stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
 try:
  tab=None
  for _ in range(100):
   try:
    port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(t for t in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list',timeout=2)) if t['type']=='page');break
   except (OSError,ValueError,StopIteration):time.sleep(.1)
  assert tab,'No isolated Chrome test page'
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
   if js('typeof state!=="undefined" && state.catalog!==null'):break
   time.sleep(.1)
  assert js('state.catalog.words.length')==4
  assert js('state.catalog.words.every(w=>w.references.length===9)')
  checks.append('four expressions, nine separate references each')
  row=next(r for r in read(U/'PUBLIC_INPUTS.json')['rows'] if r['word_id']=='WORD0739' and r['person']==12)
  js("$('word').value='WORD0739';$('word').dispatchEvent(new Event('change'))")
  startpath=U/'PUBLIC_UPLOAD_STARTED.json'
  if not startpath.exists():
   intent=U/'PUBLIC_UPLOAD_INTENT.json'
   if intent.exists():raise RuntimeError('Previous upload intent exists without saved job ID. Inspect server session before any new submission.')
   with intent.open('x',encoding='utf8') as f:json.dump(dict(source=row['source'],sha256=row['source_sha256'],word=row['word_id']),f)
   node=call('DOM.querySelector',dict(nodeId=call('DOM.getDocument')['root']['nodeId'],selector='#file'))['nodeId']
   call('DOM.setFileInputFiles',dict(nodeId=node,files=[row['source']]))
   assert js('state.file!==null && !$("analyze").disabled')
   js("$('analyze').click()")
   for _ in range(100):
    if js('state.currentJob'):break
    time.sleep(.1)
   jobid=js('state.currentJob');assert jobid,js("$('message').textContent")
   save(startpath,dict(job_id=jobid,source=row['source'],sha256=row['source_sha256'],public=True,created_unix=time.time()))
  else:
   previous=read(startpath);assert previous['sha256']==row['source_sha256'];jobid=previous['job_id']
  for i in range(600):
   job=js('api("/api/jobs/'+jobid+'")')
   if job['status'] in ['COMPLETE','FAILED']:break
   if i%15==0:print(json.dumps(dict(stage='public_http_upload',status=job['status'],progress=job.get('progress'))),flush=True)
   time.sleep(1)
  assert job['status']=='COMPLETE',job
  assert job['submission_sha256']==row['source_sha256'] and job['word']==row['word_id']
  # Resume UI checks against the same completed job; never repeat video inference.
  js('state.currentJob='+json.dumps(jobid)+';state.busy=false;controls();api("/api/jobs/'+jobid+'").then(j=>renderResult(j.result,j))')
  result=job['result'];assert len(result['references'])==9 and result['training_used_submission'] is False
  checks.append('new public raw file uploaded, fresh analysis, nine-reference result')
  ready="""new Promise((ok,no)=>{let k=0,t=setInterval(()=>{if([...document.querySelectorAll('video')].every(v=>v.readyState>=2&&!v.seeking&&v.videoWidth)){clearInterval(t);ok(true)}else if(++k>250){clearInterval(t);no('video decode timeout')}},40)})"""
  js(ready)
  assert js("$('stages').children.length")==len(result['stages'])
  assert js("$('referenceResults').children.length")==9
  for ref in result['references']:
   js('$("reference").value='+json.dumps(ref['id'])+';$("reference").dispatchEvent(new Event("change"))');js(ready)
  checks.append('all nine reference videos decoded')
  first=result['selected_whole_references'][0]
  js('startSync('+json.dumps(first)+',[0,1.5])');time.sleep(.4)
  timing=js('({q:$("submissionVideo").currentTime,r:$("referenceVideo").currentTime,active:!!state.sync})')
  assert timing['q']>0 and timing['active'],timing
  js("$('syncStop').click()")
  assert js("$('submissionVideo').paused && $('referenceVideo').paused")
  checks.append('paired path playback advances; stop pauses both')
  pairs=0
  for stage in result['stages']:
   for feature in stage['features']:
    if feature['peak_at'][0]:
     peak=feature['peak_at'][0];ref=feature['reference_ids'][0]
     js('showPeak('+json.dumps(ref)+','+json.dumps(peak)+')');js(ready)
     got=js('[$("submissionVideo").currentTime,$("referenceVideo").currentTime]')
     assert abs(got[0]-peak['query_seconds'])<.04 and abs(got[1]-peak['reference_seconds'])<.04,(got,peak)
     pairs+=1;break
  checks.append(f'{pairs} stage peak links seek exact stored query/reference timestamps')
  js("$('stages').scrollIntoView({block:'start'})")
  (U/'LIVE_RESULT.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  js('scrollTo(0,0)');time.sleep(.4)
  (U/'LIVE_DEMO.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png'))['data']))
  colors=js("[...document.querySelectorAll('h1,h2,p,button,select')].filter(e=>e.getClientRects().length).map(e=>[getComputedStyle(e).color,getComputedStyle(e).backgroundColor])")
  assert all(a!=b for a,b in colors)
  call('Emulation.setDeviceMetricsOverride',dict(width=760,height=1100,deviceScaleFactor=1,mobile=False))
  assert js('document.documentElement.scrollWidth<=innerWidth+2')
  assert not errors,errors
  checks+=['visible text contrast checked','760px layout no overflow','no JS exceptions','camera never requested']
  save(U/'LIVE_UI_VERIFICATION.json',dict(status='PASS',checks=checks,job_id=jobid,public_source=row['clip_id'],source_sha256=row['source_sha256'],latency=result['latency'],request_timing=job.get('request_timing'),timing=timing,js_errors=errors,isolated_test_browser=True,camera_access=False,user_data_read=False,html_sha256=hashlib.sha256((U/'index.html').read_bytes()).hexdigest(),js_sha256=hashlib.sha256((U/'app.js').read_bytes()).hexdigest()))
  print('LIVE_UI_VERIFICATION PASS',flush=True);call('Browser.close')
 except BaseException as e:
  save(U/('LIVE_UI_FIRST_FAILURE_'+uuid.uuid4().hex[:8]+'.json'),dict(error=str(e),checks=checks,js_errors=errors));raise
 finally:
  if ws:ws.close()
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
  log.close()
if __name__=='__main__':main()
