"""Static saved-public-result browser QA. No camera or inference."""
from pathlib import Path
import json,time,threading,subprocess,urllib.request,websocket,uuid,mimetypes,re,traceback,base64
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit,unquote
U=Path(__file__).resolve().parent
requests=[]
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_POST(self):self.send_error(403)
 def do_GET(self):
  route=unquote(urlsplit(self.path).path);requests.append(route);p=(U/('RESULT_KO.html' if route=='/' else route.lstrip('/'))).resolve()
  if not p.is_relative_to(U) or p.suffix not in ['.html','.js','.mp4','.json','.txt'] or not p.exists():self.send_error(404);return
  data=p.read_bytes();total=len(data);a=0;b=total-1;m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers.get('Range',''))
  if m:a=int(m[1]);b=min(b,int(m[2])) if m[2] else b
  self.send_response(206 if m else 200);self.send_header('Content-Type',mimetypes.guess_type(str(p))[0] or 'application/octet-stream');self.send_header('Content-Length',str(b-a+1));self.send_header('Accept-Ranges','bytes')
  if m:self.send_header('Content-Range',f'bytes {a}-{b}/{total}')
  self.end_headers()
  try:self.wfile.write(data[a:b+1])
  except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
def main():
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
 profile=U/('browser_'+uuid.uuid4().hex[:8]);profile.mkdir();log=(profile/'chrome.log').open('w');proc=subprocess.Popen(['C:/Program Files/Google/Chrome/Application/chrome.exe','--headless=new','--remote-debugging-port=0','--remote-allow-origins=http://127.0.0.1','--disable-background-networking','--disable-component-update','--no-first-run','--autoplay-policy=no-user-gesture-required','--user-data-dir='+str(profile),'about:blank'],stdout=subprocess.DEVNULL,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW);ws=None;checks=[];errors=[]
 try:
  tab=None
  for _ in range(100):
   try:
    port=int((profile/'DevToolsActivePort').read_text().splitlines()[0]);tab=next(x for x in json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list')) if x['type']=='page');break
   except (OSError,ValueError,StopIteration):time.sleep(.1)
  if tab is None:raise RuntimeError('Chrome startup failed; inspect saved chrome.log')
  ws=websocket.create_connection(tab['webSocketDebuggerUrl'],origin='http://127.0.0.1',timeout=20);seq=0
  def call(method,params=None):
   nonlocal seq
   seq+=1;ws.send(json.dumps(dict(id=seq,method=method,params=params or {})))
   while True:
    z=json.loads(ws.recv())
    if z.get('method')=='Runtime.exceptionThrown':errors.append(z)
    if z.get('id')==seq:assert not z.get('error'),z;return z.get('result',{})
  def js(s):
   z=call('Runtime.evaluate',dict(expression=s,returnByValue=True,awaitPromise=True));assert not z.get('exceptionDetails'),z;return z.get('result',{}).get('value')
  def until(s):
   end=time.monotonic()+20
   while time.monotonic()<end:
    if js(s):return
    time.sleep(.08)
   raise AssertionError(s)
  def check(name,b):assert b,name;checks.append(name)
  call('Runtime.enable');call('Page.enable');call('Emulation.setDeviceMetricsOverride',dict(width=1440,height=1200,deviceScaleFactor=1,mobile=False));call('Page.navigate',dict(url=base));until("typeof current!=='undefined'&&current&&$('ref').readyState>=2&&$('query').readyState>=2")
  check('24 saved cases12 public clips',js('VIEW_DATA.length===24&&clips.length===12'))
  check('omission initial actual video decoded',js("current.kind==='omit_final_event'&&$('ref').videoWidth>0&&$('query').videoWidth>0&&$('query').duration<$('ref').duration"))
  check('new missing candidate and baseline differs',js("current.decisions.final_attempt_supervised_42.cards.some(c=>c.status==='missing_candidate')&&JSON.stringify(current.decisions.final_normal_only_42.cards.map(c=>c.status))!==JSON.stringify(current.decisions.final_attempt_supervised_42.cards.map(c=>c.status))"))
  js("$('play').click()");until("playing&&!$('query').paused");a=js("$('query').currentTime");time.sleep(.4);check('query one clock advances reference paused',js("$('query').currentTime")>a and js("$('ref').paused"));js('stop()')
  js("for(const v of [.2,.5,.8]){$('seek').value=v;$('seek').oninput()}");until("!$('ref').seeking&&!$('query').seeking");time.sleep(.1);check('latest scrub source correspondence',js("Math.abs($('ref').currentTime-current.source_indices[Math.min(current.source_indices.length-1,Math.round($('query').currentTime*30))]/30)<.04"))
  js('showReference(2)');until("referenceOnly&&!$('ref').paused");check('absent reference event not fabricated in query',js("$('query').paused"));js('stop()');check('reference event handler cleaned',js('referenceHandler===null'))
  js("$('seek').value=.8;$('seek').oninput()");until("!$('ref').seeking&&!$('query').seeking");time.sleep(.15)
  (U/'VIEW_OMISSION.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=True))['data']))
  js("$('kind').value='mask_final_event';loadCase()");until("$('ref').readyState>=2&&$('query').readyState>=2");check('mask unchanged RGB and no false absence',js("current.video===current.reference&&current.decisions.final_attempt_supervised_42.cards.every(c=>c.status!=='missing_candidate')&&current.decisions.final_attempt_supervised_42.cards.some(c=>c.display_status==='uncertain')"))
  js("$('seek').value=.8;$('seek').oninput()");until("!$('ref').seeking&&!$('query').seeking");time.sleep(.15)
  (U/'VIEW_MASK.png').write_bytes(base64.b64decode(call('Page.captureScreenshot',dict(format='png',captureBeyondViewport=True))['data']))
  for cid in js('clips'):
   js("$('clip').value="+json.dumps(cid)+";loadKinds();$('kind').value='full_normal';loadCase()");until("$('ref').readyState>=2&&$('query').readyState>=2")
   check('normal decoded '+cid,js("$('query').videoWidth>0&&$('cards').textContent.indexOf('undefined')<0&&current.decisions.final_attempt_supervised_42.cards.every(c=>c.display_status!=='difference')"))
  check('phase verdict separate length',js("$('phaseBar').children.length===current.phase.length&&$('verdictBar').children.length===current.phase.length"))
  check('no overflow desktop',js('document.documentElement.scrollWidth<=innerWidth'))
  call('Emulation.setDeviceMetricsOverride',dict(width=700,height=1000,deviceScaleFactor=1,mobile=False));time.sleep(.1);check('no overflow700',js('document.documentElement.scrollWidth<=innerWidth'))
  check('no camera endpoint',not any(x.startswith(('/api/','/preview')) for x in requests));check('noJSexceptions',not errors)
  (U/'BROWSER_CHECK.json').write_text(json.dumps(dict(status='PASS',checks=checks,errors=errors,actual_public_video=True,camera=False,models_run=False,temporary_server_closed=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('PASS',len(checks))
 except BaseException:
  (U/f'BROWSER_ERRORS_{time.time_ns()}.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
  (U/f'BROWSER_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
 finally:
  if ws:ws.close()
  if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
  log.close();server.shutdown();server.server_close()
if __name__=='__main__':main()
