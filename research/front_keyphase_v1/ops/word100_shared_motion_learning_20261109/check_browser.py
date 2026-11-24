"""Camera-free isolated replay of saved public results; no model imports or inference."""
from pathlib import Path
import json,time,threading,subprocess,urllib.request,websocket,uuid,mimetypes,re,traceback
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit
U=Path(__file__).resolve().parent;D=U.parent/'word100_action_feedback_demo_20261108'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
requests=[]
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_POST(self):self.send_error(403)
 def do_GET(self):
  route=urlsplit(self.path).path;requests.append(route);obj=None;p=None
  if route=='/api/status':obj=dict(state='ready',has_result=False)
  elif route=='/api/meta':
   f=read(D/'FROZEN.json');obj=dict(token='no-camera-test',words=[dict(id=w,label=f['labels'][w]) for w in f['words']],examples=read(D/'EXAMPLES.json'))
  elif route.startswith('/ref/'):
   word=route.split('/')[-1];obj=read(D/'REFERENCES.json')[word];obj['video']='/media/'+word+'.mp4'
  elif route in ['/','/app.js','/style.css']:p=D/({'/':'index.html','/app.js':'app.js','/style.css':'style.css'}[route])
  elif route.startswith(('/media/','/examples/')):
   p=(D/route.lstrip('/')).resolve()
   if not p.is_relative_to(D.resolve()) or p.suffix not in ['.json','.mp4']:self.send_error(403);return
  else:self.send_error(404);return
  if obj is not None:data=json.dumps(obj,ensure_ascii=False).encode();mime='application/json'
  elif p and p.exists():data=p.read_bytes();mime=mimetypes.guess_type(str(p))[0] or 'application/octet-stream'
  else:self.send_error(404);return
  total=len(data);a=0;b=total-1;m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers.get('Range',''))
  if m:a=int(m[1]);b=min(b,int(m[2])) if m[2] else b
  self.send_response(206 if m else 200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(b-a+1));self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store')
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
  if tab is None:raise RuntimeError('Headless Chrome did not start; inspect saved chrome.log. No camera or models were loaded.')
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
  call('Runtime.enable');call('Page.enable');call('Page.navigate',dict(url=base));until("typeof meta!=='undefined'&&meta&&$('ref').readyState>=2")
  js("document.querySelector('#examples button').click()");until("result&&$('ref').readyState>=2&&$('query').readyState>=2")
  check('real public videos decoded',js("$('ref').videoWidth>0&&$('query').videoWidth>0"))
  js("$('play').click()");until("playing&&!$('query').paused");check('reference never independently playing',js("$('ref').paused"));a=js("$('query').currentTime");time.sleep(.6);check('query advances',js("$('query').currentTime")>a)
  js("pause();showAt(.3);showAt(.6);showAt(.9)");until("!$('ref').seeking&&!$('query').seeking");time.sleep(.1)
  check('latest scrub aligns',js("Math.abs($('ref').currentTime-reference.timestamp_us[Math.min(mapRef(nearest(result.timestamp_us,view*1e6)),reference.timestamp_us.length-1)]/1e6)<.05"))
  check('no verdict enabled',js('result.overall_pass===false&&result.correction_validated===false'))
  js('focusCard(0)');until("!$('ref').seeking");js("$('loop').click()");until('playing');check('paired segment has one clock',js("referenceOnly||$('ref').paused"));js('pause()')
  check('no camera endpoints',not any(x.startswith(('/api/open','/api/start','/preview')) for x in requests));check('no JS exceptions',not errors)
  (U/'REPLAY_BROWSER_CHECK.json').write_text(json.dumps(dict(status='PASS',checks=checks,errors=errors,actual_public_video=True,camera=False,models_run=False,server_port=server.server_port),ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('PASS',len(checks))
 except BaseException:
  (U/f'BROWSER_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
 finally:
  if ws:ws.close()
  if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
  log.close();server.shutdown();server.server_close()
if __name__=='__main__':main()
