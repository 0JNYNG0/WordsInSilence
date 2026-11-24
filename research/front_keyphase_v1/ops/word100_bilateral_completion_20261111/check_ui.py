from bilateral_core import *
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit
import threading,mimetypes,re

class Fixture(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_POST(self):self.send_error(403,'Camera and recording prohibited in public replay check')
 def do_GET(self):
  route=urlsplit(self.path).path
  if route=='/api/meta':
   f=read(D/'FROZEN.json');obj=dict(token='fixture',words=[dict(id=w,label=f['labels'][w]) for w in f['words']],examples=read(D/'EXAMPLES.json'))
  elif route=='/api/status':obj=dict(state='ready',has_result=False)
  elif route.startswith('/ref/'):
   obj=copy.deepcopy(read(D/'REFERENCES.json')[route.split('/')[-1]]);obj['video']='/media/'+route.split('/')[-1]+'.mp4'
  else:obj=None
  if obj is not None:
   data=json.dumps(obj,ensure_ascii=False).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
  p=U/'web'/dict([('/', 'index.html'),('/app.js','app.js'),('/style.css','style.css')]).get(route,'missing')
  if route.startswith(('/media/','/examples/')):p=(D/route.lstrip('/')).resolve()
  if not p.is_file() or not (p.is_relative_to(D) or p.is_relative_to(U/'web')):self.send_error(404);return
  size=p.stat().st_size;start=0;end=size-1;m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers.get('Range',''))
  if m:start=int(m[1]);end=min(end,int(m[2])) if m[2] else end
  self.send_response(206 if m else 200);self.send_header('Content-Type',mimetypes.guess_type(str(p))[0] or 'application/octet-stream');self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
  if m:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  try:
   with p.open('rb') as f:f.seek(start);self.wfile.write(f.read(end-start+1))
  except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass

def main():
 server=ThreadingHTTPServer(('127.0.0.1',0),Fixture);threading.Thread(target=server.serve_forever,daemon=True).start()
 src=(D/'check_browser.py').read_text('utf-8').replace('from task import *','from bilateral_core import *').replace("BASE='http://127.0.0.1:8814'",f"BASE='http://127.0.0.1:{server.server_port}'")
 extra='''
  check('two separate timeline lanes',js("$('stageBands').children.length>0 && $('bands').children.length>0 && $('stageBands').getBoundingClientRect().top < $('bands').getBoundingClientRect().top"))
  check('completion distinct from shape',js("$('components').textContent.includes('행동 전체 완료 · 미확인')"))
  js("window.__unchanged=JSON.stringify(result);$('handZoom').open=true;showAt(result.cards[0].query_seconds[0])");until("!$('ref').seeking&&!$('query').seeking")
  check('display leaves prediction data intact',js('JSON.stringify(result)===__unchanged'))
  shot('UI_COMPONENTS.png')
  for idx in [2,3,4,5]:
   js(f"(async()=>{{await loadResult(await get(meta.examples[{idx}].url));return true}})()")
   until("$('query').readyState>=2&&$('ref').readyState>=2")
   js("window.__unchanged=JSON.stringify(result);focusCard(0)");until("!$('ref').seeking&&!$('query').seeking")
   check(f'public result {idx} preserved',js('JSON.stringify(result)===__unchanged'))
   check(f'public result {idx} bounded tracks',js("[...document.querySelectorAll('.trackpart')].every(e=>parseFloat(e.style.left)>=0&&parseFloat(e.style.left)<=100&&parseFloat(e.style.width)<=100)"))
   if idx==2:
    js("showAt(result.timestamp_us[result.cards[0].motion_evidence.span[1]]/1e6)");until("!$('ref').seeking&&!$('query').seeking")
    check('observed wrist arrow labeled',js("$('arrowNote').textContent.includes('관측된 손목 이동')"))
    check('no hypothetical correction path',js("$('arrowNote').textContent.includes('새로 그린 것이 아님')"))
   shot(f'UI_PUBLIC_{idx}.png')
 '''
 src=src.replace("js(\"document.querySelectorAll('#examples button')[1].click()\")",extra+"\n  js(\"(async()=>{await loadResult(await get(meta.examples[1].url));return true})()\")")
 generated=U/'browser_check_generated.py';generated.write_text(src,'utf-8')
 try:exec(compile(src,str(generated),'exec'),dict(__name__='__main__'))
 finally:server.shutdown();server.server_close()
if __name__=='__main__':main()
