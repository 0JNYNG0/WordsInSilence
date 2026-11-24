"""Loopback-only research report; explicit public-video allowlist, no capture."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import json,time,urllib.parse
U=Path(__file__).resolve().parent
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(U),**kw)
 def do_GET(self):
  path=urllib.parse.urlsplit(self.path).path
  if path.startswith('/video/'):
   alias=path.split('/')[-1];rows=json.loads((U/'LEARNING_INPUTS.json').read_text(encoding='utf8'))['rows'];r=next((r for r in rows if r['alias']==alias),None)
   if not r:self.send_error(404);return
   p=Path(r['source']);n=p.stat().st_size;start,end=0,n-1
   ran=self.headers.get('Range')
   if ran and ran.startswith('bytes='):
    ab=ran[6:].split('-');start=int(ab[0] or 0);end=min(int(ab[1]) if ab[1] else n-1,n-1)
   if not 0<=start<=end<n:self.send_error(416);return
   self.send_response(206 if ran else 200);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
   if ran:self.send_header('Content-Range',f'bytes {start}-{end}/{n}')
   self.end_headers()
   try:
    with p.open('rb') as f:
     f.seek(start);left=end-start+1
     while left:
      b=f.read(min(left,262144));self.wfile.write(b);left-=len(b)
   except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
   return
  if '..' in urllib.parse.unquote(path).split('/') or not Path(self.translate_path(path)).resolve().is_relative_to(U.resolve()):self.send_error(403);return
  super().do_GET()
 def do_POST(self):
  if self.path!='/ui-measurement':self.send_error(404);return
  n=int(self.headers.get('Content-Length',0))
  if n>100000:self.send_error(413);return
  d=json.loads(self.rfile.read(n));allowed={'payload_fetch_and_two_animation_frames_ms','ui_queries','user_agent','payload_bytes'};d={k:v for k,v in d.items() if k in allowed};d['recorded_epoch']=time.time()
  with (U/'UI_MEASUREMENTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(d)+'\n')
  self.send_response(204);self.end_headers()
 def log_message(self,*args):pass
if __name__=='__main__':
 print('Research viewer http://127.0.0.1:8769/RESULT_KO.html',flush=True);ThreadingHTTPServer(('127.0.0.1',8769),Handler).serve_forever()
