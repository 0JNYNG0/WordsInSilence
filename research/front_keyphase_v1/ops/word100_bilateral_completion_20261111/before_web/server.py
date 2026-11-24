from task import *
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit,unquote
import secrets,threading,mimetypes,re
from capture import CaptureService
PORT=8814;SERVICE=CaptureService();TOKEN=secrets.token_urlsafe(24)
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def send_json(self,x,status=200):
  b=json.dumps(x,ensure_ascii=False,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def file(self,p):
  if not p.is_file():self.send_error(404);return
  size=p.stat().st_size;start=0;end=size-1;header=self.headers.get('Range','');m=re.fullmatch(r'bytes=(\d+)-(\d*)',header)
  if m:start=int(m[1]);end=min(end,int(m[2])) if m[2] else end
  if start>end:self.send_error(416);return
  self.send_response(206 if m else 200);self.send_header('Content-Type',mimetypes.guess_type(str(p))[0] or 'application/octet-stream');self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(end-start+1))
  if m:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  with p.open('rb') as f:
   f.seek(start);remain=end-start+1
   while remain:
    b=f.read(min(65536,remain))
    if not b:break
    self.wfile.write(b);remain-=len(b)
 def do_GET(self):
  try:
   route=unquote(urlsplit(self.path).path)
   if route=='/api/meta':
    f=read(U/'FROZEN.json');self.send_json(dict(token=TOKEN,words=[dict(id=w,label=f['labels'][w]) for w in f['words']],cards=read(U/'CARDS.json')['words'],examples=read(U/'EXAMPLES.json') if (U/'EXAMPLES.json').exists() else [],scope='연구용 관절 비교 · 자동 교정 검증 전'));return
   if route=='/api/status':self.send_json(SERVICE.status());return
   if route=='/api/result':self.send_json(SERVICE.result or {});return
   if route=='/preview.mjpeg':
    self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.send_header('Cache-Control','no-store');self.end_headers();last=-1
    while SERVICE.state in ['opening','preview','countdown','recording']:
     with SERVICE.preview_condition:
      SERVICE.preview_condition.wait_for(lambda:SERVICE.jpeg_seq!=last or SERVICE.state not in ['opening','preview','countdown','recording'],timeout=1)
      if SERVICE.jpeg is None or SERVICE.jpeg_seq==last:continue
      b=SERVICE.jpeg;last=SERVICE.jpeg_seq
     self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(b)).encode()+b'\r\n\r\n'+b+b'\r\n');self.wfile.flush()
    return
   if route=='/preview.jpg':
    b=SERVICE.jpeg
    if b is None:self.send_error(404);return
    self.send_response(200);self.send_header('Content-Type','image/jpeg');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
   if route=='/recorded.mp4' and SERVICE.directory:self.file(SERVICE.directory/'recorded.mp4');return
   if route.startswith('/ref/'):
    w=route.split('/')[-1]
    if w not in read(U/'FROZEN.json')['words']:self.send_error(404);return
    r=read(U/'REFERENCES.json')[w];r['video']='/media/'+w+'.mp4';self.send_json(r);return
   roots={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
   if route in roots:self.file(U/roots[route]);return
   if route.startswith(('/media/','/examples/')):
    p=(U/route.lstrip('/')).resolve()
    if p.is_relative_to(U.resolve()) and p.suffix in ['.mp4','.json']:self.file(p);return
   self.send_error(404)
  except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
 def do_POST(self):
  origin=self.headers.get('Origin');host=self.headers.get('Host','')
  if host!=f'127.0.0.1:{PORT}' or (origin and origin!=f'http://127.0.0.1:{PORT}') or self.headers.get('X-Session-Token')!=TOKEN:self.send_json({'error':'Local session token required'},403);return
  try:
   length=int(self.headers.get('Content-Length','0'))
   if length>4096:raise ValueError('Request too large')
   body=json.loads(self.rfile.read(length) or '{}');route=urlsplit(self.path).path
   if route=='/api/open':SERVICE.open_camera()
   elif route=='/api/start':SERVICE.start(body.get('word'))
   elif route=='/api/finish':SERVICE.finish_requested=True
   elif route=='/api/close':SERVICE.stop_event.set()
   else:self.send_error(404);return
   self.send_json({'ok':True})
  except (ValueError,RuntimeError) as e:self.send_json({'error':str(e)},400)
if __name__=='__main__':
 server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler);threading.Thread(target=SERVICE.prepare,daemon=True).start();write(U/'SERVER.json',dict(pid=os.getpid(),port=PORT,url=f'http://127.0.0.1:{PORT}/',camera_at_start=False));server.serve_forever()
