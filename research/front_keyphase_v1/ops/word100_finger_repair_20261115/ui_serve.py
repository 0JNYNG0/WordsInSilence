from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parent/'ui_check'
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*a,**k):super().__init__(*a,directory=str(ROOT),**k)
 def log_message(self,*a):pass
 def do_GET(self):
  p=Path(self.translate_path(self.path)).resolve()
  if p.suffix!='.mp4' or not p.is_file()or not p.is_relative_to(ROOT):return super().do_GET()
  size=p.stat().st_size;a=0;b=size-1;m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers.get('Range',''))
  if m:a=int(m[1]);b=min(b,int(m[2]))if m[2]else b
  self.send_response(206 if m else 200);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(b-a+1))
  if m:self.send_header('Content-Range',f'bytes {a}-{b}/{size}')
  self.end_headers()
  with p.open('rb')as f:f.seek(a);self.wfile.write(f.read(b-a+1))
ThreadingHTTPServer(('127.0.0.1',8831),Handler).serve_forever()
