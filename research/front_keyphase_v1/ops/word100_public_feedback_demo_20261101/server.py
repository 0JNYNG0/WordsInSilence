"""Local read-only public replay server. No uploads, cameras, inference or file browser."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
from pathlib import Path
import json,mimetypes,re,os
U=Path(__file__).resolve().parent;O=U.parent;PORT=8801
ALLOWED={'/':U/'index.html','/index.html':U/'index.html','/app.js':U/'app.js','/data.json':U/'DEMO_DATA.json','/report.txt':U/'RESULT_KO.txt'}
data=json.loads((U/'DEMO_DATA.json').read_text(encoding='utf8'))
ALLOWED.update({'/media/'+a:O/f'word100_core_stage_evidence_20261031/media/{a}.webm' for a in data['clips']})
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):self.serve(True)
 def do_HEAD(self):self.serve(False)
 def do_POST(self):self.send_error(405)
 def serve(self,body):
  path=urlparse(self.path).path
  if path=='/health':
   b=json.dumps(dict(unit=U.name,pid=os.getpid(),port=PORT,read_only=True,camera=False,inference=False)).encode()
   self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers()
   if body:self.wfile.write(b)
   return
  p=ALLOWED.get(path)
  if p is None or not p.is_file():self.send_error(404);return
  total=p.stat().st_size;start=0;end=total-1;partial=False
  if self.headers.get('Range'):
   m=re.fullmatch(r'bytes=(\d*)-(\d*)',self.headers['Range'])
   if not m or not any(m.groups()):self.send_error(416);return
   if m[1]:start=int(m[1]);end=min(end,int(m[2]) if m[2] else end)
   else:start=max(0,total-int(m[2]))
   if start>end or start>=total:self.send_error(416);return
   partial=True
  self.send_response(206 if partial else 200)
  mime=mimetypes.guess_type(str(p))[0] or 'application/octet-stream'
  if mime.startswith('text/') or p.suffix in ['.js','.json']:mime+='; charset=utf-8'
  self.send_header('Content-Type',mime);self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
  if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
  self.end_headers()
  if body:
   try:
    with p.open('rb') as f:
     f.seek(start);remaining=end-start+1
     while remaining:
      b=f.read(min(262144,remaining))
      if not b:break
      self.wfile.write(b);remaining-=len(b)
   except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
def main():
 server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
 (U/'SERVER_STATUS.json').write_text(json.dumps(dict(pid=os.getpid(),port=PORT,url=f'http://127.0.0.1:{PORT}/',camera=False,inference=False)),encoding='utf8')
 print(f'READY http://127.0.0.1:{PORT}/',flush=True);server.serve_forever()
if __name__=='__main__':main()
