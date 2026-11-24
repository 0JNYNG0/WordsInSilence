from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
import json,mimetypes,os
U=Path(__file__).resolve().parent;PORT=8798
def main():
 vi=json.loads((U/'VISUAL_INDEX.json').read_text('utf8'));allowed={'/':U/'index.html','/index.html':U/'index.html','/RESULT_KO.html':U/'RESULT_KO.html'}
 for s in vi['samples']:
  for path in [s['sheet']]+[f['file'] for f in s['frames']]:
   p=Path(path).resolve();allowed['/'+p.relative_to(U).as_posix()]=p
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_HEAD(self):self.serve(False)
  def do_GET(self):self.serve(True)
  def serve(self,body):
   path=urlparse(self.path).path
   if path=='/health':b=json.dumps(dict(unit=U.name,pid=os.getpid(),read_only=True,inference=False,camera=False)).encode();mime='application/json'
   else:
    p=allowed.get(path)
    if p is None or not p.is_file():self.send_error(404);return
    b=p.read_bytes();mime=mimetypes.guess_type(p.name)[0] or 'application/octet-stream'
   self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.end_headers()
   if body:
    try:self.wfile.write(b)
    except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
 server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler);(U/'SERVER_STATUS.json').write_text(json.dumps(dict(pid=os.getpid(),port=PORT,url=f'http://127.0.0.1:{PORT}/')),encoding='utf8');print(f'READY http://127.0.0.1:{PORT}/',flush=True);server.serve_forever()
if __name__=='__main__':main()
