from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse
import json,hashlib,mimetypes,re,os
U=Path(__file__).resolve().parent;MEDIA=U.parent/'word100_core_stage_evidence_20261031/media';PORT=8795
def main():
 rows=json.loads((U/'AUTHORIZATION.json').read_text(encoding='utf8'))['rows'];allowed={r['alias']:MEDIA/(r['alias']+'.webm') for r in rows}
 for a,p in allowed.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads((MEDIA/(a+'.json')).read_text(encoding='utf8'))['output_sha256']
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_HEAD(self):self.serve(False)
  def do_GET(self):self.serve(True)
  def serve(self,body):
   path=urlparse(self.path).path
   if path=='/health':
    b=json.dumps(dict(unit=U.name,pid=os.getpid(),port=PORT,read_only=True,camera=False,inference=False)).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers()
    if body:self.wfile.write(b)
    return
   p=({'/':U/'index.html','/index.html':U/'index.html','/data.json':U/'REPORT_DATA.json','/REPORT_DATA.json':U/'REPORT_DATA.json'}).get(path)
   if path.startswith('/media/'):p=allowed.get(path[7:])
   if p is None or not p.is_file():self.send_error(404);return
   start=0;end=p.stat().st_size-1;total=end+1;part=False
   if self.headers.get('Range'):
    m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers['Range'])
    if not m:self.send_error(416);return
    start=int(m[1]);end=min(end,int(m[2]) if m[2] else end);part=True
    if start>end:self.send_error(416);return
   self.send_response(206 if part else 200);self.send_header('Content-Type',mimetypes.guess_type(p.name)[0] or 'application/octet-stream');self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store')
   if part:self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
   self.end_headers()
   if body:
    try:
     with p.open('rb') as f:
      f.seek(start);remaining=end-start+1
      while remaining:
       b=f.read(min(262144,remaining));self.wfile.write(b);remaining-=len(b)
    except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
 server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler);(U/'SERVER_STATUS.json').write_text(json.dumps(dict(pid=os.getpid(),port=PORT,url=f'http://127.0.0.1:{PORT}/')),encoding='utf8');print(f'READY http://127.0.0.1:{PORT}/',flush=True);server.serve_forever()
if __name__=='__main__':main()
