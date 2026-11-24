"""Local read-only evidence viewer. Exact public allowlist, no camera/upload/inference."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,unquote
import json,mimetypes,re,hashlib,os,time
U=Path(__file__).resolve().parent
OLD=U.parent/'word100_semantic_episode_pilot_20261031'
PORT=8791
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rows=read(U/'AUTHORIZATION.json')['rows'];media={r['alias']:Path(r['source']) for r in rows}
 for r in rows:assert sha(r['source'])==r['source_sha256']
 documents={'/':U/'index.html','/index.html':U/'index.html','/previous.html':U/'previous_fixed.html','/phase_data.json':U/'PHASE_DATA.json'}
 for p in ['EVALUATION.json','EMPIRICAL_RANGE.json','CONTROLS.json','FINAL_GATE.json','WEB_RESULT.json','review/STORED_RESULT_VERIFICATION.json','review/EMPIRICAL_RESULT_VERIFICATION.json','review/WEB_RESULT_VERIFICATION.json']:
  documents['/'+p]=OLD/p
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_HEAD(self):self.serve(False)
  def do_GET(self):self.serve(True)
  def serve(self,body):
   path=unquote(urlparse(self.path).path)
   if path=='/health':
    b=json.dumps(dict(unit=U.name,pid=os.getpid(),port=PORT,read_only=True,public_aliases=list(media),camera=False,model_inference=False)).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers()
    if body:self.wfile.write(b)
    return
   if path.startswith('/media/'):
    alias=path[len('/media/'):];p=media.get(alias)
    if p is not None and (U/'media'/(alias+'.webm')).is_file() and (U/'media'/(alias+'.json')).is_file():p=U/'media'/(alias+'.webm')
   else:p=documents.get(path)
   if p is None or not p.is_file():self.send_error(404);return
   size=p.stat().st_size;start=0;end=size-1;rng=self.headers.get('Range');partial=False
   if rng:
    match=re.fullmatch(r'bytes=(\d+)-(\d*)',rng)
    if not match:self.send_error(416);return
    start=int(match[1]);end=min(size-1,int(match[2]) if match[2] else size-1);partial=True
    if start>end or start>=size:self.send_error(416);return
   self.send_response(206 if partial else 200);self.send_header('Content-Type',mimetypes.guess_type(p.name)[0] or 'application/octet-stream');self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
   if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
   self.end_headers()
   if body:
    try:
     with p.open('rb') as f:
      f.seek(start);remaining=end-start+1
      while remaining:
       b=f.read(min(262144,remaining))
       if not b:break
       self.wfile.write(b);remaining-=len(b)
    except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass # video seek cancels its old range request
 server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
 (U/'SERVER_STATUS.json').write_text(json.dumps(dict(pid=os.getpid(),port=PORT,url=f'http://127.0.0.1:{PORT}/',started=time.time(),public_clips=len(media),read_only=True),indent=2),encoding='utf8')
 print(f'PUBLIC_REPORT_READY http://127.0.0.1:{PORT}/',flush=True)
 server.serve_forever()
if __name__=='__main__':main()
