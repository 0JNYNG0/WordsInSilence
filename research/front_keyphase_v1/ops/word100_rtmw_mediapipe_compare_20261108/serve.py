from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit,unquote
import json,re,mimetypes
U=Path(__file__).resolve().parent
rows=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'))['rows']
media={'/media/'+r['clip_id']+'.mp4':Path(r['source']) for r in rows}
allowed={'/':'RESULT_KO.html','/RESULT_KO.html':'RESULT_KO.html','/viewer.js':'viewer.js','/viewer_data.json':'viewer_data.json','/VISUAL_INDEX.json':'VISUAL_INDEX.json','/SPEED_SUMMARY.json':'SPEED_SUMMARY.json','/VISUAL_REVIEW.json':'VISUAL_REVIEW.json','/VERIFICATION.json':'VERIFICATION.json','/METRICS.json':'METRICS.json','/RESULT_KO.txt':'RESULT_KO.txt'}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_HEAD(self):self.send_file(False)
 def do_GET(self):self.send_file(True)
 def send_file(self,body):
  url=unquote(urlsplit(self.path).path);p=media.get(url)
  if p is None and url in allowed:p=U/allowed[url]
  if p is None and re.fullmatch(r'/visuals/[A-Za-z0-9_]+\.jpg',url):p=U/url.lstrip('/')
  if p is None or not p.is_file():self.send_error(404);return
  size=p.stat().st_size;start=0;end=size-1;rg=self.headers.get('Range')
  if rg:
   m=re.fullmatch(r'bytes=(\d+)-(\d*)',rg)
   if not m:self.send_error(416);return
   start=int(m[1]);end=min(int(m[2]) if m[2] else end,end)
   if start> end:self.send_error(416);return
  self.send_response(206 if rg else 200);self.send_header('Content-Type',mimetypes.guess_type(str(p))[0] or 'application/octet-stream');self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store')
  if rg:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  if body:
   try:
    with p.open('rb') as f:
     f.seek(start);remaining=end-start+1
     while remaining:
      buf=f.read(min(1024*1024,remaining));self.wfile.write(buf);remaining-=len(buf)
   except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
if __name__=='__main__':
 print('Public comparison only: http://127.0.0.1:8813/',flush=True)
 ThreadingHTTPServer(('127.0.0.1',8813),Handler).serve_forever()
