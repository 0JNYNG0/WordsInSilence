"""Local companion UI. Keeps last recording accessible when preview reopens."""
from pathlib import Path
from collections import OrderedDict
import sys,json,threading,os,re,argparse
D=Path(__file__).resolve().parent;O=D.parent.parent
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'))
import server
from role_runtime import CaptureService
RECORDINGS=OrderedDict()

def remember():
 # Explicit bounded retention of a finished session only; never enumerate user folders.
 if server.ATTEMPT and server.SERVICE.state=='done' and server.SERVICE.directory:
  key=server.ATTEMPT['id'];p=server.SERVICE.directory/'recorded.mp4'
  if p.is_file():
   RECORDINGS[key]=p
   while len(RECORDINGS)>4:RECORDINGS.popitem(last=False)

class Handler(server.Handler):
 def do_GET(self):
  route=server.unquote(server.urlsplit(self.path).path)
  if not self.valid_host():self.reply({'error':'Local host only'},403);return
  if server.PORT==8822 and route.startswith('/qa/'):
   p=(D/route.lstrip('/')).resolve()
   if p.is_relative_to((D/'qa').resolve()) and p.is_file() and p.suffix in ['.json','.jpg']:
    self.file(p);return
  if route.startswith('/session-recording/'):
   query=server.parse_qs(server.urlsplit(self.path).query)
   if query.get('token',[''])[0]!=server.TOKEN:self.reply({'error':'Token required'},403);return
   match=re.fullmatch(r'/session-recording/([a-f0-9]{24})\.mp4',route)
   if not match:self.send_error(404);return
   key=match[1]
   with server.LOCK:
    remember();p=RECORDINGS.get(key)
   if not p:self.send_error(404);return
   self.file(p);return
  rel='index.html' if route=='/' else route.lstrip('/')
  p=(D/'site'/rel).resolve()
  if p.is_relative_to((D/'site').resolve()) and p.is_file() and p.suffix in ['.html','.js','.css']:
   self.file(p);return
  super().do_GET()
 def do_POST(self):
  # Retain before /api/open resets ATTEMPT and starts a new preview directory.
  if self.authorized() and server.urlsplit(self.path).path in ['/api/open','/api/close']:
   with server.LOCK:remember()
  super().do_POST()

if __name__=='__main__':
 args=argparse.ArgumentParser();args.add_argument('--port',type=int,default=8821);a=args.parse_args();server.PORT=a.port
 server.SERVICE=CaptureService();http=server.LocalServer(('127.0.0.1',server.PORT),Handler)
 threading.Thread(target=server.SERVICE.prepare,daemon=True).start()
 (D/f'SERVER_{a.port}.json').write_text(json.dumps(dict(pid=os.getpid(),port=a.port,entrypoint=str(Path(__file__).resolve()),camera_at_start=False)),encoding='utf-8')
 print(f'Local http://127.0.0.1:{server.PORT}/ — camera off until user enters practice/quest',flush=True)
 http.serve_forever()
