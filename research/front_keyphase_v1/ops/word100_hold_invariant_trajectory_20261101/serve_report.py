"""Read-only localhost report server; explicit result/video allowlists only."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, unquote
import argparse, json, gzip, re, os, time

U = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(); p.add_argument('--port', type=int, default=8800)
    args = p.parse_args()
    tasks = json.loads((U/'TASKS.json').read_text(encoding='utf8'))
    manifest = json.loads((U/'RESULT_MANIFEST.json').read_text(encoding='utf8'))
    results = {r['id']: Path(r['path']) for r in manifest['results']}
    videos = {a: Path(r['source']) for a, r in tasks['rows'].items()}
    static = {'/': ('RESULT_KO.html', 'text/html; charset=utf-8'),
              '/RESULT_KO.html': ('RESULT_KO.html', 'text/html; charset=utf-8'),
              '/viewer.js': ('viewer.js', 'text/javascript; charset=utf-8'),
              '/viewer.css': ('viewer.css', 'text/css; charset=utf-8'),
              '/VIEW_DATA.json': ('VIEW_DATA.json', 'application/json; charset=utf-8')}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if args and str(args[1] if len(args)>1 else '') not in ('200', '206'):
                super().log_message(fmt, *args)

        def do_HEAD(self): self.respond(False)
        def do_GET(self): self.respond(True)

        def respond(self, body):
            name = unquote(urlsplit(self.path).path)
            if name in static:
                f, mime = static[name]; data = (U/f).read_bytes()
                return self.bytes(data, mime, body)
            if name.startswith('/data/') and name[6:] in results:
                return self.bytes(gzip.decompress(results[name[6:]].read_bytes()),
                                  'application/json; charset=utf-8', body)
            if name.startswith('/video/') and name[7:] in videos:
                return self.video(videos[name[7:]], body)
            self.send_error(404)

        def bytes(self, data, mime, body):
            self.send_response(200); self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-cache'); self.end_headers()
            if body:
                try: self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError): pass

        def video(self, path, body):
            total = path.stat().st_size; start, end = 0, total-1
            raw = self.headers.get('Range'); partial = bool(raw)
            if raw:
                m = re.fullmatch(r'bytes=(\d*)-(\d*)', raw.strip())
                if not m or not any(m.groups()):
                    self.send_error(416); return
                a,b = m.groups()
                if not a:
                    if int(b)==0: self.send_error(416); return
                    start = max(0, total-int(b))
                else:
                    start=int(a); end=min(total-1,int(b)) if b else total-1
                if start>=total or start>end:
                    self.send_response(416); self.send_header('Content-Range', f'bytes */{total}')
                    self.end_headers(); return
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type','video/mp4'); self.send_header('Accept-Ranges','bytes')
            self.send_header('Content-Length',str(end-start+1))
            if partial: self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
            self.end_headers()
            if body:
                try:
                    with path.open('rb') as f:
                        f.seek(start); remaining=end-start+1
                        while remaining:
                            data=f.read(min(256*1024,remaining))
                            if not data: break
                            self.wfile.write(data); remaining-=len(data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError): pass

    server=ThreadingHTTPServer(('127.0.0.1',args.port), Handler)
    (U/'REPORT_SERVER.json').write_text(json.dumps(dict(pid=os.getpid(),port=args.port,
        url=f'http://127.0.0.1:{args.port}/',started=time.time(),read_only=True,
        video_allowlist=list(videos),computation_running=False),ensure_ascii=False,indent=2),encoding='utf8')
    print(f'http://127.0.0.1:{args.port}/',flush=True)
    server.serve_forever()

if __name__=='__main__': main()
