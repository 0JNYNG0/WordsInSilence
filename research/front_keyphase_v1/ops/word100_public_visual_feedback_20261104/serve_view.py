"""Loopback saved-public-results viewer. No inference or upload endpoints."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import hashlib
import json
import re
import os
import argparse

U=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*args): pass
    def send_bytes(self,content,mime,status=200):
        self.send_response(status)
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(len(content)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Permissions-Policy','camera=(), microphone=()')
        self.end_headers()
        if self.command!='HEAD': self.wfile.write(content)
    def do_POST(self):
        self.close_connection=True
        self.send_bytes(b'{"error":"read-only saved public viewer"}','application/json',405)
    do_PUT=do_POST
    do_DELETE=do_POST
    def do_HEAD(self): self.do_GET()
    def do_GET(self):
        try:self.get()
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError): pass
    def get(self):
        path=urlsplit(self.path).path
        routes={'/':('index.html','text/html; charset=utf-8'),'/viewer.js':('viewer.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8'),'/api/catalog':('data/catalog.json','application/json; charset=utf-8')}
        if path in routes:
            file,mime=routes[path];return self.send_bytes((U/file).read_bytes(),mime)
        if path=='/api/health':return self.send_bytes(json.dumps(dict(mode='saved-public-display-only',pid=os.getpid(),cases=len(self.server.media),training=False,inference=False,camera=False,goal_paused=True)).encode(),'application/json')
        if path.startswith('/api/case/'):
            key=path.removeprefix('/api/case/')
            if key in self.server.media:return self.send_bytes((U/'data'/f'{key}.json').read_bytes(),'application/json; charset=utf-8')
        if path.startswith('/media/'):
            key=path.removeprefix('/media/')
            if key in self.server.media:
                file=Path(self.server.media[key]['path']);length=file.stat().st_size
                start,end=0,length-1;status=200
                requested=self.headers.get('Range')
                if requested:
                    m=re.fullmatch(r'bytes=(\d+)-(\d*)',requested)
                    if not m:return self.send_bytes(b'Unsupported range','text/plain',416)
                    start=int(m[1]);end=min(int(m[2]) if m[2] else end,end)
                    if not 0<=start<=end<length:return self.send_bytes(b'Out of range','text/plain',416)
                    status=206
                self.send_response(status);self.send_header('Content-Type','video/mp4')
                self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
                self.send_header('Cache-Control','private, max-age=3600')
                if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{length}')
                self.end_headers()
                if self.command!='HEAD':
                    with file.open('rb') as f:
                        f.seek(start);left=end-start+1
                        while left:
                            block=f.read(min(left,65536))
                            if not block:break
                            self.wfile.write(block);left-=len(block)
                return
        self.send_bytes(b'Not found','text/plain',404)

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8811);args=p.parse_args()
    build=read(U/'BUILD.json');media=read(U/'MEDIA.json')
    assert build['new_inference']==build['new_training']==0
    assert sha(U/'MEDIA.json')==build['media_manifest_sha256']
    for path,h in build['output_pins'].items():assert sha(path)==h,path
    for row in media.values():assert sha(row['path'])==row['sha256'],row['path']
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.daemon_threads=True;server.media=media
    (U/'VIEWER_RUNNING.json').write_text(json.dumps(dict(pid=os.getpid(),port=args.port,url=f'http://127.0.0.1:{args.port}/',training=False,inference=False,public_only=True),indent=2),encoding='utf-8')
    print(f'Saved public viewer http://127.0.0.1:{args.port}/ pid={os.getpid()}',flush=True)
    server.serve_forever()

if __name__=='__main__':main()
