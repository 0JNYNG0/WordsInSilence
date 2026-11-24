"""Loopback-only public whitelist. No camera, uploads, arbitrary paths or user files."""
from pipeline import *
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

def make_server(engine):
 class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def reply(self,code,obj):
        b=json.dumps(obj,ensure_ascii=False,allow_nan=False).encode();self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(b)
    def do_GET(self):
        p=urlparse(self.path)
        if p.path=='/':
            b=(U/'index.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
        elif p.path=='/status':
            try:self.reply(200,engine.public(parse_qs(p.query)['key'][0]))
            except (KeyError,ValueError):self.reply(404,{'error':'unknown run'})
        elif p.path.startswith('/video/'):
            alias=p.path.split('/')[-1];row=engine.rows.get(alias) or engine.refrows.get(alias)
            if row is None:self.reply(404,{'error':'not whitelisted'});return
            source=Path(row['source']);size=source.stat().st_size;start=0;end=size-1;partial=False
            if self.headers.get('Range'):
                import re
                m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers['Range'])
                if not m:self.reply(416,{'error':'unsupported range'});return
                start=int(m[1]);end=min(int(m[2]) if m[2] else size-1,size-1);partial=True
                if start>end:self.reply(416,{'error':'invalid range'});return
            self.send_response(206 if partial else 200);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
            if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with source.open('rb') as f:
                    f.seek(start);left=end-start+1
                    while left:
                        b=f.read(min(left,65536));self.wfile.write(b);left-=len(b)
            except (BrokenPipeError,ConnectionResetError):pass
        else:self.reply(404,{'error':'not found'})
    def do_POST(self):
        try:
            # Same-origin only; no permissive CORS and no input payload files.
            origin=self.headers.get('Origin')
            assert origin is None or origin==f'http://127.0.0.1:{self.server.server_port}'
            n=int(self.headers.get('Content-Length','0'));assert 0<n<8192
            body=json.loads(self.rfile.read(n))
            if self.path=='/start':self.reply(200,dict(key=engine.start(body['alias'],body['mode'],body['repeat'])))
            elif self.path=='/ack':self.reply(200,engine.acknowledge(body.pop('key'),body))
            else:self.reply(404,{'error':'not found'})
        except Exception as e:self.reply(400,dict(error=str(e)))
 return ThreadingHTTPServer(('127.0.0.1',0),Handler)
