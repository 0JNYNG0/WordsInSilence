from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,secrets,socket,re,traceback,os
from backend import CaptureApp,U
from prepare import sha
def main():
    plan=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'))
    for p,h in plan['protected'].items():assert sha(p)==h,p
    app=CaptureApp();token=secrets.token_urlsafe(32);rows={r['clip_id']:r for r in plan['rows']}
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def setup(self):
            super().setup();self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        def log_message(self,*args):pass
        def send(self,raw,ctype='application/json',code=200,extra=None):
            self.send_response(code);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store')
            for k,v in (extra or {}).items():self.send_header(k,v)
            self.end_headers();self.wfile.write(raw)
        def json(self,o,code=200):self.send(json.dumps(o,ensure_ascii=False,allow_nan=False).encode('utf-8'),code=code)
        def do_GET(self):
            if self.path=='/':return self.send((U/'test.html').read_bytes(),'text/html; charset=utf-8')
            if self.path=='/capture_client.js':return self.send((U/'capture_client.js').read_bytes(),'text/javascript; charset=utf-8')
            if self.path=='/state':return self.json(dict(token=token,clips=[dict(id=k,label=v['label']) for k,v in rows.items()],ready=True))
            if self.path.startswith('/source/') and self.path[8:] in rows:
                p=Path(rows[self.path[8:]]['source']);size=p.stat().st_size;start=0;end=size-1;code=200;headers={'Accept-Ranges':'bytes'}
                rg=self.headers.get('Range')
                if rg:
                    m=re.fullmatch(r'bytes=(\d+)-(\d*)',rg)
                    if not m:return self.json({'error':'Range'},416)
                    start=int(m[1]);end=min(int(m[2]) if m[2] else end,end);code=206
                    if start>end:return self.json({'error':'Range'},416)
                    headers['Content-Range']=f'bytes {start}-{end}/{size}'
                with p.open('rb') as f:f.seek(start);data=f.read(end-start+1)
                return self.send(data,'video/mp4',code,headers)
            return self.json({'error':'not found'},404)
        def do_POST(self):
            host=f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host')!=host or self.headers.get('Origin') not in [None,'http://'+host] or self.headers.get('X-Session-Token')!=token:
                self.close_connection=True;return self.json({'error':'Local authorization required'},403)
            try:
                if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length',[]))!=1:raise ValueError('Length')
                n=int(self.headers['Content-Length']);limit=64*1024*1024 if self.path=='/video' else 4096*2160*4
                if not 0<n<=limit:raise ValueError('Body size')
                self.connection.settimeout(15);body=self.rfile.read(n)
                if len(body)!=n:raise ValueError('Incomplete body')
                if self.path=='/start':
                    j=json.loads(body);result=app.start(int(j['width']),int(j['height']))
                elif self.path=='/frame':result=app.frame(self.headers['X-Capture-ID'],int(self.headers['X-Frame-ID']),int(self.headers['X-Timestamp-US']),body)
                elif self.path=='/finish':
                    j=json.loads(body);result=app.finish(j['session_id'],int(j['expected_frames']),bool(j.get('cancel',False)))
                elif self.path=='/video':result=app.save_video(self.headers['X-Capture-ID'],body)
                elif self.path=='/trace':
                    sid=self.headers['X-Capture-ID'];j=json.loads(body)
                    if sid not in app.completed or j['id']!=sid:raise ValueError('Trace session')
                    with (U/'runs'/sid/'BROWSER_TRACE.json').open('x',encoding='utf-8') as f:json.dump(j,f,ensure_ascii=False,indent=2)
                    result=dict(saved=True)
                else:return self.json({'error':'not found'},404)
                return self.json(result)
            except (ValueError,KeyError,TypeError) as e:return self.json({'error':str(e)},400)
            except Exception:
                with (U/'SERVER_ERRORS.log').open('a',encoding='utf-8') as f:f.write(traceback.format_exc())
                return self.json({'error':'Capture operation failed'},500)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
    (U/'SERVER.json').write_text(json.dumps(dict(pid=os.getpid(),run_id=os.environ.get('WIS_BROWSER_RUN_ID'),url=f'http://127.0.0.1:{server.server_port}',ready=True)),encoding='utf-8')
    print('READY',server.server_port,flush=True)
    try:server.serve_forever()
    finally:server.server_close();app.close()
if __name__=='__main__':main()
