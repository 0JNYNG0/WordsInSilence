"""Loopback raw-frame bridge, mountable by the existing web app.

No camera access, training, model downloads, or educational verdicts.
Browser ImageData RGBA preserves pixels; this avoids JPEG re-encoding here.
"""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import threading,secrets,json,time,uuid
import numpy as np
from stream_runtime import StreamSession,reuse_observations,U

class Application:
    def __init__(self,engine,output_root):
        self.engine=engine;self.root=Path(output_root);self.root.mkdir(parents=True,exist_ok=True)
        self.token=secrets.token_urlsafe(32);self.lock=threading.Lock();self.session=None;self.sid=None
    def start(self,width,height):
        with self.lock:
            if self.session is not None:raise RuntimeError('A capture is active')
            self.session=StreamSession(self.engine,width,height);self.sid=uuid.uuid4().hex
            return dict(session_id=self.sid,width=width,height=height,pixel_format='RGBA',feedback_model_ready=False)
    def frame(self,sid,fid,ts,body):
        with self.lock:
            s=self.session
            if s is None or sid!=self.sid:raise RuntimeError('Wrong capture session')
            if len(body)!=s.width*s.height*4:raise ValueError('Raw RGBA payload size mismatch')
            rgba=np.frombuffer(body,np.uint8).reshape(s.height,s.width,4)
            bgr=np.ascontiguousarray(rgba[:,:,:3][:,:,::-1])
            return s.submit(fid,ts,bgr)
    def finish(self,sid,expected_frames):
        with self.lock:
            if self.session is None or sid!=self.sid:raise RuntimeError('Wrong capture session')
            s=self.session;p=self.root/sid;t=time.perf_counter()
            try:
                metadata=s.finish(p)
                before=self.engine.calls;features=reuse_observations(p)
                np.savez_compressed(p/'analysis_input.npz',**{k:v for k,v in features.items() if isinstance(v,np.ndarray)})
                result=dict(session_id=sid,status=metadata['status'],received=metadata['received'],completed=metadata['completed'],
                    dropped=metadata['dropped'],expected_frames=expected_frames,
                    input_complete=metadata['status']=='COMPLETE' and expected_frames==metadata['received'],
                    reuse_detector_calls=self.engine.calls-before,finish_reuse_ms=(time.perf_counter()-t)*1000,
                    artifact=str(p),feedback_model_ready=False,meaning='관절 관측 입력 준비 결과. 수어 정오/교정 결과 아님.')
                with (p/'HTTP_RESULT.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
                return result
            finally:self.session=None;self.sid=None

def make_server(app,port=0):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def reply(self,value,status=200):
            raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(raw)
        def do_POST(self):
            host=f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host')!=host or self.headers.get('Origin') not in [None,'http://'+host] or self.headers.get('X-Session-Token')!=app.token:
                self.close_connection=True;return self.reply(dict(error='Local authenticated request required'),403)
            if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length',[]))!=1:
                self.close_connection=True;return self.reply(dict(error='Invalid length'),400)
            try:
                n=int(self.headers['Content-Length']);assert 0<n<=4096*2160*4
            except Exception:
                self.close_connection=True;return self.reply(dict(error='Payload too large or invalid'),413)
            try:
                self.connection.settimeout(5);body=self.rfile.read(n)
                if len(body)!=n:raise ValueError('Truncated payload')
                if self.path=='/start':
                    j=json.loads(body);result=app.start(int(j['width']),int(j['height']))
                elif self.path=='/frame':
                    result=app.frame(self.headers['X-Capture-ID'],int(self.headers['X-Frame-ID']),int(self.headers['X-Timestamp-US']),body)
                elif self.path=='/finish':
                    j=json.loads(body);result=app.finish(j['session_id'],int(j['expected_frames']))
                else:return self.reply(dict(error='Not found'),404)
                self.reply(result,200 if result.get('accepted',True) else 429)
            except (ValueError,KeyError,TypeError):self.reply(dict(error='Invalid capture input'),400)
            except RuntimeError:self.reply(dict(error='Capture state conflict'),409)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    return server
