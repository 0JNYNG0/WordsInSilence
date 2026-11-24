"""Isolated local UI adapter; reuses frozen 8814 capture/model code read-only."""
from pathlib import Path
import sys, json, threading, secrets, time, os, mimetypes, re, copy
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit, unquote, parse_qs
U=Path(__file__).resolve().parent;O=U.parent;OLD=O/'word100_action_feedback_demo_20261108';PREVIEW=O/'web_tutorial_stage_preview_20261113'/'site'
sys.dont_write_bytecode=True
sys.path.insert(0,str(OLD))
from capture import CaptureService
from contract import QuestProgress, SUPPORTED, TARGETS
sys.path.insert(0,str(O/"word100_actionable_feedback_20261113"))
from coaching import attach
PORT=8821;TOKEN=secrets.token_urlsafe(32)
SERVICE=CaptureService();QUEST=QuestProgress();LOCK=threading.RLock();ATTEMPT=None;RESULT=None;OWNER=None
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))

def finalize_result():
    global RESULT
    if ATTEMPT and SERVICE.state=='done' and SERVICE.result is not None and RESULT is None:
        r=attach(SERVICE.result)
        r.update(attempt_id=ATTEMPT['id'],mode=ATTEMPT['mode'],video='/recorded.mp4?attempt='+ATTEMPT['id'])
        outcome=QUEST.consume(ATTEMPT,r)
        RESULT=dict(result=r,quest=QUEST.state(),outcome=outcome)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,x,status=200):
        b=json.dumps(x,ensure_ascii=False,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def file(self,p):
        if not p.is_file():self.send_error(404);return
        size=p.stat().st_size;start=0;end=size-1;m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers.get('Range',''))
        if m:start=int(m[1]);end=min(end,int(m[2])) if m[2] else end
        if start>end:self.send_error(416);return
        self.send_response(206 if m else 200);self.send_header('Content-Type',mimetypes.guess_type(str(p))[0] or 'application/octet-stream');self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(end-start+1))
        if m:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        with p.open('rb') as f:
            f.seek(start);remain=end-start+1
            while remain:
                b=f.read(min(65536,remain))
                if not b:break
                self.wfile.write(b);remain-=len(b)
    def valid_host(self):return self.headers.get('Host')==f'127.0.0.1:{PORT}'
    def authorized(self):return self.valid_host() and self.headers.get('X-Session-Token')==TOKEN
    def do_GET(self):
        if not self.valid_host():self.reply({'error':'Local host only'},403);return
        route=unquote(urlsplit(self.path).path);query=parse_qs(urlsplit(self.path).query)
        try:
            if route=='/api/meta':
                self.reply(dict(token=TOKEN,words=[dict(id=w,label=l) for w,l in SUPPORTED.items()],planned_words=TARGETS,cards=read(OLD/'CARDS.json')['words'],quest=QUEST.state(),examples=read(OLD/'EXAMPLES.json'),model='기존 RTMW TCN + 관절 전용 시간합성곱·GRU 연구 모델',privacy='촬영은 이 PC에서 분석하며 학습·튜닝에 사용하지 않습니다.'))
                return
            if route.startswith('/api/'):
                if not self.authorized():self.reply({'error':'Local session token required'},403);return
                with LOCK:
                    finalize_result()
                    if route=='/api/status':self.reply(dict(**SERVICE.status(),attempt_id=ATTEMPT['id'] if ATTEMPT else None,quest=QUEST.state()));return
                    if route=='/api/result':
                        if not RESULT or query.get('attempt',[''])[0]!=ATTEMPT['id']:self.reply({'error':'결과가 없거나 이전 촬영입니다.'},409);return
                        self.reply(RESULT);return
                self.send_error(404);return
            if route=='/preview.mjpeg':
                if query.get('token',[''])[0]!=TOKEN:self.reply({'error':'Token required'},403);return
                self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.send_header('Cache-Control','no-store');self.end_headers();last=-1
                while SERVICE.state in ['opening','preview','countdown','recording']:
                    with SERVICE.preview_condition:
                        SERVICE.preview_condition.wait_for(lambda:SERVICE.jpeg_seq!=last or SERVICE.state not in ['opening','preview','countdown','recording'],timeout=1)
                        if SERVICE.jpeg is None or SERVICE.jpeg_seq==last:continue
                        b=SERVICE.jpeg;last=SERVICE.jpeg_seq
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(b)).encode()+b'\r\n\r\n'+b+b'\r\n');self.wfile.flush()
                return
            if route=='/recorded.mp4':
                if query.get('token',[''])[0]!=TOKEN or not ATTEMPT or query.get('attempt',[''])[0]!=ATTEMPT['id'] or not SERVICE.directory:self.send_error(404);return
                self.file(SERVICE.directory/'recorded.mp4');return
            if route.startswith('/ref/'):
                word=route.split('/')[-1]
                if word not in SUPPORTED:self.send_error(404);return
                r=read(OLD/'REFERENCES.json')[word];r['video']='/media/'+word+'.mp4';self.reply(r);return
            if route.startswith('/public-replay/'):
                word=Path(route).stem
                if word not in SUPPORTED:self.send_error(404);return
                self.file(OLD/'parallel'/word/'video.mp4');return
            if route.startswith(('/media/','/examples/')):
                p=(OLD/route.lstrip('/')).resolve()
                if p.is_relative_to(OLD.resolve()) and p.suffix in ['.mp4','.json']:
                    if p.suffix=='.json' and p.name=='result.json':self.reply(attach(read(p)))
                    else:self.file(p)
                    return
                self.send_error(404);return
            rel='index.html' if route=='/' else route.lstrip('/')
            for base in [U/'site',PREVIEW]:
                p=(base/rel).resolve()
                if p.is_relative_to(base.resolve()) and p.is_file() and p.suffix in ['.html','.js','.css','.glb','.ttf','.woff2','.png','.jpg']:
                    self.file(p);return
            self.send_error(404)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
    def do_POST(self):
        global ATTEMPT,RESULT,OWNER
        origin=self.headers.get('Origin')
        if not self.authorized() or (origin and origin!=f'http://127.0.0.1:{PORT}'):
            self.reply({'error':'Local session token required'},403);return
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<=n<=4096:raise ValueError('Request too large')
            b=json.loads(self.rfile.read(n) or '{}');route=urlsplit(self.path).path;client=b.get('client')
            if not isinstance(client,str) or not re.fullmatch('[a-zA-Z0-9-]{16,80}',client):raise ValueError('Invalid client')
            with LOCK:
                if OWNER and OWNER!=client and SERVICE.state in ['opening','preview','countdown','recording','analysing','finishing']:raise ValueError('다른 창에서 카메라를 사용하고 있어요.')
                if route=='/api/open':
                    SERVICE.open_camera();OWNER=client;ATTEMPT=None;RESULT=None
                elif route=='/api/start':
                    if OWNER!=client:raise ValueError('먼저 이 창에서 카메라를 켜 주세요.')
                    if ATTEMPT or SERVICE.record_requested:raise ValueError('이미 촬영을 시작했어요.')
                    mode=b.get('mode');target=b.get('word')
                    if mode not in ['practice','quest'] or target not in SUPPORTED:raise ValueError('아직 연결되지 않은 단어예요.')
                    if mode=='quest' and (QUEST.index>=len(QUEST.steps) or target!=QUEST.steps[QUEST.index]):raise ValueError('현재 퀘스트 단어가 아니에요.')
                    SERVICE.start(target);ATTEMPT=dict(id=secrets.token_hex(12),mode=mode,target=target,generation=QUEST.generation)
                elif route=='/api/finish':
                    if OWNER!=client or SERVICE.state!='recording':raise ValueError('현재 촬영 중이 아니에요.')
                    SERVICE.finish_requested=True
                elif route=='/api/close':
                    if OWNER and OWNER!=client:raise ValueError('다른 창의 촬영이에요.')
                    SERVICE.stop_event.set();ATTEMPT=None;RESULT=None
                elif route=='/api/quest/reset':
                    if SERVICE.state in ['opening','preview','countdown','recording','analysing','finishing']:raise ValueError('촬영을 닫은 뒤 다시 시작해 주세요.')
                    QUEST.reset();ATTEMPT=None;RESULT=None
                elif route=='/api/public-replay':
                    if SERVICE.state not in ['ready','done','error'] or SERVICE.consumer is None:raise ValueError('촬영을 닫고 분석 준비가 끝난 뒤 확인해 주세요.')
                    word=b.get('word')
                    if word not in SUPPORTED:raise ValueError('Unsupported public input')
                    import numpy as np
                    with np.load(OLD/'parallel'/word/'rtmw/observations.npz') as z:
                        result=SERVICE.consumer.raw(z,word)
                    result=attach(result)
                    result.update(source='public_integration_replay',video='/public-replay/'+word+'.mp4',training_use=False)
                    self.reply(dict(result=result,quest=QUEST.state(),outcome=dict(advanced=False,reason='public_replay')));return
                else:self.send_error(404);return
                self.reply(dict(ok=True,attempt_id=ATTEMPT['id'] if ATTEMPT else None,quest=QUEST.state()))
        except (ValueError,RuntimeError,TypeError) as exc:self.reply({'error':str(exc)},400)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass

if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    threading.Thread(target=SERVICE.prepare,daemon=True).start()
    (U/'SERVER.json').write_text(json.dumps(dict(pid=os.getpid(),port=PORT,url=f'http://127.0.0.1:{PORT}/',camera_at_start=False),indent=2),encoding='utf-8')
    print(f'Local http://127.0.0.1:{PORT}/ camera starts only on user command',flush=True)
    server.serve_forever()
