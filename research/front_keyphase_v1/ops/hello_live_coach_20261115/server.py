"""Local, separate live-coaching experiment. Explicit camera start, RAM only.

Public replay feeds cached PUBLIC RTMW points in native order, not a camera.
Camera inference uses a one-frame mailbox instead of building a delayed queue;
replaced frames and observation age are visible diagnostics, not hidden drops.
No writes of user images, coordinates, or decisions. No training/import of TCN.
"""
from pathlib import Path
import sys, os, json, time, threading, secrets, mimetypes, re, copy
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit, parse_qs
import cv2
import numpy as np
from diagnostic_ledger import DiagnosticLedger

U=Path(__file__).resolve().parent;O=U.parent
sys.dont_write_bytecode=True
PORT=int(os.environ.get('WIS_HELLO_PORT','8822'))
TOKEN=secrets.token_urlsafe(32)
# ops -> front_keyphase_v1 -> runs (fixed, never user-supplied).
REFERENCE=O.parent/'runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4'
HAND_EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
BODY_EDGES=[(5,6),(5,7),(7,9),(6,8),(8,10),(0,1),(0,2),(1,3),(2,4)]
COLORS={'unknown':(166,160,153),'pending':(166,160,153),'ok':(110,160,15),'wrong':(45,143,222)}

def annotate(frame,xy,ob):
    im=frame.copy();h,w=im.shape[:2];p=np.asarray(xy)
    valid=np.isfinite(p).all(1)&(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)
    palette={}
    for item in ob.get('highlights',[]):
        for idx in item.get('indices',[]):
            if isinstance(idx,int) and 0<=idx<133:palette[idx]=COLORS.get(item.get('state'),COLORS['unknown'])
    edges=BODY_EDGES+[(a+s,b+s)for s in [91,112]for a,b in HAND_EDGES]
    for a,b in edges:
        if valid[a]and valid[b]:
            color=palette.get(b,palette.get(a,COLORS['unknown']))
            cv2.line(im,tuple(p[a].astype(int)),tuple(p[b].astype(int)),color,2,cv2.LINE_AA)
    for i in list(range(11))+list(range(23,133)):
        if valid[i]:cv2.circle(im,tuple(p[i].astype(int)),1 if 23<=i<91 else 2,palette.get(i,COLORS['unknown']),-1,cv2.LINE_AA)
    return im

class LiveService:
    def __init__(self):
        self.lock=threading.RLock();self.condition=threading.Condition(self.lock)
        self.state='preparing';self.error=None;self.source=None;self.seq=0;self.generation=0
        self.countdown=0;self.observation=None;self.jpeg=None;self.metrics={}
        self.stop_event=threading.Event();self.reset_event=threading.Event();self.worker=None
        self.coach=None;self.engine=None;self.camera_available=False
        self.diagnostics=DiagnosticLedger()

    def prepare(self):
        try:
            import torch
            torch.set_num_threads(2)
            from coach import Coach
            self.coach=Coach()
            # Compile/warm at startup, never open the physical camera here.
            sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'))
            from stream_runtime import Engine
            self.engine=Engine()
            self.camera_available=True
            with self.lock:self.state='ready'
        except Exception as exc:
            with self.lock:
                self.state='ready' if self.coach is not None else 'error'
                self.error='카메라 분석기를 준비하지 못했어요. 공개 영상 확인은 사용할 수 있어요.' if self.coach is not None else '분석기를 준비하지 못했어요.'
            print('PREPARE_ERROR',type(exc).__name__,str(exc),flush=True)

    def status(self):
        with self.lock:
            return dict(state=self.state,error=self.error,source=self.source,seq=self.seq,generation=self.generation,
                countdown=self.countdown,observation=copy.deepcopy(self.observation),metrics=dict(self.metrics),diagnostics=self.diagnostics.snapshot(),
                camera_available=self.camera_available,training_use=False,video_saved=False,
                source_scope='한 사람의 공개 참고에 대한 단계별 실험. 전체 수어 정확도 검증은 아닙니다.')

    def start(self,mode):
        with self.lock:
            if mode not in ['camera','public_replay']:raise ValueError('지원하지 않는 입력이에요.')
            if self.coach is None or self.state=='preparing':raise ValueError('분석기를 준비하고 있어요.')
            if self.worker and self.worker.is_alive():raise ValueError('현재 연습을 종료한 뒤 다시 시작해 주세요.')
            if mode=='camera'and not self.camera_available:raise ValueError('카메라 분석기를 준비하지 못했어요.')
            self.stop_event.clear();self.reset_event.clear();self.coach.reset()
            self.generation+=1;self.seq=0;self.observation=None;self.jpeg=None;self.metrics={};self.error=None
            self.source=mode;self.state='opening' if mode=='camera' else 'replay';self.countdown=0
            self.diagnostics.begin(mode,self.generation)
            self.worker=threading.Thread(target=self.run,args=(mode,),daemon=True,name='hello-coaching')
            self.worker.start()

    def reset(self):
        with self.lock:
            if self.worker and self.worker.is_alive():
                self.reset_event.set();self.generation+=1;self.seq=0;self.observation=None;self.jpeg=None
                self.diagnostics.begin(self.source,self.generation)
                self.condition.notify_all();return
            mode=self.source
        if mode in ['camera','public_replay']:self.start(mode)

    def stop(self):
        self.stop_event.set()
        with self.lock:self.condition.notify_all()

    def publish(self,frame,xy,ob,metrics,expected_generation):
        im=annotate(frame,xy,ob)
        # Small preview does not alter inference coordinates or input crop.
        if im.shape[1]>800:im=cv2.resize(im,(800,round(im.shape[0]*800/im.shape[1])))
        ok,jpg=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,82])
        if ok:
            with self.condition:
                if expected_generation!=self.generation or self.stop_event.is_set()or self.reset_event.is_set():return
                self.jpeg=jpg.tobytes();self.seq+=1;self.observation=ob;self.metrics=metrics
                self.diagnostics.observe(ob,metrics);self.condition.notify_all()

    def run(self,mode):
        try:
            while True:
                if mode=='public_replay':self.replay()
                else:self.camera()
                # A reset arriving at the final input must restart, not vanish at EOF.
                with self.lock:
                    if self.reset_event.is_set()and not self.stop_event.is_set():
                        self.reset_event.clear();self.coach.reset()
                        self.state='replay' if mode=='public_replay'else'opening'
                        continue
                    self.state='ready' if self.stop_event.is_set() else 'done'
                    break
        except Exception as exc:
            with self.lock:self.state='error';self.error=str(exc)
            print('SESSION_ERROR',type(exc).__name__,str(exc),flush=True)
        finally:
            with self.condition:
                self.diagnostics.end('stopped' if self.stop_event.is_set() else self.state)
                if self.stop_event.is_set():self.observation=None;self.jpeg=None;self.countdown=0
                self.condition.notify_all()

    def replay(self):
        with np.load(U/'source/reference.npz',allow_pickle=False)as z:raw={k:z[k].copy() for k in z.files}
        cap=cv2.VideoCapture(str(REFERENCE))
        try:
            if not cap.isOpened():raise RuntimeError('참고 영상을 읽을 수 없어요.')
            # Accept only known source formats; normalize key names without guessing timings.
            xy=raw.get('xy_px',raw.get('xy'));scores=raw.get('score',raw.get('scores'))
            ts=raw['timestamp_us'];n=len(ts);i=0;start=time.perf_counter();frames=[]
            while i<n and not self.stop_event.is_set():
                if self.reset_event.is_set():
                    self.reset_event.clear();self.coach.reset();cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                    i=0;start=time.perf_counter();frames=[]
                with self.lock:generation=self.generation
                delay=(ts[i]-ts[0])/1e6-(time.perf_counter()-start)
                if delay>0 and self.stop_event.wait(delay):break
                ok,bgr=cap.read()
                if not ok:raise RuntimeError('공개 영상의 프레임 수가 관절 자료와 다릅니다.')
                h,w=bgr.shape[:2];t=time.perf_counter();ob=self.coach.update(xy[i],scores[i],int(ts[i]),w,h)
                frames.append((int(ts[i]),bgr.copy(),xy[i].copy()));frames=frames[-8:]
                assessed=ob.get('assessed_timestamp_us')
                if assessed is None:assessed=int(ts[i])
                match=min(frames,key=lambda f:abs(f[0]-assessed))
                self.publish(match[1],match[2],ob,dict(input_frame=i,observed_frames=i+1,total_public_frames=n,
                    observation_ms=round((time.perf_counter()-t)*1000,1),cached_public_points=True,
                    elapsed_seconds=round((ts[i]-ts[0])/1e6,2),camera_opened=False),generation)
                i+=1
        finally:cap.release()

    def camera(self):
        # Physical device access is ONLY reached by authenticated explicit /api/start camera.
        mailbox={'value':None,'received':0,'error':None};mlock=threading.Lock();ready=threading.Event();reader_stop=threading.Event()
        def read_camera():
            cap=None
            try:
                cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
                if not cap.isOpened():raise RuntimeError('카메라를 열지 못했어요. 다른 촬영 창을 닫고 다시 시도해 주세요.')
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
                cap.set(cv2.CAP_PROP_FPS,30);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
                while not reader_stop.is_set()and not self.stop_event.is_set():
                    ok,bgr=cap.read();captured=time.perf_counter()
                    if not ok:raise RuntimeError('카메라 영상이 끊겼어요.')
                    with mlock:
                        mailbox['received']+=1;mailbox['value']=(mailbox['received'],captured,bgr)
                    ready.set()
            except Exception as exc:
                with mlock:mailbox['error']=str(exc)
                ready.set()
            finally:
                if cap is not None:cap.release()
        reader=threading.Thread(target=read_camera,daemon=True,name='hello-camera-reader');reader.start()
        try:
            if not ready.wait(8):raise RuntimeError('카메라 응답이 늦어요. 연습을 다시 시작해 주세요.')
            t0=time.perf_counter();countdown_end=t0+3;last_id=0;last_ts=-1;observed=0;replaced=0;frames=[]
            with self.lock:self.state='countdown'
            while not self.stop_event.is_set():
                if self.reset_event.is_set():
                    self.reset_event.clear();self.coach.reset();t0=time.perf_counter();countdown_end=t0+3;last_ts=-1;frames=[]
                    with self.lock:self.state='countdown'
                with self.lock:generation=self.generation
                with mlock:error=mailbox['error'];item=mailbox['value']
                if error:raise RuntimeError(error)
                if not item or item[0]==last_id:
                    self.stop_event.wait(.005);continue
                fid,captured,bgr=item;replaced+=max(0,fid-last_id-1);last_id=fid
                if time.perf_counter()<countdown_end:
                    with self.lock:self.countdown=max(1,int(np.ceil(countdown_end-time.perf_counter())))
                    self.publish(bgr,np.full((133,2),np.nan),dict(stage_index=0,completed=[False,False],phase='countdown',message='편하게 준비해 주세요.',checks=[],highlights=[]),dict(camera_opened=True),generation)
                    self.stop_event.wait(.03);continue
                with self.lock:self.state='live';self.countdown=0
                ts=max(last_ts+1,round((captured-t0)*1e6));last_ts=ts
                r=self.engine.infer(bgr);observed+=1;h,w=bgr.shape[:2]
                t=time.perf_counter();ob=self.coach.update(r['xy'],r['score'],ts,w,h)
                frames.append((ts,captured,bgr.copy(),r['xy'].copy()));frames=frames[-8:]
                assessed=ob.get('assessed_timestamp_us')
                if assessed is None:assessed=ts
                match=min(frames,key=lambda f:abs(f[0]-assessed))
                self.publish(match[2],match[3],ob,dict(observed_frames=observed,captured_frames=fid,mailbox_replaced_frames=replaced,
                    observation_ms=round((time.perf_counter()-t)*1000,1),rtmw_ms=round(r['compute_ms'],1),
                    displayed_observation_age_ms=round((time.perf_counter()-match[1])*1000,1),
                    elapsed_seconds=round(ts/1e6,1),camera_opened=True,video_saved=False),generation)
                if generation==self.generation and not self.reset_event.is_set()and ob.get('stage_index')==2 and all(ob.get('completed',[])):break
        finally:
            reader_stop.set();reader.join(3)
            if reader.is_alive():raise RuntimeError('카메라 종료가 지연되고 있어요. 서버를 다시 실행해 주세요.')

SERVICE=LiveService()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def host_ok(self):return self.headers.get('Host')==f'127.0.0.1:{PORT}'
    def authorized(self):return self.host_ok() and self.headers.get('X-Session-Token')==TOKEN
    def reply(self,value,status=200):
        b=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def file(self,p):
        if not p.is_file():self.send_error(404);return
        size=p.stat().st_size;a=0;b=size-1;m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers.get('Range',''))
        if m:a=int(m[1]);b=min(b,int(m[2]))if m[2]else b
        if a>b:self.send_error(416);return
        self.send_response(206 if m else 200);self.send_header('Content-Type',mimetypes.guess_type(str(p))[0]or'application/octet-stream')
        self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(b-a+1))
        if m:self.send_header('Content-Range',f'bytes {a}-{b}/{size}')
        self.end_headers()
        with p.open('rb')as f:
            f.seek(a);remaining=b-a+1
            while remaining:
                block=f.read(min(65536,remaining))
                if not block:break
                self.wfile.write(block);remaining-=len(block)
    def do_GET(self):
        if not self.host_ok():self.reply({'error':'Local host only'},403);return
        route=urlsplit(self.path).path;q=parse_qs(urlsplit(self.path).query)
        try:
            if route=='/api/meta':
                self.reply(dict(token=TOKEN,reference='/media/reference.mp4',ready=SERVICE.coach is not None,
                    scope='공개 참고 1명의 핵심 동작을 이용한 실시간 코칭 시험 · 전체 수어 정확도는 미검증',
                    scope_detail='손가락 모양은 학습된 손모양 신경망으로 봅니다. 팔을 따라 쓸어내리는 방향과 양손의 위·아래 이동은 기존 시간 합성곱+GRU와 추가 학습한 방향 판정부로 확인합니다. 단계 순서·반복 횟수·시작과 끝의 충족은 별도 진행 규칙으로 연결합니다. 한 사람의 공개 인사 영상을 보강했으며, 엄지·정확한 손바닥 방향·실제 접촉과 여러 사용자의 정확도는 아직 완전 검증하지 않았습니다.',
                    action_model_scope=dict(connected=True,learned_direction_axes=[2,0],diagnostic_only_direction_axes=[],greeting_stage_training=False,movement_head_finetuned=True,word_tcn_connected=False),
                    stages=[dict(label='팔을 따라 손목 쪽으로 쓸기',guide='편 손으로 반대 팔을 따라 손목 쪽으로 이동해요.',image='/media/stage1.jpg',start_seconds=1.1),dict(label='두 주먹을 나란히 움직이기',guide='두 손을 주먹으로 만들고 함께 움직여요.',image='/media/stage2.jpg',start_seconds=1.6)]))
            elif route=='/api/status':
                if not self.authorized():self.reply({'error':'Token required'},403);return
                self.reply(SERVICE.status())
            elif route=='/preview.mjpeg':
                if q.get('token',[''])[0]!=TOKEN:self.reply({'error':'Token required'},403);return
                generation=int(q.get('generation',[SERVICE.generation])[0]);last=-1
                self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.send_header('Cache-Control','no-store');self.end_headers()
                while generation==SERVICE.generation:
                    with SERVICE.condition:
                        SERVICE.condition.wait_for(lambda:SERVICE.seq!=last or SERVICE.state in ['ready','done','error']or generation!=SERVICE.generation,timeout=1)
                        if generation!=SERVICE.generation:return
                        payload=SERVICE.jpeg;seq=SERVICE.seq;state=SERVICE.state
                    if payload is not None and seq!=last:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(payload)).encode()+b'\r\n\r\n'+payload+b'\r\n');self.wfile.flush();last=seq
                    if state in ['ready','done','error']:return
            elif route=='/media/reference.mp4':self.file(REFERENCE)
            elif route in ['/media/stage1.jpg','/media/stage2.jpg']:
                self.file(U/'source'/('sweep_start.jpg'if'stage1'in route else'fists_start.jpg'))
            else:
                rel='index.html'if route=='/'else route.lstrip('/');p=(U/'site'/rel).resolve()
                if not p.is_relative_to((U/'site').resolve())or p.suffix not in ['.html','.js','.css','.svg','.png','.jpg']:self.send_error(404);return
                self.file(p)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        except Exception as exc:self.reply({'error':str(exc)},500)
    def do_POST(self):
        origin=self.headers.get('Origin')
        if not self.authorized()or(origin and origin!=f'http://127.0.0.1:{PORT}'):
            self.reply({'error':'Local session required'},403);return
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<=n<=1024:raise ValueError('Request too large')
            data=json.loads(self.rfile.read(n)or'{}');route=urlsplit(self.path).path
            if route=='/api/start':SERVICE.start(data.get('mode'))
            elif route=='/api/reset':SERVICE.reset()
            elif route=='/api/stop':SERVICE.stop()
            else:self.send_error(404);return
            self.reply(SERVICE.status())
        except (ValueError,RuntimeError)as exc:self.reply({'error':str(exc)},409)
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass

def main():
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    threading.Thread(target=SERVICE.prepare,daemon=True,name='hello-prepare').start()
    print(f'HELLO_COACH http://127.0.0.1:{PORT}/ camera=explicit-start-only',flush=True)
    try:server.serve_forever()
    finally:SERVICE.stop();server.server_close()

if __name__=='__main__':main()
