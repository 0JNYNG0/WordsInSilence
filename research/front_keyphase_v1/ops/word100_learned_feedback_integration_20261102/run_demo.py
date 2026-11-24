"""Loopback-only learned feedback demo. Private submissions are inference-only."""
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse,parse_qs
from pathlib import Path
import json,uuid,threading,time,traceback,secrets,os
from learner_adapter import U,O,read,write,analyze,warmup,WORDS
PORT=8807;TOKEN=secrets.token_urlsafe(32);LOCK=threading.Lock();POOL=ThreadPoolExecutor(max_workers=1)
JOBS={};STATE={'ready':False,'message':'모델을 준비하고 있습니다.'};FUTURE=None
LABELS=dict(zip(WORDS,['순종','잘하다','조상','이집트']))
manifest=read(U.parent/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json')['rows']
REFERENCES={w:Path(next(r['source'] for r in manifest if r['word_id']==w and r['person']==1)) for w in WORDS}
PUBLIC={w:O/'word100_learned_feature_retention_20261102/video_controls'/f'{w}_reencoded_normal.mp4' for w in WORDS}
def init():
 try:STATE.update(ready=True,message='새 촬영을 분석할 수 있습니다.',warmup=warmup())
 except Exception:STATE.update(ready=False,message='모델 준비에 실패했습니다.',error=traceback.format_exc());(U/'DEMO_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
def perform(jid,source,word,condition):
 job=JOBS[jid]
 try:
  result=analyze(source,Path(job.get('analysis_directory',str(U/'private_sessions'/jid/'analysis'))),word,condition,lambda x:job.update(progress=x))
  job.update(status='complete',result=result)
 except Exception as e:job.update(status='failed',error=str(e))
def serve():
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def json(self,value,status=200):
   b=json.dumps(value,ensure_ascii=False,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(b)
  def file(self,path,ctype):
   if not path.is_file():return self.json({'error':'파일 없음'},404)
   size=path.stat().st_size;lo,hi=0,size-1;partial=False
   range_header=self.headers.get('Range')
   if range_header:
    import re
    match=re.fullmatch(r'bytes=(\d+)-(\d*)',range_header)
    if not match:return self.json({'error':'Unsupported range'},416)
    lo=int(match[1]);hi=min(int(match[2]) if match[2] else hi,hi);partial=True
    if lo>hi:return self.json({'error':'Invalid range'},416)
   self.send_response(206 if partial else 200);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(hi-lo+1));self.send_header('Accept-Ranges','bytes')
   if partial:self.send_header('Content-Range',f'bytes {lo}-{hi}/{size}')
   self.send_header('Cache-Control','no-store');self.end_headers()
   try:
    with path.open('rb') as f:
     f.seek(lo);remaining=hi-lo+1
     while remaining:
      b=f.read(min(1048576,remaining))
      if not b:break
      self.wfile.write(b);remaining-=len(b)
   except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
  def do_GET(self):
   route=urlparse(self.path).path
   if route=='/':return self.file(U/'demo.html','text/html; charset=utf-8')
   if route=='/app.js':return self.file(U/'demo.js','text/javascript; charset=utf-8')
   if route=='/api/state':return self.json(dict(STATE,token=TOKEN,words=LABELS,busy=FUTURE is not None and not FUTURE.done()))
   if route.startswith('/api/job/'):
    j=JOBS.get(route.rsplit('/',1)[-1]);return self.json(j if j else {'error':'Unknown job'},200 if j else 404)
   if route.startswith('/api/public/'):
    w=route.rsplit('/',1)[-1]
    if w not in WORDS:return self.json({'error':'Unknown word'},404)
    p=U/'demo_public'/w/'RESULT.json'
    return self.json(read(p)) if p.exists() else self.json({'error':'Public check not complete'},409)
   if route.startswith('/api/control/'):
    w=route.rsplit('/',1)[-1]
    if w not in WORDS:return self.json({'error':'Unknown word'},404)
    return self.json(read(U/'demo_controls'/w/'RESULT.json'))
   if route.startswith('/api/order/'):
    w=route.rsplit('/',1)[-1]
    if w not in WORDS:return self.json({'error':'Unknown word'},404)
    return self.json(read(U/'demo_order'/w/'RESULT.json'))
   if route.startswith('/media/order/'):
    w=route.rsplit('/',1)[-1]
    if w not in WORDS:return self.json({'error':'Unknown word'},404)
    return self.file(U/'order_videos'/f'{w}_reverse_main.mp4','video/mp4')
   if route.startswith('/media/control/'):
    w=route.rsplit('/',1)[-1]
    if w not in WORDS:return self.json({'error':'Unknown word'},404)
    return self.file(O/'word100_learned_feature_retention_20261102/video_controls'/f'{w}_action_cut.mp4','video/mp4')
   if route.startswith('/media/reference/'):
    p=REFERENCES.get(route.rsplit('/',1)[-1]);return self.file(p,'video/mp4') if p else self.json({},404)
   if route.startswith('/media/public/'):
    p=PUBLIC.get(route.rsplit('/',1)[-1]);return self.file(p,'video/mp4') if p else self.json({},404)
   if route.startswith('/media/session/'):
    jid=route.rsplit('/',1)[-1];j=JOBS.get(jid)
    return self.file(U/'private_sessions'/jid/'input.video',j['mime']) if j else self.json({},404)
   return self.json({'error':'Not found'},404)
  def do_POST(self):
   global FUTURE
   u=urlparse(self.path)
   if u.path not in ('/api/analyze','/api/analyze-public'):return self.json({},404)
   if self.headers.get('X-Session-Token')!=TOKEN:return self.json({'error':'잘못된 세션'},403)
   origin=self.headers.get('Origin')
   if origin and origin not in (f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}'):return self.json({},403)
   q=parse_qs(u.query);word=q.get('word',[''])[0];condition=q.get('condition',[''])[0]
   if word not in WORDS or condition not in ('learned_motion',):return self.json({'error':'지원하지 않는 조건'},400)
   if u.path=='/api/analyze-public':
    with LOCK:
     if not STATE['ready'] or (FUTURE is not None and not FUTURE.done()):return self.json({'error':'분석이 끝난 뒤 다시 시도해 주세요.'},409)
     jid=uuid.uuid4().hex;d=U/'public_runtime_runs'/jid;d.mkdir(parents=True,exist_ok=False)
     write(d/'INTENT.json',dict(source=str(PUBLIC[word]),word=word,condition=condition,training=False,public_only=True))
     JOBS[jid]=dict(id=jid,status='running',progress={'stage':'starting'},mime='video/mp4',analysis_directory=str(d/'analysis'),public_only=True)
     FUTURE=POOL.submit(perform,jid,PUBLIC[word],word,condition)
    return self.json({'id':jid},202)
   try:length=int(self.headers.get('Content-Length','0'))
   except ValueError:return self.json({},400)
   if not 0<length<=64*1024*1024:return self.json({'error':'영상은 64MB 이하로 선택해 주세요.'},413)
   with LOCK:
    if not STATE['ready'] or (FUTURE is not None and not FUTURE.done()):return self.json({'error':'분석이 끝난 뒤 다시 시도해 주세요.'},409)
    body=self.rfile.read(length)
    if len(body)!=length:return self.json({'error':'업로드가 중단됐습니다.'},400)
    jid=uuid.uuid4().hex;d=U/'private_sessions'/jid;d.mkdir(parents=True,exist_ok=False);(d/'input.video').write_bytes(body)
    write(d/'INFERENCE_ONLY.json',dict(training=False,tuning=False,candidate_selection=False,source='user submitted through local demo',word=word,condition=condition))
    mime=self.headers.get('X-Video-Type','video/webm');mime=mime if mime in ('video/webm','video/mp4') else 'application/octet-stream'
    JOBS[jid]=dict(id=jid,status='running',progress={'stage':'starting'},mime=mime)
    FUTURE=POOL.submit(perform,jid,d/'input.video',word,condition)
   return self.json({'id':jid},202)
 server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
 write(U/'DEMO_LAUNCH.json',dict(pid=os.getpid(),url=f'http://127.0.0.1:{PORT}/',new_reference_comparisons=0,training=False))
 threading.Thread(target=init,daemon=True).start();print(f'http://127.0.0.1:{PORT}/',flush=True);server.serve_forever()
if __name__=='__main__':serve()
