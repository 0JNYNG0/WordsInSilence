"""Start the local demo without opening any camera; reuse an existing server."""
from pathlib import Path
import json,sys,time,subprocess,urllib.request,urllib.error,webbrowser
ROOT=Path(__file__).resolve().parent
OPS=ROOT/'research/front_keyphase_v1/ops'
PYTHON=OPS/'word100_signrep_feasibility_20261029/runtime/Scripts/python.exe'
SERVER=OPS/'capture_feedback_completion_20261116/public_shape_endpoint_learning_20261116/serve.py'
URL='http://127.0.0.1:8821'
def get(path,token=None):
 req=urllib.request.Request(URL+path,headers={'X-Session-Token':token}if token else{})
 with urllib.request.urlopen(req,timeout=3)as r:return json.load(r)
def main():
 check='--check-only'in sys.argv
 try:meta=get('/api/meta')
 except (OSError,ValueError):
  if check:raise RuntimeError('시험 서버가 꺼져 있습니다. START_WEB_DEMO.cmd로 켜 주세요.')
  if not PYTHON.is_file()or not SERVER.is_file():raise RuntimeError('프로젝트 실행 파일을 찾지 못했습니다.')
  logs=OPS/'capture_feedback_completion_20261116/demo_scope_20261116';logs.mkdir(exist_ok=True)
  with (logs/'launcher_stdout.log').open('a',encoding='utf-8')as out,(logs/'launcher_stderr.log').open('a',encoding='utf-8')as err:
   subprocess.Popen([str(PYTHON),'-X','utf8','-B',str(SERVER)],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  meta=None
  for _ in range(20):
   try:meta=get('/api/meta');break
   except (OSError,ValueError):time.sleep(.5)
  if meta is None:raise RuntimeError('서버 시작을 확인하지 못했습니다. launcher_stderr.log를 확인해 주세요.')
 ids={r['id']for r in meta.get('words',[])}
 if not {'WORD0099','WORD1290'}.issubset(ids)or'SEN0354'in ids:raise RuntimeError('현재 실행된 서버는 최신 사과·감사 시연 구성이 아닙니다. 진행 중인 촬영을 닫고 서버 갱신이 필요합니다.')
 if [s['id']for s in meta['quest']['steps']]!=['WORD0099','WORD1290']:raise RuntimeError('실전 퀘스트 구성이 다릅니다.')
 if not check:print('분석기를 준비하고 있습니다. 카메라는 화면에서 직접 켜 주세요.',flush=True)
 deadline=time.monotonic()+100
 while True:
  status=get('/api/status',meta['token'])
  if status['state']!='preparing':break
  if time.monotonic()>deadline:raise RuntimeError('분석기 준비 시간이 길어지고 있습니다. 서버 로그를 확인해 주세요.')
  time.sleep(.5)
 if status['state']=='error':raise RuntimeError(status.get('error')or'분석기 준비 오류')
 if not status.get('hand_form_active'):raise RuntimeError('이전 판정부 서버가 실행 중입니다. 촬영을 닫은 뒤 최신 서버로 다시 시작해 주세요.')
 print(json.dumps(dict(url=URL+'/',state=status['state'],camera_active=status.get('camera_thread_active',False),quest=['사과','감사'],hello_in_demo=False),ensure_ascii=False))
 if not check and '--prepare-only'not in sys.argv:webbrowser.open(URL+'/')
if __name__=='__main__':
 try:main()
 except Exception as e:print(str(e),file=sys.stderr);sys.exit(1)
