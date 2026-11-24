"""Local-only HTTP streaming test with synthetic JPEGs; no camera or models."""
from pathlib import Path
import sys,threading,time,json,urllib.request,cv2,numpy as np
U=Path(__file__).resolve().parent;D=U.parent/'word100_action_feedback_demo_20261108';sys.path.insert(0,str(D));import server
def main():
 svc=server.SERVICE;svc.state='preview';http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler);threading.Thread(target=http.serve_forever,daemon=True).start();port=http.server_address[1]
 def feed():
  for i in range(45):
   _,encoded=cv2.imencode('.jpg',np.full((120,160,3),i*4,np.uint8))
   with svc.preview_condition:svc.jpeg=encoded.tobytes();svc.jpeg_seq+=1;svc.preview_condition.notify_all()
   time.sleep(1/30)
  svc.state='ready'
  with svc.preview_condition:svc.preview_condition.notify_all()
 thread=threading.Thread(target=feed);thread.start();count=0;start=time.perf_counter()
 with urllib.request.urlopen(f'http://127.0.0.1:{port}/preview.mjpeg',timeout=4) as r:
  assert r.headers['Content-Type'].startswith('multipart/x-mixed-replace')
  while True:
   line=r.readline()
   if not line:break
   if line.strip()!=b'--frame':continue
   length=0
   while True:
    line=r.readline()
    if line==b'\r\n':break
    if line.lower().startswith(b'content-length:'):length=int(line.split(b':')[1])
   payload=r.read(length);assert len(payload)==length and payload[:2]==b'\xff\xd8';r.read(2);count+=1
 elapsed=time.perf_counter()-start;thread.join();http.shutdown();http.server_close();result=dict(status='PASS' if count>=35 and count/elapsed>15 else 'FAIL',frames=count,seconds=elapsed,preview_fps=count/elapsed,single_http_stream=True,hardware_camera=False,real_gpu=False)
 (U/'PREVIEW_VERIFIED.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result);assert result['status']=='PASS'
if __name__=='__main__':main()
