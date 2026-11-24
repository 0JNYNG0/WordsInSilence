"""Separate port8822, public fixture only. Cannot open a device camera."""
from bootstrap import *
import importlib.util,time,json,av,cv2,threading,os
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'))
spec=importlib.util.spec_from_file_location('g6_public_web',O/'web_tutorial_live_integration_20261113/server.py');web=importlib.util.module_from_spec(spec);spec.loader.exec_module(web)
web.PORT=8822;web.ACCEPTANCE_PATH=U/'PUBLIC_BROWSER_ACCEPTANCE.json';web.SERVICE.public_fixture=True
import capture
word=sys.argv[1] if len(sys.argv)>1 else 'WORD0739'
assert word in ['WORD0739','WORD1133']
source=O/'word100_t3f_heldout_20261019/data/clips'/f'NIA_SL_{word}_REAL17_F/source.mp4'
with av.open(str(source)) as c:frames=[cv2.resize(f.to_ndarray(format='bgr24'),(640,360)) for f in c.decode(video=0)]
class PublicCamera:
 def __init__(self,*args):self.index=0;self.start=None
 def isOpened(self):return True
 def set(self,*args):return True
 def get(self,k):return {cv2.CAP_PROP_FRAME_WIDTH:640,cv2.CAP_PROP_FRAME_HEIGHT:360,cv2.CAP_PROP_FPS:30}.get(k,0)
 def read(self):
  time.sleep(1/30)
  if web.SERVICE.state=='recording':
   if self.start is None:self.start=time.perf_counter()
   self.index=round((time.perf_counter()-self.start)*30)
   f=frames[min(self.index,len(frames)-1)]
   if self.index>len(frames)+90:web.SERVICE.finish_requested=True
   return True,f.copy()
  return True,frames[0].copy()
 def release(self):pass
capture.cv2.VideoCapture=PublicCamera
server=web.ThreadingHTTPServer(('127.0.0.1',web.PORT),web.Handler)
threading.Thread(target=web.SERVICE.prepare,daemon=True).start()
(U/'PUBLIC_BROWSER_SERVER.json').write_text(json.dumps(dict(pid=os.getpid(),port=web.PORT,physical_camera=False,word=word)),encoding='utf-8')
server.serve_forever()
