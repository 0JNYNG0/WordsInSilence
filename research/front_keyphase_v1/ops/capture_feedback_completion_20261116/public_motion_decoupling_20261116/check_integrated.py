from pathlib import Path
import sys,importlib.util
E=Path(__file__).resolve().parent;G=E.parent;D=G/'public_shape_endpoint_learning_20261116'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
web=load('current_motion_consumer_check',D/'check_web.py')
sys.path.insert(0,str(E))
from motion_web_adapter import FeedbackConsumer,PriorConsumer,CaptureService
web.D=E;web.FeedbackConsumer=FeedbackConsumer;web.PriorConsumer=PriorConsumer
if __name__=='__main__':
 if '--capture'in sys.argv:
  sys.modules['check_web']=web;m=load('motion_capture_fixture',G/'public_visibility_roles_20261116/check_capture.py');m.CaptureService=CaptureService;m.main()
 else:web.main()
