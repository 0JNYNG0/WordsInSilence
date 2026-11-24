from pathlib import Path
import sys,importlib.util
J=Path(__file__).resolve().parent;G=J.parent;D=G/'public_shape_endpoint_learning_20261116';E=G/'public_motion_decoupling_20261116'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
web=load('current_view_consumer_check',D/'check_web.py')
sys.path.insert(0,str(J))
from view_web_adapter import FeedbackConsumer,PriorConsumer,CaptureService
web.D=J;web.FeedbackConsumer=FeedbackConsumer;web.PriorConsumer=PriorConsumer
if __name__=='__main__':
 if '--capture'in sys.argv:
  sys.modules['check_web']=web;m=load('view_capture_fixture',G/'public_visibility_roles_20261116/check_capture.py');m.CaptureService=CaptureService;m.main()
 elif '--cards'in sys.argv:
  m=load('view_cards',E/'check_cards.py');m.FeedbackConsumer=FeedbackConsumer;m.E=J;m.main()
 else:web.main()
