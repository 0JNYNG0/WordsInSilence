from check_web import *
from form_web_adapter import CaptureService
import importlib.util
if __name__=='__main__':
 source=G/'public_visibility_roles_20261116/check_capture.py'
 spec=importlib.util.spec_from_file_location('fixed_public_fixture',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.D==D
 m.CaptureService=CaptureService
 m.main()
