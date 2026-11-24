from pathlib import Path
import importlib.util
p=Path(__file__).resolve().parent.parent/'common.py'
s=importlib.util.spec_from_file_location('_joint_parent_common',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
globals().update({k:v for k,v in vars(m).items() if not k.startswith('__')})
U=Path(__file__).resolve().parent
