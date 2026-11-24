from pathlib import Path
import json,hashlib,sys,importlib.util
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];P=O/'word100_finger_repair_20261115';D=O/'word100_clause_finger_diagnostics_20261114';J=O/'word100_rtmw_joint_learning_20261108'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item()if isinstance(x,np.generic)else x.tolist())+'\n','utf-8')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
