from pathlib import Path
import json,hashlib,sys,importlib.util
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
N=O/'word100_finger_error_split_20261115';P=O/'word100_finger_repair_20261115';Q=O/'word100_finger_quality_transition_20261115';H=O/'word100_finger_holdout_20261115';D=O/'word100_clause_finger_diagnostics_20261114';J=O/'word100_rtmw_joint_learning_20261108'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else v.item())+'\n','utf-8')
def load_channels(p):
 z=np.load(p);key=next(k for k in ['feedback','tcn280','full']if k in z);x=z[key];assert x.ndim==2 and x.shape[1]==280;return x,z['frame_id']
