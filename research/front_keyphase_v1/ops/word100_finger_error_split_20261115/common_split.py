from pathlib import Path
import json,hashlib,sys,importlib.util
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];Q=O/'word100_finger_quality_transition_20261115';P=O/'word100_finger_repair_20261115';H=O/'word100_finger_holdout_20261115';D=O/'word100_clause_finger_diagnostics_20261114';J=O/'word100_rtmw_joint_learning_20261108'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else v.item())+'\n','utf-8')
def source(cid):
 for base in [O/'word100_wrist_only_inputs_20261026/sources',O/'word100_label_only_audit_20261018/videos',O/'word100_role_dtw_20261017/public_videos']:
  p=base/f'{cid}.mp4'
  if p.exists():return p
 p=O/'word100_autonomous_recognition_20261026/fresh199/data/clips'/cid/'source.mp4'
 if p.exists():return p
 return None
