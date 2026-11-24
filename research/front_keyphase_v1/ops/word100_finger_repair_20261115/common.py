from pathlib import Path
import json,hashlib,sys
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_clause_finger_diagnostics_20261114';E=O/'word100_feature_expansion_20261113';J=O/'word100_rtmw_joint_learning_20261108'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
