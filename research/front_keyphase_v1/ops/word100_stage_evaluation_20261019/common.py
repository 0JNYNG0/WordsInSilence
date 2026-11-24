from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,sys
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
B=O/'word100_body_boundary_20261017';AUD=O/'word100_sequence_input_audit_20261019'
M3=O/'word100_multi_position_aux_train_20261019'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def write(p,x):
 p=Path(p);assert not p.exists(),p
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def check(pins):
 for p,h in pins.items():assert sha(p)==h,p
