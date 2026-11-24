"""Local isolated F-G16 experiment; no execution on import."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,sys,os,time
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
BASE=O/'word100_body_boundary_20261017';T=O/'word100_frozen_c_temporal_20261019';P=O/'word100_frozen_order_aux_preflight_20261019'
VIEWS=('masked','sensor');SEEDS=(20261013,20261014,20261015);ARM='FG16M3'
F=O/'word100_frozen_order_aux_train_20261019';R=O/'word100_auxiliary_supervision_plan_20261019'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def write(p,x,new=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if new:assert not p.exists(),str(p)
 temp=p.with_name(p.name+'.writing');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
 for attempt in range(8):
  try:os.replace(temp,p);return
  except PermissionError:
   if attempt==7:raise
   time.sleep(.05*(attempt+1))
def check(pins):
 for p,h in pins.items():assert sha(p)==h,('protected_hash_changed',p)
def statehash(state):
 h=hashlib.sha256()
 for k,t in sorted(state.items()):
  h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def preflight_hash(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def event(**kwargs):
 x={'at':now(),**kwargs};write(U/'STATUS.json',x)
 with (U/'EVENTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 print(json.dumps(x,ensure_ascii=False),flush=True)

A=O/'word100_auxiliary_supervision_train_20261019';D=O/'word100_auxiliary_order_diagnosis_20261019'
