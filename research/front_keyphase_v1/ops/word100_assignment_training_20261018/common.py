"""Local P/Q research unit. No training or detector invocation on import."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,sys,importlib.util,os
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
BASE=O/'word100_body_boundary_20261017';OBS=O/'word100_uncertain_hand_preservation_20261018'
F=O/'word100_forearm_contour_20261018';D=O/'word100_duplicate_wrist_quality_20261018'
PLAN=O/'word100_assignment_training_plan_20261018'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def write(p,x,new=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if new:assert not p.exists(),str(p)
 t=p.with_name(p.name+'.writing');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');os.replace(t,p)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def check(pins):
 for p,h in pins.items():assert sha(p)==h,('input_hash_changed',p)
def event(**kw):
 x={'at':now(),**kw};write(U/'STATUS.json',x);print(json.dumps(x,ensure_ascii=False),flush=True)
 with (U/'EVENTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
