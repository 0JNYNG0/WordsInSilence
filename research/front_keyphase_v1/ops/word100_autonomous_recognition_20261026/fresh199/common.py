from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,sys,os
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parents[1];ROOT=O.parents[2]
B=O/'word100_body_boundary_20261017';T=O/'word100_frozen_c_temporal_20261019'
DATA=U/'data';SEEDS=(20261013,20261014,20261015);VIEWS=('masked','sensor')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def write(p,x,new=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if new:assert not p.exists(),str(p)
 q=p.with_name(p.name+'.writing');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');os.replace(q,p)
def check(pins):
 for p,h in pins.items():assert sha(p)==h,('changed',p)
def event(**kw):
 r={'at':now(),**kw};write(U/'STATUS.json',r)
 with (U/'EVENTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(json.dumps(r,ensure_ascii=False),flush=True)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
