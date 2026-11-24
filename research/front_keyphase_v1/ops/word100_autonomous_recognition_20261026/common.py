from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,sys
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
B=O/'word100_body_boundary_20261017';T=O/'word100_frozen_c_temporal_20261019';E=O/'word100_t3f_heldout_20261019'
SEEDS=(20261013,20261014,20261015);VIEWS=('sensor','masked')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x,new=True):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x' if new else 'w',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for q in iter(lambda:f.read(1<<20),b''):h.update(q)
 return h.hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def softmax(z):
 a=np.asarray(z,dtype=np.float64);a=np.exp(a-a.max(axis=-1,keepdims=True));return a/a.sum(axis=-1,keepdims=True)
def combine(logits):
 a=np.asarray(logits,dtype=np.float64);assert a.shape==(3,100) and np.isfinite(a).all()
 return softmax(a).mean(axis=0)
def ranking(p):return sorted(range(len(p)),key=lambda i:(-float(p[i]),i))
