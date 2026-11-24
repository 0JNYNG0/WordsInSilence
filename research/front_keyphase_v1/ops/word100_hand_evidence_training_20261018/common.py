"""Isolated, local public-data experiment. No devices or training on import."""
import json, hashlib, sys, os
from pathlib import Path
from datetime import datetime, timezone, timedelta
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent
O=U.parent
ROOT=O.parents[2]
BASE=O/'word100_body_boundary_20261017'
PLAN=O/'word100_hand_evidence_training_plan_20261017'
CURRENT=O/'word100_current_extent_20261017'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
def write(p,x,new=False):
 p=Path(p)
 if new and p.exists():raise FileExistsError(p)
 t=p.with_name(p.name+'.writing');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)
def checkpins(pins):
 for p,h in pins.items():assert sha(p)==h, 'Changed pinned file: '+p
def loadnp(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

