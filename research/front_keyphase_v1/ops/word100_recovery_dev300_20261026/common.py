import sys,json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
OLD=O/'word100_yolo_hand_fourway_20261025'
B=O/'word100_body_boundary_20261017';T=O/'word100_frozen_c_temporal_20261019'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def same(a,b):return np.array_equal(a,b,equal_nan=True)
rule=module('dev300_frozen_geometry',O/'word100_fullframe_hand_merge_20261020/candidate.py')
def points(d):return np.asarray(d['xy'],dtype=float)
def roi(p,v,b):
    keep=v&np.isfinite(p).all(-1)&(p[...,0]>=b[0])&(p[...,0]<b[2])&(p[...,1]>=b[1])&(p[...,1]<b[3])
    return np.where(keep[...,None],p,np.nan).astype(np.float32),keep
def event(x):
    x={'at':now(),**x}
    with (U/'EVENTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(x,ensure_ascii=False),flush=True)
