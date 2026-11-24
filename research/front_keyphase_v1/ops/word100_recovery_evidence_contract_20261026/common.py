import sys,json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
V=O/'word100_recovery_dev300_20261026';R=O/'word100_public_recovery_bridge_20261026'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def same(a,b):return np.array_equal(a,b,equal_nan=True)
