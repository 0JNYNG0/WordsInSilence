from pathlib import Path
import sys, json, hashlib, datetime, importlib.util
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
GRAPH=HERE.parent; OPS=GRAPH.parent; PRIOR=GRAPH/'candidate_description_supervision'
sys.dont_write_bytecode=True
sys.path.insert(0,str(PRIOR))
import projected_direction_core as old
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8');q.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
def reverse_interval(x,ts,a,b):
    """Reflect the observed action in time, retaining outside times and coordinates."""
    out=x.copy(); tt=np.asarray(ts).copy()
    out[a:b+1]=x[a:b+1][::-1]
    tt[a:b+1]=ts[a]+ts[b]-np.asarray(ts[a:b+1])[::-1]
    # Only the swapped block's dt values need reconstruction; frame a follows
    # the same outside frame, endpoints and following interval are unchanged.
    delta=np.r_[float(x[0,279]),np.diff(tt).astype(float)/1e6]
    out[a:b+1,279]=delta[a:b+1]
    assert np.all(np.diff(tt)>0)
    return out,tt

