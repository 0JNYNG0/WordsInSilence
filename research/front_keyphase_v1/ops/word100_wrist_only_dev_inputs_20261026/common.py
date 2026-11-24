import sys,json,hashlib,importlib.util,copy
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
P=O/'word100_wrist_only_training_preflight_20261026'
B=O/'word100_body_boundary_20261017'
V=O/'word100_recovery_dev300_20261026'
R=O/'word100_public_recovery_bridge_20261026'
OLD=O/'word100_yolo_hand_fourway_20261025'
Y=O/'word100_yolo_selective_wrist_20261025'
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
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def equal(a,b):return np.array_equal(a,b,equal_nan=True)
rule=module('wrist_frozen_geometry',O/'word100_fullframe_hand_merge_20261020/candidate.py')
def points(d):return np.asarray(d['xy'],dtype=float)
def roi(p,v,b):
    keep=v&np.isfinite(p).all(-1)&(p[...,0]>=b[0])&(p[...,0]<b[2])&(p[...,1]>=b[1])&(p[...,1]<b[3])
    return np.where(keep[...,None],p,np.nan).astype(np.float32),keep
def event(x):
    x={'at':now(),**x}
    with (U/'EVENTS.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    print(json.dumps(x,ensure_ascii=False),flush=True)
def restore(ds):
    ds=copy.deepcopy(ds)
    for d in ds:d['xy']=np.asarray(d['xy'],dtype=float)
    return ds

def wrist_channels(view,base,old,wrists,valid):
    result={k:v.copy() for k,v in old.items()};changed={}
    for temporal in view['original_spans']:
        lo,hi=view['original_spans'][temporal];x=result['x_'+temporal]
        assert x.shape==(hi-lo+1,280)
        scale=view['original_scales'][temporal]['body'];assert np.isfinite(scale) and scale>0
        count=0
        for i in range(lo,hi+1):
            if not base['usable'][i,[11,12]].all():continue
            origin=(base['xy_px'][i,11]+base['xy_px'][i,12])/np.float32(2)
            for side in (0,1):
                if not valid[i,side]:continue
                assert not base['usable'][i,33+21*side:54+21*side].any()
                j=9+21*side;cols=[2*j,2*j+1,186+j]
                assert not old['x_'+temporal][i-lo,186+j]
                xy=(wrists[i,side]-origin)/scale
                assert np.isfinite(xy).all()
                x[i-lo,cols]=[float(xy[0]),float(xy[1]),1.0];count+=1
        changed[temporal]=count
    return result,changed
