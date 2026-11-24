from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
U=Path(__file__).resolve().parent; O=U.parent; ROOT=O.parents[2]
OLD=O/'word100_yolo_hand_fourway_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def module(name,p):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
rule=module('selective_original_merge',O/'word100_fullframe_hand_merge_20261020/candidate.py')
def points(d):return np.asarray(d['xy'],dtype=float)
def roi(p,v,box):
    x0,y0,x1,y1=box
    keep=v & np.isfinite(p).all(-1)&(p[...,0]>=x0)&(p[...,0]<x1)&(p[...,1]>=y0)&(p[...,1]<y1)
    return np.where(keep[...,None],p,np.nan).astype(np.float32),keep
def native_m1(z,i,mprec):
    p=z['native_xy_px'][i,33:].reshape(2,21,2).copy();v=z['native_usable'][i,33:].reshape(2,21).copy()
    for q in mprec['decisions']:
        if q['accepted']:
            s=q['side'];assert not v[s].any()
            a=points(mprec['full'][q['index']]);good=np.asarray(q['joint_supported'],bool)
            p[s]=np.where(good[:,None],a,np.nan);v[s]=good
    # Reconstruct-before-ROI check; don't confuse cropped-out resting hands with misses.
    a,b=roi(p,v,z['roi_box']);vv=z['redetected_roi_usable'][i,33:].reshape(2,21)
    assert np.array_equal(b,vv)
    assert np.allclose(a[b],z['redetected_roi_xy_px'][i,33:].reshape(2,21,2)[b],atol=.0002,rtol=0)
    return p,v
