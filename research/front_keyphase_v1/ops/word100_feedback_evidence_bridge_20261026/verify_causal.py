"""Independently rebuild oracle numerical components from public raw intervals.

No candidate descriptor/mapping/bridge imports; verify at common-supported cells.
"""
from pathlib import Path
import json, numpy as np
U=Path(__file__).resolve().parent;O=U.parent
P=O/'word100_point_sequence_split_20261022';I=O/'word100_cross_person_motion_support_20261022'/'inputs'
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def interp(a,x):
    first=np.floor(x).astype(int);last=np.ceil(x).astype(int);w=(x-first).reshape((-1,)+(1,)*(a.ndim-1))
    return a[first]*(1-w)+a[last]*w
def components(a,b):
    # three moments at first axis -> five independent components at last axis
    return np.moveaxis(np.concatenate((a,(a[1]-b[0])[None],(a[2]-b[1])[None]),axis=0),0,-1)
r=json.loads((U/'RESULT.json').read_text('utf-8'))
total=0;maximum=0.0
for row in r['causal']:
    cid=row['name'].split('__',1)[1];e=load(I/(cid+'.npz'));saved=load(U/'causal'/(row['name']+'.npz'))
    old=load(P/'known'/(row['name']+'.npz'));ix=old['source_index'];n=len(e['core'])
    position=np.concatenate((e['centre'],e['arm_xy'].reshape(n,2,4)),axis=-1)
    radius=np.concatenate((np.repeat(e['radius'][:,:,None],2,axis=-1),np.zeros((n,2,4))),axis=-1)
    lo=np.concatenate((e['angle_low'],position-radius),axis=-1)
    hi=np.concatenate((e['angle_high'],position+radius),axis=-1)
    querylo=interp(lo,ix);queryhi=interp(hi,ix)
    actual=np.array([np.median(old['path'][old['path'][:,1]==j,0]) for j in range(n)])
    unique=sorted(set(ix.tolist()));where=[np.flatnonzero(ix==x).mean() for x in unique]
    oracle=np.interp(np.arange(n),unique,where)
    assert np.allclose(actual,saved['actual_index']) and np.allclose(oracle,saved['oracle_index'])
    for j in range(2,n-2):
        good=saved['common'][j]
        if not good.any():continue
        indices=np.array([j-2,j,j+2]);al=components(lo[indices],hi[indices]);ah=components(hi[indices],lo[indices])
        for alignment_index,mapped in enumerate((actual,oracle)):
            qi=mapped[indices]
            source=np.interp(qi,np.arange(len(ix)),ix)
            for sampling,(ql,qh) in enumerate(((interp(querylo,qi),interp(queryhi,qi)),(interp(lo,source),interp(hi,source)))):
                bl=components(ql,qh);bh=components(qh,ql)
                # Center/radius equivalent, distinct arithmetic from candidate gap.
                midA=(al+ah)/2;midB=(bl+bh)/2;radA=(ah-al)/2;radB=(bh-bl)/2
                gap=np.maximum(np.abs(midA-midB)-radA-radB,0)
                expected=saved['gaps'][alignment_index*2+sampling,j]
                err=np.abs(gap[good]-expected[good]);assert np.isfinite(err).all()
                maximum=max(maximum,float(err.max()));total+=err.size
                assert (err<1e-10).all(),(row['name'],j,alignment_index,sampling,float(err.max()))
out=dict(status='PASS',cases=len(r['causal']),component_scalars_verified=total,max_absolute_error=maximum,
    verification='Same Codex separate raw-interval, interpolation, mapping and center-radius arithmetic; no candidate helper imports.',
    limitation='Verifies diagnostic construction, not unknown real-user correspondence or human sign correctness.')
with (U/'CAUSAL_VERIFIED.json').open('x',encoding='utf-8') as h:json.dump(out,h,indent=2)
print(json.dumps(out))
