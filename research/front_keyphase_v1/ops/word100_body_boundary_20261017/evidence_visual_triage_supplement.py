"""Read-only low-waist transition supplement for an ambiguous source sequence."""
import json
from pathlib import Path
import numpy as np
import av
from evidence_visual_triage import U, read, sha, sheet

cid='NIA_SL_WORD1085_REAL06_F'
m=read(U/f'evidence_visual_triage_{cid}_v2.json')
assert sha(m['source'])==m['source_sha256']
assert sha(m['observation'])==m['observation_sha256']
with np.load(m['observation'],allow_pickle=False) as z:
    xy,us,ob,ts=(z[k] for k in ('xy_px','usable','observed','timestamp_us'))
old=us&ob&np.isfinite(xy).all(-1)
new=old&(xy[:,:,0]>=0)&(xy[:,:,0]<1920)&(xy[:,:,1]>=0)&(xy[:,:,1]<m['bottom'])
core=(ts>=round(m['core_seconds'][0]*1e6))&(ts<=round(m['core_seconds'][1]*1e6))
ids=list(range(70,92,2))+[85,87,89,91]
ids=sorted(set(ids));images={}
with av.open(m['source']) as source:
    for i,f in enumerate(source.decode(video=0)):
        if i in ids:images[i]=f.to_ndarray(format='rgb24')
outs=[]
for off in range(0,len(ids),12):
    outs.append(sheet(cid,m['label'],m['bottom'],ids[off:off+12],images,xy,old,ts,core,~new[:,33:75].any(1),f'low_transition_{off//12+1}'))
out={'clip_id':cid,'source_sha256':m['source_sha256'],'observation_sha256':m['observation_sha256'],
     'images':{str(p):sha(p) for p in outs},'bottom':m['bottom'],
     'frames':[{'frame':int(i),'seconds':float(ts[i])/1e6,'right_native_points':int(old[i,54:75].sum()),
                'right_roi_points':int(new[i,54:75].sum()),'right_points_removed':np.flatnonzero(old[i,54:75]&~new[i,54:75]).tolist()} for i in ids],
     'detector_runs':0,'training_runs':0}
with (U/'evidence_visual_triage_LOW_TRANSITION.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
