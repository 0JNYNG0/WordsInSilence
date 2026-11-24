"""Side-symmetric observation quarantine. Not a hand-pose repair or visibility oracle.

No word/target/classifier input. Preserve all times and raw coordinates. Only exclude
weak, geometrically inconsistent estimates; no interpolation or trajectory smoothing.
"""
import numpy as np
from geometry import features
VERSION='rtmw_weak_geometry_v1'
CONFIG=dict(weak_score_ratio=.60,palm_arm=.80,span_arm=1.40,rootgap_arm=.50,
            edge_arm=.12,down_cos=.65,support_us=120000,support_frames=3)

def quarantine(xy,score,timestamp_us,width,height):
    q=np.asarray(xy,np.float32);s=np.asarray(score,np.float32);ts=np.asarray(timestamp_us,np.int64)
    if q.shape!=(len(ts),133,2) or s.shape!=(len(ts),133):raise ValueError('observation shape')
    if len(ts) and (not np.all(np.diff(ts)>0)):raise ValueError('nonmonotonic time')
    cleaned=q.copy();hand_mask=np.zeros((len(ts),2),bool);wrist_mask=np.zeros_like(hand_mask);reasons=np.zeros((len(ts),2),np.uint8)
    fs=features(q,s);shoulder=np.linalg.norm(q[:,5]-q[:,6],axis=1)
    for side,(sh,el,wr,start) in enumerate([(5,7,9,91),(6,8,10,112)]):
        g=fs[side];upper=np.linalg.norm(q[:,sh]-q[:,el],axis=1);unit=np.maximum(upper,shoulder*.5)
        reliable_arm=np.isfinite(q[:,[sh,el]]).all((1,2))&(unit>1)
        weak=np.isfinite(g['score_ratio'])&(g['score_ratio']<CONFIG['weak_score_ratio'])
        # RTMW scores are uncalibrated activations. This relative indicator alone
        # can never remove a hand; geometry and/or boundary/temporal evidence required.
        malformed=(g['palm_arm']>CONFIG['palm_arm'])|(g['span_arm']>CONFIG['span_arm'])|(g['rootgap_arm']>CONFIG['rootgap_arm'])
        wrist=q[:,wr];edge=np.minimum.reduce([wrist[:,0],width-wrist[:,0],wrist[:,1],height-wrist[:,1]])<CONFIG['edge_arm']*unit
        fore=wrist-q[:,el];down=(fore[:,1]>0)&(fore[:,1]/np.maximum(np.linalg.norm(fore,axis=1),1e-6)>CONFIG['down_cos'])&(q[:,el,1]>q[:,sh,1])
        boundary=weak&reliable_arm&edge&down
        supported=np.zeros(len(ts),bool)
        for i in np.flatnonzero(boundary):
            lo=np.searchsorted(ts,ts[i]-CONFIG['support_us']);hi=np.searchsorted(ts,ts[i]+CONFIG['support_us'],side='right')
            supported[i]=np.count_nonzero(boundary[lo:hi])>=CONFIG['support_frames']
        bad=(weak&reliable_arm&malformed)|supported
        # Preserve a plausible interior arm wrist even if its finger pattern is bad.
        # Boundary wrist estimates are not claimed as visible body landmarks.
        bw=bad&edge&down
        cleaned[bad,start:start+21]=np.nan;cleaned[bw,wr]=np.nan
        hand_mask[:,side]=bad;wrist_mask[:,side]=bw
        reasons[:,side]=np.where(weak&reliable_arm&malformed,1,0)|np.where(supported,2,0)
    return dict(xy_px=cleaned,hand_quarantined=hand_mask,wrist_quarantined=wrist_mask,reason_bits=reasons,version=VERSION)
