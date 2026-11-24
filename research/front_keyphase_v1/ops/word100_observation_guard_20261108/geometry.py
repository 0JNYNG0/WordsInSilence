import numpy as np
def features(xy,score):
    q=np.asarray(xy,dtype=np.float32);s=np.asarray(score,dtype=np.float32)
    shoulder=np.linalg.norm(q[:,5]-q[:,6],axis=1)
    out=[]
    for sh,el,wr,start in [(5,7,9,91),(6,8,10,112)]:
        upper=np.linalg.norm(q[:,sh]-q[:,el],axis=1)
        denom=np.maximum(upper,shoulder*.5)
        hand=q[:,start:start+21]
        palm=np.linalg.norm(hand[:,[5,9,13,17]]-hand[:,0,None],axis=2)
        size=np.median(palm,axis=1)
        span=np.max(np.linalg.norm(hand-hand[:,0,None],axis=2),axis=1)
        rootgap=np.linalg.norm(q[:,wr]-hand[:,0],axis=1)
        arm_score=np.median(s[:,[sh,el,wr]],axis=1)
        hand_score=np.median(s[:,start:start+21],axis=1)
        out.append(dict(palm_arm=size/np.maximum(denom,1e-6),span_arm=span/np.maximum(denom,1e-6),rootgap_arm=rootgap/np.maximum(denom,1e-6),score_ratio=hand_score/np.maximum(arm_score,1e-6),palm_spokes_arm=palm/np.maximum(denom[:,None],1e-6)))
    return out
