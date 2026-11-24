"""Keep the previous feature architecture; isolate this trial's data and outputs."""
from common import *
previous=load('_recovery_previous_fine',PREV/'model_core.py')
torch=previous.torch;nn=previous.nn;F=previous.F;old=previous.old
FineNet=previous.FineNet;hand_features=previous.hand_features
batch=previous.batch;retention=previous.retention;supervision=previous.supervision

def rows_load(role=None):
    out=[]
    for m in read(U/'DATA_READY.json')['rows']:
        if role and m['role']!=role:continue
        assert sha(m['path'])==m['sha256'],m['clip_id']
        with np.load(m['path'],allow_pickle=False) as z:out.append(dict(m,**{k:z[k].copy() for k in z.files}))
    return out

def balanced_fine_loss(o,b):
    """Equal state, then clip, then observed hand contributions. Unknown has no loss.

    This is a new predeclared FIT-only objective, not a post-score threshold change.
    Finger/head structure and output readout stay unchanged.
    """
    logits=o['fingers'];targets=b['fy'];ce=F.cross_entropy(logits.reshape(-1,3),targets.reshape(-1),ignore_index=-1,reduction='none').reshape_as(targets)
    states=[]
    for state in range(3):
        clips=[]
        for i in range(len(targets)):
            hands=[]
            for side in (0,1):
                m=targets[i,:,side]==state
                if m.any():hands.append(ce[i,:,side][m].mean())
            if hands:clips.append(torch.stack(hands).mean())
        if clips:states.append(torch.stack(clips).mean())
    return torch.stack(states).mean() if states else logits.sum()*0
