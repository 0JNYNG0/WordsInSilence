"""Only CROSS-SIDE identity conflicts quarantine anchors; a same-side match confirms an existing hand."""
import numpy as np
from identity_v2 import PARAMS,build_anchors,get_tracks,assess,choose
def conflict_mask(records,clip_index,shape):
    held=np.zeros(shape,bool)
    for r in records:
        if r['clip_index']!=clip_index:continue
        for p in r['pairs']:
            if p['hold_new'] and p['existing_side']!=r['side']:
                held[r['frame'],r['side']]=True
                held[r['frame'],p['existing_side']]=True
    return held
