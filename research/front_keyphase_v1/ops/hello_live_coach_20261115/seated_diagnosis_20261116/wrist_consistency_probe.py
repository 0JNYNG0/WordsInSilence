"""Offline public ablation: controller and action model must measure the same wrist.

No production change, model fitting, threshold selection, personal input or camera.
"""
from pathlib import Path
import sys,json
import numpy as np
V=Path(__file__).resolve().parent;U=V.parent;sys.path.insert(0,str(U))
from action_runtime import ActionObserver
from coach import Coach
from public_probe import run_case,write

class ConsistentWristAction(ActionObserver):
    def update(self,xy,score,ts,width,height):
        p=np.asarray(xy,np.float32).copy();s=np.asarray(score,np.float32).copy()
        scale=np.linalg.norm(p[5]-p[6])
        for hand,body in [(91,9),(112,10)]:
            q=p[[hand,body]]
            valid=np.isfinite(q).all() and (q[:,0]>=0).all() and (q[:,0]<width).all() and (q[:,1]>=0).all() and (q[:,1]<height).all()
            if valid and np.isfinite(scale) and scale>0 and np.linalg.norm(p[hand]-p[body])<=.28*scale:
                p[body]=p[hand];s[body]=s[hand]
        return super().update(p,s,ts,width,height)

if __name__=='__main__':
    out=V/'WRIST_CONSISTENCY_PROBE.json'; assert not out.exists()
    write(V/'WRIST_PROBE_PLAN.json',dict(scope='Offline mechanism ablation only',reason='Controller uses hand wrist133 slots91/112 while motion features use body wrist9/10; compare measured channels without weakening shape/direction gates.',private_data=False,training=False,thresholds_changed=False))
    m=Coach(action=ConsistentWristAction());rows=[]
    for name,path in [('native',U/'source/reference.npz'),('letterbox',V/'public_video_640_letterbox.npz'),('upper_body',V/'public_video_upper_body_640.npz')]:
        with np.load(path) as z:a={k:z[k].copy()for k in z.files}
        rows.append(run_case(m,'wrist_'+name,a))
    write(out,dict(rows=rows,production_applied=False,training=False,private_data=False))
