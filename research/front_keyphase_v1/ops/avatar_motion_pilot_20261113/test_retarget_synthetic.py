from pathlib import Path
import json
import numpy as np
from retarget_math import body_frame,map_to_rig,solve_arm

U=Path(__file__).resolve().parent
# Artificial body fixture: explicitly not a person or a sign recording.
body=np.zeros((25,3));body[2]=[-.2,1,0];body[5]=[.2,1,0];body[1]=[0,1,0];body[8]=[0,.5,0]
p=np.array([[.1,.8,.2],[-.2,.7,.3]])
origin=np.array([0,106.,0]);basis=np.eye(3);width=25.
base=map_to_rig(p,body,origin,basis,width)
theta=.4;R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
changed=map_to_rig((p@R.T)*4+[20,-8,6],(body@R.T)*4+[20,-8,6],origin,basis,width)
assert np.allclose(base,changed,atol=1e-10)
rng=np.random.default_rng(11);max_length_error=0;reachable=0
for i in range(100):
    shoulder=np.zeros(3);target=rng.uniform([.1,-.3,.05],[.5,.3,.3]);hint=np.array([0.,.2,-.3])
    result=solve_arm(shoulder,target,hint,.35,.3)
    error=max(abs(np.linalg.norm(result['elbow'])-.35),abs(np.linalg.norm(result['wrist']-result['elbow'])-.3))
    max_length_error=max(max_length_error,error)
    if not result['unreachable']:
        reachable+=1;assert result['target_reach_error']<1e-9
outside=solve_arm(np.zeros(3),np.array([2.,0,0]),np.array([0,1,0]),.35,.3)
assert outside['unreachable']and outside['target_reach_error']>1
out=dict(status='PASS_SYNTHETIC_RETARGET_MATH_ONLY',body_translation_scale_rotation_invariant=True,
         synthetic_arm_cases=100,reachable_cases=reachable,max_fixed_length_error=max_length_error,
         unreachable_target_explicit=True,real_motion_retargeted=False,skin_binding_validated=False,
         note='Geometry plumbing tests only. No source3D acceptance, real sign motion, or avatar accuracy implied.')
(U/'SYNTHETIC_RETARGET_VERIFIED.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps(out,ensure_ascii=False))
