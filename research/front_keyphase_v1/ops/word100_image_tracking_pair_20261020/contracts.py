import json
from pathlib import Path
import cv2,numpy as np
from tracker import optical_step,Tracker,PARAMS
cv2.setNumThreads(1)
rng=np.random.default_rng(31)
im=rng.integers(0,256,(160,240),dtype=np.uint8)
im=cv2.GaussianBlur(im,(3,3),0)
pts=cv2.goodFeaturesToTrack(im,32,.01,4,mask=None)
pts=pts[:,0];pts=pts[(pts[:,0]>25)&(pts[:,0]<210)&(pts[:,1]>25)&(pts[:,1]<130)]
shifted=cv2.warpAffine(im,np.float32([[1,0,3],[0,1,2]]),(240,160))
out,info=optical_step(im,shifted,pts,[110,80],80)
assert out is not None and np.linalg.norm(out['center']-[113,82])<.3
blank,reason=optical_step(im,np.zeros_like(im),pts,[110,80],80)
assert blank is None
t=Tracker();pose=np.zeros((33,2));mask=np.zeros(33,bool)
o=t.step(im,np.zeros((0,21,2)),[],pose,mask,0)
assert o['estimate']==[None,None] and o['observed_finger_points_added']==0
o=t.step(im,np.zeros((0,21,2)),[],pose,mask,1)
assert o['proposal']==[] and o['estimate']==[None,None]
try:t.step(im,np.zeros((0,21,2)),[],pose,mask,.5);assert False
except ValueError:pass
Path(__file__).with_name('CONTRACTS.json').write_text(json.dumps(dict(status='PASS',
    checks=['known_translation','blank_rejected','no_hand_no_creation','no_body_no_seed',
            'missing_fingers_never_created','nonmonotonic_time_rejected'],params=PARAMS),indent=2)+'\n',encoding='utf8')
print('CONTRACTS PASS')
