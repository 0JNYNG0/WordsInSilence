import json
from pathlib import Path
import numpy as np,cv2
from candidate import center_transform,Tracker,PARAMS,PALM
cv2.setNumThreads(1)
checks=[]
p=np.array([[80,70],[90,65],[105,65],[120,70],[125,85],[110,95],[90,90],[78,82]],np.float32)
palm=np.array([[85,95],[80,65],[100,60],[125,65],[130,90]],np.float32);c=palm.mean(0).tolist()
m=cv2.getRotationMatrix2D((100,80),25,1.1);m[:,2]+=[3,4];q=p@m[:,:2].T+m[:,2]
r,l=center_transform(p,q,c,palm);assert r is not None and np.allclose(r['center'],np.array(c)@m[:,:2].T+m[:,2],atol=1e-4)
checks.append('known_rotation_scale_translation_center')
bad,bl=center_transform(np.array([[i,0] for i in range(8)],np.float32),np.array([[i+2,0] for i in range(8)],np.float32),[4,2],palm)
assert bad is None;checks.append('collinear_texture_not_anatomical_center')
r,l=center_transform(p,q,None,None);assert r is None;checks.append('no_center_from_region_only')
im=np.random.default_rng(31).integers(0,256,(180,240),dtype=np.uint8)
pose=np.zeros((33,2));usable=np.zeros(33,bool)
for idx,xy in {11:[50,20],12:[160,20],13:[60,50],15:[85,95],14:[170,50],16:[175,95]}.items():pose[idx]=xy;usable[idx]=True
hand=np.tile([100.,80.],(21,1));hand[PALM]=palm
t=Tracker();o=t.step(im,np.array([hand]),[0],pose,usable,0,[c,None])
assert o['center_used'][0]==c and o['seed'][0];checks.append('current_arm_observation_seed')
current=[c[0]+5,c[1]+3];hand2=hand+np.array([5,3]);pose[15]+=np.array([5,3])
o=t.step(im,np.array([hand2]),[0],pose,usable,1/30,[current,None])
assert o['center_used'][0]==current and o['source'][0]=='observed';checks.append('current_measurement_wins_over_history')
o=t.step(im,np.zeros((0,21,2)),[],pose,usable,1,[None,None])
assert o['center_estimate']==[None,None];checks.append('expired_track_not_invented')
o=t.step(im,np.array([hand2]),[0],pose,usable,1.1,[current,None])
assert o['recheck'][0]['gap_frames']==1 and o['center_used'][0]==current;checks.append('reappearance_current_reset')
t=Tracker();mask=np.zeros(33,bool)
o=t.step(im,np.array([hand]),[0],pose,mask,0,[c,None])
assert o['observed_center'][0]==c and not any(o['seed']);checks.append('uncertain_identity_preserves_observation_not_tracking_label')
o=t.step(im,np.zeros((0,21,2)),[],pose,mask,.03,[None,None])
assert o['center_used']==[None,None] and o['observed_finger_points_added']==0;checks.append('no_hand_no_shape_creation')
try:t.step(im,np.zeros((0,21,2)),[],pose,mask,.02,[None,None]);raise AssertionError()
except ValueError:checks.append('noncausal_time_rejected')
# Current geometry can reset anatomy without throwing away valid tracked pixels.
t=Tracker();o=t.step(im,np.array([hand]),[0],pose,usable,0,[c,None]);assert o['seed'][0]
original_features=cv2.goodFeaturesToTrack
try:
    cv2.goodFeaturesToTrack=lambda *args,**kwargs:None
    o=t.step(im,np.array([hand]),[0],pose,usable,1/30,[c,None])
    assert o['seed'][0] and o['seed_reason'][0]=='current_palm_revalidated_existing_pixels'
    assert o['center_used'][0]==c
finally:cv2.goodFeaturesToTrack=original_features
checks.append('preserve_valid_region_pixels_while_resetting_current_center')
Path(__file__).with_name('CONTRACTS.json').write_text(json.dumps(dict(status='PASS',checks=checks,params=PARAMS),indent=2)+'\n',encoding='utf8')
print('CONTRACTS PASS',len(checks))
