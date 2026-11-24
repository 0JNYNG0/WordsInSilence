"""Synthetic safety/nonregression tests, no real footage or inference."""
import numpy as np,json
from experiment import U,rule,roi,single_side,combine
tests=[]
def ck(name,x):assert x,name;tests.append(name)
pose=np.zeros((33,2));use=np.ones(33,bool)
pose[11]=[600,300];pose[12]=[1000,300];pose[13]=[700,500];pose[14]=[900,500]
# Anatomical left deliberately on screen-left and then right; no screen-half rule.
pose[15]=[1050,550];pose[16]=[750,550]
def hand(x,y):return np.array([[x+(k%4)*3,y+(k//4)*3] for k in range(21)],float)
def det(p,s=0):return {'xy':p,'valid':np.ones(21,bool),'score':.9,'box':[0,0,1920,1080],'requested_side':s}
a,b=hand(*pose[15]),hand(*pose[16])
xy,v,s,r=single_side([det(a),det(b)],pose,use,1920,1080)
ck('Crossing sides use anatomy, not left/right image half',s==[0,1] and v.all())
xy,v,s,r=single_side([det(a),det(a+1)],pose,use,1920,1080)
ck('Multiple same-arm detections withheld',not v.any())
bad=use.copy();bad[[15,16]]=False
ck('Missing arm wrists do not force handedness',not single_side([det(a)],pose,bad,1920,1080)[1].any())
q=rule.merge([det(a)],[det(a)],np.empty((0,21,2)),[],pose,use,1920,1080)
ck('Missing hand can be added with current arm and scales agreeing',q[0]['accepted'])
q=rule.merge([det(a)],[det(a+100)],np.empty((0,21,2)),[],pose,use,1920,1080)
ck('Different scales disagree, no addition',not q[0]['accepted'])
q=rule.merge([det(a)],[det(a)],np.array([a]),[0],pose,use,1920,1080)
ck('Existing hand retained, not duplicated',not q[0]['accepted'])
q=rule.merge([det(a)],[det(a)],np.array([a]),[1],pose,use,1920,1080)
ck('Contradictory existing identity not silently replaced',not q[0]['accepted'] and q[0]['reason']=='existing_identity_conflict')
q=rule.merge([det(a),det(a+1)],[det(a)],np.empty((0,21,2)),[],pose,use,1920,1080)
ck('Two new candidates cannot occupy same side',not any(t['accepted'] for t in q))
xx=np.array([[[[20.,30.],[20.,900.]]]]);vv=np.ones(xx.shape[:-1],bool)
cut,valid=roi(xx,vv,[0,0,1920,800])
ck('ROI applied after detection and no coordinates rescaled',np.array_equal(cut[valid],xx[valid]) and valid.sum()==1)
ck('Outside ROI becomes missing, not zero observation',np.isnan(cut[~valid]).all())
z={'native_xy_px':np.zeros((1,75,2),np.float32),'native_usable':np.zeros((1,75),bool),'native_observed':np.zeros((1,75),bool),
   'raw_raw_count':np.array([1]),'raw_hand_xy_px':np.array([[b]]),'assignment_sides':np.array([[1,-1]]),'width':np.array([1920]),'height':np.array([1080]),'raw_pose_xy_px':pose[None],'raw_pose_usable':use[None]}
z['native_xy_px'][0,54:]=b;z['native_usable'][0,54:]=True;z['native_observed'][0,54:]=True
out,valid,q=combine([det(a)],[det(a)],z,0)
ck('Supplement preserves every existing opposite-hand point',np.array_equal(out[1],b.astype(np.float32)) and valid[1].all())
(U/'CONTRACT_TESTS.json').write_text(json.dumps({'status':'PASS','checks':tests,'count':len(tests),'training':0},indent=2),encoding='utf8')
print('PASS',len(tests))
