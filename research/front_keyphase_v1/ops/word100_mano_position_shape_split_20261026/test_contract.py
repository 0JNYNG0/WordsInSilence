from pathlib import Path
import json,numpy as np
from candidate import assess,seed_points,nearest_anchors
U=Path(__file__).resolve().parent
hand=np.array([[50+(i%5)*3,50+(i//5)*3] for i in range(21)],float);pose=np.zeros((33,2));pose[11]=[20,20];pose[12]=[120,20];pose[15]=hand[0];pose[16]=[150,50];pv=np.ones(33,bool)
def tr(a):return dict(anchor=a,target=5,initial=np.ones((8,2)),xy=np.tile([55.,55.],(8,1)),valid=np.ones(8,bool),qualified=True,reason='tracked_region_not_joint')
c=[{'pass':False},{'pass':False}];tracks={0:[tr(4),tr(6)],1:[]};a=assess(hand,pose,pv,100,tracks,c);assert a['identity_proposal']==0 and not a['finger_shape_confirmed'] and not a['precise_palm_center_confirmed']
b=assess(hand,pose,pv,100,{0:[tr(4),tr(6)],1:[tr(4),tr(6)]},c);assert b['identity_proposal']==-1
d=assess(hand,pose,pv,100,tracks,[{'pass':False},{'pass':True}]);assert d['identity_proposal']==-1
assert assess(hand,pose,pv,100,{0:[tr(4)],1:[]},c)['identity_proposal']==-1
assert len(seed_points(np.zeros((100,100),np.uint8),hand,None,1))==0
assert len(seed_points(np.random.default_rng(1).integers(0,256,(100,100),dtype=np.uint8),hand,hand,1))==0
anchors=np.zeros((8,2),bool);anchors[2,0]=True;anchors[4,0]=True;anchors[6,0]=True;assert nearest_anchors(5,anchors,np.arange(8)/30)[0]==[4,6]
out=dict(status='PASS',tests=['position support never confirms fingers','competing regions hold identity','contour disagreement holds identity','single-sided flow holds identity','blank image seeds no hand','shared hand projection excluded from seed','nearest eligible anchors only'],count=7)
(U/'CONTRACT_TESTS.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(out)
