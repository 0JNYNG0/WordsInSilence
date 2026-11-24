from common import *
from candidate import trigger,supplement
import copy
pose=np.zeros((33,2));pose[11]=[150,40];pose[12]=[250,40];pose[13]=[140,70];pose[14]=[260,70];pose[15]=[140,100];pose[16]=[260,100]
usable=np.ones(33,bool)
def hand(x):
    p=np.array([[x,100]]+[[x+(i%4)*2,100+(i//4+1)*3] for i in range(20)],float)
    return p
base=np.full((2,21,2),np.nan);valid=np.zeros((2,21),bool);base[0]=hand(140);valid[0]=True
def det(p,crop=False):
    d={'xy':p,'valid':np.isfinite(p).all(-1)}
    if crop:d.update(requested_side=1,box=[200,40,320,180])
    return d
full=[det(hand(260))];crop=[det(hand(260),True)]
def calc(ff=full,cc=crop,bb=base,vv=valid,pp=pose):return supplement(ff,cc,bb,vv,[base[0]],[0],pp,usable,400,300,{1:[200,40,320,180]})
tests=[]
def check(name,truth):assert truth,name;tests.append(name)
sp,sv,pp,pv,dec=calc();check('strict_full_hand_supported',sv[1].all() and not dec[0]['position_only'])
c=copy.deepcopy(crop);c[0]['xy'][17:]+=20
sp,sv,pp,pv,dec=calc(cc=c);check('shape_disagreement_wrist_only',not sv[1].any() and pv[1,0] and pv[1].sum()==1)
check('no_invented_angles',not pv[1,1:].any());check('normal_hand_exact',np.array_equal(pp[0],base[0]))
c=copy.deepcopy(crop);c[0]['xy']+=20
check('wrist_disagreement_withheld',not calc(cc=c)[3][1].any())
c=copy.deepcopy(crop);c[0]['xy'][17:]+=20
check('duplicate_crop_hold',not calc(cc=c+c)[3][1].any())
check('duplicate_full_hold',not calc(ff=full+full)[3][1].any())
check('missing_crop_hold',not calc(cc=[])[3][1].any())
b=base.copy();v=valid.copy();b[1]=hand(260);v[1]=True
check('existing_m1_never_replaced',np.array_equal(calc(bb=b,vv=v)[2],b))
p=pose.copy();p[16]=[260,250]
check('rest_outside_roi_no_call',not trigger(base,valid,p,usable,400,300,[0,0,400,200]))
v=np.ones((2,21),bool);check('both_present_no_call',not trigger(b,v,pose,usable,400,300,[0,0,400,200]))
# Normal crossing uses anatomical arms, not screen halves.
p=pose.copy();p[[13,15]]=[(260,70),(260,100)];p[[14,16]]=[(140,70),(140,100)]
check('crossing_anatomical_assignment',rule.arm(hand(140),p,usable,100)==1 and rule.arm(hand(260),p,usable,100)==0)
f=read(U/'FROZEN.json');frames=0
for r in f['old_rows']+f['new_rows']:
    z=load(r['arrays']);rr=read(r['source_assets']['detections'])['frames']
    for i in range(r['frames']):native_m1(z,i,rr[i]);frames+=1
write(U/'CONTRACT_CHECK.json',{'status':'PASS','tests':tests,'native_before_roi_reconstructed_frames':frames,'note':'Same Codex implementation checks, not independent expert validation'})
print('PASS',len(tests),frames)
