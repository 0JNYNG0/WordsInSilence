import cv2,json
import numpy as np
from pathlib import Path
from candidate import propose,decide,PARAMS
U=Path(__file__).resolve().parent
checks=[]
def check(name,p):
    assert p,name
    checks.append(name)
def scene(cross=False):
    im=np.full((260,300,3),35,np.uint8)
    cv2.rectangle(im,(105,0),(195,120),(100,100,100),-1)
    e=[[90,90],[210,90]];w=[[75,220],[225,220]] if not cross else [[210,220],[90,220]]
    for a,b in zip(e,w):cv2.line(im,tuple(a),tuple(b),(190,190,190),18)
    return im,e,w,120.
im,e,w,s=scene();a=propose(im,e,w,s,[1,0],[-1,-1])
check('separated_visible_tubes_support_anatomical_sides',a['evidence_side']==[0,1])
check('occupied_destination_does_not_force_swap',a['sides']==[1,0])
b=propose(im,e,[w[0]],s,[1],[-1]);check('single_wrong_hint_can_be_corrected',b['sides']==[0])
check('confirmed_side_never_overwritten',propose(im,e,[w[0]],s,[1],[1])['sides']==[1])
check('withheld_side_not_promoted',propose(im,e,[w[0]],s,[-1],[-1])['sides']==[-1])
rev=propose(im,e,w[::-1],s,[0,1],[-1,-1]);check('raw_order_independent',rev['evidence_side'][::-1]==a['evidence_side'])
check('no_wrist_no_synthetic_hand',propose(im,e,[None],s,[1],[-1])['evidence_side']==[-1])
check('no_elbows_abstain',propose(im,[None,None],w,s,[0,1],[-1,-1])['evidence_side']==[-1,-1])
check('blank_pixels_cannot_assign',propose(np.zeros_like(im),e,w,s,[1,0],[-1,-1])['evidence_side']==[-1,-1])
check('contrast_reversal_same_contour',propose(255-im,e,w,s,[0,1],[-1,-1])['evidence_side']==a['evidence_side'])
ee=[[299-x,y] for x,y in e];ww=[[299-x,y] for x,y in w]
check('screen_mirror_not_screen_half_identity',propose(im[:,::-1].copy(),ee,ww,s,[0,1],[-1,-1])['evidence_side']==a['evidence_side'])
dup=propose(im,e,[w[0],w[0]],s,[1,1],[-1,-1]);check('duplicate_hand_same_arm_not_forced',dup['sides']==[1,1] and dup['reason']==['multiple_candidates_for_one_arm']*2)
x,ee,ww,ss=scene(True);y=propose(x,ee,ww,ss,[0,1],[-1,-1]);check('true_crossing_no_wrong_support',all(q in [-1,i] for i,q in enumerate(y['evidence_side'])))
gap=im.copy();gap[145:170]=35
check('large_gap_not_filled',propose(gap,e,[w[0]],s,[1],[-1])['evidence_side']==[-1])
wide=np.full_like(im,35);cv2.rectangle(wide,(30,0),(270,259),(190,190,190),-1)
check('torso_wide_region_not_forearm',propose(wide,e,w,s,[0,1],[-1,-1])['evidence_side']==[-1,-1])
im.setflags(write=False);propose(im,e,w,s,[0,1],[-1,-1]);check('read_only_input_accepted',True)
(U/'CONTRACTS.json').write_text(json.dumps({'status':'PASS','count':len(checks),'checks':checks,'parameters':PARAMS},indent=2),encoding='utf8');print('CONTRACTS',len(checks),'PASS')
