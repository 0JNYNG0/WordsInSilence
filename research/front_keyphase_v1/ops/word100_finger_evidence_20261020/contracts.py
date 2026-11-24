"""Behavioral properties; no public/user sample is used to choose a threshold."""
import json
import sys
import numpy as np
from pathlib import Path
from candidate import compare_shape, safe_new_shape_channels, PALM

U=Path(__file__).resolve().parent
tests=[]
hand=np.array([[0,0],[-.35,.25],[-.65,.5],[-.9,.75],[-1.1,1.],
               [-.5,1],[-.5,1.6],[-.5,2.2],[-.5,2.8],
               [0,1.2],[0,1.9],[0,2.6],[0,3.3],
               [.5,1],[.5,1.65],[.5,2.3],[.5,2.95],
               [.9,.8],[.9,1.3],[.9,1.8],[.9,2.3]],float)*50+300
valid=np.ones(21,bool)
r=compare_shape(hand,valid,hand,valid)
assert r["pair_available"] and not r["point_uncertain"].any()
tests.append("identical_observations_preserved")
for angle in (0.3,1.5,3.):
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    b=(hand-[300,300])@rot*1.8+[920,770]
    r=compare_shape(hand,valid,b,valid)
    assert r["pair_available"] and not r["point_uncertain"].any()
    assert np.max(r["residual_px"])<1e-8
tests.append("translation_rotation_uniform_size_do_not_flag_shape")
r=compare_shape(hand,valid)
assert not r["pair_available"] and not r["point_uncertain"].any()
tests.append("single_observation_is_unassessed_not_wrong")
missing=valid.copy();missing[PALM[2]]=False
r=compare_shape(hand,valid,hand,missing)
assert not r["pair_available"]
tests.append("missing_palm_prevents_unsupported_alignment")
missing=valid.copy();missing[8]=False
r=compare_shape(hand,valid,hand,missing)
assert r["pair_available"] and r["finger_missing"][1] and not r["point_uncertain"].any()
tests.append("one_missing_tip_does_not_discard_other_fingers")
b=hand.copy();b[8]=2*hand[7]-hand[8]
r=compare_shape(hand,valid,b,valid)
assert r["finger_uncertain"][1] and r["point_uncertain"][8]
assert not r["finger_uncertain"][[0,2,3,4]].any()
tests.append("reversed_distal_bone_marks_affected_finger_only")
b=hand.copy();b[8]=b[7]
r=compare_shape(hand,valid,b,valid)
assert r["point_uncertain"][8]
tests.append("collapsed_bone_not_precise_shape")
# Whole hand can be a well-observed fist; no prescribed open/closed-hand angle.
b=hand.copy();b[7]=b[6]+[4,3];b[8]=b[7]+[-3,-2]
r=compare_shape(b,valid,b,valid)
assert not r["point_uncertain"].any()
tests.append("matching_folded_finger_not_rejected_by_angle")
# Position retention plus consistent distal-coordinate masking.
x=np.ones((1,280),np.float32);q=np.zeros((1,2,21),bool);added=np.ones_like(q)
q[0,0,[5,6,7,8]]=True
y=safe_new_shape_channels(x,q,added)
assert np.array_equal(x[:,:18],y[:,:18]) and y[0,2*(9+5)]==1
assert y[0,102+2*5]==0 and y[0,237+5]==0
assert y[0,2*(9+8)]==0 and y[0,186+9+8]==0
assert np.all(x==1)
tests.append("keep_palm_position_hold_uncertain_distal_both_paths")
y=safe_new_shape_channels(x,q,np.zeros_like(q))
assert np.array_equal(x,y)
tests.append("baseline_observations_not_silently_filtered")
with (U/(sys.argv[1] if len(sys.argv)>1 else "CONTRACTS.json")).open("x",encoding="utf8") as fp:
    json.dump(dict(status="PASS",count=len(tests),tests=tests),fp,ensure_ascii=False,indent=2)
print("PASS",len(tests))
