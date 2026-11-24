import json
from pathlib import Path
import numpy as np
from candidate import angle_evidence,compare_aligned_angles,TRIPLES
U=Path(__file__).resolve().parent
def base():
    x=np.array([[j,j*j/23] for j in range(21)],float)[None,None]
    v=np.ones((1,1,21),bool);p=np.ones((1,1),bool)
    a=np.ones((1,1,15),bool);u=np.zeros_like(a)
    return x,v,p,a,u
passed=[]
def check(name,fun):fun();passed.append(name)
x,v,p,a,u=base();saved=x.copy();r=angle_evidence(x,v,p,a,u)
check("input_coordinates_unchanged",lambda:np.testing.assert_array_equal(x,saved))
check("all_computable_open_chain",lambda:np.testing.assert_equal(r["compare_eligible"].sum(),15))
for k in range(15):
    z=u.copy();z[...,k]=True;s=angle_evidence(x,v,p,a,z)
    assert s["compare_eligible"].sum()==14 and s["angle_state"][0,0,k]==3
    assert np.array_equal(s["angle_deg"],r["angle_deg"])
passed.append("each_of_15_flags_only_changes_its_own_angle")
w=v.copy();w[...,8]=False;s=angle_evidence(x,w,p,a,u)
assert np.flatnonzero(~s["computable"][0,0]).tolist()==[5]
passed.append("missing_index_tip_does_not_remove_other_fingers_or_unrelated_index_angles")
s=angle_evidence(x,v,np.zeros_like(p),a,u);assert s["computable"].all() and not s["compare_eligible"].any()
passed.append("no_second_observation_is_unassessed_not_missing_or_wrong")
s=angle_evidence(x,v,p,np.zeros_like(a),u);assert np.all(s["angle_state"]==2)
passed.append("no_angle_evidence_is_unassessed")
y=x.copy();y[...,2,:]=y[...,1,:];s=angle_evidence(y,v,p,a,u)
assert np.flatnonzero(s["angle_state"][0,0]==1).tolist()==[0,1]
passed.append("zero_shared_bone_only_degenerates_its_actual_dependent_angles")
y=x.copy();y[...,20,:]=np.nan;s=angle_evidence(y,v,p,a,u)
assert not s["computable"][0,0,14] and s["computable"].sum()==14
passed.append("nonfinite_coordinate_never_compared")
theta=.71;rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
s=angle_evidence(x@rot*2.3+[2,-8],v,p,a,u)
assert np.allclose(r["angle_deg"],s["angle_deg"],atol=1e-10)
passed.append("rotation_translation_uniform_scale_invariance")
q=angle_evidence(x,v,p,a,np.ones_like(u));d=compare_aligned_angles(q,r,np.ones_like(a))
assert d["mean_difference_deg"] is None and np.isnan(d["difference_deg"]).all() and d["coverage"]==0
passed.append("insufficient_evidence_is_not_zero_difference")
z=u.copy();z[...,6]=True;q=angle_evidence(x,v,p,a,z)
d=compare_aligned_angles(q,r,np.ones_like(a))
assert d["expected_count"]==15 and d["compared_count"]==14 and d["coverage"]==14/15
passed.append("missing_query_does_not_shrink_expected_reference")
assert d["mean_difference_deg"]==0 and np.isnan(d["difference_deg"][0,0,6])
passed.append("known_match_separate_from_unknown_slot")
with (U/"CONTRACTS.json").open("x",encoding="utf8") as fp:
    json.dump(dict(status="PASS",tests=passed,count=len(passed)),fp,indent=2)
print("PASS",len(passed))
