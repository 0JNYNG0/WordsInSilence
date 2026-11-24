"""Behavioral edge cases for missing scales, partial hands and coordinate transforms."""
import json
from dataclasses import replace
import numpy as np
from input_candidate import U, O, Observation, three_conditions, evidence, angles

f=json.loads((U/"FROZEN.json").read_text("utf8"))
source=Observation.load(f["rows"][0]["old_masked"])
def copy(o):
    return replace(o,**{k:getattr(o,k).copy() for k in
                       ["xy_px","timestamp_us","frame_id","observed","usable","conf","hand_score"]},
                   sidecar=dict(o.sidecar))
def clear(o,indices):
    o.xy_px[:,indices]=np.nan;o.observed[:,indices]=False;o.usable[:,indices]=False;o.conf[:,indices]=np.nan
tests=[]
# A missing per-clip scale must not borrow merged estimates or the other hand.
a=copy(source);clear(a,slice(33,54))
ch,_,_=three_conditions(a,source)
assert not ch["merge_fixed_scale"]["x"][:,102:144].any()
assert not ch["merge_fixed_scale"]["x"][:,237:258].any()
assert ch["merge_recomputed_scale"]["x"][:,237:258].any()
tests.append("missing_baseline_scale_stays_missing_in_fixed_condition")
# Palm location is available without declaring all finger bends available.
p=copy(source);keep={0,5,9,13,17};clear(p,[33+j for j in range(21) if j not in keep])
rec=dict(frames=[dict(decisions=[],core=False) for _ in range(p.n_frames)])
e=evidence(p,p,rec)
assert e["palm_available"][:,0].any() and not e["bend_available"][:,0].any()
tests.append("palm_position_does_not_imply_finger_bends")
# Distal bends may be computable without wrist; local XY remains masked.
p=copy(source);clear(p,[33]);e=evidence(p,p,rec);ch,_,_=three_conditions(p,p)
assert not e["palm_available"][:,0].any() and not e["wrist_available"][:,0].any()
assert e["bend_available"][:,0].any()
assert not ch["full_before"]["x"][:,237:258].any()
tests.append("wrist_missing_keeps_distal_evidence_separate_from_local_coordinates")
# Translation and scale leave geometrical angles invariant.
xy=source.xy_px[:,33:].reshape(-1,2,21,2);valid=source.usable[:,33:].reshape(-1,2,21)
d,m=angles(xy,valid);d2,m2=angles(xy*2,valid)
assert np.array_equal(m,m2) and np.allclose(d[m],d2[m],atol=.1)
tests.append("uniform_hand_scale_does_not_fix_angle_disagreement")
# A zero-length bone is not a valid 0-degree bend.
xy=np.zeros((1,21,2),np.float32);v=np.ones((1,21),bool)
d,m=angles(xy,v);assert not m.any() and np.isnan(d).all()
tests.append("collapsed_bones_are_unavailable_not_perfect_angles")
# Added evidence is never injected into the 280 model channels as a confidence score.
ch,_,_=three_conditions(source,source)
assert all(z["x"].shape[1]==280 for z in ch.values())
assert np.array_equal(ch["full_before"]["x"],ch["merge_fixed_scale"]["x"])
assert np.array_equal(ch["full_before"]["x"],ch["merge_recomputed_scale"]["x"])
tests.append("no_addition_all_three_conditions_equal")
with (U/"CONTRACT_CHECKS.json").open("x",encoding="utf8") as fp:
    json.dump(dict(status="PASS",count=len(tests),tests=tests),fp,ensure_ascii=False,indent=2)
print("PASS",len(tests))
