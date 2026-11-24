"""Summarize frozen diagnostic artifacts, without rerunning candidate/inference."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent;Q=U.parent/"word100_merge_scale_input_20261020"
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
c=Counter();rows=[]
for r in read(U/"FROZEN.json")["rows"]:
    cid=r["clip_id"];e=load(U/"evidence"/(cid+".npz"))
    old=load(Q/"inputs"/cid/"full_before.npz")["x"]
    merged=load(Q/"inputs"/cid/"merge_fixed_scale.npz")["x"]
    safe=load(U/"inputs"/(cid+".npz"))["x"]
    prior=load(Q/"evidence"/(cid+".npz"))
    added=e["source_added"].any(-1);paired=added&e["pair_available"]
    flags=added&e["finger_uncertain"].any(-1)
    core=prior["core"][:,None]
    counts=dict(new_hand_frames=int(added.sum()),
        new_paired_without_flags=int((paired&~flags).sum()),
        new_paired_with_flags=int(flags.sum()),
        new_unpaired=int((added&~e["pair_available"]).sum()),
        core_new_hands=int((added&core).sum()),
        core_new_flagged_hands=int((flags&core).sum()),
        core_new_unpaired_hands=int((added&core&~e["pair_available"]).sum()),
        added_body_points_before=int((merged[:,186:237]-old[:,186:237]).sum()),
        added_body_points_after=int((safe[:,186:237]-old[:,186:237]).sum()),
        added_local_points_before=int((merged[:,237:279]-old[:,237:279]).sum()),
        added_local_points_after=int((safe[:,237:279]-old[:,237:279]).sum()),
        new_angles_pair_comparable=int((e["angle_available"]&added[...,None]).sum()),
        new_angles_indeterminate=int((e["angle_unresolved"]&added[...,None]).sum()),
        new_fingers_with_missing_coordinates=int((e["finger_missing"]&added[...,None]).sum()),
        existing_observations_flagged_diagnostic_only=int((e["finger_uncertain"]&~added[...,None]).sum()))
    c.update(counts);rows.append(dict(clip_id=cid,**counts))
v=read(U/"VISUAL_TARGETS.json")
controls={}
for group in ("기존 정상","기존 교차"):
    a=[x for x in v["controls"] if x["group"]==group]
    controls[group]=dict(frames=len(a),frames_with_shape_flags=sum(x["uncertain_fingers"]>0 for x in a),
                        frames_with_new_shape_hold=sum(x["new_shape_hold"]>0 for x in a))
result=dict(counts=dict(c),controls=controls,rows=rows,
    interpretation="Shape uncertainty is evidence quality, not wrong signing. Correct side assignment is not a precise finger ground truth.",
    optional_adapter="Existing observations unchanged; fixed scale; only added ambiguous/unassessed shape held. Position anchors retained.",
    rollout="Research files only, not model or DTW/web. No training or model evaluation.")
with (U/"EFFECT_AUDIT.json").open("x",encoding="utf8") as fp:json.dump(result,fp,ensure_ascii=False,indent=2)
print(json.dumps(dict(counts=dict(c),controls=controls),ensure_ascii=False))
