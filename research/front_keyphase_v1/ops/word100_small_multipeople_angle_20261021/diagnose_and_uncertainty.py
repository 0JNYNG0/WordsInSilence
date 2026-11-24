"""Describe remaining blockers without changing any rules or scoring thresholds."""
from study import U,read,load,write,adapter
from collections import Counter
import copy
import numpy as np

rows=read(U/"PREPARED_INDEX.json"); diagnostics=[]; count=Counter(); examples=[]
for row in rows:
    cid=row["clip_id"];q=load(row["prepared"]);result=read(U/"queries"/(cid+".json"))
    quality=load(row["quality"]);frames=read(row["detections"])["frames"]
    required=result["required"];hs=[("L","R").index(s) for s in required]
    reasons=Counter(); ambiguous=[]; lowered=[]
    raw=load(row["raw"])
    for i in np.flatnonzero(q["core"]):
        for h in hs:
            states=q["angle_state"][i,h]
            if np.any(states==2):
                ds=frames[i]["decisions"]
                confirms=[d for d in ds if d["reason"]=="confirm_existing_hand" and d["side"]==h]
                adds=[d for d in ds if d["accepted"] and d["side"]==h]
                if quality["pair_available"][i,h]:
                    reason="pair_exists_but_local_angle_unavailable"
                elif confirms or adds:
                    reason="paired_candidate_lacks_valid_anchor_set"
                elif any(d["reason"]=="current_arm_uncertain" for d in ds):
                    reason="no_side_correspondence_with_current_arm_uncertain_candidate"
                    ambiguous.append(dict(index=int(i),frame_id=int(q["frame_id"][i]),side=h))
                elif not ds:
                    reason="fresh_detector_found_no_hand"
                else:
                    reason="no_accepted_same_side_reconfirmation"
                reasons[reason]+=int((states==2).sum())
            # Diagnostic only: wrist outside fixed ROI in the original native pose,
            # not a semantic claim that this hand was optional.
            wrist=raw["pose_xy_px"][i,15+h]
            if np.isfinite(wrist).all() and wrist[1]>=row["box"][3]:
                lowered.append(dict(index=int(i),frame_id=int(q["frame_id"][i]),side=h))
    diagnostics.append(dict(clip_id=cid,word_code=row["word_code"],label=row["label"],
        unassessed_angle_frame_reasons=dict(reasons),arm_uncertain_examples=ambiguous,
        pose_wrist_below_roi_in_annotation=lowered))
    originals=[];refs=[];people=[]
    for pid in result["pairs"]:
        p=load(U/"pairs"/(pid+".npz"));m=read(U/"pairs"/(pid+".json"));p["rows"]=m["rows"]
        originals.append(p);refs.append(load(U/"prepared"/(m["reference"]+".npz")))
        people.append(int(m["reference"].split("_REAL")[1][:2]))
    base=adapter.aggregate(q,originals,required,people)
    # Choose a currently assessed angle, preferably one with five reference votes.
    choices=[(int(base["votes"][i,h,a]),int(i),h,a)
             for i in np.flatnonzero(q["core"]) for h in hs for a in range(15)
             if q["angle_state"][i,h,a]==4]
    assert choices,cid
    votes,i,h,a=max(choices)
    modified=copy.deepcopy(q);modified["angle_state"][i,h,a]=3
    pp=[adapter.pair(modified,r,p["path"],required,p["rows"][0]["alignment_ok"])
        for r,p in zip(refs,originals)]
    ag=adapter.aggregate(modified,pp,required,people)
    assert np.array_equal(modified["xy_px"],q["xy_px"],equal_nan=True)
    assert np.array_equal(modified["angle_deg"],q["angle_deg"],equal_nan=True)
    assert ag["votes"][i,h,a]==0
    for k in range(30):
        if k!=h*15+a:assert ag["rows"][k]==base["rows"][k]
    assert any(x["first_row"]<=i<=x["last_row"] for x in ag["rows"][h*15+a]["reasons"]["query_conflict"])
    count["single_angle_uncertainty_no_neighbor_loss"]+=1
    count["had_five_refs_before_uncertainty"]+=int(votes>=5)
    # If one reference loses an entire hand, expected range still remains.
    changed=copy.deepcopy(refs[0]);changed["angle_state"][:,h]=0;changed["angle_deg"][:,h]=np.nan
    p=adapter.pair(q,changed,originals[0]["path"],required,originals[0]["rows"][0]["alignment_ok"])
    assert np.array_equal(p["expected"],originals[0]["expected"])
    assert not p["strict"][:,h].any()
    count["reference_missing_keeps_expected_range"]+=1
    examples.append(dict(clip_id=cid,index=i,frame_id=int(q["frame_id"][i]),side=h,angle=a,
                         before_reference_votes=votes,after_reference_votes=0))

write(U/"DIAGNOSIS.json",dict(rows=diagnostics,
    interpretation="Reconfirmation provenance failures are not action errors; current_arm_uncertain may coexist with other causes. Pose wrist below ROI is a geometry flag, not linguistic optional-hand annotation."))
write(U/"UNCERTAINTY_CONTROLS.json",dict(status="PASS",counts=dict(count),examples=examples,
    limitations="Frozen path and angle metadata interventions, not new RGB occlusion experiments or classification performance."))
print(dict(count))
