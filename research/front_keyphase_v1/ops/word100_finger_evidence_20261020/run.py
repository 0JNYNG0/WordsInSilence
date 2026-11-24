import hashlib
import json
import os
import shutil
import sys
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from candidate import compare_shape, safe_new_shape_channels, PALM, EPS

U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
P=O/"word100_fullframe_hand_merge_20261020"
Q=O/"word100_merge_scale_input_20261020"
def read(p):return json.loads(Path(p).read_text("utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,v):
    with Path(p).open("x",encoding="utf8") as fp:json.dump(v,fp,ensure_ascii=False,indent=2,allow_nan=False)
def inside(x,b):
    return np.isfinite(x).all(-1)&(x[...,0]>=b[0])&(x[...,0]<b[2])&(x[...,1]>=b[1])&(x[...,1]<b[3])


def pair_for(frame,side,old_present,position,mask,box):
    """Only unambiguous SAME-side correspondence. No new identity decisions."""
    if not mask.any():return None,None,0
    if not old_present:
        qs=[d for d in frame["decisions"] if d["accepted"] and d["side"]==side]
        assert len(qs)<=1
        if len(qs)==1:
            q=qs[0];b=np.array(frame["crop"][q["crop_index"]]["xy"])
            return b,mask&inside(b,box),2
        return None,None,0
    qs=[d for d in frame["decisions"] if d["reason"]=="confirm_existing_hand" and d["side"]==side]
    if len(qs)!=1:return None,None,0
    q=qs[0]
    assert q["matched_old"] and all(frame["baseline_sides"][j]==side for j in q["matched_old"])
    b=np.array(frame["full"][q["index"]]["xy"])
    return b,inside(b,box),1


def freeze():
    assert read(U/"CONTRACTS_V2.json")["status"]=="PASS"
    old=read(Q/"FROZEN.json");pins=dict(old["pins"])
    for d in read(Q/"INDEX.json"):
        pins.update(d["outputs"])
    for p in [U/"candidate.py",U/"contracts.py",U/"CONTRACTS_V2.json",U/"run.py",U/"verify.py",
              Q/"FINAL_AUDIT.json",Q/"VISUAL_REVIEW.json",Q/"ANGLE_REVIEW_QUEUE.json",
              P/"VISUAL_REVIEW.json"]:
        pins[str(p)]=sha(p)
    for p,h in {**pins,**old["protected"]}.items():assert sha(p)==h,p
    assert not (Q/"RUNNING.json").exists() and not (P/"RUNNING.json").exists()
    (U/"before").mkdir()
    for p in [ROOT/"WordsInSilence_PROJECT_CONTEXT.md",ROOT/"NEW_CHAT_HANDOFF.md",O/"STATE.json"]:
        shutil.copyfile(p,U/"before"/p.name)
    write(U/"FROZEN.json",dict(
        at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        authorization="User approved per-finger uncertainty candidate, position and normal/crossing preservation, public validation. No training until review.",
        rows=old["rows"],pins=pins,protected=old["protected"],scope="All36 completed TRAIN clips/23words/9people/4641frames; no detector rerun or1199 preparation claim.",
        geometry="Proper palm similarity alignment; midpoint discs with radius half between-prediction residual; unresolved bone when endpoint discs touch/overlap or length degenerates. Angle also unresolved if envelope covers0..180. No fitted angle threshold.",
        eps=EPS,threshold_selection="EPS arithmetic only. Radius ratio1 follows disc intersection. No labels or public result based search. 10synthetic tests before public run.",
        model_adapter="Research optional fixed-scale input only. Old observations untouched. New ambiguous OR unassessed finger coordinates held. Body wrist+MCP anchors preserved as location evidence; distal points held in both body/local blocks. No production model/input edit.",
        meaning="Flags express disagreement/resolution of hypotheses, not biological wrongness or calibrated accuracy. Both predictions can agree and be wrong.",
        gates=["Raw coordinates/frame times/ROI/palm position identical","Old valid model channels unchanged",
               "3normal/53crossing saved controls unchanged","Independent geometry and saved-input verification",
               "View largest prior12disagreements plus deterministic diverse newly held public examples",
               "No hidden-coordinate fill, no word-specific exception, no new source data or training"],
    ))


def run():
    f=read(U/"FROZEN.json")
    for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
    write(U/"RUNNING.json",dict(pid=os.getpid()))
    (U/"evidence").mkdir();(U/"inputs").mkdir()
    total=Counter();index=[];perclip=[];held_review=[]
    fields=["comparable","point_uncertain","bone_available","bone_unresolved","bone_discrepancy_ratio",
            "angle_available","angle_unresolved","angle_envelope_low_deg","angle_envelope_high_deg",
            "finger_available_count","finger_uncertain","finger_missing","residual_px"]
    prior=read(Q/"ANGLE_REVIEW_QUEUE.json")[:12]
    prior_results=[]
    for r in f["rows"]:
        cid=r["clip_id"];obs=load(r["merged"]);old=load(r["old_masked"])
        saved=read(r["record"]);previous=load(Q/"evidence"/(cid+".npz"))
        x=load(Q/"inputs"/cid/"merge_fixed_scale.npz")
        xy=obs["xy_px"][:,33:].reshape(-1,2,21,2)
        mask=obs["usable"][:,33:].reshape(-1,2,21)
        oldmask=old["usable"][:,33:].reshape(-1,2,21)
        t=len(xy);arrays={};kind=np.zeros((t,2),np.uint8);available=np.zeros((t,2),bool)
        pair_b=np.full((t,2,21,2),np.nan,np.float64);pair_v=np.zeros((t,2,21),bool)
        aligned=np.full_like(pair_b,np.nan);scale=np.full((t,2),np.nan)
        for i in range(t):
            for s in range(2):
                b,vb,k=pair_for(saved["frames"][i],s,oldmask[i,s].any(),xy[i,s],mask[i,s],r["box"])
                result=compare_shape(xy[i,s],mask[i,s],b,vb)
                if not arrays:
                    for field in fields:
                        sample=result[field]
                        fill=False if sample.dtype==bool else (0 if np.issubdtype(sample.dtype,np.integer) else np.nan)
                        arrays[field]=np.full((t,2)+sample.shape,fill,dtype=sample.dtype)
                for field in fields:arrays[field][i,s]=result[field]
                kind[i,s]=k;available[i,s]=result["pair_available"]
                scale[i,s]=result["alignment_scale"];aligned[i,s]=result["aligned_b"]
                if b is not None:pair_b[i,s]=b;pair_v[i,s]=vb
        added=previous["point_origin"]==2
        added_hand=added.any(-1)
        # Unassessed points are not called wrong; the optional new-shape adapter
        # conservatively withholds them rather than inventing precise shape.
        hold=(arrays["point_uncertain"]|~arrays["comparable"])&added
        safe=safe_new_shape_channels(x["x"],hold,added)
        npz=dict(**arrays,pair_kind=kind,pair_available=available,pair_b=pair_b,pair_valid=pair_v,
                 aligned_b=aligned,alignment_scale=scale,shape_hold_new=hold,source_added=added,
                 position_xy_px=obs["xy_px"],position_observed=obs["observed"],position_usable=obs["usable"],
                 wrist_available=previous["wrist_available"],palm_available=previous["palm_available"],
                 palm_xy_px=previous["palm_xy_px"],timestamp_us=obs["timestamp_us"],frame_id=obs["frame_id"])
        ep=U/"evidence"/(cid+".npz");np.savez_compressed(ep,**npz)
        ip=U/"inputs"/(cid+".npz")
        np.savez_compressed(ip,x=safe,frame_id=obs["frame_id"],timestamp_us=obs["timestamp_us"],
                            body_valid=safe[:,186:237],hand_valid=safe[:,237:279])
        counts=dict(frames=t,observed_hand_frames=int(mask.any(-1).sum()),
                    paired_hand_frames=int(available.sum()),paired_new_hand_frames=int((available&added_hand).sum()),
                    paired_existing_hand_frames=int((available&oldmask.any(-1)).sum()),
                    new_finger_instances=int(added_hand.sum()*5),
                    new_uncertain_finger_instances=int((arrays["finger_uncertain"]&added_hand[...,None]).sum()),
                    existing_uncertain_finger_instances=int((arrays["finger_uncertain"]&oldmask.any(-1)[...,None]).sum()),
                    new_uncertain_hand_frames=int((arrays["finger_uncertain"].any(-1)&added_hand).sum()),
                    new_unassessed_hand_frames=int((~available&added_hand).sum()),
                    held_new_point_frames=int(hold.sum()),
                    safe_input_value_changes=int((safe!=x["x"]).sum()))
        total.update(counts)
        perclip.append(dict(clip_id=cid,label=r["label"],**counts))
        for i,s in np.argwhere(arrays["finger_uncertain"].any(-1)&added_hand):
            ratios=arrays["bone_discrepancy_ratio"][i,s]
            held_review.append(dict(clip_id=cid,label=r["label"],index=int(i),side=int(s),
                frame_id=int(obs["frame_id"][i]),fingers=np.flatnonzero(arrays["finger_uncertain"][i,s]).tolist(),
                max_ratio=float(np.nanmax(ratios)),core=bool(previous["core"][i])))
        for p in prior:
            if p["clip_id"]!=cid:continue
            i,s=p["index"],p["side"]
            angle_index=int(np.nanargmax(previous["two_scale_angle_difference_deg"][i,s]))
            prior_results.append(dict(**p,angle_index=angle_index,pair_available=bool(available[i,s]),
                    selected_angle_unresolved=bool(arrays["angle_unresolved"][i,s,angle_index]),
                    same_finger_uncertain=bool(arrays["finger_uncertain"][i,s,angle_index//3]),
                    any_uncertain=bool(arrays["finger_uncertain"][i,s].any())))
        index.append(dict(clip_id=cid,outputs={str(ep):sha(ep),str(ip):sha(ip)}))
    write(U/"INDEX.json",index)
    write(U/"SUMMARY.json",dict(status="PREPARED_PENDING_VERIFICATION",counts=dict(total),clips=36,
             rows=perclip,prior12=prior_results,
             caveat="Uncertainty distinguished, not corrected coordinates or measured true shape/classification accuracy."))
    held_review.sort(key=lambda x:(-x["max_ratio"],x["clip_id"],x["frame_id"],x["side"]))
    write(U/"REVIEW_QUEUE.json",held_review)
    for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
    (U/"RUNNING.json").unlink()
    print(json.dumps(dict(total),ensure_ascii=False))


if __name__=="__main__":
    try:freeze() if sys.argv[1]=="freeze" else run()
    except BaseException:
        if not (U/"FIRST_ERROR.json").exists():write(U/"FIRST_ERROR.json",dict(error=traceback.format_exc()))
        raise
