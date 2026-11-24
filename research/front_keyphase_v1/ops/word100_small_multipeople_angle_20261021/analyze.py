from study import U,read,load,write,adapter
from collections import Counter
import numpy as np
rows=read(U/"PREPARED_INDEX.json"); summary=[]; clipstats=[]; overall=Counter()
for word in sorted({r["word_code"] for r in rows}):
    count=Counter(); failures=Counter(); reasons=Counter(); coverage=[]; details=[]
    for r in rows:
        if r["word_code"]!=word:continue
        cid=r["clip_id"];q=load(r["prepared"]);m=read(U/"queries"/(cid+".json"))
        before=read(U/"history"/(cid+".json"));v=load(U/"queries"/(cid+".npz"))["votes"]
        required=m["required"];hs=[("L","R").index(s) for s in required]
        core=q["core"];states=q["angle_state"][core][:,hs]
        count.update("state_"+str(k) for k in states.ravel())
        denom=states.size
        count["core_angle_frames"]+=denom
        count["five_reference_frames"]+=int((v[core][:,hs]>=5).sum())
        one=Counter(x["status"] for x in m["rows"])
        count.update("status_"+x["status"] for x in m["rows"])
        count.update("historical_"+x["status"] for x in before["rows"])
        count["raw_summary"]+=sum(x["raw_only_median_deg"] is not None for x in m["rows"])
        count["strict_summary"]+=sum(x["compared_median_deg"] is not None for x in m["rows"])
        for h in hs:
            ss=q["angle_state"][:,h];votes=v[:,h]
            for name,mask in (("not_observed",ss==0),("degenerate",ss==1),
                              ("not_reconfirmed",ss==2),("reconfirmation_conflict",ss==3),
                              ("reference_or_alignment",(ss==4)&(votes<5))):
                reasons[name]+=int(mask[core].sum())
        ready_pairs=0
        for pid in m["pairs"]:
            p=read(U/"pairs"/(pid+".json"))
            failures["alignment_pairs_total"]+=1
            if p["alignment_support"]<.35:failures["alignment_pairs_below_floor"]+=1
            for x in p["rows"]:
                if not x["required"]:continue
                raw,strict=x["raw"],x["strict"]
                failures["angle_pair_total"]+=1
                if raw["ready"] and not strict["ready"]:failures["raw_ready_but_strict_not"]+=1
                if not strict["ready"]:
                    if not x["alignment_ok"]:failures["alignment"]+=1
                    elif strict["coverage"] is None or strict["coverage"]<.5:failures["coverage_below_half"]+=1
                    elif min(strict["query_frames"],strict["reference_frames"])<8:failures["fewer_than_8"]+=1
                else:ready_pairs+=1
                if strict["coverage"] is not None:coverage.append(strict["coverage"])
        record=dict(clip_id=cid,label=r["label"],signer_id=r["signer_id"],word_code=word,
                    states=dict(Counter(map(int,states.ravel()))),status=dict(one),
                    compared_angles=sum(x["compared_median_deg"] is not None for x in m["rows"]),
                    required_angles=15*len(hs),five_reference_frame_ratio=float((v[core][:,hs]>=5).mean()),
                    query_state4_ratio=float((states==4).mean()),strict_ready_pairs=ready_pairs)
        details.append(record);clipstats.append(record)
    summary.append(dict(word_code=word,label=details[0]["label"],clips=9,counts=dict(count),
                        hold_reasons=dict(reasons),pair_failure_reason=dict(failures),
                        median_strict_pair_coverage=float(np.median(coverage)),clips_detail=details))
    overall.update(count)
prep=read(U/"INPUT_VERIFY.json")
added=prep["new_added_hand_frames"]
write(U/"SUMMARY.json",dict(scope=dict(words=3,people=9,clips=27,existing_reused=5,newly_processed=22,
    saved_frames=3299,pairs=216,source="public TRAIN only"),
    totals=dict(overall),words=summary,clips=clipstats,
    new_in_roi_hand_frames=sum(x["fresh"] for x in added),
    new_in_roi_core_hand_frames=sum(x["fresh"] and x["core"] for x in added),
    legacy_existing_added_hand_frames=sum(not x["fresh"] for x in added),
    coordinates_preserved=prep["counts"]["existing_usable_points_preserved"],
    meaning="Availability and abstention audit, not recognition/correctness performance",
    decision="PREPARATION_AND_SELECTIVE_ABSTENTION_VERIFIED_NORMAL_AVAILABILITY_INSUFFICIENT",
    training=0,user_recordings=0,model_changes=0))
print(json_summary:= [(s["label"],s["counts"]["strict_summary"],s["counts"]["raw_summary"],
                       s["hold_reasons"],s["pair_failure_reason"]) for s in summary])
