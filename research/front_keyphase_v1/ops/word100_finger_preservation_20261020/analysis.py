"""Frozen descriptive information-loss audit; no performance-based selection."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
CHAINS=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
TRIPLES=[ch[j:j+3] for ch in CHAINS for j in range(3)]
def angles(xy,v):
    angle=np.full((*v.shape[:-1],15),np.nan);valid=np.zeros(angle.shape,bool)
    for k,(a,b,c) in enumerate(TRIPLES):
        x=xy[...,a,:]-xy[...,b,:];y=xy[...,c,:]-xy[...,b,:]
        ok=v[...,[a,b,c]].all(-1)&(np.linalg.norm(x,axis=-1)>1e-6)&(np.linalg.norm(y,axis=-1)>1e-6)
        d=np.degrees(np.arctan2(abs(x[...,0]*y[...,1]-x[...,1]*y[...,0]),(x*y).sum(-1)))
        angle[...,k]=np.where(ok,d,np.nan);valid[...,k]=ok
    return angle,valid
def runs(mask):
    x=np.pad(np.asarray(mask,np.int8),(1,1));d=np.diff(x)
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)-1))

def main():
    f=read(U/"FROZEN.json");total=Counter();rows=[];events=[];controls=[];motion_total=0.;motion_lost=0.
    for rank,r in enumerate(f["rows"]):
        out=U/"outputs"/r["clip_id"];e=load(out/"evidence.npz");x=load(out/"inputs.npz")
        obs=load(out/"merged.npz");xy=obs["xy_px"][:,33:].reshape(-1,2,21,2)
        rawmask=obs["usable"][:,33:].reshape(-1,2,21)
        core=e["core"];new=e["source_added"].any(-1)
        before=x["before"][:,237:279].reshape(-1,2,21).astype(bool)
        merged=x["merged"][:,237:279].reshape(-1,2,21).astype(bool)
        safe=x["safe"][:,237:279].reshape(-1,2,21).astype(bool)
        a,av=angles(xy,merged);b,bv=angles(xy,safe)
        _,ov=angles(xy,before)
        lost=av&~bv
        unknown=~e["pair_available"]&new
        ambiguous=e["finger_uncertain"].any(-1)&new
        all_lost=av.any(-1)&~bv.any(-1)
        visible_hands=rawmask.any(-1)
        counts=dict(frames=len(core),core_frames=int(core.sum()),
            observed_hand_frames=int(visible_hands.sum()),
            old_hand_frames=int((e["point_origin"]==1).any(-1).sum()),
            added_hand_frames=int(new.sum()),added_core_hand_frames=int((new&core[:,None]).sum()),
            new_paired_without_flags=int((new&e["pair_available"]&~ambiguous).sum()),
            new_uncertain_hand_frames=int(ambiguous.sum()),new_unpaired=int(unknown.sum()),
            new_uncertain_fingers=int((new[...,None]&e["finger_uncertain"]).sum()),
            body_points_removed=int((x["merged"][:,186:237]-x["safe"][:,186:237]).sum()),
            local_points_removed=int((merged&~safe).sum()),
            before_local_points=int(before.sum()),merged_local_points=int(merged.sum()),safe_local_points=int(safe.sum()),
            before_angles=int(ov.sum()),merged_angles=int(av.sum()),safe_angles=int(bv.sum()),
            lost_angles=int(lost.sum()),lost_core_angles=int((lost&core[:,None,None]).sum()),
            core_merged_angles=int((av&core[:,None,None]).sum()),
            core_safe_angles=int((bv&core[:,None,None]).sum()),
            core_hand_frames_with_shape_loss=int((lost.any(-1)&core[:,None]).sum()),
            core_hand_frames_losing_all_computable_angles=int((all_lost&core[:,None]).sum()),
            core_frames_losing_all_hands_angles=int((core&av.any((1,2))&~bv.any((1,2))).sum()),
            crossing_proxy_frames=int(((xy[:,0,0,0]<xy[:,1,0,0])&rawmask[:,0,0]&rawmask[:,1,0]&core).sum()))
        # A change in adjacent measured angle is a proxy, not a true semantic cue.
        transition=av[1:]&av[:-1]&core[1:,None,None]&core[:-1,None,None]
        kept=bv[1:]&bv[:-1]
        variation=np.where(transition,abs(np.nan_to_num(a[1:]-a[:-1])),0.)
        lost_variation=np.where(transition&~kept,variation,0.)
        energy=float(variation.sum());erased=float(lost_variation.sum())
        motion_total+=energy;motion_lost+=erased
        dt=np.diff(e["timestamp_us"])/1e6
        step=float(np.median(dt)) if len(dt) else 0.
        per=[]
        for side in range(2):
            for start,end in runs(lost[:,side].any(-1)&core):
                duration=float((e["timestamp_us"][end]-e["timestamp_us"][start])/1e6+step)
                per.append(dict(clip_id=r["clip_id"],label=r["label"],side=side,
                    start_index=int(start),end_index=int(end),duration_s=duration,
                    lost_angle_frames=int(lost[start:end+1,side].sum()),
                    all_angles_lost_frames=int(all_lost[start:end+1,side].sum()),
                    middle_index=int((start+end)//2)))
        events.extend(per)
        core_ids=np.flatnonzero(core)
        middle=int(core_ids[len(core_ids)//2]) if len(core_ids) else len(core)//2
        if rank%5==0:controls.append(dict(clip_id=r["clip_id"],label=r["label"],index=middle,selection="systematic every5word core midpoint"))
        rows.append(dict(clip_id=r["clip_id"],label=r["label"],word_code=r["word_code"],signer_id=r["signer_id"],
            **counts,angle_variation_deg=energy,removed_angle_variation_deg=erased,
            longest_core_loss_s=max((z["duration_s"] for z in per),default=0.),
            largest_removed_transition=(dict(index=int(np.unravel_index(np.argmax(lost_variation),lost_variation.shape)[0])+1,
                side=int(np.unravel_index(np.argmax(lost_variation),lost_variation.shape)[1]),
                angle_index=int(np.unravel_index(np.argmax(lost_variation),lost_variation.shape)[2]),
                delta_deg=float(lost_variation.max())) if lost_variation.any() else None)))
        total.update(counts)
        np.savez_compressed(out/"loss_audit.npz",angles=a,angle_valid_before=ov,angle_valid_merged=av,
            angle_valid_safe=bv,lost_angles=lost,core=core,
            removed_angle_variation=lost_variation,all_angles_lost=all_lost)
    events.sort(key=lambda e:(-e["duration_s"],-e["lost_angle_frames"],e["clip_id"],e["side"]))
    selected=[];seen=set()
    for e in events:
        if e["clip_id"] not in seen:
            selected.append(e);seen.add(e["clip_id"])
            if len(selected)==12:break
    variation=sorted((r for r in rows if r["largest_removed_transition"]),key=lambda r:-r["largest_removed_transition"]["delta_deg"])
    result=dict(status="DESCRIPTIVE_LOSS_AUDIT_COMPLETE",scope="100new clips/100words/TRAIN only",
        counts=dict(total),clips_with_core_angle_loss=sum(r["lost_core_angles"]>0 for r in rows),
        clips_with_core_all_angle_loss=sum(r["core_hand_frames_losing_all_computable_angles"]>0 for r in rows),
        longest_core_loss_s=max((r["longest_core_loss_s"] for r in rows),default=0),
        core_angle_variation_deg=motion_total,removed_core_angle_variation_deg=motion_lost,
        limitation="Measured angle changes may be detector jitter. Core interval is not per-finger semantic annotation. These metrics expose possible information loss, not recognition accuracy or required-phase ground truth.",
        rows=rows,top_loss_events=selected,top_removed_variation=variation[:8],normal_controls=controls)
    with (U/"ANALYSIS.json").open("x",encoding="utf8") as fp:json.dump(result,fp,ensure_ascii=False,indent=2)
    with (U/"ALL_LOSS_EVENTS.json").open("x",encoding="utf8") as fp:json.dump(events,fp,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ("rows","top_loss_events","top_removed_variation","normal_controls")},ensure_ascii=False))
if __name__=="__main__":main()
