import hashlib,json,os,shutil,sys,traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from candidate import angle_evidence
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
F=O/"word100_finger_evidence_20261020";W=O/"word100_finger_preservation_20261020"
Q=O/"word100_merge_scale_input_20261020"
def read(p):return json.loads(Path(p).read_text("utf-8-sig"))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open("x",encoding="utf8") as fp:json.dump(d,fp,ensure_ascii=False,indent=2,allow_nan=False)
def freeze():
    assert read(U/"CONTRACTS.json")["status"]=="PASS"
    assert not any((p/"RUNNING.json").exists() for p in [F,W,U])
    rows=[];pins={}
    for source in [F,W]:
        prior=read(source/"FROZEN.json")
        for r in prior["rows"]:
            cid=r["clip_id"];q=dict(clip_id=cid,label=r["label"],signer_id=r["signer_id"],word_code=r["word_code"],
                cohort="previous36" if source==F else "broader100")
            if source==F:
                q.update(obs=r["merged"],quality=str(F/"evidence"/(cid+".npz")),
                    inputs=str(Q/"inputs"/cid/"merge_fixed_scale.npz"),input_key="x",
                    old_hold=str(F/"inputs"/(cid+".npz")),old_hold_key="x",core=str(Q/"evidence"/(cid+".npz")))
            else:
                p=W/"outputs"/cid
                q.update(obs=str(p/"merged.npz"),quality=str(p/"evidence.npz"),
                    inputs=str(p/"inputs.npz"),input_key="merged",old_hold=str(p/"inputs.npz"),
                    old_hold_key="safe",core=str(p/"evidence.npz"))
            assert q["signer_id"] in range(1,10)
            for k in ["obs","quality","inputs","old_hold","core"]:pins[q[k]]=sha(q[k])
            rows.append(q)
        for name in ["FROZEN.json","FINAL_AUDIT.json","VERIFY.json","INDEX.json"]:
            pins[str(source/name)]=sha(source/name)
    assert len(rows)==len({r["clip_id"] for r in rows})==136
    for name in ["candidate.py","contracts.py","CONTRACTS.json","run.py","verify.py"]:pins[str(U/name)]=sha(U/name)
    protected=read(W/"FROZEN.json")["protected"]
    for p,h in protected.items():assert sha(p)==h,p
    cross=O/"word100_image_arm_connectivity_20261018/CROSSING.json"
    normal=O/"word100_image_arm_connectivity_20261018/ANNOTATED.json"
    for p in [cross,normal,W/"VISUAL_TARGETS.json",W/"REVIEW_SUPPLEMENT.json",W/"VISUAL_REVIEW.json"]:
        pins[str(p)]=sha(p)
    (U/"before").mkdir()
    for p in [R/"WordsInSilence_PROJECT_CONTEXT.md",R/"NEW_CHAT_HANDOFF.md",O/"STATE.json"]:
        shutil.copyfile(p,U/"before"/p.name)
    write(U/"FROZEN.json",dict(at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),rows=rows,pins=pins,protected=protected,
        authorization="User approves coordinate-preserving per-angle evidence candidate and public validation only; no training until joint review.",
        rule="Coordinates/masks/timestamps/features unchanged. Per-angle presence,degeneracy,unassessed,unresolved,assessed-no-conflict. Existing frozen two-prediction diagnostics reused unchanged; no new thresholds.",
        interpretation="Eligibility is absence of known conflict in detector evidence, not semantic correctness or calibrated accuracy.",
        comparison="Contrast old new-point exclusion with per-angle eligibility, stratified old/new and previous36/broader100. Entire model input must be equal to merged fixed-scale source.",
        controls=dict(crossing=str(cross),normal=str(normal),systematic20=str(W/"VISUAL_TARGETS.json")),
        visual_policy="Reuse previous raw92-frame audit for unchanged coordinates; review representative recovery/remaining gaps with per-angle masks. No new ground-truth labels.",
        gates=["No coordinate or model feature changes","No propagation via deleted point","No unknown angle represented as match0","Normal/crossing coordinate preservation","Independent saved-angle/status verification","Report existing-angle uncertainty separately"],
        forbidden=["training","classifier scoring","detector rerun","user data","downloads","external transfer","Git","production C/T3F/DTW/web changes"]))
def run():
    f=read(U/"FROZEN.json")
    for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
    write(U/"RUNNING.json",dict(pid=os.getpid()))
    (U/"outputs").mkdir();index=[];rows=[];totals={k:Counter() for k in ["previous36","broader100","all"]}
    for r in f["rows"]:
        obs=load(r["obs"]);e=load(r["quality"]);inputs=load(r["inputs"]);old=load(r["old_hold"])
        x=inputs[r["input_key"]];oldx=old[r["old_hold_key"]];core=load(r["core"])["core"]
        xy=obs["xy_px"][:,33:].reshape(-1,2,21,2);v=x[:,237:279].reshape(-1,2,21).astype(bool)
        result=angle_evidence(xy,v,e["pair_available"],e["angle_available"],e["angle_unresolved"])
        oldresult=angle_evidence(xy,oldx[:,237:279].reshape(-1,2,21).astype(bool),
            e["pair_available"],e["angle_available"],e["angle_unresolved"])
        added=e["source_added"].any(-1)[...,None]
        restore=result["compare_eligible"]&~oldresult["computable"]
        lost=result["computable"]&~oldresult["computable"]
        out=dict(**result,xy_px=obs["xy_px"],observed=obs["observed"],usable=obs["usable"],conf=obs["conf"],
            x=x,frame_id=obs["frame_id"],timestamp_us=obs["timestamp_us"],core=core,source_added=e["source_added"],
            old_point_filter_computable=oldresult["computable"],restored_eligible=restore)
        path=U/"outputs"/(r["clip_id"]+".npz");np.savez_compressed(path,**out)
        counts=dict(frames=len(x),retained_raw_point_frames=int(obs["observed"].sum()),
            retained_model_values=int(x.size),raw_computable=int(result["computable"].sum()),
            eligible=int(result["compare_eligible"].sum()),old_point_filter_computable=int(oldresult["computable"].sum()),
            point_filter_lost=int(lost.sum()),restored_eligible=int(restore.sum()),
            core_restored_eligible=int((restore&core[:,None,None]).sum()),
            core_point_filter_lost=int((lost&core[:,None,None]).sum()))
        for k,name in enumerate(["missing","degenerate","unassessed","unresolved","assessed_no_conflict"]):
            counts[name]=int((result["angle_state"]==k).sum())
            counts["new_"+name]=int(((result["angle_state"]==k)&added).sum())
            counts["existing_"+name]=int(((result["angle_state"]==k)&~added).sum())
        for key in [r["cohort"],"all"]:totals[key].update(counts)
        rows.append(dict(clip_id=r["clip_id"],label=r["label"],cohort=r["cohort"],**counts))
        index.append(dict(clip_id=r["clip_id"],path=str(path),sha256=sha(path)))
    write(U/"INDEX.json",index)
    write(U/"SUMMARY.json",dict(status="PREPARED_PENDING_VERIFICATION",totals={k:dict(v) for k,v in totals.items()},rows=rows))
    (U/"RUNNING.json").unlink()
    print(json.dumps({k:dict(v) for k,v in totals.items()},ensure_ascii=False))
if __name__=="__main__":
    try:freeze() if sys.argv[1]=="freeze" else run()
    except BaseException:
        if not (U/"FIRST_ERROR.json").exists():write(U/"FIRST_ERROR.json",dict(error=traceback.format_exc()))
        raise
