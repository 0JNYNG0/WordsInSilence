import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import sys, json, shutil, traceback
from pathlib import Path
from collections import Counter
import importlib.util
import numpy as np
sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent; O = U.parent; ROOT = O.parents[2]
S = O/"word100_small_multipeople_angle_20261021"
A = O/"word100_angle_evidence_20261021"
P = O/"word100_fullframe_hand_merge_20261020"
B = O/"word100_body_boundary_20261017"

def module(name, p):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m); return m
s = module("handtime_previous", S/"study.py")
candidate = module("handtime_candidate", U/"candidate.py")
metric = module("handtime_metric", O/"word100_metric_evidence_20261020/candidate.py")
read, load, write, sha, now = s.read, s.load, s.write, s.sha, s.now
adapter, angle, pipe, Obs = s.adapter, s.angle, s.pipe, s.Observation
from wis_lab.core.features import body_frame


def freeze():
    assert not (U/"FROZEN.json").exists()
    old = read(S/"FROZEN.json"); pipe.check(old["protected"])
    rows = read(S/"PREPARED_INDEX.json")
    have = {r["clip_id"] for r in rows}
    for r in rows: r["primary"] = True
    for r in read(A/"FROZEN.json")["rows"]:
        cid = r["clip_id"]
        if cid in have: continue
        rec = read(B/"data/records"/(cid+".json"))
        dpath = P/"records"/(cid+".json") if r["cohort"]=="previous36" else Path(r["obs"]).parent/"detections.json"
        rows.append(dict(clip_id=cid, label=r["label"], word_code=r["word_code"], signer_id=r["signer_id"],
            primary=False, raw=rec["native_capture"]["raw"], native=rec["native_capture"]["candidate"],
            assignment=str(Path(rec["native_capture"]["candidate"]).parent/"assignment.npz"),
            old_masked=rec["views"]["masked"]["observation"], observation=r["obs"], quality=r["quality"],
            detections=str(dpath), box=rec["calibration"]["box_source_pixels"]))
    pins = {}
    for r in rows:
        assert r["signer_id"] in range(1,10)
        for k in ("raw","native","assignment","old_masked","observation","quality","detections","prepared"):
            if k in r: pins[r[k]] = sha(r[k])
    for p in [U/"candidate.py",U/"study.py",S/"PREPARED_INDEX.json",A/"VERIFY.json",
              P/"candidate.py",pipe.F/"candidate.py",s.A/"candidate.py",s.M/"adapter.py",
              O/"word100_metric_evidence_20261020/candidate.py",S/"FINAL_AUDIT.json"]:
        pins[str(p)] = sha(p)
    for p in (S/"pairs").glob("*"): pins[str(p)] = sha(p)
    (U/"before").mkdir()
    roots=[ROOT/"WordsInSilence_PROJECT_CONTEXT.md",ROOT/"NEW_CHAT_HANDOFF.md",O/"STATE.json"]
    for p in roots: shutil.copyfile(p,U/"before"/p.name)
    write(U/"FROZEN.json",dict(at=now(),rows=rows,pins=pins,protected=old["protected"],roles=old["roles"],
        authorization="User 2026-10-21: separate existing-hand corroboration and per-hand needed time; public verification, no training.",
        scope="27 clips /3 words /9 TRAIN people primary; 131 additional frozen public controls, 158 unique clips/100 words total.",
        conditions=["baseline","recheck_only","schedule_only","combined"],
        recheck="Only unavailable existing-hand evidence: reciprocal unique geometry, original frozen geometry limits; current-arm contradiction/duplicate/ambiguous veto. No coordinate, side, new-hand or old valid evidence changes. Shape corroboration is not proof of anatomical identity.",
        schedule="Reference-only leave-query-signer-out. Existing positive full-source stable-lowering rule (.3sec/8frames/.05shoulder). >=5 people support, any in-ROI hand point veto. Unknown retained required. No query geometry, mask, state or path used to build reference denominators.",
        alignment="Saved 216 primary DTW paths reused; thresholds .35 alignment/.5 coverage/8frames/5people unchanged. Reference-reference paths within 8 peers only for range. Query annotation is offline timing aid, not live capability.",
        validation="All158 geometry/state preservation; old80 control rows; inspect first newly corroborated core per clip and all rest interval boundaries in primary27. Independent expected/coverage arithmetic. Whole-hand query deletion with fresh DTW, fixed-path short angle absence. Normal lowering/optional hand/true crossing not sign error. No pass/fail sign verdict.",
        forbidden=["training","classifier inference","user/mentor data","DEV/heldout","runtime changes","camera","downloads","external transfer","Git"]))
    print("FROZEN",len(rows),flush=True)


def recheck(r):
    raw=load(r["raw"]); sides=load(r["assignment"])["sides"]
    obs=load(r["observation"]); old=load(r["old_masked"]); q=load(r["quality"])
    frames=read(r["detections"])["frames"]
    xy=obs["xy_px"][:,33:].reshape(-1,2,21,2); v=obs["usable"][:,33:].reshape(-1,2,21)
    ov=old["usable"][:,33:].reshape(-1,2,21)
    pa=q["pair_available"].copy(); av=q["angle_available"].copy(); au=q["angle_unresolved"].copy()
    changes=[]; checks=[]; pair_b=q["pair_b"].copy(); pair_v=q["pair_valid"].copy()
    for i,fr in enumerate(frames):
        n=int(raw["raw_count"][i])
        proposals=candidate.existing_correspondence(fr["full"],raw["hand_xy_px"][i,:n],sides[i,:n].tolist(),
            raw["pose_xy_px"][i],raw["pose_usable"][i],int(raw["width"][i]),int(raw["height"][i]),pipe.merge)
        for p in proposals:
            h=p["side"]; p=dict(p,index=i,used=False)
            if p["accepted"] and h in (0,1) and not pa[i,h] and ov[i,h].any() and v[i,h].any():
                b=np.array(fr["full"][p["full_index"]]["xy"]); vb=pipe.roi._inside(b,r["box"])
                result=pipe.shape.compare_shape(xy[i,h],v[i,h],b,vb)
                if result["pair_available"]:
                    pa[i,h]=True; av[i,h]=result["angle_available"]; au[i,h]=result["angle_unresolved"]
                    pair_b[i,h]=b; pair_v[i,h]=vb; p["used"]=True; changes.append(p)
            checks.append(p)
    before=angle.angle_evidence(xy,v,q["pair_available"],q["angle_available"],q["angle_unresolved"])
    after=angle.angle_evidence(xy,v,pa,av,au)
    assert np.array_equal(before["angle_deg"],after["angle_deg"],equal_nan=True)
    assert np.all(after["angle_state"][before["angle_state"]!=2]==before["angle_state"][before["angle_state"]!=2])
    return before,after,dict(pair_available=pa,angle_available=av,angle_unresolved=au,
        pair_b=pair_b,pair_valid=pair_v),changes,checks


def run():
    f=read(U/"FROZEN.json");pipe.check(f["pins"]);pipe.check(f["protected"])
    assert not (U/"RUNNING.json").exists()
    write(U/"RUNNING.json",dict(at=now(),pid=os.getpid()))
    for sub in ("rechecked","schedules","pairs","queries"): (U/sub).mkdir()
    index=[];counts=Counter(); samples={}; states={}
    for r in f["rows"]:
        cid=r["clip_id"];before,after,e,changed,checks=recheck(r)
        np.savez_compressed(U/"rechecked"/(cid+".npz"),**after,**e,before_state=before["angle_state"])
        write(U/"rechecked"/(cid+".json"),dict(changes=changed,checks=checks))
        counts["clips"]+=1;counts["new_reconfirmed_hand_frames"]+=len(changed)
        counts["new_eligible_angle_frames"]+=int(((before["angle_state"]==2)&(after["angle_state"]==4)).sum())
        if r["primary"]:
            z=load(r["prepared"]); zz=dict(z); zz.update(after);samples[cid]=(z,zz)
            native=Obs.load(r["native"]);origin,scale,_=body_frame(native)
            src=metric.source_states(load(r["native"]),load(r["observation"]),native.timestamp_us,
                r["box"][3],r["box"][2],origin,scale,r["intervals"])
            active=np.stack([load(r["observation"])["usable"][:,33+21*h:54+21*h].any(1) for h in (0,1)],1)
            states[cid]=dict(core=z["core"], rest=np.stack([src[h]["offline_full_view_state"]==3 for h in ("L","R")],1),
                             active=active)
            np.savez_compressed(U/"schedules"/(cid+"__source.npz"),**states[cid])
        index.append(dict(clip_id=cid,label=r["label"],primary=r["primary"],changes=len(changed)))
    write(U/"RECHECK_INDEX.json",index)
    primary=[r for r in f["rows"] if r["primary"]];summary={}
    for qrow in primary:
        cid=qrow["clip_id"];required=f["roles"].get(qrow["word_code"],{}).get("required_sides",["L","R"])
        refs=[r for r in primary if r["word_code"]==qrow["word_code"] and r["signer_id"]!=qrow["signer_id"]]
        assert len(refs)==8
        schedules={}
        for rr in refs:
            rid=rr["clip_id"];others=[r for r in refs if r["clip_id"]!=rid]
            paths=[load(S/"pairs"/(rid+"__"+x["clip_id"]+".npz"))["path"] for x in others]
            sch=candidate.reference_schedule(states[rid],[states[x["clip_id"]] for x in others],paths)
            schedules[rid]=sch
            np.savez_compressed(U/"schedules"/(cid+"__"+rid+".npz"),**sch)
        for condition in f["conditions"]:
            quality=condition in ("recheck_only","combined")
            timed=condition in ("schedule_only","combined")
            pairs=[]
            for rr in refs:
                rid=rr["clip_id"];pid=cid+"__"+rid
                saved=load(S/"pairs"/(pid+".npz"));support=read(S/"pairs"/(pid+".json"))["alignment_support"]
                r=samples[rid][int(quality)]; q=samples[cid][int(quality)]
                ex=schedules[rid]["expected"] if timed else np.broadcast_to(r["core"][:,None],(len(r["core"]),2))
                p=candidate.pair(q,r,saved["path"],required,support>=.35,ex,adapter)
                np.savez_compressed(U/"pairs"/(condition+"__"+pid+".npz"),
                    **{k:p[k] for k in ("expected","raw","strict","difference_deg","path")})
                write(U/"pairs"/(condition+"__"+pid+".json"),dict(rows=p["rows"],alignment_support=support))
                pairs.append(p)
            out=candidate.summarize(pairs,required,adapter)
            write(U/"queries"/(condition+"__"+cid+".json"),dict(rows=out,reference_people=[r["signer_id"] for r in refs]))
            c=summary.setdefault(condition,dict(total=Counter(),words={}))
            for dest in [c["total"],c["words"].setdefault(qrow["label"],Counter())]:
                dest.update(row["status"] for row in out)
        print("COMPARED",cid,flush=True)
    write(U/"RESULT.json",dict(at=now(),counts=dict(counts),comparison=summary))
    pipe.check(f["pins"]);pipe.check(f["protected"]);(U/"RUNNING.json").unlink()
    print("DONE",dict(counts),flush=True)


if __name__=="__main__":
    try: {"freeze":freeze,"run":run}[sys.argv[1]]()
    except BaseException:
        if not (U/"FIRST_ERROR.txt").exists(): (U/"FIRST_ERROR.txt").write_text(traceback.format_exc(),encoding="utf8")
        raise
