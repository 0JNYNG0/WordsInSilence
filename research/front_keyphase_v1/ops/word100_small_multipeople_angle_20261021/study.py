"""Complete, small public cohort; frozen detector/merge/angle/DTW rules."""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import sys, json, hashlib, shutil, time, traceback
from pathlib import Path
from collections import Counter
import importlib.util
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
O = U.parent
ROOT = O.parents[2]
W = O / "word100_finger_preservation_20261020"
A = O / "word100_angle_evidence_20261021"
M = O / "word100_angle_multiref_dtw_20261021"
B = O / "word100_body_boundary_20261017"
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
pipe = module("small_cohort_pipeline", W / "pipeline.py")
pipe.U = U  # All pipeline outputs are redirected to THIS new research unit.
adapter = module("adapter", M / "adapter.py")
oldrun = module("small_cohort_alignment", M / "run.py")
angle = module("small_cohort_angle", A / "candidate.py")
read, load, write, sha, now = pipe.read, pipe.load, pipe.write, pipe.sha, pipe.stamp
Observation, F = oldrun.Observation, oldrun.F


def freeze():
    previous = read(M / "FROZEN.json")
    pipe.check(previous["protected"])
    manifest = read(B / "PREP_CONTEXT.json")
    # Deterministic role/control selection; no new scores or quality results.
    roles = read(O / "word100_role_dtw_20261017/HAND_ROLES.json")["words"]
    both = min(w for w, v in roles.items() if v["required_sides"] == ["L", "R"])
    single = min(w for w, v in roles.items() if len(v["required_sides"]) == 1)
    crosses = read(O / "word100_image_arm_connectivity_20261018/CROSSING.json")
    crossing = min(x["clip_id"].split("_")[2] for x in crosses)
    words = sorted({both, single, crossing})
    assert words == ["WORD0003", "WORD0185", "WORD0949"]
    prior = {r["clip_id"]: r for r in read(A / "FROZEN.json")["rows"]}
    aindex = {r["clip_id"]: r for r in read(A / "INDEX.json")}
    candidates = sorted([r for r in manifest["ordered_rows"]
                         if r["role"] == "train_candidate" and r["word_code"] in words],
                        key=lambda r: (r["word_code"], r["signer_id"]))
    assert len(candidates) == 27
    pins, rows = {}, []
    for r in candidates:
        cid = r["clip_id"]; rp = B / "data/records" / (cid + ".json"); rec = read(rp)
        assert 1 <= r["signer_id"] <= 9 and rec["role"] == "train_candidate"
        native = rec["native_capture"]["candidate"]
        vsrc = r.get("retained_video")
        if not vsrc or not Path(vsrc).exists() or sha(vsrc) != rec["source"]["sha256"]:
            vsrc = None
        row = dict(clip_id=cid, label=r["canonical_label"], word_code=r["word_code"],
                   signer_id=r["signer_id"], role="train_candidate", frames=rec["frames"],
                   intervals=r["intervals"], box=rec["calibration"]["box_source_pixels"],
                   raw=rec["native_capture"]["raw"], native=native,
                   assignment=str(Path(native).parent / "assignment.npz"),
                   old_masked=rec["views"]["masked"]["observation"], source=rec["source"],
                   retained_video=vsrc, record_path=str(rp), reuse=cid in prior)
        if cid in prior:
            ar = prior[cid]
            row.update(observation=ar["obs"], quality=ar["quality"], angle_saved=aindex[cid]["path"])
            assert sha(row["angle_saved"]) == aindex[cid]["sha256"]
            row["detections"] = str((Path(ar["obs"]).parent / "detections.json")
                                    if ar["cohort"] == "broader100" else
                                    (O / "word100_fullframe_hand_merge_20261020/records" / (cid + ".json")))
        for k in ("raw", "assignment", "native", "old_masked", "record_path",
                  "observation", "quality", "angle_saved", "detections"):
            if k in row:
                p = Path(row[k]); assert p.is_file(), (cid, k, p)
                pins[str(p)] = sha(p)
                side = p.with_suffix(".sidecar.json")
                if side.exists(): pins[str(side)] = sha(side)
        if vsrc: pins[vsrc] = sha(vsrc)
        rows.append(row)
    for word in words:
        assert {r["signer_id"] for r in rows if r["word_code"] == word} == set(range(1, 10))
    for p in [Path(__file__), W/"pipeline.py", A/"candidate.py", A/"FROZEN.json",
              A/"INDEX.json", A/"VERIFY.json", M/"adapter.py", M/"run.py", M/"FROZEN.json",
              M/"VERIFY.json", pipe.P/"candidate.py", pipe.F/"candidate.py",
              pipe.Q/"input_candidate.py", pipe.MODEL,
              O/"word100_fullframe_roi_pair_20261016/roi_adapter.py",
              O/"word100_role_dtw_20261017/HAND_ROLES.json",
              O/"word100_image_arm_connectivity_20261018/CROSSING.json"]:
        pins[str(p)] = sha(p)
    (U/"before").mkdir()
    for p in [ROOT/"WordsInSilence_PROJECT_CONTEXT.md", ROOT/"NEW_CHAT_HANDOFF.md", O/"STATE.json"]:
        shutil.copyfile(p, U/"before"/p.name)
    write(U/"FROZEN.json", dict(at=now(), rows=rows, pins=pins, protected=previous["protected"],
        authorization="2026-10-21 user: prepare same rechecking for small complete multiperson public cohort; verify, no training before joint review.",
        selection=dict(words=words, basis="lowest public role code for both/single hand, lowest prior crossing control word; all TRAIN01-09. No new results used.", clips=27, people=9),
        roles=roles, detector_parameters=read(W/"FROZEN.json")["detector_parameters"],
        conditions=previous["conditions"],
        comparisons=["historical available metadata, same final coordinates/path",
                     "complete metadata, same final coordinates/path",
                     "raw finite angles, same final coordinates/path"],
        implementation="Reuse frozen whole-frame redetection + absent-hand crop corroboration + merge BEFORE constant ROI; old hand coordinates/identities never overwritten. Same old scales. Angle status only; legacy safe-point output unused.",
        visual_policy="All27 at public core25/50/75%; all newly added in-ROI hand frames; selected prior crossing controls; top per-word withheld computable cases. Images first, then evidence. No thresholds changed.",
        interpretation="No error/correctness labels. Enough metadata != enough reliable evidence. Report normal availability failures explicitly, not just abstention-safety PASS. Public timing oracle; not deployment.",
        forbidden=["training","classifier inference","DEV/heldout","user/mentor data","camera","downloads","external transfer","Git","runtime changes"]))
    print("FROZEN", Counter(r["reuse"] for r in rows), sum(r["frames"] for r in rows), flush=True)


def prepare():
    f = read(U/"FROZEN.json"); pipe.check(f["pins"]); pipe.check(f["protected"])
    assert not (U/"RUNNING.json").exists() and not (U/"PREPARED_INDEX.json").exists()
    write(U/"RUNNING.json", dict(pid=os.getpid(), at=now(), stage="prepare"))
    for sub in ("videos", "outputs", "prepared"): (U/sub).mkdir()
    archives = {}; index = []; start = time.monotonic()
    opts = pipe.vision.HandLandmarkerOptions(
        base_options=pipe.BaseOptions(model_asset_path=str(pipe.MODEL)),
        running_mode=pipe.vision.RunningMode.IMAGE, **f["detector_parameters"])
    try:
        with pipe.vision.HandLandmarker.create_from_options(opts) as lm:
            for n, row in enumerate(f["rows"]):
                r = dict(row); cid = r["clip_id"]
                video = pipe.video_path(r, archives)
                if not r["reuse"]:
                    result = pipe.process(lm, r, video)
                    r.update(observation=str(U/"outputs"/cid/"merged.npz"),
                             quality=str(U/"outputs"/cid/"evidence.npz"),
                             detections=str(U/"outputs"/cid/"detections.json"),
                             elapsed_seconds=result["seconds"])
                obs = Observation.load(r["observation"]); quality = load(r["quality"])
                xy = obs.xy_px[:,33:].reshape(-1,2,21,2); usable = obs.usable[:,33:].reshape(-1,2,21)
                ev = angle.angle_evidence(xy, usable, quality["pair_available"],
                                         quality["angle_available"], quality["angle_unresolved"])
                if r["reuse"]:
                    prior = load(r["angle_saved"])
                    for k in ("angle_deg","angle_state"):
                        assert np.array_equal(ev[k],prior[k],equal_nan=True)
                seq = F.sequence_features(obs, trim=False)
                core = np.zeros(obs.n_frames,bool)
                for a,b in r["intervals"]:
                    core |= (obs.timestamp_us >= a*1e6)&(obs.timestamp_us <= b*1e6)
                prepared = dict(features=seq["features"], present=seq["present"],
                    **ev, xy_px=obs.xy_px, usable=obs.usable, observed=obs.observed,
                    conf=obs.conf, frame_id=obs.frame_id, timestamp_us=obs.timestamp_us, core=core)
                np.savez_compressed(U/"prepared"/(cid+".npz"),**prepared)
                r.update(video=str(video), prepared=str(U/"prepared"/(cid+".npz")))
                r["output_hashes"]={r[k]:sha(r[k]) for k in ("observation","quality","detections","prepared")}
                index.append(r)
                (U/"PROGRESS.json").write_text(json.dumps(dict(complete=n+1,total=27,at=now(),last=cid,
                    seconds=time.monotonic()-start),ensure_ascii=False,indent=2),encoding="utf8")
                print("PREPARED",n+1,27,cid,"reuse" if r["reuse"] else "new",flush=True)
        write(U/"PREPARED_INDEX.json",index); pipe.check(f["pins"]); pipe.check(f["protected"])
        print("PREPARED_ALL",round(time.monotonic()-start,1),flush=True)
    finally:
        for z in archives.values(): z.close()
        (U/"RUNNING.json").unlink()


def compare():
    f=read(U/"FROZEN.json"); rows=read(U/"PREPARED_INDEX.json")
    pipe.check(f["pins"]); pipe.check(f["protected"])
    for r in rows: pipe.check(r["output_hashes"])
    for sub in ("pairs","queries","history"): (U/sub).mkdir()
    count=Counter(); index=[]; history=[]
    for word in f["selection"]["words"]:
        rr=[r for r in rows if r["word_code"]==word]
        samples={r["clip_id"]:load(r["prepared"]) for r in rr}
        historical={}
        for r in rr:
            z=dict(samples[r["clip_id"]])
            if not r["reuse"]:
                z["angle_state"]=np.where(z["angle_state"]>=2,2,z["angle_state"])
            historical[r["clip_id"]]=z
        required=f["roles"].get(word,{}).get("required_sides",["L","R"])
        for row in rr:
            cid=row["clip_id"]; q=samples[cid]; pairs=[]; oldpairs=[]; people=[]; pids=[]
            for ref in rr:
                if ref["signer_id"]==row["signer_id"]: continue
                rid=ref["clip_id"]; r=samples[rid]
                path,support=oldrun.alignment(q,r,required)
                p=adapter.pair(q,r,path,required,support>=.35)
                oldp=adapter.pair(historical[cid],historical[rid],path,required,support>=.35)
                pid=cid+"__"+rid
                np.savez_compressed(U/"pairs"/(pid+".npz"),
                    **{k:p[k] for k in ("expected","raw","strict","difference_deg","path")})
                write(U/"pairs"/(pid+".json"),dict(query=cid,reference=rid,required=required,
                    alignment_support=support,rows=p["rows"],history_rows=oldp["rows"]))
                pairs.append(p); oldpairs.append(oldp); people.append(ref["signer_id"]); pids.append(pid)
            ag=adapter.aggregate(q,pairs,required,people)
            agold=adapter.aggregate(historical[cid],oldpairs,required,people)
            np.savez_compressed(U/"queries"/(cid+".npz"),votes=ag["votes"],expected_votes=ag["expected_votes"])
            write(U/"queries"/(cid+".json"),dict(**row,required=required,pairs=pids,rows=ag["rows"],
                accepted=None,sign_error=None,whole_hand_score=None))
            np.savez_compressed(U/"history"/(cid+".npz"),votes=agold["votes"],expected_votes=agold["expected_votes"])
            write(U/"history"/(cid+".json"),dict(rows=agold["rows"]))
            count.update("angle_"+v["status"] for v in ag["rows"])
            count["pairs"]+=len(pairs);count["query_clips"]+=1
            index.append({k:row[k] for k in ("clip_id","label","word_code","signer_id")})
        print("COMPARED",word,dict(count),flush=True)
    write(U/"INDEX.json",index)
    write(U/"RUN_RESULT.json",dict(status="COMPLETE_PENDING_VERIFY",counts=dict(count),at=now()))


if __name__=="__main__":
    pipe.cv2.setNumThreads(1)
    try:
        {"freeze":freeze,"prepare":prepare,"compare":compare}[sys.argv[1]]()
    except BaseException:
        p=U/"FIRST_ERROR.txt"
        if not p.exists(): p.write_text(traceback.format_exc(),encoding="utf8")
        raise
