"""Broader public TRAIN audit with frozen merge/shape rules, no classifier."""
import importlib.util
import json
import hashlib
import shutil
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
P=O/"word100_fullframe_hand_merge_20261020"
Q=O/"word100_merge_scale_input_20261020"
F=O/"word100_finger_evidence_20261020"
B=O/"word100_body_boundary_20261017"
sys.path[:0]=[str(O.parent/"src"),str(ROOT/"research/sign_model_pilot_v1/src")]
from wis_lab.core.observation import Observation
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
merge=module("preserve_merge",P/"candidate.py")
shape=module("preserve_fingers",F/"candidate.py")
channels=module("preserve_channels",Q/"input_candidate.py")
roi=channels.roi
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as fp:
        for block in iter(lambda:fp.read(2**20),b""):h.update(block)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,v):
    with Path(p).open("x",encoding="utf8") as fp:json.dump(v,fp,ensure_ascii=False,indent=2,allow_nan=False)
def stamp():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def check(pins):
    for p,h in pins.items():assert sha(p)==h,p
MODEL=ROOT/"Tools/models/hand_landmarker.task"


def freeze():
    previous=read(F/"FROZEN.json")
    check(previous["protected"])
    assert not (F/"RUNNING.json").exists() and not (P/"RUNNING.json").exists()
    manifest=read(B/"PREP_CONTEXT.json")
    candidates=[r for r in manifest["ordered_rows"] if r["role"]=="train_candidate"]
    assert len(candidates)==899
    prior_ids={r["clip_id"] for r in previous["rows"]}
    words=sorted({r["word_code"] for r in candidates});assert len(words)==100
    selected=[]
    for n,word in enumerate(words):
        preferred=1+n%9
        choices=[r for r in candidates if r["word_code"]==word and r["clip_id"] not in prior_ids]
        r=min(choices,key=lambda x:((x["signer_id"]-preferred)%9,x["clip_id"]))
        selected.append(r)
    pins={}
    rows=[]
    for r in selected:
        cid=r["clip_id"];rp=B/"data/records"/(cid+".json");rec=read(rp)
        assert rec["role"]=="train_candidate" and rec["signer_id"] in range(1,10)
        native=rec["native_capture"]["candidate"]
        ass=str(Path(native).parent/"assignment.npz")
        assert Path(ass).is_file(),ass
        vsrc=r.get("retained_video")
        if not vsrc or not Path(vsrc).is_file() or sha(vsrc)!=rec["source"]["sha256"]:vsrc=None
        row=dict(clip_id=cid,label=r["canonical_label"],word_code=r["word_code"],
            signer_id=r["signer_id"],role="train_candidate",frames=rec["frames"],
            intervals=r["intervals"],box=rec["calibration"]["box_source_pixels"],
            raw=rec["native_capture"]["raw"],assignment=ass,native=native,
            old_masked=rec["views"]["masked"]["observation"],source=rec["source"],
            retained_video=vsrc,record_path=str(rp))
        for k in ("raw","assignment","native","old_masked","record_path"):
            p=Path(row[k]);pins[str(p)]=sha(p)
            if k in ("native","old_masked"):
                sp=p.with_suffix(".sidecar.json");pins[str(sp)]=sha(sp)
        if vsrc:pins[vsrc]=sha(vsrc)
        rows.append(row)
    for p in [P/"candidate.py",F/"candidate.py",F/"CONTRACTS_V2.json",F/"verify.py",
              F/"FINAL_AUDIT.json",F/"VERIFY.json",F/"INDEX.json",
              Q/"input_candidate.py",Q/"verify.py",MODEL,
              O/"word100_fullframe_roi_pair_20261016/roi_adapter.py",
              ROOT/"research/sign_model_pilot_v1/src/pilot/data.py",
              ROOT/"research/sign_model_pilot_v1/src/pilot/web_input.py",
              B/"PREP_CONTEXT.json",U/"pipeline.py",U/"analysis.py",U/"verify.py"]:
        pins[str(p)]=sha(p)
    # Prior36 artifacts remain separate controls, never recomputed.
    for item in read(F/"INDEX.json"):pins.update(item["outputs"])
    for p in [F/"EFFECT_AUDIT.json",F/"VISUAL_REVIEW.json",P/"FROZEN.json",
              O/"word100_image_arm_connectivity_20261018/CROSSING.json",
              O/"word100_image_arm_connectivity_20261018/ANNOTATED.json"]:
        pins[str(p)]=sha(p)
    (U/"before").mkdir()
    for p in [ROOT/"WordsInSilence_PROJECT_CONTEXT.md",ROOT/"NEW_CHAT_HANDOFF.md",O/"STATE.json"]:
        shutil.copyfile(p,U/"before"/p.name)
    write(U/"FROZEN.json",dict(at=stamp(),rows=rows,pins=pins,protected=previous["protected"],
        authorization="User broader public shape-preservation audit, no training until review; keep C/T3-F and no user data.",
        selection="100word codes sorted; one NEW TRAIN clip per word, cyclic preferred signer1..9, next available excluding previous36. No score/quality selection.",
        previous_scope="36 clips reused as separate saved controls; new100 clips/100words/9TRAIN signers; not all899/1199, no external vocabulary/heldout claim.",
        rules="Frozen fullframe merge and per-finger rule imported unchanged. Same MediaPipe task/IMAGE parameters. Reuse native whole-video observation, merge current-image full/crop before fixed ROI. Crop is detection scale only.",
        comparisons=["before_merge","merged_no_shape_hold","merged_new_shape_hold"],
        normalization="Same old per-clip hand scales for both merged variants. No scale refit or point interpolation.",
        analysis="Core label interval used as task-active proxy, not proof of essential semantic phase. Count coordinate/angle losses, longest gaps, raw angle variation evidence, left/right and position invariants. No recognition model evaluation.",
        visual_policy="Systematic20 raw core-midpoint controls (every5th word in sorted100); top12 clip-distinct core-loss spans with adjacent raw frames and top removed angular-change examples; no rule changes after results. Raw panels inspected before overlay details.",
        interpretation="Discrepancy flags are not ground truth. Preservation false positives require visible evidence; no annotation means no precision/recall or all-vocabulary correctness.",
        gates=["All100 processed, no silent exclusion","Native and existing feature values unchanged",
               "Wrist/MCP body-position anchors preserved","Existing crossing53/normal3 controls reused",
               "Independent saved input/geometry checks","Actual core information losses reported, including adverse examples"],
        detector_parameters=dict(num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5),
        forbidden=["training","classifier","DEV/heldout","user/mentor recordings","camera","download","external upload","Git","production changes"]))
    print("FROZEN",len(rows),sum(r["frames"] for r in rows),flush=True)


def detect(lm,rgb,box=None):
    h,w=rgb.shape[:2]
    rr=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)))
    x0,y0=box[:2] if box else (0,0)
    out=[]
    for pp,ss in zip(rr.hand_landmarks,rr.handedness):
        d=dict(xy=[[float(p.x*w+x0),float(p.y*h+y0)] for p in pp],
               label=ss[0].category_name,score=float(ss[0].score))
        if box:d["box"]=box
        out.append(d)
    return out


def video_path(r,archives):
    if r["retained_video"]:
        p=Path(r["retained_video"]);assert sha(p)==r["source"]["sha256"];return p
    p=U/"videos"/(r["clip_id"]+".mp4")
    assert not p.exists()
    src=r["source"];archive=src["archive"]
    if archive not in archives:archives[archive]=zipfile.ZipFile(archive)
    z=archives[archive];info=z.getinfo(src["member"])
    assert info.CRC==src["crc32"] and info.file_size==src["bytes"]
    with z.open(info) as fp,p.open("xb") as out:shutil.copyfileobj(fp,out)
    assert sha(p)==src["sha256"],r["clip_id"]
    return p


def process(lm,r,video):
    cid=r["clip_id"];z=load(r["raw"]);ass=load(r["assignment"])["sides"]
    native=Observation.load(r["native"]);old=Observation.load(r["old_masked"])
    remask=roi.apply_roi(native,r["box"])
    for k in ("xy_px","usable","observed","frame_id","timestamp_us"):
        assert np.array_equal(getattr(remask,k),getattr(old,k),equal_nan=True),(cid,k)
    obs=Observation(**{k:getattr(native,k).copy() for k in
        ("xy_px","timestamp_us","frame_id","observed","usable","conf","hand_score")},sidecar=dict(native.sidecar))
    cap=cv2.VideoCapture(str(video));assert cap.isOpened()
    frames=[];fid=0;start=time.perf_counter()
    for i,target in enumerate(z["frame_id"]):
        while fid<=int(target):
            ok,bgr=cap.read();assert ok;fid+=1
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        h,w=rgb.shape[:2];assert (w,h)==(int(z["width"][i]),int(z["height"][i]))
        n=int(z["raw_count"][i]);sides=ass[i,:n].tolist()
        full=detect(lm,rgb)
        boxes=merge.crop_boxes(z["pose_xy_px"][i],z["pose_usable"][i],sides,w,h)
        crop=[]
        for side,box in boxes.items():
            for d in detect(lm,rgb[box[1]:box[3],box[0]:box[2]],box):
                d["requested_side"]=side;crop.append(d)
        decisions=merge.merge(full,crop,z["hand_xy_px"][i,:n],sides,z["pose_xy_px"][i],z["pose_usable"][i],w,h)
        for d in decisions:
            if not d["accepted"]:continue
            side=d["side"];sl=slice(33+21*side,54+21*side)
            assert not native.observed[i,sl].any()
            valid=np.array(d["joint_supported"])
            obs.xy_px[i,sl]=np.where(valid[:,None],np.array(full[d["index"]]["xy"],np.float32),np.nan)
            obs.observed[i,sl]=valid;obs.usable[i,sl]=valid
            obs.hand_score[i,side]=min(full[d["index"]]["score"],crop[d["crop_index"]]["score"])
        frames.append(dict(index=i,frame_id=int(target),timestamp_us=int(z["timestamp_us"][i]),
            core=any(a<=z["timestamp_us"][i]/1e6<=b for a,b in r["intervals"]),
            baseline_sides=sides,full=full,crop=crop,decisions=decisions))
    cap.release();obs.validate();masked=roi.apply_roi(obs,r["box"])
    saved=dict(clip_id=cid,label=r["label"],signer_id=r["signer_id"],video=str(video),frames=frames)
    ch,hs,_=channels.three_conditions(old,masked)
    prior=channels.evidence(old,masked,saved)
    xy=masked.xy_px[:,33:].reshape(-1,2,21,2);mask=masked.usable[:,33:].reshape(-1,2,21)
    oldmask=old.usable[:,33:].reshape(-1,2,21);t=len(xy);arrays={};pair_kind=np.zeros((t,2),np.uint8)
    pair_b=np.full((t,2,21,2),np.nan);pair_v=np.zeros((t,2,21),bool)
    keys=("comparable","point_uncertain","bone_available","bone_unresolved","bone_discrepancy_ratio",
          "angle_available","angle_unresolved","angle_envelope_low_deg","angle_envelope_high_deg",
          "finger_available_count","finger_uncertain","finger_missing","residual_px","aligned_b")
    available=np.zeros((t,2),bool);scale=np.full((t,2),np.nan)
    for i,fr in enumerate(frames):
        for side in range(2):
            b=vb=None;k=0
            if mask[i,side].any():
                if not oldmask[i,side].any():
                    ds=[d for d in fr["decisions"] if d["accepted"] and d["side"]==side]
                    assert len(ds)<=1
                    if ds:
                        b=np.array(fr["crop"][ds[0]["crop_index"]]["xy"])
                        vb=mask[i,side]&roi._inside(b,r["box"]);k=2
                else:
                    ds=[d for d in fr["decisions"] if d["reason"]=="confirm_existing_hand" and d["side"]==side]
                    if len(ds)==1:
                        d=ds[0];assert d["matched_old"] and all(fr["baseline_sides"][j]==side for j in d["matched_old"])
                        b=np.array(fr["full"][d["index"]]["xy"]);vb=roi._inside(b,r["box"]);k=1
            result=shape.compare_shape(xy[i,side],mask[i,side],b,vb)
            if not arrays:
                for key in keys:
                    a=result[key];fill=False if a.dtype==bool else (0 if a.dtype.kind in "iu" else np.nan)
                    arrays[key]=np.full((t,2)+a.shape,fill,dtype=a.dtype)
            for key in keys:arrays[key][i,side]=result[key]
            available[i,side]=result["pair_available"];scale[i,side]=result["alignment_scale"];pair_kind[i,side]=k
            if b is not None:pair_b[i,side]=b;pair_v[i,side]=vb
    added=prior["point_origin"]==2;hold=(arrays["point_uncertain"]|~arrays["comparable"])&added
    safe=shape.safe_new_shape_channels(ch["merge_fixed_scale"]["x"],hold,added)
    out=U/"outputs"/cid;out.mkdir()
    obs.save(str(out/"merged_native.npz"));masked.save(str(out/"merged.npz"))
    np.savez_compressed(out/"inputs.npz",before=ch["full_before"]["x"],merged=ch["merge_fixed_scale"]["x"],safe=safe,
        frame_id=masked.frame_id,timestamp_us=masked.timestamp_us,
        old_scales=np.array([hs["left"],hs["right"]],float))
    np.savez_compressed(out/"evidence.npz",**arrays,pair_kind=pair_kind,pair_b=pair_b,pair_valid=pair_v,
        pair_available=available,alignment_scale=scale,source_added=added,shape_hold_new=hold,
        **{k:prior[k] for k in ("point_origin","wrist_available","palm_available","palm_xy_px","bend_degrees","bend_available","core","frame_id","timestamp_us")})
    write(out/"detections.json",saved)
    result=dict(clip_id=cid,frames=t,seconds=time.perf_counter()-start,
        native_new_hand_frames=sum(d["accepted"] for fr in frames for d in fr["decisions"]),
        roi_new_hand_frames=int(added.any(-1).sum()),new_uncertain_hand_frames=int((arrays["finger_uncertain"].any(-1)&added.any(-1)).sum()),
        outputs={str(p):sha(p) for p in out.iterdir() if p.is_file()})
    write(out/"DONE.json",result);return result


def run():
    f=read(U/"FROZEN.json");check(f["pins"]);check(f["protected"])
    assert not (U/"RUNNING.json").exists() and not (U/"INDEX.json").exists()
    write(U/"RUNNING.json",dict(at=stamp(),pid=os.getpid()))
    (U/"outputs").mkdir();(U/"videos").mkdir();archives={};index=[]
    opts=vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE,**f["detector_parameters"])
    try:
        with vision.HandLandmarker.create_from_options(opts) as lm:
            for n,r in enumerate(f["rows"]):
                v=video_path(r,archives);result=process(lm,r,v);index.append(result)
                (U/"PROGRESS.json").write_text(json.dumps(dict(at=stamp(),complete=n+1,total=100,
                    frames=sum(x["frames"] for x in index),elapsed_seconds=sum(x["seconds"] for x in index),
                    last=r["clip_id"]),ensure_ascii=False,indent=2),encoding="utf8")
                print("DONE",n+1,100,r["clip_id"],round(result["seconds"],1),"s",flush=True)
    finally:
        for z in archives.values():z.close()
    write(U/"INDEX.json",index);check(f["pins"]);check(f["protected"]);(U/"RUNNING.json").unlink()
    print("COMPLETE100",flush=True)

if __name__=="__main__":
    cv2.setNumThreads(1)
    try:freeze() if sys.argv[1]=="freeze" else run()
    except BaseException:
        if not (U/"FIRST_ERROR.json").exists():write(U/"FIRST_ERROR.json",dict(at=stamp(),error=traceback.format_exc()))
        raise
