"""Mask-only fixed selection, then raw public RGB rendering for blind review.

Does not read predicted coordinates, confidence scores, classifier outputs or
pixel values during selection. Blinded panels contain no prediction overlay.
No training, detector inference, classification, or user/mentor recording access.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

UNIT=Path(__file__).resolve().parent
OPS=UNIT.parent
BUNDLE=OPS/"word100_temporal_public_bundle_20261025"
PILOT=OPS/"word100_temporal_refiner_pilot_20261025"
PREVIOUS=OPS/"word100_temporal_refiner_autonomous_20261025"
PREFIX="public-recovery-visible-benchmark-20261025-v1:"
TARGETS={"A":8,"B":8,"C":4,"D":4}

def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()
def digest(text):return hashlib.sha256(text.encode("utf-8")).hexdigest()
def write_new(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def runs(flags):
    edges=np.diff(np.r_[False,flags,False].astype(np.int8))
    return zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))

def select():
    old24=PILOT/"VISIBLE_SAMPLE_SELECTION.json"
    old12=PREVIOUS/"independent_visible/SELECTION.json"  # before labels, no coordinate answers
    pin_paths=[Path(__file__),BUNDLE/"MANIFEST.json",old24,old12]
    excluded={(r["clip_id"],r["frame_id"]) for r in read(old24)["cases"]}
    excluded.update((r["clip_id"],r["frame_id"]) for r in read(old12)["selection"])
    assert len(excluded)==36
    columns=("clip_id","signer_id","arrays","arrays_sha256","frames","rgb_path",
             "rgb_sha256","video_dimensions","selection_history")
    rows=[{k:r[k] for k in columns} for r in read(BUNDLE/"MANIFEST.json") if 1<=r["signer_id"]<=6]
    assert len(rows)==39
    source_pins={str(p.resolve()):sha(p) for p in pin_paths}
    candidates={k:[] for k in TARGETS};decisions=[];field_reads=Counter()
    for row in rows:
        path=Path(row["arrays"]);assert sha(path)==row["arrays_sha256"]
        source_pins[str(path.resolve())]=row["arrays_sha256"]
        with np.load(path,allow_pickle=False) as z:
            # Explicit allowlist: no xy/conf/hand_score/source-derived error.
            b0=z["baseline_roi_usable"].copy();b1=z["redetected_roi_usable"].copy()
            ids=z["frame_id"].copy();timestamps=z["timestamp_us"].copy()
        for k in ("baseline_roi_usable","redetected_roi_usable","frame_id","timestamp_us"):field_reads[k]+=1
        estimate_path=PREVIOUS/"runs/B2_seed17/natural_estimates"/(row["clip_id"]+".npz")
        source_pins[str(estimate_path.resolve())]=sha(estimate_path)
        with np.load(estimate_path,allow_pickle=False) as z:
            estimated=z["estimated"].copy();observed=z["usable_observed"].copy();estimate_ids=z["frame_id"].copy()
        for k in ("estimated","usable_observed","frame_id"):field_reads["candidate_"+k]+=1
        n=row["frames"]
        assert b0.shape==b1.shape==estimated.shape==(n,75)
        assert np.array_equal(ids,np.arange(n)) and np.array_equal(ids,estimate_ids)
        assert np.array_equal(observed,b1) and not np.any(estimated&b1)
        for side in (0,1):
            wrist=33+side*21;hand=slice(wrist,wrist+21);fingers=slice(wrist+1,wrist+21)
            masks={"A":~b0[:,wrist]&b1[:,wrist],
                   "B":~b1[:,wrist]&estimated[:,wrist],
                   "C":b1[:,wrist]&((~b1[:,fingers])&estimated[:,fingers]).any(axis=1),
                   "D":b0[:,hand].all(axis=1)&b1[:,hand].all(axis=1)&~estimated[:,hand].any(axis=1)}
            for stratum,flags in masks.items():
                for start,stop in runs(flags):
                    start,stop=int(start),int(stop);middle=(start+stop-1)//2
                    key=f"{stratum}|{row['clip_id']}|{side}|{start}|{stop}"
                    item=dict(candidate_id=digest(PREFIX+key),stratum=stratum,
                        clip_id=row["clip_id"],signer_id=row["signer_id"],anatomical_side=side,
                        interval_half_open=[start,stop],frame_id=middle,
                        timestamp_us=int(timestamps[middle]),context_frames=[middle-3,middle,middle+3],
                        historical_crossing_metadata="crossing_control" in row["selection_history"],
                        rgb_path=row["rgb_path"],rgb_sha256=row["rgb_sha256"],source_dimensions=row["video_dimensions"])
                    reason=None
                    if middle<3 or middle+3>=n:reason="three_frame_context_outside_source"
                    elif any((row["clip_id"],i) in excluded for i in item["context_frames"]):reason="prior36_exact_frame_in_center_or_context"
                    decisions.append(dict(candidate_id=item["candidate_id"],stratum=stratum,
                        clip_id=row["clip_id"],anatomical_side=side,interval_half_open=[start,stop],
                        frame_id=middle,eligible=reason is None,exclusion_reason=reason))
                    if reason is None:candidates[stratum].append(item)
    selected=[];used_clip=Counter();used_person=Counter();used_side=Counter();used_frames=set()
    def pick(stratum,require_cross=False):
        available=[c for c in candidates[stratum] if used_clip[c["clip_id"]]<2
                   and (c["clip_id"],c["frame_id"]) not in used_frames
                   and (not require_cross or c["historical_crossing_metadata"])]
        if not available:return None
        within=Counter(c["signer_id"] for c in selected if c["stratum"]==stratum)
        available.sort(key=lambda c:(within[c["signer_id"]],used_person[c["signer_id"]],
            used_side[(c["clip_id"],c["anatomical_side"])],used_clip[c["clip_id"]],c["candidate_id"]))
        c=available[0];selected.append(c);used_clip[c["clip_id"]]+=1;used_person[c["signer_id"]]+=1
        used_side[(c["clip_id"],c["anatomical_side"])]+=1;used_frames.add((c["clip_id"],c["frame_id"]))
        return c
    cross_selected=pick("D",True) is not None
    while sum(c["stratum"]=="D" for c in selected)<TARGETS["D"]:
        if pick("D") is None:break
    for stratum in ("A","B","C"):
        for _ in range(TARGETS[stratum]):
            if pick(stratum) is None:break
    selected.sort(key=lambda c:digest(PREFIX+"blind-order:"+c["candidate_id"]))
    for i,c in enumerate(selected,1):
        c["sample_id"]=f"R{i:02d}"
        c["image_path"]=str(UNIT/"images"/(c["sample_id"]+".png"))
        c["grid_path"]=str(UNIT/"images"/(c["sample_id"]+"_grid.png"))
        c["triptych_path"]=str(UNIT/"triptychs"/(c["sample_id"]+".jpg"))
    counts=Counter(c["stratum"] for c in selected)
    assert len(selected)<=24 and max(used_clip.values(),default=0)<=2
    assert len({(c["clip_id"],c["frame_id"]) for c in selected})==len(selected)
    for c in selected:
        assert 1<=c["signer_id"]<=6
        assert all((c["clip_id"],f) not in excluded for f in c["context_frames"])
    return dict(schema="wis_mask_selected_recovery_image_benchmark_v1",status="FROZEN_BEFORE_RGB_DECODE_OR_PREDICTION_COORDINATE_REVIEW",
        seed_prefix=PREFIX,selection_uses_only=["boolean usability/estimate masks","frame metadata","signer metadata","historical crossing selection tag","fixed SHA sorting"],
        prohibited_selection_inputs=["predicted coordinates","confidence","classifier logits","wrist/shape error","pixel appearance","model outcomes"],
        rule="Each contiguous mask interval contributes only its lower middle frame. Three source frames t-3,t,t+3 must all avoid prior36 exact frames. ReserveD first incl1 historical-crossing-tag if available, then A,B,C. Within stratum prefer least represented person then overall person, unusedclip/side, clipcount, SHA. Global clipmax2, no repeated clip/frame. No replacement after visual review.",
        strata={"A":"B0 wrist absent -> B1 wrist usable","B":"B1 wrist absent -> B2seed17 wrist estimated",
                "C":"B1 wrist usable, one or more finger points absent -> B2seed17 estimates",
                "D":"all original21 hand points usable and retained, no estimates; metadata-crossing control when eligible"},
        requested=TARGETS,eligible_intervals={k:len(v) for k,v in candidates.items()},
        selected_counts={k:counts[k] for k in TARGETS},shortfalls={k:TARGETS[k]-counts[k] for k in TARGETS},
        selected_people=dict(used_person),selected_clips=len(used_clip),selected_scenes=len(selected),
        historical_crossing_control_included=cross_selected,field_reads=dict(field_reads),
        prior_exact_frames=sorted([dict(clip_id=c,frame_id=f) for c,f in excluded],key=lambda x:(x["clip_id"],x["frame_id"])),
        selection=selected,eligibility_decisions=decisions,source_pins=source_pins,
        limitations=["Masks identify computational recovery, not true visibility or correctness.","SelectedTRAIN people were previously inspected; this is independent image assessment, not unseen-person validation.","ExistingV05 is excluded and retained only as prior diagnostic, never fresh validation.","No labels or images from this benchmark may train/tune/select models."],
        training=0,model_inference=0,user_recording_access=0)

def render(selection):
    import cv2
    images=UNIT/"images";triptychs=UNIT/"triptychs";images.mkdir();triptychs.mkdir()
    output=[]
    for row in selection["selection"]:
        assert sha(row["rgb_path"])==row["rgb_sha256"]
        cap=cv2.VideoCapture(row["rgb_path"]);assert cap.isOpened()
        needed=set(row["context_frames"]);frames={}
        try:
            for frame in range(max(needed)+1):
                ok,image=cap.read();assert ok,(row["sample_id"],frame,"decode failed")
                if frame in needed:frames[frame]=image.copy()
        finally:cap.release()
        current=frames[row["frame_id"]];h,w=current.shape[:2]
        assert [w,h]==row["source_dimensions"]==[1920,1080]
        assert cv2.imwrite(row["image_path"],current)
        grid=current.copy()
        for x in range(0,w,100):
            cv2.line(grid,(x,0),(x,h-1),(160,160,160),1)
            cv2.putText(grid,str(x),(x+3,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        for y in range(0,h,100):
            cv2.line(grid,(0,y),(w-1,y),(160,160,160),1)
            cv2.putText(grid,str(y),(3,min(y+18,h-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        assert cv2.imwrite(row["grid_path"],grid)
        panels=[]
        for offset,frame in zip((-3,0,3),row["context_frames"]):
            panel=cv2.resize(frames[frame],(960,540),interpolation=cv2.INTER_AREA)
            panel=cv2.copyMakeBorder(panel,32,0,0,0,cv2.BORDER_CONSTANT,value=(24,24,24))
            text=f"{row['sample_id']} | anatomical {'LEFT' if row['anatomical_side']==0 else 'RIGHT'} | t{offset:+d}"
            cv2.putText(panel,text,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
            panels.append(panel)
        assert cv2.imwrite(row["triptych_path"],np.concatenate(panels,axis=1),[cv2.IMWRITE_JPEG_QUALITY,94])
        output.append(dict(sample_id=row["sample_id"],anatomical_side=row["anatomical_side"],
            side_name="left" if row["anatomical_side"]==0 else "right",original_dimensions=[w,h],
            triptych_dimensions=[2880,572],context_offsets_frames=[-3,0,3],
            image_path=row["image_path"],grid_path=row["grid_path"],triptych_path=row["triptych_path"],
            image_sha256=sha(row["image_path"]),grid_sha256=sha(row["grid_path"]),triptych_sha256=sha(row["triptych_path"])))
    for path,pin in selection["source_pins"].items():assert sha(path)==pin,(path,"changed during prepare")
    blind=dict(status="READY_FOR_BLIND_IMAGE_ONLY_REVIEW",review_instructions="Open only this index and its images. No stratum, detector availability, predictions, scores, word label or clip filename supplied. Judge visible anatomy, occlusion and handshape uncertainty independently; do not invent hidden joints.",
               selection_sha256=sha(UNIT/"SELECTION.json"),schema_version=1,scenes=output)
    write_new(UNIT/"BLIND_INDEX.json",blind)
    write_new(UNIT/"IMAGE_CHECK.json",dict(status="PASS",scenes=len(output),image_files=len(output)*3,
        source_dimensions=[1920,1080],triptych_dimensions=[2880,572],prediction_overlays=False,
        selection_frozen_before_decode=True,all_source_pins_unchanged=True,model_inference=0,training=0,
        selection_sha256=blind["selection_sha256"],blind_index_sha256=sha(UNIT/"BLIND_INDEX.json")))

def main():
    selection=select()
    write_new(UNIT/"SELECTION.json",selection)  # irrevocable selection before pixels are viewed
    write_new(UNIT/"SOURCE_PINS.json",selection["source_pins"])
    write_new(UNIT/"SCHEMA.json",dict(schema_version=1,
        evaluator_only="SELECTION.json contains strata/masks/source clip identifiers; keep hidden until independent image labels freeze.",
        blinded="BLIND_INDEX.json provides sample ID, anatomical side, original/triptych image paths and hashes only.",
        images="Original current RGB1920x1080 PNG; separate originalpixel100pxgrid; t-3/t/t+3 triptychJPEG2880x572 with each image scaled1/2.",
        future_labels="Mark occlusion, wrist interval/point with uncertainty and visible fingers first. Freeze both reviewers before revealing predictions.",
        insufficiency="Unsupported/ambiguous scenes stay in denominator as unevaluable; never replace after viewing results.",
        training_use=False))
    render(selection)
    print(json.dumps({k:selection[k] for k in ("selected_scenes","selected_counts","eligible_intervals","shortfalls","selected_people","selected_clips")},ensure_ascii=False))

if __name__=="__main__":main()
