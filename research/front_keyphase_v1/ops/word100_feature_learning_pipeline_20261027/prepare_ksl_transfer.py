"""Read-only public KSL transfer inputs. Existing AI drafts are not training targets."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

U=Path(__file__).resolve().parent;OPS=U.parent;ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=U/"data/KSL_TRANSFER.json";assert not out.exists()
    bundlepath=OPS/"word100_shape_sequence_bundle_20261027/BUNDLE.json";bundle=read(bundlepath)
    fpath=OPS/"word100_shared_feature_bundle_20261027/FROZEN.json";records={r["clip_id"]:r for r in read(fpath)["rows"]}
    folder=U/"data/ksl_hand_crops";folder.mkdir(exist_ok=True)
    model=ROOT/"Tools/models/hand_landmarker.task"
    protocol={"bundle_sha256":sha(bundlepath),"source_records_sha256":sha(fpath),"hand_model_sha256":sha(model),
       "scope":"Public KSL36-window descriptive transfer only; no fitting, tuning, candidate selection, educational verdict",
       "crop":"Existing whole-frame native observed21-hand bounding box;25%margin each direction,clamp image; absent observed hand is missing crop, not inferred fill",
       "detector":"Same MediaPipe IMAGE default0.5,2candidate setting as external data; closest detected palm to expected native palm in crop",
       "uncertainty":"Two candidates within0.02normalized palm distance flag; preserve all existing whole-frame inputs unchanged",
       "drafts":"Existing direct2features only descriptive agreement; not independent physical or linguistic truth",
       "source_target_side":"Inherited existing native assignment, not newly human-certified identity",
       "all13frames_kept":True,"training_allowed":False}
    (U/"KSL_TRANSFER_PROTOCOL.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    detector=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)),running_mode=vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5))
    rows=[];xyz=[];observed=[];start=time.perf_counter();pins={}
    try:
        for seq in bundle["sequences"]:
            sr=records[seq["clip_id"]];rp=Path(sr["record"]);assert sha(rp)==sr["record_sha256"]
            record=read(rp);p=Path(record["views"]["native"]["observation"]);sc=read(p.with_suffix('.sidecar.json'))
            assert sha(p)==sc["observation_sha256"];pins[str(p)]=sha(p)
            with np.load(p,allow_pickle=False) as z:xy=z["xy_px"].copy();mask=z["usable"].copy();timestamps=z["timestamp_us"].copy()
            side={"L":0,"R":1}[seq["side"]];off=33+21*side
            for fr in seq["frames"]:
                fi=fr["frame_id"];assert abs(int(timestamps[fi])-fr["pts_us"])<=1
                image=Image.open(fr["full_image"]).convert('RGB');w,h=image.size
                points=xy[fi,off:off+21];valid=mask[fi,off:off+21]&np.isfinite(points).all(-1)
                r={"case_id":seq["case_id"],"clip_id":seq["clip_id"],"word_id":seq["word_id"],"side":seq["side"],
                   "person":seq["recorded_person_id"],"frame_id":fi,"pts_us":fr["pts_us"],"full_image":fr["full_image"],
                   "full_image_sha256":sha(fr["full_image"]),"draft_only":fr["direct"],"source_observed_points":int(valid.sum()),"image":None,
                   "detected":False,"identity_uncertain":False,"identity_source":"inherited_native_assignment_not_independent_truth"}
                coords=np.zeros((21,3),np.float32);seen=False
                if valid.all():
                    lo=points.min(0);hi=points.max(0);size=np.maximum(hi-lo,32.);lo=np.floor(lo-.25*size).astype(int);hi=np.ceil(hi+.25*size).astype(int)
                    lo=np.maximum(lo,0);hi=np.minimum(hi,[w,h]);box=(*lo,*hi)
                    if np.all(hi>lo):
                        crop=image.crop(tuple(int(x) for x in box));dest=folder/f'{seq["case_id"]}_{fi}.png'
                        assert not dest.exists();crop.save(dest);r.update({"image":str(dest),"image_sha256":sha(dest),"crop_xyxy":list(map(int,box))})
                        found=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(crop).copy()))
                        candidates=[np.array([[p.x,p.y,p.z] for p in hand],np.float32) for hand in found.hand_landmarks]
                        expected=(points[[0,5,9,13,17]].mean(0)-lo)/(hi-lo)
                        distances=[float(np.linalg.norm(c[[0,5,9,13,17],:2].mean(0)-expected)) for c in candidates]
                        r["candidates"]=[c.tolist() for c in candidates];r["candidate_palm_distances"]=distances
                        if candidates:
                            ix=int(np.argmin(distances));coords=candidates[ix];seen=bool(np.isfinite(coords).all())
                            r["selected"]=ix;r["detected"]=seen
                            if len(candidates)>1:r["identity_uncertain"]=abs(distances[0]-distances[1])<=.02
                rows.append(r);xyz.append(coords);observed.append(seen)
    finally:detector.close()
    assert len(rows)==468
    np.savez_compressed(U/"data/KSL_TRANSFER_JOINTS.npz",raw_xyz=np.array(xyz),observed=np.array(observed))
    result={"status":"DESCRIPTIVE_PUBLIC_TRANSFER_INPUTS_ONLY","rows":rows,"counts":{"frames":468,"windows":36,"crops":sum(bool(r['image']) for r in rows),"redetected":sum(observed),"ambiguous":sum(r['identity_uncertain'] for r in rows)},
            "source_pins":pins,"protocol_sha256":sha(U/"KSL_TRANSFER_PROTOCOL.json"),"elapsed_seconds":time.perf_counter()-start,"training":False,"model_selection_use":False}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(result['counts']))

if __name__=="__main__":main()
