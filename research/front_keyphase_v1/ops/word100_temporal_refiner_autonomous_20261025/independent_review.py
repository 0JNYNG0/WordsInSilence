"""Independent checks and image-only diagnostic sampling, never training.

The visible sample is selected from source metadata before predicted hand
coordinates are read. It is a diagnostic sample, not anatomical ground truth.
"""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import importlib.util
import numpy as np

UNIT = Path(__file__).resolve().parent
OPS = UNIT.parent
BUNDLE = OPS / "word100_temporal_public_bundle_20261025"
PILOT = OPS / "word100_temporal_refiner_pilot_20261025"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def prepare_visible():
    rows = read(BUNDLE / "MANIFEST.json")
    prior = read(PILOT / "VISIBLE_SAMPLE_SELECTION.json")
    old_frames = {(r["clip_id"], r["frame_id"]) for r in prior["cases"]}
    chosen = []
    for person in range(1, 7):
        candidates = [r for r in rows if r["signer_id"] == person]
        candidates.sort(key=lambda r: hashlib.sha256(("independent-visible-20261025:" + r["clip_id"]).encode()).hexdigest())
        assert len(candidates) >= 2
        for order, row in enumerate(candidates[:2]):
            start, end = max(row["evaluation_annotation_intervals_only"], key=lambda a: a[1]-a[0])
            fraction = (order + 1) / 3.0
            frame = min(row["frames"]-1, max(0, round((start+(end-start)*fraction)*row["frame_cadence_fps"])))
            while (row["clip_id"], frame) in old_frames:
                frame += 1
                assert frame < row["frames"]
            chosen.append({"sample_id": f"N{len(chosen)+1:02d}", "clip_id": row["clip_id"],
                "signer_id": person, "frame_id": frame, "anatomical_side": order,
                "side_name": "left" if order == 0 else "right", "rgb_path": row["rgb_path"],
                "source_sha256": row["rgb_sha256"], "source_dimensions": row["video_dimensions"],
                "selection_fraction_of_longest_annotated_interval": fraction})
    result = {"status": "SELECTED_BEFORE_IMAGE_OR_PREDICTION_REVIEW", "at": datetime.now().astimezone().isoformat(),
        "rule": "TRAIN01-06 two distinct clips each by fixed SHA256 ordering; first/second at1/3 and2/3 of longest source interval; anatomical left/right; avoid prior24 exact frames",
        "purpose": "Image-only coarse wrist region diagnostic; never training target, rule calibration, or independent final accuracy test",
        "test_signers_accessed": [], "coordinates_read_during_selection": False,
        "selection": chosen, "image_only_labels": [], "not_ground_truth": True}
    folder = UNIT / "independent_visible"
    folder.mkdir(exist_ok=False)
    # Pin selection before decoding images or inspecting any predictions.
    write_new(folder / "SELECTION.json", result)
    import cv2
    for row in chosen:
        assert sha(row["rgb_path"]) == row["source_sha256"]
        cap = cv2.VideoCapture(row["rgb_path"])
        assert cap.isOpened()
        image = None
        try:
            for frame in range(row["frame_id"]+1):
                ok, image = cap.read()
                assert ok
        finally:
            cap.release()
        assert [image.shape[1], image.shape[0]] == row["source_dimensions"]
        dest = folder / (row["sample_id"] + ".png")
        assert cv2.imwrite(str(dest), image)
        row["image_path"] = str(dest)
        row["image_sha256"] = sha(dest)
    result["status"] = "IMAGES_READY_FOR_BLIND_REVIEW"
    write_new(UNIT / "VISIBLE_INDEPENDENT_NEW.json", result)
    print(json.dumps({"images": len(chosen), "selection": [{k:r[k] for k in ("sample_id","clip_id","frame_id","side_name")} for r in chosen]}, ensure_ascii=False, indent=2))


def module_at(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_weights(base, policy):
    """Separate arithmetic implementation; no calls to teacher internals."""
    xy = base["xy_px"].astype(np.float64)
    valid = base["usable"] & np.isfinite(xy).all(axis=2)
    shoulder_ok = valid[:,11] & valid[:,12]
    # Match the frozen source's float32 body-scale convention, then compute
    # residuals independently in float64. Promoting the scale first changes
    # its final bits without changing any quality decisions.
    lengths = np.sqrt(np.square(base["xy_px"][:,11]-base["xy_px"][:,12]).sum(axis=1))
    scale = float(np.median(lengths[shoulder_ok & (lengths > 1)]))
    rules = policy["thresholds"]
    result = valid.astype(np.float32)
    geometry = np.zeros(valid.shape, bool)
    wrist_concern = np.zeros(valid.shape, bool)
    wrist_values = []
    temporal_values = []
    for side, offset in enumerate((33,54)):
        body_wrist = 15+side
        ok = valid[:,body_wrist] & valid[:,offset] & base["conf_available"][:,body_wrist] & (base["conf"][:,body_wrist] >= .5)
        distances = np.sqrt(np.square(xy[:,body_wrist]-xy[:,offset]).sum(axis=1))/scale
        wrist_values.extend(distances[ok].tolist())
        wrist_concern[:,offset:offset+21] = (ok & (distances > rules["wrist_disagreement_shoulder_units"]))[:,None]
        for start in (1,5,9,13,17):
            chain = [0,start,start+1,start+2,start+3]
            for a,b in zip(chain[:-1], chain[1:]):
                usable = valid[:,offset+a] & valid[:,offset+b]
                length = np.sqrt(np.square(xy[:,offset+a]-xy[:,offset+b]).sum(axis=1))/scale
                bad = usable & ((length < rules["projected_bone_short_shoulder_units"]) | (length > rules["projected_bone_long_shoulder_units"]))
                geometry[:,offset+a] |= bad
                geometry[:,offset+b] |= bad
    temporal_large = np.zeros(valid.shape, bool)
    time = base["timestamp_us"].astype(np.float64)
    for i in range(1, len(xy)-1):
        usable = valid[i-1] & valid[i] & valid[i+1]
        ratio = (time[i]-time[i-1])/(time[i+1]-time[i-1])
        expected = (1-ratio)*xy[i-1]+ratio*xy[i+1]
        distance = np.sqrt(np.square(xy[i]-expected).sum(axis=1))/scale
        temporal_large[i] = usable & (distance > rules["isolated_excursion_shoulder_units"])
        temporal_values.extend(distance[33:][usable[33:]].tolist())
    concern = wrist_concern | geometry
    result[valid & concern] = .75
    result[valid & concern & temporal_large] = .5
    return result, wrist_values, temporal_values


def review():
    quality = module_at(UNIT/"teacher_reliability.py", "independent_teacher_under_review")
    loader = module_at(BUNDLE/"load_inputs.py", "independent_input_loader")
    policy = read(UNIT/"TEACHER_POLICY.json")
    assert policy["implementation_sha256"] == sha(UNIT/"teacher_reliability.py")
    assert policy["train_signer_ids"] == [1,2,3,4,5,6]
    rows = read(BUNDLE/"MANIFEST.json")
    expected_train = {r["clip_id"] for r in rows if r["signer_id"] <= 6}
    assert {r["clip_id"] for r in policy["threshold_fit_clips"]} == expected_train
    all_wrist, all_time = [], []
    totals = {"weight_1":0,"weight_075":0,"weight_050":0,"valid_hand_targets":0}
    for i,row in enumerate(rows):
        base = loader.load_observations(i,"B0")
        before = {k:v.copy() for k,v in base.items()}
        expected,w,e = independent_weights(base,policy)
        actual = quality.weights_for(base,policy)
        assert np.array_equal(actual,expected), (row["clip_id"],"independent arithmetic mismatch")
        for k in before:
            assert np.array_equal(before[k],base[k],equal_nan=True)
        h = actual[:,33:]
        totals["weight_1"] += int((h==1).sum())
        totals["weight_075"] += int((h==.75).sum())
        totals["weight_050"] += int((h==.5).sum())
        totals["valid_hand_targets"] += int((h>0).sum())
        if row["signer_id"] <= 6:
            all_wrist.extend(w); all_time.extend(e)
    assert np.isclose(np.quantile(all_wrist,.99),policy["fitting"]["wrist_train_q99"],rtol=0,atol=1e-12)
    assert np.isclose(np.quantile(all_time,.99),policy["fitting"]["excursion_train_q99"],rtol=0,atol=1e-12)
    assert len(all_wrist)==policy["fitting"]["wrist_samples"] and len(all_time)==policy["fitting"]["excursion_samples"]
    audit = read(UNIT/"TEACHER_AUDIT.json")
    assert all(audit["totals"][k]==v for k,v in totals.items())
    # Known-geometry regression: crossing and fast reversals do not imply bad labels.
    b={"xy_px":np.zeros((5,75,2),np.float32),"usable":np.zeros((5,75),bool),
       "conf":np.ones((5,75),np.float32),"conf_available":np.ones((5,75),bool),
       "timestamp_us":np.arange(5,dtype=np.int64)*33333}
    b["xy_px"][:,11]=[400,200];b["xy_px"][:,12]=[600,200];b["usable"][:,[11,12,15,16]]=True
    for side,offset in enumerate((33,54)):
        for t,x in enumerate(([350,550,750,550,350] if side==0 else [750,550,350,550,750])):
            wrist=np.array([x,400.],np.float32);b["xy_px"][t,15+side]=wrist
            b["xy_px"][t,offset]=wrist
            for finger,start in enumerate((1,5,9,13,17)):
                for joint in range(4):
                    b["xy_px"][t,offset+start+joint]=wrist+[5+finger*5,10+joint*10]
        b["usable"][:,offset:offset+21]=True
    assert np.all(quality.weights_for(b,policy)[:,33:]==1), "Crossing/fast reversal alone discounted"
    missing={k:v.copy() for k,v in b.items()};missing["usable"][2,41]=False
    w=quality.weights_for(missing,policy)
    assert w[2,41]==0 and np.all(w[:,33:][missing["usable"][:,33:]]>=.5)
    glossary=read(BUNDLE/"GLOSS_REFERENCE_INDEX.json")["glossary"]
    assert all(r["phonological_attributes"] is None for r in glossary)
    assert all(r["pointwise_anatomical_ground_truth"]=="NOT_AVAILABLE" and r["hidden_3d_ground_truth"]=="NOT_AVAILABLE" for r in rows)
    visible=read(UNIT/"VISIBLE_INDEPENDENT_NEW.json")
    assert visible["status"]=="FROZEN_IMAGE_ONLY_DIAGNOSTIC" and len(visible["image_only_labels"])==12
    assert not visible["prediction_coordinates_read_before_freeze"]
    result={"status":"PASS_FOR_BOUNDED_ROBUST_PROXY_EXPERIMENT_NOT_ANATOMICAL_GT",
        "at":datetime.now().astimezone().isoformat(),"reviewer":"temporal_bundle_inventory",
        "files_reviewed":{n:sha(UNIT/n) for n in ("train_compare.py","refiner_model.py","rgb_features.py","teacher_reliability.py","TEACHER_POLICY.json")},
        "independent_teacher_arithmetic":{"clips":len(rows),"train_fit_clips":len(expected_train),"train_wrist_samples":len(all_wrist),"train_excursion_samples":len(all_time),"counts":totals},
        "synthetic_regressions":{"normal_crossing_and_fast_reversal_retained":True,"missing_joint_not_supervised_as_zero":True},
        "static_contract_review":{"synthetic_mask_after_B1":True,"hidden_hand_not_RGB_crop_center":True,"cached_original_not_recovery_prediction":True,"teacher_weight_loss_only_not_input":True,"test08_09_no_rule_fit_or_epoch_selection":True,"DEV07_only_epoch_selection":True,"unweighted_wrist_and_relative_shape_reported":True,"known_coordinates_preserved":True,"natural_estimates_not_observations":True,"whole_ROI_crop_fallback_explicit":True},
        "resolved_hold_scope":"Prior pilot hold remains unchanged. This is a new robust pseudo-target experiment, not a declaration that suspect teacher coordinates became correct.",
        "known_unresolved":"The prior V05 image mismatch can retain teacher weight1 because body/hand errors are correlated. No V05-specific threshold change is justified. Robust loss bounds influence but does not establish anatomical truth.",
        "visible_diagnostic":{"new_images":12,"train_people_only":[1,2,3,4,5,6],"labels_sha256":sha(UNIT/"VISIBLE_INDEPENDENT_NEW.json"),"use":"diagnostic only; not training/threshold tuning/final accuracy","new_wrist_fully_occluded_cases":0,"limits":"One AI reviewer coarse2D wrist ranges. Contact/overlap2 and low/lowered hands2, but no precise21-joint or hidden3D truth."},
        "stage4_optional_handshape_supervision":{"ready":False,"evidence":{"current_bundle_61_independent21point_truth":0,"glossary100_phonological_annotations":0,"new_visible12_finger_truth":0},"decision":"Do not force optional handshape training; Gloss/word labels do not uniquely label handshape. This is a current-bundle evidence limit, not a claim no suitable dataset exists anywhere."},
        "runtime_preflight":"Pending root execution; this review does not report model-forward/training success",
        "training":0,"ML_inference":0,"user_recording_access":0,"blocking_findings_remaining":[]}
    write_new(UNIT/"REVIEW.json",result)
    print(json.dumps({"status":result["status"],"counts":totals,"new_images":12,"stage4_ready":False},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare-visible","review"))
    args = parser.parse_args()
    if args.command == "prepare-visible":
        prepare_visible()
    elif args.command == "review":
        review()
