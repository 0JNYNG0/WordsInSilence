"""Prepare fixed public SHuBERT feasibility inputs; no inference or training.

Existing original-frame MediaPipe observations are retained verbatim. Official
crop geometry and body normalization are loaded as AST function definitions,
without importing the official scripts or their command-line side effects.
Missing hand images use the official previous-crop fallback, separately marked
as stale input rather than a newly observed hand.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np


UNIT = Path(__file__).resolve().parent
OPS = UNIT.parent
ROOT = OPS.parents[2]
BUNDLE = OPS / "word100_shared_feature_bundle_20261027"
FACE_BASE = OPS / "word100_face_body_stage_20261027/data"
VENDOR = UNIT / "vendor/shubert"
OUT = UNIT / "inputs"
ALIASES = [f"V{i:02d}" for i in range(1, 13)]
BODY_IDS = [0, 11, 12, 13, 14, 15, 16]
BODY_REQUIRED = [0, 2, 11, 12, 13, 14, 15, 16]
HAND_OFFSETS = [33, 54]
STREAM_NAMES = ["left_hand", "right_hand", "face"]
CURRENT, PREVIOUS_COPY, BLANK = 1, 2, 0


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def numeric_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def official_functions(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    namespace = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def load_official():
    hands_path = VENDOR / "dataset/crop_hands.py"
    face_path = VENDOR / "dataset/crop_face.py"
    body_path = VENDOR / "features/body_features.py"
    hands = official_functions(hands_path, ["get_bounding_box", "adjust_bounding_box", "crop_frame", "resize_frame"])
    face = official_functions(face_path, ["calculate_bounding_box", "cues_on_grey_background", "resize_frame"])
    body = official_functions(body_path, ["normalize_pose_keypoints"])
    return hands, face, body


def frozen_selection():
    frozen_path = BUNDLE / "FROZEN.json"
    index_path = BUNDLE / "visual_inputs/INDEX.json"
    frozen, index = read(frozen_path), read(index_path)
    by_alias = {r["alias"]: r for r in index["rows"]}
    chosen = {r["alias"]: r for r in frozen["rows"] if r["alias"] in ALIASES}
    assert len(chosen) == 12
    rows = []
    for alias in ALIASES:
        row = dict(chosen[alias])
        visual = by_alias[alias]
        assert row["clip_id"] == visual["clip_id"]
        assert row["recorded_person_id"] in [1, 2, 3]
        source = Path(visual["source"])
        # Some public clips were already extracted into the earlier frozen
        # public face unit. Authorize by the pinned AIHub archive provenance
        # and byte hash, not by requiring the extracted cache to remain on W:.
        assert "AIHub_103" in Path(row["source"]["archive"]).parts
        assert source.suffix.lower() == ".mp4"
        assert visual["source_sha256"] == row["source"]["sha256"]
        record = Path(row["record"])
        assert sha(record) == row["record_sha256"]
        rec = read(record)
        face = FACE_BASE / row["clip_id"] / "face_roi.npz"
        assert source.exists() and face.exists()
        assert sha(source) == visual["source_sha256"]
        row.update(source_video=str(source), source_video_sha256=visual["source_sha256"],
                   native_observation=rec["views"]["native"]["observation"],
                   masked_observation=rec["views"]["masked"]["observation"],
                   face_observation=str(face), source_record=rec)
        rows.append(row)
    return rows, {str(frozen_path): sha(frozen_path), str(index_path): sha(index_path)}


def protocol(rows, pins):
    return {
        "schema": "public_shubert_inputs_v1",
        "scope": "Frozen feature-extraction feasibility only; no semantic or educational accuracy claim",
        "fixed_aliases": ALIASES,
        "public_clips": 12, "expressions": 4, "recorded_people": [1, 2, 3],
        "whole_source_video": True, "sampling": "Every original 30 fps frame; no sparse16 substitute",
        "source_manifest_pins": pins,
        "official_sources": {str(p): sha(p) for p in [VENDOR / "dataset/crop_hands.py", VENDOR / "dataset/crop_face.py", VENDOR / "features/body_features.py"]},
        "crop_geometry": {
            "hands": "Official get_bounding_box(scale_factor=1.5), adjust_bounding_box, INTER_AREA resize224",
            "face": "Official eye and mouth rectangles on grey128 background, INTER_AREA resize224; no expression label or verdict",
            "color": "RGB uint8 NHWC decoded by PyAV. Official decord RGB -> cvtColor -> OpenCV mp4 writer returns RGB after video decoding; we avoid lossy intermediate encoding.",
            "hand_current_gate": "All 21 existing native observed & usable & finite points; fixed existing anatomical L/R slots, no reassignment",
            "missing": "Previous RGB crop copied exactly, or black before any observation; source_kind and origin_frame_index identify this as imputed/stale image input",
        },
        "body14": "Official normalize_pose_keypoints on full-frame x/width,y/height; select0,11,12,13,14,15,16; float32. Nose,left-eye,shoulders,elbows,wrists must be observed/usable; otherwise official prior-vector/-9999 fallback separately marked.",
        "source_kind": {"0": "blank_or_body_sentinel", "1": "current_saved_observation", "2": "previous_observation_copy_NOT_current_evidence"},
        "intentional_official_differences": [
            "No person YOLO crop; saved full-frame MediaPipe extraction reused, not official detector rerun.",
            "Existing corrected native L/R identities retained. Official independent nearest-pose-wrist select_hands is not rerun because it may select one candidate for both hands.",
            "Conservative current-hand availability checks saved observed and usable masks; official JSON presence alone has no equivalent per-point mask.",
            "Original frame indices preserved even before first pose. Official script can skip initial result_dict=None frames.",
            "Lossless RGB memmap instead of mp4v intermediate hand/face encoding.",
            "Face uses existing current-image head-ROI recovery cache mapped back to original coordinates, not official face extraction.",
            "Full-frame normalized coordinates retained for body calculation. Official prior person crop would change normalized x/y aspect scaling; this trial is not a full preprocessing reproduction.",
            "No target-word answer, dictionary expected shape, or user video is supplied in input arrays. IDs and expression labels are manifest metadata only.",
        ],
        "observed_vs_inferred": "Original native/masked masks, coordinates, face cache, crop sources and timestamps remain separate. Repeated or blank crops never set current-observation masks true.",
        "excluded": ["private user footage", "mentor footage", "new detector inference", "new model download", "training", "C/T3F replacement", "operational DTW/web changes"],
        "limitations": ["No independently verified handshape/stage/error labels", "Existing public TCN TRAIN replay, not fresh C/T3F evaluation", "All12 results retained; no score-based subset selection", "Face visual stream may carry nonmanual information but no expression analysis is claimed"],
        "array_schema": {
            "left_hand_rgb.npy/right_hand_rgb.npy/face_rgb.npy": "uint8[T,224,224,3] RGB, no labels",
            "body14.npy": "float32[T,14], no labels",
            "native_observation.npz/masked_observation.npz": "verbatim numeric source arrays including observed,usable,xy_px,timestamp_us,frame_id",
            "face_observation.npz": "verbatim face cache arrays",
            "provenance.npz": "source_kind[T,3],origin_frame_index[T,3],crop_bbox_xywh[T,3,4],hand_point_valid[T,2,21],body_point_valid[T,7],body_source_kind[T],body_origin_frame_index[T],decode_pts_us[T],timestamp_us[T],frame_id[T]",
        },
        "selected_clips": [{k: r[k] for k in ["alias", "clip_id", "word_id", "label", "recorded_person_id", "role"]} for r in rows],
    }


def check_existing(directory, row):
    done = directory / "DONE.json"
    if not directory.exists():
        return None
    if not done.exists():
        raise RuntimeError(f"Partial output preserved; inspect before any recovery: {directory}")
    record = read(done)
    assert record["source_video_sha256"] == row["source_video_sha256"]
    assert record["prepare_script_sha256"] == sha(__file__), "Completed inputs came from another script revision"
    for p, h in record["source_pins"].items():
        assert sha(p) == h, p
    for p, h in record["output_pins"].items():
        assert sha(p) == h, p
    return record


def prepare_clip(row, hands, face_functions, body_functions):
    directory = OUT / row["alias"]
    existing = check_existing(directory, row)
    if existing:
        print(f"{row['alias']}: verified reuse {existing['frames']} frames", flush=True)
        return existing
    native = numeric_npz(row["native_observation"])
    masked = numeric_npz(row["masked_observation"])
    face = numeric_npz(row["face_observation"])
    n = len(native["frame_id"])
    assert native["xy_px"].shape == (n, 75, 2)
    assert np.array_equal(native["frame_id"], np.arange(n))
    for other in [masked, face]:
        assert np.array_equal(other["frame_id"], native["frame_id"])
        assert np.array_equal(other["timestamp_us"], native["timestamp_us"])
    assert np.isin(np.diff(native["timestamp_us"]), [33333, 33334]).all()
    valid = native["observed"] & native["usable"] & np.isfinite(native["xy_px"]).all(-1)
    directory.mkdir(parents=True)
    arrays = [np.lib.format.open_memmap(directory / f"{s}_rgb.npy", mode="w+", dtype=np.uint8, shape=(n,224,224,3)) for s in STREAM_NAMES]
    body14 = np.lib.format.open_memmap(directory / "body14.npy", mode="w+", dtype=np.float32, shape=(n,14))
    kinds = np.zeros((n,3),np.uint8)
    origin = np.full((n,3),-1,np.int32)
    boxes = np.full((n,3,4),-1,np.int32)
    hand_valid = np.stack([valid[:,o:o+21] for o in HAND_OFFSETS],axis=1)
    body_valid = valid[:,BODY_IDS].copy()
    body_kind = np.zeros(n,np.uint8)
    body_origin = np.full(n,-1,np.int32)
    decode_pts = np.zeros(n,np.int64)
    previous_crops = [None,None,None]
    previous_index = [-1,-1,-1]
    previous_boxes = [None,None,None]
    previous_body = None
    previous_body_index = -1
    center_pose_distance = np.full((n,2),np.nan,np.float32)
    hand_bbox_center = np.full((n,2,2),np.nan,np.float32)
    errors = []
    count = 0
    shape = None
    with av.open(row["source_video"]) as container:
        stream = container.streams.video[0]
        # Container average_rate may reflect a shortened final packet duration
        # (V01 reports 452/15). Per-frame original PTS correspondence below is
        # the relevant contract, not that aggregate container estimate.
        container_average_rate = str(stream.average_rate)
        container_base_rate = str(stream.base_rate)
        for frame in container.decode(stream):
            i = count
            assert i < n, "Video has extra frames"
            rgb = frame.to_ndarray(format="rgb24")
            ih,iw = rgb.shape[:2]
            shape = [iw,ih]
            assert list(face["image_size"]) == shape
            assert frame.pts is not None
            decode_pts[i] = round(float(frame.pts * frame.time_base) * 1_000_000)
            assert abs(int(decode_pts[i])-int(native["timestamp_us"][i])) <= 1
            normalized = native["xy_px"][i] / np.array([iw,ih])
            for s,offset in enumerate(HAND_OFFSETS):
                current = None
                box = None
                if bool(hand_valid[i,s].all()):
                    lm = normalized[offset:offset+21]
                    box = hands["get_bounding_box"](lm,rgb.shape,scale_factor=1.5)
                    box = hands["adjust_bounding_box"](box,rgb.shape)
                    current = hands["resize_frame"](hands["crop_frame"](rgb,box),(224,224))
                    assert current is not None and current.shape == (224,224,3)
                    assert box[0] >= 0 and box[1] >= 0
                    assert box[0]+box[2] <= iw and box[1]+box[3] <= ih
                    hand_bbox_center[i,s] = np.array(box[:2])+np.array(box[2:])/2
                    pose_wrist = 15 if s==0 else 16
                    center_pose_distance[i,s] = np.linalg.norm(normalized[offset]-normalized[pose_wrist])
                if current is not None:
                    arrays[s][i]=current; kinds[i,s]=CURRENT; origin[i,s]=i; boxes[i,s]=box
                    previous_crops[s]=current;previous_index[s]=i;previous_boxes[s]=box
                elif previous_crops[s] is not None:
                    arrays[s][i]=previous_crops[s];kinds[i,s]=PREVIOUS_COPY;origin[i,s]=previous_index[s];boxes[i,s]=previous_boxes[s]
                else:
                    arrays[s][i]=0
            current = None
            face_box = None
            if bool(face["found"][i]) and np.isfinite(face["xyz_normalized"][i]).all():
                lm = face["xyz_normalized"][i]
                current = face_functions["resize_frame"](face_functions["cues_on_grey_background"](rgb,lm),(224,224))
                assert current is not None and current.shape == (224,224,3)
                # This bbox describes the face-landmark extent, not the official composited eye/mouth crop extent.
                lo = np.floor(lm[:,:2].min(0)*[iw,ih]).astype(int)
                hi = np.ceil(lm[:,:2].max(0)*[iw,ih]).astype(int)
                face_box = [int(lo[0]),int(lo[1]),int(hi[0]-lo[0]),int(hi[1]-lo[1])]
            s=2
            if current is not None:
                arrays[s][i]=current;kinds[i,s]=CURRENT;origin[i,s]=i;boxes[i,s]=face_box
                previous_crops[s]=current;previous_index[s]=i;previous_boxes[s]=face_box
            elif previous_crops[s] is not None:
                arrays[s][i]=previous_crops[s];kinds[i,s]=PREVIOUS_COPY;origin[i,s]=previous_index[s];boxes[i,s]=previous_boxes[s]
            else:
                arrays[s][i]=0
            if bool(valid[i,BODY_REQUIRED].all()):
                assert np.linalg.norm(normalized[11]-normalized[12])>1e-8
                transformed=np.asarray(body_functions["normalize_pose_keypoints"](normalized[:25]))
                vec=transformed[BODY_IDS].reshape(14).astype(np.float32)
                assert np.isfinite(vec).all()
                body14[i]=vec;body_kind[i]=CURRENT;body_origin[i]=i
                previous_body=vec;previous_body_index=i
            elif previous_body is not None:
                body14[i]=previous_body;body_kind[i]=PREVIOUS_COPY;body_origin[i]=previous_body_index
            else:
                body14[i]=-9999
            count += 1
    assert count == n, f"Video/observation length mismatch {count}/{n}"
    for a in arrays:
        a.flush()
    body14.flush()
    np.savez_compressed(directory / "native_observation.npz", **native)
    np.savez_compressed(directory / "masked_observation.npz", **masked)
    np.savez_compressed(directory / "face_observation.npz", **face)
    np.savez_compressed(directory / "provenance.npz",source_kind=kinds,origin_frame_index=origin,
        crop_bbox_xywh=boxes,hand_point_valid=hand_valid,body_point_valid=body_valid,
        body_source_kind=body_kind,body_origin_frame_index=body_origin,decode_pts_us=decode_pts,
        timestamp_us=native["timestamp_us"],frame_id=native["frame_id"],
        hand_wrist_to_assigned_pose_wrist_normalized_distance=center_pose_distance,
        hand_bbox_center_px=hand_bbox_center)
    # Contract checks: copied RGB never counted as newly observed, and identity slots never reassigned.
    assert np.array_equal(kinds[:,:2] == CURRENT,hand_valid.all(-1))
    for s in range(3):
        for i in np.flatnonzero(kinds[:,s] == PREVIOUS_COPY):
            assert origin[i,s] < i
            assert np.array_equal(arrays[s][i],arrays[s][origin[i,s]])
    for target, original in [("native_observation.npz",native),("masked_observation.npz",masked),("face_observation.npz",face)]:
        retained=numeric_npz(directory/target)
        for key in original:
            if original[key].dtype.kind in 'fc':assert np.array_equal(retained[key],original[key],equal_nan=True)
            else:assert np.array_equal(retained[key],original[key])
    files=sorted(directory.glob("*.np*"))
    pins={p:sha(p) for p in [row['source_video'],row['record'],row['native_observation'],row['masked_observation'],row['face_observation']]}
    result={"alias":row['alias'],"clip_id":row['clip_id'],"frames":n,"image_size":shape,
        "source_video_sha256":row['source_video_sha256'],"source_pins":pins,
        "container_average_rate":container_average_rate,"container_base_rate":container_base_rate,
        "prepare_script_sha256":sha(__file__),"output_pins":{str(p):sha(p) for p in files},
        "crop_source_counts":{s:{str(k):int((kinds[:,j]==k).sum()) for k in [0,1,2]} for j,s in enumerate(STREAM_NAMES)},
        "body_source_counts":{str(k):int((body_kind==k).sum()) for k in [0,1,2]},
        "decode_pts_max_difference_us":int(np.abs(decode_pts-native['timestamp_us']).max()),
        "hand_slot_order":"anatomical_left_then_right_existing_native_NO_reassignment",
        "hand_crop_center_coordinate_check":"within source pixels; no source-frame cropping transform",
        "hand_pose_distance_ge_official_0_1": [int((center_pose_distance[:,s]>=0.1).sum()) for s in [0,1]],
        "face_bbox_note":"provenance bbox is face-landmark extent, not composited eye-mouth crop bounds",
        "output_bytes":sum(p.stat().st_size for p in files),"array_preservation_pass":True,
        "rgb_copy_not_observation_pass":True,"full_frame_preservation_pass":True}
    write(directory/"DONE.json",result)
    print(f"{row['alias']}: {n} frames, RGB sources {result['crop_source_counts']}",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    assert args.prepare != args.verify, "Choose --prepare or --verify"
    rows,pins=frozen_selection()
    proposal=protocol(rows,pins)
    protocol_path=UNIT/"PUBLIC_INPUT_PROTOCOL.json"
    if protocol_path.exists():
        assert read(protocol_path)==proposal,"Existing frozen input protocol differs"
    elif args.prepare:
        write(protocol_path,proposal)
    else:
        raise FileNotFoundError(protocol_path)
    OUT.mkdir(exist_ok=True)
    if args.prepare:
        hands,faces,body=load_official()
        results=[prepare_clip(r,hands,faces,body) for r in rows]
    else:
        results=[check_existing(OUT/r['alias'],r) for r in rows]
        assert all(results)
    total_bytes=sum(r['output_bytes'] for r in results)
    assert total_bytes < 1_000_000_000
    manifest={"status":"PUBLIC_INPUTS_PREPARED_NOT_MODEL_VALIDATED", "created_at":datetime.now(timezone.utc).isoformat(),
        "protocol_sha256":sha(protocol_path),"prepare_script_sha256":sha(__file__),"clips":len(results),
        "frames":sum(r['frames'] for r in results),"output_bytes":total_bytes,
        "all_source_hashes_verified":True,"all_native_masked_face_arrays_unchanged":True,
        "all_frame_ids_and_pts_matched":True,"labels_only_in_manifest_not_arrays":True,
        "source_kind_legend":proposal['source_kind'],"semantic_ground_truth_available":False,
        "training_performed":False,"model_inference_performed":False,
        "rows":[{**{k:r[k] for k in ['alias','clip_id','word_id','label','recorded_person_id','role','source_video']},
                 "input_dir":str(OUT/r['alias']),"done_sha256":sha(OUT/r['alias']/"DONE.json")} for r in rows],
        "summary":results}
    if args.prepare:
        write(OUT/"MANIFEST.json",manifest)
    else:
        old=read(OUT/"MANIFEST.json")
        for key in ['protocol_sha256','prepare_script_sha256','clips','frames','output_bytes','rows','summary']:
            assert old[key]==manifest[key],key
        write(OUT/"VERIFY.json",{"status":"PASS_SOURCE_OUTPUT_HASHES_FRAME_AND_ARRAY_CONTRACT", "at":datetime.now(timezone.utc).isoformat(),"manifest_sha256":sha(OUT/"MANIFEST.json"),"clips":12,"frames":manifest['frames'],"not_independent_human_ground_truth":True})
    print(json.dumps({k:manifest[k] for k in ['status','clips','frames','output_bytes']},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except Exception as exc:
        OUT.mkdir(exist_ok=True)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        write(OUT/f"ERROR_{stamp}.json",{"type":type(exc).__name__,"message":str(exc),"partial_outputs_preserved":True})
        raise
