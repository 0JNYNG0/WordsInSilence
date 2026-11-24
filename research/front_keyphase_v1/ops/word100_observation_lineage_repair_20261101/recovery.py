"""Isolated current-image recovery on the previously fixed public 303 frames.

Reuses the existing fullframe_hand_merge rule without changing its parameters.
No old runner is imported or executed. Prepare/selftest do not load a detector.
The run command needs a frozen manifest and writes only this new unit.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np

UNIT = Path(__file__).resolve().parent
OPS = UNIT.parent
ROOT = OPS.parents[2]
OLD = OPS / "word100_fullframe_hand_merge_20261020"
MERGE_PATH = OLD / "candidate.py"
OBSERVER_PATH = OPS / "word100_guide_video_feedback_20261028/motion_observer.py"
MODEL = ROOT / "Tools/models/hand_landmarker.task"
CONTRACT = OPS.parent / "config/contract.json"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


rule = module(MERGE_PATH, "fixed_recovery_merge_rule")


def read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def check_pins(pins):
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError("Pinned source changed: " + path)


def inside_box(points, box):
    p = np.asarray(points)
    return (np.isfinite(p).all(axis=-1) & (p[..., 0] >= box[0]) & (p[..., 0] < box[2])
            & (p[..., 1] >= box[1]) & (p[..., 1] < box[3]))


def eligibility(native_observed, pose, pose_usable, box, width, height, base_sides):
    """Point availability and current pose only; no word/guide/frame GT input."""
    observed = np.asarray(native_observed)
    if observed.shape != (75,) or observed.dtype.kind != "b":
        raise ValueError("Expected 75 original observed flags")
    good, scale = rule.context(np.asarray(pose), np.asarray(pose_usable), width, height)
    result = []
    for side, (elbow, wrist) in enumerate(((13, 15), (14, 16))):
        sl = slice(33+21*side, 54+21*side)
        if observed[sl].any():
            reason = "EXISTING_NATIVE_HAND_PRESERVED"
        elif side in base_sides:
            reason = "EXISTING_RAW_ASSIGNED_SIDE_PRESERVED"
        elif scale <= 1 or not good[[elbow, wrist]].all():
            reason = "CURRENT_POSE_ARM_UNSUPPORTED"
        elif not inside_box(np.asarray(pose)[wrist], box):
            reason = "POSE_WRIST_OUTSIDE_FIXED_ROI"
        else:
            reason = "ELIGIBLE_NATIVE_SIDE_UNOBSERVED_CURRENT_ARM_IN_ROI"
        result.append(dict(side=side, eligible=reason.startswith("ELIGIBLE_"), reason=reason,
                           native_observed_points=int(observed[sl].sum())))
    return result


def merge_frame(native_frame, raw_hands, base_sides, pose, pose_usable, width, height,
                box, full, crops, hand_score_threshold):
    """Pure same-frame merge. Keeps every previously observed or partial side."""
    output = {k: np.asarray(v).copy() for k, v in native_frame.items()}
    eligible = eligibility(output["observed"], pose, pose_usable, box, width, height, base_sides)
    allowed = {r["side"] for r in eligible if r["eligible"]}
    decisions = deepcopy(rule.merge(full, crops, raw_hands, base_sides, pose, pose_usable, width, height))
    additions = []
    for q in decisions:
        q["old_merge_accepted"] = bool(q["accepted"])
        q["applied"] = False
        if not q["accepted"]:
            continue
        side = q["side"]
        if side not in allowed:
            q["application_reason"] = "INELIGIBLE_SIDE_PRESERVED"
            continue
        sl = slice(33+21*side, 54+21*side)
        if output["observed"][sl].any():
            raise ValueError("Attempt to overwrite an existing hand")
        detection, crop = full[q["index"]], crops[q["crop_index"]]
        scores = [float(detection["score"]), float(crop["score"])]
        if not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
            raise ValueError("Actual full/crop handedness scores required")
        score = min(scores)
        points = np.asarray(detection["xy"], dtype=np.float32)
        supported = np.asarray(q["joint_supported"], dtype=bool) & np.isfinite(points).all(axis=1)
        # Keep the original quality rule separate from the two-scale agreement.
        usable = supported & (score >= hand_score_threshold)
        output["xy_px"][sl] = np.where(supported[:, None], points, np.nan)
        output["observed"][sl] = supported
        output["usable"][sl] = usable
        output["hand_score"][side] = score
        q.update(applied=True, application_reason="NEW_CURRENT_IMAGE_OBSERVATIONS_ONLY",
                 actual_full_handedness_score=scores[0], actual_crop_handedness_score=scores[1],
                 stored_hand_score=score, supported_points=int(supported.sum()),
                 final_roi_usable_points=int((usable & inside_box(points, box)).sum()))
        additions.append(dict(side=side, full_index=q["index"], crop_index=q["crop_index"],
                              observed_points=int(supported.sum()), usable_points=int(usable.sum()),
                              roi_usable_points=q["final_roi_usable_points"]))
    for side in (0, 1):
        sl = slice(33+21*side, 54+21*side)
        if np.asarray(native_frame["observed"])[sl].any():
            for key in ("xy_px", "observed", "usable", "conf"):
                if not np.array_equal(output[key][sl], np.asarray(native_frame[key])[sl], equal_nan=True):
                    raise AssertionError("Existing side altered")
            if not np.array_equal(output["hand_score"][side], np.asarray(native_frame["hand_score"])[side], equal_nan=True):
                raise AssertionError("Existing score altered")
    for key in ("xy_px", "observed", "usable", "conf"):
        if not np.array_equal(output[key][:33], np.asarray(native_frame[key])[:33], equal_nan=True):
            raise AssertionError("Existing pose altered")
    return output, dict(eligibility=eligible, decisions=decisions, additions=additions)


def lock_candidate_features(old_x, old_mask, x, mask, selected):
    """Preserve every original supported value and lock all out-of-scope cells."""
    x, mask = np.asarray(x).copy(), np.asarray(mask).copy()
    if not (x.shape == old_x.shape == mask.shape == old_mask.shape):
        raise ValueError("Original feature opportunities changed")
    if not (mask[old_mask].all() and np.array_equal(x[old_mask], old_x[old_mask])):
        raise AssertionError("Original supported feature values/masks changed")
    in_scope = np.zeros(len(x), dtype=bool); in_scope[selected] = True
    discarded = int((mask[~in_scope] & ~old_mask[~in_scope]).sum())
    x[~in_scope] = old_x[~in_scope]; mask[~in_scope] = old_mask[~in_scope]
    assert np.array_equal(x[~in_scope], old_x[~in_scope])
    assert np.array_equal(mask[~in_scope], old_mask[~in_scope])
    return x, mask, discarded


def prepare():
    """Audit selected input/cache metadata; no image decoding or feature extraction."""
    dest = UNIT / "RECOVERY_PREPARED.json"
    if dest.exists():
        raise FileExistsError(dest)
    scope = read(UNIT / "INPUT_SCOPE.json")
    rows = {r["alias"]: r for r in read(OPS / "word100_stage_feedback_cycle_20261031/INPUTS.json")["rows"]}
    selected = {}
    for window in scope["windows"]:
        for axis in ("query_axis", "reference_axis"):
            a = window[axis]
            for frame, pts in zip(a["reviewed_frame_ids"], a["reviewed_timestamp_us"]):
                old = selected.setdefault(a["alias"], {}).setdefault(int(frame), int(pts))
                assert old == pts
    assert len(selected) == 8 and sum(map(len, selected.values())) == 303
    pins, clips, opportunities, skip = {}, [], [], Counter()
    def pin(p, expected=None):
        p = str(Path(p).resolve()); digest = sha(p)
        if expected is not None and digest != expected:
            raise ValueError("Source identity changed: " + p)
        pins[p] = digest
        return p
    for p in (Path(__file__), MERGE_PATH, OBSERVER_PATH, MODEL, CONTRACT,
              UNIT / "AUTHORIZATION.json", UNIT / "INPUT_SCOPE.json",
              UNIT / "review/LINEAGE_DATA_AUDIT.json", OLD / "FROZEN.json", OLD / "INDEX.json"):
        pin(p)
    old_cache = read(OLD / "FROZEN.json")
    assert old_cache["params"] == rule.PARAMS
    for source in (MERGE_PATH, MODEL):
        assert sha(source) == old_cache["pins"][str(source)]
    cache_ids = {r["clip_id"] for r in old_cache["rows"]}
    for alias in sorted(selected):
        r = rows[alias]
        assert r["role"] in ("fit", "development") and r["person"] in (1, 4)
        record_path = pin(r["record"], r["record_sha256"])
        record = read(record_path); cap = record["native_capture"]
        # Reused P045 raw capture and final joint_v3 assignment have different folders.
        assignment = str(Path(cap["candidate"]).with_name("assignment.npz"))
        trace = str(Path(cap["candidate"]).with_name("trace.json"))
        files = dict(video=pin(r["source"], r["source_sha256"]), input=pin(r["input"], r["input_sha256"]),
                     native=pin(r["observation"], record["outputs"].get(r["observation"])),
                     raw=pin(cap["raw"]), assignment=pin(assignment), trace=pin(trace), record=record_path)
        pin(Path(cap["candidate"]).with_suffix(".sidecar.json"))
        z, native, raw, sides = arrays(files["input"]), arrays(files["native"]), arrays(files["raw"]), arrays(files["assignment"])["sides"]
        for key in ("frame_id", "timestamp_us"):
            assert np.array_equal(z[key], native[key]) and np.array_equal(raw[key], native[key])
        lookup = {int(f): i for i, f in enumerate(native["frame_id"])}
        indices = []
        for fid, pts in sorted(selected[alias].items()):
            i = lookup[fid]; assert int(native["timestamp_us"][i]) == pts
            indices.append(i)
            nraw = int(raw["raw_count"][i]); base_sides = sides[i, :nraw].tolist()
            el = eligibility(native["observed"][i], raw["pose_xy_px"][i], raw["pose_usable"][i],
                             r["box"], int(raw["width"][i]), int(raw["height"][i]), base_sides)
            assert all(e["side"] not in base_sides for e in el if e["eligible"])
            skip.update(x["reason"] for x in el)
            opportunities.append(dict(alias=alias, source_index=i, frame_id=fid, timestamp_us=pts,
                                      eligibility=el, raw_count=nraw, baseline_sides=base_sides,
                                      inference_required=any(x["eligible"] for x in el)))
        # Exact old detector result reuse is only possible with the same source clip.
        # None of these eight clips is in the previously completed merge cohort.
        if r["clip_id"] in cache_ids:
            raise RuntimeError("Matching completed cache discovered: audit and pin its exact inputs before any new inference")
        clips.append(dict(alias=alias, clip_id=r["clip_id"], files=files, box=r["box"], indices=indices,
                          completed_cache_match=False, frame_count=len(native["frame_id"])))
    write(dest, dict(schema="wis_fixed303_current_image_recovery_preparation_v1", status="PREPARED_NOT_EXECUTED",
        clips=clips, opportunities=opportunities, source_pins=pins,
        runtime=dict(python=sys.executable, versions={name: importlib.metadata.version(name)
                     for name in ("mediapipe", "numpy", "av")}),
        selection=dict(clips=8, unique_frames=303, hand_frames=606,
                       eligible_frames=sum(r["inference_required"] for r in opportunities),
                       eligible_hand_frames=sum(x["eligible"] for r in opportunities for x in r["eligibility"]),
                       reasons=dict(skip)),
        method=dict(merge_source=str(MERGE_PATH), parameters=rule.PARAMS, image_mode="IMAGE", num_hands=2,
                    min_hand_detection_confidence=.5, min_hand_presence_confidence=.5, min_tracking_confidence=.5,
                    old_parameters_unchanged=True, detector_asset=str(MODEL),
                    eligible="Native side wholly unobserved AND absent from original raw assignment sides, current supported arm, pose wrist inside original fixed ROI",
                    baseline="Saved same native observations/69 features; existing partial and quality-rejected sides preserved",
                    score_semantics="Actual handedness category score is a quality proxy, not landmark accuracy or the detector confidence option",
                    decoder="Original MP4 source PTS; JPEGs and nonselected image frames are not detector inputs",
                    cache="No matching clip IDs among existing completed fullframe_hand_merge cache; all eight audited",
                    output="Selected original303 frames only; velocity computed on intact original full timeline before selection; all outside-scope x/mask locked to saved baseline and never exported as candidate gains",
                    new_signrep_or_classifier_or_dtw=False),
        limitations=["Two same-model image scales can agree on inaccurate landmarks.",
                     "Missing support causes are not educational errors; added support is not anatomical accuracy.",
                     "No parameter tuning or repeat run is authorized after observing this candidate's results."]))
    print(json.dumps(read(dest)["selection"]))


def freeze():
    prep = read(UNIT / "RECOVERY_PREPARED.json")
    assert all(importlib.metadata.version(name) == version for name, version in prep["runtime"]["versions"].items())
    check_pins(prep["source_pins"])
    test = read(UNIT / "RECOVERY_SELFTEST.json")
    method = read(UNIT / "review/RECOVERY_METHOD_REVIEW.json")
    assert str(test["status"]).startswith("PASS") and str(method["status"]).startswith("PASS")
    assert test["module_sha256"] == sha(__file__)
    assert test["selftest_sha256"] == sha(UNIT / "recovery_selftest.py")
    pins = dict(prep["source_pins"])
    for p in (UNIT / "RECOVERY_PREPARED.json", UNIT / "RECOVERY_SELFTEST.json", UNIT / "recovery_selftest.py", UNIT / "review/RECOVERY_METHOD_REVIEW.json"):
        pins[str(p)] = sha(p)
    write(UNIT / "RECOVERY_FROZEN.json", dict(status="FROZEN_BEFORE_NEW_IMAGE_DETECTOR_RUN", pins=pins,
          method=prep["method"], scope=prep["selection"], time_unix=time.time(),
          execution_owner="root", new_training=0, new_classifier=0, new_alignment=0, production_changed=False))
    print("RECOVERY_FROZEN")


def selected_rgb(path, wanted):
    """Yield only requested exact original frame IDs and actual video PTS."""
    import av
    remaining = set(wanted); origin = None; last_us = -1; kept = 0; dropped = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                if origin is None:
                    origin = frame.pts
                pts = int(round(float((frame.pts-origin)*stream.time_base)*1e6))
                if pts <= last_us:
                    dropped += 1
                    continue
                last_us = pts; frame_id = kept+dropped; kept += 1
                if frame_id in remaining:
                    if pts != wanted[frame_id]:
                        raise ValueError("Decoded source PTS mismatch")
                    yield frame_id, pts, frame.to_ndarray(format="rgb24")
                    remaining.remove(frame_id)
                if not remaining:
                    return
    if remaining:
        raise ValueError("Original selected frame missing: " + str(sorted(remaining)))


def detect(lm, rgb, box=None):
    import mediapipe as mp
    h, w = rgb.shape[:2]; tick = time.perf_counter()
    rr = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    elapsed = time.perf_counter()-tick; result = []; x0, y0 = box[:2] if box is not None else (0, 0)
    for points, labels in zip(rr.hand_landmarks, rr.handedness):
        d = dict(xy=[[float(p.x*w+x0), float(p.y*h+y0)] for p in points],
                 label=labels[0].category_name, score=float(labels[0].score))
        if box is not None:
            d["box"] = box
        result.append(d)
    return result, elapsed


def run():
    frozen = read(UNIT / "RECOVERY_FROZEN.json"); check_pins(frozen["pins"])
    if (UNIT / "RECOVERY_RUNNING.json").exists() or (UNIT / "RECOVERY_RESULT_MANIFEST.json").exists():
        raise RuntimeError("Existing recovery execution/output; inspect instead of rerun")
    prep = read(UNIT / "RECOVERY_PREPARED.json")
    assert all(importlib.metadata.version(name) == version for name, version in prep["runtime"]["versions"].items())
    write(UNIT / "RECOVERY_RUNNING.json", dict(pid=os.getpid(), started=time.time()))
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    observer = module(OBSERVER_PATH, "fixed_recovery_observer")
    point_rule = read(CONTRACT)["usable_rule"]
    options = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=.5,
        min_hand_presence_confidence=.5, min_tracking_confidence=.5)
    started = time.perf_counter(); init_start = time.perf_counter()
    lm = vision.HandLandmarker.create_from_options(options); init_seconds = time.perf_counter()-init_start
    totals = Counter({key:0 for key in ("new_full_image_calls", "new_crop_image_calls", "added_native_hand_frames",
        "added_roi_usable_joint_frames", "new_supported_selected_feature_cells", "preserved_original_selected_feature_cells",
        "out_of_scope_derived_support_discarded")})
    timings = Counter(); records = []; all_frames = []
    try:
        for clip in prep["clips"]:
            alias, f = clip["alias"], clip["files"]
            native, raw, old = arrays(f["native"]), arrays(f["raw"]), arrays(f["input"])
            sides = arrays(f["assignment"])["sides"]
            candidate = {k: a.copy() for k, a in native.items()}
            points = [p for p in prep["opportunities"] if p["alias"] == alias]
            eligible = {p["frame_id"]: p for p in points if p["inference_required"]}
            frame_records = {p["frame_id"]: dict(p, status="SKIPPED_BY_FIXED_ELIGIBILITY", full=[], crop=[],
                decisions=[], additions=[], full_seconds=0., crop_seconds=0., full_calls=0, crop_calls=0) for p in points}
            decoded = 0; decode_tick = time.perf_counter()
            if eligible:
                for fid, pts, rgb in selected_rgb(f["video"], {fid:p["timestamp_us"] for fid,p in eligible.items()}):
                    decoded += 1; p = eligible[fid]; i = p["source_index"]
                    w, h = int(raw["width"][i]), int(raw["height"][i])
                    assert rgb.shape[:2] == (h, w)
                    full, fs = detect(lm, rgb)
                    n = int(raw["raw_count"][i]); bs = sides[i, :n].tolist()
                    boxes = rule.crop_boxes(raw["pose_xy_px"][i], raw["pose_usable"][i], bs, w, h)
                    allowed = {e["side"] for e in p["eligibility"] if e["eligible"]}
                    crops, cs, calls = [], 0., 0
                    for side, box in boxes.items():
                        if side not in allowed:
                            continue
                        got, elapsed = detect(lm, rgb[box[1]:box[3], box[0]:box[2]], box)
                        for det in got:
                            det["requested_side"] = side
                        crops.extend(got); cs += elapsed; calls += 1
                    before = {k: native[k][i] for k in ("xy_px", "observed", "usable", "conf", "hand_score")}
                    mt = time.perf_counter()
                    merged, ledger = merge_frame(before, raw["hand_xy_px"][i, :n], bs,
                        raw["pose_xy_px"][i], raw["pose_usable"][i], w, h, clip["box"], full, crops,
                        point_rule["hand_score_threshold"])
                    assert ledger["eligibility"] == p["eligibility"]
                    timings["merge_seconds"] += time.perf_counter()-mt
                    for key in merged:
                        candidate[key][i] = merged[key]
                    frame_records[fid].update(status="NEW_IMAGE_CANDIDATE_EXECUTED", full=full, crop=crops,
                        decisions=ledger["decisions"], additions=ledger["additions"], boxes={str(k):v for k,v in boxes.items() if k in allowed},
                        full_seconds=fs, crop_seconds=cs, full_calls=1, crop_calls=calls)
                    totals["new_full_image_calls"] += 1; totals["new_crop_image_calls"] += calls
                    totals["added_native_hand_frames"] += len(ledger["additions"])
                    totals["added_roi_usable_joint_frames"] += sum(a["roi_usable_points"] for a in ledger["additions"])
                    timings["full_image_seconds"] += fs; timings["crop_image_seconds"] += cs
            timings["decode_and_detector_loop_seconds"] += time.perf_counter()-decode_tick
            assert decoded == len(eligible)
            # This feature calculation is the new candidate only. No previous
            # comparison, learned posterior, classifier or alignment is rerun.
            native_candidate_usable = candidate["usable"].copy()
            roi_inside = inside_box(candidate["xy_px"], clip["box"])
            candidate["usable"] &= roi_inside
            tick = time.perf_counter(); features = observer.extract_features(candidate)
            timings["candidate_feature_seconds"] += time.perf_counter()-tick
            x = np.nan_to_num(features["x"]).astype(np.float32); mask = features["mask"]
            selected = np.asarray(clip["indices"], dtype=np.int64)
            # Adding a wrist can support velocity at the next source frame.
            # Out-of-scope derivatives are locked back to the original input,
            # never silently counted, exported or used as candidate evidence.
            x, mask, derived_outside = lock_candidate_features(old["x"], old["mask"], x, mask, selected)
            totals["out_of_scope_derived_support_discarded"] += derived_outside
            gained = mask[selected] & ~old["mask"][selected]
            totals["new_supported_selected_feature_cells"] += int(gained.sum())
            totals["preserved_original_selected_feature_cells"] += int(old["mask"][selected].sum())
            path = UNIT / "recovery_outputs" / (alias + ".npz"); path.parent.mkdir(exist_ok=True)
            with path.open("xb") as out:
                np.savez_compressed(out, x=x[selected], mask=mask[selected], baseline_x=old["x"][selected],
                    baseline_mask=old["mask"][selected], newly_supported=gained,
                    frame_id=native["frame_id"][selected], timestamp_us=native["timestamp_us"][selected],
                    source_index=selected, xy_px=np.where((candidate["observed"] & roi_inside)[selected, :, None], candidate["xy_px"][selected], np.nan),
                    observed=(candidate["observed"] & roi_inside)[selected], usable=candidate["usable"][selected],
                    native_xy_px=candidate["xy_px"][selected], native_observed=candidate["observed"][selected],
                    native_usable=native_candidate_usable[selected], hand_score=candidate["hand_score"][selected])
            ledger_path = UNIT / "recovery_records" / (alias + ".json")
            frames = [frame_records[p["frame_id"]] for p in points]
            write(ledger_path, dict(alias=alias, clip_id=clip["clip_id"], frames=frames, source=clip,
                features=features["feature_names"], new_feature_cells=int(gained.sum()),
                original_support_and_values_preserved=True, excluded_points_not_promoted=True,
                outside_scope_features_locked_to_original=True, outside_scope_derived_support_discarded=derived_outside,
                all_source_frames_preserved=True, educational_verdict=None))
            records.append(dict(alias=alias, output=str(path), output_sha256=sha(path), ledger=str(ledger_path),
                                ledger_sha256=sha(ledger_path), frames=len(selected)))
            all_frames.extend(frames)
    finally:
        lm.close()
    check_pins(frozen["pins"])
    assert len(all_frames) == 303
    write(UNIT / "RECOVERY_RESULT_MANIFEST.json", dict(status="COMPLETE_PENDING_INDEPENDENT_AND_VISUAL_REVIEW",
        records=records, counts=dict(totals), timings=dict(timings), initialization_seconds=init_seconds,
        wall_seconds=time.perf_counter()-started, selected_frames=303,
        skipped_frames=sum(r["status"].startswith("SKIPPED") for r in all_frames),
        executed_frames=sum(r["status"] == "NEW_IMAGE_CANDIDATE_EXECUTED" for r in all_frames),
        frozen_sha256=sha(UNIT / "RECOVERY_FROZEN.json"), new_training=0, new_classifier=0, new_dtw=0,
        old_detector_results_overwritten=False, source_results_recomputed=False,
        educational_accuracy=None, landmark_accuracy=None, production_changed=False))
    (UNIT / "RECOVERY_RUNNING.json").rename(UNIT / "RECOVERY_RUN_FINISHED.json")
    print(json.dumps(dict(status="COMPLETE_PENDING_REVIEW", counts=dict(totals))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("prepare", "freeze", "run"))
    args = parser.parse_args()
    try:
        globals()[args.stage]()
    except BaseException:
        target = UNIT / ("RECOVERY_FIRST_ERROR_" + args.stage + ".txt")
        if not target.exists():
            target.write_text(traceback.format_exc(), encoding="utf-8")
        raise
