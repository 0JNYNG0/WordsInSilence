"""Frozen image-consistency comparison of saved public recovery coordinates.

No detector/model inference or learning. Real source coordinates are read only
after a separate root authorization pins BOTH independent blind reviews.
The pure self-test mode never opens real arrays, labels, images or predictions.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

UNIT = Path(__file__).resolve().parent
OPS = UNIT.parent
BUNDLE = OPS / "word100_temporal_public_bundle_20261025"
PREVIOUS = OPS / "word100_temporal_refiner_autonomous_20261025"
SEEDS = (17, 29, 43)
CONDITIONS = ("B0", "B1", "Linear") + tuple(f"{c}_seed{s}" for c in ("B2", "B3") for s in SEEDS)
VISIBILITY = {"visible", "occluded", "blur", "out_of_frame", "ambiguous"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_new(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def valid_box(box):
    if box is None:
        return False
    q = np.asarray(box, dtype=float)
    return bool(q.shape == (4,) and np.isfinite(q).all() and q[0] < q[2] and q[1] < q[3])


def boxes_overlap(a, b):
    return bool(valid_box(a) and valid_box(b) and max(a[0], b[0]) < min(a[2], b[2])
                and max(a[1], b[1]) < min(a[3], b[3]))


def inside_box(point, box):
    return bool(valid_box(box) and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])


def inside_roi(points, roi):
    q = np.asarray(points)
    x0, y0, x1, y1 = np.asarray(roi).reshape(-1)[:4]
    return (np.isfinite(q).all(axis=-1) & (q[..., 0] >= x0) & (q[..., 0] < x1)
            & (q[..., 1] >= y0) & (q[..., 1] < y1))


def distance_to_box(point, box):
    """Pixel distance to a coarse rectangle, NOT anatomical joint error."""
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return float(np.hypot(dx, dy))


def agreement(a, b):
    reasons = []
    if not a["identity_confident"] or not b["identity_confident"]:
        reasons.append("anatomical_identity_not_confirmed_by_both")
    if a["visibility"] != "visible" or b["visibility"] != "visible":
        reasons.append("wrist_not_clearly_visible_to_both")
    if not valid_box(a["wrist_box"]) or not valid_box(b["wrist_box"]):
        reasons.append("coarse_wrist_box_missing")
    elif not boxes_overlap(a["wrist_box"], b["wrist_box"]):
        reasons.append("reviewer_boxes_do_not_overlap")
    return dict(scorable=not reasons, reasons=reasons,
                both_visible=a["visibility"] == b["visibility"] == "visible",
                both_identity_confident=a["identity_confident"] and b["identity_confident"],
                boxes_overlap=boxes_overlap(a["wrist_box"], b["wrist_box"]))


def score_point(point, available, a, b):
    agree = agreement(a, b)
    if not agree["scorable"]:
        return dict(status="unscorable", agreement=agree, inside_A=None, inside_B=None,
                    distance_to_A_px=None, distance_to_B_px=None)
    if not available:
        return dict(status="unavailable", agreement=agree, inside_A=None, inside_B=None,
                    distance_to_A_px=None, distance_to_B_px=None)
    q = np.asarray(point, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("Available wrist must be a finite 2D point")
    ia, ib = inside_box(q, a["wrist_box"]), inside_box(q, b["wrist_box"])
    state = "inside_both" if ia and ib else "inside_one_ambiguous" if ia or ib else "outside_both"
    return dict(status=state, agreement=agree, inside_A=ia, inside_B=ib,
                distance_to_A_px=distance_to_box(q, a["wrist_box"]),
                distance_to_B_px=distance_to_box(q, b["wrist_box"]))


def box_roi_relation(box, roi):
    if not valid_box(box):
        return "unknown"
    r = np.asarray(roi).reshape(-1)[:4]
    if box[2] < r[0] or box[0] >= r[2] or box[3] < r[1] or box[1] >= r[3]:
        return "outside"
    if r[0] <= box[0] and box[2] < r[2] and r[1] <= box[1] and box[3] < r[3]:
        return "inside"
    return "boundary"


def ranges(flags):
    edges = np.diff(np.r_[False, flags, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def fixed_body_basis(base):
    """Same baseline-only shoulder basis as frozen train_compare.py."""
    valid = base["usable"][:, 11] & base["usable"][:, 12]
    left, right = base["xy_px"][valid, 11], base["xy_px"][valid, 12]
    width = np.linalg.norm(left - right, axis=1)
    good = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1) & (width > 1)
    if not good.any():
        raise ValueError("No baseline shoulder basis; cannot substitute hidden hands")
    return np.median((left[good] + right[good]) / 2, axis=0).astype(np.float32), float(np.median(width[good]))


def transparent_linear(base, basis):
    """Frozen natural-gap policy, same 11-frame windows and original times.

    This duplicates only the arithmetic control, never runs the old model.
    All complete-hand bounded gaps <=7 are retained; points already usable
    stay exact. ROI rejection and unbounded gaps remain unavailable.
    """
    xy = base["xy_px"].copy()
    estimated = np.zeros(base["usable"].shape, bool)
    proposed = np.zeros_like(estimated)
    rejected_roi = np.zeros_like(estimated)
    center, scale = basis
    n = len(xy)
    records = []
    for side in (0, 1):
        sl = slice(33 + 21 * side, 54 + 21 * side)
        complete = (base["usable"][:, sl] & base["observed"][:, sl]).all(axis=1)
        for start0, stop0 in ranges(~complete):
            start, stop = int(start0), int(stop0)
            if stop - start > 7 or start == 0 or stop == n:
                continue
            missing = ~base["usable"][start:stop, sl]
            if not missing.any():
                continue
            middle = (start + stop - 1) // 2
            lo = max(0, min(middle - 5, n - 11))
            if n < 11 or not (lo <= start - 1 and stop < lo + 11):
                continue
            indices = np.arange(lo, lo + 11)
            normalized = (base["xy_px"][indices, sl] - center) / scale
            valid = base["usable"][indices, sl]
            times = base["timestamp_us"][indices].astype(np.float64)
            interpolated = np.empty_like(normalized)
            for joint in range(21):
                known = np.flatnonzero(valid[:, joint])
                if len(known) < 2:
                    raise ValueError("Frozen bounded window lacks two temporal anchors")
                for axis in range(2):
                    interpolated[:, joint, axis] = np.interp(times, times[known], normalized[known, joint, axis])
            local = (interpolated.astype(np.float32) * scale + center)[start-lo:stop-lo]
            inside = inside_roi(local, base["roi_box"])
            add = missing & inside
            xy[start:stop, sl][add] = local[add]
            estimated[start:stop, sl] |= add
            proposed[start:stop, sl] |= missing
            rejected_roi[start:stop, sl] |= missing & ~inside
            records.append(dict(side=side, range=[start, stop], window=[lo, lo+11],
                                estimated_joint_frames=int(add.sum())))
    assert not np.any(estimated & base["usable"])
    assert np.array_equal(xy[base["usable"]], base["xy_px"][base["usable"]])
    return dict(xy_px=xy, estimated=estimated, usable_observed=base["usable"].copy(),
                observed=base["observed"].copy(), proposed=proposed, rejected_roi=rejected_roi,
                intervals=records, frame_id=base["frame_id"], timestamp_us=base["timestamp_us"])


def validate_review(doc, ids):
    if not isinstance(doc, dict) or not isinstance(doc.get("rows"), list):
        raise ValueError("Expected frozen review object with rows")
    rows = doc["rows"]
    assert len(rows) == len(ids) and {r["sample_id"] for r in rows} == set(ids)
    assert len({r["sample_id"] for r in rows}) == len(ids)
    for r in rows:
        assert isinstance(r["identity_confident"], bool)
        assert r["visibility"] in VISIBILITY
        assert r["wrist_box"] is None or valid_box(r["wrist_box"])
        assert r["palm_box"] is None or valid_box(r["palm_box"])
        assert r["visible_fingertips"] is None or (isinstance(r["visible_fingertips"], int) and 0 <= r["visible_fingertips"] <= 5)
        assert isinstance(r["reason"], str)
        for key in ("wrist_box", "palm_box"):
            if r[key] is not None:
                assert 0 <= r[key][0] < r[key][2] <= 1920 and 0 <= r[key][1] < r[key][3] <= 1080
    return {r["sample_id"]: r for r in rows}


def required_authorization_paths():
    return [UNIT / name for name in ("BLIND_REVIEW_A.json", "BLIND_REVIEW_B.json", "SELECTION.json",
            "BLIND_INDEX.json", "SCOPE.json", "evaluate_benchmark.py", "EVALUATOR_PLAN.json")]


def validate_authorization(path):
    """Root writes this only after both independent reviews freeze."""
    p = Path(path).resolve()
    assert p.parent == UNIT and p.name == "EVALUATION_AUTHORIZATION.json"
    auth = read(p)
    assert auth["status"] == "APPROVED_AFTER_BOTH_BLIND_REVIEWS_FROZEN"
    assert auth["evaluation_allowed"] is True
    assert auth["both_blind_reviews_frozen_before_prediction_access"] is True
    assert auth["training_allowed"] is False and auth["model_inference_allowed"] is False
    pins = {str(Path(k).resolve()): v for k, v in auth["source_pins"].items()}
    for f in required_authorization_paths():
        assert str(f.resolve()) in pins, (str(f), "root authorization must pin this file")
    for f, h in pins.items():
        assert sha(f) == h, (f, "changed after blind freeze")
    return auth, pins


def base_view(payload, condition, b0):
    origin = np.where(payload["usable"], np.where(b0["usable"], "original_observed", "image_redetected"), "unavailable")
    return dict(xy_px=payload["xy_px"], available=payload["usable"], observed=payload["observed"],
                estimated=np.zeros_like(payload["usable"]), provenance=origin,
                usable_observed=payload["usable"], condition=condition)


def estimated_view(saved, b1, condition):
    estimated = saved["estimated"]
    assert estimated.shape == b1["usable"].shape
    assert np.array_equal(saved["usable_observed"], b1["usable"])
    assert np.array_equal(saved["observed"], b1["observed"])
    assert np.array_equal(saved["frame_id"], b1["frame_id"])
    assert np.array_equal(saved["timestamp_us"], b1["timestamp_us"])
    assert not np.any(estimated & b1["usable"])
    assert np.array_equal(saved["xy_px"][b1["usable"]], b1["xy_px"][b1["usable"]])
    assert np.isfinite(saved["xy_px"][estimated]).all()
    assert inside_roi(saved["xy_px"][estimated], b1["roi_box"]).all()
    assert not estimated[:, :33].any(), "Body observations must never be estimated in this hand pilot"
    origin = np.where(estimated, "linear_estimate" if condition == "Linear" else "learned_temporal_estimate",
                      np.where(b1["usable"], np.where(b1["point_origin"] == 1, "original_observed", "image_redetected"), "unavailable"))
    return dict(xy_px=saved["xy_px"], available=b1["usable"] | estimated,
                observed=b1["observed"], estimated=estimated, provenance=origin,
                usable_observed=b1["usable"], condition=condition,
                linear_rejected_roi=saved.get("rejected_roi"))


def summary(rows):
    counts = Counter(r["score"]["status"] for r in rows)
    availability = sum(r["available"] for r in rows)
    scorable = sum(r["score"]["agreement"]["scorable"] for r in rows)
    evaluated = sum(r["score"]["status"] in ("inside_both", "inside_one_ambiguous", "outside_both") for r in rows)
    assert sum(counts.values()) == len(rows)
    return dict(total_scenes=len(rows), available=availability, unavailable=len(rows)-availability,
                jointly_scorable_scenes=scorable, available_and_scorable=evaluated,
                statuses={k:counts[k] for k in ("inside_both", "inside_one_ambiguous", "outside_both", "unavailable", "unscorable")},
                provenance=dict(Counter(r["provenance"] for r in rows)),
                inside_both_over_all_pct=100 * counts["inside_both"] / len(rows) if rows else None,
                inside_both_over_evaluable_pct=100 * counts["inside_both"] / evaluated if evaluated else None,
                denominator_note="All selected scenes retained. Unscorable/missing are not zero-error or successful cases.")


def run(authorization):
    auth, auth_pins = validate_authorization(authorization)  # before real labels/coordinates
    selection = read(UNIT / "SELECTION.json")
    blind = read(UNIT / "BLIND_INDEX.json")
    scenes = selection["selection"]
    assert len(scenes) == 24 and len(blind["scenes"]) == 24
    ids = [s["sample_id"] for s in scenes]
    a = validate_review(read(UNIT / "BLIND_REVIEW_A.json"), ids)
    b = validate_review(read(UNIT / "BLIND_REVIEW_B.json"), ids)
    for s in scenes:
        assert 1 <= s["signer_id"] <= 6
    manifest = {r["clip_id"]: r for r in read(BUNDLE / "MANIFEST.json")}
    cids = sorted({s["clip_id"] for s in scenes})
    pins = dict(auth_pins)
    pins[str(Path(authorization).resolve())] = sha(authorization)
    fixed_paths = [BUNDLE / "MANIFEST.json", BUNDLE / "load_inputs.py", PREVIOUS / "FROZEN_PROTOCOL.json",
                   PREVIOUS / "train_compare.py", PREVIOUS / "downstream/INPUT_PINS.json", UNIT / "SOURCE_PINS.json"]
    for f in fixed_paths:
        pins[str(f.resolve())] = sha(f)
    for path, expected in selection["source_pins"].items():
        assert sha(path) == expected, (path, "selection source changed")
        pins[str(Path(path).resolve())] = expected
    old_protocol = read(PREVIOUS / "FROZEN_PROTOCOL.json")
    historical_pins = {str(Path(k).resolve()): v for k,v in read(PREVIOUS / "downstream/INPUT_PINS.json").items()}
    for f in (BUNDLE / "load_inputs.py", PREVIOUS / "train_compare.py"):
        assert sha(f) == old_protocol["source_pins"][str(f.resolve())]
    for cid in cids:
        row = manifest[cid]
        assert sha(row["arrays"]) == row["arrays_sha256"]
        pins[str(Path(row["arrays"]).resolve())] = row["arrays_sha256"]
        for condition in CONDITIONS[3:]:
            f = PREVIOUS / "runs" / condition / "natural_estimates" / (cid + ".npz")
            actual = sha(f)
            assert actual == historical_pins[str(f.resolve())], (str(f), "saved candidate changed since prior downstream freeze")
            pins[str(f.resolve())] = actual
    for row in blind["scenes"]:
        for field in ("image", "grid", "triptych"):
            p = row[field + "_path"]
            assert sha(p) == row[field + "_sha256"]
            pins[str(Path(p).resolve())] = row[field + "_sha256"]
    output = UNIT / "evaluation"
    output.mkdir()  # refuse overwrite of a previous evaluation
    write_new(output / "INPUT_PINS.json", pins)  # pin all sources before coordinate loading
    spec = importlib.util.spec_from_file_location("recovery_benchmark_safe_adapter", BUNDLE / "load_inputs.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    results = []
    preservation = []
    linear_intervals = []
    for cid in cids:
        with np.load(manifest[cid]["arrays"], allow_pickle=False) as z:
            b0, b1 = adapter.from_arrays(z, "B0"), adapter.from_arrays(z, "B1")
            native_xy, native_usable = z["native_xy_px"].copy(), z["native_usable"].copy()
        assert np.all(b1["usable"][b0["usable"]])
        assert np.array_equal(b0["xy_px"][b0["usable"]], b1["xy_px"][b0["usable"]])
        assert np.array_equal(b0["timestamp_us"], b1["timestamp_us"])
        assert np.array_equal(b0["roi_box"], b1["roi_box"])
        assert np.array_equal(b0["frame_id"], np.arange(len(b0["xy_px"])))
        assert np.array_equal(b0["frame_id"], b1["frame_id"])
        views = {"B0": base_view(b0, "B0", b0), "B1": base_view(b1, "B1", b0)}
        linear = transparent_linear(b1, fixed_body_basis(b0))
        linear_intervals.append(dict(clip_id=cid, intervals=linear["intervals"]))
        views["Linear"] = estimated_view(linear, b1, "Linear")
        for condition in CONDITIONS[3:]:
            f = PREVIOUS / "runs" / condition / "natural_estimates" / (cid + ".npz")
            with np.load(f, allow_pickle=False) as z:
                saved = {k:z[k].copy() for k in z.files}
            views[condition] = estimated_view(saved, b1, condition)
        for condition in CONDITIONS:
            view = views[condition]
            source = b0 if condition == "B0" else b1
            same = np.array_equal(view["xy_px"][source["usable"]], source["xy_px"][source["usable"]])
            assert same
            preservation.append(dict(clip_id=cid, condition=condition, observed_usable_joint_frames=int(source["usable"].sum()),
                                     coordinates_exact=same, observed_mask_unchanged=np.array_equal(view["observed"], source["observed"]),
                                     estimates_promoted_to_observed=int((view["estimated"] & view["observed"]).sum())))
        for s in [q for q in scenes if q["clip_id"] == cid]:
            i, side = s["frame_id"], s["anatomical_side"]
            wrist, sample_id = 33 + 21*side, s["sample_id"]
            roi = np.asarray(b1["roi_box"]).reshape(-1)[:4].tolist()
            native_exists = bool(native_usable[i, wrist])
            native_outside = native_exists and not bool(inside_roi(native_xy[i,wrist],roi))
            for condition in CONDITIONS:
                view = views[condition]
                available = bool(view["available"][i, wrist])
                point = view["xy_px"][i, wrist].tolist() if available else None
                assert not available or bool(inside_roi(point, roi))
                row = dict(sample_id=sample_id, condition=condition, stratum=s["stratum"], clip_id=cid,
                    signer_id=s["signer_id"], anatomical_side=side, frame_id=i, timestamp_us=int(b1["timestamp_us"][i]),
                    historical_crossing_metadata=s["historical_crossing_metadata"],
                    available=available, wrist_xy=point, observed=bool(view["observed"][i,wrist]),
                    estimated=bool(view["estimated"][i,wrist]), provenance=str(view["provenance"][i,wrist]),
                    roi_box=roi, point_inside_roi=bool(inside_roi(point,roi)) if available else None,
                    original_fullframe_wrist_available=native_exists, original_fullframe_wrist_outside_roi=native_outside,
                    reviewer_A_roi_relation=box_roi_relation(a[sample_id]["wrist_box"],roi),
                    reviewer_B_roi_relation=box_roi_relation(b[sample_id]["wrist_box"],roi),
                    candidate_rejection_detail=("linear_proposal_outside_roi" if condition == "Linear" and linear["rejected_roi"][i,wrist]
                                               else "not_saved_or_not_eligible_reason_not_inferred" if not available else None),
                    score=score_point(point,available,a[sample_id],b[sample_id]))
                results.append(row)
    assert len(results) == 24 * len(CONDITIONS)
    by_condition = {c:summary([r for r in results if r["condition"]==c]) for c in CONDITIONS}
    by_stratum = {c:{s:summary([r for r in results if r["condition"]==c and r["stratum"]==s]) for s in "ABCD"} for c in CONDITIONS}
    by_id = {(r["sample_id"],r["condition"]):r for r in results}
    transitions = {c:dict(Counter(by_id[s,"B1"]["score"]["status"]+" -> "+by_id[s,c]["score"]["status"] for s in ids)) for c in CONDITIONS}
    agreement_rows = [dict(sample_id=s, **agreement(a[s],b[s]), reviewer_A=a[s],reviewer_B=b[s]) for s in ids]
    family = {}
    for name in ("B2", "B3"):
        selected = [by_condition[f"{name}_seed{seed}"] for seed in SEEDS]
        family[name] = dict(all_seeds=list(SEEDS), no_best_seed_selection=True,
            inside_both_counts=[x["statuses"]["inside_both"] for x in selected],
            mean_inside_both_count=float(np.mean([x["statuses"]["inside_both"] for x in selected])),
            outside_both_counts=[x["statuses"]["outside_both"] for x in selected],
            available_counts=[x["available"] for x in selected])
    result = dict(status="COMPLETED_FROZEN_SAVED_COORDINATE_IMAGE_CONSISTENCY_COMPARISON", selected_scenes=24,
        conditions=list(CONDITIONS), evaluated_rows=len(results), by_condition=by_condition, by_stratum=by_stratum,
        family_seed_summary=family, paired_transition_from_B1=transitions,
        jointly_scorable_scenes=sum(r["scorable"] for r in agreement_rows),
        unscorable_scenes=[r["sample_id"] for r in agreement_rows if not r["scorable"]],
        preservation_checks=len(preservation), all_usable_originals_preserved=all(r["coordinates_exact"] for r in preservation),
        estimates_promoted_to_observed=sum(r["estimates_promoted_to_observed"] for r in preservation),
        limitations=["Coarse visible wrist consistency, not 21-joint accuracy or hidden joint truth.",
          "Two independent AI visual reviews are not expert anatomical annotations.",
          "TRAIN01-06 people and prior-trained candidates: diagnostic, not new-person generalization.",
          "Scene selection uses B2 seed17 availability; not an unbiased benchmark of all possible gaps.",
          "Classifiers and DTW were not rerun; no sign-recognition or feedback-quality improvement inferred.",
          "Both-box outside is image-range disagreement, not proof the signer performed a sign incorrectly.",
          "C-stratum wrist is already observed; wrist agreement does not validate estimated finger points."],
        training=0, model_inference=0, user_recording_access=0, production_changed=False)
    for f,h in pins.items():
        assert sha(f) == h, (f, "changed during evaluation")
    result["source_pins_unchanged"] = True
    write_new(output / "ROWS.json", results)
    write_new(output / "AGREEMENT.json", agreement_rows)
    write_new(output / "PRESERVATION.json", preservation)
    write_new(output / "LINEAR_INTERVALS.json", linear_intervals)
    write_new(output / "RESULT.json", result)
    render_overlays(output, scenes, blind, a, b, results)
    print(json.dumps(dict(status=result["status"], scenes=24, rows=len(results),
                         jointly_scorable_scenes=result["jointly_scorable_scenes"], by_condition=by_condition),ensure_ascii=False))


def render_overlays(output, scenes, blind, a, b, rows):
    """Post-freeze evaluation overlays, never replace the blind source images."""
    import cv2
    out = output / "overlays"
    out.mkdir()
    image_index = {r["sample_id"]:r for r in blind["scenes"]}
    result_index = {(r["sample_id"],r["condition"]):r for r in rows}
    files = []
    for scene in scenes:
        sid = scene["sample_id"]
        source = cv2.imread(image_index[sid]["image_path"])
        assert source is not None and source.shape[:2] == (1080,1920)
        panels = []
        for condition in CONDITIONS:
            panel = source.copy()
            r = result_index[sid,condition]
            for reviewer,color in ((a[sid],(70,220,70)),(b[sid],(240,180,40))):
                if valid_box(reviewer["wrist_box"]):
                    x0,y0,x1,y1=map(lambda x:int(round(x)),reviewer["wrist_box"])
                    cv2.rectangle(panel,(x0,y0),(x1,y1),color,4)
            if r["available"]:
                p=tuple(int(round(v)) for v in r["wrist_xy"])
                color=(40,80,250) if r["estimated"] else (255,255,255)
                cv2.drawMarker(panel,p,color,cv2.MARKER_CROSS,26,4)
            panel=cv2.resize(panel,(640,360),interpolation=cv2.INTER_AREA)
            panel=cv2.copyMakeBorder(panel,55,0,0,0,cv2.BORDER_CONSTANT,value=(24,24,24))
            cv2.putText(panel,f"{sid} {condition}: {r['score']['status']}",(9,20),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(panel,r["provenance"],(9,42),cv2.FONT_HERSHEY_SIMPLEX,.46,(200,200,200),1,cv2.LINE_AA)
            panels.append(panel)
        montage=np.concatenate([np.concatenate(panels[i:i+3],axis=1) for i in (0,3,6)],axis=0)
        path=out/(sid+".jpg")
        assert cv2.imwrite(str(path),montage,[cv2.IMWRITE_JPEG_QUALITY,94])
        files.append(dict(sample_id=sid,path=str(path),sha256=sha(path),dimensions=[1920,1245]))
    write_new(output/"OVERLAY_INDEX.json",dict(created_only_after_blind_freeze=True,
        legend="Green reviewerA / blue reviewerB / white observed wrist / red temporal estimate. Rectangles never adjusted to predictions.",rows=files))


def selftest():
    """Synthetic only. Never accesses actual labels, source arrays or models."""
    def label(box, visibility="visible", confident=True):
        return dict(identity_confident=confident,visibility=visibility,wrist_box=box)
    a,b=label([10,10,20,20]),label([15,15,25,25])
    tests=[]
    def check(name,condition):
        assert condition,name
        tests.append(name)
    check("both boxes",score_point([17,17],True,a,b)["status"]=="inside_both")
    check("one box ambiguous",score_point([11,11],True,a,b)["status"]=="inside_one_ambiguous")
    check("outside both",score_point([30,30],True,a,b)["status"]=="outside_both")
    check("missing not zero error",score_point(None,False,a,b)["status"]=="unavailable" and score_point(None,False,a,b)["distance_to_A_px"] is None)
    check("occluded withheld",score_point([17,17],True,a,label([15,15,25,25],"occluded"))["status"]=="unscorable")
    check("identity uncertainty withheld",score_point([17,17],True,a,label([15,15,25,25],confident=False))["status"]=="unscorable")
    check("nonoverlap withheld",score_point([17,17],True,a,label([30,30,40,40]))["status"]=="unscorable")
    check("touching edges not overlap",not boxes_overlap([0,0,10,10],[10,0,20,10]))
    check("rectangle edge included",inside_box([10,20],[10,10,20,20]))
    check("ROI right excluded",not bool(inside_roi([20,10],[0,0,20,20])))
    check("coarse rectangle distance",distance_to_box([23,24],[10,10,20,20])==5)
    n=17
    xy=np.ones((n,75,2),np.float32)*10
    xy[:,:,0]+=np.arange(n,dtype=np.float32)[:,None]
    usable=np.ones((n,75),bool)
    usable[6:9,33:54]=False
    xy[~usable]=0
    base=dict(xy_px=xy,usable=usable,observed=usable.copy(),timestamp_us=np.arange(n,dtype=np.int64)*40000,
              frame_id=np.arange(n),roi_box=np.array([0,0,100,100]))
    linear=transparent_linear(base,(np.array([0,0],np.float32),10.))
    check("threeframe holes filled",int(linear["estimated"].sum())==3*21)
    check("transparent interpolation",np.allclose(linear["xy_px"][6:9,33:54,0],np.arange(16,19)[:,None]))
    check("original coordinates preserved",np.array_equal(linear["xy_px"][usable],xy[usable]))
    check("original observed preserved",np.array_equal(linear["observed"],usable))
    check("opposite hand untouched",not linear["estimated"][:,54:75].any())
    longer={k:v.copy() for k,v in base.items()}
    longer["usable"][:8,33:54]=False;longer["observed"]=longer["usable"].copy();longer["xy_px"][~longer["usable"]]=0
    check("unbounded/long gap not filled",not transparent_linear(longer,(np.zeros(2,np.float32),10.))["estimated"][:9,33:54].any())
    irregular={k:v.copy() for k,v in base.items()}
    irregular["timestamp_us"]=np.array([0,1,2,3,4,5,6,10,20,30,31,32,33,34,35,36,37])*1000
    irregular_linear=transparent_linear(irregular,(np.zeros(2,np.float32),10.))
    expected=np.interp(irregular["timestamp_us"][6:9],irregular["timestamp_us"][[5,9]],[15,19])
    check("timestamps not index interpolation",np.allclose(irregular_linear["xy_px"][6:9,33,0],expected))
    check("unavailable reason retained with unscorable",score_point(None,False,a,label(None,"occluded"))["status"]=="unscorable")
    return dict(status="PASS",tests=tests,count=len(tests),real_labels_read=0,real_coordinates_read=0,
                model_inference=0,training=0,method="Small independent synthetic rectangles and time-series arrays")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("selftest","run"))
    parser.add_argument("--authorization",type=Path)
    args=parser.parse_args()
    if args.mode=="selftest":
        report=selftest();write_new(UNIT/"EVALUATOR_SELFTEST.json",report);print(json.dumps(report))
    else:
        if args.authorization is None:
            parser.error("Root-pinned EVALUATION_AUTHORIZATION.json required after both blind reviews freeze")
        run(args.authorization)
