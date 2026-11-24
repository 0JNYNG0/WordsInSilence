"""Audit the saved P065f48 recovery without detector/producer reexecution.

Reuses only previously independent scalar merge-checker functions. Reconstructs
69 features independently from saved source coordinates, one frame at a time.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import sys
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent.parent
O = U.parent
PINS = {}
CHECKS = 0
FAILURES = []


def check(ok, name, detail=None):
    global CHECKS
    CHECKS += 1
    if not bool(ok):
        FAILURES.append(dict(check=name, detail=detail))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path, expected=None):
    path = Path(path).resolve()
    h = sha(path)
    if expected is not None:
        check(h == expected, "source pin " + str(path))
    PINS[str(path)] = h
    return path


def read(path):
    return json.loads(pin(path).read_text(encoding="utf-8-sig"))


def load(path, expected=None):
    with np.load(pin(path, expected), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def exact(a, b, name):
    check(np.array_equal(a, b, equal_nan=True), name)


def feature_row(points, observed, usable, timestamp, frame, i):
    """Scalar source equations, no motion_observer calls or binary labels."""
    p = np.asarray(points[i], dtype=np.float64)
    valid = observed[i] & usable[i] & np.isfinite(p).all(axis=1)
    center = (p[11] + p[12]) / 2
    scale = math.hypot(*(p[11] - p[12]))
    body = bool(valid[11] and valid[12] and math.isfinite(scale) and scale > 1e-8)
    norm = (p - center) / scale if body else np.full((75, 2), np.nan)
    nv = valid & body
    values, masks = [], []

    def emit(v, good):
        ok = bool(good and math.isfinite(float(v)))
        values.append(float(v) if ok else 0.0)
        masks.append(ok)

    def angle(ids):
        a, b, c = ids
        if not all(valid[j] for j in ids):
            return 0., False
        ux, uy = p[a] - p[b]
        vx, vy = p[c] - p[b]
        denom = math.hypot(ux, uy) * math.hypot(vx, vy)
        if not math.isfinite(denom) or denom <= 1e-12:
            return 0., False
        return math.acos(max(-1., min(1., (ux * vx + uy * vy) / denom))) / math.pi, True

    triplets = [(1, 2, 3), (2, 3, 4)]
    for j in [5, 9, 13, 17]:
        triplets.extend([(0, j, j + 1), (j, j + 1, j + 2), (j + 1, j + 2, j + 3)])
    for hand, shoulder, elbow, wrist in [(33, 11, 13, 15), (54, 12, 14, 16)]:
        for point in [hand, wrist, elbow, hand + 8]:
            for axis in [0, 1]:
                emit(norm[point, axis], nv[point])
        palm = [hand + j for j in [0, 5, 9, 13, 17]]
        for axis in [0, 1]:
            emit(sum(norm[j, axis] for j in palm) / 5., all(nv[j] for j in palm))
        emit(*angle((shoulder, elbow, wrist)))
        for ids in triplets:
            emit(*angle(tuple(hand + j for j in ids)))
        direction = p[hand + 9] - p[hand]
        length = math.hypot(*direction)
        axis_ok = bool(valid[hand] and valid[hand + 9] and math.isfinite(length) and length > 1e-8)
        for v in direction:
            emit(v / length if axis_ok else 0, axis_ok)
        emit(math.hypot(*(norm[hand + 5] - norm[hand + 17])), nv[hand + 5] and nv[hand + 17])
    delta = norm[54] - norm[33]
    pair_known = nv[33] and nv[54]
    for v in delta:
        emit(v, pair_known)
    emit(math.hypot(*delta), pair_known)
    for hand, wrist in [(33, 15), (54, 16)]:
        emit(math.hypot(*(norm[hand] - norm[wrist])), nv[hand] and nv[wrist])
    for hand in [33, 54]:
        velocity = [0., 0.]
        ok = False
        if i > 0:
            dt = (int(timestamp[i]) - int(timestamp[i - 1])) / 1e6
            previous = np.asarray(points[i - 1], dtype=np.float64)
            good = observed[i - 1] & usable[i - 1] & np.isfinite(previous).all(axis=1)
            width = math.hypot(*(previous[11] - previous[12]))
            ok = bool(frame[i] - frame[i - 1] == 1 and 0 < dt <= .15 and nv[hand]
                      and good[11] and good[12] and good[hand]
                      and math.isfinite(width) and width > 1e-8)
            if ok:
                past = (previous[hand] - (previous[11] + previous[12]) / 2) / width
                velocity = (norm[hand] - past) / dt
        for v in velocity:
            emit(v, ok)
        speed = math.hypot(*velocity)
        moving = ok and math.isfinite(speed) and speed > 1e-8
        for v in velocity:
            emit(v / speed if moving else 0, moving)
    return np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=bool)


def main():
    frozen = read(U / "FROZEN_ONE_RECOVERY.json")
    result = read(U / "ONE_RECOVERY_RESULT.json")
    for path, h in frozen["pins"].items():
        pin(path, h)
    old_math = pin(O / "word100_observation_lineage_repair_20261101/review/verify_recovery_math_v2.py")
    spec = importlib.util.spec_from_file_location("independent_reviewer_scalar_math_readonly", old_math)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # Only independent scalar checker functions are called, never its old verify/main.
    source = result["source"]
    record = read(source["record"])
    native = load(source["observation"])
    raw = load(record["native_capture"]["raw"])
    assignment = load(Path(record["native_capture"]["candidate"]).with_name("assignment.npz"))
    baseline = load(source["input"], source["input_sha256"])
    saved = load(U / "P065_f48_recovered.npz", result["output_sha256"])
    contract = read(O.parent / "config/contract.json")
    threshold = contract["usable_rule"]["hand_score_threshold"]
    check(threshold == .5, "original score threshold")
    i, fid, pts = 48, 48, 1600000
    check(result["alias"] == frozen["alias"] == "P065", "alias")
    check(result["frame_id"] == frozen["frame_id"] == fid, "frozen frame")
    check(result["timestamp_us"] == frozen["timestamp_us"] == pts, "frozen PTS")
    for name, arrays in [("native", native), ("raw", raw), ("input", baseline)]:
        check(int(arrays["frame_id"][i]) == fid and int(arrays["timestamp_us"][i]) == pts, name + "exact identity")
    exact(saved["frame_id"], [fid], "only one exported source frame")
    exact(saved["timestamp_us"], [pts], "only exact original PTS")
    w, h = int(raw["width"][i]), int(raw["height"][i])
    count = int(raw["raw_count"][i])
    sides = assignment["sides"][i, :count].tolist()
    check(count == frozen["baseline_raw_count"] == 2, "both original raw detections")
    check(sides == frozen["baseline_sides"] == [-1, 1], "unassigned raw retained in duplicate checks")
    pose, pose_valid, box = raw["pose_xy_px"][i], raw["pose_usable"][i], source["box"]
    eligible = helper.eligibility(native["observed"][i], pose, pose_valid, box, w, h, sides)
    check(eligible == frozen["eligibility"] == result["eligibility"] == result["merge"]["eligibility"], "independent eligibility")
    check([x["side"] for x in eligible if x["eligible"]] == [0], "only unobserved left eligible")
    params = frozen["parameters"]
    check(params == dict(arm_gate=.25, arm_margin=.10, point_agreement=.08, duplicate_median=.10,
                         duplicate_wrist=.20, ambiguous_overlap=.50, ambiguous_wrist=.30,
                         crop_half_shoulder=.60, detector_confidence=.5), "unchanged generic parameters")
    full, crops = result["full"], result["crops"]
    expected = helper.expected_decisions(full, crops, raw["hand_xy_px"][i, :count], sides,
                                         pose, pose_valid, w, h, params)
    good, scale = helper.geometry_context(pose, pose_valid, w, h)
    px, py = pose[15]
    radius = scale * .6
    crop_box = [max(0, math.floor(px - radius)), max(0, math.floor(py - radius)),
                min(w, math.ceil(px + radius)), min(h, math.ceil(py + radius))]
    check(result["boxes"] == {"0": crop_box}, "current left-arm crop geometry")
    for crop in crops:
        check(crop["requested_side"] == 0 and crop["box"] == crop_box, "same frame requested crop")
        check(0 <= crop["score"] <= 1, "actual crop handedness score")
    candidate_xy = native["xy_px"].copy()
    candidate_observed = native["observed"].copy()
    candidate_usable = native["usable"].copy()
    additions, point_rows = [], []
    check(len(expected) == len(result["merge"]["decisions"]), "all full detections retained")
    for dec, actual in zip(expected, result["merge"]["decisions"]):
        for key, value in dec.items():
            check(actual[key] == value, "decision " + str(dec["index"]) + "/" + key)
        check(actual["old_merge_accepted"] == dec["accepted"], "old acceptance retained")
        side = dec["side"]
        applied = bool(dec["accepted"] and eligible[side]["eligible"])
        check(actual["applied"] == applied, "application eligibility")
        if not applied:
            continue
        full_point = np.asarray(full[dec["index"]]["xy"], dtype=np.float32)
        crop = crops[dec["crop_index"]]
        score = min(full[dec["index"]]["score"], crop["score"])
        supported = np.asarray(dec["joint_supported"], bool) & np.isfinite(full_point).all(axis=1)
        usable = supported & (score >= threshold)
        sl = slice(33 + 21 * side, 54 + 21 * side)
        candidate_xy[i, sl] = np.where(supported[:, None], full_point, np.nan)
        candidate_observed[i, sl] = supported
        candidate_usable[i, sl] = usable
        roi_count = sum(bool(usable[j]) and helper.finite_inside(full_point[j], box) for j in range(21))
        values = dict(actual_full_handedness_score=full[dec["index"]]["score"],
                      actual_crop_handedness_score=crop["score"], stored_hand_score=score,
                      supported_points=int(supported.sum()), final_roi_usable_points=roi_count,
                      application_reason="NEW_CURRENT_IMAGE_OBSERVATIONS_ONLY")
        for key, value in values.items():
            check(actual[key] == value, "applied measurement " + key)
        additions.append(dict(side=side, full_index=dec["index"], crop_index=dec["crop_index"],
                              observed_points=int(supported.sum()), usable_points=int(usable.sum()), roi_usable_points=roi_count))
        for j in range(21):
            difference = helper.distance(full[dec["index"]]["xy"][j], crop["xy"][j])
            point_rows.append(dict(joint=j, full_crop_distance_pixels=difference,
                limit_pixels=params["point_agreement"] * scale,
                within_agreement=difference <= params["point_agreement"] * scale,
                crop_point_inside=helper.finite_inside(crop["xy"][j], crop_box),
                supported=bool(supported[j]), roi_usable=bool(usable[j] and helper.finite_inside(full_point[j], box))))
    check(additions == result["merge"]["additions"], "addition ledger")
    check(len(additions) == 1 and additions[0]["observed_points"] == 12, "one left partial hand12points")
    # Native coordinates/observed flags remain untouched outside the added slot.
    for key, candidate in [("xy_px", candidate_xy), ("observed", candidate_observed)]:
        exact(candidate[:i], native[key][:i], "reconstruction prior frames " + key)
        exact(candidate[i + 1:], native[key][i + 1:], "reconstruction later frames " + key)
        exact(candidate[i, :33], native[key][i, :33], "pose unchanged " + key)
        exact(candidate[i, 54:], native[key][i, 54:], "right unchanged " + key)
    roi = np.array([[helper.finite_inside(point, box) for point in row] for row in candidate_xy])
    candidate_usable &= roi
    for field, rebuilt in [("xy_px", candidate_xy), ("observed", candidate_observed), ("usable", candidate_usable)]:
        exact(saved[field], rebuilt[i:i + 1], "saved " + field)
    for field in ["xy_px", "observed", "usable"]:
        exact(saved["original_" + field], native[field][i:i + 1], "original native snapshot " + field)
    exact(saved["usable"][0, 54:], native["usable"][i, 54:] & roi[i, 54:], "right ROI support preserved")
    exact(saved["baseline_x"], baseline["x"][i:i + 1], "all baseline feature values")
    exact(saved["baseline_mask"], baseline["mask"][i:i + 1], "all baseline feature masks")
    check(saved["x"].shape == saved["mask"].shape == (1, 69), "all69 opportunities retained")
    old_known = baseline["mask"][i]
    exact(saved["x"][0, old_known], baseline["x"][i, old_known], "all original supported values exact")
    check(saved["mask"][0, old_known].all(), "all original support retained")
    exact(saved["x"][0, 28:56], baseline["x"][i, 28:56], "right features exact all cells")
    exact(saved["mask"][0, 28:56], baseline["mask"][i, 28:56], "right feature support exact")
    row_values, row_mask = feature_row(candidate_xy, candidate_observed, candidate_usable,
                                      native["timestamp_us"], native["frame_id"], i)
    exact(saved["mask"][0], row_mask, "independent all69 masks")
    errors = np.abs(saved["x"][0].astype(float) - row_values.astype(float))
    for col in range(69):
        check(errors[col] <= 2e-6, "independent feature value " + source["feature_names"][col], float(errors[col]))
    gains = saved["mask"][0] & ~old_known
    check(int(gains.sum()) == result["added_feature_cells"] == 11, "11 newly supported features")
    check(int(old_known.sum()) == result["preserved_feature_cells"] == 38, "38 prior supported features")
    check([name for name, gain in zip(source["feature_names"], gains) if gain] == result["gained_feature_names"], "exact gained names")
    check(not saved["mask"][0, 11:25].any(), "no left finger-angle support invented")
    old_classifier_support = bool(baseline["mask"][i, :2].all() and baseline["mask"][i, 11:25].sum() >= 10)
    new_shape_input_support = bool(saved["mask"][0, :2].all() and saved["mask"][0, 11:25].sum() >= 10)
    check(not old_classifier_support and not new_shape_input_support, "left shape remains unsupported")
    # The only possible derivative dependency beyond f48 is f49.
    _, next_mask = feature_row(candidate_xy, candidate_observed, candidate_usable,
                               native["timestamp_us"], native["frame_id"], i + 1)
    discarded = int((next_mask & ~baseline["mask"][i + 1]).sum())
    check(discarded == result["out_of_scope_derivative_support_discarded"] == 0, "next-frame support not gained or exported")
    for key in ["new_classifier", "new_training", "new_dtw"]:
        check(result[key] == frozen[key] == 0, key)
    check(result["full_calls"] == result["crop_calls"] == 1, "one full and one crop call")
    check(result["old_prediction_masks_not_promoted"] is True, "posterior support contract")
    check(result["landmark_accuracy"] is None, "no landmark accuracy claim")
    prior_verify = read(U / "review/VERIFY_POINT.json")
    for path, h in prior_verify["source_pins"].items():
        pin(path, h)
    # Verifies the existing diagnostic and its probabilities/masks have not changed.
    check(sha(U / "POINT_DIAGNOSIS.json") == prior_verify["source_pins"][str(U / "POINT_DIAGNOSIS.json")], "point diagnostic unchanged")
    addendum = read(U / "review/PRODUCT_REVIEW_ADDENDUM.json")
    check(sha(addendum["prior_review"]) == addendum["current_prior_file_sha256"], "prior disclosed pin state unchanged")
    for path, h in list(PINS.items()):
        check(sha(path) == h, "read-only preservation " + path)
    PINS[str(Path(__file__).resolve())] = sha(__file__)
    report = dict(schema="causal_pilot_one_recovery_verification.v1",
        status="PASS" if not FAILURES else "FAIL", created_utc=datetime.now(timezone.utc).isoformat(),
        checks=CHECKS, failures=FAILURES, alias="P065", frame_id=fid, timestamp_us=pts,
        baseline_raw_count=count, baseline_raw_sides=sides,
        added_left_points=12, added_features=int(gains.sum()), preserved_features=int(old_known.sum()),
        gained_feature_names=result["gained_feature_names"], point_agreement_audit=point_rows,
        maximum_independent_feature_absolute_error=float(errors.max()), feature_numeric_tolerance=2e-6,
        left_finger_angle_support=int(saved["mask"][0, 11:25].sum()),
        old_shape_support=old_classifier_support, recovered_shape_geometric_support=new_shape_input_support,
        no_old_posterior_mask_promotion=True, independent_scalar_merge_helper=str(old_math),
        source_pins=PINS, actions=dict(new_inference=0, producer_import=0, producer_reexecution=0,
            old_experiment_rerun=0, new_training=0, new_path=0, RGB_decode_or_review=0, original_edits=0),
        limitations=["Two-scale and current-arm agreement are observation-quality checks, not anatomical landmark accuracy.",
            "Onlyf48 is exported. Other-frame preservation is supported by unchanged original full-file hashes, saved one-frame scope, producer range locks and independent dependency reconstruction; no unsaved runtime array is claimed to have been directly inspected.",
            "Saved output omits conf/hand_score; score provenance is verified from saved full/crop records and ledger, not claimed as independently inspected output fields.",
            "Exact original video decode is supported by frozen video hash, native PTS and producer static contract; this verifier does not decode RGB or rerun detection.",
            "Left shape support remains false; new geometry must not reclassify baseline-only saved posteriors as new-input model evidence.",
            "Prior product-review pin mismatch remains explicitly disclosed in PRODUCT_REVIEW_ADDENDUM; no completion ledger was rewritten."])
    with (U / "review/VERIFY_RECOVERY.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:report[k] for k in ["status", "checks", "failures", "maximum_independent_feature_absolute_error"]}, ensure_ascii=False))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
