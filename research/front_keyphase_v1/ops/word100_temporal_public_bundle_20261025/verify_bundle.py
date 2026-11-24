"""Independent saved-bundle audit; never imports or calls the preparation code.

Reads public sources and packaged arrays only. No detector/model/training code.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
OPS = HERE.parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(4 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def same(left, right, context):
    assert left.shape == right.shape and left.dtype == right.dtype, context
    assert np.array_equal(left, right, equal_nan=True), context


def fullpath(path):
    return str(Path(path).resolve())


def gap_spans(mask):
    result = []
    opening = None
    for i, absent in enumerate(mask.tolist() + [False]):
        if absent and opening is None:
            opening = i
        elif not absent and opening is not None:
            result.append((opening, i))
            opening = None
    return result


def main():
    assert (HERE / "PREPARE_RESULT.json").is_file(), "Preparation incomplete"
    assert not (HERE / "VERIFY_RESULT.json").exists(), "Do not overwrite verification"
    manifest = read(HERE / "MANIFEST.json")
    summary = read(HERE / "PREPARE_RESULT.json")
    links = read(HERE / "GLOSS_REFERENCE_INDEX.json")
    gaps = read(HERE / "GAP_CANDIDATES.json")
    queue = read(HERE / "REVIEW_QUEUE.json")
    source_pins = read(HERE / "SOURCE_PINS.json")
    protected = read(HERE / "PROTECTED_BEFORE.json")
    source_full = read(OPS / "word100_fullframe_hand_merge_20261020/FROZEN.json")
    source_track = read(OPS / "word100_image_tracking_pair_20261020/FROZEN.json")
    source_small = read(OPS / "word100_small_multipeople_angle_20261021/PREPARED_INDEX.json")
    source_small_frozen = read(OPS / "word100_small_multipeople_angle_20261021/FROZEN.json")
    full_index = read(OPS / "word100_fullframe_hand_merge_20261020/INDEX.json")
    labels = read(OPS / "word100_training_20261013/MANIFEST.json")
    labels_by = {r["clip_id"]: r for r in labels}
    full_by = {r["clip_id"]: r for r in source_full["rows"]}
    small_by = {r["clip_id"]: r for r in source_small}
    track_by = {r["clip_id"]: r for r in source_track["rows"]}
    index_by = {r["clip_id"]: r for r in full_index}
    ids = {r["clip_id"] for r in manifest}
    assert len(manifest) == len(ids) == 61
    assert ids == set(full_by) | set(small_by)
    assert len(set(full_by) & set(small_by)) == 2
    assert len({r["word_code"] for r in manifest}) == 24
    historical = {}
    historical_collisions = []

    def history_add(path, value):
        p = fullpath(path)
        if p in historical and historical[p] != value:
            historical_collisions.append(p)
        historical[p] = value

    for snapshot in (source_full, source_track, source_small_frozen):
        for p, value in snapshot["pins"].items():
            history_add(p, value)
    for row in full_index:
        history_add(row["path"], row["sha256"])
        for p, value in row["outputs"].items():
            history_add(p, value)
    for row in source_small:
        for p, value in row.get("output_hashes", {}).items():
            history_add(p, value)
    # Only pins actually used by this bundle must share an unambiguous history.
    assert not (set(historical_collisions) & set(source_pins)), "Conflicting source history"
    historical_matches = 0
    for p, value in source_pins.items():
        assert digest(p) == value, ("source_changed", p)
        if p in historical:
            assert value == historical[p], ("historical_mismatch", p)
            historical_matches += 1
    for p, value in protected.items():
        assert digest(p) == value, ("protected_changed", p)

    count = Counter()
    expected_gaps = []
    expected_review_frames = set()
    palm_indices = [33, 38, 42, 46, 50]
    feature_keys = {"frame_id", "timestamp_us", "roi_box", "width", "height", "assignment_sides", "point_origin_roi"}
    channel_fields = ("xy_px", "observed", "usable", "conf", "hand_score")
    raw_fields = ("raw_count", "hand_xy_px", "raw_label", "raw_score", "hand_finite", "pose_xy_px", "pose_usable", "pose_conf")
    feature_keys.update(prefix + "_" + k for prefix in ("native", "baseline_roi", "redetected_roi") for k in channel_fields)
    feature_keys.update("raw_" + k for k in raw_fields)
    for row in manifest:
        cid = row["clip_id"]
        label = labels_by[cid]
        source = full_by.get(cid, small_by.get(cid))
        t = track_by[cid]["source"] if cid in track_by else source["source"]
        assert label["role"] == row["classifier_prior_role"] == "train_candidate"
        assert row["signer_id"] == label["signer_id"] in range(1, 10)
        assert row["word_code"] == label["word_code"] == source["word_code"]
        assert row["display_label"] == label["canonical_label"] == source["label"]
        assert row["gloss_key"] == "AIHub_103:" + row["word_code"]
        assert row["source_group"] == cid
        assert row["bundle_role"] == "known_public_development_regression"
        assert row["refiner_training_use"] == "NOT_ASSIGNED_NOT_TRAINED"
        assert row["pointwise_anatomical_ground_truth"] == "NOT_AVAILABLE"
        assert row["hidden_3d_ground_truth"] == "NOT_AVAILABLE"
        assert fullpath(row["rgb_path"]) == fullpath(source["video"])
        assert digest(row["rgb_path"]) == row["rgb_sha256"] == t["sha256"]
        assert row["source_archive"] == t["archive"] and row["source_member"] == t["member"]
        assert Path(row["rgb_path"]).stat().st_size == row["rgb_bytes"] == t["bytes"]
        assert row["evaluation_annotation_intervals_only"] == source["intervals"]
        assert row["roi_box_source_pixels"] == source["box"]
        assets = row["source_assets"]
        for key in ("raw", "native", "assignment", "old_masked"):
            assert fullpath(assets[key]) == fullpath(source[key])
        if cid in full_by:
            assert fullpath(assets["detections"]) == fullpath(index_by[cid]["path"])
            assert fullpath(assets["merged_roi"]) == fullpath(OPS / "word100_fullframe_hand_merge_20261020/outputs" / cid / "masked.npz")
        else:
            assert fullpath(assets["detections"]) == fullpath(source["detections"])
            assert fullpath(assets["merged_roi"]) == fullpath(source["observation"])
        for p, value in row["source_pins"].items():
            assert source_pins[p] == value
        assert row["historical_pins_matched"] == sum(p in historical for p in row["source_pins"])
        assert digest(row["arrays"]) == row["arrays_sha256"]
        z = arrays(row["arrays"])
        assert set(z) == feature_keys, (cid, "model_array_fields", set(z) ^ feature_keys)
        assert all(value.dtype.kind not in "OUS" for value in z.values()), "Unexpected label/text/object arrays"
        n = row["frames"]
        assert n == row["frame_decode_count"]
        assert np.array_equal(z["frame_id"], np.arange(n))
        times = z["timestamp_us"]
        assert len(times) == n and times[0] == 0 and np.all(np.diff(times) > 0)
        assert 0 <= row["max_clock_error_us"] <= 1
        assert row["clock_authority"].startswith("decoded per-frame video PTS")
        assert row["frame_cadence_fps"] > 0 and row["container_average_fps"] > 0
        assert np.array_equal(z["roi_box"], source["box"])
        source_arrays = {key: arrays(assets[key]) for key in ("native", "old_masked", "merged_roi", "raw", "assignment")}
        for prefix, name in (("native", "native"), ("baseline_roi", "old_masked"), ("redetected_roi", "merged_roi")):
            original = source_arrays[name]
            for k in channel_fields:
                same(z[prefix + "_" + k], original[k], (cid, prefix, k))
                count["channel_scalar_values_verified"] += z[prefix + "_" + k].size
            for k in ("frame_id", "timestamp_us"):
                same(z[k], original[k], (cid, name, k))
            assert z[prefix + "_xy_px"].shape == (n, 75, 2)
        for k in raw_fields:
            same(z["raw_" + k], source_arrays["raw"][k], (cid, "raw", k))
        for k in ("width", "height", "frame_id", "timestamp_us"):
            same(z[k], source_arrays["raw"][k], (cid, "raw clock/dimensions", k))
        same(z["assignment_sides"], source_arrays["assignment"]["sides"], (cid, "assignment"))
        assert z["assignment_sides"].shape == (n, 2)
        x0, y0, x1, y1 = source["box"]
        for prefix in ("baseline_roi", "redetected_roi"):
            xy = z[prefix + "_xy_px"]
            inside = (xy[..., 0] >= x0) & (xy[..., 0] < x1) & (xy[..., 1] >= y0) & (xy[..., 1] < y1)
            assert not np.any(z[prefix + "_usable"] & ~inside), (cid, "ROI leak")
        keep = z["baseline_roi_usable"]
        assert np.all(z["redetected_roi_usable"][keep])
        assert np.array_equal(z["redetected_roi_xy_px"][keep], z["baseline_roi_xy_px"][keep], equal_nan=True)
        assert np.array_equal(z["redetected_roi_xy_px"][:, :33], z["baseline_roi_xy_px"][:, :33], equal_nan=True)
        origin = np.zeros((n, 75), dtype=np.uint8)
        origin[z["redetected_roi_usable"]] = 2
        origin[keep] = 1
        same(origin, z["point_origin_roi"], (cid, "provenance"))
        det = read(assets["detections"])
        assert len(det["frames"]) == n
        for i, d in enumerate(det["frames"]):
            assert d["frame_id"] == i and d["timestamp_us"] == int(times[i]) and "full" in d
        core = np.zeros(n, dtype=bool)
        for start, end in row["evaluation_annotation_intervals_only"]:
            core = core | ((times / 1e6 >= start) & (times / 1e6 <= end))
        assert core.any()
        selected = set([0, n - 1] + [int(v) for v in np.quantile(np.flatnonzero(core), [0.25, 0.5, 0.75])])
        expected_review_frames.update((cid, i, int(times[i])) for i in selected)
        for side in (0, 1):
            columns = [i + side * 21 for i in palm_indices]
            present = np.all(z["baseline_roi_usable"][:, columns], axis=1)
            native_present = np.all(z["native_usable"][:, columns], axis=1)
            merged_present = np.all(z["redetected_roi_usable"][:, columns], axis=1)
            for first, end in gap_spans(~present):
                expected_gaps.append({"clip_id": cid, "side": side, "start_index": first, "end_index_exclusive": end,
                    "start_us": int(times[first]), "last_us": int(times[end - 1]), "frames": end - first,
                    "core_frames": int(core[first:end].sum()), "full_native_palm_frames": int(native_present[first:end].sum()),
                    "redetected_roi_palm_frames": int(merged_present[first:end].sum()),
                    "reappears": bool(end != n and present[end]), "reason": "UNREVIEWED_AVAILABILITY_GAP_NOT_AN_OCCLUSION_LABEL"})
        count.update(clips=1, frames=n, rgb_bytes=row["rgb_bytes"], array_bytes=Path(row["arrays"]).stat().st_size,
                     retained_usable_roi_points=int(keep.sum()), added_available_points=int((z["redetected_roi_usable"] & ~keep).sum()))
    assert expected_gaps == gaps, "Independent gap tally differs"
    assert queue["status"] == "UNANNOTATED_NOT_GROUND_TRUTH"
    got_review = {(s["clip_id"], s["frame_id"], s["timestamp_us"]) for s in queue["samples"]}
    assert got_review == expected_review_frames and len(got_review) == len(queue["samples"])
    assert all(s["visibility"] == "UNANNOTATED" and s["keypoints_2d"] is None for s in queue["samples"])

    expected_links = {}
    for query in manifest:
        expected_links[query["clip_id"]] = {r["clip_id"] for r in manifest if r["word_code"] == query["word_code"] and r["signer_id"] != query["signer_id"]}
    assert len(links["reference_links"]) == 61
    assert {r["query_clip"] for r in links["reference_links"]} == ids
    for row in links["reference_links"]:
        assert set(row["reference_clips"]) == expected_links[row["query_clip"]]
        assert len(row["reference_clips"]) == len(set(row["reference_clips"]))
        expected_status = "PUBLIC_REFERENCE_LINKS_READY" if row["reference_clips"] else "NO_OTHER_SIGNER_IN_THIS_BUNDLE"
        assert row["status"] == expected_status
    glossary = links["glossary"]
    assert len(glossary) == 100
    byword = defaultdict(list)
    for r in labels:
        byword[r["word_code"]].append(r)
    for row in glossary:
        original = byword[row["source_word_id"]]
        assert {(x["canonical_label"], x["label_index"]) for x in original} == {(row["display_label"], row["classifier_index"])}
        assert row["gloss_key"] == "AIHub_103:" + row["source_word_id"]
        assert set(row["in_bundle_clips"]) == {x["clip_id"] for x in manifest if x["word_code"] == row["source_word_id"]}
        assert row["cross_corpus_equivalence"] == "UNVERIFIED_NO_AUTOMATIC_MERGE" and row["phonological_attributes"] is None
    for k, value in summary["counts"].items():
        assert value == count[k], ("count_mismatch", k, value, count[k])
    assert summary["words"] == 24 and summary["signers"] == list(range(1, 10))
    assert summary["person_clip_counts"] == {str(k): v for k, v in Counter(r["signer_id"] for r in manifest).items()}
    assert summary["reference_links"] == sum(len(r) for r in expected_links.values())
    assert summary["queries_without_other_person_reference"] == sum(not r for r in expected_links.values())
    assert summary["gap_candidate_intervals"] == len(gaps) and summary["review_queue_frames"] == len(queue["samples"])
    assert summary["source_files_pinned"] == len(source_pins) and summary["protected_files_unchanged"] == len(protected)
    assert summary["new_training"] == summary["new_ml_inference"] == summary["user_recording_access"] == 0

    # Independent bounded RGB decode, not visual annotation or a second all-video decode.
    import cv2
    decoded = []
    clock_edge = next(r for r in manifest if r["clip_id"] == "NIA_SL_WORD0949_REAL01_F")
    for row in (manifest[0], manifest[len(manifest) // 2], clock_edge):
        cap = cv2.VideoCapture(row["rgb_path"])
        assert cap.isOpened()
        n = 0
        pts = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert [image.shape[1], image.shape[0]] == row["video_dimensions"]
            pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000.0)
            n += 1
        cap.release()
        assert n == row["frames"]
        actual_clock_error = float(np.max(np.abs(arrays(row["arrays"])["timestamp_us"] - np.asarray(pts))))
        assert actual_clock_error <= 1
        assert abs(actual_clock_error - row["max_clock_error_us"]) < 1e-5
        decoded.append({"clip_id": row["clip_id"], "frames": n, "dimensions": row["video_dimensions"], "decoded_pts_max_error_us": actual_clock_error})
    return {"status": "PASS", "at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "author": "temporal_bundle_inventory separate verifier; did not import/call prepare.py",
        "scope": "Saved public bundle source/array/role/provenance/reference and arithmetic verification, not anatomical ground truth or model performance",
        "counts": dict(count), "source_hashes_rechecked": len(source_pins), "historical_pins_matched": historical_matches,
        "protected_hashes_rechecked": len(protected), "reference_links": summary["reference_links"],
        "queries_without_reference": summary["queries_without_other_person_reference"],
        "gap_intervals_recomputed": len(gaps), "review_queue_frames": len(queue["samples"]),
        "independent_rgb_decode": decoded, "rgb_decode_frames": sum(r["frames"] for r in decoded),
        "model_array_field_allowlist": sorted(feature_keys), "new_training": 0, "new_inference": 0,
        "resolved_preparation_check_issue": "Preserved FIRST_ERROR and first_attempt. Average container FPS was not frame PTS; verified original stored timestamps unchanged against sources and decoded edge-video per-frame PTS.",
        "limitations": ["All 61 are known TRAIN public source clips, not new held-out people", "No new independent pointwise anatomical labels", "Gap records are availability gaps, not confirmed occlusions", "Three-video independent RGB decode only; preparation recorded all61 decode"]}


if __name__ == "__main__":
    try:
        result = main()
    except Exception:
        result = {"status": "FAIL", "at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "traceback": traceback.format_exc()}
        fail_path = HERE / "VERIFY_RESULT_FIRST_ERROR.json"
        with fail_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        print(result["traceback"])
        raise
    with (HERE / "VERIFY_RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
