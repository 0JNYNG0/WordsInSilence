"""Local independent audit of two new source clips using existing production APIs.

No source registration, classifier inference, fitting, held-out use or production
edits. No action annotations exist for these clips; none are synthesized here.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
WEB = ROOT / "research/front_keyphase_v1"
PILOT = ROOT / "research/sign_model_pilot_v1"
EVIDENCE = Path(__file__).resolve().parent
OUT = WEB / "runs/ksl77_input_prep_v1"
ACQUISITION = WEB / "evidence/additional_source_labels_20260918/ksl77_acquisition.json"
sys.path[:0] = [str(WEB / "src"), str(PILOT / "src")]

import av
import numpy as np
from wis_lab.capture.landmarker import Landmarker
from wis_lab.capture.video_source import extract
from wis_lab.service.neural import _channels
from pilot.web_input import load_web_clip


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def runs(mask):
    padded = np.r_[False, mask, False].astype(np.int8)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    return [[int(a), int(b - 1)] for a, b in zip(starts, ends)]


def verify(obs_path, channel_path, source, contract):
    with np.load(obs_path, allow_pickle=False) as z:
        xy, seen, usable, conf, score = (z[k] for k in ("xy_px", "observed", "usable", "conf", "hand_score"))
        ts, fid = z["timestamp_us"], z["frame_id"]
    with np.load(channel_path, allow_pickle=False) as z:
        x, bv, hv, cts, cfid = (z[k] for k in ("x", "body_valid", "hand_valid", "timestamp_us", "frame_id"))
    side = json.loads(obs_path.with_suffix(".sidecar.json").read_text(encoding="utf-8"))
    assert side["observation_sha256"] == sha(obs_path)
    assert side["source_sha256"] == sha(source["path"]) == source["sha256"]
    assert xy.shape == (len(ts), 75, 2) and xy.dtype == np.float32
    assert ts.dtype == np.int64 and ts[0] == 0 and np.all(np.diff(ts) > 0)
    np.testing.assert_array_equal(fid, np.arange(len(ts)))
    assert side["decoded_frames"] == len(ts) and side["dropped_non_monotonic_pts"] == 0
    # Read actual frame PTS independently, without replacing them with index/fps.
    with av.open(source["path"]) as container:
        stream = container.streams.video[0]
        times = [frame.pts * frame.time_base for frame in container.decode(stream)]
        avg_rate = str(stream.average_rate)
    expected_ts = np.array([round(float(t - times[0]) * 1e6) for t in times], dtype=np.int64)
    np.testing.assert_array_equal(ts, expected_ts)
    assert not (usable & ~seen).any() and np.isnan(xy[~seen]).all()
    assert np.isnan(conf[:, 33:]).all(), "Hand score must not become per-joint confidence"
    rule = contract["usable_rule"]
    width, height, margin = side["width"], side["height"], rule["bounds_margin_ratio"]
    bounds = (np.isfinite(xy).all(axis=2) & (xy[:, :, 0] >= -width * margin)
              & (xy[:, :, 0] <= width * (1 + margin)) & (xy[:, :, 1] >= -height * margin)
              & (xy[:, :, 1] <= height * (1 + margin)))
    eligible = np.concatenate([conf[:, :33] >= rule["pose_visibility_threshold"],
                               np.repeat(score >= rule["hand_score_threshold"], 21, axis=1)], axis=1)
    np.testing.assert_array_equal(usable, seen & bounds & eligible)
    indices = np.flatnonzero(usable[:, 33:].any(axis=1))
    assert len(indices)
    lo, hi = int(indices[0]), int(indices[-1])
    np.testing.assert_array_equal(cts, ts[lo:hi + 1])
    np.testing.assert_array_equal(cfid, fid[lo:hi + 1])
    selection = [0, 11, 12, 13, 14, 15, 16, 23, 24] + list(range(33, 75))
    coords, present = xy[lo:hi + 1, selection], usable[lo:hi + 1, selection]
    shoulders = present[:, 1] & present[:, 2]
    body_scale = np.median(np.linalg.norm(coords[shoulders, 1] - coords[shoulders, 2], axis=1))
    assert np.isfinite(body_scale) and body_scale > 0
    expected_bv = present & shoulders[:, None]
    np.testing.assert_array_equal(bv, expected_bv)
    origin = (coords[:, 1] + coords[:, 2]) / 2
    expected_body = np.zeros_like(coords)
    for j in range(51):
        good = expected_bv[:, j]
        expected_body[good, j] = (coords[good, j] - origin[good]) / body_scale
    np.testing.assert_allclose(x[:, :102].reshape(-1, 51, 2), expected_body, rtol=1e-6, atol=1e-6)
    scales = {}
    for name, start, dest in (("left", 9, 0), ("right", 30, 21)):
        lens = np.linalg.norm(coords[:, start + np.array([5, 9, 13, 17])] - coords[:, start, None], axis=2)
        good_lens = present[:, start, None] & present[:, start + np.array([5, 9, 13, 17])]
        hs = float(np.median(lens[good_lens])) if good_lens.any() else None
        scales[name] = hs
        expected_hv = present[:, start, None] & present[:, start:start + 21]
        if hs is None or hs <= 1e-6:
            expected_hv[:] = False
        np.testing.assert_array_equal(hv[:, dest:dest + 21], expected_hv)
        expected_hand = np.zeros((len(cts), 21, 2), dtype=np.float32)
        if hs is not None and hs > 1e-6:
            for j in range(21):
                good = expected_hv[:, j]
                expected_hand[good, j] = (coords[good, start + j] - coords[good, start]) / hs
        np.testing.assert_allclose(x[:, 102:186].reshape(-1, 42, 2)[:, dest:dest + 21],
                                   expected_hand, rtol=1e-6, atol=1e-6)
    assert x.shape == (len(cts), 280) and x.dtype == np.float32 and np.isfinite(x).all()
    np.testing.assert_array_equal(x[:, 186:237], bv)
    np.testing.assert_array_equal(x[:, 237:279], hv)
    assert x[0, 279] == 0
    np.testing.assert_allclose(x[1:, 279], np.diff(cts) / 1e6, rtol=1e-6)
    saved_adapter = load_web_clip(str(obs_path))
    np.testing.assert_array_equal(saved_adapter["x"], x)
    hands = {}
    for name, sl in (("left", slice(33, 54)), ("right", slice(54, 75))):
        all_points = usable[:, sl].all(axis=1)
        any_points = usable[:, sl].any(axis=1)
        hands[name] = {"all_21_usable_frames": int(all_points.sum()),
                       "any_usable_frames": int(any_points.sum()),
                       "all_21_usable_fraction": round(float(all_points.mean()), 4),
                       "no_usable_hand_frame_ranges": runs(~any_points),
                       "not_all_21_usable_frame_ranges": runs(~all_points),
                       "per_joint_usable_frames": usable[:, sl].sum(axis=0).tolist()}
    missing = ~usable[:, 33:].any(axis=1)
    return {"status": "PASS_INPUT_CONTRACT_ONLY_NOT_RECOGNITION", "raw_frames": len(ts),
            "tcn_frames": len(cts), "channels": x.shape[1], "active_span_frame_indices": [lo, hi],
            "active_span_means": "first-to-last usable hand; NOT a verified sign-action boundary",
            "ground_truth_action_window": None, "action_window_hand_quality": "UNASSESSED_NO_BOUNDARY_LABEL",
            "timestamps_first_last_us": [int(ts[0]), int(ts[-1])], "average_rate": avg_rate,
            "actual_pts_exact": True,
            "gap_min_median_max_us": [int(np.diff(ts).min()), float(np.median(np.diff(ts))), int(np.diff(ts).max())],
            "hands": hands, "both_hands_unusable_frames": int(missing.sum()),
            "both_hands_unusable_frame_ranges": runs(missing),
            "body_scale_px": float(body_scale), "hand_scales_px": scales,
            "extractor_id": side["extractor_id"], "quality": side["quality"],
            "handedness_label_disagreements": side["handedness_label_disagreements"],
            "handedness_label_comparisons": side["handedness_label_comparisons"],
            "unassigned_hand_detections": side["unassigned_hand_detections"],
            "checks": ["source_and_observation_hash", "independent_actual_frame_pts", "no_frame_loss",
                       "usable_rule_recomputed", "missing_nan_preserved", "42_hand_points_preserved",
                       "body_and_hand_normalization_recomputed", "masks_dt_recomputed", "persisted_live_input_exact"],
            "coordinate_accuracy": "NOT_VALIDATED_AGAINST_JOINT_GROUND_TRUTH",
            "recognition_accuracy": "NOT_MEASURED", "training_approved": False}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    contract_path = WEB / "config/contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    acquisition = json.loads(ACQUISITION.read_text(encoding="utf-8"))
    rows = [r for r in acquisition["files"] if r["triage_allowed"]]
    assert {r["member"] for r in rows} == {"16/00_16.MP4", "48/00_48.MP4"}
    assert acquisition["training_approved"] is False
    pin_paths = [contract_path, Path(__file__), ACQUISITION] + [Path(r["path"]) for r in rows] + [
        WEB / "src/wis_lab" / p for p in ("capture/landmarker.py", "capture/video_source.py", "core/observation.py", "service/neural.py")
    ] + [PILOT / "src/pilot" / p for p in ("web_input.py", "data.py")]
    pins = {str(p): sha(p) for p in pin_paths}
    result = {"schema": "wis_ksl77_input_preparation_v1", "status": "RUNNING", "clips": [],
              "started_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "source_hashes_before": pins,
              "versions": {k: importlib.metadata.version(k) for k in ("numpy", "av", "mediapipe")},
              "scope": "Only prefix00; separate diagnostic cache; no annotations invented or training registration",
              "classifier_inference_calls": 0, "training_runs": 0, "heldout_runs": 0,
              "other_34_clips_media_processed": 0, "production_changes": 0}
    try:
        for source in rows:
            assert source["split"] == "UNASSIGNED" and source["identity_verified"] is False
            assert sha(source["path"]) == source["sha256"]
            clip_id = "KSL77_00_" + str(source["source_label_id"])
            print("EXTRACT", clip_id, flush=True)
            with Landmarker(contract) as lm:
                obs = extract(source["path"], lm, clip_id)
            provenance = {**source, "training_approved": False, "semantic_target_id": None,
                          "ground_truth_action_window": None, "review_exposed": True}
            obs.sidecar.update(provenance=provenance, diagnostic_only=True)
            obs_path = OUT / (clip_id + ".npz")
            obs.save(str(obs_path))
            channels, why, support = _channels(obs)
            assert channels is not None, why
            indices = np.flatnonzero(obs.usable[:, 33:].any(axis=1))
            span = slice(int(indices[0]), int(indices[-1]) + 1)
            channel_path = OUT / (clip_id + ".tcn_input.npz")
            np.savez_compressed(channel_path, x=channels["x"], body_valid=channels["body_valid"],
                                hand_valid=channels["hand_valid"], timestamp_us=obs.timestamp_us[span],
                                frame_id=obs.frame_id[span])
            save(channel_path.with_suffix(".json"), {"schema": "wis_diagnostic_tcn_input_v1", "source": provenance,
                 "observation_sha256": sha(obs_path), "tcn_input_sha256": sha(channel_path),
                 "extractor_id": obs.sidecar["extractor_id"], "support": support,
                 "temporal_policy": "existing first-to-last usable hand span; full observation retained",
                 "annotation_policy": "no ground-truth action window exists; none invented"})
            review = verify(obs_path, channel_path, source, contract)
            result["clips"].append({"clip_id": clip_id, "target_candidate": source["target_candidate"],
                                    "observation_path": str(obs_path), "tcn_input_path": str(channel_path),
                                    "verification": review})
            save(EVIDENCE / "result.json", result)
            print("VERIFIED", clip_id, "raw", review["raw_frames"], "TCN", review["tcn_frames"], flush=True)
        assert {str(p): sha(p) for p in pin_paths} == pins
        result.update(status="PASS_INPUT_CONTRACT_ONLY_NOT_TRAINING_OR_GENERALIZATION", source_hashes_unchanged=True,
                      finished_kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                      elapsed_seconds=round(time.perf_counter() - started, 3),
                      output_bytes=sum(p.stat().st_size for p in OUT.iterdir()))
    except Exception:
        result.update(status="FAIL_FIRST_RUN_PRESERVED", error=traceback.format_exc())
        raise
    finally:
        save(EVIDENCE / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "output_bytes")}), flush=True)


if __name__ == "__main__":
    main()
