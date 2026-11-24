"""Local operational audit: invoke existing extractors; never register/train these clips."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
WEB = ROOT / "research/front_keyphase_v1"
PILOT = ROOT / "research/sign_model_pilot_v1"
OUT = WEB / "runs/isolated_word_input_prep_v1"
EVIDENCE = Path(__file__).resolve().parent
AUDIT = Path("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/TCN_일반화감사_20260918/word_input_audit.json")
sys.path[:0] = [str(WEB / "src"), str(PILOT / "src")]

import numpy as np
from wis_lab.capture.landmarker import Landmarker
from wis_lab.capture.video_source import extract
from wis_lab.service.neural import _channels
from pilot.web_input import load_web_clip


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def verify(obs_path, channels_path, source, contract):
    """Recompute contract invariants from saved raw arrays, without production helpers."""
    with np.load(obs_path, allow_pickle=False) as z:
        xy, observed, usable = z["xy_px"], z["observed"], z["usable"]
        ts, fid, conf, score = z["timestamp_us"], z["frame_id"], z["conf"], z["hand_score"]
    with np.load(channels_path, allow_pickle=False) as z:
        x, bv, hv, cts, cfid = z["x"], z["body_valid"], z["hand_valid"], z["timestamp_us"], z["frame_id"]
    side = json.loads(obs_path.with_suffix(".sidecar.json").read_text(encoding="utf-8"))
    assert side["observation_sha256"] == sha(obs_path)
    assert side["source_sha256"] == source["video_sha256"]
    assert xy.shape == (len(ts), 75, 2) and xy.dtype == np.float32
    assert ts.dtype == np.int64 and ts[0] == 0 and np.all(np.diff(ts) > 0)
    assert np.all(np.diff(fid) == 1) and fid[0] == 0
    assert side["decoded_frames"] == len(ts) and side["dropped_non_monotonic_pts"] == 0
    assert not (usable & ~observed).any() and np.isnan(xy[~observed]).all()
    assert np.isnan(conf[:, 33:]).all()  # handedness score is NOT a joint confidence
    rule = contract["usable_rule"]
    w, h, margin = side["width"], side["height"], rule["bounds_margin_ratio"]
    bounds = (np.isfinite(xy).all(axis=2) & (xy[:, :, 0] >= -w*margin)
              & (xy[:, :, 0] <= w*(1+margin)) & (xy[:, :, 1] >= -h*margin)
              & (xy[:, :, 1] <= h*(1+margin)))
    eligible = np.concatenate([conf[:, :33] >= rule["pose_visibility_threshold"],
                               np.repeat(score >= rule["hand_score_threshold"], 21, axis=1)], axis=1)
    np.testing.assert_array_equal(usable, observed & bounds & eligible)
    hand_rows = np.flatnonzero(usable[:, 33:].any(axis=1))
    lo, hi = int(hand_rows[0]), int(hand_rows[-1])
    np.testing.assert_array_equal(cts, ts[lo:hi+1])
    np.testing.assert_array_equal(cfid, fid[lo:hi+1])
    selection = [0, 11, 12, 13, 14, 15, 16, 23, 24] + list(range(33, 75))
    coords = xy[lo:hi+1, selection]
    present = usable[lo:hi+1, selection]
    shoulders = present[:, 1] & present[:, 2]
    body_scale = np.median(np.linalg.norm(coords[shoulders, 1]-coords[shoulders, 2], axis=1))
    assert np.isfinite(body_scale) and body_scale > 0
    expected_bv = present & shoulders[:, None]
    np.testing.assert_array_equal(bv, expected_bv)
    expected_body = np.zeros_like(coords)
    origin = (coords[:, 1] + coords[:, 2]) / 2
    for j in range(51):
        good = expected_bv[:, j]
        expected_body[good, j] = (coords[good, j] - origin[good]) / body_scale
    np.testing.assert_allclose(x[:, :102].reshape(-1, 51, 2), expected_body, rtol=1e-6, atol=1e-6)
    scales = {}
    for name, start, dest in [("left", 9, 0), ("right", 30, 21)]:
        lens = np.linalg.norm(coords[:, start+np.array([5, 9, 13, 17])]-coords[:, start, None], axis=2)
        valid_lens = present[:, start, None] & present[:, start+np.array([5, 9, 13, 17])]
        hs = float(np.median(lens[valid_lens])) if valid_lens.any() else None
        scales[name] = hs
        expected_hv = present[:, start, None] & present[:, start:start+21]
        if hs is None or hs <= 1e-6:
            expected_hv[:] = False
        np.testing.assert_array_equal(hv[:, dest:dest+21], expected_hv)
        expected_hand = np.zeros((len(cts), 21, 2), dtype=np.float32)
        if hs is not None and hs > 1e-6:
            for j in range(21):
                good = expected_hv[:, j]
                expected_hand[good, j] = (coords[good, start+j]-coords[good, start])/hs
        np.testing.assert_allclose(x[:, 102:186].reshape(-1, 42, 2)[:, dest:dest+21],
                                   expected_hand, rtol=1e-6, atol=1e-6)
    assert x.shape == (len(cts), 280) and x.dtype == np.float32 and np.isfinite(x).all()
    np.testing.assert_array_equal(x[:, 186:237], bv)
    np.testing.assert_array_equal(x[:, 237:279], hv)
    assert x[0, 279] == 0
    np.testing.assert_allclose(x[1:, 279], np.diff(cts)/1e6, rtol=1e-6)
    # Separate persisted-input adapter must agree with the live service adapter.
    persisted_adapter = load_web_clip(str(obs_path))
    np.testing.assert_array_equal(persisted_adapter["x"], x)
    in_annotation = (ts >= source["annotation_start_s"]*1e6) & (ts <= source["annotation_end_s"]*1e6)
    assert in_annotation.any()
    per_hand = {}
    for name, sl in [("left", slice(33, 54)), ("right", slice(54, 75))]:
        per_hand[name] = {
            "full_clip_all_21_usable_frames": int(usable[:, sl].all(axis=1).sum()),
            "active_span_all_21_usable_frames": int(usable[lo:hi+1, sl].all(axis=1).sum()),
            "annotation_all_21_usable_frames": int(usable[in_annotation, sl].all(axis=1).sum()),
            "active_span_per_joint_usable_frames": usable[lo:hi+1, sl].sum(axis=0).tolist(),
        }
    return {"status": "PASS_INPUT_CONTRACT_ONLY", "raw_frames": len(ts), "tcn_frames": len(cts),
            "tcn_channels": 280, "active_span_frame_indices": [lo, hi],
            "timestamp_first_last_us": [int(ts[0]), int(ts[-1])],
            "gap_min_median_max_us": [int(np.diff(ts).min()), float(np.median(np.diff(ts))), int(np.diff(ts).max())],
            "annotation_frames": int(in_annotation.sum()), "hands": per_hand,
            "body_scale_px": float(body_scale), "hand_scales_px": scales,
            "extractor_id": side["extractor_id"], "quality": side["quality"],
            "handedness_label_disagreements": side["handedness_label_disagreements"],
            "handedness_label_comparisons": side["handedness_label_comparisons"],
            "unassigned_hand_detections": side["unassigned_hand_detections"],
            "checks": ["source_and_observation_hash", "monotonic_pts_no_frame_loss", "usable_rule_recomputed",
                       "missing_nan_preserved", "full_42_hand_points", "body_normalization_recomputed",
                       "hand_normalization_recomputed", "masks_and_dt_recomputed", "persisted_live_adapter_exact_match"],
            "recognition_accuracy": "NOT_MEASURED", "semantic_equivalence": "UNASSESSED",
            "training_approved": False}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    contract_path = WEB / "config/contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    pin_paths = [contract_path, Path(__file__), AUDIT] + [WEB / "src/wis_lab" / x for x in
        ["capture/landmarker.py", "capture/video_source.py", "core/observation.py", "service/neural.py"]] + [
        PILOT / "src/pilot/web_input.py", PILOT / "src/pilot/data.py"]
    pins = {str(p): sha(p) for p in pin_paths}
    rows = [r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["source_rows"] if r.get("local_video")]
    assert {r["clip_id"] for r in rows} == {"NIA_SL_WORD1544_REAL01_F", "NIA_SL_WORD2485_REAL01_F"}
    result = {"schema": "wis_isolated_word_input_preparation_v1", "status": "RUNNING", "clips": [],
              "started_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "source_hashes_before": pins,
              "versions": {k: importlib.metadata.version(k) for k in ["numpy", "av", "mediapipe"]},
              "scope": "local diagnostic cache only; no manifest registration, classifier inference, training or held-out use"}
    try:
        for source in rows:
            assert source["signer_id"] == 1 and source["view"] == "F"
            assert source["training_approved"] is False and source["semantic_target_id"] is None
            clip_id = source["clip_id"]
            print("EXTRACT", clip_id, flush=True)
            with Landmarker(contract) as lm:
                obs = extract(source["local_video"], lm, clip_id)
            assert obs.sidecar["source_sha256"] == source["video_sha256"]
            obs.sidecar["provenance"] = source
            obs.sidecar["diagnostic_only"] = True
            obs_path = OUT / (clip_id + ".npz")
            obs.save(str(obs_path))
            channels, why, support = _channels(obs)
            assert channels is not None, why
            idx = np.flatnonzero(obs.usable[:, 33:].any(axis=1))
            span = slice(int(idx[0]), int(idx[-1])+1)
            channels_path = OUT / (clip_id + ".tcn_input.npz")
            np.savez_compressed(channels_path, x=channels["x"], body_valid=channels["body_valid"],
                                hand_valid=channels["hand_valid"], timestamp_us=obs.timestamp_us[span],
                                frame_id=obs.frame_id[span])
            meta = {"schema": "wis_diagnostic_tcn_input_v1", "source": source,
                    "observation_sha256": sha(obs_path), "tcn_input_sha256": sha(channels_path),
                    "extractor_id": obs.sidecar["extractor_id"], "frames": channels["frames"],
                    "body_scale": channels["body_scale"], "hand_scales": channels["hand_scales"],
                    "support": support, "semantic_target_id": None, "training_approved": False,
                    "temporal_policy": "existing web first-to-last usable hand span; original full observation retained",
                    "annotation_policy": "metadata only; not treated as verified full-action boundaries"}
            write_json(channels_path.with_suffix(".json"), meta)
            review = verify(obs_path, channels_path, source, contract)
            result["clips"].append({"clip_id": clip_id, "raw_label": source["raw_label"],
                                    "observation_path": str(obs_path), "tcn_input_path": str(channels_path),
                                    "verification": review})
            write_json(EVIDENCE / "result.json", result)
            print("VERIFIED", clip_id, "raw", review["raw_frames"], "TCN", review["tcn_frames"], flush=True)
        after = {str(p): sha(p) for p in pin_paths}
        assert after == pins, "operational input/source changed during run"
        result.update(status="PASS_INPUT_CONTRACT_ONLY", source_hashes_unchanged=True,
                      finished_kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                      elapsed_seconds=round(time.perf_counter()-started, 3),
                      output_bytes=sum(p.stat().st_size for p in OUT.iterdir()))
    except Exception:
        result.update(status="FAIL_FIRST_RUN_PRESERVED", error=traceback.format_exc())
        raise
    finally:
        write_json(EVIDENCE / "result.json", result)
    print(json.dumps({k: result[k] for k in ["status", "elapsed_seconds", "output_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
