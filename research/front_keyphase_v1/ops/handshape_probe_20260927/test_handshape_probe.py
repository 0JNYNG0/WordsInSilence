"""Synthetic-only stdlib unittest suite for handshape_probe. No MediaPipe, no camera, no real data."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

import handshape_probe as hp

W, H, T = 320, 240, 8
ROWS = [0, 2, 3, 5, 7]


def lm(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def hand_lms(wx, wy):
    return [lm(wx, wy)] + [lm(wx + 0.005 * k, wy - 0.005 * k, -0.01) for k in range(1, 21)]


def cat(name, score, index=-1):
    return SimpleNamespace(category_name=name, score=score, index=index, display_name="")


def fake_result(*hands):
    """hands: (wrist_x_norm, wrist_y_norm, [gesture cats], raw_handedness_name)."""
    return SimpleNamespace(gestures=[list(h[2]) for h in hands],
                           handedness=[[cat(h[3], 0.9, 0)] for h in hands],
                           hand_landmarks=[hand_lms(h[0], h[1]) for h in hands],
                           hand_world_landmarks=[hand_lms(0.0, 0.0) for _ in hands])


def pose(n=T):
    xy = np.full((n, 75, 2), np.nan)
    usable = np.zeros((n, 75), bool)
    xy[:, 11], xy[:, 12], xy[:, 0] = (100.0, 200.0), (200.0, 200.0), (150.0, 150.0)
    usable[:, [0, 11, 12]] = True
    return xy, usable


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def fake_frames(video_path, ordinals):
    return {o: (np.full((H, W, 3), (o * 20) % 256, np.uint8), o * 33333) for o in ordinals}


class Recorder:
    def __init__(self, results):
        self.results, self.calls = list(results), []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.results[len(self.calls) - 1]


def factory(rec):
    return lambda model_path: (rec, lambda: None)


def make_fixture(tmp, mutate=None):
    root = os.path.join(tmp, "root")
    os.mkdir(root)
    paths = {k: os.path.join(root, n) for k, n in
             (("video", "clip.mp4"), ("npz", "clip.npz"), ("sidecar", "clip.json"), ("model", "g.task"))}
    for k in ("video", "sidecar", "model"):
        with open(paths[k], "wb") as fh:
            fh.write(f"synthetic {k}".encode())
    xy, usable = pose()
    fid = np.arange(T, dtype=np.int64)
    np.savez(paths["npz"], xy_px=xy, usable=usable, observed=usable.copy(), frame_id=fid,
             timestamp_us=fid * 33333, hand_score=np.zeros((T, 2)))
    paths["app"] = os.path.join(tmp, "app_source.py")
    with open(paths["app"], "wb") as fh:
        fh.write(b"# protected app source")
    cfg = {"root": root, "model_path": paths["model"], "model_sha256": sha(paths["model"]),
           "model_url": "https://example.invalid/g.task", "policy": dict(hp.FIXED_POLICY),
           "protected_files": {paths["app"]: sha(paths["app"])},
           "inputs": [{"clip_id": "NIKL_TEST_0001", "word_id": "W0001", "display_name": "synthetic",
                       "video_path": paths["video"], "video_sha256": sha(paths["video"]),
                       "npz_path": paths["npz"], "sha256": sha(paths["npz"]),
                       "sidecar_path": paths["sidecar"], "sidecar_sha256": sha(paths["sidecar"]),
                       "width": W, "height": H,
                       "samples": [{"row": r, "frame_id": r, "timestamp_us": r * 33333} for r in ROWS]}]}
    if mutate:
        mutate(cfg)
    cfg_path = os.path.join(tmp, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return cfg_path, cfg, paths


def strict_load(path):
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.read(), parse_constant=hp._reject_constant)


class RelativePositionTests(unittest.TestCase):
    def rel(self, xy, us, wrist, row=1):
        w, _, reason = hp.clip_shoulder_width(xy, us)
        return hp.relative_positions(list(wrist), hp.frame_anchors(xy[row], us[row]), w, reason)

    def test_known_value_and_translation_scale_invariance(self):
        xy, us = pose()
        xy[3, 12] = (500.0, 200.0)  # one outlier row: median must ignore it
        base = self.rel(xy, us, (170.0, 260.0))
        # shoulder mid (150,200), width 100 -> (0.2,0.6); nose (150,150) -> (0.2,1.1)
        np.testing.assert_allclose(base["shoulder_position"], [0.2, 0.6])
        np.testing.assert_allclose(base["nose_position"], [0.2, 1.1])
        for k, (tx, ty) in ((2.5, (37.0, -12.0)), (0.4, (-5.0, 90.0))):
            got = self.rel(xy * k + (tx, ty), us, (170.0 * k + tx, 260.0 * k + ty))
            np.testing.assert_allclose(got["shoulder_position"], base["shoulder_position"], atol=1e-9)
            np.testing.assert_allclose(got["nose_position"], base["nose_position"], atol=1e-9)

    def test_missing_nose_only_nose_null(self):
        xy, us = pose()
        us[1, 0] = False
        r = self.rel(xy, us, (170.0, 260.0))
        self.assertIsNone(r["nose_position"])
        self.assertEqual(r["nose_position_reason"], "nose_not_usable_in_frame")
        self.assertIsNotNone(r["shoulder_position"])

    def test_missing_shoulder_in_frame_vs_whole_clip(self):
        xy, us = pose()
        us[1, 11] = False
        r = self.rel(xy, us, (170.0, 260.0))
        self.assertIsNone(r["shoulder_position"])
        self.assertIsNotNone(r["shoulder_position_reason"])
        self.assertIsNotNone(r["nose_position"])  # width still known from other rows
        us[:, 11] = False
        r = self.rel(xy, us, (170.0, 260.0))
        self.assertIsNone(r["shoulder_position"])
        self.assertIsNone(r["nose_position"])
        self.assertTrue(r["shoulder_position_reason"] and r["nose_position_reason"])

    def test_nan_with_usable_true_is_unusable_and_usable_false_is_respected(self):
        xy, us = pose()
        xy[1, 0] = np.nan
        self.assertIsNone(self.rel(xy, us, (170.0, 260.0))["nose_position"])
        xy, us = pose()
        xy[:, 11] = np.nan  # usable stays True but coordinates NaN everywhere
        w, n, reason = hp.clip_shoulder_width(xy, us)
        self.assertIsNone(w)
        self.assertEqual(n, 0)
        self.assertIsNotNone(reason)
        r = self.rel(pose()[0], pose()[1], (float("nan"), 1.0))
        self.assertIsNone(r["shoulder_position"])
        self.assertEqual(r["shoulder_position_reason"], "wrist_landmark_not_finite")


class ParseAndSummaryTests(unittest.TestCase):
    def anchors(self):
        xy, us = pose()
        return hp.frame_anchors(xy[0], us[0])

    def test_two_hands_positions_attach_to_their_own_detection(self):
        res = fake_result((0.5, 0.6, [cat("Victory", 0.7)], "Left"),
                          (0.25, 0.5, [cat("Closed_Fist", 0.6)], "Left"))
        rec = hp.parse_result(res, 400, 400, self.anchors(), 100.0, None)
        a, b = rec["hands"]
        self.assertEqual((a["detector_index"], b["detector_index"]), (0, 1))
        self.assertEqual(a["top_category"]["category_name"], "Victory")
        self.assertEqual(b["top_category"]["category_name"], "Closed_Fist")
        # wrist px (200,240)->(0.5,0.4); (100,200)->(-0.5,0.0); same raw handedness, no side matching
        np.testing.assert_allclose(a["body_relative"]["shoulder_position"], [0.5, 0.4])
        np.testing.assert_allclose(b["body_relative"]["shoulder_position"], [-0.5, 0.0])

    def test_no_hand_differs_from_none_category(self):
        empty = hp.parse_result(fake_result(), W, H, self.anchors(), 100.0, None)
        none = hp.parse_result(fake_result((0.5, 0.5, [cat("None", 0.9), cat("Victory", 0.1)], "Right")),
                               W, H, self.anchors(), 100.0, None)
        self.assertEqual(empty["detection_status"], "no_hand_detected")
        self.assertEqual(empty["n_hands"], 0)
        self.assertEqual(none["detection_status"], "hands_detected")
        self.assertEqual(none["hands"][0]["top_category"]["category_name"], "None")
        s = hp.summarize([empty, none])
        self.assertEqual(s["frames_no_hand"], 1)
        self.assertEqual(s["hand_detections"], 1)
        self.assertEqual(s["top_category_counts"], {"None": 1})

    def test_summary_counts_hand_computed(self):
        a = self.anchors()
        frames = [hp.parse_result(r, W, H, a, 100.0, None) for r in (
            fake_result(),
            fake_result((0.3, 0.5, [cat("Victory", 0.8)], "Left"), (0.7, 0.5, [cat("Closed_Fist", 0.5)], "Right")),
            fake_result((0.3, 0.5, [cat("Victory", 0.4), cat("Open_Palm", 0.3)], "Left")),
            fake_result((0.3, 0.5, [], "Left")))]
        s = hp.summarize(frames)
        self.assertEqual((s["frames_total"], s["frames_no_hand"], s["frames_one_hand"], s["frames_two_hands"]), (4, 1, 2, 1))
        self.assertEqual(s["hand_detections"], 4)
        self.assertEqual(s["detections_without_top_category"], 1)
        self.assertEqual(s["top_category_counts"], {"Closed_Fist": 1, "Victory": 2})
        self.assertEqual(hp.dumps_strict(s), hp.dumps_strict(hp.summarize(frames)))

    def test_nan_score_sanitized_and_json_strict(self):
        with self.assertRaises(ValueError):
            hp.dumps_strict({"x": float("nan")})
        rec = hp.parse_result(fake_result((0.5, 0.5, [cat("Victory", float("nan")), cat("None", 0.2)], "Left")),
                              W, H, self.anchors(), 100.0, None)
        h = rec["hands"][0]
        self.assertIsNone(h["gesture_categories"][0]["score"])
        self.assertEqual(h["top_category"]["category_name"], "None")
        json.loads(hp.dumps_strict(rec), parse_constant=hp._reject_constant)

    def test_inconsistent_result_lengths_raise(self):
        res = fake_result((0.5, 0.5, [cat("Victory", 0.8)], "Left"))
        res.handedness = []
        with self.assertRaises(hp.ProbeError):
            hp.parse_result(res, W, H, self.anchors(), 100.0, None)


class PreflightAndOutputTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_row_binding_mismatch(self):
        _, cfg, paths = make_fixture(self.tmp)
        npz = hp.load_npz(paths["npz"])
        bad = [dict(s) for s in cfg["inputs"][0]["samples"]]
        bad[1]["timestamp_us"] += 1
        with self.assertRaises(hp.ProbeError):
            hp.bind_rows(npz, bad, "c")
        bad = [dict(s) for s in cfg["inputs"][0]["samples"]]
        bad[2]["frame_id"] = 6  # row 3 holds frame 3
        with self.assertRaises(hp.ProbeError):
            hp.bind_rows(npz, bad, "c")

    def test_row_mismatch_run_fails_without_result(self):
        def mutate(cfg):
            cfg["inputs"][0]["samples"][1]["frame_id"] = 99
        cfg_path, _, _ = make_fixture(self.tmp, mutate)
        rec = Recorder([])
        out = os.path.join(self.tmp, "out")
        self.assertEqual(hp.run(cfg_path, out, factory(rec), fake_frames, expected_clips=1), 1)
        self.assertTrue(os.path.exists(os.path.join(out, "FAILED.json")))
        self.assertFalse(os.path.exists(os.path.join(out, "result.json")))
        self.assertEqual(rec.calls, [])

    def test_hash_mismatch_fails_before_inference(self):
        cfg_path, _, _ = make_fixture(self.tmp, lambda c: c["inputs"][0].__setitem__("sha256", "0" * 64))
        rec = Recorder([])
        out = os.path.join(self.tmp, "out")
        self.assertEqual(hp.run(cfg_path, out, factory(rec), fake_frames, expected_clips=1), 1)
        self.assertEqual(strict_load(os.path.join(out, "FAILED.json"))["technical_status"], "failed")
        self.assertFalse(os.path.exists(os.path.join(out, "result.json")))
        self.assertEqual(rec.calls, [])

    def test_existing_out_dir_refused_and_untouched(self):
        cfg_path, _, _ = make_fixture(self.tmp)
        out = os.path.join(self.tmp, "out")
        os.mkdir(out)
        marker = os.path.join(out, "keep.txt")
        with open(marker, "w") as fh:
            fh.write("keep")
        self.assertEqual(hp.run(cfg_path, out, factory(Recorder([])), fake_frames, expected_clips=1), 2)
        self.assertEqual(os.listdir(out), ["keep.txt"])

    def test_out_dir_equal_to_source_refused(self):
        cfg_path, _, paths = make_fixture(self.tmp)
        before = sha(paths["video"])
        self.assertEqual(hp.run(cfg_path, paths["video"], factory(Recorder([])), fake_frames, expected_clips=1), 2)
        self.assertEqual(sha(paths["video"]), before)

    def test_safe_write_never_overwrites_or_escapes(self):
        _, _, paths = make_fixture(self.tmp)
        protected = {hp._norm(paths["video"])}
        out = hp.prepare_out_dir(os.path.join(self.tmp, "out"), protected)
        hp.safe_write(out, "a.txt", b"x", protected)
        with self.assertRaises((hp.ProbeError, FileExistsError)):
            hp.safe_write(out, "a.txt", b"y", protected)
        with self.assertRaises(hp.ProbeError):
            hp.safe_write(out, os.path.join("..", "root", "clip.mp4"), b"z", protected)
        with open(os.path.join(out, "a.txt"), "rb") as fh:
            self.assertEqual(fh.read(), b"x")

    def test_policy_and_clip_id_validation(self):
        _, cfg, _ = make_fixture(self.tmp)
        self.assertEqual(len(hp.validate_config(cfg, 1)), 5)  # model, video, npz, sidecar, app
        with self.assertRaises(hp.ProbeError):
            hp.validate_config(cfg, 18)
        for mutate in (lambda c: c["policy"].__setitem__("num_hands", 1),
                       lambda c: c["inputs"][0].__setitem__("clip_id", "../evil"),
                       lambda c: c["inputs"][0].__setitem__("video_path", "rtsp://cam/0")):
            bad = json.loads(json.dumps(cfg))
            mutate(bad)
            with self.assertRaises(hp.ProbeError):
                hp.validate_config(bad, 1)

    def test_pts_tolerance_and_dimensions(self):
        s = [{"row": 0, "frame_id": 3, "timestamp_us": 99999}]
        img = np.zeros((H, W, 3), np.uint8)
        hp.verify_frames(s, {3: (img, 99999 + 2)}, W, H, 2)
        with self.assertRaises(hp.ProbeError):
            hp.verify_frames(s, {3: (img, 99999 + 3)}, W, H, 2)
        with self.assertRaises(hp.ProbeError):
            hp.verify_frames(s, {3: (np.zeros((H, W + 1, 3), np.uint8), 99999)}, W, H, 2)


class RunTests(unittest.TestCase):
    def test_model_call_receives_only_a_copied_image(self):
        xy, us = pose()
        npz = {"xy_px": xy, "usable": us}
        samples = [{"row": r, "frame_id": r, "timestamp_us": r * 33333} for r in ROWS]
        decoded = fake_frames(None, ROWS)
        originals = {k: v[0].copy() for k, v in decoded.items()}

        def mutating(*args, **kwargs):
            calls.append((args, kwargs))
            args[0][:] = 255
            return fake_result()
        calls = []
        clip = {"clip_id": "c", "word_id": "SECRET_WORD", "width": W, "height": H}
        hp.infer_clip(clip, npz, samples, decoded, mutating)
        self.assertEqual(len(calls), 5)  # once per selected frame, no reruns
        for args, kwargs in calls:
            self.assertEqual(kwargs, {})
            self.assertEqual(len(args), 1)
            self.assertIsInstance(args[0], np.ndarray)
            self.assertEqual((args[0].dtype, args[0].shape), (np.uint8, (H, W, 3)))
        for k, v in decoded.items():
            np.testing.assert_array_equal(v[0], originals[k])

    def test_end_to_end_synthetic(self):
        had_mp = "mediapipe" in sys.modules
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path, _, paths = make_fixture(tmp)
            hashes = {k: sha(p) for k, p in paths.items()}
            rec = Recorder([
                fake_result(),
                fake_result((0.5, 0.9, [cat("Victory", 0.8), cat("None", 0.2)], "Left")),
                fake_result((0.3, 0.5, [cat("Closed_Fist", 0.7)], "Left"), (0.7, 0.5, [cat("Victory", 0.6)], "Right")),
                fake_result((0.5, 0.5, [cat("None", 0.9)], "Right")),
                fake_result()])
            out = os.path.join(tmp, "out")
            self.assertEqual(hp.run(cfg_path, out, factory(rec), fake_frames, expected_clips=1), 0)
            r = strict_load(os.path.join(out, "result.json"))
            self.assertEqual(r["technical_status"], "complete")
            t = r["totals"]
            self.assertEqual((t["frames_total"], t["frames_no_hand"], t["hand_detections"]), (5, 2, 4))
            self.assertEqual(t["top_category_counts"], {"Closed_Fist": 1, "None": 1, "Victory": 2})
            fr = r["clips"][0]["frames"][1]
            # wrist (160,216) - shoulder mid (150,200) over width 100
            np.testing.assert_allclose(fr["hands"][0]["body_relative"]["shoulder_position"], [0.1, 0.16])
            with open(os.path.join(out, r["clips"][0]["review_sheet"]), "rb") as fh:
                self.assertEqual(fh.read(8), b"\x89PNG\r\n\x1a\n")
            with open(os.path.join(out, "RESULT.md"), encoding="utf-8") as fh:
                self.assertIn("sheets/review_NIKL_TEST_0001.png", fh.read())
            self.assertFalse(any(n.endswith(".pending.json") for n in os.listdir(out)))
            self.assertEqual({k: sha(p) for k, p in paths.items()}, hashes)
            self.assertEqual(len(rec.calls), 5)
            self.assertTrue(all(len(a) == 1 and not kw for a, kw in rec.calls))
        if not had_mp:
            self.assertNotIn("mediapipe", sys.modules)


if __name__ == "__main__":
    unittest.main()
