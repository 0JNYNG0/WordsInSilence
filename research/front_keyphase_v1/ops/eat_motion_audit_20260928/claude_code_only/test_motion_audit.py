"""Synthetic unit tests for motion_audit.py (no real data, no video, no subprocess).

Run: python -m unittest -v test_motion_audit
"""
import json
import os
import tempfile
import unittest
from fractions import Fraction

import numpy as np

import motion_audit as ma

L, R = 33, 54


def make_clip(T, ts=None):
    """Nose (100,100); shoulders (50,200),(150,200) -> scale 100 px.
    All L hand points at (60,150) -> rel (-0.4, 0.5); R at (140,150) -> rel (0.4, 0.5)."""
    xy = np.zeros((T, 75, 2), dtype=np.float64)
    usable = np.ones((T, 75), dtype=bool)
    xy[:, 0] = (100.0, 100.0)
    xy[:, 11] = (50.0, 200.0)
    xy[:, 12] = (150.0, 200.0)
    xy[:, L:L + 21] = (60.0, 150.0)
    xy[:, R:R + 21] = (140.0, 150.0)
    ts = np.arange(T, dtype=np.int64) * 33333 if ts is None else np.asarray(ts, dtype=np.int64)
    return xy, usable, ts


def set_hand(xy, base, row, x, y):
    xy[row, base:base + 21] = (x, y)


class MeasureTests(unittest.TestCase):
    def assertJsonSafe(self, res):
        json.dumps(res, allow_nan=False)

    def test_constant_xy_path_zero(self):
        xy, u, ts = make_clip(5)
        res = ma.measure(xy, u, ts, [0, 4])
        self.assertJsonSafe(res)
        self.assertEqual(res["scale_px"], 100.0)
        for side in "LR":
            s = res["summary"][side]
            self.assertEqual(s["valid_frames"], 5)
            self.assertEqual(s["adjacent_valid_pairs"], 4)
            self.assertEqual(s["path_total"], 0.0)
            self.assertEqual(s["max_speed"], 0.0)
            self.assertEqual(s["mean_path_per_covered_second"], 0.0)
            self.assertEqual(s["largest_invalid_run_frames"], 0)
        self.assertAlmostEqual(res["series"]["L"]["rel_x"][2], -0.4)
        self.assertAlmostEqual(res["series"]["L"]["rel_y"][2], 0.5)

    def test_nonuniform_timestamps_exact_speed(self):
        xy, u, ts = make_clip(3, ts=[0, 100000, 400000])
        for row, x in enumerate((60.0, 70.0, 80.0)):  # +0.1 shoulder widths per step
            set_hand(xy, L, row, x, 150.0)
        res = ma.measure(xy, u, ts, [0, 2])
        s = res["summary"]["L"]
        self.assertEqual(res["series"]["time_s"], [0.0, 0.1, 0.4])
        self.assertEqual(res["series"]["delta_s"][0], None)
        self.assertAlmostEqual(res["series"]["delta_s"][2], 0.3)
        self.assertAlmostEqual(s["covered_seconds"], 0.4)
        self.assertAlmostEqual(s["path_total"], 0.2)
        self.assertAlmostEqual(s["max_speed"], 1.0)          # 0.1 / 0.1 s
        self.assertAlmostEqual(s["mean_path_per_covered_second"], 0.5)
        self.assertAlmostEqual(s["x_range"][0], -0.4)
        self.assertAlmostEqual(s["x_range"][1], -0.2)

    def test_missing_middle_frame_no_bridge(self):
        xy, u, ts = make_clip(4, ts=[0, 100000, 200000, 300000])
        for row, x in enumerate((60.0, 70.0, 80.0, 90.0)):
            set_hand(xy, L, row, x, 150.0)
        u[2, L + 0] = False  # wrist missing on row 2
        res = ma.measure(xy, u, ts, [0, 3])
        s = res["summary"]["L"]
        self.assertIsNone(res["series"]["L"]["rel_x"][2])
        self.assertEqual(res["series"]["L"]["rel_valid"], [True, True, False, True])
        self.assertEqual(s["valid_frames"], 3)
        self.assertEqual(s["adjacent_valid_pairs"], 1)
        self.assertAlmostEqual(s["path_total"], 0.1)       # 1->3 bridge would give 0.3
        self.assertAlmostEqual(s["covered_seconds"], 0.1)
        self.assertEqual(s["largest_invalid_run_frames"], 1)
        # delta_s is reported for every row, including invalid ones
        self.assertEqual(len(res["series"]["delta_s"]), 4)
        self.assertAlmostEqual(res["series"]["delta_s"][2], 0.1)
        self.assertEqual(res["summary"]["R"]["adjacent_valid_pairs"], 3)

    def test_one_invalid_mcp_invalidates_center(self):
        xy, u, ts = make_clip(3)
        u[1, L + 5] = False                 # unusable MCP
        xy[1, R + 9, 0] = np.nan            # usable but non-finite MCP
        res = ma.measure(xy, u, ts, [0, 2])
        for side in "LR":
            ser = res["series"][side]
            self.assertEqual(ser["palm_all5_valid"], [True, False, True])
            self.assertIsNone(ser["palm_px_x"][1])
            self.assertIsNone(ser["rel_x"][1])
            self.assertEqual(res["summary"][side]["adjacent_valid_pairs"], 0)
            self.assertIsNone(res["summary"][side]["path_total"])
        self.assertJsonSafe(res)

    def test_palm_center_is_mean_of_exact_five(self):
        xy, u, ts = make_clip(1)
        xy[0, L:L + 21] = (1000.0, 1000.0)          # non-palm points far away
        for k, x in zip(ma.PALM_LOCAL, (10.0, 20.0, 30.0, 40.0, 50.0)):
            xy[0, L + k] = (x, 2 * x)
        u[0, L + 8] = False                          # non-palm point irrelevant
        res = ma.measure(xy, u, ts, [0, 0])
        self.assertAlmostEqual(res["series"]["L"]["palm_px_x"][0], 30.0)
        self.assertAlmostEqual(res["series"]["L"]["palm_px_y"][0], 60.0)

    def test_unavailable_shoulder_scale(self):
        for mode in ("unusable", "zero", "nan"):
            xy, u, ts = make_clip(4)
            if mode == "unusable":
                u[:, 12] = False
            elif mode == "zero":
                xy[:, 12] = xy[:, 11]
            else:
                xy[:, 11, 1] = np.nan
            res = ma.measure(xy, u, ts, [0, 3])
            self.assertIsNone(res["scale_px"], mode)
            for side in "LR":
                self.assertEqual(res["series"][side]["rel_x"], [None] * 4)
                self.assertEqual(res["series"][side]["palm_all5_valid"], [True] * 4)
                s = res["summary"][side]
                self.assertFalse(s["scale_available"])
                for k in ("valid_frames", "adjacent_valid_pairs", "x_range", "y_range",
                          "covered_seconds", "path_total", "mean_path_per_covered_second",
                          "max_speed", "largest_invalid_run_frames"):
                    self.assertIsNone(s[k], (mode, k))
            self.assertJsonSafe(res)

    def test_scale_is_median_over_full_clip(self):
        xy, u, ts = make_clip(3)
        xy[0, 12] = (250.0, 200.0)  # distance 200 on row 0, 100 elsewhere
        u[2, 11] = False            # row 2 excluded
        res = ma.measure(xy, u, ts, [1, 1])  # span must not restrict scale
        self.assertEqual(res["scale_px"], 150.0)
        self.assertEqual(res["scale_frames"], 2)

    def test_inclusive_span_clipping(self):
        xy, u, ts = make_clip(6)
        res = ma.measure(xy, u, ts, [-3, 10])
        self.assertEqual(res["span_used"], [0, 5])
        self.assertEqual(res["summary"]["L"]["span_frames"], 6)
        self.assertEqual(res["summary"]["L"]["adjacent_valid_pairs"], 5)
        res = ma.measure(xy, u, ts, [2, 3])
        self.assertEqual(res["span_used"], [2, 3])
        self.assertEqual(res["summary"]["L"]["span_frames"], 2)
        self.assertEqual(res["summary"]["L"]["adjacent_valid_pairs"], 1)
        self.assertEqual(len(res["series"]["time_s"]), 6)  # series stay full length
        res = ma.measure(xy, u, ts, [7, 9])
        self.assertIsNone(res["span_used"])
        self.assertEqual(res["summary"]["L"]["span_frames"], 0)
        self.assertIsNone(res["summary"]["L"]["path_total"])
        with self.assertRaises(ValueError):
            ma.measure(xy, u, ts, [3, 2])
        with self.assertRaises(ValueError):
            ma.measure(xy, u, ts, [0.0, 2])

    def test_never_mutates_inputs(self):
        xy, u, ts = make_clip(5)
        xy[2, L + 5, 0] = np.nan
        u[3, 0] = False
        copies = (xy.copy(), u.copy(), ts.copy())
        for a in (xy, u, ts):
            a.flags.writeable = False
        ma.measure(xy, u, ts, [1, 3])
        np.testing.assert_array_equal(xy, copies[0])
        np.testing.assert_array_equal(u, copies[1])
        np.testing.assert_array_equal(ts, copies[2])

    def test_empty_and_single_frame(self):
        xy, u, ts = make_clip(0)
        res = ma.measure(xy, u, ts, [0, 5])
        self.assertJsonSafe(res)
        self.assertEqual(res["T"], 0)
        self.assertIsNone(res["span_used"])
        self.assertIsNone(res["scale_px"])
        self.assertEqual(res["series"]["time_s"], [])
        xy, u, ts = make_clip(1)
        res = ma.measure(xy, u, ts, [0, 0])
        s = res["summary"]["L"]
        self.assertEqual((s["valid_frames"], s["adjacent_valid_pairs"]), (1, 0))
        self.assertEqual(s["covered_seconds"], 0.0)
        self.assertIsNone(s["path_total"])
        self.assertIsNone(s["max_speed"])
        self.assertEqual(res["series"]["delta_s"], [None])

    def test_rejects_non_increasing_timestamps(self):
        xy, u, ts = make_clip(3, ts=[0, 100, 100])
        with self.assertRaises(ValueError):
            ma.measure(xy, u, ts, [0, 2])


class HelperTests(unittest.TestCase):
    def test_to_jsonable(self):
        obj = {"a": np.float32(np.nan), "b": np.int64(3), "c": np.array([1.0, np.inf]),
               "d": np.bool_(True), 5: float("-inf")}
        out = ma.to_jsonable(obj)
        self.assertEqual(out, {"a": None, "b": 3, "c": [1.0, None], "d": True, "5": None})
        json.dumps(out, allow_nan=False)

    def test_dense_rows(self):
        self.assertEqual(ma.dense_rows(20, [2, 4]), list(range(0, 10)))
        self.assertEqual(ma.dense_rows(20, [15, 18]), list(range(10, 20)))
        self.assertEqual(ma.dense_rows(0, [0, 3]), [])

    def test_pts_to_us(self):
        self.assertEqual(ma.pts_to_us(1001, Fraction(1, 30000)), 33367)
        self.assertEqual(ma.pts_to_us(0, Fraction(1, 90000)), 0)

    def test_tile_top_is_unmarked_crop(self):
        from PIL import Image
        img = Image.new("RGB", (1000, 800), (10, 20, 30))
        img.putpixel((500, 400), (200, 0, 0))
        xy, u, _ = make_clip(1)
        xy[0, L:L + 21] = (500.0, 400.0)
        tile, box = ma.make_tile(img, xy[0], u[0], ["row 0"], None)
        self.assertEqual(box, (150, 40, 850, 760))
        th = round(720 * 360 / 700)
        self.assertEqual(tile.size, (360, 2 * th + ma.LABEL_H))
        expect = img.crop(box).resize((360, th), Image.BILINEAR)
        self.assertEqual(list(tile.crop((0, 0, 360, th)).getdata()), list(expect.getdata()))
        self.assertNotEqual(list(tile.crop((0, th, 360, 2 * th)).getdata()),
                            list(expect.getdata()))

    def test_refuses_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                ma.main(["--config", os.path.join(d, "none.json"), "--out", d])


if __name__ == "__main__":
    unittest.main()
