"""Stage 0 checks on synthetic arrays and the policy file only. No model, no source npz, no camera.

Run: py -3.13 ops/framerate_pair_20260927/tests_stage0.py
"""
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(H), str(L.parent / "sign_model_pilot_v1" / "src")]
import framerate_impl as F                                    # noqa: E402
from pilot.data import PilotDataError, build_channels         # noqa: E402

L_SH, R_SH = 5, 2                                            # BODY_25 slots of the shoulders


def clip(n, shoulders=None):
    """Synthetic native clip: n frames, 67 points, 30 Hz nominal clock, every point present."""
    xy = np.zeros((n, 67, 2), dtype=np.float32)
    for t in range(n):
        xy[t, :, 0] = 100 + np.arange(67) + t
        xy[t, :, 1] = 200 + 2 * np.arange(67)
    xy[:, L_SH] = [300, 100]
    xy[:, R_SH] = [100, 100]
    present = np.ones((n, 67), dtype=bool)
    if shoulders is not None:
        present[:, [L_SH, R_SH]] = shoulders[:, None]
    ts = np.round(np.arange(n) * F.NOMINAL_STEP_US).astype(np.int64)
    return xy, present, ts


class Decimate(unittest.TestCase):
    def setUp(self):
        self.cam = patch("cv2.VideoCapture", side_effect=AssertionError("REAL CAMERA FORBIDDEN"),
                         create=True)
        try:
            self.cam.start()
        except ModuleNotFoundError:
            self.cam = None

    def tearDown(self):
        if self.cam is not None:
            self.cam.stop()

    def test_stride1_is_an_exact_copy(self):
        xy, pr, ts = clip(5)
        a, b, c = F.decimate(xy, pr, ts, stride=1)
        for x, y in ((a, xy), (b, pr), (c, ts)):
            self.assertTrue(np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))
            self.assertEqual(x.dtype, y.dtype)
            self.assertFalse(np.shares_memory(x, y))

    def test_even_and_odd_end_frames(self):
        for n, kept in ((6, [0, 2, 4]), (7, [0, 2, 4, 6])):
            xy, pr, ts = clip(n)
            _, _, t = F.decimate(xy, pr, ts)
            self.assertEqual(t.tolist(), ts[kept].tolist(), n)

    def test_frame0_kept_and_no_interpolation(self):
        xy, pr, ts = clip(5)
        a, _, _ = F.decimate(xy, pr, ts)
        self.assertTrue(np.array_equal(a, xy[[0, 2, 4]]))

    def test_mask_and_nan_carried_verbatim(self):
        xy, pr, ts = clip(6)
        xy[2, 30] = np.nan
        pr[2, 30] = False
        pr[3, 40] = False                                    # dropped frame: its hole goes with it
        a, b, _ = F.decimate(xy, pr, ts)
        self.assertTrue(np.isnan(a[1, 30]).all())
        self.assertFalse(b[1, 30])
        self.assertTrue(np.array_equal(b, pr[[0, 2, 4]]))
        self.assertTrue(b[:, 40].all())

    def test_inputs_unchanged_and_not_aliased(self):
        xy, pr, ts = clip(7)
        before = (xy.copy(), pr.copy(), ts.copy())
        a, b, c = F.decimate(xy, pr, ts)
        a[:] = -1
        b[:] = False
        c[:] = 0
        for x, y in zip((xy, pr, ts), before):
            self.assertTrue(np.array_equal(x, y))

    def test_one_frame_and_empty(self):
        xy, pr, ts = clip(1)
        a, b, c = F.decimate(xy, pr, ts)
        self.assertEqual(len(c), 1)
        out = build_channels(a, b, c)
        self.assertEqual(out["frames"], 1)
        self.assertEqual(float(out["x"][0, -1]), 0.0)          # dt of the first frame is 0
        with self.assertRaises(ValueError):
            F.decimate(xy[:0], pr[:0], ts[:0])

    def test_bad_arguments(self):
        xy, pr, ts = clip(4)
        for kw in ({"stride": 0}, {"stride": 2.0}, {"offset": 2}, {"offset": -1}):
            with self.assertRaises(ValueError):
                F.decimate(xy, pr, ts, **kw)
        with self.assertRaises(ValueError):
            F.decimate(xy[:3], pr, ts)

    def test_shoulder_scale_can_vanish_after_decimation(self):
        # shoulders visible only on odd frames: the original has a scale, the decimated clip not
        xy, pr, ts = clip(6, shoulders=np.array([0, 1, 0, 1, 0, 1], dtype=bool))
        build_channels(xy, pr, ts)
        a, b, c = F.decimate(xy, pr, ts)
        with self.assertRaises(PilotDataError):
            build_channels(a, b, c)

    def test_scales_are_re_estimated_from_kept_frames(self):
        xy, pr, ts = clip(4)
        xy[1, L_SH] = [500, 100]                             # a wider shoulder on a dropped frame
        xy[3, L_SH] = [500, 100]
        full = build_channels(xy, pr, ts)["body_scale"]
        kept = build_channels(*F.decimate(xy, pr, ts))["body_scale"]
        self.assertNotEqual(full, kept)                      # stated in the policy, not hidden

    def test_nominal_clock_doubles(self):
        xy, pr, ts = clip(9)
        _, _, c = F.decimate(xy, pr, ts)
        self.assertTrue(F.kept_intervals_are_sums(ts, c))
        self.assertTrue(F.nominal_dt_doubles(ts, c))
        out = build_channels(*F.decimate(xy, pr, ts))["x"][1:, -1]
        self.assertTrue(np.allclose(out, 2 / 30, atol=2e-6))
        irregular = ts.copy()
        irregular[4:] += 10_000
        self.assertFalse(F.nominal_dt_doubles(irregular, irregular[::2]))

    def test_recording_id(self):
        self.assertEqual(F.recording_id("NIA_SL_SEN0005_REAL11_F_a00_050"), "NIA_SL_SEN0005_REAL11_F")
        self.assertEqual(F.recording_id("NO_SUFFIX"), "NO_SUFFIX")

    def test_web_gate_information_only(self):
        _, pr, _ = clip(3)
        self.assertTrue(F.web_gate_information(pr, 4, 4)["under_web_minimums"])
        _, pr, _ = clip(6)
        pr[:, 25:67] = False
        pr[:2, 30] = True
        info = F.web_gate_information(pr, 4, 4)
        self.assertEqual(info["observed_hand_frames"], 2)
        self.assertTrue(info["under_web_minimums"])


class Loader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name) / "allowed"
        self.root.mkdir()
        xy, pr, ts = clip(4)
        self.path = self.root / "a.npz"
        np.savez(self.path, xy_native=xy, coordinate_present=pr, timestamp_nominal_us=ts)
        self.row = {"id": "S_REAL11_F_a00_001", "role": "validation", "signer_id": 11,
                    "path": str(self.path), "sha256": F.sha256_file(self.path)}

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, row):
        return F.load_native(row, ["validation"], [11, 12, 13], self.root)

    def test_valid_row(self):
        out = self.load(self.row)
        self.assertEqual(out["xy"].shape, (4, 67, 2))
        self.assertEqual(out["sha256"], self.row["sha256"])

    def test_refusals(self):
        outside = pathlib.Path(self.tmp.name) / "b.npz"
        np.savez(outside, x=np.zeros(1))
        for change in ({"role": "train"}, {"signer_id": 14}, {"signer_id": 3},
                       {"sha256": "0" * 64}, {"path": str(outside), "sha256": F.sha256_file(outside)}):
            with self.assertRaises(F.LoaderRefused, msg=str(change)):
                self.load(dict(self.row, **change))

    def test_missing_field_refused(self):
        bad = self.root / "c.npz"
        np.savez(bad, xy_native=np.zeros((1, 67, 2)))
        with self.assertRaises(F.LoaderRefused):
            self.load(dict(self.row, path=str(bad), sha256=F.sha256_file(bad)))


class Policy(unittest.TestCase):
    def test_policy_is_the_single_fixed_condition(self):
        p = json.loads((H / "comparison_policy.json").read_text(encoding="utf-8"))
        self.assertEqual((p["transform"]["stride"], p["transform"]["offset"]), (2, 0))
        self.assertEqual(p["material"]["allowed_roles"], ["validation"])
        self.assertEqual(p["material"]["allowed_signers"], [11, 12, 13])
        self.assertEqual(p["stage1_decision_rule"]["proceed_to_stage1_design_if"], "net_loss >= 7")
        self.assertTrue(any("KSL77" in s for s in p["material"]["never_read"]))
        self.assertEqual(p["current_baseline"]["sha256"][:8], "83212868")
        self.assertEqual(p["execution"]["ksl77"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
