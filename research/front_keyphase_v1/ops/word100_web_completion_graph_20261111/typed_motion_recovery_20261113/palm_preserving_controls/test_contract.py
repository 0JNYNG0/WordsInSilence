"""Toy-coordinate contracts only. No donor files, models, inference, or camera.

Run after review: python -B <this-file>. Uses the existing pure adapter only.
"""
import sys
sys.dont_write_bytecode = True
import importlib.util
from pathlib import Path
import unittest
import numpy as np

from palm_control import (CHAINS, DISTAL, PROTECTED, make_control,
                          rebuild_channel_pair, replace_distal_four_fingers)


def hand(closed=False):
    h = np.zeros((21, 2), np.float64)
    h[1:5] = [[-14, -5], [-22, -12], [-27, -20], [-30, -28]]
    for m, x in zip((5, 9, 13, 17), (-18, -6, 7, 19)):
        h[m] = [x, -30 + abs(x) * .2]
        vectors = [[1, -9], [2, 6], [-1, 7]] if closed else [[0, -14], [0, -10], [0, -8]]
        h[m + 1:m + 4] = h[m] + np.cumsum(vectors, axis=0)
    return h + [150, 160]


def fixture(dtype=np.float32):
    xy = np.full((4, 133, 2), [200, 200], dtype=dtype)
    xy[:, 5] = [240, 150]
    xy[:, 6] = [140, 150]
    for i in range(4):
        xy[i, 91:112] = hand(False) + [150 + 2 * i, i]
        xy[i, 112:133] = hand(False) + [3 * i, 2 * i]
    score = np.linspace(.1, .9, 4 * 133, dtype=np.float32).reshape(4, 133)
    ts = np.array([1001000, 1034000, 1071000, 1100000], np.int64)
    fid = np.array([7, 9, 11, 15], np.int64)
    return xy, score, ts, fid


def control(pack, side="R", **kwargs):
    options = dict(side=side, donor_side="R", frame_indices=[1, 2], width=640, height=480)
    options.update(kwargs)
    return make_control(*pack, hand(True), **options)


class Contracts(unittest.TestCase):
    def test_exact_protected_and_input_immutability(self):
        donor, target = hand(True), hand(False).astype(np.float32)
        before = donor.tobytes(), target.tobytes()
        donor.flags.writeable = target.flags.writeable = False
        out = replace_distal_four_fingers(donor, target, donor_side="R", target_side="R")
        self.assertEqual(out[list(PROTECTED)].tobytes(), target[list(PROTECTED)].tobytes())
        self.assertEqual(before, (donor.tobytes(), target.tobytes()))
        self.assertFalse(np.array_equal(out[list(DISTAL)], target[list(DISTAL)]))
        self.assertFalse(np.shares_memory(out, target))

    def test_preserves_projected_bone_lengths_and_donor_angles(self):
        donor, target = hand(True), hand(False)
        out = replace_distal_four_fingers(donor, target, donor_side="R", target_side="R")
        for chain in CHAINS:
            db, tb, ob = [np.diff(h[list(chain)], axis=0) for h in (donor, target, out)]
            np.testing.assert_allclose(np.linalg.norm(ob, axis=1), np.linalg.norm(tb, axis=1), atol=1e-12)
            du = db / np.linalg.norm(db, axis=1)[:, None]
            ou = ob / np.linalg.norm(ob, axis=1)[:, None]
            np.testing.assert_allclose(du @ du.T, ou @ ou.T, atol=1e-12)

    def test_same_and_opposite_hand_mirror_equivariance(self):
        donor, target = hand(True), hand(False)
        def mirror(h):
            h = h.copy(); h[:, 0] = 640 - h[:, 0]; return h
        right = replace_distal_four_fingers(donor, target, donor_side="R", target_side="R")
        both_left = replace_distal_four_fingers(mirror(donor), mirror(target), donor_side="L", target_side="L")
        left_from_right = replace_distal_four_fingers(donor, mirror(target), donor_side="R", target_side="L")
        np.testing.assert_allclose(both_left, mirror(right), atol=1e-12)
        np.testing.assert_allclose(left_from_right, mirror(right), atol=1e-12)

    def test_common_rotation_translation_scale_equivariance(self):
        donor, target = hand(True), hand(False)
        angle = .37
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        transform = lambda h: h @ rotation * 1.4 + [22, -12]
        out = replace_distal_four_fingers(donor, target, donor_side="R", target_side="R")
        moved = replace_distal_four_fingers(transform(donor), transform(target), donor_side="R", target_side="R")
        np.testing.assert_allclose(moved, transform(out), atol=1e-12)

    def test_identity_donor_is_identity_with_rounding_tolerance(self):
        target = hand(False)
        out = replace_distal_four_fingers(target, target, donor_side="R", target_side="R")
        np.testing.assert_allclose(out, target, atol=1e-12)

    def test_full_pack_preserves_metadata_other_frames_and_other_hand(self):
        for side in ("L", "R"):
            pack = fixture()
            before = [x.tobytes() for x in pack]
            out = control(pack, side)
            self.assertEqual(before, [x.tobytes() for x in pack])
            self.assertEqual(out["applied_frame_count"], 2)
            self.assertEqual(int(out["synthetic_point"].sum()), 24)
            self.assertEqual(pack[0][~out["synthetic_point"]].tobytes(), out["xy_px"][~out["synthetic_point"]].tobytes())
            for key, value in zip(("score", "timestamp_us", "frame_id"), pack[1:]):
                self.assertEqual(out[key].tobytes(), value.tobytes())

    def test_unavailable_and_nonfinite_frames_unchanged(self):
        pack = fixture()
        pack[0][1, 112 + 9] = np.nan
        available = np.ones((4, 133), bool); available[2, 112 + 8] = False
        out = control(pack, source_available=available)
        self.assertEqual(out["applied_frame_count"], 0)
        self.assertEqual(pack[0].tobytes(), out["xy_px"].tobytes())

    def test_missing_thumb_is_not_fabricated(self):
        pack = fixture()
        pack[0][:, 113:117] = np.nan
        out = control(pack)
        self.assertEqual(out["applied_frame_count"], 2)
        self.assertEqual(pack[0][:, 113:117].tobytes(), out["xy_px"][:, 113:117].tobytes())

    def test_degenerate_palm_and_zero_bone_rejected(self):
        for kind in ("palm", "finger"):
            target = hand(False)
            if kind == "palm":
                target[[5, 9, 13, 17]] = target[0] + [0, -10]
            else:
                target[6] = target[5]
            with self.assertRaises(ValueError):
                replace_distal_four_fingers(hand(True), target, donor_side="R", target_side="R")

    def test_out_of_bounds_generated_point_leaves_entire_frame_unchanged(self):
        pack = fixture()
        # All source fingers extend upward inside the image, whereas the donor
        # closes back beyond a palm only 4px above the bottom image boundary.
        h = hand(False)
        for chain in CHAINS:
            h[list(chain[1:])] = h[chain[0]] + np.cumsum([[0, -14], [0, -30], [0, -30]], axis=0)
        h[:, 1] += 476 - h[:, 1].max()
        pack[0][:, 112:133] = h
        out = control(pack)
        self.assertEqual(out["applied_frame_count"], 0)
        self.assertEqual(out["xy_px"].tobytes(), pack[0].tobytes())
        self.assertTrue(all("image bounds" in r["reason"] for r in out["frame_report"]))

    def test_bad_index_time_side_rejected(self):
        for indices in ([1, 1], [-1], [4], [1.0], [True]):
            with self.assertRaises(ValueError):
                control(fixture(), frame_indices=indices)
        with self.assertRaises(ValueError):
            control(fixture(), side="right")
        pack = fixture(); pack[2][2] = pack[2][1]
        with self.assertRaises(ValueError):
            control(pack)

    def test_canonical_rebuild_exact_body_palm_thumb_masks_and_time(self):
        ops = Path(__file__).resolve().parents[3]
        path = ops / "word100_rtmw_shared_input_20261108" / "adapter.py"
        spec = importlib.util.spec_from_file_location("palm_contract_adapter", path)
        adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
        for side in ("L", "R"):
            pack = fixture()
            out = control(pack, side)
            pair = rebuild_channel_pair(pack[0], out, convert=adapter.convert)
            a, b = pair["baseline"]["tcn280"], pair["control"]["tcn280"]
            self.assertTrue(pair["protected_channels_identical"])
            self.assertEqual(a[:, 186:].tobytes(), b[:, 186:].tobytes())
            self.assertFalse(np.array_equal(a, b))
            # A five-point palm centroid is exactly unchanged in common coords.
            base = 9 if side == "L" else 30
            ids = base + np.array([0, 5, 9, 13, 17])
            p, q = a[:, :102].reshape(-1, 51, 2), b[:, :102].reshape(-1, 51, 2)
            self.assertEqual(p[:, ids].mean(1).tobytes(), q[:, ids].mean(1).tobytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
