"""Contract checks only; SYNTHETIC test labels never become public ground truth.

Run: python -m unittest test_observation_input -v
The saved-public checks read only prior public provenance/native arrays. There
is no video decoding, backbone inference, optimizer, training, or download.
"""
import copy
import json
import unittest

import numpy as np

from observation_input import (
    BODY_IDS, BODY_REQUIRED, HAND_OFFSETS, append_saved_context,
    build_observation_input, join_public_saved_context, load_public_saved_input, pad_observation_batch,
    partial_label_cross_entropy,
)


def synthetic_fixture(n=5):
    """Explicit synthetic sensor fixture, not a KSL annotation."""
    times = np.array([0, 33333, 90000, 125000, 250000], dtype=np.int64)[:n]
    xy = np.arange(n * 75 * 2, dtype=np.float32).reshape(n, 75, 2) / 10
    observed = np.ones((n, 75), dtype=bool)
    observed[:min(2, n), 33:54] = False
    if n > 1:
        observed[1, 59] = False
    if n > 3:
        observed[3, 40] = False  # Left index geometry missing; wrist stays valid.
        observed[3, 0] = False  # Body14 copy but remaining native body points survive.
    native = {"frame_id": np.arange(n, dtype=np.int64), "timestamp_us": times,
              "xy_px": xy, "observed": observed, "usable": np.ones_like(observed)}
    valid = observed & native["usable"] & np.isfinite(xy).all(axis=-1)
    hand_valid = np.stack([valid[:, o:o + 21] for o in HAND_OFFSETS], axis=1)
    current = np.column_stack((hand_valid.all(axis=-1), np.ones(n, dtype=bool)))
    index = np.broadcast_to(np.arange(n)[:, None], (n, 3))
    origins = np.maximum.accumulate(np.where(current, index, -1), axis=0)
    kinds = np.where(current, 1, np.where(origins >= 0, 2, 0)).astype(np.uint8)
    body_current = valid[:, BODY_REQUIRED].all(axis=-1)
    body_origin = np.maximum.accumulate(np.where(body_current, np.arange(n), -1))
    body_kind = np.where(body_current, 1, np.where(body_origin >= 0, 2, 0)).astype(np.uint8)
    provenance = {
        "frame_id": native["frame_id"].copy(), "timestamp_us": times.copy(),
        "decode_pts_us": times.copy(), "source_kind": kinds,
        "origin_frame_index": origins.astype(np.int32), "hand_point_valid": hand_valid,
        "body_point_valid": valid[:, BODY_IDS], "body_source_kind": body_kind,
        "body_origin_frame_index": body_origin.astype(np.int32),
    }
    return native, provenance


def synthetic_observation(n=5):
    return build_observation_input(*synthetic_fixture(n), image_size=(1920, 1080))


class ObservationContractTests(unittest.TestCase):
    def test_shapes_finite_input_and_no_labels(self):
        item = synthetic_observation()
        self.assertEqual(item["observation_features"].shape, (5, 2, 108))
        self.assertTrue(np.isfinite(item["observation_features"]).all())
        self.assertIsNone(item["targets"])
        self.assertIsNone(item["target_availability"])
        self.assertIn("not_independently_verified", item["identity_status"])

    def test_never_observed_origin_is_not_frame_zero(self):
        item = synthetic_observation()
        np.testing.assert_array_equal(item["hand_origin_frame_index"][:2, 0], [-1, -1])
        np.testing.assert_array_equal(item["hand_origin_timestamp_us"][:2, 0], [-1, -1])
        self.assertTrue(np.isnan(item["hand_rgb_age_s"][:2, 0]).all())
        self.assertFalse(item["hand_rgb_age_known"][:2, 0].any())
        self.assertTrue(item["hand_rgb_age_known"][:, 1].all())

    def test_previous_age_uses_actual_timestamps(self):
        item = synthetic_observation()
        self.assertEqual(item["hand_source_kind"][3, 0], 2)
        self.assertAlmostEqual(item["hand_rgb_age_s"][3, 0], 0.035)
        self.assertAlmostEqual(item["hand_rgb_age_s"][1, 1], 0.033333)
        np.testing.assert_allclose(item["frame_delta_s"], [0, 0.033333, 0.056667, 0.035, 0.125])

    def test_missing_finger_preserves_current_wrist_body_and_other_hand(self):
        item = synthetic_observation()
        self.assertFalse(item["finger_geometry_valid"][3, 0, 1])
        self.assertTrue(item["finger_geometry_valid"][3, 0, 2])
        self.assertTrue(item["hand_wrist_valid"][3, 0])
        self.assertTrue(item["pose_wrist_valid"][3, 0])
        self.assertTrue(item["finger_geometry_valid"][3, 1].all())
        self.assertFalse(item["hand_rgb_current"][3, 0])
        self.assertEqual(item["body14_source_kind"][3], 2)
        self.assertFalse(item["body_point_valid"][3, 0])
        self.assertTrue(item["body_point_valid"][3, 1:].all())

    def test_missing_hand_never_means_sequence_padding(self):
        item = synthetic_observation()
        self.assertFalse(item["sequence_padding_mask"].any())
        batch = pad_observation_batch([item, synthetic_observation(3)])
        self.assertEqual(int(batch["sequence_padding_mask"].sum()), 2)
        self.assertEqual(int(batch["hand_missing_rgb_mask"].sum()), 4)
        self.assertFalse(batch["sequence_padding_mask"][0].any())
        self.assertTrue(batch["hand_point_valid"][0, 0, 1].all())
        self.assertFalse(batch["hand_missing_rgb_mask"][1, 3:].any())

    def test_inputs_are_not_mutated(self):
        native, provenance = synthetic_fixture()
        before = copy.deepcopy((native, provenance))
        build_observation_input(native, provenance, (1920, 1080))
        for original, after in zip(before, (native, provenance)):
            for key in original:
                np.testing.assert_array_equal(original[key], after[key])

    def test_unknown_source_kind_rejected(self):
        native, provenance = synthetic_fixture()
        provenance["source_kind"][2, 0] = 3
        with self.assertRaisesRegex(ValueError, "source_kind"):
            build_observation_input(native, provenance, (1920, 1080))

    def test_blank_origin_zero_future_and_stale_origins_rejected(self):
        for frame, hand, origin in [(0, 0, 0), (3, 0, 4), (4, 0, 2)]:
            with self.subTest(frame=frame, origin=origin):
                native, provenance = synthetic_fixture()
                provenance["origin_frame_index"][frame, hand] = origin
                with self.assertRaisesRegex(ValueError, "origin"):
                    build_observation_input(native, provenance, (1920, 1080))

    def test_strict_shapes_and_dtypes(self):
        for key, value in [("source_kind", np.ones((5, 2), dtype=np.uint8)),
                           ("source_kind", np.ones((5, 3), dtype=np.float32)),
                           ("hand_point_valid", np.ones((5, 2, 21), dtype=np.uint8))]:
            with self.subTest(key=key, shape=value.shape, dtype=value.dtype):
                native, provenance = synthetic_fixture()
                provenance[key] = value
                with self.assertRaises(ValueError):
                    build_observation_input(native, provenance, (1920, 1080))

    def test_time_and_frame_mismatch_rejected(self):
        for key, value in [("timestamp_us", 0), ("frame_id", 100), ("decode_pts_us", 1_000_000)]:
            with self.subTest(key=key):
                native, provenance = synthetic_fixture()
                provenance[key][2] = value
                with self.assertRaises(ValueError):
                    build_observation_input(native, provenance, (1920, 1080))
        native, provenance = synthetic_fixture()
        native["timestamp_us"][2] = native["timestamp_us"][1]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_observation_input(native, provenance, (1920, 1080))
        native, provenance = synthetic_fixture()
        native["timestamp_us"] = native["timestamp_us"].astype(np.uint64)
        with self.assertRaisesRegex(ValueError, "signed integer"):
            build_observation_input(native, provenance, (1920, 1080))

    def test_mask_and_current_source_disagreement_rejected(self):
        native, provenance = synthetic_fixture()
        provenance["hand_point_valid"][3, 0, 7] = True
        with self.assertRaisesRegex(ValueError, "hand_point_valid"):
            build_observation_input(native, provenance, (1920, 1080))
        native, provenance = synthetic_fixture()
        provenance["source_kind"][3, 0] = 1
        provenance["origin_frame_index"][3, 0] = 3
        with self.assertRaisesRegex(ValueError, "all-21"):
            build_observation_input(native, provenance, (1920, 1080))

    def test_invalid_xy_zeroed_only_with_separate_validity_mask(self):
        native, provenance = synthetic_fixture()
        native["xy_px"][3, 40] = np.nan
        item = build_observation_input(native, provenance, (1920, 1080))
        self.assertFalse(item["hand_point_valid"][3, 0, 7])
        np.testing.assert_array_equal(item["hand_xy_normalized"][3, 0, 7], [0, 0])
        self.assertTrue(item["hand_wrist_valid"][3, 0])

    def test_saved_context_append_preserves_values_without_repair_claim(self):
        item = synthetic_observation()
        context = np.arange(20, dtype=np.float32).reshape(5, 4)
        joined = append_saved_context(item, context)
        self.assertEqual(joined.shape, (5, 2, 112))
        np.testing.assert_array_equal(joined[:, 0, -4:], context)
        np.testing.assert_array_equal(joined[:, 1, -4:], context)
        self.assertIn("stale RGB", " ".join(item["limitations"]))
        with self.assertRaises(ValueError):
            append_saved_context(item, context[:4])

    def test_batch_rejects_malformed_shapes_dtypes_order_frames_and_nonfinite(self):
        cases = [
            ("source_broadcast", "hand_source_kind", np.ones((5, 1), dtype=np.uint8)),
            ("points_broadcast", "hand_point_valid", np.ones((5, 1, 21), dtype=bool)),
            ("scalar_padding", "sequence_padding_mask", np.asarray(False)),
            ("reverse_hands", "hand_order", ["anatomical_right", "anatomical_left"]),
            ("frame_order", "frame_id", np.array([0, 2, 1, 3, 4], dtype=np.int64)),
            ("float_source", "hand_source_kind", np.ones((5, 2), dtype=np.float32)),
            ("unknown_source", "hand_source_kind", np.full((5, 2), 9, dtype=np.uint8)),
            ("integer_points", "hand_point_valid", np.ones((5, 2, 21), dtype=np.uint8)),
            ("integer_padding", "sequence_padding_mask", np.zeros(5, dtype=np.uint8)),
            ("scalar_frame", "frame_id", np.asarray(0)),
            ("float_frame", "frame_id", np.arange(5, dtype=np.float32)),
            ("backward_time", "timestamp_us", np.array([0, 100, 50, 200, 300], dtype=np.int64)),
        ]
        for name, key, value in cases:
            with self.subTest(case=name):
                item = synthetic_observation()
                item[key] = value
                with self.assertRaises(ValueError):
                    pad_observation_batch([item])
        for dtype, fill in [(np.float32, np.nan), (np.float64, 0)]:
            with self.subTest(dtype=dtype, fill=fill):
                item = synthetic_observation()
                item["observation_features"] = np.full(item["observation_features"].shape, fill, dtype=dtype)
                with self.assertRaises(ValueError):
                    pad_observation_batch([item])

    def test_body14_copy_status_changes_actual_numeric_side_input(self):
        before = synthetic_observation()
        native, provenance = synthetic_fixture()
        native["observed"][4, 2] = False  # Required by body14, absent from seven visible body slots.
        provenance["body_source_kind"][4] = 2
        provenance["body_origin_frame_index"][4] = 2
        after = build_observation_input(native, provenance, (1920, 1080))
        self.assertEqual(before["body14_source_kind"][4], 1)
        self.assertEqual(after["body14_source_kind"][4], 2)
        self.assertAlmostEqual(after["body14_age_s"][4], 0.16)
        self.assertFalse(np.array_equal(before["observation_features"], after["observation_features"]))

    def test_face_copy_status_changes_actual_numeric_side_input(self):
        before = synthetic_observation()
        native, provenance = synthetic_fixture()
        provenance["source_kind"][4, 2] = 2
        provenance["origin_frame_index"][4, 2] = 3
        after = build_observation_input(native, provenance, (1920, 1080))
        self.assertAlmostEqual(after["face_rgb_age_s"][4], 0.125)
        self.assertFalse(np.array_equal(before["observation_features"], after["observation_features"]))


class PartialLabelLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def fixture(self):
        torch = self.torch
        # SYNTHETIC logits and class labels; no semantic classes are asserted.
        logits = torch.arange(24, dtype=torch.float64).reshape(1, 2, 2, 2, 3) / 10
        logits.requires_grad_()
        targets = torch.tensor([[[[1, -1], [0, 2]], [[-1, -1], [-1, -1]]]])
        confirmed = targets >= 0
        padding = torch.zeros((1, 2), dtype=torch.bool)
        return logits, targets, confirmed, padding

    def test_unknown_item_zero_gradient_other_hand_and_item_preserved(self):
        logits, targets, confirmed, padding = self.fixture()
        result = partial_label_cross_entropy(logits, targets, confirmed, padding)
        self.assertEqual(result["supervised_items"], 3)
        result["loss"].backward()
        self.assertEqual(int(self.torch.count_nonzero(logits.grad[~confirmed])), 0)
        self.assertTrue((logits.grad[confirmed].abs().sum(dim=-1) > 0).all().item())

    def test_confirmed_target_survives_missing_sensor(self):
        logits, targets, confirmed, padding = self.fixture()
        sensor = synthetic_observation()["hand_rgb_current"][:2]
        self.assertFalse(sensor[0, 0])
        self.assertTrue(confirmed[0, 0, 0, 0].item())
        result = partial_label_cross_entropy(logits, targets, confirmed, padding)
        result["loss"].backward()
        self.assertGreater(float(logits.grad[0, 0, 0, 0].abs().sum()), 0)

    def test_ignored_nan_logits_do_not_contaminate_loss(self):
        logits, targets, confirmed, padding = self.fixture()
        with self.torch.no_grad():
            logits[~confirmed] = float("nan")
        result = partial_label_cross_entropy(logits, targets, confirmed, padding)
        self.assertTrue(self.torch.isfinite(result["loss"]).item())
        result["loss"].backward()
        self.assertTrue(self.torch.isfinite(logits.grad).all().item())
        self.assertEqual(int(self.torch.count_nonzero(logits.grad[~confirmed])), 0)

    def test_all_unknown_returns_connected_zero_even_with_nan(self):
        logits, targets, confirmed, padding = self.fixture()
        confirmed[:] = False
        with self.torch.no_grad():
            logits[:] = float("nan")
        result = partial_label_cross_entropy(logits, targets, confirmed, padding)
        self.assertEqual(float(result["loss"].detach()), 0)
        self.assertEqual(result["supervised_items"], 0)
        result["loss"].backward()
        self.assertEqual(int(self.torch.count_nonzero(logits.grad)), 0)

    def test_sequence_padding_masks_targets_not_sensor_hand(self):
        logits, targets, confirmed, padding = self.fixture()
        padding[0, 0] = True
        result = partial_label_cross_entropy(logits, targets, confirmed, padding)
        self.assertEqual(result["supervised_items"], 0)
        result["loss"].backward()
        self.assertEqual(int(self.torch.count_nonzero(logits.grad)), 0)

    def test_loss_matches_manual_selected_mean_and_reductions(self):
        torch = self.torch
        logits, targets, confirmed, padding = self.fixture()
        expected = torch.nn.functional.cross_entropy(logits[confirmed], targets[confirmed])
        result = partial_label_cross_entropy(logits, targets, confirmed, padding)
        torch.testing.assert_close(result["loss"], expected)
        sum_result = partial_label_cross_entropy(logits, targets, confirmed, padding, "sum")
        torch.testing.assert_close(sum_result["loss"], expected * 3)
        none_result = partial_label_cross_entropy(logits, targets, confirmed, padding, "none")
        torch.testing.assert_close(none_result["loss"].sum(), expected * 3)

    def test_invalid_confirmed_target_or_shape_rejected(self):
        logits, targets, confirmed, padding = self.fixture()
        targets[0, 0, 0, 0] = 3
        with self.assertRaisesRegex(ValueError, "valid class"):
            partial_label_cross_entropy(logits, targets, confirmed, padding)
        with self.assertRaisesRegex(ValueError, "match"):
            partial_label_cross_entropy(logits, targets[:, :1], confirmed, padding)


class SavedPublicInputTests(unittest.TestCase):
    def test_all_twelve_saved_ffn_join_with_matching_provenance_and_pins(self):
        total = 0
        for i in range(1, 13):
            item = load_public_saved_input(f"V{i:02d}")
            joined = join_public_saved_context(item)
            self.assertEqual(joined.shape, (len(item["frame_id"]), 2, 876))
            self.assertTrue(np.isfinite(joined).all())
            np.testing.assert_array_equal(joined[:, :, :108], item["observation_features"])
            np.testing.assert_array_equal(joined[:, 0, 108:], joined[:, 1, 108:])
            total += len(joined)
        self.assertEqual(total, 1438)
        print("PUBLIC_SAVED_CONTEXT_COUNTS=" + json.dumps({"clips": 12, "frames": total,
              "side_input_dimensions": 108, "saved_context_dimensions": 768,
              "joined_per_hand_dimensions": 876, "new_inference": False}, sort_keys=True))

    def test_all_twelve_saved_public_inputs_have_no_generated_targets(self):
        rows, totals = [], {"frames": 0, "current_hand_rgb": 0, "previous_hand_rgb": 0, "blank_hand_rgb": 0}
        clips = []
        for i in range(1, 13):
            item = load_public_saved_input(f"V{i:02d}")
            clips.append(item)
            kinds = item["hand_source_kind"]
            counts = {"frames": len(kinds), "current_hand_rgb": int((kinds == 1).sum()),
                      "previous_hand_rgb": int((kinds == 2).sum()), "blank_hand_rgb": int((kinds == 0).sum())}
            for key, value in counts.items():
                totals[key] += value
            rows.append({"alias": item["public_alias"], **counts})
            self.assertIsNone(item["targets"])
            self.assertIsNone(item["target_availability"])
            self.assertFalse(item["sequence_padding_mask"].any())
            self.assertEqual(int((item["hand_rgb_age_s"][kinds == 1] != 0).sum()), 0)
            self.assertTrue((item["hand_rgb_age_s"][kinds == 2] > 0).all())
        self.assertEqual(totals, {"frames": 1438, "current_hand_rgb": 2764, "previous_hand_rgb": 112, "blank_hand_rgb": 0})
        batch = pad_observation_batch(clips)
        self.assertEqual(int((~batch["sequence_padding_mask"]).sum()), 1438)
        self.assertEqual(int(batch["hand_missing_rgb_mask"].sum()), 0)
        print("PUBLIC_INPUT_COUNTS=" + json.dumps({"totals": totals, "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
