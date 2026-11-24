"""Unit tests for the multiview review adapter (synthetic data only)."""
import copy, os, sys, unittest  # noqa: E401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiview_review_adapter_v1 import POSE_MAP, adapt_frame  # noqa: E402

W, H = 640.0, 480.0


def raw_pose():
    return [[0.01 * i, 0.002 * i, 0.5, 0.9] for i in range(33)]


def enriched_pose():
    return [{"norm_x": 0.01 * i, "norm_y": 0.002 * i, "norm_z": 0.5,
             "visibility": 0.9, "world_x": 7} for i in range(33)]


def raw_hand(label, score=0.8, base=0.0):
    return {"handedness": label, "score": score,
            "norm": [[base + 0.01 * j, 0.003 * j, 0.2] for j in range(21)]}


def enriched_hand(side, score=0.8, base=0.0):
    return {"anatomical_side": side, "mp_score": score,
            "landmarks": [{"norm_x": base + 0.01 * j, "norm_y": 0.003 * j,
                           "norm_z": 0.2, "px": 1} for j in range(21)]}


def raw_frame():
    hands = [raw_hand("Left"), raw_hand("Right", base=0.5)]
    return {"pose_norm": raw_pose(), "hands": hands, "timestamp": 3}


def enriched_frame():
    return {"pose": enriched_pose(),
            "hands": {"left": enriched_hand("left"),
                      "right": enriched_hand("right", base=0.5)}}


class AdapterTest(unittest.TestCase):

    def test_pose_body25_mapping_and_non_square_scaling(self):
        frame, diag = adapt_frame(raw_frame(), [W, H], "raw")
        pose = frame["people"][0]["pose_keypoints_2d"]
        self.assertEqual(len(pose), 75)
        for mp_index, body_index in POSE_MAP.items():
            off = body_index * 3
            self.assertAlmostEqual(pose[off], 0.01 * mp_index * W)
            self.assertAlmostEqual(pose[off + 1], 0.002 * mp_index * H)
            self.assertAlmostEqual(pose[off + 2], 0.9)
        for body_index in (0, 1, 8, 24):
            self.assertEqual(pose[body_index * 3:body_index * 3 + 3],
                             [0.0, 0.0, 0.0])
        self.assertEqual(diag["copied_pose_joints"], 8)

    def test_hand_uses_shared_detector_score_and_keeps_joint_order(self):
        frame, _ = adapt_frame(raw_frame(), [W, H], "raw")
        person = frame["people"][0]
        left = person["hand_left_keypoints_2d"]
        self.assertEqual(len(left), 63)
        self.assertEqual(len(person["hand_right_keypoints_2d"]), 63)
        for j in range(21):
            self.assertAlmostEqual(left[j * 3], 0.01 * j * W)
            self.assertAlmostEqual(left[j * 3 + 1], 0.003 * j * H)
            self.assertAlmostEqual(left[j * 3 + 2], 0.8)
        thumb_x = [left[j * 3] for j in range(1, 5)]
        self.assertEqual(thumb_x, sorted(thumb_x))
        self.assertAlmostEqual(person["hand_right_keypoints_2d"][0], 0.5 * W)

    def test_raw_and_enriched_agree_on_equivalent_data(self):
        raw_out, raw_diag = adapt_frame(raw_frame(), [W, H], "raw")
        enr_out, enr_diag = adapt_frame(enriched_frame(), [W, H], "enriched")
        self.assertEqual(raw_out, enr_out)
        self.assertEqual(raw_diag["side_policy"], "raw_declared_unverified")
        self.assertEqual(enr_diag["side_policy"],
                         "enriched_declared_anatomical_unverified")
        for diag in (raw_diag, enr_diag):
            self.assertEqual(diag["schema"], "wis_multiview_review_adapter_v1")
            self.assertEqual(diag["quality"], "UNASSESSED")
            self.assertFalse(diag["enrollment_eligible"])
            self.assertEqual(diag["coordinate_space"], "pixel_xy")
            self.assertFalse(diag["depth_used"])
            self.assertEqual(diag["hand_confidence_basis"],
                             "detector_score_shared_across_21_joints")

    def test_missing_pose_and_hands(self):
        empty, diag = adapt_frame({}, [W, H], "raw")
        self.assertEqual(empty, {"people": []})
        self.assertEqual(diag["ambiguous_sides"], [])
        for kind, hands in (("raw", []), ("enriched", {}),
                            ("enriched", {"left": None, "right": None})):
            out, _ = adapt_frame({"hands": hands}, [W, H], kind)
            self.assertEqual(out, {"people": []})
        pose_only, _ = adapt_frame({"pose_norm": raw_pose()}, [W, H], "raw")
        person = pose_only["people"][0]
        self.assertEqual(person["hand_left_keypoints_2d"], [0.0] * 63)
        self.assertEqual(person["hand_right_keypoints_2d"], [0.0] * 63)
        hands_only, _ = adapt_frame({"pose": None, "hands": {"left": None,
                                     "right": enriched_hand("right")}},
                                    [W, H], "enriched")
        person = hands_only["people"][0]
        self.assertEqual(person["pose_keypoints_2d"], [0.0] * 75)
        self.assertEqual(person["hand_left_keypoints_2d"], [0.0] * 63)
        self.assertAlmostEqual(person["hand_right_keypoints_2d"][62], 0.8)

    def test_bad_confidence_zeroed_and_z_unconsumed(self):
        pose = raw_pose()
        pose[11][3] = float("nan")
        pose[12][3] = 0.0
        pose[13][3] = True
        pose[14][2] = float("inf")
        pose[15][0] = "0.4"
        frame, diag = adapt_frame({"pose_norm": pose}, [W, H], "raw")
        keypoints = frame["people"][0]["pose_keypoints_2d"]
        for mp_index in (11, 12, 13, 15):
            off = POSE_MAP[mp_index] * 3
            self.assertEqual(keypoints[off:off + 3], [0.0, 0.0, 0.0])
        off = POSE_MAP[14] * 3
        self.assertAlmostEqual(keypoints[off], 0.14 * W)
        self.assertEqual(diag["copied_pose_joints"], 4)

    def test_raw_duplicate_side_is_dropped_not_selected(self):
        frame = raw_frame()
        frame["hands"].append(raw_hand("Left", score=0.99, base=0.9))
        out, diag = adapt_frame(frame, [W, H], "raw")
        person = out["people"][0]
        self.assertEqual(diag["ambiguous_sides"], ["left"])
        self.assertEqual(person["hand_left_keypoints_2d"], [0.0] * 63)
        self.assertAlmostEqual(person["hand_right_keypoints_2d"][0], 0.5 * W)
        self.assertAlmostEqual(person["pose_keypoints_2d"][5 * 3], 0.11 * W)

    def test_enriched_side_mismatch_and_unknown_labels_rejected(self):
        bad = {"hands": {"left": enriched_hand("right")}}
        self.assertRaises(ValueError, adapt_frame, bad, [W, H], "enriched")
        unknown_key = {"hands": {"middle": enriched_hand("left")}}
        self.assertRaises(ValueError, adapt_frame, unknown_key, [W, H],
                          "enriched")
        unknown_side = {"hands": {"left": dict(enriched_hand("left"),
                                               anatomical_side="LEFT")}}
        self.assertRaises(ValueError, adapt_frame, unknown_side, [W, H],
                          "enriched")
        raw_label = {"hands": [dict(raw_hand("Left"), handedness="left")]}
        self.assertRaises(ValueError, adapt_frame, raw_label, [W, H], "raw")

    def test_malformed_frame_and_lengths_rejected(self):
        self.assertRaises(ValueError, adapt_frame, [], [W, H], "raw")
        self.assertRaises(ValueError, adapt_frame, {}, [W, H], "openpose")
        self.assertRaises(ValueError, adapt_frame,
                          {"pose_norm": raw_pose()[:32]}, [W, H], "raw")
        self.assertRaises(ValueError, adapt_frame,
                          {"pose_norm": [[0.1, 0.2, 0.3]] * 33}, [W, H], "raw")
        short = raw_hand("Right")
        short["norm"] = short["norm"][:20]
        self.assertRaises(ValueError, adapt_frame, {"hands": [short]},
                          [W, H], "raw")
        self.assertRaises(ValueError, adapt_frame, {"hands": [None]},
                          [W, H], "raw")
        self.assertRaises(ValueError, adapt_frame,
                          {"pose": raw_pose()}, [W, H], "enriched")

    def test_malformed_resolution_rejected(self):
        for resolution in ([W], [0, H], [W, -1], [True, H], [W, "480"],
                           [W, float("inf")], "640x480"):
            self.assertRaises(ValueError, adapt_frame, raw_frame(),
                              resolution, "raw")

    def test_input_is_not_mutated(self):
        for frame, kind in ((raw_frame(), "raw"),
                            (enriched_frame(), "enriched")):
            before = copy.deepcopy(frame)
            resolution = [W, H]
            adapt_frame(frame, resolution, kind)
            self.assertEqual(frame, before)
            self.assertEqual(resolution, [W, H])


if __name__ == "__main__":
    unittest.main()
