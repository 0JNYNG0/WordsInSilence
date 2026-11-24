"""Independent synthetic arm-chain counterexamples, fixed before real replay."""
from __future__ import annotations

import unittest
import numpy as np
import assignment

HandAssigner = assignment.HandAssigner


def hand(x, y):
    return np.array([[0, 0], [-7, -3], [-12, -7], [-16, -12], [-20, -16],
        [-8, -17], [-9, -26], [-10, -34], [-11, -41], [0, -20], [0, -31],
        [0, -40], [0, -48], [8, -17], [9, -27], [10, -35], [11, -42],
        [15, -12], [18, -21], [20, -28], [22, -34]], float) + [x, y]


def pose(shift=0, wrists=True):
    p = np.full((33, 2), np.nan)
    u = np.zeros(33, bool)
    p[11], p[12] = [260, 150+shift], [140, 150+shift]
    p[13], p[14] = [260, 220+shift], [140, 220+shift]
    p[15], p[16] = [300, 260+shift], [100, 260+shift]
    u[[11, 12, 13, 14]] = True
    u[[15, 16]] = wrists
    return p, u


def observe(a, hs, labels, t, p, u, scores=None):
    return a.assign(np.asarray(hs).reshape(-1, 21, 2), labels,
                    [.99]*len(hs) if scores is None else scores, p, u, t, 640, 480)


def seed(a):
    p, u = pose()
    r, _ = observe(a, [hand(300, 260), hand(100, 260)], [0, 1], 0, p, u)
    assert r == {0: 0, 1: 1}


def tie_input():
    p, u = pose(20, False)
    return [hand(100, 280), hand(120, 280)], [1, 1], p, u


class ArmChainTests(unittest.TestCase):
    def test_arm_chain_resolves_competing_raw_candidates(self):
        a = HandAssigner(); seed(a)
        hs, labels, p, u = tie_input()
        r, d = observe(a, hs, labels, 33333, p, u)
        self.assertEqual(d['baseline_assigned_sides'], [None, None])
        self.assertEqual(r, {0: 1, 1: None})
        self.assertEqual(d['reasons'][0], 'arm_chain_supported_recovery')

    def test_arm_chain_does_not_modify_legacy_tracks_or_following_results(self):
        a, b = HandAssigner(), assignment._base.HandAssigner()
        seed(a); seed(b)
        hs, labels, p, u = tie_input()
        for t in [33333, 66667, 100000, 133333, 200000, 600000]:
            _, d = observe(a, hs, labels, t, p, u)
            rb, db = observe(b, hs, labels, t, p, u)
            self.assertEqual(d['baseline_assigned_sides'], list(rb.values()))
            self.assertEqual(d['baseline_reasons'], db['reasons'])
            self.assertEqual(set(a.tracks), set(b.tracks))
            for s in a.tracks:
                for name in ['palm', 'velocity']:
                    np.testing.assert_array_equal(getattr(a.tracks[s], name), getattr(b.tracks[s], name))
                for name in ['time', 'frame', 'velocity_known']:
                    self.assertEqual(getattr(a.tracks[s], name), getattr(b.tracks[s], name))

    def test_recoveries_never_refresh_arm_anchor_and_frame_expiry(self):
        a = HandAssigner(); seed(a)
        hs, labels, p, u = tie_input()
        for t in [33333, 66667]:
            r, _ = observe(a, hs, labels, t, p, u)
            self.assertEqual(r[0], 1)
            self.assertEqual(a.arm_anchors[1].time, 0)
            self.assertEqual(a.arm_anchors[1].frame, 0)
        r, d = observe(a, hs, labels, 99999, p, u)
        self.assertEqual(r, {0: None, 1: None})
        self.assertIn(1, d['arm_chain']['expired_sides'])

    def test_elapsed_time_expiry_even_with_only_one_frame_gap(self):
        a = HandAssigner(); seed(a)
        hs, labels, p, u = tie_input()
        r, d = observe(a, hs, labels, 100001, p, u)
        self.assertEqual(r, {0: None, 1: None})
        self.assertIn(1, d['arm_chain']['expired_sides'])

    def test_unknown_arm_history_never_inferred_from_shoulder_elbow(self):
        a = HandAssigner()
        hs, labels, p, u = tie_input()
        r, d = observe(a, hs, labels, 33333, p, u)
        self.assertEqual(r, {0: None, 1: None})
        self.assertFalse(d['arm_chain']['recoveries'])

    def test_no_raw_hand_never_generates_observation(self):
        a = HandAssigner(); seed(a)
        p, u = pose(20, False)
        r, d = observe(a, [], [], 33333, p, u)
        self.assertEqual(r, {})
        self.assertFalse(d['coordinates_generated'])
        self.assertEqual(d['missing_sides'], [0, 1])
        self.assertEqual(a.arm_anchors[1].time, 0)

    def test_missing_or_nonfinite_current_arm_blocks_recovery(self):
        for joint in [11, 12, 14]:
            for invalid_kind in ['mask', 'nan', 'outside']:
                with self.subTest(joint=joint, kind=invalid_kind):
                    a = HandAssigner(); seed(a)
                    hs, labels, p, u = tie_input()
                    if invalid_kind == 'mask': u[joint] = False
                    elif invalid_kind == 'nan': p[joint] = np.nan
                    else: p[joint] = [10000, 10000]
                    _, d = observe(a, hs, labels, 33333, p, u)
                    self.assertFalse(d['arm_chain']['recoveries'])

    def test_zero_upper_arm_or_large_arm_change_blocks_recovery(self):
        for kind in ['zero', 'large']:
            a = HandAssigner(); seed(a)
            hs, labels, p, u = tie_input()
            p[14] = p[12] if kind == 'zero' else [500, 300]
            _, d = observe(a, hs, labels, 33333, p, u)
            self.assertFalse(d['arm_chain']['recoveries'])

    def test_hand_label_disagreement_not_overridden(self):
        a = HandAssigner(); seed(a)
        hs, labels, p, u = tie_input()
        r, d = observe(a, hs, [0, 0], 33333, p, u)
        self.assertFalse(d['arm_chain']['recoveries'])
        self.assertNotIn(1, r.values())

    def test_close_duplicate_palms_cannot_fill_two_sides(self):
        a = HandAssigner(); seed(a)
        _, _, p, u = tie_input()
        r, d = observe(a, [hand(100, 280), hand(101, 280)], [1, 0], 33333, p, u)
        self.assertTrue(d['close_palms'])
        self.assertFalse(d['arm_chain']['recoveries'])
        self.assertLessEqual(len(set(s for s in r.values() if s is not None)), 1)

    def test_reset_and_timestamp_rewind_forget_arm_identity(self):
        for mode in ['explicit', 'equal', 'rewind']:
            a = HandAssigner(); seed(a)
            hs, labels, p, u = tie_input()
            if mode == 'explicit': a.reset()
            t = 33333 if mode == 'explicit' else (0 if mode == 'equal' else -1)
            _, d = observe(a, hs, labels, t, p, u)
            self.assertFalse(d['arm_chain']['recoveries'])
            self.assertFalse(a.arm_anchors)

    def test_raw_reversal_keeps_physical_selected_hand(self):
        for reverse in [False, True]:
            a = HandAssigner(); seed(a)
            hs, labels, p, u = tie_input()
            if reverse: hs, labels = hs[::-1], labels[::-1]
            r, _ = observe(a, hs, labels, 33333, p, u)
            self.assertEqual(r[1 if reverse else 0], 1)
            self.assertIsNone(r[0 if reverse else 1])

    def test_translation_and_uniform_scale_preserve_association(self):
        for scale, delta in [(1., [25., 15.]), (.75, [40., 50.])]:
            a = HandAssigner()
            p, u = pose()
            hs = [hand(300, 260), hand(100, 260)]
            observe(a, [h*scale+delta for h in hs], [0, 1], 0, p*scale+delta, u)
            hs, labels, p, u = tie_input()
            r, _ = observe(a, [h*scale+delta for h in hs], labels, 33333, p*scale+delta, u)
            self.assertEqual(r, {0: 1, 1: None})

    def test_invalid_or_low_score_hand_cannot_be_recovered(self):
        for kind in ['nan', 'low_score', 'palm_outside']:
            a = HandAssigner(); seed(a)
            hs, labels, p, u = tie_input()
            scores = [.99, .99]
            if kind == 'nan': hs[0][4, 0] = np.nan
            elif kind == 'low_score': scores[0] = .49
            else: hs[0][5] = [-1000, -1000]
            r, d = observe(a, hs, labels, 33333, p, u, scores)
            self.assertIsNone(r[0])
            self.assertFalse(any(x['raw_index'] == 0 for x in d['arm_chain']['recoveries']))

    def test_inputs_not_mutated_and_output_coordinates_not_synthesized(self):
        a = HandAssigner(); seed(a)
        hs, labels, p, u = tie_input()
        h = np.asarray(hs)
        before = [x.copy() for x in [h, p, u]]
        _, d = observe(a, h, labels, 33333, p, u)
        for actual, expected in zip([h, p, u], before):
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(d['arm_chain']['coordinates_generated'])
        self.assertFalse(d['arm_chain']['baseline_tracks_modified'])

    def test_reliable_current_pose_assignments_remain_untouched(self):
        a = HandAssigner(); seed(a)
        hs, labels, p, u = tie_input()
        p[15], p[16] = [100, 280], [120, 280]
        u[[15, 16]] = True
        r, d = observe(a, hs, labels, 33333, p, u)
        for i, side in enumerate(d['baseline_assigned_sides']):
            if side is not None: self.assertEqual(r[i], side)

    def test_missing_opposite_history_is_not_required_when_own_support_strong(self):
        a = HandAssigner(); seed(a)
        del a.arm_anchors[0]
        del a.tracks[0]
        hs, labels, p, u = tie_input()
        r, d = observe(a, hs, labels, 33333, p, u)
        self.assertEqual(r, {0: 1, 1: None})
        self.assertEqual(d['arm_chain']['recoveries'][0]['side'], 1)

    def test_recent_opposite_actual_hand_cannot_be_ignored_when_prediction_moves(self):
        a = HandAssigner(); seed(a)
        other = a.tracks[0]
        other.palm = hand(100, 280)[assignment.PALM]
        other.velocity = np.array([1200., 0.])
        other.velocity_known = True
        hs, labels, p, u = tie_input()
        r, d = observe(a, hs, labels, 33333, p, u)
        self.assertIsNone(r[0])
        self.assertTrue(any('opposite_observed_hand_not_excluded' in e['reasons']
                            for e in d['arm_chain']['rejected'] if e['raw_index'] == 0))

    def test_current_pose_track_conflict_is_not_promoted_by_old_arm(self):
        a = HandAssigner(); seed(a)
        p, u = pose(20, True)
        p[15], p[16] = [100, 280], [300, 280]
        r, d = observe(a, [hand(100, 280)], [0], 33333, p, u)
        self.assertTrue(a.pending_conflicts)
        self.assertEqual(r, {0: None})
        self.assertFalse(d['arm_chain']['recoveries'])

    def test_seed_without_real_full_chain_not_created(self):
        for absent in [13, 14, 15, 16]:
            a = HandAssigner()
            p, u = pose()
            u[absent] = False
            observe(a, [hand(300, 260), hand(100, 260)], [0, 1], 0, p, u)
            self.assertNotIn(absent % 2 == 0, a.arm_anchors)

