"""Observation -> role evidence -> sampled reference -> feedback eligibility regressions."""
from pathlib import Path
import sys, unittest, copy
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from wis_lab.core import features as F
from wis_lab.core.observation import Observation, empty_frames
from wis_lab.core.recognize import _feedback_support, AXIS_GROUPS, build_reference

def observation(left_active=False, right_active=True, n=20):
    a = empty_frames(n)
    # Detailed hand landmarks remain present to separate feedback missingness from POSE wrist
    # missingness. The shoulders are 100px apart; a resting pose wrist is 180px below them.
    a['xy_px'][:] = np.arange(75, dtype=np.float32)[None, :, None] + 10
    a['xy_px'][:, 11] = (-50, 0)
    a['xy_px'][:, 12] = (50, 0)
    a['xy_px'][:, 15] = (-60, 50 if left_active else 180)
    a['xy_px'][:, 16] = (60, 50 if right_active else 180)
    a['observed'][:] = a['usable'][:] = True
    a['timestamp_us'][:] = np.arange(n) * 33333
    a['frame_id'][:] = np.arange(n)
    obs = Observation(**a)
    obs.validate()
    return obs

def role_evidence(left_active=False, right_active=True):
    obs = observation(left_active, right_active)
    origin, scale, _ = F.body_frame(obs)
    return F.feedback_hand_roles(obs, origin, scale, 0, obs.n_frames - 1)

class RoleSupport(unittest.TestCase):
    def support(self, q, r):
        groups = {'bend_L':{'mean_distance':.8,'compared_steps':2,'unobserved_steps':8},
                  'bend_R':{'mean_distance':.2,'compared_steps':8,'unobserved_steps':2}}
        return _feedback_support(groups, AXIS_GROUPS['handshape'], q, r,
                                  {'feedback':{'exclude_both_resting_hands':True}})

    def test_raw_evidence_survives_and_coverage_uses_eligible_groups(self):
        q = F.uniform_time_sample(F.sequence_features(observation()), 8)
        r = build_reference(observation(), 'synthetic_clip', 'SYNTHETIC', 8)
        self.assertEqual(q['feedback_hand_roles'], r['feedback_hand_roles'])
        self.assertEqual(q['feedback_hand_roles']['hands']['L']['paired_frames'], 20)
        s = self.support(q,r)
        self.assertEqual(s['excluded_groups'], ['bend_L'])
        self.assertEqual(s['missing_groups'], [])
        self.assertTrue(s['per_group']['bend_L']['measured'])
        self.assertEqual(s['per_group']['bend_L']['mean_distance'], .8)
        self.assertEqual(s['per_group']['bend_L']['aggregate_weight'], 0)
        self.assertEqual(s['time_fraction_min'], .8)
        self.assertTrue(s['partial'])

    def test_legacy_false_or_missing_role_is_not_rest(self):
        for value in (None, {}, {'L':False}, {'L':np.bool_(False)}, {'L':0}, {'L':'false'}):
            with self.subTest(value=value):
                s = self.support({'participating':value}, {'feedback_hand_roles':role_evidence()})
                self.assertEqual(s['excluded_groups'], [])

    def test_insufficient_pose_wrist_is_unknown_despite_legacy_false(self):
        obs = observation()
        obs.usable[4:, 15] = False
        q = F.uniform_time_sample(F.sequence_features(obs), 8)
        r = build_reference(observation(), 'c', 'SYNTHETIC', 8)
        self.assertIs(q['participating']['L'], False)
        self.assertEqual(q['feedback_hand_roles']['hands']['L']['state'], 'UNKNOWN')
        self.assertEqual(self.support(q,r)['excluded_groups'], [])

    def test_shoulders_and_wrist_must_be_observed_together(self):
        obs = observation()
        obs.usable[5:, 11:13] = False
        obs.usable[:5, 15] = False
        q = F.uniform_time_sample(F.sequence_features(obs), 8)
        r = build_reference(observation(), 'c', 'SYNTHETIC', 8)
        d = q['feedback_hand_roles']['hands']['L']
        self.assertEqual(d['paired_frames'], 0)
        self.assertEqual(d['state'], 'UNKNOWN')
        self.assertEqual(self.support(q,r)['excluded_groups'], [])

    def test_both_unobserved_pose_wrists_are_not_two_resting_hands(self):
        obs = observation(right_active=False)
        obs.usable[4:, 15:17] = False
        q = F.uniform_time_sample(F.sequence_features(obs), 8)
        r = build_reference(observation(right_active=False), 'c', 'SYNTHETIC', 8)
        self.assertEqual(q['participating'], {'L':False, 'R':False})
        self.assertEqual([q['feedback_hand_roles']['hands'][s]['state'] for s in ('L','R')],
                         ['UNKNOWN','UNKNOWN'])
        self.assertEqual(self.support(q,r)['excluded_groups'], [])

    def test_low_fraction_is_unknown_even_with_five_frames(self):
        obs = observation(n=30)
        obs.usable[5:, 15] = False
        d = F.sequence_features(obs)['feedback_hand_roles']['hands']['L']
        self.assertEqual(d['state'], 'UNKNOWN')

    def test_nonfinite_shoulders_do_not_count(self):
        obs = observation()
        origin, scale, _ = F.body_frame(obs)
        obs.xy_px[4:, 11] = np.nan
        ev = F.feedback_hand_roles(obs, origin, scale, 0, 19)
        self.assertEqual(ev['hands']['L']['paired_frames'], 4)
        self.assertFalse(F.observed_resting_hand(ev, 'L'))

    def test_claim_of_rest_without_valid_evidence_does_not_exclude(self):
        ref = {'feedback_hand_roles':role_evidence()}
        cases = [None, {}, {'schema':F.FEEDBACK_ROLE_SCHEMA,'hands':{'L':{'state':'RESTING'}}}]
        for key, value in [('paired_frames',4), ('paired_fraction',.1), ('median_height_sw',None),
                           ('vertical_amplitude_sw',float('nan')), ('vertical_amplitude_sw',.5),
                           ('state','UNKNOWN')]:
            ev = copy.deepcopy(ref['feedback_hand_roles'])
            ev['hands']['L'][key] = value
            cases.append(ev)
        for ev in cases:
            with self.subTest(evidence=ev):
                self.assertEqual(self.support({'feedback_hand_roles':ev},ref)['excluded_groups'], [])

    def test_active_in_either_sequence_remains_eligible(self):
        rest = {'feedback_hand_roles':role_evidence()}
        active = {'feedback_hand_roles':role_evidence(left_active=True)}
        self.assertEqual(self.support(active,rest)['excluded_groups'], [])
        self.assertEqual(self.support(rest,active)['excluded_groups'], [])
        self.assertEqual(F.GROUPS['bend_R'][1], 0)
        self.assertTrue(self.support(rest,rest)['per_group']['bend_R']['included_in_aggregate'])

    def test_active_but_unobserved_hand_remains_missing(self):
        s = _feedback_support({'bend_R':{'mean_distance':.2,'compared_steps':8}},
             AXIS_GROUPS['handshape'], {'feedback_hand_roles':role_evidence(left_active=True)},
             {'feedback_hand_roles':role_evidence()},
             {'feedback':{'exclude_both_resting_hands':True}})
        self.assertEqual(s['missing_groups'], ['bend_L'])
        self.assertEqual(s['excluded_groups'], [])
        self.assertIsNone(s['per_group']['bend_L']['mean_distance'])
        self.assertTrue(s['partial'])

    def test_default_is_identical_and_does_not_mutate_evidence(self):
        from wis_lab.core.recognize import _axis_support
        q = {'feedback_hand_roles':role_evidence()}
        before = copy.deepcopy(q)
        groups = {'bend_L':{'mean_distance':.8,'compared_steps':2}}
        s = _feedback_support(groups, AXIS_GROUPS['handshape'], q, q, {})
        self.assertEqual(s, _axis_support(groups, AXIS_GROUPS['handshape']))
        self.assertEqual(q, before)

if __name__ == '__main__':
    unittest.main()
