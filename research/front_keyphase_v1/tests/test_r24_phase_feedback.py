from pathlib import Path
import copy, json, sys, unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wis_lab.core.observation import Observation, empty_frames
from wis_lab.core import features as F
from wis_lab.core.phase_feedback import inspect_path
from wis_lab.core.recognize import build_reference
from wis_lab.core.trajectory import location_path
from wis_lab.core.dtw import _SLICES

def observation():
    a = empty_frames(12)
    a['xy_px'][:] = 50
    a['xy_px'][:, 11] = (-50, 0); a['xy_px'][:, 12] = (50, 0)
    a['xy_px'][:, 0] = (0, -60)
    a['xy_px'][:, 9] = (-5, -35); a['xy_px'][:, 10] = (5, -35)
    a['xy_px'][:, 13] = (-75, 75); a['xy_px'][:, 14] = (75, 75)
    for side, offset in ((33, -60), (54, 60)):
        a['xy_px'][:, side] = (offset, 70)
        for finger in range(5):
            for joint in range(4):
                a['xy_px'][:, side + 1 + 4 * finger + joint] = (offset + 3*joint + 4*finger, 65 - 5*joint)
    a['observed'][:] = a['usable'][:] = True
    a['usable'][:, 23:25] = False
    a['timestamp_us'][:] = np.arange(12) * 33333
    a['frame_id'][:] = np.arange(12)
    obs = Observation(**a); obs.validate(); return obs

def sample(obs=None):
    return build_reference(obs or observation(), 'SYNTHETIC', 'SYNTHETIC', 12)

class PhaseFeedback(unittest.TestCase):
    def test_translation_and_uniform_scale_preserve_body_context_and_metrics(self):
        obs = observation(); shifted = copy.deepcopy(obs)
        shifted.xy_px = shifted.xy_px * 2.5 + np.array([300, 500], dtype=np.float32)
        a, b = sample(obs), sample(shifted)
        # Existing float32 acos near a straight joint differs by 0.0198 degrees under
        # rescaling. Geometry has a tight tolerance; angle tolerance is numerical only,
        # not an execution-quality threshold. No existing angle implementation is changed.
        for group in F.GROUP_ORDER:
            tol = 0.03 / 180 if group.startswith('bend_') else 1e-6
            np.testing.assert_allclose(a['features'][:,_SLICES[group]],
                                       b['features'][:,_SLICES[group]], atol=tol, equal_nan=True)
        for key, pts in a['feedback_body_context']['points'].items():
            np.testing.assert_allclose(pts, b['feedback_body_context']['points'][key], atol=1e-6)
        report = inspect_path(a,b,[(i,i) for i in range(12)])
        np.testing.assert_allclose(report['steps'][0]['hands']['R']['shoulder_position']['delta'], 0, atol=1e-6)

    def test_no_torso_measurement_is_not_body_calibration(self):
        s = sample(); n = s['feedback_body_context']['normalization']
        self.assertEqual(n['paired_torso_frames'],0)
        self.assertEqual(n['paired_shoulders_frames'],12)
        self.assertFalse(n['arm_length_calibrated']); self.assertFalse(n['torso_length_calibrated'])
        obs=observation();obs.usable[:,23:25]=True
        n=sample(obs)['feedback_body_context']['normalization']
        self.assertEqual(n['paired_torso_frames'],12)
        self.assertFalse(n['torso_length_calibrated'])

    def test_unusable_finite_anchor_is_not_drawn(self):
        obs=observation();obs.usable[3,0]=False
        a=sample(obs)
        self.assertIsNone(a['feedback_body_context']['points']['nose'][3])
        self.assertIsNotNone(a['feedback_body_context']['points']['mouth'][3])

    def test_missing_shoulders_invalidate_all_anchors_on_that_frame(self):
        obs=observation();obs.usable[3,11]=False
        a=sample(obs)
        self.assertTrue(all(v[3] is None for v in a['feedback_body_context']['points'].values()))

    def test_unknown_hand_bend_remains_null_not_zero(self):
        a,b=sample(),sample();a['present'][0,F.GROUP_ORDER.index('bend_R')]=False
        out=inspect_path(a,b,[(0,0)])['steps'][0]['hands']['R']
        self.assertIsNone(out['finger_angles_deg']['index']['delta'])
        self.assertIsNotNone(out['shoulder_position']['delta'])

    def test_degree_units_finger_mapping_and_observed_zero(self):
        a,b=sample(),sample();sl=_SLICES['bend_R']
        a['features'][0,sl]=.5;b['features'][0,sl]=.5
        a['features'][0,sl.start+3:sl.start+6]=1
        out=inspect_path(a,b,[(0,0)])['steps'][0]['hands']['R']
        self.assertEqual(out['finger_angles_deg']['index']['delta'],[90,90,90])
        self.assertEqual(out['finger_angles_deg']['thumb']['delta'],[0,0,0])
        self.assertEqual(out['spread_angles_deg']['delta'],[0,0,0,0])

    def test_existing_path_is_retained_with_repeated_reference_frames(self):
        a,b=sample(),sample();path=[(0,0),(1,0),(2,1),(2,2)]
        before=copy.deepcopy(a)
        out=inspect_path(a,b,path)
        self.assertEqual([(s['i'],s['j']) for s in out['steps']],path)
        np.testing.assert_array_equal(a['features'],before['features'])
        self.assertEqual(a['feedback_body_context'],before['feedback_body_context'])
        json.dumps(out,allow_nan=False)

    def test_legacy_sample_without_context_does_not_invent_anchors(self):
        a,b=sample(),sample();del a['feedback_body_context']
        out=inspect_path(a,b,[(0,0)])
        self.assertEqual(out['inputs']['query']['body_points'],{})
        self.assertIsNone(out['inputs']['query']['normalization'])

    def test_feature_change_in_position_is_not_normalized_away(self):
        obs=observation();changed=copy.deepcopy(obs);changed.xy_px[:,54:75,1]+=30
        out=inspect_path(sample(changed),sample(obs),[(0,0)])
        delta=out['steps'][0]['hands']['R']['shoulder_position']['delta']
        np.testing.assert_allclose(delta,[0,.3],atol=1e-6)

    def test_default_path_carries_the_same_interval_review_as_the_old_opt_in(self):
        # 2026-09-26 observational_detail_v1: this test used to assert that the DEFAULT contract
        # produced no interval_review (it was opt-in only), which left the ordinary web page
        # without the selected-interval panel. The detail is now attached on every path. What
        # the old test protected still holds and is asserted here: the report reads the path
        # the trajectory already walked, the opt-in flag changes nothing, and no verdict appears.
        a,b=sample(),sample()
        default=location_path(a,b,{})
        opt=location_path(a,b,{'feedback':{'selected_interval_review':True}})
        self.assertEqual(json.dumps(default,sort_keys=True),json.dumps(opt,sort_keys=True))
        report=default['interval_review']
        self.assertEqual(report['status'],'AVAILABLE')
        self.assertEqual(report['detail_policy'],'observational_detail_v1')
        self.assertEqual([(s['i'],s['j']) for s in report['steps']],
                         [(s['i'],s['j']) for s in default['steps']])
        # no preset window unless a config names one; nothing is inferred
        self.assertEqual(report['review_window'],{'status':'NOT_CONFIGURED'})
        self.assertEqual(report['execution_assessment'],'UNCONFIGURED')
        self.assertIsNone(report['accepted'])
        json.dumps(report,allow_nan=False)

if __name__=='__main__': unittest.main()
