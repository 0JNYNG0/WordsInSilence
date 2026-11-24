"""Semantic tests for observed movement, not implementation-mirroring score assertions."""
from pathlib import Path
import copy, sys, unittest
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).parent)]
from wis_lab.core import features as F, movement_feedback as M, recognize as R
from test_r22_feedback_role import observation

def signals(n=81):
    vec=np.zeros((n,F.FEATURE_WIDTH),np.float32)
    mask=np.ones((n,len(F.GROUP_ORDER)),bool)
    times=np.arange(n,dtype=np.int64)*33333
    slices={}; k=0
    for g in F.GROUP_ORDER:
        slices[g]=slice(k,k+F.GROUPS[g][0]); k+=F.GROUPS[g][0]
    return vec,mask,times,slices

class MovementEvidence(unittest.TestCase):
    def test_wrist_visible_shape_unseen_is_not_zero_shape_change(self):
        x,p,t,sl=signals()
        p[:,F.GROUP_ORDER.index('shape_R')]=False
        h=M.describe(M.movement_source(x,p,t))['hands']['R']
        self.assertEqual(h['wrist_speed']['mean'],0)
        self.assertEqual(h['wrist_speed']['observed_steps'],80)
        self.assertIsNone(h['shape_change']['mean'])
        self.assertEqual(h['shape_change']['unknown_steps'],80)
        self.assertEqual(h['vertical_turns'],0)

    def test_observed_unchanged_shape_is_a_real_zero(self):
        x,p,t,sl=signals()
        h=M.describe(M.movement_source(x,p,t))['hands']['R']
        self.assertEqual(h['shape_change']['mean'],0)
        self.assertEqual(h['shape_change']['observed_steps'],80)

    def test_a_missing_finger_breaks_both_adjacent_shape_differences_only(self):
        x,p,t,sl=signals()
        p[40,F.GROUP_ORDER.index('shape_R')]=False
        h=M.describe(M.movement_source(x,p,t))['hands']['R']
        self.assertEqual(h['shape_change']['unknown_steps'],2)
        self.assertEqual(h['shape_change']['observed_steps'],78)
        self.assertEqual(h['wrist_speed']['observed_steps'],80)

    def test_every_missing_wrist_breaks_a_run_even_across_a_short_gap(self):
        x,p,t,sl=signals(17)
        p[1::2,F.GROUP_ORDER.index('handpos_R')]=False
        h=M.describe(M.movement_source(x,p,t))['hands']['R']
        self.assertIsNone(h['vertical_turns'])
        self.assertEqual(h['completeness'],'UNCOUNTABLE')
        self.assertEqual(h['observed_seconds'],0)

    def test_same_resolved_path_at_different_speeds_has_same_turns(self):
        x,p,t,sl=signals()
        path=np.interp(np.arange(81),[0,20,40,60,80],[0,1,0,1,0])
        x[:,sl['handpos_R'].start+1]=path
        for factor in (1,4):
            h=M.describe(M.movement_source(x,p,t*factor))['hands']['R']
            self.assertEqual(h['vertical_turns'],3)
            self.assertEqual(h['completeness'],'COMPLETE')

    def test_nonfinite_values_do_not_create_observed_change(self):
        x,p,t,sl=signals()
        x[:,sl['shape_R'].start]=np.nan
        x[:,sl['handpos_R']]=np.nan
        h=M.describe(M.movement_source(x,p,t))['hands']['R']
        self.assertIsNone(h['wrist_speed']['mean'])
        self.assertIsNone(h['shape_change']['mean'])
        self.assertIsNone(h['horizontal_turns'])

    def test_real_timestamp_gap_is_not_a_continuous_step(self):
        x,p,t,sl=signals()
        t[40:]+=500000
        h=M.describe(M.movement_source(x,p,t))['hands']['R']
        self.assertEqual(h['runs'],2)
        self.assertEqual(h['wrist_speed']['unknown_steps'],1)
        self.assertEqual(h['shape_change']['unknown_steps'],1)
        self.assertEqual(h['completeness'],'PARTIAL')

    def test_source_survives_sampling_and_reference_build_without_resampling(self):
        obs=observation(n=20)
        seq=F.sequence_features(obs)
        q=F.uniform_time_sample(seq,8)
        ref=R.build_reference(obs,'c','SYNTHETIC',8)
        self.assertLess(len(q['features']),20)
        self.assertEqual(len(q['movement_observation']['timestamp_us']),20)
        self.assertEqual(q['movement_observation'],ref['movement_observation'])

    def test_legacy_input_without_original_timeline_remains_unavailable(self):
        report=M.compare_sources(None,None)
        self.assertEqual(report['status'],'UNAVAILABLE')
        self.assertIsNone(report['accepted'])

    def test_arrays_unchanged_and_observed_output_is_json_finite(self):
        import json
        x,p,t,sl=signals()
        before=copy.deepcopy((x,p,t))
        source=M.movement_source(x,p,t)
        json.dumps(M.compare_sources(source,source),allow_nan=False)
        for actual,old in zip((x,p,t),before):
            np.testing.assert_array_equal(actual,old)

if __name__=='__main__': unittest.main()
