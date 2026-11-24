import unittest
import json
import numpy as np
from candidate_motion_evidence import candidate_motion_evidence as measure
from candidate_motion_evidence import attach_candidate_motion_evidence as attach


def data(values, times=None, mask=None):
    x=np.asarray(values, float)
    t=np.arange(len(x),dtype=np.int64)*33333 if times is None else np.asarray(times)
    m=np.ones(len(x),bool) if mask is None else np.asarray(mask,bool)
    return measure(x,m,t)


class MotionEvidenceContract(unittest.TestCase):
    def test_jump_and_distributed_motion_are_measured_not_classified(self):
        jump=data([0,0,0,1,1,1]); motion=data(np.linspace(0,1,6))
        j,m=jump['dimensions'][0],motion['dimensions'][0]
        self.assertEqual(j['endpoint_difference'],m['endpoint_difference'])
        self.assertEqual(j['largest_step_fraction_of_variation'],1)
        self.assertAlmostEqual(m['largest_step_fraction_of_variation'],.2)
        self.assertEqual(j['effective_change_steps'],1)
        self.assertAlmostEqual(m['effective_change_steps'],5)
        for r in [jump,motion]:
            self.assertIsNone(r['observation_discontinuity'])
            self.assertIsNone(r['physical_motion_complete'])

    def test_short_real_movement_is_never_dropped(self):
        r=data([0,.5,1],[0,4000,8000])
        self.assertEqual(r['samples'],3)
        self.assertEqual(r['dimensions'][0]['exact_nonzero_neighbor_changes'],2)
        self.assertEqual(r['duration_us'],8000)
        self.assertIsNone(r['guide_requirement_satisfied'])

    def test_two_samples_remain_ambiguous_even_when_motion_is_real(self):
        r=data([0,1],[0,7000])
        self.assertEqual(r['dimensions'][0]['sampled_change_pattern'],'one_observed_change')
        self.assertIsNone(r['observation_discontinuity'])

    def test_occlusion_does_not_become_intermediate_motion(self):
        r=data([0,0,999,1,1],mask=[1,1,0,1,1]); d=r['dimensions'][0]
        self.assertEqual(d['observed_neighbor_pairs'],[[0,1],[3,4]])
        self.assertEqual(d['connected_absolute_variation'],0)
        self.assertEqual(d['endpoint_difference'],1)
        self.assertEqual(d['observation_state'],'gapped_samples')
        self.assertFalse(d['endpoint_difference_is_path_evidence'])

    def test_time_scaling_preserves_path_and_scales_velocity(self):
        a=data([0,.3,1],[0,10000,30000]); b=data([0,.3,1],[0,20000,60000])
        da,db=a['dimensions'][0],b['dimensions'][0]
        self.assertEqual(da['connected_absolute_variation'],db['connected_absolute_variation'])
        self.assertEqual(da['effective_change_steps'],db['effective_change_steps'])
        np.testing.assert_allclose(da['observed_neighbor_velocity_per_second'],np.array(db['observed_neighbor_velocity_per_second'])*2)

    def test_large_time_gap_is_exposed_not_silently_normalized(self):
        r=data([0,.5,1],[0,33333,600000]); d=r['dimensions'][0]
        self.assertEqual(r['sample_dt_us'],[33333.,566667.])
        self.assertLess(d['observed_neighbor_velocity_per_second'][1],d['observed_neighbor_velocity_per_second'][0])
        self.assertFalse(r['intersample_path_observed'])

    def test_same_endpoints_can_contain_repeated_movement(self):
        d=data([0,.5,1,.5,0])['dimensions'][0]
        self.assertEqual(d['endpoint_difference'],0)
        self.assertEqual(d['connected_absolute_variation'],2)
        self.assertEqual(d['positive_variation'],1)
        self.assertEqual(d['negative_variation'],1)

    def test_side_mirroring_changes_sign_not_support(self):
        a=data([0,.3,.9])['dimensions'][0]; b=data([0,-.3,-.9])['dimensions'][0]
        self.assertEqual(a['connected_absolute_variation'],b['connected_absolute_variation'])
        self.assertEqual(a['observed_neighbor_pairs'],b['observed_neighbor_pairs'])
        self.assertEqual(a['endpoint_difference'],-b['endpoint_difference'])

    def test_dimensions_and_missingness_stay_separate(self):
        r=measure([[0,0],[.5,np.nan],[1,1]],np.ones((3,2),bool),[0,10,20])
        self.assertEqual(r['dimensions'][0]['neighbor_pair_count'],2)
        self.assertEqual(r['dimensions'][1]['neighbor_pair_count'],0)
        self.assertEqual(r['dimensions'][1]['claimed_observed_nonfinite'],1)
        json.dumps(r,allow_nan=False)

    def test_every_candidate_preserved_without_mutation(self):
        candidates=[dict(span=[0,0],eligible=False),dict(span=[1,2],score=.1),dict(a=0,b=3)]
        x=np.arange(4.); original=x.copy(); result=attach(candidates,x,np.ones(4,bool),np.arange(4)*1000)
        self.assertEqual(len(result),3)
        self.assertFalse(result[0]['eligible'])
        self.assertEqual(result[0]['motion_observation_evidence']['samples'],1)
        self.assertEqual(result[1]['span'],[1,2])
        self.assertNotIn('motion_observation_evidence',candidates[0])
        np.testing.assert_array_equal(x,original)

    def test_all_missing_and_single_samples_stay_unknown(self):
        for x,m,state in [([1,2],[0,0],'none'),([1],[1],'one_sample')]:
            r=data(x,mask=m)
            self.assertEqual(r['dimensions'][0]['observation_state'],state)
            self.assertIsNone(r['physical_motion_complete'])

    def test_invalid_contract_is_rejected(self):
        with self.assertRaises(ValueError): measure([0,1],[.8,.9],[0,1])
        with self.assertRaises(ValueError): data([0,1],[1,1])
        with self.assertRaises(ValueError): data([0,1],np.array([2,1],dtype=np.uint64))
        with self.assertRaises(ValueError): measure([0,1],np.ones(2,bool),[0,1],1,0)
        with self.assertRaises(ValueError): attach([dict(span=[.5,1])],[0,1],np.ones(2,bool),[0,1])


if __name__=='__main__':unittest.main(verbosity=2)
