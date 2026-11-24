"""Adversarial source-span checks for the common progress component."""
import unittest
import numpy as np
from motion_evidence import movement_evidence,repetition_feedback

class EvidenceContractTests(unittest.TestCase):
    def test_progress_without_endpoint_is_retained_not_success(self):
        e=movement_evidence([1]*12,[0]*3+[1]*5+[0]*4,[],[True]*12)
        self.assertEqual(e['confirmed_arrival_count'],0)
        self.assertEqual(e['unconfirmed_progress_count'],1)
        self.assertFalse(repetition_feedback(e,2)['lesson_complete'])
    def test_boundary_predictions_without_real_window_are_not_evidence(self):
        e=movement_evidence([1]*6,[1]*6,[],[True]*6)
        self.assertEqual(e['movement_candidates'],[])
    def test_pose_hold_is_not_movement(self):
        self.assertEqual(movement_evidence([2]*15,[0]*15,[],[True]*15)['movement_candidates'],[])
    def test_missing_cannot_join_real_windows(self):
        mask=np.ones(16,dtype=bool);mask[8]=False
        e=movement_evidence([1]*16,[1]*16,[],mask)
        self.assertTrue(all(mask[a:b+1].all() for a,b in (r['source_span'] for r in e['movement_candidates'])))
    def test_background_is_not_bound_to_required_action(self):
        e=movement_evidence([0]*16,[1]*16,[],[True]*16)
        self.assertEqual(e['movement_candidates'],[])
    def test_no_word_completion_even_with_two_arrivals(self):
        e=movement_evidence([2]*20,[1]*20,[{'start':2,'end':8},{'start':11,'end':17}],[True]*20)
        self.assertFalse(repetition_feedback(e,2)['lesson_complete'])
        self.assertTrue(all(not c['description_bound'] for c in e['movement_candidates']))
    def test_invalid_arrival_rejected(self):
        with self.assertRaises(ValueError):movement_evidence([2]*10,[1]*10,[{'start':2,'end':20}],[True]*10)
    def test_missing_arrival_not_verified(self):
        mask=[True]*10;mask[6]=False
        e=movement_evidence([2]*10,[1]*10,[{'start':3,'end':6}],mask)
        self.assertEqual(e['confirmed_arrival_count'],0)
    def test_missing_required_count_cannot_complete(self):
        with self.assertRaises(ValueError):repetition_feedback({},0)
    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):movement_evidence([1]*10,[1]*9,[],[True]*10)

if __name__=='__main__':unittest.main()
