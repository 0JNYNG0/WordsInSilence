import unittest
from labels import assign_extent,direct_candidates,validate_aux_admission

def annotation():
    return dict(clip_id='PUBLIC_FIT',axis=0,moving='R',source_sha256='source',
        events=[dict(onset=[2,3],offset=[8,9],direction=2),dict(onset=[13,14],offset=[18,19],direction=0)],
        holds=[[9,12]],background=[[0,1],[20,24]])

class LabelContract(unittest.TestCase):
    def test_full_and_partial_are_different(self):
        a=annotation()
        self.assertEqual(assign_extent(3,8,0,'R',[a])[0],2)
        self.assertEqual(assign_extent(4,7,0,'R',[a])[0],1)

    def test_hold_not_missingness(self):
        self.assertEqual(assign_extent(10,12,0,'R',[annotation()])[0],0)
        self.assertEqual(assign_extent(10,12,0,'L',[annotation()])[0],-1)

    def test_two_distinct_events_have_multi_label(self):
        label,ids,_=assign_extent(2,19,0,'R',[annotation()])
        self.assertEqual(label,3);self.assertEqual(len(set(ids)),2)

    def test_one_event_plus_terminal_hold_is_not_multi_gold(self):
        self.assertEqual(assign_extent(2,12,0,'R',[annotation()])[0],-1)

    def test_overlap_conflict_is_not_silently_hold(self):
        a=annotation();a['events'][0]['offset']=[8,10]
        self.assertEqual(assign_extent(9,10,0,'R',[a])[2],'annotation_conflict')

    def test_single_observation_does_not_prove_temporal_extent(self):
        a=annotation();a['events']=[dict(onset=[2,4],offset=[4,6],direction=2)]
        self.assertEqual(assign_extent(4,4,0,'R',[a]),(-1,[],'single_observation_no_temporal_extent'))

    def test_two_and_three_sample_motion_are_not_removed_for_length(self):
        a=annotation();a['events']=[dict(onset=[2,3],offset=[3,4],direction=2)]
        self.assertEqual(assign_extent(2,3,0,'R',[a])[0],2)
        self.assertEqual(assign_extent(2,4,0,'R',[a])[0],2)

    def test_unknown_extent_is_excluded(self):
        self.assertEqual(assign_extent(5,16,0,'R',[annotation()])[0],-1)

    def test_direct_proposals_do_not_invent_unannotated_side(self):
        c=direct_candidates([annotation()])
        self.assertTrue(c);self.assertTrue(all(r['moving']=='R' for r in c))
        self.assertIn('derived_within_event_partial',{r['source'] for r in c})

    def test_witness_is_not_automatic_aux_admission(self):
        ad=dict(clip_id='PUBLIC_FIT',axis=0,moving='R',event_index=0,left_source_frame=4,right_source_frame=7,
            observed_removed_frames=[5,6],rgb_evidence=dict(path='public',sha256='proof'),source_sha256='source',reviewed_mid_motion=True)
        with self.assertRaises(ValueError):validate_aux_admission(ad,[annotation()],25)
        ad['admission']='approved_source_edit_aux_only'
        _,_,left,right=validate_aux_admission(ad,[annotation()],25)
        self.assertEqual((left,right),(4,7))
        ad['left_source_frame']=2
        with self.assertRaises(ValueError):validate_aux_admission(ad,[annotation()],25)

if __name__=='__main__':unittest.main(verbosity=2)
