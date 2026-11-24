from pathlib import Path
import copy,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from wis_lab.core.phase_feedback import bind_review_window

class ReferenceWindow(unittest.TestCase):
    def setUp(self):
        self.frames={'sampled_feature_sha256':'abc','frame_ids':[2,4,7,9,11,14]}
        self.d={'clip_id':'CLIP','word_id':'WORD','sampled_feature_sha256':'abc',
                'frame_id_range':[5,12],'status':'VISUALLY_INSPECTED_CENTRAL_WINDOW_NOT_EXPERT_ANNOTATION'}
    def bind(self,d=None,frames=None):
        return bind_review_window(self.d if d is None else d,self.frames if frames is None else frames,'CLIP','WORD')
    def test_window_maps_original_frame_ids_to_samples(self):
        out=self.bind();self.assertEqual(out['status'],'AVAILABLE')
        self.assertEqual(out['reference_sample_range'],[2,4]);self.assertEqual(out['actual_sampled_frame_ids'],[7,9,11])
        self.assertFalse(out['is_a_verdict'])
    def test_another_clip_never_supplies_the_window(self):
        d=dict(self.d,clip_id='OTHER');self.assertEqual(self.bind(d)['reason'],'reference_identity_mismatch')
    def test_another_word_never_supplies_the_window(self):
        d=dict(self.d,word_id='OTHER');self.assertEqual(self.bind(d)['reason'],'reference_identity_mismatch')
    def test_changed_features_do_not_silently_use_old_annotation(self):
        d=dict(self.d,sampled_feature_sha256='xyz');self.assertEqual(self.bind(d)['reason'],'reference_features_changed')
    def test_bad_ranges_do_not_fall_back_to_full_span(self):
        for bounds in ([12,5],[5,5],[-1,7],[False,7],[4.5,7],[1,9],[5,20],[5,6]):
            self.assertEqual(self.bind(dict(self.d,frame_id_range=bounds))['status'],'UNAVAILABLE')
    def test_no_profile_does_not_claim_a_reviewed_interval(self):
        self.assertEqual(bind_review_window(None,self.frames,'CLIP','WORD'),{'status':'NOT_CONFIGURED'})
    def test_annotation_status_cannot_be_relabelled_as_an_expert_pass(self):
        self.assertEqual(self.bind(dict(self.d,status='EXPERT_APPROVED'))['reason'],'unsupported_annotation_status')
    def test_source_annotation_does_not_mutate(self):
        before=copy.deepcopy(self.d);out=self.bind();out['authored_frame_id_range'][0]=10
        self.assertEqual(self.d,before)

if __name__=='__main__':unittest.main()
