"""Focused tests of misleading-success, scope and identity failure modes."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from compose_feedback import attach, digest


def fixture(order='OBSERVED_A_BEFORE_B', high=True, gaps=False):
    axis=[100,101,102];ts=[900000,950000,1000000]
    evidence=dict(input_frames=axis,native_seconds=[.9,.95,1.])
    base=dict(word_id='synthetic',source_sha256='a'*64,frames=3,
        stages=[dict(guide_id='step')],guide_feature_comparison=dict(items=[]),
        partial_order_feedback=dict(items=[dict(first={'guide_ids':['step']},second={'guide_ids':['after']},
            observation_status=order,first_evidence=evidence,second_evidence=evidence)]))
    obs=dict(guide_id='step',target_index=0,frames=[100,102],
        fixed_scope=dict(frame_ids=axis,timestamp_us=ts),high_confirmation=high,
        unavailable_native_frames=[101] if gaps else [],whole_action_complete=None,educational_error=None)
    local=dict(word_id='synthetic',native_frames=3,observations=[obs],image_unavailable_frames=[101] if gaps else [])
    identity=dict(word_id='synthetic',video_sha256='a'*64,native_frame_ids=axis,timestamp_us=ts,
        motion_scope_family='test_fixed_scope',base_value_sha256=digest(base),local_value_sha256=digest(local))
    return base,local,identity


class Contract(unittest.TestCase):
    def test_local_high_cannot_override_reverse(self):
        b,l,i=fixture('OBSERVED_B_BEFORE_A');r=attach(b,l,i);p=r['combined_evidence_feedback']
        self.assertIn('RELATED_PAIR_OBSERVED_IN_REVERSE_ORDER',p['motion_events'][0]['flags'])
        self.assertIsNone(p['whole_sign_correctness']);self.assertIsNone(p['normality'])
        self.assertEqual({k:v for k,v in r.items() if k!='combined_evidence_feedback'},b)

    def test_gaps_remain_with_high_probability(self):
        b,l,i=fixture(gaps=True);p=attach(b,l,i)['combined_evidence_feedback']
        self.assertEqual(p['native_denominator'],3)
        self.assertEqual(p['global_image_unavailable_frames'],[101])
        self.assertIn('OBSERVATION_GAP_RETAINED',p['motion_events'][0]['flags'])

    def test_high_and_forward_is_not_completion(self):
        b,l,i=fixture();p=attach(b,l,i)['combined_evidence_feedback']
        self.assertIsNone(p['motion_events'][0]['whole_action_complete'])
        self.assertFalse(p['all_requirements_verified'])

    def test_low_probability_is_not_absence(self):
        b,l,i=fixture(high=False);p=attach(b,l,i)['combined_evidence_feedback']
        self.assertIsNone(p['motion_events'][0]['missing_action'])
        self.assertFalse(p['correction_enabled'])

    def test_interleaving_is_not_relabelled_reverse(self):
        b,l,i=fixture('PAIR_INTERLEAVED_EVIDENCE');e=attach(b,l,i)['combined_evidence_feedback']['motion_events'][0]
        self.assertIn('RELATED_PAIR_ORDER_UNRESOLVED',e['flags'])
        self.assertNotIn('RELATED_PAIR_OBSERVED_IN_REVERSE_ORDER',e['flags'])

    def test_no_order_query_does_not_invent_repetition(self):
        b,l,i=fixture();b['partial_order_feedback']['items']=[];i['base_value_sha256']=digest(b)
        e=attach(b,l,i)['combined_evidence_feedback']['motion_events'][0]
        self.assertIn('NO_DISTINCT_PAIR_QUERY',e['flags'])

    def test_identity_and_original_time_rejected(self):
        for mutation in ['word','sha','time','scope','content']:
            b,l,i=fixture()
            if mutation=='word':i['word_id']='different'
            if mutation=='sha':i['video_sha256']='b'*64
            if mutation=='time':i['timestamp_us'][1]+=1
            if mutation=='scope':
                l['observations'][0]['fixed_scope']['frame_ids']=[100,102]
                i['local_value_sha256']=digest(l)
            if mutation=='content':l['observations'][0]['high_confirmation']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):attach(b,l,i)

    def test_does_not_mutate_and_rejects_duplicate(self):
        b,l,i=fixture();saved=deepcopy((b,l,i));out=attach(b,l,i)
        self.assertEqual((b,l,i),saved)
        with self.assertRaises(ValueError):attach(out,l,i)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    p=Path(__file__).with_name('COMPOSER_CHECKS.json')
    with p.open('x',encoding='utf-8') as f:json.dump(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),model_calls=0),f,indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
