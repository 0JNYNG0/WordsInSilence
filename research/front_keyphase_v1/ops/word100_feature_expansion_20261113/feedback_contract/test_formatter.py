"""Synthetic contract tests, not public-video/model-performance measurements."""
import copy,json,unittest,hashlib
from pathlib import Path
from formatter import format_feedback
U=Path(__file__).resolve().parent
OBS=dict(word_id='TOY_WORD',stage_id='TOY_STAGE',hand='R',finger='index',state='F',native_interval_us=[1000000,1300000],attribute='finger_state',visibility='observed',evidence_type='validated_model',evidence_ids=['SYNTHETIC_NOT_REAL'],synthetic_example=True)
BIND=dict(binding_id='SYNTHETIC_BINDING',word_id='TOY_WORD',stage_id='TOY_STAGE',hand='R',finger='index',reference_state='E',action_phrase='손을 앞으로 옮기는',reference_anchors=[dict(kind='synthetic_fixture_not_rgb')],stage_interval_provenance=dict(kind='synthetic_fixture_not_source'),comparison_scope='sparse_reference_anchor_not_mandatory_word_rule',clause_ids=['SYNTHETIC_CLAUSE'])
VALID=dict(attribute_validated=True,validated_states=['E','B','F'],phase_binding_validated=True,reference_comparison_validated=True)
def o(**kw):return dict(OBS,**kw)
def b(**kw):return dict(BIND,**kw)
class ContractTests(unittest.TestCase):
    def test_no_validation_no_claim(self):self.assertEqual(format_feedback(OBS,BIND)['status'],'unknown')
    def test_observation_without_source(self):self.assertEqual(format_feedback(OBS,None,VALID)['status'],'observation')
    def test_difference_not_error(self):
        x=format_feedback(OBS,BIND,VALID);self.assertEqual(x['status'],'reference_difference');self.assertIsNone(x['educational_correctness']);self.assertTrue(x['synthetic_example']);self.assertNotIn('틀렸',x['text'])
    def test_matching_component_not_word(self):self.assertEqual(format_feedback(o(state='E'),BIND,VALID)['status'],'reference_match')
    def test_hidden_is_unknown(self):self.assertEqual(format_feedback(o(visibility='uncertain'),BIND,VALID)['status'],'unknown')
    def test_unknown_is_not_folded(self):self.assertEqual(format_feedback(o(state='U'),BIND,VALID)['status'],'unknown')
    def test_thumb_b_f_rejected(self):
        for state in ['B','F']:self.assertEqual(format_feedback(o(finger='thumb',state=state),b(finger='thumb'),VALID)['status'],'unknown')
    def test_temporal_guess_is_not_direct(self):self.assertEqual(format_feedback(o(evidence_type='temporal_inferred'),BIND,VALID)['status'],'unknown')
    def test_identity_guards(self):
        for kw in [{'word_id':'OTHER'},{'stage_id':'OTHER'},{'hand':'L'},{'finger':'ring'}]:self.assertEqual(format_feedback(OBS,b(**kw),VALID)['status'],'observation')
    def test_phase_unknown_cannot_compare(self):self.assertEqual(format_feedback(OBS,BIND,dict(VALID,phase_binding_validated=False))['status'],'observation')
    def test_provenance_missing_cannot_compare(self):self.assertEqual(format_feedback(OBS,b(reference_anchors=[]),VALID)['status'],'observation')
    def test_spread_and_contact_unsupported(self):
        for a in ['finger_spread','exact_contact']:self.assertEqual(format_feedback(o(attribute=a),BIND,VALID)['status'],'unknown')
    def test_bad_native_time(self):
        for t in [[2,1],[-1,2],[float('nan'),2],[True,2],None]:self.assertEqual(format_feedback(o(native_interval_us=t),BIND,VALID)['status'],'unknown')
    def test_both_hands_share_format(self):
        x=format_feedback(o(hand='L',finger='little'),b(hand='L',finger='little'),VALID);self.assertEqual(x['status'],'reference_difference');self.assertIn('왼손 새끼손가락이',x['text'])
    def test_softmax_not_a_correctness_rate(self):
        x=format_feedback(o(probability=.99),BIND,VALID);self.assertNotIn('%',x['text']);self.assertIsNone(x['educational_correctness'])
    def test_reference_timing_not_query_timing(self):
        x=format_feedback(OBS,BIND,VALID);self.assertEqual(x['evidence_interval_us'],OBS['native_interval_us'])
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests);result=unittest.TextTestRunner(verbosity=1).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),scope='synthetic input-contract behavior only; no inference accuracy or real feedback validation',formatter_sha256=hashlib.sha256((U/'formatter.py').read_bytes()).hexdigest())
    (U/'CONTRACT_TESTS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
    fixtures=[dict(label='가짜 입력: 참고와 다른 접힘',output=format_feedback(OBS,BIND,VALID)),dict(label='가짜 입력: 같은 펼침',output=format_feedback(o(state='E'),BIND,VALID)),dict(label='가짜 입력: 손가락 가림',output=format_feedback(o(visibility='uncertain'),BIND,VALID)),dict(label='가짜 입력: 왼손도 같은 기능',output=format_feedback(o(hand='L',finger='little'),b(hand='L',finger='little'),VALID)),dict(label='가짜 입력: 엄지 굽힘 미지원',output=format_feedback(o(finger='thumb',state='B'),b(finger='thumb'),VALID)),dict(label='가짜 입력: 구간 연결 미검증',output=format_feedback(OBS,BIND,dict(VALID,phase_binding_validated=False)))]
    (U/'SYNTHETIC_EXAMPLES.json').write_text(json.dumps(dict(kind='SYNTHETIC_ONLY_NOT_PUBLIC_OR_USER_RESULTS',examples=fixtures),ensure_ascii=False,indent=2)+'\n','utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
