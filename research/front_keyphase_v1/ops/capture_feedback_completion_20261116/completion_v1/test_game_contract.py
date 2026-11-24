from pathlib import Path
import sys,json,copy,unittest
V=Path(__file__).resolve().parent;O=V.parents[1];sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'))
import contract
from feedback_acceptance import feedback_gate
contract.SUPPORTED['WORD1290']='감사'
def public(key):return json.loads((V/'web'/f'{key}.json').read_text('utf-8'))
class BasicGameTests(unittest.TestCase):
 def test_actual_public_outputs(self):
  for key,expected in[('normal',True),('partial',False),('other_word',False)]:self.assertEqual(feedback_gate(public(key),'WORD1290')['accepted'],expected)
 def test_all_six_requirements_mandatory(self):
  for criterion in public('normal')['lesson_acceptance']['required_ids']:
   q=public('normal');q['lesson_acceptance']['required_ids'].remove(criterion);q['lesson_acceptance']['criteria']=[c for c in q['lesson_acceptance']['criteria']if c['id']!=criterion];self.assertFalse(feedback_gate(q,'WORD1290')['accepted'])
 def test_unknown_difference_and_unqualified_never_pass(self):
  for state in['unknown','difference']:
   for c in range(6):
    q=public('normal');q['lesson_acceptance']['criteria'][c]['status']=state;self.assertFalse(feedback_gate(q,'WORD1290')['accepted'])
  q=public('normal');q['lesson_acceptance']['qualified']=False;self.assertFalse(feedback_gate(q,'WORD1290')['accepted'])
 def test_positive_quest_unit_fixture_exactly_once(self):
  # Public model output copied into a controller fixture, never a webcam claim.
  g=contract.QuestProgress(['WORD1290']);q=public('normal');q.update(source='user_inference_only',stop_reason='arms_returned')
  a=dict(id='unitfixture',mode='quest',target='WORD1290',generation=g.generation)
  self.assertTrue(g.consume(a,q)['advanced']);self.assertFalse(g.consume(a,q)['advanced']);self.assertEqual(g.completions,1)
 def test_public_replay_cannot_advance(self):
  g=contract.QuestProgress(['WORD1290']);a=dict(id='public',mode='quest',target='WORD1290',generation=g.generation)
  self.assertFalse(g.consume(a,public('normal'))['advanced'])
 def test_outside_source_span_rejected(self):
  q=public('normal');q['lesson_acceptance']['criteria'][0]['source_spans']=[[0,len(q['timestamp_us'])]];self.assertFalse(feedback_gate(q,'WORD1290')['accepted'])
if __name__=='__main__':unittest.main()
