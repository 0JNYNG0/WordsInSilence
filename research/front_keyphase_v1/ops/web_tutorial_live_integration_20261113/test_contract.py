import unittest,copy
from contract import QuestProgress
def attempt(q,w,mode='quest',id='a'):return dict(id=id,mode=mode,target=w,generation=q.generation)
def result(w,source='user_inference_only',stop='arms_returned'):
 # Contract fixture only; matching TCN by itself is deliberately insufficient.
 return dict(source=source,stop_reason=stop,timestamp_us=[0,33000,66000,99000],tcn_top3=[dict(word=w,probability=.8)],lesson_acceptance=dict(target=w,schema='observed_lesson_requirements_v1',qualified=True,required_ids=['fixture_requirement'],criteria=[dict(id='fixture_requirement',status='supported',source_spans=[[1,2]])]))
class ProgressTests(unittest.TestCase):
 def test_valid_progress_exactly_once(self):
  q=QuestProgress()
  for i,w in enumerate(q.steps):
   a=attempt(q,w,id=str(i));self.assertTrue(q.consume(a,result(w))['advanced']);self.assertFalse(q.consume(a,result(w))['advanced'])
  self.assertEqual(q.completions,1);self.assertTrue(q.state()['completed'])
 def test_public_practice_wrong_and_timeout_never_advance(self):
  for mode,source,stop,word in [('practice','user_inference_only','manual',None),('quest','public_integration_replay','manual',None),('quest','user_inference_only','time_limit',None),('quest','user_inference_only','manual','OTHER')]:
   q=QuestProgress();w=q.steps[0];self.assertFalse(q.consume(attempt(q,w,mode),result(word or w,source,stop))['advanced']);self.assertEqual(q.index,0)
 def test_reset_and_stage_mismatch(self):
  q=QuestProgress();w=q.steps[0];a=attempt(q,w);q.reset();self.assertFalse(q.consume(a,result(w))['advanced'])
  self.assertFalse(q.consume(attempt(q,q.steps[1],id='b'),result(q.steps[1]))['advanced'])
 def test_invalid_empty_and_missing_coordinates(self):
  for change in [{'tcn_top3':[]},{'timestamp_us':[]},{'tcn_top3':[{'word':'WORD1567','probability':float('nan')}]},{'source':'synthetic'}]:
   q=QuestProgress();r=result(q.steps[0]);r.update(change);self.assertFalse(q.consume(attempt(q,q.steps[0]),r)['advanced'])
 def test_new_goals_not_silently_mapped(self):
  q=QuestProgress()
  for w in ['SEN0354','WORD0099','WORD1290','5']:self.assertNotIn(w,q.steps)
 def test_word_recognition_without_feedback_cannot_pass(self):
  q=QuestProgress();w=q.steps[0];r=result(w);r.pop('lesson_acceptance');self.assertFalse(q.consume(attempt(q,w),r)['advanced'])
if __name__=='__main__':unittest.main()
