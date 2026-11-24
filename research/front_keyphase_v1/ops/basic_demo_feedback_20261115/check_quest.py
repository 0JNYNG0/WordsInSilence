"""Application tests and saved public replay, not a new model-accuracy score."""
from prepare import *
import sys,copy
sys.path.insert(0,str(O/'web_tutorial_live_integration_20261113'))
from contract import QuestProgress

def fixture():
 q=QuestProgress();w=q.steps[0];a=dict(id='attempt',mode='quest',generation=q.generation,target=w)
 r=dict(source='user_inference_only',stop_reason='rest',timestamp_us=list(range(20)),tcn_top3=[dict(word=w,probability=.9)],cards=[],lesson_acceptance=dict(schema='observed_lesson_requirements_v1',target=w,qualified=True,required_ids=['shape','motion'],criteria=[dict(id=k,status='supported',source_spans=[[3,15]])for k in ['shape','motion']]))
 return q,a,r

def main():
 checks=[]
 def case(name,edit,expected):
  q,a,r=fixture();edit(q,a,r);before=q.index;ans=q.consume(a,r);ok=ans['advanced']==expected and q.index==before+int(expected);assert ok,(name,ans)
  checks.append(dict(name=name,passed=ok,reason=ans['reason']))
 case('all explicit qualified requirements',lambda q,a,r:None,True)
 case('TCN alone cannot advance',lambda q,a,r:r.pop('lesson_acceptance'),False)
 case('shape difference overrides recognition',lambda q,a,r:r.update(cards=[{'display_status':'difference'}]),False)
 case('qualified underlying difference overrides empty visible cards',lambda q,a,r:r.update(qualified_evidence={'joint_cards':[{'state':'attributable_feature_difference'}]}),False)
 case('unqualified candidate cannot advance',lambda q,a,r:r['lesson_acceptance'].update(qualified=False),False)
 case('unknown observation cannot advance',lambda q,a,r:r['lesson_acceptance']['criteria'][0].update(status='unknown'),False)
 case('partial cannot advance',lambda q,a,r:r['lesson_acceptance']['criteria'][1].update(status='partial'),False)
 case('different target cannot advance',lambda q,a,r:r['lesson_acceptance'].update(target='OTHER'),False)
 case('empty requirements cannot advance',lambda q,a,r:r['lesson_acceptance'].update(required_ids=[],criteria=[]),False)
 case('duplicate requirements cannot advance',lambda q,a,r:r['lesson_acceptance'].update(required_ids=['shape','shape']),False)
 case('out of range evidence cannot advance',lambda q,a,r:r['lesson_acceptance']['criteria'][0].update(source_spans=[[3,21]]),False)
 case('missing evidence cannot advance',lambda q,a,r:r['lesson_acceptance']['criteria'][0].update(source_spans=[]),False)
 case('public replay cannot advance',lambda q,a,r:r.update(source='public_replay'),False)
 case('time limit cannot advance',lambda q,a,r:r.update(stop_reason='time_limit'),False)
 case('wrong recognition cannot advance',lambda q,a,r:r.update(tcn_top3=[dict(word='OTHER',probability=.9)]),False)
 q,a,r=fixture();assert q.consume(a,r)['advanced'];assert q.consume(a,r)['reason']=='duplicate';checks.append(dict(name='duplicate accepted attempt',passed=True))
 # Existing disclosed conditions, no re-inference and no user recordings.
 audits=read(O/'word100_game_acceptance_20261115/PUBLIC_ACCEPTANCE_AUDIT.json')['rows'];rr=[]
 for item in audits:
  q,a,r=fixture();word=item['clip_id'].split('_')[2];q.steps=[word];a['target']=word
  saved=read(item['source_result'])['after'];r.update(tcn_top3=item['tcn_top3'],qualified_evidence=saved);r.pop('lesson_acceptance')
  ans=q.consume(a,r);assert not ans['advanced'];rr.append(dict(clip_id=item['clip_id'],kind=item['kind'],old_name_only_advance=item['recognition_matches'],qualified_correction=item['lesson']['status']=='correct',new_reason=ans['reason'],advanced=False))
 write(U/'QUEST_VERIFIED.json',dict(passed=True,contract_cases=checks,public_cases=rr,
  prevented_name_only_with_correction=sum(r['old_name_only_advance']and r['qualified_correction']for r in rr),
  actual_positive_completion_examples=0,positive_unit_fixture_only=True,
  limitation='This fixes false progression. The new vocabulary completion producer still requires public qualification; blocking all unknowns is NOT a completed game or accuracy improvement.',
  changed_files={str(O/'web_tutorial_live_integration_20261113'/n):sha(O/'web_tutorial_live_integration_20261113'/n)for n in ['contract.py','feedback_acceptance.py','site/live.js']}))
 print('QUEST CONTRACT',len(checks),'PUBLIC REPLAY',len(rr),'contradictory progression prevented',sum(r['old_name_only_advance']and r['qualified_correction']for r in rr))
if __name__=='__main__':main()
