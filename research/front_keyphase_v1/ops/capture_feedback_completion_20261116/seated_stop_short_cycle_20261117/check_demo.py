from check_runtime import *
import copy,hashlib
from contract import QuestProgress
def main():
 words=['WORD0099','WORD1290'];quest=QuestProgress(steps=words);rows=[]
 for i,word in enumerate(words):
  result=read(D/'web'/f'normal_NIA_SL_{word}_REAL01_F.json');result.update(source='user_inference_only',stop_reason='arms_returned')
  result['tcn_top3'][0].update(word=word,probability=.9)
  for status in ['difference','unknown']:
   bad=copy.deepcopy(result);bad['lesson_acceptance']['criteria'][0]['status']=status;bad['lesson_acceptance']['overall_pass']=False;bad['overall_pass']=False
   attempt=dict(id=f'{word}_{status}',mode='quest',generation=quest.generation,target=word)
   q=quest.consume(attempt,bad);rows.append(dict(word=word,case=status,passed=not q['advanced'] and quest.index==i))
  attempt=dict(id=word+'_normal',mode='quest',generation=quest.generation,target=word)
  q=quest.consume(attempt,result);rows.append(dict(word=word,case='normal_after_retry',passed=q['advanced']and quest.index==i+1))
  rows.append(dict(word=word,case='duplicate',passed=not quest.consume(attempt,result)['advanced']))
 rows.append(dict(case='quest_completed_once',passed=quest.completions==1 and quest.index==2))
 assets=[]
 roots=[O/'web_tutorial_live_integration_20261113/site/sign-demos',O/'web_tutorial_stage_preview_20261113/site/assets']
 for root in roots:
  for p in root.glob('*.glb'):assets.append(dict(path=str(p),exists=p.is_file(),on_c=p.drive.upper()=='C:'))
 write(D/'checks/DEMO_CHECK.json',dict(passed=all(r['passed']for r in rows)and all(x['exists']and x['on_c']for x in assets),rows=rows,assets=assets,synthetic_application_contract=True,physical_camera=False,physical_hdd_disconnect=False,scope='Frozen public results injected into isolated quest state; no user/live quest result fabricated'))
 print('DEMO',len(rows),all(r['passed']for r in rows))
if __name__=='__main__':main()
