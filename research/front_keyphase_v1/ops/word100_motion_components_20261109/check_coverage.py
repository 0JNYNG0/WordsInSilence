from common import *
from component_coverage import apply_coverage
import copy

def main():
 n=0
 for word in ['ANY_A','ANY_B','unseen_word_123']:
  for hand in ['L','R']:
   component='fingertip_convergence:'+hand
   req=dict(component=component,label='손끝 모양',scope='late_core',source_applicability='verified')
   base=dict(word=word,cards=[dict(id='arbitrary',status='supported',display_status='similar',message='기존 특징 확인',checked_components=['손 이동'],coverage_requirements=[req])])
   out=apply_coverage(base);c=out['cards'][0];assert c['display_status']=='uncertain' and c['checked_components']==['손 이동'] and not c['educational_wrong'];assert base['cards'][0]['status']=='supported';n+=2
   obs=dict(component=component,scope='late_core',observation_sufficient=True,validated=True,state='supported')
   b=copy.deepcopy(base);b['cards'][0]['component_observations']={component:obs};assert apply_coverage(b)['cards'][0]['display_status']=='similar';n+=1
   for key,val in [('scope','initial'),('component','hand_separation'),('validated',False),('observation_sufficient',False)]:
    bb=copy.deepcopy(b);bb['cards'][0]['component_observations'][component][key]=val;assert apply_coverage(bb)['cards'][0]['display_status']=='uncertain';n+=1
   bb=copy.deepcopy(b);bb['cards'][0]['coverage_requirements'][0]['source_applicability']='unresolved_variant';assert apply_coverage(bb)['cards'][0]['display_status']=='uncertain';n+=1
   bb=copy.deepcopy(b);bb['cards'][0]['component_observations'][component]['state']='contradicted';assert apply_coverage(bb)['cards'][0]['display_status']=='difference';n+=1
   bb=copy.deepcopy(base);bb['cards'][0].update(status='difference_candidate',display_status='difference',difference_components=['movement']);c=apply_coverage(bb)['cards'][0];assert c['display_status']=='difference' and c['difference_components']==['movement'];n+=1
 plain=dict(cards=[dict(id='unchanged',status='supported')]);assert apply_coverage(plain)['cards']==plain['cards'];n+=1
 write(U/'COVERAGE_CHECK.json',dict(passed=True,checks=n,arbitrary_word_ids=True,left_and_right=True,missing_vs_wrong_separate=True,no_camera=True));print(n)
if __name__=='__main__':main()
