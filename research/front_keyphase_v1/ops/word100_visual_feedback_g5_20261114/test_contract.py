"""Synthetic UI contract probes, NOT model performance scores or labels."""
from prepare import U,read,write
from adapter import project
from copy import deepcopy
base=read(U/'data/NIA_SL_WORD1133_REAL18_F__normal.json')['cards'][0]['source_card']
guides=[dict(id='shared_action',instruction='손을 펴고 팔을 따라 움직여 보세요.',hands=['R','L'])]
checks=[]
for feature,hand in [(1,'R'),(2,'L'),(3,'R'),(4,'L')]:
    s=deepcopy(base);s.update(card='shared_action',state='attributable_feature_difference',order_evidence=[],drafts=[],differences=[dict(task='shape',feature=feature,expected=1,state='difference',span=[2,5])])
    r=dict(joint_cards=[s],original_candidates=[dict(card='shared_action',chosen=dict(span=[1,6]))]);c=project(r,guides,8)[0]
    assert c['status']=='difference' and c['cues'][0]['hands']==[hand] and c['cues'][0]['spans']==[[2,5]]
    checks.append(f'feature{feature}:correct anatomical hand and exact frames')
    s['state']='evidence_conflict';c=project(r,guides,8)[0];assert c['status']=='uncertain' and not c['cues'];checks.append(f'feature{feature}:conflicting raw difference suppressed')
    s['state']='action_link_uncertain';c=project(r,guides,8)[0];assert c['status']=='uncertain' and not c['cues'];checks.append(f'feature{feature}:unlinked raw difference suppressed')
write(U/'CONTRACT_VERIFIED.json',dict(status='PASS',tests=checks,scope='Fabricated adapter inputs to test shared hand mapping and withholding. No claim of actual handshape-error accuracy. No training or data-label change.'))
print('12 display-contract checks passed; synthetic inputs only.')
