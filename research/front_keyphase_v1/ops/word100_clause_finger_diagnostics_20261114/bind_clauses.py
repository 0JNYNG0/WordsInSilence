from pathlib import Path
import json
from clause_feedback import compose
U=Path(__file__).resolve().parent
read=lambda n:json.loads((U/n).read_text('utf-8-sig'))
write=lambda n,v:(U/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
actions=read('CLAUSE_CONTRACT.json')['actions']
phrases={
 'open_front':[('shape','오른손을 펴서'),('body_region','얼굴 앞에 세워')],
 'close_chest':[('shape_change','손을 닫으면서'),('movement','가슴 쪽으로 내려')],
 'raise_forearm':[('shape_maintenance','닫힌 손을 유지하며'),('movement','팔뚝을 세워')],
 'brush_arm':[('shape','오른손을 펴고'),('body_relation','왼팔을 따라'),('movement','손목 쪽으로 쓸어')],
 'alternating':[('bilateral_sequence','오른손과 왼손을 번갈아'),('movement','위로 올려')],
 'gather_hands':[('fingertip_relation','양손의 손끝을'),('movement','가운데로 모아')],
 'spread_hands':[('movement','두 손을 양옆으로 벌려')]
}
for a in actions:
 text=a['display_instruction'];a['clauses']=[]
 for typ,phrase in phrases[a['action_id']]:
  i=text.index(phrase);a['clauses'].append(dict(kind=typ,text=phrase,char_span=[i,i+len(phrase)],action_id=a['action_id'],same_action=True,source='existing project-reviewed display instruction; not official verbatim quote'))
 a['clause_order_is_not_required_action_order']=True
 a['expected_fingers_from_reviewed_descriptor_not_from_sentence_parser']=True
write('CLAUSE_LINKS.json',dict(actions=actions,all_clauses_hand_structured=True,automatic_observed_interval=True,
 warning='Co-occurring phrase fragments are not separate temporal actions. A dictionary context phrase alone is not an RGB observation label.',
 supported_description_actions=len(actions),supported_unrestricted_sentences=False))
# Component contracts, not learner accuracy measurements.
a=next(a for a in actions if a['action_id']=='brush_arm')
finger=dict(hand='R',finger='index',expected='extended',span=[10,20],state='difference',probability=.02)
direction=dict(span=[10,20],state='opposite_direction')
both=compose(a,[finger],[direction],True);assert {x['kind']for x in both['cues']}=={'finger','direction'}
for reason in ['uncertain','observation_unknown']:
 r=compose(a,[{**finger,'state':reason}],[direction],True);assert all(x['kind']!='finger'for x in r['cues'])and len(r['unknown'])==1
assert not compose(a,[finger],[direction],False)['cues']
assert not compose(a,[{**finger,'finger':'thumb'}],[],True)['cues']
left={**a,'finger_requirements':[dict(hand='L',finger='index',expected='extended')]};assert compose(left,[{**finger,'hand':'L'}],[],True)['cues'][0]['hand']=='L'
for a in actions:
 for c in a['clauses']:assert a['display_instruction'][slice(*c['char_span'])]==c['text']
write('CLAUSE_CONTRACT_CHECKS.json',dict(pass_all=True,software_contract_tests=7,source_clause_offsets_checked=sum(len(a['clauses'])for a in actions),model_accuracy_test=False,web_changed=False))
print('clause actions',len(actions),'contract checks passed')
