from pathlib import Path
import json,hashlib,collections
from goal_contract import summarize_component
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
 with (U/name).open('x',encoding='utf-8')as f:json.dump(value,f,ensure_ascii=False,indent=2)
cards_path=O/'word100_action_feedback_demo_20261108/CARDS.json';cards=read(cards_path)['words']
bindings=[]
definitions=[('WORD0739','close_chest','forearm',2,'NOT_A_SUFFICIENT_GOAL_PROXY','가슴 쪽 손 위치와 손 닫힘이 필요하며 전완 가로 방향만으로 대신할 수 없음.'),
 ('WORD0739','raise_forearm','forearm',3,'HOLD_FOR_STATE_DEFINITION_REVIEW','비스듬한 팔 자세에서 가로/세로 라벨 경계와 관측의 불일치. 손 닫힘도 별도.'),
 ('WORD1133','brush_arm','arm_region',4,'COMPONENT_EVIDENCE_ONLY','상대 팔에서 손목 쪽으로 이동한 관계만 학습. 손 펴기/경로/접촉과 교육적 허용 범위는 별도.'),
 ('WORD1567','alternating','height',5,'HOLD_FOR_CROSSING_UNCERTAINTY','양손 높이가 바뀜은 회전 궤도나 필요한 반복 횟수 전체가 아님. 교차 시 보류1개.'),
 ('WORD2523','spread_hands','separation',7,'COMPONENT_EVIDENCE_ONLY','양손 벌어짐만 관측. 손끝 오므림·접촉·표현 변이 판정과 전체완료 아님.')]
for word,cid,domain,family,status,note in definitions:
    card=next((c for c in cards[word]if c['id']==cid),None)
    if card is None:
        # Fail visibly rather than invent a guide card ID.
        raise ValueError((word,cid,[c['id']for c in cards[word]]))
    bindings.append(dict(word_id=word,card_id=cid,family=family,domain=domain,instruction=card['instruction'],
        status=status,limitations_ko=note,source=str(cards_path),source_sha256=sha(cards_path),
        instruction_is_existing_project_card=True,whole_guide_learned=False))
dense=read(U/'DENSE_REPLAY.json');hints=[]
for r in dense['rows']:
    h=summarize_component(r['result'],r['available'],r['states'],r['domain'])
    assert h['semantic_completion']is None and h['educational_error']is None and not h['display_enabled']
    hints.append(dict(group_id=r['group_id'],variant=r['variant'],split=r['split'],hint=h))
write('GUIDE_STATE_BINDINGS.json',dict(rows=bindings,not_arbitrary_word_exceptions=True,
    explanation='The models take only anatomical relations. Word-to-guide expected-component lookup is declarative; unlearned clause requirements stay explicit.',
    common100_maintained=True,common100_retrained=False,web_changed=False))
write('COACHING_CONTRACT.json',dict(rows=hints,checks=len(hints),display_enabled=False,
    primitive_supported_domains=['arm_region','separation'],overall_completion_validated=False))
# Preserve discrepancies instead of relabelling the probe to fit the model.
ev=read(U/'EVALUATION.json');issues=[p for p in ev['predictions']if p['kind']=='mlp'and p['split']=='exposed_probe'and p['truth']is not None and p['decision']!=p['truth']]
write('LIMITATION_AUDIT.json',dict(issues=issues,probe_labels_changed=False,retraining_after_probe=False,
    forearm='The phrase wrist_above_elbow and approximately across can overlap for oblique arms. Direct RGB class boundary is not an independent educational goal. Three disagreements on REAL08 need adjudication; do not call them three proven learner/model errors.',
    sparse_scope='186 reviewed domain/hand observations on 120 review slots (deduplicated by frame/task/hand); 164 assigned,22 uncertain. Not framewise dense annotation of 32 videos.',
    partial_scope='Start-only and goal-only are controlled edited posture tests; true intermediate learner stops and acceptable 70% sweeps are not labelled educational errors.',
    dense_scope='Dense source replay expands input coverage, not ground truth coverage; source-edited controls are not independent people.',
    missing_scope='Unknown when required coordinates absent or classification uncertain; distinct reasons. No imputed goal.',
    paired_start_end_limit='Two endpoint states alone do not prove the continuous path, finger shape, contact or repetition. Goal channel cannot set overall completion.',
    promotion='Two component domains pass the fixed visible probe, but new person/manual-error and automatic-span validation are insufficient. No web/default replacement.'))
print('Guide bindings and non-decisional feedback wording verified',len(hints))
