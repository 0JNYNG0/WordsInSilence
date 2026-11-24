from pathlib import Path
import json,datetime,hashlib
R=Path.cwd();O=R/'research/front_keyphase_v1/ops';D=O/'capture_feedback_completion_20261116/moa_coaching_room_20261117';now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
ck=json.loads((D/'qa/MIDDLE_CAPTURE.json').read_text(encoding='utf-8'));assert ck['passed'] and not ck['physical_camera']
c=json.loads((D/'CHECKPOINT.json').read_text(encoding='utf-8'));old=c['ui_review']
summary='사용자가 매끈한 직선 기둥 시안을 반려하고 이전 연기 질감과 세모/ㅗ 볼륨을 요청. 정지 A/B 비교 후 사용자 직접 A와 B 중간 형태 선택. 이에 따라 MOA와 연결된 두툼한 위쪽에서 아래로 갈수록 넓어지는 둥근 연기 볼륨을 비행에 적용한 별도 시안 생성. 상승 직전 시작/퇴장 뒤 확산/전환 후 오른쪽 복귀/대화 숨김 유지. 실제 렌더 GIF·이미지 확인, JS오류/실제카메라 호출0. 기존8821 미변경; 형태선택 승인됨, 실제 데모 반영 전 시안 확인 대기.'
next_action='사용자가 선택한 세모형과 ㅗ형 중간 연기 애니메이션의 최종 확인을 받으면 기존 UI에 반영한다. 그동안 남은 공개 판정 후보 admission/동결해시 및 녹화 보존 HTTP 검증을 마무리하고 통과분만 연결한다. 기존8821/C·T3-F 유지, 사용자촬영 학습·튜닝·선택 금지.'
rec=dict(at=now,status='VOLUMETRIC_MIDDLE_FORM_SELECTED_ANIMATED_PREVIEW_PENDING',summary=summary,next_action=next_action,form_choice='A와 B의 중간 형태',form_choice_source='Explicit user reply to call_cf86245c31ac4fc592c14fc3442a6048',production_changed=False,training_runs=0,private_data_read=False,storyboard=str(D/'qa/MOA_VOLUMETRIC_MIDDLE_STORYBOARD.png'),animation=str(D/'qa/MOA_VOLUMETRIC_MIDDLE_PREVIEW.gif'),shape_options=str(D/'qa/MOA_SMOKE_SHAPE_OPTIONS.png'),capture=str(D/'qa/MIDDLE_CAPTURE.json'),files={f:hashlib.sha256((D/'site'/f).read_bytes()).hexdigest()for f in ['scene.js','hologram.js']})
(D/'VOLUMETRIC_MIDDLE_REVIEW.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
c['ui_review']=rec;c['storyboard']=rec['storyboard'];c['animation']=rec['animation'];c['pending'][0]='Confirm animated application of user-selected middle smoke form before production UI replacement';(D/'CHECKPOINT.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
p=R/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8-sig');s=s.replace(old['summary'],summary,1).replace(old['next_action'],next_action,1).replace('### 2026-11-17 MOA에 붙어 올라가는 중간 폭 효과 시안·확인 대기','### 2026-11-17 사용자 선택 세모/ㅗ 중간 연기 형태·비행 시안',1).replace(str(D/'ATTACHED_PLUME_REVIEW.json'),str(D/'VOLUMETRIC_MIDDLE_REVIEW.json'),2);p.write_text(s,encoding='utf-8')
s=json.loads((O/'STATE.json').read_text(encoding='utf-8-sig'));s['next_single_action']=next_action;k=s['capture_feedback_completion_20261116'];k['next_single_action']=next_action;k['updated_at']=now;k['moa_coaching_room_20261117']=c;(O/'STATE.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
p=R/'NEW_CHAT_HANDOFF.md';ls=p.read_text(encoding='utf-8-sig').splitlines();ls[0]='# 최신 인수인계 — '+now
for i,l in enumerate(ls):
 if l.startswith('사용자 최신 요청:'):ls[i]='사용자 최신 요청: '+summary
 if l.startswith('시안:'):ls[i]='시안: '+rec['storyboard']
 if l.startswith('움직임:'):ls[i]='움직임: '+rec['animation']
 if l.startswith('다음 한 작업:'):ls[i]='다음 한 작업: '+next_action
 if l.startswith('근거: D/ATTACHED_PLUME'):ls[i]='근거: D/VOLUMETRIC_MIDDLE_REVIEW.json, D/qa/MIDDLE_CAPTURE.json. 형태 선택은 사용자 명시 승인, 동작 시안 확인 후 데모 반영. 모델/카메라/학습 변경 없음.'
p.write_text('\n'.join(ls)+'\n',encoding='utf-8')
for p in [D/'VOLUMETRIC_MIDDLE_REVIEW.json',O/'STATE.json']:json.loads(p.read_text(encoding='utf-8'))
print('USER_SELECTED_MIDDLE_FORM_RECORDED; production unchanged')
