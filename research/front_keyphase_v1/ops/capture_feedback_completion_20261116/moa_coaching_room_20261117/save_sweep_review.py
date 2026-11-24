from pathlib import Path
import json,datetime,hashlib
R=Path.cwd();O=R/'research/front_keyphase_v1/ops';D=O/'capture_feedback_completion_20261116/moa_coaching_room_20261117';now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
capture=json.loads((D/'qa/SWEEP_CAPTURE.json').read_text(encoding='utf-8'));assert capture['passed'] and capture['physical_camera'] is False
summary='MOA가 앞으로 다가오는 동안 흰 배경을 유지하고, 상승 직전부터 화면 아래→위로 은은한 페이드가 번지는 별도 시안으로 수정. 기존 중심 확산/가는 연기 없음. 비행 후 위로 퇴장하고 화면 전환이 완료된 뒤 오른쪽에서 재등장. 전환 중 대화/메뉴 숨김. 실제 렌더 이미지/GIF 확인, 기존8821 UI·판정 미변경, 사용자 적용 승인 대기.'
next_action='상승 직전 아래→위 페이드 시안의 사용자 확인을 반영한다. UI 승인 전8821 교체 금지. 기존 공개 판정 후보의 admission/동결해시 및 녹화 보존 HTTP 검증을 마무리하고 통과분만 연결한다. 사용자촬영은 학습·튜닝·후보선택 금지.'
rec=dict(at=now,status='BOTTOM_UP_SWEEP_PREVIEW_AWAITING_USER_APPROVAL',summary=summary,storyboard=str(D/'qa/MOA_BOTTOM_UP_STORYBOARD.png'),animation=str(D/'qa/MOA_BOTTOM_UP_PREVIEW.gif'),capture=str(D/'qa/SWEEP_CAPTURE.json'),production_changed=False,camera_opened=False,training_runs=0,private_data_read=False,files={f:hashlib.sha256((D/'site'/f).read_bytes()).hexdigest()for f in ['scene.js','hologram.js']},next_action=next_action,check_scope='Actual scene rendered with controlled animation time and mocked camera API; no JS errors, no physical camera request. Existing 14-control browser check predates this effect-only revision.')
(D/'BOTTOM_UP_REVIEW.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
c=json.loads((D/'CHECKPOINT.json').read_text(encoding='utf-8'));old=c['ui_review'];c['ui_review']=rec;c['storyboard']=rec['storyboard'];c['animation']=rec['animation'];c['pending'][0]='User approval of upward fade beginning before ascent';(D/'CHECKPOINT.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
p=R/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8-sig');s=s.replace(old['summary'],summary,1).replace(old['next_action'],next_action,1).replace('### 2026-11-17 넓은 홀로그램 페이드 시안 수정·확인 대기','### 2026-11-17 상승 직전 아래→위 페이드 시안·확인 대기',1);s=s.replace(str(D/'HOLOGRAM_FADE_REVIEW.json'),str(D/'BOTTOM_UP_REVIEW.json'),2);p.write_text(s,encoding='utf-8')
s=json.loads((O/'STATE.json').read_text(encoding='utf-8-sig'));s['next_single_action']=next_action;k=s['capture_feedback_completion_20261116'];k['next_single_action']=next_action;k['updated_at']=now;k['moa_coaching_room_20261117']=c;(O/'STATE.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
p=R/'NEW_CHAT_HANDOFF.md';s=p.read_text(encoding='utf-8-sig');lines=s.splitlines();lines[0]='# 최신 인수인계 — '+now
for i,l in enumerate(lines):
 if l.startswith('사용자 최신 요청:'):lines[i]='사용자 최신 요청: '+summary
 if l.startswith('시안:'):lines[i]='시안: '+rec['storyboard']
 if l.startswith('움직임:'):lines[i]='움직임: '+rec['animation']
 if l.startswith('다음 한 작업:'):lines[i]='다음 한 작업: '+next_action
 if l.startswith('근거: D/HOLOGRAM_FADE'):lines[i]='근거: D/BOTTOM_UP_REVIEW.json, D/qa/SWEEP_CAPTURE.json. 기존14브라우저제어검사는 직전버전이며 이번은 전환효과만 수정·렌더확인. 이전 시안 보존.'
p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
for p in [D/'BOTTOM_UP_REVIEW.json',D/'CHECKPOINT.json',O/'STATE.json']:json.loads(p.read_text(encoding='utf-8'))
assert Path(rec['storyboard']).is_file() and Path(rec['animation']).is_file()
print('SWEEP_CHECKPOINT_OK; production unchanged')
