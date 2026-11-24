from pathlib import Path
import datetime,json,hashlib
R=Path.cwd();O=R/'research/front_keyphase_v1/ops';D=O/'capture_feedback_completion_20261116/moa_coaching_room_20261117';now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
check=json.loads((D/'qa/PLUME_CAPTURE.json').read_text(encoding='utf-8'));assert check['passed'] and not check['physical_camera']
c=json.loads((D/'CHECKPOINT.json').read_text(encoding='utf-8'));old=c['ui_review']
summary='사용자 요청에 따라 전체 화면 폭의 아래→위 페이드 대신, MOA 아랫부분에서 나와 비행 궤적을 따라 위로 끌려가는 중간 폭의 부드러운 효과로 수정. 이전 가는 연기보다 두껍고 직전 전체 페이드보다 좁음. 상승 직전에 시작하고 로봇이 퇴장한 뒤 넓어져 전환 마무리, 이후 오른쪽 재등장. 별도 시안 실제 렌더/카메라 모의 검증 통과. 기존8821 UI·판정 미변경, 적용 승인 대기.'
next_action='MOA에 붙어 올라가는 중간 폭 효과 시안의 사용자 확인을 반영한다. UI 승인 전8821 교체 금지. 기존 공개 판정 후보의 admission/동결해시 및 녹화 보존 HTTP 검증을 마무리하고 통과분만 연결한다. 사용자촬영은 학습·튜닝·후보선택 금지.'
rec=dict(at=now,status='ATTACHED_MEDIUM_PLUME_PREVIEW_AWAITING_USER_APPROVAL',summary=summary,next_action=next_action,storyboard=str(D/'qa/MOA_ATTACHED_PLUME_STORYBOARD.png'),animation=str(D/'qa/MOA_ATTACHED_PLUME_PREVIEW.gif'),capture=str(D/'qa/PLUME_CAPTURE.json'),production_changed=False,camera_opened=False,private_data_read=False,training_runs=0,files={f:hashlib.sha256((D/'site'/f).read_bytes()).hexdigest()for f in ['hologram.js','scene.js']},verification='Actual scene capture at controlled animation time; no JS errors or physical camera request. UI effect only; no model/regression rerun required.')
(D/'ATTACHED_PLUME_REVIEW.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
c['ui_review']=rec;c['storyboard']=rec['storyboard'];c['animation']=rec['animation'];c['pending'][0]='User approval of medium-width plume attached to MOA';(D/'CHECKPOINT.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
p=R/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf-8-sig');s=s.replace(old['summary'],summary,1).replace(old['next_action'],next_action,1).replace('### 2026-11-17 상승 직전 아래→위 페이드 시안·확인 대기','### 2026-11-17 MOA에 붙어 올라가는 중간 폭 효과 시안·확인 대기',1).replace(str(D/'BOTTOM_UP_REVIEW.json'),str(D/'ATTACHED_PLUME_REVIEW.json'),2);p.write_text(s,encoding='utf-8')
s=json.loads((O/'STATE.json').read_text(encoding='utf-8-sig'));s['next_single_action']=next_action;k=s['capture_feedback_completion_20261116'];k['next_single_action']=next_action;k['updated_at']=now;k['moa_coaching_room_20261117']=c;(O/'STATE.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
p=R/'NEW_CHAT_HANDOFF.md';ls=p.read_text(encoding='utf-8-sig').splitlines();ls[0]='# 최신 인수인계 — '+now
for i,l in enumerate(ls):
 if l.startswith('사용자 최신 요청:'):ls[i]='사용자 최신 요청: '+summary
 if l.startswith('시안:'):ls[i]='시안: '+rec['storyboard']
 if l.startswith('움직임:'):ls[i]='움직임: '+rec['animation']
 if l.startswith('다음 한 작업:'):ls[i]='다음 한 작업: '+next_action
 if l.startswith('근거: D/BOTTOM_UP'):ls[i]='근거: D/ATTACHED_PLUME_REVIEW.json, D/qa/PLUME_CAPTURE.json. 이번은 전환효과만 수정·실제 렌더 확인. 이전 시안 보존.'
p.write_text('\n'.join(ls)+'\n',encoding='utf-8')
print('ATTACHED_PLUME_RECORDED; original demo unchanged')
