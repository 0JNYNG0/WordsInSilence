"""Record verified capture-only maintenance; do not resume paused learning."""
from pathlib import Path
import json,hashlib,datetime
A=Path(__file__).resolve().parent;U=A.parent;O=U.parent;R=U.parents[3]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
old=read(A/'before/EXECUTION_FROZEN.json')['pins'];new=read(U/'EXECUTION_FROZEN.json')['pins']
changed=[p for p,h in old.items() if sha(p)!=h]
assert set(changed)=={str((U/n).resolve()) for n in ['logic.js','demo.js','demo.html','test_logic.js']},changed
assert all(sha(p)==h for p,h in new.items())
cap=read(A/'CAPTURE_VERIFIED.json');ui=read(A/'UI_LOGIC_VERIFIED.json');http=read(A/'HTTP_VERIFIED.json')
assert cap['status']==ui['status']=='PASS' and all(http[k] is True for k in ['ready','logic.js','demo.js','demo.html'])
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
summary=('8812 자동 종료를 기존 웹의 손 관측 후 양손 미검출 0.75초 방식으로 복원했다. '
'이전 통합화면은 낮은 손/손목 증거를 요구했고 높은 추정 손목으로 종료를 거부했으며 650ms 초과 관측간격마다 대기를 초기화했다. '
'한손 빠른 이탈·양손 빠른 이탈·추정 손목 고잔류·800ms 느린 관측·짧은 손출현 5조건에서 이전 실패를 합성 재현하고 현재 종료를 확인했다. '
'촬영 프레임 시각으로 시간을 재며 오류/오래된 응답/2초 초과 관측공백은 종료 근거로 누적하지 않는다. '
'처음부터 안 보이는 손/한손이라도 남아 있는 상태/0.75초 미만 짧은 미검출은 종료하지 않는다. '
'이는 촬영 조작 규칙이므로 중간에 양손을 0.75초 이상 못 찾으면 조기 종료될 수 있다. '
'단어별 예외·학습·사용자 자료 열람 없이 공통 처리만 수정했다. '
'포착논리26+표시14+실제 JS 가상 UI43 검사 통과, 서버 ready 및 3개 배포파일 exact 확인. 실제 웹캠 성공은 아직 사용자 재시험 필요. '
'기존580핀 중 UI/검사4파일만 변경, 나머지576 및 C/T3F·SignRep·DTW 보존. 서버 재시작 없음. AO 연구 일시중단 유지.')
alignment=('DTW는 학습에도 사용됐다. train.py assemble은 공개 FIT01~09의 저장된 DTW 대응을 canonical_index로 읽고 원래 시간별 관측을 유지한 채 정렬손실로 학습 가능한 64차원 단어별 prototype을 함께 최적화한다. '
'단계 라벨은 ID01 직접 구간과 다른 FIT의 DTW 약한 연결을 구분(학습가중치1/.2)한다. '
'공통 시간축의 길이·원점과 설명 구간은 ID01에서 시작하므로 완전히 사람 독립적인 평균 시간축이라고 주장할 수 없다. '
'실행 engine.compare는 새 영상 latent를 학습된 prototype에 DTW1회 연결한다. 개인 참고 영상9회 비교가 아니다. '
'prepare_display는 ID01을 같은 canonical index의 실제 참고 재생시간으로 표시한다. '
'따라서 참고화면은 ID01이지만 평가 특징은 FIT9명의 영향을 받은 학습 모델이다. '
'시간정렬=오류의 의미/정오 검증 완료가 아니며, 학습 구간의 약한 연결과 단계평균 비교의 한계는 그대로다.')
nxt=('같은 http://127.0.0.1:8812/ 를 새로고침하여 3초 후 촬영→한손/양손 수행→양손 화면 아래→자동 종료를 사용자 추론 전용으로 재확인한다. '
'이전 촬영을 열람하거나 학습·튜닝하지 않는다. 새 학습/실험/AO 연구는 사용자 재개 전까지 중단 유지.')
report=dict(status='COMPLETE_CAPTURE_AUTOSTOP_V3',at=now,summary=summary,alignment_facts=alignment,next=nxt,
 changed_files=changed,preserved_pin_count=len(old)-len(changed),capture_tests=cap,ui_tests=ui,http=http,
 remaining_limits=['Actual physical webcam not opened or tested this turn.','A sustained hand-detector dropout can resemble hands-down, as in old capture UX.','Canonical indexing originates from public ID01; not an independently learned neutral time grid.'],
 failures_preserved=['before/ and BEFORE_REPRODUCTION.json retain original implementation/five failures.',
 'Initial read-only HTTP checks in sandbox failed due socket policy; require_escalated GET-only checks then passed. No server restart.',
 'Initial broad source search returned missing server.py path and an unrelated denied historical directory; narrowed to named current sources.'],
 implementation='Codex',verification='Same Codex separate synthetic capture/UI tests and read-only HTTP checks; not independent human accuracy ground truth.',
 new_training=0,new_feedback_inference=0,camera_opened=0,user_material_access=0)
write(A/'RESULT.json',report)
(A/'RESULT_KO.txt').write_text(summary+'\n\n학습·DTW 확인\n'+alignment+'\n\n다음\n'+nxt+'\n\n근거\n'+
 '기존 종료: src/wis_lab/service/camera.py END_QUIET_S 및 _record_frame\n'+
 '학습: word100_learned_guide_feedback_20261102/train.py assemble/run\n'+
 '실행: 같은 단위 engine.py compare, word100_learned_feature_retention_20261102/retention_engine.py\n'+
 '참고 재생: word100_integrated_test_20261104/prepare_display.py, logic.js mapping\n',encoding='utf-8')
state_path=O/'STATE.json';state=read(state_path)
assert state['current_stage']=='PAUSED_BY_USER_AFTER_MOTION_ORDER_INPUT_PREPARATION'
state['word100_integrated_test_20261104']['capture_autostop_v3']={k:report[k] for k in ['status','at','summary','alignment_facts','next']}
state['next_single_action']=nxt;write(state_path,state)
p=R/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf-8')
title='### 2026-11-04 자동 종료 복원과 학습 시간축 확인'
assert title not in text
for section,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary+'\n\n'+alignment),('## 9. 다음 단일 작업',nxt)]:
 assert section in text
 text=text.replace(section,section+'\n\n'+title+'\n\n'+body,1)
p.write_text(text,encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n'+alignment+'\n\n'+nxt+
 '\n\n유지보수 단위: word100_integrated_test_20261104 / capture_autostop_v3. before/에 수정 전 파일과 동결 보존. '+
 '앞선 ui_capture_v2의 웹캠 기본화·결과 후 관절 표시·분석 로딩 유지. 8812 GET ready=true/busy=false; 원래 서버를 재시작하지 않았다. '+
 '검증은 같은 Codex의 별도 모의 검사이며 독립 사람 정답 검증이 아니다.\n\n'+
 'AO word100_motion_aware_order_20261103 일시중단 유지. FIT72(36원본 ID01~09), DEV76(12원본 ID11~13), exact 재사용28, pending AH120(native Y snapshot48 포함), '+
 '별도최종입력검토·실제 preflight·새학습/평가 미완료. 사용자/멘토 자료 학습·튜닝·선택 금지. 원본/과거 동결 읽기 전용.\n',encoding='utf-8')
assert read(state_path)['current_stage']=='PAUSED_BY_USER_AFTER_MOTION_ORDER_INPUT_PREPARATION'
print('Capture v3 complete; 576 other pins preserved; records synchronized; research remains paused')
