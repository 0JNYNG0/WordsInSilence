"""Record this UI integration, without rerunning training or frozen evaluations."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import ast, hashlib, json, shutil, urllib.request

U = Path(__file__).resolve().parent
ROOT = U.parents[3]
O = U.parent
now = datetime.now(timezone(timedelta(hours=9))).isoformat()
def read(p): return json.loads(p.read_text('utf-8-sig'))
def write(p, data): p.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

protected = read(U/'PROTECTED.json')
hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in protected.items()}
assert all(hashes.values())
for p in (U/'server.py', U/'contract.py', U/'check_http.py', U/'check_public_capture.py'):
    ast.parse(p.read_text('utf-8'))
http = read(U/'HTTP_VERIFIED.json')
capture = read(U/'PUBLIC_CAPTURE_VERIFIED.json')
assert http['passed'] and capture['passed']
meta = json.load(urllib.request.urlopen('http://127.0.0.1:8821/api/meta'))
req = urllib.request.Request('http://127.0.0.1:8821/api/status', headers={'X-Session-Token':meta['token']})
status = json.load(urllib.request.urlopen(req))
assert status['state']=='ready' and not status['camera_thread_active']
assert meta['quest']['index']==0
assert [x['label'] for x in meta['planned_words']]==['안녕하세요','사과','감사']
server = read(U/'SERVER.json')

browser_checks = {
    'practice_3d_and_camera_controls': True,
    'camera_not_opened_by_agent': True,
    'wrong_avatar_gesture_disabled_for_other_words': True,
    'selected_public_reference_video_loaded': True,
    'public_result_disclosed_and_not_quest_success': True,
    'two_videos_skeleton_and_timeline_visible': True,
    'unmatched_query_shown_as_unknown_not_invented_match': True,
    'practice_result_cleared_when_entering_quest': True,
    'new_three_targets_separately_marked_as_preparation': True,
    'desktop_and_narrow_layout_checked': True,
    'temporary_viewport_override_reset': True,
    'console_error_or_warning_seen': False,
}
write(U/'BROWSER_VERIFIED.json', {'at':now,'actor':'same root Codex via CUA; not independent human validation','checks':browser_checks,
    'screenshots':['PRACTICE_READY.png','FEEDBACK_PUBLIC.png','FEEDBACK_UNCERTAIN.png','QUEST_READY.png','NARROW_READY.png'],
    'limitations':['Physical webcam test is pending user action.','Existing model feedback accuracy was not improved or newly validated by this UI task.']})
write(U/'VERIFIED.json', {
    'at':now,'status':'INTEGRATION_VERIFIED_PUBLIC_INPUT_PHYSICAL_WEBCAM_PENDING',
    'implementation_actor':'root Codex','verification_actor':'same Codex separate code/API/public replay/CUA checks; not independent KSL annotation',
    'contract_tests':{'passed':5,'source':'test_contract.py; run during implementation'},
    'http_checks':len(http['checks']),'http_passed':http['passed'],
    'public_capture_passed':capture['passed'],'public_capture_stop_reason':capture['stop_reason'],
    'public_capture_frames':capture['frames'],'public_capture_second_detection':False,
    'javascript_syntax_files_passed':['scene.js','live.js','feedback.js'],
    'protected_hashes':hashes,'final_service_state':status['state'],'camera_thread_active':status['camera_thread_active'],
    'quest_index_after_public_checks':meta['quest']['index'],'server':server,
    'new_training_runs':0,'private_recordings_read':0,'physical_camera_opened':False,
    'new_model_downloads':0,'baseline_model_replaced':False,
    'first_errors_and_repairs':[
        'Final read-only localhost status query from restricted environment failed with WinError10013 before any documentation write; closeout was rerun in the normal local environment.',
        'Generated scene.js initially retained a dangling if after removing simulated quest events; fixed generator event-line removal and passed final Node syntax checks.',
        'UI inspection found unmatched feedback could show an unrelated query frame; added explicit unknown placeholder and reference-only mode.',
        'Desktop robot/control overlap and narrow dialogue/button overlap corrected; viewport reset after verification.'
    ],
    'remaining':['New three-word recognition/feedback preparation is not training completion.','Avatar gesture currently prepared only for public 조상 trial.','G2 semantic/segment evidence research remains incomplete.','Physical webcam latency and end-to-end interaction need the user trial.']
})

report = '''웹 가상 연습실 실제 촬영 연결 — 2026-11-13

완료한 범위
http://127.0.0.1:8821/ 에 기존 3D 연출과 실제 촬영/분석 경로를 연결했다.
연습: 단어 선택 → 카메라 켜기 → 3초 준비 → 촬영 중 RTMW → 손을 내리면 자동 종료 → 이미 뽑은 관절로 TCN·시간 합성곱+GRU·DTW 결과.
관측 차이는 해당 순간의 손·팔과 짧은 안내로 표시한다. 확인 부족은 회색이며 오답으로 바꾸지 않는다. 대응 구간이 없으면 엉뚱한 사용자 프레임을 붙이지 않고 확인 부족 안내를 보여준다.
실전: 실제 촬영의 TCN 1순위가 현재 목표 단어와 같을 때만 다음 원/선이 채워진다. 이는 단어 인식 성공이며 수어 전체 자세가 정확하다는 판정은 아니다. 공개 예제/중복 응답/이전 시도/취소/시간초과는 성공을 만들지 않는다.

현재 지원과 추가 목표
바로 연결한 범위: 조상·이집트·순종·잘하다. 기존 상세 피드백 모델의 지원 범위를 그대로 사용했다.
사용자 확정 확장 목표: AIHub 안녕하세요(SEN0354) → 과일 사과(WORD0099) → 감사(WORD1290). 지숫자5 제외. 기존4단어 삭제 없음.
- 안녕하세요: 기존 C: 공개 예시 SEN0354_REAL01_F 한 개 확인. 여러 사람의 동일 표현 RGB 학습자료 확보는 아직 필요. 국립국어원 인사와 임의 통합 금지.
- 사과: 공개 자료 준비 이력은 있으나 기존104클래스 후보는 기존100단어 개발 회귀가 있어 미채택. ID05 표현 변이는 보류. 이번에 미검증 후보를 웹 기본으로 올리지 않았다.
- 감사: 기존 TCN100에는 포함. 세부 피드백 가이드/시점 정답과 검증은 아직 필요.
세 단어를 준비 대상으로 기록했으며 새 학습을 완료했다고 주장하지 않는다.
캐릭터의 동작은 조상 공개5뷰 적용 시험만 준비됐다. 다른 단어를 선택하면 잘못된 캐릭터 동작 버튼을 막고 해당 공개 참고 영상을 제공한다. 교육용 아바타 완성 판정은 아니다.

검증
서버 진행 계약 5검사, HTTP/API 42검사 통과. 공개 관절 기록4개를 실제 모델/API로 처리해 TCN·피드백·DTW 반환과 퀘스트 미진행을 확인했다. 이는 연결 검사이며 새 정확도 평가가 아니다.
공개 조상 영상을 카메라 대신 입력한 실제 촬영 루프 시험 통과: 3초 준비, 135프레임 녹화/관측 수 일치, arms_returned 자동 종료, 두 번째 관절 재검출 없이 TCN/피드백 반환. 실제 카메라 장치를 열지 않았다.
CUA로 연습/실전/참고 영상/두 영상 비교/확인 부족 표시/넓고 좁은 화면을 확인했다. 최종 JS3개 구문 및 기존 보호8파일 해시 통과. 신규 서버는 준비 완료·카메라 꺼짐 상태다.
작성과 검증 모두 root Codex. 별도 사람 또는 수어 전문가의 독립 정답 검증이 아니다.

한계와 다음
이번 결과는 연결 완료다. 모델의 손모양/방향/순서 판별 성능 향상이나 실제 사용자 촬영의 자동종료·대기시간 통과를 뜻하지 않는다. 실제 웹캠 시험은 사용자가 버튼을 눌러 진행한다.
새3단어는 공개 자료의 의미/사람 분리/설명과 시점 정답을 먼저 확정하고, 별도 후보 학습 및 기존100단어 회귀 검증을 거친 뒤 연결한다. 감사 세부 정답과 사과 변이 검토는 기존 자료로 준비할 수 있고, 안녕하세요는 다인 원본 확보 조건부터 확인한다.
피드백 본선 word100_feedback_recovery_20261113의 G2 재개점 보존. 이미 마친6+18학습 재실행 없음.
기존 C·T3-F, 8814 코드/모델, 8820 연출 미리보기 보존. 사용자 촬영 학습·튜닝·선택 없음. 신규 촬영은 이 PC의 임시 폴더에 저장해 결과 재생에 쓰며 외부 전송하지 않는다.

근거
PLAN_KO.txt, contract.py/test_contract.py, HTTP_VERIFIED.json, PUBLIC_CAPTURE_VERIFIED.json, BROWSER_VERIFIED.json, VERIFIED.json.
새3단어 사실: ../demo_scope_portability_review_20261113/RESULT_KO.txt, ../demo_handshape_learning_20261113/RESULT_KO.txt.
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf-8')

next_action = ('8821 실제 촬영 연결은 공개 입력/API/화면 검증 완료, 실제 카메라는 사용자가 시험한다. '
    '확장 목표는 기존4단어 유지+AIHub 안녕하세요(SEN0354)/과일사과(WORD0099)/감사(WORD1290), 숫자5제외. '
    '안녕하세요 다인 원본 확보 조건과 사과 표현변이/감사 세부 시점정답을 준비한 뒤 후보학습·기존100회귀검증; 미지원 성공처리 금지. '
    '기존 G2 연구 재개점과 완료6+18학습 보존, C·T3-F/사용자자료 보호 유지.')
summary = ('3D 가상 연습실8821에 기존 RTMW 촬영중 관절추출→재사용 TCN/관절 시간합성곱+GRU/DTW 피드백 연결. '
    '연습은 양쪽 영상·손팔 강조·짧은 안내·불확실 분리, 실전은 서버가 실제 촬영 top1 목표일치로만 진행. '
    '지원4단어 유지, AIHub 안녕하세요/사과/감사 추가준비·5제외. 계약5/API42/공개촬영루프135프레임 자동종료/브라우저 검증 통과. '
    '물리웹캠 미시험, 새학습/성능향상/교육용완료 아님. 기존 모델과8820보존.')

backup=U/'before_closeout';backup.mkdir(exist_ok=True)
paths=[ROOT/'NEW_CHAT_HANDOFF.md',ROOT/'WordsInSilence_PROJECT_CONTEXT.md',O/'STATE.json']
for p in paths:
    dest=backup/p.name
    if not dest.exists():shutil.copy2(p,dest)

state=read(O/'STATE.json');stage=state['current_stage']
state['web_tutorial_live_integration_20261113']={
    'status':'PUBLIC_INTEGRATION_VERIFIED_USER_WEBCAM_PENDING','updated_at':now,'summary':summary,
    'server':{**server,'execution':'normal local process; verified listener','physical_camera_opened_by_agent':False},
    'result':str(U/'RESULT_KO.txt'),'verification':str(U/'VERIFIED.json'),
    'supported_feedback_words':['WORD1567','WORD2523','WORD0739','WORD1133'],
    'planned_words':meta['planned_words'],'new_training_runs':0,'model_replaced':False,
    'next':next_action,'author_and_verifier':'same root Codex; not independent KSL expert'
}
state['next_single_action']=next_action
assert state['current_stage']==stage
write(O/'STATE.json',state)

context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text('utf-8-sig')
title='### 2026-11-13 웹 가상 연습실 실제 촬영 연결 · AIHub 안녕하세요 확정'
assert title not in text, 'Already closed out; do not prepend duplicates'
evidence='근거: `research/front_keyphase_v1/ops/web_tutorial_live_integration_20261113/RESULT_KO.txt`, `VERIFIED.json`.'
for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
    assert marker in text
    text=text.replace(marker,marker+'\n\n'+title+'\n'+body+'\n'+evidence+'\n',1)
context.write_text(text,encoding='utf-8')

handoff=f'''# 최신 인수인계 — 2026-11-13 웹 실제 촬영 연결 완료·새3단어 준비

{summary}

현재 단위: {U}
실제 테스트: http://127.0.0.1:8821/ (PID {server['pid']}, 일반 실행환경, 카메라 버튼 전 장치 미개방)
기존 8820은 연출 미리보기로 보존. 8814 코드는 읽기전용 재사용했으며 서버/모델 원본 변경 없음.
새 단위 server.py 실행 경로는 word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. 시작 전에 실제8821 PID/포트 확인, 중복 실행 금지.

구현: server.py/contract.py/site(scene.js,live.js,feedback.js,index.html,live.css). 연습 단어 선택·3초 카운트다운·자세복귀 종료·분석 로딩·TCN상위3·DTW 재생·관측근거 안내. 실전은 실제촬영 top1 일치만 진행, 공개시험/중복/이전시도/취소/시간초과 진행금지. 카메라와 관절처리는 로컬이다. 새 촬영은 임시폴더 결과재생용이며 학습 금지.

검증: contract5, API42, 공개 촬영루프135프레임 자동종료·재검출0, 실제 CUA 넓은/좁은화면·비교·불확실 표시 확인. root Codex 작성 및 별도검사이며 독립사람/전문가검증 아님. 실제 사용자 웹캠 환경과 교육적 교정 정확도는 미검증. 새학습 없음.
근거: RESULT_KO.txt, HTTP_VERIFIED.json, PUBLIC_CAPTURE_VERIFIED.json, BROWSER_VERIFIED.json, VERIFIED.json. 스크린샷 PRACTICE_READY.png, FEEDBACK_PUBLIC.png, FEEDBACK_UNCERTAIN.png, QUEST_READY.png, NARROW_READY.png.

사용자 확정: 기존4 유지+안녕하세요/사과/감사 추가 목표, 숫자5 제외. 인사는 AIHub 안녕하세요 SEN0354 선택. NIKL인사와 임의병합 금지. 안녕하세요 기존공개예시1개/다인RGB미확보, 사과 기존104후보회귀미채택·ID05변이보류, 감사TCN100포함·상세피드백미지원. 준비 완료/추가학습완료로 과장하지 않는다. 캐릭터는 조상 적용시험만 준비; 다른단어 동작 버튼 차단하고 공개참고영상 제공.

다음 한 작업: {next_action}
본선 current_stage=word100_web_completion_graph_20261111 유지. word100_feedback_recovery_20261113/NEXT_PLAN_KO.txt G2보완→G3실제통합→G4평가 순서와 완료6+18학습 재실행금지 유지. 기존C·T3-F/운영/원본 보존, 사용자·멘토자료 학습·튜닝·선택 금지. 아바타 얼굴관계/접촉/큰회전도 미해결.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
assert read(O/'STATE.json')['current_stage']==stage
for p in [U/'RESULT_KO.txt',U/'VERIFIED.json',U/'BROWSER_VERIFIED.json',*paths]: assert p.is_file()
print('Closeout verified; current_stage retained; no training/evaluation rerun.')
