"""Finalize this bounded usability repair from its actual public evidence."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil

R = Path(__file__).resolve().parents[5]
O = R / 'research/front_keyphase_v1/ops'
U = O / 'hello_live_coach_20261115'
V = Path(__file__).resolve().parent
assert R.name == 'WordsInSilence', R
stamp = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

admitted = read(V / 'ADMITTED.json')
for rel, expected in admitted['code_hashes'].items():
    assert hashlib.sha256((U / rel).read_bytes()).hexdigest() == expected, rel
checks = {}
for name in ['PUBLIC_COMPARISON_V2.json', 'INDEPENDENT_CHECKS_V2.json',
             'INDEPENDENT_HOLD_LEDGER_V2.json', 'LEDGER_CHECK.json', 'SERVICE_FIXTURE_CHECK.json']:
    data = read(V / name)
    assert data['passed'], name
    checks[name] = {key:data[key] for key in ['passed', 'passes', 'total'] if key in data}
web = read(V / 'WEB_SERVICE_CHECK.json')
status = web['status']
assert status['source'] == 'public_replay' and status['state'] == 'done'
assert status['diagnostics']['session']['last_reason']['stage'] == 2
assert len(status['diagnostics']['recent_sessions']) == 1
assert not status['training_use'] and not status['video_saved']
assert (V / 'WEB_VERIFIED.jpg').stat().st_size > 1000
models = {
    'shared_conv_bigru': (O / 'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt',
        '80aba3b28006270308834bf5ba88277d7b26c8dce01c11fc962ce5c21b8cea17'),
    'previously_finetuned_motion_head': (U / 'action_integration/finetune/CANDIDATE.pt',
        '810f3b0dd922e83ac12045e848e660e4c2d6ef8b6190a6eb49a15cac542ef378'),
}
for path, sha in models.values():
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, path

next_action = ('수정된 8822에서 평소 앉은 웹캠 환경의 두 단계 진행과 안내를 확인한다. '
               '막히면 RAM 진단의 누락/모양 불확실/조건 미충족/지연을 읽고 공개 자료로 재현한다. '
               '사용자 촬영을 학습·튜닝·후보 선택에 쓰지 않는다. 실제 웹캠 편의성과 전체 지연은 아직 미검증이며 C·T3-F는 유지한다.')
limitations = [
    '24조건은 공개 SEN0354 REAL01 한 영상과 좌표·시간 변형이다. 24명/24독립영상의 정확도가 아니다.',
    '정지 프레임 복제와 누락 주입은 공통 진행 규칙의 강건성 검사이며 실제 수어 전문가 정답 검증이 아니다.',
    '실제 웹캠을 열지 않았다. 사용자 관측 어려움의 모든 원인이 해결됐다고 주장할 수 없다.',
    '15fps 공개 조건은 유지되지만 10fps에서 짧은 첫 동작을 보류하는 기존 한계는 남는다.',
    '엄지, 정확한 손바닥 방향, 실제 접촉, 다인 일반화와 전체 단어 정답은 미완료다.',
    '관절 좌표의 실제 오류와 손모양 모델 오판 자체를 이번 변경으로 학습 교정한 것은 아니다.',
]
verified = {
    'at':stamp, 'status':'PUBLIC_VERIFIED_WEB_APPLIED_CAMERA_RETEST_PENDING',
    'implementers':{'coach':'Codex coach_gap_repair', 'ui':'Codex diagnostic_ui',
                    'ledger_server_public_comparison_browser_records':'Codex root'},
    'independent_reviewer':'Codex usability_contract_review',
    'changes':['role-specific required observations', '250ms hold without hidden-motion credit',
               'latest actual equal-extremum query anchor', 'RAM diagnostic ledger and stable UI reason'],
    'checks':checks, 'ui_self_checks':57,
    'browser':{'public_replays':2, 'both_finished_two_stages':True,
               'previous_summary_visible':True, 'physical_camera_opened':False,
               'screenshot':str(V / 'WEB_VERIFIED.jpg')},
    'preserved_first_failure':{'file':'PUBLIC_COMPARISON.json','passes':22,'total':24,
                             'repair':'PLATEAU_ANCHOR_DECISION.json'},
    'code_hashes':admitted['code_hashes'],
    'frozen_model_hashes':{k:v[1] for k,v in models.items()},
    'training_this_task':False, 'private_training_tuning_selection':False,
    'C_T3_F_changed':False, 'whole_word_correct':None,
    'scope':'Separate guided practice 8822 only; existing models unchanged',
    'limitations':limitations, 'next_single_action':next_action,
    'server_pid':web['listener_pid'], 'url':'http://127.0.0.1:8822/',
}
write(V / 'VERIFIED.json', verified)

report = f'''実行記録 / 안녕하세요 실시간 코칭 진단·진행 보완
완료 시각: {stamp}
상태: 공개 자료 검증 후 별도 8822에 적용. 실제 사용자 웹캠 재확인은 남음.

무엇을 바꿨나
1. 관절 누락, 손모양 불확실, 확인된 손모양 조건 미충족, 위치·순서 조건 미충족, 움직임 불확실, 관측 지연을 구분한다.
   불확실은 오답으로 표시하지 않으며, 큰 안내 한 문장과 안정된 이유 표시를 유지한다.
2. 팔 쓸기에서는 움직이는 오른손의 펼침과 왼팔 기준점을 확인한다. 왼손 손가락이 모두 펴져야 시작하는 불필요한 조건을 제거했다.
   왼팔·손목 기준점은 계속 필요하다. 두 주먹 단계에서는 양손 모양과 양손 움직임을 모두 요구한다.
   이는 공통 역할별 관측 구조이며, 다른 모든 수어에서 지지손 모양이 중요하지 않다는 뜻이 아니다.
3. 0.25초 이내의 누락·불확실은 이미 관측한 진행을 잠시 보관한다. 안 보이는 사이의 이동·반복은 더하지 않는다.
   다시 보인 순간만으로 완료하지 않으며, 관측된 반대 모양이나 긴 누락은 미완료 시도를 끊는다. 완료한 단계는 보존한다.
4. 반복 중 잠시 같은 위치에 머문 뒤 움직일 때, 오래된 정지 시점부터 방향을 묻던 문제를 고쳤다.
   같은 극점의 가장 최근 실제 관측을 시작점으로 쓴다. 방향 확률·변위·양손 조건은 낮추지 않았다.
5. 현재와 직전 최대 5회 연습의 이유별 횟수·최근 이유 변화를 서버 메모리에만 남긴다.
   다음 연습을 눌러도 이전 요약이 보이지만 서버 종료 시 사라진다. 이 진단 요약에는 영상·관절 좌표가 없다.
   횟수는 중복되지 않은 관측 업데이트 수이며 오류 횟수/정확도가 아니다. 재시작 조건 횟수도 실제로 잃은 진도 수가 아니다.

이전 → 현재
- 지지손 손가락 누락 / 지지손 손모양 변경: 불필요한 조건으로 두 단계 미완료 → 필요한 팔 기준점이 보이면 완료.
- 반복 사이 같은 자세를 잠깐 유지 / 그 유지 중 한 번 누락: 두 번째 동작 미완료 → 완료.
- 역방향 쓸기, 첫 동작만 하고 멈춤, 주먹 유지뿐인 경우, 한 손만 반복: 잘못 완료시키지 않는 동작 유지.
- 가려진 사이 위로 돌아온 움직임: 반복 횟수에 추가하지 않음 유지.

어떻게 검증했나
- 공개 SEN0354 REAL01 원본 1개와 통제 변형 총 24조건. 최종 예상 결과 24/24 일치.
  정상 완료만 통과라는 뜻이 아니라, 오류·누락 조건에서 완료를 막는 기대 결과까지 포함한다.
- 최초 22/24 실패 기록을 보존하고, 두 정지 구간 실패의 오래된 극점 시각 원인을 수정한 뒤 24조건 재검증.
- 다른 에이전트의 상태 흐름 검사 34개, 극점·진단 검사 16개 통과. 합성 상태 검사이며 신경망 정확도가 아니다.
- RAM 진단과 공개 모의 카메라 서비스 흐름 검사, UI 자체 57검사 및 실제 브라우저 공개 재생 2회 확인.
  브라우저에서 두 단계 완료와 직전 연습 요약 보존 확인. 실제 웹캠은 열지 않음.

변경하지 않은 것
이번 추가 학습 없음. 공유 시간 합성곱+GRU 및 이전 보강 방향 판정부와 손모양 가중치는 유지.
C·T3-F / 기존 웹 경로 유지. 사용자 촬영 학습·튜닝·후보 선택 없음. 원본 공개 자료 수정 없음.
실시간 연습의 단계 순서·반복 횟수·시작끝은 여전히 규칙으로 연결하며, 전체 단어 정답은 보장하지 않는다.

남은 확인
{chr(10).join('- '+x for x in limitations)}
다음: {next_action}
웹: http://127.0.0.1:8822/ 새로고침. 서버 listener PID {web['listener_pid']}; 카메라는 사용자 버튼으로만 열림.
근거: VERIFIED.json, PUBLIC_COMPARISON_V2.json, INDEPENDENT_CHECKS_V2.json,
INDEPENDENT_HOLD_LEDGER_V2.json, UI_REVIEW.json, WEB_SERVICE_CHECK.json, WEB_VERIFIED.jpg.
'''
report = report.replace('実行記録 / ', '')
(V / 'RESULT_KO.txt').write_text(report, encoding='utf-8')
if not (V / 'before/RESULT_KO.txt').exists():
    shutil.copy2(U / 'RESULT_KO.txt', V / 'before/RESULT_KO.txt')
(U / 'RESULT_KO.txt').write_text(report, encoding='utf-8')

graph = read(U / 'TASK_GRAPH.json')
graph['at'] = stamp
graph['latest_repair'] = str(V / 'VERIFIED.json')
graph['new_training'] = 'None in usability repair; earlier copied movement-head training preserved'
graph['usability_repair'] = [
    {'id':'R1','label':'막힌 이유를 구분해 기록','status':'complete'},
    {'id':'R2','label':'손 역할별 조건·짧은 누락·정지 후 방향 보완','status':'complete'},
    {'id':'R3','label':'공개 정상·오류와 별도 검토','status':'complete_controlled_24_conditions'},
    {'id':'R4','label':'웹 적용·공개 재생·직전 진단 표시','status':'complete'},
    {'id':'R5','label':'평소 앉은 웹캠 환경에서 자연스러운 진행 확인','status':'pending_user_retest'},
]
for node in graph['nodes']:
    if node['id'] == 'E':
        node['status'] = 'public_repair_verified_user_retest_pending'
        node['evidence'] = str(V / 'VERIFIED.json')
write(U / 'TASK_GRAPH.json', graph)

state_path = O / 'STATE.json'
state = read(state_path)
state['updated_at'] = stamp
state['last_updated'] = stamp
state['next_single_action'] = next_action
entry = {'at':stamp, 'status':verified['status'], 'report':str(V / 'RESULT_KO.txt'),
         'verification':str(V / 'VERIFIED.json'), 'graph':str(U / 'TASK_GRAPH.json'),
         'url':verified['url'], 'pid':web['listener_pid'], 'public_conditions':24,
         'source_videos':1, 'training_this_task':False, 'private_training':False,
         'physical_camera_tested':False, 'C_T3_F_changed':False,
         'next_single_action':next_action}
state['hello_live_usability_repair_20261116'] = entry
state['hello_live_coach_20261115'].update(entry)
state['hello_live_coach_20261115']['independent_review'] = str(V / 'INDEPENDENT_CHECKS_V2.json')
state['hello_live_coach_usability_review_20261116']['followup_implementation'] = 'hello_live_usability_repair_20261116'
write(state_path, state)

context_path = R / 'WordsInSilence_PROJECT_CONTEXT.md'
context = context_path.read_text(encoding='utf-8-sig')
heading8 = '## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
heading9 = '## 9. 다음 단일 작업'
latest = '''### 2026-11-16 실시간 진단·역할별 조건·순간 누락 보완 / 공개 검증 후 웹 적용
8822에 누락·모양 불확실·모양 조건·진행 조건·움직임 불확실·관측 지연을 구분하는 RAM 진단을 추가했다. 현재+직전 최대 5회 요약만 서버 메모리에 보관하고 영상/좌표를 진단에 저장하지 않는다. 서버 종료 시 삭제되며 횟수는 관측 수이지 오류율이 아니다.
팔 쓸기는 움직이는 손의 펼침과 반대 팔 기준점을 요구하되 지지손 손가락까지 필수였던 조건을 제거. 두 주먹 반복의 양손 요구는 유지. 250ms 이내 누락·불확실은 진행 보관하되 가림 중 이동/반복/완료를 인정하지 않는다. 정지 후 반복 방향 조회에 오래된 극점 시각을 쓰던 문제도 최신 실제 동률 극점으로 보완했다. 모델/확률/변위 기준 변경·추가 학습 없음.
한 공개 SEN0354 원본과 통제 변형 24조건 최종 유지. 최초 22/24와 수정 근거 보존. 별도 상태 검사 34개·극점/진단 16개, 서비스 공개 모의·브라우저 공개 재생 2회 완료/이전 요약 확인. 24명/24독립영상 정확도가 아니며 실제 웹캠 재확인은 남음. 15fps 유지/10fps 짧은 첫 동작 보류 한계 유지. 엄지·방향·접촉·다인/전체정답 미완료.
근거 hello_live_coach_20261115/usability_repair_20261116/VERIFIED.json·RESULT_KO.txt. Codex root 통합/공개비교/웹, coach_gap_repair 구현, diagnostic_ui 화면, usability_contract_review 독립 검토. 기존 C·T3-F·공유/손모양/방향 가중치 유지, 사용자 촬영 학습·튜닝·후보선택 없음. 현재 서버는 SERVER.json과 실제 포트로 확인.
'''
assert heading8 in context and heading9 in context
context = context.replace(heading8, heading8 + '\n\n' + latest, 1)
context = context.replace(heading9, heading9 + '\n\n### 2026-11-16 진단·진행 보완 후 실제 웹캠 재확인\n' + next_action + '\n', 1)
context_path.write_text(context, encoding='utf-8')

handoff = f'''# 최신 목표 — 학습된 동작 모델을 이용한 한국수어 피드백 게임

## 마지막 완료 — {stamp}
안녕하세요 실시간 코칭의 관절 누락·손모양 불확실·진행 조건을 나눠 진단하고, 과한 양손 조건과 순간 누락 초기화를 보완했다. 공개 검증을 통과한 코드를 별도 8822에 적용했다. 실제 웹캠 재확인은 남았다.

## 이번 변경
- 팔 쓸기: 움직이는 오른손의 펼침 + 왼팔/손목 기준점. 지지손 손가락까지 모두 펴야 하던 조건 제거. 두 주먹 반복에서는 양손 모양·움직임 모두 유지.
- 250ms 이내 누락/불확실은 이미 관측한 진행만 보관. 가림 속 이동·반복/복귀 순간은 완료 근거로 쓰지 않음. 확인된 반대 모양·긴 누락은 미완료 시도 초기화, 완료 단계 유지.
- 정지 후 움직임 조회 시작을 최신 실제 동일 극점으로 갱신. 확률·변위·양손 조건 완화 없음.
- 현재+직전 최대 5회 진단 요약 RAM 보관. 영상/좌표는 진단에 저장하지 않음. 서버 재시작 시 삭제. 큰 문장+이유 표시 안정화, 상세에 이유별 관측 횟수. 횟수는 오류율/정확도가 아님.

## 검증·한계
공개 SEN0354 REAL01 1개와 좌표/시간 통제 변형 24조건 기대 결과 일치. 최초 22/24 실패를 보존하고 정지 구간 시각 원인 보완 후 재검증. 별도 상태 검사 34/34, 극점·진단 16/16, UI 자체 57항목, 서비스 공개 모의 검사, 브라우저 공개 재생 2회 완료·직전 요약 보존 확인. 독립 24영상/24명 정확도가 아니다. 물리 카메라를 열지 않았다.
15fps 유지, 10fps 짧은 첫 동작 보류 한계. 실제 RTMW 좌표 오류·손모양 오판을 이번에 재학습한 것은 아니다. 엄지·정확한 손방향·접촉·다인·전체 수어 정답 미완료. 사용자 앞선 어려운 촬영의 과거 상세 기록은 복구되지 않았으며 원인을 단정하지 않는다.

## 모델 역할·금지 범위
기존 100단어 공통 시간 합성곱+양방향GRU 특징, 손모양 보조, 이전 공개 보강 방향 판정부 유지. 팔 쓸기 축2/양손 높이 축0에 학습 방향을 요구하되 단계 순서·반복 횟수·시작끝은 명시 규칙. SEN0354 전체를 모델이 자율 이해한 상태가 아님. whole_word_correct=null. 단어 TCN은 이 안내 연습에 미사용; 기존 C·T3-F/8821 유지.
이번 학습 없음. 사용자 촬영 학습·튜닝·후보 선택 없음. 기존 공개 원본·동결 산출물 유지. 새 모델/자료 다운로드·Git staging/commit/push 없음.

## 다음 한 작업
{next_action}

## 실행과 근거
http://127.0.0.1:8822/ 새로고침. 일반 실행 listener PID {web['listener_pid']}, launcher 31072. 카메라는 사용자 버튼에서만 열림. 재개 시 실제 포트/명령 및 SERVER.json 확인. 일반 서버는 sandbox에서 안 보일 수 있음.
단위: research/front_keyphase_v1/ops/hello_live_coach_20261115/
최신 결과: usability_repair_20261116/RESULT_KO.txt, VERIFIED.json, ADMITTED.json, PUBLIC_COMPARISON_V2.json, INDEPENDENT_CHECKS_V2.json, INDEPENDENT_HOLD_LEDGER_V2.json, WEB_SERVICE_CHECK.json, WEB_VERIFIED.jpg. 그래프 TASK_GRAPH.json. 서버 로그 usability_repair_20261116/server.out.log·server.err.log.
UI_REVIEW.json은 UI 담당자 자체 검사 당시 기록이며 최종 웹 적용/검증은 root VERIFIED.json을 따른다. 최초 실패 PUBLIC_COMPARISON.json, before/ 및 first_attempt_coach.py 보존.
원본 공유 모델 SHA 80aba3b28006270308834bf5ba88277d7b26c8dce01c11fc962ce5c21b8cea17.
이전 방향 머리 SHA 810f3b0dd922e83ac12045e848e660e4c2d6ef8b6190a6eb49a15cac542ef378.
구현: root 서버/진단/공개비교/웹/기록, coach_gap_repair 진행 로직, diagnostic_ui 화면. 독립 검증: usability_contract_review. 수어 전문가 검수 아님.
Python: research/front_keyphase_v1/ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe -X utf8 -B.
웹 STATE: research/front_keyphase_v1/ops/STATE.json. 현재 키 hello_live_usability_repair_20261116. 루트 Ops/STATE는 Unity 계보.

## 큰 목표와 자료 준비
기존 4단어는 공통 회귀 범위, 실제 시연 목표는 안녕하세요 SEN0354/사과 WORD0099/감사 WORD1290. 인사 참고 1개는 C의 runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4. W의 1TB는 WORD 33ZIP, SEN 다인 원본은 아직 없음. 감사·사과 후속은 basic_demo_feedback_20261115/acquisition_motion_followup. 본 실시간 보완을 전체 프로젝트 완료로 기록하지 않는다.
'''
(R / 'NEW_CHAT_HANDOFF.md').write_text(handoff, encoding='utf-8')
print(json.dumps({'status':verified['status'], 'report':str(V / 'RESULT_KO.txt'),
                  'pid':web['listener_pid'], 'records_synchronized':True}, ensure_ascii=False))
