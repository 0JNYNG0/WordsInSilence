"""Persist user-requested pause and graph only; never run project experiments."""
from pathlib import Path
import json, shutil
from datetime import datetime,timezone,timedelta
HERE=Path(__file__).resolve().parent
OPS=HERE.parent
ROOT=HERE.parents[3]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
next_action='사용자 재개 요청 후 B부터: 감사 설명의 손모양·손면 방향·상대 도착 위치·반복 순서에 공개 원본 정답과 관측 가능성 연결. C에서 관절/모양/구간연결 문제에 맞게 보완·필요한 학습, D 공개 정상/오류/부분/불확실 회귀, E 정상통과·교정·재시도 연결, F 실제 촬영 시연. 현재 부분 움직임 보존 부품의 오프라인45파생검사와10계약검사는 완료했으나 전체 과제·웹 연결 완료는 아니다.'
plan=read(HERE/'plan.json')
plan.update(status='PAUSED_BY_USER',paused_at=now,next=next_action,
    predecessor_graph='../basic_demo_feedback_20261115/TASK_GRAPH.json',
    stop_boundary='진단 및 부분 관측 보존 부품의 오프라인 비교 종료. 사용자 요청 이후 새 실험/학습/서버 변경 없음.',
    completed_this_turn=['촬영후 경로와 실제 양성완료 생성부 누락 감사','감사 필수6항목 명세','기존 관측창·부분 영상 진단','공통 부분관측 보존 부품 작성','공개5영상×9파생조건45검사 및10계약검사'],
    remaining='B의 원본 정답/범위 검증과 C의 실제 손모양·끝상태 판정, D 전체 과제 정상/오류 검증, E/F')
for node in plan['nodes']:
    if node['id']=='C':node['status']='PARTIAL_PAUSED'
    if node['id']=='D':node['status']='COMPONENT_CHECKS_ONLY_FULL_GATE_PENDING'
write(HERE/'plan.json',plan)
write(HERE/'PAUSE.json',dict(status='PAUSED_BY_USER',at=now,next_action=next_action,
    active_training_or_test_started_by_this_turn=None,
    user_cameras_opened=False,web_modified=False,model_training_runs=0,
    implementation_actor='Codex root',validation_actor='same Codex; not independent review',
    graph='TASK_GRAPH.png',plan='plan.json',results='PROGRESS_COMPARISON.json'))
report='''촬영 후 피드백 완성 경로 / 사용자 요청으로 일시 중단

방향: 촬영 중 RTMW 관측 재사용 → 촬영 종료 → TCN 단어 인식 및 시간 합성곱+GRU/손가락 분석 → 설명과 연결된 근거·DTW 비교 재생 → 교정 또는 과제 완료.
실시간 코칭 확장은 보류한다. 기존 공통 보완을 폐기하지 않으며 동일 부품의 회귀가 통과할 때만 두 방식에 반영한다.

이번 완료
1. 기존 피드백 카드와 게임 성공조건을 검토했다. 실제 모든 필수조건의 긍정 근거를 생성하는 연결은 미완료다.
2. 감사 설명을 손모양 좌/우, 손면 방향 좌/우, 상대 도착 관계, 반복 순서의 6필수 항목으로 명세했다. 정답 검증 완료를 뜻하지 않는다.
3. 공개 감사18번을 첫 도착 직후까지 잘랐을 때, 움직임 판정은 남지만 도착 상태가 보류되어 반복0회로만 요약되는 것을 재현했다. 전체 영상에서는 같은 구간의 가까움 판정이 살아 있다. 끝 문맥과 관측창이 영향을 주는 것으로 보이며, 이것만으로 정확한 원인 기여도를 분리했다고 하지 않는다.
4. 수행한 움직임의 실제 구간과 도착 확인을 별도 보존하는 공통 부품을 작성했다. 0회→1회로 임의 채우지 않는다. '움직임은 보였지만 끝은 확인 필요'로 남긴다.
5. 이미 사용한 공개5영상의 정상/첫동작만/고정/누락/시간·크기변환/역재생/도착누락 45파생조건은 선언한 관측보존 계약을 통과했다. 별도10계약검사 통과. 45개 독립영상 또는 오류45개 인식 성능이 아니다. 정상5영상의2회 접근 판정 유지. 실제 새로운 오수행자에 대한 정확도 미측정.

이번 미실행: 추가학습, 새 가중치/임계값, 사용자 촬영 사용, 전체 단어 정답 승인, 새 웹 기능 연결, 서버 재시작. C·T3-F 변경 없음. 모델 정확도 향상으로 보고하지 않는다.
실제 남은 것: 손모양 불일치, 손면 방향·상대 도착 정답, 부분과 완료의 최종 구분, 모든 필수항목 기반 긍정 완료 생산부, 정상/오류 전체 회귀, 웹 성공·교정·재시도, 실제 촬영 대기시간·휴대성.
다음 재개: plan.json B·C. 감사부터 완성 단위를 연결하고 사과·승인된 안녕하세요로 확장한다. 기존4단어는 공통 회귀; 시연 단어로 임의 대체하지 않는다.
검토·실행: Codex root. 새 독립 에이전트·전문가 검토 없음.
'''
(HERE/'RESULT_KO.txt').write_text(report,encoding='utf-8')
before=HERE/'before';before.mkdir(exist_ok=True)
paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json']
for p in paths:
    if not (before/p.name).exists():shutil.copy2(p,before/p.name)
state=read(OPS/'STATE.json')
state['current_stage']='capture_feedback_completion_20261116'
state['next_single_action']='일시 중단: 사용자 재개 요청 전 실행하지 않는다. '+next_action
state['capture_feedback_completion_20261116']=dict(status='PAUSED_BY_USER',updated_at=now,
    graph=str(HERE/'TASK_GRAPH.png'),plan=str(HERE/'plan.json'),report=str(HERE/'RESULT_KO.txt'),
    component_tests=dict(public_clips=5,derived_cases=45,passed=45,contract_tests=10),
    whole_word_accuracy=None,web_admitted=False,training_runs=0,private_data_used=False,
    next_single_action=next_action)
write(OPS/'STATE.json',state)
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=ctx.read_text(encoding='utf-8-sig')
header='### 2026-11-16 촬영후 평가 우선순위 복귀·작업 그래프 정리 / 사용자 요청 일시 중단'
entry=header+'\n사용자 요청으로 촬영후 피드백·정상통과 완성을 중심에 두고 실시간 코칭 확장은 보류. 기존 공유 RTMW/100공통Conv+GRU/손가락/TCN·DTW는 유지한다. 정상 과제완료 생산부가 없는 현황을 확인, A~F 검증루프와 감사 필수6항목 명세를 작성했다. 공개 첫동작만 구간의 접근0회 요약 누락을 재현하고 움직임·도착을 분리 보존하는 공통부품을 격리 작성했다. 공개5영상45파생관측계약 및10계약검사 통과, 전체정확도/새웹완료 아님. 추가학습·사용자자료·서버변경·C·T3-F변경 없음. 손면·손모양·전체완료 검증 및 웹 연결은 남음. 구현/검증Codex root,새독립검토 없음. 근거 ops/capture_feedback_completion_20261116/plan.json·PAUSE.json·PROGRESS_COMPARISON.json·RESULT_KO.txt.\n**현재 사용자가 작업 중단 요청. 그래프/중단기록 정리만 완료하고 기다린다. 재개 요청 전 추가 실행 금지.**\n\n'
marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
assert marker in s
if header not in s:s=s.replace(marker,marker+'\n'+entry,1)
next_header='### 2026-11-16 작업 일시 중단 / 촬영후 평가 그래프 B·C에서 재개'
marker='## 9. 다음 단일 작업\n'
assert marker in s
if next_header not in s:s=s.replace(marker,marker+'\n'+next_header+'\n사용자 재개 요청 전 작업하지 않는다. '+next_action+'\n\n',1)
ctx.write_text(s,encoding='utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — 한국수어 피드백 게임

## 현재: 사용자 요청으로 일시 중단 ({now})
사용자 최신 요청: "작업 그래프만 정리하고 작업은 잠시 중단. 이따 재개 요청하면 그래프대로 진행."
새 학습·검증·구현·서버 변경을 시작하지 않는다. 이번 실행 중 프로세스 없음(이번 명령 모두 종료). 기존 웹서버는 중지/재시작하지 않았으며 이전PID를 실행중이라고 가정하지 않는다.

## 확정 방향
촬영 중 RTMW 추출 재사용 → 종료 후 TCN/시간 합성곱+GRU·손가락 분석 → 설명 행동·DTW 재생 → 검증된 교정/관측부족/과제완료. 실시간8822는 기능 확장 보류, 기존 공유부품을 버리지 않는다. 촬영후8821 우선. 실시간 전용규칙을 자동이식하지 않는다.
기초시연 감사→사과→사용자가 선택한SEN0354안녕하세요. 기존4단어는 회귀용이며 최종시연에 임의대체하지 않는다. 인사는다인자료여전히미확보.

## 이번 완료
기존 경로 감사, 실제 긍정 lesson_acceptance 생성부 미완료 확인. A~F그래프/감사필수6항목명세 작성.
공개감사14~18의 첫접근구간 비교:18번은움직임판단있지만끝상태보류→도달0회요약. 움직임 후보를 실제관측범위로보존하고끝상태와분리하는 공통부품 작성.
공개5영상×9파생조건45관측계약+10자체계약검사 통과. 정상5개2회접근유지. 기존손모양문제/도달정답해결 아님. 45독립영상이나전체정답률아님.
새모델학습0,가중치/임계값변경0,사용자자료미사용,웹반영없음,C·T3-F유지. Codex root구현·자체검증,새독립검토없음.

## 재개 위치
{next_action}

## 근거
research/front_keyphase_v1/ops/capture_feedback_completion_20261116/
plan.json · TASK_GRAPH.png · requirements.json · PAUSE.json · RESULT_KO.txt
TERMINAL_DIAGNOSIS.json · PROGRESS_FROZEN.json · PROGRESS_COMPARISON.json
motion_evidence.py · test_evidence.py (10통과)
이번공개자료이미사용한개발/진단자료:새최종평가로재명명금지.
다음검증전과거실험파일덮어쓰기금지. verify_progress.py는완료결과덮어쓰기방지함.

## 남은 공통 문제
공통손모양 old/expanded 불일치와 옆손가락 관절추적오류는여전함. 지난수치붕괴가드 두경로적용보존. 근거 hello_live_coach_20261115/handshape_cause_review_20261116/.
손면방향/상대도착/필수항목완료와게임긍정통과미완료. 정면영상의관측가능한수행을판단하며정확3D접촉거리/모든언어적정확성을보장하지않음. LLM/새3D시범/범위확장 없음.
''',encoding='utf-8')
print('PAUSE_AND_GRAPH_RECORD_SAVED',now)
