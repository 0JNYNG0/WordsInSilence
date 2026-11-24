"""Synchronize verified Stage2 outcome; no model/production mutation."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
U=Path(__file__).resolve().parent;LAB=U.parents[1];ROOT=LAB.parents[1]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
assert not (U/'COMPLETION.json').exists(),'Completed documentation already synchronized'
raw=read(U/'RAW_CAPTURE_RESULT.json');replay=read(U/'REPLAY_RESULT.json');ind=read(U/'INDEPENDENT_VERIFICATION.json');change=read(U/'CHANGE_AUDIT.json')
assert raw['conditions']==replay['conditions']==24
assert ind['independent_behavior_tests']==22 and ind['failures']==ind['errors']==0
assert ind['raw_output_audit']['status']=='PASS'
assert replay['counts']['side_changed']==9 and replay['counts']['newly_assigned']==3 and replay['counts']['dropped']==31
assert change['summary']['actual_counts']=={k:replay['counts'][k] for k in ['unchanged','dropped','side_changed','newly_assigned']}
for context in ['RAW_CAPTURE_CONTEXT.json','REPLAY_CONTEXT.json']:
    for p,h in read(U/context)['pins'].items():assert sha(p)==h,p
for row in replay['rows']:
    for p,h in row['outputs'].items():assert sha(p)==h,p
for p in ['RESULT_KO.txt','EMPIRICAL_REVIEW.md','CODE_REVIEW.md']:assert (U/p).exists()
now=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
summary='研究용 좌우 손 배정·미검출 처리 후보 구현 및 비교 완료. 공개 학습 REAL01 12영상 원본/crop 24조건, 2,976프레임·3,910검출의 같은 raw 비교. 좌우 변경9건/7프레임은 올바른 교정, 실제 추가 왼손 회복1건. 배정 제외31건에 정상 관측 회귀가 포함: 공항 원본63–65 약0.100초 및 crop78 관측 손실, 마음에들다47 raw 선택의 형상 회귀 우려. 동작/연결 검사22/22 및2,997파일 해시 보존 PASS는 보존·논리 검증이며 성능 통과가 아님. 후보 미채택·학습 준비 보류. 새 학습/분류 추론/운영·웹·DTW 변경0.'
summary=summary.replace('研究용','연구용')
next_action='사용자와 2단계의 교정·회귀 결과를 검토한 뒤, 정상 손의 pose/handedness 충돌 거절 및 중복 raw 선택 문제를 같은 입력 기능의 별도 수정 후보로 보완한다. 현재 후보로 전체 재추출·학습·운영 교체를 자동 시작하지 않는다.'
decision={'status':'COMPLETE_WITH_REGRESSIONS_NOT_ADOPTED_AWAITING_USER_REVIEW','updated_kst':now,
 'unit':'ops/word100_hand_assignment_20261016','report':'ops/word100_hand_assignment_20261016/RESULT_KO.txt',
 'summary':summary,'next_single_action':next_action,'behavior_tests_passed':22,
 'conditions':24,'frames':2976,'raw_hand_detections':3910,'corrected_side_detections':9,'corrected_side_frames':7,
 'additional_real_hand_observations':1,'raw_new_assignments':3,'raw_dropped_assignments':31,
 'known_counterexamples':['airport_native63_to65_visible_R_dropped_0.100s','airport_crop78_visible_R_dropped','like_crop47_shape_selection_concern'],
 'training_ready':False,'adopted':False,'training':0,'classifier_inference':0,'production_changed':False,
 'running_processes':[],'implementation':'Codex root','independent_verification':['assignment_review','assignment_verify','raw_capture']}
write_new(U/'DECISION.json',decision)
paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',LAB/'ops/STATE.json',ROOT/'NEW_CHAT_HANDOFF.md',ROOT/'Docs/WEBCAM_IMPROVEMENT_TODO.md']
(U/'before').mkdir(exist_ok=True)
for p in paths:
    dest=U/'before'/p.name
    with dest.open('xb') as f:f.write(p.read_bytes())
context=paths[0].read_text('utf-8')
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
h9='## 9. 다음 단일 작업\n'
assert context.count(h8)==context.count(h9)==1
context=context.replace(h8,h8+'\n### 2026-10-16 웹캠 보완 2단계 비교 완료 — 좌우 교정 확인, 관측 회귀로 후보 미채택\n'+summary+'\n근거 `ops/word100_hand_assignment_20261016/RESULT_KO.txt`, `COMPLETION.json`, `Docs/WEBCAM_IMPROVEMENT_TODO.md`. 같은 raw 기록을 비교했고, 이전 원본/crop24조건 관절 배열과도 전부 일치. 입력 검출 재수행163.73초, 모델 학습과 무관.\n',1)
context=context.replace(h9,h9+'\n### 2026-10-16 다음 — 2단계 결과 상호 검토 후 정상 관측 보존 보완\n'+next_action+'\n사용자는 이번 단계 결과를 함께 확인한 다음 학습으로 넘어가도록 요청했다. 후보는 그대로 보존하고 코드 검증 PASS를 학습 적격으로 바꾸지 않는다. 기존 모델·사용자 자료 분리 유지.\n',1)
paths[0].write_text(context,'utf-8')
state=read(paths[1]);state['updated_date_kst']=now;state['current_stage']='WORD100_HAND_ASSIGNMENT_STAGE2_COMPARED_REGRESSIONS_NOT_ADOPTED';state['next_single_action']=next_action
state['word100_hand_assignment_20261016']=decision
paths[1].write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8')
paths[2].write_text(f'''# WordsInSilence 최신 인수인계

갱신 {now}. 목표는 근거리 웹캠의 상반신·화면 밖 손 조건에 맞춘 인식 보완이다.

## 사용자 진행 방식
기능별 작업 → 결과·문제 보고 → 사용자와 검토 → 다음 단계. 사용자는 좌우 손 배정·미검출 처리를 먼저 보완하고 결과를 함께 확인한 다음 학습하도록 요청했다.

## 마지막 완료와 판정
{summary}
원시 캐시는 같은 공개 학습 REAL01 12영상의24조건만 다시 검출했다. 이전 관절 배열24조건 모두 완전 일치. 동일 raw에서 기존·후보 배정을 비교했으며 사용자/개발/보류 자료 배열은 사용하지 않았다. 무검출871프레임 좌표 생성0, 시간/마스크 보존.
새 배정3건 중 믿다49만 실제 추가손 회복, 마음에들다47·멀다88은 같은 오른손의 raw 교체다. 제외31건을 모두 오류 제거로 해석하지 않는다. 정상 수행 손을 알고리즘이 추가로 제외한 것은 자연적인 화면 밖 누락과 구분한다.
자료: `ops/word100_hand_assignment_20261016/RESULT_KO.txt`, `DECISION.json`, `COMPLETION.json`, `EMPIRICAL_REVIEW.md`, `INDEPENDENT_VERIFICATION.json`, `CHANGE_AUDIT.json`.

## 다음 한 작업
{next_action}
알려진 문제: 가까운 pose 손목과 충돌하는 잘못된 높은 handedness가 정상 손을 거절함; 같은 손의 복수 검출 경쟁으로 전부 버리거나 형상이 덜 맞는 raw를 고름; 엄격한 손바닥 경계가 부분 관측을 추가 제외함.
기존 후보·결과를 덮어쓰거나 최종 성능에 맞춘 계수 탐색 금지. 후속 수정 후보/조건은 별도로 기록한다. 이 결과가 사용자 확인된 것으로 먼저 간주하지 않는다.

## 작업 목록
`Docs/WEBCAM_IMPROVEMENT_TODO.md`. 1단계 완료, 2단계 구현·비교 완료/회귀로 채택 보류. 3단계 비교조건/자료준비와4단계 TCN학습은 대기. DTW 필수손 처리와 웹 재시험은 그 이후 별도 기능 단위.

## 보존 기준
TCN `runs/word100_tcn_v1/whole_context/best.pt`, SHA e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f. 100단어 학습9명899영상/개발3명300영상. 96.67%는 개발검증이다. 보류10/14–18 미사용.
사용자12회시험 Top1 6/12(50%),Top5 8/12(66.67%)는 개발참여자1인 실사용 기록. 학습·튜닝·후보선택·임계값 조정 금지.
운영 웹/DTW/모델, 원본ZIP/관절과 이전 동결 결과 보존. 이번 프로세스 모두 종료. 학습/분류추론/카메라/장치/다운로드/외부전송/Git 변경0. 발표8장 PDF와 대본포함8장 PDF는 완료 상태 유지.
''','utf-8')
todo=paths[3].read_text('utf-8')
todo=todo.replace('| 다음 작업·검토 후 |','| 구현·비교 완료 / 회귀로 채택 보류 |')
todo=todo.replace('이번 승인으로 1단계부터 실행한다. 2단계 이후는 1단계 결과를 사용자와 검토한 후 진행한다.','1단계 검토 후 사용자 승인으로 2단계 후보 구현·비교까지 실행했다. 2단계 회귀 결과를 함께 검토하고 정상 관측 보존을 보완하기 전 학습으로 넘어가지 않는다.')
todo+='\n## 2026-10-16 2단계 결과\n\n'+summary+'\n\n다음 한 작업: '+next_action+'\n\n근거: `research/front_keyphase_v1/ops/word100_hand_assignment_20261016/RESULT_KO.txt`.\n'
paths[3].write_text(todo,'utf-8')
for p in [paths[1],U/'DECISION.json']:read(p)
assert all(p.exists() for p in paths)
assert 'REGRESSIONS_NOT_ADOPTED' in read(paths[1])['current_stage']
write_new(U/'COMPLETION.json',dict(decision,documents={str(p):sha(p) for p in paths},
 evidence={str(U/p):sha(U/p) for p in ['RESULT_KO.txt','DECISION.json','RAW_CAPTURE_RESULT.json','REPLAY_RESULT.json','INDEPENDENT_VERIFICATION.json','CHANGE_AUDIT.json','EMPIRICAL_REVIEW.md','CODE_REVIEW.md']},
 preservation_verified=True))
print('Stage2 partial/non-adopted outcome and all4 authority documents synchronized. No training.')
