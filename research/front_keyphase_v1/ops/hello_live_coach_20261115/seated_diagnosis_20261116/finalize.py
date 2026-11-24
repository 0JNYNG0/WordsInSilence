"""Record operational diagnosis separately from public-only model evidence."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib
V=Path(__file__).resolve().parent;U=V.parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
review=read(V/'DIAGNOSTIC_AGGREGATE.json'); compare=read(V/'CANDIDATE_COMPARISON.json'); web=read(V/'WEB_SERVICE_CHECK.json'); admission=read(V/'VERTICAL_ADMITTED.json')
assert compare['passed'] and compare['public_passes']==28 and compare['old_dev_normal_correct']==32
assert web['state']=='done' and web['completed']==[True,True]
assert web['provenance']['vertical_motion_head_sha256']==admission['sha256']
for name in ['INTEGRATION_CHECK.json','STATE_REGRESSION.json','HOLD_LEDGER_REGRESSION.json']:
 assert read(V/name)['passed']
base=O/'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='80aba3b28006270308834bf5ba88277d7b26c8dce01c11fc962ce5c21b8cea17'
assert hashlib.sha256((U/'action_integration/finetune/CANDIDATE.pt').read_bytes()).hexdigest()=='810f3b0dd922e83ac12045e848e660e4c2d6ef8b6190a6eb49a15cac542ef378'
next_action=('공개 자료에서 손모양 불확실을 관절 관측/손가락 분류/동작 전환으로 나눠 검토한다. '
 '8822에는 검증된 상하 방향 보강과 단계·손가락별 진단이 적용됐다. 실제 사용자 편의성 개선은 재촬영 전까지 미확인; '
 '필요 시 자연스러운 한 차례 재시험의 진단만 확인하고 사용자 촬영은 학습·튜닝·후보 선택에 쓰지 않는다. C·T3-F 유지.')
report=f'''앉은 자세 연습 5회 진단 및 공개 재현 보완
시각: {at}

확인한 사용자 기록
- 카메라 세션 5회가 남아 있었고 모두 두 단계 완료로 끝났다. 공개 재생 기록은 이 집계에서 제외했다.
- 첫 판정부터 완료까지 약 54.8 / 127.4 / 93.3 / 72.3 / 102.4초. 이 시간은 자세 조정·재시도 시간이지 모델 계산 한 번의 대기시간이 아니다.
- 총 6,901 관측 중: 손모양 불확실 2,276(33.0%), 방향 불확실 1,053(15.3%), 진행 조건 미충족 803(11.6%), 요구 모양과 다르게 판정 363(5.3%), 관절 누락 220(3.2%), 진행 관측 2,186(31.7%).
- 불확실/미충족으로 기록된 관측만 보면 손모양 약48%, 방향22%, 진행조건17%, 모양조건8%, 누락5%다. 오류율·정확도·실제 사용자 오답 비율이 아니다.
- 지연 보류 분류는 0회였고, 세션별 최대 관측 시점 지연은 약315~370ms였다. 화면 렌더링까지 포함한 전체 지연 측정이 아니다.
- 영상/관절 좌표를 가져오거나 학습·튜닝·후보 선택에 사용하지 않았다. 스칼라 진단 요약만 별도 보존했다.
  이전 진단은 마지막24 이유 전환만 남아 초기 단계의 상세 구간·손가락·모델확률을 소급 복구할 수 없다.
  유한한 잘못된 관절 좌표는 누락으로 집계되지 않으므로, 누락이 적다고 관절 추정이 정확하다는 뜻도 아니다.

공개 자료에서 재현한 문제
같은 공개 인사 영상을 640x480 상반신 화면으로 바꾸어 RTMW를 새로 추출했다. 원본은 완료됐지만 이 조건은 첫 단계만 완료했고, 짧은 상향 복귀를 정지/불확실로 보아 두 번째 단계가 막혔다.
16초 동안 공개 시작 자세를 유지한 뒤 같은 동작을 하는 파생 조건에서도 두 번째 단계가 막혔다.
이는 사용자의 동작 잘못만으로 볼 수 없는 모델/입력 조건 민감성의 근거다. 사용자 세션에서 똑같은 원인이 발생했다고 확정한 것은 아니다.

실제 보완
1. 기존 공개 FIT와 RGB로 이미 검토한9개 공개 인사 방향·정지 구간에 화면 크기/대기 흐름 변형을 추가했다.
   기존 공유 시간 합성곱+GRU와 손모양 모델은 동결하고 복사한 작은 방향 판정부만 seed42/400step 1회 지도학습했다.
   전체 판정부 교체는 기존 팔 방향의 정지1구간이 나빠져 제외했다. 최초 실패 MOTION_RESULTS.json을 보존했다.
   실제 적용은 상하 방향(axis0)만 별도 보강 판정부를 사용하고, 팔 쓸기 등 다른 방향은 기존 판정부 그대로다. 단어 이름 예외는 없다. 손목 교체 실험도 개선없어 적용하지 않았다.
2. 관절이 잡혔는데 손모양 모델이 불확실할 때 계속 '잘 보이게 해 주세요'라고만 하던 문구를 수정했다.
   이제 '손가락 모양 판정이 불확실해요. 참고 모양을 잠깐 유지해 주세요.'라고 구분한다. 확률·변위·양손·누락 안전 조건은 낮추지 않았다.
3. 진단에 단계별 관측 수, 단계가 바뀐 시각, 어느 손가락이 관측 불가/판정 불확실/반대 모양인지, 방향 조회가 정지/불확실/자료부족인지 구분하는 스칼라 집계를 추가했다.
   개인 영상·좌표·모델 특징/확률 배열은 진단에 저장하지 않는다. 기본 웹 화면은 큰 안내 한 문장과 기존 요약을 유지하며 더 세밀한 집계는 진단 API에 있다.

검증 결과
- 기존 정상·오류24조건 유지. 상반신 원본 재추출/대기/추가 화면 조건까지 총28조건 기대 결과 일치.
- 상반신 확대 및16초 대기 조건: 기존 두 번째 단계 미완료 → 현재 완료.
- 학습에 넣지 않은 다른 공개 화면 크기/자르기 조건도 기존 미완료 → 완료. 같은 사람·같은 원본의 변형이며 새 사람 정확도가 아니다.
- 기존 공개 개발 검증32개 정상 방향 구간32개 유지. 확률 기준 이상의 정답31개도 유지.
- 자체 통합27항목, 기존 독립 작성 상태 검사34항목·유지/진단16항목을 root가 재실행해 통과. 이번 별도 에이전트 검토는 수행하지 않았다.
- 실제 브라우저 공개 재생에서 두 단계 완료 및 서버의 새 상하 판정부 해시·상세 진단 확인. 카메라를 새로 열지 않음.

남은 문제
손모양의 불확실성이 이번 작업으로 해결됐다고 말할 수 없다. 손모양 가중치는 그대로이며, 실제 사용자 편의성 향상도 아직 재시험 전이다.
엄지·정확한 손바닥 방향·실제 접촉·다인 일반화/전체 단어 정답, 10fps 짧은 첫 동작 보류 한계는 남는다.
이번 결과는 관측·판정 조건을 개선한 제한된 실시간 연습 보완이며 전체 피드백 게임 완성이 아니다.

웹: http://127.0.0.1:8822/ 새로고침. 일반 실행 listener PID {web['listener_pid']}.
기존 C·T3-F, 공유 인코더, 원래 방향/손모양 체크포인트, 기존 웹 경로 유지. 새 외부 다운로드·Git 변경 없음.
다음: {next_action}
근거: DIAGNOSTIC_AGGREGATE.json(진단 전용), PUBLIC_PROBE.json, MOTION_FROZEN_PLAN.json,
MOTION_RESULTS.json, AXIS_SCOPE_DECISION.json, CANDIDATE_COMPARISON.json, INTEGRATION_CHECK.json, WEB_SERVICE_CHECK.json.
'''
(V/'RESULT_KO.txt').write_text(report,encoding='utf-8')
verified=dict(at=at,status='PUBLIC_REPRODUCED_VERTICAL_REPAIR_APPLIED_USER_RETEST_PENDING',
 report=str(V/'RESULT_KO.txt'),personal_diagnostic_only=True,private_training_tuning_selection=False,
 personal_sessions=5,personal_accuracy=None,public_conditions=28,public_source_people=1,
 old_dev_normal_direction_correct=32,old_dev_normal_total=32,shared_base_frozen=True,
 trained_parameters='copied motion_head, applied only axis0',training_steps=400,training_seed=42,
 unrestricted_candidate_rejected=True,C_T3_F_changed=False,thresholds_changed=False,
 implementation_and_verification='Codex root; existing independently authored regression tests re-executed; no new independent agent',
 physical_camera_opened_by_agent=False,actual_usability_improvement_verified=False,
 url='http://127.0.0.1:8822/',pid=web['listener_pid'],vertical_sha=admission['sha256'],
 code_hashes={n:hashlib.sha256((U/n).read_bytes()).hexdigest()for n in ['coach.py','action_runtime.py','diagnostic_ledger.py','server.py']},
 next_single_action=next_action)
write(V/'VERIFIED.json',verified)
(U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
g=read(U/'TASK_GRAPH.json');g['at']=at;g['latest_repair']=str(V/'VERIFIED.json');g['new_training']='Public motion-head copy400steps; applied vertical axis0 only. Original shared/shape/forearm heads frozen.'
g['seated_diagnosis']=[
 dict(id='S1',label='5회 진단 확인: 실제 사용자 오답률과 구분',status='complete'),
 dict(id='S2',label='공개 화면 조건·대기 흐름에서 막힘 재현',status='complete'),
 dict(id='S3',label='상하 방향 보강·기존 정상오류 검증',status='complete_public_scope'),
 dict(id='S4',label='웹 적용·상세 진단·공개 연결 확인',status='complete'),
 dict(id='S5',label='손모양 불확실 원인 보강·실사용 편의성',status='remaining')]
for n in g['nodes']:
 if n['id']=='E':n.update(status='five_sessions_difficult_completion_reviewed_vertical_repair_retest_pending',evidence=str(V/'VERIFIED.json'))
write(U/'TASK_GRAPH.json',g)
s=read(O/'STATE.json');s['updated_at']=at;s['last_updated']=at;s['next_single_action']=next_action
entry=dict(at=at,status=verified['status'],report=str(V/'RESULT_KO.txt'),verification=str(V/'VERIFIED.json'),
 graph=str(U/'TASK_GRAPH.json'),pid=web['listener_pid'],url=verified['url'],next_single_action=next_action,
 training_this_task=True,training_scope='public-only vertical readout',private_training=False,public_conditions=28,
 user_usability_report='5 camera sessions complete but55-127sec retry effort; not stable usability',physical_camera_tested=False)
s['hello_live_seated_diagnosis_20261116']=entry;s['hello_live_coach_20261115'].update(entry)
s['hello_live_coach_20261115']['independent_review']='No new agent review this turn; root reran prior independently authored state/hold contracts.'
write(O/'STATE.json',s)
cpath=R/'WordsInSilence_PROJECT_CONTEXT.md';c=cpath.read_text('utf-8-sig')
section='''### 2026-11-16 앉은 연습5회 진단·공개 재현 후 상하 방향 보강
카메라5회 모두 완료됐지만 첫 판정~완료55~127초의 자세조정/재시도. 처리1회 대기시간 아님. 6901관측에서 모양불확실33%,방향불확실15%,진행조건12%,모양조건5%,누락3%; 오류율/정확도가 아니다. 영상/관절 좌표 없이 진단 요약만 확인·별도 보존. 개인 학습/튜닝/후보선택 없음.
공개SEN0354를640상반신으로 다시RTMW추출/16초 대기흐름에서 두 번째 짧은상승이정지로판정되어막힘 재현. 공개기존FIT+검토된9구간의화면/대기변형으로복사방향머리1회400step보강. 전체머리교체는기존팔정지1구간회귀로미채택,상하axis0만적용해기존팔·다른축동결. 정상/오류24+추가4=28조건유지,기존DEV정상32/32유지,다른학습미사용화면조건도미완료→완료. 같은사람원본파생이지독립다인정확도아님.
손모양불확실 안내를관측누락과구분하고 단계별·손가락별·방향원인 RAM진단추가. 자체통합27/기존독립작성상태34·유지진단16을root재실행. 별도에이전트재검토없음. 웹공개재생완료와새머리해시확인,이번실제카메라미개방. 손모양가중치/C·T3-F/공유인코더/기존체크포인트유지. 실제사용편의성개선·손모양불확실해결은미검증.
근거 hello_live_coach_20261115/seated_diagnosis_20261116/VERIFIED.json·RESULT_KO.txt. 현재서버와PID는SERVER.json확인.
'''
c=c.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+section,1)
c=c.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n### 2026-11-16 앉은 연습 진단 보완 후속\n'+next_action+'\n',1);cpath.write_text(c,encoding='utf-8')
handoff=f'''# 최신 목표 — 학습된 행동 모델로 한국수어 피드백 게임

## 현재 — {at}
사용자가 평소 앉은 자세로 약5회 시험했고 스트레스를 보고했다. RAM 카메라5세션 모두완료였으나55~127초 조정/재시도가 필요했다. 계산지연과 혼동금지. 전체6901관측의 모양불확실33%,방향불확실15%,진행조건12%,모양조건5%,누락3%,진행32%. 이 수치는 정확도/사용자오답률이 아니다. 개인영상·좌표를 가져오거나 학습·튜닝·후보선택에 쓰지 않았고 스칼라 진단요약만 진단용으로 보존했다.

## 마지막 보완
공개인사1개를640상반신으로 다시RTMW추출 및16초 대기후 같은행동에서 두번째단계 막힘을 재현. 공개기존FIT+RGB검토된9방향·정지구간을 화면/대기변형해서 복사한motion_head만400step/seed42 1회 학습했다. 전체머리교체는 이전팔정지1구간을퇴행시켜미채택. 최종은 상하axis0만새머리,팔쓸기/기타축은원래머리그대로. 학습에넣지않은다른공개화면조건도미완료→완료. 손목좌표교체 실험은개선없어미적용.
기존24조건+추가4조건=28유지,기존DEV정상방향32/32유지. 모두공개원본1명의파생조건이라새사용자정확도아님. 자체통합27개,기존독립작성상태34/유지진단16을root재실행. 이번별도에이전트검토없음.
손모양불확실 문구를 '참고 모양을 잠깐 유지'로 변경하여 관측누락안내와분리. RAM진단에단계별횟수/최초시각/손가락별모양불확실/방향hold·unknown·자료부족추가. UI는기존간단한한문장/요약,상세집계는API. 현재+직전5회 RAM,서버재시작시지움;이번5회의스칼라요약은진단결과에별도보존(학습금지).

## 모델과 금지
공통100단어 시간합성곱+양방향GRU 특징·손모양모델은동결. 이전보강방향머리는원래파일그대로이며새상하머리만분기. 순서/반복횟수/시작끝은명시규칙. 전체수어정답whole_word_correct=null. 단어TCN은이실시간연습에미사용. C·T3-F/기존8821/원본영상·모델보존. 사용자학습·튜닝·후보선택금지.
공유 SHA80aba3b28006270308834bf5ba88277d7b26c8dce01c11fc962ce5c21b8cea17.
기존방향 SHA810f3b0dd922e83ac12045e848e660e4c2d6ef8b6190a6eb49a15cac542ef378.
새상하 SHA{admission['sha256']}. seated_diagnosis_20261116/VERTICAL_ADMITTED.json이새상하적용기준. action_integration/ADMITTED.json은원래방향기준으로계속유효. 이전usability ADMITTED 코드해시는당시기록이다.

## 남음 / 다음 한 작업
{next_action}
손모양불확실은아직해결안됨. 실제RTMW오추정과모델오판분리필요. 엄지·정확한손바닥방향·접촉·다인/전체정답·10fps짧은첫동작한계남음. 이번개선이사용자에게편해졌다고재시험없이단정금지.

## 실행·근거
http://127.0.0.1:8822/ 일반실행 listener PID {web['listener_pid']}, launcher34204. 서버시작만하고카메라는사용자버튼으로만연다. 이번root는공개재생만브라우저확인,실제카메라미개방. 재개시SERVER.json과실제프로세스확인. 제한환경에서일반서버가안보일수있음.
단위 research/front_keyphase_v1/ops/hello_live_coach_20261115/seated_diagnosis_20261116/
최신 RESULT_KO.txt·VERIFIED.json·VERTICAL_ADMITTED.json·CANDIDATE_COMPARISON.json·INTEGRATION_CHECK.json·WEB_SERVICE_CHECK.json. 진단전용DIAGNOSTIC_ONLY/AGGREGATE는학습코드에연결금지. 학습script motion_candidate.py는이파일들을읽지않음. 첫전체머리회귀 MOTION_RESULTS.json, 수정범위 AXIS_SCOPE_DECISION.json.
기록/구현/공개검증/웹:Codex root. 그래프는상위TASK_GRAPH.json. 로그이폴더server.out.log/server.err.log. Python: ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe -X utf8 -B.
현재웹STATE research/front_keyphase_v1/ops/STATE.json의 hello_live_seated_diagnosis_20261116. root Ops/STATE는Unity계보.

## 큰 범위
시연목표 안녕하세요SEN0354/사과WORD0099/감사WORD1290. 기존4단어는공통회귀. 안녕하세요원본1개는 C의 runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4. W1TB는WORD33ZIP, SEN다인원본미확보. 감사·사과후속 basic_demo_feedback_20261115/acquisition_motion_followup. 전체프로젝트완료로기록금지.
'''
(R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
assert read(O/'STATE.json')['hello_live_coach_20261115']['verification']==str(V/'VERIFIED.json')
print(json.dumps({'status':verified['status'],'pid':web['listener_pid'],'records_synchronized':True},ensure_ascii=False))
