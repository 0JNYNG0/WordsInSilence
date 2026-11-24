"""Record verified web opt-in handoff, preserving prior project history."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone, timedelta
U=Path(__file__).resolve().parent
LAB=U.parents[1]; ROOT=LAB.parents[1]
stamp=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
review=read(U/'INDEPENDENT_FINAL_REVIEW.json')
for p,h in review['source_pins'].items():assert sha(p)==h,p
assert sha(review['additional_readonly_backend_pin']['path'])==review['additional_readonly_backend_pin']['sha256']
assert read(U/'RUNTIME_SMOKE_001.json')['status']=='PASS'
assert read(U/'HTTP_READY.json')['status']=='PASS'
report='''후보 C 웹 시험 준비 완료

주소: http://127.0.0.1:8765/practice.html?model=c
기존 단독 모드: http://127.0.0.1:8765/practice.html?model=m0
기본 URL은 기존 M0 유지. C는 명시 선택한다. 같은 촬영의 C와 M0 결과를 나란히 비교한다.
100단어 분류 모델, 현재 선택 목록은 기존 일상 표현12개 유지.

사용 순서
1. 후보 C 선택을 확인한다. 다시보기가 필요하면 카메라 연결 전에 다시보기 저장을 체크한다.
2. 카메라 연결, 참고 영상 확인, 촬영 시작. 3초 준비 후 C 검출 준비가 끝나 촬영 중으로 바뀌면 동작한다.
3. 머리와 움직이는 손이 보이게 한다. 한 손 수어의 쉬는 손은 아래 화면 밖에 있어도 된다.
4. 동작을 끝내고 촬영 종료를 누른다. C는 중간 손 누락으로 자동 종료하지 않으며 최대10초이다.
5. 같은 촬영의 C/M0 1위와 목표 단어 순위를 비교한다. 매번 결과는 자동 기록되고 확인 기록은 본인 메모다.
성공만 남기지 말고 실수/불확실 시도도 해당 표시로 기록한다. 촬영 자료는 학습·튜닝·모델 선택에 쓰지 않는다.

C의 차이
같은 280채널 TCN 구조와 같은100단어/학습899개 분리를 유지했다. 모델을 크게 바꾼 것이 아니라 연습 환경을 바꿨다.
기존 영상의 아래쪽 손이 잘 보이는 환경과 책상 웹캠의 아래쪽 손이 안 보이는 환경 차이를 줄인다.
전체 RGB에서 검출 후 상반신 모델 범위만 남기는 관측과, 실제 아래쪽 RGB를 잘라 검출기부터 손을 못 보게 한 관측을 학습 때50:50으로 사용했다.
화면 밖 좌표는 값/관측마스크 모두 차단한다. 내부 빈 프레임과 실제 시간 간격은 유지하며 손을 가짜로 채우지 않는다.
손 좌우 배정은 고정 joint_v3의 현재 몸손목·어깨 기준 및 이전 손 이동 근거를 사용한다. 겹침이 해결되지 않으면 보류할 수 있다.
실제 웹캠은 보이는 전체 영상만 검출하고 그 안 좌표만 사용한다. 화면 밖의 내려놓은 손목을 추정하거나 얼굴·머리 위를 추가로 자르지 않는다.

성능 근거와 한계
이전 완료 개발검증의 같은 실제 잘림300개: 기존M0 159/300=53.00%, 이번 고정C(seed20261013)295/300=98.33%, +45.33퍼센트포인트.
3seed C평균98.67%(98.00~99.67), 기존M0 대비+45.67퍼센트포인트. 원본영상 마스크 보기 평균98.44%.
전체 처리·재학습 효과가 포함된 비교다. B(마스크만 학습) 대비 C(실제 검출 누락까지 혼합)의 추가효과 평균+1.11퍼센트포인트이며 이번 고정seed는 B와 동률이다.
C의 Top5 평균은 B보다0.11퍼센트포인트 낮고 원본보기 개별회귀도 있었다. 모든 경우 개선을 보장하지 않는다.
DEV300/3사람은 후보 선택에도 쓴 개발검증이며 실제 웹캠/새 사용자 정확도가 아니다. 이번 서버연결 작업에서 이 점수를 재채점하지 않았다.
화면 안의 손도 MediaPipe가 놓칠 수 있고 겹침/좌우 배정 오류는 남아 있다. 잘린 필수 동작을 복원하지 못한다.
TCN이 단어 이름을 맞혀도 손모양까지 정답이라는 뜻은 아니다.
DTW 수치와 학습9명 참고은행은 원본 그대로다. C 가중치와 직접 비교하는 수치가 아니며 한손 누락에 따른 판단보류 문제는 별도 후속과제다.
두 검출기를 같은 RGB에 실행하므로 실제 웹캠 속도도 사용자 시험에서 확인해야 한다. 카메라는 아직 사용자가 켜기 전이다.

검증
후보 구현 자체10검사: 기존TRAIN3관측410프레임/54,320채널 값 동결처리와 일치.
별도 담당 독립 합성통합19/19PASS: 같은 촬영·시계, 분리저장, hash, 오류격리, 경합, 숨긴점 불변, DTW 원핀 유지.
실제 실행 연동: 검은 합성RGB4장으로 MediaPipe 생성/처리/종료 확인, 사전고정TRAIN 냉커피1개로 C와M0 각1번 추론 정상. 카메라0/DEV0/학습0.
기존 session시간4검사와 결과저장실패 회귀검사 ALLPASS. frontend DOM smoke 및 실제 브라우저 C↔M0 선택 확인.
서버 API에서 두 모델 READY 확인. 모델 가중치·DTW 코드/기준은행 불변.
구현 Codex root/roi_training_protocol/roi_data_budget, 독립검토 roi_missing_protocol.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')
result=dict(status='READY_FOR_USER_WEBCAM_OPT_IN',updated_kst=stamp,
    unit='ops/word100_candidate_c_web_20261017',url='http://127.0.0.1:8765/practice.html?model=c',
    baseline_preserved=True,baseline_default=True,same_capture_pipeline_comparison=True,
    C_seed=20261013,C_epoch=77,C_sha256='283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab',
    M0_sha256='e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f',
    user_data_use='evaluation_only_no_training_tuning_or_selection',new_training=0,DEV_rescoring=0,agent_camera_access=0,
    server=read(U/'SERVER.json'),independent_checks=19,self_checks=10,
    runtime_smoke='RUNTIME_SMOKE_001.json',independent_review='INDEPENDENT_FINAL_REVIEW.json',
    browser_verified=True,webcam_accuracy_verified=False,
    remaining=['actual webcam evaluation and capture speed','inside-frame detector occlusion/assignment errors','unchanged DTW missing-hand support limits'],
    next_single_action='User records with candidate C; aggregate frozen paired C/M0 stored results without training, tuning, threshold or candidate selection.')
paths=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',LAB/'ops/STATE.json']
for p in paths:
    dest=U/'before'/p.name
    if not dest.exists():dest.write_bytes(p.read_bytes())
(U/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
(U/'ENVIRONMENT_NOTES.json').write_text(json.dumps(dict(
    pytest_unavailable='No install; existing standalone scripts used instead.',
    legacy_test_first_run='No output while using default tempfile location, own PID47412 stopped. Re-run with workspace tempfile directory passed ALL checks.',
    server_first_pid=49304,server_current_pid=read(U/'SERVER.json')['pid'],
    actual_camera_not_opened=True),ensure_ascii=False,indent=2),'utf-8')
state=read(paths[2]);state['updated_date_kst']=stamp
state['current_stage']='WORD100_C_WEB_OPT_IN_READY_WAITING_USER_TRIALS'
state['next_single_action']=result['next_single_action']
state['word100_candidate_c_web_20261017']=result
paths[2].write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8')
context=paths[0].read_text('utf-8')
summary=f'''\n### 2026-10-17 후보 C 웹 선택 시험 연결 완료\n갱신 {stamp}. 사용자 최신 승인으로 C(seed20261013/epoch77/SHA283a54bd) 명시 선택 웹 모드를 연결했다. 기존M0/SHAe39d8ac4와 기본모드 보존. 같은 RGB 촬영을 각 동결 입력처리로 C/M0 비교, 별도NPZ·SHA·모델정보 저장. C매촬영 fresh detector+joint_v3+실제 가시화면 strictROI; 내부미검출 보존, C수동종료/최대10초. 기존DTW 원코드/은행 유지. 사용자자료 학습/튜닝/선택0. 자체10검사/TRAIN3관측410프레임·54,320값 동등, 독립합성19/19PASS, 실제blank4프레임검출+TRAIN1개 C/M0forward PASS, 브라우저선택/API두모델READY. 새학습/DEV재채점/카메라접근0. PID{result['server']['pid']} 일반실행8765, 카메라는 사용자클릭 때만. 근거 ops/word100_candidate_c_web_20261017/RESULT.json, RESULT_KO.txt, INDEPENDENT_FINAL_REVIEW.json. 실제웹캠 정확도/속도와 DTW한손지원은 미검증·후속.\n'''
marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
assert marker in context
context=context.replace(marker,marker+'\n'+summary,1)
marker='## 9. 다음 단일 작업'
next_text='''\n### 2026-10-17 후보 C 사용자 웹 시험 및 저장결과 집계\nhttp://127.0.0.1:8765/practice.html?model=c 에서 사용자가 직접 촬영한다. 같은 촬영 C/M0의 정답순위·실패·손관측 상태를 저장결과로 집계하되 사용자자료를 학습·튜닝·임계값·후보선택에 쓰지 않는다. 성공만 선별하지 않는다. 기존M0/DTW는 보존, 전역모델교체·재학습·보류사람평가 없음. 서버 PID/HTTP상태부터 확인해 중복실행하지 않는다. 이전 비교완료 대기 지시는 이 후속승인으로 갱신되었다.\n'''
assert marker in context
paths[0].write_text(context.replace(marker,marker+'\n'+next_text,1),'utf-8')
paths[1].write_text(f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

## 현재 목표와 마지막 완료
사용자 승인: 후보 C를 웹 시험에서 선택, 기존 모델 보존·비교, 본인 촬영 학습·튜닝 금지.
C 웹 선택 연결 완료. URL http://127.0.0.1:8765/practice.html?model=c . 기본 URL/M0 선택은 기존 그대로.
고정 C seed20261013 epoch77 SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab. M0 SHAe39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f.
같은 RGB 촬영을 별도 detector·입력처리로 C/M0 비교. C는 fresh per-take joint_v3+strict 실제 화면경계, 내부누락 유지. 완료입력의 기존 active_span→280채널 유지. C만 수동종료/최대10초로 중간손누락 자동종료 방지.
원본관측 .npz는 M0/DTW용, .candidate_c.npz는 C용. 별도 hash와 모델정보를 결과에 저장. 기존 DTW 원코드/참고은행은 복원·원핀검증됐고 새 word100_web_analysis wrapper로 C분류와 분리했다.

## 검증과 실행
구현 자체10검사, 기존TRAIN3 raw410프레임/54,320채널 동결입력동등. 독립합성19/19PASS, 실제MediaPipe 검은RGB4장 및 고정TRAIN1개 C/M0forward PASS. 실제사용자/카메라/DEV추론/새학습0. 기존시계4검사/저장실패회귀 ALLPASS. frontend DOM smoke 및 실제브라우저 C↔M0 선택, API두모델READY.
일반실행 서버 PID{result['server']['pid']}, 포트8765, 카메라미개방. 로그 ops/word100_candidate_c_web_20261017/server.v2.*.log 및 SERVER.json/HTTP_READY.json.

## 다음 한 작업
사용자가 직접 시험한 저장결과를 같은촬영 C/M0로 집계·보고한다. 촬영기록을 학습/튜닝/후보선택/임계값수정에 사용하지 않는다. 각 시도의 성공/실수 모두 기록. 서버 상태 먼저 확인, 중복실행금지. 사용자는 동작하는손·머리가 화면에 보이게, 쉬는손은 밖이어도 됨. C촬영중 표시 확인 후 수행하고 종료버튼(최대10초).

## 이전 완료 비교의 정확한 해석
동결DEV300/3사람 실제잘림 M0 53.00%→C3seed평균98.67%(98.00~99.67), 전체처리효과+45.67%p. 웹고정seed C는295/30098.33%, B와동률. B→C 추가혼합효과평균+1.11%p. 원본보기 C평균98.44%, 일부개별회귀3/1/4; C Top5는B보다0.11%p낮음. 개발선택에도 쓴 자료이며 새웹캠정확도 아님. 영역내검출누락·겹침·DTW한손근거부족 잔존. 각영상노출은 동결코드+저장합계검산, 배치별실측없음.

## 근거와 유지조건
새단위 ops/word100_candidate_c_web_20261017/RESULT.json, RESULT_KO.txt, INDEPENDENT_FINAL_REVIEW.json, RUNTIME_SMOKE_001.json.
기존연구 ops/word100_body_boundary_20261017/FINAL_RESULT.json 및 동결자료 원본보존. TRAIN899/DEV300/보류14~18분리 유지. 기존 st-gcn 후속자동화 삭제상태유지.
Git·다운로드·외부전송·원본/W:/sources수정 없음. 새 운영기본교체/추가학습/최고seed선택 승인아님. 구현root/roi_training_protocol/roi_data_budget, 독립roi_missing_protocol.
''','utf-8')
for p in (U/'RESULT.json',LAB/'ops/STATE.json'):read(p)
print('RESULT and current context/STATE/HANDOFF synchronized',stamp)
