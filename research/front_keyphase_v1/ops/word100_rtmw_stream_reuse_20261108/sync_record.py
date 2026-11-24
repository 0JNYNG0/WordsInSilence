from pathlib import Path
import json,shutil
from datetime import datetime,timezone
U=Path(__file__).resolve().parent;R=U.parents[3];key=U.name
state_path=U.parent/'STATE.json';context_path=R/'WordsInSilence_PROJECT_CONTEXT.md';handoff_path=R/'NEW_CHAT_HANDOFF.md'
state=json.loads(state_path.read_text(encoding='utf-8'));v=json.loads((U/'VERIFICATION.json').read_text(encoding='utf-8'));hv=json.loads((U/'HTTP_VERIFICATION.json').read_text(encoding='utf-8'))
assert v['status']==hv['status']=='PASS'
if key in state:raise RuntimeError('Already synchronized; inspect rather than duplicate')
before=U/'before';before.mkdir(exist_ok=False)
for p in [state_path,context_path,handoff_path]:shutil.copy2(p,before/p.name)
next_action='실제 브라우저의 촬영 프레임 전송·녹화 PTS와 RTMW 관절시각을 연결하고 공개 재생으로 누락/지연/취소를 검증한다. 기존8812 통합은 아직이며 관절전용 피드백망은 입력·감독정답 조건 고정 후 별도학습. C·T3F/사용자자료제한 유지; 과거AO학습 자동재개하지 않음.'
state[key]=dict(status='STREAM_CORE_AND_LOOPBACK_VALIDATED_BROWSER_INTEGRATION_PENDING',at=datetime.now(timezone.utc).isoformat(),
    report=str(U/'RESULT_KO.txt'),plan=str(U/'PLAN_KO.txt'),verification=str(U/'VERIFICATION.json'),http_verification=str(U/'HTTP_VERIFICATION.json'),
    first_stage=v['totals'],http_frames=hv['frames'],http_post_capture_ms=hv['post_capture_ms'],
    new_training=0,user_material_used=False,existing_web_changed=False,camera_used=False,
    chosen_feedback_direction='pose-only GuideFeedbackNet temporal-convolution + GRU candidate; SignRep excluded from always-on new candidate, old checkpoint preserved; not yet trained',
    authority='2026-11-08 user requested full plan and immediate RTMW streaming/reuse implementation',
    previous_research_pause_preserved=True,next_single_action=next_action)
state['current_stage']='RTMW_STREAM_CORE_VALIDATED_BROWSER_AND_POSE_FEEDBACK_PENDING'
state['next_single_action']=next_action
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=context_path.read_text(encoding='utf-8')
status='''### 2026-11-08 촬영 중 RTMW 입력·재사용 부품 및 로컬 전송 검증

사용자가 게임의 빠른 흐름을 우선해 SignRep 상시 RGB 분석 제외 방향과 촬영 중 RTMW 추출·재사용부터 구현을 승인했다. 기존 GuideFeedbackNet의 시간 합성곱+GRU를 활용한 관절전용 후보→학습된 흐름/DTW→기존8812 연결 순서로 PLAN_KO.txt를 기록했다. SignRep 입력만 끄는 방식이 아니며 관절전용 판정부 학습은 아직 실행하지 않았다. 이전 AO 학습 자동재개가 아닌 새 입력 단위다.

word100_rtmw_stream_reuse_20261108에서 GPU 상주엔진·최대8프레임큐·원본시각·명시적누락감사·관절재사용을 구현했다. 공개6영상(순종02/잘하다02/조상02/수어02/후원05/남아02)×2회1414프레임 전부 처리, 기존동일픽셀 RTMW좌표 최대차이0px, 재사용추가검출0. 입력→좌표완성p95 31.53~32.34ms, 마지막입력→저장/재사용53.21~67.66ms. 공개3영상360프레임 실제로컬HTTP 전송도 픽셀/좌표/시각 보존, 마지막예정촬영시각→입력응답77.92~92.01ms. 모두 소스 프레임 사전디코딩 조건이며 웹캠/브라우저읽기/게임/최종피드백 시간은 아니다. 초기GPU준비4.43초+예열3.87초 별도. 좌표보존은 엄지 정확도 개선이 아니다.

합성13검사·별도127260상대좌표 재계산·HTTP saved output검산 통과, 기존23보호파일불변. 첫npz핸들정리 오류 수정/최초기록보존. HTTP 제한환경10013은 일반환경의 카메라없는 공개로컬시험으로 해결. 같은Codex별도검산이지 독립사람검증 아님. 8812/C·T3F/SignRep/DTW 변경0, 카메라/개인자료/학습/다운로드/Git0. 시험서버 종료. 결과 RESULT_KO.txt와 VERIFICATION/HTTP_VERIFICATION.json.

'''
action='''### 2026-11-08 RTMW 스트리밍 이후 다음 단일 작업

입력부·공개 실시간 재생·로컬HTTP 검증은 완료했다. 완료시험/과거비교를 반복하지 않는다. 다음은 실제 브라우저 촬영 프레임 전송과 MediaRecorder 녹화 PTS, 관절 timestamp를 맞추는 연결 작업. 공개 영상의 브라우저 재생으로 먼저 확인하고 카메라는 사용자 조작으로만 연다. 브라우저 canvas 읽기·전송·렌더링과 기존 C·T3F에 필요한 MP 입력 보존 비용까지 포함한 지연/누락을 측정한다. RTMW 점 존재로 자동종료를 판단하지 않는다. 8812는 아직 변경하지 않았고 학습된 관절전용 피드백도 아직 없다. 그 다음 공개 FIT/DEV와 시간별 감독정답·가림마스크·얼굴부위관계 조건을 고정해 관절전용 시간합성곱+GRU 후보를 학습한다. 기존운영교체/사용자학습 금지, AO 과거중단보존.

'''
text=text.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+status,1)
text=text.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n'+action,1)
context_path.write_text(text,encoding='utf-8')
handoff='''# WordsInSilence 최신 인수인계

2026-11-08 사용자 승인: 전체 계획 정리 후 촬영 중 RTMW 추출·재사용부터 바로 구현. 최신 단위 word100_rtmw_stream_reuse_20261108. PLAN_KO.txt / RESULT_KO.txt / FROZEN.json / COMPLETE.json / VERIFICATION.json / HTTP_COMPLETE.json / HTTP_VERIFICATION.json을 따른다. 완료실행 재추론 금지.

새 방향: 게임의 촬영 후 대기시간 우선. RTMW로 촬영 중 관절과 실제시각을 모아 종료 후 다시 검출하지 않고 사용한다. 새 기본후보는 기존 GuideFeedbackNet의 시간합성곱+GRU를 활용한 관절전용 신경망. SignRep의 상시 RGB 실행은 새 후보에서 제외하고 기존 학습 결과는 비교기준으로 보존. SignRep을 끄거나0입력으로 기존모델을 그대로 쓰는 것/SignRep 가중치가 관절전용으로 자동이전되는 것이 아니다. 새 관절전용 학습은 아직0.

완료: stream_runtime.py 상주GPU/시각입력큐/명시적포화/누락감사, 원시133점+점수+PTS 보존, 관절만으로 몸/손 상대좌표·계산가능마스크 재사용. 마스크는 가시성정답 아님. 손가락시간복원은 아직 없음. stream_http.py는 로컬인증 RGBA프레임 입력 API로 기존웹에 연결할 부품이며8812에아직연결안됨. 카메라열기없음.

공개6영상707원본프레임×2=1414처리, 누락0, 기존동일픽셀RTMW좌표0px차이, 재사용추가검출0. 입력→좌표p95 31.53~32.34ms; 마지막입력→저장/재사용53.21~67.66ms. 공개3영상360프레임HTTP전송도 픽셀/좌표/점수/시각동일; 마지막예정촬영시각→응답77.92~92.01ms. 두시험모두사전디코딩프레임이며 브라우저픽셀읽기/카메라/녹화/게임/최종피드백제외. 최종피드백0.1초라고말하면안됨. GPU컴파일4.43초/예열3.87초별도.

합성13검사, 별도127260상대좌표 재계산최대1.91e-6차이, 기존23핀보존. HTTP값/시각독립스크립트확인. HTTP클라이언트원시RTT배열없으므로시간은러너집계이며독립재계산주장금지. 같은Codex별도검산이지사람정답검증아님. 최초npz잠금오류와HTTP샌드박스10013기록보존;일반환경에서카메라없는로컬HTTP검증완료. 시험프로세스/서버종료.

다음: 실제브라우저 프레임전송·MediaRecorder PTS·관절시각연결을공개영상으로검증. 8812에통합하기전브라우저읽기/전송/게임부하와C·T3F용기존MP입력보존비용까지측정. RTMW는화면밖손에도점을출력할수있어점존재만으로자동종료하면안됨. 취소/지속가림/한손·양손완료확인. 그뒤관절전용학습용FIT/DEV/감독정답/얼굴관계/마스크고정후학습. 사용자카메라자동열기/개인자료학습·튜닝·후보선택금지.

기존8812코드·C·T3F·SignRep·DTW변경0. 이전RTMW비교단위 word100_rtmw_mediapipe_compare_20261108의24공개영상/2738프레임과후보모델을읽기전용사용. 그결과는손누락보완과엄지퇴행이함께있으며정확도전면우세/지터개선주장불가. SimCC raw score는확률아님. 기존AO word100_motion_aware_order_20261103 중단은유지;이번새입력작업이AO학습재개아님.
'''
handoff_path.write_text(handoff,encoding='utf-8')
assert key in context_path.read_text(encoding='utf-8')
assert json.loads(state_path.read_text(encoding='utf-8'))[key]['new_training']==0
print('Synchronized context, STATE, latest handoff')
