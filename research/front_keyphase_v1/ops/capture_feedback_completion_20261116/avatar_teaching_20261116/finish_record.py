from pathlib import Path
import json,hashlib,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
U=Path(__file__).resolve().parent;G=U.parent;O=G.parent;R=O.parents[2];at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();rel=str(U.relative_to(R)).replace('\\','/')
def write(name,x):(U/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
pack=json.loads((U/'PACKAGED.json').read_text('utf-8'));motion=json.loads((U/'MOTION_VERIFICATION.json').read_text('utf-8'));pres=json.loads((U/'PRESERVATION_VERIFIED.json').read_text('utf-8'));assert pres['status']=='PASS'
visual=dict(reviewer='Codex root; same implementer performing source/side-view review, not independent human sign-language expert',
 evidence=['WORD0099_COMPARISON.jpg','WORD1290_COMPARISON.jpg','WORD0099_FULL_FLOW.jpg','WORD1290_FULL_FLOW.jpg','WEB_APPLE.jpg','WEB_APPLE_WIPE.jpg','WEB_THANKS.jpg'],
 checks={
 'WORD0099':{'shape':'Extended right index before mouth sweep; then left closed fist and right curved open palm. Source native F/L/R/U/D key frames reviewed.','location':'Character mouth-relative index target corrected after initial shoulder-only transfer was too low. Wipe shown in front of torso.','direction_order':'Mouth sweep toward performer right precedes downward wipes over left fist; original 116-frame order retained.','corrections':'Mouth landmark and left-fist palm-local hold through overlap; explicitly inferred/artist corrected, not measured occluded finger ground truth.'},
 'WORD1290':{'shape':'Left hand extended with back upward, right hand extended with little-finger side downward; front and oblique browser view reviewed.','location':'Both hands before upper torso; right hand descends toward back of left.','direction_order':'Two principal down/up motions in source and animation (around frames60/61 and74); initial small dip51 is lead-in, not a third required tap. Original135-frame order retained.','corrections':'No word-specific pose correction; character arm-length transfer only.'}},
 limitations=['Camera intrinsics/extrinsics estimated from videos, not official calibrated camera data. Reprojection is detector consistency, not measured3D accuracy.','No exact mesh-contact distance ground truth; appearance/contact and fine joint smoothness remain approximate.','Body/face expression remain static; this is a two-word hand/arm demonstration, not full non-manual sign reproduction.','No expert linguistic certification or new-user webcam validation performed.'])
write('VISUAL_REVIEW.json',visual)
web=dict(status='PASS_OBSERVED_LOCAL_UI',checks=[
 'Apple/gratitude GLB correctly selected on practice word change; no automatic camera opening.',
 'Standalone and practice-modal source/character comparison loaded both source videos and corresponding GLBs.',
 'Play/pause, restart,0.5/1 speed, timeline seek and chapter seek, rotation, zoom and reset inspected.',
 'Apple chapter2 source2.166666s vs rendered pose2.167s at paused browser observation; continuous playback also followed video time.',
 'Parent scene pauses while comparison/feedback dialog open; comparison iframe unloads to about:blank on close.',
 'Existing captureLocked GPU-priority freeze retained; no physical capture test this turn.',
 'Desktop1365px practice and default1280px comparison checked. Narrow319px practice comparison button initially overlapped dialogue; shortened main controls, reloaded and observed modal open=true. Full small-phone visual polish not certified.',
 'Existing public apple-normal result displayed seven similar cards and correct summary; gratitude cut result retained uncertainty/grey summary, not false success. Existing cached public results, not a new accuracy benchmark.',
 'No console errors in inspected practice/comparison pages; node syntax checks passed scene/live/sign-demo.',
 'No webcam opened; no private recordings read or trained on.'],
 snapshots=['WEB_APPLE.jpg','WEB_APPLE_WIPE.jpg','WEB_THANKS.jpg'],
 reviewer='Codex direct browser UI actions and read-only DOM checks')
write('WEB_VERIFICATION.json',web)
admit=dict(status='ADMITTED_TWO_WORD_DEMONSTRATION_WITH_SOURCE_COMPARISON',at=at,scope=['WORD0099','WORD1290'],
 assets=pack['catalog'],motion_verification='MOTION_VERIFICATION.json',visual_review='VISUAL_REVIEW.json',web_verification='WEB_VERIFICATION.json',preservation='PRESERVATION_VERIFIED.json',
 source='Existing public AIHub02_real_word_video.zip five native views, same clip per word; front detections reused and four auxiliary views extracted with existing RTMW/OpenVINO GPU.',
 frame_counts={'WORD0099':116,'WORD1290':135,'new_pose_inference_frames':1004,'reused_front_frames':251},
 failures_corrected=['Apple initial camera fit had unstable limb lengths; retained failed derivative and used seed/refit camera solution on apple itself.','Apple mouth transfer too low and occluded left fist opening: separately recorded retarget-only corrections.','GLB export started at Blenderframe1=1/30s; normalized to sourceframe0 and verified every exported joint/native frame.','Narrow practice controls overlapped dialogue; main compact controls corrected, full controls remain in comparison viewer.'],
 training_runs=0,recognition_or_feedback_model_changed=False,camera_opened=False,private_data_used=False,expert_review=False,
 next='User may now watch demonstrations and later capture apple/gratitude; actual webcam automatic termination/end-to-end feedback remains deferred user validation, not animation task incompletion.')
write('ADMISSION.json',admit);pack['status']='ADMITTED_WITH_REVIEW_LIMITATIONS';write('PACKAGED.json',pack)
write('CHECKPOINT.json',dict(status='COMPLETE_ANIMATION_SOURCE_REVIEW_AND_PRACTICE_CONNECTION',at=at,stages={x:'COMPLETE_WITH_DOCUMENTED_LIMITATIONS'for x in ['source_five_view_audit','motion_retarget','source_handshape_location_direction_order_review','web_practice_and_comparison']},active_process=None,server='http://127.0.0.1:8821/',camera='not opened',next=admit['next']))
report='''사과·감사 3D 시범 제작과 연습 화면 연결 완료

사용 방법
http://127.0.0.1:8821/ 새로고침 → 02 연습 → 사과/감사 선택 → 동작 재생.
「원본과 나란히」를 누르면 공개 원본과 캐릭터를 같은 시점으로 볼 수 있다.
0.5배/1배, 반복, 처음부터, 시점 슬라이더, 주요 동작 바로가기, 회전·확대 제공.
촬영 중에는 기존처럼 3D 배경을 멈춰 RTMW 관측에 GPU를 우선 사용한다.

원본과 제작
기존 공개 AIHub 5방향 영상에서 사과116프레임/감사135프레임(30fps)을 사용했다.
정면251프레임 관측은 같은 원본/시점임을 대조해 재사용하고, 다른4방향1004프레임은 기존RTMW GPU로 추출했다.
영상에서 추정한 카메라로3D를 복원하고 기존Ch29/Mixamo65뼈대·외형에 옮겼다. 공식 카메라 보정값 또는 측정된3D정답은 아니다.
사과와 감사의 REAL01은 서로 다른 실제 사람이다. 같은 단어 안의5방향만 같은 촬영 기록으로 묶었다.

원본 대조와 수정
사과: 검지로 입술 아래 스치기 → 왼 주먹 위를 오른 손바닥으로 아래로 닦기 순서.
캐릭터 목/얼굴 비율 때문에 입보다 손이 낮아져 입 기준 위치를 보정했다.
겹침 중 왼손 주먹이 풀려 보이는 문제는 원본의 겹치기 전후 주먹 유지 근거로 손모양을 유지했다.
이는 시범 애니메이션 보정이며, 가려진 손가락을 실제로 측정했다고 주장하거나 학습 정답으로 쓰지 않았다.
감사: 왼손등이 위, 오른손을 세운 상태에서 아래로 두 번 이어지는 흐름을 정면/옆면과 원본으로 대조했다.
원본과GLB의 한 프레임 시작차이를 수정하고 두 동작의 모든프레임×65뼈 위치가 Blender 결과와 일치함을 검산했다.
이 검산은 파일 변환 정확도이지 실제 수어 동작 정확도가100%라는 뜻이 아니다.

웹 확인
두 단어 선택, 동기 재생, 느리게, 회전·확대, 동작별 이동, 모달 열기/닫기를 실제 브라우저에서 확인했다.
사과 정상 공개 결과의7항목 비슷함, 감사 중간 종료 편집결과의 확인 부족 표시도 기존처럼 나왔다.
촬영/TCN/피드백/퀘스트 처리는 그대로이며 live.js는 시범 선택을 전달하는 onScene만 변경했다.
서버 코드와 기존 사과·감사 판정ADMISSION 해시 유지. C·T3-F·신경망 가중치·판정 기준 교체 없음. 개인자료 접근/학습0.

범위와 다음 확인
사용자가 따라 볼 수 있는 두 단어 시범/웹 연결을 완료했다. 모든 단어 또는 전체 프로젝트 완성 판정은 아니다.
손 접촉과 미세한 관절 움직임은 추정 복원/캐릭터 체형의 한계가 있고 표정·몸통 표현은 고정이다.
Codex가 원본과 화면을 대조했으며 농인강사 등 외부 전문가 검수는 하지 않았다.
사용자의 실제 촬영 자동 종료·새 행동 교정·전체 대기시간 확인은 기존처럼 사용자가 나중에 촬영할 때 진행한다.
이번 작업으로 인식률 향상이나 실제 사용자 정확도를 새로 입증했다고 보고하지 않는다.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')
before=U/'before_closeout';before.mkdir(exist_ok=True)
for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json',R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']:
 if not(before/p.name).exists():shutil.copy2(p,before/p.name)
summary='사과·감사 공개5방향 기반 Ch29 시범2개 제작, 원본 손모양/신체위치/방향/순서 대조, 8821 연습실 단어별 시범 및 원본·캐릭터 동기 비교 연결 완료. 사과 입 위치/겹친 왼주먹은 시범만 보정했고GLB1프레임시차도 수정. 새RTMW보조4뷰1004프레임/정면251재사용; 새학습0. 기존TCN/피드백/C·T3-F 보존. Codex 직접검산·시각검수이며 외부수어전문가/실제웹캠검증 아님.'
next_action='사용자가 8821에서 사과·감사 캐릭터/원본을 본 뒤 촬영할 때 자동종료·정상/교정/재시도와 실제대기시간 확인. 현재 사용자촬영은 나중에 한다고 했으므로 준비 유지. 개인자료 학습/튜닝/선택 금지. 새로운 단어·모델 확대나 완료애니메이션 재생성은 요구/구체문제 없이 반복하지 않는다.'
context=R/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text('utf-8');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업';assert h8 in s and h9 in s
s=s.replace(h8,h8+'\n\n### 2026-11-16 사과·감사 3D 시범과 원본 비교 연결 완료\n'+summary+'\n근거 '+rel+'/ADMISSION.json·MOTION_VERIFICATION.json·VISUAL_REVIEW.json·WEB_VERIFICATION.json·PRESERVATION_VERIFIED.json. 원본검증/추정복원/시범보정을 구분. 카메라는 열지 않음. 다음 실제촬영확인은 사용자 대기.\n',1)
s=s.replace(h9,h9+'\n\n### 2026-11-16 시범을 보고 실제 촬영 확인 / 준비 유지\n'+next_action+'\n이번 시범 구현 완료범위는 '+rel+'/CHECKPOINT.json. 전체피드백 목표/외부수어검수/전체HDD분리실행까지 완료로 확대하지 않음.\n',1);context.write_text(s,'utf-8')
statefile=O/'STATE.json';state=json.loads(statefile.read_text('utf-8'));state['next_single_action']=next_action;state['last_updated']=at;task=state['capture_feedback_completion_20261116'];task['next_single_action']=next_action;task['teaching_reference_review'].update(animation_complete=True,web_modified=True,status=admit['status'],report=str(U/'RESULT_KO.txt'),admission=str(U/'ADMISSION.json'),verification=str(U/'WEB_VERIFICATION.json'),limitations=visual['limitations'],training_runs_this_task=0,source_frames=251,auxiliary_pose_frames=1004,camera_opened=False,updated_at=at)
statefile.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {at}
## 현재 목표와 완료
{summary}
사과→감사 시연 유지, 안녕하세요 제외. 기존4단어는 연구 회귀 선택지. 이번 사용자 명시요청은 두 단어3D시범→원본검증→연습연결이며 완료했다.
## 지금 확인할 근거
{rel}/CHECKPOINT.json·RESULT_KO.txt·ADMISSION.json.
MOTION_VERIFICATION:251프레임×65뼈GLB변환 일치. 실제3D정답/수어정확도100%가 아님.
사과guided_v2·감사baseline 사용. 사과의 최초 불안정 카메라fit은 미사용, calibration_seed_v2를 사과자체자료로 재적합. 입위치와 왼주먹유지 보정은 애니메이션만 적용. TIME_ORIGIN_CORRECTION.json에 GLB1프레임차이 정정. VISUAL_REVIEW·WEB_VERIFICATION에 한계와 실제검수 기록.
## 웹
http://127.0.0.1:8821/ →02연습→사과/감사→동작 재생/원본과 나란히.
별도 비교: /sign-demo.html?word=WORD0099 또는WORD1290. 원본과캐릭터 같은시간 재생/0.5배/회전·확대/구간선택. 모델/서버 재시작 없이정적UI연결; 마지막브라우저응답정상. 다음재개시서버실제상태확인. 루트START_WEB_DEMO.cmd 유지.
PRESERVATION_VERIFIED.json: 서버와기존판정ADMISSION 해시 유지, live.js는onScene시범선택신호만 변경. 촬영GPU우선정지 유지. 공개사과정상/감사부분시험 기존결과표시 확인. 카메라열기/개인촬영/학습/새다운로드0.
## 다음
{next_action}
진행중 추출/Blender/학습프로세스 없음. 외부수어전문가검수·실제사용자웹캠·전체HDD분리실행은 미완료. 사용자촬영을 자동시작하지 않는다. 애니메이션GLB/원본참고는C:에서서빙하되 전체시연HDD독립을 새로보장하지 않는다.
''','utf-8')
hist=R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt';hist.write_text(hist.read_text('utf-8')+f'\n\n{at} 사과·감사 따라 배울 3D 시범 연결\n사용자가 동작을 몰라 촬영테스트를 할 수 없다는 문제에서, 모델교체보다 따라 볼 시범을 먼저 준비했다. '+summary+'\n원본5뷰→추정카메라/삼각복원→기존Mixamo65뼈대→원본/정면/옆면 검수→GLB변환검산→웹 단어선택/동기비교. 시범 제작과 AI피드백학습은 다른 작업이며 이번은 새학습0. 원본은5뷰지만 정확한3D를측정했다고과장하지 않음. 실제촬영검증은 사용자대기. 근거 '+rel+'/RESULT_KO.txt\n','utf-8')
assert json.loads(statefile.read_text('utf-8'))['capture_feedback_completion_20261116']['teaching_reference_review']['animation_complete']
for n in ['ADMISSION.json','VISUAL_REVIEW.json','WEB_VERIFICATION.json','CHECKPOINT.json','PACKAGED.json']:json.loads((U/n).read_text('utf-8'))
print('Recorded complete bounded animation/UI task, preservation, limitations, and deferred actual webcam verification.')
