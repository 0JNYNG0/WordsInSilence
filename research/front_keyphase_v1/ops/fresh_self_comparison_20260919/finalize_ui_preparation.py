"""Record verified operations; no application implementation or model work."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

ops=Path(__file__).resolve().parent
root=ops.parents[3]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
audit=read(ops/'applied_ui_audit.json')
assert audit['status']=='PASS' and audit['passed']==21
flow=read(ops/'ui_flow_verification_v4.json')
assert flow['passed']==flow['total']==33
history={}
for name in ['claude_completion_after_launcher_timeout','claude_html_v2_completion','claude_js_v2_completion','claude_fixes_v3_completion','claude_fixes_v4_completion','claude_preview_v5_completion']:
    history[name]=read(ops/(name+'.json'))
progress={
 'status':'CAPTURE_UI_FUNCTIONALLY_READY_CAMERA_CLOSED_MINOR_COSMETIC_ISSUE',
 'at_kst':stamp,
 'protocol_id':'self_frozen_20260919_v1',
 'planned_words':18,'rounds':3,'planned_slots':54,'frozen_models':3,
 'implementation_author':'Claude Code; exact generated v4 files applied by Codex',
 'claude_execution_history':history,
 'claude_active_worker':None,
 'initial_response_disposition':'Child terminated; initial response incomplete and unusable. Preserved, not applied. Completed v2/v3/v4 supersede it.',
 'ui_flow_checks':[33,33], 'applied_boundary_and_mock_event_checks':[21,21],
 'applied_sha256':audit['applied_sha256'],
 'browser_report':'BROWSER_REVIEW.md',
 'web_server':{'tool_session':18876,'url':'http://127.0.0.1:8765/frozen-capture.html','last_verified_camera':None,'last_verified_camera_session_id':None,'left_running_for_user':True},
 'mock_server':{'port':8766,'pid':34008,'state':'STOPPED_AFTER_EVENT_EVIDENCE_SAVED','synthetic_trials':2,'hardware_calls':0},
 'actual_camera_capture_verification':'NOT_RUN_USER_READINESS_REQUIRED',
 'camera_sessions_created':0,'new_human_trials':0,'new_training_runs':0,'new_inference_calls':0,
 'existing_models_backend_default_ui_and_protocol_unchanged':True,
 'cosmetic_issue':{'issue':'When camera is closed, preview without src shows a broken icon and mirrored alt text. Does not activate camera or block capture controls.','fix_status':'NOT_APPLIED','claude_v5_exit_code':1,'error':'ConnectionRefused','same_condition_retry':False},
 'not_yet_done':['Participant readiness and actual new captures','Offline three-model evaluator for new cohort including incomplete slots','Three-model comparison on new observations','Minor empty-preview visual cleanup'],
 'next_single_action':'Present the camera-off page. Participant explicitly starts only when ready; do not activate camera on their behalf. Prepare offline comparison for new54-slot cohort using original server reviews and same observations across frozen parent/control/treatment. Preserve errors, cancellations and unconfirmed/missing attempts. No existing18 replay, training, default promotion or full-test repeat.'
}
(ops/'PROGRESS.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
report=f'''# 本人 새 촬영 비교 화면 준비 결과

{stamp}. 기능 준비 완료이며 실제 사용자 촬영은 아직 시작하지 않았다.

## 완료

- Claude Code가 별도 `web/frozen-capture.html`/`.js`를 작성했고 Codex가 적용·독립 검증했다. 같은 카메라·MediaPipe·관절 저장·참가자 확인 API를 사용한다.
- 18단어 × 3차례 = 54슬롯과 기존 visibility 부모/특징 추출부 고정 대조군/사과9건 추가군을 촬영 전에 고정했다. 같은 촬영을 세 모델에 적용하며 촬영 중 예측은 보여주지 않는다.
- 합성 흐름33/33: 중복 생성 방지, 저장 실패/복구, 취소·오류 보존, 종료 이후 늦은 응답, 새로고침 비재개와 전체54슬롯을 확인했다. 실제 Chrome에서도 주요 흐름을 조작했고 이벤트/보존 경계21/21을 확인했다. 검사는 화면 기능의 근거이며 새 인식 성능 결과가 아니다.
- 준비 화면에서 첫 공부 시범이 보이고 카메라는 꺼져 있다. 실제 서버는 `http://127.0.0.1:8765/frozen-capture.html`에 남겼다. 합성 서버8766은 종료했다.
- 보호 대상13개(세 모델, backend, 입력 처리, 기존 화면, 고정 계획)의 해시는 같았다. 기존 검증·학습·추론을 반복하지 않았고 사용자 촬영0, 새 학습0, 새 추론0이다.

## 실패 보존과 남은 제한

첫 Claude 대기 보조 종료는 실제 자식 종료와 달랐다. 자식은 이후 종료했으나 응답이 불완전해 적용하지 않았다. 완료된 v2와 최소 v3/v4 수정으로 대체했고 각 실패 증거를 보존했다. 현재 해당 worker가 실행 중이라는 과거21:40 기록은 더 이상 현재 상태가 아니다.

카메라 시작 전 빈 미리보기에 깨진 아이콘/반전된 대체 텍스트가 보이는 작은 표시 문제가 남았다. 별도 CSS 요청 v5는 ConnectionRefused로 종료1이어서 적용하지 않았고 같은 조건에서 재시도하지 않았다. 촬영 버튼/저장 흐름에는 영향을 주지 않는다. 실제 카메라를 연 촬영 검증은 사용자 준비 후에 진행한다.

## 다음

사용자가 준비됐을 때 화면에서 직접 카메라를 열고 새 촬영을 진행한다. 사용자 기록은 평가용으로만 사용한다. 새54슬롯을 원래 서버 확인 기록과 대조하고 동일 관측을 세 모델에 넣을 오프라인 평가 연결은 아직 남아 있다. 부모step220과 두 후보step660, 취소·오류·미확인·미촬영 분모를 처리해야 하므로 기존18건 재생 러너를 그대로 돌리지 않는다.

한 학습 미참여자의 새로운 수행에 대한 개발 비교이며 다인 일반화나 올바른 수행 판정은 아니다. 기본 모델은 자동 승격하지 않는다. 다른 참가자 시험은 사용자 결정대로 이후에 진행한다.

근거: `PROTOCOL.md`, `protocol_LOCAL_ONLY.json`, `model_readiness.json`, `ui_flow_verification_v4.json`, `BROWSER_REVIEW.md`, `browser_mock_events.json`, `post_ui_health.json`, `applied_ui_audit.json`, `PROGRESS.json`.
'''
(ops/'RESULT.md').write_text(report.replace('本人','본인'),encoding='utf-8')
state_path=ops.parent/'STATE.json'
state=read(state_path)
state['updated_date_kst']=stamp
state['current_stage']='FRESH_SELF_CAPTURE_UI_READY_CAMERA_CLOSED'
state['next_single_action']=progress['next_single_action']+' Minor preview CSS request failed ConnectionRefused; do not repeat without changed evidence or change implementation author.'
state['stage_results']['fresh_self_comparison_20260919']=progress
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
context_path=root/'WordsInSilence_PROJECT_CONTEXT.md'
context=context_path.read_text(encoding='utf-8')
status=f'''### {stamp} 최신 — 새 촬영 비교 화면 적용·기능 검증 완료, 카메라 닫힘

- Claude 완료된 v4 별도 화면을 적용했다. 합성33/33과 실제 Chrome 주요 흐름·보존 경계21/21 통과. 촬영 중 예측 비노출, 명시적 본인 확인, 저장 실패 차단, 취소/오류·54슬롯 보존, 새로고침 비재개를 확인했다. 세 모델/기존 backend·입력·기본화면/계획13개 해시 불변, 새 학습·추론·사용자촬영0.
- 준비 화면은 http://127.0.0.1:8765/frozen-capture.html 이며 첫 공부 시범 표시/실제 카메라null/소유세션null 확인. 사용자 직접 준비 전 카메라를 열지 않는다. 모의8766은 종료했다.
- 이전PID29716은 종료했고 불완전 응답을 보존한 뒤 완료된v2/v3/v4로 해결했다. 현재 Claude worker 없음. 카메라 닫힘 상태의 깨진 미리보기 대체표시만 남았으며 CSS 요청v5는 ConnectionRefused 종료1로 미적용; 같은 조건 재시도/구현 담당 변경0. 기능 검증과 실제 카메라 촬영·성능 검증을 구분한다.
- 본인 새54슬롯의 세 모델 오프라인 비교 연결과 실제 촬영은 아직 미완료다. 기존18건 재생·완료 검사를 반복하지 않는다. 상세 `research/front_keyphase_v1/ops/fresh_self_comparison_20260919/RESULT.md`.

'''
next_text=f'''### {stamp} 다음 단일 작업 — 사용자 준비 후 새 촬영, 고정 세 모델 평가 연결

- 준비 화면을 제시하고 사용자가 직접 카메라 열기로 준비를 확인할 때 촬영한다. 본인 새18×3회54슬롯, 예측 비노출, 같은 관측을 세 모델에 적용한다. 실제 촬영0 상태를 완료로 표시하지 않는다.
- 새로운 촬영 집합용 오프라인 평가 연결을 준비한다. 기존33/21 검사·모델 READY·고정 계획은 재사용한다. 원래 서버 참가자 확인과 관측 해시를 기준으로 parent220/control660/treatment660의18제한/전체93을 비교하고 취소·오류·미확인·미촬영 분모를 모두 보존한다. 기존18건 러너는 가정이 달라 그대로 재실행하지 않는다.
- 빈 미리보기 표시 정리는 부차적이며 Claude 연결 실패를 같은 조건으로 반복하지 않는다. 새 촬영 준비를 추가 다운로드·다인 모집·표시 정리 완료에 불필요하게 묶지 않는다. 사용자기록 비학습·보류 보호·전체18목표·원본/모델 보존·기본모델 승격 전 사용자 검토·기존 구현 역할을 유지한다.

'''
marker8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'
marker9='## 9. 다음 단일 작업\n\n'
assert context.count(marker8)==context.count(marker9)==1
context=context.replace('최종 갱신: 2026-09-19 20:18 KST','최종 갱신: '+stamp,1)
context=context.replace('### 2026-09-19 21:40 KST 최신 —','### 2026-09-19 21:40 KST 이전 기록 —',1)
context=context.replace(marker8,marker8+status,1).replace(marker9,marker9+next_text,1)
context_path.write_text(context,encoding='utf-8')
handover=root/'research/HANDOVER_2026-09-18.md'
text=handover.read_text(encoding='utf-8')
line=f'> {stamp} 최우선 최신: Claude v4 새 촬영 화면 적용/합성33·브라우저 주요 흐름 및 보존 경계21 통과. http://127.0.0.1:8765/frozen-capture.html 카메라닫힘/사용자촬영0. 기존 모델/입력/기본UI13개 해시동일. 이전Claude worker는 모두종료했으며 PID29716 대기를 재개하지 않는다. 빈미리보기 표시 정리v5만 ConnectionRefused로 미적용. 다음은 사용자 직접 준비 후 새18×3회 및 같은관측 세모델 오프라인 비교 연결(아직미완료). 추가학습/추론·다운로드·문의·삭제·백업·빌드·예약0. 상세 ops/fresh_self_comparison_20260919/RESULT.md 및 루트8·9/실제STATE.\n\n'
handover.write_text(line+text,encoding='utf-8')
print(json.dumps({'status':progress['status'],'at_kst':stamp,'state_updated':True,'ui_report':str(ops/'RESULT.md')},ensure_ascii=False))
