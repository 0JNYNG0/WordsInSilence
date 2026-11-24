"""Record verified runtime recovery and user-directed return to existing UI."""
import json, hashlib, urllib.request
from datetime import datetime
from pathlib import Path

root = Path('C:/MyUnityProjects/WordsInSilence')
web = root / 'research/front_keyphase_v1'
ops = web / 'ops/fresh_self_comparison_20260919'
here = Path(__file__).resolve().parent
stamp = datetime.now().strftime('%Y-%m-%d %H:%M') + ' KST'
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def save(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def api(path):
    with urllib.request.urlopen('http://127.0.0.1:8765/api/v1/'+path,timeout=15) as response:
        return json.load(response)

health = api('health')
session = api('sessions/ses_e1e9d0a79bc6')
journal_path = Path('C:/Users/local-user/Downloads/self_frozen_20260919_v1_progress.json')
journal = read(journal_path)
assert journal['session_id']=='ses_4d643cbf408e'
assert all(not s.get('trial_id') for s in journal['slots'])
evidence = {
    'at_kst':stamp,'status':'EXISTING_UI_CAMERA_RECOVERED',
    'user_direction':'Use the existing test screen; no separate newly designed screen required.',
    'first_failed_session':'ses_4d643cbf408e',
    'first_failed_session_trials':0,'first_failed_journal':str(journal_path),
    'first_failed_journal_sha256':hashlib.sha256(journal_path.read_bytes()).hexdigest(),
    'existing_ui_reproduced_error':'CameraError: cannot open camera index 0',
    'same_device_unrestricted_probe':{'device_index':0,'backend':'DSHOW','opened':True,'frame_received':True,'shape':[480,640,3],'saved_frames':0,'exit_code':0},
    'stopped_server':{'pid':24448,'tool_session':17679,'camera_was_closed':True},
    'new_server':{'pid':33832,'tool_session':74341,'host':'127.0.0.1','port':8765,'execution_context':'require_escalated'},
    'observed_preview':{'naturalWidth':1280,'naturalHeight':720,'complete':True,'url':'/api/v1/sessions/ses_e1e9d0a79bc6/preview.jpg'},
    'health':health,'session':session,
    'root_cause_evidence':'Same 0/DSHOW device failed under restricted server; opened and delivered frames in general execution; unchanged server then provided1280x720 preview after restart in that context.',
    'new_frozen_page_tab':'CLOSED_AFTER_EXPORT','existing_ui_url':'http://127.0.0.1:8765/',
    'production_code_changes':0,'agent_triggered_trials':0,'independent_recount_preparation':'NOT_STARTED_CAMERA_RECOVERY_TOOK_PRIORITY',
    'comparison_caveat':'Existing UI shows results and lacks the frozen54-slot journal. Its new development recordings must not be presented as completion of the blinded54-slot protocol or sent unchanged to its strict evaluator.'}
save(here/'CAMERA_RECOVERY_LOCAL_ONLY.json',evidence)
next_action = 'Keep existing web UI as the only user-facing tester. Preserve any new observations/reviews in ses_e1e9d0a79bc6; do not interrupt active camera/session. After session close, compare frozen parent/control/treatment on identical confirmed records with a compatible audited offline runner. Original visible-results development flow is not the uncompleted blinded54-slot protocol. Do not build another UI.'
p = read(ops/'PROGRESS.json')
p.update(status='EXISTING_UI_CAMERA_RECOVERED_FRESH_COMPARISON_PENDING',at_kst=stamp,next_single_action=next_action)
p['web_server'].update(pid=33832,tool_session=74341,url='http://127.0.0.1:8765/',state='RUNNING_EXISTING_UI',last_verified_camera=health['camera'],last_verified_camera_session_id=health['camera_session_id'])
p['camera_recovery_20260920']={'status':evidence['status'],'evidence':'resume_20260920/CAMERA_RECOVERY_LOCAL_ONLY.json','failed_frozen_session':'ses_4d643cbf408e','existing_ui_session':'ses_e1e9d0a79bc6','trial_count_at_check':session['trial_count']}
p['frozen54_protocol_status']='NOT_COMPLETED_USER_CHOSE_EXISTING_UI; preserve plan/code/evidence, do not silently apply to incompatible records'
p['actual_camera_capture_verification']='CAMERA_AND_PREVIEW_VERIFIED; new trial performance not evaluated'
p['not_yet_done']=['Fresh recorded observations and original participant reviews','Compatible offline comparison of three frozen models on existing-UI records','Independent recount and user review; no automatic promotion']
save(ops/'PROGRESS.json',p)
state=read(web/'ops/STATE.json')
state.update(updated_date_kst=stamp,current_stage='EXISTING_UI_CAMERA_RECOVERED_AWAITING_FRESH_DEVELOPMENT_COMPARISON',next_single_action=next_action,
             running_processes='General-context web server PID33832 / tool session74341. Existing UI session ses_e1e9d0a79bc6 opened, camera preview verified; do not interrupt user.',
             serve_command='Existing server http://127.0.0.1:8765/; do not duplicate. Camera requires normal execution context, restricted server failed.',
             human_test_status='Camera1280x720 preview verified on existing screen. Fresh comparison not yet evaluated; failed frozen session0trials preserved.')
state['stage_results']['fresh_self_comparison_20260919']=p
save(web/'ops/STATE.json',state)
context=root/'WordsInSilence_PROJECT_CONTEXT.md'
t=context.read_text(encoding='utf-8')
t=t.replace(next(x for x in t.splitlines() if x.startswith('최종 갱신:')),'최종 갱신: '+stamp,1)
marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
t=t.replace(marker,marker+f'''\n### {stamp} 최우선 — 사용자 선택으로 기존 시험 화면 복귀, 카메라 실제 복구\n\n- 사용자가 기존 테스트 화면 유지를 요청했다. 별도 화면의 결과 비노출/고정순서 목적이 별도 UI를 필수로 만들지는 않으므로 기존 `http://127.0.0.1:8765/`를 계속 사용한다. 새 화면 탭은 실패 기록 JSON 저장 후 닫았고 소스/평가도구/계획은 보존한다.\n- 사용자 시작의 첫 세션ses_4d643cbf408e는 카메라 실패·촬영0·정상 종료였다. 기존 화면에서도0번 카메라 열기 실패를 재현했다. 같은0번/DSHOW 장치는 일반 실행에서 프레임을 받았고, 제한 서버PID24448을 안전 종료한 뒤 일반 환경의PID33832/핸들74341로 재시작하여 기존 화면1280×720 미리보기·오류null을 확인했다. production수정0이다.\n- 기존 화면 세션ses_e1e9d0a79bc6는 카메라를 열어 사용자에게 남겼다. 에이전트의 촬영 시작0, 영상저장 꺼짐이다. 사용자의 진행을 방해하는 새로고침/서버재시작/카메라 종료를 하지 않는다.\n- 기존 화면은 결과를 즉시 보여주고 새54슬롯 journal을 만들지 않는다. 이 기록을 별도 화면의 맹검54회 완료로 표시하거나 strict evaluator에 억지로 맞추지 않는다. 다음은 기존 화면 새 기록에 맞는 동일관측 세 모델 후처리 비교이며 아직 미실행이다. 근거: `research/front_keyphase_v1/ops/fresh_self_comparison_20260919/resume_20260920/CAMERA_RECOVERY_LOCAL_ONLY.json`.\n''',1)
marker9='## 9. 다음 단일 작업\n'
t=t.replace(marker9,marker9+f'''\n### {stamp} 다음 단일 작업 — 기존 화면을 유지한 새 기록 비교\n\n- 사용자는 기존 시험 화면을 선택했고 실제 웹캠 미리보기를 복구했다. 기존 세션을 임의 종료/새로고침하지 않으며 별도 화면 재개나 또 다른 UI를 선행하지 않는다. 실제 세션 상태와 사용자의 완료/중단을 확인한다.\n- 기존 화면에서 저장되는 관측과 원래 참가자 확인으로 동일입력 세 모델 비교를 준비한다. 예측 비노출54슬롯 방식과 다르다는 평가 한계를 명시하고, 필요한 최소 후처리 연결만 Claude 작성·Codex 검증 역할로 진행한다. 이미 통과한 도구 검사는 해당 계약 범위에서만 재사용한다.\n- 사용자기록 비학습·18목표·보류/원본/모델 보존·임시 단어 예외 및 임계값 완화 금지·다운로드/문의/삭제/백업/Unity빌드/예약 금지는 유지한다. 카메라 복구를 인식성능 검증 완료로 세지 않는다.\n''',1)
context.write_text(t,encoding='utf-8')
print(json.dumps({'status':evidence['status'],'session':session['session_id'],'trial_count':session['trial_count'],'camera_present':health['camera'] is not None,'updated':stamp}))
