"""Publish the bounded persistence correction and derive new runtime pins explicitly."""
import hashlib,json
from datetime import datetime
from pathlib import Path
here=Path(__file__).resolve().parent;web=here.parents[1];root=web.parents[1]
evaluation=web/'ops/existing_ui_three_model_20260920'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now().strftime('%Y-%m-%d %H:%M')+' KST'
v=read(here/'verification_after_indent_corrected.json');live=read(here/'runtime_verification.json')
assert v['status']=='PASS' and v['passed']==v['total']==10 and live['status']=='PASS_LIVE_EXPIRY_SNAPSHOT'
app=web/'src/wis_lab/service/app.py';appsha=sha(app)
assert appsha==v['app_sha256']==live['app_sha256']
cfg=read(evaluation/'config_LOCAL_ONLY.json')
changed=[p for p,h in cfg['protected_sha256'].items() if sha(Path(p))!=h]
assert [Path(p).resolve() for p in changed]==[app.resolve()]
oldsha=cfg['protected_sha256'][changed[0]]
cfg['protected_sha256'][changed[0]]=appsha
for p in (here/'application_corrected.json',here/'verification_after_indent_corrected.json',here/'runtime_verification.json'):
    cfg['protected_sha256'][str(p)]=sha(p)
cfg['prepared_at_kst']=datetime.now().isoformat()
cfg['runtime_dependency_amendment']={'reason':'Persist expired-session CLOSED/cameraFalse/exact trial list; same existing UI and inference path',
    'file':str(app),'old_sha256':oldsha,'new_sha256':appsha,'evidence':str(here/'runtime_verification.json'),
    'frozen_model_and_input_hashes_unchanged':True,'original_config_preserved':True}
cfg['usage_note']='This is a template pinned to the corrected runtime; original session has0trials. Derive a new file with the actual future CLOSED nonempty session_id.'
with (evaluation/'config_after_lease_fix_LOCAL_ONLY.json').open('x',encoding='utf-8') as f:json.dump(cfg,f,ensure_ascii=False,indent=2)
record={'status':'PASS_LEASE_SNAPSHOT_PERSISTENCE','at_kst':stamp,'app_sha256':appsha,'before_sha256':oldsha,
    'implementation_author':'Claude Code; exact generated insertion applied by Codex with corrected placement indentation',
    'initial_reproduction':[6,10],'final_focused_checks':[10,10],'runtime_check':live['status'],
    'only_protected_file_changed':str(app),'other_pins_unchanged':len(changed)==1,
    'new_config':'ops/existing_ui_three_model_20260920/config_after_lease_fix_LOCAL_ONLY.json',
    'previous_server_pid':33832,'server_pid':35088,'server_tool_session':90445,
    'latest_real_session':'ses_e1e9d0a79bc6','latest_real_session_state':'CLOSED','latest_real_trial_count':0,
    'synthetic_runtime_probe_session':live['probe_session_id'],'real_camera_opens':0,'real_model_inference':0,'training':0,'model_promotions':0}
save(here/'RESULT.json',record)
pfile=web/'ops/fresh_self_comparison_20260919/PROGRESS.json';p=read(pfile)
next_action='Await actual new recordings and participant reviews in the existing UI. Do not evaluate0-trial sessions or synthetic_expiry_probe. On a nonempty CLOSED session derive a new per-session config from config_after_lease_fix_LOCAL_ONLY.json with identical models/input contracts, then prepare/execute/audit once. Reuse passed checks; do not re-create a test screen.'
p.update(at_kst=stamp,status='COMPARISON_READY_AWAITING_NEW_CAPTURE',next_single_action=next_action,
         existing_models_backend_default_ui_and_protocol_unchanged=False,
         runtime_persistence_amendment=record)
p['web_server'].update(pid=35088,tool_session=90445,state='RUNNING_EXISTING_UI_CAMERA_CLOSED',last_verified_camera=None,last_verified_camera_session_id=None)
save(pfile,p)
state=read(web/'ops/STATE.json')
state.update(updated_date_kst=stamp,current_stage='COMPARISON_READY_AWAITING_NEW_CAPTURE',next_single_action=next_action,
             running_processes='Web server PID35088 / tool session90445 in normal camera-capable context. Camera closed. Claude finished. No training/inference job.',
             human_test_status='Real session ses_e1e9d0a79bc6 expired with0trials; standard close API synchronized its disk snapshot. Runtime probe is synthetic/no camera/no trials and excluded.')
state['stage_results']['fresh_self_comparison_20260919']=p
state['stage_results']['lease_snapshot_fix_20260920']=record
save(web/'ops/STATE.json',state)
context=root/'WordsInSilence_PROJECT_CONTEXT.md';t=context.read_text(encoding='utf-8')
t=t.replace(next(x for x in t.splitlines() if x.startswith('최종 갱신:')),'최종 갱신: '+stamp,1)
mark='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
t=t.replace(mark,mark+f'''\n### {stamp} 최우선 — 자동 종료 세션 저장 불일치 수정·실서버 확인, 새 촬영 대기\n\n- 실제 API는0건 세션ses_e1e9d0a79bc6를 LEASE_EXPIRED/CLOSED로 반환했지만 session.json은OPEN/카메라true였다. 사용자가 직접 종료했다고 추정하지 않는다. 자동 만료 경로가 디스크 스냅샷을 저장하지 않은 코드 근거를 확인했다.\n- Claude가 작성한6줄을 기존 `Store.reap_expired`에 적용해 카메라false와 종료된 세션 전체/시도목록을 저장한다. 초기 재현6/10, 적용 중 들여쓰기 오류는 서버 반영 전 발견·보존·교정했다. 최종 관련10/10 PASS, 실서버 카메라 없는 합성세션의 API/파일 CLOSED 일치 확인. 화면/인식수식/모델 수정0.\n- 기존 빈 세션은 원래close API로 이유LEASE_EXPIRED를 보존해 파일을 동기화했다. 서버는PID35088/핸들90445로 교체했고 카메라는 닫혀 있다. 합성probe ses_81b5e3576d24는 실제 참가자 기록으로 세지 않는다. 새 실제관측/모델추론/학습0이다.\n- 비교의 원래config/모델계획은 보존하고, app.py의 검증된 저장 수정 해시만 반영한 `ops/existing_ui_three_model_20260920/config_after_lease_fix_LOCAL_ONLY.json`을 별도로 만들었다. 그외 보호파일 해시는 동일하다. 실제 다음 세션ID로 별도config를 만들어 사용하며0건 세션은 평가하지 않는다. 상세 `ops/lease_snapshot_fix_20260920/RESULT.md`.\n''',1)
mark9='## 9. 다음 단일 작업\n'
t=t.replace(mark9,mark9+f'''\n### {stamp} 다음 단일 작업 — 실제 새 촬영이 생기면 기존 화면 기록 비교\n\n- 기존 UI·동결 세 모델·학습/입력 계약 유지. 새 사용자 관절 기록과 본인 확인이 아직 없어 성능 비교를 실행할 수 없다. 준비 질문의 답변/새 촬영을 기다리고 카메라를 자동으로 열지 않는다.\n- 비어 있지 않은 CLOSED 세션을 확인하면 `config_after_lease_fix_LOCAL_ONLY.json`에서 실제 세션ID의 새config를 만들고 기존 `session_replay_v2.py` prepare→execute1회→`audit_saved_result.py` 순서로 진행한다. 초기config는 수정 전 app.py 해시이므로 그대로 쓰지 않는다.\n-0건 세션·합성probe를 성능 표본으로 포함하지 않는다. 이미 통과한 검사와 과거18건 성능을 반복하지 않으며 새 화면·추가학습/다운로드/문의·삭제/백업/Unity빌드/예약을 선행하지 않는다. 전체 웹/일반화/수행피드백 목표는 여전히 미완료다.\n''',1)
t=t.replace('\n-0건 세션','\n- 0건 세션')
context.write_text(t,encoding='utf-8')
print(json.dumps({'status':record['status'],'at_kst':stamp,'only_changed_dependency':'app.py','server_pid':35088,'real_trials':0}))
