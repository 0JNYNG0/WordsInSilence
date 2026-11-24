"""Publish verified tool readiness; don't create a cohort from zero observations."""
import hashlib,json,urllib.request
from datetime import datetime
from pathlib import Path
here=Path(__file__).resolve().parent;web=here.parents[1];root=web.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now().strftime('%Y-%m-%d %H:%M')+' KST'
v1=read(here/'verification_v1.json');v2=read(here/'verification_v2.json')
recount=read(here/'synthetic_recount.json');fault=read(here/'recount_fault_verification.json')
assert v1['passed']==24 and v1['total']==26 and v2['status']=='PASS' and v2['passed']==11
assert recount['status']=='PASS_SAVED_OUTPUT_RECOUNT' and fault['status']=='PASS_FAULT_DETECTED'
with urllib.request.urlopen('http://127.0.0.1:8765/api/v1/sessions/ses_e1e9d0a79bc6',timeout=15) as response:session=json.load(response)
status='TOOL_VERIFIED_AWAITING_NONEMPTY_NEW_SESSION' if session['state']=='CLOSED' and session['trial_count']==0 else 'TOOL_VERIFIED_AWAITING_CLOSED_CAPTURE'
record={'status':status,'at_kst':stamp,'implementation_author':'Claude Code; generic interfaces only','independent_verification':'Codex',
        'initial_checks':[24,26],'corrected_checks':[11,11],'unaffected_checks_reused':24,
        'synthetic_independent_recount':[recount['passed'],recount['total']],'deliberately_wrong_summary_rejected':True,
        'files_sha256':{name:sha(here/name) for name in ('session_replay_v2.py','audit_saved_result.py','config_LOCAL_ONLY.json')},
        'session_observation':{k:session[k] for k in ('session_id','state','trial_count','active_trial_id','camera','store_video')},
        'new_user_observation_arrays_read':0,'real_model_inference':0,'training':0,'model_promotion':0,
        'production_changes':0,'source_cli_completed':read(here/'claude_completion.json'),'fix_cli_completed':read(here/'fix_completion.json')}
save(here/'READINESS_LOCAL_ONLY.json',record)
next_action='Await participant new captures in the existing UI. The observed session ses_e1e9d0a79bc6 is closed with0trials; do not score it. For the actual next nonempty CLOSED session create a new config preserving frozen model/word/pin choices, prepare session_replay_v2 once, execute once, independently audit_saved_result. Reuse completed checks and do not build another screen.'
pfile=web/'ops/fresh_self_comparison_20260919/PROGRESS.json';p=read(pfile)
p.update(status=status,at_kst=stamp,next_single_action=next_action,
         existing_ui_offline_evaluator={'status':status,'path':'ops/existing_ui_three_model_20260920','code_sha256':record['files_sha256'],'synthetic_checks':'24 unaffected reused +11 corrected/affected PASS','actual_new_cohort_execution':'NOT_RUN_NO_OBSERVATIONS'},
         latest_actual_session=record['session_observation'])
p['web_server'].update(state='RUNNING_EXISTING_UI_CAMERA_CLOSED',last_verified_camera=None,last_verified_camera_session_id=None)
save(pfile,p)
state=read(web/'ops/STATE.json')
state.update(updated_date_kst=stamp,current_stage=status,next_single_action=next_action,
             human_test_status='Latest existing-UI session ses_e1e9d0a79bc6 CLOSED,0trials,camera=false; no fresh performance result.',
             running_processes='Web server PID33832 / tool session74341 remains. Latest session closed/camera released. Claude calls completed exit0; no inference/training running.')
state['stage_results']['fresh_self_comparison_20260919']=p
state['stage_results']['existing_ui_three_model_20260920']=record
save(web/'ops/STATE.json',state)
context=root/'WordsInSilence_PROJECT_CONTEXT.md';t=context.read_text(encoding='utf-8')
t=t.replace(next(x for x in t.splitlines() if x.startswith('최종 갱신:')),'최종 갱신: '+stamp,1)
mark='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
t=t.replace(mark,mark+f'''\n### {stamp} 최우선 — 기존 화면용 세 모델 후처리·독립 검산 준비 완료, 실제 새 기록 없음\n\n- 기존 화면과 추론 어댑터는 그대로다. 과거18건/두 모델 고정 러너를 무리하게 재실행하지 않고 Claude가 작성한 `ops/existing_ui_three_model_20260920/session_replay_v2.py`로 임의 횟수의 기존 화면 세션 전체를 부모/대조/추가군에 연결한다. 화면을 추가하지 않았다.\n- 최초24/26 중2실패를 보존하고 그2함수만 수정했다. 영향 없는24검사 재사용, 수정/연결11/11 PASS. 원래 확인기록·모든시도·누락·오류 분모 유지와 같은관측3모델 연결을 합성 검증했다. Codex 독립 검산143/143 일치, 정답 수를 부풀린 결과는 FAIL로 거부했다. 이는 인식성능 검증이 아니다.\n- 기존 화면 세션ses_e1e9d0a79bc6는 CLOSED·촬영0·카메라 해제로 확인됐다. 새 사용자 관측 배열 읽기/실제모델 추론/학습/승격0이다. 현재config는 그0건 세션의 준비 기록이므로 다음 실제세션ID로 별도config를 만든 후 사용한다.\n- 상세/명령: `research/front_keyphase_v1/ops/existing_ui_three_model_20260920/RESULT.md`, 실제 상태 `READINESS_LOCAL_ONLY.json`. 서버PID33832/핸들74341 유지. 사용자 준비 질문을 보냈으며 직접 새촬영 전에는 더 진행할 성능 비교 입력이 없다.\n''',1)
mark9='## 9. 다음 단일 작업\n'
t=t.replace(mark9,mark9+f'''\n### {stamp} 다음 단일 작업 — 기존 화면의 실제 새 촬영 확보 후 준비된 비교 실행\n\n- 기존 화면 유지. 사용자가 새 세션에서 촬영·실제 표현 확인을 저장하고 종료하면 실제 세션ID로 새config를 만든다. 모델3개/해시/18단어/보호파일 선택은 바꾸지 않는다. 현재0건 CLOSED세션을 평가하거나 촬영했다고 세지 않는다.\n- `ops/existing_ui_three_model_20260920/RESULT.md`에 따라 prepare→manifest대조→execute1회→원래확인/호출로그 기반 독립검산 순서로 진행한다. 같은 관측에서18제한/전체93의 정답·회귀·실패·지원범위를 보고한다.\n- 현재도구 합성/보존 검사와 지난 모델 시험은 재사용한다. 화면의 결과를 본 개발 시험이라는 한계를 유지하며 맹검54회/다인일반화/올바른 수행 판정으로 확대하지 않는다. 사용자기록 비학습·보류보호·원본/모델보존·기본모델 자동승격 금지·추가다운로드/문의/삭제/백업/Unity빌드/예약 금지는 유지한다.\n''',1)
context.write_text(t,encoding='utf-8')
print(json.dumps({'status':status,'at_kst':stamp,'actual_trial_count':session['trial_count'],'code_sha256':record['files_sha256']['session_replay_v2.py']}))
