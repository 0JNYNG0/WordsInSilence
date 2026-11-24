"""Persist actual partial completion and the live worker identity for safe continuation."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json

OPS=Path(__file__).resolve().parent
WEB=OPS.parents[1]
ROOT=WEB.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
now=datetime.now(timezone(timedelta(hours=9)))
stamp=now.strftime('%Y-%m-%d %H:%M KST')
assert read(OPS/'model_readiness.json')['status']=='PASS_THREE_FROZEN_MODELS_READY'
assert len(read(OPS/'protocol_LOCAL_ONLY.json')['slots'])==54
progress={'status':'PROTOCOL_AND_MODEL_READINESS_COMPLETE_UI_WORKER_PENDING',
 'at_kst':now.isoformat(),'previous_goal_turn_classification':'PROGRESS: two real head-only candidates trained and independently verified.',
 'current_goal_turn_progress':['Fixed complete18-word/54-slot new-capture protocol before inputs exist','Verified same93-class order and web-adapter loading for all three frozen models, no inference','Started existing localhost service without camera','Dispatched generic-only Claude static UI implementation; actual child process still running after launcher wait limit'],
 'web_server':{'tool_session':18876,'url':'http://127.0.0.1:8765/','camera_opened':False},
 'claude':{'launcher_tool_session':76840,'launcher_exit_code':1,'launcher_end_reason':'Python subprocess.TimeoutExpired at600 seconds; this is NOT proof of child termination',
            'child_pid':29716,'child_started_local':'2026-09-19T21:25:51.4199618+09:00','live_child_verified':True,
            'monitor_tool_session':55595,'monitor_status':'RUNNING as confirmed by write_stdin','response_bytes_at_last_check':0,'stderr_bytes_at_last_check':0,
            'response_file':str(OPS/'claude_response.json'),'completion_record':str(OPS/'claude_completion_after_launcher_timeout.json'),
            'restart_allowed_based_only_on_timeout':False},
 'not_yet_done':['Claude UI response/code review','New page application','Synthetic UI failure-path tests','Browser UI validation','Actual new participant captures','Three-model evaluation of new observations'],
 'camera_sessions_created':0,'new_human_trials':0,'new_training_runs':0,'new_inference_calls':0,
 'data_and_model_default_unchanged':True}
with (OPS/'PROGRESS.json').open('x',encoding='utf-8') as f:json.dump(progress,f,ensure_ascii=False,indent=2)
state_path=WEB/'ops/STATE.json'
state=read(state_path)
state['updated_date_kst']=stamp
state['updated_at']=now.isoformat()
state['current_stage']='FRESH_SELF_PROTOCOL_FROZEN_UI_CLAUDE_WORKER_LIVE'
state['next_single_action']='Continue the CURRENT Claude UI worker, do not launch a duplicate. Launcher76840 timed out at600s but actual child PID29716 (start21:25:51 KST) was verified alive. Monitor tool session55595 waits for that exact process; poll that handle, or inspect completion JSON/actual PID if handle missing. Response currently0 bytes. After terminal response, stage_claude_response.py parses into ops/claude_generated; review before copying new frozen-capture.html/js into web. Existing protocol/model readiness is complete and must be reused. Use mock_capture_server.py on8766 for synthetic browser failure-path tests, no real camera. Verify real localhost8765 pre-start page without opening camera, then make the concrete capture page available for participant readiness. Full54 slots, all18 words, three fixed models, same observations, hidden prediction feedback. No user training, candidate promotion, downloads, deletion, backup, Unity build, schedule or repeated prior model tests.'
state['stage_results']={'fresh_self_comparison_20260919':progress,**state['stage_results']}
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md'
text=context.read_text(encoding='utf-8')
current=f'''### {stamp} 최신 — 새 촬영 프로토콜·세 모델 입력 준비 완료, 화면 Claude 프로세스 실행 중

- 새 촬영 전18단어×3회54슬롯 순서와 기존 visibility 부모/새 head-only 대조·추가군 해시를 고정했다. 세 모델이 같은 웹 어댑터에서 READY이고93클래스 순서가 동일함을 확인했다. 부모step220/새두군660 구분, 새추론·학습0. `ops/fresh_self_comparison_20260919/protocol_LOCAL_ONLY.json`, `model_readiness.json`.
- 기존 서버는127.0.0.1:8765, 실행핸들18876로 시작했고 카메라null/소유세션null 확인. 새 촬영 세션·사용자촬영0. 기존 화면은 결과를 즉시 노출하므로 기존 API를 재사용하는 별도 결과 비노출 시험 화면을 Claude에 일반 명세만 전달해 작성 중이다. backend/기본화면/기본모델은 그대로다.
- Claude 호출 보조76840은600초 대기제한으로 종료1이지만 실제 자식PID29716(21:25:51 시작)은 살아 있음을 확인했다. 같은프로세스 종료대기 핸들55595를 이어간다. 마지막 응답/오류 파일은0바이트. 시간초과를 실패·중단으로 오해해 같은요청을 재시작하지 않는다. 아직 화면코드/검증/새촬영 결과는 없다. 상세 `ops/fresh_self_comparison_20260919/PROGRESS.json`.

'''
anchor='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'
assert text.count(anchor)==1
text=text.replace(anchor,anchor+current,1)
next_text=f'''### {stamp} 다음 단일 작업 — 현재 Claude 실행을 이어받아 새 촬영 화면 적용·검증

- 핸들55595/자식PID29716의 실제 상태와 완료JSON을 확인한다. 응답이 완성되면 `stage_claude_response.py`로 staging 후 코드를 검토·적용한다. 응답 미완성 또는 관측 시간초과만으로 재호출하지 않는다. 원래 호출76840은 보조 대기제한 종료이며 실제 자식 종료 판정이 아니다.
- 새화면은 기존API로만 촬영/본인확인을 저장하고 예측을 촬영중 숨긴다. 합성 `mock_capture_server.py`로 카메라 없이 저장실패/중복방지/취소/새로고침을 검증한 후 실제8765의 시작전 화면을 확인한다. 검증 후 사용자에게 준비된 화면을 보여주되 직접 준비 확인 전 카메라는 열지 않는다.
- 이미 완료된 모델 로드/고정54슬롯 계획과 직전 학습 검증은 재사용한다. 완료와 구현중을 구분하며 사용자기록 비학습·전체18목표·보류보호·모델기본값보존·구현역할·다운로드/문의/삭제/백업/빌드/예약 금지를 유지한다.

'''
anchor='## 9. 다음 단일 작업\n\n'
assert text.count(anchor)==1
context.write_text(text.replace(anchor,anchor+next_text,1),encoding='utf-8')
handover=ROOT/'research/HANDOVER_2026-09-18.md'
text=handover.read_text(encoding='utf-8')
anchor='<!-- current-continuation-20260919 -->\n'
note=f'> {stamp} 최우선 진행중: 본인 새촬영18×3=54순서/세모델해시 고정, 같은93클래스·웹입력 READY 확인(추론0). 새 결과비노출 화면 Claude 요청76840은 보조600초TimeoutExpired 종료1이지만 실제 자식PID29716 살아있음; 현재 모니터핸들55595를 이어가며 중복재시작 금지. 응답0바이트/화면구현·검증 미완료. 기존서버18876/localhost8765 카메라닫힘·새사람촬영0. 다음은 실제응답을 stage/review/apply하고 가짜카메라UI검사 후 사용자촬영화면 제시. 이전학습검증·프로토콜반복 금지. 상세 ops/fresh_self_comparison_20260919/PROGRESS.json 및 실제STATE/루트8·9절.\n\n'
assert text.count(anchor)==1
handover.write_text(text.replace(anchor,anchor+note,1),encoding='utf-8')
print(json.dumps({'status':progress['status'],'updated_at_kst':stamp,'same_live_process_monitor':55595,'new_human_captures':0}))
