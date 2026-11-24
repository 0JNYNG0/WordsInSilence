"""Record the user's explicit pause and verified process termination."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
ops=Path(__file__).resolve().parent
root=ops.parents[3]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
workers={}
for role in ('core','driver'):
    completion=read(ops/f'claude_offline_{role}_completion.json')
    workers[role]={**completion,'response_bytes':(ops/f'claude_offline_{role}_response.json').stat().st_size,
                   'stop_reason':'EXPLICIT_USER_PAUSE','code_applied':False,'behavior_verified':False}
pause={
 'at_kst':stamp,'active':True,'reason':'User cannot test today and explicitly asked to stop after current work and wait.',
 'resume_requires':'Explicit user resume; camera additionally requires participant readiness.',
 'previous_goal_turn':'PROGRESS: capture UI applied and browser/flow verification completed.',
 'current_turn_progress':'Prepared offline evaluator interface contracts, reviewed existing adapter/storage boundaries, wrote evidence-based project brief, safely stopped owned processes. No evaluator code completion claimed.',
 'offline_evaluator_status':'SPECIFICATIONS_PREPARED_CODE_REQUESTS_CANCELLED_ON_USER_PAUSE_NOT_IMPLEMENTED',
 'claude_workers':workers,'claude_owned_processes_remaining':0,
 'process_termination':{'initial_standard_attempt':'ACCESS_DENIED; its unconditional success text was not evidence of termination','elevated_owner_context':'Both actual processes29760/21008 verified absent; launcher handles74324/20456 terminal'},
 'web_server':{'pid':3104,'port':8765,'state':'STOPPED','tool_session18876_exit':1,'camera_before_stop':None,'camera_session_id_before_stop':None},
 'mock_server':'Previously stopped8766','capture_tab':'Closed',
 'new_human_trials':0,'new_model_inference':0,'new_training':0,
 'new_downloads':0,'new_backups':0,'unity_builds':0,'scheduled_tasks':0,
 'project_brief':'research/PROGRESS_BRIEF_2026-09-19.md',
 'next_after_resume':'Finish and independently test offline54-slot three-model evaluator using existing adapter; address empty-preview cosmetic separately; then fresh capture when participant ready. Do not replay completed historical tests or alter frozen models.'
}
(ops/'USER_PAUSE_20260919.json').write_text(json.dumps(pause,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
progress=read(ops/'PROGRESS.json')
progress['status']='USER_PAUSED_CAPTURE_UI_READY_OFFLINE_EVALUATOR_NOT_IMPLEMENTED'
progress['at_kst']=stamp
progress['user_pause']=pause
progress['web_server']['left_running_for_user']=False
progress['web_server']['state']='STOPPED_ON_USER_REQUEST'
progress['offline_evaluator_status']=pause['offline_evaluator_status']
progress['next_single_action']='WAIT_FOR_EXPLICIT_USER_RESUME. No camera, training, downloads, further implementation or automatic testing today. '+pause['next_after_resume']
(ops/'PROGRESS.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state_path=ops.parent/'STATE.json'
state=read(state_path)
state['updated_date_kst']=stamp
state['current_stage']='USER_PAUSED_AFTER_CAPTURE_UI_PREPARATION'
state['user_pause']={'active':True,'at_kst':stamp,'scope':'All further autonomous project work until explicit user resume; user cannot test today.','evidence':'research/front_keyphase_v1/ops/fresh_self_comparison_20260919/USER_PAUSE_20260919.json'}
state['next_single_action']=progress['next_single_action']
state['stage_results']['fresh_self_comparison_20260919']=progress
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
context_path=root/'WordsInSilence_PROJECT_CONTEXT.md'
context=context_path.read_text(encoding='utf-8')
lines=context.splitlines()
for i,line in enumerate(lines):
    if line.startswith('최종 갱신:'):
        lines[i]='최종 갱신: '+stamp
        break
context='\n'.join(lines)+'\n'
status=f'''### {stamp} 최우선 — 사용자 요청으로 작업 중단, 오늘 촬영 없음

- 사용자가 오늘 테스트 불가를 알리고 현재 작업까지만 정리한 뒤 기다리라고 요청했다. 새촬영0이며 실제 카메라null/소유세션null 확인 후 서버8765/PID3104를 종료했다. 준비용Chrome탭도 닫았다. 모의8766은 이미 종료됐다.
- 새54슬롯 평가의 집계/실행 명세를 준비했고 Claude 요청2개를 시작했으나 사용자 중단에 맞춰 실제PID29760/21008을 종료·확인했다. 두 응답은 미완료, 새평가코드 적용·검증0. 작성요청을 구현완료로 세지 않는다. 현재 이 작업의 실행worker 없음.
- 기존 화면 기능33/33·브라우저/보존경계21/21과 모델 결과는 유지한다. 새 학습/추론·다운로드·백업·Unity빌드·예약0. 전체 현황과 남은 인식/피드백/Unity 단계를 `research/PROGRESS_BRIEF_2026-09-19.md`에 정리했다. 전체 목표는 미완료이며 일시중단을 완료나 기술적 실패로 바꾸지 않는다.

'''
next_text=f'''### {stamp} 다음 단일 작업 — 명시적 재개 요청까지 대기

- 오늘 자동 후속 구현·학습·조회·사람 테스트를 이어가지 않는다. 사용자 재개 후에만 새54슬롯 오프라인 평가 도구 구현/독립검증을 마무리한다. 두 Claude 미완료 요청을 살아 있는 작업으로 오해하지 않는다.
- 촬영 화면 기능 검사는 완료돼 재실행하지 않는다. 고정 세 모델과 기존 결과를 유지하고 실제 촬영은 사용자가 준비됐을 때 진행한다. 전체 인식 일반화·차이 설명 타당성·현재 웹 기능 Unity 통합은 여전히 미완료다.
- 종료·응답크기·재개 경계: `research/front_keyphase_v1/ops/fresh_self_comparison_20260919/USER_PAUSE_20260919.json`.

'''
context=context.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+status,1)
context=context.replace('## 9. 다음 단일 작업\n\n','## 9. 다음 단일 작업\n\n'+next_text,1)
context_path.write_text(context,encoding='utf-8')
handover=root/'research/HANDOVER_2026-09-18.md'
old=handover.read_text(encoding='utf-8')
handover.write_text(f'> {stamp} 최우선 중단: 사용자가 오늘 테스트불가·현작업정리후대기를 요청했다. 실제camera닫힘 확인 후8765종료/준비탭닫힘/Claude평가도구 요청2개 PID29760·21008 실제종료. 새평가도구는 명세만 준비·구현/검증미완료; 기존촬영화면33+21기능검증과모델결과유지. 명시재개전 자동작업0. 전체중간보고 research/PROGRESS_BRIEF_2026-09-19.md, 종료근거 ops/fresh_self_comparison_20260919/USER_PAUSE_20260919.json.\n\n'+old,encoding='utf-8')
report=root/'research/PROGRESS_BRIEF_2026-09-19.md'
with report.open('a',encoding='utf-8') as f:
    f.write(f'\n## 중단 확인 — {stamp}\n\n카메라가 닫혀 있음을 확인한 뒤 웹 서버8765를 종료하고 준비 탭을 닫았다. 새 평가 도구는 집계/실행 명세까지 준비했으며, 작성 응답이 완료되기 전에 사용자 요청으로 두 Claude 프로세스를 실제 종료했다. 코드 적용·동작 검증은 미완료다. 이 작업에서 남아 있는 Claude 실행은 없으며 사용자 재개 전 자동 작업을 이어가지 않는다.\n')
print(json.dumps({'paused':True,'workers':workers,'state':state['current_stage'],'report':str(report)},ensure_ascii=False))
