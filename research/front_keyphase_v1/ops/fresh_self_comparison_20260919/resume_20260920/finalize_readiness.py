"""Operational evidence/document update; no training, inference or capture."""
import hashlib, json, urllib.request
from datetime import datetime
from pathlib import Path

root = Path('C:/MyUnityProjects/WordsInSilence')
web = root / 'research/front_keyphase_v1'
ops = web / 'ops/fresh_self_comparison_20260919'
out = ops / 'resume_20260920'
stamp = datetime.now().strftime('%Y-%m-%d %H:%M') + ' KST'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p, obj): p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

v = read(out / 'verification_v2.json')
i = read(out / 'verification_integrity.json')
assert (v['status'], v['passed'], v['total']) == ('PASS',47,47)
assert (i['status'], i['passed'], i['total']) == ('PASS',6,6)
plan = read(ops / 'protocol_LOCAL_ONLY.json')
pins = {p: {'expected': digest, 'actual': sha(Path(p))} for p,digest in plan['protected_sha256'].items()}
assert len(pins) == 13 and all(x['expected'] == x['actual'] for x in pins.values())
code = read(out / 'staging_v2.json')['files']
assert all(sha(out / 'generated_v2' / name) == digest for name,digest in code.items())
with urllib.request.urlopen('http://127.0.0.1:8765/api/v1/health', timeout=15) as response:
    health = json.load(response)
assert health['status'] == 'OK' and health['camera'] is None and health['camera_session_id'] is None
save(out / 'health_prestart_LOCAL_ONLY.json', health)
css_response = read(out / 'last_fix_response.json')['result'].strip()
css = json.loads(css_response.removeprefix('```json').removesuffix('```').strip())['preview_css']
assert css in (web / 'web/frozen-capture.html').read_text(encoding='utf-8')
applied = {name: sha(web / 'web' / name) for name in ('frozen-capture.html','frozen-capture.js')}
assert applied['frozen-capture.js'] == '5f21b13cbd9e0fcbe2fe414324f4dd85ed3705338e847f6ae6037a3bad8f594e'
report = {'status':'VERIFIED_AWAITING_FRESH_CAPTURE','at_kst':stamp,'user_resume_quote':'빨리 다음 진행해줘.',
          'implementation_author':'Claude Code; Codex applied and independently verified',
          'evaluator_code_sha256':code,'synthetic_checks':[47,47],'integrity_checks':[6,6],
          'reused_capture_checks':[33,33],'reused_prior_browser_boundary_checks':[21,21],
          'protected_files_unchanged':pins,'applied_ui_sha256':applied,
          'browser_prestart':{'browser':'Chrome','tab_id':'199983501','url':'http://127.0.0.1:8765/frozen-capture.html',
                             'observed':'Study demo video visible; round1 slot1/54; no session; camera-open enabled; capture disabled; broken preview absent',
                             'camera_button_clicked_by_agent':False},
          'web_server':{'pid':24448,'tool_session':17679,'port':8765,'camera':None,'camera_session_id':None},
          'fresh_human_trials':0,'real_model_inference_calls':0,'new_training_runs':0,'model_promotions':0,
          'real_capture_and_three_model_execution':'NOT_RUN',
          'claude_completions':{p.name:read(p) for p in out.glob('*completion.json')}}
save(out / 'READINESS_LOCAL_ONLY.json',report)

next_action = 'Participant starts fresh 18-word x3 capture when ready. Close session, export progress JSON; prepare and pin new cohort using generated_v2/frozen_replay.py, execute once, independently recount three-model results. Do not replay old development records or promote a model automatically.'
p = read(ops / 'PROGRESS.json')
p.update(status='OFFLINE_EVALUATOR_VERIFIED_AWAITING_FRESH_CAPTURE',at_kst=stamp,
         implementation_author='Claude Code capture UI v4 and offline evaluator corrected v2; Codex applied and independently verified',
         offline_evaluator_status='SYNTHETIC_47_AND_INTEGRITY_6_PASS_REAL_COHORT_NOT_RUN',
         next_single_action=next_action,applied_sha256=applied,
         not_yet_done=['Participant readiness and fresh observations','Three-model execution and independent recount','User review of model results before any default replacement'])
p['user_pause'].update(active=False,resumed_at_kst=stamp,resume_quote='빨리 다음 진행해줘.',historical_record='USER_PAUSE_20260919.json')
p['web_server'].update(tool_session=17679,pid=24448,left_running_for_user=True,state='RUNNING_CAMERA_CLOSED',last_verified_camera=None,last_verified_camera_session_id=None)
p['cosmetic_issue'].update(fix_status='APPLIED_CLAUDE_CSS_BROWSER_VERIFIED',verified_at_kst=stamp)
p['offline_evaluator']={'directory':'resume_20260920/generated_v2','code_sha256':code,'synthetic_checks':[47,47],'integrity_checks':[6,6], 'readiness':'resume_20260920/READINESS_LOCAL_ONLY.json', 'actual_new_cohort_execution':'NOT_RUN'}
p['resume_20260920']={'at_kst':stamp,'status':report['status'],'browser_tab_id':'199983501','protected_files_unchanged':13,'real_inference':0,'training':0,'fresh_human_trials':0}
save(ops / 'PROGRESS.json',p)
state = read(web / 'ops/STATE.json')
state.update(updated_date_kst=stamp,current_stage='FRESH_SELF_EVALUATOR_VERIFIED_AWAITING_NEW_CAPTURE',next_single_action=next_action,
             resume_command='Read ops/fresh_self_comparison_20260919/resume_20260920/RESULT.md; preserve frozen inputs and reuse passed checks.',
             running_processes='Web server8765 PID24448 / tool session17679 running, camera closed at prestart check. Claude authoring completed; no training job.',
             serve_command='Existing server: http://127.0.0.1:8765/frozen-capture.html (do not start duplicate)',
             human_test_status='Fresh self_frozen_20260919_v1 captures0. Existing18-record development comparison retained. Await participant readiness.')
state['stage_results']['fresh_self_comparison_20260919'] = p
if 'user_pause' in state: state['user_pause'].update(active=False,resumed_at_kst=stamp,resume_quote='빨리 다음 진행해줘.')
save(web / 'ops/STATE.json',state)

context = root / 'WordsInSilence_PROJECT_CONTEXT.md'
text = context.read_text(encoding='utf-8')
old = next(line for line in text.splitlines() if line.startswith('최종 갱신:'))
text = text.replace(old,'최종 갱신: '+stamp,1)
marker = '## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
assert text.count(marker) == 1
latest = f'''\n### {stamp} 최우선 — 재개 후 새 촬영 평가 도구 검증 완료, 촬영 준비\n\n- 사용자 “빨리 다음 진행해줘.”에 따라 Sep19 일시중단을 해제했다. Claude Code 실제 반환 코드와 수정안을 적용해 새54슬롯 집계/실행 도구를 완성했다. 합성47/47·무결성6/6 PASS이며 실제 새 촬영/세 모델 성능 비교는 아직0이다. 기존 화면33/33·브라우저21/21과 완료된 모델 비교는 재사용했다.\n- 원래 서버 확인 기록을 정답 근거로 사용하고, 취소·오류·미확인·미촬영을 모두 보존한다. 같은 새 관측에 부모/대조/추가군을 적용하며 모델·관측·구현 해시가 바뀌면 거부한다. 분류 결과를 올바른 수행 판정으로 바꾸지 않는다.\n- Claude의 빈 미리보기 CSS를 적용하고 실제 Chrome에서 시범 표시·카메라 닫힘·세션 없음·시작 버튼 활성화를 확인했다. 준비 URL http://127.0.0.1:8765/frozen-capture.html, 서버PID24448/핸들17679. 사용자 직접 준비 전 카메라를 열지 않았다. 보호파일13개 해시 불변, 새 학습/실제추론/촬영/승격0.\n- 상세 실행법과 검증: `research/front_keyphase_v1/ops/fresh_self_comparison_20260919/resume_20260920/RESULT.md`. 다음은 사용자 준비 시 새 촬영→종료/진행기록 저장→1회 오프라인 비교·독립 재집계다. 프로젝트 전체 완료나 다인 일반화 검증 완료를 뜻하지 않는다.\n'''
text = text.replace(marker,marker+latest,1)
marker9 = next(line for line in text.splitlines() if line.startswith('## 9.'))+'\n'
next_text = f'''\n### {stamp} 다음 단일 작업 — 준비된 화면에서 새 촬영 후 고정 세 모델 비교\n\n- Claude 평가 도구 `ops/fresh_self_comparison_20260919/resume_20260920/generated_v2/` 구현·검증 완료. 이미 통과한53개 새 검사와 이전33/21·모델 결과를 반복하지 않는다.\n- 사용자 준비 시 카메라 열기→고정18단어×3회54슬롯→실제로 한 표현 확인→종료 후 진행 기록 JSON 저장. 중단된 슬롯도 유지한다. 새로고침으로 재개하거나 과거18건을 새 시험으로 대체하지 않는다.\n- 서버 CLOSED·원본 확인 기록·관측 파일을 묶어 prepare로 고정하고 execute를1회 실행한다. 같은 관측의 세 모델18제한/전체93 결과, 개선/회귀와 누락 분모를 독립 재집계한다. 그전까지 새 성능은 UNVERIFIED다.\n- 사용자기록 비학습·보류시험 보호·전체18목표·기본모델 보존·Claude 작성 역할 유지. 추가다운로드/문의·삭제·백업·Unity빌드·예약작업 없음. 결과가 나온 후 다음 개선을 결정하며 피드백 확장/Unity 통합을 선행하지 않는다.\n'''
text = text.replace(marker9,marker9+next_text,1)
context.write_text(text,encoding='utf-8')
print(json.dumps({'status':report['status'],'at_kst':stamp,'checks':[47,6],'protected_unchanged':len(pins),'camera':health['camera'],'server_pid':24448}))
