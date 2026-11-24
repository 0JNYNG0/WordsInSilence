"""Finalize verified research results and synchronize current project records."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib, shutil
U=Path(__file__).resolve().parent; ROOT=U.parents[3]; OPS=U.parent
KEY=U.name
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def exclusive(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

assert not (U/'COMPLETION.json').exists()
frozen=read(U/'FROZEN.json'); ev=read(U/'EVALUATION.json')
for collection in ['pins','protected']:
    for path,h in frozen[collection].items():assert sha(path)==h,path
verify=read(U/'review/RESULT_VERIFICATION.json')
assert verify['status']=='PASS' and verify['failure_count']==0
qas=sorted((U/'qa').glob('BROWSER_QA*.json'),key=lambda p:p.stat().st_mtime)
qa=qas[-1]; assert read(qa)['status']=='PASS'
assert ev['technical_failure_count']==0 and ev['natural']['complete']==293 and ev['natural']['unknown']==292
assert not (U/'RUNNING.json').exists()
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
server=read(U/'REPORT_SERVER.json')
files=['RESULT_KO.txt','RESULT_KO.html','VIEW_DATA.json','viewer.js','viewer.css','build_report.py',
       'serve_report.py','qa_browser.py','finalize.py','NEXT_PLAN_KO.txt','FROZEN.json','EVALUATION.json','RESULT_MANIFEST.json',
       'review/RESULT_VERIFICATION.json','review/PREEXECUTION_REVIEW.json','review/POSTEXECUTION_REVIEW.json',
       str(qa.relative_to(U))]
pins={str(U/f):sha(U/f) for f in files}
next_action='관측 공백과 기존 부모 범위를 보존하면서 연속 분절선 비교 후보 하나로 같은 경로의 표본 간격 영향과 실제 꺾임·순서 차이를 분리 검증한다. 전체 가속 결과를 0으로 맞추지 않으며, 공백/범위 밖 변경은 보류한다. NEXT_PLAN_KO.txt 참조. 아직 실행하지 않음.'
status='COMPLETE_VERIFIED_RESEARCH_COMPARISON_NOT_PRODUCTION_ADOPTED'
summary=dict(public_words=4,public_clips=16,natural_pairs=36,parent_instances=117,
    natural_components=585,complete=293,unknown=292,ambiguous_complete_paths=291,
    result_records=2825,missing_component_opportunities=10,control_cases=160,computed_control_cases=158,
    hold_complete_zero=123,hold_edited_complete_zero=39,speed075_complete_zero=123,
    speed125_complete_positive=123,speed125_same_endpoints_positive=99,swap_outside_crop_components=95,
    independent_checks=verify['checks'],independent_failures=0,browser_checks=len(read(qa)['checks']),
    no_proven_accuracy_gain=True,training=0,model_inference=0,production_changed=False)
completion=dict(status=status,at_kst=now,summary=summary,report_url=server['url'],
    report_server=dict(pid=server['pid'],port=server['port'],exec_session=35320,read_only=True),
    code_review='same Codex separate agent, not independent human KSL truth',
    implementation='root + semantic_feedback',method_review='semantic_model',
    independent_verification='semantic_data_audit',browser_qa='root installed Edge isolated headless',
    protected_files_unchanged=True,user_mentor_access=False,external_download=False,
    geometry_comparison_complete=True,educational_accuracy=None,recognition_accuracy=None,
    first_failure_preserved='FIRST_PREP_FAILURE.json',
    ui_environment_recovery='cua/node kernel assets unavailable; first local socket/timeout QA preserved; verified original server session stopped then ordinary-context read-only server started',
    next_single_action=next_action,pins=pins)
exclusive(U/'COMPLETION.json',completion)
write(U/'CURRENT_EXECUTION.json',dict(status=status,active=None,training=0,model_inference=0,
    report_server=completion['report_server'],report_url=server['url'],next_action=next_action))

for path in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',OPS/'STATE.json',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']:
    backup=U/'before'/path.name
    if not backup.exists():shutil.copy2(path,backup)
state=read(OPS/'STATE.json')
state.update(current_stage='HOLD_INVARIANT_TRAJECTORY_COMPLETE_NOT_ADOPTED',updated_date_kst=now,
    updated_at=now,active=None,activity_state='IDLE_REPORT_AVAILABLE',next_single_action=next_action,
    resume_command=next_action,latest_visual_review=str(U/'RESULT_KO.html'),
    goal_status='INCOMPLETE: trajectory component and visual comparison verified; educational feedback accuracy and production integration remain unvalidated.')
state[KEY]=dict(status=status,at=now,active=None,unit=str(U),report=str(U/'RESULT_KO.html'),
    report_url=server['url'],summary=summary,verification=str(U/'review/RESULT_VERIFICATION.json'),
    browser_qa=str(qa),report_server=completion['report_server'],new_training=0,new_inference=0,
    production_changed=False,private_access=False,next_single_action=next_action)
write(OPS/'STATE.json',state)
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=context.read_text(encoding='utf-8-sig')
entry8='''
### 2026-11-01 정지 시간·이동 경로 분리 비교 완료 — HTML 공개, 정확도 향상 미입증
`word100_hold_invariant_trajectory_20261101`의 승인된 네 단계 완료. 기존global/좌표/마스크/좌우·교차/손모양출력 보존 후 공통5성분 단조min-max 후보 하나를 별도 비교. 공개4표현16영상36쌍117부모×5=585기회, 완전293/보류292. 자연완전293중291은최적경로복수로정확시점확정불가.
기술160사례중158계산+2미제공 보존. 결과2825+missing10. hold·slow123완전관측각각0(hold실제변경39); 기존도0이므로개선증거X. 정상speed125123양수중99동일endpoint이며sampling영향미해결. 순서교환95성분변형일부범위밖을원장유지. 생긴차이를교육오류로바꾸지않음.
자체149PASS, 별도Codex 전수3,054,883검사오류0, 브라우저15검사PASS. 같은Codex검산≠독립수어전문가정답. 새학습/모델추론0, 기존완료경로/점수재실행0(새기술입력64의고정baseline만1회). C·T3F·운영·사용자/멘토·원본·과거동결보존. HTML은이번사용자명시요청.
보고서 `http://127.0.0.1:8800/`에서 원본2영상/이전·현재경로/궤적/공백/기술변형을구분. 일반실행읽기전용서버exec35320,계산종료. 첫입력경로오류와UI실행환경오류보존후검증복구. RESULT_KO.txt/COMPLETION.json/review/POSTEXECUTION_REVIEW.json/qa/BROWSER_QA_03.json 참조. 프로젝트전체·교육피드백완성아님.

'''
entry9='''
### 2026-11-01 다음 한 작업 — 표본 간격과 실제 이동 경로의 차이 분리
현재고정비교/전수검산/HTML보고는완료. 다음은연속관측의분절선경로후보하나로같은직선위재표본·정지중복과실제꺾임/순서차이를분리한다. 공백parent전체null유지/좋은block만골라전체정상승격금지. 모든speed125를0으로만드는조건X;꺾임점삭제는실제관측경로차이일수있음. 기존global·부모범위·손모양출력보존/범위밖변형은보류. 기술시험과실제교육오류분리. 새모델순회·점수튜닝·사용자자료·운영교체X.
세부 `research/front_keyphase_v1/ops/word100_hold_invariant_trajectory_20261101/NEXT_PLAN_KO.txt`. 다음후보아직구현·실행안함. 이번완료결과반복실행X. 확인된공통기하한계만좁게검증후손모양·양손관계·표현설명연결로이어간다.

'''
for prefix,entry in [('## 8.',entry8),('## 9.',entry9)]:
    line=next(s for s in txt.splitlines() if s.startswith(prefix))
    txt=txt.replace(line+'\n',line+'\n'+entry,1)
context.write_text(txt,encoding='utf8')
handoff=f'''# WordsInSilence 최신 인수인계

현재 단위 `{KEY}` 완료. 계산·학습 실행중 없음. 보고서만 일반 실행 읽기전용 서버 `{server['url']}` (exec35320, PID {server['pid']})로 열어둠. 재개시 실제 상태 확인, 중복실행 금지.

목표는 설명과 여러 참고를 이용해 중요한 손모양·관계·이동·순서를 이해하고 정상 변이와 구분하는 피드백. 이번은 정지 시간과 이동 경로를 분리하는 공통 기하 후보 하나의 검증이며 전체 목표 완성 아님.
공개4표현16영상36쌍117부모/5성분585기회 중 완전293·보류292. 최적경로복수291/293. 기존global·관측·좌우/교차·3seed손모양/양축출력그대로.
기술160사례158계산+2없음, 결과2825+missing10. hold/slow123complete각각0(hold편집39)이나 기존도0. speed125123positive(99동일끝점) sampling영향,swap95성분변형일부구간밖. 숫자감소≠정렬/교육정확도향상. 연구후보운영미채택.
자체149PASS, 다른Codex방법/수치전수검산3,054,883오류0, Edge UI15PASS. 수어전문가정오와다름. 새학습/모델추론0. 기존경로/점수재실행0, 신규기술입력64baseline만1회. 사용자/멘토자료접근0. C·T3F·운영웹/DTW·원본W:/sources/과거동결보존.
첫준비경로오류 FIRST_PREP_FAILURE.json 보존후공개원본명시경로/hash로복구. cua/node도구kernelpath실패와sandbox localhost접속실패는qa에보존;기존서버세션종료확인후일반환경서버/설치Edge로QA완료. 다운로드·외부전송·카메라·Git없음.

근거: `{U}`의 RESULT_KO.html/txt, COMPLETION.json, EVALUATION.json, review/RESULT_VERIFICATION.json, POSTEXECUTION_REVIEW.json, qa/BROWSER_QA_03.json.
구현root/semantic_feedback,방법semantic_model,별도검산semantic_data_audit. HTML에실제원본영상과저장좌표변형을구분,이전평균/최대/새최소최대의다른집계명시. 관측없음을잘맞았음으로표시하지않음.

다음 한 작업(아직 미실행): {next_action}
임계값순회/새모델/학습으로이한계를감추지않음. 새샘플밀도분석도전단어효과나교육판정안전성을미리보장하지않음. 목표변천사는 Docs/PROJECT_DIRECTION_HISTORY_KO.txt.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
with history.open('a',encoding='utf8') as f:
    f.write(f'''\n\n2026-11-01 정지 시간을 강요하지 않는 이동 경로 비교\n이동과 짧은 유지가 연결됐다는 이유만으로 정상 수어를 틀렸다고 하면 안 되어, 기존 정렬을 보존한 별도 궤적 비교 하나를 구현했다. 공개4표현16영상36쌍과기술160사례를고정검증. 자연585성분중293전체관측/292보류, 정지·느림보호는기존도성공해서새성능향상으로부르지않았다. 정상가속표본삭제123에서도차이가남고291자연경로는최적대응복수여서정답시점이나교정을확정하지못했다. 전체목표를바꾼것아니며정확한교육피드백을위해기하차이와언어오류를분리한단계다. 새학습0/운영미적용. 사용자요청으로이전/현재영상·그래프HTML제공,전수검산과브라우저QA완료. 다음은같은경로의샘플간격과실제꺾임차이만좁게검증하고손모양·표현설명연결로이어간다. 근거 {U}/RESULT_KO.txt 및 COMPLETION.json.\n''')
checks=dict(frozen_and_protected_hashes=len(frozen['pins'])+len(frozen['protected']),
    independent_status=verify['status'],browser_qa=str(qa),state_status=read(OPS/'STATE.json')[KEY]['status'],
    completion_files_exist=all(Path(p).exists() for p in pins),new_training=0,new_model_inference=0,
    state_context_handoff_history_synchronized=True,report_server_exec_session=35320,
    screenshots_visually_reviewed=['desktop_comparison.png','candidate_endpoint.png','trajectory_graphs.png','technical_speed.png','mobile_summary.png'])
assert checks['completion_files_exist'] and checks['state_status']==status
exclusive(U/'FINAL_VERIFICATION.json',dict(status='PASS',at_kst=now,checks=checks,completion_sha256=sha(U/'COMPLETION.json')))
print(json.dumps(dict(status=status,report=server['url'],summary=summary),ensure_ascii=False))
