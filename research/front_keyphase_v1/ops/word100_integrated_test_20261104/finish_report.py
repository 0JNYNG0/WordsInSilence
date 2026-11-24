"""Read-only artifact verification, then synchronize the current handoff."""
from pathlib import Path
import ast,json,hashlib,datetime,subprocess
U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(n,x):assert x,n;checks.append(n)
smoke=read(U/'PUBLIC_SMOKE.json');browser=read(U/'BROWSER_VERIFIED.json');http=read(U/'HTTP_VERIFIED.json')
check('actual public integration PASS',smoke['status']=='PASS')
check('browser PASS',browser['status']=='PASS' and browser['count']==37)
check('http PASS',http['status']=='PASS' and http['count']==16)
check('actual integrated output hash',sha(smoke['result'])==smoke['result_sha256'])
r=read(smoke['result']);oldrun=read(O/'word100_local_event_runtime_20261103/PUBLIC_RUN_RESULT.json');old=read(oldrun['result'])
for field in ['path','path_axes','stages','overlay','source_sha256','word_id']:
 check('preserved public '+field,r[field]==old[field])
check('one learned DTW',r['learned_prototype_comparisons']==1)
check('no individual reference comparisons',r['runtime_individual_reference_video_comparisons']==0)
check('no user recordings accessed or made',not (U/'private_sessions').exists())
check('no new training',smoke['training']==0 and r['training'] is False)
check('normal and unknown separated',r['whole_sign_correctness'] is None)
for p,h in read(O/'word100_local_event_demo_20261103/EXECUTION_FROZEN.json')['pins'].items():check('prior pin '+Path(p).name,sha(p)==h)
for p in U.glob('*.py'):ast.parse(p.read_text(encoding='utf8'));checks.append('parse '+p.name)
logic=subprocess.check_output(['node',str(U/'test_logic.js')],text=True,encoding='utf8');logic=json.loads(logic);check('capture/display logic',logic['status']=='PASS')
launches=[read(p) for p in U.glob('DEMO_LAUNCH_*.json')];pid=launches[-1]['pid'] if launches else None
date=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
summary=('새 통합 촬영 페이지 word100_integrated_test_20261104 / http://127.0.0.1:8812 구현·검증 완료. '
 '3초 카운트다운→촬영→손내림 근거 확인 시 자동 종료→TCN+기존 학습 SignRep/DTW 분석을 연결했다. '
 '잠깐의 가림만으로 종료하지 않으며 수동 종료/19초 상한도 유지한다. TCN은 목표를 주지 않고 기존100개 순위를 계산하고 상위3개만 가로로 표시, 목표단어 색 강조, C/T3F 선택 가능. '
 '참고/촬영 병렬 재생·학습 canonical DTW 시점·손/팔 강조·주요/준비/마무리/사이 후보 막대와 유사/차이후보/확인어려움 막대를 분리했다. '
 '새 공개 통합1회 118native frame, 약9.309초(업로드·cold boot 제외), GPU.0 SignRep 사용. 기존AK 공개 path/overlay/stages exact, C/T3F 순위 exact 확인. '
 '실제 새 입력 학습기 추론과 웹 연결 검사이며 정답률 개선 주장은 아니다. 논리45·격리브라우저37·HTTP16 검사, 실제앱 공개재생 확인. '
 '같은Codex 구현/별도저장검산이며 독립사람 검증이 아니다. 실사용 카메라 동작은 사용자가 확인할 차례다. '
 '첫 browser 대기검사 오류, 빠른seek의 참고시점 누락을 보존·수정했고 HTTP 과대본문 전송의 연결리셋은 헤더선거부 검사로 확인. '
 '지원 피드백4표현, 전체정오/접촉/반복/모든중요차이 미검증; 추정 연결구간을 의미없음으로 확정하지 않음. '
 'C·T3F/기존AM/8811/원본 보존, 사용자·멘토 접근/학습/튜닝/선택0, 새학습0. AO 연구 일시중단 유지.')
next_action=('앞으로 통합 웹 시험은 8812와 word100_integrated_test_20261104를 기준으로 계속 수정한다. '
 '사용자가 카메라 켜기→3초 후 촬영을 눌러 새 촬영을 추론 전용으로 확인한다. 카메라는 에이전트가 켜지 않았으며 자동종료의 실제환경 성공을 보장하지 않는다. '
 '공개/가짜 촬영 검사는 완료되어 새 문제 없이 반복하지 않는다. 사용자 촬영은 학습·튜닝·후보선택 금지. '
 '연구 AO의 FIT72/DEV76 및 pending AH120, 실제 preflight/훈련/평가0은 사용자 재개 전 그대로 보류. 기존모델 운영교체 없음.')
report=dict(status='COMPLETE_ISOLATED_INTEGRATED_WEB_TEST',at=date,url='http://127.0.0.1:8812/',pid=pid,
 summary=summary,next=next_action,checks=checks,check_count=len(checks),logic_checks=logic['checks'],browser_checks=browser['count'],http_checks=http['count'],
 actual_public_inference=1,user_content_access=0,training=0,model_accuracy_improvement_claim=False,
 verification='Implementation and separate artifact checks by same Codex; no independent human correctness labels')
write(U/'FINAL_VERIFIED.json',report)
(U/'RESULT_KO.txt').write_text(summary+'\n\n'+next_action+'\n\n'+
 '사용법: 연습 단어 선택 → 카메라 켜기 → 3초 후 촬영 → 동작 후 손내림. 자동 종료가 안 되면 촬영 종료. 분석 후 가로 상위3개, 동작구간, 비교근거를 확인. 공개 예시 보기로 촬영 없이 화면 확인 가능.\n'+
 '참고영상4개는 기존 W: 공개 파일을 읽으므로 W: 연결이 필요하다. 사용자 영상은 새 private_sessions에 로컬 추론용으로만 저장되고 기존 사용자폴더는 스캔하지 않는다.\n'+
 '검사 세부: PUBLIC_SMOKE.json, BROWSER_VERIFIED.json, HTTP_VERIFIED.json, FINAL_VERIFIED.json. UI_1366.png는 실제 공개 저장 결과 화면; UI_SYNTHETIC_DIFFERENCE.png는 UI 검사전용 합성 상태이며 모델 성능 증거가 아니다.\n',encoding='utf8')
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text(encoding='utf8')
for header,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',next_action)]:
 title='### 2026-11-04 통합 촬영·TCN·학습형 동작 피드백 페이지 완료'
 idx=text.index(header);end=text.index('\n',idx)
 text=text[:end+1]+'\n'+title+'\n\n'+body+'\n'+text[end+1:]
context.write_text(text,encoding='utf8')
s=read(O/'STATE.json');s['next_single_action']=next_action;s['word100_integrated_test_20261104']={k:report[k] for k in ['status','at','url','pid','summary','next','logic_checks','browser_checks','http_checks','actual_public_inference','training']};s['active_web_test']='word100_integrated_test_20261104';write(O/'STATE.json',s)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+date+'\n\n'+summary+'\n\n'+next_action+'\n\n'+
 f'새 서버 URL http://127.0.0.1:8812/ , PID {pid}. 실제 포트/프로세스부터 확인하고 중복 실행하지 않는다. 실행파일 `{U / "run_demo.py"}`, Python은 word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. 일반 실행환경에서 서버 시작, 카메라는 사용자 버튼으로만 시작. 이전8811/AM서버·모델은 보존.\n\n'+
 'AO 중단 상태: word100_motion_aware_order_20261103 / PAUSE_CLOSEOUT.json. 공개 FIT72(36원본 ID01~09), DEV76(12원본 ID11~13), exact 재사용28, pending AH120(native Y snapshot48 포함), 별도입력검토·실제preflight·새학습/평가 미완료. 이번 웹 통합은 연구 재개가 아니다.\n\n'+
 '고정모델·출처·시각을 보존하고 사용자 촬영은 추론 전용. 목표는 여러 수행자로 배운 동작과 설명을 바탕으로 새로운 사용자의 중요한 차이를 알려주는 것. 한 사람 영상과의 실행시 반복비교로 돌아가지 않는다. 현재 결과를 전체 한국수어 정답 판정으로 과장하지 않는다.\n',encoding='utf8')
check('state parses',read(O/'STATE.json')['active_web_test']==U.name)
print('COMPLETE',len(checks),'pid',pid)
