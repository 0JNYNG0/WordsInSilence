"""Finalize completed read-only analysis/UI work; no model execution."""
import json
import hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
U=Path(__file__).resolve().parent;B=U.parents[1];R=B.parents[1]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
checks=read(U/'CHECK.json');ui=read(U/'UI_CHECK.json')
assert checks['status']==ui['status']=='PASS'
guard=read(U/'ORIGINAL_GUARD.json')
changed=[p for p,h in guard.items() if sha(p)!=h]
allowed=str(B/'runs/ses_79977f57893d/session.json')
assert changed==[allowed],changed
assert read(allowed)['state']=='CLOSED'
write(U/'FINAL_GUARD.json',dict(status='PASS',original_capture_npz_results_reviews_and_model_unchanged=True,
    changed_session_metadata=changed,reason='Explicit idle session close during server update',
    before_session_metadata='before/session_before_server_restart.json'))
server=read(U/'SERVER_PROCESS.json')
files=['src/wis_lab/service/word100.py','src/wis_lab/service/word100_analysis.py','src/wis_lab/service/app.py',
       'web/practice.html','web/practice.css','web/practice.js','config/word100_analysis_references.json']
done=dict(status='COMPLETE_VERIFIED_DESCRIPTIVE_ANALYSIS',at_kst=now,implementer='Codex local',
    verification='Codex separate trigonometry/closed-form checks, camera-blocked APIs and browser QA; not an independent expert',
    checks=len(checks['checks']),ui_check='UI_CHECK.json',original_guard='FINAL_GUARD.json',
    changes=['Top5 from same100 logits','0-100 model-score bars; not execution correctness',
             'Read-only requested-demo DTW:2D angles, body-relative location, relative trajectory',
             'Top3 reference video row','All3 saved user attempts and replay selectable','Observation coverage and frame gaps'],
    model_sha256='e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f',
    model_retrained=False,user_inference_rerun=False,user_data_tuning=False,real_camera_opened_by_agent=False,
    reference_check=read(U/'REFERENCE_CHECK.json'),url='http://127.0.0.1:8765/practice.html',server=server,
    limitations=['DTW compares one2D reference; no pass/fail calibration or expert handshape correctness validation.',
        'Visible-hand span includes preparation/return while hands remain visible; not a semantic-core segmenter.',
        'Missing hands are not zero error; reported support is aligned-pair coverage, not original elapsed time.',
        'No causal explanation or accuracy gain established from these3 user attempts.'],
    files={p:sha(B/p) for p in files})
write(U/'COMPLETION.json',done)
context=R/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text('utf-8')
entry8='''### 2026-10-13 100단어 웹 상세 분석 완료 — 기존 세 촬영 보존
`ops/word100_analysis_20261013/COMPLETION.json` COMPLETE_VERIFIED_DESCRIPTIVE_ANALYSIS. 기존2×2 위쪽 유지,03은5순위로 확대. 아래에 전체100개 점수에서 계산한 상위5 가로 막대/선택표현 전체순위/1·2위 차이, 요청 참고동작 DTW 수치, 상위3 참고영상1×3, 촬영기록 목록·다시보기 추가.
사용자 실제 기록 ses_79977f57893d: 수어1회/감사2회 모두 npz·capture·원 결과 보존, 확인메모는 수어/두 번째 감사2개. 첫 감사1위 유성구주민지원센터51.1점·감사44.9점, 두 번째 감사85.7점. 점수는 정답률/일치율 아님. 원100logits 재표시이며 사용자추론 재실행·학습·튜닝0.
DTW는 예측1위가 아닌 촬영 때 바인딩한 요청 참고영상과 비교. 동일MediaPipe 관절캐시, 최대96개 실제시간 최근접표본/밴드0.2/기존관측근거가드 재사용. 이전18단어 교정척도·합격선·손참여휴식 제외 규칙 미사용.2D손가락마디 평균차이(도), 손목위치/첫 관측위치 대비궤적 차이(어깨너비100기준), 손별 대응쌍 비율·손가락별 각도 표시. 준비/복귀 포함 첫~마지막 손 관측구간 비교이며 의미핵심검출 아님. 실제3D각도/수형정답/인과원인 미입증.
상위3 어떤 예측에도 영상이 있도록100클래스 각각 최저학습자ID01의 정면영상1개·기존관절 해시검증. 추가407,221,246바이트, 총461,354,948바이트 W의 기존참고캐시. 다운로드·재추출0, 연습선택은12개 유지, 평가/보류사용0.
38개 수치·API검사PASS(자기차이0/평행이동·크기불변/별도삼각함수/결측null/기존결과불변), 실제브라우저 기존기록·5순위·DTW·세영상·저장재생 확인. Codex 구현/별도계산검산이며 다른 전문가검수 아님. 모델 e39d8ac4/기존기본 불변, 정확도개선 주장 없음.
서버11368/8765 일반환경으로 갱신. 기존 세션은 촬영중 아님 확인 후 종료, session.json 상태만CLOSED로 갱신(before보존), 나머지 원본/결과/확인기록/모델 해시 유지. 카메라 자동재개 없음. 기존 사용자창은 새로고침 필요.
'''
entry9='''### 2026-10-13 다음 — 저장된 감사 두 시도의 상세 분석 사용자 확인
웹 practice.html의 결과 분석→촬영 기록에서 첫 감사/두 번째 감사를 선택해 원 순위·DTW 차이·상위3영상을 비교한다. 추가 촬영 없이 확인 가능. 새촬영은 사용자가 카메라 연결을 눌러 시작. 기존 메모 없는 첫 시도도 자동기록에 보존됨.
서버 PID11368/8765 재개 시 실제확인. 모델 학습·튜닝·원본결과덮어쓰기·보류채점 금지. 진단수치가 큰 이유를 수어 오류로 단정하거나 단어별 임계값을 만들지 않는다. 현재참고1인·2D가림·준비/복귀/수형변형 한계를 유지하고, 실제정답판정/교정 검증은 별도 범위로 남긴다. 발표 자료는 계속 유예.
'''
for heading,entry in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',entry8),('## 9. 다음 단일 작업',entry9)]:
    assert entry.split('\n')[0] not in s
    s=s.replace(heading+'\n',heading+'\n\n'+entry+'\n',1)
context.write_text(s,'utf-8')
statefile=B/'ops/STATE.json';state=read(statefile)
state['updated_date_kst']=now;state['current_stage']='WORD100_WEB_TOP5_DTW_ANALYSIS_READY'
state['next_single_action']='사용자가 기존 감사2회 기록의5순위·DTW·상위3영상을 확인. 카메라/학습/튜닝 자동실행 없음.'
state['word100_analysis_20261013']=done
write(statefile,state)
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계
갱신 {now}. AGENTS/CONTEXT 최신§3·§8·§9/WORKFLOW 우선.

## 마지막 완료
`ops/word100_analysis_20261013` COMPLETE_VERIFIED_DESCRIPTIVE_ANALYSIS. 사용자 요청5순위/막대그래프/DTW/상위3영상/기존2×2유지 완료. 실제 URL http://127.0.0.1:8765/practice.html.
상단4칸 유지, 하단상세분석 추가. 기존세촬영(ses_79977f57893d, 수어1/감사2) 원100logits로5순위 확장, 재추론/학습/튜닝0. 확인메모는2개지만 세번 모두 원 관절/결과/다시보기 저장됨. 촬영기록 선택으로 첫실패/둘째성공 모두 볼 수 있음. 단어맞춤과 수형정답은 구분.
DTW는 requested reference와만 비교: 2D손가락각도(도), 손목위치/처음위치기준이동궤적(어깨너비100기준거리).96개실제시간최근접표본/밴드0.2. 기존18합격척도/휴식손제외/임계값 가져오지 않음. 결측null+대응쌍근거표시, 원 촬영관측률은 별도분모. 준비/복귀포함 손관측범위이며 의미핵심 자동검출아님. 정답/오답검증 및 원인인과해석은 미완료.
100개 학습자ID01 참고영상·기존MediaPipe관절 검증, W:/WordsInSilence/data/AIHub_103/word100_web_references_20261013 총461,354,948바이트(이번추가407,221,246). 연습메뉴12, 인식은전체100.
CHECK38개PASS + UI_CHECK + FINAL_GUARD. Codex 구현/별도수치·API검산/브라우저QA, 외부전문가/다른에이전트검수아님. 단위SAVED_ANALYSES는 원본을 덮어쓰지 않은 파생진단이며 정확도평가아님.

## 서버 / 다음
새 PID11368/8765 일반환경(require_escalated), 로그는 위 단위server_stdout/stderr. 재개 때실제확인후재사용. 이전62560 종료. 세션촬영없음확인후close, session.json만CLOSED로 갱신(before보존). 나머지 원 관절·capture·결과·메모·모델해시 유지.
카메라 새로 열지 않음. 기존창 새로고침하면 최근저장촬영 로드. 다음은 사용자가 두감사 기록의막대/수치/영상을 비교검토하는것. 추가학습·보류시험·사용자튜닝·운영기본교체 자동시작없음.
실제 카메라 새 촬영 연결의 새코드 UI흐름은 이번에는 재촬영안함. 기존촬영저장재생/API는 검증됨. 외장W 연결필요. 발표문서는 마지막으로유예.

## 앞선 완료 — 반복 금지
word100_web_20261013:12참고/100분류 연결+2×2UI,22검사PASS.
word100_training_20261013:AIHub-only fresh MaskedSignTCN100/280채널/37,636매개변수.899학습ID01~09/300개발ID11~13, 전체1,199영상150,476프레임.
전체277/300=92.33%, 전체·핵심혼합290/300=96.67%,+4.33pp(개선17/회귀4). 개발로epoch선택도했음→독립최종시험아님. 기존93과직접비교금지.
새연습모델 whole_context e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f. 기본기존모델보존, NIKL/KSL/사용자/멘토·ID10/14~18 학습미사용. 캐시 W:/WordsInSilence/data/AIHub_103/word100_inputs_20261013.
''','utf-8')
assert read(statefile)['current_stage']=='WORD100_WEB_TOP5_DTW_ANALYSIS_READY'
print(json.dumps({'status':'RECORDED','checks':len(checks['checks']),'server_pid':server['pid']}))
