from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
U=Path(__file__).resolve().parent;B=U.parents[1];ROOT=B.parents[1]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
numeric=read(U/'INDEPENDENT_CHECK.json');api=read(U/'API_CHECK.json');web=read(U/'WEB_CHECK.json')
assert numeric['status']==api['status']=='PASS'
assert web['metricRows']==3 and web['handRows']==6 and not web['overflow'] and web['cameraOff']
assert all(sha(ROOT/p)==h for p,h in read(U/'ORIGINAL_GUARD.json').items())
baseline=read(U/'BASELINE.json');assert all(sha(p)==h for p,h in baseline['code_sha256'].items())
assert sha(U/'BASELINE.json')==numeric['baseline_sha256']==api['baseline_sha256']
rep=ROOT/'Docs/interim_report_20261014/web';html=(rep/'index.html').read_text('utf-8')
assert 'id="test-plan"' not in html and '실사용36회안내' not in html and '새 실사용 시험 아님' in html
files=['src/wis_lab/service/word100_analysis.py','src/wis_lab/service/word100_multiref.py','web/practice.js','web/practice.html','web/practice.css','config/word100_multiref.json']
out=dict(status='COMPLETE_VERIFIED_READY_FOR_USER_TEST',at_kst=now,implementation='Codex local',
    verification='Codex separate median/ratio recomputation,144 source-array checks, API and actual browser; not external expert review',
    numeric_checks=len(numeric['checks']),api_checks=len(api['checks']),practice_words=12,training_clips=108,
    training_performers=9,training_ordered_pairs=864,reused_pairs=96,new_pairs=768,
    development_clips=36,development_calculable_axes=108,
    development_result_meaning='coverage and numeric consistency only, not sign-correctness accuracy',
    baseline_sha256=sha(U/'BASELINE.json'),model_sha256=baseline['model_sha256'],
    user_training=False,user_calibration=False,model_change=False,original_files_preserved=True,
    current_user_trials=0,retrospective_trials=3,real_camera_started=False,
    graph='all3 axes + left/right; common dynamic0..ceil(max ratio) display range; fixed1x marker and unmodified numeric ratios',
    limitations=['2D projected flexion and wrist differences, not full handshape semantics',
        'reference8 leave-one-out vs live9 median comparisons; descriptive ratio not calibrated acceptance threshold',
        'camera, style and legitimate variant differences may increase distance; no correctness probability',
        'both hands required for combined; absence remains null'],
    server=read(U/'SERVER_PROCESS.json'),
    artifacts=dict(numeric='INDEPENDENT_CHECK.json',api='API_CHECK.json',browser='WEB_CHECK.json',
        correction='CORRECTION.json',screenshot='DTW_GRAPHS.png'),
    code_sha256={p:sha(B/p) for p in files},presentation_sha256={p:sha(rep/p) for p in ['index.html','report.js']},
    next='User performs actual tests. Aggregate all attempts after frozen start, including unconfirmed/misrecognised/invalid/cancelled, report denominators separately and replace same presentation with actual results. Never tune from user data.')
write(U/'COMPLETION.json',out)
write(U/'TEST_READY.json',dict(at_kst=now,model_sha256=baseline['model_sha256'],baseline_sha256=sha(U/'BASELINE.json'),
    browser_participant=web['participant'],existing_result_ids=[r['trial_id'] for r in read(U/'SAVED_ANALYSES.json')],
    purpose='학습 미포함 개발참여자1인 실사용 평가; 독립된 새 사람 일반화의 증명으로 확대하지 않음',
    recommended_trials='현재12표현을3라운드, 표현당3회. 실제 실시 수를 집계하며 계획을 결과로 적지 않음',
    aggregation='all attempted trials including non-success; separate Top1/Top3/Top5 among model-ready attempts and failure/cancellation counts; DTW descriptive distributions and unmeasured counts separately',
    forbidden=['training','threshold calibration','candidate selection','best-take-only reporting']))
(U/'RESULT_KO.txt').write_text('''학습자료 대비 DTW 상대 차이 표시 완료

현재 연습12표현 × 학습 촬영자9명 =108영상으로 비교 기준을 구성했다.
각 학습자는 자신을 뺀8명과 비교하며, 손별 차이 중앙값→양손 평균→9명 중앙값을 항목별1배로 삼는다.
사용자는 같은 단어의9명과 비교한다. 손가락 굽힘/손 위치/이동 궤적 모두 같은 의미의 상대 차이를 전체·왼손·오른손 가로막대로 표시한다.
단위가 다른 원래 수치는 보조로 보존한다. 정답률·합격선·TCN 모델 가중치와의 거리는 아니다.

864쌍 중 과거검산96쌍 재사용/768쌍 새 계산. 기준에 포함하지 않은 공개 개발36영상의108항목 계산 가능.
별도통계 검산16개/API검사19개 통과.144개 캐시의 숫자배열은 원본 변환과 정확히 일치.
실제브라우저3항목/6손그래프/공통눈금/기준1배/넘침없음 확인. 과거3촬영 새분석은 별도 파일에만 추가했고 원결과·모델SHA 보존.
구현/검산은Codex 로컬. 전문가 정상/오류 정답 검증이나 실사용 정확도 향상이 입증된 것은 아니다.

최초문제: 제한환경Windows asyncio소켓 대기→카메라/재추론차단 일반환경 검산.
연결문제: 캐시의불필요한object메타데이터를읽으려함→pickle허용없이필요한숫자배열/마스크만읽도록수정.
기준의수치/자료/계수변경0, 이전기준파일및검사보존. CORRECTION.json 참조.

발표웹에서개인테스트계획부록/36회계획문구삭제. DTW설명은완료한기능으로갱신하고 기존감사촬영의새분석임을명시.
아직새실사용시험0회. 사용자가시험을마치면실제모든시도를집계해같은웹발표에반영한다.
사용자기록학습/튜닝/후보선택금지. 진행중학습0. 카메라는사용자버튼으로만시작.
''','utf-8')
statepath=B/'ops/STATE.json';state=read(statepath)
state['updated_date_kst']=now;state['current_stage']='WORD100_TRAINING_RELATIVE_DTW_READY_FOR_USER_TEST'
state['word100_multiref_20261014']={k:out[k] for k in ['status','at_kst','practice_words','training_clips','training_performers','training_ordered_pairs','development_clips','baseline_sha256','model_sha256','user_training','model_change','current_user_trials','retrospective_trials','next']}
state['word100_multiref_20261014'].update(unit='ops/word100_multiref_20261014',completion='ops/word100_multiref_20261014/COMPLETION.json',server=out['server'])
state['next_single_action']=out['next'];write(statepath,state)
ctxpath=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';ctx=ctxpath.read_text('utf-8')
head8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
new8='''

### 2026-10-14 학습자료 대비 DTW 세 항목·양손 공통 그래프 완료
사용자 승인에 따라연습12표현×학습9명108영상으로기준생성.864사람간비교(기존96재사용/768새계산),본인제외8명중앙값→양손평균→9명중앙값을1배로고정.사용자는9참고와비교하며8/9차이명시.손가락굽힘·손위치·이동궤적을전체/좌우가로막대공통눈금,1배점선으로표시.관측부족/분모0은보류,정답률·합격선아님.
별도숫자검산16/API19통과,144캐시원입력일치,개발36영상108항목계산가능.실제브라우저3항목/6손그래프확인. Codex구현·별도계산검산이며전문가타당성/성능향상검증아님.근거ops/word100_multiref_20261014/COMPLETION.json.캐시object읽기오류는필요숫자배열만안전로딩하도록수정,기준수치보존.
TCN e39d8ac4·원본촬영보존/학습0/튜닝0.과거3촬영분석은새sidecar로추가,새실사용0.일반환경서버65612/8765,카메라꺼짐.발표8766개인테스트계획삭제,현재DTW구현설명으로수정(과거사례명시).TEST_READY의시작시각·기준SHA에따라이후실제시도전체집계예정.
'''
head9='## 9. 다음 단일 작업'
new9='''

### 2026-10-14 다음 — 고정 모델·DTW 기준으로 사용자 실사용 후 집계
http://127.0.0.1:8765/practice.html 사용자시험대기.새시도기준은ops/word100_multiref_20261014/TEST_READY.json;현재브라우저참가자self_eval_20261014/다시보기체크,카메라미연결.12표현×3회권고는계획이며발표결과로쓰지않는다.사용자완료통보후모든시도(확인미저장/오인식/분석실패/취소포함)를구분집계,Top1/3/5분모·DTW차이와미관측수를보고하고동일발표웹에실제결과반영.학습미포함개발참여자1인실사용으로표기;독립새사용자전체성능으로확대금지.사용자자료로학습/척도/튜닝/후보선택없음.새학습/다운로드자동재개없음.
'''
assert head8 in ctx and head9 in ctx
ctx=ctx.replace(head8,head8+new8,1).replace(head9,head9+new9,1);ctxpath.write_text(ctx,'utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {now}. AGENTS/CONTEXT 최신§3·§8·§9/WORKFLOW 우선.

## 현재 목표·완료
사용자 요구: DTW세항목을학습자료대비동일한해석으로낮을수록가깝게표시→사용자실시험→실제결과집계후발표웹갱신.
ops/word100_multiref_20261014/COMPLETION.json COMPLETE_VERIFIED_READY_FOR_USER_TEST.
12표현×학습9명108영상,864사람간비교(96재사용/768새계산).손별참고중앙값→양손평균→학습9명중앙값이1배.본인제외8참고기준/사용자9참고차이명시.
손가락굽힘/손위치/이동궤적전체와좌우가로막대.공통 표시 범위는3이상·전체최대비율올림,고정1배점선.관측부족은null,정답률/합격선아님.
숫자검산16/API19 PASS,144캐시숫자배열원본일치,공개개발36개108항목계산가능.실제웹3항목/6손그래프확인.
Codex구현·별도코드검산,외부전문가타당성검수아님.모델/원본SHA보존,사용자학습/튜닝0.
CORRECTION: sandbox asyncio소켓대기→일반환경검산;cacheobjectmetadata읽기오류→pickle없이features/present숫자만읽도록수정.기준수치/계수보존.

## 서버·결과
연습 http://127.0.0.1:8765/practice.html 일반환경PID65612.카메라는사용자가연결버튼을누를때만시작.재개시실제PID확인;중복서버금지.
발표 http://127.0.0.1:8766/ 정적PID58776.개인시험계획부록/36회목표문구삭제,7장DTW완료설명으로갱신.
3과거촬영새DTW는별도 sidecar에추가,원래점수/촬영/확인기록불변.감사2차상대차이1.646/4.743/4.127배는과거촬영의새분석이며새시험아님.

## 다음 한 작업
사용자실시험대기.카메라자동시작금지.현재준비탭참가자self_eval_20261014/다시보기체크(새로고침하면기본self로돌아갈수있음).
TEST_READY.json에기준SHA·모델SHA·시작시각·기존3trialID동결.사용자완료통보후그이후실제모든시도를집계.
확인미저장/오인식/분석불가/취소도누락하지않고별도분모. Top1/3/5실제정답비율+DTW상대차이/관측부족을구분해같은발표웹내용교체.
권고12표현×3회이나실제실시수를보고.‘학습미포함개발참여자1인실사용’이며독립새사람일반화증명아님.
사용자자료학습/튜닝/척도/후보선택금지.향상미입증이며손모양정답판정도미검증.

## 고정 모델·데이터
runs/word100_tcn_v1/whole_context/best.pt SHA e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f.
AIHub-only100단어/899학습01~09·300개발11~13.92.33→96.67%개발결과,속도94.67%미채택.최종보류10/14~18미사용.
말뭉치주석329,401개조사/2025목록11ZIP601.64GB확인은보존.새다운로드·학습자동재개없음.
''','utf-8')
assert read(statepath)['current_stage']=='WORD100_TRAINING_RELATIVE_DTW_READY_FOR_USER_TEST'
print(json.dumps(dict(status=out['status'],numeric_checks=out['numeric_checks'],api_checks=out['api_checks'],server_pid=out['server']['pid'],model_sha256=baseline['model_sha256'])))
