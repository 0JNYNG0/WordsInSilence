from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib

U=Path(__file__).resolve().parent
ROOT=U.parents[3]
LAB=U.parents[1]
stamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
def read(p): return json.loads(p.read_text('utf-8-sig'))
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
scope=read(U/'SCOPE.json'); check=read(U/'DISPLAY_CHECK.json')
assert check['status']=='PASS'
web_check={
    'at_kst':stamp,'status':'PASS','browser':'IAB existing practice tab4 and report tab3',
    'training_metrics':3,'left_right_graphs':6,'fixed_axis':[0,50,100],
    'retrospective_example_scores':[62.213864960328856,82.58884425876133,80.49500354649042],
    'practice_horizontal_overflow':False,'report_slides':9,'report_slide_overflow':False,
    'camera_off':True,'connect_enabled':True,'participant':'self_eval_20261014','replay_checked':True,
    'ready_word':'수어','history_has_dates':True,
    'proof':['DTW_SCREEN.png','WEB_FULL.png','REPORT_SCREEN.png'],
    'note':'Initial screenshot clip used document coordinates and was blank; final proof cropped from actual full-page screenshot. Initial label locator matched region and select; corrected to unique combobox without camera activation.'}
write(U/'WEB_CHECK.json',web_check)
files=[LAB/'web/practice.html',LAB/'web/practice.js',LAB/'web/practice.css',LAB/'web/dtw-display.js',ROOT/'Docs/interim_report_20261014/web/index.html']
next_action='사용자가 현재12표현을가능하면3라운드촬영하고완료를알리면,고정기준이후모든시도·취소·오인식을구분집계하여동일발표웹에실제결과반영.사용자자료학습·튜닝·척도조정금지.'
completion={
    'at_kst':stamp,'status':'UI_VERIFIED_AWAITING_USER_TRIALS','implementation':'Codex local',
    'verification':'Codex separate native-unit arithmetic checks, original hashes and actual browser checks; no external reviewer or expert correctness validation',
    'display_version':scope['display_version'],'formula':scope['formula'],
    'model_sha256':scope['model_sha256'],'baseline_sha256':scope['baseline_sha256'],
    'changed_files':{str(p.relative_to(ROOT)):sha(p) for p in files},
    'checks':['DISPLAY_CHECK.json','WEB_CHECK.json'],'trial_protocol':'TEST_PROTOCOL.json',
    'new_capture_by_agent':False,'training_or_model_change':False,'user_trial_results_pending':True,
    'presentation':'Method updated on slide7; old example clearly retrospective. Actual new trial statistics pending.',
    'next':next_action}
write(U/'COMPLETION.json',completion)
(U/'RESULT_KO.txt').write_text('TCN의 기존순위·정답일치판정 보존. DTW세항목은동일단어학습9명과비교한기존거리/기준을0~100차이점수로표시.0가까울수록유사,50학습자간대표차이이며합격선아님.관측부족은수치보류.전체/좌우9그래프확인.27측정값독립식검산,원본17파일과기준/계산코드SHA보존.표시변경이며인식성능향상실험아님.발표7장설명갱신,새사용자결과미생성.실제촬영후전체시도집계예정.\n','utf-8')
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8')
state_entry='''### 2026-10-14 DTW 차이 점수 0–100 표시·사용자 시험 준비 완료
사용자 승인에 따라 기존 학습9명 비교는 그대로 두고 세 항목/좌우 그래프를 고정0–100 차이 점수로 수정했다. r=원래차이/학습자간대표차이, 점수=100r/(1+r). 낮을수록 유사하며50은학습자간대표차이,정답률·오답비율·합격선아님. 관측부족은보류.원단위는접힌측정근거에보존.촬영기록에날짜추가.
ops/word100_readable_dtw_20261014/COMPLETION.json UI_VERIFIED_AWAITING_USER_TRIALS.27값별도식검산/결측·단조성확인,원본17파일·모델·DTW기준·계산코드SHA불변.실제브라우저3전체+6손그래프/발표9장넘침없음확인. Codex구현·별도검산,전문가판정검증아님.발표7장표시방법수정,과거감사예시명시;새실사용통계는아직없음.
연습8765참가자self_eval_20261014/다시보기체크/첫표현수어/카메라꺼짐.새시험시작기준은기존TEST_READY를보존해누락방지.범위와집계는새TEST_PROTOCOL.json.촬영은사용자가진행하며완료통보후동일발표8766에실제수치반영.

'''
next_entry='''### 2026-10-14 다음 — 변경 화면에서 사용자 촬영 후 실제 결과 집계
표시수정·검증완료,사용자실시험대기.현재12표현을한번씩3라운드권고,계획36회를결과로쓰지않음.모든발급trial과결과/취소로그/확인기록을함께집계;화면최근30개/확인기록있는행만사용하지않음.학습미포함개발참여자1인실사용으로보고하며독립새사람일반화증명아님.Top1/3/5정답비율과DTW차이점수는별도통계,실패/미관측분모명시.모델·DTW기준동결,본인자료학습/튜닝/후보/척도조정금지.완료후동일Docs/interim_report_20261014/web발표내용을실제결과로갱신.준비만끝난상태를시험완료로말하지않음.

'''
for marker,entry in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n',state_entry),('## 9. 다음 단일 작업\n\n',next_entry)]:
    assert marker in text and entry.splitlines()[0] not in text
    text=text.replace(marker,marker+entry,1)
p.write_text(text,'utf-8')
p=LAB/'ops/STATE.json';state=read(p)
state['current_stage']='WORD100_READABLE_DTW_VERIFIED_USER_TRIALS_PENDING'
state['next_single_action']=next_action
state['word100_readable_dtw_20261014']=dict(completion,unit='ops/word100_readable_dtw_20261014')
write(p,state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {stamp}. AGENTS/CONTEXT 최신§3·§8·§9/WORKFLOW 우선.

## 현재 완료·사용자 요청
DTW의 1배/2배가 직관적이지 않다는 사용자 요청으로 세 항목 모두 0–100 차이 점수에 표시. 같은 단어의 실제 학습 촬영자9명과 비교하는 계산/기준은 유지, TCN 순위·목표단어 일치도 유지.
ops/word100_readable_dtw_20261014/COMPLETION.json UI_VERIFIED_AWAITING_USER_TRIALS. r=기존거리/학습자간대표거리, score=100r/(1+r).0은측정차이0,50학습자간대표차이,100가까울수록큼.정답률/틀린비율/합격선아님.전체+좌우의9그래프고정눈금,관측부족null.실제거리와기준은측정근거에유지.사용자자료로계수맞춤없음.
Codex구현·별도식27값검산/결측·단조성/원본17파일·모델·기준·DTW코드SHA보존.브라우저3항목/6손그래프와발표9장시각검증.인식률향상이나전문가타당성검증아님.

## 다음 한 작업
사용자가 직접 촬영 후 완료를 알리면 전체시도 집계하고 같은 발표 웹내용 갱신. TEST_PROTOCOL.json 참조.권고 현재12표현×3라운드36회,실제수를보고.실수/오인식도확인기록,확인버튼안누른시도도자동결과를포함.모든trial_ids/trial_closed/result를사용하고화면최근30개제한에의존하지않음.발급전카운트다운취소와실제trial취소구분.
기존ops/word100_multiref_20261014/TEST_READY.json(20:29KST)의기준과제외3trialID유지.새표시버전training_difference_100_v1.현재새사용자시험통계없음.집계후Top1/3/5분모와DTW차이분포/미관측을별도로보고.학습미포함개발참여자1인실사용이며독립새사람일반화증명아님.학습/튜닝/척도/후보선택에본인자료금지.

## 화면·서버
연습 http://127.0.0.1:8765/practice.html 일반서버PID65612(재개시확인).기존탭4에참가자self_eval_20261014,다시보기저장체크,수어선택,카메라꺼짐.재시작/카메라자동활성화하지않음.새로고침시참가자/체크가초기화될수있음.
발표 http://127.0.0.1:8766/#s7 정적PID58776,기존탭3.7장방법설명과대본을0–100점수로갱신.기존감사2차의62.2/82.6/80.5점은과거촬영예시,새시험결과아님.개인시험목표슬라이드는추가하지않음.실제시험후별도실적반영.

## 보존된 기준
모델runs/word100_tcn_v1/whole_context/best.pt SHA {scope['model_sha256']}.
AIHub100단어899학습01~09/300개발11~13,개발96.67%.최종보류10/14~18미사용.학습·DTW基準変更なし.
DTW BASELINE SHA {scope['baseline_sha256']}.12표현108학습영상/864사람간비교.학습자는본인제외8명,사용자는9참고;각손유효참고최소5.기존3촬영새분석은sidecar,원래result/NPZ보존.
새학습/다운로드/카메라/운영모델교체/Git변경없음.말뭉치주석329401개와2025목록11ZIP601.64GB조사이력보존.
'''.replace('학습·DTW基準変更なし.','학습·DTW 기준 변경 없음.'),'utf-8')
assert read(LAB/'ops/STATE.json')['current_stage']=='WORD100_READABLE_DTW_VERIFIED_USER_TRIALS_PENDING'
print(json.dumps({'status':completion['status'],'next':'User records trials; aggregate and update existing presentation after completion'},ensure_ascii=False))
