import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from html import escape

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPO=ROOT.parents[1]
r=json.loads((HERE/'RESULT.json').read_text('utf-8'))
now=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
report='''후보 C 실제 사용자 촬영 비교 — 2026-10-17

1. 비교 방법
사용자 1명이 12개 단어를 처음 한 번씩 수행한 12회를 본 집계로 사용했다.
확인 기록을 저장하지 않은 '마음에 들다' 첫 시도도 자동 저장되어 포함했다.
마지막 재시도 1회는 별도로 표시했다. 성공한 재시도로 첫 실패를 대체하지 않았다.
각 촬영의 같은 프레임 번호·시각을 사용한 기존 M0와 C의 저장 결과를 비교했다.
12개 연습 단어 중 고르는 시험이 아니라, 학습된 100개 후보 전체에서 정답 순위를 확인한 시험이다.
비교 대상은 모델과 관절 처리까지 포함한 전체 처리 방식이다. 입력 보완 하나의 효과만 분리한 실험은 아니다.

2. 처음 12회 결과
정답 1위: 기존 8/12 (66.67%) → C 11/12 (91.67%), +25.00%p, 3회 증가.
정답 상위 5위: 기존 10/12 (83.33%) → C 11/12 (91.67%), +8.33%p, 1회 증가.
기존 1위였던 8개는 모두 C에서도 1위였다.
개선: 멀다 29위→1위 / 공항 11위→1위 / 별 3위→1위.
회귀: 마음에 들다 첫 시도 2위→8위. 기존과 C 모두 첫 시도 정답 1위에는 실패했다.
추가 재시도: 마음에 들다는 기존 2위, C 2위. C의 1위는 강진군수어통역센터였다.
참고로 13회를 모두 합치면 기존 8/13 (61.54%) → C 11/13 (84.62%). 한 단어를 두 번 수행한 불균형 집계이므로 주 결과로 쓰지 않는다.

3. 무엇이 개선되었나
개선된 세 촬영에서 왼손 관측률은 동작 구간 기준 멀다 0%, 공항 3.23%, 별 0%였다.
오른손 관측률은 각각 97.62%, 98.39%, 97.06%였다. 여기서 관측은 해당 손의 사용 가능한 관절이 하나 이상 있는 프레임을 뜻한다.
반대손이 거의 보이지 않아도 C는 세 단어를 모두 1위로 골랐다. 화면 밖의 쉬는 손을 허용하려던 보완 방향을 지지하는 관찰이다.
손 누락을 완전히 없앤 것이 아니라, 이런 입력에서도 분류가 가능해진 사례다.
다만 관절 배정·입력 처리·학습 모델이 함께 달라졌으므로 어느 한 요소가 개선을 전부 만들었다고 단정할 수 없다.

4. 남은 문제
마음에 들다는 두 시도 모두 아직 1위로 인식하지 못했다.
첫 시도 C는 '수어'에 94.88% 모델 점수를 주었다. 이것은 높은 점수로도 오답을 낼 수 있음을 보여주며, 점수를 수행 정확도나 정답 확률로 해석하면 안 된다.
첫 시도 동작 구간 39프레임, 재시도 48프레임 모두 적어도 한 손이 관측됐다. 첫 시도 오른손은 100% 관측됐으며, 재시도는 양손 동시 관측이 오히려 66.67%→35.42%로 줄었는데 정답 순위는 8위→2위로 올랐다.
따라서 이 실패를 '양손이 모두 사라져서 생긴 오류'로 설명할 수는 없다. 관측이 있다고 손가락 모양·좌우 배정이 정확하다는 보장도 없다.
저장된 RGB 표본에서 두 시도 모두 손을 얼굴/가슴 쪽으로 올린 뒤 양손이 만나는 흐름이 보였다. 참고 영상과 프레임별 위치·손 모양·속도는 동일하지 않다. 수어 수행의 정오나 오답의 단일 원인을 이 관찰만으로 확정하지 않는다.
활성 구간 길이는 2.22초→2.90초로 달랐다. 원인 후보를 설명하는 관찰이며, 사용자 촬영으로 보정값이나 학습 조건을 맞추지 않는다.

DTW는 C와 별개인 기존 관측/학습 참고은행을 그대로 사용한다.
멀다·공항·별 3회는 PARTIAL이며, 손가락 굽힘·손 위치·이동 궤적의 종합값이 모두 INSUFFICIENT였다. 나머지 10회(재시도 포함)는 계산상 OK였다.
OK는 계산할 근거가 있다는 뜻이며 올바른 수어 수행 판정이 아니다. 한손만 쓰는 상황에서 필요한 손 위주로 비교하는 DTW 설계는 아직 남아 있다.
마음에 들다 DTW 손가락 굽힘 차이는 32.95도→31.13도, 손 위치 차이는 어깨너비 대비65.71%→64.88%, 이동 궤적 차이는57.49%→52.11%였다. 이는 기존 DTW의 저장된 2D 차이 지표이며 C의 학습 가중치 설명이나 '틀린 비율'이 아니다. 여기서 임계값을 새로 정하지 않았다.

5. 실행 상태와 검증
13회 모두 두 모델 READY. 프레임 번호·시간 배열의 완전 일치와 각 관측 파일 해시를 확인했다.
저장된 100개 출력 점수로 순위·표시 점수를 재계산했고, 별도 표준 라이브러리 계산으로 다시 검산했다. 같은 Codex 담당자의 독립 계산이며 별도 전문가/에이전트 검토는 아니다.
진단 이미지 579개의 저장 해시도 모두 일치했다. 이미지는 최대10fps 표본이므로 영상의 모든 순간을 시각 검증한 것은 아니다.
관측 속도는 약17.0~20.4프레임/초, 최대 프레임 간격113.97ms였다. 종료 요청부터 서버 결과 조립까지54.95~115.31ms였으며, DTW 상세 계산/브라우저 표시까지 포함한 지연은 아니다.
원본 촬영·모델 해시는 유지됐다. 새 추론·학습·튜닝·임계값 변경·후보 선택·웹 수정·카메라 접근은 수행하지 않았다.

6. 해석과 다음 단계
이번 사용자·12단어 첫 시도에서는 C가 기존보다 분명히 더 많이 맞혔다. 하지만 1명×단어당1회이므로 일반 사용자 전체의 정확도91.67%를 입증한 것은 아니다.
사용자는 학습에 포함되지 않았지만, 과거 시험과 환경이 보완 방향 논의에 이미 반영됐다. 완전히 처음 만나는 독립 사용자에 대한 맹검 평가와 구분한다.
현재 결과를 보존하고 사용자와 확인한 뒤, 공개 학습/개발 자료에서 혼동 및 한손 DTW 근거 부족을 별도 작업으로 점검할 수 있다. 이번 촬영은 평가 근거로만 남기며 학습·튜닝·선택에 쓰지 않는다.
'''
(HERE/'RESULT_KO.txt').write_text(report,'utf-8')
rows=''
for x in r['rows']:
    a,b=x['M0']['rank'],x['C']['rank']; cls='gain' if b<a else 'loss' if b>a else ''
    mark='재시도 · 별도' if x['group']=='retry' else str(x['number'])
    rows+=f'<tr class="{cls}"><td>{mark}</td><td>{escape(x["label"])}</td><td>{a}위</td><td><strong>{b}위</strong></td><td>{"확인 저장" if x["confirmation_saved"] else "자동 저장 · 확인 미저장"}</td></tr>'
html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C 후보 실사용 비교</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f7;color:#17304a;font-family:'Malgun Gothic',sans-serif;line-height:1.7}main{max-width:1050px;margin:40px auto;padding:0 24px}h1{font-size:32px;line-height:1.35}h2{font-size:22px;margin-top:0}.eyebrow{font-size:14px;color:#4b627b;font-weight:bold}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:white;border:1px solid #dbe3ed;border-radius:18px;padding:26px;margin:20px 0}.grid .card{margin:0}b.big{font-size:38px;display:block;line-height:1.3;color:#0f766e}.bar{height:24px;border-radius:6px;background:#e7edf4;margin:8px 0 16px}.bar i{display:block;height:100%;border-radius:6px;background:#8190a4}.bar.green i{background:#149d88}table{width:100%;border-collapse:collapse;text-align:left}td,th{padding:10px;border-bottom:1px solid #e4eaf1}.gain strong{color:#087b68}.loss strong{color:#ba4a12}.note{border-left:4px solid #cf8236;padding:12px 18px;background:#fff6e8}a{color:#036980}details{background:#fff;padding:20px;border-radius:14px;margin-bottom:30px}pre{white-space:pre-wrap;font:14px/1.8 'Malgun Gothic',sans-serif}p{margin:10px 0}footer{font-size:13px;color:#53677d;margin-bottom:30px}@media(max-width:700px){.grid{display:block}.grid .card{margin:14px 0}h1{font-size:26px}td,th{padding:7px;font-size:13px}}@media print{body{background:white}main{margin:0}details{display:none}.card{break-inside:avoid}}
</style><main><div class="eyebrow">WordsInSilence · 2026.10.17 · 동일 촬영 비교</div><h1>처음 12회에서<br>기존 8회 → C 후보 11회 정답</h1><p>사용자 1명 · 12개 단어 각 1회 · 100개 후보 중 순위 확인<br>‘마음에 들다’ 재시도 1회는 본 집계와 분리했습니다.</p>
<div class="grid"><section class="card"><h2>정답이 1위인 비율</h2><b class="big">+25.00%p</b><p>기존 8/12 · 66.67%</p><div class="bar"><i style="width:66.6667%"></i></div><p>C 후보 11/12 · 91.67%</p><div class="bar green"><i style="width:91.6667%"></i></div></section><section class="card"><h2>정답이 상위 5위에 든 비율</h2><b class="big">+8.33%p</b><p>기존 10/12 · 83.33%</p><div class="bar"><i style="width:83.3333%"></i></div><p>C 후보 11/12 · 91.67%</p><div class="bar green"><i style="width:91.6667%"></i></div></section></div>
<section class="card"><h2>같은 촬영에서 정답 순위가 어떻게 달라졌나</h2><table><thead><tr><th>순서</th><th>수행 단어</th><th>기존 M0</th><th>C 후보</th><th>기록</th></tr></thead><tbody>__ROWS__</tbody></table><p>확인 버튼을 누르지 않은 첫 실패도 포함했습니다. 재시도 결과로 첫 실패를 바꾸지 않았습니다.</p></section>
<div class="grid"><section class="card"><h2>개선 방향을 지지하는 근거</h2><p><strong>멀다 · 공항 · 별</strong>은 반대손이 거의 또는 전혀 관측되지 않았지만 C에서 모두 1위가 됐습니다.</p><p>기존에서 1위였던 나머지 8개도 모두 1위를 유지했습니다.</p><p>쉬는 손이 화면 밖이어도 분류할 수 있게 하려던 방향과 맞는 결과입니다. 전체 처리 방식의 효과이며 한 요소만의 효과를 분리한 시험은 아닙니다.</p></section><section class="card"><h2>남은 두 가지 문제</h2><p><strong>① 마음에 들다:</strong> 첫 시도 8위, 재시도 2위. 아직 1위에 들지 못했습니다. 양손 전체 미검출로만 설명할 수 없습니다.</p><p><strong>② DTW:</strong> 멀다·공항·별은 한손 근거 부족으로 종합 차이 수치가 계산 불충분입니다. 분류 개선과 별도로 보완이 필요합니다.</p></section></div>
<section class="card"><h2>결과를 이렇게 설명하면 정확합니다</h2><div class="note">“학습에 사용하지 않은 사용자 1명이 12개 단어를 한 번씩 수행했을 때, 동일 촬영에서 기존 66.67% 대비 C 후보는 91.67%를 1위로 인식했다. 한손이 주로 보이는 세 촬영에서 개선을 확인했지만, 일부 단어 혼동과 한손 DTW 분석 한계가 남았다.”</div><p>1명·12회의 탐색적 시험입니다. 모든 새 사용자의 성능이나 수어 수행의 올바름을 보장하는 수치는 아닙니다. 과거 사용자 시험이 보완 방향 논의에 반영되었으므로 완전한 맹검 평가와도 구분합니다.</p><p>사용자 촬영은 평가 전용으로 보존했습니다. 학습·튜닝·임계값 수정·모델 선택을 하지 않았습니다.</p></section><details><summary>검증 방법과 관측 한계 자세히 보기</summary><pre>__TEXT__</pre></details><footer>13쌍 관측 해시·프레임 번호·시간 일치 확인 / 100개 저장 점수 재계산 + 별도 방식 검산 / 신규 학습·추론 0<br><a href="RESULT_KO.txt">상세 설명</a> · <a href="RESULT.json">집계 근거</a> · <a href="INDEPENDENT_RECOMPUTE.json">별도 계산 검산</a></footer></main></html>'''
(HERE/'RESULT_KO.html').write_text(html.replace('__ROWS__',rows).replace('__TEXT__',escape(report)),'utf-8')

statep=ROOT/'ops/STATE.json'; state=json.loads(statep.read_text('utf-8'))
state['updated_date_kst']=now
state['current_stage']='WORD100_C_USER_PAIRED_EVALUATION_COMPLETE_WAITING_REVIEW'
state['word100_c_user_paired_20261017']={
 'status':'COMPLETE_SAVED_RESULTS_VERIFIED','updated_kst':now,'unit':'ops/word100_c_user_paired_20261017',
 'session':'ses_b7aef2518a30','participants':1,'initial_trials':12,'separate_retries':1,
 'initial_M0_top1':8,'initial_C_top1':11,'initial_M0_top5':10,'initial_C_top5':11,'top1_delta_pp':25.0,
 'top1_gains':['멀다','공항','별'],'rank_regression':'마음에 들다 first M0rank2 Crank8; retry both rank2',
 'DTW_partial_trials':3,'DTW_partial_words':['멀다','공항','별'],
 'same_frame_ids_timestamps_and_hashes_verified':13,'saved_capture_hashes_verified':579,
 'user_data_use':'evaluation_only_never_training_tuning_threshold_or_model_selection',
 'new_training':0,'new_inference':0,'web_changes':0,'camera_access':0,
 'verification':'rank/softmax and observations checked; separate stdlib count-greater/hash recomputation; same Codex operator, no independent agent review',
 'limitations':['one participant,12 first trials','not untouched blind evaluation: prior environment informed method design','full-pipeline comparison,not individual causal ablation','correct word does not validate correct sign execution'],
 'report':str(HERE/'RESULT_KO.html'),'result':str(HERE/'RESULT.json'),
 'next_single_action':'Review paired results with user. No automatic retraining, tuning, DTW replacement or default model change. Preserve user trials for evaluation only.'}
state['word100_candidate_c_web_20261017']['paired_user_evaluation']='ops/word100_c_user_paired_20261017/RESULT.json'
state['word100_candidate_c_web_20261017']['status']='READY_OPT_IN_FIRST_USER_PAIRED_EVALUATION_COMPLETE'
state['word100_candidate_c_web_20261017']['webcam_accuracy_verified']='limited one-user12-first-take paired evaluation; not general population accuracy'
state['word100_candidate_c_web_20261017']['next_single_action']=state['word100_c_user_paired_20261017']['next_single_action']
statep.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n','utf-8')

ctx=REPO/'WordsInSilence_PROJECT_CONTEXT.md'; text=ctx.read_text('utf-8')
entry8='''### 2026-10-17 후보 C 실사용 동일 촬영 12회 비교 완료
- 사용자1명/12단어 첫1회 기준 M0 Top1 8/12 66.67%→C 11/12 91.67%(+25.00%p); Top5 10/12 83.33%→11/12 91.67%(+8.33%p). 멀다29→1/공항11→1/별3→1. 기존Top1 8개 유지. 마음에들다 첫시도 M0 2→C8위 회귀; 추가재시도 양쪽2위, 본집계와 분리. 확인미저장 첫실패도 자동기록으로 포함.
- 세 개선촬영의 C 왼손 관측률0%/3.23%/0%, 오른손97.62%/98.39%/97.06%; 쉬는손 미관측 대응 방향을 지지하지만 전체처리 짝비교이며 요소별 인과효과 아님. DTW는 이3회 PARTIAL/3항목 INSUFFICIENT. C분류와 기존DTW의 남은 문제 분리.
- 13쌍 저장관측 hash/프레임번호/시간 일치,100logits순위·점수검산,별도stdlib재계산PASS/진단579이미지hash일치. 동일Codex담당 검산, 에이전트독립검토아님. 원본/모델유지; 새추론/학습/튜닝/카메라/웹변경0. 사용자1명 탐색평가이며 과거 환경이 설계논의에 반영된 점을 밝혀 맹검/일반사용자정확도와 구분.
- 근거 `research/front_keyphase_v1/ops/word100_c_user_paired_20261017/RESULT_KO.html`, `RESULT.json`, `INDEPENDENT_RECOMPUTE.json`. session ses_b7aef2518a30 CLOSED_BY_CLIENT. 서버 재시작/카메라조작 없음.

'''
head='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'
assert head in text
text=text.replace(head,head+'\n'+entry8,1)
lines=text.splitlines(keepends=True); idx=next(i for i,l in enumerate(lines) if l.startswith('## 9.'))
entry9='''
### 2026-10-17 동일 촬영 C/M0 비교 결과 사용자 검토 대기
완료된 첫12회 비교(8→11정답)와 마음에들다 회귀·재시도, 한손DTW근거부족을 사용자와 확인한다. 기존M0기본/명시적C선택과 모델핀 유지. 촬영13회는 평가전용이며 학습·튜닝·후보선택·임계값수정 금지. 이번 집계만 근거로 추가학습/운영교체/DTW수정 자동시작하지 않는다. 후속 보완은 공개TRAIN/DEV로 조건을 고정한 별도 작업으로 정한다. 비교 완료, 기존st-gcn자동화 삭제상태 유지.

'''
lines.insert(idx+1,entry9); ctx.write_text(''.join(lines),'utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {now}

## 현재 목표와 마지막 완료
사용자 요청: 직접 촬영한 처음12회와 마음에들다 추가1회를 같은 촬영의 C/기존M0로 비교. 학습·튜닝 금지.
분석 완료: ses_b7aef2518a30의13회 자동저장 확인. 첫12회 M0 Top1 8/12 66.67%→C11/12 91.67%(+25.00%p),Top5 10→11(+8.33%p). 확인미저장 첫실패도 포함.
멀다29→1,공항11→1,별3→1 개선. 기존1위8개 유지. 마음에들다 첫2→8위 회귀, 마지막재시도 양쪽2위(별도집계). 실패를재시도로대체하지않음.
改善3촬영의반대손거의미관측 상황은 보완방향을지지하나 1명12회 탐색평가. 과거사용자환경이 설계논의에반영돼 독립맹검/전체새사용자성능으로확대금지. 100후보전체순위.

## 검증과 남은 문제
13쌍관측 hash/프레임번호/시간완전일치, 저장100logits순위·softmax검산. 별도stdlib count-greater계산PASS, 진단JPEG579개hash일치. 동일Codex담당 독립계산이며 별도에이전트/전문가검토아님.
마음에들다 첫시도 C수어94.88%오답: 점수=수행정확도아님. 활동구간에최소1손항상관측,재시도양손관측률더낮아도순위개선; 단순전체손누락으로원인단정불가.
DTW멀다/공항/별은PARTIAL,3항목종합INSUFFICIENT. 나머지10(재시도포함)계산OK는수행정답판정아님. DTW는기존관측/고정9TRAIN참고은행유지,C가중치와비교아님.
관측17.0~20.4fps,최대간격113.97ms;종료→서버조립54.95~115.31ms(DTW상세/화면전달지연제외).

## 다음 한 작업
완료 결과를 사용자와 확인하고 다음 범위를 결정한다. 추가 학습·튜닝·기본교체·DTW수정은 이번 분석에서 하지 않았다. 사용자촬영을 학습/튜닝/모델선택/임계값변경에사용금지. 원본13회·모델·웹유지.

## 웹과 모델
C http://127.0.0.1:8765/practice.html?model=c / M0 ?model=m0 / 기본 M0 유지.
Cseed20261013 epoch77 SHA283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab.
M0 SHAe39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f.
원본.npz=M0/DTW, .candidate_c.npz=C. 전체pipeline짝비교이며가중치만의비교아님. C fresh-per-take joint_v3+실제화면밖마스크/내부누락보존,수동종료/최대10초.
마지막시작서버PID34640/8765 일반실행, 로그ops/word100_candidate_c_web_20261017/server.v2.*. 이번분석은서버재시작/장치접근없음. 촬영session파일 CLOSED_BY_CLIENT/camera false 확인, 실제서버상태는필요시별도확인.

## 근거
ops/word100_c_user_paired_20261017/RESULT_KO.html,RESULT_KO.txt,RESULT.json,INDEPENDENT_RECOMPUTE.json,EVALUATION_SCOPE.json.
기존후보연구 ops/word100_body_boundary_20261017/FINAL_RESULT.json, 웹연결 ops/word100_candidate_c_web_20261017/RESULT.json 보존.
TRAIN899/DEV300/보류14~18분리 유지. st-gcn후속자동화삭제상태유지. 새학습/추론/웹수정/다운로드/외부전송/Git0.
'''
(REPO/'NEW_CHAT_HANDOFF.md').write_text(handoff.replace('改善3','개선3'),'utf-8')
print(json.dumps({'report':str(HERE/'RESULT_KO.html'),'state':'synced','updated':now},ensure_ascii=False))
