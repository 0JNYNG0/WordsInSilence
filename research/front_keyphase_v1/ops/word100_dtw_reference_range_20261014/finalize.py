from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
from PIL import Image
U=Path(__file__).resolve().parent;LAB=U.parents[1];ROOT=U.parents[3]
def read(p): return json.loads(p.read_text('utf-8-sig'))
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
checks=read(U/'VERIFICATION.json');web=read(U/'WEB_CHECK.json')
assert checks['status']=='PASS' and len(web['metrics'])==3 and web['cameraOff'] and not web['overflow']
Image.open(U/'WEB_FULL.png').crop((632,1015,1187,1470)).save(U/'REFERENCE_SCREEN.png')
changed=[LAB/'web/dtw-display.js',LAB/'web/practice.js',LAB/'web/practice.html',LAB/'web/practice.css',ROOT/'Docs/interim_report_20261014/web/index.html']
next_action='사용자가 고정 모델·학습 참고 범위로 촬영한 뒤, 실제 모든 시도의 TCN 정확도와 DTW 참고 범위 초과율을 구분 집계해 기존 발표 웹에 반영한다. 참고 범위 초과를 수어 오답률로 부르지 않는다.'
completion=dict(at_kst=stamp,status='REFERENCE_ENVELOPE_IMPLEMENTED_CORRECTNESS_NOT_VALIDATED',implementation='Codex local',verification='Codex independent native-value recomputation and actual browser verification; no external expert correctness validation',rule='Maximum of9 training leave-one-signer-out distances for each word/metric/hand',reference_distributions_checked=108,development_coverage={'handshape':[30,36],'location':[35,36],'movement':[33,36]},interpretation='Empirical reference membership only. No verified error examples, therefore no correctness sensitivity/specificity or pass threshold claim.',model_or_baseline_change=False,user_tuning=False,camera_started=False,changed_files={str(p.relative_to(ROOT)):sha(p) for p in changed},evidence=['SCOPE.json','DEVELOPMENT_COVERAGE.json','VERIFICATION.json','WEB_CHECK.json','REFERENCE_SCREEN.png'],next=next_action)
write(U/'COMPLETION.json',completion)
(U/'RESULT_KO.txt').write_text('학습 촬영자 9명을 본인 제외 비교한 차이의 최댓값을 단어·항목·손별 참고 상한으로 표시했다. 참고 범위 이내 / 범위 초과·확인 필요 / 판단 보류를 전체와 좌우에 표시한다. 검증용 같은 단어36영상의 범위 이내는 굽힘30, 위치35, 궤적33개다. 이 결과는 정답 판정 정확도가 아니며, 정상 수행/오류 수행의 전문가 라벨이 없어 수어 합격선으로 사용할 수 없다. 모델·기준자료·사용자 원본 불변, 사용자 자료로 상한 조정 없음. 실제 촬영 후 집계 대기.\n','utf-8')
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8')
entry='''### 2026-10-14 DTW 항목별 학습 참고 범위 표시·별도 촬영자 점검 완료
사용자가 숫자만이 아니라 잘 따라한 범주/수정 확인 범주를 요청했다. 정상·오류 합격선을 임의로 만들지 않고, 단어·항목·손별 학습9명의 본인제외 비교 최댓값을 참고 상한으로 고정했다. 화면에 참고 범위 이내 / 범위 초과·확인 필요 / 판단 보류와 실제 상한 점선을 표시. 범위 이내는 수어 전체 정답, 초과는 오류 확정이 아니다.
사전 고정 규칙으로 기존 개발3명36영상 저장 비교를 검산한 결과, 전체 기준 범위 이내는 굽힘30/36(83.33%), 위치35/36(97.22%), 궤적33/36(91.67%). 동일 단어 영상도 초과하므로 정답 판정 민감도/특이도 또는 정상 수행 허용선으로 주장하지 않는다. 개발/사용자 결과를 보고 상한을 수정하지 않았다.108개 기준분포·경계/결측 검산,원본17파일·모델·DTW 기준/계산코드SHA 보존. 실제 웹 전체/좌우 범주 확인, 발표7장 방법 설명 일치. 근거 ops/word100_dtw_reference_range_20261014/COMPLETION.json.
기존0–100 표시와 원래 거리 유지, 카메라/새학습 없음. 사용자 실사용 집계는 아직 대기이며 기존 TEST_READY 경계와 TEST_PROTOCOL을 유지한다.

'''
follow='''### 2026-10-14 다음 — 참고 범주가 표시되는 화면에서 실사용 후 집계
연습8765 카메라꺼짐, 참가자self_eval_20261014/다시보기체크/수어선택. 사용자 촬영 완료 후 실제 전체 시도를 집계하고 동일 발표8766에 반영한다. TCN Top1/3/5 정답 비율, DTW 참고 범위 이내·초과·보류 건수는 별도이며 DTW 초과를 수어 오답으로 세지 않는다. 사용자 자료 학습/척도/상한/후보 조정 금지. 실제 정상·오류 수행을 구별하는 교정 기준의 검증은 전문가 라벨과 별도 평가가 필요한 미완료 과제다.

'''
for marker,add in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n',entry),('## 9. 다음 단일 작업\n\n',follow)]:
    assert marker in text and add.splitlines()[0] not in text
    text=text.replace(marker,marker+add,1)
p.write_text(text,'utf-8')
p=LAB/'ops/STATE.json';state=read(p);state['current_stage']='WORD100_DTW_REFERENCE_ENVELOPE_READY_CORRECTNESS_UNVALIDATED';state['next_single_action']=next_action;state['word100_dtw_reference_range_20261014']=dict(completion,unit='ops/word100_dtw_reference_range_20261014');write(p,state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {stamp}. AGENTS / CONTEXT 최신 §3·§8·§9 / WORKFLOW 우선.

## 현재 요구와 완료
사용자는 DTW 숫자로 ‘잘 따라한 범주인지, 수정 확인이 필요한지’를 알고 싶다. 숫자 변환만으로 부족하다는 지적을 반영해 학습 자료의 참고 범주를 추가했다.
ops/word100_dtw_reference_range_20261014/COMPLETION.json: REFERENCE_ENVELOPE_IMPLEMENTED_CORRECTNESS_NOT_VALIDATED.
단어·항목·손마다 학습9명의 본인 제외 비교 차이 최댓값을 참고 상한으로 사용. 전체/좌우에 ‘참고 범위 이내 / 범위 초과·확인 필요 / 판단 보류’를 표시한다. 원거리, DTW 계산, 모델, 기존0–100 변환(100r/(1+r))은 그대로다. 점선은 이제 각 항목의 실제 참고 상한이고50에 고정되지 않는다.
50은 여전히 학습자 간 대표 차이 수준. 범위 이내는 측정된 특징 차이가 학습 상한 이하라는 뜻이며 수어 정답 확정이 아니다. 초과는 동작 또는 관측 조건을 확인할 후보이며 오류 확정이 아니다. 실제 정상·오류 라벨 검증이 남아 있다.

## 확인한 근거와 한계
규칙을 SCOPE에 먼저 고정하고 기존 개발3명36개 저장 결과로 검산했다. 범위 이내: 굽힘30/36, 위치35/36, 궤적33/36. 같은 단어의 개발 영상도 초과하므로 올바른 수행의 허용 범위나 오류검출 정확도라고 주장하지 않는다. 이 결과로 상한/계수를 바꾸지 않았다.
108개 기준분포의 원값 재검산과 경계·결측·표본부족 검사 PASS. 원본17파일·모델·BASELINE·DTW계산코드 SHA 보존. Codex 구현 및 별도 계산 검증; 전문가 독립 타당성 검증은 아님. 실제 브라우저 전체3/좌우6 범주·점선·넘침 없음 확인. 발표7장도 방법 설명 일치.

## 다음 한 작업
사용자 직접 실사용 촬영 완료 통보 후 모든 실제 시도를 집계하고 같은 발표 웹내용을 갱신한다. 이전 ops/word100_readable_dtw_20261014/TEST_PROTOCOL.json 및 word100_multiref_20261014/TEST_READY.json(20:29KST 경계/과거3trial제외) 유지. 범주 표시버전 training_observed_max9_v1 추가.
12표현 한 번씩3라운드36회 권고, 실제 횟수로 보고. 확인 버튼을 누르지 않은 시도·오인식·취소·관측부족도 누락 금지. session trial_ids, timing trial_closed, result, self-report를 함께 확인하며 UI최근30개에 의존하지 않는다.
TCN Top1/3/5 정답률과 DTW 참고 범위 초과율은 서로 다른 지표다. DTW 초과를 수어 오답으로 세지 않는다. 학습 미포함 개발참여자1인 실사용이며 독립 새사람 일반화 증명 아님. 사용자 자료 학습/튜닝/상한/척도/후보선택 금지.

## 서버와 화면
연습 http://127.0.0.1:8765/practice.html 일반환경 PID65612(이번 턴 실프로세스 확인). 기존탭4 참가자self_eval_20261014, 다시보기체크, 수어선택, 카메라꺼짐. 새로고침 시 설정이 초기화될 수 있다. 카메라는 사용자 연결 버튼으로만 시작.
발표 http://127.0.0.1:8766/#s7 정적PID58776, 기존탭3. DTW 기존 감사촬영의62.2/82.6/80.5점은 새 사용자시험 결과가 아닌 과거 구현 예시다. 이번 상한60.9/56.1/59.2점도 학습 자료만으로 계산. 예시의 왼손굽힘은 참고 이내, 오른손은 초과. 카메라 새촬영 없음.

## 고정 기준
모델 runs/word100_tcn_v1/whole_context/best.pt SHA e39d8ac46f5a5f8442c48a141498dc810ccbfad242e3f104e1b8633988515c2f.
AIHub100단어899학습01–09 / 300개발11–13, 개발96.67%. 최종보류10/14–18 미사용.
BASELINE SHA 0af5e66be355ab0a781fc0b7bfcbcee551fd92f9f1cf3bf2097f4c87d34440cf. 12표현108학습영상/864사람간비교. 학습자는 본인제외8명, 사용자/개발은9참고. 각손 최소5개 유효 참고, 범주 상한은9명 기준값이 모두 있어야 표시한다.
학습/다운로드/운영모델교체/Git변경 없음. 말뭉치 조사 등 과거 결과 보존.
''','utf-8')
print(json.dumps({'status':completion['status'],'next':'Actual user recordings and aggregate; correctness labels still unvalidated'},ensure_ascii=False))
