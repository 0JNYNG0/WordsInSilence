"""Plan-only review using saved outputs; no ML imports, inference, or training."""
from pathlib import Path
import json,hashlib,shutil
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];P=O/'word100_short_motion_completion_20261111'
def read(p):return json.loads(p.read_text('utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def overlap(a,b):return max(0,min(a[1],b[1])-max(a[0],b[0])+1)/(max(a[1],b[1])-min(a[0],b[0])+1)
ds=[r for r in read(P/'DIRECTION_RESULTS.json') if r['seed']==42 and r['group']=='reserved_id13' and r['kind']=='normal'];audit=[]
for r in ds:
 z=read(P/'fresh_baseline'/f"{r['clip_id']}_normal.json");t=r['truth'];same=lambda es:[e for e in es if e['axis']==t['axis'] and e['moving']==t['moving']]
 best=lambda es:max([overlap([e['a'],e['b']],t['span']) for e in same(es)]+[0.])
 audit.append(dict(clip_id=r['clip_id'],truth=t,manual_correct=r['manual']['after']['eligible'] and r['manual']['after']['prediction']==t['direction'],auto_matched=r['automatic']['combined']['matched'],best_retained_overlap=best(z['baseline']['phase_events']+z['baseline']['phase_partials']),best_selected_overlap=best(z['selected'])))
assert len(audit)==9 and sum(a['manual_correct'] for a in audit)==8 and sum(a['auto_matched'] for a in audit)==4
assert sum(a['manual_correct'] and not a['auto_matched'] for a in audit)==4
write(U/'EVIDENCE_AUDIT.json',dict(rows=audit,manual_inputs=['time interval','moving hand','relationship axis','anchor hand'],manual_score_is_direction_only=True,probability_is_not_calibrated_feedback_accuracy=True,coarse_wrong_interval_is_not_zero_detection=True,all_preselection_logits_not_saved_for_fresh6=True,review='Saved result arithmetic and code review by same Codex; no independent expert labels.'))
sources=[dict(title='ASRF / Alleviating Over-segmentation Errors by Detecting Action Boundaries',url='https://arxiv.org/abs/2007.06866',supports='Separate action labeling and boundary prediction, then refine segmentation; does not establish KSL pedagogical feedback performance.'),dict(title='Temporal Convolutional Networks for Action Segmentation and Detection',url='https://arxiv.org/abs/1611.05267',supports='Temporal convolutions can learn fine-grained temporal segmentation; no evidence that our current model already solves it.'),dict(title='ActionFormer: Localizing Moments of Actions with Transformers',url='https://arxiv.org/abs/2202.07925',supports='Multiscale temporal representation and boundary estimation; architectural reference only, no recommendation to download or substitute model.')]
write(U/'SOURCES.json',sources)
plan='''현재 능력과 자동 동작 탐색 — 계획 재검토 결과

검토 범위
코드의 실제 입력/출력, 저장된 새9항목 결과와 자동후보, 실제 피드백 문장, 이전 계획, 관련 원논문의 설계 근거를 확인했다. 이번에는 학습·추론·관절추출·웹수정·카메라·다운로드를 실행하지 않았다. 사용자 요청에 따라 이 계획을 먼저 보고한다.

현재 할 수 있는 일
이번 이동 판정부는 구간 시작/끝뿐 아니라 어느 손의 어떤 관계(높이, 다른 손과의 거리, 전완을 따르는 위치 등)를 볼지도 지정받는다. 상대좌표 흐름을 보고 증가/유지/감소3종과 내부 확신값을 낸다. 관측이 부족하면 별도로 보류한다. 손가락 모양과 교육적 정오까지 이3종 출력 하나로 판단하는 것은 아니다.
최근6영상9이동은 위 조건의 방향 판단8/9다. 전체 영상의 손모양·순서·누락·교정 문장을 모두 맞힌8/9가 아니다. 단일 작은 표본·과거 공통개발 노출·같은Codex 주석이므로 실사용88.9%정확도라고 표현하지 않는다.
전체 시스템에는 별도 손모양/구간/가이드 연결 부품이 있고, 그 판정에 맞춰 준비된 피드백 문장을 선택한다. 신경망이 자유롭게 설명글을 쓰는 언어모델은 아니다. 공개 관절변형 시험의 실제 저장 예: ‘오른손이 닫힌 모양으로 보여요. 펴는 부분을 확인해 보세요.’ / ‘손목에서 팔꿈치 쪽으로 움직인 것으로 보여요. 참고처럼 팔꿈치 쪽에서 손목 쪽으로 쓸어 보세요.’ 자연 촬영의 모든 주먹/역방향 오류를 검출한다는 증거는 아니다. 잘하다 역순의 잘못된비슷함1건도 남았다.

자동 판단에 대한 정확한 결론
자동 구간 탐색은 이미 구현되어 있다. 다만 충분한 품질이 아니다. 새9항목에서 자동구간+방향4/9,정확한구간+방향8/9이며4항목은 구간을 바르게 주면 맞힌다.1항목은 판단과탐색 모두 남았다.
추가 저장값 점검: WORD0678의 보존 후보는 정답범위와 겹침0.474로0.5평가기준에 못 미친다. 완전히 아무 동작도 발견하지 못한 경우와 다르다. WORD1139의양손거리와WORD1290의세짧은높이이동은 같은축의 보존/최종선택 후보에 겹침이 없다. 그러나 새6영상에는 새검출기의 모든 기각 전 점수가 저장되지 않았으므로 생성실패인지 선택실패인지 지금 단정할 수 없다.

최종 결정
이전4단계의 방향은 유지한다. 다만 곧바로 더 학습하는 대신,1단계 앞에 ‘후보가 어디서 사라지는지’ 진단을 넣고 자동탐색과 최종교정의 성공조건을 분리한다. 새 대형모델이나LLM으로 교체할 근거는 현재 부족하다. 기존 가벼운 시간합성곱+GRU와 관절특징을 보완하는 설계를 우선한다. ASRF와TCN동작분할 연구는 이 접근의 기술적타당성을 지지하지만 한국수어 성능이나 최적모델임을 증명하지 않는다.

1. 누락 단계 진단과 시간별 학습 정답 보강
- 학습FIT 기록01~09의 공개자료에서 관측→활동점수→시작/끝 후보→후보선택→가이드연결을 모두 저장한다. 기존 동결평가를 다시 추론해서 점수를 바꾸지 않는다.
- 정답을 ‘팔을 움직임’ 하나로 붙이지 않고 손/신체관계/이동/방향전환/관계유지/준비마무리/관측불가로 나눈다. 위치유지와 손가락모양유지는 별도다. 유지 끝의 미세흔들림과 진짜 짧은 왕복을 함께 포함한다.
- 설명의 행동 역할과 시작·끝 관계도 관측 가능한 구간에 붙인다. 가림으로 모르면 보류하며 단순히 정답영상이라는 이유로 추정 정답을 채우지 않는다.
- 현재10표현24FIT의 감독을 바탕으로,같은100단어FIT 안에서 부족한 관계·지속시간·교차 조합의 다른 기록을 보강한다. 실패한 평가단어를 학습자료에 추가하거나 단어별 문턱을 만들지 않는다.
- 진단 분기: 후보가있는데 선택에서사라짐→선택/이어붙이기를우선보완; 활동/경계후보자체없음→해당탐색출력을추가감독; 정확한구간에서도틀림→행동판정부/관측정답보완. 모든경우를재학습하겠다고먼저정하지않는다.

2. 전체 시간 흐름에서 행동을 자동으로 찾도록 학습
- 정규화된RTMW관절 흐름과 관측신뢰도를 기존 시간합성곱+GRU가 읽는다. 이동/유지와 시작·끝·방향전환을 구분하는 출력을 함께 보완한다. 여러길이맥락은FIT지속시간분포로 사전고정한다.
- 짧은움직임을 평탄화하거나 반복을한개로합치지 않는다. 반대로 흔들림을 여러동작으로쪼개지도 않게 지도한다. 새손모양/반대방향이 나와도 관측한행동후보에서 버리지 않는다.
- 출력은 ‘어느손/어떤관계/언제부터언제까지/어떤변화/관측상태’의 후보 목록이다. 여기까지 정답단어·기대방향으로 관측값을 바꾸지 않는다. 한 구간을 두 반복으로 중복 사용하지 않는다.
- 이 단계는 촬영마다 사람이 구간을 지정하지 않게 만드는 기능이다. 기존전체파이프라인과 분리한 연구경로에서 검증한다.

3. 찾은 행동을 설명에 연결하고 교정 근거를 구성
- 목표단어의 가이드를 선택하고 관측후보의 앞·중간·뒤 관계를 연결한다. 시작자세가 닮았다는 이유만으로 앞행동을 뒤행동에 배정하지 않는다.
- 가이드는 손·관계·방향·순서·필수조건을 제공한다. 구체조건은 확인된 설명/주석에만 둔다. 수행자한명의좌표/속도/반복횟수를 모든사람의 필수정답으로 강제하지 않는다.
- DTW는 관측순서를 정렬하는 보조로 유지한다. 뒤집힌순서나반대방향까지 정답순서로고쳐연결하지 않고,대응안됨/추가행동/관측부족을 남긴다. 의미상완료는 시작·진행·도착관계의 근거가 갖춰져야 한다. 국소정지/구간겹침/손내림만으로 완료판정하지 않는다.
- 사용자출력은 ‘표시된 팔쓸기 구간에서 손을 펴보세요’처럼 구간+부위+한가지교정으로 구성한다. 방향/모양 근거가 불확실하면 오답문구 대신 확인불가로 표시한다. 이것은 설계예시이며 아직 새실사용성능을 얻었다는 뜻이 아니다.

4. 자동촬영입력부터 최종피드백까지 동일조건 비교
- 기준 / 움직임보완만 / 탐색연결보완만 / 결합4조건을 비교한다. 정확한구간시험은진단,자동구간시험이주성공기준이다.
- 지표를분리:행동검출누락/가짜행동,시작끝오차,반복수,가이드오배정,최종교정정확성,정상에잘못된경고,잘못된비슷함,불확실비율,전체대기시간.
- 새평가 공개기록은세부노출/SHA감사후점수전에예약한다. 이전결과는개발회귀. 같은Codex검토와 독립전문가정답은구분한다.
- 통과조건:기존정상/속도/교차경고증가0,손모양방향기존성과유지,짧은행동누락감소,틀린동작의정상승격감소,실제어느구간을어떻게고칠지설명개선. 불확실표시만늘어나는경우채택하지않는다. 학습난수3회로재현한다.
- 위근거가확보된경우에만 기존시각화테스트페이지에후보를연결한다. 최종운영교체는승인범위에따른다. 현재웹과C·T3-F는그대로다.

현재준비상태
이번은계획검토완료이고후속학습미시작이다. 사용자확인후위순서로진행한다.100단어공통학습/4단어피드백우선검증유지. 사용자·멘토촬영은학습/튜닝/모델선택금지.

논문근거(공통설계근거이며우리한국수어성능근거는아님)
ASRF: https://arxiv.org/abs/2007.06866
TCN for Action Segmentation: https://arxiv.org/abs/1611.05267
ActionFormer(여러시간길이의표현/경계추정참고,도입결정아님): https://arxiv.org/abs/2202.07925
'''
(U/'PLAN_KO.txt').write_text(plan,'utf-8')
write(U/'VERIFICATION.json',dict(saved_result_arithmetic_passed=True,rows=9,manual_correct=8,automatic_matches=4,manual_correct_auto_missing=4,body_wrist_only_confirmed=True,hand_and_axis_are_given=True,code_review_no_model_run=True,sources=sources,result_pins={str(p):sha(p) for p in [P/'RESULT.json',P/'DIRECTION_RESULTS.json',P/'FULLPATH_COMPLETE.json',P/'motion_core.py']},same_codex_not_independent_expert=True))
summary='현재능력·자동탐색 계획 재검토완료(학습/추론0).8/9는구간+손+관계축을지정한방향부품시험이며전체교정정확도가아님.자동탐색존재하나4/9;새5미매칭중1은넓은후보겹침.474,4는동일축보존/최종후보미겹침.생성/기각단계추가로그필요.기존4단계유지하되후보소실진단우선→시간별활동·경계감독→가이드역할/완료연결→최종피드백비교로구체화.사용자계획확인후재개.'
next_action='사용자 계획 확인 후 공개FIT에서 후보 생성·선택·가이드 연결 단계별 진단부터 실행. 상세 '+str(U/'PLAN_KO.txt')+' . 현재 학습/실험 재개하지 않음.'
(U/'before').mkdir(exist_ok=True)
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8')
for sec,body in [('## 8. 현재 상태',summary),('## 9. 다음 단일 작업',next_action)]:
 i=t.index('\n',t.index(sec))+1;t=t[:i]+'\n### 2026-11-11 자동탐색 능력·다음 설계 재검토\n\n'+body+'\n근거 '+str(U/'PLAN_KO.txt')+'\n'+t[i:]
p.write_text(t,'utf-8');state=read(O/'STATE.json');state['current_stage']=U.name;state['next_single_action']=next_action;state[U.name]=dict(status='PLAN_REVIEW_COMPLETE_AWAITING_USER_CONFIRMATION',summary=summary,previous_experiment=str(P),plan=str(U/'PLAN_KO.txt'),training=False,inference=False,private_data=0,web_changed=False);write(O/'STATE.json',state)
handoff=(ROOT/'NEW_CHAT_HANDOFF.md').read_text('utf-8');handoff=handoff.replace('# WordsInSilence 최신 인수인계','# WordsInSilence 최신 인수인계\n\n최신 요청: 현재 능력과 다음 계획을 먼저 보고, 확인 후 학습 재개.\n'+summary+'\n'+next_action+'\n후속 계획 단위 '+str(U)+'\n',1);(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,'utf-8')
with (ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf-8') as f:f.write('\n\n[2026-11-11 자동탐색 능력 설명·계획 검토]\n'+summary+'\n8/9를전반피드백정확도로오해하지않게입력힌트와출력3종을명시.논문설계원리와로컬성능근거구분.새모델교체보다기존구간탐색의생성/기각/연결어디서실패하는지검사하도록계획보강.\n근거 '+str(U/'PLAN_KO.txt')+'\n')
for p,h in read(U/'VERIFICATION.json')['result_pins'].items():assert sha(Path(p))==h
assert read(O/'STATE.json')['current_stage']==U.name
print(json.dumps(dict(status='plan_review_complete',manual=8,automatic=4,denominator=9,training_run=False,report=str(U/'PLAN_KO.txt')),ensure_ascii=False))
