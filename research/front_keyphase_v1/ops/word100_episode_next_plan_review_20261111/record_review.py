"""Planning audit only. Reads saved results and writes project documentation."""
from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
P=O/'word100_action_episode_supervision_20261110'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2),'utf-8')
r=read(P/'RESULT.json');v=read(P/'VERIFICATION.json')
assert v['passed'] and v['result_sha256']==sha(P/'RESULT.json')
primary=[x for x in r['diagnostic'] if x['scope']=='additional' and x['seed']==42 and x['kind']=='all'][0]
assert primary['manual_old_family']==primary['manual_new_family']==primary['manual_direction']==14
assert primary['old']['joint']==13 and primary['new']['joint']==12
g={x['kind']:x for x in r['groups'] if x['seed']==42}
assert g['reversed_core_auto_order']['before_warnings']==4 and g['reversed_core_auto_order']['after_warnings']==5
assert g['reversed_core_auto_order']['after_similar']==1
assert all(g[k]['changed']==0 for k in ['full_normal','speed075','speed125'])
report='''다음 작업 계획 재검토 — 2026-11-11

판단
가장 먼저 해결할 것은 실제 움직임의 구간과 설명의 행동을 연결하는 과정이다. 현재 모델 전체를 바꾸거나 학습량부터 늘리는 것보다, 기존에 잘 찾던 움직임을 유지하면서 이 연결부를 보완하는 것이 현재 결과에 부합한다.
다만 큰 행동 종류·방향의 작은 표본 결과로 손가락 모양 학습이 완성됐다고 판단하면 안 된다. 모양 판정의 회귀 시험은 따로 유지한다.

이번 검토 범위
이전 RESULT/VERIFICATION 해시 일치 확인, 직접구간/자동구간 비교 수치, 원본 RGB와 대조한 원인 기록, 초기값3개 결과, 기존 NEXT_PLAN을 다시 읽고 대조했다. 새 학습·모델 추론·카메라·사용자 영상 열람은 하지 않았다. 같은 Codex의 기록 검토이며 독립 전문가 검증은 아니다.

확인된 문제
1) 준비 동작이 주요 동작에 섞이거나 끝부분이 잘린다. 추가 정상 이집트 영상에서 실제 펼침 구간을 기존·새 검출기 모두 충분히 잡지 못했다.
2) 반복 움직임 하나가 조각나거나 앞뒤 역할을 혼동한다. 조상 전체 역재생에서는 새 검출기가 반복1개를 놓쳤고, 순종 역순 시험은 행동 역할 충돌로 보류됐다.
3) 도착 모습만 비슷해도 전체 행동을 잘했다고 표시하는 경우가 남았다. 잘하다 역순 시험1개는 계속 비슷함이다. 실제 이동을 뒷받침하는 구간보다 급변이 섞인 부분 구간이 연결돼 있다.
4) 마지막 행동을 지운6조건은 보류다. 잘했다고 오인하지는 않지만, 무엇이 빠졌는지를 짚는 피드백은 아직 못 한다.
5) 새 후보의 이집트 역방향 경고는 실제 두 손 접근과 일치하나, 편집 연결점을 피하면서 나온 개선이다. 자연스러운 실제 사용자 오류까지 해결했다는 증거가 아니다.

수치의 의미
추가 공개4영상에서 확인한7개 움직임을 원본/전체 역재생으로 비교한14조건이다. 직접 구간을 주면 기존·새 모델 모두14/14의 큰 행동 종류·방향을 맞혔고, 자동 구간에서는 기존13/14→새12/14였다. 14개의 독립 영상도 아니고, 세밀한 손가락 모양 정확도도 아니다.
이 자료의 기록ID14는 이번 피드백 학습에서 제외했으나 과거 TCN 평가에 사용됐다. 이제 피드백 평가 결과도 봤으므로 앞으로 새 평가자료라고 부르지 않는다.
이전 주요 동작 역순9조건의 경고4→5는 제한 개선이다. 정상12/속도24의 기존 표시는 유지했지만, 정상도 일부 보류여서 정상100% 판정으로 보고하지 않는다.

1단계 — 문제를 함께 분리하고, 최소한의 정답을 보강
학습용 공개 영상에 행동별 사건ID와 실제 시작/방향전환/끝의 시간 범위를 연결한다. 프레임을 전부 완벽히 표시하는 작업이 아니라, 비교를 망치는 경계와 연결에 집중한다.
각 조각에는 ‘준비 / 주요 행동 진행 / 도착 후 유지 / 다음 행동으로 전환 / 마무리 / 관측 불가’를 붙인다. 서로 이웃한 두 조각에는 ‘같은 행동의 이어짐 / 반복의 다음 회차 / 서로 다른 행동 / 확인 불가’를 붙인다. 반복의 왕복과 모으기→펼치기 같은 구별 동작은 합치지 않는다.
원본 RGB로 빠른 실제 변화인지, 검출 좌표 오류인지, 역재생 시험의 편집점인지 구분한다. 편집점 정보는 시험 감사에만 사용하며 실사용 판정부에 입력하지 않는다.
기존4표현과 이미 경계 자료가 있는 다른6표현을 포함한다. 다른6표현의 현재 경계 주석은 주로 한 기록ID이므로, 이것만으로 사람 일반화 검증을 했다고 하지 않는다. 기존 공개 자료 안에서 학습·비교 기록ID를 분리해 추가 확인할 수 있는지 먼저 감사한다.
완료 조건: 누락/분리/역할 오류에 대해 해당 원본과 학습 정답이 연결돼 있고, 평가 자료가 학습/튜닝에 섞이지 않아야 한다. 보이지 않는 정답을 추측해 채우지 않는다.

2단계 — 행동의 조각을 연결하고, 시도한 행동은 빠뜨리지 않기
기존 검출기 후보를 보존한 채 공통 연결부를 만든다. 몸 기준 상대 위치, 움직임의 연속성, 방향 전환, 양손 관계, 관측 품질로 조각을 연결한다.
손모양이 정답과 다르거나 방향이 반대여도 ‘시도한 행동 후보’로 남긴다. 정답을 잘 닮은 구간만 골라 비교하지 않는다.
도착점만 보였다는 이유로 행동 완료로 올리지 않는다. 시작→이동→도착의 근거, 반복 횟수, 진행 순서를 함께 기록한다. 구간 하나를 두 번의 반복 정답에 중복 연결하지 않는다.
DTW는 같은 행동 안의 속도 차이를 맞추는 역할을 유지한다. 없는 행동을 다른 행동으로 늘려 채우거나, 실제 잘못된 순서를 정답 순서로 재배치하지 않는다. 연결되지 않은 시도와 기대 행동도 결과에 남긴다.
완료 조건: 기존에 잡던 주요 행동을 잃지 않고, 빠른 실제 변화·반복을 합쳐 지우지 않아야 한다. 관측 불가와 실제 행동 생략을 구분하지 못하면 보류한다.

3단계 — 필요한 연결부를 지도학습하고 원인별 효과 비교
기존 손모양·방향 판정부와 현재 운영 모델은 기준으로 고정한다. 1단계에서 검증한 정답으로 시작/끝 및 조각 연결/분리 판정부를 학습한다. 의미 역할 자체가 틀리는 사례가 확인되면 그 역할 정답도 함께 보강한다.
100단어899영상의 공통 학습을 유지하되 정답이 없는 세부 구간은 지도 정답처럼 취급하지 않는다. 학습용 자료 안에서 기록ID를 분리해 개발 비교하고, 외부 평가 점수로 규칙·모델을 고르지 않는다.
기존 방식 / 구간 보완만 / 설명 역할 연결만 / 필요한 둘을 결합한 방식으로 원인별 효과와 상호 영향을 구분한다. 범위가 큰 새 모델을 여러 개 가져오는 비교가 아니다. 판정부 학습은 필요한 조건에만, 사전 고정한 초기값3개로 수행한다.
완료 조건: 무엇 때문에 결과가 좋아졌는지 분리해 설명할 수 있고, 학습용 내부 검증에서 주요 행동 누락·정상 경고가 늘지 않아야 한다. 직접 지정한 구간에서도 틀리면 경계 문제만이라고 가정하지 않고 모양/행동 판정부를 점검한다.

4단계 — 실제 피드백 결과로 채택 여부 결정
기존78조건은 회귀용으로 유지한다. 다른 공개 평가 영상은 원본 중복·기존 노출을 확인한 후 사용 전에 예약한다. 확보할 수 없으면 일반화 검증이 부족함을 명시한다.
정상·속도차이·교차·유지·짧은변화·방향반대·순서변경·중단/누락·관측부족을 따로 집계한다. 편집으로 생기는 급변에 기대지 않도록 편집점 없는 원본과 연속 역재생 진단을 별도로 본다. 합성 오류 성능을 실제 학습자 오류 성능이라고 부르지 않는다.
구간 검출 점수 외에 ‘중요한 오류를 비슷하다고 넘겼는가’, ‘정상에 잘못된 교정을 했는가’, ‘어느 행동이 빠졌는지 짚었는가’, ‘정상까지 모두 보류하지는 않았는가’를 함께 본다. 실패·미검출도 분모에 포함한다.
통과 조건: 정상/속도/교차의 잘못된 경고 증가 없음, 기존 모양/방향 성능 저하 없음, 자동 구간에서 중요한 차이의 실제 검출 개선, 보류율 악화로 만든 가짜 개선 없음, 초기값별 안정성 확인. 처리시간은 새 연결부 시간과 촬영 후 전체 대기시간을 분리해 측정한다.
통과한 근거만 ‘이 구간에서 오른손을 손목 쪽으로 이어 내려 주세요’처럼 해당 설명과 짧은 문장·색상으로 연결한다. 손모양을 확실히 관측하지 못했으면 손모양 교정을 단정하지 않는다. 웹 기본 교체는 결과 확인 후다.

진행 분기
• 직접 구간은 맞고 자동 구간만 틀림 → 경계/조각 연결 우선.
• 직접 구간도 틀림 → 행동/모양 정답과 판정부 보강.
• 관절 자체가 틀리거나 안 보임 → 관측 품질 처리 및 보류. 구간 연결로 덮지 않음.
• 움직임은 잡았으나 설명의 다른 행동으로 연결됨 → 앞뒤 맥락과 설명 역할 연결 정답 보강.
• 두 원인이 함께 있음 → 각각과 결합 조건을 비교해 함께 수정할지 결정.

유지 사항
C·T3-F와 운영웹 유지. 사용자·멘토 촬영은 학습·튜닝·후보 선택에 사용하지 않음. 새 외부 모델 다운로드 없음. 이번 계획 검토에서는 학습/재추론하지 않음. 과거 동결 결과는 수정하지 않음.

이 계획의 판단 범위
현재 기록으로는 구간·역할 연결이 우선 병목이라는 근거가 있다. 이 보완만으로 모든 손모양/순서/누락이 해결된다는 근거는 아직 없다. 그래서 작은 원인 분리 비교를 먼저 하고, 필요한 학습만 수행한 뒤 전체 피드백으로 통과 여부를 판단한다.
'''
(U/'PLAN_KO.txt').write_text(report,'utf-8')
write(U/'REVIEW.json',dict(status='PLAN_REVIEWED_NO_EXECUTION',date='2026-11-11',evidence={str(P/f):sha(P/f) for f in ['RESULT.json','VERIFICATION.json','CAUSAL_AUDIT.json','NEXT_PLAN.json']},findings=['Boundary/linkage is supported priority, not proven sole cause.','Manual14/14 is coarse family/direction, not fine handshape.','Synthetic reverse gain may benefit from excluding edit splice; do not generalize to real errors.','6 extra words need more recordIDs for person generalization.','Preserve attempted wrong movement and unmatched segments through DTW.','Evaluate actual feedback and uncertainty coverage, not only boundary overlap.'],next_steps=['FIT error/episode pair annotations + source split audit','Generic proposal continuity/completion/role linkage','Necessary supervised linkage/role training with component and combined comparison','Frozen regression/generalization/latency and precise feedback check'],training_started=False,models_changed=False,personal_data_accessed=False,independent_expert_review=False,report=str(U/'PLAN_KO.txt')))
summary='저장한 결과·원인 기록을 재검토해 다음4단계 계획을 구체화했다. 직접구간14/14는 큰 행동종류·방향이며 손모양 완성 증거 아님. 구간/역할 연결이 우선 병목이나 유일 원인으로 단정하지 않음. 인위적 편집점·다른6표현의 사람 다양성·보류율 및 실제 피드백 기준을 추가했다. 새 학습/추론/웹변경/개인자료 접근 없음.'
nexttext='PLAN_KO의1→4단계: 공개FIT 사건·조각관계 정답/평가분리 감사→기존 후보를 보존하는 공통 연결·완수 근거→경계만/역할만/결합 비교와 필요한 지도학습→고정 회귀·추가공개·시간·피드백 검증. 현재는 계획 검토 완료, 후속 실행 전. C·T3-F/운영웹 및100단어 공통학습 보존.'
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text('utf-8')
for sec,title,body in [('## 8. 현재 상태','### 2026-11-11 다음 작업 계획 재검토 완료',summary),('## 9. 다음 단일 작업','### 2026-11-11 구간·역할 연결 보강4단계',nexttext)]:
 a=text.index('\n',text.index(sec))+1
 text=text[:a]+'\n'+title+'\n\n'+body+'\n근거 '+str(U/'PLAN_KO.txt')+'\n'+text[a:]
context.write_text(text,'utf-8')
state=read(O/'STATE.json');state['current_stage']=U.name;state['updated_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();state['next_single_action']=nexttext
state[U.name]=dict(status='PLAN_REVIEWED_NO_EXECUTION',summary=summary,next=nexttext,report=str(U/'PLAN_KO.txt'),review=str(U/'REVIEW.json'),last_experiment=str(P/'RESULT_KO.txt'),implementation_started=False)
write(O/'STATE.json',state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n마지막 실험: '+str(P/'RESULT_KO.txt')+'\n역순9 경고4→5의 제한 개선. 자동구간 추가14 기존13→새12, 전체교체 미채택. 정상/속도/교차 기존 표시 유지. 잘하다 비슷함 오판1,누락6보류 등 잔존.\n\n다음 한 작업: '+nexttext+'\n근거: '+str(U/'PLAN_KO.txt')+'\n이번에는 계획만 검토·문서화. 새 학습/추론 프로세스 없음. 웹 서버 상태는 변경하지 않음.\n','utf-8')
with (ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf-8') as f:f.write('\n\n2026-11-11 다음 작업 근거 재검토\n'+summary+'\n'+nexttext+'\n기존 계획 정밀화: 손모양 완성으로의 과잉해석 방지, 편집점 영향 분리, 조각의 동일행동/반복/다음행동 주석, 결합 원인 비교, 실제 교정·보류율 평가. 학습 실행 아님.\n근거 '+str(U/'PLAN_KO.txt')+'\n')
for p,h in read(U/'REVIEW.json')['evidence'].items():assert sha(p)==h
assert read(O/'STATE.json')['current_stage']==U.name
for p in [ROOT/'NEW_CHAT_HANDOFF.md',ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']:assert str(U/'PLAN_KO.txt') in p.read_text('utf-8')
write(U/'DELIVERY_CHECK.json',dict(passed=True,evidence_hashes_unchanged=True,documents_synchronized=True,training_or_inference=False))
print('Plan reviewed and recorded. No training or inference.')
