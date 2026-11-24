from core import *
from graph import record
ROOT=O.parents[2]

def main():
 assert (W/'VERIFICATION.json').exists();decision=read(W/'DECISION.json');V=W/'parent_consistency_repair'
 next_task=('G2 다음 단일 작업: 기존 공개 FIT의 SOURCE_EPISODES_DENSE.json과 저장된 행동 후보를 연결해 후보 자체의 설명 소속 정답을 준비한다. '
 '실제 행동 내부 / 행동 일부 / 다른 행동 혼입 / 준비·복귀 / 경계 불확실을 분리하고, 실제 RGB로 확인된 범위만 정답화한다. '
 '방향·손모양이 틀린 시도도 비교에서 빠지지 않도록 행동 소속과 정오를 분리한다. 동일 행동 관계 판정부의 개선은 보존하되 그 확률을 전체 구간의 정답률로 쓰지 않는다. '
 'FIT 내 사람 분리·클래스/표현 편향 검증 후 후보-설명 소속과 구간 범위를 직접 학습하고, 기존 단계 점수/관계 점수/공동 선택을 함께 비교한다. '
 'DEV 점수로 계수·경계 임계값을 탐색하지 않는다. 정상·속도·교차·짧은 변화와 순서/누락/관측부족 회귀를 통과해야 G3로 이동. '
 '반복의 개별 회차는 mixed-label 학습 표현이1개뿐이므로 다른 표현의 직접 정답을 보강하기 전 범용 반복완료로 주장하지 않는다. '
 'C·T3-F/공통100/웹 보존, 사용자·멘토 미접근, 최종8 미평가, 이미 완료된 실험 재실행 금지.')
 summary=('A~D의 정답 연결·학습·공동 구간 선택·저장결과 검산을 실행했고 A/B/C 원인별 보완을 거쳤다. '
 '공개30영상에서393표본쌍(159학습/234해당 판정부 제외 비교)을 기존100단어 관측 특징에 연결했다. '
 'full/upper는 같은 자료의2보기이며 독립영상으로 세지 않는다. 불확실한 개별회차36쌍은 해당손실에서 제외했다. '
 '단어만으로도 높게 나오는 편향을 확인해 단어 내부의 양·음성 균형과 별도 전이대조를 도입했다. '
 '반복의 같은행동 정답을 처음 제외한 문제가 드러나 보존학습으로 복원했다. 동일행동 판단은 full보기234쌍에서 3seed 각각183/166/165→234/232/234로 개선됐다. '
 '이는 이번 관계판정부 비교이며 신규사용자·전체피드백 정확도가 아니다. 상류 공통100모델은 비교 수행자를 예전에 보았다.')
 remaining=('구간 연결은 제한된 순서교환3영상에서 다른 행동 표본 혼입이6→3, 역순으로 분리되는영상이1→2로 개선됐다. '
 '하지만 보완모델의seed42는 한 영상에서 실제 행동 대신 준비동작 후보를 골라 주요행동 진행표본 포함이6→5로 떨어졌다. '
 '두 순간의 관계를 잘 맞히는 것만으로 설명에 맞는 전체행동 구간을 보장할 수 없었다. '
 '정상/속도45카드는 모두 후보를 유지하고 제한된 정상6행동 표본을 보존했으나, 변경된 구간으로 손모양·방향·교차를 종합 판정하는 회귀는 아직 통과가 아니다. '
 '개별 반복회차의 타표현 일반화도 부족하다. G2 REWORK, G3~G6 미통과, 웹 미반영.')
 text_report=f'''A~D 실행 결과 — 같은 행동을 묶는 학습은 개선, 전체 구간 선택은 보완 중

쉽게 말하면
AI가 두 장면을 보고 “하나의 행동이 이어지는 중이구나”라고 구분하는 능력은 좋아졌습니다.
하지만 영상 전체에서 “바로 이 구간을 이 설명과 비교하자”라고 고르는 능력은 아직 부족합니다.
그래서 실제 웹은 기존 상태로 유지했습니다.

A. 준비한 정답을 실제 관측에 연결
공개30영상의393표본쌍을 연결했습니다.159쌍은 학습,234쌍은 이 판정부 학습에서 제외한 비교입니다.
표본쌍은 영상2개가 아니라, 같은 영상 안의 두 순간입니다. full/upper 보기는 독립 자료가 아닙니다.
단어 이름만으로 높은 점수가 가능한 편향을 발견해 단어 내부 양·음성 균형과 평가를 보완했습니다.
개별 회차가 불확실한36쌍은 그 정답을 지어내지 않고 손실에서 제외했습니다.

B. 작은 관계 신경망 학습 및 원인 보완
기존100단어 공통 관측부는 고정하고 새 관계판정부(912→48→2,44천여 가중치)를 지도학습했습니다.
기존 신경망 전체를 다시 학습하거나 C·T3-F를 바꾼 것이 아닙니다. 각 후보3seed,고정1200단계입니다.
첫 후보가 반복동작을 서로 다른 큰 행동으로 나누는 문제를 확인했습니다.
이미 검토한 “반복해도 같은 큰 행동” 정답을 보존학습으로 복원했습니다.
234쌍 중 동일행동 판단:183/166/165 → 234/232/234. 각 숫자는 초기값을 바꾼3회 결과입니다.
상류모델은 이 수행자 자료를 이전에 보았습니다. 새 사용자 정확도나 수어정오 정확도가 아닙니다.
개별 반복회차는 양·음성을 모두 가진 학습표현이1개뿐이라 다른 표현 일반화가 아직 부족합니다.

C. 기존 연결과 비교
기존방식 / 구간내 관계만 / 구간끼리 관계만 / 둘을 함께 사용:4조건을 같은 저장입력으로 비교했습니다.
공개12영상에서 만든 정상·속도·순서교환·누락·관측부족·잘못된 손모양 등82조건입니다.
두 학습후보 각3seed를 비교했으며, 이전 관절추출과 공통모델 추론은 다시 실행하지 않았습니다.
제한된 순서교환3영상:다른 행동 확인표본 혼입6→3, 역순 구분1영상→2영상.
정상/속도45카드는 모두 후보가 있고, 직접 확인된 정상6행동의 진행표본을 보존했습니다.
하지만 보완후보의1seed에서는 준비동작을 본동작으로 골라6행동 중1행동의 진행표본을 놓쳤습니다.
전체 정확한 경계·누락·반복완료가 해결된 것이 아니므로 웹에 적용하지 않았습니다.

D. 별도 검산과 보호 확인
저장 확률,혼동행렬,후보 점수,선택 구간,진행표본 포함,다른 행동 혼입을 다른 계산 경로로 검산했습니다.
손 관측이 모두 없는 입력을 손모양/접촉의 관측근거로 채택한 항목은0입니다.
이는 누락검출 정확도100%라는 뜻이 아닙니다. 몸통 관계가 보여도 손 정답으로 올리지 않았다는 뜻입니다.
C·T3-F/공통100/기존웹 포함 보호파일477개 해시 불변. 사용자·멘토자료 접근·학습·튜닝 없음.
검산 담당은 같은 Codex이며 독립 사람/한국수어 전문가 검증과 다릅니다.
추가 관계계산·후보선택은 저장관측 기준 평균 약14ms,p95약19ms. 촬영/RTMW/전체웹대기시간은 아닙니다.
최종평가용 공개8영상과 실제 카메라 시험은 하지 않았습니다.

남은 문제와 다음 한 작업
{next_task}

기록
PRETRAIN_VERIFICATION.json,A_REWORK.json,FROZEN_V2.json,TRAINING_COMPLETE.json
parent_consistency_repair/CAUSE_AND_REPAIR.json,TRAINING_COMPLETE.json,COMPARISON.json
DECISION.json,D_REGRESSION_DETAIL.json,VERIFICATION.json
'''
 (W/'RESULT_KO.txt').write_text(text_report,encoding='utf-8')
 (W/'NEXT_PLAN_KO.txt').write_text('다음 루프 — 관계 판단의 개선을 정확한 행동구간 선택으로 연결\n\n'+next_task+'\n\n1. FIT 후보를 직접 검토된 행동 소속/불확실 경계에 연결. 일부만 보였다고 수어오답으로 라벨하지 않음.\n2. 후보의 소속과 관측된 범위 정답을 분리하여 학습. 예상 순서는 입력/선택 제약에 쓰지 않음.\n3. 구간 내부/구간 간 관계를 함께 비교. 준비동작으로 바꾸거나 반복을 잘라 점수만 높이지 않는지 확인.\n4. 정상·오류가 겹친 공개 개발 비교와 별도 검산. 통과한 뒤에만 상위G3통합을 시작.\n',encoding='utf-8')
 record('G2','REWORK',dict(normal_preserved=False,false_support_reduced=True,not_abstention_only=True,three_seeds=True),[W/'DECISION.json',W/'VERIFICATION.json',W/'COMPARISON.json',V/'COMPARISON.json'],'A-D execution complete with A shortcut repair/B parent-consistency repair/C contract repair. Relation component improved; combined selected span loses one true movement in repaired seed42. All normal shape/direction/cross integration not established. No web adoption.')
 sg=read(L/'SUBGRAPH.json');sg['updated']=stamp();sg['parent_status']='REWORK';updates={'A':('PASS_INPUT_MAPPING_WITH_SCOPE_LIMITS','393 source pairs bound/verified; word shortcut and supervision mask repaired.'),'B':('PARTIAL_GAIN_PARENT_RELATION','Parent234-probe gain retained as component; atomic transfer still limited.'),'C':('COMPARED_REWORK_CANDIDATE_IDENTITY','Four selection conditions compared; preparation replaces true motion in one repaired seed.'),'D':('REGRESSION_EXECUTED_NOT_PASSED','82 saved public scenarios x3 seeds per learned variant; arithmetic/protections pass, integrated gate incomplete.')}
 for n in sg['nodes']:
  st,note=updates[n['id']];n.setdefault('attempts',[]).append(dict(at=stamp(),status=st,note=note,evidence=str(W/'DECISION.json')));n['status']=st;n['next']=next_task if n['id'] in ['A','C','D'] else 'Preserve improved parent component; extend genuine repeat-cycle sources before generality claim.'
 sg['next_single_action']=next_task;write(L/'SUBGRAPH.json',sg)
 ce=read(U/'CURRENT_EXECUTION.json');ce.update(status='G2_HIERARCHICAL_RELATION_COMPARED_REWORK',at=stamp(),training_running=False,process_running=False,current_command=None,last_completed=str(W/'VERIFICATION.json'),next=next_task,web_changed=False,private_access=False,camera=False);ce['completed_experiments']+=['hierarchical_relation_learning','hierarchical_relation_learning/parent_consistency_repair'];write(U/'CURRENT_EXECUTION.json',ce)
 s=read(O/'STATE.json');s.update(current_stage=U.name,next_single_action=next_task,last_updated=stamp(),updated_at=stamp(),activity_state='A_D_VERIFIED_REWORK_REPORTED');s[U.name].update(status=ce['status'],at=stamp(),training_running=False,web_changed=False,private_access=False,resume_requires_user=False,latest_evidence=str(W/'VERIFICATION.json'),progress=summary,remaining=remaining+' '+next_task);write(O/'STATE.json',s)
 title='2026-11-12 A~D 실행·관계학습 개선 — 전체 구간 연결은 G2 보완'
 evidence='근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/hierarchical_relation_learning/RESULT_KO.txt`, `DECISION.json`, `VERIFICATION.json`'
 cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';contents=cp.read_text(encoding='utf-8-sig');assert title not in contents
 for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary+'\n'+remaining),('## 9. 다음 단일 작업',next_task)]:
  assert marker in contents;contents=contents.replace(marker,marker+'\n\n### '+title+'\n'+body+'\n'+evidence+'\n',1)
 cp.write_text(contents,encoding='utf-8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — A~D 관계학습/연결 비교 완료, G2 보완\n\n'+summary+'\n\n'+remaining+'\n\n'+next_task+'\n\n현재작업: '+str(U)+'\n이번 결과: '+str(W/'RESULT_KO.txt')+'\n별도 검산: '+str(W/'VERIFICATION.json')+'\n\n학습/평가 프로세스 없음. 이번 A~D 비교 및 부모행동 보존학습 완료, 재실행 금지. 승인된 후속 자율보완은 유지되며 추가 재승인 요청 아님. 사용자·멘토자료 미접근, C·T3-F/공통100/웹 보호477해시 불변, 최종공개8 미평가. 실제 촬영시험 없음.\n\n담당 root Codex 구현+별도 산술검산(독립 사람/한국수어 전문가 검증 아님). 과거 에이전트3개 실제 사용량 제한 기록은 보존하되 현재 장애라고 추측하거나 반복호출하지 않음. 옛 목표도구 paused와 현재 사용자 승인 실행범위를 혼동하지 않음.\n',encoding='utf-8')
 history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
 with history.open('a',encoding='utf-8') as f:f.write('\n\n'+title+'\n'+summary+'\n'+remaining+'\n연구 판단: 두 장면의 동일행동 관계와 후보 전체의 설명 소속을 별도 학습 대상으로 구분. 반복행동 보존을 버려 편향을 줄이는 대신, 표현 내부의 대조학습과 정상보존을 함께 학습. 부분구성 개선을 실제 웹피드백 완성으로 과장하지 않음.\n'+next_task+'\n'+evidence+'\n')
 # Delivery consistency, no repeated model tests.
 assert read(O/'STATE.json')['next_single_action']==read(U/'CURRENT_EXECUTION.json')['next']==read(L/'SUBGRAPH.json')['next_single_action']
 assert title in cp.read_text(encoding='utf-8') and summary in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf-8')
 assert all(Path(p).is_file() for p in [W/'RESULT_KO.txt',W/'NEXT_PLAN_KO.txt',W/'VERIFICATION.json'])
 write(W/'DELIVERY_CHECK.json',dict(at=stamp(),status='PASS_DOCUMENTS_SYNCHRONIZED',root=str(ROOT),state=str(O/'STATE.json'),context=str(cp),handoff=str(ROOT/'NEW_CHAT_HANDOFF.md'),history=str(history),process_running=False,web_changed=False,private_access=False))
 print('Synced result, A-D graph, main graph, context8/9, state, handoff and research history.',flush=True)

if __name__=='__main__':main()
