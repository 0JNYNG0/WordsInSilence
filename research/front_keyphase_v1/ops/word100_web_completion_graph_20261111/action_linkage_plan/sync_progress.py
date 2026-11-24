"""Persist exact graph progress, controlled comparison and next runnable unit."""
from pathlib import Path
import sys
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent;ROOT=U.parents[3]
sys.path.insert(0,str(U))
from graph import read,write,stamp,record
now=stamp();V=U/'relation_context_ablation'
assert read(V/'VERIFICATION.json')['status']=='PASS_ARITHMETIC_CANDIDATE_REJECTED'
assert read(W/'LABEL_VERIFICATION.json')['status']=='PASS_SOURCE_RECONSTRUCTION_PARTIAL_TRAINING_READINESS'
title='2026-11-12 행동 연결 세부 그래프·관계 정답 확장 — G2 보완 지속'
progress='전체 그래프는 G0·G1 기존 범위 통과, G2 행동구간-설명 연결 보완, G3~G6 미통과다. 기존 단계 판정부의 공개 표본 개선은 보존한다. 이번에는 G2를 A근거 보강/B행동 구분/C공동 연결/D통합 회귀로 나누고 실패 원인별 복귀 경로를 만들었다. 관계 정답 감사에서 반복/별개 주요행동이 각각 특정1단어에 몰린 편향을 발견했다. 공개 FIT12영상의 RGB표본을 추가/재검토하여 두 관계의 학습 출처를 각각3표현으로 확장했다. 큰 행동의 동일성과 개별 움직임 회차의 동일성을 분리한393표본쌍을 만들었고, 그중36쌍의 개별 회차 정답은 불확실하므로 비워 뒀다. 이는 영상393개나 새 모델 정확도가 아니다.'
trial='문맥 의존 원인분리를 위해 같은962학습항목/468판정항목·3seed·1200단계로 신체 상대관계의 가까운 앞뒤 흐름만 남기는 비교 학습을 완료했다. 기존462/465/465→451/450/451로 표본 판정이 내려갔고, 정상/속도45카드 중 후보누락0→3/2/2, 재배열6행동의 진행표본 포함6→4였다. 따라서 문맥을 단순 제거하는 후보는 채택하지 않는다. 기존 문맥 포함 단계모델을 유지하며, 의미 있는 관계 정답으로 연결을 보완한다. 모든 수치는 공개 개발진단/해당 작은 판정부 범위로 신규 사용자나 전체 피드백 정확도가 아니다.'
nextwork='다음 단일 작업: action_linkage_plan/MULTILEVEL_RELATION_TARGETS.json의159train/234probe표본쌍을 기존 공통100의 신체 정규화 관측 특징에 연결하고, 단어명/영상 길이/단순 시간간격만으로 답을 외울 수 있는지 감사한다. 사람·원본 해시 분리와 관측마스크를 검산한 뒤, 기존 시간모델을 보존하는 작은 부모행동/개별회차 관계 판정부의 고정 학습 계약을 작성한다. 현재쌍의 구간 시작·끝 또는 의미상 완료 정답은 따로 없으므로 같은 행동표본을 전체 완료로 승격하지 않는다. 필요 경계만 실제 RGB 확대검토. 입력검증 후 학습/공동연결/복합오류 회귀를 그래프대로 계속한다. 과거 완료실험 재실행·최종8평가·웹교체는 하지 않는다.'

plan='''웹 피드백 완성 작업 계획 — 2026-11-12

현재 위치
기존 그래프 전체 완료 아님. G0 범위고정/G1 기존 설명·주석 계약 통과.
G2 행동 구간과 설명 연결이 보완 중. G3 통합 판단, G4 별도 공개평가,
G5 웹 시각화 반영, G6 촬영·자동종료·전체 지연 확인은 남아 있다.
이 상태는 기존 웹 기능이 전혀 없다는 뜻이 아니라 새 완료기준을 아직 통과하지 않았다는 뜻이다.

왜 이 계획인가
1) 원본의 실제 움직임 표본을 포함하는 별도 후보는 이미6진단쌍 모두 존재했다.
   그런데 독립적으로 최고점만 고르면 다른 행동까지 포함한 긴 구간을 고르는 문제가 남았다.
   이는 알려진 정답으로 후보 존재를 확인한 감사이며, 모델 정확도6/6이라는 뜻이 아니다.
2) 같은 관절 위치도 전체 순서가 바뀌면 시간문맥 특징과 출력이 달랐다.
   직접 비교한 결과 문맥을 제거하는 것만으로는 해결되지 않았고 정상 판정도 나빠졌다.
3) 관계를 가르칠 정답이 단어별로 몰려 있었다. 이 편향을 보강하지 않은 학습은 근거가 약하다.

A. 한 행동과 반복 회차의 근거를 보강한다
- 원본의 직접 확인한 시작/진행/방향전환/끝·유지와, 설명의 해당 행동을 연결한다.
- 같은 큰 행동 / 그 안의 다른 반복 회차 / 별개 행동 / 동시에 가능한 특징 / 불확실을 구별한다.
- 큰 행동이 같다는 이유로 같은 회차라고 학습하지 않는다. 정답 없는 항목은 손실에서 제외한다.
- 이미 완료: 주석 분포 감사,12개 공개 FIT RGB표본 추가/재검토,393쌍의 두 수준 정답 및 별도 재구성 검산.
- 남은 일: 해당 관측 특징 연결, 시간간격·단어만으로 맞히는 편향 검사, 필요한 정확한 경계만 확대검토.
- 통과 기준: 출처·불확실성·수행자 분리·관측마스크 일치. 관측 안 된 손/접촉은 정답으로 만들지 않는다.

B. 행동 종류를 구분하는 학습을 보완한다
- 기존 공통100 시간 합성곱+GRU 및 좋아진 단계판정부를 기준으로 유지한다.
- 사전학습 모델을 또 새로 내려받지 않고, 필요한 관계를 출력하는 작은 학습부를 비교한다.
- 출력은 같은 큰 행동인지/같은 회차인지와 불확실 근거이며, 수어의 전체 정오가 아니다.
- 단어 ID·원본 프레임 번호를 모델의 답 힌트로 넣지 않는다. 목표 표현은 설명과 요구 특징 선택에 쓴다.
- 원본과 의미를 보존한 시간변형은 같은 쪽, 실제 별개 움직임은 다른 쪽으로 가르친다.
- 학습 전 자료·loss·3seed·최종 checkpoint·기준모델·검증조건 고정. 개발점수를 보고 경계/가중치 탐색 금지.
- 완료: 문맥 제거 가설의 고정 비교 및 별도 검산. 정상 회귀로 후보 미채택, 문맥 포함 기준 유지.

C. 설명과 후보 구간을 함께 연결한다
- 각 설명을 따로 최고점에 붙이는 방식과, 여러 후보의 관계를 함께 보는 방식을 비교한다.
- 반복 회차는 지우지 않고, 동시에 발생하는 손모양·방향·위치는 겹쳐도 허용한다.
- 실제 별개 행동은 같은 관측을 중복 근거로 사용하지 않도록 학습된 관계와 관측을 함께 본다.
- 목표의 기대순서로 먼저 강제 정렬하지 않는다. 연결한 다음 순서가 달랐는지 검사한다.
- 적절한 연결 없음(null), 부분 수행, 관측 부족을 완료와 분리한다.
- A의 입력 준비와 B의 기준모델이 있어야 시작한다. B/C 각각의 성능 상승을 먼저 요구하는 순환 의존을 만들지 않는다.
- 기준 / 학습 보완만 / 연결 보완만 / 둘 다의 기여를 분리하되, 새 조건만 실행하고 기존 저장출력은 재사용한다.

D. 실제 피드백 경로에서 함께 검증한다
- 정상·느림·빠름·교차·짧은 변화와 방향 반대·순서 교환·누락·관측 누락의 단독/복합 조건을 비교한다.
- '해당 행동을 찾았는가'와 '다른 행동까지 섞였는가'를 함께 센다. 보류만 늘려 경고를 줄이는 것은 실패다.
- 기존 손모양·방향 판정은 파라미터 보존뿐 아니라 새 구간을 넣었을 때의 결과도 확인한다.
- 관측 부족이면 불확실 표시, 충분한 근거가 있을 때만 원본 시간의 손·팔과 짧은 교정 문장을 연결한다.
- 통과하면 G3 통합→동결한 G4 별도 공개8→G5 웹→사용자가 시작하는 G6 실제 촬영 확인.

루프 규칙
정답/범위 불확실→A, 행동 종류 오인→B, 후보는 맞으나 붙이는 곳이 틀림→C.
여러 원인이 겹치면 함께 비교한다. 실패 근거는 남기고 검증된 부품은 보존한다.
같은 실패를 입력·가설 변경 없이 반복하지 않는다. 미사용 최종평가로 튜닝하지 않는다.
원본에서조차 판단 불가능한 경우는 강제 정답으로 해결하지 않는다.

범위와 검증 한계
4단어는 우선 웹 완료범위이며 공통100 학습은 유지. 추가 주석은 같은100 공개 FIT 안에서 확보.
C·T3-F/공통100/운영 웹 변경 없음. 사용자·멘토자료 접근/학습/튜닝/선택 없음.
Codex의 실제 이미지 검토와 별도 산술검산이며 독립 사람·한국수어 전문가 정답은 아니다.
에이전트는 기존 실제 사용량 제한을 기록해 둔 상태. 이번 구현·검산은 root Codex가 수행했다.
신규 관계판정부의 데이터는 상류 모델이 이미 본 공개 수행자이므로 새 사람 일반화로 포장하지 않는다.
'''
(W/'WORK_PLAN_KO.txt').write_text(plan,encoding='utf-8')
(W/'RESULT_KO.txt').write_text(progress+'\n\n'+trial+'\n\n'+nextwork+'\n\n근거: LABEL_VERIFICATION.json / ../relation_context_ablation/VERIFICATION.json / SUBGRAPH.json\n',encoding='utf-8')
sg=read(W/'SUBGRAPH.json')
for n in sg['nodes']:
 if n['id']=='A':n.update(status='SOURCE_LABELS_VERIFIED_INPUTS_PENDING',evidence=n['evidence']+[str(W/'LABEL_VERIFICATION.json')],next=nextwork)
 if n['id']=='B':n.update(status='DIAGNOSTIC_COMPLETE_BASELINE_RETAINED',evidence=n['evidence']+[str(V/'VERIFICATION.json')],next='Preserve full-context baseline. AfterA input audit, supervised relation/context contrast rather than removing context. A/B component availability permits C experiments; D judges integrated success.')
sg['updated']=now;write(W/'SUBGRAPH.json',sg)
g=read(U/'TASK_GRAPH.json')
if not any('Relation hierarchy expanded' in a.get('note','') for n in g['nodes'] for a in n['attempts']):
 record('G2','REWORK',dict(normal_preserved=False,false_support_reduced=False,not_abstention_only=False,three_seeds=True),[V/'VERIFICATION.json',W/'LABEL_VERIFICATION.json',W/'SUBGRAPH.json'],'Relation hierarchy expanded across expressions; local-only context ablation regressed and was rejected. Improved previous full-context component retained. No G2 completion or runtime adoption.')
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=ctx.read_text(encoding='utf-8-sig')
if '### '+title not in t:
 for sec,body in [('8.',progress+'\n'+trial),('9.',nextwork)]:
  lines=t.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if l.startswith('## '+sec));lines.insert(i+1,'\n### '+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/action_linkage_plan/RESULT_KO.txt`, `WORK_PLAN_KO.txt`, `LABEL_VERIFICATION.json`\n\n');t=''.join(lines)
 ctx.write_text(t,encoding='utf-8')
state=read(O/'STATE.json');state.update(last_updated=now,updated_at=now,current_stage=U.name,next_single_action=nextwork);state[U.name].update(status='G2_RELATION_TARGETS_READY_INPUT_AUDIT_NEXT',at=now,resume_requires_user=False,training_running=False,web_changed=False,private_access=False,latest_evidence=str(W/'LABEL_VERIFICATION.json'),progress=progress,remaining=nextwork);write(O/'STATE.json',state)
e=read(U/'CURRENT_EXECUTION.json');e.update(status='G2_RELATION_INPUT_AUDIT_NEXT',at=now,training_running=False,process_running=False,current_command=None,last_completed=str(W/'LABEL_VERIFICATION.json'),next=nextwork,subgraph=str(W/'SUBGRAPH.json'));e['completed_experiments']+=['relation_context_ablation'] if 'relation_context_ablation' not in e['completed_experiments'] else [];write(U/'CURRENT_EXECUTION.json',e)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — G2 세부 그래프·관계 정답 보강\n\n'+progress+'\n\n'+trial+'\n\n'+nextwork+'\n\n현재작업: '+str(U)+'\n계획: '+str(W/'WORK_PLAN_KO.txt')+'\n그래프: '+str(U/'presentation/web_completion_graph_20261112.png')+'\n학습/평가 프로세스 없음. 승인된 자율 진행 계속; 중단/재승인 요청 아님. 완료 실험 재실행 금지. 새 관계판정부는 아직 학습하지 않음. 현재393쌍은 큰 행동 동일성/개별 회차를 구분하는 정답이며 정확한 전체 경계/교육적 완료 정답 아님. 사용자·멘토자료 미접근, C·T3-F/공통100/웹 보존, 최종공개8 미평가.\n\n이전 에이전트3개 실제 사용량 제한. 이번 root가 구현/별도 산술검산; 독립 사람/KSL 검증 아님. 옛 목표도구 paused 상태는 현재 승인 실행범위를 반영하지 못함. 중복 목표/자동화 생성 금지.\n',encoding='utf-8')
history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
if title not in history.read_text(encoding='utf-8-sig'):
 with history.open('a',encoding='utf-8') as f:f.write('\n'+title+'\n'+progress+'\n'+trial+'\n'+nextwork+'\n')
write(U/'RESUMED_PROGRESS.json',dict(at=now,status='ACTIVE_G2_REWORK',progress=progress,trial=trial,next=nextwork,training_running=False,web_changed=False))
print('Synchronized graph, context8/9, STATE, handoff and research history. G2 remains REWORK.')
