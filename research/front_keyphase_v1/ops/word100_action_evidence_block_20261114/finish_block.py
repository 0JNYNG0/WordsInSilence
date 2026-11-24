from pathlib import Path
import json,hashlib,datetime,sys,shutil,importlib.util
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
v=read(U/'VERIFIED_V2.json');assert v['passed']and v['g2_pass']and all(v['g2_checks'].values())
assert not(U/'RESULTS_REVIEW.json').exists()
write(U/'AUDIT_CORRECTION.json',dict(
 first_verification_preserved=str(U/'VERIFIED.json'),corrected=str(U/'VERIFIED_V2.json'),
 cause='The first checker attributed a prior source audit to JOINT_CORRESPONDENCE_RESULTS, but the audit source hash actually names JOINT_ORDER_RESULTS. Record13 was already unknown in the intermediate snapshot.',
 fix='Read the exact originally audited JOINT_ORDER_RESULTS snapshot. No model, input, label, probability, threshold or gate changed.',
 records='Old audited source-assignment conflicts2 cases x3 seeds: supported->reverse. In the intermediate correspondence snapshot one was already unknown; that is not counted as a new false-positive reduction.',
 other_errors=['Read attempts for nonexistent compare.py/core.py were corrected by inspecting filenames; no data changed.',
 'PowerShell checkpoint update used nonexistent stage property; status is the actual field. It did not start or repeat training.'],
 first_two_failed_candidates_preserved=['EVALUATION.json:19/24','metric_recovery/EVALUATION.json:20/24'],
 final_development_candidate='state_path/EVALUATION.json:24/24',
 no_independent_test_claim=True))
report='''행동 구간과 설명 연결(G2) — 개발 검증 기준 통과

이번에는 작은 실험 하나가 끝날 때마다 종료하지 않고, 원본 확인 → 추가 학습 → 실패 원인 보완 → 자동 구간 연결 → 별도 검산까지 한 칸을 마무리했다.
전체 피드백 기능이나 웹 적용이 완료된 것은 아니다. 다음 칸 G3에서 요구한 행동 전체의 수행 여부와 교정 문장을 통합 검증해야 한다.

이전 → 현재

1. 손을 그대로 쥐고 팔만 움직여도 ‘손모양이 바뀌었다’고 하던 문제
   이전: 공개 개발 자료의 같은 24조건에서 22조건 일치, 정상 유지 2조건 오판.
   현재: 24조건 모두 일치. 먼저 손이 어떤 모양인지 보고, 앞뒤 모양의 차이로 변화 방향을 계산하도록 학습 구조를 보완했다.
   24조건은 공개 6영상에서 파생한 실제 구간·역재생·고정 자세 대조이다. 새 사용자 24명이나 전체 수어 정확도가 아니다.

2. 손을 닫기 시작한 것과 끝 모양까지 확인한 것을 분리
   공개 2영상으로 정상·짧은 수행·속도·고정 자세·역순·가림·중간 잘림 27조건을 확인했다.
   사전에 정한 23확인 항목을 통과했고, 나머지 4가림 조건도 별도 검산했다. 짧아도 펼친 손에서 닫힌 손까지 관측되면 근거를 남긴다.
   중간에 자른 5조건은 완료로 올리지 않는다. 짧은 입력의 모양 추정 자체에는 오류가 남아 있어, 기존 끝 모양 관측과 서로 맞지 않으면 확인 불가로 남긴다.
   ‘중간에 멈췄다는 실제 사용자 오류’를 모두 판별했다는 뜻은 아니다. 이 결과만으로 손을 더 접으라고 안내하지 않는다.

3. 사람이 지정한 구간에서만 되는지 확인
   기존 AI가 자동으로 찾은 공개 12영상의 82처리조건 × 3모델 결과, 총 246결과에 새 근거를 연결했다.
   정상 3영상의 자동 구간에서도 손 닫기와 닫힌 손 유지가 구분됐다. 잘하다·조상·이집트의 기존 판단도 덮어쓰지 않았다.
   정상 속도 차이의 기존 순서 근거 81쌍을 유지했다. 81은 9행동쌍 × 3속도조건 × 3모델이다.
   이전에 원본 주석과 충돌한다고 확인했던 2구간 연결 사례는 세 모델 모두에서 충돌 없는 역순 관측으로 바뀌었다. 모든 부분 역재생을 수어 오답으로 가정하지 않았다.
   손모양과 방향을 동시에 바꾼 3원본의 시험에서는 기존 손모양 차이를 유지했고, 2원본의 반대 방향 근거도 유지했다. 1원본은 여전히 방향 근거가 부족하다.

작업과 검증

- FIT 기록ID02~07의 원본 화면 216장을 확인하고, 유지되는 모양·보이는 중간 굽힘 등의 직접 정답을 보강했다. 기록ID를 사람 외형으로 동일인 판정하지 않았다.
- 작은 보조 판정부를 3방법 × 3초기값, 총 9회 학습했다. 첫 두 방법의 19/24, 20/24 실패를 보존하고 마지막 구조의 24/24와 후속 연결을 검증했다. 같은 실행을 다시 돌리지 않았다.
- 기존 공통100단어 모델을 다시 학습하거나 교체한 것이 아니다. 단어 이름·절대 위치·예상 순서를 입력하지 않는 공통 손모양 변화 부품을 연구 경로에 추가했다.
- 처음과 끝이 같은 모양이어도 중간에 바뀔 수 있으므로, ‘처음과 끝이 같음’과 ‘내내 같은 모양 유지’를 구분했다.
- 저장된 확률·분모·원본 대응·자료 분리·보호 파일을 별도 코드로 검산했다. 동일 Codex의 별도 검산이지 독립 한국수어 전문가 검증은 아니다.
- 첫 검산에서 과거 비교 파일 버전을 잘못 지정한 오류를 보존하고 실제 감사 대상 파일로 바로잡았다. 학습 결과·정답·통과 조건은 바꾸지 않았다.

G2 완료가 의미하는 것

정해 둔 개발 기준인 정상 유지, 확인된 잘못된 구간 연결 감소, 보류만 늘리지 않는 긍정적인 차이 확인, 세 모델 검증을 통과했다.
‘이 구간에서 이런 변화가 관측됐다’를 설명의 행동 항목에 전달하는 연구 연결 단계가 완료된 것이다.
‘그 단어를 끝까지 올바르게 했다’거나 ‘사용자를 정확히 교정할 수 있다’는 최종 합격은 아니다.

남은 것 — 다음 G3 한 칸의 계획

1. 손모양 변화, 손과 신체의 관계, 팔의 방향을 같은 행동 구간에 함께 묶는다. 손을 닫았다는 사실만으로 가슴 쪽 이동·팔 세움까지 완료했다고 하지 않는다.
2. 기존 정상에서도 놓친 신체 영역/팔 자세 근거를 먼저 FIT 원본으로 확인한다. 관절 오검출·관측 불가·모델 오판을 구분하고, 필요한 경우에만 해당 공통 정답과 판정부를 보강한다.
3. 정상·짧게 충분히 한 동작·중간 잘림·순서 변경·모양/방향 동시 오류를 고정 비교한다. 기존 모양·방향 근거와 정상 속도 차이·교차를 보존하고, 보류 증가만으로 성공 처리하지 않는다.
4. 같은 구간의 관측으로 뒷받침되는 문장만 준비한다. ‘어떻게 고치기’와 ‘잘 보이지 않아 확인하기’를 분리한다. G3 통과 뒤에만 봉인된 새 공개자료 평가와 웹 적용으로 이동한다.

현재 한계

- 개발에 여러 번 노출된 자료다. 최종 미사용 공개8자료는 아직 열지 않았다.
- 손가락 4개의 굽힘/펼침은 엄지, 손끝 모으기, 정확한 접촉, 깊이까지 해결한 것이 아니다.
- 자동 구간 일부의 행동 정체·끝 상태가 불확실하다. 이를 실제 누락 오류라고 단정하지 않는다.
- 관측 부족은 오답이 아니며, 모델 간 불일치는 정상 또는 완료로 자동 승격하지 않는다.
- 실제 촬영 전체 정확도와 대기시간은 이번 연구 계산만으로 새로 주장하지 않는다.

적용 및 보호

연구 부품·학습 결과·연결 검증에 반영. 운영 웹/기존 C·T3-F/공통100/원본 끝 모양 모델 그대로 유지.
사용자·멘토 자료 접근/학습/튜닝/후보 선택 없음. 카메라·다운로드·Git·외부전송·서버 재시작 없음.
이 작업의 학습과 비교 프로세스는 모두 종료됐다. 새 기능이 웹에서 이미 켜진 것으로 안내하지 않는다.

근거: PLAN.json, SOURCE_REVIEW.json, state_path/EVALUATION.json,
EXTENDED_COMPARISON.json, AUTO_INTEGRATION.json, ASSEMBLED_EVIDENCE.json,
VERIFIED_V2.json, AUDIT_CORRECTION.json.
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
write(U/'RESULTS_REVIEW.json',dict(status='G2_PASS_DEVELOPMENT_SCOPE_G3_PENDING',at=stamp,
 implementer='Codex',verifier='Same Codex separate saved-output/source arithmetic; not independent human KSL expert',
 graph_box='G2',block_complete=True,whole_project_complete=False,web_changed=False,
 training_runs=9,new_state_probe=dict(previous=22,current=24,total=24,independent_test=False),
 source_review=dict(sheets=6,panels=216),extended_cases=27,extended_predeclared_checks=23,
 automatic_scenarios=82,automatic_rows=246,automatic_original_videos=12,
 source_assignment_conflict_repairs=6,source_assignment_original_cases=2,normal_order_observations_preserved=81,
 g2_checks=v['g2_checks'],verification=str(U/'VERIFIED_V2.json'),protected_files=v['protected_file_count'],
 current_exposure='FIT02-07; exposed probe08/09; previously exposed development11-13. No unexposed final evaluation.',
 next_box='G3',next_plan=['Joint shape/body/forearm evidence on the same action',
 'FIT diagnosis of remaining body-region/forearm unknowns before any new training',
 'Fixed normal/short/partial/missing/reverse/combined-error preservation',
 'Evidence-specific correction vs observation request; no web before G3/G4'],
 unresolved=['Learned cutoff state may be wrong; endpoint disagreement remains unknown',
 'New state branch does not establish full sign correctness','Fine thumb/contact/aperture unsupported',
 'Some candidate identity/end-state evidence absent; no forcing semantic completion'],
 no_private_access=True,final8_unopened=True,process_running=False,
 evidence={str(U/n):sha(U/n)for n in ['RESULT_KO.txt','VERIFIED_V2.json','ASSEMBLED_EVIDENCE.json','AUDIT_CORRECTION.json']}))
# Graph transition through its own gate API; preserve previous failed attempts.
backup=U/'before_finish';backup.mkdir()
for p in [G/'TASK_GRAPH.json',G/'CURRENT_EXECUTION.json',O/'STATE.json',R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md']:
 if p.exists():shutil.copy2(p,backup/p.name)
spec=importlib.util.spec_from_file_location('completion_graph_api',G/'graph.py');graph=importlib.util.module_from_spec(spec);spec.loader.exec_module(graph)
graph.record('G2','PASS',v['g2_checks'],[U/'RESULTS_REVIEW.json',U/'VERIFIED_V2.json',U/'ASSEMBLED_EVIDENCE.json'],
 'Original development gates met after learned configuration/change recovery and automatic-interval integration. Exact audited source-assignment conflicts only, not all partial reversals. G3 whole-requirement completion, final evaluation and web rollout remain pending.')
write(U/'CURRENT_EXECUTION.json',dict(status='G2_COMPLETE',at=stamp,training_running=False,block_complete=True,next_box='G3',web_changed=False,final8_unopened=True))
write(G/'CURRENT_EXECUTION.json',dict(status='G2_PASS_G3_NEXT',at=stamp,active_unit=str(U),running=False,next_box='G3',evidence=str(U/'RESULTS_REVIEW.json')))
state=read(O/'STATE.json');state['current_stage']=G.name;state['current_followup']=U.name
next_action='G3 한 칸을 연속 진행: 같은 행동 구간의 모양 변화·신체 관계·팔 방향/끝 상태를 통합하고, 정상/짧은 충분수행/부분/가림/복합오류를 고정 비교한다. FIT 근거가 필요한 부분만 보강·학습. G3/G4 전 웹 교체와 Final8 평가 금지.'
state['next_single_action']=next_action
state[U.name].update(status='G2_PASS_DEVELOPMENT_SCOPE_G3_PENDING',completed=stamp,block_complete=True,training_runs=9,training_running=False,report=str(U/'RESULT_KO.txt'),verification=str(U/'VERIFIED_V2.json'),next_box='G3',web_changed=False)
state[G.name].update(status='G2_PASS_G3_PENDING',at=stamp,active_unit=str(U),latest_evidence=str(U/'RESULTS_REVIEW.json'),progress='G2 original development gates passed; graph is not complete and web unchanged.',remaining=next_action,training_running=False,web_changed=False)
write(O/'STATE.json',state)
context=R/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text('utf-8-sig')
title='### 2026-11-14 G2 행동·설명 연결 개발 검증 통과 / G3 종합 완료 판단은 다음 단계'
entry8=title+'\n\n공개 FIT 원본216패널 확인, 같은 손모양 유지와 실제 변화 정답 보강 후 3후보×3seed 학습. 첫19/24·20/24 실패 보존, 최종 변화 판정부는 기존22→24/24(노출 개발6영상 파생조건). 짧음/부분/가림27조건과 23사전확인 통과; 중간잘림5는 완료로 올리지 않음. 자동82조건×3모델 246결과에 구간·관측·설명 소유권을 분리 연결했고 기존 정상속도81순서쌍/모양·방향 근거 보존. 과거 감사로 확인된 구간 연결 충돌2사례×3seed 해소, 실제 사용자 오류 정확도 아님. 별도 저장검산 통과, 보호477파일 유지. G2의 원래 개발 기준 통과로 칸 완료. G3 전체 행동/끝 상태/교정 의미 검증, Final8, 웹 적용은 미완료. 짧은 입력의 모양 추정 오류는 기존 끝 모양과 불일치하면 보류하므로 여전히 한계. C·T3-F/공통100/웹 불변, 개인자료 접근 없음.\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`, `RESULTS_REVIEW.json`, `VERIFIED_V2.json`.\n\n'
entry9=title+'\n\n'+next_action+'\n단순 자세/순서 관측으로 전체 완료를 올리지 않고, 명확히 보이는 차이와 관측 부족을 구분한다. 세부 실험마다 작업 종료하지 않고 G3 칸의 통과 기준까지 검토한다. 현재 실행 프로세스 없음. 완료된9학습/기존비교 재실행금지.\n\n'
for header,entry in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',entry8),('## 9. 다음 단일 작업',entry9)]:
 assert text.count(header)==1;text=text.replace(header+'\n',header+'\n\n'+entry,1)
context.write_text(text,encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# G2 개발 검증 완료, 다음 G3 — {stamp}
목표: 현재4단어 웹 피드백. 공통100 및 기존 C·T3-F 보존. 사용자 요청대로 세부 실험별 종료 대신 그래프 칸 단위로 진행.
현재 단위: {U}
G2 원래 개발 기준 PASS. RESULT_KO.txt / RESULTS_REVIEW.json / VERIFIED_V2.json 확인. 전체 웹/피드백 완성 아님.
이번 3후보×3seed=9학습 완료. 첫19/24·20/24 실패 보존, 최종 변화24/24(기존22/24). 노출 probe08/09, 새로운 독립평가 아님.
자동82조건×3모델246결과 연결, 기존정상81순서쌍/모양·방향 유지. 구간충돌2사례×3seed 이전 감사대상 비교로 해소. 모든 부분역재생을 오답으로 보지 않음.
27확장조건/23사전확인 및 추가가림검산. 5중간잘림은 끝 상태 불일치로 완료하지 않음; 짧은 입력의 모양 모델 오류 자체가 해결된 것은 아님.
첫 검산의 과거 파일 버전 착오는 AUDIT_CORRECTION.json에 보존. VERIFIED.json 실패 대신 VERIFIED_V2.json이 최종; 정답/모델/통과조건 변경 없음.
다음 한 칸: {next_action}
G3 미통과, Final8 미개봉, 웹/C/T3F/공통100/원본끝모양 모델 유지. 새로운 연구 부품은 웹에 활성화하지 않음.
사용자·멘토자료 접근/학습/튜닝/선택 금지. 카메라/다운로드/Git/외부전송/서버재시작 없음.
모든 학습/비교 프로세스 종료. 완료9회 및 이전학습/평가 재실행하지 말고 저장출력 사용.
동일 Codex 별도 검산은 독립 사람/KSL전문가 정답 검증과 다름.
''',encoding='utf-8')
# A compact presentation of actual graph states, not an invented completion percentage.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/malgun.ttf');navy='#162D4A';grey='#EDF1F5';ink='#26364A'
fig,ax=plt.subplots(figsize=(14,6.4),dpi=150);fig.patch.set_facecolor('white');ax.set(xlim=(0,14),ylim=(0,6.4));ax.axis('off')
ax.text(.4,5.85,'피드백 기능 작업 현황',fontproperties=font,fontsize=23,weight='bold',color=navy)
ax.text(.4,5.33,'행동과 설명을 연결하는 G2 개발 검증 완료 · 전체 웹 기능은 다음 단계 진행 필요',fontproperties=font,fontsize=13,color=ink)
boxes=[(.45,3.35,'G0·G1  완료','자료·설명·정답 정리',True),(4.95,3.35,'G2  이번 완료','행동 구간 ↔ 설명 연결',True),(9.45,3.35,'G3  다음 단계','끝까지 했는지 종합 판단',False),(.45,1.1,'G4  예정','새 공개자료 최종 평가',False),(4.95,1.1,'G5  예정','교정 문장·영상 강조 적용',False),(9.45,1.1,'G6  예정','실제 촬영·지연 검증',False)]
for x,y,t,b,done in boxes:
 ax.add_patch(FancyBboxPatch((x,y),4.05,1.35,boxstyle='round,pad=0.08,rounding_size=0.12',facecolor=navy if done else grey,edgecolor=navy if t.startswith('G3')else'none',linewidth=2))
 ax.text(x+.2,y+.88,t,fontproperties=font,fontsize=16,weight='bold',color='white'if done else navy)
 ax.text(x+.2,y+.35,b,fontproperties=font,fontsize=14,color='white'if done else ink)
for a,b in [((4.58,4.02),(4.84,4.02)),((9.08,4.02),(9.34,4.02)),((4.58,1.77),(4.84,1.77)),((9.08,1.77),(9.34,1.77))]:
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=18,color=navy,lw=1.6))
ax.annotate('',xy=(.75,2.55),xytext=(13.15,3.15),arrowprops=dict(arrowstyle='->',color='#8492A5',connectionstyle='angle,angleA=-90,angleB=0,rad=8',lw=1.2))
ax.text(.45,.45,'G2 통과 = 개발자료에서 연결·관측 근거 검증 완료  /  사용자 교정 정확도·최종 웹 적용 완료를 뜻하지 않음',fontproperties=font,fontsize=11,color=ink)
fig.savefig(U/'PROGRESS_GRAPH.png',bbox_inches='tight',pad_inches=.15);plt.close(fig)
write(U/'RECORDS_SYNCED.json',dict(at=stamp,graph_status=next(n['status']for n in read(G/'TASK_GRAPH.json')['nodes']if n['id']=='G2'),
 paths={str(p):sha(p)for p in [G/'TASK_GRAPH.json',O/'STATE.json',context,R/'NEW_CHAT_HANDOFF.md',U/'RESULT_KO.txt',U/'PROGRESS_GRAPH.png']},process_running=False))
print(json.dumps(dict(status='G2_PASS',report=str(U/'RESULT_KO.txt'),graph=str(U/'PROGRESS_GRAPH.png'),next='G3'),ensure_ascii=False))
