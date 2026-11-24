"""Record completed L evidence and move to a separate observed-order audit."""
from common import *

U=HERE/'full_bag_rate_rank_learning'
evidence=[U/'INDEPENDENT_SAVED_FIT_VERIFICATION.json',U/'evaluation/INDEPENDENT_SAVED_VERIFICATION.json',U/'changed_span_shadow/VERIFICATION.json']
assert all(p.exists() for p in evidence)
summary='어려운 비교 후보의 학습 노출을 보강한 L을 학습·비교·별도 검산했다. FIT 460bag 모두 3seed에서 fullpositive를 골라 기존6실패를 회복했다. 그러나 DEV의 빠른 REAL11에서 약33ms 범위 겹침이 남아 미채택이다. L단독 정상9/9·느림9/9·빠름8/9, 고정반반결합 빠름7/9. 순서편집 실제 구간18/18·역순9/9 개선과 정상 손모양/방향 상태는 유지됐다. 구간 겹침을 수어 오답으로 해석하지 않는다.'
nxt='C5: 기존 FIT의 직접 진행 witness와 저장 BoundaryHead 진행 출력을 먼저 검산하여, 실제 행동의 관측 순서와 전체 범위 경계 불확실성을 분리할 최소 보완을 설계한다. 같은 몸 입력의 anchor 재학습 반복·약33ms 허용폭 조정·DEV 맞춤 임계값 변경 금지. G3의 양손 교대 필수 공백과 명시 구간/모델 계보 연결은 병행 준비한다. 원래 G2 정상 관측순서 유지 기준, 기존 범위 회귀 기록·누락/부분/완료 구분을 보존한다. 새 설계의 FIT 검증 후에만 필요한 학습/비교를 진행한다.'
decision=dict(at=stamp(),status='NOT_ADOPTED_CONTINUE_OBSERVED_ORDER_CAUSE_AUDIT',G2_pass=False,
    fit=dict(bags=460,public_clips=24,persons=6,passed_by_seed=[460,460,460],prior_failures_recovered=6,new_regressions=0,not_new_user_accuracy=True),
    development=dict(scenarios=82,seeds=[42,43,44],normal=[9,9],slow=[9,9],fast=[8,7],denominator=9,
                     dynamic_witness=[18,18],dynamic_witness_denominator=18,dynamic_reverse=[9,9],
                     conditions=['full_bag_rate_rank','old_local_full_plus_fullbag_half'],remaining='REAL11 speed125 actual33333us overlap, single seed42 / half seed42,43'),
    semantics=dict(normal_speed_state_changes=0,new_normal_warnings=0,completion=None,all_missing_shape='observation_unknown',omission_completion_promotions=0),
    interpretation='Full extent overlap, observed action order and completion are distinct. Existing G2 result remains failed; new observed-order evidence would need its own preregistered checks, not a tolerance waiver.',
    next=nxt,evidence=[dict(path=str(p),sha256=sha(p))for p in evidence],review='Root implementation/execution plus separate Codex saved arithmetic review, not independent human/KSL ground truth.',
    no_private=True,no_final8=True,web_changed=False)
assert not(U/'DECISION.json').exists();write(U/'DECISION.json',decision)
e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),status='ACTIVE_C5_OBSERVED_ORDER_DESIGN_AUDIT',current_command='Read-only FIT progression/order contract audit; G3 integration/capability preparation',jobs=[],session_id=None,other_session_id=None,training_running=False,process_running=False,next=nxt);write(GRAPH/'CURRENT_EXECUTION.json',e)
s=read(HERE/'SUBGRAPH.json');s.update(at=stamp(),status='ACTIVE_C5_OBSERVED_ORDER_DESIGN_AUDIT')
for n in s['nodes']:
    if n['id']=='C4':n.update(status='L_FIT_GAIN_DEV_EXTENT_REWORK',evidence=[str(U/'DECISION.json')])
if not any(n['id']=='C5'for n in s['nodes']):s['nodes'].append(dict(id='C5',status='FIT_SAVED_PROGRESS_AUDIT',task=nxt))
write(HERE/'SUBGRAPH.json',s)
g=read(GRAPH/'TASK_GRAPH.json');g2=next(n for n in g['nodes']if n['id']=='G2')
assert not any(a.get('experiment_id')=='L_FULL_BAG_RATE_REWORK'for a in g2['attempts'])
g2['attempts'].append(dict(at=stamp(),experiment_id='L_FULL_BAG_RATE_REWORK',status='REWORK',checks=dict(normal_preserved=False,false_support_reduced=True,not_abstention_only=True,three_seeds=True),evidence=[dict(path=str(U/'DECISION.json'),sha256=sha(U/'DECISION.json'))],note=summary));g2['status']='REWORK';write(GRAPH/'TASK_GRAPH.json',g)
st=read(OPS/'STATE.json');st.update(updated_at=stamp(),last_updated=stamp(),next_single_action=nxt,next_single_task=nxt)
st['word100_web_completion_graph_20261111'].update(at=stamp(),status='ACTIVE_C5_OBSERVED_ORDER_DESIGN_AUDIT',training_running=False,progress=summary,remaining=nxt,latest_evidence=str(U/'DECISION.json'));write(OPS/'STATE.json',st)
root=OPS.parents[2];p=root/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8-sig');title='### 2026-11-13 어려운 예제 학습 검증 완료 · 관측 순서와 경계 분리 검토'
assert title not in t
for head,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:
    assert head in t;t=t.replace(head,head+'\n\n'+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113/full_bag_rate_rank_learning/DECISION.json`.\n',1)
p.write_text(t,'utf-8')
(root/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — 2026-11-13 새벽 자율 작업 계속

{summary}

{nxt}

L train88933/evaluate88729/changedsemantic 모두 exit0. 완료 재실행 금지. FIT별도검산509630, 평가9215, 의미9818 검사는 산술 항목 수이지 사람/영상 표본 수가 아니다. L은 K와 동일 J 초기3모델·동일1200단계·동일source선택일정에 전체 저장음성최대512개와 유효속도 모두를 함께 보여준 비교이다. FIT6실패 음성노출0~3→56~73회, FIT3seed460/460. 추가 손실항2개의 묶음효과이며 각 독립효과나 신규사용자 정확도로 주장하지 않는다.

L 조건의 새 의미16movement·1attribute만 추론. 기존8semantic조건 보존. 정상/속도 상태변화0, allmissing 손모양144개 unknown, 모든completion None, omit/mask 완료승격0. 정상 손모양이 유지됐다는 것과 전체행동 완료는 다르다.

GATES stage2는 observed order support 유지이다. 현재 extent 비중첩 판정 실패를 기록하고, 실제 progression 근거의 순서를 별도 구성할 수 있는지 검토한다. 기대순서 매칭·33ms 허용문턱·검증결과 맞춤 가중치 금지. 기존BoundaryHead 제3채널도 sparse 진행정답을 이미 배웠으므로 동일학습을 새 이름으로 반복하지 않는다. motion agent=FIT 기존 진행출력 검산/최소계약, palm agent=G3 양손교대 기존모델/정답 재사용계약, absence agent=L 저장검산 완료, root=설계검토·기록.

G3_CAPABILITY_SCOPE_AUDIT: 연구5움직임 카드 vs웹8설명, coupled42 vs웹pose_transfer42/EvidenceHeads 계보. 교대 자체는필수인데 sequence_required 미지원. 이집트끝손끝모음 명시미검증, four_open으로대체불가. 웹 along_forearm은phase재범위선택하므로명시구간API필요, late뒤1/2 vs1/3차이도확인. 원래 모델과상태를조용히바꾸지않음.

R/A/PD/H/N/P/S/B/M/J/K/L 완료학습·평가재실행금지. allFIT손모델/최종8/웹미실행. 공통100/C·T3F유지. G0/G1PASS,G2REWORK,G3이후보류. 사용자/멘토자료·카메라·다운로드·Git·외부전송·운영기본교체금지. 같은Codex별도검산≠독립사람/KSL정답. 사용자새벽계속진행승인유효,중단상태아님.

H=research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/typed_motion_recovery_20261113. Python=ops/word100_signrep_feasibility_20261029/runtime/Scripts/python.exe. 현재 실행 프로세스 없음, 에이전트 읽기감사 진행. 실제CURRENT_EXECUTION/완료파일 먼저확인하고 중복실행하지않는다.
''','utf-8')
print('L recorded; active C5, no training process; G2 REWORK retained.')
