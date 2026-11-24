"""Persist completed stage/contrast work and the exact remaining association problem."""
from graph import U,read,write,stamp,record
ROOT=U.parents[3];O=U.parent;W=U/'episode_stage_contrast';now=stamp()
assert read(W/'VERIFICATION.json')['status']=='PASS_SAVED_ARITHMETIC_G2_REWORK'
title='2026-11-12 동작/유지 구분 학습 개선 — 전체 행동 연결은 G2 보완 중'
progress='공개 FIT24영상의 실제 RGB 표본을 재검토해 44개 세부 움직임·30개 행동 묶음과 유지/경계 불확실성을 기록했다. 16영상 추가학습·8영상 해당 판정부 제외 비교로 움직임/마지막 자세 유지/그밖을 구분하는 작은 판정부3seed를 학습했다. 이어 준비·다른 자세·손내림 대조정답을264개(train)·128개(probe) 보강하고 같은1200최적화단계로 비교했다. 기존340개 판정 항목은338/336/336→338/339/340, 새128개 대조항목은78/80/80→124/126/125로 개선됐다. 입력 상류 모델은 이 수행자를 이전에 보았으므로 신규인물 성능이나 전체 피드백 정확도가 아니다.'
remaining='전체 영상 연결은 아직 미통과다. 움직임만으로 전체 행동구간을 채점하면 마지막 유지자세가 감점되어 정상구간이 누락되는 계약 문제를 발견해 분리 기록했다. 움직임+해당 유지자세로 바로잡은 저장결과 계산에서 정상/속도45구간 모두 후보를 얻고, 제한된 정상6동작 및 재배열6동작의 실제 진행표본을 모두 포함했다. 그러나 재배열의2/3영상은 한 구간에 다른 행동까지 포함했고, seed43은 정상1영상에서도 혼합이 남았다. 이는 정확한 구간/순서 피드백 성공이 아니다. 새 판정부와 연결식은 격리 후보로 보존하며 웹 미반영, G2 REWORK다.'
cause='동일 원본을 순서만 옮긴18개 확인장면은 관절위치·관측표시·현재 신체관계가 동일했으나 저장된 시간문맥 특징과 판정은 달라졌다. 이는 순서 문맥 민감성 진단이며 특정 모델 원인/수어 오답의 확정은 아니다. 단일 행동별 최고점만 고르면 다른 행동과 겹친 긴 구간을 중복 사용할 수 있다. 손이 모두 빠진 입력의 몸통 움직임 제안은 남아도 손특징의 관측 근거는0이며, 이를 손모양/접촉 정답으로 사용하지 않는다.'
nextwork='G2 다음 단일 작업: 겹친 행동 구간의 중복 근거 사용과 시간문맥 의존을 분리 진단한다. 기존 저장 후보에 올바른 별도 구간이 존재하는지 먼저 확인하고, 동시에 가능한 손모양/위치 특징과 서로 별개인 행동 발생을 구분하는 출처 기반 관계 정답을 준비한다. 기대순서 강제나 모든 겹침 금지로 해결하지 않는다. 필요 시 공개 FIT의 완전 행동블록을 내부 순서 그대로 옮긴 학습 대조를 준비하되 접합부/경계는 마스킹하고 정답 의미를 다시 검증한다. 새 조건은 고정 후 한번만 학습·비교. 정상 속도·교차·짧은 변화 및 누락/순서/관측부족을 함께 통과해야 G3로 이동한다. 최종8자료는 아직 열지 않는다.'
evidence=[U/'episode_stage/SAVED_VERIFICATION.json',W/'VERIFICATION.json',W/'BINDING_V2_SCORES.json',W/'CONTEXT_DIAGNOSTIC.json']
g=read(U/'TASK_GRAPH.json')
if not any('Stage/contrast component improved' in a.get('note','') for n in g['nodes'] for a in n['attempts']):
 record('G2','REWORK',dict(normal_preserved=False,false_support_reduced=False,not_abstention_only=True,three_seeds=True),evidence,'Stage/contrast component improved held-out-from-head sparse labels, but global action assignment still mixes distinct actions. V1 motion-only span score was incompatible with terminal holds; V2 retains action evidence yet fails false-support gate. No runtime adoption.')
report='작업 현황 — 웹 피드백 완성 그래프\n\n'+progress+'\n\n'+remaining+'\n\n'+cause+'\n\n'+nextwork+'\n\n검증: 같은 Codex의 별도 산술구현으로 혼동행렬·옛 출력 재사용·246개 저장예측·두 연결 계산·보호파일477개 SHA256 확인. 독립 사람/수어 전문가 검증 아님. 기존 실험 재추론/재학습하지 않았으며, 확장된 probe의 새 입력만 과거 모델을 추가 평가했다. 대조정답의 동일/유사 프레임과 full/upper보기는 독립 영상 개수로 세지 않는다. 기존 C·T3-F/공통100/웹 보존. 사용자·멘토자료 접근·학습·튜닝·선택 없음. 카메라·서버·다운로드·Git 없음. 게임병합은 웹완료 후.\n'
(W/'RESULT_KO.txt').write_text(report,encoding='utf-8')
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=ctx.read_text(encoding='utf-8-sig')
if '### '+title not in t:
 for sec,body in [('8.',progress+'\n'+remaining),('9.',nextwork)]:
  lines=t.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if l.startswith('## '+sec));lines.insert(i+1,'\n### '+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/episode_stage_contrast/RESULT_KO.txt`, `VERIFICATION.json`, `BINDING_V2_SCORES.json`\n\n');t=''.join(lines)
 ctx.write_text(t,encoding='utf-8')
state=read(O/'STATE.json');state.update(last_updated=now,updated_at=now,current_stage=U.name,next_single_action=nextwork)
state[U.name].update(status='G2_REWORK_JOINT_ASSOCIATION',at=now,resume_requires_user=False,training_running=False,web_changed=False,private_access=False,latest_evidence=str(W/'VERIFICATION.json'),progress=progress,remaining=remaining);write(O/'STATE.json',state)
execution=dict(status='G2_JOINT_ASSOCIATION_DIAGNOSIS',at=now,training_running=False,process_running=False,web_changed=False,private_access=False,camera=False,completed_experiments=['context_role_training','local_motion_quality','episode_membership','episode_stage','episode_stage_contrast'],next=nextwork,reviewer='root Codex; three delegated reviewers returned quota errors, no repeated retries; separate arithmetic is not independent human truth',resume_authorized=True,goal_tool_note='Old SignRep objective remains paused in goal API; explicit user resume authorizes ongoing local work. API cannot resume/edit that goal; do not create duplicate or claim complete.')
write(U/'CURRENT_EXECUTION.json',execution)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — G2 장면학습 개선, 행동 연결 보완\n\n'+progress+'\n\n'+remaining+'\n\n'+cause+'\n\n'+nextwork+'\n\n현재 작업: '+str(U)+'\n마지막 완료: episode_stage_contrast/verify.py 및 context_diagnostic.py. 학습·평가 프로세스 없음. 완료 결과 재실행 금지. G2 미통과, G3~G6 미진행. 사용자 재개 지시는 유효하며 과거 PAUSE는 이력이다. 옛 목표 도구 상태는 paused이고 현재 실행범위를 반영하지 못하므로 파일/사용자 승인 기준으로 이어간다.\n\n별도 에이전트3개 사용량 제한을 실제 확인했으며 이후 root가 구현·별도 산술검산 담당. 독립 사람/수어전문가 정답 아님. 원본/보호파일477개 해시 일치. 사용자·멘토자료 미접근, C·T3-F·공통100·웹 미변경, 최종공개8 미평가, 게임병합 보류.\n',encoding='utf-8')
history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
if title not in history.read_text(encoding='utf-8-sig'):
 with history.open('a',encoding='utf-8') as f:f.write('\n'+title+'\n'+report+'\n')
write(U/'RESUMED_PROGRESS.json',dict(at=now,status='ACTIVE_G2_REWORK',progress=progress,remaining=remaining,cause=cause,next=nextwork,training_running=False,web_changed=False))
print('Synchronized context8/9, STATE, handoff, history and graph. G2 remains REWORK; component learning gain retained.')
