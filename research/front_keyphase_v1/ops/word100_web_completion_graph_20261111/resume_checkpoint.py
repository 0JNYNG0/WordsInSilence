"""Record explicit resume, preserving the pause and all experiment results."""
from graph import *

O=U.parent; ROOT=O.parents[2]; now=stamp()
event=dict(at=now,event='USER_EXPLICIT_RESUME',request='저장한 중단 기록부터 작업 그래프 순서대로 검토·검증하며 계속 진행',checkpoint=str(U/'PAUSE_CHECKPOINT.json'))
write(U/'RESUME_CHECKPOINT.json',event)
g=read(U/'TASK_GRAPH.json');g.update(status='ACTIVE',resumed_at=now,resume_requires_user=False);write(U/'TASK_GRAPH.json',g)
write(U/'CURRENT_EXECUTION.json',dict(status='RUNNING_REVIEW',at=now,pid=None,training_running=False,web_changed=False,camera=False,private_access=False,resume_requires_user=False,next='dynamic block independent audit and FIT block definitions',graph=str(U/'TASK_GRAPH.json')))
with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
progress='사용자 명시적 재개 요청으로 중단 기록부터 이어간다. 완료된 역할 판정부6조건 학습·평가는 재실행하지 않는다. 같은8분류 기준64/67과 정상24비교구간 유지가 확인된 상태이며, G2는 아직 미통과다.'
nextwork='dynamic_blocks 저장 결과의 별도 검산과 FIT6영상 완전행동 블록 정의부터 진행한다. 검산 통과 후에만 새 학습 후보를 고정하고 비교한다. 최종 공개8영상 평가·웹 반영·카메라 시험은 아직 하지 않았다.'
title='2026-11-11 명시적 재개 — 웹 완성 그래프 G2 검산부터'
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=ctx.read_text(encoding='utf-8-sig')
for section,body in [('8.',progress),('9.',nextwork)]:
 ls=t.splitlines(keepends=True);i=next(i for i,l in enumerate(ls) if l.startswith('## '+section));ls.insert(i+1,'\n### '+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESUME_CHECKPOINT.json`\n\n');t=''.join(ls)
ctx.write_text(t,encoding='utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 웹 완성 그래프 재개\n\n'+progress+'\n\n'+nextwork+'\n\n그래프: '+str(U/'TASK_GRAPH.json')+'\n재개점: '+str(U/'RESUME_CHECKPOINT.json')+'\n\n사용자/멘토 자료 접근·학습·튜닝 금지. 기존 C·T3-F와 웹 보존. 같은 Codex 별도 검산과 독립 사람 정답을 구분한다. 게임 병합은 웹 완료 후.\n',encoding='utf-8')
s=read(O/'STATE.json');s.update(last_updated=now,updated_at=now,current_stage=U.name,next_single_action=nextwork);s[U.name].update(status='RUNNING_REVIEW',resumed_at=now,resume_requires_user=False);write(O/'STATE.json',s)
with (ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf-8') as f:f.write('\n'+title+'\n'+progress+'\n'+nextwork+'\n')
print('Resume recorded; G2 remains REWORK. No training/evaluation started by this script.')
