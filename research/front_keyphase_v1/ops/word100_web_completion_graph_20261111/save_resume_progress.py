"""Synchronize current evidence after the resumed controlled comparison."""
from graph import *
O=U.parent;ROOT=O.parents[2];now=stamp()
progress='재개 후 같은 정답·학습 횟수를 맞춘 역할 판정부 추가학습6조건과 공개 개발82조건 평가·별도 검산을 완료했다. 기존8분류 정답점64/67과 정상/속도 비교구간은 유지됐지만,246쌍 비교의 실제 선택구간·순서 변화0으로 추가학습 이득은 없었다. 웹에 반영하지 않았다.'
cause='후보 품질만 우대하면 한 사례 개선과 다른 사례 악화가 함께 생겼다. 기존 경계 판정부의 공개82입력2802후보를 저장·검산했고, 검토한3영상의B/C6움직임은 정상·재배열 모두 실제 진행 witness를 포함한 후보6/6이 존재했다. 대표 시작·끝 점이 RGB불확실범위와 모두맞은 것은5/6으로, 남은1개는종료대표점이1프레임앞선 경우다. 실제순서/수어완료 정확도가아니다. 올바른후보를고르는문제와움직임/유지/관측불연속을나누는문제가남아G2미통과.'
nextwork='학습 횟수만 늘리지 않는다. 검토된 FIT30영상33관계의 후보범위 정답을 점검하고, 동일 요약 입력의 재학습 대조B와 기존Localizer시간Conv+GRU복사C의 최소 비교를 준비한다. 새 local_motion_quality는 코드·입력준비 단계이며 아직 학습하지 않았다. 합성편집은RGB진행근거를별도승인하기전aux생성보류. 합성에동작완료/오답정답을주지않는다. 준비자료·코드검산후조건동결→학습→공개개발비교→G2판정순서. 최종공개8·웹반영·카메라는아직미실행.'
title='2026-11-11 재개 검산·원인 분리 — 추가 역할학습 효과 없음, 실제 움직임 후보 선택 보완'
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=ctx.read_text(encoding='utf-8-sig')
if '### '+title not in t:
 for sec,body in [('8.',progress+'\n'+cause),('9.',nextwork)]:
  ls=t.splitlines(keepends=True);i=next(i for i,l in enumerate(ls) if l.startswith('## '+sec));ls.insert(i+1,'\n### '+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/context_role_training/EVALUATION_COMPLETE.json`, `reviews/context_evaluation_verification.json`, `boundary_candidate_probe/SOURCE_RECALL_AUDIT.json`\n\n');t=''.join(ls)
 ctx.write_text(t,encoding='utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 웹 완성 그래프 G2 계속 진행\n\n'+progress+'\n\n'+cause+'\n\n'+nextwork+'\n\n현재 작업: '+str(U)+'\n학습/평가세션9884·59687은exit0완료. 완료실험재실행금지. 사용자명시적재개 RESUME_CHECKPOINT.json, 과거PAUSE_CHECKPOINT.json은이력이다.\n\n기존C·T3-F/웹/공통100보존. 개인·멘토자료접근·학습·튜닝금지, 원본읽기전용. 같은Codex별도검산과독립사람/수어전문가정답을구분. Unity는웹완료뒤.\n',encoding='utf-8')
state=read(O/'STATE.json');state.update(last_updated=now,updated_at=now,current_stage=U.name,next_single_action=nextwork)
state[U.name].update(status='G2_REWORK_LOCAL_QUALITY_PREPARATION',at=now,resume_requires_user=False,training_running=False,web_changed=False,private_access=False,latest_evidence=str(U/'context_role_training/EVALUATION_COMPLETE.json'));write(O/'STATE.json',state)
write(U/'CURRENT_EXECUTION.json',dict(status='LOCAL_MOTION_QUALITY_PREPARATION',at=now,training_running=False,web_changed=False,private_access=False,camera=False,agents=['graph_binding_design implementation','graph_label_audit label preflight','sequence_target_review source recall audit'],next=nextwork))
with (ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf-8') as f:f.write('\n'+title+'\n'+progress+'\n'+cause+'\n'+nextwork+'\n')
write(U/'RESUMED_PROGRESS.json',dict(at=now,status='ACTIVE_G2_REWORK',progress=progress,cause=cause,next=nextwork,training_running=False,web_changed=False))
print('Current state/handoff/history synchronized. Work remains active at G2.')

if __name__=='__main__':pass
