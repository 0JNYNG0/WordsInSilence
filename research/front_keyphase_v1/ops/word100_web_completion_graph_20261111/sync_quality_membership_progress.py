"""Checkpoint completed experiments without rerunning or promoting them."""
from graph import U, read, write, stamp, record

ROOT=U.parents[3]; O=U.parent; now=stamp()
evidence=[U/'local_motion_quality/EVALUATION_COMPLETE.json',U/'reviews/local_quality_saved_evaluation_verification.json',U/'episode_membership/EVALUATION_COMPLETE.json',U/'reviews/episode_membership_verification.json',U/'episode_membership/coverage_review/COARSE_RGB_REVIEW.json']
assert all(p.is_file() for p in evidence)
title='2026-11-11 再開後の実験完了'.replace('再開後の実験完了','재개 후 추가 비교 완료 — 부분 근거와 전체 행동 구간 분리')
progress='후보범위 판정부 B/C 각3seed 학습·2926저장후보 비교를 완료했다. 제한된 직접정답16개에서14개 일치했으나 전부 같은범주로 답하는 기준도14/16이어서 개선 근거가 아니다. 설명-구간 소속 판정부 mean/ordered 각3seed도 학습·82조건 비교를 완료했으나 정상/속도72구간 중23~25구간이 달라지고 일부 정상누락·반복동작 축소가 생겨 미채택했다. 기존 웹/C·T3-F/공통100은 변경하지 않았다.'
cause='동작 일부의 존재확률을 전체 비교구간 선택점수로 사용하면 짧은 조각을 전체행동으로 오인할 수 있다. 부분존재·전체범위·반복연속성을 분리하고 직접 구간정답을 보강한다. 두 새 비교는 완료되어 재실행하지 않는다. G2는 REWORK이며 최종공개8·웹반영·카메라는 미실행이다.'
nextwork='공개 FIT36영상 중 기존 세밀구간12영상 외 자료를 확인한다. 기록ID02/04의4단어8영상 overview·dense 이미지 준비완료, 실제 RGB 검토 후 불확실 경계·진행·유지·반복그룹을 구분한다. 기존04주석과 충돌은 보존하고 해결 전 강한 정답에서 제외한다. 정답·범위 검산 후 부분존재와 전체범위 선택을 분리한 다음 후보만 동결하여 비교한다. 이미 본 DEV로 임계값·학습횟수 튜닝하지 않는다.'
g=read(U/'TASK_GRAPH.json')
if not any(a.get('note','').startswith('Completed local quality and episode membership') for n in g['nodes'] for a in n['attempts']):
 record('G2','REWORK',dict(normal_preserved=False,false_support_reduced=False,not_abstention_only=False,three_seeds=True),evidence[:4],'Completed local quality and episode membership comparisons: no reliable quality gain; membership selection regresses normal coverage. Neither deployed. Expand source-reviewed FIT coverage and separate presence from complete interval.')
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=ctx.read_text(encoding='utf-8-sig')
if '### '+title not in t:
 for sec,body in [('8.',progress+'\n'+cause),('9.',nextwork)]:
  ls=t.splitlines(keepends=True);i=next(i for i,l in enumerate(ls) if l.startswith('## '+sec));ls.insert(i+1,'\n### '+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/episode_membership/EVALUATION_COMPLETE.json`, `reviews/episode_membership_verification.json`, `local_motion_quality/EVALUATION_COMPLETE.json`\n\n');t=''.join(ls)
 ctx.write_text(t,encoding='utf-8')
state=read(O/'STATE.json');state.update(last_updated=now,updated_at=now,current_stage=U.name,next_single_action=nextwork)
state[U.name].update(status='G2_REWORK_PUBLIC_FIT_COVERAGE',at=now,resume_requires_user=False,training_running=False,web_changed=False,private_access=False,checkpoint=str(U/'RESUME_CHECKPOINT.json'),latest_evidence=str(evidence[3]));write(O/'STATE.json',state)
write(U/'CURRENT_EXECUTION.json',dict(status='PUBLIC_FIT_DENSE_RGB_REVIEW',at=now,training_running=False,process_running=False,web_changed=False,private_access=False,camera=False,next=nextwork,completed_experiments=['context_role_training','local_motion_quality','episode_membership'],reviewer='root Codex; prior agents quota exhausted; root separate arithmetic is not independent human truth'))
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — G2 공개 FIT 구간정답 보완 중\n\n'+progress+'\n\n'+cause+'\n\n'+nextwork+'\n\n현재 작업: '+str(U)+'\n학습·평가 프로세스 없음. 완료 실험 재실행 금지. 마지막 prepare_dense.py 완료, COARSE_RGB_REVIEW.json 및 dense8장 존재. 사용자 명시적 재개 유효. PAUSE_CHECKPOINT는 과거 이력.\n\n별도 에이전트3개는 실제 사용량 한도로 종료. 이후 구현·산술검산은 root가 담당하며 독립 사람/수어 전문가 검토로 부르지 않는다. 초기 import·JSON 직렬화·경로 실패 기록 보존. 사용자·멘토자료 접근/학습/튜닝/선택 없음. 원본읽기전용. 게임병합은 웹완료뒤.\n',encoding='utf-8')
history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
if title not in history.read_text(encoding='utf-8-sig'):
 with history.open('a',encoding='utf-8') as f:f.write('\n'+title+'\n'+progress+'\n'+cause+'\n'+nextwork+'\n')
write(U/'RESUMED_PROGRESS.json',dict(at=now,status='ACTIVE_G2_REWORK',progress=progress,cause=cause,next=nextwork,training_running=False,web_changed=False))
print('Synchronized completed experiments and active public FIT review; G2 remains REWORK.')
