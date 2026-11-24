from local_sequence_core import *
from graph import record
ROOT=O.parents[2]
assert read(Q/'VERIFICATION.json')['protected_unchanged']
summary='후보 구간 내부의 신체상대 관계·손좌표만 보는 작은 시간합성곱+GRU를 공개24원본의 같은 구간정답으로3회 학습·비교했다. 같은 행동블록을 다른 시점으로 옮겨도 입력차0, 저장확률차 약0.0000005 이하라 구간 밖 문맥의 영향은 제거됐다. 하지만 정상 행동을 잘못 묶거나 일부를 놓치는 오류가 늘어 미채택이다. 관측문맥 제거만으로 해결되지 않는다는 결과를 보존했다.'
remaining='올바른 후보범위가 있어도 확률순위가 잘못되는 사례와 편집 접합부를 실제 움직임처럼 고르는 사례가 함께 있다. 새 임계값을 개발결과에 맞추지 않는다. 기존100단어에서 학습한 관측 인코더를 각 후보 내부에서만 다시 계산하는 제한된 전이 비교를 진행한다. 인코더·C·T3-F·웹은 고정하고 작은 판정부만 학습한다. G2 보완 중, 최종8·웹교체 미진행.'
nxt='전이 관측 후보의 고정3초기값 비교를 마친 뒤 기존 손모양·방향 판정을 연결해 검산한다. 정상/속도/교차를 보존하면서 순서·누락 개선이 함께 확인돼야 다음 검증으로 이동한다. 실패 시 후보 탐색/학습근거/관측 불연속 원인을 분리하고 결과를 덮어쓰지 않는다. 발표자료16개 PNG/SVG와 실제공개 DTW재생영상·슬라이드 구성안 준비; 오전7시 이후 자료 마감 우선.'
(Q/'RESULT_KO.txt').write_text(summary+'\n\n'+remaining+'\n\n'+nxt+'\n\n검산: 저장배열 재집계·고정선택 재계산·보호477해시 PASS. 동일Codex 별도산술이며 독립사람/수어전문가검증아님.\n',encoding='utf-8')
record('G2','REWORK',dict(normal_preserved=False,false_support_reduced=False,not_abstention_only=False,three_seeds=True),[Q/'VERIFICATION.json',Q/'FAILURE_DIAGNOSIS.json'],'Local candidate sequence is context-invariant but regresses normal sources. Next test common100 pretrained observation features locally, not a new word-specific rule.')
ce=read(U/'CURRENT_EXECUTION.json');ce.update(at=stamp(),status='G2_LOCAL_PRETRAINED_COMPARISON_RUNNING',training_running=False,process_running=True,current_command='evaluate_transfer_local.py',session_id=54804,last_completed=str(Q/'VERIFICATION.json'),next=nxt);ce['completed_experiments']=list(dict.fromkeys(ce['completed_experiments']+['local_sequence_candidate']));write(U/'CURRENT_EXECUTION.json',ce)
s=read(O/'STATE.json');s.update(current_stage=U.name,next_single_action=nxt,last_updated=stamp(),updated_at=stamp(),activity_state='OVERNIGHT_CONTINUING');s[U.name].update(status=ce['status'],at=stamp(),latest_evidence=ce['last_completed'],progress=summary,remaining=remaining+' '+nxt);write(O/'STATE.json',s)
title='2026-11-12 후보 내부 시간학습 비교 — 문맥 독립성 확인·정상 회귀 미통과';p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';v=p.read_text(encoding='utf-8-sig');assert title not in v
for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary+'\n'+remaining),('## 9. 다음 단일 작업',nxt)]:v=v.replace(marker,marker+'\n\n### '+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/candidate_description_supervision/local_sequence_candidate/RESULT_KO.txt`, `VERIFICATION.json`, `FAILURE_DIAGNOSIS.json`.\n',1)
p.write_text(v,encoding='utf-8');(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 새벽 자율작업 계속\n\n'+summary+'\n\n'+remaining+'\n\n'+nxt+'\n\n현재 evaluate_transfer_local.py exec54804 진행중. 중복실행금지. 완료학습/비교재실행금지. Codex로컬구현·별도검산. 에이전트quota실패로반복호출없음. 발표자료 Docs/final_check_20261112. 브라우저file프로토콜정책차단으로자동브라우저검수미실행,PNG와영상프레임검수수행. 사용자/멘토/카메라/다운로드/외부전송/Git 없음.\n',encoding='utf-8')
with (ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf-8') as f:f.write('\n\n'+title+'\n'+summary+'\n'+remaining+'\n'+nxt+'\n')
print('Local sequence checkpoint synchronized')
