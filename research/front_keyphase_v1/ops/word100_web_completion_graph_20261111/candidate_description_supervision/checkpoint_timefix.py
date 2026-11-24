from transfer_local_timefix_core import *
from graph import record
ROOT=O.parents[2]
assert read(TQ/'VERIFICATION.json')['protected_unchanged']
summary='공통100단어 관측 인코더를 후보 안에서만 재사용하는 비교를 완료했다. 새 실험에서 시간간격 입력을 진행률로 잘못 넣은 오류를 찾아 첫 실험은 제외·보존하고, 실제 시간간격으로 별도 학습·비교·검산했다. 수정 비교는 순서를 바꾼 공개 편집시험에서 개선됐지만 모든 초기값에서 정상 구간과 손모양 검사를 보존하지 못해 웹에 적용하지 않았다.'
remaining='같은 촬영의 잘라낸 정상 화면과 손모양만 바꾼 시험을 정확히 짝지어 확인하니, 손모양 변경 이전부터 후보 범위 점수가 낮은 사례가 있었다. 기존 전체화면과 잘라낸 시험화면 비교는 입력조건도 달라 손모양만의 영향으로 해석할 수 없다. 실제 잘라낸 공개 FIT 관측을 보강하고, 후보 양끝의 짧은 주변 흐름을 함께 보는 효과를 분리 비교한다. 정답 진행표본 전체포함은 교육적 완료 판정과 다르다.'
nxt='기존 ROI 방식으로 공개 FIT24영상의 상반신 관측을 준비한다. 원본 프레임·시각 대응과 관측 부족 마스크를 검증한 뒤 실제 잘라낸 관측 추가/후보 경계 주변133ms 문맥 추가를 분리한 고정 비교를 진행한다. 정상·속도·교차·손모양·방향 보존과 순서·누락 개선이 함께 확인되기 전 웹/최종8로 넘어가지 않는다. 오전7시부터 발표자료 마감 우선.'
(TQ/'RESULT_KO.txt').write_text(summary+'\n\n'+remaining+'\n\n'+nxt+'\n\n저장결과 별도 검산 PASS, 보호477파일 불변. 동일Codex 검산이며 독립사람/수어전문가 검증 아님.\n',encoding='utf-8')
record('G2','REWORK',dict(normal_preserved=False,false_support_reduced=True,not_abstention_only=True,three_seeds=True),[TQ/'VERIFICATION.json',TQ/'PAIRED_SHAPE_AUDIT.json',W/'transfer_local_candidate/INPUT_SEMANTICS_CORRECTION.json'],'Corrected time-unit comparison helps edited order in some seeds but regresses normal/shape. No deployment; next isolate actual framing and boundary context.')
ce=read(U/'CURRENT_EXECUTION.json');ce.update(at=stamp(),status='G2_PUBLIC_CROP_AND_BOUNDARY_CONTEXT_PREPARATION',training_running=False,process_running=False,current_command=None,session_id=None,last_completed=str(TQ/'VERIFICATION.json'),next=nxt);ce['completed_experiments']=list(dict.fromkeys(ce['completed_experiments']+['transfer_local_candidate_INVALID_TIME_UNIT','transfer_local_timefix']));write(U/'CURRENT_EXECUTION.json',ce)
s=read(O/'STATE.json');s.update(current_stage=U.name,next_single_action=nxt,last_updated=stamp(),updated_at=stamp(),activity_state='OVERNIGHT_CONTINUING');s[U.name].update(status=ce['status'],at=stamp(),latest_evidence=ce['last_completed'],progress=summary,remaining=remaining+' '+nxt);write(O/'STATE.json',s)
title='2026-11-12 전이 후보 입력정정·비교 완료 — 관측 화면과 경계 문맥 분리'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';v=p.read_text('utf-8-sig');assert title not in v
for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary+'\n'+remaining),('## 9. 다음 단일 작업',nxt)]:
 v=v.replace(marker,marker+'\n\n### '+title+'\n'+body+'\n근거: `research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/candidate_description_supervision/transfer_local_timefix/RESULT_KO.txt`, `VERIFICATION.json`, `PAIRED_SHAPE_AUDIT.json`.\n',1)
p.write_text(v,encoding='utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계 — 새벽 자율작업 계속\n\n'+summary+'\n\n'+remaining+'\n\n'+nxt+'\n\n현재 학습/평가 실행 없음. 완료 실험 재실행 금지. G2 보완중, 기존 웹/C·T3-F/공통100 보존. 사용자·멘토·최종8·카메라 미접근. 검토 에이전트3회 실제 quota 오류로 재호출하지 않고 Codex 로컬 검산. 발표자료 Docs/final_check_20261112, 실제 공개영상 DTW 그래프·스켈레톤·재생영상 포함. 브라우저 file 프로토콜 정책차단을 우회하지 않고 정적 PNG/영상 검수.\n',encoding='utf-8')
with (ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf-8') as f:f.write('\n\n'+title+'\n'+summary+'\n'+remaining+'\n'+nxt+'\n')
print('Time-unit recovery checkpoint synchronized')
