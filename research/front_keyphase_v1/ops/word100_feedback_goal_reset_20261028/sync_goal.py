from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,shutil
U=Path(__file__).resolve().parent;R=U.parents[3]
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
summary='사용자가 V4를 잠시 제외하고 최종 목표를 재정의했다. 목표는 한국수어 가이드와 참고 영상에서 손별 모양·방향·신체 관계·이동·반복·순서를 이해하고 사용자 수행의 정상 변이/중요 불일치/관측 부족을 구분하는 근거 기반 피드백이다. 기존 기록과 공식 문헌(LexSign, 영상언어 모델, 행동 시간 검출)을 검토했다. V4 추가실험/도입은 보류하며 중심 판단 모델은 아직 미완성이다. 구현은 제안 단계, 이번에 새 학습·추론·영상전송·자료/모델 다운로드·운영변경 없음. 사전 설명과 AI 프레임 관찰 초안을 구분하고 사용자 예시의 반복수/접촉부위를 검증 없이 필수규칙으로 확정하지 않는다.'
nxt='목표와 설계 설명을 확인한 뒤 현재 공개 한국수어 자료의 작은 묶음에서 가이드-주요단계 대응, 손별 실제 관측, 정상/오류/불확실 근거를 먼저 고정하고 중심 비교 기능의 전체 경로를 검증한다. V4후보 추가시험을 자동재개하지 않는다. 새 외부 자료·모델 다운로드는 출처/용도/규모/이용조건을 먼저 설명하고 확인받는다. 기존C/T3F/웹/DTW 유지, 사용자멘토자료 학습·튜닝·선택 금지. 새로운모델의학습/다운로드는 이번목표상담만으로 시작하지 않는다.'
paths=[R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',U.parent/'STATE.json']
(U/'before').mkdir(exist_ok=True)
for p in paths:
 if not (U/'before'/p.name).exists():shutil.copy2(p,U/'before'/p.name)
ctx=paths[0].read_text(encoding='utf-8-sig')
title='2026-10-28 목표 재정의 — 가이드와 영상을 연결하는 단계·관계 피드백 / V4 보류'
assert title not in ctx
for marker,heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',title,summary),('## 9. 다음 단일 작업','2026-10-28 중심 피드백 목표 확인과 한국수어 소규모 전체 경로 검증',nxt)]:
 ctx=ctx.replace(marker,marker+'\n\n### '+heading+'\n'+now+'\n'+body+'\n근거: research/front_keyphase_v1/ops/'+U.name+'/PLAN_KO.txt.\n',1)
paths[0].write_text(ctx,encoding='utf-8')
paths[1].write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음 한 작업: '+nxt+'\n\n최근 완료 결과: CPU 외부손모양 9회비교에서 V4 59.90%, V3대비9.15%p; 한국수어피드백개선증거아님. 오차행렬488장분석완료(204모두정답/167혼재/117모두오답). 이전보고/모델보존.\n\n작업단위: research/front_keyphase_v1/ops/'+U.name+'\n실행상태: 활성학습/추론없음. 목표상담과문헌확인만완료.\n',encoding='utf-8')
state=json.loads(paths[2].read_text(encoding='utf-8-sig'))
state.update(current_stage='GUIDE_VIDEO_FEEDBACK_GOAL_RESET_V4_ON_HOLD',updated_at=now,updated_date_kst=now,next_single_action=nxt)
state[U.name]=dict(status='GOAL_CLARIFIED_IMPLEMENTATION_PROPOSED',at=now,summary=summary,next=nxt,v4_work_on_hold=True,downloads_require_prior_explanation_and_confirmation=True,training=False,inference=False,operating_changed=False,user_mentor_used=False,report=str(U/'PLAN_KO.txt'))
paths[2].write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
assert json.loads(paths[2].read_text(encoding='utf-8'))[U.name]['v4_work_on_hold']
assert (U/'PLAN_KO.txt').exists()
print('Goal and constraints synchronized; no model or operating changes.')
