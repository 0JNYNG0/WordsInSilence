from pathlib import Path
from datetime import datetime
import json,shutil
U=Path(__file__).resolve().parent
ROOT=U.parents[3]
now=datetime.now().astimezone().isoformat()
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';state=U.parent/'STATE.json';handoff=ROOT/'NEW_CHAT_HANDOFF.md'
before=U/'before';before.mkdir(exist_ok=True)
for p in (context,state,handoff):
    if not (before/p.name).exists():shutil.copy2(p,before/p.name)
next_action='사용자 최신 자율승인에 따라 공통 신뢰도 가중/강건 대리학습 계약 검증→B2/B3 각3seed 및 B0/B1 비교→고정 C·T3-F와 DTW downstream 영향검증까지 지속. 이전 V05 HOLD/원자료는 보존하고 새 단위에서 대리좌표 복원과 실제 영상 정확도를 분리. 사소한검증은자율해결,진행근거확보불가능한큰문제만보고. 사용자촬영학습/튜닝/선택/접근 금지; 운영교체금지.'
entry='### 2026-10-25 RGB·관절·시간 보완 자율 비교 진행\n- 사용자가 단계별 재승인 없이 현재 계획 마지막까지 진행하도록 승인. 불확실한 정답문제도 자율분석·보완하고 도저히 해결할 수 없는 큰 문제만 중단보고.\n- 새 격리단위 `research/front_keyphase_v1/ops/word100_temporal_refiner_autonomous_20261025`. 이전 pilot HOLD 결과와 24장면 고정범위 유지.\n- 불완전 MP좌표를 해부학 정답으로 간주하지 않고, 공통 신뢰도 가중과 robust loss로 대리복원 시험. 원관측 좌표 보존/추정 표식 분리. 후보 성능·실제 위치·downstream 결과 별도 보고.\n- 작성: root 학습/특징, refiner_review 신뢰도, bundle_inventory 독립검토, bundle_split downstream. 기존 C·T3-F/운영웹 및 사용자자료 보호.\n\n'
c=context.read_text(encoding='utf-8-sig')
assert 'RGB·관절·시간 보완 자율 비교 진행' not in c
c=c.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+entry,1)
c=c.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n### 2026-10-25 현재 계획 자율 완료\n\n'+next_action+'\n\n',1)
context.write_text(c,encoding='utf-8')
s=json.loads(state.read_text(encoding='utf-8-sig'))
s.update(current_stage='TEMPORAL_REFINER_AUTONOMOUS_IN_PROGRESS',updated_at=now,next_single_action=next_action)
s[U.name]={'status':'PREPARING_ROBUST_PROXY_EXPERIMENT','unit':str(U),'current_execution':'RGB feature exec session 88014; root owns training; do not duplicate','training_runs_completed':0,'old_visible_hold_preserved':True,'user_data_access':False}
state.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
handoff.write_text('# WordsInSilence 최신 인수인계\n\n갱신 '+now+'\n\n현재 작업: '+U.name+'\n\n'+next_action+'\n\n진행: RGB 특징 exec 88014 확인 후 이어받기. teacher_reliability.py/TEACHER_POLICY는 refiner_review, independent_review.py는 bundle_inventory, downstream_compare.py는 bundle_split. root train_compare.py/rgb_features.py. 이전결과쓰기금지. 새 AUTHORIZATION/RESEARCH_CONTRACT/RGB_FROZEN을 먼저 확인. 학습 전 가중기준/모델조건/해시 고정. 학습 및 TEST결과 본 후 기준변경금지. 단순 대리좌표오차향상≠실제관절정확도≠인식률. 모델/특징 프로세스 중복실행금지.\n',encoding='utf-8')
print('Root documents synchronized; prior versions preserved.')
