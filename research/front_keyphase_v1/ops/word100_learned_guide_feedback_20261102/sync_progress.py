from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,sys
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
status='LEARNED_GUIDE_NEURAL_FEEDBACK_TRAINING_PREPARATION'
summary=('사용자가 지도학습+가중치 미세조정한 피드백 신경망 제작을 승인했고, 이전 앱 목표를 삭제한 뒤 새 계속 목표를 active로 등록했다. '
 '공개4표현48영상(FIT36/DEV12),144관찰 중앙시점,13설명 연결 구간/11출처 문구를 확인했다. '
 'DTW로 옮긴 구간은 약한 학습 근거이며 독립 정답으로 평가하지 않는다. GPU로 SignRep 마지막 투영층 직전 특징을 준비하고 '
 '본체고정/마지막층 rank8 미세조정 각3seed 비교를 고정했다. 전체 본체 미세조정이나 성능 향상 완료가 아니다.')
next_action=('word100_learned_guide_feedback_20261102/PROJECTION_PROGRESS 및 실제 세션을 확인해 중복 실행하지 않는다. '
 '투영층 입력 준비 후 model.py/train.py의 마스크·목표누출·실제가중치 업데이트·DTW 축을 검증하고, '
 'FROZEN_TRAINING 조건의 6회 학습을 진행한다. 고정 DEV 공개 영상의 물리적 단계 근거는 예측을 보지 않고 확인한다. '
 '동일 검증을 반복하지 않고 저장 결과 검산→학습된 기준 DTW/문구 피드백→격리 데모로 이어간다. '
 '사용자/멘토 자료 학습·튜닝·선택 금지, C·T3-F/8804와 과거 동결 결과 보존. 새 외부 다운로드 없음.')
record=dict(status=status,at=now,summary=summary,next_action=next_action,
 unit=str(U),goal_registered=True,goal_status_at_registration='active',
 preparation=str(U/'PREPARED.json'),frozen=str(U/'FROZEN_TRAINING.json'),
 projection_exec_session=35188,model_training_started=False,user_mentor_used=False)
(U/'CURRENT_EXECUTION.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
p=O/'STATE.json';d=json.loads(p.read_text(encoding='utf-8-sig'));d['current_stage']=status;d['next_single_action']=next_action;d['updated_date_kst']=now[:10];d[U.name]=record
p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
p=R/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig')
for h,title,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','지도학습·실제 부분 미세조정 실행 착수',summary),('## 9. 다음 단일 작업','학습 입력·6회 비교·가이드 근거·실제 데모 순서로 실행',next_action)]:
 assert h in t
 t=t.replace(h,h+'\n\n### 2026-11-02 '+title+'\n\n'+body,1)
p.write_text(t,encoding='utf8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_action+
 '\n\n현재 단위: '+str(U)+'\nGPU 준비 exec35188, 완료/실패/진행 파일과 세션을 먼저 확인. 기존 서버 생존 확인은 아직 안 함. '+
 '새 목표 앱 active 등록 성공. 현재 실제 영상 모델은 기존 동결 SignRep이고 이번 부분미세조정은 fc.1의rank8 변화; '
 '새 큰모델/사적영상 사용 없음. 이전 단위 CURRENT_EXECUTION이 완료여도 새 단위를 완료로 오인하지 않는다.\n',encoding='utf8')
with (R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt').open('a',encoding='utf8') as f:f.write('\n\n'+now+' — 학습된 가이드 피드백 신경망 실행 승인\n'+summary+'\n'+next_action+'\n')
print(status)
