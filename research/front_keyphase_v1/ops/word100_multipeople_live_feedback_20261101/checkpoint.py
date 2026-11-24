from pathlib import Path
import json,datetime
U=Path(__file__).resolve().parent; O=U.parent; R=O.parents[2]
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
status='MULTIPEOPLE_LIVE_DEMO_VALIDATION_IN_PROGRESS'
summary='사용자 승인으로 공개 4표현 FIT ID01–09/DEV ID11–13의 새 개발 실험과 실제 새촬영 데모를 준비 중이다. 기존 3인 180센터와 추가 9인 252센터 조건을 각각 3seed 학습 완료. 36개 DEV센터의 지원 항목 평균은 98.7586%→98.4039%로 개선 미확인, 보호 기존 판정부 유지. 전체 수어/피드백 정확도가 아니다. 저장검산22870 PASS. 48공개 입력, 직접 RGB 라벨144센터(DEV36 전부 별도 확인), 9인 참고 비교·새업로드 서버8802 구현 및 최종검증 진행. 실제 공개 원본 4.7초 영상 냉시작 분석19.42초, 서버 초기화 재사용 후 실측은 아직. 사용자 영상은 격리 추론 전용, C/T3F·운영 보존.'
next_action='최종 비교 엔진 검산 후 참고은행36개와 FIT 한사람 제외 범위 산출, DEV12/대조36/속도·누락 기술시험, 실제 공개 업로드·화면 검증, 보고·문서 동기화. 완료 학습 재실행X. 새 9인 후보 미채택. 이전3인 평가역할을 새 프로토콜에서 명시적으로 종료했으며 과거 실험은 보존; 최종 독립시험으로 주장X.'
context=R/'WordsInSilence_PROJECT_CONTEXT.md'
t=context.read_text(encoding='utf-8-sig')
title='### 2026-11-01 다인 학습 비교 완료·실제 새촬영 데모 검증 진행'
if title not in t:
 t=t.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+title+'\n'+summary,1)
 t=t.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n### 2026-11-01 다음 단일 작업 — 9인 참고와 새촬영 연결 검증\n'+next_action,1)
 context.write_text(t,encoding='utf-8')
s=O/'STATE.json';d=json.loads(s.read_text(encoding='utf-8-sig'));d['current_stage']=status;d['next_single_action']=next_action
d[U.name]=dict(status=status,updated_at=now,summary=summary,next_action=next_action,unit=str(U),training_complete=True,candidate_adopted=False,production_changed=False,user_training=False)
s.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n'+next_action+'\n\n현재 학습/입력준비 프로세스 종료. 8801은 기존 기록재생서버이며 새8802 실제서버는 아직 미시작. 현재 단위 engine/build_bank/server 최종검토 담당 agent 진행 여부를 먼저 확인. 최신 evidence는 해당 단위 FROZEN/TRAINING_REPORT/review/SAVED_TRAINING_VERIFICATION. 학습 완료결과 재추론X, 기준 문서 과거8801단계 지시로 현재새업로드 승인 되돌리기X. 최종 문서/성과보고 미완료.\n',encoding='utf8')
print(status)
