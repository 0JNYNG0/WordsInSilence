from pathlib import Path
import json, datetime
U=Path(__file__).resolve().parent
R=U.parents[3]
name=U.name
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
ctx=R/'WordsInSilence_PROJECT_CONTEXT.md'
s=ctx.read_text(encoding='utf-8-sig')
clarify='''### 2026-11-01 사용자 확인: 목표 유지, 지원 단어 피드백의 우선순위 정정

최종 목표는 위의 설명·다중 참고 기반 피드백 그대로다. 게임이 제공하는 **알려진 연습 단어**의 여러 공개 참고와 실제 수행을 비교하고, 같은 지원 단어를 다른 사람이 했을 때의 중요한 차이·정상 변이·관측 부족을 주평가로 삼는다. 미학습 단어로 공통 특징이 전이되는 시험은 보조 진단이며 지원 단어 기능의 보편적 선결관문으로 삼지 않는다. 이전 동결 시험의 FAIL을 소급 PASS로 바꾸지 않는다.

목표 단어·설명은 참고와 중요한 특징을 선택하는 근거다. 실제 관측 좌표·손모양을 기대 정답으로 덮어쓰는 입력이 아니다. TCN 추측이 없어도 게임의 연습 단어로 비교를 시작할 수 있고, TCN 결과는 보조 정보로 보존한다. 원래 목적·기술 경로·우선순위 변화와 실제 성과/한계는 `Docs/PROJECT_DIRECTION_HISTORY_KO.txt`에 이어서 기록해 멘토/교수 설명 자료로 쓴다. 이 기록은 기존 보고서/실패를 대체하지 않는다.

'''
anchor='### 2026-10-28 사용자 확정: 현재 AI 피드백의 기준 목표'
assert anchor in s and '2026-11-01 사용자 확인: 목표 유지' not in s
s=s.replace(anchor,clarify+anchor,1)
s=s.replace('최종 갱신: 2026-10-28 KST','최종 갱신: 2026-11-01 KST',1)
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'
h9='## 9. 다음 단일 작업'
p8='''### 2026-11-01 목표 단어를 아는 피드백 연결 — 승인된 네 단계 실행 중

사용자 요청으로 목표/진행 변천사를 기록하고, 알려진 지원 단어의 다른 사람 비교를 우선하는 연결부를 실행한다. 단위 `word100_target_known_feedback_20261101`. 공개20단어 FIT60/DEV20 참고은행 검산 완료; 설명후보12단어, 실제 단계/자동정렬 지원4단어16영상. 저장된 원3손모양 전체시점/3seed 및 두 자동정렬을 재사용한다. 시험영상의 정답 주석을 정렬 입력으로 쓰지 않으며, 부분 근거와 미관측을 같은 분모에 남긴다. 신규학습/새모델추론/운영교체 없음. 계획 `PLAN_KO.txt`, 방법검토 `review/METHOD_PLAN.json`, 구현 `COMPARATOR_FROZEN.json`. 직전6학습 미채택 판정은 불변.

'''
p9='''### 2026-11-01 다음 단일 작업 — 알려진 연습 단어의 자동 비교 연결·검산 완료

`word100_target_known_feedback_20261101/PLAN_KO.txt` 네 단계를 완료한다. 이력기록→참고은행→기존 관측/자동정렬 연결→36고정쌍×2방법×3seed 기술/공개비교와 별도검산. 에이전트 semantic_feedback 구현 중, semantic_model 방법/코드검토, semantic_data_audit 이력/입력검산. 중복학습/완료점수 재추론 금지. 부분 관측의 이용 가능성과 교육적 정오를 구분해 결과를 기록하고 root가 §8§9/STATE/HANDOFF를 마감한다. 미학습단어 성능 향상을 제품의 선결관문으로 다시 끌어오지 않는다.

'''
assert h8 in s and h9 in s
s=s.replace(h8,h8+'\n\n'+p8.rstrip(),1).replace(h9,h9+'\n\n'+p9.rstrip(),1)
ctx.write_text(s,encoding='utf8')
statepath=U.parent/'STATE.json'
state=json.loads(statepath.read_text(encoding='utf-8-sig'))
state.update(updated_date_kst=now,current_stage='TARGET_KNOWN_FEEDBACK_INTEGRATION_RUNNING',next_single_action='Complete word100_target_known_feedback_20261101 reviewed four-step integration and independent verification; no new training/inference.')
state[name]=dict(status=state['current_stage'],at=now,unit=str(U),plan=str(U/'PLAN_KO.txt'),active='semantic_feedback implementation; semantic_model method/code review; semantic_data_audit history/input review',supported_words=20,fit_references=60,dev_queries=20,phase_supported_words=4,goal_unchanged=True,primary_evaluation='Known supported word, different person; unknown word supplementary',new_training=0,new_inference=0,production_changed=False,user_mentor_used=False)
statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(R/'NEW_CHAT_HANDOFF.md').write_text('''# WordsInSilence 최신 인수인계

현재 word100_target_known_feedback_20261101 실행 중. 최신 사용자: 목표/진행 변천사 기록 후 검토된 모든 계획을 바로 실행. 최종 한국수어 설명+다중 참고 기반 중요한 차이/정상 변이 피드백 목표는 유지. 이번 정정은 알려진 연습 단어를 다른 사람이 하는 시험을 주평가, 미학습 단어 전이를 보조 진단으로 둔 것. 직전 semantic_feature_cycle6학습/FAIL/부분범위 한계는 불변; 재실행하지 않는다.

PLAN_KO.txt 네 단계: 1이력기록 2공개20단어 FIT60/DEV20 참고은행 3저장 원3손모양 전체시점/3seed와두자동DTW 연결 4공개36쌍(FIT24/DEV12)과기술반례/별도검산. 설명후보12단어, 실제단계 지원4단어16영상;20전체동작이해완성 아님. query정답주석을 정렬에 쓰지 않고 사후검산만. 목표단어는 참고선택만, 관측 덮어쓰기 금지. 미관측이 분모를 줄이거나 다른 부분의 관측을 모두 버리게 하지 않는다. 신규7의 추가4는3시점출력이라 전체시점복사 금지.

root 기준문서/이력; semantic_feedback comparator구현; semantic_model 방법/코드/결과검토; semantic_data_audit 이력/입력/결과검산. REFERENCE_BANK독립1278검사PASS. COMPARATOR_FROZEN 제안root승인, 실행전코드/자동path/입력핀. 실제새추론/학습0. 진행에이전트/실행상태 확인 후 이어받고 중복실행X.

이력초안 근거 review/HISTORY_AUDIT.json. Docs/PROJECT_DIRECTION_HISTORY_KO.txt로root완성예정. 웹기본M0, C/T3F선택경로보존; 새SignRep/semantic후보오프라인. 사용자/멘토촬영접근·학습·튜닝·선택/새다운로드/Git/카메라/외부전송/운영교체X. HTML명확한성능개선없으면생략. 별도Codex검산≠독립수어전문가정답. 완료후RESULT_KO.txt와§8§9/STATE/HANDOFF동기화.
''',encoding='utf8')
print(json.dumps({'status':'RUNNING_DOCS_SYNCED','unit':name,'at':now}))
