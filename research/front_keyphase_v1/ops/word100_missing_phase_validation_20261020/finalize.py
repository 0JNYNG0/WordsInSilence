from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
U=Path(__file__).resolve().parent;OPS=U.parent;B=OPS.parent;ROOT=B.parents[1];P=OPS/'word100_metric_evidence_20261020';O=OPS/'word100_multiref_observation_compare_20261019'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
f=read(U/'FROZEN.json');s=read(U/'SUMMARY.json');v=read(U/'VERIFICATION.json');qa=read(U/'BROWSER_QA.json');stamp=datetime.now(timezone(timedelta(hours=9))).isoformat()
assert not (U/'FINAL_AUDIT.json').exists() and not (U/'RUNNING.json').exists()
assert v['status']==qa['status']==read(U/'SELFTEST.json')['status']=='PASS'
assert sha(U/'audit.py')==f['runner_sha256'] and sha(P/'candidate.py')==f['candidate_sha256'] and sha(O/'FROZEN.json')==f['old_frozen_sha256']
for p,h in f['protected'].items():assert sha(p)==h
for name,h in f['previous_artifact_sha256'].items():assert sha(P/name)==h
write(U/'VISUAL_REVIEW.json',{'status':'PASS','inspected':['QA_DESKTOP.png','QA_RESULTS.png','visual/self_mask.png','visual/real_gap.png'],'notes':'Chart labels, counts, observation-mask break, actual-source witness timeline visually inspected. No new RGB review or semantic phase labels claimed. Mobile overflow checked by browser.','reviewer':'Codex'})
execution=read(U/'CURRENT_EXECUTION.json');execution.update(run_exit_code=0,verify_exit_code=0,active_processes=[],status='COMPLETE');write(U/'CURRENT_EXECUTION.json',execution)
write(U/'STATUS.json',{'status':'COMPLETE_VALIDATION_NOT_DEPLOYED','at':stamp})
artifact_names=['FROZEN.json','SELFTEST.json','SUMMARY.json','ROWS.json','ACTUAL_GAPS.json','SELF_CONTROLS.json','INPUT_PINS.json','VERIFICATION.json','VISUAL_REVIEW.json','BROWSER_QA.json','RESULT_KO.html','RESULT_KO.txt','audit.py','verify.py','report.py']
final={'status':'PASS','at':stamp,'protected_files_unchanged':len(f['protected']),'previous_artifacts_unchanged':True,'implementation':'Codex local','verification':'Same operator, separate scalar reproduction; not a separate agent','scope':s['scope'],'new_training':0,'classifier_inference':0,'private_or_mentor_recordings_read':0,'heldout_14_18_inputs_read':0,'production_changes':0,'semantic_phase_labels_added':0,'artifact_sha256':{n:sha(U/n) for n in artifact_names}}
write(U/'FINAL_AUDIT.json',final)
next_action='사용자와 전체손·짧은구간 관측누락 검증 결과 확인. 다음은 연구용 항목별 실제시간 관측지도/연속미관측 표시와 정상속도·반복·손내림 대조 검증 계획. 센서미관측을 수행오류로 단정하지 않는다. C/T3-F·운영DTW·웹 유지, 인식보조·학습은 별도 확인 전 금지.'
entry={'status':'COMPLETE_VALIDATION_NOT_DEPLOYED','at':stamp,'unit':str(U),'scope':s['scope'],'whole_hand_unavailable_all_metrics':591,'short_peak10':s['conditions']['peak10']['metrics'],'actual_gap':s['actual_gap'],'self_control':s['self_control'],'verification':v,'protected_files_unchanged':29,'new_training':0,'private_data_read':0,'production_changes':0,'report':str(U/'RESULT_KO.html'),'execution':'run5703/verify70073 exit0; no active processes','next_single_action':next_action}
state=read(OPS/'STATE.json');state.update(current_stage='WORD100_MISSING_PHASE_VALIDATED_REVIEW',updated_date_kst='2026-10-20',updated_at=stamp,activity_state='AWAITING_USER_REVIEW',next_single_action=next_action,resume_command=next_action);state['word100_missing_phase_validation_20261020']=entry;write(OPS/'STATE.json',state)
title='### 2026-10-20 전체손·짧은구간 관측누락 검증 완료 — 낮은 차이와 전체수행 확인은 별개'
section=f'''{title}
갱신 {stamp}. ops/word100_missing_phase_validation_20261020. 사용자 승인 공개100단어 관측누락 검증 완료. 기존 C/T3-F/웹/운영DTW 유지. TRAIN899참고/DEV300질의,591손단위(기존역할9제외),sensor경로·기존DTW정렬 고정. 조건은 FROZEN에서 선고정; 새 원래결과 재채점/정렬/학습 없음.
- 전체손 관측제거591단위 모두 굽힘/위치/궤적 None,0차이로대체없음. 짧은 중앙10%/최대손목이동10%·20% 관측제거. 최대이동구간은필수구별단계정답아님. 모든단어공통규칙; 실제동작생략라벨없음.
- peak10(주석구간10%,약0.058~0.337초): 관측실제제거·전후수치가능 굽힘424중195/위치445중215/궤적445중220에서차이가감소. 남은부분평균/유효참고구성영향이며개선아님. 자기대조에서관측제거후15각도모두0인435손단위/위치0인455손단위; 실제정확도아님.
- 실제저장관측: 주석구간원본영역내손근거&sensor손점없음355손프레임/132손단위,그중굽힘전체수치104/위치115계속가능. 원본도MediaPipe추정,미관측원인·필수단계정답아님. 정상내림별도구분/자동오답없음.
- 자체8검사PASS. 같은Codex별도scalar검산:1199입력해시/200단어파일/2364조건손단위/21252쌍조건/42552출력/591자기대조PASS. 최대수치오차1.43e-14. 보호29/이전결과불변,보고3그림·화면QA PASS. 다른agent/전문가검증아님.
- run5703/verify70073 종료0,잠금없음. 추가학습0/인식보조0/개인·멘토자료읽기0/보류14~18입력읽기0/운영변경0. RESULT_KO.html/txt,SUMMARY,VERIFICATION,FINAL_AUDIT. 결과확인후실제시간관측지도·연속미관측표시및정상수행대조검토. 실제필수동작생략·인식률개선·100단어외일반성미검증.

'''
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=context.read_text('utf-8-sig');assert title not in txt
txt=txt.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+section,1)
txt=txt.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n'+title+'\n'+next_action+'\n\n',1);context.write_text(txt,'utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

현재 목표: 공개다중참고 DTW의 관측누락과 차이수치를 분리해 검증. 마지막 완료: 전체손·짧은구간 관측누락검증. 운영미적용.

- 단위 `{U}`. run5703/verify70073 종료0;활성작업/잠금없음. 완료계산 재실행불필요.
- 공개100단어TRAIN899참고/DEV300질의,591손단위,4가림조건. sensor관측·기존정렬고정. 전면미관측591모두3항목보류,0대체없음.
- peak10 짧은누락(주석동작10%,중앙0.1581초):전후측정가능굽힘424중195/위치445중215/궤적445중220에서차이오히려감소. 자기대조435손15각도전부0/455위치0. 못본부분을평가한게아님,성능향상아님.
- 실제원본영역내근거&sensor누락355손프레임/132손단위. 그중굽힘104/위치115는수치계속가능. 원본도추정/필수단계정답없음,정상내림별도,자동오답처리없음.
- 자체8/별도scalar2364행·21252쌍조건·42552출력·591자기대조검산PASS. 같은Codex별도코드,다른agent아님.1199입력/200단어파일해시검사,보호29·이전결과불변. 보고3그림+브라우저QA PASS.
- C/T3-F·운영DTW·웹유지,사용자멘토미접근,학습/인식보조0,보류14~18미접근. FROZEN/SUMMARY/VERIFICATION/FINAL_AUDIT/RESULT_KO.html 확인.
- 한계:기존정렬고정측정진단,새웹전체경로아님. 관측제거이지언어오류아님. 고이동은필수단계라벨아님. 일반인식향상/잘못된수어거절성능미검증.

다음 한 작업: {next_action}
''','utf-8')
assert read(OPS/'STATE.json')['word100_missing_phase_validation_20261020']['status']=='COMPLETE_VALIDATION_NOT_DEPLOYED'
assert title in context.read_text('utf-8') and next_action in (ROOT/'NEW_CHAT_HANDOFF.md').read_text('utf-8')
print('FINALIZED',stamp)
