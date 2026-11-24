"""Close the completed experiment from saved evidence; never run a model."""
from common import *
import re

assert not (U/'FINAL_AUDIT.json').exists()
ctx=read(U/'TRAIN_CONTEXT.json');ready=read(U/'READY.json')
v=read(U/'COMPARISON_INDEPENDENT.json');d=read(U/'DIAGNOSTIC_INDEPENDENT.json')
a=read(U/'ANALYSIS.json');c=read(U/'CONCLUSION.json');n=read(U/'NEXT_PLAN.json')
execution=read(U/'CURRENT_EXECUTION.json');exit_record=read(U/'EXIT.json')
assert ready['context_sha256']==sha(U/'TRAIN_CONTEXT.json')
assert ready['preflight_sha256']==sha(U/'PREFLIGHT.json')
assert v['status']=='PASS_SAVED_SCORE_TRACE_AND_FROZEN_T3F_VERIFICATION'
assert v['code_sha256']==sha(U/'verify.py')
assert a['comparison_sha256']==sha(U/'COMPARISON_INDEPENDENT.json')
assert a['diagnostic_sha256']==sha(U/'DIAGNOSTIC.json')
assert d['status']=='PASS' and d['diagnostic_sha256']==sha(U/'DIAGNOSTIC.json')
assert d['verifier_sha256']==sha(U/'verify_diagnostic.py')
assert execution['status']=='COMPLETE' and not execution['running']
assert exit_record['exit_code']==0 and exit_record['runs']==3
assert not (U/'TRAIN_RUNNING.json').exists()
assert not c['gate_pass'] and not n['automatic_training'] and not n['automatic_adoption']
assert c['epochs']==[13,78,15] and c['selected_epochs']==[1,66,3]
assert a['actual_batches']==3074 and a['actual_source_exposures']==95294
assert v['saved_predictions_verified']==9000
assert v['auxiliary_verification']['extra_saved_predictions_verified']==3600
assert read(U/'LAYOUT_QA.json')['status']=='PASS'
check(ctx['pins']);check(v['pins'])
for seed in SEEDS:
    r=read(U/f'{ARM}_{seed}_RESULT.json')
    assert r['checkpoint_sha256']==sha(U/f'{ARM}_{seed}_best.pt')
for p in U.glob('*.json'):read(p)
page=(U/'RESULT_KO.html').read_text(encoding='utf8')
for href in re.findall(r'href="([^"]+)"',page):
    if not href.startswith(('https:','http:','#')):assert (U/href).exists(),href

stamp=now();key=U.name
heading='### 2026-10-19 다중 시점 보조 3회 비교 완료 — 소폭 개선·새 회귀로 미채택'
next_action='이번 다중 시점 후보의 비교 결과를 사용자와 검토한다. 기존 C·T3-F·웹·DTW는 유지한다. 다음 후보 학습 전에 기존 공개100단어 감사 근거로 관측 가능한 동작 단계·순서·정상 반복·가림을 구분하는 평가 기준부터 설계한다. 필수 수어 단계·생략 오류의 정답은 수어 검토 없이 단정하지 않는다. 추가 학습·계수/epoch 탐색·운영 교체 자동 실행 없음. 사용자·멘토 촬영은 실사용 시험 전용으로 학습·튜닝·후보선택에 사용하지 않는다.'
body=f'''{heading}
갱신 {stamp}. ops/{key}. 사용자 “그 보완방법대로 진행해줘”로 승인한 구현·사전검증·3seed학습·기존모델비교·별도검산·보고 완료. 실행27200 종료0,진단16413 종료0,진행중없음.
- 변경: 기존 T3-F47236개 전체 동결, GRU16 시간별 기억을 기존 elapsed 삼각 앞·중간·뒤로 읽어48→100. 새7200개(직전4000보다3200증가), 총54436. 입력280·ROI·마스크·시간·CE(b+a)+.1CE(a)·일정·선택 기준 유지. 구조와 용량 효과 완전 분리 불가. 단어별 예외 없음.
- 공개 TRAIN899(ID01~09)/DEV300(ID11~13), 같은300으로3회 개발 비교. sensor C98.67%/T3-F99.22%/직전AUX99.22%/이번99.33%(296/300,300/300,298/300). T3-F대비+0.11%p이나1/3회만 개선. masked 이번99.00%(295,299,297),TF대비0.00%p/직전AUX대비-0.11%p. top5두조건99.89%로TF와동일. 실제웹캠·새사람·모든수어 성능 아님.
- 2회차 별→고양 혼동을 고쳤으나 masked의 기존 성남시수어통역센터 정답이 전주시수어통역센터로 바뀜. 3회중2회개선/촬영자13/other집합 보존 기준 미달로 미채택. 직전AUX의3회차masked속다 개선도 유지 안됨. sensor 잔여오답5고유영상/6회:속다·동대문·상처·백육십이·남해군지부.
- 보조 단독 sensor정확도 직전21/27.67/45.33%→80.67/98.67/92.33%. 고정TRAIN진단의 끝1/3만 보여줄 때 같은답 유지78.64~88.88%→32.70~43.60%로 후반 의존 완화 근거. 반면 세구간역순 유지8.12~18.91%→37.71~49.83%로 모든 순서 변화에 더 민감해진 것은 아님. 가공특징 문맥/선택epoch/용량 차이와 언어적 정답 부재 때문에 순서 이해나 인과 단정 금지.
- 선택epoch1/66/3,종료13/78/15,106epoch·3074실제배치·95294원본노출;학습시간합계116.99초(구현·검산·보고제외). 61~80전체창은없어 후반안정성 전반 개선 판정 불가.
- 초기10788출력exact·NumPy사전GRU풀링1.62e-7·패딩배치3.82e-6,저장9000예측+base/aux3600·실제배치ID·동결tensor·선택epoch검산PASS. 별도훈련된GRU진단240출력 NumPy최대3.08e-6. 같은Codex별도산술검산이며독립agent/전문가검토아님. 준비도구4오류 근거보존·학습전복구,실제학습오류없음.
- C/T3-F·웹·DTW·원본/과거동결자료 보존. 사용자/멘토촬영읽기·학습·튜닝·선택0,ID14~18원시접근0. 자동추가학습없음. RESULT_KO.html/txt,CONCLUSION,NEXT_PLAN,COMPARISON_INDEPENDENT,DIAGNOSTIC_INDEPENDENT,FINAL_AUDIT.

'''
docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
before_close={str(p):sha(p) for p in docs}
text=docs[0].read_text(encoding='utf8')
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
assert h8 in text and h9 in text and heading not in text
text=text.replace(h8,h8+'\n'+body,1).replace(h9,h9+'\n'+heading+'\n'+next_action+'\n\n',1)
state=read(docs[2]);assert state['current_stage']=='WORD100_MULTI_POSITION_AUX_TRAINING'
state.update(current_stage='WORD100_MULTI_POSITION_AUX_COMPLETE_NOT_PROMOTED',updated_at=stamp,updated_date_kst=stamp[:10],next_single_action=next_action)
state[key].update(status=c['status'],at=stamp,report_pending=False,running=False,exit_code=0,running_processes=[],
    report=str(U/'RESULT_KO.html'),next_plan=str(U/'NEXT_PLAN.json'),automatic_training=False,automatic_adoption=False,
    sensor_mean=c['sensor_mean'],masked_mean=c['masked_mean'],sensor_delta_vs_T3F_pp=c['sensor_delta_vs_T3F_pp'],
    masked_delta_vs_T3F_pp=0.0,masked_delta_vs_AUX01_pp=c['masked_delta_vs_AUX01_pp'],
    gate_pass=False,failed_checks=c['failed_checks'],completed_epochs=c['epochs'],selected_epochs=c['selected_epochs'],
    actual_batches=a['actual_batches'],actual_source_exposures=a['actual_source_exposures'],
    predictions_verified=9000,extra_components_verified=3600,numpy_diagnostic_outputs=240,
    user_recordings_read=0,heldout_raw_read=0,production_changed=False)
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

현재 목표: 앞·중간·뒤 기억을 함께 읽는 보조 후보 F-G16-M3의 승인된3회학습·기존C/T3-F비교 완료. 소폭 개선이지만 채택 기준 미달. 기존 모델·웹은 그대로다.

- 현재 단위 research/front_keyphase_v1/ops/{key}. AUTHORIZATION/FROZEN_PROTOCOL/TRAIN_CONTEXT/READY/PREFLIGHT/EVALUATION_EVIDENCE/CONCLUSION/NEXT_PLAN/RESULT_KO.html/txt/FINAL_AUDIT.
- TRAIN_CONTEXT SHA {sha(U/'TRAIN_CONTEXT.json')}. 공개100단어 TRAIN899(ID01~09),DEV300(ID11~13);3seed20261013/14/15. 같은 개발 자료를 반복 사용했으므로 새사람 또는 실사용 정확도가 아니다. ID14~18원시접근0,사용자/멘토촬영접근0.
- 기존T3-F전체47236동결. 새GRU16의 모든 실제시간 hidden을 elapsed삼각앞중간뒤로 읽어48→100zerohead. 새7200/총54436. 직전보다출력부3200개증가하므로구조/용량인과완전분리불가. 입력280/손·ROI/시간/미검출/손실CE(b+a)+.1CE(a)/일정/선택 유지,단어별예외없음.
- sensor평균 C98.67%,T3F99.22%,직전AUX99.22%,M3 99.33%. M3각회296/300,300/300,298/300. TF대비+0.11%p이고1/3회만개선. masked M3 99.00%(295,299,297),TF99.00%와동일/직전AUX99.11%보다-0.11%p. 상위5평균99.89%두조건으로TF동일.
- 별을2회차양조건에서고쳤지만 masked 성남시수어통역센터가전주시수어통역센터로새오답. 촬영자13/other집합회귀·2/3회개선실패로미채택. 직전AUX의3회차masked속다개선도유지못함. sensor잔여5영상/6회오답은속다·동대문·상처·백육십이·남해군지부.
- 보조단독sensor정확도80.67/98.67/92.33%(직전21/27.67/45.33%). 마지막1/3만보여줄때같은보조답유지32.70~43.60%(직전78.64~88.88%)로후반의존완화근거. 앞2구간교환시72.30~93.99%(직전90.43~94.66%). 그러나세구간역순유지37.71~49.83%(직전8.12~18.91%)이므로모든순서반응개선아님. TRAIN진단/TCN문맥/모델용량·선택차이/정상수어정답없음 주의.
- 선택1/66/3,종료13/78/15. 실제106epoch·3074배치·95294원본노출. 학습만116.99초. 61~80전체창없어후반안정성일반개선주장금지.
- 검증완료: 준비10788초기exact/NumPy1.62e-7/패딩3.82e-6;저장9000최종예측+base/aux3600/실제배치ID/동결tensor/선택검산;별도훈련GRU240출력NumPy3.08e-6. 동일Codex별도계산이며독립agent나수어전문가검토아님. 결과재채점·재학습금지.
- 준비도구4오류(LFCRLF,tuple/list,누락import,float32순열exact)를 FIRST_PREP_ERROR/SECOND_PREP_ERROR/PREP_ERRORS/RECOVERY/NUMERIC_RELABELLING_DIAGNOSIS/first_attempt에보존. 학습전검사도구만수정. 모델·계수·분할변경없음,실제학습오류없음.
- 실행 prepare17823/train27200(PID41116)/diagnostic16413 모두종료0. CURRENT_EXECUTION COMPLETE/runningfalse,TRAIN_RUNNING없음. 진행중없음. 기존완료검산재실행불필요.
- C/T3-F/웹/DTW/W:/sources/과거동결단위보존. Git·카메라·다운로드·외부프로젝트전송없음. 개인자료는실사용테스터전용으로학습/튜닝/후보선택금지.

다음 한 작업: {next_action}
'''
for p,h in before_close.items():assert sha(p)==h,'Concurrent document change'
docs[0].write_text(text,encoding='utf8');docs[1].write_text(handoff,encoding='utf8');write(docs[2],state)
assert all(key in p.read_text(encoding='utf8') for p in docs)
check(ctx['pins']);check(v['pins'])
write(U/'FINAL_AUDIT.json',{
    'status':'PASS_COMPLETE_VERIFIED_NOT_PROMOTED','at':stamp,'runs':3,'training_exit_code':0,
    'running':False,'training_context_sha256':sha(U/'TRAIN_CONTEXT.json'),
    'protected_hash_count':len(ctx['pins']),'saved_predictions_verified':9000,'extra_components_verified':3600,
    'actual_batches':3074,'actual_source_exposures':95294,'diagnostic_numpy_outputs':240,
    'verification':'Same Codex, separate saved-score arithmetic, actual batch traces, frozen tensors and NumPy GRU. No independent agent/expert.',
    'layout_and_visual_review':'PASS desktop1280/mobile390 no overflow or script errors; desktop top and comparison screenshots inspected.',
    'first_errors_preserved':['FIRST_PREP_ERROR.json','SECOND_PREP_ERROR.json','PREP_ERRORS.jsonl','RECOVERY.json','NUMERIC_RELABELLING_DIAGNOSIS.json','first_attempt'],
    'training_error':False,'user_recordings_read':0,'heldout_raw_read':0,'existing_models_preserved':True,
    'production_changed':False,'automatic_further_training':False,'adopted':False,
    'docs_before_close':before_close,'docs_synced':{str(p):sha(p) for p in docs},
    'files':{p.name:sha(p) for p in U.iterdir() if p.is_file()}
},new=True)
print('PASS: three-run comparison complete; sources preserved; documents synced; no adoption or further training.')
