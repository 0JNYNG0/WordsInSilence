from common import *
assert read(U/'READY.json')['training_allowed']
current=read(U/'CURRENT_EXECUTION.json');current['exec_session_id']=65626;current['prepare_session_id']=52949
write(U/'CURRENT_EXECUTION.json',current)
stamp=now();key=U.name
heading='### 2026-10-19 승인된 보조 단어 학습 신호 3회 비교 진행 — 기존 C/T3-F 보존'
next_action=f'ops/{key}의 exec65626/상태/잠금을 먼저 확인해 중복 없이 이어받고, 3회 완료 후 verify.py로 저장 점수·실제 배치ID·고정tensor·보조점수 합산을 검산한다. lambda0.1/입력/초기값/선택 기준 변경 없음. 사용자·멘토 자료는 테스터 자료로 학습·튜닝·선택 금지. 단어 공통 규칙이나 검증은100단어 범위; 운영 교체 없음.'
body=f'''{heading}
갱신 {stamp}. 사용자 “계획 진행 순서에 맞게 모두 작업 진행” 승인으로 직전 계획 대기 해소. AUTHORIZATION/FROZEN_PROTOCOL/TRAIN_CONTEXT/READY 고정. 모델 본체는 기존 검증된 frozen_order_aux.py 그대로이며 별도 objective CE(b+a,y)+0.1 CE(a,y). T3-F47236·BN고정/보조4000만. 공개899TRAIN/300DEV·seed20261013/14/15·같은C280입력·스케줄/batch32/AdamW.001/wd.001/clip1/max80/patience12/최종2view CE선택. preflight exec52949종료0, 이전초기해시3개동일/수식오차<9e-16/단어번호순열불변·7/100/137class공통loss검증/부품gradient전달검증PASS,사전optimizer0. 현재exec65626 PID{current['pid']}, 중복실행금지. 새평가결론아직없음. 출력100class는이번검증범위이며나머지수어확대는자료/출력종류/학습/검증이추가필요;범용설계와범용성입증구분. 사용자실사용테스터자료접근0. 기존C/T3F/FG16·웹·DTW보존,원본수정없음. 완료후저장점수/aux분해/실제trace독립산술검산·보고. 구현/검산모두Codex별도코드,독립agent아님.

'''
ctxpath=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';context=ctxpath.read_text(encoding='utf8')
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
assert heading not in context and h8 in context and h9 in context
context=context.replace(h8,h8+'\n'+body,1).replace(h9,h9+'\n'+heading+'\n'+next_action+'\n\n',1)
ctxpath.write_text(context,encoding='utf8')
state=read(O/'STATE.json');assert state['current_stage']=='WORD100_AUXILIARY_SUPERVISION_PLAN_REVIEWED_NO_TRAINING'
state.update(current_stage='WORD100_AUXILIARY_SUPERVISION_TRAINING',updated_date_kst=stamp[:10],next_single_action=next_action)
state[key]={'status':'APPROVED_THREE_RUNS_IN_PROGRESS','at':stamp,'exec_session_id':65626,'pid':current['pid'],'training_authorized':True,'context_sha256':sha(U/'TRAIN_CONTEXT.json'),'lambda_aux':0.1,'train':899,'dev':300,'user_data_used':False,'protected_models_unchanged':True,'no_word_specific_rules':True}
write(O/'STATE.json',state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

현재 목표: 승인된 F-G16-AUX01 보조 단어 학습 신호의 3회 비교를 완료한다. 기존모델보존·범용설계·사용자실사용테스터분리를유지한다.

- 단위 research/front_keyphase_v1/ops/{key}. AUTHORIZATION/FROZEN_PROTOCOL/TRAIN_CONTEXT/READY/PREFLIGHT 참조. 직전계획의대기는이번사용자승인으로해소됨.
- 이전검증모델구조사용,새objective만 CE(b+a,y)+.1CE(a,y). 각seed원래T3F47236/BN고정,새GRU16+head4000. 이전FG16이어학습아님,정확히동일초기해시. lambda·모델·입력·스케줄변경금지.
- TRAIN899/DEV300·100단어·9명/3명·seeds20261013/14/15·batch32·AdamWlr.001/wd.001·clip1·max80/patience12·최종combined양viewCE선택. 보조단독성적은진단만. source채널동결재사용,사용자자료읽기0/ID14~18원시접근0.
- prepare52949종료0. 학습exec65626/PID{current['pid']}. CURRENT_EXECUTION/STATUS/TRAIN_RUNNING/training.log와실제exec확인후이어받기. PID없음만으로잠금지우지않음. 완료면재시작/재채점금지.
- verify.py는 C/T3F/FG16/새후보저장점수7200산술+새aux/base3600분해검산,실제배치ID/입력일정/가중치고정/optimizer4000/선택epoch검사. 최종재추론없음. 동일Codex별도검산,독립agent아님.
- 범용목표: loss는단어명/ID/개인별예외없고class순열/크기7/100/137합성검사PASS. 실제모델출력100개이며모든수어성능입증아님;추가단어학습은새자료·출력범위·검증필요.
- 사용자·멘토는실사용테스터. 자료읽기/학습/튜닝/계수/모델선택금지. C/T3F/FG16와웹DTW/원본불변,Git/외부전송/카메라/다운로드없음.
- 최종보고는평균/범위/%p/각seed회귀/사람·미관측집합/보조성적과최종성적구분. 개선없어도최초조건유지후보고,재탐색/운영교체자동진행금지.

다음 한 작업: {next_action}
''',encoding='utf8')
print('Recorded authorization and actual run session65626.')
