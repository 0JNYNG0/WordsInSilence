from common import *
import shutil
assert read(U/'READY.json')['training_allowed']
lock=read(U/'TRAIN_RUNNING.json');cs=sha(U/'TRAIN_CONTEXT.json');assert lock['context_sha256']==cs
write(U/'CURRENT_EXECUTION.json',{'session_id':62419,'pid':lock['pid'],'context_sha256':cs,'at':now(),'command':'train.py run','process_evidence':'live exec62419 and TRAIN_RUNNING; CIM unavailable in sandbox, not interpreted as absent'},new=True)
docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
backup=U/'before';backup.mkdir(exist_ok=False)
for p in docs:shutil.copy2(p,backup/p.name)
stamp=now();key=U.name
next_action='승인된 F-G16 보조3seed 학습 exec62419를 이어받는다. TRAIN_RUNNING/CURRENT_EXECUTION/로그와실제세션을확인해중복실행금지. 완료후 verify.py로저장점수/실제배치추적/동결tensor를검산하고C·T3F비교·회귀/후반변동을보고한다. 기존C/T3F·웹·DTW보존,사용자자료학습/튜닝금지,추가재탐색·운영교체없음.'
heading='### 2026-10-19 사용자 승인 F-G16 순서 기억 보조3회 학습·비교 진행'
body=f'''{heading}
갱신 {stamp}. ops/{key}. 사용자 “그 계획대로 진행해줘”로 직전 구현검증의 학습전검토대기가 해소되어 3seed학습과비교승인. AUTHORIZATION/FROZEN_PROTOCOL/TRAIN_CONTEXT/READY 기록. 모델은검증된 P/frozen_order_aux.py를그대로읽어사용. T3-F각seed기존47236전체고정+GRU16출력4000만학습. TRAIN899/DEV300·seed20261013/14/15·80최대epoch·patience12·AdamW.001/wd.001/clip1·C동일입력일정유지. prepare exec30007종료0,run exec62419/PID{lock['pid']}진행. 신규결과없음. 기존preflight소스/입력해시·초기3state검산재사용,승인상태만새단위에기록하고과거동결단위불변. 입력TRAIN/DEV분리검증PASS. 최종결과전에조건변경없음. 진행상태는CURRENT_EXECUTION/STATUS/training.log. 검산기는verify.py 별도산술/실제batchID/optimizer·checkpoint·선택epoch연결검사. 사용자촬영접근0/ID14~18원시접근0/기존웹·모델교체없음.

'''
ctx=docs[0].read_text(encoding='utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n';assert h8 in ctx and h9 in ctx
ctx=ctx.replace(h8,h8+'\n'+body,1).replace(h9,h9+'\n'+heading+'\n'+next_action+'\n\n',1)
state=read(docs[2]);state.update(current_stage='WORD100_FROZEN_ORDER_AUX_TRAINING',updated_date_kst=stamp[:10],next_single_action=next_action)
state[key]={'status':'TRAINING_APPROVED_THREE_SEEDS','at':stamp,'unit':key,'session_id':62419,'pid':lock['pid'],'context_sha256':cs,'train':899,'dev':300,'training_authorized':True,'user_data_used':False,'C_T3F_preserved':True}
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

현재 목표: 사용자 “그 계획대로 진행해줘”에 따라 F-G16 보조 부분만3회학습하고기존C·T3-F와비교한다. 이전학습전대기는이번승인으로해소됨.

- 단위: research/front_keyphase_v1/ops/{key}
- AUTHORIZATION/FROZEN_PROTOCOL/TRAIN_CONTEXT/READY 동결. 검증된 ../word100_frozen_order_aux_preflight_20261019/frozen_order_aux.py 그대로사용. 기존T3-F47236전체·BN고정+GRU16/출력4000만학습. 입력/검출기/ROI/정규화조건불변.
- TRAIN899/DEV300,seed20261013/14/15,같은C샘플링일정,batch32,최대80/patience12,AdamWlr.001/wd.001/clip1,평균DEV두viewCE선택. 3개결과전부보고,최고seed만선택금지. 개인촬영0/ID14~18원시접근0. C/T3F/웹/DTW교체없음.
- prepare exec30007종료0. train exec62419 PID{lock['pid']}진행. CURRENT_EXECUTION.json/STATUS.json/training.log/TRAIN_RUNNING 먼저확인. CIM접근불가이므로PID열거없다는이유로잠금지우지않음. 진행중학습중복실행금지.
- 완료후 verify.py 실행: 기존C·T3F저장점수재사용,FG16저장점수재검산만(재추론없음),전배치899ID노출/일정/동결hash/optimizer새4000/선택checkpoint검산. 평균·범위/정확도%·%p/사람·미검출회귀/61~80후반변동보고. 일찍종료되면해당창없음을명시.
- 결과에맞춘재학습/조건조정금지. 최초오류보존. 검증이끝나면근거보고서+§8§9/STATE/HANDOFF동기화. 운영교체승인없음.

다음 한 작업: {next_action}
'''
docs[0].write_text(ctx,encoding='utf8');docs[1].write_text(handoff,encoding='utf8');write(docs[2],state)
print('Recorded authorized running session62419; previous summaries backed up')
