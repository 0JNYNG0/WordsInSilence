from common import *
assert read(U/'READY.json')['training_allowed']
assert not (U/'CURRENT_EXECUTION.json').exists() and not (U/'TRAIN_RUNNING.json').exists()
key=U.name;stamp=now();p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf8')
heading='### 2026-10-19 다중 시점 GRU 후보 사전검증 통과 — 승인된 3회 비교 시작'
msg=f'사용자 “그 보완방법대로 진행해줘”로 직전 계획 검토 대기 해소. ops/{key}. AUTHORIZATION/FROZEN_PROTOCOL/TRAIN_CONTEXT/READY 고정. T3-F47236 전체고정,GRU16시간별기억을기존elapsed3삼각가중치로읽고zero48→100,새7200/총54436. 같은GRU초기해시3개/입력280/CE(b+a)+.1CE(a)/기존899TRAIN·300DEV·3seed·일정·선택기준유지. 사전검증10788입력초기출력exact/NumPyGRU및풀링1.62e-7/패딩배치3.82e-6/gradient와동결검사PASS. 준비도구오류4회(LFCRLF/JSONtuple/list/누락import/float32순열exact검사)근거보존후수정,모델및학습조건변경없음. 현재READY,이어서train.py run3회→verify.py저장검산→보고. 사용자/멘토자료및ID14~18원시접근금지,C/TF/웹/DTW변경금지. 추가순서손실없음/사후계수탐색없음. 상태·잠금먼저확인,중복실행금지.'
assert heading not in text
for marker in ['## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 9. 다음 단일 작업\n']:
 assert marker in text;text=text.replace(marker,marker+'\n'+heading+'\n'+msg+'\n\n',1)
p.write_text(text,encoding='utf8')
state=read(O/'STATE.json');assert state['current_stage']=='WORD100_AUXILIARY_ORDER_DIAGNOSIS_COMPLETE_PLAN_REVIEW'
state.update(current_stage='WORD100_MULTI_POSITION_AUX_READY_APPROVED',updated_date_kst=stamp[:10],next_single_action=msg)
state[key]={'status':'READY_APPROVED_THREE_RUNS','at':stamp,'context_sha256':sha(U/'TRAIN_CONTEXT.json'),'unit':key,'train':899,'dev':300,'seeds':list(SEEDS),'trainable':7200,'input_changes':False,'user_data_used':False,'production_changed':False,'report_pending':True}
write(O/'STATE.json',state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'# WordsInSilence 최신 인수인계\n\n갱신 {stamp}\n\n{msg}\n\n- 현재단위 {U}\n- 핵심문서 AUTHORIZATION/FROZEN_PROTOCOL/TRAIN_CONTEXT/READY/PREFLIGHT/EVALUATION_EVIDENCE.\n- 실행할명령 train.py run. 이미시작됐으면 CURRENT_EXECUTION/TRAIN_RUNNING/STATUS/실제exec확인후이어받기.\n- 기존AUX01/C/TF점수재사용,완료run재추론금지. 공개899/300반복개발비교이며새사람일반화시험아님. 동일후보3seed이외추가학습금지.\n- 다음 비교 후 정확도%/%p·회귀·계산검산을보고하고3문서동기화. 사용자는실사용테스터로촬영접근/학습/튜닝/선택금지.\n',encoding='utf8')
print('Synced approved READY state.')
