from common import *
import re
assert not (U/'FINAL_AUDIT.json').exists()
ctx=read(U/'TRAIN_CONTEXT.json');check(ctx['pins'])
r=read(U/'COMPARISON_INDEPENDENT.json');a=read(U/'ANALYSIS.json');c=read(U/'CONCLUSION.json');plan=read(U/'NEXT_PLAN.json')
assert r['status'].startswith('PASS') and r['auxiliary_verification']['status']=='PASS'
assert r['code_sha256']==sha(U/'verify.py');check(r['pins'])
assert a['comparison_sha256']==sha(U/'COMPARISON_INDEPENDENT.json')
assert read(U/'EXIT.json')['exit_code']==0 and not (U/'TRAIN_RUNNING.json').exists()
current=read(U/'CURRENT_EXECUTION.json');assert not current['running'] and current['status']=='COMPLETE'
assert c['runs']==3 and c['epochs']==[13,13,20] and c['selected_epochs']==[1,1,8]
assert c['actual_batches_verified']==1334 and c['actual_source_exposures_verified']==41354
assert c['sensor_gain_events']==c['sensor_loss_events']==0 and c['masked_gain_events']==1 and c['masked_loss_events']==0
assert not c['gate_pass'] and not plan['automatic_further_training']
assert all(x['views'][v]['base_saved_logit_max_error']==0 for x in r['auxiliary_verification']['per_seed'] for v in VIEWS)
assert read(U/'LAYOUT_QA.json')['status']=='PASS'
page=(U/'RESULT_KO.html').read_text(encoding='utf8')
for href in re.findall(r'href="([^"]+)"',page):
 if not href.startswith(('http:','https:','#')):assert (U/href).exists(),href
for p in U.glob('*.json'):read(p)
stamp=now();key=U.name
next_action='승인된 F-G16-AUX01 구현·사전검사·3seed 학습·C/T3-F/FG16 비교·저장검산·보고 완료. 주sensor99.22%(TF대비+0.00%p),masked99.11%(+0.11%p,속다1영상1seed개선),채택기준미달→기존모델유지. 결과를 사용자와 검토하고 다음 단일 설계를 결정; 자동추가학습/계수재탐색/운영교체/heldout사용 없음. 모든 수어에 쓸 공통방법 목표이나 현재증거100단어 한정. 사용자·멘토는실사용테스터이며촬영읽기/학습/튜닝/후보선택금지.'
heading='### 2026-10-19 보조 단어 학습 신호 3회 비교 완료 — 주성능 동일, 보조조건만 소폭 개선'
body=f'''{heading}
갱신 {stamp}. ops/{key}. 사용자의 기존모델보존/범용보완/사용자실사용테스터 승인에 따라 F-G16-AUX01 3회 완료. 기존TF47236/BN고정·같은GRU16/head4000에 CE(b+a,y)+.1CE(a,y). TRAIN899/DEV300·동일초기3해시/입력280/관측·구간일정/batch32/AdamW.001/wd.001/clip1/max80/patience12/최종DEV2view CE선택. 종료13/13/20,선택1/1/8. 실제학습합계46.92초(준비검산보고제외). sensor정답296/299/298=99.22%범위98.67~99.67,T3F/FG16동일+0.00%p;masked295/299/298=99.11%,TF/FG16대비+0.11%p. C평균sensor98.67/masked98.44대비+0.56/+0.67%p이나주sensor이득은기존TF계승. sensor top5TF/FG/새모두99.89%. TF대비sensor새정답0/새오답0,masked3회차속다ID12한영상새정답1/새오답0. 사람이바뀐독립900영상이나모든수어검증아님. 사람·미관측집합주성능동일. sensor남은고유6영상/반복7오답:속다·동대문·상처·남해군지부·별·백육십이. 보조혼자도7건모두오답이고5건정답margin악화. 실제보조첫→마지막sensor정답률21→45.67/27.67→53/31.33→54.33%,선택시점21/27.67/45.33. 배움은확인됐으나기존오답을고칠상보정보부족;인과확정아님. 두번째마지막합친정답299→298,세번째마지막299지만CE더나빠미선택;최고정답epoch사후선택금지. 61~80후반창없음. 46epoch/1334실제배치/41354영상노출검산,최종저장7200+새base/aux3600예측산술/합산정확/teacher저장값0오차/고정tensor·optimizer4000·선택epoch PASS,최종재추론0. sameCodex별도코드검산,독립agent아님. preflight52949/train65626/verify종료0;오류없음/진행중없음. 새후보미채택,기존C/TF/FG/웹DTW해시보존. 사용자/멘토촬영읽기학습튜닝선택0/ID14~18원시접근0/Git·외부전송·카메라·다운로드0. 단어예외없는공통식(합성7/100/137class·순열검증),현재모델출력100;미학습수어확대는자료/출력종류/새검증필요. 다음은결과검토,계수epoch재탐색자동진행없음;공개단계/순서/생략평가기준및어휘확대자료분리계획검토제안. RESULT_KO.html/COMPARISON_INDEPENDENT/ANALYSIS/CONCLUSION/NEXT_PLAN/FINAL_AUDIT.

'''
docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json'];before={str(p):sha(p) for p in docs}
context=docs[0].read_text(encoding='utf8');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
assert h8 in context and h9 in context and heading not in context
context=context.replace(h8,h8+'\n'+body,1).replace(h9,h9+'\n'+heading+'\n'+next_action+'\n\n',1)
state=read(docs[2]);assert state['current_stage']=='WORD100_AUXILIARY_SUPERVISION_TRAINING'
state.update(current_stage='WORD100_AUXILIARY_SUPERVISION_COMPLETE_NOT_PROMOTED',updated_date_kst=stamp[:10],next_single_action=next_action)
state[key]={'status':'COMPLETE_VERIFIED_PRIMARY_NO_GAIN_KEEP_C_T3F','at':stamp,'unit':key,'candidate':ARM,'lambda_aux':.1,'runs':3,'train':899,'dev':300,'seeds':list(SEEDS),
 'epochs':[13,13,20],'selected_epochs':[1,1,8],'sensor_top1':[296,299,298],'masked_top1':[295,299,298],'sensor_mean_percent':c['sensor_mean'],'masked_mean_percent':c['masked_mean'],
 'delta_vs_T3F_sensor_pp':0.,'delta_vs_T3F_masked_pp':1/9,'sensor_new_correct':0,'sensor_new_wrong':0,'masked_new_correct':1,'masked_new_wrong':0,
 'gate_pass':False,'auxiliary_learning_confirmed':True,'no_word_specific_rules':True,'all_sign_language_validated':False,'user_role':'Real-world tester only; never training/tuning/model selection',
 'existing_models_web_DTW_preserved':True,'raw_user_recordings_read':0,'saved_final_predictions_verified':7200,'saved_components_verified':3600,'actual_batches':1334,'actual_source_exposures':41354,
 'running_processes':[],'training_session':65626,'exit_code':0,'report':str(U/'RESULT_KO.html'),'next_plan':str(U/'NEXT_PLAN.json'),'automatic_further_training':False}
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

현재 목표: 승인된 F-G16-AUX01 보조 분류 신호 구현·3회 학습·검산·보고 완료. 주검증 추가개선없어기존 C/T3-F 유지. 사용자 결과 검토 대기.

- 단위 research/front_keyphase_v1/ops/{key}. AUTHORIZATION/FROZEN_PROTOCOL/TRAIN_CONTEXT/READY/PREFLIGHT/COMPARISON_INDEPENDENT/ANALYSIS/CONCLUSION/NEXT_PLAN/FINAL_AUDIT.
- 원래TF47236/BN전체고정,새GRU16/head4000에CE(b+a,y)+.1CE(a,y). 기존P모델소스그대로,새objective만별도. 첫초기state3개FG16동일. 단어/사람예외없음. 입력/ROI/관측/배정/정규화/스케줄유지.
- 공개100단어TRAIN899(ID01~09)/DEV300(ID11~13),seed20261013/14/15,종료13/13/20회·선택1/1/8회. batch32/AdamWlr.001/wd.001/clip1/max80/patience12/최종2viewCE기준고정. 학습합계46.92초(전체작업시간아님).
- sensor C295/299/294(98.67%)→TF296/299/298(99.22%)→FG16동일→AUX동일. masked C294/299/293(98.44)→TF295/299/297(99.00)→FG동일→AUX295/299/298(99.11). 대TF주성능+0.00%p,보조조건+0.11%p. C대비주성능+0.56%p는기존TF계승으로새효과아님.
- TF대비sensor새정답0/새오답0,masked새정답1(속다/ID12/seed15)/새오답0. 주채택gate두조건실패(평균sensor상승/2seed상승). 새후보미채택. 회귀없음은선택모델의정답여부이며점수/학습중회귀없다는뜻아님.
- sensor남은6영상에서반복합7오답:속다→외상값,동대문→칠천/이집트,상처→믿다,남해군지부→수어,별→고양,백육십이→육백만. 보조단독도7건모두오답,5건정답margin더나빠짐. 사람·미관측집합주정답수TF동일.
- 보조단독sensor첫→마지막21→45.67/27.67→53/31.33→54.33%,선택시점21/27.67/45.33. 실제기울기전달/가중치갱신확인. 강한보조학습이상보적인정답을보장않음. 첫2run검증CE악화/두번째마지막합친정답299→298. 세번째마지막299지만CE더나빠선택안함. 사후최고top1선택금지. 61~80창없어후반안정성개선판정불가.
- 저장최종7200예측및새base/aux3600별도산술검산,합산exact/기존TF저장logits와0오차. 46epoch·1334실제batchID·41354영상노출/선택epoch·체크포인트·optimizer새4000·동결tensor검산PASS. 최종재추론0. sameCodex별도검산,독립agent아님.
- 범용목표는유지: 공통loss(class순열/합성7/100/137수식검증)이나현재출력100개/검증100단어. 모든수어일반화입증아님,나머지단어는공개자료/출력종류/학습/사람분리검증필요. 고유100class를사후늘려이번결과와혼합하지않음.
- 사용자는실사용테스터이며사용자·멘토촬영읽기/학습/튜닝/계수/선택금지. 이번전혀접근안함. ID14~16미사용299개유지/ID17~18원시접근없음. 웹·C·TF·FG·DTW·원본불변. Git/카메라/다운로드/외부전송없음.
- prepare52949종료0/train65626종료0/verify종료0. CURRENT_EXECUTION완료/TRAIN_RUNNING없음. 추가학습/계수재탐색/운영교체/heldout평가자동진행없음. 기존완료점수재채점금지.
- 다음제안은결과검토후공개구별단계/순서/생략평가기준및어휘확대자료분리계획. 같은DEV로계수epoch반복탐색하지않기. 단어라벨만으로필수동작완수능력주장불가;임의역순모두오류라벨금지.
- 보고서: {U/'RESULT_KO.html'}

다음 한 작업: {next_action}
'''
for p,h in before.items():assert sha(p)==h,'Concurrent doc edit'
docs[0].write_text(context,encoding='utf8');docs[1].write_text(handoff,encoding='utf8');write(docs[2],state)
assert all(key in p.read_text(encoding='utf8') for p in docs)
check(ctx['pins'])
write(U/'FINAL_AUDIT.json',{'status':'PASS_COMPLETE_COMPARISON_NOT_PROMOTED','at':stamp,'protected_files_verified':len(ctx['pins']),
 'training_runs':3,'training_exit_code':0,'prepare_exit_code':0,'verify_exit_code':0,'sessions':{'prepare':52949,'train':65626},
 'frozen_model_verified':True,'selected_score_and_trace_verification':True,'general_word_rule':True,'all_signs_performance_proven':False,
 'user_recordings_read':0,'production_changed':False,'gate_pass':False,'no_final_reinference':True,'running_processes':[],
 'docs_synced':{str(p):sha(p) for p in docs},'files':{p.name:sha(p) for p in U.iterdir() if p.is_file()},
 'verification':'Codex separate numerical reference, saved score/actual trace/frozen-state verifier, HTML render and visual review; no independent agent/expert.'},new=True)
print('PASS: completed3runs, verified and documented; existing models preserved and no automatic follow-up training.')
