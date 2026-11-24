from common import *
import re
assert not (U/'FINAL_AUDIT.json').exists()
ctx=read(U/'TRAIN_CONTEXT.json');check(ctx['pins'])
r=read(U/'COMPARISON_INDEPENDENT.json');a=read(U/'ANALYSIS.json');c=read(U/'CONCLUSION.json');plan=read(U/'NEXT_PLAN.json')
assert r['status'].startswith('PASS') and read(U/'LAYOUT_QA.json')['status']=='PASS'
assert not (U/'TRAIN_RUNNING.json').exists() and read(U/'EXIT.json')['exit_code']==0
assert not c['gate_pass'] and c['sensor_delta_vs_T3F_pp']==0
assert c['actual_batches']==1885 and c['actual_source_exposures']==58435
assert c['epochs']==[13,14,38] and c['selected_epochs']==[1,2,26]
assert a['source_comparison_sha256']==sha(U/'COMPARISON_INDEPENDENT.json')
check(r['pins'])
assert r['code_sha256']==sha(U/'verify.py')
for view in VIEWS:
 assert r['summary']['T3F'][view]['top1_counts']==r['summary']['FG16'][view]['top1_counts']
 assert all(not x['gains'] and not x['losses'] for x in r['paired']['T3F'][view])
assert all(not x['full_window'] and x['epochs_available']==0 for x in r['late_per_seed']['FG16'])
page=(U/'RESULT_KO.html').read_text(encoding='utf8')
for href in re.findall(r'(?:src|href)="([^"]+)"',page):
 if not href.startswith(('http:','https:','#')):assert (U/href).exists(),href
for p in U.glob('*.json'):read(p)
stamp=now();key=U.name
current=read(U/'CURRENT_EXECUTION.json');current.update(status='COMPLETE',exit_code=0,completed_at=stamp,running=False);write(U/'CURRENT_EXECUTION.json',current)
event(stage='COMPLETE_VERIFIED_NO_ADDITIONAL_ACCURACY_GAIN',runs=3,keep_existing_C_T3F=True)
docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json'];before={str(p):sha(p) for p in docs}
next_action=plan['next_single_proposal']
next_action+=' 이번 승인3회학습·비교는완료. F-G16은추가정확도개선없어미채택, C/T3-F·웹·DTW유지. 다음학습자동실행없음. 사용자·멘토자료학습/튜닝/후보선택금지,ID14~16미사용유지.'
heading='### 2026-10-19 F-G16 보조3회 학습·비교·검산 완료 — T3-F 대비 추가 정확도 개선 없음'
body=f'''{heading}
갱신 {stamp}. ops/{key}. 최신사용자 “그 계획대로 진행해줘”의3회학습비교완료. T3-F각seed기존47236변수·BN전체고정, GRU16+출력4000만 AdamW로학습. TRAIN899/DEV300·100단어·9명/3명·같은C입력일정/선택규칙유지. seed20261013/14/15각13/14/38epoch종료,선택1/2/26(12회개선없음조기종료). 실제학습합계63.03초(준비/검산/보고제외). 선택sensor정답C295/299/294→T3F296/299/298→FG16동일296/299/298;3seed평균98.67→99.22→99.22%,FG16범위98.67~99.67%. masked C294/299/293→TF295/299/297→FG16동일,평균98.44→99.00→99.00%. sensor상위5 C99.67/TF99.89/FG16 99.89%. FG16대TF+0.00%p,대C+0.56%p는기존TF개선계승이며새GRU효과아님. 대TF새정답0/새오답0양view;seed13동대문오답후보응접실→칠천변화있어출력완전동일주장금지. 사람·미관측집합정답수같음. sensor남은고유6영상/3seed합7오답:속다·동대문·상처·남해군지부·별·백육십이. 믿다/사육장은공개DEV각3영상×3seed이미9/9,실사용혼동해결근거아님. 평균sensor CE.035206→.034982소폭변화이나정확도상승아님;seed13 CE소폭악화. 최초TRAIN CE1.58e-5~3.09e-5와작은기울기는포화된기존점수위잔차분류신호약함가능성지지,인과확정아님/AdamW상작은기울기만으로학습안됨단정금지. 새head가중치와점수실제변함. 미래순서/완수라벨없어그능력검증안됨. 전실험61~80창없어후반안정성향상주장금지. 저장5400예측·1885실제배치·58435학습영상노출·스케줄·선택epoch/체크포인트·고정tensor·optimizer4000별도검산PASS,최종재추론0. 같은Codex별도코드검산,독립agent아님. 주채택조건sensor평균상승/2seed상승실패→FG16미채택기존유지. prepare30007/run62419/verify종료0,CIM권한불가를프로세스없음으로해석않고실제exec확인. 사용자자료읽기0/학습튜닝0/ID14~18원시접근0/운영교체0. RESULT_KO.html/COMPARISON_INDEPENDENT/ANALYSIS/CONCLUSION/NEXT_PLAN/FINAL_AUDIT. 다음은보조출력자체단어학습손실을주는설계검토제안,아직추가구현·학습없음.

'''
context=docs[0].read_text(encoding='utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
assert h8 in context and h9 in context and heading not in context
context=context.replace(h8,h8+'\n'+body,1).replace(h9,h9+'\n'+heading+'\n'+next_action+'\n\n',1)
state=read(docs[2]);assert state['current_stage']=='WORD100_FROZEN_ORDER_AUX_TRAINING'
state.update(current_stage='WORD100_FROZEN_ORDER_AUX_COMPLETE_NO_EXTRA_GAIN',updated_date_kst=stamp[:10],next_single_action=next_action)
state[key]={'status':'COMPLETE_VERIFIED_NO_ADDITIONAL_ACCURACY_GAIN_KEEP_C_T3F','at':stamp,'unit':key,'training_runs':3,'train':899,'dev':300,'seeds':list(SEEDS),'epochs':[13,14,38],'selected_epochs':[1,2,26],
 'sensor_top1_counts':[296,299,298],'sensor_mean_accuracy_percent':99.22222222222223,'sensor_range_percent':[98.66666666666667,99.66666666666667],
 'masked_top1_counts':[295,299,297],'masked_mean_accuracy_percent':99.0,'delta_vs_T3F_pp':0.0,'delta_vs_C_pp':.5555555555555555,
 'delta_C_attribution':'Inherited T3F gain, not new GRU improvement','newly_correct_vs_T3F':0,'newly_incorrect_vs_T3F':0,'wrong_label_swap_count':1,
 'frozen_T3F_preserved':True,'new_trainable_parameters':4000,'actual_batches_verified':1885,'actual_source_exposures_verified':58435,'saved_predictions_verified':5400,'late61_80_available':False,
 'gate_pass':False,'user_recordings_read':0,'user_data_used':False,'C_T3F_web_DTW_preserved':True,'new_blind_test':False,'running_processes':[],
 'report':str(U/'RESULT_KO.html'),'next_plan':str(U/'NEXT_PLAN.json'),'next_training_authorized':False}
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

현재 목표: 승인된 F-G16 순서 기억 보조3회 학습·기존C/T3-F비교·검산 완료. 추가 정확도 개선없음, 기존모델유지. 결과검토후다음설계결정.

- 단위: research/front_keyphase_v1/ops/{key}
- 사용자 “그 계획대로 진행해줘”로승인된3회완료. AUTHORIZATION/FROZEN_PROTOCOL/TRAIN_CONTEXT/READY동결,직전P단위frozen_order_aux.py그대로사용. T3-F기존47236변수/BN고정, GRU16/출력4000만학습. 입력/관절/ROI/정규화변경없음.
- TRAIN899(100단어/ID01~09)/DEV300(ID11~13),seed20261013/14/15. 같은C샘플링/AdamW.001,wdecay.001,clip1,batch32,max80,patience12,평균2view CE선택. 13/14/38epoch종료,선택1/2/26. 실제학습63.03초;준비/검산/보고시간제외.
- sensor정답C295/299/294(98.67%)→TF296/299/298(99.22%)→FG16동일296/299/298(99.22%,범위98.67~99.67). masked C294/299/293(98.44)→TF295/299/297(99.00)→FG16동일. sensor상위5 C99.67/TF99.89/FG16 99.89. 대TF+0.00%p. 대C+0.56%p는기존TF개선계승,새GRU효과아님.
- 대TF새로맞힌0/새로틀린0양view모든seed. 다만seed13sensor동대문은잘못고른후보가응접실→칠천으로변함;모든출력/점수동일주장금지. 사람별/미관측집합정답수동일. sensor고유6영상/반복합7오답:속다→외상값,동대문→칠천/이집트,상처→믿다,남해군지부→수어,별→고양,백육십이→육백만.
- 믿다/사육장은공개DEV각3영상×3seed모두C/TF/FG16 9/9라이번DEV로실사용혼동해결주장불가. 사용자촬영읽기0/재평가0/학습튜닝0.
- 초기TRAIN CE약1.6e-5~3.1e-5로이미아주작음;새가중치/점수는변했으나정답순위개선0. 기존확신높은점수와합친단어loss만으로보조학습신호약함가능성. 원인분리증명아님;AdamW상작은기울기만으로학습안됨단정금지. sensor평균CE.035206→.034982미세변화,1회차는악화. 정확도개선으로승격금지.
- 전run이61epoch전에종료되어예정후반61~80창없음. 후반흔들림줄었다고주장금지. 정해진조기종료,실패/중단/임의종료아님. 재학습/조건재탐색없음.
- verify.py 저장5400예측산술(최종재추론0)/전1885실제배치ID·58435영상노출·입력일정·checkpoint/선택epoch·동결tensor·optimizer새4000확인PASS. 같은Codex별도검산이며독립agent/전문가아님.
- 주검토gate(T3F대비)실패:평균sensor상승없고2seed상승없음. 나머지회귀조건통과. FG16미채택,C/T3F·웹·DTW불변. 추가학습자동진행금지.
- 다음제안NEXT_PLAN: 보조출력자체도단어를맞히게별도보조loss를주는설계검토. 계수/조건은다음학습전에고정하고결과검토후. 필수단계완수판정은별도전문가공개정상/생략/순서오류근거필요. 모델크기/epoch늘리기자동확대없음.
- ID14~16정상299개미사용유지,ID17/18기평가재사용을새독립성능으로부르지않음. 이번ID14~18원시접근0. 사용자·멘토자료선택/학습금지. Git/외부전송/다운로드/카메라없음.
- 실행prepare30007=0/train62419=0/verify=0. TRAIN_RUNNING없음,CURRENT_EXECUTION완료,진행중없음. 결과HTML데스크톱/모바일ChromeQA PASS. 원본/과거완료단위해시보존.
- 보고서: {U/'RESULT_KO.html'}

다음 한 작업: {next_action}
'''
for p,h in before.items():assert sha(p)==h,'Concurrent documentation changes'
docs[0].write_text(context,encoding='utf8');docs[1].write_text(handoff,encoding='utf8');write(docs[2],state)
assert all(key in p.read_text(encoding='utf8') for p in docs)
assert read(O/'STATE.json')[key]['training_runs']==3
out={'status':'PASS','at':stamp,'training_complete':True,'comparison_verified':True,'engineering_checks_pass':True,'accuracy_progression_gate_pass':False,
 'protected_pins':len(ctx['pins']),'files':{p.name:sha(p) for p in U.iterdir() if p.is_file()},'docs_synced':{str(p):sha(p) for p in docs},
 'raw_user_data_read':0,'existing_models_web_DTW_changed':False,'running_processes':[],'implementation':'Codex local','verification':'Separate saved arithmetic/actual trace/tensor verification by same Codex','training_sessions':{'30007':0,'62419':0}}
write(U/'FINAL_AUDIT.json',out,new=True)
print('PASS: complete comparison, protected evidence, report and current documents synchronized; no promotion')
