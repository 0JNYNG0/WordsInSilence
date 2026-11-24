from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,re,shutil
D=Path(__file__).resolve().parent;O=D.parent;ROOT=O.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (D/'FINAL_AUDIT.json').exists()
s=read(D/'SCOPE.json');v=read(D/'INDEPENDENT.json');a=read(D/'ANALYSIS.json');c=read(D/'CONCLUSION.json');n=read(D/'NEXT_PLAN.json')
assert v['status']=='PASS' and sha(D/'ANALYSIS.json')==v['analysis_sha256']
assert sha(D/'diagnose.py')==v['diagnostic_code_sha256'] and sha(D/'verify_analyze.py')==v['verifier_code_sha256']
for p,h in s['pins'].items():assert sha(Path(p))==h,p
assert read(D/'LAYOUT_QA.json')['status']=='PASS'
assert c['training_runs']==c['new_DEV_inference']==c['user_data_read']==0
assert not n['training_authorized_now'] and c['sensor_error_events']==7
assert c['negative_correction_events']==5 and c['positive_insufficient_events']==2
assert n['proposed_candidate']['parameter_budget']['GRU']+n['proposed_candidate']['parameter_budget']['new_head']==7200
page=(D/'RESULT_KO.html').read_text(encoding='utf8')
for href in re.findall(r'href="([^"]+)"',page):
 if not href.startswith(('https:','http:','#')):assert (D/href).exists(),href
for p in D.glob('*.json'):read(p)
stamp=datetime.now(timezone(timedelta(hours=9))).isoformat();key=D.name
heading='### 2026-10-19 보조 오답 원인·순서 진단 완료 — 후반 특징 의존 확인, 다중 시점 읽기 제안'
next_action='공개자료 보조원인 진단·검산·보고 완료. 사용자와 NEXT_PLAN을 검토한 뒤 별도 F-G16-M3(가칭)의 다중시점 GRU 읽기 사전검증부터 진행할지 결정한다. 이번 승인에 새모델구현/추가학습 없음. 제안은 T3-F전체고정·같은280입력/관측/시간/CE+.1auxCE/단어예외없음, 마지막16읽기→앞중간뒤48읽기,새부품4000→7200. 승인·사전검증후에만 동일3seed 학습비교. 별도순서손실 동시추가/계수재탐색/웹교체/보류자료평가 없음. 사용자·멘토는실사용테스터로 읽기·학습·튜닝·선택 금지. 다른수어에쓸공통목표이나검증100단어 한정.'
body=f'''{heading}
갱신 {stamp}. ops/{key}. 공개TRAIN899·100단어·기존AUX01선택3seed·masked/sensor2조건의 고정진단 완료. 저장DEV300 component1800묶음만검산(새DEV추론0),추가학습0/사용자접근0/보류원시0. 기존sensor고유6영상/반복7오답에서aux단독정답0;기존오답과비교한보조점수차5건음수/2건양수이나격차부족. 보조가중치단순확대근거없음. TCN원본특징을한번계산하고GRU가읽는특징열6조건고정: 앞두구간교환·끝유지시보조답유지90.43~94.66%,세구간역순8.12~18.91%,끝1/3만78.64~88.88%(각899분모·3seed범위). masked동일경향89.54~96.77/6.45~17.02/80.65~94.10. 앞교환점수상대변화중앙값0.95~4.30%. 후반특징의존근거이나TCN특징은최대61프레임문맥이있어원시앞동작무시/순서정답/오답유일원인확정금지. 동일TRAINwhole고정추론base/합산899/899모든seed·view,auxsensor251/303/443;TRAIN성능을일반화로보고금지. 저장학습이력보조계속배워도합산CE악화/일부정답수회귀,선택epoch변경없음. 단어정답만으로필수단계/생략판정을가르치지않았음. 기존T3F이미시간3구간사용하므로순서전혀모름도틀림. 별도NumPyGRU360출력최대차3.2922e-7,저장합산exact,2494보호파일유지. 진단exec55909종료0/검산0. 보고검사single-summary복사도구가2개summary에strict오류1회→FIRST_LAYOUT_ERROR보존후각각검사수정·desktop/mobilePASS/시각확인. 실험오류없음. 같은Codex별도검산이며독립agent아님. 다음F-G16-M3가칭은같은GRU시간별hidden을기존elapsed삼각3구간으로읽고48→100zerohead,기존TF47236고정·새7200(기존4000보다3200증가,구조+용량효과분리불가). 입력·손실·일정유지/단어별분기없음. 아직미구현/미학습,결과검토후사전검증→승인된3회비교만. 관측단계/가림/반복평가기준별도,필수수어단계는수어검토필요;임의역순을틀린수어라라벨금지. C/T3F/웹/DTW불변;기대개선%미확정/모든수어검증아님. RESULT_KO.html/txt,ANALYSIS,INDEPENDENT,NEXT_PLAN,CONCLUSION,FINAL_AUDIT.

'''
docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json'];before={str(p):sha(p) for p in docs}
backup=D/'before';backup.mkdir(exist_ok=False)
for p in docs:shutil.copyfile(p,backup/p.name)
ctx=docs[0].read_text(encoding='utf8');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
assert h8 in ctx and h9 in ctx and heading not in ctx
ctx=ctx.replace(h8,h8+'\n'+body,1).replace(h9,h9+'\n'+heading+'\n'+next_action+'\n\n',1)
state=read(docs[2]);assert state['current_stage']=='WORD100_AUXILIARY_SUPERVISION_COMPLETE_NOT_PROMOTED'
state.update(current_stage='WORD100_AUXILIARY_ORDER_DIAGNOSIS_COMPLETE_PLAN_REVIEW',updated_date_kst=stamp[:10],next_single_action=next_action)
state[key]={'status':'COMPLETE_DIAGNOSIS_NO_TRAINING_PENDING_PLAN_REVIEW','at':stamp,'unit':key,
 'public_train_clips':899,'words':100,'seeds':[20261013,20261014,20261015],'conditions':6,'views':2,
 'saved_DEV_components_verified':1800,'numpy_GRU_samples_verified':360,'max_numeric_error':v['max_logit_absolute_error'],
 'sensor_error_events':7,'unique_error_clips':6,'aux_error_corrections':0,'negative_correction_events':5,'positive_insufficient_events':2,
 'early_swap_retention_range':c['swap_early_aux_label_retention_range_pct'],'reversed_blocks_retention_range':c['reversed_blocks_aux_label_retention_range_pct'],
 'training_runs':0,'new_DEV_inference':0,'user_recordings_read':0,'heldout_read':0,'production_changed':False,
 'candidate_proposed_not_implemented':'F-G16-M3 multi-position GRU readout; trainable4000->7200, loss/input frozen',
 'running_processes':[],'session':55909,'exit_code':0,'report':str(D/'RESULT_KO.html'),'next_plan':str(D/'NEXT_PLAN.json'),'automatic_training':False}
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

현재 목표: 보조가 학습됐는데도 기존 오답을 고치지 못한 원인을 공개 자료로 진단하고 다음 공통 계획 보고. 완료했으며 추가 학습은 결과 검토 후다.

- 현재 단위 research/front_keyphase_v1/ops/{key}. SCOPE/COMPLETE/ANALYSIS/INDEPENDENT/CONCLUSION/NEXT_PLAN/RESULT_KO.html/txt/FINAL_AUDIT.
- 최신 사용자 범위: 특정 단어 맞춤 금지, 기존 C/T3-F 유지, 사용자 촬영 학습·튜닝 금지, 추가 학습 전 결과·계획 검토. 이번 사용자/멘토 촬영 접근0, 원시ID14~18 접근0, 새DEV추론0, 추가학습0.
- 입력은 기존 AUX01의 공개 TRAIN899·100단어·3seed·masked/sensor2조건, x_whole 고정. 기존 selected checkpoint만 사용. 보조GRU가읽는 TCN특징열 original/swap_first_two_thirds/reverse_thirds/prefix_two_thirds/suffix_third/last_feature 6조건 결과를 보기 전 고정했다. 최종DEV 재채점없음.
- 핵심 신규근거: sensor 앞2구간순서교환/끝유지 시보조답유지90.43~94.66%,끝까지바뀌는3구간역순8.12~18.91%,끝1/3만78.64~88.88%. masked도89.54~96.77/6.45~17.02/80.65~94.10. 앞교환상대centered점수RMS중앙값0.95~4.30%. 후반특징의존을지지하지만가공특징최대61프레임문맥때문에원시앞동작무시로해석금지. thirds는semantic단계아님;역순의수어정답여부미정. 이수치는정확도아님.
- 기존AUX01 저장sensor고유6영상/3seed반복7오답에서aux도전부오답.5건정답보다기존오답에점수추가,2건정답방향이나격차미달. 가중치만키울근거없음. 단어정답만학습했고동작단계/순서/생략정답은없음. T3F도이미3구간요약하므로순서전혀못봄은아님.
- 이번고정TRAINwhole원본base/합산899/899모든6조합;auxsensor251/303/443. 학습자료이며새사람성능아님. 선택후보조분류늘어도합산CE악화한기존이력재사용;epoch/계수재선택안함.
- 직전실험 U=word100_auxiliary_supervision_train_20261019: sensor296/299/298=99.22%(TF와동일),masked295/299/298=99.11%(TF+0.11%p),선택1/1/8·종료13/13/20. 새후보미채택/C·TF그대로. 원래실험/검산/학습trace 재실행불필요.
- 다음제안F-G16-M3(가칭/미구현): T3F47236전체eval동결,새GRU16의시간별hidden을기존elapsed삼각앞중간뒤로요약하여3x16=48읽기/zero Linear48->100. 기존last16하나읽기를대체. 새파라미터2400+4800=7200(기존4000보다3200증가),전체54436;개선시구조와용량효과완전분리불가명시. 현재280입력/ROI/마스크/시간/CE(b+a)+.1CE(a)/스케줄/선택유지. 단어별경계/보너스없음. 승인후사전검증→같은3seed비교만;현재학습승인아님. 기존저장기준재사용;실패시자동재탐색없음.
- 별도평가: 기존공개100참고감사를재사용해관측가능변화/정상반복/실제가림/화면밖을구분. 필수수어단계·생략오류는수어전문검토전정답단정금지. 순서전용학습신호는별도가설로첫비교에추가하지않고타임스탬프/위치지름길·문맥누출·정지/반복을검토해야함.
- 검산: exec55909종료0,별도NumPyGRU360출력 최대3.2922e-7,저장DEV1800 base/aux/final묶음정합,보호2494파일해시유지. 같은Codex별도코드이며독립agent/수어전문가아님. reportQA단일summary복사검사오류1회 FIRST_LAYOUT_ERROR보존→둘각각검사수정후desktop/mobilePASS·시각확인. 모델/진단결과오류없음.
- 진행중없음. C/T3F/웹/DTW변경없음. 원본/W:/sources/과거동결단위읽기전용. Git/카메라/다운로드/외부프로젝트전송없음. 공개논문웹조회만함. 사용자는실사용테스터이며촬영을학습/튜닝/선택에사용금지.
- 공통적용목표이나현재100단어검증범위,미학습수어자동인식/향상률보장없음. ID14~16미사용자료를새튜닝에사용하지않음.

다음 한 작업: {next_action}
'''
for p,h in before.items():assert sha(Path(p))==h,'Concurrent document change'
docs[0].write_text(ctx,encoding='utf8');docs[1].write_text(handoff,encoding='utf8');write(docs[2],state)
assert all(key in p.read_text(encoding='utf8') for p in docs)
for p,h in s['pins'].items():assert sha(Path(p))==h,p
write(D/'FINAL_AUDIT.json',{'status':'PASS_DIAGNOSIS_COMPLETE_PLAN_REVIEW','at':stamp,'training_runs':0,'user_recordings_read':0,'DEV_new_inference':0,
 'existing_models_preserved':True,'protected_hash_count':len(s['pins']),'source_scope_sha256':sha(D/'SCOPE.json'),
 'diagnosis_session':55909,'exit_code':0,'running':False,'verification':'Same Codex separate NumPy reference and saved arithmetic; no independent agent.',
 'report_layout_issue':'FIRST_LAYOUT_ERROR.json preserved, checker corrected and passed; results unaffected.',
 'visual_review':'Desktop top and diagnostic chart inspected; desktop/mobile width and all details checked.',
 'docs_synced':{str(p):sha(p) for p in docs},'files':{p.name:sha(p) for p in D.iterdir() if p.is_file()}})
print('PASS: diagnosis complete, documents synchronized, no training or production change')
