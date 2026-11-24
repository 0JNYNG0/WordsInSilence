from common import *
import re
assert not (U/'FINAL_AUDIT.json').exists()
s=read(U/'SCOPE.json');v=read(U/'INDEPENDENT.json');a=read(U/'ANALYSIS.json');c=read(U/'CONCLUSION.json');n=read(U/'NEXT_PLAN.json')
check(s['pins'])
assert v['status']=='PASS' and v['analysis_sha256']==sha(U/'ANALYSIS.json')
assert v['verifier_sha256']==sha(U/'verify_analyze.py')
assert v['guarded_sha256']==sha(U/'GUARDED_GEOMETRY.json')
assert read(U/'COMPLETE.json')['exit_code']==0 and not (U/'RUNNING.json').exists()
assert c['new_training_runs']==c['new_model_inference']==c['user_recordings_read']==0
assert c['semantic_phase_contracts_reviewed']==0 and c['semantic_error_detection_accuracy'] is None
assert read(U/'LAYOUT_QA.json')['status']=='PASS'
for name in ['FIRST_EXECUTION_ERROR.json','FIRST_GUARD_ERROR.json']:assert not (U/name).exists()
assert not n['training_authorized_now'] and not n['automatic_adoption']
for p in U.glob('*.json'):read(p)
page=(U/'RESULT_KO.html').read_text(encoding='utf8')
for href in re.findall(r'href="([^"]+)"',page):
 if not href.startswith(('https:','http:','#','file:')):assert (U/href).exists(),href
manifest=read(AUD/'VISUAL_MANIFEST.json');sheets=['reference_01.jpg','reference_02.jpg','reference_10.jpg','reference_11.jpg']
for name in sheets:assert sha(AUD/'visuals'/name)==manifest['files'][name]

stamp=now();key=U.name
heading='### 2026-10-19 공개 단계·순서 평가 기준 검증 완료 — 관측/반복/필수단계 분리, 추가학습 없음'
next_action='이번 평가 기준과 공개 자료 진단 결과를 사용자와 검토한다. 다음 단일 작업은 공개 영상의 관측 가능한 단계 후보와 필요한 손·가림을 기록하는 검토표를 구체화하고, 수어 검토로 필수/선택 단계·선후 관계·허용 반복을 확인할 소규모 라벨 검증 계획을 정하는 것이다. 확인되지 않은 기계적 삭제/역순을 수어 오류 정답으로 학습하지 않는다. C·T3-F·웹·DTW 유지, 추가학습은 결과 함께 확인 후. 사용자·멘토 촬영은 학습·튜닝·후보선택에 사용하지 않는다.'
body=f'''{heading}
갱신 {stamp}. ops/{key}. 공개 자료의 누락·순서 평가 기준 정리/기술 검증 완료. 추가학습0·새모델추론0·새DEV추론0·사용자/멘토촬영접근0·ID14~18원시접근0. 기존C/T3-F·웹·DTW보존.
- 공개100단어 TRAIN899/ID01~09. 고정참고100영상의9조건×2관측=1800쌍 + 다른촬영자799×2=1598쌍,합3398pair/view. 서로독립3398영상아님. 시계0.7/1.3/구간별변경600쌍,동일관측프레임유지200쌍에서기존거리/보류상태불변. 실제촬영속도에따른검출안정성검증아님.
- 기준: 단어정체성/필수단계·선후/관측근거를분리. 절대초·프레임수·반복횟수일괄감점금지. 같은자세유지와실제움직임주기반복구분,반복은검토된허용규칙만적용. 준비/마무리전체손관측을요구하지않고필수단계별근거확인. 미검출은누락으로판정하지않음.
- 최초탐색거리의공통로컬손목원점퇴화발견: masked다른촬영자799쌍중534쌍0,직접검사표본에서손목한점으로모든프레임대응가능확인. 기존모델/운영DTW버그로해석금지. FIRST_PROBE_FINDING/GEOMETRY보존. EVIDENCE_GUARD_PROTOCOL로좌우분리/global21/local비손목20전체관절근거진단추가. 엄격한진단요건이며최적근거임계값아님,입력변경없음.
- 공개sensor오른손위치에서중간1/3삭제100/100·앞두구간교환80/100이같은단어다른사람차이중앙값이하. 순서있는평균거리공통합격선불충분근거. 삭제·교환·반복은기계적탐침이며수어오류정답아님. 관측부거리만으로삭제/미검출마스킹구분불가,관측상태별도유지. 왼손비교가능67단어/나머지33오류로세지않음.
- 합성구조12사례/이름치환300검사PASS. 저장13592손·항목지표,별도스칼라192·작은경로전수192검산최대2.22e-16. 동일Codex별도계산이며독립agent/수어전문가아님. 실행41653·90219종료0,진행중없음.
- 기존공개8영상64표본시각재검토와100단어검토표초안완료. 성남/전주후반유사한위치변화관찰하나도시구별정확단계미확정;한글명칭분할로단계정답만들지않음. 전문가확인단계규칙0개,수어오류검출정확도미산출. NEXT_PLAN따라라벨기준확인후에만추가학습설계. RESULT_KO.html/txt,SCENE_REVIEW,PHASE_REVIEW_CATALOG,ANALYSIS,INDEPENDENT,CONCLUSION,NEXT_PLAN,FINAL_AUDIT.

'''
docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json'];before={str(p):sha(p) for p in docs}
text=docs[0].read_text(encoding='utf8');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
assert h8 in text and h9 in text and heading not in text
text=text.replace(h8,h8+'\n'+body,1).replace(h9,h9+'\n'+heading+'\n'+next_action+'\n\n',1)
state=read(docs[2]);assert state['current_stage']=='WORD100_MULTI_POSITION_AUX_COMPLETE_NOT_PROMOTED'
state.update(current_stage='WORD100_STAGE_EVALUATION_COMPLETE_SEMANTIC_REVIEW_PENDING',updated_at=stamp,updated_date_kst=stamp[:10],next_single_action=next_action)
state[key]={'status':c['status'],'at':stamp,'unit':key,'public_words':100,'public_train_clips':899,'pair_views':3398,
 'clock_control_pairs':600,'dwell_control_pairs':200,'saved_hand_metrics':13592,'independent_numeric_checks':192,'exhaustive_paths':192,
 'max_numeric_error':v['max_numeric_error'],'expert_reviewed_phase_contracts':0,'semantic_error_detection_accuracy':None,
 'new_training_runs':0,'new_model_inference':0,'user_recordings_read':0,'heldout_raw_read':0,'production_changed':False,
 'original_probe_issue_preserved':True,'sessions':[41653,90219],'exit_code':0,'running_processes':[],
 'automatic_training':False,'automatic_adoption':False,'report':str(U/'RESULT_KO.html'),'next_plan':str(U/'NEXT_PLAN.json')}
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

현재 목표: 공개 자료의 동작 누락·순서 평가 기준을 만들고 정상 속도·반복을 오류로 단정하지 않는지 검증. 기술 검증 완료, 수어 의미상 정답표 검토가 남았다. 추가 학습은 결과 함께 확인 후다.

- 현재 단위 research/front_keyphase_v1/ops/{key}. SCOPE/EXECUTION_CONTEXT/COMPLETE/EVIDENCE_GUARD_PROTOCOL/CONTRACT_TESTS/GEOMETRY/GUARDED_GEOMETRY/ANALYSIS/INDEPENDENT/PHASE_REVIEW_CATALOG/SCENE_REVIEW/CONCLUSION/NEXT_PLAN/RESULT_KO.html/txt/FINAL_AUDIT.
- 범위: 공개TRAIN899(ID01~09)·100단어·sensor/masked. 고정참고100×9조건×2=1800,다른사람같은단어799×2=1598,합3398쌍. 새모델추론0/학습0/새DEV추론0/개인·멘토접근0/ID14~18원시0. C/T3-F·웹·DTW불변. 서로독립3398영상아님.
- 기준은단어정체성,필수단계/선후관계,관측근거분리. 절대시간·프레임수로감점안함. 같은상태유지와움직임주기반복구분. 허용반복/필수·선택은수어검토필요,미확정은보류. 전체시작끝손관측요구아니며필수단계별로필요한손근거확인. 미검출을누락이나0차이로단정금지.
- 시계만0.7/1.3배·구간별변경600쌍,동일관측유지200쌍에서기존거리및보류상태불변. 실제새촬영프레임속도에서의MediaPipe성능검증이아님. 합성단계구조12사례/이름치환300검사PASS;한국수어정답시험아님.
- 최초탐색거리local손목원점퇴화: masked다른사람799쌍중534쌍0,표본직접확인시공통손목만으로전프레임대응가능. 기존모델/운영DTW버그라고주장금지. FIRST_PROBE_FINDING와GEOMETRY보존,원결과덮어쓰지않음. 후속고정guard는좌우분리/global21점/local비손목20점전체관측만비교. 엄격하여부분정상손도보류할수있는진단요건이며제품합격선아님. 기존모델입력/계수변경0.
- sensor오른손위치에서중간삭제100/100·앞두구간교환80/100이같은단어다른사람차이중앙값이하. 평균거리공통합격선불충분. 기계적3등분은의미단계가아님,삭제/교환/반복을수어오류정답으로전환금지. 동일관측부분에서는삭제/가림마스크거리같아별도관측근거가필수. 왼손완전관측비교67단어,나머지33단어를오류로세지않음.
- 공개8영상64표본기존contactsheet시각재검토,100단어PHASE_REVIEW_CATALOG초안. 전문가확인필수단계/반복정답0개. 따라서오류검출정확도·정상오판율·수어완수정확도는미산출. 성남/전주후반유사위치변화는관찰되지만도시구분단계/필수손모양미확정. 한글명칭을잘라수어단계를만들지않음.
- 별도검산저장13592지표·스칼라192·작은정렬경로전수192,최대2.22e-16. 같은Codex별도산술이며독립agent/전문가아님. Chrome desktop1280/mobile390 QA및시각검사PASS. 실행41653/90219종료0,진행중없음. 계산/검산재실행불필요.
- 직전M3후보는미채택보존: sensor99.33%(TF99.22%대비+.11%p),masked99.00%(TF동일/직전AUX-.11%p),1/3회만개선·masked성남새오답. 이번작업은그모델을재학습/교체하지않음.
- 다음: 공개영상의관측가능단계후보기록→수어검토로필수/선택/선후/허용반복확인→기준고정후정상오판/검토된오류검출/판단보류별평가→결과함께확인뒤에만별도추가학습설계. 파생영상은원본과같은분할,개인자료선택금지. 범용구조목표이나100단어밖정확도보장없음.
- 원본/W:/sources/과거동결파일읽기전용. Git/카메라/다운로드/외부프로젝트전송없음. 공개논문조회만함. 사용자·멘토는실사용테스터로촬영학습/튜닝/후보선택금지.

다음 한 작업: {next_action}
'''
for p,h in before.items():assert sha(p)==h,'Concurrent document change'
docs[0].write_text(text,encoding='utf8');docs[1].write_text(handoff,encoding='utf8');docs[2].write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
check(s['pins'])
assert all(key in p.read_text(encoding='utf8') for p in docs)
write(U/'FINAL_AUDIT.json',{'status':'PASS_TECHNICAL_CRITERIA_COMPLETE_SEMANTIC_LABELS_PENDING','at':stamp,'protected_files':len(s['pins']),
 'running':False,'sessions':[41653,90219],'exit_code':0,'new_training':0,'new_model_inference':0,'user_recordings_read':0,'heldout_raw_read':0,
 'existing_models_preserved':True,'production_changed':False,'semantic_phase_contracts_reviewed':0,'semantic_error_accuracy':None,
 'first_probe_issue_preserved':True,'verification':'Same Codex independent scalar computations and exhaustive small alignment paths. Not independent agent or KSL expert.',
 'visual_review':'Four existing sheets(8selectedrows64frames), desktop report top and chart/table screenshot; mobile overflow check.',
 'doc_hashes':{str(p):sha(p) for p in docs},'files':{p.name:sha(p) for p in U.iterdir() if p.is_file()}})
print('PASS: criteria and public diagnostics complete; semantic labels pending; docs synchronized; no model changes or training.')
