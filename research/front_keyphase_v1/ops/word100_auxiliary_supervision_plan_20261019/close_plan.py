from pathlib import Path
import json, hashlib, re
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert not (U/'FINAL_AUDIT.json').exists()
scope=read(U/'SCOPE.json');plan=read(U/'PROPOSED_PROTOCOL.json');review=read(U/'REVIEW.json')
for p,h in scope['source_pins'].items():assert sha(p)==h,p
assert not scope['training_authorized_this_turn'] and not review['training_ready']
assert review['training_runs']==0 and read(U/'MATH_REVIEW.json')['optimizer_steps']==0
assert read(U/'LAYOUT_QA.json')['status']=='PASS'
page=(U/'RESULT_KO.html').read_text(encoding='utf8')
for href in re.findall(r'href="([^"]+)"',page):
    if not href.startswith(('http:','https:','#')):assert (U/href).exists(),href
for p in U.glob('*.json'):read(p)
stamp=datetime.now(timezone(timedelta(hours=9))).isoformat();key=U.name
docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
for p in docs:assert sha(p)==sha(U/'before'/p.name),'Concurrent changes: '+str(p)
heading='### 2026-10-19 보조 단어 분류 신호 설계 검토 완료 — F-G16-AUX01 제안, 추가 학습 없음'
next_action='사용자와 F-G16-AUX01 제안(PROPOSED_PROTOCOL.json)을 검토한 뒤, 승인되면 동일 초기값·입력으로 CE(b+a,y)+0.1 CE(a,y)만 추가하는 별도 후보를 구현/검증하고 3seed 비교한다. 현재는 설계·가상 수식 검토까지이며 학습 승인/READY 아님. C/T3-F/F-G16 저장 점수 재사용, 기존 모델·웹·DTW 유지. 사용자·멘토 자료 및 ID14~18 원시자료 사용 금지. 결과를 본 뒤 계수·epoch·선택 기준 변경 금지.'
body=f'''{heading}
갱신 {stamp}. ops/{key}. 최신 사용자 요청은 별도 보조 학습 신호 검토와 추가 학습 전 구체적 계획 보고. 이전 F-G16은 sensor 평균99.22%, T3-F대비+0.00%p/새정답0. 최초 TRAIN CE약1.58e-5~3.09e-5와 실제 head변화 근거로 기존 높은 확신이 보조 신호를 작게 만들 가능성 검토; 인과확정/학습미실행 오해 금지. 다음 단일 후보 F-G16-AUX01: 기존 T3-F47236/BN고정, 같은 GRU16+같은 출력4000만 학습; 최종CE+0.1×보조단독CE. 정답은 같은 공개 단어라벨이며 순서/완수 라벨 아님. 0.1은 검증된최적값/10%영향이 아닌 사전 제안값, 초기 보조항 지배 가능성/기존 정답 뒤집힘 위험 명시. 별도head/모델크기/입력/가중합 변경 없음. 899TRAIN(ID01~09)/300DEV(ID11~13),3seed20261013/14/15,각 원래TF+동일 신생보조초기해시부터,학습FG이어받기금지. 동일 batch32/AdamW.001/weightdecay.001/clip1/max80/patience12; 선택은 최종combined 양view DEV CE만. 저장lambda0 FG16 및 TF주비교/C보조. sensor평균상승/2seed상승+기타 평균회귀없음 gate,모든seed회귀보고,보조성적만개선성공불가. DEV반복사용/천장효과/3seed는독립900개아님/순서이해미검증 명시. review_math.py는 가상100클래스CE 수식/유한차분/torch기울기일치최대9.75e-11,zerohead첫단계memorygradient0→사전고정nonzerohead에서전달 확인; 프로젝트모델로드0/optimizer0/자료읽기0. 이는 모델준비PASS아님. 같은Codex별도수식검산,독립agent아님. 이전완료파일/기존모델/웹source해시보존. 모델/학습코드수정0,추가학습0,웹/DTW교체0. RESULT_KO.html/PROPOSED_PROTOCOL/REVIEW/MATH_REVIEW/SCOPE. 공개 PyTorch2.12 CE와 DSN원논문 일반근거확인,프로젝트데이터외부전송0. 다음은사용자계획검토후승인범위실행.

'''
text=docs[0].read_text(encoding='utf-8-sig');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n'
assert h8 in text and h9 in text and heading not in text
text=text.replace(h8,h8+'\n'+body,1).replace(h9,h9+'\n'+heading+'\n'+next_action+'\n\n',1)
state=read(docs[2]);assert state['current_stage']=='WORD100_FROZEN_ORDER_AUX_COMPLETE_NO_EXTRA_GAIN'
state.update(current_stage='WORD100_AUXILIARY_SUPERVISION_PLAN_REVIEWED_NO_TRAINING',updated_date_kst=stamp[:10],next_single_action=next_action)
state[key]={'status':'PLAN_REVIEW_COMPLETE_WAIT_USER','at':stamp,'unit':key,'candidate':'F-G16-AUX01','lambda_aux':0.1,'loss':'CE(b+a,y)+0.1 CE(a,y)',
 'proposed_runs':3,'training_runs':0,'optimizer_steps':0,'training_authorized':False,'training_ready':False,'candidate_implemented':False,
 'math_verification':'PASS_SYNTHETIC_ONLY','base_parameters_frozen_proposed':47236,'aux_parameters_proposed':4000,
 'train_videos_proposed':899,'dev_videos_proposed':300,'seeds_proposed':[20261013,20261014,20261015],
 'user_recordings_read':0,'C_T3F_web_DTW_preserved':True,'model_inferences':0,'running_processes':[],
 'report':str(U/'RESULT_KO.html'),'plan':str(U/'PROPOSED_PROTOCOL.json'),'source_scope':str(U/'SCOPE.json')}
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {stamp}

현재 목표: 보조 부분 자체의 단어 분류 학습 신호 설계 검토 완료. 사용자 계획 확인 전이며 추가 학습 없음.

- 단위: research/front_keyphase_v1/ops/{key}. SCOPE/PROPOSED_PROTOCOL/REVIEW/MATH_REVIEW/RESULT_KO.html/FINAL_AUDIT.
- 최신 승인 범위는 검토·계획 보고. 학습/모델 수정/운영 교체는 이번에 하지 않음. 모델 로드·영상 읽기·optimizer step 모두0.
- 직전 완료 F-G16 3회 sensor296/299/298=99.22%, TF와동일(+0.00%p), C295/299/294=98.67%보다+0.56%p는TF기존효과. 새정답0/새오답0,seed13동대문오답후보만변화. 옛 unit word100_frozen_order_aux_train_20261019는읽기전용. 저장검산/학습재실행금지.
- 기존 높은확신으로 combined CE학습신호가약할가능성(첫TRAIN CE1.58e-5~3.09e-5). 가중치는실제변했음; 유일원인확정아님/AdamW기울기크기만으로학습여부단정금지.
- 제안 F-G16-AUX01: L=mean CE(b+a,y)+0.1 mean CE(a,y). b=원래matching-seed T3F전부고정, a=기존GRU16→같은100출력head. 별도head/파라미터추가없고 새4000만. 추론b+a 유지. 0.1은 사전제안값/최적값아님/최종영향10%아님. 초기별도신호커져도 성능보장없음.
- 같은공개100단어 TRAIN899(ID01~09)/DEV300(ID11~13),seed20261013/14/15,동일입력280/ROI/좌우/정규화/whole-context-view일정. 각 원래TF+동일초기보조상태로새로시작; 학습된FG이어학습금지. 3회계획이나 아직승인/READY아님.
- batch32,AdamWlr.001/weightdecay.001,clip1,max80,patience12,최종combined2view DEV CE>1e-6 개선선택. 보조DEV성적은진단만/선택금지. 조기종료가보조학습전에일어날수있어도결과보고조건변경금지.
- 주비교TF/FG16(lambda0)저장점수,C보조. sensor평균상승및>=2/3seed상승, maskedtop1/양viewtop5/사람·미검출집합평균회귀없음,개별회귀모두보고. 보조단독정확도상승/확신상승만으로성공금지. 3seed는독립900영상아님.
- 허구의100class logit으로 수식/NumPy미분/유한차분/torch자동미분확인PASS(max9.75e-11). 실제후보검증아님. zerohead첫단계memory기울기0은유지,head가먼저학습해야이후전달가능. 실제학습update0. 동일Codex별도검산,독립agent아님.
- 승인 후 순서: 별도후보loss/logging구현→해시/동결/gradient/preflight→3회→저장점수·실제배치trace검산→비교보고. 기존C/TF/FG/웹/DTW보존. 초기해시/입력검산기존근거재사용,변경영향만검사.
- 별도단어loss는 순서/완수/잘못된수어 정답라벨을새로주는게아님. 부분모양암기·동결특징손실·보조점수로정답뒤집힘·개발집합반복사용한계유지. 사용자/멘토촬영읽기·학습·튜닝·선택금지. ID14~16새heldout299개보존,ID17/18은이미과거평가되어새blind근거아님; 이번ID14~18원시사용0.
- 이전학습exec62419완료/EXIT0/TRAIN_RUNNING없음 확인. 이번진행학습없음. 공개공식기술문서열람만했으며프로젝트자료외부전송/다운로드/Git/카메라없음.

다음 한 작업: {next_action}
'''
docs[0].write_text(text,encoding='utf8');docs[1].write_text(handoff,encoding='utf8');write(docs[2],state)
assert all(key in p.read_text(encoding='utf8') for p in docs)
for p,h in scope['source_pins'].items():assert sha(p)==h
write(U/'FINAL_AUDIT.json',{'status':'PASS_PLAN_REVIEW_ONLY','at':stamp,'source_files_verified':len(scope['source_pins']),
 'new_training_runs':0,'optimizer_steps':0,'candidate_implemented':False,'model_inferences':0,'raw_user_recordings_read':0,
 'protected_evidence_unchanged':True,'training_ready':False,'docs_synced':{str(p):sha(p) for p in docs},
 'files':{p.name:sha(p) for p in U.iterdir() if p.is_file()},'verification':'Same Codex evidence hashes, separate analytic/finite-difference math and HTML layout/visual review; no independent agent'})
print('PASS: plan review complete, documents synchronized; no training and no model changes.')
