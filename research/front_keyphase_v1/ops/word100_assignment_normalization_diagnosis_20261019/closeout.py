"""Synchronize the current handoff without altering any prior frozen experiment."""
from pathlib import Path
import json,hashlib,re
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;ROOT=U.parents[3];O=U.parent;P=O/'word100_assignment_training_20261018';B=O/'word100_body_boundary_20261017'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
now=datetime.now(timezone(timedelta(hours=9))).isoformat();key=U.name
assert not (U/'FINAL_AUDIT.json').exists()
verification=read(U/'VERIFICATION.json');assert verification['status']=='PASS'
for name,h in verification['result_pins'].items():assert sha(U/name)==h
models={}
for a in ['C','P','Q']:
 for s in [20261013,20261014,20261015]:
  prefix=B/f'training_C_{s}' if a=='C' else P/f'{a}_{s}';r=read(str(prefix)+'_RESULT.json');ck=Path(str(prefix)+'_best.pt')
  assert sha(ck)==r['checkpoint_sha256'];models[str(ck)]=sha(ck)
state_path=O/'STATE.json';state=read(state_path)
next_action='손 배정·크기 정규화 진단 및 NEXT_PLAN의 두 교차 조건×3seed(6회) 통제 비교를 사용자와 검토. 새 학습/운영교체는 아직 승인되지 않음. 기존 C·웹·DTW 유지, 사용자/멘토 학습·튜닝·선택 금지. 완료 진단/DEV 채점 재실행 불필요.'
old=state['word100_assignment_training_20261018'];old['training_runs']=6;old['next_single_action']='P/Q 6회 학습·비교 완료, 미채택. 후속 공개 진단 완료: '+key+' 참조. 추가 학습은 새 계획 사용자 확인 후.';old['metadata_reconciled_at']=now
state.update(updated_date_kst=now,updated_at=now,current_stage='WORD100_ASSIGNMENT_NORMALIZATION_DIAGNOSIS_COMPLETE',activity_state='AWAITING_USER_PLAN_REVIEW',next_single_action=next_action)
state[key]={'status':'COMPLETE_VERIFIED_DIAGNOSIS_PLAN_PENDING','at':now,'unit':str(U),'contract':str(U/'CONTRACT.json'),'result':str(U/'RESULT_KO.html'),'next_plan':str(U/'NEXT_PLAN.json'),
 'training_allowed':False,'new_training_runs':0,'DEV_reinference':0,'public_TRAIN_sources':899,'public_DEV_saved_sources':300,'frozen_C_train_sensitivity_forward_samples':540,
 'changed_train_sources':{'P':7,'Q':17},'real_changed_tensors':{'P':15,'Q':30},'normalization_only_propagated_tensor_frames':{'P':591,'Q':1061},
 'sensor_saved_gain_loss':{'P':[3,3],'Q':[3,7]},'new_sensor_losses_all_same_DEV_input':True,'scale_only_fixed_C_rank_changes':0,'scale_only_pairs':135,
 'causality':'Input propagation proven; normalization as sole cause of retrained performance regression unproven. Corrected-word C sensor DEV already21/21 and42/42 eachseed.',
 'verification':str(U/'VERIFICATION.json'),'independent_human_or_agent_review':False,'user_data_used':False,'production_changed':False,'running':False,'next_single_action':next_action}
write(state_path,state)
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=ctx.read_text(encoding='utf8');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
entry8='''### 2026-10-19 손 배정·크기 정규화 영향 공개 진단 완료 — 추가 학습 없음
ops/word100_assignment_normalization_diagnosis_20261019. 공개 TRAIN899의 손 크기 지원량/구간·관측조건 차이, 저장 C/P/Q DEV300×3seed의 오답 전이, 실제 변경 TRAIN 텐서45개×4분리조건×기존C3개=540건 고정모델 민감도 분석. P의 실제 변경원본7/899(7단어),Q17/899(14단어); 이 단어들의 C sensor DEV는 각seed21/21·42/42로 이미정답. sensor P새정답3/새오답3,Q3/7; 새오답모두검증입력동일이라재학습가중치/선택경로관여,단독원인미확정. 정규화로관측불변텐서프레임P591/Q1061값변화,전체절대차합7.20/4.75%이며정확도기여율아님. 최대Q분모−7.80%,같은좌표+8.47%. 크기만바꾼135쌍1위변경0(학습자료민감도,일반화증명아님). P선택epochC와전부동일;Q2seed선택시점차이. 별도산술검산PASS(180규모/5400저장쌍/540출력/4545핀). 이전16/36입력조합중변경없는구간1/6포함정정,실제변경15/30;611/1108변경프레임·27684/57318채널값은기존과동일. 모델9개해시보존,C/웹/DTW/사용자자료불변,학습0/DEV재추론0. 결과 RESULT_KO.html,VERIFICATION.json,NEXT_PLAN.json. 같은Codex별도계산이며독립사람·에이전트검토아님.
'''
entry9='''### 2026-10-19 정규화 통제 비교안 사용자 검토 — 추가 학습 대기
완료한 진단·기존DEV채점을반복하지않는다. NEXT_PLAN.json의 C/Q 저장각3seed재사용+두교차조건(Q배정/기존유효크기,기존배정/Q유효크기)각3seed총6추가학습은제안단계이며아직승인없음. 둘다유효한손에서만분모교환,한쪽계산불가면해당관측원래미관측·크기유지,손좌표합성금지. 관측·숫자크기영향을분리한원인진단용이며과거오배정크기를운영정답으로채택하지않음. 영향재현시현재관측에서믿을수있는크기를전체/핵심구간에공유하는후보를TRAIN지원량부터설계;미재현시크기재탐색중지하고남은공개혼동/학습안정성대상계획. 기존C/웹/DTW유지. 사용자·멘토촬영학습/튜닝/선택금지. 운영교체·새학습은사용자계획확인후.
'''
assert key not in text;text=text.replace(h8,h8+'\n\n'+entry8,1).replace(h9,h9+'\n\n'+entry9,1);ctx.write_text(text,encoding='utf8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신: {now}

현재 완료: ops/{key}. 사용자 요청 공개 자료 원인 진단·다음 계획까지 완료, 추가 학습 사용자 확인 대기.

- 기존 P/Q 각3seed 학습은 완료·미채택. sensor 평균 C98.6667%,P98.6667%,Q98.2222%; masked98.4444/98.7778/98.0000%. 기존 C·웹·DTW 유지.
- 새 진단: 실제 학습입력 변경 원본P7/899(7단어),Q17/899(14단어). 이 단어 C sensor DEV는 각seed21/21·42/42로 이미정답. 새 sensor 정답/오답 P3/3,Q3/7(300×3반복건수). 새오답모두DEV입력동일 → 재학습경로 관여, 단독원인 미확정.
- 손 크기 분모가 관측 불변 텐서프레임P591/Q1061로전파. 최대Q−7.80%분모→동일좌표+8.47%. 다수채널값변화는영향크기·정확도기여율과다름(전파프레임절대차합P7.20/Q4.75%).
- 기존C3개 고정, 변경된 TRAIN45조합×4조건=540건 추론,크기만바꾼135쌍1위변화0. 학습자료민감도라새사람/재학습성능보장안됨. P선택epochC와77/76/75로동일,Q77/78/78. 정규화가 하락단독원인이라고말하지않음.
- 별도산술검산VERIFICATION.json PASS(4545핀,180손크기,5400저장예측쌍,540출력). 같은Codex,독립사람/에이전트검토아님. 기존9체크포인트해시유지. 학습0,DEV재추론0,사용자자료접근/튜닝0,카메라/다운로드/외부전송/Git0.
- COUNT_RECONCILIATION: 이전P16/Q36은실제변경없는선택구간1/6포함. 진짜비동일텐서P15/Q30. 변경프레임611/1108·채널27684/57318은이전과일치. 이전실험결과파일수정안함. STATE의이전작업training_runs0/준비중문구만완료6으로동기화.
- RESULT_KO.html/txt,DIAGNOSIS_SUMMARY.json,NEXT_PLAN.json. 현재실행없음.

다음 한 작업: 사용자와 결과 및 제안된 6회 통제학습 검토. 기존 C/Q 저장결과재사용, Q배정+기존유효크기 / 기존배정+Q유효크기 각3seed 제안. 양쪽유효분모만교환, 미관측은해당관측분기유지. 원인진단용으로운영대체아님. 아직추가학습승인없음. 규모안정화후보는영향재현후TRAIN지원량부터계약. 완료진단/채점반복금지.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
# Verify synchronized files and report links.
state=read(state_path);assert state[key]['new_training_runs']==0 and not state[key]['training_allowed']
assert key in ctx.read_text(encoding='utf8') and key in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
page=(U/'RESULT_KO.html').read_text(encoding='utf8')
for link in re.findall(r'href="([^"]+)"',page):assert (U/link).is_file()
for p in U.glob('*.json'):read(p)
write(U/'FINAL_AUDIT.json',{'status':'COMPLETE_VERIFIED_PLAN_PENDING','at':now,'checkpoints_unchanged':models,'verification':verification,
 'report_and_plan_pins':{n:sha(U/n) for n in ['RESULT_KO.html','RESULT_KO.txt','DIAGNOSIS_SUMMARY.json','NEXT_PLAN.json','COUNT_RECONCILIATION.json']},
 'synced_documents':{str(p):sha(p) for p in [ctx,ROOT/'NEW_CHAT_HANDOFF.md',state_path]},'training_runs':0,'DEV_reinference':0,'user_or_mentor_data_used':False,'production_changed':False,'active_analysis_process':False})
print('COMPLETE_VERIFIED_PLAN_PENDING; nine original checkpoints unchanged; documents synchronized')
