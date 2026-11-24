"""Audit and synchronize the completed authorized diagnosis, never runs models."""
import json,hashlib,re
from pathlib import Path
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
at=datetime.now(timezone(timedelta(hours=9))).isoformat();f=read(U/'FROZEN.json');a=read(U/'ANALYSIS.json')
assert read(U/'VERIFY.json')['status']=='PASS'
assert read(U/'AGGREGATE_CHECK.json')['status']=='PASS'
ir=read(U/'INTERPRETATION_REVIEW.json')
assert not (U/'RUNNING.json').exists()
for p,h in f['protected'].items():assert sha(Path(p))==h,p
recovery=read(U/'RECOVERY.json')
for p,h in recovery['immutable'].items():assert sha(Path(p))==h,p
runtime=read(U/'RUNTIME_EXPLANATION.json')
for p,h in runtime['source_sha256'].items():assert sha(Path(p))==h,p
html=(U/'RESULT_KO.html').read_text('utf-8')
links=re.findall(r'(?:src|href)="([^"]+)"',html)
for link in links:assert (U/link).is_file(),link
write(U/'VISUAL_REVIEW.json',{'at':at,'status':'PASS','actor':'parent Codex','images':{p.name:sha(p) for p in (U/'visuals').glob('*.png')},'observed':'All four figures viewed via view_image. Korean labels, legends, values and limits readable; no clipping seen. Report HTML responsive structure/local links checked.','browser_rendered':False,'user_camera_access':False})
evidence=['FROZEN.json','RECOVERY.json','RESULT.json','VERIFY.json','FIXTURE_RESULT.json','ANALYSIS.json','AGGREGATE_CHECK.json','INTERPRETATION_REVIEW.json','RUNTIME_EXPLANATION.json','PROVENANCE_NOTE.json','VISUAL_REVIEW.json','RESULT_KO.html','RESULT_KO.txt','study.py','study_recovery.py','verify.py','analyze.py','report.py']
audit={'at':at,'status':'COMPLETE_DIAGNOSIS_RESEARCH_ONLY_NO_ADOPTION','scope':{'public_words':3,'public_people':9,'clips':27,'short_variants':81,'pair_methods':1296,'normal_pair_methods':2160},'protected_pins_verified':len(f['protected']),'recovery_immutable_pins_verified':len(recovery['immutable']),'prior_frozen_pins_independently_verified':read(U/'VERIFY.json')['pinned_old_files_checked'],'runtime_code_bank_pins_verified':len(runtime['source_sha256']),'first_error_preserved':True,'recovery_partial_cases_preserved':432,'sessions':{'original':{'id':63674,'exit':1},'recovery':{'id':43257,'exit':0},'independent_verify':{'id':91015,'exit':0},'analysis':{'chunk':'d448ff','exit':0},'report':{'chunk':'10f410','exit':0}},'actors':{'implementation_execution':'parent Codex','diagnostic_verifier':'phase_design_review (separate agent); parent executed','analysis_plots':'omission_summary (separate agent); parent executed','runtime_and_interpretation':'phase_evaluation_review (separate agent)','aggregate_visual_publication':'parent Codex'},'not_done':['semantic essential-stage adjudication','new training','model inference','new MediaPipe','new DTW search','user/mentor data access','operating model/web change','all100word generalization validation'],'pins':{n:sha(U/n) for n in evidence}}
write(U/'FINAL_AUDIT.json',audit)
next_action='사용자와 짧은 누락 원인 진단 및 운영 DTW/연구 참고범위의 차이를 공동 검토한다. 다음 제안은 여러 공개 참고를 각각 연속 동작으로 유지하고 짧은 구간의 앞·중간·뒤 순서와 관측 부족을 함께 비교하는 별도 후보 계획이다. 단일 참고 정답화·최소차이 참고만 선택·이웃 허용 일괄삭제·문턱 완화는 하지 않는다. 손 위치/모양 근거 분리, 고정 참고시간 및 원촬영 미관측 시간표를 유지한다. 정상 속도·손내림·한 손·교차·반복 대조와 짧은 제거 대조를 사전 고정하고 동일 자료에서 검증한 뒤 별도 공개 묶음으로 확인한다. 의미상 필수 단계 정답과 미검출 원인을 단정하지 않는다. C·T3-F·운영 웹 유지, 사용자·멘토 자료 접근/학습/튜닝/선택 금지. 추가 학습은 결과 공동 확인 후.'
entry='''### 2026-10-22 짧은 누락 원인 분리 진단 — 관측 부족·참고 범위·이웃 허용 영향 확인, 운영 미적용
- 단위 `word100_short_omission_cause_20261022`. 공개 TRAIN 3단어(수어·후원·멀다)×9명27영상, 짧은3입력프레임 제거81변형×8참고×2저장정렬=1296비교, 원속도+4속도 대조2160비교를 저장자료로 독립 검산. 새DP/검출/분류/학습0. 100단어 전체/의미상 필수누락 정답/실사용 정확도 평가 아님.
- 기존정렬 새차이없음12: 범위대조11·이웃대조10·경로대조0. 이전 위치·방향 보조정렬 새차이없음9: 범위6·이웃7·경로0(6중복,2미설명). 요인은 공통관측칸에서 조건변경 민감도이며 배타적 원인분할/개선 정확도 아님. 전체81에는 경로민감도11 있어 정렬 정상 증명도 아님. held 경로는 삭제전연결의 최근접 잔존시점 대조이지 정답정렬 아님.
- 새9변형 삭제전 입력손관측42/48손프레임, 신뢰각도375/720·손바닥56/96·팔96/96. 일부손관측없음2·일부각도부족7·각도전체부족2; 모든관절없음0. 미설명 후원REAL02/03 75%는 입력손관측6/6이나 각도0/90·손바닥0/12, 팔근거는남음. 입력observed는 usable&observed 손점 하나 이상이며 RGB가시성/원시21점 정답 아님.
- 새9 모두 해당구간 원본일부차이/관측부족이 있었고 삭제후8개에는 차이잔존. 1개만 해당구간8참고 차이0이나 unknown존재. ‘추가차이없음’을 ‘전체정상’으로 승격금지. 원본수치변화 존재는 의미상필수단계 증명 아님.
- 새저장경로 정상1080쌍: 비교가능칸 차이 pool+neighbor106465/1146665=9.28%, neighbor제거150827/동일분모=13.15%, singleanchor721413/1126099=64.06%. 조건별분모도다르고 언어오류율 아님. anybad1080/1080 포화라 경고영상수만으로 안전결론금지. 단일참고/이웃삭제 미채택.
- 고정참고범위·정상손내림마스크·원본좌표/교차 보존, 이전135누락검증 재사용. 새로운 손내림/교차 개선 실증 아님. 전단위 손내림→활동 연결증가121→156 문제 유지.
- 운영코드 정적감사: 12표현×9TRAIN참고 각자DTW 후 손별 유효참고 차이중앙값(최소5명), 시간/각도내평균·필요양손평균 별도. 가상평균관절 동작 아님. 연구8사람 시점별minmax와 다름·운영미적용. 서버실측/사용자기록 접근없음.
- 첫exec63674 손역할기본값 누락 종료1 보존. 기존상속 기본양손을복구하고432저장파일 동일검증/보존; recovery43257 종료0. 독립verify91015 종료0,9합성검산PASS, 별도해석68집계일치, 부모집계/4그림확인/보호32핀 통과. C·T3-F·웹·운영DTW불변. 보고서/근거U; 다음은 결과공동검토 후 연속참고구간 후보 계획, 추가학습미승인.
'''
paths=[R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']
for p in paths:assert sha(p)==sha(U/'before'/p.name),'Root changed concurrently: '+str(p)
before={str(p):sha(p) for p in paths}
ctx=paths[0].read_text('utf-8-sig')
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
assert ctx.count(h8)==1 and ctx.count(h9)==1
ctx=ctx.replace(h8,h8+'\n\n'+entry,1).replace(h9,h9+'\n\n### 2026-10-22 짧은 누락 원인 진단 공동 검토\n'+next_action+'\n',1)
paths[0].write_text(ctx,encoding='utf-8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: DTW의 정상 개인차 허용을 지키면서 짧은 차이와 관측 부족이 묻히는 원인을 구분한다. TCN 단어 인식 학습 작업이 아니다.
마지막 완료: `word100_short_omission_cause_20261022` 저장 공개자료 진단·독립검산·보고. 새차이없음9 중 넓은참고6/이웃확장7 민감도,2개미설명. 정상차이증가 때문에 단순기준축소 미채택. 기존 C·T3-F·운영웹 유지.
다음 한 작업: {next_action}

{entry}

구현/실행: 부모 Codex. 독립 수치 검산: phase_design_review 구현 verify.py/부모 실행. 집계·그림: omission_summary 구현 analyze.py/부모 실행. 운영 코드 및 해석 교차검토: phase_evaluation_review. 부모 별도 집계·시각확인·동기화 완료. 수어 전문가 의미검증 아님.
현재 실행/잠금 없음. 완료 진단/독립검산 재실행 금지. 원study.py의 첫오류는 보존했고 복구study_recovery.py로 완료했다. FROZEN/RECOVERY의 조건과 핀을 변경하지 않는다. FINAL_AUDIT 및 SYNC_CHECK 참고.
근거 폴더: research/front_keyphase_v1/ops/word100_short_omission_cause_20261022. RESULT_KO.html/txt, ANALYSIS, VERIFY, AGGREGATE_CHECK, INTERPRETATION_REVIEW, RUNTIME_EXPLANATION, PROVENANCE_NOTE, VISUAL_REVIEW.
원본/W:/sources/과거동결결과 읽기전용. 사용자·멘토촬영 접근/학습/튜닝/후보선택0. 추가 학습은 사용자 공동 확인 전 금지.
'''
paths[1].write_text(handoff,encoding='utf-8')
s=read(paths[2]);s.update(updated_date_kst=at,updated_at=at,current_stage='WORD100_SHORT_OMISSION_CAUSE_DIAGNOSED_NO_ADOPTION',next_single_action=next_action,resume_command=next_action)
s[U.name]={'status':'VERIFIED_CAUSE_SENSITIVITY_NO_ADOPTION','at':at,'scope':audit['scope'],'missed':{m:a['cohorts'][m+'_missed'] for m in ('old','new')},'normal':read(U/'AGGREGATE_CHECK.json')['normal_totals'],'report':str(U/'RESULT_KO.html'),'verification':str(U/'VERIFY.json'),'final_audit':str(U/'FINAL_AUDIT.json'),'next':next_action,'protected_models_preserved':True,'user_access':0,'training':0,'adopted':False,'meaning':'Overlapping diagnostic sensitivities, not semantic causes or recognition accuracy.'}
paths[2].write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(paths[2])['current_stage']=='WORD100_SHORT_OMISSION_CAUSE_DIAGNOSED_NO_ADOPTION'
for p in paths:assert U.name in p.read_text('utf-8-sig')
write(U/'SYNC_CHECK.json',{'at':at,'status':'PASS','before':before,'after':{str(p):sha(p) for p in paths},'latest_unit':U.name,'next_action':next_action,'no_pending_execution':True,'report_sha256':sha(U/'RESULT_KO.html'),'final_audit_sha256':sha(U/'FINAL_AUDIT.json')})
print('FINAL AUDIT AND SYNC PASS')
