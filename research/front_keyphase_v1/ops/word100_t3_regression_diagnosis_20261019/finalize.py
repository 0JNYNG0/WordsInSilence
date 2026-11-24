"""Record verified diagnosis; no scientific model/input edits."""
from pathlib import Path
import json,hashlib,datetime,re
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert not (U/'FINAL_AUDIT.json').exists()
a=read(U/'ARITHMETIC_VERIFICATION.json');assert a['status']=='PASS'
assert read(U/'REPORT_LAYOUT_QA.json')['pass']
assert not read(U/'NEXT_PLAN.json')['training_authorized_now']
pins={**read(U/'INPUT_ANALYSIS.json')['pins'],**read(U/'HEAD_DECOMPOSITION.json')['pins']}
for p,h in pins.items():assert sha(p)==h,p
assert len(pins)==691
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
 assert sha(p)==sha(U/'before'/p.name),'Concurrent root edit: '+str(p)
txt=(U/'RESULT_KO.html').read_text(encoding='utf8')
for link in re.findall(r'(?:src|href)="([^"]+)"',txt):
 if not link.startswith(('https:','http:','#')):assert (U/link).exists(),link
assert '5.13%' in txt and '97.44%' in txt and '추가 학습 없음' in txt
at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
next_action='T3 회귀 원인과 NEXT_PLAN.json의 T3-F(C 체크포인트·BN·dropout 고정, 시간보조head만 학습) 비교안을 사용자와 검토한다. 아직 후보 구현/추가학습 미승인·미실행. 검토 승인 후 동등성/동결 검증→조건부3seed 비교. C/웹/DTW 유지, 사용자·멘토자료 제외. 시간요약 계열 반복 재탐색 금지.'
entry='''
### 2026-10-19 T3 상처·재등장 회귀 진단 완료 — 기존 C 유지, 추가학습 없음
ops/word100_t3_regression_diagnosis_20261019. 공개 저장입력132개(DEV87+TRAIN45),동결T3내부값522건(87×2view×3seed),RGB9영상표본72장면+3영상12시점(일부중복) 검토. 저장logit재구성최대6.6757e-6·순위일치/NumPy별도산술PASS/691원본근거해시보존. 기존C/T3입력동일. T3global은원래C가아니라공동으로새학습된경로. 회귀20예측중global이미오답12/추가head정답뒤집음8(sensor4/3,masked8/5);동일영상·seed/view중복포함. 상처REAL12는재등장그룹아님,C2/3→T3 0/3양view,global도모두믿다;TRAIN01/DEV13과반복·마무리시점차이표본관찰,언어적정오미판정. sensor재등장123→121/123감소전부성남REAL13 2seed이며내린왼손공백;masked긴공백없어도같은영상2회오답. masked재등장136→133/138=수영-3/백삼십육-1/감히+1. 수영영역내겹친왼손주석관측masked2/39(5.13%)vs sensor38/39(97.44%);T3masked오답3/sensor정답3,C같은부족입력정답3. 수영global오답1/추가head뒤집음2로진행설명세분정정. 백삼십육은종료후내린손경계점1개재등장,감히는비슷한공백에도개선. 시간위치≠의미단계·누락단독원인아님;인과확정·새사용자보장불가. 전체sensor98.67→98.78(+.11%p)/masked98.44→98.00(-.44%p)기존결과유지. T3-F계획은이미학습된C·BN·dropout고정하고기존보조시간head만학습하는3seed공학비교안,아직구현/학습안함·사용자검토대기;추가점수최종회귀가능/원래freshT3대비단일변수인과실험아님. C/웹/DTW·기존자료보존,사용자멘토자료미접근. 같은Codex별도산술검산이며독립agent/전문가아님. RESULT_KO.html/txt,NEXT_PLAN,VISUAL_REVIEW,ARITHMETIC_VERIFICATION,FINAL_AUDIT. 학습0,새MediaPipe0,내부분해용eval522(기존평가파일교체/재선택없음). desktop/mobile보고서6그림·가로넘침0시각확인완료.

'''
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=p.read_text(encoding='utf8')
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';h9='## 9. 다음 단일 작업\n';assert s.count(h8)==s.count(h9)==1
s=s.replace(h8,h8+entry,1).replace(h9,h9+'\n### 2026-10-19 T3 회귀 진단·후속안 검토\n'+next_action+'\n\n',1);p.write_text(s,encoding='utf8')
state=read(O/'STATE.json');state.update(updated_date_kst=at,current_stage='WORD100_T3_REGRESSION_DIAGNOSIS_COMPLETE_C_RETAINED',activity_state='AWAITING_USER_PLAN_REVIEW_NO_TRAINING',next_single_action=next_action)
state[U.name]={'status':'COMPLETE_PUBLIC_DIAGNOSIS_NO_TRAINING','at':at,'unit':U.name,'operator':'same Codex; separate NumPy/stored-score arithmetic; not an independent agent/expert','public_input_clips':132,'DEV_clips_decomposed':87,'fixed_model_feature_passes':522,'checkpoint_values_changed':False,'max_saved_logit_error':6.67572021484375e-6,'source_pins_verified':691,'regressions':{'sensor':{'total':7,'global_already_wrong':4,'extra_head_flip':3},'masked':{'total':13,'global_already_wrong':8,'extra_head_flip':5}},'sensor_reentry_C_T3_correct':[123,121],'sensor_reentry_denominator':123,'masked_reentry_C_T3_correct':[136,133],'masked_reentry_denominator':138,'scope_limit':'public repeatedDEV; sampledRGB not expert all-frame labels; causal reason not fully isolated','new_training_runs':0,'new_detector_runs':0,'user_mentor_data_used':False,'production_changed':False,'proposal':'T3-F frozen trained C + existing phase head; unimplemented/unapproved','report':str(U/'RESULT_KO.html'),'next_single_action':next_action};write(O/'STATE.json',state)
handoff=f'''# WordsInSilence 최신 인수인계

갱신: {at}

완료: word100_t3_regression_diagnosis_20261019. 사용자 요청 T3 상처·재등장 회귀 공개자료 원인 분석과 다음 계획 보고. 기존 C 유지, 추가 학습 0.

- 공개 저장입력132개(DEV87+관련TRAIN45)·동결T3내부값522건·원본RGB9영상표본+3영상특정시점 확인. 새로운 MediaPipe/카메라/사용자·멘토자료/보류ID 접근 없음.
- C/T3 입력동일. global은이미학습된C동결경로가아니라T3와공동으로새학습된경로. T3전체를C기억그대로+추가head로설명하지말것.
- 회귀20예측=global이미오답12+추가시간점수정답뒤집음8. sensor4+3/ masked8+5. 동일영상의seed·view중복포함,독립영상20개아님.
- 상처REAL12 원래재등장그룹아님. C2/3→T3 0/3양view,global-only도믿다3/3. 표본상오른손→왼손뒤마무리vsTRAIN01/DEV13재차오른손. 언어적정오판정안함. sensor이동중심TRAIN우41.0~47.7/좌41.1~57.8% 대대상26.8/73.1%;시간의존가능성이나인과확정아님.
- sensor재등장123→121/123 감소전부성남REAL13 2seed. 내려놓은왼손공백,masked긴공백없어도같은영상2회오답.
- masked재등장136→133/138:수영-3/백삼십육-1/감히+1. 수영주석왼손관측2/39vs sensor38/39,영역내겹침관측차이. T3global오답1/extrahead뒤집음2로진행설명정정. 백삼십육종료뒤왼손경계점1개재등장. 감히는비슷한공백에도개선.
- 전체기존결과sensor C98.67→T3 98.78%(+.11%p),masked98.44→98.00%(-.44%p)동일. 새사용자성능아님,3seed는3명이아님.
- 검산:저장logit재구성max6.6757e-6·순위일치/NumPy기여·margin합/691근거해시PASS. 같은Codex별도산술검산,독립agent/전문가아님. 완료평가파일·모델재선택없음. 기존15checkpoint포함보존. 분석eval522은수행했으므로추론0으로쓰지말것.
- 산출물RESULT_KO.html/txt,INPUT_ANALYSIS,HEAD_DECOMPOSITION,SUMMARY,TIME_POSITION_AUDIT,VISUAL_REVIEW,ARITHMETIC_VERIFICATION,NEXT_PLAN,CONCLUSION,FINAL_AUDIT. 로컬렌더링1280/420가로넘침0·그림6확인,top/점수표시각확인.
- T3-F 제안:학습된C를각seed별고정(proj/blocks/head/BN/eval),기존9600보조시간head만학습. 초기동등성·동결검증후조건부3seed. 아직구현/학습안함. 기존freshT3와단일변수인과실험아님. 추가점수로최종회귀가능,상승보장없음. 또회귀하면시간요약계열보류;DEV가중치재탐색금지.
- 현재진행세션/학습잠금없음. 완료분석반복금지. 원본/W:/sources/기존모델/웹/DTW변경0. 카메라/다운로드/외부전송/Git0.

다음 한 작업: {next_action}
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
protected=read(O/'word100_temporal_phase_training_20261019/FINAL_AUDIT.json')['original_checkpoint_pins']
for p,h in protected.items():assert sha(p)==h
artifacts={str(p):sha(p) for p in U.rglob('*') if p.is_file() and 'before' not in p.parts and 'public_video_cache' not in p.parts}
roots={str(p):sha(p) for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']}
write(U/'FINAL_AUDIT.json',{'at':at,'status':'COMPLETE_PUBLIC_DIAGNOSIS_KEEP_C','artifact_pins':artifacts,'root_document_pins':roots,'original_checkpoint_pins':protected,'source_pins_verified':len(pins),'arithmetic_status':'PASS_SAME_OPERATOR_SEPARATE_CALCULATION','report_visual_inspected':['report_preview_top.png','report_preview_scores.png','timing_plot.png','9 initial contact sheets','3 paired contact sheets'],'new_training':0,'new_MediaPipe':0,'fixed_T3_diagnostic_forward_passes':522,'original_saved_scores_overwritten':False,'user_mentor_accessed':False,'production_changed':False,'running_processes_from_this_unit':[],'next_single_action':next_action})
for p,h in roots.items():assert sha(p)==h
assert read(O/'STATE.json')['next_single_action']==next_action
print('Finalized public diagnosis. C retained; no additional training; next plan awaiting user review.')
