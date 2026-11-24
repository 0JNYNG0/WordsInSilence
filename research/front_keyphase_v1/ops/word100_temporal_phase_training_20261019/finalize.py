"""Finalize verified evidence and synchronize authoritative handoff documents."""
from common import *
import re

assert not (U/'FINAL_AUDIT.json').exists() and not (U/'TRAIN_RUNNING.json').exists()
r=read(U/'COMPARISON_INDEPENDENT.json');a=read(U/'ALL_CLASS_ANALYSIS.json');ready=read(U/'READY.json');con=read(U/'CONCLUSION.json')
assert r['status']=='PASS_SAVED_SCORES_AND_ACTUAL_TRACES' and ready['status']=='PASS' and not con['review_gate']['pass']
assert read(U/'EXIT.json')['train_exit']==0 and read(U/'REPORT_LAYOUT_QA.json')['pass']
check(ready['pins']);check(read(U/'TRAIN_CONTEXT.json')['pins']);check(r['pins']);check(a['pins'])
protected=read(O/'word100_balanced_batch_20261019/FINAL_AUDIT.json')['original_checkpoint_pins'];check(protected)
assert len(a['all_classes'])==100
for v in VIEWS:
 for arm in ['C','T3']:
  assert sum(x[v][arm] for x in a['all_classes'])==sum(x['correct'] for x in r['metrics'][arm][v])
  assert sum(x['count'] for x in a['confusions'][v][arm])==900-sum(x['correct'] for x in r['metrics'][arm][v])
 assert sum(1 if x['change']=='gain' else -1 for x in a['paired_events'][v])==sum(x[v]['delta_correct'] for x in a['all_classes'])
assert r['summary']['C']['sensor']['top1_counts']==[295,299,294] and r['summary']['T3']['sensor']['top1_counts']==[296,297,296]
assert r['summary']['T3']['masked']['top1_counts']==[295,296,291]
html=(U/'RESULT_KO.html').read_text(encoding='utf8')
for link in re.findall(r'(?:href|src)="([^"]+)"',html):
 if not link.startswith(('https:','http:','#')):assert (U/link).exists(),link
assert '98.67%' in html and '98.78%' in html and '98.44%' in html and '98.00%' in html
assert '해당 구간 없음 · 43회 종료' in html and '해당 구간 없음 · 59회 종료' in html
at=now();next_action='T3 3회 비교 결과를 사용자와 검토한다. sensor 평균+0.11%p이나 masked -0.44%p 및 촬영자/재등장그룹 회귀로 미채택, 기존 C/웹/DTW 유지. 새 실험·추가학습·운영교체는 별도 계획 검토 후 진행. 사용자/멘토자료 제외.'
state=read(O/'STATE.json');state.update(updated_date_kst=at,current_stage='WORD100_TEMPORAL_PHASE_TRAINING_COMPLETE_C_RETAINED',activity_state='AWAITING_USER_RESULT_REVIEW',next_single_action=next_action)
state[U.name]={'status':'COMPLETE_NOT_ADOPTED_KEEP_C','at':at,'unit':U.name,'operator':'Codex local implementation/execution + separate arithmetic by same operator; no independent agent/expert','train':899,'dev':300,'new_runs':3,'seeds':[20261013,20261014,20261015],'completed_epochs':[43,67,59],'selected_epochs':[31,55,47],'sensor_counts_C':[295,299,294],'sensor_counts_T3':[296,297,296],'sensor_mean_C':r['summary']['C']['sensor']['accuracy_mean'],'sensor_mean_T3':r['summary']['T3']['sensor']['accuracy_mean'],'sensor_delta_pp':r['contrasts']['sensor']['mean_pp'],'masked_mean_C':r['summary']['C']['masked']['accuracy_mean'],'masked_mean_T3':98.0,'masked_delta_pp':r['contrasts']['masked']['mean_pp'],'review_gate_pass':False,'ready_pass':True,'initial_output_difference':0,'verified_saved_predictions':3600,'actual_batches':4901,'actual_source_exposures':151931,'late_three_seed_comparison':'unavailable: two runs stop before61 by unchanged patience12','baseline_C_reinference':0,'completed_scores_reinference':0,'user_mentor_data_used':False,'production_changed':False,'report':str(U/'RESULT_KO.html'),'next_single_action':next_action}
write(O/'STATE.json',state)
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=ctx.read_text(encoding='utf8')
entry='''
### 2026-10-19 T3 시간 단계요약 3회 비교 완료 — 미채택·기존 C 유지
ops/word100_temporal_phase_training_20261019. 승인계획대로 기존 C global요약 + soft 앞/중간/뒤 대비96차원/zero-init head9600 구현. 동일 입력2398파일/4196텐서·초기3seed 출력오차0/RNG·시간/패딩/미검출/기울기 검증9종 PASS. 원래32×28+3배치/동일899TRAIN·300DEV/학습률·CE선택 유지. 새3회 43/67/59회 정상조기종료, 선택31/55/47. sensor C[295,299,294]→T3[296,297,296]/300, 평균98.6667→98.7778%(+0.1111%p); masked98.4444→98.0000%(-0.4444%p). sensor900예측 순증1, masked순감4. 동대문/백육십이/흡연실 개선, 상처/성남시수어통역센터 등 회귀. ID13·재등장그룹 회귀로 사전gate미통과. 후반61~80은2실행이61전종료하여3회안정성결론불가; 2회째만61~67보조비교. 저장3600예측/실제4901배치·151931노출/선택epoch·checkpoint 별도검산PASS. 기존15checkpoint 보존. C/웹/DTW 유지, 사용자/멘토자료미접근. 같은Codex 별도산술검증이며 독립사람/agent 아님. 상대시간요약≠의미단계, 추가파라미터효과와순서효과분리불가. RESULT_KO.html/txt·COMPARISON_INDEPENDENT·ALL_CLASS_ANALYSIS·FINAL_AUDIT. 학습오류없음; 보고서headless sandbox실행실패증거보존후 일반환경렌더링복구, desktop/mobile/그림확인PASS.

'''
s=s.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n'+entry,1)
s=s.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n### 2026-10-19 T3 비교 결과 검토\n'+next_action+'\n\n',1);ctx.write_text(s,encoding='utf8')
handoff=f'''# WordsInSilence 최신 인수인계

갱신: {at}

완료: word100_temporal_phase_training_20261019. 사용자 승인 T3 구현→검증→3회 학습→저장 C 비교→산술 검증·보고까지 완료. 기존 C 유지, T3 미채택.

- T3: 기존 C 입력·trunk·global head 유지 + 상대시간 삼각가중 앞/중간/뒤 대비96차원, bias없는 zero-init head9600. 전체37636→47236. 의미단계검출/손누락복원 아님.
- 입력2398파일/4196텐서·3seed 초기output오차0/RNG·시간/패딩/미검출/기울기9종PASS. 공개899TRAIN(1~9)/300DEV(11~13), 원래32×28+3배치·학습률·80/patience12·CE선택 동일. 새추출 없음.
- 학습 session95506 exit0; 43/67/59회 완료, 선택31/55/47회. 준비 session94793 exit0. 실행/잠금없음. 완료 run/점수 재실행금지.
- sensor C295/299/294→T3 296/297/296(각300), 평균98.67→98.78%(+0.11%p), 2seed개선1회귀. masked C294/299/293→295/296/291,98.44→98.00%(-0.44%p). ID13·중간재등장그룹 회귀로gateFAIL.
- sensor동대문7→9/9,백육십이8→9,흡연실8→9 개선;상처8→6,성남시수어통역센터9→7 회귀. 3영상×3seed이고9명아님. 새사람/웹캠정확도 아님.
- 후반61~80은1/3회째61전종료;3회평균안정성비교불가. 2회째61~67만 C0.9444→T3 0.4444%p 인접평균변화,전체안정화증거아님. 조기종료는12회CE무개선규칙, 실행오류아님.
- 저장3600예측/실제4901batch·151931원본노출/선택epoch·checkpoint 별도검산PASS. 실제trace로그 있음. 기존15checkpoint 해시보존. 추가headnonzero/finite 확인. 초기동등성·최종score 재추론없음.
- 사용자/멘토자료미접근/학습·튜닝·선택미사용. 기존 C/웹/DTW변경0, 카메라/다운로드/외부자료전송/Git0. 구현·검산 같은Codex,독립사람/agent검증아님.
- 보고서 RESULT_KO.html/txt, READY,FROZEN_PROTOCOL,COMPARISON_INDEPENDENT,LATE_MATCHED_WINDOW_AUDIT,ALL_CLASS_ANALYSIS,CONCLUSION,FINAL_AUDIT. 보고서만 첫headless sandbox오류 보존 후 일반환경 재실행성공;desktop1280/mobile420 가로넘침0,그림2정상,시각확인완료.

다음 한 작업: {next_action}
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
event(stage='COMPLETE_NOT_ADOPTED_KEEP_C',runs=3,sensor_delta_pp=r['contrasts']['sensor']['mean_pp'],masked_delta_pp=r['contrasts']['masked']['mean_pp'])
artifact_pins={str(p):sha(p) for p in U.iterdir() if p.is_file() and p.suffix in ('.py','.json','.html','.txt','.pt','.png','.svg','.cjs') and p.name!='FINAL_AUDIT.json'}
root_pins={str(p):sha(p) for p in [ctx,O/'STATE.json',ROOT/'NEW_CHAT_HANDOFF.md']}
write(U/'FINAL_AUDIT.json',{'at':at,'status':'COMPLETE_VERIFIED_C_RETAINED','implementation_and_arithmetic_operator':'same Codex','artifact_pins':artifact_pins,'root_document_pins':root_pins,'original_checkpoint_pins':protected,'training_errors':0,'report_render_error_preserved_and_recovered':True,'report_visual_inspected':['report_preview_top.png','report_preview_changes.png','late_accuracy.png'],'new_model_training_runs':3,'baseline_retrained_or_reinferred':False,'user_mentor_data_used':False,'production_changed':False,'no_running_training_lock':True,'next_single_action':next_action},new=True)
check(root_pins);check(protected)
for p in [ctx,ROOT/'NEW_CHAT_HANDOFF.md']:assert U.name in p.read_text(encoding='utf8')
assert read(O/'STATE.json')['current_stage']=='WORD100_TEMPORAL_PHASE_TRAINING_COMPLETE_C_RETAINED'
print('Completed and synchronized: C retained; T3 gate failed.')
