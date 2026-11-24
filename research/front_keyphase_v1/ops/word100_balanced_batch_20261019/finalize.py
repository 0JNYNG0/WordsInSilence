"""Final evidence checks and authoritative status synchronization; no model execution."""
from common import *
import re

def main():
 r=read(U/'COMPARISON_INDEPENDENT.json');m=read(U/'LATE_MATCHED_WINDOW_AUDIT.json');prep=read(U/'PREP_CONTEXT.json');ctx=read(U/'TRAIN_CONTEXT.json')
 assert not (U/'TRAIN_RUNNING.json').exists() and not list(U.glob('FIRST_*_ERROR.json'))
 assert len(list(U.glob('BAL31_*_RESULT.json')))==3
 check(prep['pins']);check(ctx['pins']);check(read(U/'READY.json')['pins']);check(r['pins']);check(m['pins'])
 assert r['summary']['C']['sensor']['top1_counts']==[295,299,294]
 assert r['summary']['BAL31']['sensor']['top1_counts']==[293,297,297]
 assert r['selected_epochs']['BAL31']==[79,76,65]
 assert [x['epochs'] for x in r['traces'].values()]==[80,80,77]
 for pair in m['paired']:
  for arm in ['C','BAL31']:
   hp=BASE/f"training_C_{pair['seed']}_history.json" if arm=='C' else U/f"BAL31_{pair['seed']}_history.json"
   h=[x for x in read(hp) if pair['window'][0]<=x['epoch']<=pair['window'][1]]
   for v in VIEWS:
    x=np.array([z[v+'_top1'] for z in h],dtype=float)/3;a=pair[arm][v]
    assert abs(float(np.mean(np.abs(np.diff(x))))-a['adjacent_mean_pp'])<1e-12
    assert abs(float(np.std(x))-a['sd_pp'])<1e-12
 html=(U/'RESULT_KO.html').read_text(encoding='utf-8')
 for link in re.findall(r'(?:src|href)="([^"]+)"',html):assert (U/link).is_file(),link
 assert '79·76·65' in (U/'RESULT_KO.txt').read_text(encoding='utf-8')
 at=now();key=U.name
 facts={'sensor_old_counts':[295,299,294],'sensor_new_counts':[293,297,297],
  'sensor_mean_old':r['summary']['C']['sensor']['accuracy_mean'],'sensor_mean_new':r['summary']['BAL31']['sensor']['accuracy_mean'],'sensor_delta_pp':r['contrasts']['sensor']['mean_pp'],
  'masked_delta_pp':r['contrasts']['masked']['mean_pp'],'matched_adjacent_change_old_pp':m['three_seed_mean']['C']['sensor']['adjacent_mean_pp'],'matched_adjacent_change_new_pp':m['three_seed_mean']['BAL31']['sensor']['adjacent_mean_pp'],
  'epochs_completed':[80,80,77],'selected_epochs':[79,76,65],'actual_batches':sum(x['batches'] for x in r['traces'].values()),'actual_source_exposures':sum(x['actual_source_exposures'] for x in r['traces'].values()),'review_gate_pass':r['review_gate']['pass']}
 audit={'at':at,'status':'COMPLETE_VERIFIED_NO_ADOPTION_C_RETAINED','facts':facts,'original_checkpoint_pins':read(O/'word100_confusion_late_training_diagnosis_20261019/FINAL_AUDIT.json')['original_checkpoint_pins'],
  'pins':{str(U/n):sha(U/n) for n in ['AUTHORIZATION.json','FROZEN_PROTOCOL.json','READY.json','PREP_CONTEXT.json','TRAIN_CONTEXT.json','train.py','balanced.py','verify_results.py','COMPARISON_INDEPENDENT.json','LATE_MATCHED_WINDOW_AUDIT.json','CONCLUSION.json','RESULT_KO.html','RESULT_KO.txt','late_accuracy.png']},
  'execution':{'preparation_session':92425,'training_session':57158,'training_exit':0,'verification_exit':0,'running':False},
  'verification_owner':'Codex local separate arithmetic, actual traces, initial states and saved checkpoint binding; no separate agent/expert',
  'model_inference_repeated':False,'user_mentor_data_used':False,'production_changed':False,'training_allowed_after_completion':False,
  'qualification':'Primary frozen61-80 window retained with coverage80/80/77; secondary overlap61-80,61-80,61-77 added for fair descriptive comparison, no retuning.'}
 write(U/'FINAL_AUDIT.json',audit,new=True)
 summary='ops/'+key+'. 동일 C 입력2398파일 해시/4196텐서, 3seed×80회 일정215760건 검증PASS. 899개→31×29, 원본·순서·29step·관측일정·모델·lr·선택규칙 고정. 새3회 학습80/80/77회 종료, 선택79/76/65회. sensor C[295,299,294]→[293,297,297]/300, 평균98.6667→98.5556%(-0.1111%p); masked98.4444→98.1111%(-0.3333%p), sensor top5양쪽99.6667%. 동등구간 보조 후반 인접변화1.1524→1.0475%p, 실행별1.6140→0.7193 / 0.9474→0.9649 / 0.8958→1.4583%p. 3회째77조기종료여서 같은C도61~77로 비교; 원래61~80 결과 보존. 일관된 인식·안정화 개선 없음, 미채택·C/웹/DTW유지. 저장3600예측 산술검산 및 실제6873batch/213063노출·checkpoint·선택epoch검산PASS. 사용자/멘토미사용. 기존15checkpoint 보존. 별도산술은 같은Codex, 독립에이전트/사람 아님. RESULT_KO.html/txt·COMPARISON_INDEPENDENT·LATE_MATCHED_WINDOW_AUDIT·FINAL_AUDIT.'
 next_action='균등 배치3회 결과를 사용자와 검토한다. 현재C 유지, 새조건미채택. 후속 제안은 반복 혼동의 동작 단계·순서 보존을 위한 공통 시간 구간 요약 후보 설계(효과미확정); 입력·비교계획 확인 전 추가학습/운영교체 없음. 사용자·멘토 자료 학습/튜닝/선택 제외.'
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig')
 pos=t.index('\n',t.index('## 8. 현재 상태'));t=t[:pos+1]+'\n### 2026-10-19 균등 배치3회 비교 완료 — 미채택·기존C유지\n'+summary+'\n\n'+t[pos+1:]
 pos=t.index('\n',t.index('## 9. 다음 단일 작업'));t=t[:pos+1]+'\n### 2026-10-19 균등 배치 결과 검토\n'+next_action+'\n\n'+t[pos+1:];p.write_text(t,encoding='utf-8')
 p=O/'STATE.json';s=read(p);s.update(current_stage='WORD100_BALANCED_BATCH_COMPLETE',activity_state='AWAITING_USER_RESULT_REVIEW',updated_date_kst=at,next_single_action=next_action)
 s[key]={'status':audit['status'],'at':at,'unit':key,'facts':facts,'training_completed':True,'new_runs':3,'training_running':False,'training_allowed':False,'production_changed':False,'user_mentor_data_used':False,'existing_C_preserved':True,'training_session':57158,'training_exit_code':0,'verification':'PASS saved scores / actual trace / checkpoint binding; same Codex separate arithmetic','artifacts':['RESULT_KO.html','RESULT_KO.txt','READY.json','COMPARISON_INDEPENDENT.json','LATE_MATCHED_WINDOW_AUDIT.json','FINAL_AUDIT.json'],'next_single_action':next_action};write(p,s)
 handoff=f'''# WordsInSilence 최신 인수인계

갱신: {at}

현재 완료: ops/{key}. 사용자 승인899개→31개씩29묶음 검증 및 새3seed 학습·기존C 비교 완료. 개선후보 미채택, C·웹·DTW 유지.

- 입력2398파일 해시동일,4196텐서 검증,3seed×80회215760일정 검산PASS. 실제 학습자료899와DEV300 분리, 초기값/순서/관측조건/모델/lr0.001/CE선택/earlystop12 유지. eval묶음32 유지. 변경은 학습 배치 경계뿐.
- seed20261013/14/15: 학습80/80/77, 선택79/76/65. C sensor295/299/294→새293/297/297(각300), 평균98.67→98.56%(-0.11%p). masked98.44→98.11%(-0.33%p), sensorTop5동일99.67%. 2seed정확도하락/1seed개선, gateFAIL. 통계적열등확정이 아니라 개선근거부족.
- 고정61~80후반 실제기록 인접변화1.1228→1.0475%p(새3회째17회기록). 공정한 보조동일구간:61~80/61~80/61~77, 평균1.1524→1.0475%p(-약9.1%). 하지만1.61→0.72 /0.95→0.96 /0.90→1.46%p로 세번째악화. 후반SD1.06→1.03%p. 일관된안정화아님. 마지막3개배치만의문제라는설명지지못함, BN단독인과판정불가.
- 추가학습0(승인3회외없음), 저장점수재추론0. 3600저장예측 산술 및 실제6873batch/213063source노출·selectedcheckpoint 연결검산PASS. 같은Codex 별도계산, 독립에이전트/전문가아님.
- 사용자/멘토촬영 분석·학습·튜닝·선택0, 카메라0, 새추출0, 다운로드0, 외부전송0, Git0. 기존15checkpoint 해시보존. 원본/sources/W: 및 이전동결단위변경없음.
- 준비exec92425종료0, 학습exec57158종료0, 검산종료0. 실행잠금/오류없음. 새3회째77회종료는고정patience12정상종료. 재실행/재채점금지.
- RESULT_KO.html/txt, late_accuracy.png/svg, READY, COMPARISON_INDEPENDENT, LATE_MATCHED_WINDOW_AUDIT, FINAL_AUDIT 참조. HTML은보고서, 운영웹수정아님.

다음 한 작업: {next_action}
'''
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
 assert read(O/'STATE.json')[key]['facts']==facts
 for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md']:assert key in p.read_text(encoding='utf-8')
 write(U/'DOCUMENT_SYNC.json',{'at':now(),'status':'PASS','pins':{str(p):sha(p) for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']},'current_stage':s['current_stage']},new=True)
 event(stage='COMPLETE_VERIFIED_C_RETAINED',new_runs=3,sensor_delta_pp=facts['sensor_delta_pp'],production_changed=False)
 print(json.dumps(facts,ensure_ascii=False))
if __name__=='__main__':main()
