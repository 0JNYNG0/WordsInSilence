"""Finalize verified experiment and synchronize the three authoritative documents."""
from common import *
import re

def main():
 assert read(U/'EXIT.json')['train_exit']==0 and not (U/'TRAIN_RUNNING.json').exists()
 r=read(U/'COMPARISON_INDEPENDENT.json');c=read(U/'CONCLUSION.json');ready=read(U/'READY.json');qa=read(U/'REPORT_LAYOUT_QA.json')
 assert r['status']=='PASS_SAVED_SCORES_ACTUAL_TRACE_AND_FROZEN_C' and r['review_gate']['pass'] and qa['pass']
 assert len(r['traces'])==3 and r['completed_epochs']['T3F']==[80,80,80]
 assert r['saved_predictions_verified']==5400 and c['actual_batches']==6960
 pins={}
 for f in ['PREP_CONTEXT.json','TRAIN_CONTEXT.json','READY.json','COMPARISON_INDEPENDENT.json']:
  for p,h in read(U/f)['pins'].items():
   assert p not in pins or pins[p]==h
   pins[p]=h
 check(pins)
 html=(U/'RESULT_KO.html').read_text(encoding='utf8')
 links=re.findall(r'(?:href|src)="([^"]+)"',html)
 assert all((U/x).is_file() for x in links)
 assert not (U/'FIRST_TRAIN_ERROR.json').exists()
 timestamp=now();unit=U.name
 next_action='T3-F 3회 비교 결과를 사용자와 검토한다. 사전 공학 기준은 통과했으나 수영·별·백육십이 일부 회귀가 남고 반복 DEV 결과다. 다음은 모델을 고정한 별도 평가 계획을 합의한다. 추가 학습·시간요약 재탐색·운영 C/웹/DTW 교체는 진행하지 않는다. 사용자·멘토 자료는 평가 외 학습/튜닝/선택 금지.'
 summary='''ops/word100_frozen_c_temporal_20261019. 사용자 승인대로 각 seed의 학습된 C 체크포인트(77/76/75epoch)를 고정하고 기존 T3 시간요약 보조 가중치9,600개만 학습. C 가중치37,636개·BN통계·dropout eval 고정. 준비11검사 PASS, 초기 출력오차0, 기존2,398입력파일/4,196텐서 동일. 3회 모두80epoch완료·선택69/70/78. 실제6,960배치/215,760원본노출 로그·모든 배치 C해시·240epoch 기준출력변화0·선택/마지막 C텐서불변 검산. 저장 C/T3/T3-F 예측5,400건을 별도 산술 검산 PASS; 기준점수 재추론 없음. sensor C[295,299,294]→T3-F[296,299,298]/각300, 평균98.6667→99.2222%(+0.5556%p), 범위98.6667~99.6667%. masked C[294,299,293]→[295,299,297], 98.4444→99.0000%(+0.5556%p), 범위98.3333~99.6667%. T3대비 sensor+0.4444%p/masked+1.0000%p. C대비 sensor개선6/회귀1·masked개선8/회귀3(각900반복예측). 사전gate PASS이나개별회귀있음: 백육십이REAL12 seed13 양view, 수영REAL12 seed13 masked, 별REAL11 seed14 masked. 상처는각view C8→T3 6→T3-F8/9로C수준회복, 성남9→7→9/9. 재등장 sensor123/123,masked137/138; 원래C123/123·136/138. 후반61~80 인접epoch변동 평균 C sensor1.1228/masked1.2222%p→T3-F0.0175/0%p, T3동일후반구간부족으로3회비교미적용. 새후보는C사전학습출발이라freshT3대비단일변수인과실험아님. 반복DEV·선택사용자료이며새사용자/손모양정확도보장아님. 기존C/웹/DTW유지,사용자멘토자료미접근·보류ID14~18미사용. 같은Codex구현+별도산술/텐서검산,독립agent아님. 학습전문법오류·줄바꿈복사해시불일치 최초증거보존후수정,학습오류없음. 보고서sandbox렌더오류보존후일반환경렌더PASS,desktop/mobile가로넘침0·그림확인완료. RESULT_KO.html/txt,CONCLUSION,COMPARISON_INDEPENDENT,READY,FINAL_AUDIT. 추가학습/자동교체없음.'''
 context_path=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';context=context_path.read_text(encoding='utf8')
 h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
 assert context.count(h8)==1 and context.count(h9)==1
 heading8='### 2026-10-19 T3-F 동결 보조 학습 3회 비교 완료 — 평균 개선·기존 C 유지'
 assert heading8 not in context
 context=context.replace(h8,h8+'\n\n'+heading8+'\n'+summary,1)
 context=context.replace(h9,h9+'\n\n### 2026-10-19 T3-F 결과 검토 후 별도 평가 계획\n'+next_action,1)
 context_path.write_text(context,encoding='utf8')
 sp=O/'STATE.json';s=read(sp);s.update(updated_date_kst=timestamp[:10],current_stage='WORD100_FROZEN_C_TEMPORAL_COMPLETE_AWAITING_REVIEW',next_single_action=next_action)
 s[unit]={'at':timestamp,'status':'COMPLETE_AWAITING_USER_RESULT_REVIEW','authorization':'AUTHORIZATION.json','report':str(U/'RESULT_KO.html'),'audit':str(U/'FINAL_AUDIT.json'),'training_runs':3,'completed_epochs':[80,80,80],'selected_epochs':[69,70,78],'training_exit':0,'exec_session_finished':14764,'freeze_tests':11,'all_batch_frozen_hashes_preserved':True,'initial_output_max_error':0,'base_probe_max_error':0,'trainable_parameters':9600,'frozen_parameters':37636,'actual_batches':6960,'actual_source_exposures':215760,'independently_recounted_predictions':5400,'summary':r['summary'],'contrasts':r['contrasts'],'engineering_gate':r['review_gate'],'sensor_gains_losses_vs_C':[6,1],'masked_gains_losses_vs_C':[8,3],'remaining_errors':['C correct→T3-F wrong: sensor 백육십이/REAL12/seed13','masked 수영/REAL12/seed13,백육십이/REAL12/seed13,별/REAL11/seed14','상처 C 수준 회복이나 각view1/9오답 남음','DEV 반복/선택 사용, 새사용자 일반화 미검증'],'operator':'Codex local implementation and separate arithmetic/tensor verification; no independent agent','production_changed':False,'user_mentor_data_used':False,'next_single_action':next_action}
 write(sp,s)
 handoff=f'''# WordsInSilence 최신 인수인계

갱신: {timestamp}

현재 목표: 기존 C를 고정한 시간요약 보조 후보 T3-F의 3회 비교 완료. 사용자 결과 검토 대기.

- 단위: research/front_keyphase_v1/ops/{unit}
- 완료: 동결11검사 PASS → 보조9,600가중치만3회(각80epoch)학습 → C/T3 저장점수와 비교·별도검산. 선택epoch69/70/78. exec14764 종료0, 잠금없음·진행중학습없음. 재실행하지 않는다.
- sensor 정확도: C98.6667% / T398.7778% / T3-F99.2222%. T3-F 정답296/299/298(각300), 범위98.67~99.67%. C대비+0.5556%p.
- masked 정확도: C98.4444% / T398.0000% / T3-F99.0000%. 정답295/299/297, 범위98.33~99.67%. C대비+0.5556%p.
- 사전 공학gate PASS, 모든 C가중치·BN·기준출력 불변. 실제6,960배치/215,760원본노출·저장5,400예측 검산. 같은Codex 별도 산술/텐서검증이며 독립사람/agent아님.
- 개선/회귀: C대비 sensor오답→정답6/정답→오답1, masked8/3(각300영상×3모델). 상처·성남은T3회귀에서C수준회복. 수영masked1회·별masked1회·백육십이양view각1회 C정답을뒤집음. 그룹평균PASS가전영상보존을뜻하지않음.
- 준비문법/줄바꿈복사해시오류 및 sandbox렌더오류 증거보존후복구. 학습오류없음. 보고서desktop/mobile렌더·이미지확인PASS.
- 기존 C·웹·DTW 변경0. 사용자/멘토자료미접근·학습/튜닝/선택미사용, 보류ID14~18미사용. 원본입력/기존18체크포인트 해시보존.
- 보고: RESULT_KO.html/txt, CONCLUSION.json, COMPARISON_INDEPENDENT.json, FINAL_AUDIT.json. 반복DEV/모델선택에쓰인자료이므로새사용자성능아님. T3-F는사전학습C출발이라freshT3와한변수인과실험아님.

다음 한 작업: {next_action}

FROZEN_PROTOCOL은 이전 제안 내용을 상속하므로 일부 ‘아직 미구현’ 문구는 당시 계획 문맥이다. 현재 실행 권한/결과는 AUTHORIZATION, protocol상단 training_authorized_now=true, READY, EXIT, 본 최신 상태를 따른다. 동결 파일은 소급 수정하지 않았다.
'''
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
 current=read(U/'CURRENT_EXECUTION.json');current.update(status='COMPLETE',exit_code=0,finished_at=read(U/'EXIT.json')['finished_at']);write(U/'CURRENT_EXECUTION.json',current)
 event(stage='COMPLETE_AWAITING_USER_RESULT_REVIEW',runs=3,engineering_gate_pass=True,production_changed=False)
 # Parse and verify document consistency; no re-training or new inference.
 assert read(sp)[unit]['summary']==r['summary']
 assert unit in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
 assert '99.2222' in context_path.read_text(encoding='utf8')
 artifacts=['READY.json','PREP_CONTEXT.json','TRAIN_CONTEXT.json','COMPARISON_INDEPENDENT.json','CONCLUSION.json','RESULT_KO.html','RESULT_KO.txt','comparison.png','REPORT_LAYOUT_QA.json','report_preview_top.png','report_preview_scores.png','EXIT.json','CURRENT_EXECUTION.json','FIRST_SETUP_ERROR.json','FIRST_PREP_ERROR.json','PREP_RECOVERY.json','FIRST_RENDER_ERROR.txt','report.py','finalize.py']
 protected_checkpoints=[p for p in pins if p.endswith('_best.pt') and not str(U) in p]
 assert len(protected_checkpoints)==18
 audit={'at':now(),'status':'PASS_COMPLETE_KEEP_PRODUCTION_C','source_and_code_pins_verified':len(pins),'original_checkpoints_preserved':protected_checkpoints,'final_frozen_check':'READY, all6960 batch hashes,240epoch base probe,selected/last exact tensor equality; reused completed verification','results_verification_reused':'COMPARISON_INDEPENDENT.json','saved_predictions_verified':5400,'new_inference_during_finalization':0,'engineering_gate_pass':True,'individual_regressions_disclosed':{'sensor':1,'masked':3},'original_C_kept':True,'user_mentor_data_used':False,'heldout_signers_used':False,'no_more_training':True,'training_exit':0,'layout_qa':qa,'visual_review':'top,focus/regressions tables,andcomparison chart inspected; readable/no clipping','first_errors_preserved':['FIRST_SETUP_ERROR.json','FIRST_PREP_ERROR.json','PREP_RECOVERY.json','FIRST_RENDER_ERROR.txt'],'first_errors_resolved_before_completion':True,'training_errors':0,'protocol_inherited_proposal_wording':'historical unimplemented wording retained; current authorization explicit, no retrospective frozen edits','artifact_pins':{str(U/f):sha(U/f) for f in artifacts},'document_pins':{str(p):sha(p) for p in [sp,context_path,ROOT/'NEW_CHAT_HANDOFF.md']},'verification_operator':'Same Codex, separate arithmetic and tensor-level audit, not independent person/agent'}
 write(U/'FINAL_AUDIT.json',audit,new=True)
 print(json.dumps({'status':audit['status'],'protected_checkpoints':len(protected_checkpoints),'source_code_pins':len(pins),'documents_synchronized':3},ensure_ascii=False))

if __name__=='__main__':main()
