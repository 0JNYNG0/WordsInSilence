"""Finalize completed six-run comparison and synchronize current project state."""
from common import *
import re
def main():
 assert not (U/'TRAIN_RUNNING.json').exists() and not (U/'PREP_RUNNING.json').exists()
 assert not (U/'FINAL_AUDIT.json').exists();result=read(U/'REPORT_DATA.json');verified=read(U/'COMPARISON_INDEPENDENT.json');interpret=read(U/'INTERPRETATION.json')
 assert result['six_new_runs_complete'] and len(verified['traces'])==6
 check(read(U/'TRAIN_CONTEXT.json')['pins']);check(read(U/'PREP_CONTEXT.json')['pins']);check(read(U/'READY.json')['pins'])
 models={}
 for a in ['C','P','Q','R10','R01']:
  for seed in [20261013,20261014,20261015]:
   prefix=BASE/f'training_C_{seed}' if a=='C' else PREV/f'{a}_{seed}' if a in ['P','Q'] else U/f'{a}_{seed}'
   r=read(str(prefix)+'_RESULT.json');ck=Path(str(prefix)+'_best.pt');assert sha(ck)==r['checkpoint_sha256'];models[str(ck)]=sha(ck)
 page=(U/'RESULT_KO.html').read_text(encoding='utf8')
 for href in re.findall(r'href="([^"]+)"',page):assert (U/href).is_file(),href
 for p in U.glob('*.json'):read(p)
 at=now();key=U.name;statepath=O/'STATE.json';state=read(statepath)
 next_action='완료한6회 비교와원인해석을사용자와검토. 기존C·웹·DTW유지,교차조건은진단용미배포. INTERPRETATION의다음계획은제안이며추가학습승인없음. 완료학습/채점재실행·사용자/멘토튜닝금지.'
 state.update(updated_date_kst=at,updated_at=at,current_stage='WORD100_ASSIGNMENT_SCALE_FACTORIAL_COMPLETE',activity_state='AWAITING_USER_RESULT_REVIEW',next_single_action=next_action)
 state[key].update(status='COMPLETE_VERIFIED_DIAGNOSTIC_C_RETAINED',at=at,running=False,training_allowed=False,training_runs=6,training_exit_code=0,
  training_exec_session=16208,actual_epochs=result['actual_epochs'],actual_batches=result['actual_batches'],actual_source_exposures=result['actual_source_exposures'],summary=result['summary'],contrasts=result['contrasts'],interaction=result['interaction'],
  result=str(U/'RESULT_KO.html'),comparison=str(U/'COMPARISON_INDEPENDENT.json'),interpretation=str(U/'INTERPRETATION.json'),user_data_used=False,production_changed=False,next_single_action=next_action)
 state['word100_assignment_normalization_diagnosis_20261019']['approved_followup']=key;write(statepath,state)
 sm=result['summary'];numbers=' / '.join(f"{a} {sm[a]['sensor']['accuracy_mean']:.4f}%" for a in ['C','R10','R01','Q']);nm=' / '.join(f"{a} {sm[a]['masked']['accuracy_mean']:.4f}%" for a in ['C','R10','R01','Q'])
 heading8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';heading9='## 9. 다음 단일 작업';p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=p.read_text(encoding='utf8')
 entry8=f'''### 2026-10-19 손 배정·크기 분리 6회 비교 완료 — 진단용, C 유지
ops/{key}. R10(Q배정+C유효분모)/R01(C배정+Q유효분모)각3seed총6회완료. {result['actual_epochs']}epoch/{result['actual_batches']}실제batch/{result['actual_source_exposures']}원본노출. 원C초기화·일정·모델·CE선택동일,기존C/Q저장각3회재사용·재추론없음. sensor평균 {numbers}; masked {nm}. {interpret['summary']} 입력8392/155821120채널검산(마스크/global/시간/구간불변),실제배치/선택epoch/checkpoint/logit별도검산PASS. 양쪽분모유효할때만교환,미관측원래유지. 교차조건운영미적용. 기존C/웹/DTW/사용자자료불변. 작성/별도산술검산같은Codex,독립사람평가아님. 결과RESULT_KO.html,COMPARISON_INDEPENDENT.json,INTERPRETATION.json.
'''
 entry9=f'''### 2026-10-19 분리 비교 결과 검토·추가 학습 대기
{next_action} 단위ops/{key}. 손크기공식이아니라관측정정전후분모교환실험임을구분. 개별조건의개발정확도상승을새사람일반화나원인기여율로보고하지않는다. 새후보는자료·판정조건을먼저동결하고사용자계획확인후진행.
'''
 txt=txt.replace(heading8,heading8+'\n\n'+entry8,1).replace(heading9,heading9+'\n\n'+entry9,1);p.write_text(txt,encoding='utf8')
 h=f'''# WordsInSilence 최신 인수인계

갱신: {at}

현재 완료: ops/{key}. 승인된두교차조건각3회총6회학습·기존C/Q저장비교·별도검산완료. 추가학습/운영교체없이사용자결과검토대기.

- R10=Q관측/배정+C유효손크기, R01=C관측/배정+Q유효손크기. 크기공식은동일하며계산된분모만교환. 양쪽유효일때만교환,한쪽없으면각관측원래미검출/크기유지. 숨은좌표보충/단어별예외없음.
- 공개TRAIN899/DEV300,100단어. 원래C와초기화/순서/4epoch일정/280채널/37636파라미터/최적화기/최대80·patience12/평균DEV CE선택동일. 실행환경동일. 실제{result['actual_epochs']}epoch/{result['actual_batches']}batch/{result['actual_source_exposures']}원본노출.
- sensor3seed평균 {numbers}. masked {nm}. 세부seed/사람/그룹/교호작용은COMPARISON_INDEPENDENT.json.
- {interpret['summary']}
- 입력전수8392/155821120채널검산,마스크·globalXY·시간·구간불변. 독립코드저장logit/실제배치ID/선택epoch/모델해시검산PASS. 같은Codex별도산술,독립사람/에이전트검토아님. C/Q완료점수재추론0.
- prepare20625종료0,verify80268종료0,train16208종료0. 현재잠금/실행없음. 운영C/웹/DTW/사용자·멘토자료불변. 촬영자료학습·튜닝·선택0,카메라/다운로드/외부전송/Git0.
- RESULT_KO.html/txt,REPORT_DATA.json,INTERPRETATION.json,FINAL_AUDIT.json.

다음한작업: {next_action}
'''
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(h,encoding='utf8');ex=read(U/'CURRENT_EXECUTION.json');ex.update(stage='COMPLETE_VERIFIED',training_exit_code=0,running=False);write(U/'CURRENT_EXECUTION.json',ex)
 assert read(statepath)[key]['training_runs']==6 and not read(statepath)[key]['training_allowed']
 write(U/'FINAL_AUDIT.json',{'status':'COMPLETE_VERIFIED_C_RETAINED','at':at,'models_checked':models,'original_C_P_Q_checkpoints_unchanged':9,'new_runs_verified':6,'saved_original_scores_reinferred':False,'user_data_used':False,'production_changed':False,
  'result_pins':{n:sha(U/n) for n in ['RESULT_KO.html','RESULT_KO.txt','REPORT_DATA.json','INTERPRETATION.json','COMPARISON_INDEPENDENT.json','EFFECT_DETAILS.json','effect_details.py','INPUTS_INDEPENDENT.json','READY.json','FROZEN_PROTOCOL.json','TRAIN_CONTEXT.json']},
  'synced_document_pins':{str(x):sha(x) for x in [p,ROOT/'NEW_CHAT_HANDOFF.md',statepath]},'first_errors':[x.name for x in U.glob('FIRST*')],'running':False})
 print('COMPLETE_VERIFIED_C_RETAINED')
if __name__=='__main__':main()
