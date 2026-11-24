"""Close the verified, one-time public held-out comparison and sync current state."""
from common import *
import re

def main():
 assert read(U/'PREP_EXIT.json')['exit_code']==0
 recovery=read(U/'EVALUATION_RECOVERY.json')
 assert read(U/'EVAL_EXIT.json')['exit_code']==1 and recovery['status']=='PASS_SAVED_RESULTS_COMPLETE_STATUS_ONLY_RECOVERY' and recovery['new_inference']==0
 check(recovery['score_pins']);assert sha(U/'EVAL_EXIT.json')==recovery['original_exit_sha256']
 assert not any((U/f).exists() for f in ['PREP_RUNNING.json','EVAL_RUNNING.json','FIRST_PREP_ERROR.json'])
 ctx=read(U/'EVAL_CONTEXT.json');check(ctx['pins'])
 model_source=read(U/'MODEL_SOURCE_AUDIT.json');assert model_source['status']=='PASS_ORIGINAL_TRAINING_SOURCE_UNCHANGED';check(model_source['pins'])
 vi=read(U/'INPUTS_VERIFIED.json');r=read(U/'COMPARISON_INDEPENDENT.json');c=read(U/'CONCLUSION.json');qa=read(U/'REPORT_LAYOUT_QA.json')
 assert vi['status']=='PASS' and vi['records']==200 and r['status']=='PASS_SAVED_SCORE_RECOUNT' and r['predictions_verified']==2400 and qa['pass']
 check(vi['record_pins']);check(r['pins']);check(vi['verifier_pins'])
 assert c['summary']==r['summary'] and c['primary_improvement_maintained']==r['primary_improvement_maintained']
 assert r['verifier_sha256']==sha(U/'verify.py')
 html=(U/'RESULT_KO.html').read_text(encoding='utf8');links=re.findall(r'(?:href|src)="([^"]+)"',html)
 assert all((U/x).is_file() for x in links)
 s=r['summary'];p=r['pairs'];delta={v:s['T3F'][v]['mean']-s['C'][v]['mean'] for v in VIEWS}
 gains_losses={v:[sum(x['gains'] for x in p[v]),sum(x['losses'] for x in p[v])] for v in VIEWS}
 failures={v:len(vi['input_failures'][v]) for v in VIEWS}
 result_text=' / '.join(f"{v} C{s['C'][v]['counts']}→T3-F{s['T3F'][v]['counts']}/각200, 평균{s['C'][v]['mean']:.4f}→{s['T3F'][v]['mean']:.4f}%({delta[v]:+.4f}%p), T3-F범위{s['T3F'][v]['range'][0]:.2f}~{s['T3F'][v]['range'][1]:.2f}%" for v in ['sensor','masked'])
 next_action='C와 T3-F의 고정 모델 미사용 공개200개 평가 결과를 사용자와 검토한다. 기존 C와 웹/DTW를 유지한다. 이번 ID17/18은 평가에 사용했으므로 이후 튜닝 뒤 새 맹검 자료로 재사용하지 않는다. ID14~16 정상연결299개는 미사용 유지. 추가 학습·모델/seed선택·운영교체는 별도 계획 확인 전 진행하지 않는다. 사용자·멘토 촬영은 학습/튜닝/선택 금지.'
 summary=f'''ops/{U.name}. 사용자 승인대로 기존 C/T3-F 각3개 체크포인트·epoch·전처리를 먼저 고정한 뒤, 학습/모델선택 이력이 없는 공식 AI Hub 2.Validation ID17/18의 현재100단어×2촬영자ID=200영상 전수평가. 기존TRAIN899(ID01~09)/DEV300(ID11~13)과 영상ID·CRC/크기·SHA중복0, 공개실행이력/추출파일미사용감사 PASS. 메타목록은이전에읽었으나학습/성능선택미사용. 같은말뭉치2ID이며외부신원인증/실제웹캠일반화는미검증. 전처리는C의joint_v3/하단경계1.477334276/280채널/관측active_span그대로. native전체검출후masked와fresh상반신RGB검출sensor 비교. 입력200건·{vi['matched_channel_values']:,}채널값·{vi['hidden_points_perturbed']:,}숨긴점교란 별도산술검산 PASS, 입력실패{failures}도200분모유지. 추가학습0, 모델6개×2view×200=2400예측 각1회저장·별도산술검산 PASS. {result_text}. C오답→T3-F정답/역회귀(각600반복예측): {gains_losses}. 사전주평가개선유지={r['primary_improvement_maintained']},보조/ID/관측그룹확장일관성={r['stronger_consistency_pass']}; 개별회귀/남은혼동은COMPARISON_INDEPENDENT/RESULT_KO에전체공개. 기존C/T3-F체크포인트SHA와상태불변, 운영/사용자자료변경0. ID14~16정상연결299개미사용; 기존식별불명확마음에들다REAL14_U_F1개는범위밖그대로. 평가한17/18은이제미사용시험으로재사용불가. 같은Codex 구현+별도입력/점수산술검산(독립agent아님). RESULT_KO.html/txt, CONCLUSION, INPUTS_VERIFIED, COMPARISON_INDEPENDENT, FINAL_AUDIT. 준비exec22995종료0, 평가종료0, 잠금없음. 학습/재채점/운영교체추가실행없음.'''
 summary=summary.replace('평가종료0','평가6모델점수저장후상태파일갱신WinError5(exit1)증거보존→저장2400건검산PASS→상태만복구(재추론0)')
 summary+=' 주평가C2개가이미200/200이라2쌍동률·1쌍개선. 평균개선+0.3333%p이지만사전2/3상승기준미충족을그대로보고. masked신규회귀전주→성남(seed14)1건,기존믿다→무산(seed15)오답1건유지.'
 cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=cp.read_text(encoding='utf8')
 h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
 assert text.count(h8)==1 and text.count(h9)==1
 heading='### 2026-10-19 T3-F 미사용 공개 200영상 비교 완료 — 기존 C 유지'
 assert heading not in text
 text=text.replace(h8,h8+'\n\n'+heading+'\n'+summary,1)
 text=text.replace(h9,h9+'\n\n### 2026-10-19 T3-F 별도 공개 평가 결과 검토\n'+next_action,1)
 cp.write_text(text,encoding='utf8')
 timestamp=now();sp=O/'STATE.json';st=read(sp)
 st.update(current_stage='WORD100_T3F_HELDOUT_COMPLETE_AWAITING_REVIEW',next_single_action=next_action,updated_date_kst=timestamp[:10])
 st[U.name]={'at':timestamp,'status':'COMPLETE_AWAITING_USER_RESULT_REVIEW','report':str(U/'RESULT_KO.html'),'audit':str(U/'FINAL_AUDIT.json'),'frozen_models':6,'new_training':0,'heldout_signer_ids':[17,18],'heldout_videos':200,'classes':100,'input_failures':failures,'input_values_verified':vi['matched_channel_values'],'predictions_verified':2400,'summary':s,'delta_pp':delta,'gains_losses':gains_losses,'primary_improvement_maintained':r['primary_improvement_maintained'],'stronger_consistency_checks':r['stronger_consistency_checks'],'remaining_errors':r['remaining_errors'],'heldout17_18_now_consumed':True,'heldout14_16_eligible_untouched':299,'unresolved14_outside_cohort':1,'production_changed':False,'user_mentor_data_used':False,'next_single_action':next_action}
 write(sp,st)
 handoff=f'''# WordsInSilence 최신 인수인계

갱신: {timestamp}

현재 목표: T3-F를 고정하고 학습·선택 미사용 공개 자료에서 C와 비교 완료. 결과 검토 대기.

- 단위: research/front_keyphase_v1/ops/{U.name}
- 완료: 이력감사 → 모델6개·200개 전수 범위 고정 → 두 입력 조건 준비 → 입력 독립산술검산 → 각 모델 1회 평가 → 저장2,400예측 별도검산. 추가 학습0.
- 자료: 공식Validation ID17/18 각100영상. 기존TRAIN899/DEV300과 중복0. 기존개발검증300개 결과와 구분. ID14~16 정상연결299개는 계속 미사용(기존식별문제1개 별도보류).
- 결과: {result_text}.
- 개선/회귀: {gains_losses}(각200영상×3모델). 주평가 개선유지={r['primary_improvement_maintained']},확장일관성={r['stronger_consistency_pass']}. 평균만으로 개별회귀가 사라진 것은 아님. 상세단어/ID/group/순위는보고서.
- 해석: 두조건평균각+0.3333%p. sensor는C2개가이미200/200이라2쌍동률·1쌍상승(198→200),새회귀0. 사전‘2/3상승’기준을사후완화하지않아미충족. masked는개선3/회귀1; 전주→성남seed14신규회귀, 믿다→무산seed15기존오답유지.
- 입력검산: 200개, {vi['matched_channel_values']:,}채널값. 실패={failures}. 실패도분모에서제외안함.
- C와T3-F checkpoint/epoch/좌우배정/하단경계/채널 고정. 평가로재선택안함. 기존C·웹·DTW유지,사용자/멘토촬영미접근·학습/튜닝/선택미사용.
- 한계: 같은말뭉치2촬영자ID이며실사용웹캠/인구전체일반화증명아님. 3회모델반복을600개독립영상으로세지않음. 이번17/18은이제평가사용됨. 다시맹검으로부르지않음.
- 실행종료: 준비exec22995종료0. 평가6개점수저장후STATUS파일갱신WinError5로EVAL_EXIT1; 최초증거와종료코드보존. 저장2400건검산PASS로계산완료확인하고상태만복구(EVALUATION_RECOVERY). 재추론0. PREP/EVAL잠금없음. 진행중학습없음. 완료추출/추론/검산재실행하지않는다.
- 구현/검산: Codex로컬+별도산술코드. 독립사람/agent검토아님. FINAL_AUDIT/REPORT_LAYOUT_QA확인.
- 근거: FROZEN_PROTOCOL, EXPOSURE_AUDIT, EVAL_CONTEXT, INPUTS_VERIFIED, COMPARISON_INDEPENDENT, CONCLUSION, RESULT_KO.html/txt, FINAL_AUDIT.

다음 한 작업: {next_action}
'''
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
 cur=read(U/'CURRENT_EXECUTION.json');cur.update(status='COMPLETE_SAVED_RESULTS_VERIFIED_STATUS_RECOVERED',finished_at=timestamp,prep_exit=0,evaluation_exit=1,evaluation_recovery='EVALUATION_RECOVERY.json',new_inference_on_recovery=0);write(U/'CURRENT_EXECUTION.json',cur)
 event(stage='COMPLETE_AWAITING_USER_RESULT_REVIEW',videos=200,models=6,predictions=2400,primary_improvement_maintained=r['primary_improvement_maintained'],production_changed=False)
 assert read(sp)[U.name]['summary']==r['summary'] and U.name in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf8')
 artifacts=['AUTHORIZATION.json','FROZEN_PROTOCOL.json','EXPOSURE_AUDIT.json','EVAL_CONTEXT.json','PREFLIGHT.json','PREP_COMPLETE.json','PREP_EXIT.json','INPUTS_VERIFIED.json','EVAL_EXIT.json','COMPARISON_INDEPENDENT.json','CONCLUSION.json','RESULT_KO.html','RESULT_KO.txt','comparison.png','REPORT_LAYOUT_QA.json','CURRENT_EXECUTION.json','verify.py','report.py','finalize.py','EVALUATION_RECOVERY.json','FIRST_EVALUATION_ERROR.txt','MODEL_SOURCE_AUDIT.json','VERIFIER_SELFCHECK.json']
 audit={'at':now(),'status':'PASS_COMPLETE_KEEP_PRODUCTION_C','code_and_source_pins':len(ctx['pins']),'model_checkpoints_unchanged':ctx['models'],'inputs_verified':200,'predictions_verified':2400,'training':0,'reselection':False,'original_C_kept':True,'user_mentor_data_used':False,'heldout17_18_now_consumed':True,'heldout14_16_eligible_untouched':299,'primary_improvement_maintained':r['primary_improvement_maintained'],'stronger_consistency_checks':r['stronger_consistency_checks'],'gains_losses':gains_losses,'layout_qa':qa,'visual_review':'Desktop top/results and remaining-problem table reviewed; no clipping. Mobile no horizontal document overflow.','artifact_pins':{str(U/f):sha(U/f) for f in artifacts},'document_pins':{str(p):sha(p) for p in [cp,sp,ROOT/'NEW_CHAT_HANDOFF.md']},'verification_operator':'Same Codex, separate arithmetic reconstruction; not independent agent/person'}
 audit.update(evaluation_original_exit=1,evaluation_recovery=recovery,new_inference_during_recovery=0,mean_improved_in_both_views=all(d>0 for d in delta.values()),sensor_C_ceiling_ties=2)
 audit['artifact_pins'].update({str(U/f):sha(U/f) for f in ['FIRST_RENDER_ERROR.txt','render_qa.cjs','report_preview_top.png','report_preview_scores.png','report_preview_remaining.png']})
 write(U/'FINAL_AUDIT.json',audit,new=True)
 print(json.dumps({'status':audit['status'],'primary_improvement_maintained':r['primary_improvement_maintained'],'documents_synced':3},ensure_ascii=False))

if __name__=='__main__':main()
