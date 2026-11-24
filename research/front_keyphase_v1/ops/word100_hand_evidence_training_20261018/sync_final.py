from common import *
d=read(U/'COMPARISON.json');status=read(U/'STATUS.json');assert status['stage']=='COMPLETE_VERIFIED'
v=read(U/'SCORES_TRACE_INDEPENDENT.json');assert v['status']=='PASS'
s=d['summary'];passed=d['gate']['pass_to_next_review'];stamp=now()
numbers=[]
for view in ('sensor','masked'):
 text=[]
 for a in ('C','C0','C1'):
  q=s[a][view]['top1_correct'];text.append(f"{a}{q['percent_mean']:.2f}%({q['percent_range'][0]:.2f}~{q['percent_range'][1]:.2f})")
 pp=s['C1'][view]['top1_correct']['percent_mean']-s['C'][view]['top1_correct']['percent_mean']
 numbers.append(view+': '+', '.join(text)+f'; C1-기존C{pp:+.2f}%p.')
decision='후속 검토 조건 통과(자동적용 아님)' if passed else '보완C1미채택, 기존C유지'
text8='### 2026-10-18 손 관측 근거 TCN 학습·비교 완료 — '+decision+'\n'
text8+='ops/word100_hand_evidence_training_20261018. 사용자 승인 범위의 기존280채널 보존+30상태추가310채널 입력1199검증 완료. 공개TRAIN899/DEV300만 사용, C0(추가단서0)/C1(실제단서)각3seed 총6회 완료. '+ ' '.join(numbers)+'\n'
text8+='사전고정진행gate '+('PASS' if passed else 'FAIL')+'. 저장점수5400행 독립재집계, 배치원본ID/보기/선택epoch/checkpoint검산PASS. 총'+str(sum(t['epochs'] for t in d['trace']))+'epoch, 실제배치'+str(v['batches'])+'개/노출'+str(v['logged_video_exposures'])+'건은 동일899개반복. 초기출력3seed차이0. 기존280채널77,910,560값/추가손상태556,504검산, 실제C정상대조40보존. 대표96TRAIN영상시각검토.\n'
text8+='새 정보로 실제 손가락 복원 불가, 현재영역/겹침추정오차·미검출잔존. DEV반복선택지표이며웹캠/독립새사람정확도아님. 사용자/멘토/보류14~18자료미사용, 기존C/M0/DTW/웹보존. 구현Codex로컬/동일담당자별도계산코드검산, 타에이전트검토주장없음. 최초오류및같은조건복구보존. 상세RESULT_KO.txt/COMPARISON.json/SCORES_TRACE_INDEPENDENT.json.\n\n'
text9='### 2026-10-18 비교 결과 사용자 검토 — 추가 학습·운영 교체 없음\n'
text9+='현재단위 완료, 실행중프로세스/잠금없음. '+decision+'. 사용자에게평균·범위·%p·각그룹회귀·미검증한계를보고하고대기. 결과를보고기준/계수완화·추가학습·최고seed선택금지. 다음보완후보는별도계획확인후한변경씩검토, 기존C는유지. 완료입력/학습/저장점수재실행불필요. 사용자촬영학습·튜닝금지계속유지.\n\n'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text(encoding='utf-8-sig');assert text8.split('\n')[0] not in t
for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n',text8),('## 9. 다음 단일 작업\n',text9)]:assert marker in t;t=t.replace(marker,marker+'\n'+body,1)
p.write_text(t,encoding='utf-8')
state=read(O/'STATE.json');state['updated_date_kst']=stamp;state['current_stage']='WORD100_HAND_EVIDENCE_COMPARISON_COMPLETE_'+('PASS_REVIEW' if passed else 'NOT_ADOPTED')
state['next_single_action']='사용자와 완료 비교 결과 검토. 기존C 유지, 추가학습/웹교체 없음. 후속후보는 별도 계획.'
state['word100_hand_evidence_training_20261018']={'status':status['stage'],'decision':decision,'at':stamp,'unit':str(U.relative_to(ROOT)).replace('\\','/'),'authorization':str(U/'AUTHORIZATION.json'),'prep_context_sha256':sha(U/'PREP_CONTEXT.json'),'training_context_sha256':sha(U/'TRAIN_CONTEXT.json'),'summary':s,'gate':d['gate'],'verification':v,'runs':6,'process_running':False,'training_exec_session':71189,'training_exec_exit_code':0,'user_mentor_data_used':False,'heldout_used':False,'production_changed':False,'report':str(U/'RESULT_KO.txt'),'comparison':str(U/'COMPARISON.json')}
write(O/'STATE.json',state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n갱신 '+stamp+'\n\n'+text8+text9+'실행/검증 근거: READY.json, PREP_CONTEXT.json, TRAIN_CONTEXT.json, INPUTS_INDEPENDENT.json, VISUAL_REVIEW.json, C0/C1 각3 RESULT/best.pt/history/traces. 기존C 저장점수재사용;새모델완료후재채점0. 과거계획의 승인대기문구는이력이며이번사용자진행승인으로6회완료.\n\n보존: 기존 C ?model=c / M0 ?model=m0 / 기본M0, 필요손DTW 유지. 서버/카메라조작없음. st-gcn자동화삭제유지. W/sources/원본/기존동결결과읽기전용, Git/외부전송/다운로드없음.\n',encoding='utf-8')
write(U/'CURRENT_EXECUTION.json',{'at':stamp,'status':'COMPLETE','session_id':71189,'exit_code':0,'running':False,'next':'User result review; no further automatic training.'})
assert not (U/'TRAIN_RUNNING.json').exists()
for p in (O/'STATE.json',U/'STATUS.json',U/'CURRENT_EXECUTION.json'):read(p)
checkpins(read(U/'PREP_CONTEXT.json')['pins']);checkpins(read(U/'TRAIN_CONTEXT.json')['pins'])
write(U/'FINAL_AUDIT.json',{'status':'PASS','at':now(),'protected_source_and_model_hashes_unchanged':True,'documents_synchronized':True,'training_process_exit':0,'no_lock':True,'report_sha256':sha(U/'RESULT_KO.txt'),'comparison_sha256':sha(U/'COMPARISON.json')},new=True)
print('Final documents and protected hashes verified.')
