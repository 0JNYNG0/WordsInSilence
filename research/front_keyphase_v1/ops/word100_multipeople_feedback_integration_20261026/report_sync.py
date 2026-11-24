from pathlib import Path
import json,datetime,hashlib,shutil,html
R=Path('C:/MyUnityProjects/WordsInSilence');O=R/'research/front_keyphase_v1/ops';U=O/'word100_multipeople_feedback_integration_20261026';FC=O/'word100_feedback_evidence_consumer_20261026'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.datetime.now().astimezone().isoformat();v=read(U/'VERIFIED.json');c=read(U/'CAUSE_VERIFIED.json');a=read(U/'REQUEST_AUDIT.json');fv=read(FC/'VERIFIED.json');n=a['counts'];pct=100*n['no_retry_requested']/n['unassessed']
assert v['status']=='PASS_REAL600_EVIDENCE_PRESERVATION' and c['status']=='PASS_CAUSES_ALL600' and fv['status']=='PASS_ALL4500_CONSUMER_RECEIPTS'
note=f'''추가로 확인한 재확인 요청의 구조
손가락 각도를 계산할 수 있지만 재확인하지 못했던 원본 핵심구간323,572개 중320,912개({pct:.2f}%)는 그 프레임에서 재검출을 요청하지 않았던 경우였다. 재검출은 했지만 비교할 두 번째 손 근거를 확보하지 못한 경우2,628개, 손은 대응됐으나 해당 각도를 확인할 수 없는 경우32개다. 세 범주는 겹치지 않는다.
이유는 기존 재검출이 손 전체 미검출 복구용으로 설계됐기 때문이다. 이미 손이 검출된 정상 프레임의 손가락 모양을 전부 재확인하는 작업은 실행하지 않았다. 따라서 미확인은 대다수가 호출 범위의 공백이며, 실제 검출 실패나 틀린 수어로 해석하면 안 된다. 앞서 연속 비교의527,230/542,217은 참고 두 개의 시간구간을 조합한 채널·시점 수이고, 여기323,572는 원본300영상의 개별 각도·프레임 수이므로 분모가 다르다.
다음에는 통째로 미검출인 손을 찾는 기존 경로는 유지하고, 손이 보이는데 좌표가 불안정한 경우를 위한 별도 재확인 요청 기준을 공개TRAIN 관측으로 정리한다. 특정 단어·목표 정답·사용자 결과를 조건으로 사용하지 않는다. 재확인하지 않은 상태와 확인 결과가 엇갈린 상태를 구분하고, 추가 관측의 일치만으로 해부학 정답을 선언하지 않는다.
'''
backup=U/'before_source_explanation';backup.mkdir()
for p in [U/'RESULT_KO.html',U/'RESULT_KO.txt',U/'REPORT_CHECK.json']:shutil.copy2(p,backup/p.name)
(U/'REQUEST_CAUSE_KO.txt').write_text(note,encoding='utf-8')
p=U/'RESULT_KO.txt';p.write_text(p.read_text(encoding='utf-8')+'\n'+note,encoding='utf-8')
p=U/'RESULT_KO.html';s=p.read_text(encoding='utf-8');s=s.replace('<h2>공개 영상 비교 탐색</h2>','<section class="card"><h2>왜 재확인 자료가 적었나</h2><p>원본에서 재확인이 없던 각도 중 <strong>'+f'{pct:.2f}%'+'</strong>는 재검출을 요청하지 않은 경우였습니다. 기존 경로는 손 전체 미검출을 복구하는 용도였고, 이미 보이는 손을 모두 다시 확인하지 않았습니다.</p><p>따라서 <b>미확인 ≠ 틀린 손 모양</b>입니다. 좌표로 계산한 차이는 보존하면서 요청 여부와 실제 불일치를 구분합니다.</p><details><summary>원본 기록으로 확인한 원인</summary><pre>'+html.escape(note)+'</pre><a href="REQUEST_AUDIT.json">원인 집계 근거</a></details></section><h2>공개 영상 비교 탐색</h2>',1);p.write_text(s,encoding='utf-8')
summary='연구용 피드백 연결부4500조건 전수검산 완료(FC exec83419/78479 종료0). 서로다른공개ID11/12/13·100단어DEV300·600방향쌍 비교/검산 완료(exec69077/93317 종료0). 기존연속비교893415·품질349886셀 모두보존,134품질감소는원래대응유지. 원래순간차이·부족시간·각도별근거와두참고개별결과 유지,저장C294/300·T3F295/300결과만연결,새인식개선주장없음. 손가락연속543231중재확인1014;부족542217중527230(97.24%)는두번째관측없음만해당. 원본미확인323572중320912(99.18%)는재검출호출자체없었음:기존trigger가손전체미검출복구용이므로정상손모양확인미실행. 이를오류/저신뢰로자동판정금지. 별도원인검산완료. 100단어연구HTML탐색기/그래프완료,Node3000상태검사PASS·PNG눈검토,실브라우저렌더미검증. 학습/새추론0·운영변경0.'
nextjob='손전체미검출복구와 보이는손의형태재확인 요청을분리하는 공통 연구후보를준비한다. 먼저 기존공개TRAIN899 관측/근거에서 재등장·좌우충돌·손모양기하/시간불안정·정상교차와통제구간을정리해,단어/목표정답/사용자결과 없이요청기준과비교범위동결. 기존native좌표/순간별차이보존,재확인없음과실제불일치분리,같은모델일치=정답아님. 동결후실제이미지재확인효과와비용/회귀를공개자료로비교하고,검증된보완만TCN인식/DTW근거에각각연결. 이번600쌍은DEV진단이며새최종시험아님. 사용자/멘토/추가199·200 금지,C/T3F/E3·운영·W:/sources보존. 기존완료결과재추론/재채점금지. 목표active,작은시험PASS로전체완료금지.'
paths=[R/'WordsInSilence_PROJECT_CONTEXT.md',O/'STATE.json',R/'NEW_CHAT_HANDOFF.md'];bak=U/'before_completed_sync';bak.mkdir()
for p in paths:shutil.copy2(p,bak/p.name)
p=paths[0];s=p.read_text(encoding='utf-8-sig')
for prefix,title,text in [('## 8. 현재 상태','2026-10-26 인식·다중 참고 피드백 연결과 재확인 공백 검증',summary),('## 9. 다음 단일 작업','2026-10-26 보이는 손의 재확인 요청을 별도 보완',nextjob)]:
 line=next(x for x in s.splitlines() if x.startswith(prefix));s=s.replace(line,line+'\n\n### '+title+'\n'+now+'\n'+text+'\n근거: research/front_keyphase_v1/ops/word100_multipeople_feedback_integration_20261026/RESULT_KO.html,VERIFIED.json,CAUSE_VERIFIED.json,REQUEST_AUDIT.json.\n',1)
p.write_text(s,encoding='utf-8')
p=paths[1];s=read(p);s.update(updated_date_kst=now,current_stage='AUTONOMOUS_FEEDBACK_REAL600_VERIFIED_VISIBLE_HAND_RECHECK_SCOPE_NEXT',next_single_action=nextjob)
s['word100_feedback_evidence_consumer_20261026']=dict(at=now,status='COMPLETED_VERIFIED_RESEARCH_ONLY',run_session=83419,verify_session=78479,exit_code=0,active_process=False,counts=fv['counts'],report=str(FC/'RESULT_KO.html'),production_changed=False)
s['word100_multipeople_feedback_integration_20261026']=dict(at=now,status='COMPLETED_VERIFIED_RESEARCH_ONLY',run_session=69077,verify_session=93317,cause_session=73287,exit_code=0,active_process=False,summary=summary,next=nextjob,report=str(U/'RESULT_KO.html'),production_changed=False,semantic_feedback_validated=False)
p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
paths[2].write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음 단일 작업: '+nextjob+'\n\n현재 모든 세션 종료0,활성잠금없음. FC83419생성/78479검산;실제600쌍69077생성/93317검산;원인73287생성,verify_causes와request_audit종료0. 보고서상세/Node HTML3000상태검사통과,정적PNG눈검토완료,브라우저렌더검증아님. 보고서본문의두가지원인비율은분모가다름. 품질부족대부분이재확인을요청하지않았던구간으로,성능실패로해석금지. 원본몸/손좌표,정상교차불변은수치보존검산이며정답해부학검증아님.\n\n이전증거유지:W3seed인식미채택(sensor -0.56%p,masked -0.22%p). Q4500근거보존/정상속도시간개선,EP연속단독1회귀실패보존. C/T3F/E3·운영웹/DTW·원본불변. 신규학습/검출/분류추론없음. 모델순위300개저장결과재사용,600쌍두배분모금지. 사용자/멘토/노출된추가199·200 접근금지. 카메라/다운로드/외부전송/Git0. 같은Codex별도산술검산과독립사람정답혼동금지. 전체goal계속.\n',encoding='utf-8')
for folder,sessions in [(FC,[83419,78479]),(U,[69077,93317,73287])]:
 cur=folder/'CURRENT_EXECUTION.json'
 if cur.exists():shutil.copy2(cur,bak/(folder.name+'_CURRENT_EXECUTION.json'))
 cur.write_text(json.dumps(dict(at=now,status='complete',sessions=sessions,exit_code=0,active_process=False)),encoding='utf-8')
 assert not(folder/'RUNNING.json').exists()
for p in paths:
 if p.suffix=='.json':read(p)
checks=read(U/'REPORT_CHECK.json');checks.update(static_png_visually_reviewed=True,html_script_3000_states_pass=True,browser_rendered=False,source_request_addendum=True,at=now);checks['files']={str(p):sha(p) for p in [U/'RESULT_KO.html',U/'RESULT_KO.txt',U/'reference_example.png',FC/'RESULT_KO.html',FC/'RESULT_KO.txt']};(U/'REPORT_CHECK_V2.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
(U/'SYNC_CHECK.json').write_text(json.dumps(dict(at=now,status='PASS',paths={str(p):sha(p) for p in paths},reports=checks['files'],active_process=False,goal_complete=False),ensure_ascii=False,indent=2),encoding='utf-8')
print('FEEDBACK_REPORT_SYNC_COMPLETE',pct)
