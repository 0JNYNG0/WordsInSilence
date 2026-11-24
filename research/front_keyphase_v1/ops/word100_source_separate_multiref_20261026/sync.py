from pathlib import Path
import sys,json,shutil,datetime
ROOT=Path('C:/MyUnityProjects/WordsInSilence');O=ROOT/'research/front_keyphase_v1/ops';U=O/'word100_source_separate_multiref_20261026';sys.path.insert(0,str(U));import prepare as R
at=R.now();v=R.read(U/'INTEGRATION_VERIFIED.json');c=R.read(U/'CAUSES.json');p=R.read(U/'PREPARED.json');f=R.read(U/'FROZEN.json');R.check(f['protected']);R.check(f['pins']);R.check(p['outputs']);R.check(R.read(U/'PAIRS_DONE.json')['pins'])
summary='公開TRAIN100단어·300영상(기존query100+다른기록ID참고200),200쌍 원본DTW순간/연속차이 유지하며 MP각도별재확인과YOLO감사를분리해연결. 새참고200/6749장면exec92749종료0,검산exec9307종료0. 원본순간각도211158·연속176955보존,양쪽MP현재영상근거순간0→19647(9.3%),연속0→7650(4.3%),철회0. 이전0은예전DTW부재가아닌이번전체/확대/원위치검사근거부재. 미확인191511중153978(80.4%)은미요청만원인. 300입력/200쌍별도수식검산·한각도/한손숨김각200PASS. 실제이미지4장확인,독립21점정답아님. HTML4600선택조합Node검사PASS,실브라우저렌더미검증. 학습/TCN재채점/운영교체0,사용자·멘토·추가199/200접근0. 전체goal계속.'
summary=summary.replace('公開','공개')
nextstep='같은공개300영상의주석핵심구간중원본각도는계산되지만아직재확인미요청인시간을공통규칙으로추가확인한다. 대상선정에단어별예외·인식점수·사용자자료금지. 현재동결결과읽기전용,새단위에freeze하고동일MP/원본위치/팔연결규칙. 완료한전체/확대재검출은동일이미지/box/가중치해시로재사용,미완료호출만실행. 예상최대15045손프레임/10704영상프레임/25749호출;실제재사용고려예산동결후실행. 전후같은200고정path/원본값/분모,MP근거와YOLO감사분리,연속부족·새불일치·정상손/교차보존검산. 전체호출은오프라인근거범위시험이며웹캠실시간속도보장아님. 독립사람정답없이정밀각도교정/수어정답/인식향상으로보고금지. 실제통합피드백목표계속,C/T3F/E3·운영·원본보존,사용자/멘토/추가199/200금지.'
ev='research/front_keyphase_v1/ops/word100_source_separate_multiref_20261026/RESULT_KO.html,REFERENCES_VERIFIED.json,INTEGRATION_VERIFIED.json,CAUSES.json,VISUAL_REVIEW.json'
R.write(U/'HTML_SCRIPT_VERIFIED.json',dict(at=at,status='PASS4600_SOURCE_SCRIPT_STATES',states=4600,queries=100,references=200,canvas_calls=535069,script_sha256=R.sha(Path('C:/Users/local-user/.codex/visualizations/2026/09/18/01a0b4a2-0a79-7b80-b823-d6469869041f/multiref_html_check.js')),html_sha256=R.sha(U/'RESULT_KO.html'),browser_rendered=False))
R.write(U/'NEXT.json',dict(at=at,action=nextstep,prior_results_readonly=True,upper_bound=dict(hand_frames=15045,image_frames=10704,calls=25749),authorization='Existing user autonomous continuation,publicTRAIN only,operating preserved',goal_complete=False))
before=U/'before';before.mkdir()
for name in ['WordsInSilence_PROJECT_CONTEXT.md','NEW_CHAT_HANDOFF.md']:
 q=ROOT/name;shutil.copy2(q,before/name)
shutil.copy2(O/'STATE.json',before/'STATE.json')
q=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=q.read_text('utf-8-sig')
for num,body in [(8,summary),(9,nextstep)]:
 start=s.index(f'## {num}.');pos=s.index('\n',start)+1;s=s[:pos]+f'\n### 2026-10-26 공개 다중참고의 출처별 관측근거 연결\n{at}\n{body}\n근거: {ev}.\n\n'+s[pos:]
q.write_text(s,encoding='utf8')
q=O/'STATE.json';state=R.read(q);state.update(current_stage='AUTONOMOUS_SOURCE_SEPARATE_MULTIREf200_VERIFIED_CORE_RECHECK_NEXT',next_single_action=nextstep);state[U.name]=dict(at=at,status='RESEARCH300_INPUTS200_PAIRS_VERIFIED_NOT_PRODUCTION',summary=summary,next=nextstep,evidence=ev,counts=v['pair_counts'],input_counts=v['input_counts'],causes=c['counts'],active_sessions=[],training=0,operating_changed=False,goal_complete=False);q.write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'# WordsInSilence 최신 인수인계\n\n{at}\n\n{summary}\n\n다음 단일 작업: {nextstep}\n\n근거: {ev}\n\n현재단위 {U.name},모든실행종료0·잠금없음. REFERENCES_DONE/REFERENCES_VERIFIED(200),PREPARED(300),PAIRS_DONE/INTEGRATION_VERIFIED(200),CAUSES,VISUAL_REVIEW,HTML_SCRIPT_VERIFIED완료재사용. exec50783첫10→exec92749나머지190,verify9307,prepare87720,paired11238. 새이미지앞에목록/규칙동결;완료재실행금지. source분리:MP현재영상근거,YOLO불일치audit별도,몸/위치는원본관측,정밀정답아님. 기존순간/연속값유지;학습0. 전후0→19647은추가관측범위이지정답률/정확도향상아님. 기존재확인은전체손누락용이라이미보이는손의동일3쌍검사부재였음. 다음채움시험후근거범위가실제충분한지확인하고통합연구피드백으로이어감. 부족원인80.4%미호출;다른불일치도함께보존. 조회/학습금지자료:사용자·멘토·추가199/200. C/T3F/E3/운영웹·DTW/원본보존. goal active,작은성과만으로완료금지. 검산=같은Codex별도수식,독립사람정답아님. 브라우저fileURL열기안전거절이력은우회하지않고로컬그림/소스검사만사용.\n',encoding='utf8')
q=U/'CURRENT_EXECUTION.json';q.write_text(json.dumps(dict(at=at,stage='completed_verified',exec_history=[dict(exec=50783,exit_code=0),dict(exec=92749,exit_code=0),dict(exec=9307,exit_code=0),dict(exec=87720,exit_code=0),dict(exec=11238,exit_code=0)],active_sessions=[],next=nextstep),ensure_ascii=False),encoding='utf8')
for q in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md']:assert summary in q.read_text('utf8') and nextstep in q.read_text('utf8')
assert R.read(O/'STATE.json')[U.name]['summary']==summary;assert not(U/'RUNNING.json').exists() and not(U/'PAIR_RUNNING.json').exists();R.check(f['protected']);R.write(U/'SYNC_CHECK.json',dict(at=at,status='PASS',state_key=U.name,active_sessions=[],goal_complete=False,summary=summary,next=nextstep))
print('SYNC_COMPLETE',at,flush=True)
