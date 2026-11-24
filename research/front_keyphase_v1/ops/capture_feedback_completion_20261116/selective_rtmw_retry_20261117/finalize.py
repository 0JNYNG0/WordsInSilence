from pathlib import Path
import json,hashlib,datetime,re
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;ROOT=O.parents[2]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    v=read(D/'PUBLIC_VALIDATION.json');capture=read(D/'web/CAPTURE_CHECK.json');http=read(D/'web/HTTP_CHECK.json');admission=read(D/'web/ADMISSION.json');guards=read(D/'GUARD_CHECK.json')
    assert all(q['passed']for q in[v,capture,http,guards])
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in admission['pins'].items())
    rows=[r for r in v['controls']if r['kind']in['missing_hand','missing_one_finger']]
    attempted=[r for r in rows if r['retry']['requested']]
    now=datetime.datetime.now().astimezone().isoformat()
    result=dict(status='PUBLIC_VERIFIED_SELECTIVE_REOBSERVATION_WEB_APPLIED',at=now,actor='Codex local implementation and separate checks; no independent expert',scope='Front F camera; avatar five-view scope unchanged',normal_preserved=len(v['normal']),dropout_test_clips=8,dropout_test_cases=len(rows),unknown_cases_before=sum(r['unknown_before']>0 for r in rows),unknown_cases_after=sum(r['unknown_after']>0 for r in rows),unknown_criteria_before=sum(r['unknown_before']for r in rows),unknown_criteria_after=sum(r['unknown_after']for r in rows),recovered_cases=v['recovered_cases'],attempted_cases=len(attempted),added_seconds_attempted=[r['retry']['seconds']for r in attempted],guard_and_error_controls=len(v['controls'])-len(rows),software_contracts=len(guards['rows']),natural_reextractions=v['natural'],public_capture=capture['rows'],http=http,prior_pins_preserved=admission['prior_pins_preserved'],active_pins=len(admission['pins']),training_runs=0,weights_changed=False,private_training=False,private_tuning=False,private_candidate_selection=False,private_footage_read=False,physical_user_recheck=False,
        changes=['Only short missing observations near localized important-action evidence trigger retry. Complete observations or classification disagreement alone do not trigger.', 'Warm RTMW-l uses two upper-body context boxes on each selected frame, at most6missing frames plus two anchors and16pose calls. Missing points only, original arrays retained.', 'Reject cross-hand overlap, frame edge, long/unbounded gaps, crop disagreement, observed-point contradiction, blank image and budget overrun.', 'Re-evaluate feedback with qualified observations; keep original TCN top3 and all previously known supported/difference criteria, including error locations.', 'Capture preparation, pacing, return-to-rest and UI unchanged. Startup waits until recovery admission verification completes.'],
        limits=['14 unknown cases came from16 controlled dropout tests on8 known public frontal videos, not14 independent naturally failing webcam recordings.', 'Four natural re-extractions at640x480 had no unknown before or after. Natural detector failure recovery has not been established.', '7 controlled unknown cases remain: projected overlap, crop disagreement or time budget. Hidden fingers and complete absence of an important event are not inferred.', 'Left/right proposal symmetry verified as software contract; actual recovered footage here was right-hand gaps. Left-hand pose-recovery accuracy is not established.', 'The1.2s retry budget stops new work and rejects late candidates; an in-flight GPU/decode call is not forcibly cancelled, so it is not a hard wall-time guarantee.', 'Recorded public capture latency excludes physical webcam, pre-stop rest wait and browser paint. Synthetic observation errors are not actual learner recordings.'],
        initial_issues=['Initial FIT selector inspected surviving core only; short missing runs adjacent to truncated events were excluded. Added bounded300ms event adjacency before separate verification.', 'Verification harness first had a syntax typo and a source-path key mismatch; corrected before completed validation.', 'Restricted-environment socket visibility did not establish old server absence. First start hit occupied port; confirmed old entrypoint/camera-off in ordinary environment, then replaced exact server tree.', 'Startup exposed ready during added admission hash checks; kept preparing through validation and verified final HTTP readiness.'],
        next='Return to A/B/C demo-finalization plan; new actual user webcam check and physical HDD disconnect remain pending. Do not automatically train or expand recovery to hidden hands.')
    write(D/'RESULT.json',result)
    report=f'''중요 구간 선택적 RTMW 재검출 — 공개 검증 및 웹 적용 완료

달라진 점
촬영 중 관절값은 그대로 재사용한다. 이미 관측한 중요한 행동 구간과 가까운 지점에서 관절이 짧게 빠지고 결과가 확인 불가일 때만 해당 영상의 몇 장을 다시 본다.
최대6개 누락 프레임과 앞뒤 장면, 두 가지 상반신 영역으로 기존 GPU RTMW-l을 재사용한다. 최대16회 포즈 추정, 추가1.2초 작업 예산이며 전체 영상의 관절을 다시 추출하지 않는다.
영상에서 새로 얻은 관절만 쓰며 보간 좌표를 정답처럼 채우지 않는다. 기존에 확인된 정상·오류와 오류 시점을 바꾸면 새 결과를 버린다. TCN 후보는 원래 결과를 유지한다.
손끼리 겹침·화면 가장자리·긴 누락·두 검출의 불일치·시간 부족이면 기존 보류로 돌아간다. 관절은 충분한데 모양 모델끼리 의견이 다른 경우는 재검출 대상이 아니다.

결과
기존 공개 정면 정상32영상: 판정과 TCN후보 모두 유지. 재검출0회, 조건 검사 약0.0004초 수준.
별도 공개8영상에서 관절 누락을 인위적으로 만든16시험: 확인 불가가 생긴14건 →7건. 판단 항목 수로는15개 →7개. 7건이 복구됐다. 실제 사용자 실패14건이 아니다.
별도 제어조건28개: 손 영역을 검게 가린 영상, 긴 누락, 중요 구간 밖 누락, 기존 손모양 오류·부분 수행 등의 보존 확인. 역순 관절 편집은 의미오류 정답이 없어 기존 판정 유지 진단으로만 썼다.
소프트웨어 보호검사8개: 기존 정상/오류/오류위치 보존, 모양판정 불확실만으로 호출 금지, 좌우 공통 선택, 영상읽기 실패 시 복귀 통과.
자연 관절 재추출4조건(공개2영상×전체화면축소/상반신): 원래도 확인 불가0, 적용후0. 이 검사에서 자연 검출 개선은 확인되지 않았다.

시간
선택적 재검출을 시도한 별도 공개누락10시험의 추가시간 약0.55~0.99초. 입력 영상 크기·읽기 비용에 따라 다르다.
실제 촬영 코드에 공개 영상을 공급한 정상 사과/감사: 자동종료, 종료후계산 약0.332/0.336초. 추가재검출 없음.
같은 촬영 경로의 짧은 누락: 감사는 재검출 추가0.530초, 총0.905초에 정상 근거 복구. 사과는 손겹침 근거로 재검출하지 않고 보류 유지, 총0.374초.
이는 웹캠 실측·브라우저 표시시간이 아니다. 1.2초 예산은 실행 중 GPU 호출의 강제 취소를 보장하지 않는다.

적용
http://127.0.0.1:8821/ 최신 서버에 적용. 기존 화면·촬영준비·자동종료 유지. 공개API 사과/감사 정상 및 공개재생 퀘스트진행 금지 확인. 시작도구 START_WEB_DEMO.cmd가 새 서버를 가리킨다.
서버 실제PID {http['server']['pid']}, 일반 실행환경, 카메라꺼짐 확인. 77개 기존 고정파일 보존, 현재85개 검증핀.
신경망 새학습0, 가중치/C·T3-F/100단어 본체 변경0, 사용자촬영 열람·학습·튜닝·후보선택0.

남은 범위
실제 웹캠에서 자연스럽게 발생한 검출 실패의 복구 효과는 아직 확인하지 않았다. 가려진 손가락이나 아예 찾지 못한 주요 동작까지 복원하는 기능이 아니다.
검증상 복구된 사례는 오른손; 왼손 공통 선택 로직 검산은 했지만 실제 왼손 재검출 정확도를 별도로 입증하지 않았다.
새 실제 촬영과 HDD 분리 확인은 사용자 가능 시 진행한다. 그 전에도 기존 A/B/C 화면 점검·시연 실행 준비·발표 근거 정리는 가능하다.
작성·실행·별도 출력/해시/경로 검산은Codex, 수어 전문가나 독립 검증자 검증은 아니다.
'''
    (D/'RESULT_KO.txt').write_text(report,encoding='utf-8')
    graph=read(G/'web_demo_finish_plan_20261117/PLAN.json')
    graph.update(updated_at=now,current_public_repair=str(D/'RESULT.json'),current_runtime=str(D/'serve.py'),status='SELECTIVE_REOBSERVATION_PUBLIC_VERIFIED_DEMO_FINALIZATION_REMAINS')
    graph['nodes'].insert(1,dict(id='R_RETRY',label='중요 구간 선택적 RTMW 재관측',status='complete_public_verified_web_applied',evidence=str(D/'RESULT.json'),limits='Controlled public dropout improvement; actual webcam retry pending'))
    graph['edges']=[['P','R_RETRY']]+[['R_RETRY'if a=='P'else a,b]for a,b in graph['edges']]
    write(D/'GRAPH.json',graph)
    sp=O/'STATE.json';s=read(sp);t=s['capture_feedback_completion_20261116'];next_action='선택적 재관측은 공개 검증 후8821에 적용 완료. 기존 A/B/C 시연 화면 근거·로컬 실행 준비·발표 초안을 이어간다. 사용자 가능 시 D에서 실제 정상→오류→교정 재시도와 HDD 분리 확인. 자연 웹캠 검출 실패 복구는 아직 미검증이며 개인 촬영은 진단만, 학습/튜닝/후보선택 금지.'
    s['current_stage']=t['status']=result['status'];s['next_single_action']=t['next_single_action']=next_action
    t['selective_rtmw_retry_20261117']={k:result[k]for k in ['status','at','normal_preserved','dropout_test_clips','dropout_test_cases','unknown_cases_before','unknown_cases_after','recovered_cases','training_runs','weights_changed','physical_user_recheck','private_footage_read','prior_pins_preserved','active_pins']}
    t['selective_rtmw_retry_20261117'].update(report=str(D/'RESULT_KO.txt'),result=str(D/'RESULT.json'),admission=str(D/'web/ADMISSION.json'),http=str(D/'web/HTTP_CHECK.json'),pid=http['server']['pid'])
    t['current_graph']=str(D/'GRAPH.json');t['active_feedback_runtime']=str(D/'serve.py');t['report']=str(D/'RESULT_KO.txt');t['updated_at']=now
    write(sp,s)
    cp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';c=cp.read_text(encoding='utf-8-sig')
    e8='### 2026-11-17 重要区間選択的 RTMW 재검출 공개 검증·웹 적용\n'.replace('重要区間選択的','중요 구간 선택적')+f'중요 행동의 관측 부족에만 기존GPU RTMW-l로 짧은 구간 재관측. 최대6누락프레임/앞뒤포함16회/추가1.2초예산, 보간입력 없음. 공개32정상·TCN보존, 공개8영상16누락시험 중 확인불가14건→7건(인위적관절누락), 제어28/소프트웨어8통과. 자연재추출4조건은전후확인불가0으로자연검출개선주장없음. 실제촬영코드공개4시험자동종료: 정상0.332/0.336초, 감사누락복구총0.905초(추가0.530), 사과겹침보류0.374. 별도누락재검출추가0.55~0.99초. 사용자웹캠/화면표시시간아님. 학습0/가중치·C·T3-F보존/개인자료사용0. 최신8821 PID{http["server"]["pid"]},카메라꺼짐,HTTP통과. 기존77핀보존/현재85핀. 보고 {D / "RESULT_KO.txt"}, 그래프 GRAPH.json. A/B/C 및 실제D/E남음. 작성·검산Codex,독립전문가아님.\n\n'
    e9='### 2026-11-17 선택적 재관측 적용 이후\n'+next_action+' 최신 그래프 '+str(D/'GRAPH.json')+'.\n\n'
    for section,entry in [(8,e8),(9,e9)]:
        c,n=re.subn(rf'(^## {section}[^\n]*\n\n)',lambda m:m.group(1)+entry,c,count=1,flags=re.M);assert n==1
    cp.write_text(c,encoding='utf-8')
    handoff=f'''# 최신 인수인계 — {now}
목표: 정면 웹캠 사과(과일)·감사 시범→촬영→자동종료→정상/교정/관측부족→재시도/퀘스트. 5방향은 아바타전용.
최신 완료: {D/'RESULT_KO.txt'} / RESULT.json. 중요한 구간 주변의 짧은 관절 누락에만 기존GPU RTMW-l 재관측. 두 상반신 영역/앞뒤 검증, 최대6누락프레임·16포즈호출·추가1.2초작업예산. 누락점만 실제 재검출로 교체, 보간정답 없음. 기존 TCN후보·정상/오류/오류시점 보호. 겹침·긴누락·검출불일치·시간부족은 원래 보류. 모델불확실만으로 재검출하지 않음.
검증: 공개32정상 유지. 별도8영상16인위적누락시험 중 확인불가14건→7건,판정항목15→7. 제어28/소프트웨어8통과. 자연재추출4조건은전후불확실0으로자연검출개선주장불가. 공개촬영코드4시험자동종료; 정상0.332/0.336초,감사누락복구0.905초(추가0.530),사과겹침보류0.374초. 별도재검출추가0.55~0.99초. 물리웹캠/브라우저표시시간아님.
보존: C·T3-F/100단어 본체/모델가중치 유지,새학습0,총이력75유지,개인촬영열람·학습·튜닝·후보선택0. 기존77핀유지/현재85핀. 구현·검증Codex,별도전문가아님.
실행: START_WEB_DEMO.cmd → frontal_demo_launcher.py → {D/'serve.py'}. 실제서버PID {http['server']['pid']},8821,일반환경,카메라꺼짐·공개HTTP정상·공개재생퀘스트진행금지·최신JS보존확인. 재개시PID/포트를확인하고중복실행금지. 제한환경의연결실패를서버없음으로간주하지말것.
초기실패기록: 핵심구간만보면누락으로잘린경계가빠져FIT단계에서300ms인접후보범위보완. 검증스크립트문법/소스키오류수정. 기존서버포트중복은정상환경에서실제PID/카메라꺼짐확인후교체. 추가해시검사중ready노출은preparing유지로수정후최종HTTP통과. 동결실험출력보존.
현재그래프 {D/'GRAPH.json'}: 기존P공개검증 및R_RETRY완료; A/B/C촬영없는시연준비, D실제촬영/HDD분리, E고정남음.
다음 한 작업: {next_action}
한계: 복구성공은주로공개관측누락재현이며자연사용자실패회복미검증. 실제복구오른손사례;왼손선택대칭코드검산만완료. 가려진손가락/아예없는행동추정안함. 예산은실행중GPU강제취소가아님. 기존사용자8촬영진단은과거기록보존,새후보로재평가하지않음.
'''
    (ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
    assert read(sp)['capture_feedback_completion_20261116']['training_runs']==75
    assert all(Path(x).is_file()for x in [t['current_graph'],t['active_feedback_runtime'],t['report']])
    write(D/'CLOSEOUT_CHECK.json',dict(passed=True,at=now,state_json_valid=True,report_links_valid=True,prior_pins_preserved=True,training_total_preserved=75,web_applied=True,private_data=False))
    print(json.dumps({'status':result['status'],'unknown_cases':[result['unknown_cases_before'],result['unknown_cases_after']],'pid':http['server']['pid'],'report':str(D/'RESULT_KO.txt')},ensure_ascii=False))
if __name__=='__main__':main()
