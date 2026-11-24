from pathlib import Path
import json,hashlib,statistics,ast
from datetime import datetime
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;ROOT=O.parents[2]
P=G/'selective_rtmw_retry_20261117';I=G/'selective_rtmw_followup_20261117'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
at=datetime.now().astimezone().isoformat()
v=read(D/'PUBLIC_VALIDATION.json');old=read(P/'PUBLIC_VALIDATION.json');cap=read(D/'web/CAPTURE_CHECK.json');guards=read(D/'GUARD_CHECK.json');http=read(D/'web/HTTP_CHECK.json');admission=read(D/'web/ADMISSION.json');private=read(D/'private_diagnostic_only/RESULT.json')
assert all(x['passed'] for x in [v,cap,guards,http]) and admission['admitted']
assert all(sha(f)==h for f,h in admission['pins'].items())
oldmap={(r['id'],r['kind']):r for r in old['controls']}
pairs=[(oldmap[r['id'],r['kind']],r)for r in v['controls']if r['kind']in['missing_hand','missing_one_finger']and(r['id'],r['kind'])in oldmap]
assert len(pairs)==16
before=sum(a['unknown_after']>0 for a,b in pairs);after=sum(b['unknown_after']>0 for a,b in pairs)
assert (before,after)==(7,4)
assert all(b['unknown_after']<=a['unknown_after'] for a,b in pairs)
remaining=[dict(id=b['id'],kind=b['kind'],retry=b['retry'])for a,b in pairs if b['unknown_after']]
changed=[dict(id=b['id'],kind=b['kind'],old_reason=a['retry']['reason'],new_reason=b['retry']['reason'],seconds=b['retry']['seconds'])for a,b in pairs if b['unknown_after']<a['unknown_after']]
times=[b['retry']['seconds']for a,b in pairs if b['retry']['pose_calls']]
assert len(private['rows'])==8 and all(r['known_judgments_preserved'] for r in private['rows'])
assert all(r['retry']['pose_calls']==0 for r in private['rows'])
assert [r['attempt']for r in private['rows']if r['after_overall_pass']]==[1,5]
assert all('difference'in r['after_current_retry'].values()for r in private['rows']if r['attempt']not in[1,5])
report=dict(status='PUBLIC_VERIFIED_RETRY_TIMING_AND_BRACKET_REPAIR_WEB_APPLIED',at=at,implementation='Codex local',verification='Separate public source/hash, normal/error/blank/capture/HTTP tests by same Codex; no independent human expert',changes=['Bounded2-thread exact-PTS video decoding; convert selected frame toRGB once','Do not treat overlap of already observed bracket frame as occlusion of earlier missing frames','Check both recovered hand poses against other hand at every repaired frame; all prior confidence, disagreement and time gates retained'],original16_controlled_trials=dict(previous_unknown=before,current_unknown=after,newly_recovered=changed,remaining=remaining),expanded_public_controls=dict(videos=16,dropout_trials=32,unknown_before=24,unknown_after=7,recovered=17,other_controls=52,normal_preserved=32,software_guards=10),capture=cap,selective_retry_seconds=dict(min=min(times),max=max(times)),private_diagnosis=dict(recordings=8,normal_attempts_now_pass=[1,5],shape_differences_in_other_attempts=6,retry_invocations=0,benefit_from_retry_not_demonstrated=True,other_unknown_criteria_remain=True,source=str(D/'private_diagnostic_only/RESULT.json'),training_use=False,tuning_use=False,candidate_selection_use=False),preservation=dict(C_T3F=True,model_weights=True,common100=True,training_runs=0,historical_training_runs_unchanged=75,previous_pins=90,current_pins=len(admission['pins']),old_recordings_unchanged=True),runtime=str(D/'serve.py'),server=http['server'],camera_started=False,limitations=['The original7 are trials on4 public clips with synthetic missing coordinates, not7 independent natural user errors.','Remaining4 are2 treatments on2 clips: projected overlap risk in a missing frame; crop disagreement at middle fingertip. Do not call these all proven true occlusion.','Expanded set includes prior development/public training people; not unseen-user accuracy.','Same pixels preserved by decoding checks; wall-clock timings are single runs with machine-load variation.','Public capture latency excludes browser painting and waiting for automatic-stop trigger.','Private inference was done only after public admission, never for selecting code or thresholds. Live auto-stop/readiness cannot be retroactively verified from saved motion videos.','No claim of reconstructing genuinely hidden finger state.'],next='Continue existing graph A/B/C presentation and local demo readiness; actual new capture and HDD-disconnected acceptance remain D, then freeze E.')
write(D/'RESULT.json',report)
body=f'''선택적 RTMW 재검출 후속 결과 — {at}

무엇이 바뀌었나
1. 영상을 읽는 작업을2개 처리 흐름으로 나누고 같은 프레임 색상 변환을 중복하지 않게 했다. 프레임·해상도·판정 기준은 그대로다.
2. 관절이 빠진 순간과 이미 관측된 앞뒤 순간을 구분했다. 뒤의 정상 관측에서 손이 겹쳤다고 앞의 보이는 손까지 복구를 막지 않는다.
3. 실제로 새 좌표를 넣을 순간에는 두 번의 재검출 모두 다른 손과 겹치지 않는지 확인한다. 진짜 가림·검출 불일치는 강제 통과시키지 않는다.

이전7건 → 현재4건 확인 불가
- 시간 부족1건 복구.
- 누락 직후의 관측된 순간이 겹쳐서 앞의 누락까지 차단하던2건 복구.
- 남은2건: 누락 구간의 겹침 위험. 실제 가림이라고 단정하지 않지만 안전하게 복구할 근거가 부족하여 보류.
- 남은2건: 두 화면 범위로 재검출한 중지 끝 위치가 서로 다름. 다른 손가락을 맞추려고 이 불일치를 무시하지 않음.
이7건은 공개 영상4개의 관절 일부를 시험용으로 비운 조건이며, 자연 사용자 실패7영상을 뜻하지 않는다.

정상·오류·속도
공개 정상32영상 판정·TCN순위 유지. 확장16영상32누락조건은 확인 불가24→7, 그 외52제어조건 및10소프트웨어보호 검사 통과.
공개 촬영 경로4시험 모두 자동 종료. 정상 종료→계산 약0.25초, 감사 누락복구 약0.64초(그중 추가 재검출 약0.38초), 사과 겹침은 보류 약0.27초.
별도 기존 비교조건의 재검출 작업은 약{min(times):.2f}~{max(times):.2f}초. 영상 전체 재검출이 아니며 추가1.2초작업예산·16호출 한도 유지.
실제 웹캠 화면 표시 대기시간이나 새 사용자 정확도 수치가 아니다.

사용자 촬영 재확인
저장된8촬영을 공개 검증·적용 결정 뒤에 진단 전용으로 다시 계산했다.
각 단어 첫 정상 시도(1·5)는 현재 통과. 나머지6시도에는 손모양 차이가 안내되고 일부 다른 항목은 확인 부족이 남는다.
이 개선은 앞서 공개 검증한 손겹침·손가락 판단 보완의 결과다. 이번 재검출은8건 모두 실행 조건에 해당하지 않아 전후 판정이 같다.
긴 누락·겹침 위험 또는 관절 누락이 아닌 판단 불확실은 이번 짧은 누락 복구 대상이 아니다. 재검출이 사용자 영상에서 실제 도움을 줬다고 주장하지 않는다.
사용자 자료 학습·튜닝·후보 선택0, 원래 영상과 결과 보존. 새 자동 종료·브라우저 표시시간은 새 촬영에서 확인해야 한다.

최신 화면 http://127.0.0.1:8821/ (새로고침)
모델 재학습0, C·T3-F와100단어 본체 및 모든 신경망 가중치 유지. 정면 자료만 사용. 시행·검증Codex, 독립 전문가 인증 아님.
남은 시연 마감: A화면·B로컬실행·C발표근거, D실제새촬영/HDD분리, E최종고정. 이번 재검출 완료를 전체 시연 완료로 간주하지 않는다.
'''
(D/'RESULT_KO.txt').write_text(body,encoding='utf-8')
graph=read(P/'GRAPH.json');graph.update(updated_at=at,status='SELECTIVE_REOBSERVATION_FOLLOWUP_COMPLETE_DEMO_ACCEPTANCE_REMAINS',current_public_repair=str(D/'RESULT.json'),current_runtime=str(D/'serve.py'))
graph['nodes'].insert(2,dict(id='R_RETRY_FOLLOWUP',label='남은 재검출 실패 원인·시간·앞뒤 겹침 보완',status='complete_public_verified_web_applied',evidence=str(D/'RESULT.json'),result='기존 확인불가7→4; 정상32유지; 사용자8건은 진단만, 재검출0회'))
graph['edges'].append(['R_RETRY','R_RETRY_FOLLOWUP']);graph['edges']+= [['R_RETRY_FOLLOWUP','A'],['R_RETRY_FOLLOWUP','B']]
graph['review'].update(performed_this_turn='Public implementation, guards, captures, hash/HTTP checks; post-admission private inference diagnosis',runtime_modified=True,new_training=0)
write(D/'GRAPH.json',graph)
write(I/'RESULT.json',dict(status='VERIFIED_INTERMEDIATE_SUPERSEDED_BY_ANCHOR_REPAIR',result='Decoder2workers and sharedRGB; originalunknown7→6; private8diagnoses no retry trigger',superseded_by=str(D/'RESULT.json'),private_training=False))
statep=O/'STATE.json';state=read(statep);next_action='새 재검출 보완은 공개 검증 후8821에 적용 완료(기존 확인불가7→4). A화면 근거/B같은 노트북 실행/C발표자료를 이어간다. 저장된 사용자8촬영 진단은 완료했으나 재검출 실행0회이며 자연 실패 복구 증거가 아니다. 실제 새 촬영·자동종료·화면대기·HDD분리는 D, 모두 통과 후 E고정. 사용자 자료는 진단만.'
state['current_stage']=report['status'];state['next_single_action']=next_action
work=state['capture_feedback_completion_20261116'];work.update(status=report['status'],updated_at=at,current_graph=str(D/'GRAPH.json'),active_feedback_runtime=str(D/'serve.py'),next_single_action=next_action)
work['selective_rtmw_followup_20261117']=dict(status=report['status'],report=str(D/'RESULT.json'),text=str(D/'RESULT_KO.txt'),intermediate=str(I),public_remaining=4,previous_remaining=7,training_runs=0,private_diagnosis_only=True,web_admission=str(D/'web/ADMISSION.json'),server=http['server'])
write(statep,state)
contextp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=contextp.read_text(encoding='utf-8')
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
paragraph=f'''\n\n### 2026-11-17 선택적 재검출7건 원인 보완·사용자8촬영 진단
기존 공개 인위누락16시험의 남은7건→4건. 영상읽기2스레드/선택프레임색변환재사용으로시간초과1복구;이미관측된뒷경계의겹침으로앞누락을막던2복구. 실제복구프레임의새양손겹침은두crop모두차단. 남은4는누락순간겹침위험2/중지끝재검출불일치2,실제가림확정으로과장금지. 공개32정상/TCN유지,확장32누락조건24→7,기타제어52·소프트웨어10통과. 공개캡처4자동종료:정상약0.25초,감사복구총0.64/추가0.38초;사용자화면시간아님. 6프레임/300ms/16호출/1.2초예산·가중치·C·T3-F보존,학습0/총이력75유지.
공개판정고정후저장사용자8촬영진단:첫정상사과/감사통과,나머지6손모양차이;일부다른항목확인부족유지. 재검출8건모두0회,최신개선은이전공개손겹침/손가락보완의효과이며재검출실사용효과아님. 개인학습/튜닝/후보선택0·원본보존. 최신8821 PID{http['server']['pid']},카메라꺼짐,HTTP통과,現在95핀/이전90핀보존. 보고 {D/'RESULT_KO.txt'};GRAPH.json. 구현·별도검산Codex,독립전문가아님. A/B/C와실제D/E남음.
'''.replace('現在','현재')
assert h8 in text and h9 in text
text=text.replace(h8,h8+paragraph,1);text=text.replace(h9,h9+f'\n\n### 2026-11-17 재검출 실패 후속 보완 이후\n{next_action}\n최신 그래프 {D/"GRAPH.json"}.\n',1);contextp.write_text(text,encoding='utf-8')
handoff=f'''# 최신 인수인계 — {at}
목표: 정면웹캠 사과(과일)·감사 시범→촬영→자동종료→정상/오류/확인부족→재시도/퀘스트. 5방향은아바타전용.
최신 완료: {D/'RESULT_KO.txt'} / RESULT.json. 원래 공개 인위누락시험 남은7→4: decoder최적화1복구, 이미관측된뒷경계겹침과누락프레임겹침구분2복구. 새관측을넣을순간양손겹침은두crop모두차단. 남은4=누락프레임겹침위험2+중지끝crop불일치2;실제가림4라고단정금지.
검증: 정상32/TCN유지,확장16영상32누락조건24→7,기타제어52·소프트웨어10통과. 공개캡처4자동종료;정상약0.25초,감사복구총0.645초(추가0.380),사과겹침보류0.270. 실제웹캠/화면표시시간아님. 6누락프레임·300ms·16호출·1.2초예산과해상도동일.
추가사용자요청으로공개admission고정후8저장촬영진단. 첫정상사과/감사통과,나머지6손모양차이안내,다른항목불확실남음. 재검출은전부0회:긴누락/겹침위험/관절누락없는판단불확실. 이전공개보완효과와이번재검출효과구분필수. {D/'private_diagnostic_only/RESULT.json'} 개인자료는진단만,학습/튜닝/후보선택절대금지;모든원본보존.
실행 START_WEB_DEMO.cmd→frontal_demo_launcher.py→{D/'serve.py'}; select_front retry v3 (정확한VERSION=selective_front_rtmw_retry_v3). 실제8821 PID{http['server']['pid']},일반환경,카메라꺼짐·HTTP/공개퀘스트오진행방지통과. 시작시실제PID/상태재확인,중복실행금지. localhost제한환경실패로서버없다고추정금지.
보존: C·T3-F/100단어본체/신경망가중치동일,이번학습0·총이력75유지,현재95핀/이전90핀보존. 구현·검증Codex로독립전문가아님. 중간v2 {I} 보존,최종v3위경로.
그래프 {D/'GRAPH.json'}: P/R_RETRY/R_RETRY_FOLLOWUP완료. A/B/C시연마감준비와D새실제촬영/HDD분리,E최종고정은미완료. {next_action}
이번사용자영상진단후기준조정없음. 현재재검출은짧은관절누락만복구;실제로숨은손가락/없는행동을상상해통과시키지않음. 추가전면자료나실제새진단없이남은4를억지로풀기위한임계값변경금지.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
for p in D.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
for p in [statep,D/'RESULT.json',D/'GRAPH.json',D/'web/ADMISSION.json']:read(p)
assert str(D/'serve.py') in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf-8')
assert 'selective_rtmw_anchor_repair_20261117' in (ROOT/'frontal_demo_launcher.py').read_text(encoding='utf-8')
assert all(sha(f)==h for f,h in admission['pins'].items())
write(D/'CLOSEOUT_CHECK.json',dict(passed=True,document_paths_exist=True,json_parsed=True,admission_pins_unchanged=len(admission['pins']),private_after_public_freeze=True,training_runs=0,actual_camera_opened=False,server=http['server']))
print('CLOSED',report['status'],'7 -> 4','pins',len(admission['pins']))
