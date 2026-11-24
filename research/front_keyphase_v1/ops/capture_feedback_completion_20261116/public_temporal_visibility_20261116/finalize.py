from pathlib import Path
import json,hashlib,datetime,ast
L=Path(__file__).resolve().parent;G=L.parent;O=G.parent;R=O.parents[2];K=G/'public_view_arrival_support_20261116'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 now=datetime.datetime.now().astimezone().isoformat();comp=read(L/'qualified_checks/RESULT.json');reg=read(L/'qualified_regression/RESULT.json');extra=read(L/'verified_extra/RESULT.json');http=read(L/'web/HTTP_CHECK.json');cap=read(L/'web/CAPTURE_CHECK.json');pins=read(L/'web/ADMISSION.json')['pins'];parentpins=read(K/'web/ADMISSION.json')['pins']
 assert comp['regressions']==0 and reg['shape_new_wrong']==reg['shape_lost_correct']==reg['distinction_failed']==reg['temporal_failed']==0 and not reg['known_error_loss']and extra['passed']
 for n in ['CONSUMER_CHECK','CARD_CHECK','ENDPOINT_CARD_CHECK','UNCERTAIN_LOCATION_CHECK','CAPTURE_CHECK']:assert read(L/'web'/f'{n}.json')['passed'],n
 assert http['temporal_visibility_active']and not http['camera_active']and all(r['overall_pass']for r in http['rows'])
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items())and all(pins[p]==h for p,h in parentpins.items())
 for p in list(L.glob('*.py'))+list((L/'runtime').glob('*.py')):ast.parse(p.read_text(encoding='utf8'),filename=str(p))
 changes=[r for r in comp['rows']if r['before']!=r['after']];training=read(L/'palm/FROZEN.json');times=[{k:r[k]for k in ['phase','word','auto_end','frames','observed','redetection','stop_to_result_seconds']}for r in cap['rows']]
 def timing(word):
  a,b=[r for r in times if r['word']==word];return f"{a['stop_to_result_seconds']:.3f} → {b['stop_to_result_seconds']:.3f}초"
 remaining=[]
 for r in comp['rows']:
  unknown=[k for k,v in r['after'].items()if v=='unknown']
  if unknown:remaining.append(dict(id=r['id'],unknown=unknown))
 result=dict(status='COMPLETE_TEMPORAL_VISIBILITY_WEB_APPLIED',at=now,stats=dict(known_public_conditions=len(comp['rows']),all_required_supported_before=comp['full_before'],all_required_supported_after=comp['full_after'],changed_conditions=len(changes),new_normal_warnings=0,lost_supported=0,all_previous_event_bounds_preserved=True,added_observed_cycles=sum(r['events_after']-r['events_before']for r in comp['rows']),shape_truth_anchors=237,shape_new_wrong=0,shape_lost_correct=0,error_controls=40,distinct_pattern_controls=6,temporal_controls=48,side_partial_hold_controls=8,missing_digit_controls=12,additional_camera_views=2),training=dict(actual_runs=3,admitted_runs=3,coarse_palm_labels=len(training['rows']),prior_labels=190,new_labels=32,fit_people=list(range(1,10)),new_label_sources='Existing public FIT people2/5 L/R, native RGB separated palms reviewed. No exact hidden-finger labels.',method='Fixed three-seed supervised fine-tuning of hand-local temporal coarse-form head, 500steps/seed. Original negatives rehearsed; source-normalized features and confidence.85 preserved.',frozen='C-T3F,100-word TCN, common Conv+GRU backbone, exact finger observers and old archived checkpoints'),changes=changes,corrections=['Add qualified whole-palm evidence only with independently observed extended finger, no distinct-pattern/projection conflict or missing digit geometry. Individual hidden digits never filled.','Separate learned approach from observed target arrival; reject explicit non-action preparation and preserve old cycles. No expected-count-driven events.','Preserve the reliability of uncertain spans when adding events; new interval is not new evidence of correctness.','Localize gray unknown display to actual uncertain spans rather than a neighbouring supported interval.'],rejected_or_corrected=['Initial unqualified palm recovery generated two new wrong anchors. Added independent digit/observer disagreement qualification; final237 has zero extra wrong/lost correct.','Initial distance-only endpoint extension counted preparation; source-trained body action posture qualification fixed it.','Interval recomposition promoted a known ambiguous palm to difference; exact uncertainty provenance now retained.'],remaining=remaining,limitations=['50 conditions are reused public clips/camera/resolution variants, not50 independent people or an unbiased accuracy estimate.','Additional4L/R camera views were unused at the first candidate check; after that check they are reused verification, not final untouched holdout. Person4 frontal is FIT, no new-person generalization claim.','A heavily foreshortened palm remains unknown; hidden individual finger state is not established.','Some side-view repetitions share an extended event interval or have insufficient near/motion observations; remaining12conditions are not forced to pass.','Cutting directly at an arrival can remove the following temporal window context. A60-frame public partial fixture retains the first full cycle and remains incomplete.','Real webcam and browser-paint latency, expert educational correctness, and all-word coverage not established by these tests.'],capture_timing=times,web=dict(url='http://127.0.0.1:8821/',pid=http['server']['pid'],entrypoint=str(L/'serve.py'),launcher='START_WEB_DEMO.cmd -> temporal_demo_launcher.py',camera_active=False),pins=len(pins),parent_pins=len(parentpins),private_recordings_read_or_used=False,implementation_actor='Codex',verification_actor='Same Codex separate source review, executable comparisons and serialized output/hash checks; no separate agent or KSL expert',evidence=dict(comparison='qualified_checks/RESULT.json',regression='qualified_regression/RESULT.json',extra='verified_extra/RESULT.json',capture='web/CAPTURE_CHECK.json',uncertainty='web/UNCERTAIN_LOCATION_CHECK.json',endpoint='web/ENDPOINT_CARD_CHECK.json',http='web/HTTP_CHECK.json'))
 write(L/'RESULT.json',result);write(L/'PRESERVATION_VERIFIED.json',dict(passed=True,pins=len(pins),parent_pins=len(parentpins),all_hashes_match=True,private_data=False))
 graph=dict(at=now,nodes=[dict(id='A',label='원본·관절·손모양·반복 끝 원인 분리',status='complete'),dict(id='B',label='공개 정답 보강·손모양 학습·도착 연결 보완',status='complete'),dict(id='C',label='정상·오류·부분 수행·가림 비교와 발견 회귀 수정',status='complete'),dict(id='D',label='웹 표시·공개 촬영 자동 종료·시간 확인 후 적용',status='complete'),dict(id='LIMIT',label='심한 가림·일부 옆시점 반복은 확인 부족 유지',status='remaining_observation_limit'),dict(id='USER',label='실제 웹캠 정상→오류→교정 재시도',status='pending_user_only_not_training')]);write(L/'GRAPH.json',graph)
 text=f'''남은 손모양·반복 끝 보완 결과

바뀐 점
1. 옆에서 짧게 보이는 편 손을 구분하도록 공개 손모양 정답222개(기존190+추가32)로 작은 손 전체 모양 판정부를3번 미세조정했습니다. 기존 손가락별 판단은 유지했습니다. 새 손바닥 판정만 믿어 생긴 추가 오판2개도 확인했고, 실제 편 손가락의 별도 관측과 함께 확인하도록 고쳐 최종 추가 오판은0개였습니다.
2. 손이 다가온 사실과 목표 위치까지 도착한 사실을 나누어 연결했습니다. 공개 옆시점11L·13L에서 한 번만 찾던 반복을 두 번 확인했습니다.13L의 왼손 모양은 여전히 불확실하므로 전체 통과시키지는 않습니다. 준비 동작을 반복으로 세던 초기 문제도 학습된 양손 관계로 걸러냈습니다.
3. 불확실한 손모양을 다시 오류로 바꾸는 연결 문제를 고쳤습니다. 화면의 회색 표시도 엉뚱한 정상 구간 대신 실제 확인이 부족한 구간(공개13L의53~57프레임)을 가리키게 했습니다.

이전 → 현재
- 알고 있던 공개50조건 중 모든 요구 항목 확인:35 → 38.50명의 정확도가 아니라 촬영 각도·해상도 등을 포함한 기존 공개 조건 비교입니다.
- 새로 보완된 손모양3조건, 반복2조건. 기존 정면17조건과 모든 기존 동작 구간 유지.
- 손모양237정답 추가오판/정답소실0. 오류구분40조건, 별도손가락패턴6조건, 속도·크기·부분·누락48조건, 측면부분/정지8조건, 손가락누락12조건 통과.
- 추가공개4번수행자의좌우2영상 확인. 정면은 학습에 들어간 사람이고, 초기후보 점검 후 재사용했으므로 미사용 새사람 일반화 시험으로 과장하지 않습니다.
- 두 번 수행한 공개 영상은 통과, 한 번만 완료한 시험은1회확인/전체미통과. 알려진 손모양 차이는1.83~2.30초의 오른손 강조와 짧은 교정 문장으로 유지.
- 공개영상 카메라 재생에서 종료→결과계산:사과 {timing('WORD0099')},감사 {timing('WORD1290')}. 자동종료·3초시작·정상통과 유지,122/124수신프레임 모두관측,재검출0. 실제 웹캠/브라우저 표시시간은 아닙니다.

남은 것
가려진 손가락의 실제 상태를 상상해서 채우지 않습니다. 심한 옆시점의 손모양, 두 반복이 하나의 긴 구간처럼 잡히는 사례, 끝부분 관측이 부족한 사례는 아직 확인 부족으로 남습니다. 모든 단어·각도가 완성됐다는 결과가 아닙니다.공통 코드·학습은 단어/사람ID 예외가 없지만 현재 웹 시연 검증 범위는 사과·감사입니다.

최신 화면 http://127.0.0.1:8821/ (새로고침). 카메라 꺼짐. 기존 C·T3-F/100단어 TCN/공통Conv+GRU본체 유지. 내 촬영 읽기·학습·튜닝·후보선택0. 새다운로드·외부전송·Git작업0.
구현과 검증:Codex 직접 수행, 별도 에이전트/수어전문가 검증은 아님. 근거:같은 폴더 RESULT.json 및 각검증JSON.
''';(L/'RESULT_KO.txt').write_text(text,encoding='utf8')
 next_action='최신8821 사과·감사 실제 촬영 정상→명확한 오류→교정 재시도를 사용자가 시작하면 진단한다(학습/튜닝/선택 금지). 공개 측면의 남은 중복된 구간·끝관측은 실제 떼기/도착 시점 정답과 몸/관측손바닥 정규화 근거로 추가연구하며, 가려진 손가락을 정답으로 채우지 않는다. 이번 검증을 이유없이 반복하지 않고 실제시연/발표 결과를 우선한다.'
 state=read(O/'STATE.json');p=state['capture_feedback_completion_20261116'];accounted=p.get('public_temporal_visibility_20261116',{}).get('training_runs_accounted',False);state.update(current_stage='PUBLIC_TEMPORAL_VISIBILITY_WEB_APPLIED',last_updated=now,next_single_action=next_action);p.update(status=state['current_stage'],updated_at=now,report=str(L/'RESULT_KO.txt'),training_runs=p['training_runs']+(0 if accounted else 3),training_runs_current_task=3,training_runs_current_task_admitted=3,current_work_graph=str(L/'GRAPH.json'),current_checkpoint=str(L/'PROGRESS.json'),next_single_action=next_action,web_admitted=True,private_data_used=False);p['public_temporal_visibility_20261116']=dict(status=result['status'],at=now,path=str(L),result=str(L/'RESULT.json'),report=str(L/'RESULT_KO.txt'),graph=str(L/'GRAPH.json'),stats=result['stats'],training_runs_accounted=True,pid=http['server']['pid'],session_id=None,private_data=False,next_single_action=next_action);write(O/'STATE.json',state);write(L/'PROGRESS.json',dict(status=result['status'],at=now,live_server_pid=http['server']['pid'],training_and_checks_finished=True,next_single_action=next_action))
 context=R/'WordsInSilence_PROJECT_CONTEXT.md';t=context.read_text(encoding='utf-8-sig');heading='### 2026-11-16 가림·반복 끝 원인 분리 / 검증된 보완 웹 적용';update=f'\n\n{heading}\n공개222손모양정답(기존190+FIT2/5좌우32)3seed미세조정,독립손가락근거/충돌/누락검사와결합. 초기추가오판2건을수정해237정답추가오판/정답소실0. 학습된접근+실제목표도착과몸동작관계연결로11L/13L반복1→2,13L손모양은unknown유지. 준비동작횟수화/불확실→오류재승격을수정.50공개조건전체확인35→38,정면17/기존모든경계보존.40오류/6패턴/48시간·누락/8측면부분/12손가락누락통과. 추가4LR재사용검증(새사람평가아님). 웹12쌍TCN/DTW유지,오류교정/부분1회/불확실53~57프레임위치표시확인. 공개촬영사과{timing("WORD0099")},감사{timing("WORD1290")}·자동종료유지;실제웹캠표시시간아님.8821 PID{http["server"]["pid"]} ready/cameraoff,temporal_demo_launcher.부모163/전체177핀보존,C·T3-F/100단어TCN/공통Conv+GRU/정확손가락동결,개인자료0.심한가림/일부옆시점반복관측남음.근거 {L.relative_to(R).as_posix()}/RESULT.json·RESULT_KO.txt·GRAPH.json.\n';assert heading not in t;t=t.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'+update,1);t=t.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n### 2026-11-16 가림·반복 보완 적용 이후\n'+next_action+'\n',1);context.write_text(t,encoding='utf8')
 (R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — {now}

## 완료·현재 적용
{L}
RESULT.json / RESULT_KO.txt / GRAPH.json이 이번 근거. A원인분리→B공개정답·학습/연결→C회귀수정/검증→D웹적용 완료.
TemporalVisibilityLesson:공개222손모양정답3seed,편손가락의독립근거가있는전체손모양회복.개별가린손가락추론없음.학습된접근+실제도착,몸관계로준비제외,기존모든이벤트보존.기존불확실성유지하며회색표시정확구간화.도착기준은기존FIT17프레임기준유지,단어/사람ID예외없음.공통Conv+GRU/100단어TCN/C·T3-F/정확손가락동결.

## 결과·한계
50알려진공개조건 전체확인35→38,손모양3조건/반복2조건개선.정면17정상/모든기존구간유지.237손모양추가오판/소실0,40오류+6패턴+48시간/누락+8측면부분/정지+12손가락누락통과.4LR추가공개시점은첫점검후재사용/정면FIT사람이므로새사람holdout아님.11L전체2회통과,공개60프레임부분은1회미완료.13L반복2회이나왼손unknown유지/전체미통과.불확실표시66~68에서실제53~57로정정.가린손가락/16L끝관측/6LR같은중첩구간한계남음.성공으로강제하지않음.
초기후보는손모양새오판2개/준비동작횟수화/불확실→오류승격이있어수정.최종qualified_*가판정근거;shape/endpoint/combined/admission_checks/regression은중간후보이력.

## 웹·검증
http://127.0.0.1:8821/ PID{http['server']['pid']} ready/cameraoff,entrypoint {L/'serve.py'}
START_WEB_DEMO.cmd -> temporal_demo_launcher.py. 이전K/arrival_demo_launcher포함177핀불변(부모163). 모델훈련3회전부적용,누적STATE69회.웹소비12쌍TCN/DTW유지.공개실제RTMW재생:사과{timing('WORD0099')},감사{timing('WORD1290')} 종료→결과계산,122/124모든수신프레임/재검출0/3초시작·자동종료.실제웹캠/브라우저표시시간아님.웹cap/cards/endpoint/location/HTTP JSON모두통과.
Codex구현+별도출력/해시/실행검증,별도에이전트나수어전문가없음.사용자촬영읽기/학습/튜닝/후보선택0.새다운로드/외부전송/커밋0.학습/검증실행종료,서버만유지.

## 다음 한 작업
{next_action}
''',encoding='utf8')
 print('FINALIZED',result['stats'],len(pins),flush=True)
if __name__=='__main__':main()
