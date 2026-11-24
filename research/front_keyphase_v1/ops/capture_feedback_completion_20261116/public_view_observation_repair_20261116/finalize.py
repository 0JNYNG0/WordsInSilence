from pathlib import Path
import json,hashlib,datetime,ast
J=Path(__file__).resolve().parent;G=J.parent;O=G.parent;R=O.parents[2];E=G/'public_motion_decoupling_20261116'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,q):Path(p).write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 now=datetime.datetime.now().astimezone().isoformat();rows=read(J/'final_checks/RESULT.json')['rows'];reg=read(J/'final_regression/RESULT.json');capture=read(J/'web/CAPTURE_CHECK.json');consumer=read(J/'web/CONSUMER_CHECK.json');cards=read(J/'web/CARD_CHECK.json');http=read(J/'web/HTTP_CHECK.json');admission=read(J/'web/ADMISSION.json');parent=read(E/'web/ADMISSION.json')
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in admission['pins'].items())
 assert all(admission['pins'][p]==h for p,h in parent['pins'].items())
 assert not any(r['lost']or r['new_warning']for r in rows)
 assert reg['shape_new_wrong']==reg['shape_lost_correct']==reg['distinction_failed']==reg['temporal_failed']==0 and not reg['known_error_loss']
 assert read(J/'final_regression/confirmation/PATTERN_CHECK.json')['passed']and read(J/'confirmation/RESULT.json')['passed']and read(J/'final_checks/CONFIRMATION.json')['passed']
 assert capture['passed']and consumer['passed']and cards['passed']and http['view_posture_active']and not http['camera_active']
 for p in list(J.glob('*.py'))+list((J/'runtime').glob('*.py')):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
 front=[r for r in rows if r['id'].endswith('_F')];assert len(front)==17 and all(all(v=='supported'for v in r['after']['criteria'].values())for r in front)
 stats=dict(public_conditions=len(rows),all_required_supported_before=sum(all(v=='supported'for v in r['before']['criteria'].values())for r in rows),all_required_supported_after=sum(all(v=='supported'for v in r['after']['criteria'].values())for r in rows),normal_warning_items_before=sum(v=='difference'for r in rows for v in r['before']['criteria'].values()),normal_warning_items_after=sum(v=='difference'for r in rows for v in r['after']['criteria'].values()),warning_conditions_before=sum(any(v=='difference'for v in r['before']['criteria'].values())for r in rows),normal_front_original_clips=len(front),new_warning=0,lost_supported=0,shape_anchors=237,new_shape_error=0,lost_shape_correct=0,distinction_controls=40,retained_previous_error_items=True,synthetic_pattern_controls=6,temporal_controls=48,side_partial_hold_controls=8,unselected_public_side_clips=2,consumer_pairs=len(consumer['rows']))
 remaining=[dict(id=r['id'],events=r['after']['events'],unknown=[k for k,v in r['after']['criteria'].items()if v=='unknown'],difference=[k for k,v in r['after']['criteria'].items()if v=='difference'])for r in rows if not all(v=='supported'for v in r['after']['criteria'].values())]
 timing=[{k:r[k]for k in['phase','word','frames','observed','auto_end','redetection','stop_to_result_seconds']}for r in capture['rows']]
 result=dict(status='COMPLETE_VIEW_POSTURE_REPAIR_WEB_APPLIED',at=now,stats=stats,training=dict(admitted='Projected hand-plane head: 54 existing front anchors + 48 source-reviewed FIT side anchors, 102 rows / 9 people / 15 views, three fixed-seed fine-tunes.',not_admitted='Additional palm fine-tuning (190 rows, 3 seeds) and relation fine-tuning (162 rows, 3 seeds) showed no additional improvement in the scoped comparison; retain deployed versions.',actual_runs=9,admitted_runs=3,whole100_TCN_and_ConvGRU_unchanged=True),changes='Only posture head weights and isolated integration. Old handshape/finger/movement weights, event boundaries, C-T3F, UI and auto-stop retained. No word/person exceptions or confidence threshold relaxation.',remaining=remaining,residual_causes=dict(finger='Example13_L: native source and keypoints show foreshortened/occluded left fingers; do not assert hidden extension or infer fist solely from projected curl.',endpoint='Example11_L: temporal motion detects approach/separation, but body/palm posture head stays separated1 rather than near2 around visible arrivals. New right-view relation fine-tuning did not fix it; no invented completion.',target='Examples03_L/12_L: narrow projected palm and endpoint samples do not provide two frames within existing FIT-calibrated target-region envelope. Unknown retained, not physical-contact rejection.'),checks=dict(public='final_checks/RESULT.json',regression='final_regression/RESULT.json',new_views='confirmation/RESULT.json',final_confirmation='final_checks/CONFIRMATION.json',cards='web/CARD_CHECK.json',consumer='web/CONSUMER_CHECK.json',capture='web/CAPTURE_CHECK.json',http='web/HTTP_CHECK.json',native_audit='audit_INDEX.json',residual_numeric='RESIDUAL_NUMERIC.json'),web=dict(url='http://127.0.0.1:8821/',pid=http['server']['pid'],entrypoint=str(J/'serve.py'),launcher='START_WEB_DEMO.cmd -> view_demo_launcher.py',camera_active=False),capture_timing=timing,pins=len(admission['pins']),preserved_parent_pins=len(parent['pins']),private_data_read_or_trained=False,external_downloads=0,verification_actor='Codex implementation and source/serialized-output/hash/runtime checks; no separate agent or KSL expert validation.',limitations=['Public conditions include reused views/resolutions, not 46 independent people or learner accuracy.','Unselected10L/R are public camera-view confirmation, not proof of arbitrary new-person generalization.','Physical webcam/browser-paint latency and real learner normal/error/retry remain unverified this turn.','Projected plane describes visible 2D posture, not calibrated3D palm orientation or physical contact.','Prior apple4-frame synthetic handshape miss and fully HDD-unplugged demo are outside this completed repair.'])
 write(J/'RESULT.json',result);write(J/'PRESERVATION_VERIFIED.json',dict(at=now,passed=True,pins=len(admission['pins']),parent_pins=len(parent['pins']),all_hashes_match=True,private_data_used=False))
 graph=[dict(id='A',work='공개 원본·관절·판정값 원인 분리',status='complete'),dict(id='B',work='학습용 옆시점 주석 보강·자세 판정부 미세조정',status='complete'),dict(id='C',work='정상·오류·짧은 동작·새 공개시점 비교',status='complete'),dict(id='D',work='통과한 부품만 웹 연결·자동 종료·속도 확인',status='complete'),dict(id='REMAIN',work='옆시점 손가락 가림·도착 상태/위치 근거 부족',status='unresolved_not_forced_to_pass'),dict(id='USER',work='실제 웹캠 정상→오류→교정→재시도',status='pending_user_started_capture_not_training')]
 write(J/'GRAPH.json',dict(at=now,nodes=graph,scope='이번 옆시점 보완 루프와 남은 한계. 프로젝트 전체 완성률 아님.'))
 def times(word):
  a,b=[r for r in timing if r['word']==word];return f"{a['stop_to_result_seconds']:.3f} → {b['stop_to_result_seconds']:.3f}초"
 report=f'''옆시점 보완 결과

옆시점 문제란?
같은 정상 수어도 카메라가 옆으로 가면 손가락이 겹쳐 보이거나 손바닥이 짧게 보입니다. 이번에는 관절이 비교적 잘 잡혔는데도 세운 손을 눕힌 손으로 분류한 문제가 확인됐습니다. 모든 옆시점 문제가 단순 관절 누락이라는 뜻은 아닙니다.

이번에 바꾼 것
- 공개 학습 사람 9명의 기존 정면 정답54개에, 그중3명의 좌우 시점 정답48개를 더해 손 자세 판정부를 미세조정했습니다. 총102개는 영상 수가 아니라 특정 프레임의 한 손 자세 정답 수입니다.
- 추가 손모양·반복 판정부 학습도 시험했지만 개선 근거가 없어서 배포하지 않았습니다. 최종 적용은 효과가 확인된 손 자세 판정부3seed입니다. 신경망 학습을 실제로 했으며, 특정 단어에서 무조건 통과시키는 예외나 느슨한 임계값을 넣은 것이 아닙니다.
- 100단어 TCN, 기존 공통Conv+GRU 본체, 손가락/손모양/반복 판정부, C·T3-F와 기존 동작 구간은 유지했습니다.

이전 → 현재
- 46개 공개 영상 조건의 정상 자세 오답 경고: {stats['normal_warning_items_before']} → {stats['normal_warning_items_after']}개. 이전 경고는 {stats['warning_conditions_before']}개 조건에 걸친 개별 손 자세 경고입니다.
- 모든 요구 항목을 확인한 조건: {stats['all_required_supported_before']} → {stats['all_required_supported_after']}/46. 이는 사용자 정확도나 서로 다른46명이라는 뜻이 아닙니다. 정면 원본17개는 모두 정상 확인 유지.
- 손모양237정답에 추가 오판/정답 소실0. 기존 남아 있던 손모양 오판을 모두 해결했다는 뜻은 아닙니다.
- 기존 오류 구분40조건, 다른 손가락 모양6합성조건, 속도·크기·중간정지·누락48조건 및 옆시점 부분수행·정지8조건 유지. 기존 명확한 오류 지적을 확인부족으로 숨긴 경우도 없음. 역재생5조건은 정답이 달라질 수 있어 진단만.
- 후보 학습/선택 전 쓰지 않았던 공개10번 수행자 좌우 영상2개를 추가 확인. 오른쪽 시점의 왼손 자세 확인부족이 해소됐고, 왼쪽 시점의 목표위치 확인부족은 그대로 표시. 이후 최종 부품 축소 검증에는 이미 알려진 자료로 구분.
- 실제 웹 소비부12쌍에서 TCN 후보와 DTW 연결 유지. 다른 손가락 모양을 넣은 공개 합성 장면은 1.83~2.30초의 오른손 강조/교정 문장, 불확실한 왼손은 오답과 다른 확인부족 문장 유지.
- 공개 영상 카메라 재생 시험의 종료→결과 계산: 사과 {times('WORD0099')}, 감사 {times('WORD1290')}. 3초 시작·손 내림 자동 종료·정상 통과 유지, 수신123/125프레임 모두 관측, 재검출0. 실제 사람 웹캠/브라우저 표시시간 측정은 아닙니다.

남은 문제
일부 옆시점은 가려진 손가락 모양과 도착 위치/반복 끝을 아직 확정하지 못합니다. 11_L은 접근·떼는 움직임은 잡지만 끝 자세를 '가까워짐'으로 분류하지 못하는 문제가 남았습니다. 13_L은 왼손가락 겹침이 큽니다. 03_L/12_L은 끝 위치의 확인 가능한 관측이 부족합니다. 이들을 완료나 오답으로 억지 결정하지 않았습니다. 모든 수어 단어·촬영 환경에 대한 성능 보장도 아닙니다.

웹 적용: http://127.0.0.1:8821/ 새로고침. START_WEB_DEMO.cmd로도 최신 버전 시작. 카메라는 꺼진 상태입니다.
기존 {len(parent['pins'])}개 파일 해시 보존, 최신 {len(admission['pins'])}개 검증. 사용자 촬영 읽기/학습/튜닝/선택0. 새 다운로드·외부전송·커밋0. Codex 자체 원본검토/실행검증이며 독립 수어전문가 검증은 아닙니다.
'''
 (J/'RESULT_KO.txt').write_text(report,encoding='utf-8')
 next_action='최신 웹8821의 사과·감사 정상→명확한 손모양/방향 오류→교정 재시도를 사용자가 시작하면 진단한다. 사용자 자료는 학습·튜닝·선택 금지. 옆시점 반복/목표위치가 필요하면 공개FIT에서 실제 도착/떼기 경계와 가림 정답을 먼저 보강하고, 기존17정면·오류/부분/속도 검증을 유지하는 독립 후보만 비교한다. 이번 자세 보완은 검증 완료이므로 근거 없이 재학습하지 않는다. 숨은 손가락/3D접촉을 추측하여 통과시키지 않는다. 실제웹캠 표시시간·HDD분리시연은 미검증.'
 state=read(O/'STATE.json');state.update(current_stage='PUBLIC_VIEW_POSTURE_REPAIR_WEB_APPLIED',next_single_action=next_action,last_updated=now);parentstate=state['capture_feedback_completion_20261116'];prior=parentstate.get('public_view_observation_repair_20261116',{});already=prior.get('status')==result['status']
 parentstate.update(status=state['current_stage'],updated_at=now,report=str(J/'RESULT_KO.txt'),training_runs=parentstate['training_runs']+(0 if already else 9),training_runs_current_task=9,training_runs_current_task_admitted=3,current_work_graph=str(J/'GRAPH.json'),current_checkpoint=str(J/'PROGRESS.json'),next_single_action=next_action,web_admitted=True,private_data_used=False)
 parentstate['public_view_observation_repair_20261116']=dict(status=result['status'],at=now,report=str(J/'RESULT_KO.txt'),result=str(J/'RESULT.json'),graph=str(J/'GRAPH.json'),stats=stats,pid=http['server']['pid'],session_id=None,training_private=False,next_single_action=next_action);write(O/'STATE.json',state)
 context=R/'WordsInSilence_PROJECT_CONTEXT.md';text=context.read_text(encoding='utf-8-sig');heading='### 2026-11-16 옆시점 손 자세 오판 보완 / 검증 부품만 웹 적용';assert heading not in text
 update=f'\n\n{heading}\n공개 원본과 관절을 대조해 관측 부족과 자세 분류 오류 분리. 정면54+FIT3명좌우48=102개 한손 자세 정답(전체9명15영상),3seed 미세조정 적용. 추가 palm190행/관계162행 각3seed는 개선 없어 미적용;총9학습run중3적용.46공개조건 정상 자세 경고{stats["normal_warning_items_before"]}→0,모든항목확인{stats["all_required_supported_before"]}→{stats["all_required_supported_after"]};17정면정상·237손모양·40오류구분·6패턴·56속도/부분/누락유지.미사용10좌우2개확인(새사람일반화보장아님).웹소비부12쌍TCN/DTW유지,오류강조/불확실구분확인.공개캡처종료후사과{times("WORD0099")}/감사{times("WORD1290")}·자동종료유지(실제웹캠표시시간아님).8821 PID{http["server"]["pid"]},view_posture_active/카메라off,view_demo_launcher.기존142핀/전체153핀검증,C·T3-F/100본체/기존손모양·반복유지,개인자료0.남은옆시점은손가락가림/도착상태/목표위치불확실.근거 {J.relative_to(R).as_posix()}/RESULT.json·RESULT_KO.txt·GRAPH.json.\n'
 marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';assert marker in text;text=text.replace(marker,marker+update,1);marker='## 9. 다음 단일 작업';assert marker in text;text=text.replace(marker,marker+'\n\n### 2026-11-16 검증된 옆시점 자세 보완 적용 이후\n'+next_action+'\n',1);context.write_text(text,encoding='utf-8')
 handoff=f'''# 최신 인수인계 — {now}

## 현재 결과
이번 옆시점 원인분리→공개FIT 주석/학습→회귀/미사용 공개시점→웹 연결 루프 완료.
근거 폴더: {J}
RESULT.json / RESULT_KO.txt / GRAPH.json / final_checks / final_regression / confirmation / web.

## 실제 적용
final_candidate.PlaneViewLesson. 공개 정면54 + FIT1/3/7 좌우48 =102개 한손 자세 정답, 총9명15영상. 기존PlaneHead42특징→24→2의3seed 미세조정. 같은confidence/실행조건. 세운 손/눕힌 손의 2D 관측 분류이며3D접촉/숨은손가락복원아님.
새palm190행3seed와관계162행3seed는 비교결과 추가효과없어 미적용. 실제9run중3run적용. 100단어TCN/공통Conv+GRU본체,C·T3-F,기존손모양/개별손가락/반복모델과구간그대로.

## 근거와 한계
46조건 정상자세경고5→0(4조건),모든요구항목31→34. 정면원본17정상유지. 237손모양추가오판/정답소실0(이전오판1해결주장아님).40오류조건+6패턴+48속도/크기/부분/누락+8옆시점부분/정지통과;기존오류지적소실0.역재생5진단만.
미사용10L/R 공개영상2개는후보선택전확인;10R왼손자세unknown회복,10L목표위치unknown그대로.최종후보부품축소후에는이미알려진확인자료로구분.일반사용자정확도아님.
잔여:11L움직임은검출하나끝자세head가near2아닌separated1을내므로반복미완료.13L왼손 foreshortening/가림으로손모양unknown.03L/12L두프레임목표위치근거부족.전체잔여는RESULT.json.remaining.이를성공으로강제하지않음.기존사과4프레임합성손모양누락은이전한계유지.
웹소비부12쌍TCN/DTW유지.공개합성오류1.83~2.30초오른손강조/문장과불확실왼손회색구분.공개카메라재생사과{times('WORD0099')},감사{times('WORD1290')},123/125모든프레임관측/재검출0/3초시작·자동종료·정상통과유지.실제물리웹캠/브라우저시간아님.
Codex구현및원본/저장결과/해시/실행검증;별도에이전트나수어전문가검증아님.개인촬영읽기/학습/튜닝/후보선택0.

## 실행
http://127.0.0.1:8821/ 실제PID{http['server']['pid']};ready/cameraoff. 실행환경일반(require_escalated).
entrypoint: {J/'serve.py'}
START_WEB_DEMO.cmd -> view_demo_launcher.py. 이전demo_launcher.py/motion_demo_launcher.py/WEB서버/모델들은153핀안에보존(부모142핀불변).
학습/검증프로세스종료,웹서버만유지. 새worker는final_candidate. 기존E서버와그프로젝트자식만교체한기록web/ACTIVATION.json.

## 다음 한 작업
{next_action}
'''
 (R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8');write(J/'PROGRESS.json',dict(status=result['status'],at=now,graph=graph,running_sessions=[],server_pid=http['server']['pid'],next_single_action=next_action))
 print('FINALIZED',stats,'pid',http['server']['pid'],flush=True)
if __name__=='__main__':main()
