from pathlib import Path
import json,hashlib
from datetime import datetime,timezone,timedelta
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;ROOT=O.parents[2];R=G/'selective_rtmw_anchor_repair_20261117'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
j=read(D/'DIAGNOSIS.json');rows=j['rows'];pins=read(R/'web/ADMISSION.json')['pins']
assert len(rows)==3 and all(sha(f)==h for f,h in pins.items())
assert all(sha(f)==h for r in rows for f,h in r['source_hashes'].items())
assert all(r['received']==r['completed']==r['frames'] and not r['dropped'] and not r['errors'] and not r['missing_ranges'] for r in rows)
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
plan=[
 dict(id='R1',label='앉은 자세의 자동 종료 보완',priority=1,evidence='3/3manual despite readiness=true. User waited. After hands lowered, necessary rest condition lasts at most0.517/0/0.195s, below existing0.750s. Border hand estimates keep high=true.',actions=['공개 정면 영상을 기존 상반신 조건과 손이 아래로 나가는 조건으로 재생하여 같은 high 오판을 재현한다. 기존 대기 기준·손 활성 근거·종료 보류 이유를 따로 기록한다.','화면 하단의 일부 추정점과 화면 안에서 실제로 들고 움직이는 손을 구분하고, 신체 기준 팔 내림·시작 대기 자세 복귀 근거를 함께 사용한다. 화면의 고정 가로선으로 사용자 손을 잘라내거나 손 미검출만으로 종료하지 않는다.','한손/양손 정상 종료, 동작 중 교차·짧은 가림·잠깐 멈춤·낮은 위치의 동작을 함께 검증한다.'],gate='공개 재현 정상은 자동 종료, 진행 중·가림·잠깐 멈춤은 조기 종료 없음. 기존 피드백/TCN 출력·속도 유지.',avoid='사용자 영상에 맞춘 임계값·대기시간 변경, 시작 자세 원본 없이 정확한 baseline거리 실패를 지어내기'),
 dict(id='R2',label='두 번째 움직임의 마지막 상태 확인',priority=2,evidence='First thanks: movement candidate56..66 exists but endpoint_unconfirmed. Learned flow did not wholly miss second action. User says2touches; selected RGB shows2approaches. No claim of expert3D contact truth.',actions=['공개 정면에서 짧은2회 도달·보통2회·1회·가까이 가다 멈춤·근접 자세 유지의 정답 시점을 대조한다.','관절 관측 부족, 마지막 자세 분류, 움직임과 도달 구간 연결을 분리하여 어느 단계가 누락을 만드는지 확인한다. 실제 속도 때문이라고 미리 가정하지 않는다.','연결 조건 문제면 공통 연결을 수정하고, 관절이 충분한데 학습 판정이 잘못되면 해당 공통 판정부의 공개 정답만 보강해 필요한 추가 학습을 비교한다.'],gate='짧게라도 도달한 두 행동은 구별하되, 한 번·미도달·자세 유지가 두 번 완료로 바뀌지 않는다. 실제 가림은 보류 유지.',avoid='기대 횟수가2라는 이유만으로 두 번째 행동을 채워넣기, DTW로 없는 행동 확정, 사용자 촬영 학습·튜닝'),
 dict(id='R3',label='두 보완 통합 및 기존 정상·오류 검증',priority=3,actions=['두 보완을 함께 적용한 후보에서 공개32정상, 알려진 손모양 차이, 반복·누락 대조, 자동 종료 및 추가시간을 확인한다. 이미 검증된 무관한 학습 전체를 반복하지 않는다.','정상/교정/관측부족 문장과 강조 구간이 같은 근거를 가리키는지 확인하고 통과분만 현재 웹 화면에 연결한다. C·T3-F와 현재 모델 보존을 해시로 확인한다.'],gate='정상 유지·중요 오류 유지·가림 강제통과0·조기종료0. 결과 계산/자동종료대기/브라우저 표시를 분리 보고.'),
 dict(id='R4',label='두 단어 시연 흐름 마감',priority=4,actions=['사용자 가능 시 사과·감사의 정상→명확한 손모양 오류→교정 재시도를 최소 확인한다. 새촬영은 진단만 사용한다.','사과→감사 실전 퀘스트의 성공 진행과 오류 미진행, 3D시범·교정 표시·재시도를 확인한다. 로컬 실행/HDD분리 확인 뒤 버전과 발표 근거를 고정한다.'],gate='사용자가 수동종료 없이 정상 통과·오류 위치 안내·교정 재시도·퀘스트를 완주. 새 단어/LLM/Unity확장은 이후.')]
write(D/'PLAN.json',dict(status='REVIEWED_PLAN_NOT_IMPLEMENTED',at=at,implementation_authority='Current request is check only and propose plan; no runtime or training changes this turn',steps=plan,edges=[['DIAGNOSIS','R1'],['R1','R2'],['R2','R3'],['R3','R4'],['R3','R1_or_R2_if_failed']],prior_graph=str(R/'GRAPH.json'),prior_graph_mapping='R1/R2 use existing R repair branch; R3 completes public gates then A display check; R4 covers D actual capture/quest and B portability, E finalfreeze. C evidence may be prepared from verified results.',scope='All feedback source validation front-only; private recordings diagnosis only;5views avatar only'))
summary=dict(status='LATEST_USER_RETEST_DIAGNOSED_TWO_ISSUES_PLAN_ONLY',at=at,recordings=3,labels=['사과','감사','감사'],outcomes=['supported','unknown_repeat_endpoint','supported'],tcn_top1_all_selected=True,auto_stop_failed=3,manual_stops=3,frames=sum(r['frames']for r in rows),dropped=0,stop_to_browser_ack=[r['stop_to_browser_ack']for r in rows],retry_calls=0,user_reply=j['user_reply'],issues=[dict(id='rest_border_high',confirmed='Hands visibly lowered, but hand-high flag prevents uninterrupted rest; same normalized threshold requires750ms. Preview rest geometry not saved, so exact baseline distance is not known.'),dict(id='second_arrival_unconfirmed',confirmed='Saved movement model output includes second approach56..66, but endpoint unconfirmed. User states2touches. Observation/static classifier/link cause remains to be split on public material.')],plan=str(D/'PLAN.json'),training=False,tuning=False,candidate_selection=False,new_inference=False,runtime_changed=False,private_only_diagnosis=True,prior_runtime_preserved=str(R/'serve.py'),pins_unchanged=len(pins),verification='Same Codex saved result/ack/stream arrays, source snapshots and code review; no independent expert annotation',limits=['3normal-intended attempts are not a generalization accuracy test.','No intentional new error trial this time; error detection claims stay on previous evidence.','Result display timings start at manual stop and exclude waiting for failed automatic stop.','Speed is not yet established as reason for missed terminal state.'])
write(D/'RESULT.json',summary)
body='''最新촬영 확인 — 2026-11-17 (진단·계획만, 수정 없음)

결과
사과1회: 요구 항목 모두 통과. 감사1회차: 손모양·방향·상대 위치 확인, 반복의 마지막 상태는1회만 확인하여 보류. 감사2회차:2회 확인하여 통과.
세 촬영 모두 TCN1순위가 선택한 단어. 총527프레임 전부 처리, 큐누락·처리오류0. 재검출0회.
수동 종료→브라우저 표시 확인은0.88/0.79/0.54초. 종료를 기다리던 시간은 이 숫자에 포함하지 않음.

사용자 답변과 확인된 문제
세 번 모두 손을 내리고 기다려도 끝나지 않아 수동 종료. 이번에는 auto_end_calibrated=true여서 이전의 준비 실패와 다른 문제.
영상에서 손을 내린 뒤에도 화면 아래쪽에 남은 손목·손가락 추정점이 '손을 들고 있음'으로 분류됨. 공개 코드의 종료조건은 양쪽hand_high=false가750ms 유지돼야 하는데, 주요 동작 이후 필요한 조건의 최장 지속은 사과0.517초/첫감사0초/마지막감사0.195초뿐임. 시작대기좌표는저장되지않아 그 좌표와의 차이까지는 확정하지 않음.
첫감사도 실제로두번터치했다는사용자답변. 모델출력은두번째움직임56~66프레임을찾았지만 progress_endpoint_unconfirmed로 남아 한 번으로 계산함. 빠르기 자체가 원인인지, 마지막 자세/관측/연결 문제인지는 공개 자료로 분리해야 함. 무조건 사용자가 더 천천히 하도록 넘길 문제가 아님.

다음순서
1. 공개 정면의 앉은 조건으로 하단 추정점 때문에 자동 종료가 막히는 문제를 재현·공통 보완. 동작중가림·잠깐멈춤 조기종료 방지 유지.
2. 공개 정면에서 짧은2회도달/1회/미도달/자세유지의 정답시점으로 관측·마지막자세분류·구간연결 원인 분리. 필요한 부분만 수정·학습.
3. 두 보완 통합 후 정상·손모양오류·누락·자동종료·속도 검증, 통과분만 현재 화면에 적용.
4. 정상→오류→교정재시도 및 사과→감사퀘스트, 로컬실행/HDD분리 확인으로 시연본 고정.

이번에는 확인과 계획만. 사용자 촬영 학습·튜닝·후보선택 없음. C·T3-F/가중치/실행코드/서버/원본파일 변경 없음. 미사용 공개 입력을 통한 후속 검증 전에 단어별 또는 사용자별 임계값을 변경하지 않음.
'''.replace('最新','최신')
(D/'RESULT_KO.txt').write_text(body,encoding='utf-8')
next_action='새 촬영3건 진단의 계획을 사용자 확인 후 진행: R1 화면 하단 관절 때문에 손을 들었다고 판단하여 자동종료를 막는 문제를 공개 정면 앉은 조건으로 재현·보완, R2 두번째움직임의 끝 확인을 관측/자세분류/연결로 분리, R3 통합회귀 통과분만 웹연결, R4 시연마감. 이번 요청은 확인·계획만이며 아직 보완 실행 아님.'
statep=O/'STATE.json';state=read(statep);state['current_stage']=summary['status'];state['next_single_action']=next_action;work=state['capture_feedback_completion_20261116'];work['status']=summary['status'];work['updated_at']=at;work['next_single_action']=next_action
work['seated_retest_review_20261117']=dict(report=str(D/'RESULT.json'),text=str(D/'RESULT_KO.txt'),plan=str(D/'PLAN.json'),summary=summary['status'],user_confirmed_auto_end_failures=3,training_use=False,tuning_use=False,production_changed=False)
write(statep,state)
contextp=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=contextp.read_text(encoding='utf-8');h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
latest=f'''\n\n### 2026-11-17 16:49~16:50 실제 새 촬영3건 확인·다음 계획만
사과1/감사2:판정통과·반복끝확인부족·통과,TCN선택단어모두1위. 사용자세번모두자동종료안돼수동종료확인. 준비기준true이나화면하단추정손목/손가락으로hand_high가유지돼종료대기750ms를못채움(주요동작후필요조건최장0.517/0/0.195초). 실제시작대기좌표미저장으로모든거리조건은확정못함. 첫감사는사용자2터치확인;움직임56~66존재하지만끝미확인으로보류,속도원인미확정. 결과표시수동종료뒤0.882/0.794/0.536초,자동종료대기제외.527프레임처리/누락0/재검출0. 원본·현재95핀·C·T3-F·가중치·실행보존,학습/튜닝/후보선택0. 보고{D/'RESULT_KO.txt'},계획PLAN.json. 이번요청은확인·계획만,구현안함. 다음R1자동종료→R2두번째도달→R3통합검증→R4시연마감. 독립전문가검증아님.
'''
text=text.replace(h8,h8+latest,1).replace(h9,h9+f'\n\n### 2026-11-17 새 촬영 확인 후 계획 대기\n{next_action}\n계획 {D/"PLAN.json"}.\n',1);contextp.write_text(text,encoding='utf-8')
handoff=f'''# 최신 인수인계 — {at}
목표 정면웹캠 사과·감사3D시범→촬영→자동종료→빠른정상/교정/관측부족→재시도/퀘스트.5방향은아바타전용.
최신요청: 사용자“한두번더촬영,확인만하고이후계획”. 진단·계획만완료,프로그램수정/학습/새추론/서버재시작없음.
근거 {D/'RESULT_KO.txt'} / RESULT.json / DIAGNOSIS.json / PLAN.json.
새실제3촬영(16:49~16:50):사과통과,첫감사반복끝확인부족,마지막감사통과.TCN전부선택단어1위.527프레임처리/누락·오류0/재검출0. 수동종료→브라우저ACK0.882/0.794/0.536초(자동종료대기제외).
사용자답변:세번모두내리고기다려도안끝나수동종료,첫감사도손등두번터치.기록준비calibrated=true. 손내린영상과달리하단손목/손가락추정점이손높음으로남아최소750ms대기조건을못채움.주요동작후필요조건최장0.517/0/0.195초.실제previewrest좌표미저장으로그거리차는미확정. 첫감사두번째움직임후보56..66있으나끝미확인;속도때문이라단정금지.
다음계획 {next_action}
실행유지 START_WEB_DEMO.cmd→frontal_demo_launcher.py→{R/'serve.py'};8821기록PID5024,VERSIONselective_front_rtmw_retry_v3. 이번서버상태재조회/카메라시작안함. 재개시실제PID/상태확인;로컬호출일반환경필요.
직전공개보완유지:기존인위누락확인불가7→4,정상32/TCN유지,확장32누락24→7·제어52·소프트웨어10통과.공개캡처정상0.25초/감사복구0.645초.이번실사용자동종료실패는별도확인된사실이며공개시험통과로덮지말것.
모델/C·T3-F/100단어본체동결·학습0·총이력75유지,95핀동일.개인촬영은진단만·학습/튜닝/후보선택금지.새진단프레임은본폴더로컬에만,원본변경0.인위누락제어/공개정상/사용자진단을섞어정확도주장금지.
기존그래프 {R/'GRAPH.json'}의R보완분기로R1/R2,그뒤A/B/C/D/E시연마감연결.현재전체완료아님;확인만요청이므로계획승인없이보완구현재개하지않음.
'''
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf-8')
for p in [D/'RESULT.json',D/'PLAN.json',statep]:read(p)
assert all(sha(f)==h for f,h in pins.items()) and all(sha(f)==h for r in rows for f,h in r['source_hashes'].items())
write(D/'VERIFICATION.json',dict(passed=True,source_files_unchanged=True,active_pins_unchanged=95,training=False,tuning=False,candidate_selection=False,runtime_changed=False,camera_opened=False))
print('Recorded diagnosis and plan only;527frames,3manual stops,95pins unchanged')
