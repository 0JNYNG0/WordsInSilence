from pathlib import Path
from datetime import datetime
import json,hashlib,shutil
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(U/'RESULTS_VERIFIED.json')['passed'] and read(U/'TARGETS_VERIFIED.json')['passed']
assert (U/'PROGRESS_GRAPH.png').exists();now=datetime.now().astimezone().isoformat()
state=read(O/'STATE.json');assert state['current_followup']=='word100_action_linkage_repair_20261113'
files=[O/'STATE.json',R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',G/'TASK_GRAPH.json']
backup=U/'before_records';backup.mkdir()
for p in files:shutil.copy2(p,backup/p.name)
summary='설명 행동·원본 시작/진행/끝 연결과 부분/다른 행동 학습을 공개자료로 검증. 기존32영상40원천행동 연결,24FIT/8노출probe 유지. 신규 부분/타행동 no-full bag 보강 후 작은 구간선택기3seed 미세조정 완료. 순종3공개영상 끝삭제 시험의 잘못된 구간 선택8/9→0/9이나 정상·속도10선택과 손모양/방향오류18선택이 빠져 미채택. 구간 탐색과 완료판정 통로 분리 부품은246저장조건의 기존구간/순서/근거 보존 검증. 웹 모델 변경없음, G2 REWORK.'
next_task='현재 웹의 행동구간·손모양·방향 교정 입력은 고정하고, 완료 판단을 별도 통로로 진행한다. SOURCE_TRACE_TARGETS/ABSENCE_BAGS는 원본 관찰 범위 정답이지 교육적 완료/오답이 아님. 공개 FIT에서 설명의 시작상태→변화→도달상태를 직접 관찰해 라벨로 연결하고 필요한 항목만 별도 학습한다. 부분·다른행동 구분 이득을 유지하면서 정상/반대손모양/방향오류를 비교에서 삭제하지 않는지 검증. 새로운 no-full 판정으로 구간 탐색을 교체하거나 짧은 동작을 오답 처리하지 않는다. 원래 G2/G3 통과 전 봉인8/전체완료 선언 금지. 기존4단어 우선, 새단어·아바타 후순위.'
graph=dict(updated_at=now,objective='사용자 행동의 어느 부분을 어떻게 고칠지 근거 있게 알려주는 웹 피드백',
    original_graph=str(G/'TASK_GRAPH.json'),status='G2_REWORK',image=str(U/'PROGRESS_GRAPH.png'),
    current_unit=[dict(id='A',status='PASS_DIAGNOSTIC',title='속도 경계 겹침과 행동 오연결 분리',evidence='LINKAGE_DIAGNOSTIC.json'),
        dict(id='B',status='PASS_TARGET_CONTRACT',title='설명·원천 행동·부분/전체 정답 연결',evidence='TARGETS_VERIFIED.json'),
        dict(id='C',status='COMPLETE_NOT_ADOPTED',title='부분/다른 행동 학습·82조건 비교',evidence='RESULTS_VERIFIED.json'),
        dict(id='D',status='PASS_RESEARCH_WIRING_ONLY',title='완료판정이 비교구간을 지우지 못하도록 분리',evidence='COMPLETION_CHANNEL_VERIFIED.json')],
    next_steps=[dict(id='N1',title='기존 구간·교정 입력 보존',acceptance='완료판정 점수와 무관하게 오류 동작도 비교 대상으로 남음'),
        dict(id='N2',title='설명에서 요구하는 도달상태 직접 주석',acceptance='공개 FIT와 실제 관측 연결; 최소시간/정확좌표를 임의 정답으로 만들지 않음'),
        dict(id='N3',title='도달상태·변화만 별도 학습',acceptance='공통100/기존 구간탐색 가중치 고정; 부분/관측부족 분리'),
        dict(id='N4',title='종합 회귀·새 평가·웹 연결',acceptance='정상 및 오류동작 비교 보존, G2/G3 후 봉인8, 통과 항목만 화면 연결')],
    loops=[dict(from_stage='N4',to_stage='N2',when='정답·허용범위 근거 부족'),dict(from_stage='N4',to_stage='N3',when='도달상태 관측·학습 오류'),dict(from_stage='N4',to_stage='N1',when='오류동작이 비교에서 빠짐')],
    whole_project_complete=False,next=next_task)
write(U/'GRAPH.json',graph)
record=dict(updated_at=now,status='TYPED_EXTENT_TRAINED_REJECTED_COMPLETION_CHANNEL_ISOLATED',summary=summary,next=next_task,
    report=str(U/'RESULT_KO.txt'),graph=str(U/'GRAPH.json'),graph_image=str(U/'PROGRESS_GRAPH.png'),verification=str(U/'RESULTS_VERIFIED.json'),
    source_videos=32,fit_videos=24,exposed_probe_videos=8,training_runs=3,training_running=False,model_adopted=False,web_changed=False,
    private_used=False,final8_opened=False,common100_changed=False,whole_project_complete=False,
    verifier='same root Codex separate source/arithmetic verification, not independent human KSL expert')
state[U.name]=record;state['current_followup']=U.name;state['updated_at']=now;state['next_single_action']=next_task
assert state['current_stage']=='word100_web_completion_graph_20261111';write(O/'STATE.json',state)
g=read(G/'TASK_GRAPH.json');assert next(n for n in g['nodes']if n['id']=='G2')['status']=='REWORK'
g['next_task']=next_task;g.setdefault('parallel_delivery_updates',[]).append(dict(at=now,unit=U.name,summary=summary,verification=str(U/'RESULTS_VERIFIED.json'),does_not_complete_G2_G3_G4=True))
next(n for n in g['nodes']if n['id']=='G2').setdefault('attempts',[]).append(dict(at=now,experiment_id='TYPED_PARTIAL_OTHER_ACTION_EXTENT',status='REWORK',
    checks=dict(partial_source_discrimination_improved=True,normal_preserved=False,error_actions_remain_comparable=False,three_seeds=True),
    note='Coupled localization candidate rejected. Separate research completion channel preserves existing locator fields exactly; no completion validity or web adoption.',
    evidence=[dict(path=str(U/'RESULTS_VERIFIED.json'),sha256=sha(U/'RESULTS_VERIFIED.json'))]))
write(G/'TASK_GRAPH.json',g)
title='### 2026-11-14 설명 행동·부분 수행 보강 학습 / 완료판정 통로 분리'
p=R/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8-sig');assert title not in text
for header,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`, `RESULTS_VERIFIED.json`, `PROGRESS_GRAPH.png`. 기존8821 웹은 이번에 변경/재시작하지 않았고 카메라 미개방. 기록08/09는 노출probe이며 사람외형으로 동인 판정하지 않는다.'),('## 9. 다음 단일 작업',next_task+'\n이번3seed1200step 완료학습 및 전체82조건 추론 재실행 금지. 직전18학습과 이전 완료 실험도 보존. 움직임 유무·원본 전체범위·수어적 완료를 혼동하지 않는다.')]:
    marker=header+'\n\n';assert marker in text;text=text.replace(marker,marker+title+'\n\n'+body+'\n\n',1)
p.write_text(text,encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — 2026-11-14 설명 행동·부분 수행

{summary}

현재 단위: {U}
RESULT_KO.txt / RESULTS_VERIFIED.json / GRAPH.json / PROGRESS_GRAPH.png. 전체 목표 미완료; G0/G1 PASS, G2 REWORK, G3/G4 대기. 이번 단위 완료와 전체 완료를 구분한다.

이번 실제 작업
- LINKAGE_DIAGNOSTIC: 기존 full_bag_rate_rank 속도33ms겹침1사례는 살아남은 직접 시점의 타행동 혼입은 없음. 경계 실패 그대로 보존.
- OMISSION_DIAGNOSTIC: 끝을 제거한 순종3영상×3seed에서 목표 원천시점은 삭제됐지만 앞행동을 대신 잡는 선택8/9. Egypt3영상에는 해당 직접정답 없어 점수/정답 생성하지 않음.
- SOURCE_TRACE_TARGETS: 기존32공개영상40원천행동,482출처 변형 계약. 실제사람오류 정확도 아님. FIT24(02~07), 노출probe8(08/09),40그룹의 교육적완료정답0 유지. 첫준비에서 희소5행동은 내부시점1개라는 사실을 처리하지못한 오류 FIRST_ERROR 보존 후 density 명시로 복구.
- ABSENCE_BAGS/TARGETS_VERIFIED: 기존 구간 내 common100 관측만 재사용. 1895파생bag(독립영상아님). 새 FIT부분1221·타행동35bag, probe부분341·타행동12bag. 타행동은 같은 영상 내 직접 구분된 이웃행동만; 다른 단어명으로 음성라벨 생성 금지. 타행동 근거는 순종2행동에만 있음, 범용코드와 다표현검증을 구분.
- TRAINING_COMPLETE: 이전 null head에서 시작한 새3seed1200step 완료. 공통100/TCN 고정. 정지/정상범위 replay+부분/타행동 no-full bag. family/type/원천영상 균형,기존<=56negative규칙. 모델가중치 웹 미교체. 같은 학습 재실행 금지.
- LOCAL_COMPARISON: 부분/타행동 구분 개선, 온전한 구간의 기존성공22파생bag손실(15선택위치변경,7미선택). FULLPATH82×3 한 번 추론: 순종끝삭제 선택8→0/9, 정상·속도선택10손실, 손모양·방향오류선택18소실로 미채택. 수어실력향상/사용자정확도 주장 금지.
- completion_channel.py: 연구용 분리 부품. 완료 후보가 낮아도 기존구간·순서·비교근거를 그대로 유지. 246저장조건 검사 통과. 이 채널은 display_enabled=False,semantic_completion=None. 웹 미연결, 완료판정 인증 아님.
- RESULTS_VERIFIED: 별도 산술/출처/보호35174검사, 사용자자료 없음, 봉인8미개봉,477보호파일 유지. 같은 Codex검산이지 독립 사람/전문가검증 아님.

다음: {next_task}

현재 학습·평가 프로세스 종료. CURRENT_EXECUTION은 단위검증완료/다음정답보강대기. 시간/속도계산은 연구후보 전체탐색으로 실제 웹캠 지연이 아님.
웹 http://127.0.0.1:8821/ 은 이전 단위가 연결한 runtime_adapter를 사용. 이번 서버 상태는 재시작하거나 카메라로 확인하지 않았음. 재개시 실제PID/포트를 확인하고 사용자 촬영 중이면 재시작 금지. web_tutorial_live_integration_20261113/server.py; word100_action_linkage_repair_20261113/runtime_adapter.py가 기존 consumer/CaptureService에 연속방향 observer만 격리연결. 기존 TCN/GRU/DTW/RTMW/자동종료 유지.
실제웹의 phase EvidenceHeads와 이번 연구 full_bag_rank/null 후보는 같은 모델이 아니다. 연구결과를 웹성능으로 말하지 않는다. 원래 G2 기준/단일checkpoint 불완전정의는 typed_motion_recovery_20261113/G2_REFERENCE_LINEAGE_REVIEW.json 참고. 다른순서측정방법의 점수 직접비교 금지.

지속 보호: C·T3-F·공통100·원본/과거동결 읽기전용. 사용자·멘토 촬영 접근/학습/튜닝/선택금지. 다운로드·외부전송·Git금지. 기존4단어 우선(조상1567,이집트2523,순종0739,잘하다1133). 엄지/접촉/손끝/최소반복·전체완료 미검증. 새3단어 안녕하세요SEN0354/과일사과0099/감사1290 및 아바타는 후순위. 숫자5 제외. 원래 source role의 fit 표기를 보고08/09를 새학습으로 옮기지 않는다.
''',encoding='utf-8')
write(U/'CURRENT_EXECUTION.json',dict(status='UNIT_VERIFIED_NEXT_ENDSTATE_SUPERVISION_PENDING',updated_at=now,training_running=False,evaluation_running=False,
    training_runs=3,model_adopted=False,web_changed=False,next=next_task))
for p in [O/'STATE.json',G/'TASK_GRAPH.json',U/'GRAPH.json',U/'CURRENT_EXECUTION.json']:read(p)
assert read(O/'STATE.json')['next_single_action']==next_task
write(U/'RECORDS_SYNCED.json',dict(passed=True,at=now,files={str(p):sha(p)for p in files}))
print('Records synchronized. G2 REWORK, no web/model replacement, new training finished.')
