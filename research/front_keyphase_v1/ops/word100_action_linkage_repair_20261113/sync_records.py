from pathlib import Path
from datetime import datetime
import json,hashlib,shutil
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(U/'FINAL_VERIFIED.json')['passed'];now=datetime.now().astimezone().isoformat()
gfile=O/'word100_web_completion_graph_20261111/TASK_GRAPH.json'
backup=U/'before_records';backup.mkdir(exist_ok=True)
for p in [gfile,O/'STATE.json',R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md']:
    assert not (backup/p.name).exists();shutil.copy2(p,backup/p.name)
summary='방향 관측에서 떨어진 행동 구간 사이를 한 번의 이동으로 합치던 문제를 고쳐8821에 연결. 공개12영상의93저장조건/정상60상태·기존교정 유지,606계약·별도산술·4공개실추론·웹1회·화면7검사 통과. 교정범위2사례를 실제 이어진 구간으로 축소. 전체 사용자 정확도 개선 주장은 아님. 손목+전완18학습은 새손실3원천손구간으로 미채택. 속도평가에 빠진 직접시점 대응도 출처검산 후 준비(36중30정확연결,6재표본추출탈락). G2 REWORK/봉인미개봉 유지.'
next_task='G2 설명 행동 오연결과 전체 경계 겹침을 구분한다. SPEED_TRUTH_TRANSPORT의 정확히 살아남은 원본 시점만 감사에 사용하며, 없는 시점/빈 정답을 행동 부재나 실패로 바꾸지 않는다. 직접 시작·진행·끝과 인접 행동 관계가 부족하면 공개 FIT에서 보강한다. 물리이동 유무 학습만 반복하거나 기존 전체구간 겹침 기준을 임의 완화하지 않는다. 새 근거가 있는 연결 보완만 학습·정상/속도/교차/방향/순서/부분동작 통합 비교, 통과 후 봉인평가와 웹 연결. 기존4단어 우선, 새3단어·연출 후순위.'
server=read(O/'web_tutorial_live_integration_20261113/SERVER.json')
record=dict(status='LOCAL_DIRECTION_WEB_VERIFIED_G2_REWORK',updated_at=now,summary=summary,next=next_task,
    report=str(U/'RESULT_KO.txt'),verification=str(U/'FINAL_VERIFIED.json'),graph=str(U/'GRAPH.json'),server=server,
    training_runs=18,training_running=False,arm_candidate_adopted=False,model_weights_changed=False,
    direction_refinement_web_applied=True,private_used=False,full_common100_retrained=False,
    sealed_evaluation_opened=False,whole_project_complete=False,verification_actor='same root Codex separate arithmetic/contracts, not independent KSL expert')
write(U/'CURRENT_EXECUTION.json',dict(status='CURRENT_UNIT_VERIFIED_NEXT_G2_PENDING',updated_at=now,training_running=False,
    no_completed_training_rerun=True,server=server,next=next_task))
graph=dict(updated_at=now,objective='新사용자 촬영을 목표 단어의 학습된 행동 근거와 연결해 짧은 교정 안내를 제공',
    original_graph=str(gfile),current_unit=[
      dict(id='A',status='PASS_DIAGNOSTIC',title='실패 원본과 관측 근거 확인',evidence='OBSERVATION_DIAGNOSTIC.json'),
      dict(id='B',status='COMPLETE_NOT_ADOPTED',title='손목+팔 흐름 고정 학습 비교',reason='새 손실이 있어 모델 유지',evidence='ARM_SAVED_VERIFIED.json'),
      dict(id='C',status='PASS_LIMITED_COMPONENT',title='연속 행동 구간별 방향·교정 범위 연결',evidence='DIRECTION_SAVED_VERIFIED.json'),
      dict(id='D',status='PASS_LIMITED_WEB',title='공개 재생·교정 순간 표시·부분 재생',evidence='SERVED_VERIFIED.json'),
      dict(id='C2',status='PASS_PROVENANCE_ONLY',title='빠르기 변형의 직접 정답 시점 연결',evidence='SPEED_TRUTH_TRANSPORT.json')],
    remaining=[dict(id='G2',status='REWORK',title='설명 행동 소속·경계·부분동작 구분'),
      dict(id='G3',status='PENDING_INTEGRATION',title='모양·방향·순서·누락 종합 검증'),
      dict(id='G4',status='SEALED_PENDING_GATES',title='새 공개 입력 최종 평가'),
      dict(id='G5_G6',status='PARTIAL',title='웹 부품 연결 완료 / 실제 촬영 교정 검증 남음')],
    edges=[['A','B'],['B','A','새 손실이면 원인 확인'],['A','C'],['C','D','공개 보존 통과 시'],['C2','G2'],['G2','G3','정상/오류 함께 통과 시'],['G3','G4'],['G4','G5_G6']],next=next_task)
graph['objective']=graph['objective'].replace('新사용자','새 사용자');write(U/'GRAPH.json',graph)
g=read(gfile);g['next_task']=next_task
g.setdefault('parallel_delivery_updates',[]).append(dict(at=now,unit=U.name,summary=summary,verification=str(U/'FINAL_VERIFIED.json'),does_not_complete_G2_G3_G4=True))
node=next(n for n in g['nodes'] if n['id']=='G2');assert node['status']=='REWORK'
node.setdefault('attempts',[]).append(dict(at=now,experiment_id='WRIST_FOREARM_INVARIANT_CONTEXT',status='REWORK',checks=dict(normal_preserved=False,three_seeds=True),
    note='Physical translation diagnostic only; new regression means not adopted. Local direction runtime fix is a separate verified component.',
    evidence=[dict(path=str(U/'ARM_SAVED_VERIFIED.json'),sha256=sha(U/'ARM_SAVED_VERIFIED.json'))]))
write(gfile,g)
s=read(O/'STATE.json');assert s['current_stage']=='word100_web_completion_graph_20261111'
s[U.name]=record;s['updated_at']=now;s['current_followup']=U.name;s['next_single_action']=next_task
w=s['web_tutorial_live_integration_20261113'];w['server']=server;w['direction_refinement_update']=str(U/'RESULT_KO.txt');w['next']=next_task;w['updated_at']=now
write(O/'STATE.json',s)
title='### 2026-11-14 연속 구간별 방향 교정 웹 반영 / 설명 연결 원인 감사'
p=R/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8-sig');assert title not in text
for heading,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`, `FINAL_VERIFIED.json`, `GRAPH.json`. 웹8821 PID'+str(server['pid'])+'. C·T3-F/공통100/개인자료 보호. 실제 카메라 미개방.'),('## 9. 다음 단일 작업',next_task+'\n이번18학습/직전18학습/이전6+18학습 재실행 금지. 그래프 전체 미완료, 현재 통과한 방향 관측 부품만 반영.')]:
    marker=heading+'\n\n';assert marker in text;text=text.replace(marker,marker+title+'\n\n'+body+'\n\n',1)
p.write_text(text,encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — 2026-11-14 기존4단어 행동 교정 우선

{summary}

현재 단위: {U}
결과/검증: RESULT_KO.txt, FINAL_VERIFIED.json, GRAPH.json. 구현과 별도 검산은 root Codex, 독립 수어 전문가 검증 아님.
본선 current_stage=word100_web_completion_graph_20261111 유지. G0/G1 PASS, G2 REWORK, G3/G4 미완료. 방향 관측 부품의 통과를 전체 피드백 완성으로 올리지 않는다.

웹 http://127.0.0.1:8821/ . PID {server['pid']}. server.py=web_tutorial_live_integration_20261113/server.py. 재개 전 실제 PID/포트 확인, 사용자가 촬영 중이면 재시작 금지. runtime/Scripts/python.exe는 word100_signrep_feasibility_20261029 아래. -B -u 절대경로 숨김 실행. 카메라는 사용자 직접 시작만.
서버는 현재 단위 runtime_adapter.CaptureService/attach를 import한다. 기존 capture/consumer의 코드와 모델은 보존하며 복사한 함수 전역에서 refine/consumer 생성자 의존성만 격리 연결했다. 기존 TCN/GRU/DTW/RTMW 관측재사용/자동종료는 동일. localized_direction.py의 연속 phase island별 원래 방향 계산, 빈 구간 미연결, 짧은 구간 unknown 유지, 실제 반대 방향 span만 교정 안내. GUI 소스는 이번에 변경하지 않았다.

검증:12공개 영상93조건(독립93영상 아님), 정상/속도/크기변형60상태 보존 및 새경고0. 기존 오류종류 유지. 실제 소비부4공개관절 재생+서버1공개 재생+화면7계약. 실제 사용자 정확도·웹캠 전체대기시간 미검증. 이전 DOC의 서버5828은 종료했고 현재 PID를 사용한다.
속도 보강 감사: 기존82조건×3초기값 비교에서 speed 직접 truth가 빈 것을 발견. 빈 truth를 행동 부재로 해석하지 않도록 BINDING_CAUSE_AUDIT_V2로 정정(첫파일 보존). 원본좌표/해시/PTS까지24속도조건 확인, 직접정답이 있던12카드의36시점 중30정확운반,6샘플탈락 보류. SPEED_TRUTH_TRANSPORT는 새점수/채택근거 아님. 겹침 실패33ms를 지우거나 허용폭을 만들지 않는다.

학습18완료: ARM_FROZEN/ARM_TRAINED/ARM_SAVED_VERIFIED. 공개 FIT24영상6기록제외×3seed. 팔 흐름으로 순종의 작은 이동1사례 회복했지만 기존20입력 대비3원천손구간 신규손실로 미채택. 웹 모델 교체 없음. 완료 학습/이전 비교 재실행 금지.
직전 actionable_feedback의18학습, recovery의6+18, 원래그래프의C6/C7/C9/L 등 완료 결과도 읽기전용. 신경망 물리이동 유무가 설명 소속·동작 완료 정답은 아니다.

다음 한 작업: {next_task}
현재 실행중 학습 없음. 8821 연구 서버만 이 단위에서 재시작했다. 새 단어 확대보다 기존4단어 핵심 보완을 먼저 한다.

지원4단어: 조상1567/이집트2523/순종0739/잘하다1133. 엄지·손끝 모으기·접촉·필수반복·전체완료 미검증. 이집트 일부 관측 부족 유지. C·T3-F·공통100·원본 읽기전용; 사용자/멘토 촬영 접근·학습·튜닝·선택 금지. 최종8봉인 미개봉, 새 다운로드/외부전송/Git 없음.
향후3단어: AIHub 안녕하세요 SEN0354, 과일사과0099, 감사1290. 숫자5 제외. NIKL 인사와 안녕하세요를 임의 병합하지 않는다. 아바타는 조상1단어 가지이며 피드백 선행조건 아님.
''',encoding='utf-8')
for p in [gfile,O/'STATE.json',U/'GRAPH.json',U/'CURRENT_EXECUTION.json']:read(p)
assert read(O/'STATE.json')['next_single_action']==next_task
assert U.name in (R/'NEW_CHAT_HANDOFF.md').read_text('utf-8')
write(U/'RECORDS_SYNCED.json',dict(passed=True,at=now,files={str(p):sha(p) for p in [gfile,O/'STATE.json',R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md']},stale_expansion_next_action_corrected=True))
print('Current records synchronized; G2 remains REWORK; verified direction component live.')
