from pathlib import Path
from datetime import datetime
import json,hashlib,shutil
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2];G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
v=read(U/'RESULTS_VERIFIED.json');assert v['passed']and (U/'RESULT_KO.txt').exists()and (U/'PROGRESS_GRAPH.png').exists()
now=datetime.now().astimezone().isoformat();state=read(O/'STATE.json');assert state['current_followup']=='word100_action_completion_binding_20261114'
files=[O/'STATE.json',G/'TASK_GRAPH.json',R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md']
before=U/'before_records';before.mkdir()
for p in files:shutil.copy2(p,before/p.name)
summary='기존 행동구간·비교 입력을 고정하고 도달 상태 별도 지도학습·공개 비교를 완료. 공개32영상/24FIT·8노출probe,117프레임의186항목 중164직접상태정답·22보류,관절유효FIT116항목. 선형/작은신경망 각3seed 총6학습. 손목 쪽 위치4/4·양손거리6/6 직접probe와 각20연속편집조건 통과. 비스듬한 전완 주석/관측 불일치3항목과 교차높이1보류로 전체완료판정 미승격. 기존246저장조건 보존,477보호파일·8582별도검산. 웹/C·T3-F/공통100 변경없음, G2 REWORK.'
next_task='통과한 상대팔 위치·양손거리 상태 부품을 고정하고 기존 자동 행동구간에 연결해 다른 행동의 도달상태를 가져오지 않는지 먼저 검증한다. 전완은 가슴쪽 손 위치/손 닫힘과 팔 세우기를 분리해 설명의 실제 요구 관계로 공개FIT 정답을 보강한다. 비스듬한 팔을 가로/세로 각도만으로 완료 라벨화하지 않는다. 다른 표현의 같은 신체관계·실제 중간 진행 중 정지·짧은 정상·교차·관측부족을 확인하고 G2/G3 후 봉인평가/웹 승격. 기존 동작·오류 비교 보존, 손끝/접촉/반복은 미검증 유지. 이번6학습·383조건 완료결과 재실행/노출probe튜닝 금지.'
record=dict(updated_at=now,status='SEPARATE_GOAL_STATES_TRAINED_TWO_COMPONENTS_SUPPORTED_NOT_WEB_PROMOTED',summary=summary,next=next_task,
    report=str(U/'RESULT_KO.txt'),graph=str(U/'GRAPH.json'),graph_image=str(U/'PROGRESS_GRAPH.png'),verification=str(U/'RESULTS_VERIFIED.json'),
    source_videos=32,fit_videos=24,exposed_probe_videos=8,training_runs=6,training_running=False,
    component_research_pass=v['component_research_pass'],current_unit_complete=True,whole_project_complete=False,web_changed=False,model_adopted=False,
    private_used=False,final8_opened=False,common100_changed=False,TCN_changed=False,
    verifier='same root Codex separate arithmetic/provenance/contracts, not independent KSL expert')
state[U.name]=record;state['updated_at']=now;state['current_followup']=U.name;state['next_single_action']=next_task;write(O/'STATE.json',state)
g=read(G/'TASK_GRAPH.json');g['next_task']=next_task;assert next(n for n in g['nodes']if n['id']=='G2')['status']=='REWORK'
next(n for n in g['nodes']if n['id']=='G2').setdefault('attempts',[]).append(dict(at=now,experiment_id='SEPARATE_OBSERVABLE_GOAL_STATE_LEARNING',status='REWORK',
    checks=dict(localization_preserved=True,arm_region_component=True,hand_separation_component=True,forearm_state_definition=False,crossing_height_complete=False,
      semantic_completion=False,web_changed=False),note='2 coarse state domains supported by small exposed public comparison; automatic-span and educational completion remain unvalidated.',
    evidence=[dict(path=str(U/'RESULTS_VERIFIED.json'),sha256=sha(U/'RESULTS_VERIFIED.json'))]))
g.setdefault('parallel_delivery_updates',[]).append(dict(at=now,unit=U.name,summary=summary,does_not_complete_G2_G3_G4=True));write(G/'TASK_GRAPH.json',g)
title='### 2026-11-14 도달 상태 별도 학습·검증 / 두 관측 항목 제한 통과'
p=R/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8-sig');assert title not in text
for header,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary+'\n근거: `research/front_keyphase_v1/ops/'+U.name+'/RESULT_KO.txt`, `RESULTS_VERIFIED.json`, `PROGRESS_GRAPH.png`. 실제 사용자 오류/수어 전체완료/모든단어 일반화 정확도 아님. 공개영상 편집시험이며 비교정답 사후수정·재학습 없음.'),('## 9. 다음 단일 작업',next_task)]:
    marker=header+'\n\n';assert marker in text;text=text.replace(marker,marker+title+'\n\n'+body+'\n\n',1)
p.write_text(text,encoding='utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 최신 인수인계 — 2026-11-14 도달 상태 별도 학습

{summary}

현재 단위: {U}
현재 단위1~4 구현/학습/고정비교/검산 완료. 전체 목표는 미완료. G0/G1 PASS, G2 REWORK, G3 이후 PENDING. 원래 그래프는 research/front_keyphase_v1/ops/word100_web_completion_graph_20261111/TASK_GRAPH.json.
RESULT_KO.txt / GRAPH.json / PROGRESS_GRAPH.png / RESULTS_VERIFIED.json. 실제 공개영상 예시가 있는 그림은 직접 확인 완료.

실제 작업·근거
- STATE_LABELS.json: 실제RGB 검토32영상,117고유프레임/120검토칸,손·항목별186관찰.164상태정답/22보류. FIT02~07의24영상, 노출probe08/09의8영상. 08/09를 이전READY role=fit로 보고 학습시키지 않음. 기록ID로만 분리, 사람외형 동일인 판정 없음.
- LEARNING_INPUTS: 어깨 기준 정규화 좌표와 관측마스크만 입력. 단어ID·절대시간·프레임번호·미래문맥 없음. FIT116유효항목 학습. unknown 주석은 임의오답으로 사용하지 않음.
- TRAINING_FROZEN/TRAINING_COMPLETE: 선형45가중치,작은MLP473가중치 각3seed,800step1회 총6학습 완료. 기존100단어신경망/C·T3-F를 수정한 것이 아님. 기존32공개영상의새도달상태부품. 결과본후모델/경계튜닝없음, 같은학습재실행금지.
- EVALUATION: 비교용 손목쪽4/4,양손거리6/6.전완22/27일치(3불일치,2관측부족),높이5/6(1보류).선형전완20/27(1불일치6보류),다른항목동일. 신경망일반우위 주장금지. REAL08비스듬한전완3불일치는주석범주가겹칠수있음; 확정모델오류/사용자오류로말하지말것. 원본라벨/점수보존,후보에맞춰수정하지않음.
- DENSE_REPLAY: 실제연속좌표383편집조건(probe98).손목위치/양손거리 각2probe영상×10조건 통과.짧은도달prefix/시간재표본/중간2누락/끝마스킹/시작·끝고정/역순. 정지포즈합성은 실제중간에멈춘학습자오류정답아님. 부분70%허용경계/최소반복/접촉을가정하지않음. 조상REAL05 세검토점에높이양상둘다없어해당전이채점보류; 행동부재아님.
- GUIDE_STATE_BINDINGS: close_chest의전완가로는가슴위치+손닫힘의충분한대체물이아님. raise_forearm은비스듬한자세라벨보완. brush_arm/spread_hands도상대위치관측만이고손가락/궤도/전체정오아님.문장전체를이해했다고주장금지.
- goal_contract.py: 관측부족/분류불확실 이유분리,출발상태만/끝포즈만/순서관측을짧은한국어안내로분리. semantic_completion=None, educational_error=None, display_enabled=False. 웹미연결.
- 기존246조건(정상·속도108,손모양/방향오류27포함)의구간·비교입력정확보존.240좌표불변/좌우공통/미래문맥미사용검사. RESULTS_VERIFIED 별도8582산술·출처·보호PASS. 같은Codex검산,독립전문가검증아님.
- 추가상태부품만8공개영상0.001~0.0022초. RTMW/TCN/GRU/카메라/UI제외이므로전체사용자대기시간아님.
- pretraining_review에첫동결본/평가코드보존: 학습전 가시항목분모와검증기정답독립작성수정. 최종동결후6학습1회,실패재학습없음. CIM프로세스조회는권한거절; 서버상태확인/재시작안함. 이번학습프로세스모두정상종료.

다음 한 작업
{next_task}

유지사항
웹주소8821은기존서버계보이며이번실행여부확인/재시작/카메라개방안함. 기존server.py→word100_action_linkage_repair_20261113/runtime_adapter.py→consumer. 실제웹과연구full_bag/null후보·이번statehead는다른부품. 원래자동구간경계겹침/설명오연결G2미해결,새봉인8개평가금지.
직전 action_completion_binding 3seed학습은부분부재선택8/9→0/9이나정상10/오류18선택소실로미채택.이번연구channel은그실패선택기를채택한것아님. 이전18/6학습등완료결과재실행금지.
공통계산이지만이번손목위치/거리직접정답은각한표현에서왔으므로모든단어일반화완료주장금지. 다른표현검증필요.
C·T3-F/공통100/원본/과거동결읽기전용. 사용자·멘토영상접근/학습/튜닝/선택금지. 다운로드·외부전송·Git·카메라금지. 기존4단어우선,새3단어/아바타후순위. 모델·운영기본교체없음.
''',encoding='utf-8')
write(U/'CURRENT_EXECUTION.json',dict(status='UNIT_COMPLETE_RESEARCH_COMPONENTS_ONLY',updated_at=now,training_running=False,evaluation_running=False,
    training_runs=6,web_changed=False,whole_project_complete=False,next=next_task))
for p in [O/'STATE.json',G/'TASK_GRAPH.json',U/'CURRENT_EXECUTION.json']:read(p)
assert read(O/'STATE.json')['next_single_action']==next_task
write(U/'RECORDS_SYNCED.json',dict(passed=True,updated_at=now,files={str(p):sha(p)for p in files},report_sha256=sha(U/'RESULT_KO.txt'),graph_image_sha256=sha(U/'PROGRESS_GRAPH.png')))
print('Records synchronized; G2 remains REWORK; no web/model replacement.')
