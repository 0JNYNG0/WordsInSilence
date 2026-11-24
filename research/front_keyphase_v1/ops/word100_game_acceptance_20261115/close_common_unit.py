from pathlib import Path
import json,hashlib,ast
from datetime import datetime,timezone,timedelta
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
new='''    # A stationary role has no movement-window candidate. Reuse its independently
    # detected native span to observe shape, rather than silently dropping it.
    # Never merge gaps or search for the required shape inside a larger region.
    if span is None and spec['axis'] is None:
     matches=c.get('role_confirmation',{}).get('matching',[])
     unique=sorted({tuple(e['span']) for e in matches if e['family'] in spec['families']})
     if len(unique)==1:span=list(unique[0])
'''
engine=O/'word100_live_g6_20261114/engine.py';text=engine.read_text('utf-8-sig');assert text.count(new)==1
prior=text.replace(new,'').replace('\n','\r\n').encode('utf-8');frozen=read(U/'STATIC_LINK_FROZEN.json')['pins'][str(engine.resolve())]
assert hashlib.sha256(prior).hexdigest()==frozen
(U/'before/engine.py').write_bytes(prior)
write(U/'REPAIR_RECORD.json',dict(backup='before/engine.py reconstructed by reversing only admitted patch; byte hash equals pre-comparison frozen source',source_sha256_before=frozen,source_sha256_after=sha(engine),verification='Exact candidate source copied by minimal patch, paired26 plus real projection26; no model change.'))

report='''공통 연결 보완과 게임 완성 기준 정정 — 2026-11-15

사용자가 정한 우선순위
최종 캡스톤은 계획 발표가 아니라 실제 작동하는 한국수어 학습 피드백 게임을 시연하는 것.
촬영→자동종료→빠른분석→정상통과 또는 구간/이유/교정→재시도까지 완성해야 한다.
기존4단어(순종/잘하다/조상/이집트)는 공통 기능의 회귀검증 자료로만 남기며 실제게임시연에서 제외한다.
시연 단어는 기초·게임 상황에 맞춰 선정. 이전 후보 안녕하세요/사과/감사는 자료·검증가능성부터 확인.

이번에 확인하고 고친 것
전체 영상 관절과 시간모델 입력은 이미 존재했지만, 손모양 관측 연결부가 이동창의 span만 받았다.
모델이 별도로 찾은 정지/유지 역할구간은 span=None이 되어 손모양 검사에서 빠졌다.
이것은 '중요구간을 보자'는 사용자 목표의 문제가 아니라 관측결과를 전달하는 연결 누락이다.
GRU는 앞뒤 문맥을 배우지만 그 출력을 전달하는 연결까지 자동으로 고쳐주지는 않는다.
이동창이 없고 역할구간이 하나로 분명히 관측된 경우 그 원래 구간을 그대로 손모양 검사에 연결했다.
여러 조각을 임의로 합치지 않고, 정답모양을 찾아 구간을 다시 고르지 않는다. 단어명 예외 없음.

실행 결과
공개8영상에서 정상8/전체역재생8/손관측제거8/목표전중단2 = 26조건을 고정모델 전후로 비교.
정상2조건의 기존 '일부관측'이 '확인한특징일치'로 바뀌었다. 새 정상 오답경고0.
원래 찾은 구간/순서가 모두 동일하고 기존 방향·순서 차이는 유지.
중간에 끊은2조건은 앞 손모양만 더 확인했으며 나머지행동완료/전체정답으로 승격하지 않았다.
실제 웹 표시 변환기도26조건 검증했다. 검증된 코드만 현재 engine.py에 반영.
모델/문턱/GRU/TCN/DTW 재학습·변경 없음. 보호518파일 중 허용한 engine.py 1개만 변경, 나머지517동일.
이번 서버 재시작/물리웹캠은 하지 않았다. 실행 중인 서버 반영/실제시연 성공이라고 보고하지 않는다.

게임 통과의 핵심 미완료도 확인
현재 QuestProgress.consume는 TCN top1만 목표와 같으면 진행한다. 세부차이와 통과가 연결되지 않았다.
공개122조건을 현재TCN으로 다시 계산하면 목표일치97조건, 그중 교정근거가 있는17조건도 통과할 수 있다.
122는 새사용자122명이 아니라 이미 사용한 공개자료의 조건수다. 교정근거17은 모두 실제학습자오류라는 뜻 아님.
현재 내부관측이 비슷함과 '요구된 행동 끝까지 완료'는 별개다. 완수 근거 계약이 아직 연결되지 않았다.
lesson_contract.py는 긍정완수/차이/불확실을 구분하는 격리 설계부품이며 현재웹통과판정에는 연결하지 않았다.
다 막아서 통과0을 만드는 것을 성공으로 보지 않는다. 정상은 통과하고 오류는 교정되는 공통 완료조건이 필요하다.

시연 단어 자료 확인
감사 WORD1290:18정면기록 멤버정보/CRC 재대조, 기존100단어인식에 포함. 세부피드백은 새연결/검증 필요.
사과 WORD0099:18정면기록 재대조. 현재인식100밖이며 이전104후보는 미채택. ID05 표현변이 보류 유지.
안녕하세요 SEN0354:현재W 원천33ZIP에 해당SEN영상 없음. 기존C 공개참고1개 계보만 확인; 다인학습묶음 미확보.
NIKL 인사와 임의로 합치지 않는다. 숫자/새외부모델/대용량다운로드를 추가하지 않았다.

앞뒤 흐름 사용 원칙
촬영 전체 흐름→시간모델의 구간/변화 관찰→설명의 주요 행동과 연결→근거 있는 순간만 교정.
정지된 손모양 유지도 주요 행동이다. 앞뒤가 같다고 중간의 짧은 실제변화를 지워서는 안 된다.
새단어 정답에는 준비/주요유지/주요이동/전환/종료, 좌우손별모양/위치/방향/완료조건을 함께 기록한다.
한 설명의 조건들이 같은 행동구간에 함께 걸릴 수 있으므로 문구하나=구간하나로 강제하지 않는다.

다음 작업
기존4단어를 더 완성하는 대신 공통 코드/회귀자료로 유지.
시연용기초단어의 표현·자료범위·교정할핵심조건을 확정하고, 준비된감사부터 실제 정상/오류/완료의
시점정답→공통모델전이/작은판정부→공개평가→퀘스트성공/교정연결 순서로 진행한다.
사과 인식 추가와 안녕하세요 다인자료는 별도 준비상태를 확인한 뒤 진행하며 이름만 바꾸어 시연하지 않는다.
내부성공조건이 통과된 뒤 사용자가 시작한 카메라에서 자동종료/전체대기/재시도를 확인한다.
자료정리보다 기능완성이 우선이며 핵심성공·오류판정을 S-CURT로 넘기지 않는다.

구현·검산 Codex. 독립수어전문가검수 아님. 개인촬영미사용, C·T3-F/100공통가중치 보존.
근거 PLAN.json / PUBLIC_ACCEPTANCE_AUDIT.json / STATIC_LINK_RESULTS.json /
STATIC_CONNECTION_VERIFIED.json / PROJECTION_VERIFIED.json / DEMO_SCOPE_REVIEW.json.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')
write(U/'TASK_GRAPH.json',dict(at=at,goal='Actual working basic-sign learning game; old4words regression only',nodes=[
 dict(id='F0',task='기존 성공조건·근거누락 확인',status='DONE',evidence='PUBLIC_ACCEPTANCE_AUDIT.json'),
 dict(id='F1a',task='정지/유지구간 손모양 공통연결',status='DONE_SOURCE_CONNECTED',evidence='STATIC_CONNECTION_VERIFIED.json'),
 dict(id='F1b',task='정상통과/오류교정/불확실·부분완료 공통계약',status='DRAFT_NOT_CONNECTED',note='lesson_contract.py; absent completion fields are a contract gap, not measured recognition failure'),
 dict(id='F2',task='게임용기초단어 자료·정답·학습·공개평가',status='NEXT',note='감사18/사과18 원본존재;안녕하세요다인자료미확보;기존4단어시연제외'),
 dict(id='F3',task='통과한 정상/오류판정을 퀘스트·재시도에 연결',status='PENDING'),
 dict(id='F4',task='실제촬영 자동종료·대기·휴대성 검증',status='PENDING_USER_STARTED_CAPTURE'),
 dict(id='F5',task='작동하는 시연 결과를 발표자료로 정리',status='PENDING')],loops='F2에서정상실패/오류누락이면 정답·관측·구간·분류를 구분해 보완. F3에서모순이면계약수정. F4에서문제면실행경로보완. 사용자촬영으로학습/튜닝/후보선택하지않음.',constraints=['no forced success','no old4word demo mandate','no personal training','C/T3F preserved']))

state=read(O/'STATE.json');state['word100_game_acceptance_20261115']=dict(status='COMMON_STATIC_LINK_CONNECTED_BASIC_DEMO_SCOPE_NEXT',at=at,report=str(U/'RESULT_KO.txt'),graph=str(U/'TASK_GRAPH.json'),verification=str(U/'STATIC_CONNECTION_VERIFIED.json'),scope=str(U/'DEMO_SCOPE_REVIEW.json'),summary='게임 기능완성 우선으로사용자정정. 기존4단어는회귀자료/시연제외. TCN-only통과와세부판정미연결 확인(공개122중목표일치97·교정근거동반17). 역할로찾은유지구간 shape연결보완:26조건후보/순서동일,정상2특징확인추가·새오경고0·부분완료승격0. 현재engine에동일코드반영,서버/카메라미실행.517보호동일1허용수정. 감사/사과18기록재확인·안녕하세요다인자료미확보.',training_running=False,private_access=False,quest_acceptance_changed=False)
state['capstone_delivery_priority_20261115']['status']='SUPERSEDED_BY_USER_FUNCTIONAL_COMPLETION_PRIORITY'
state['capstone_delivery_priority_20261115']['superseded_by']=str(U/'PLAN.json')
state['next_single_action']='기존4단어는회귀검증전용. 게임시연 기초단어(이전후보안녕하세요/사과/감사)의자료·설명·필수행동/완료정답확정 후준비된감사부터학습·정상오류검증·퀘스트연결. 발표자료보다실제기능완성이우선.'
write(O/'STATE.json',state)
ctx=R/'WordsInSilence_PROJECT_CONTEXT.md';s=ctx.read_text('utf-8-sig');title='### 2026-11-15 사용자 목표 정정: 게임 기능 완성 우선 / 기존4단어 시연 제외'
assert title not in s
summary='\n\n'+title+'\n\n사용자: 캡스톤최종은 실제촬영→자동종료→빠른분석→정상퀘스트통과/정확한구간·이유·교정→재시도까지 작동하는 게임을 보여야한다. 자료정리우선 계획은철회. 기존4단어는공통회귀검증용이며실제시연에서제외. 기초·게임용단어선정/학습필요. 이전안녕하세요/사과/감사후보를점검하되안녕하세요다인자료미확보를임의로채우지않음.\n\n공통연결누락보완완료: 이동창없어도 모델이독립적으로찾은단일정지/유지역할span에손모양관측연결. 전체관절/GRU시간문맥삭제문제가아니라출력전달누락. 공개8영상26조건정상2추가확인·새오경고0·기존구간/순서유지·부분완료승격0,실제표시변환26검증. engine.py만적용(보호517동일),가중치/TCN/DTW불변. 서버재시작/물리카메라미실행. TCN-only퀘스트통과는여전히미수정이며통합성공계약은격리초안. 목표단어일치와행동정답을혼동금지.\n근거 `research/front_keyphase_v1/ops/word100_game_acceptance_20261115/RESULT_KO.txt`, `TASK_GRAPH.json`, `STATIC_CONNECTION_VERIFIED.json`, `DEMO_SCOPE_REVIEW.json`. 구현·별도검산Codex,개인촬영미사용.\n'
s=s.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'+summary,1)
nexttext='\n\n'+title+'\n\n다음단일작업은 게임시연기초단어 자료·설명·핵심행동·완료조건확정(F2). 기존네단어는공통기능회귀자료로만유지;이네단어완성이새단어착수의전제아님. 이전후보감사WORD1290현재100인식/18원본,사과WORD0099현100밖/18원본·ID05뜻변이,AIHub안녕하세요SEN0354공개참고1/다인미확보. 준비된감사부터시점정답→전이/작은판정부→별도공개정상·오류·부분·가림검증→통과한판정만퀘스트/교정연결. 안녕하세요를NIKL인사와임의로합치지않음. 핵심성공·오류판정을후속연구로미루지않음. 전체흐름을GRU문맥으로보되정지유지도주요행동이며,앞뒤같다고짧은실제변화를지우지않기. 개인촬영학습/튜닝/선택금지·C/T3F/공통100보존. `word100_game_acceptance_20261115/TASK_GRAPH.json`참조.\n'
s=s.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업'+nexttext,1);ctx.write_text(s,'utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('''# 최신 목표 — 게임 기능 완성 우선 / 기존4단어 시연 제외 (2026-11-15)

사용자 정정: 캡스톤은 작동하는 게임 결과물 필수. 촬영→자동종료→빠른판정→정상퀘스트통과 또는 구간/이유/교정→재시도. 자료정리우선은폐기. 핵심성공·오류판정을 S-CURT로미루지않음.
순종/잘하다/조상/이집트는공통회귀자료로만유지하고 실제시연에서제외. 기초·게임용단어를선정. 이전후보 안녕하세요/사과/감사 검토.

## 마지막 공통 보완 완료
`research/front_keyphase_v1/ops/word100_game_acceptance_20261115` 참고.
유지구간 모델역할을 찾고도 이동창없어 손모양을검사안하던연결보완. 전체영상관절/GRU문맥삭제가아님. 단일역할구간만있는경우그span재사용,임의합침/정답맞춤구간재탐색없음.
공개8영상26조건전후·실제표시변환검증:정상2손모양확인추가,새오경고0,구간/순서보존,중단2는앞손모양만확인/전체완료아님.
현재 `word100_live_g6_20261114/engine.py`에동일코드반영. 보호518중이파일1개허용수정·517동일. C·T3-F/100공통/TCN/DTW/GRU가중치그대로.
서버재시작/카메라미실행. 프로세스학습없음,26재생종료. 실행중서버반영/실제시연완료로보고하지않음.

## 중요 미완료
현재퀘스트통과는아직 TCN top1일치만사용. 공개122조건의실제TCN재계산:목표일치97중17은세부교정근거도있음. 전체동작완수근거와게임성공연결이필요.
lesson_contract.py는격리설계초안/웹미연결. 없던완료필드때문에모든결과보류가되는것을성능검증으로주장금지. 정상통과와오류교정을모두입증해야한다.
직전손가락모양·변화분리9회학습후보는여전히미채택. 당시518보호보고는그시점기록,이번공통engine수정과혼동금지.

## 다음 작업
TASK_GRAPH.json F2: 시연기초단어 자료·문장·주요행동/유지/전환/끝·완료조건 정답확정→공통부재사용학습→공개정상/오류/부분/가림검증→실제퀘스트판정연결.
DEMO_SCOPE_REVIEW.json:감사WORD1290현재100포함/18원본재확인,사과WORD0099현100밖/18원본재확인·ID05뜻변이보류. 안녕하세요SEN0354는W원천33ZIP에없음/기존C참고1개계보,다인미확보. NIKL인사와합치지않음. 새기초단어학습/게임단어목록변경은아직안함.
사용자촬영학습/튜닝/후보선택금지. 실제촬영은사용자시작으로만. 자료제출<3일·발표12월1일경잠정. 작업우선기능완성,후속세부고도화는S-CURT.

근거: RESULT_KO.txt / PUBLIC_ACCEPTANCE_AUDIT.json / STATIC_LINK_RESULTS.json / STATIC_CONNECTION_VERIFIED.json / PROJECTION_VERIFIED.json / DEMO_SCOPE_REVIEW.json.
구현·별도검산Codex,독립수어전문가검수아님.
''','utf-8')
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8-sig'))
read(O/'STATE.json');assert read(U/'PROJECTION_VERIFIED.json')['passed']
write(U/'FINAL_STATUS.json',dict(at=at,status='COMMON_LINK_COMPLETE_BASIC_DEMO_PREPARATION_NEXT',model_training=False,user_footage_used=False,server_restarted=False,live_game_success_complete=False,documents_synchronized=True,old4words_demo_excluded=True))
print('common unit and revised functional-demo scope recorded')
