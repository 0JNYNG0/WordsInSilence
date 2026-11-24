from prepare import *
import datetime

def main():
 now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();V=U/'local_shape'
 dev=read(V/'CONSENSUS_DEVELOPMENT.json');fresh=read(V/'fresh/RESULTS.json');run=read(U/'BASIC_RUNTIME_VERIFIED.json')
 write(V/'CONSENSUS_ADMISSION.json',dict(at=now,status='LIMITED_OFFLINE_OBSERVATION_QUALIFIED_NOT_LIVE',
  developer_anchors=dev,held_source_shape_anchors=fresh['summary'],
  scope='Supplement coarse hand observation only. Existing per-finger known states preserved, no invented individual-finger truth. Unknown retained.',
  not_admitted=['Standalone error/whole-word completion','Automatic success producer','New default web weights'],
  remaining='Normal P17 right hand remains unknown throughout source action. Current learned relation observer also misses its separation before second approach.',
  artifacts=['shape_consensus.py','CONSENSUS_FROZEN.json','fresh/RESULTS.json'],training_private=False))
 # Refresh only text projection from stored observations; no repeated inference.
 from feedback_preview import repeated_relation
 from basic_runtime import lesson
 for item in run['rows']:
  p=Path(item['result']);r=read(p);r['relation_feedback']=repeated_relation(lesson()['repeat'],r['events'],len(r['relation_states']),sum(s>=0 for s in r['relation_states']));write(p,r)
 graph=dict(at=now,goal='기초 단어로 실제 촬영→교정/재시도→정상 퀘스트 통과가 되는 게임 시연',
  nodes=[dict(id='F0',task='기존 성공조건·근거누락 확인',status='DONE'),
   dict(id='F1a',task='정지/유지 손모양 연결',status='PRESERVED',evidence='VERIFIED.json'),
   dict(id='F1b',task='교정·불확실을 성공과 구분하는 계약',status='SOURCE_CONNECTED_NOT_SERVER_RESTARTED',evidence='QUEST_VERIFIED.json'),
   dict(id='F2a',task='기초단어 자료와 원문 연결',status='PARTIAL',note='감사·사과 확보; 안녕하세요 다인원본 경로 사용자 질문중'),
   dict(id='F2b',task='반복과 손모양 학습·공개 검증',status='PARTIAL',note='손모양 보조 관측 개선; 추가공개17의 분리 움직임·오른손은 불확실'),
   dict(id='F3',task='정상 완료·오류·재시도를 게임에 연결',status='PARTIAL',note='오류있는TCN일치 통과 차단; 실제 양성완료 생성기는 미완료'),
   dict(id='F4',task='새촬영 자동종료·전체대기·휴대성 확인',status='PENDING_USER_STARTED_CAPTURE'),
   dict(id='F5',task='작동하는 기초단어 시연 완성',status='PENDING')],
  loop='F2에서 관측불확실/잘못된구간/동작누락을 분리. 정상의 미검출을 사용자 오류로 단정하지 않음. F3는 정상통과와오류차단 모두 증명해야 완료.',
  next='반복 사이의 실제 벌어짐/가까워짐/유지를 독립 정답으로 붙여, 자세 상태를 확신하지 못해도 움직임 근거가 살아있는지 학습 비교. P17은 이후 개발진단용으로만 사용; 다음 최종확인에 재사용하지 않음.')
 write(U/'TASK_GRAPH.json',graph)
 report='''기초 단어 시연 준비 · 이번 작업 결과

기존 유지/정지 구간의 손모양 검사 보완과 C·T3-F, 100단어 공통 모델은 유지했다.
공개 자료만 사용했다. 사용자 촬영·카메라·서버 재시작은 하지 않았다.

1. 감사: 단어 이름을 맞히는 것에서 실제 두 손의 가까워짐을 시간별로 관찰하는 단계로 확장했다.
공개 9명 학습 / 3명 개발, 원본에서149시점 정답을 검토했다. 첫 주석12개는 유지 자세를 추가 반복으로 오독해 정정했고 초기 결과도 보존했다. 100단어 시간합성곱+GRU는 동결, 작은 판정부만 별도 학습했다.
14·15번 공개 수행자에서는35시점 중34개 일치/1개보류, 두번/한번접두부/정지/미검출/시간값변경 등16조건을 통과했다. 그러나 이후16·17번에서16번은두번,17번은두번째전벌어짐을확신못해한번만확인했다. 따라서 반복 판정 전체 완료로 보고하지 않는다. 속도시험은 원프레임을 유지한 시간값변경이며 실제빠른촬영 검증은 아니다.

2. 손모양: 한쪽 손을 바꾸면 반대손 점수까지 변하는 기존 전체문맥 판정부의 한계를 확인했다.
손 한쪽의 상대관절만 보는 시간합성곱+GRU를 비교하고, 공개 수행자4~9의 오른손24시점 정답을 추가해 미세조정했다. 단독 후보는 정상 손 오판이 남아 채택하지 않았다.
기존 손가락 판단을 보존하며, 보강 전후 판정부가 일치하고 기존 손가락 근거와 충돌하지 않을 때만 보조 관측을 추가하는 조합을 검증했다. 개발237손모양 정답: 기존208맞음/1오판/28보류 → 보완227맞음/1오판/9보류. 추가 공개16·17번의17손모양 정답:10맞음/0오판/7보류 →12맞음/0오판/5보류.
전체 정답률이나 모든 손가락의 정확도를 의미하지 않는다. 검지~새끼손가락이 모두펴짐/모두굽힘/알려진혼합 3상태를 비교한 것이다. 새17번의 오른손은 여전히 주요 구간에서 보류된다. 손가락별 기존 확정값을 바꾸지 않는 설계이며 새 손가락별 정답을 임의로 채우지 않는다.

3. 게임 통과: TCN 단어명만 맞으면 통과하던 코드를 보완했다.
공개 저장122조건 중 TCN은맞지만 교정근거가있는17조건을 통과시키지 않도록 했다. 계약검사16건도 통과했다. 단, 현재 새 단어의 실제 정상완료 생성기가 미완료이므로 기존 정상 결과도 완료근거가없으면 보류한다. 이것을 게임 완성이나 정상통과 성능개선으로 표현하지 않는다. 서버재시작도 아직하지 않았다.

4. 사과 / 안녕하세요
사과는 기존100+사과 후보만 남긴 저장점수 비교에서 기존300영상의1순위293개를유지,사과개발3개일치. 새로운학습·독립최종시험·웹채택은아니다. 안녕하세요SEN0354는참고1개외다인원본미확보로사용자에게저장경로를질문했다. NIKL인사로임의대체하지않는다.

현재 완성된 것은 기본 관측 부품과 잘못된 게임 통과를 막는 계약이다. 기초 단어로 정상통과→오류교정→재시도를 끝까지 시연하는 기능은 아직 완성되지 않았다.
다음은 반복 사이에 벌어지는 움직임을 자세 모양과 분리해 배우게 하고, 중요 손모양·방향·반복의 완료근거를 함께 묶는 일이다. 손날 방향/왼손등 관계 등 전체 설명의 요구사항을 생략한 채 성공처리하지 않는다. 이후 공개 정상·오류를 함께 검증해 웹에 연결한다.

구현과 별도 검산: Codex. 독립 수어 전문가/별도 에이전트 검증은 아님.
첫 주석 오류, 점수0만 설정한 미검출 진단의 이름 오류, 실행 전 문법오류2건은 각각 정정했다. 원본/이전 결과는 보존했다. 실제 미검출은 좌표를 사용할수없게한 별도검사에서모두보류했다.
상세: TASK_GRAPH.json, VERIFIED.json, source_corrected/CONFIRMATION_RESULTS.json,
local_shape/CONSENSUS_DEVELOPMENT.json, local_shape/fresh/RESULTS.json,
BASIC_RUNTIME_VERIFIED.json, QUEST_VERIFIED.json.
'''
 (U/'RESULT_KO.txt').write_text(report,encoding='utf8')
 next_action=graph['next']+' 손모양보조관측과 함께 설명의필수항목완료생성기를검증한후웹연결. 기존4단어는회귀전용. 사용자자료학습금지.'
 state=read(O/'STATE.json');state['current_stage']='basic_demo_feedback_20261115';state['last_updated']=now;state['next_single_action']=next_action
 state['basic_demo_feedback_20261115']=dict(status='PUBLIC_COMPONENTS_VERIFIED_GAME_COMPLETION_PENDING',path=str(U),graph=str(U/'TASK_GRAPH.json'),report=str(U/'RESULT_KO.txt'),existing_static_fix_preserved=True,C_T3F_preserved=True,private_training=False,
  relation_confirmation_14_15=read(U/'source_corrected/CONFIRMATION_RESULTS.json')['summary'],relation_later_16_17_counts=[len(r['events'])for r in run['rows']],
  hand_shape_development=dev,hand_shape_confirmation=fresh['summary'],new_web_weights=False,quest_contract_source_changed=True,positive_completion_producer=False,
  source_annotation_corrections=12,learning_runs=21,active_training_processes=[],server_restarted=False,webcam=False,
  pending_question='안녕하세요SEN0354 다인원본의 다른 보관경로',next_single_action=next_action)
 write(O/'STATE.json',state)
 ctx=R/'WordsInSilence_PROJECT_CONTEXT.md';text=ctx.read_text('utf8')
 h='### 2026-11-15 기초단어 반복·손모양 공개학습 비교 / 게임 정상완료 연결은 미완료'
 assert h not in text
 current=h+'''\n- 경로 `research/front_keyphase_v1/ops/basic_demo_feedback_20261115`. 이전 유지구간손모양보완/기존C·T3-F/100공통동결 유지. 사용자촬영 미사용.\n- 감사9명FIT/3명DEV 반복상태149시점(초기오독12정정). 추가14·15의35시점34일치1보류/16진단통과. 이후16번두번/17번한번관측:17번두번째전벌어짐보류로전체반복완료아님.\n- 손모양24공개FIT시점보강·추가학습. 기존개별손가락값보존+손한쪽판정부합의 보조: DEV237개208/1/28→227/1/9(맞음/오판/보류); 새16·17의17개10/0/7→12/0/5. 17번오른손주요구간보류. 단독후보미채택, 합의보조도아직웹가중치변경없음.\n- 퀘스트TCN단독통과방지 source코드연결. 공개122저장조건중교정있는17통과차단/계약16검사. 정상완료생성기가아직없어정상도완료근거없으면보류:게임완성/정상통과개선아님. 서버재시작/카메라없음.\n- 사과101후보는저장점수상기존300중293유지/사과3일치,웹미채택. 안녕하세요다인원본미확보·저장경로질문. 총21작은학습실행(초기정답정정전6포함),진행중학습없음. 원본/실패보존.\n- 검증·담당: Codex 구현/별도검산,독립전문가아님. RESULT_KO.txt/TASK_GRAPH.json/VERIFIED.json참고.\n\n'''
 marker='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n';assert marker in text;text=text.replace(marker,marker+'\n'+current,1)
 marker9='## 9. 다음 단일 작업\n';assert marker9 in text;text=text.replace(marker9,marker9+'\n'+h+'\n'+next_action+'\n완료기준:공개정상도통과하고오류·보류를구분하는실제완료생성기.새보조관측수치만으로게임완료처리금지.\n\n',1);ctx.write_text(text,encoding='utf8')
 handoff=f'''# 최신 목표 — 기초 단어로 작동하는 피드백 게임 ({now})

캡스톤은 실제 작동 시연이 목표. 기존4단어는회귀전용. 기초후보 감사/사과/AIHub안녕하세요. 숫자5제외. C·T3-F/100공통보존,사용자촬영학습·튜닝·후보선택금지.

## 마지막 작업
경로 `research/front_keyphase_v1/ops/basic_demo_feedback_20261115`의 RESULT_KO.txt / TASK_GRAPH.json / VERIFIED.json.
이전 static-hold engine수정 보존. 전체영상이GRU에들어가며 주요구간선택때문에앞뒤맥락을없애지않음.
감사 반복state 정답149개중12개초기오독수정(기록보존),9명FIT/3명DEV,100공통모델동결+작은판정부학습. 14·15번35정답34맞음1보류/16진단통과. 추가16번두반복확인,17번은중간벌어짐을확신못해한번만관측. 실제반복은두번으로보여누락오류로단정하지않음.
손한쪽공통모델+공개FIT24시점추가/미세조정. 단독모델은오판남아미채택. 기존개별손가락근거보존+보강전후합의 보조조합은DEV237개208맞음1오판28보류→227맞음1오판9보류. 새16·17번17정답10맞음7보류→12맞음5보류,새오판0. 17번오른손은전체주요구간보류. `shape_consensus.py` 오프라인보조관측용,웹미연결.
사과101후보:기존저장점수비교300중293유지/사과3일치. 재학습·새최종시험·운영채택아님. 안녕하세요SEN0354다인원본별도경로질문대기.

## 실제 소스 변경과 미완료
web_tutorial_live_integration_20261113/contract.py, feedback_acceptance.py(신규), site/live.js:
TCN일치만으로게임통과금지. 교정근거17저장조건차단,계약16검사통과.
**정상완료생성기가없어현재정상도보류한다. 게임완성아님.** positive통과는단위검사용fixture뿐. 새기초단어웹목록/실제capture연결미완료. 서버재시작/웹캠없음. 8821현재상태는미확인(CIM조회권한없음/로컬socket10013). 서버가꺼졌다고단정하지않음.
basic_runtime.py는이미추출한RTMW를한번입력해학습된부품으로손모양/반복/원본구간결과생성. 참고9영상매번비교아님. 전체문장자동이해나전체동작완료보장은아님. 후처리~0.07초는신규2공개영상한정/관절추출·촬영·DTW·화면제외.
공개16·17번2영상을로컬HDD에서약8MB확보/259프레임RTMW추출완료. 추가네트워크다운로드없음.

## 다음 한 작업
{next_action}
새17번은이미열었으므로이후개발진단용,새최종확인으로재사용금지. 현source·pred·truth의프레임과해시를이어사용. 초기주석미수정본은학습정답으로되돌리지않음.
공통손모양보조는알려진개별손가락값을바꾸지않고미확실을남긴다. 전체완료에는손날/손등관계·반복·필수모양근거를묶는검증필요. 단어별임시예외/억지성공금지.

## 실행 / 검증
Codex 구현+별도검산,독립에이전트/전문가아님. 총21학습실행종료(초기정답정정전6포함). 진행중학습없음.
파이썬: word100_signrep_feasibility_20261029/runtime/Scripts/python.exe -X utf8 -B
실패보존:초기주석12오독,score0만넣고missing이라이름붙인진단(진짜NaN미검출은별도전부보류),실행전문법오류2건수정. 새단독손모델을정상판정으로채택하지않음.
rootSTATE/current_stage=basic_demo_feedback_20261115. 상세각출력JSON/VERIFIED,사용자자료접근없음.
'''
 (R/'NEW_CHAT_HANDOFF.md').write_text(handoff,encoding='utf8')
 print('SYNCED',now)
if __name__=='__main__':main()
