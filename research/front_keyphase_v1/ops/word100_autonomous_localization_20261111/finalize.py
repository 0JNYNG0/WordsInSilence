from pathlib import Path
import json,hashlib,collections
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2];R=U/'boundary_integrity_repair';G=U/'guide_warning_gate'
rd=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wr(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
assert rd(U/'VERIFICATION.json')['integrity_passed'] and rd(G/'VERIFICATION.json')['passed']
cfg=rd(U/'FROZEN.json');res=rd(R/'RESULT.json');raw=rd(U/'RESULT.json');gate=rd(G/'VERIFIED_RESULT.json');runs=[rd(U/'runs'/str(s)/'COMPLETE.json') for s in [42,43,44]]
summary='승인4단계 실행·3회 학습·검산 완료. 공개FIT30영상49이동28유지로 자동 시점 판정부를 보강. 직전 움직임부+구간기준 대비 기존26항목22→25(난수44는24),기존9항목4→5~6. 새공개14의6영상10항목 첫비교8→9,구간겹침계산수정후10/10(재사용개발검증). 모두 후보/방향 부품지표,전체피드백정확도아님. 최종78조건의교정은직전대비추가이득0.부분구간경고회귀2조건발견후안전연결로정상12/속도24/교차·기존오류9유지.잘하다역순비슷함1,누락6보류잔존.웹/C·T3-F/100단어공통불변·개인자료0·HTML생략.'
nxt='새 검출기를 더 붙이기보다, 공개FIT에서 준비/주요행동/복귀 구간의 역할과 후보-설명 대응 정답을 보강하고 기존quality/pair 판정부를 학습. 관측후보를 유지한 채 주요행동을 고르는 정확도와 최종교정을 함께 비교. 상세 NEXT_PLAN_KO.txt. 이번 완료 결과 재학습·재추론 금지.'
report='''자동 동작 탐색과 설명 연결 — 이번 작업 결과

핵심
약속한 1~4단계의 진단·정답 보강·학습·비교를 실행했다.
짧은 동작의 구간을 찾는 부품은 좋아졌다. 하지만 사용자에게 내보내는 최종 교정은 직전보다 더 좋아졌다고 말할 근거가 아직 없다. 새 부품을 웹 기본에 연결하지 않았다.

1. 먼저 무엇이 문제인지 확인했다
학습용 공개 24영상의 39개 이동을 점검했다. 기존 선택 결과와 맞는 것은33개, 후보가 있었으나 선택되지 않은 것은2개, 구간이 너무 넓어 제대로 대응하지 못한 것은4개였다.
놓친6개는 활동 신호가 거의 없어서라기보다, 여러 짧은 왕복을 하나의 긴 구간으로 묶는 문제가 컸다. 이 학습자료의 원인 진단을 다른 모든 실패의 원인으로 일반화하지 않는다.

2. 무엇을 가르쳤나
기존100단어899영상에서 배운 공통부는 고정했다. 세부학습은 공개30영상,49개 이동,28개 관계유지 구간을 사용했다. 이 중6영상·10이동·6유지 정답을 원본 RGB로 추가 확인했다.
어느 방향으로 움직이는지, 유지하는지, 준비/마무리 구간인지, 어디서 시작하고 끝나는지를 나눠 가르쳤다. 모르는 구간을 유지 정답으로 채우지 않았다. 손목 위치유지와 손가락 모양유지는 다르다.
기존 짧은 움직임 Conv+GRU 가중치를 가져와 시간별 탐색 출력을 붙이고 미세조정했다. 학습 난수42/43/44의3회다. 독립적인 사전학습 모델3개나 서로 다른 사람3명을 뜻하지 않는다.
100단어 전체를 이번에 처음부터 다시 학습한 것이 아니다. 세부시간 주석은10표현·기록01/03/05/07에 한정된다. 새 이동 주석의 주체는오른손이다. 왼손은 양손관계의 기준으로 포함되지만, 왼손 단독 동작의 자동분할을 폭넓게 검증한 것은 아니다.

3. 어떻게 비교했나
움직임보완만 / 구간탐색·연결보완만 / 둘의결합 / 기존기준을 구분했다. 사용자·멘토 촬영은 전혀 사용하지 않았다.
기존 공개16영상26이동: 직전 움직임보완 기준22/26 → 이번 주난수42의25/26. 다른 두 학습은25/26,24/26.
기존 공개6영상9이동:4/9 → 주난수42의5/9. 다른 두 학습은6/9,5/9.
추가 공개 기록14의6영상10이동: 첫 동결 비교에서8/10 →9/10(3학습 동일). 이후 구간겹침 계산을 고친 저장값 재검증에서는10/10.
위 숫자는 ‘자동 후보 중 정답구간과 일대일로 대응된 구간을 찾고 방향까지 맞혔는가’다. 구간 겹침0.5 기준으로 평가한다. 전체 영상의 모든 손가락·순서·누락·최종문장을 맞힌 비율이 아니다. 구간 후보가 많으므로 모든 후보의 정확도도 아니다. 아직 정답 주석이 없는 구간을 가짜동작으로 세지 않았다.
기록14도 과거 공통TCN 평가노출 이력이 있을 수 있다. 이번 세부평가 전 정답을 고정하고 FIT와 원본SHA를 분리했지만, 완전히 새로운 독립 사람 검증이라고 주장하지 않는다. 원본의 기록ID로만 구분했다.

4. 최종피드백은 무엇이 달라졌나
최종 피드백의 추가 개선은 확인하지 못했다.
공개4표현의 기존78시험조건에서 정상12개,속도변형24개(교차 포함),기존 손모양·방향 오류9개를 함께 비교했다.
중간 후보에서는 정상 빠른 동작을 잘못 경고하는2조건이 드러났다. 준비 또는 마무리의 짧은 후보가 설명의 주요행동에 일부 겹친 것만으로 교정을 내보낸 경우였다.
그 후보 자체는 지우지 않고 기록에 남겼다. 시작·끝 근거가 부족한 후보만으로 새로운 주요행동 교정을 내보내지 않도록 안전 연결을 추가했다. 기존에 확인된 경고는 보존했다. 이 보완은 결과를 본 뒤 추가한 안전수정이며 새 독립평가 성공으로 세지 않는다.
안전 연결 후3학습 모두 정상12·속도24에 추가경고0,기존 오류9개 경고유지. 전부 관측부족12개·마지막관측부족6개·마지막동작삭제6개를 새로 정상으로 올린 경우0.
주요동작 역순9시험은 직전과 동일하게 경고5,잘못된비슷함1,확인불가3이다. 잘하다 역순의 잘못된비슷함은 남았다. 마지막동작삭제6개도 구체적인 누락교정 대신 확인불가에 머문다.
이들은 공개자료 관절변형/시간변형 시험을 포함한다. 자연스러운 새 사용자 오답을 모두 정확히 잡는다는 증거가 아니다.

완료 판단에서 지킨 것
행동의 시작·끝 신체관계 범위를 FIT에서 만들고 새 후보와 연결했다. 그러나 구간이 끝났다는 것만으로 수어를 끝까지 했다고 판정하지 않았다. 손모양·필수행동·순서까지 확인할 근거가 더 필요하다. 불확실한 부분은 오답으로 확정하지 않았다.

속도
추가 탐색+설명연결 계산은 주난수42 중앙값 약0.03초였다. 관절추출·공통특징·TCN·DTW·웹화면 시간을 제외한 부품 시간이다. 사용자 전체 대기시간이0.03초가 됐다는 뜻이 아니다. 이번에 웹캠 대기시간을 다시 측정하지 않았다.

검증과 수정 이력
첫 실행에서 별도 어댑터 가중치를 전체모델로 잘못 읽어 중단했다. 예측 전이었고 로딩 순서를 수정했다.
첫 비교 뒤 경계 미세조정이 이웃 방향 구간을 침범하는 문제를 발견했다. 가중치·정답·문턱은 바꾸지 않고 경계를 자기 방향 구간 안으로 제한했다. 첫 결과는 그대로 보존했다. 신경망 재추론 없이 저장 확률을 사용했다.
별도 검산에서 실제 정상경고 회귀2조건을 확인했고 숨기지 않았다. 안전 연결 후 검산에서 선택사항 카드의 경고를 요약에서 누락한 집계문제도 수정했다. 실제 카드 경고는 사라진 적이 없으며 잘못된 첫 집계와 정정 집계를 모두 보존했다.
구현·주석·별도 검산은 같은Codex가 수행했다. 독립적인 한국수어 사용자/전문가의 정답 검증은 아니다.

현재 큰 목표와의 거리
이미 있는 것: 관절 입력,단어 인식,시간정렬,손모양·방향 판단부,피드백 시각화.
이번에 나아진 것: 짧은 행동의 자동 시점 후보와 방향을 찾는 부품.
아직 핵심인 것: 그 후보가 설명의 어느 주요행동인지 안정적으로 선택하고,끝까지 했는지/빠뜨렸는지 확인하여 정확한 교정을 내보내는 연결.
완성률을 임의의 백분율로 제시하지 않는다. 현재는 작은 공개자료 연구부품 개선이며 완성된 교육 피드백 AI가 아니다.

유지/미실행
C·T3-F,기존100단어 공통부,웹기본,기존 손모양 판정 보존. 사용자·멘토자료 접근/학습/튜닝/선택0. 새 다운로드/카메라/외부전송/Git0. 원본과과거동결결과보존. 최종교정의 추가이득이 없어HTML생략.
'''
(U/'RESULT_KO.txt').write_text(report,'utf-8')
plan='''다음 단일 목표: 발견한 움직임을 올바른 주요행동에 연결하는 판정부

근거
시점 후보/방향 점수는 올랐지만 최종78조건의 교정은 그대로다. 자동 후보가 부족한 경우와,준비/복귀 후보를 주요행동으로 잘못 연결하는 경우가 남았다. 새 모델을 계속 추가하거나100단어 전체 학습량만 늘릴 근거보다,후보와설명 연결 정답이 우선이다.

1. 기존 공개FIT에서 역할 정답 보강
4표현은 우선 검증 범위로 유지하되 단어별 예외 대신 공통 역할을 사용한다. 주행동/진입/복귀/관계유지/관측불가를 각 실제 후보에 붙인다. 오른손만의 감독 한계를 해소하기 위해 왼손과양손의 수행 주체도 구분한다.
기존FIT01/03/05/07과 중복되지 않는 FIT02/04/06/08/09의 동일4표현20영상부터,실제 주요동작과 준비·복귀의 상대관계가 무엇이 다른지 원본RGB로 확인한다. 의미상필수끝은 확인된가이드근거가 있을 때만 붙인다. 해당방향과손모양이틀린시도도 후보에서제외하지않는다.
코드실행전검사:출처SHA/행동주체/같은동작반복과다음동작구분/모호경계범위/관측부족별도. 같은Codex주석과전문가검토를혼동하지않는다.

2. 기존 후보연결 신경망에 가르치기
새무거운모델 대신 기존quality/pair와역할 판정부를 사용한다. 입력은후보의앞·중간·뒤 신체관계와관측신뢰도,주변후보의순서다.100단어공통부와손모양·기존방향부는고정한다.
출력:이후보가주요행동인가/진입복귀인가/어떤설명역할인가/같은행동의조각인가/반복또는다음행동인가/모르는가. 기대방향은후보선택에사용하지않고선택후차이판단에만사용한다.
부분후보만으로교정하지않는안전연결은유지하되,보류만늘리는개선은통과시키지않는다. 전부범용함수로만구현한다.

3. 누락/순서/피드백 연결
선택된행동은실제촬영시간순서그대로정렬한다. DTW로역순·누락을정답순서로억지로고치지않는다. 설명의앞행동을뒤행동에중복배정하지않는다.
‘팔을쓸어내리는표시구간에서손을펴보세요’처럼구간+부위+한교정을낸다. 동작이관측됐지만끝관계가충족되지않은경우와,가려져끝을모르는경우를구분한다. 없는관측으로누락을확정하지않는다.

4. 마지막 성공조건은 최종교정
지금기록11~14는개발회귀로만사용. 다음평가는미사용세부주석기록15등의공개자료를SHA/노출감사후점수전에예약한다. 기록ID만으로독립사람주장금지.
기존정상12·속도24·교차에서추가경고0,손모양방향9오류유지,주요역순9중기존5보다더많은유효교정,잘못된비슷함1감소,누락6에서가림과실제삭제구분개선이필요하다. 부품구간점수만올라가면완료가아니다.
학습3난수값,학습영상과평가영상분리. 기존최선/연결만/결합과실제문구·강조구간까지비교한다. 점수확인후문턱·정답수정금지;코드오류수정은첫결과보존과별도회귀로기록한다.

이번단위는학습·비교·안전수정·검산까지종료했다. 이문서는다음설계이며위20영상의새주석·후속학습을이미실행했다는뜻이아니다. 웹/C·T3-F 보존,개인자료학습·튜닝·선택금지.
'''
(U/'NEXT_PLAN_KO.txt').write_text(plan,'utf-8')
wr(U/'FINAL_DECISION.json',dict(status='COMPLETE_LOCALIZATION_COMPONENT_GAIN_NO_ADDITIONAL_FEEDBACK_GAIN',steps_completed=[1,2,3,4],goal_complete=False,production_adoption=False,html_created=False,primary_seed=42,initial_reserved14_before=8,initial_reserved14_after=9,initial_denominator=10,post_score_boundary_repair_after=10,post_score_repair_not_fresh=True,regressions_before_warning_gate=rd(U/'VERIFICATION.json')['scientific_checks'],final_safe_feedback_comparison=gate,training_seconds=sum(r['seconds'] for r in runs),limitations=['Sparse fine labels: 30 public videos,10expressions,49motion intervals28holds','Fine supervision moving hand R only; L occurs as relation anchor, not broad left-action proof','Candidate recall and paired direction are not whole feedback accuracy','All labels/verification same Codex, no independent human expert ground truth','Remaining reverse false-similar1 and omit6 uncertain','Unknown precision outside reviewed spans','No real-camera end-to-end latency rerun'],report=str(U/'RESULT_KO.txt'),next=str(U/'NEXT_PLAN_KO.txt')))
for p,h in cfg['protections'].items():assert sha(p)==h,p
wr(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_LOCALIZATION_COMPONENT_GAIN_NO_ADDITIONAL_FEEDBACK_GAIN',pid=None,remaining_processes=[],next=str(U/'NEXT_PLAN_KO.txt')))
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8')
for sec,body in [('## 8. 현재 상태',summary),('## 9. 다음 단일 작업',nxt)]:
 i=t.index('\n',t.index(sec))+1;t=t[:i]+'\n### 2026-11-11 자동 동작 탐색·설명 연결 4단계 완료\n\n'+body+'\n근거 '+str(U/'RESULT_KO.txt')+'\n'+t[i:]
p.write_text(t,'utf-8');s=rd(O/'STATE.json');s.update(current_stage=U.name,next_single_action=nxt);s[U.name]=dict(status='COMPLETE_LOCALIZATION_COMPONENT_GAIN_NO_ADDITIONAL_FEEDBACK_GAIN',summary=summary,next=nxt,report=str(U/'RESULT_KO.txt'),plan=str(U/'NEXT_PLAN_KO.txt'),execution=str(U/'CURRENT_EXECUTION.json'),verification=str(U/'VERIFICATION.json'),final_safe_verification=str(G/'VERIFICATION.json'),existing_models_and_web_unchanged=True,private_data_used=False);wr(O/'STATE.json',s)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n현재 단위 '+str(U)+'\n\n다음 단일 작업: '+nxt+'\n\n공통100단어899영상 기반은 고정. 세부시간 지도는30영상49이동28유지(주체R). 이 숫자는 전체영상분류/최종피드백 정확도가 아니다. 새공개14도 첫점수후수정검산은개발회귀이며독립사람/전문가검증아님.\n\n현재 실행PID없음.3회학습/158조건비교/경계수정/안전연결/검산종료;완료모델재학습/재추론금지. 최초RESULT와경계수정RESULT,안전연결VERIFIED_RESULT를구분한다. 안전연결의첫RESULT는선택카드경고집계오류가있어VERIFIED_RESULT가최종이다.\n\n보존: C·T3-F/기존웹/공통100단어/손모양. 사용자·멘토자료 접근·학습·튜닝·선택,다운로드,외부전송,Git,카메라,운영기본교체금지. 원본/W:/sources/과거동결읽기전용.\n\n보고 '+str(U/'RESULT_KO.txt')+'\n후속설계 '+str(U/'NEXT_PLAN_KO.txt')+'\n','utf-8')
h=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
with h.open('a',encoding='utf-8') as f:f.write('\n\n[2026-11-11 자동 시점 탐색과 설명 연결 4단계 실행]\n'+summary+'\n진행 이유: 정확한구간에서는판정하지만자동구간에서실패하는차이를진단하기위함.학습39이동중6누락은주로짧은왕복의과도한합침으로확인.움직임Conv+GRU를시간별상태/경계로미세조정하고후보→설명연결을따로비교했다.\n계산오류/실패도보존:어댑터로딩실패,이웃구간경계침범,정상속도변형2조건경고,안전연결요약의선택카드경고누락.안전연결후실제최종교정은직전과동일함을확인.\n목표변경없음.다음우선은새검출기/전체학습량확대가아닌후보의진입/주행동/복귀역할과설명연결지도.별도검산은같은Codex이며전문가검증아님.\n근거 '+str(U/'RESULT_KO.txt')+'\n')
docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',h]
wr(U/'DOCUMENT_CHECK.json',dict(passed=all(U.name in p.read_text('utf-8') for p in docs),files={str(p):sha(p) for p in docs},result_exists=(U/'RESULT_KO.txt').exists(),plan_exists=(U/'NEXT_PLAN_KO.txt').exists(),no_running_pid=rd(U/'CURRENT_EXECUTION.json')['pid'] is None))
print(summary)
