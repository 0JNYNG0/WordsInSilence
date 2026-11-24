from pathlib import Path
import json,hashlib,shutil,datetime

N=Path(__file__).resolve().parent;O=N.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert not (N/'RESULT.json').exists()
v=read(N/'VERIFIED.json');assert v['status']=='PASS'
for p,h in v['source_pins'].items():assert sha(p)==h
c=read(N/'COMPARISON.json');primary=c['models']['final_42']
result=dict(status='COMPLETED_PARTIAL_LOCALIZATION_GAIN_NOT_DEPLOYED',
    goal='Generic movement-candidate localization independent of finger shape; four expressions are stabilization scope, not per-word exceptions.',
    implemented=['same-view public input and supervision provenance audit','finger/palm-independent geometry Conv+BiGRU proposal branch','deployed feedback-core and geometry interval union, conflict and unassigned evidence retention','six frozen training runs,237fixed public conditions per final seed, saved verification'],
    not_completed=['new dense RGB-verified movement boundaries','reliable ordered association and completion across all cases','expert-validated learner correction','end-to-end web adoption or latency measurement'],
    data=dict(public_originals=48,fit_originals=36,historical_development_originals=12,fit_record_ids=list(range(1,10)),dev_record_ids=[11,12,13],input_conditions=564,actual_crop_fit_conditions=12,
      comparison_conditions=dict(main=144,legacy=93),independent_people_count=12,
      condition_count_is_not_people_count=True,same_originals_repeated_across_conditions=True),
    training=dict(folds=3,final_seeds=[42,43,44],primary=42,epochs_per_run=100,new_random_geometry_weights=True,existing_shape_and_direction_weights_fixed=True),
    comparison=primary['by_kind'],additional_neural_latency=primary['proposal_seconds'],
    verification=dict(saved_checks=v['checks'],input_checks=read(N/'INPUT_CHECK.json')['checks'],scope='Same Codex separate arithmetic/code verification; not independent expert labels'),
    acceptance=v['gates'],baseline_scope=read(N/'BASELINE_SCOPE.json'),
    changes=dict(web_default=False,current_TCN=False,C_T3F=False,private_data_access=False,new_detector=False,new_download=False),
    html_created=False,reason_no_html='Partial diagnostic gains with normal regression and residual false completion; no full acceptance.')
# record IDs denote dataset IDs, not visual proof of distinct people
result['data']['dataset_record_ids_count']=result['data'].pop('independent_people_count')
write(N/'RESULT.json',result)
nextplan=dict(status='NEXT_COMMON_ASSOCIATION_UNIT_DEFINED_NOT_STARTED',
    objective='Retain successful shape-independent candidate proposals while separating attempted stage, observed completion, missing observation and out-of-order movement.',
    evidence=['Normal REAL13 candidate69..78 assigned to final stage overlaps prior-stage stored anchor73; conflict with true final candidate creates extra uncertainty.',
              'Omitted/masked REAL12 final stage receives72..74, but stored anchor74 belongs to preceding stage. A tiny boundary match is still treated as completion.',
              'Geometry only stage branch does not read finger/palm points; both simple and full-hand controls give exactly identical geometry probabilities.',
              'Existing sparse/weak targets do not establish all continuous starts, ends, transitions or repeated cycles.'],
    steps=[
      dict(step=1,task='Use existing public FIT01..09 RGB and guide descriptions to verify action cores, transitions, meaningful changes and completion evidence. Keep anchor/weak/uncertain sources distinct. Counterexamples include same wrist path with wrong shape, beginning-only action, normal complete action, reversed direction and sensor masking.',pass_condition='The important action evidence is visible and consistent with original frame identity; uncertain joints do not become successful targets. No copying DEV failures as FIT truth.'),
      dict(step=2,task='Add generic temporal event association to the retained candidate bank. Link a candidate using preceding/following movement and progression, permit deletion/insertion/repetition, and retain incompatible movement as unassigned evidence. Candidate existence is separate from completion.',pass_condition='An earlier stage cannot count twice as a later completed stage; a reversed or omitted action is not normalized away by forced DTW.'),
      dict(step=3,task='Train only the association/completion component if labels support it, with the two existing observation routes fixed first. Freeze all conditions within FIT person splits before historical DEV regression. Keep primary42 and report other seeds instead of selecting by DEV.',pass_condition='Normal full/crop, speed, crossing and short changes preserved; missing-event and masked-observation cases not falsely all-similar. Include single and combined faults.'),
      dict(step=4,task='Only after acceptance, connect exact event evidence to short feedback and visual highlighting in an isolated research web candidate. Measure capture-end-to-display latency separately from model-forward time.',pass_condition='No unsupported claim of linguistic correctness or exact 3D contact. Preserve existing web/model default until verified.')],
    constraints=['No word-ID special-case fixes','C/T3F/currentTCN/web preserved','No private/mentor training tuning selection','No new model download or detector needed','No repeated frozen inference or DEV threshold tuning'],
    unresolved='Sequence/completion supervision and feature sufficiency remain research questions; this plan does not guarantee all vocabulary success.',
    user_input_needed_now=False,reason='User confirmed generic four-expression stabilization; no new scope or resources are needed for the defined next local step.')
write(N/'NEXT_PLAN.json',nextplan)
report='''공통 동작 구간 찾기 보완 결과 — 2026-11-09

결론
4단어는 먼저 안정성을 확인하는 시험 범위로 두었다. 코드는 단어별 성공 예외 없이 공통 기능으로 만들었다. 다만 범용 코드라는 말이 모든 단어를 이미 학습했다는 뜻은 아니다.
모양이 달라져도 움직인 구간을 비교에 남기는 데 일부 개선이 있었다. 정상 영상의 불필요한 보류와 동작 누락을 놓치는 사례가 남아 이번 후보를 웹 기본으로 바꾸지 않았다.

무엇을 바꿨나
기존: 손모양을 포함한 특징으로 동작 구간을 찾는다. 손모양이 달라지면 비교할 구간 자체를 못 찾을 수 있었다.
추가: 손가락과 손바닥 중심을 보지 않는 작은 시간 합성곱+GRU가 몸·팔·손목의 상대 위치와 움직임을 보고 구간 후보를 찾는다. 기존 손모양 판정부는 그대로 둔다.
결합: 두 경로의 후보와 어느 단계에도 배정하지 못한 움직임을 남긴다. 서로 다른 근거가 있으면 일단 미확인으로 구분한다. 이 후보의 존재만으로 동작을 잘했다고 확정하지 않는다.
현재 새 모델은 몸 기준 위치·관측 여부·시간 변화에서 10개 동작 구분과 배경을 배우는 구조다. 자연어 설명 전체를 읽는 언어모델이나 모든 손모양을 배운 모델은 아니다.

실제 진행 범위
1. 자료 확인 완료: 공개 원본48개, 학습36개(기록ID01~09), 기존 개발12개(ID11~13). 원본 전체, 실제 화면을 자른 뒤 추출한 관절, 좌표만 가린 시험을 구분했다. 실제 잘린 학습 입력12개를 추가했다.
2. 학습 완료: 새 움직임 분기만 사람 분리3회+고정 초기값3회, 총6회·각100회 학습. 기존 손모양/방향 모델과 TCN은 재학습하지 않았다.
3. 고정 비교 완료: 144조건과 과거93조건을 각각 비교했다. 같은 원본을 속도·가림·모양·누락 등으로 바꾼 조건이며 237명의 새 영상이 아니다. 기존 개발 자료의 회귀 확인이다.
4. 적용 판정 완료: 웹 적용 기준 미통과. 새 HTML과 웹 교체는 하지 않았다. 새로 모든 프레임의 행동 경계를 사람이 확인하는 주석 보강은 아직 완료되지 않았다.

이전 → 현재 (대표 모델42, 같은 저장 관절 입력의 판정부 비교)
- 화면을 자른 정상12개: 필수 동작의 비교 구간이 없던 영상2개 → 0개. 모든 필수 항목이 비슷함으로 나온 영상6개 → 7개, 미확인이 포함된 영상6개 → 5개.
- 원본 정상12개: 차이 경고0개 → 0개. 그러나 미확인이 포함된 영상3개 → 4개로 늘었다. 이 때문에 정상 보존 기준은 실패했다.
- 오른손 모양을 닫도록 바꾼12조건: 비교 구간을 되찾아 새 모양 차이 경고1개가 나왔다. 12개 모두가 언어적으로 틀린 행동이라는 뜻은 아니므로 이것을 오류 검출률로 부르지 않는다. 다른 초기값43/44에서는 각각2개다.
- 과거의 명확한 전체 손모양변경3조건, 역방향3조건, 둘을 함께 바꾼3조건: 각각 차이 경고3개 → 3개 유지.
- 마지막 중요 동작을 삭제한6조건: 전부 비슷함으로 잘못 넘어간 결과2개 → 1개. 아직 1개가 남았다. 누락 후보 표시는4개 → 4개였다.
- 마지막 중요 구간의 관절만 가린6조건: 전부 비슷함으로 잘못 넘어간 결과2개 → 1개. 관측 부족을 정상으로 넘기는 문제가 아직 남았다. 누락으로 잘못 표시한 결과는0개 유지.
- 양손 관절을 모두 가린12조건: 12개 모두 미확인 유지.
- 교차를 포함하는 조상 정상3개: 기존 항목별 결과 유지. 모든 종류의 교차에 대한 보장을 의미하지 않는다.
- 정상 속도 차이 조건에도 원본 정상과 같은 추가 미확인이 나타났다. 속도와 정상 보존을 모두 통과한 상태는 아니다.

‘구간을 찾음’, ‘비슷함’, ‘정답’은 다르다
구간을 찾았다는 것은 비교할 시간 범위를 출력했다는 뜻이다. 경계를 정확히 찾았거나 수어 전체가 정답이라는 뜻은 아니다. 위 차이/미확인 개수는 한 영상에 여러 항목이 있으면 서로 겹칠 수 있다.
비교는 기존 웹의 가중치와 판정 코드를 저장 관절 입력에 적용한 결과다. 촬영·RTMW 추출·TCN·DTW 재생·카메라 미관측 보호 후처리·화면까지 전부 실행한 웹 실사용 결과는 아니다(BASELINE_SCOPE.json).

남은 원인을 어디까지 확인했나
정상 순종 REAL13에서69~78프레임을 마지막 동작 후보로도 잡았다. 기존 주석의73프레임은 첫 동작이다. 그 후보의 손모양과 진짜 마지막 동작의 손모양이 다르므로 충돌 보류가 생겼다. 즉 단순히 후보를 더 찾는 것만으로는 충분하지 않았다.
마지막 동작을 삭제/가린 순종 REAL12에서도72~74프레임만으로 마지막 동작을 했다고 출력했다. 기존 주석의74프레임은 앞 동작에 속한다. 앞 동작의 끝을 뒤 동작 완수로 오인한 문제다.
이는 저장된 주석과 출력의 대조 근거다. 이번에 새로 농인 전문가가 정답을 확정한 것이 아니다. 특정 단어 예외를 넣지 않고 앞뒤 동작 연결과 실제 수행 진행을 함께 확인해야 한다.

검증과 속도
입력1381항목, 저장 결과·학습 분리·계산·해시12906항목 검산 통과. 이것은 구현/계산 검증 통과이며 모델의 적용 기준 통과가 아니다. 같은 Codex가 별도 검산한 결과이며 독립 전문가 검증은 아니다.
손가락 모양만 바꾸고 몸·손목을 유지한 시험에서는 새 움직임 모델의 출력이 정확히 같았다. 이 구조가 손모양에 끌려 변하지 않는다는 점은 확인했다.
새 움직임 분기만 사람을 나누어 검사했을 때 기존 확인시점89개 중78개, 실제 잘린 영상의32개 중25개를 맞췄다. 확인시점은 일부 영상에만 있고 연속 경계 전체의 정답률은 아니다. 전체 피드백 시스템의 독립 신규 사용자 정확도로 해석할 수 없다.
추가 움직임 신경망 계산 중앙값 약3.87ms(0.00387초), 상위95%지점 약5.99ms. 이 값은 영상 수신·관절 추출·기존 판정·DTW·화면 표시 시간을 포함하지 않는다. 전체 사용자 응답이 이 시간이라는 뜻은 아니다.

바로 이어갈 핵심
새 검출기를 또 추가하는 것보다, 이번에 되찾은 후보를 올바른 설명 단계와 순서에 연결하고 ‘조금 시작함’과 ‘끝까지 수행함’을 구분하는 보완이 우선이다.
① 공개 학습 영상에서 주요 동작 중심/전환/완수 근거를 보강한다. 모양이 틀려도 동작의 시도 구간은 남긴다.
② 기존 두 관측 경로는 유지하고, 앞뒤 연결을 보며 후보가 어느 동작인지 연결한다. 누락·추가·반복을 허용한 구조로 만들고 정답 순서에 억지로 끼워 넣지 않는다.
③ 정상·속도·교차·모양변화·방향·누락·관측부족을 함께 비교한다. 개발 결과를 보고 문턱이나 좋은 초기값을 골라 맞추지 않는다.
④ 통과한 결과만 해당 손/팔과 시간 구간을 강조하고 짧게 고치는 방법을 보여주는 웹 후보로 연결한다.

보존
기존 C·T3-F, 현재 RTMW TCN, 웹 기본 유지. 사용자/멘토 촬영 접근·학습·튜닝·선택 없음. 새 다운로드·카메라·외부 전송·Git 변경 없음.
처음 평가 모듈을 불러올 때 이름이 겹치는 오류는 첫 추론 전에 확인해 수정했고 최초 기록을 보존했다. 완료 학습이나 이전 고정 결과는 덮어쓰거나 재학습하지 않았다.
이번 단위의 학습/고정 비교/별도 검산은 종료했다. 전체 피드백 완성을 선언하지 않는다. 다음 구체 계획은 NEXT_PLAN.json에 기록했다.
'''
(N/'RESULT_KO.txt').write_text(report,encoding='utf-8')
summary='공통 팔·손목 동작후보 분기6고정학습/144+93조건 비교/12906저장검산 완료. 손가락·손바닥중심 제외로 모양변경시출력불변 확인. 대표42 실제crop정상12의필수구간없는영상2→0,전체필수비슷함6→7. 원본정상미확인3→4회귀,사건삭제/관절가림 각각6중거짓전체비슷함2→1로1잔존. 과거손모양/역방향/복합각3차이3유지. 신규분기만사람분리89확인시점78,연속경계/전체신규사용자성능아님. 원인은앞동작후보를뒤동작으로연결·짧은경계로완수오인. 웹미적용·HTML생략. 새분기3.87ms는전체응답아님. 기존C/T3F/RTMWTCN/웹보존·개인자료0. 기존가중치의저장관절판정부비교이며실제웹전체응답미측정. 주석은기존sparse/weak재사용,새연속RGB경계주석완료아님. 근거 '+N.name+'/RESULT_KO.txt.'
nexttext='공통 동작후보의 앞뒤 사건 연결·완수 판단이 다음 단일작업. 공개FIT01~09 RGB/설명으로 주요중심·전환·실제진행 정답보강→두관측분기고정한 후보연결/시도대완수 분리→누락/추가/반복을허용한순서검사→정상/속도/교차/모양/방향/관측부족회귀통과시연구웹연결. 특정단어/DEV예외금지. 완료6학습/237조건/별도검산재실행금지,새조건은FIT내동결. 세부 '+N.name+'/NEXT_PLAN.json. 전체정확도/사용자실시간개선미주장,C/T3F/현재TCN/웹유지,개인자료학습튜닝선택금지.'
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
    b=N/'before'/p.name;b.parent.mkdir(exist_ok=True)
    if not b.exists():shutil.copy2(p,b)
s=read(O/'STATE.json');s['current_stage']=N.name;s['next_single_action']=nexttext;s['updated_date_kst']='2026-11-09'
s[N.name].update(status=result['status'],summary=summary,report=str(N/'RESULT_KO.txt'),result=str(N/'RESULT.json'),verification=str(N/'VERIFIED.json'),next=nexttext,next_plan=str(N/'NEXT_PLAN.json'),running=False)
write(O/'STATE.json',s)
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf-8');title='### 2026-11-09 공통 움직임 후보 학습 — 구간 회복 일부 개선, 순서·완수 문제 잔존\n\n'
for marker,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nexttext)]:text=text.replace(marker+'\n',marker+'\n\n'+title+body+'\n',1)
p.write_text(text,encoding='utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n다음 한 작업: '+nexttext+'\n\n현재 학습/추론/검산은 모두 종료. 다음 연결/완수 단위는 정의만 했고 아직 실행하지 않았다. 질문 답변은4표현 안정화 우선·공통 기능으로 진행 승인. 이미 승인된 로컬 보완에 반복 승인 불필요. 사용자/멘토자료 금지. 과거STATE/결과를 최신 상태로 덮지 말 것.\n\n구현·실행·별도 산술 검증은 같은Codex, 외부전문가/독립사람정답 검증 아님. 보호대상과 원본·동결산출물 유지. 전체영상의 새연속경계주석·자연오류정확도·전체응답속도·웹적용은 미완료. 결과 '+str(N/'RESULT_KO.txt')+'\n',encoding='utf-8')
write(N/'CURRENT_EXECUTION.json',dict(status=result['status'],training=None,evaluation=None,verification=None,next_unit_started=False,report=str(N/'RESULT_KO.txt'),next_plan=str(N/'NEXT_PLAN.json')))
write(N/'DELIVERY_PINS.json',{str(p):sha(p) for p in [N/'RESULT.json',N/'RESULT_KO.txt',N/'NEXT_PLAN.json',N/'VERIFIED.json',N/'BASELINE_SCOPE.json',N/'BOUNDARY_DIAGNOSIS.json',ROOT/'NEW_CHAT_HANDOFF.md',ROOT/'WordsInSilence_PROJECT_CONTEXT.md',O/'STATE.json']})
print('COMPLETED: partial improvements documented, default unchanged; no active training/evaluation.')
