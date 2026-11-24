from pathlib import Path
import json,hashlib,datetime,collections
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def rd(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def wr(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=rd(U/'VERIFICATION.json');assert v['passed']
tr=rd(U/'TRAIN_MANIFEST.json');labs=rd(U/'REQUIREMENT_LABELS.json')['rows'];train=rd(U/'TRAINING_COMPLETE.json')['rows'];otr=rd(U/'ORDER_BINDING_TRAINING.json')
first=rd(U/'EVALUATION_COMPLETE.json');full=rd(U/'CORRECTED_FULLPATH_DIAGNOSTICS.json');cap=rd(U/'GUIDE_CAPABILITY.json')['rows']
result=dict(status='COMPLETE_ORDER_COMPONENT_GAIN_NO_WEB_ADOPTION',label_sources=dict(public_fit_videos=len({r['clip_id'] for r in tr}),expressions=len({r['word_id'] for r in tr}),reviewed_items=dict(collections.Counter(r['task'] for r in labs)),total_items=len(labs),derived_train_queries=len(tr),direct_clause_shape_direction_links=sum(bool(r.get('guide_clause_ids')) for r in labs if r['task']!='order'),source_provenance='Inherited Codex RGB observations, not new expert truth; fourFITpanels spot-audited'),training=dict(first_residual_runs=3,corrective_order_runs=3,epochs=40,first_parameters=train[0]['parameters'],order_parameters=otr[0]['parameters'],common100_899_frozen=True),first_candidate='Not adopted: supplied direction32/32 ->31/32,32/32,31/32 and binary nonmatch confused with opposite direction; original scores preserved',order=v['scientific'],automatic='Reversed-core9 clause pairs:5 false supports,4association-unknown for every seed. Supplying good endpoints improves comparison; autonomous association remains unresolved.',normal_speeds='Supplied12normal and24speed order comparisons maintained each seed; auto normal no newly added warnings after consumer semantic fix, but unknowns increased. Not full normal feedback equivalence.',missing='All hand missing12conditions withheld; missing final endpoint not called success/incorrect. No natural learner-error or semantic completion accuracy claimed.',shape_direction='Original deployed models and outputs retained; no new shape/direction model promoted.',capability_gaps=[r for r in cap if not r['qualified']],timing=dict(comparator_queries_only_median=float(np.median([r['seconds']+r['additional_query_seconds'] for r in full])),comparator_queries_only_p95=float(np.percentile([r['seconds']+r['additional_query_seconds'] for r in full],95)),scope='Cached observer features and requirement heads only; excludes RTMW, recording, TCN, DTW, display; not end-to-end wait time.'),verification=v,web_changed=False,C_T3F_changed=False,private_used=False,downloads=False,external_transfers=False,fresh_holdout=False,implementation_and_verification='Codex, separate saved-output arithmetic/provenance script; no independent human expert',limitations=['Historical development clips reused.','Corrected order experiment was motivated by FIT bound infeasibility after first trial; exploratory sameDEV, not fresh confirmation.','Reversed-requirement queries and synthetic temporal reversals are not natural learner errors.','Main100 observer frozen; fine supervision54videos10expressions, demonstration4words.','No exact contact, mandatory amplitude, thumb correctness, or all-sign correctness labels invented.'])
wr(U/'FINAL_RESULT.json',result)
report='''설명 요구와 실제 행동 연결 학습 — 결과

결론
좋은 구간을 주면 설명의 순서와 실제 순서가 맞는지 구별하는 작은 모델은 개선됐습니다.
하지만 자동으로 구간을 고르면 아직 다른 순간을 비교합니다. 따라서 웹과 기존 판정부는 바꾸지 않았습니다.

1. 무엇을 가르쳤나
공개 학습 영상 54개, 표현 10종에서 기존 RGB 검토 정답 479항목을 재사용해 설명-관찰 연결표를 만들었습니다.
479는 영상 수가 아닙니다: 손모양 268개 시점·특징, 방향 136구간, 순서 75쌍입니다.
모든 항목이 특정 설명에 붙은 것은 아닙니다. 손모양/방향 중 직접 연결 가능한 것은 97항목이고, 나머지는 공통 특징 학습 또는 연결 보류입니다. 순서75쌍에는 기존 설명 단계 식별자가 있습니다.
두 화면 범위·반대 요구 대조를 합친 학습 입력은3004개입니다. 이것도 서로 독립적인 영상3004개가 아닙니다.
100단어899영상으로 학습한 공통 Conv+GRU 관찰부는 고정했습니다. 이번에는 작은 비교부와 순서 연결부를 각각3회 학습했습니다.
정답은 독립 농인/강사 주석이 아니라 과거 Codex의 공개RGB 검토 기록입니다. 이번에는4개 FIT 패널과 자료·연결·분리 규칙을 재점검했습니다.

2. 무엇이 달라졌나
첫 후보는 손모양227/228을 유지했으나 방향32/32가31/32,32/32,31/32로 떨어져 채택하지 않았습니다.
원인을 확인하니 기존의 높은 점수를 조금만 수정하는 구조는 잘못된 순서 점수118개 학습 입력을 이론상 뒤집을 수 없었습니다.
그래서 순서만 별도로, 관찰된 앞/뒤 행동과 설명의 앞/뒤 요구를 직접 짝짓도록 학습했습니다. 같은 관측에서 A→B와 B→A를 동시에 맞다고 할 수 없는 공통 구조입니다. 단어별 예외는 없습니다.

공개 개발 영상의 확인된 구간에서 비교:
- 정상 순서12쌍: 기존12/12 → 새12/12 유지.
- 느리게/빠르게 만든24쌍: 기존24/24 → 새24/24 유지.
- 실제 공개 좌표의 주요 시간 순서를 뒤집은12쌍: 기존확실한정답8 + 낮은확신정답1 → 새확실한정답11 + 보류1. 세 학습 모두 같은11확인/1보류.
  보류를 억지로 둘 중 하나로 고르면11/12,12/12,11/12이지만, 실제 안내 기준은 모두11확인/1보류입니다.
- 설명 순서만 뒤집는 별도 질문12쌍: 기존은0개, 새모델은12개에서 불일치를 구분했습니다. 이 숫자는 잘못 촬영한 사람12명을 뜻하지 않습니다.

3. 짧은 움직임·관측 부족을 어떻게 다뤘나
움직임이 짧다는 이유로 오답 라벨을 붙이지 않았습니다. 원래 관측에서 잘린 구간도 곧바로 잘못된 수어 정답으로 쓰지 않았습니다.
첫 비교 코드의 '원하는 방향 점수가 낮음=반대 방향' 해석 오류를 보존 기록하고 고쳤습니다.
지금은 '반대로 움직였다는 근거', '그 구간에서 뚜렷한 움직임을 확인하지 못함', '관절 관측 부족'을 나눕니다.
시간만 짧게 바뀐 경우와 몸 기준 좌표의 크기/위치 변화 검사 통과. 이는 이동거리70%가 교육적으로 허용된다는 검증은 아닙니다.
손 관측을 지운12조건은 정상이나 오답으로 확정하지 않았습니다. 마지막 동작을 지운 시험도 없는 시점을 가까운 다른 프레임으로 채워 완료시키지 않았습니다.

4. 아직 남은 핵심 문제
자동 구간 연결에서는 역순9개 설명쌍 중5개를 여전히 비슷하다고 보고4개를 보류합니다. 위의11/12 개선은 올바른 구간을 지정했을 때의 능력이며, 완성된 웹 피드백 성능이 아닙니다.
또 기존 이집트 카드의 '벌릴 때 양손 네 손가락 펴기' 기대2항목에는 설명 단계에 직접 연결된 FIT 손모양 정답이 없었습니다. 모서리로 보이는 손을 억지로 펴짐/주먹 정답으로 만들지 않았습니다.
교차가 있는 공개 동작도 자동 비교 묶음에 포함했지만, 엄지·접촉·모든 교차의 정확도까지 입증한 것은 아닙니다.

현재 적용 상태
TCN C·T3-F, 기존 손모양·방향 모델, 웹8814 그대로입니다. 새 순서 모델은 연구 후보로 보존했습니다.
학습자 실수 촬영의 실제 교육적 정오 검증은 아직 없습니다. 사용자·멘토 촬영은 접근·학습·튜닝·선택에 사용하지 않았습니다.
저장 결과7920항목 별도 산술/해시검산 통과. 같은 Codex의 다른 검사 코드이며 독립 전문가 검증이 아닙니다.
HTML은 만들지 않았습니다. 이 단위의 학습·평가 프로세스는 종료했습니다.
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf-8')
nextplan='''다음 단일 작업: 요구 순서를 강제로 따르지 않고 실제 행동 구간을 설명에 연결

근거: 새 순서부는 지정구간 역순12쌍에서11확인/1보류지만 자동구간9쌍은5거짓지지/4연결보류. 따라서 큰 모델 추가나100단어재학습보다 구간-설명 대응의 잘못된 선택을 먼저 해결한다.

1. 공개FIT의 후보와 설명 사이 정답 만들기
기존 직접RGB 단계 시점을 사용해 후보가 어느 설명행동에 해당하는지, 준비/복귀인지, 다른 행동을 섞었는지, 근거부족인지 구분한다. DTW만으로 옮긴 구간은 약한 근거로 따로 둔다.
하나의 후보에서 어느 설명요구가 실제로 보였는지와 보이지 않았는지를 기록한다. 짧은 정상후보는 길이만으로 음성이 되지 않는다. 같은손·비슷한모양이지만 다른 단계인 어려운 대조를 FIT에서 만든다.

2. 순서를 답으로 미리 정하지 않는 공통 연결부
목표 설명별 후보를 시간순으로 억지로 끼워맞추지 않는다. 관측 특징과 설명 요구의 대응 점수로 연결하고, 연결없음/여러후보 충돌을 허용한다. 그 후 실제 연결된 시각의 순서를 따로 검사한다.
앞행동을 뒤설명에 붙이거나 서로다른설명에같은구간을복제하는 경우를 대조로 학습한다. 기존순서부·손모양·방향부는 고정하고 필요한 연결부만 학습한다. 사람·상대신체위치·양손주체 공통함수, 단어ID예외코드 금지.

3. 같은 조건에서 구간지정/자동구간 비교
정상12+속도24,교차,짧은시도,관측소실,순서역전,단계삭제를 분리한다. 신규모델 점수보기전에 분할·3seed·성공기준을 고정한다.
이번완료결과재추론금지. 동일DEV는개발회귀로만보고하고 독립자료의 실제미노출여부는 별도감사한다. 없는사람정답은생성하지않는다.
정상경고·정상보류·역순거짓지지·누락오완료가 함께개선되어야 통과. 판정문장/시간표시도원래관측근거에연결한다.

4. 통과범위만 연구웹연결 검토
새 순서판정부가 더 좋아진 것과 구간찾기까지 완성된 것을 구별한다. 좋은구간에서의성능만으로 웹전체교체하지않는다. C·T3-F/기존관측부/100단어공통부 보호. 개인촬영금지. 추가다운로드·카메라·외부전송·Git 없음.

현재 이 후속은 계획만 준비, 미착수. 이번단위의학습/평가종료, 완료결과반복금지.
'''
(U/'NEXT_PLAN_KO.txt').write_text(nextplan,encoding='utf-8')
wr(U/'CURRENT_EXECUTION.json',dict(status=result['status'],pid=None,training_running=False,evaluation_running=False,web_changed=False))
wr(U/'CODE_MANIFEST.json',{str(p):sha(p) for p in U.glob('*.py')})
print(result['status'])
