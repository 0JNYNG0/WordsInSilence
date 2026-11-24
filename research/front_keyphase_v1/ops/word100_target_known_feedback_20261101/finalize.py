from pathlib import Path
import json,hashlib,datetime,statistics
U=Path(__file__).resolve().parent;R=U.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
result=read(U/'COMPARISON_RESULTS.json')
arithmetic=read(U/'review/COMPARATOR_RESULTS_VERIFICATION.json')
method=read(U/'review/FINAL_METHOD_REVIEW.json')
history=read(U/'review/HISTORY_DOCUMENT_REVIEW.json')
counter=read(U/'COMPARATOR_CONTROLS.json')
execution=read(U/'COMPARATOR_EXECUTION.json')
assert arithmetic['failure_count']==0
assert method['status']=='READY_FOR_RESEARCH_PROTOTYPE_COMPLETION_NOT_EDUCATIONAL_VALIDATION'
assert history['status'].startswith('PASS')
for p,h in execution['pins'].items():assert sha(p)==h,p
selftest=read(U/'COMPARATOR_SELFTEST_FROZEN.json')
assert selftest['status']=='PASS' and selftest['code_sha256']==sha(U/'comparator.py')
assert str(U/'COMPARATOR_SELFTEST_FROZEN.json') in execution['pins']
assert counter['status']=='PASS_TECHNICAL_ONLY' and result['new_training']==result['new_inference']==0
assert (U/'COMPARATOR_RESULT_KO.txt').exists()
stats={}
for m in ('global_dtw','core_first'):
 rr=[result['summary'][f'development/{m}/{s}'] for s in (42,43,44)]
 cov=[r['confidence_coverage']*100 for r in rr]
 stats[m]={'local_confidence_coverage_mean_percent':statistics.mean(cov),'range_percent':[min(cov),max(cov)],'expected_feature_frame_cells_per_seed':1080,'candidate_weights_by_seed':[r['difference_candidate_weight'] for r in rr]}
report=f'''2026-11-01 목표 단어를 아는 피드백 연결 — 네 단계 실행 결과

목표와 변천 기록
원래 목표는 한국수어를 배우는 게임에서 실제 행동의 중요한 차이와 허용 가능한 차이를 알려주는 것이다. 최근 미학습 단어 일반화 시험을 크게 다뤘지만, 게임에서 먼저 필요한 것은 알려진 지원 단어를 다른 사람이 수행할 때의 분석이다. 최종 목표를 바꾼 것이 아니라 주평가와 개발 우선순위를 바로잡았다. 이전 고정 실험의 실패·미채택 결과는 그대로다.
장기 기록: {R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'}
초기 Unity/인식→다중 참고 DTW→MP·RGB·YOLO·HaMeR/MANO·제공좌표 실험→설명과 특징 연결→SHuBERT/SignRep/시간별 판정부→현재 목표 단어 중심 평가를 동기·실제 결과·한계·근거별로 정리했다. 프로젝트 기준 §2와 개요에 연결했다.

완료한 네 단계
1. 목표와 진행 변천 기록 및 별도 자료 대조.
2. 공개20단어의 참고60영상(FIT 기록ID1~3)과 개발20영상(ID4) 연결. 사전 설명 후보12단어. 목표 단어는 참고 선택에 쓰고 관측값은 바꾸지 않는다.
3. 실제 단계가 준비된 순종·잘하다·조상·이집트4단어에서 기존 시간별3항목의 저장 관측과 두 자동 시간정렬을 연결. 참고별·시점별로 비교 가능/차이 후보/근거 부족을 분리. 한 부분을 못 봐도 다른 부분의 근거는 남긴다. 나머지16단어의 단계 비교 미지원도 실제 요청 결과에 남긴다.
4. 공개36쌍(FIT24/DEV12),두 방법,세 학습seed를 모두 고정 비교하고 별도 코드·수치 검산. 새 모델 추론/학습/완료 실험 재채점은0회이며, 기존 저장 출력의 새로운 연결 비교를 실행했다. 입력·좌표·확률·정렬 경로의 기존 값은 보존했다.

무엇이 가능해졌나
이제 '이번에 연습할 단어'를 지정하면 그 단어의 각 참고를 유지한 채, 어느 시점의 어떤 손모양 항목을 비교했고 어디가 부족한지 원프레임/시간/확률/관측 근거로 추적할 수 있다. 한 영상에 불확실한 구간이 있다고 모든 국소 비교를 버리는 방식으로 처리하지 않는다. 빠진 시점과 원래 참고 범위는 분모에 남긴다. 참고 주요구간에 대응되지 않은 입력 원프레임과 시각도 QUERY_COVERAGE_LEDGER.json에 전부 남겼다. 이 시간들은 '비교하지 않음'이며 휴식이나 정상 동작으로 바꾸지 않는다.
세 항목은 오른손의 검지만 펴짐(엄지 제외), 오른손 네 손가락 펴짐, 왼손 네 손가락 펴짐이다. '네 손가락을 편 모양이 아니다'는 두 결과가 같아도 둘 다 같은 손모양이라는 뜻은 아니다. 엄지·주먹 전체·접촉·수어 전체의 옳음을 이 세 값으로 단정하지 않는다. 양손 손목의 상대 위치는 별도 측정값으로 보존하며 허용거리/틀림 문턱은 새로 만들지 않았다.

수치의 정확한 의미
개발4영상×3참고에서 각 방법·seed당1080개의 참고시점×특징 기회를 유지했다. 아래는 그 중 관측 입력이 있고 모델 확신 기준을 만족한 비율이다. 인식률이나 피드백 정확도가 아니다.
global_dtw: 3seed평균{stats['global_dtw']['local_confidence_coverage_mean_percent']:.2f}%, 범위{stats['global_dtw']['range_percent'][0]:.2f}~{stats['global_dtw']['range_percent'][1]:.2f}%.
core_first: 3seed평균{stats['core_first']['local_confidence_coverage_mean_percent']:.2f}%, 범위{stats['core_first']['range_percent'][0]:.2f}~{stats['core_first']['range_percent'][1]:.2f}%.
공개 개발 영상에 나타난 차이 후보는 global에서seed별가중치3/1/5, core_first에서3/3/4.333이다. 소수는 한 참고프레임이 여러 질의프레임과 연결될 때 무게를 나눈 결과다. 정상 수어 오경고율이나 실제 틀린 동작 검출률로 해석할 수 없다. 모든 개발 차이 후보는 양쪽의 직접 RGB 정답이 동시에 충분하지 않아 검출기 오류인지 실제 물리적 차이인지 원인을 확정하지 못했다.

확인한 중요한 한계
실제 시간 대응의 별도 주석 대조에서360개 참고프레임 기회 중 같은 관측 동작종류로 연결된 무게는 global295.792(82.16%), core_first281.833(78.29%)였다. 나머지에는 다른 단계나 경계/미확인 구간이 있다. 같은 종류 연결도 반복의 정확한 회차·수어 정오를 증명하지 않는다. 결과를 보고 둘 중 하나를 새 정답 경로로 선택하지 않았다.
특히 정상 공개 '잘하다' P048→P047에서는 한 core_first 비교가22참고프레임 중4.67만 같은 관측 종류였는데, 세 손모양 항목의 차이는 나오지 않았다. 같은 손모양을 유지한 채 다른 동작을 해도 손모양 값만으로는 놓칠 수 있다는 구체적 반례다. 이 사례는 새로 개선했다고 보고하지 않는다.
또한 설명 후보가 있는12단어와 실제 단계 검토4단어의 지원 범위가 다르다. 4단어의 세 출력 항목이 설명의 핵심 규칙을 모두 담당하지 않는다. 설명을 보고 필수 동작/허용 범위를 완전히 배웠다는 결과는 아직 아니다.

검증과 문제 처리
최종 코드 자체검사28개와 저장 출력 변형 기술시험{counter['executed']}개 통과. 적격 관측 시점이 없는12시험 항목은 지원 부족으로 그대로 남겼다. 속도 변경은 저장출력과 알려진 대응을 변형한 시험이며, 새 영상에서 자동 정렬까지 성공했다는 검증은 아니다. 실제 학습자 오류 영상 평가0건이다.
자료 검산 담당은 원래 저장 입력·확률·72경로에서 분모와 계산을 별도로 재구성했다. 방법 검토 담당은 정답 주석이 자동 정렬에 새어들지 않는지, 음성-음성 일치를 같은 수형으로 과장하지 않는지, 미관측/반복/모호한 정렬을 보존하는지 확인했다. 같은 Codex의 별도 문맥 검산이며 독립 한국수어 전문가 정답 확인은 아니다.
실행 전 코드 검토에서 정렬 경계 바깥을 확실한 대응으로 세지 않도록 했고, 옛 alias가 없는 영상들이 서로 같은 영상으로 잘못 제외되지 않도록 했다. 후보가 여러 개인 정렬도 숨기지 않는다. 실행 ledger의 최종 selftest는FROZEN/28검사/최종코드SHA가 일치함을 확인했다. 이전 CURRENT시험파일과의 혼동은 실행결과 오류나 ledger 수정이 아니다.

최종 판단
이번 네 단계의 기록·연결·비교·검산은 완료했다. 개선된 것은 부분 근거를 잃지 않는 비교 기능과 근거 추적성이다. 단어 인식률 또는 교육적 교정 정확도 향상은 이번 결과로 입증하지 못했다. 완전한 피드백 AI나 웹 적용 완료라고 보고하지 않는다. 명확한 성능 향상 주장이 없으므로 HTML은 만들지 않았다.

다음 핵심 작업
지원 단어 안에서 같은 손모양이지만 동작 단계가 다른 경우를 놓치지 않는 작은 피드백 완성본이 우선이다. (1) 알려진 목표의 참고 단계와 설명에서 확인 가능한 모양·양손 관계·이동 특징을 함께 연결한다. (2) 실제 관측은 그대로 두고, 같은 단계로 대응했는지의 근거를 별도로 표시한다. 손모양 일치만으로 잘했다고 하지 않는다. (3) 같은 표현의 다른 사람 정상 영상과, 중요한 단계가 다른 통제 사례를 같은 분모로 비교한다. 출력 변형만으로 실제 교육오류 감지를 주장하지 않는다. (4) 시점은 맞고 보이는데도 특정 특징을 잘못 분류한다는 근거가 모일 때 그 특징의 학습만 보강한다. 새 검출기/거대 모델 탐색이나 미세 흔들림 전수검사로 돌아가지 않는다.
이 결과를 토대로 다음 실행 전 기준을 고정한다. 이미 본 개발 점수로 임계값/단어 예외를 맞추지 않는다. 현재 단위에서는 추가 학습이 병목 해결에 필요한지 확인되지 않아 반복 학습하지 않았다.

보존
C·T3-F 및 운영 웹/DTW32보호핀과 과거 공개근거37개 보존. 사용자·멘토 촬영 접근·학습·튜닝·선택0,새 다운로드/외부전송/카메라/Git0. 이번은 저장 관측 재생이므로 촬영 종료→피드백 전체 대기시간을 새로 측정하지 않았다.
상세 비교/시점 예시: {U/'COMPARATOR_RESULT_KO.txt'}
실행: {U/'COMPARATOR_EXECUTION.json'}
별도 검산: {U/'review/COMPARATOR_RESULTS_VERIFICATION.json'}
방법 검토: {U/'review/FINAL_METHOD_REVIEW.json'}
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf8')
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
completion=dict(status='COMPLETE_TARGET_KNOWN_LOCAL_PROTOTYPE_NOT_EDUCATIONAL_VALIDATION',at=now,
 executed_steps=['history_record_and_review','known_target_reference_bank','saved_automatic_path_local_comparator','fixed_public_replay_and_independent_verification'],
 goal_unchanged=True,primary='Known supported word / different performer',unseen_word='Supplementary diagnostic; historical failures unchanged',
 bank_words=20,fit_references=60,development_queries=20,phase_supported_words=4,phase_unsupported_words=16,
 pairs=36,fit_pairs=24,development_pairs=12,methods=2,seeds=3,original_parent_instances=117,child_instances=144,feature_cells_total=19440,
 new_training=0,new_model_inference=0,old_completed_scores_recomputed=False,local_coverage_not_accuracy=stats,
 technical_checks=counter['executed'],unsupported_technical_cases=counter['unsupported'],final_code_selftest_count=selftest['count'],
 independent_arithmetic_checks=arithmetic['check_count'],educational_accuracy=None,word_accuracy_improvement=None,
 real_learner_errors_evaluated=0,production_adoption=False,protected_hashes=32,user_mentor_access=False,html_created=False,
 history=str(R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'),report=str(U/'RESULT_KO.txt'),
 next='Known-word reference stages: distinguish same shape/different movement and uncertain correspondence before extra training; preserve all public normal/ref/missing evidence, no target-word observation overwrite.',
 evidence_pins={str(p):sha(p) for p in [U/'COMPARISON_RESULTS.json',U/'PROJECTION_AUDIT.json',U/'COMPARATOR_EXECUTION.json',U/'COMPARATOR_RESULT_KO.txt',U/'RESULT_KO.txt',U/'review/COMPARATOR_RESULTS_VERIFICATION.json',U/'review/FINAL_METHOD_REVIEW.json',U/'review/HISTORY_DOCUMENT_REVIEW.json']})
new(U/'COMPLETION.json',completion)
ctx=R/'WordsInSilence_PROJECT_CONTEXT.md';s=ctx.read_text(encoding='utf8')
h8='## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인';h9='## 9. 다음 단일 작업'
p8='''### 2026-11-01 목표 단어 기반 연결·고정 비교 완료 — 부분 근거 보존, 교육 정확도 미검증

`word100_target_known_feedback_20261101`의 네 단계 완료. 원래 최종목표 유지/지원 단어 새수행자 주평가/미학습단어 보조진단을 §2와 `Docs/PROJECT_DIRECTION_HISTORY_KO.txt`에 기록했고 과거FAIL은 불변. 공개20단어 FIT60/DEV20 참고선택을 실제 실행, 단계지원4단어(순종·잘하다·조상·이집트)/16단어 미지원. 원3항목 전체시점 확률·두자동정렬·3seed 재사용해36쌍/117부모/144세부구간/19440셀 연결. 미관측에도 전체참고분모 보존, 국소근거는 유지. 새학습/모델추론0. 자체28/기술183검사통과·지원부족12보류, 별도 산술/방법검산 완료. 동일Codex검산≠독립수어정답.

DEV확신기준비교범위는약85~87%지만 정확도 아님. 같은 관측단계 연결은global82.16%/core_first78.29%(경계·불명·다른단계 포함한360기회). '잘하다' 일부는다른단계대응에도세손모양차이가없어 전체정오판정 불가. 음성-음성일치≠같은수형, 실제교육오류평가0. 코드연결 개선은확인/인식·교정성능향상은미입증. C/T3F·운영32핀·과거근거37개보존, 사용자/멘토접근0,HTML생략. 결과/완료: RESULT_KO.txt·COMPLETION.json. 실행중프로세스없음.

'''
p9='''### 2026-11-01 다음 단일 작업 — 같은 손모양의 다른 동작 단계를 구분하는 지원 단어 피드백

방금완료한 target_known_feedback 결과를 재실행하지 않는다. 알려진목표/개별FIT참고를유지해 주요단계의 모양+양손관계+이동 특징과 설명의 실제지원부분을 연결하고, 대응근거와차이근거를분리한다. '잘하다' 같은손모양/다른단계 반례를 공통기능반례로 유지하고 단어예외를 만들지 않는다. 같은표현다른사람·관측부족·짧은중요차이를 함께 검증한다. 현재추가학습필요 근거없음; 시점/관측문제를분리한뒤 특징분류실패와적절한정답지원이확인되면 한정학습계약을먼저동결. 미학습단어 일반화를 제품선결관문으로되돌리거나 새모델순회/미세hold전수감사를재개하지 않는다. 실제교육오류 정답 없음을 합성시험으로 숨기지 않는다. 최종목표와이력은§2/Docs/PROJECT_DIRECTION_HISTORY_KO.txt 유지.

'''
s=s.replace(h8,h8+'\n\n'+p8.rstrip(),1).replace(h9,h9+'\n\n'+p9.rstrip(),1);ctx.write_text(s,encoding='utf8')
statepath=U.parent/'STATE.json';state=read(statepath)
state.update(updated_date_kst=now,current_stage=completion['status'],next_single_action=completion['next'])
state[U.name]=completion|dict(active=None)
statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
(R/'NEW_CHAT_HANDOFF.md').write_text('''# WordsInSilence 최신 인수인계

현재 word100_target_known_feedback_20261101 COMPLETE_TARGET_KNOWN_LOCAL_PROTOTYPE_NOT_EDUCATIONAL_VALIDATION. 승인된 네 단계(이력/참고은행/기존관측+자동정렬연결/공개고정비교·별도검산)완료. 실행중없음. RESULT_KO.txt·COMPLETION.json 참조. 모델재추론/학습/완료비교재실행X.

최종목표는 한국수어설명+다중참고로 중요한차이/허용변이를알려주는것 그대로다. 이번우선순위정정: 알려진 지원단어를 다른사람이 수행하는 비교가 주평가, 미학습단어전이는 보조진단. 원semantic6학습/FAIL은 불변. 목표단어는참고/특징선택에만쓰며 관측정답을덮어쓰지않는다. TCN정답이피드백시작필수조건아님. 장기이력 Docs/PROJECT_DIRECTION_HISTORY_KO.txt(본류10단계+관측6분기),§2/개요연결. 멘토/교수보고근거로유지.

공개20단어FIT60(ID1~3)/DEV20(ID4) 실제라우팅. 설명후보12,단계지원4단어16영상(순종·잘하다·조상·이집트),나머지16단계미지원.36쌍 FIT24/DEV12,원117부모/144세부구간,2저장자동경로×3seed=19440셀. 원3손모양전체시점저장확률재사용; 신규7추가4는3시점뿐이라dense복사X. query물리주석은사후검산만. 미관측분모보존/부분관측유지/참고별모호함보존.

DEV비교가능확신범위약85~87%≠정확도. 같은관측종류대응 global295.792/360(82.16%),core281.833/360(78.29%),반복회차/언어정오정답아님. 잘하다P048→P047 corefirst22중4.67만같은종류인데3모양차이0:모양일치만으로동작정오불가. 대부분일치는둘다해당모양아님을포함하므로같은수형으로과장X. 실제DEV차이후보는양쪽직접RGB정답부족,오경고/분류오류원인확정X. 실제학습자오류평가0,교육성능미입증.

검증: 최종코드28selftest/FROZEN핀일치,기술183PASS/지원부족12,산술47062검사·방법별도검산,32보호핀/과거근거37개보존. 동일Codex문맥검산≠독립사람정답. HTML생략,C/T3F·운영웹/DTW/사용자·멘토자료유지. 새다운로드/카메라/Git/외부전송0. 웹기본M0와명시선택C/T3F,연구SignRep/판정부구분.

다음한작업: 지원단어의같은모양/다른동작단계를 공통으로구분하도록 참고단계의모양·양손관계·이동과 설명의지원부분을연결. 자동대응근거와차이근거를분리하고같은단어다른사람/누락/짧은차이를함께검증. 새모델탐색·미세hold전수감사·미학습단어선결관문으로돌아가지않기. 시점과관측이맞는데특징분류가실패한다는근거가있을때만추가학습계약고정. root만§8§9/STATE/HANDOFF갱신.
''',encoding='utf8')
historypath=R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
with historypath.open('a',encoding='utf8') as f:
 f.write('\n2026-11-01 이번 네 단계 실행 완료 추기\n참고20단어 실제선택/4단어 부분비교/36쌍 두자동경로·3seed 검산 완료. 새로운 학습0. 부분 비교 정보는 유지할 수 있게 됐지만 같은 손모양의 다른 동작 단계가 연결돼도 차이가 적게 나오는 한계가 확인됐다. 인식률·교육교정 성능 향상으로 보고하지 않았다. C·T3-F와 원FAIL/기존결과보존. 상세 '+str(U/'RESULT_KO.txt')+'\n')
print(json.dumps({'status':completion['status'],'report':str(U/'RESULT_KO.txt'),'new_training':0,'new_inference':0},ensure_ascii=False))
