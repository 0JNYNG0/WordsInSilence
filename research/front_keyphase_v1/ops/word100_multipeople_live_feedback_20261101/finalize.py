"""Summarize completed, immutable measurements; no training or inference."""
from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
if (U/'COMPLETION.json').exists():raise RuntimeError('Final report already completed; append a correction instead')
training=read(U/'TRAINING_REPORT.json');evaluation=read(U/'FEEDBACK_EVALUATION.json');guard=read(U/'GUARD_REPORT.json')
ui=read(U/'GUARD_LIVE_UI_VERIFICATION.json');active=read(U/'SERVER_ACTIVE.json')
audit_names=['SAVED_TRAINING_VERIFICATION.json','BANK_CALIBRATION_VERIFICATION.json','FEEDBACK_EVALUATION_VERIFICATION.json','GUARD_RESULT_VERIFICATION.json','LIVE_WRAPPER_SELFTEST_FINAL.json','GUARD_LIVE_VERIFICATION.json']
audits={n:read(U/'review'/n) for n in audit_names}
for n,a in audits.items():
 assert str(a['status']).startswith('PASS'),(n,a.get('status'))
 assert not a.get('failures'),(n,a.get('failures'))
assert ui['status']=='PASS' and training['adoption']['allowed'] is False and evaluation['errors']==0
assert guard['original_metrics_and_paths_unchanged'] and guard['all_source_pins_unchanged']
assert sha(U/'engine.py')=='68a94384350353db6c73025cba7c99ee406be593186c198f371d41b26fd75c11'
assert sha(U/'evidence_guard.py')==guard['code_sha256']
assert sha(U/'index.html')==ui['html_sha256'] and sha(U/'app.js')==ui['js_sha256']
same=evaluation['per_kind']['same_expression']['modes'];latency=ui['latency'];old_ui=read(U/'LIVE_UI_VERIFICATION.json')
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
report=f'''새 촬영과 여러 공개 참고를 비교하는 4표현 연구 데모 — 완료 결과
작성: {now}
실행: http://127.0.0.1:8802/   (기존 8801 기록 재생 화면은 그대로)

먼저 질문에 대한 답
여러 사람의 자료를 활용하자는 방향은 맞다. 다만 많이 넣기만 하면 좋아진다고 보장할 수는 없다.
이전에는 단어 이름만 있으면 되는 TCN 학습과 달리, 피드백 판정부에 필요한 '어느 시점에 손가락이 펴졌는가' 같은 확인된 정답이 ID01~03에 한정되어 있었다.
따라서 이미 사전학습된 SignRep을 고정하고 작은 시간별 판정부부터 확인한 시작은 타당했다. 완전히 새로운 AI를 세 사람만으로 처음부터 학습한 것은 아니다.
그러나 소규모 세부 검증을 오래 반복하고, 다인 참고와 실제 새 촬영을 연결하는 일을 늦춘 것은 우리의 진행 판단에서 개선할 점이었다.
이번에는 인원 확대가 실제로 도움이 되는지 직접 비교하고, 과거 결과 재생에 그치던 화면을 새 입력 분석으로 연결했다.

무엇이 달라졌나
이전: 미리 계산한 공개 영상 쌍의 결과를 보는 화면. 사용자의 새 영상을 분석하는 기능은 없었음.
현재: 순종·잘하다·조상·이집트 중 목표 표현을 고르고 파일을 올리거나 직접 촬영하여 분석 요청 가능.
같은 표현의 공개 참고 9개(기록 수행자 ID 기준)를 각각 비교한다. 평균 영상으로 합치지 않는다.
통합 안내는 전체 흐름이 가까운 같은 두 참고를 계속 사용한다. 구간마다 유리한 사람을 골라 붙이지 않는다.
모든 9개 비교 결과도 따로 보이며, 실제 DTW 경로로 두 영상을 함께 재생하고 큰 차이가 난 원시점으로 이동할 수 있다.
촬영은 사용자가 카메라 버튼을 눌러야 시작한다. 이번 검증은 공개 파일 업로드로 수행했으며 실제 사용자 카메라 촬영은 하지 않았다.

추가 학습과 자료
공개 4표현×12기록ID=48영상. 새 프로토콜 FIT ID01~09 36영상, DEV ID11~13 12영상.
영상별 3개 중앙시점, 총144시점의 손모양/상하관계 정답을 준비했다. 13프레임은 입력 문맥이며 13개 정답으로 부풀리지 않았다.
기존20표현의 FIT180센터를 유지하고, 지원4표현의 ID04~09 72센터를 더해252센터로 학습 범위를 늘렸다.
3인 조건/9인 조건 각각 seed42·43·44, 160epoch 고정, 총6회. SignRep 전체를 재학습한 것이 아니라 작은 시간별 판정부 지도학습이다.
검증은 두 조건 모두 같은 DEV36센터/확인 가능한194항목에서 수행했다. 58항목은 불확실하여 정답 점수에서 제외하고 따로 계수했다.
7항목 평균 정확도(3seed 평균): 3인 {100*training['aggregates']['matched_three_people']['macro_supported_target_accuracy']['mean']:.4f}% → 9인 {100*training['aggregates']['expanded_nine_people']['macro_supported_target_accuracy']['mean']:.4f}%.
차이 {100*training['adoption']['expanded_minus_matched_macro_accuracy']:.4f}%p. 개선 조건 미충족으로 9인 후보는 교체 적용하지 않고 보호된 기존 판정부를 사용한다.
이 값은 단어 인식률이나 피드백 정확도가 아니다. R_index_only 양성0, L_open_four 양성1인 작은 DEV로 모든 손모양 일반화를 주장할 수 없다.
추가 인원은 4표현에만 확대했고, 기존 다른16표현의 FIT를 모두9인으로 늘린 것은 아니다.
기록ID04~08의 과거 평가 역할은 이번 새로운 실험에서 명시적으로 종료했다. 과거 실험·결과는 변경하지 않았다.
ID11~13은 과거 TCN 개발 자료이므로 프로젝트 전체에서 처음 본 사람이나 최종 독립 평가라고 부르지 않는다.
직접 RGB 주석은 Codex의 시각 검토이며 한국수어 전문가가 만든 교육적 정오 정답은 아니다. 새 DEV36시점 전부와 새FIT15시점을 별도 다시 확인했다.

여러 참고를 활용한 효과 — 같은 표현의 다른 수행자12영상/39구간
                         참고3개       참고9개
관측한 특징이 유사함       {same['subset3']['stages']['SIMILAR_REFERENCE_RANGE']:3d}             {same['all9']['stages']['SIMILAR_REFERENCE_RANGE']:3d}
차이 후보                 {same['subset3']['stages']['DIFFERENCE_CANDIDATE']:3d}             {same['all9']['stages']['DIFFERENCE_CANDIDATE']:3d}
확인 근거 부족            {same['subset3']['stages']['INSUFFICIENT_EVIDENCE']:3d}             {same['all9']['stages']['INSUFFICIENT_EVIDENCE']:3d}
정상 공개 자료에서 차이 표시가10→7로 줄었다. 하지만 남은7개가 모두 실제 오류라는 뜻은 아니다.
이는 참고의 다양성과 FIT 정상범위 계산 자료를 함께 바꾼 개발 비교이며, 인원 수만의 순수한 인과 효과는 아니다.
서로 다른 표현을 목표로 고른36개 대조117구간은 9참고에서 차이113/근거부족4/전체구간유사0이었다.
이 대조는 다른 표현을 알아차리는 시험이며 '엄지 하나가 틀렸는가' 같은 세부 교육 정확도 시험은 아니다.
저장 특징을0.75/1.25배속으로 바꾼 대조에서는39구간당 차이6개씩(3참고11/12개). 실제 영상 재검출을 다시 한 속도 견고성 시험은 아니다.

시험 중 발견해 고친 중요한 문제
관측을 지운 주요 구간 대신, 직전에 손이 보이던1~4프레임을 주요 구간 전체에 연결하는 문제가 있었다.
원래 참고 분모가 그대로여도 실제 비교한 새 영상 시점이 잘못되면 '유사'가 나올 수 있었다.
이집트 ID11/12의 제거 대조3구간에서 이 문제가 확인됐다. 단어 이름으로 예외를 만들지 않았다.
기존 손·팔·모양 DTW는 보존하고, 손 검출 누락의 영향을 직접 받지 않는 pose 팔 위치의 별도 시간 대응을 추가했다.
원래 대응과 팔 대응에서 필요한 query-frame×항목의 합집합을 확인한다. 기존0.65 관측 기준을 못 넘으면 '유사'만 '근거 부족'으로 내린다.
새 정답·오답을 만들거나 원래 차이 수치를 바꾸지 않는다. 팔의 보조 대응도 정답 시간표가 아닌 추정이라고 명시한다.
동일96개 개발 조건에서 제거 대조의 남은 유사3개가 양 조건 모두0개가 됐다. 같은 표현39구간/273항목은 변화0.
속도 대조에서는 단계 판정 변화0, 느린 대조의 왼손모양1항목이 추가 보류됐다. 짧은 합성변형 대조의3참고 양손관계1항목도 추가 보류됐다.
이것은 발견한 반례에 대한 개발 안전성 회귀 확인이다. 새 독립 정확도나 모든 미검출 문제 해결로 주장하지 않는다.
참고 손내림에 따른 비교 제외도 정렬/수치 계산에 동일하게 적용하고 원래 항목 수와 제외 항목 수를 함께 보존했다.
손내림 근처의 손가락 동작이 언어적으로 중요하지 않다는 정답까지 확보한 것은 아니다.

실제 새 영상 처리와 속도
공개 순종 ID12 123프레임: 실제 업로드 후 전체 분석 {old_ui['latency']['total_analysis_seconds']:.2f}초.
안전장치까지 연결한 공개 이집트 ID13 121프레임: {latency['total_analysis_seconds']:.2f}초.
두 영상은 다르며 서버 준비시간을 별도 기록했다. 이 차이를 속도 개선율이나 안정적인 평균으로 해석하지 않는다.
후자 관절·디코드·저장 {latency['decode_mediapipe_assignment_save_seconds']:.2f}초, 영상특징 {latency['rgb_decode_and_signrep_seconds']:.2f}초,
작은 판정부 {latency['head_load_and_inference_seconds']:.3f}초, 안전장치 {latency['evidence_guard_seconds']:.3f}초.
실제 GPU.0/f16 사용이며 CPU fallback은 발생하지 않았다. 준비된 모델 초기화는 재사용했지만 영상별 예측은 새로 계산했다.
따라서 빠른 게임 피드백 속도 목표는 아직 미달이다. 다른 영상/전원/부하에서 이 시간이 유지된다고 보장하지 않는다.
두 단계 웹 시험에서9개 참고 영상 로딩·실제 재생·경로 동기재생·원시점 이동·가독성·좁은 화면·오류를 검사했다.
브라우저가202응답을 받기 전에 연결이 끊겨도 이미 접수한 작업을 실패로 덮거나 중복 실행하지 않게 고쳤다.

현재 데모가 할 수 없는 것
손모양 몇 항목·관절 각도·양손 상대 위치·팔 이동의 관측 차이를 보여주는 제한 데모다.
설명문 전체를 이해해서 모든 행동 규칙의 중요성과 허용 범위를 판단하는 AI가 완성된 것은 아니다.
엄지의 세밀한 모양, 손목의 정확한3D회전, 실제 피부 접촉, 정확한 반복 횟수, 수어 전체 의미·교육적 정오는 아직 확정하지 않는다.
정상39구간에도 차이 후보7/근거부족17이 남는다. '차이 후보'는 곧바로 '잘못했으니 고쳐라'라는 지시가 아니다.
원본 손·팔·얼굴 관측과 RGB가 존재한다고 정확한 관측으로 보장하지 않는다. 이번 face sidecar는 판정부 입력에 쓰지 않는다.
개발 자료 재사용 진단과 사람 전문가의 정답 검증은 별개다.

다음 작업의 우선순위
1. 실제 새 입력에서 발생하는 정상 차이 경고를 단계 대응 문제와 손 특징 오판으로 분리한다. 불확실한 시점을 더 좋은 학습 정답 없이 모델 크기/인원만 늘려 덮지 않는다.
2. 표현 설명에서 중요한 손모양·접촉·순서 항목을 제한된 검증 정답과 연결해, 관측 차이가 실제 교정할 차이인지 평가한다. 단어별 임시 문턱/예외는 만들지 않는다.
3. 처리시간을 관측/전처리/영상특징별로 더 분리한다. 현재 판정에 쓰지 않는 face 계산, 디코드 반복, 촬영 중 계산 분산을 정확도 변화와 분리해 검토한다. 프레임을 줄여 짧은 동작을 놓치는 최적화는 사전 검증 없이 적용하지 않는다.
4. 위 조건이 갖춰진 범위만 안내 문장으로 연결하고 다른 환경에서 재확인한다. 현재 결과로 전체 단어/상용 교육 정확도를 주장하지 않는다.

보존과 검증
기존 C·T3-F/운영 웹·DTW·이전 결과는 보존. 새 학습/튜닝/선택에 사용자·멘토 촬영 사용0, 새 외부 다운로드·전송0.
새 촬영은 이 컴퓨터의 격리 세션에서 추론만 하며 FIT/정상범위/참고은행에 들어가지 않는다.
구현root, 데이터·서버·후보구현semantic_data_audit/semantic_model/semantic_feedback, 별도 저장 산술검산 수행.
동일 Codex의 역할 분리 검산이지, 독립 인간 수어 정답이나 전문가 교육 평가가 아니다.
주요 근거: FROZEN.json, TRAINING_REPORT.json, BANK_PREDICTIONS.json, NORMAL_RANGES.json,
FEEDBACK_EVALUATION.json, GUARD_FROZEN.json, GUARD_REPORT.json, GUARD_LIVE_UI_VERIFICATION.json, review/의 검산들.
'''
(U/'RESULT_KO.txt').write_text(report,encoding='utf8')
status='COMPLETE_MULTIPEOPLE_NEW_INPUT_RESEARCH_DEMO_WITH_LIMITS'
summary='公開4표현 새촬영/업로드→9인 개별참고→동일2전체참고 합의→구간별 근거와 동기재생을 localhost:8802에 연결. 공개48영상/144센터 준비;3인180센터와9인252센터 각3seed 학습에서98.7586→98.4039%(좁은7특징)로 개선없어 기존보호판정부유지. 12DEV영상39구간의 정상차이표시3참고10→9참고7,유사9→15,근거부족20→17. 누락대조3구간이직전보이는시점으로도피한문제를별도팔대응근거보수검사로유사3→0(보류)수정;정상39구간변화0. 속도/짧은대조각1feature추가보류. 실제공개업로드약15.72/31.52초로속도미달. 교육정오/전체수어AI완성아님. C/T3F·운영·개인자료보존.'
summary=summary.replace('公開','공개')
next_action='현재4표현 새입력 연구데모를 기준으로 정상차이경고7/39와속도15.7~31.5초를우선해결. 기존완료학습/864기본경로/648보조경로재실행X. 지표는교육정오아님. 정확한단계/중요특징정답이확인된문제만보완하고정상회귀와경계안전성검증. 사용자촬영은격리추론만,학습/튜닝/후보선택X. 기존C/T3F·운영교체/신규다운로드확대X. 모델규모·학습인원만추가하면해결된다고가정X.'
title='### 2026-11-01 다인 학습·새촬영 피드백 데모 완료 — 근거 보완, 속도·교육 정확도 과제 유지'
ctx=R/'WordsInSilence_PROJECT_CONTEXT.md';text=ctx.read_text(encoding='utf-8-sig')
assert title not in text
text=text.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+title+'\n'+summary,1)
text=text.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업\n\n### 2026-11-01 다음 단일 작업 — 새입력 정상 경고와 실제 처리시간 개선\n'+next_action,1)
ctx.write_text(text,encoding='utf8')
st=O/'STATE.json';s=read(st);s['current_stage']=status;s['next_single_action']=next_action
s[U.name]=dict(status=status,updated_at=now,summary=summary,next_action=next_action,unit=str(U),report=str(U/'RESULT_KO.txt'),demo_url='http://127.0.0.1:8802/',server_pid=active['pid'],server_engine='live_engine',training_runs=6,candidate_adopted=False,observer='protected_baseline',reference_people_ids=list(range(1,10)),development_people_ids=[11,12,13],guard_applied_in_separate_research_demo=True,production_changed=False,user_training=False,learning_and_experiment_processes_finished=True)
save(st,s)
(R/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+now+'\n\n'+summary+'\n\n다음: '+next_action+f'\n\n실행 중은 별도 연구 서버 PID {active["pid"]}, 127.0.0.1:8802, engine live_engine뿐. 기존8801은 과거결과재생. 신규서버 실제PID/health 확인 후 재시작 여부 결정. 카메라는 사용자버튼으로만 켜짐, 파일업로드 공개2편 검증. 모든학습·평가·추가보조경로·저장검산 종료. 완료 결과 재추론·재점수화 금지.\n\n현재 단위: {U.name}. RESULT_KO.txt/COMPLETION.json/FROZEN.json/各report/review 참조. 3인으로시작한이유는시간별확인정답부족;소규모반복이길어진점은진행판단한계로기록. 과거head평가ID04~08을새실험FIT로사용함을사전명시하여옛결과보존;새학습DEV11~13는과거TCN DEV이므로전체프로젝트최종독립시험아님.\n\n원본W:/sources와기존C/T3F·운영/과거결과읽기전용,사용자자료는격리추론만. 모델후보미채택과별도데모참고확장/근거guard적용을구별. 별도Codex검산을사람전문가정답으로부르지말것. 추가HTML보고는생략하고실제데모와TXT결과만제공.\n',encoding='utf8')
history=R/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
with history.open('a',encoding='utf8') as f:f.write('\n\n'+now+' 다인 학습과 실제 새 촬영 데모\n'+summary+'\n방향성: 사용자 목표는 동일하게 유지. 과거 공개기록 재생 데모를 실제 새입력 피드백으로 오인하지 않도록 정정하고, 지원4표현의 다인 관측범위를 실제로 연결했다. 인원증가학습은 단정하지않고6회비교 후 미채택. 부족한확인정답으로3인부터시작한근거와작은검증반복으로통합을늦춘판단한계를명시. 관측분모뿐아니라query시간범위도잘못될수있음을발견,독립팔단서안전장치를추가해보이지않는근거를맞았다고승격하지않도록보완. 교육적허용규칙모델완성/출시속도달성은아직아님.\n근거: '+str(U/'RESULT_KO.txt')+'\n')
save(U/'CURRENT_EXECUTION.json',dict(status=status,server=active,model_and_evaluation_processes_finished=True,no_active_training=True,next_action=next_action))
completion=dict(status=status,completed_at=now,summary=summary,report=str(U/'RESULT_KO.txt'),server=active,source_pins={str(U/p):sha(U/p) for p in ['FROZEN.json','TRAINING_REPORT.json','FEEDBACK_EVALUATION.json','GUARD_REPORT.json','engine.py','evidence_guard.py','live_engine.py','server.py','index.html','app.js','GUARD_LIVE_UI_VERIFICATION.json','RESULT_KO.txt']},reviews={n:dict(status=a['status'],checks=a.get('checks'),sha256=sha(U/'review'/n)) for n,a in audits.items()},scope='bounded research demo and developmental comparisons completed; full project and education correctness not complete',C_T3F_preserved=True,user_or_mentor_training=False)
save(U/'COMPLETION.json',completion)
print(status)
