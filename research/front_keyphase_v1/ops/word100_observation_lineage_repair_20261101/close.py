"""Close this verified research increment and synchronize authoritative records."""
import hashlib,json,re
from datetime import datetime,timezone,timedelta
from pathlib import Path
U=Path(__file__).resolve().parent;ROOT=U.parents[3]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (U/'COMPLETION.json').exists(),'Already closed'
for name in ['RECOVERY_RUNNING.json','BRIDGE_RUNNING.json']:
    assert not (U/name).exists(),name
for name in ['RECOVERY_FROZEN.json','BRIDGE_FROZEN.json']:
    obj=read(U/name)
    for p,h in obj['pins'].items():assert sha(p)==h,p
for name in ['RECOVERY_MATH_VERIFICATION_FINAL.json','RECOVERY_SOURCE_VERIFICATION.json','BRIDGE_VERIFICATION.json','REPORT_REVIEW.json']:
    obj=read(U/'review'/name);assert obj['status'].startswith('PASS'),name
assert (U/'review/RESIDUAL_PLAN_REVIEW.json').exists()
f=read(U/'FINAL_REPORT_FACTS.json');now=datetime.now(timezone(timedelta(hours=9))).isoformat()
review=read(U/'review/REPORT_REVIEW.json')
for p,h in review.get('pins',{}).items():assert sha(p)==h,p

current={'phase':'COMPLETE','at':now,'active':None,'scope':'Fixed public303 source-frame observation recovery plus existing39-event geometry bridge.',
 'new_perception_inference_calls':42,'new_classifier_inference':0,'new_training':0,'new_DTW':0,
 'new_geometry_comparisons':11,'baseline_comparisons_reused':28,
 'saved_outputs_reinferred_or_rescored':False,'inference_exit_code':0,'bridge_exit_code':0,
 'new_html':False,'production_changed':False,'user_mentor_data_used':False,
 'unresolved_hand_slots':4,'unvalidated_finger_points':6,
 'next':'Prepare common temporal-phase correspondence evidence; no score-driven retuning, no normal/unseen data treated as corrected truth.'}
write(U/'CURRENT_EXECUTION.json',current)
status=read(ROOT/'research/front_keyphase_v1/ops/STATE.json')
status.update(current_stage='OBSERVATION_LINEAGE_REPAIR_COMPLETE',updated_date_kst=now,updated_at=now,
    next_single_action='현재 공개자료의 앞뒤 양손 정체·높이교환·이동방향과 반복 occurrence를 구분하여 같은 주요 단계 대응의 평가 근거를 준비한다. 기존 W06/W07 검토는 평가 전용이며 경로 입력에 넣지 않는다. 완료 재검출/비교는 재실행하지 않는다. word100_observation_lineage_repair_20261101/NEXT_PLAN_KO.txt 참조.')
status[U.name]={**current,'unit':str(U),'report':str(U/'RESULT_KO.txt'),
 'fixed_public_frames':303,'executed_recovery_frames':20,'eligible_hand_slots':22,
 'added_hand_frames':18,'new_supported69_cells':604,'old_supported69_exact':16154,
 'comparison_denominator':30560,'support_before':24206,'support_after':24670,
 'unavailable_before':6354,'unavailable_after':5890,'independent_checks':{'lineage':8229,'source':7182,'recovery_math':5522,'bridge':33749},
 'recognition_or_instructional_accuracy_gain':None,'fine_finger_landmark_truth':False,
 'reviewer_type':'Separate Codex agents/code; not independent human KSL truth',
 'first_verifier_failure_preserved':True,'next_plan':str(U/'NEXT_PLAN_KO.txt')}
write(ROOT/'research/front_keyphase_v1/ops/STATE.json',status)
sections={8:'''### 2026-11-01 선택적 손 재관측과 비교 연결 완료 — 관측 근거 증가
`word100_observation_lineage_repair_20261101` 완료. 최신 사용자 승인에 따라 원인 보고에서 멈추지 않고 기존 MediaPipe 재검출·새 입력·이전 비교 연결까지 진행했다. 고정10창/8공개영상303프레임 계보검사에서 잘못된 마스크 계산0; 원검출없음과 정상 상반신영역제외 구분. 기존20프레임22미관측손에 같은원본 full/crop42호출,18손/372ROI관절/604입력특징 추가. 기존16154지원값·정상손/교차·영역외손내림 불변. 추가18손 전부RGB검토에서 큰좌우교환/반대손중복 새사례없음(정밀손가락정답미검증).
기존4사례×3참고/12쌍39구간 중11새geometry·28저장재사용. 원DTW경로/양축30560분모 유지: 비교지원24206→24670(+464),미비교6354→5890. 남은미비교에순종4912포함;고정창내정상영역제외확인이나4912전체원인전수확인아님,전부검출오류로집계금지. 원지원연결차이값 exact,구학습shape/position점수baseline-only. 원classifier재추론/DTW/학습0. 인식·정렬·교육정확도향상아님.
자체31/계보8229/새입력source7182·math5522/bridge33749 PASS. 검산기native-vs-ROI schema/JSON실패보존후검산기수정,추론재실행0. 같은Codex분리검산≠독립사람정답. 4미추가손·6미지원관절 유지. P066q68↔P063r58다른교대시점대응은별도남은문제. CPU내부선택처리5.13초는8영상20프레임실험합계,사용자전체대기시간아님.
RECOVERY/BRIDGE 완료잠금없음. RESULT_KO.txt/FINAL_REPORT_FACTS/RECOVERY_ADOPTION/COMPLETION/review 참조. 새HTML없음·8800은이전보고서. C/T3F·운영·개인/멘토·원본·과거동결보존. 관련일반후속은단계마다재승인없이검증·수정후보고한다.

''',9:'''### 2026-11-01 다음 단일 작업 — 같은 주요 동작 단계의 대응 근거 준비
누락보완과기존비교연결완료,새관측/39결과재추론·재채점금지. 남은핵심은P066q68↔P063r58같은교대단계불일치이며원좌표/정렬/실제행동차이를구분해야한다. 현재공개자료의앞·중간·뒤양손관계/이동방향/손모양과개별참고를유지해대응후보를검토하고,같은영상의알려진속도변환·반복/짧은변화·교차에서단계보존기준을먼저동결한다. 낮은거리/동작생략을성공으로삼지않고미관측분모보존. 새실행전NEXT_PLAN_KO/RESIDUAL_PLAN_REVIEW와완료상태확인;정렬진실근거없으면학습정답전용X. 새모델검색/단어특정임계값순회/개인자료/운영교체없음. 최신사용자승인일반후속은재질문불필요하나현재후속정렬실행은아직없음.

'''}
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text('utf-8-sig')
for section,body in sections.items():
    m=re.search(r'^## '+str(section)+r'\.[^\n]*\n',text,re.M);assert m
    text=text[:m.end()]+'\n'+body+text[m.end():].lstrip('\n\r')
p.write_text(text,encoding='utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('''# WordsInSilence 최신 인수인계

현재 word100_observation_lineage_repair_20261101 완료. 사용자는 발견한 관련 보완도 자율 계획/실행/검증 후 보고하도록 승인. 같은 일반 후속 재승인질문X. C/T3F·운영웹/DTW·원본/과거동결보존,사용자/멘토촬영접근·학습·튜닝·선택금지. 다운로드/외부전송/카메라/Git없음.

실제 완료: 기존10창/8공개영상303프레임 계보검사→잘못된마스크결함0→원미검출선택 MediaPipe재관측20프레임42호출→18손372점604입력지원추가(원16154exact)→기존39구간11새기하/28재사용 연결. 양축30560기회그대로 지원24206→24670,미비교6354→5890. 5890중순종미지원4912포함;고정창에서정상영역제외확인했으나4912전체원인전수확인아님. 새학습/분류/DTW없음;인식/정렬/교육정확도미측정. 구shape/position점수는baseline-only.

검증: 자체31/계보8229/source7182/math5522/bridge33749PASS. 20실행frame+18추가손 모두원본RGB대조;큰좌우교환/중복새사례없음,미세손가락정확도미검증. 최초검산기schema606불일치+JSON실패보존후검산기만수정. 완료추론/결과재실행금지. CURRENT_EXECUTION COMPLETE/active없음,RECOVERY_RUNNING/BRIDGE_RUNNING없음. 세에이전트별도검산은독립수어사람정답아님.

남은4손:P048f74/P063f57,f72/P066f71. P072f70 6joint지원안됨. 강제채움금지. P066q68↔P063r58서로다른손높이시점대응별도남음. 다음은앞뒤단계대응의원본근거·평가기준준비(아직실행X),현재단위NEXT_PLAN_KO/RESIDUAL_PLAN_REVIEW참조. 낮은거리로시점일치·정오주장X. 새로운모델/가중치순회X.

RESULT_KO.txt/FINAL_REPORT_FACTS.json/RECOVERY_ADOPTION.json/BRIDGE_MANIFEST.json/COMPLETION.json이현완료근거. 새HTML없음,8800은이전hold보고서(이번결과아님). 실행전최신§8§9·STATE해당키·CURRENT_EXECUTION부터읽고중복실행하지않는다.
''',encoding='utf-8')
h=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
with h.open('a',encoding='utf-8') as out:out.write('''

2026-11-01 후속 — 관측 누락 진단에서 실제 재관측·비교 연결까지
사용자 요청: 관련 문제가 발견되면 진단 보고만 하지 말고 필요한 후속 보완도 계획·실행·검증한 뒤 보고한다.
원인: 고정10창303프레임에서 원검출없음과 정상상반신영역제외가 섞여 있었다. 기존계약과 다른 마스크 버그는 없었다. 상반신범위를 넓혀 빈칸을 지우는 대신 원래 범위 안의 통째로 빠진 손만 기존 MediaPipe로 재관측했다.
실제 증분: 원전체+손주변42호출/20프레임으로22미관측손 중18추가;원16154특징불변/604특징지원증가. 원본20프레임·추가18손을모두확인했고큰좌우교환·중복새사례없었다. 미세손가락정답을보증하지는않는다.
후속 연결: 기존4사례12쌍39구간 중11새기하/28저장재사용,원시간경로/30560분모불변. 비교지원24206→24670(+464). 정상손내림영역제외를포함하므로미비교5890개를모두오류라고부르면안된다. 기존학습shape/position출력은새입력을본것으로바꾸지않았다.
검증과 한계: 계보8229/source7182/math5522/bridge33749 별도검산통과. 최초검산기native/ROI혼동·정수직렬화오류는보존후검산기만수정,추론재실행없음. 같은Codex검산은독립사람정답아님. 4손/6관절미확인,교대동작시점대응문제는남아있다. 단어/정렬/교육정확도향상을측정한시험아니다. 기존C/T3F·운영·사용자/멘토자료보존,추가학습0.
다음 판단: 관측보강이시간정렬문제를저절로고치지않음을확인. 다음은관측된앞뒤손관계와이동흐름으로같은주요단계를대응시켰다는근거를먼저확보한다. 목표는여전히설명·개별참고로중요한동작차이와허용차이를구분하는피드백이다.
근거: research/front_keyphase_v1/ops/word100_observation_lineage_repair_20261101/RESULT_KO.txt, FINAL_REPORT_FACTS.json, review/
''')
files=['RESULT_KO.txt','FINAL_REPORT_FACTS.json','NEXT_PLAN_KO.txt','RECOVERY_ADOPTION.json','RECOVERY_RESULT_MANIFEST.json','BRIDGE_MANIFEST.json','CURRENT_EXECUTION.json',
       'review/RECOVERY_VISUAL_REVIEW.json','review/RECOVERY_SOURCE_VERIFICATION.json','review/RECOVERY_MATH_VERIFICATION_FINAL.json','review/BRIDGE_VERIFICATION.json','review/REPORT_REVIEW.json','review/RESIDUAL_PLAN_REVIEW.json']
pins={str(U/f):sha(U/f) for f in files}
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',ROOT/'research/front_keyphase_v1/ops/STATE.json',h]:pins[str(p)]=sha(p)
write(U/'COMPLETION.json',{'status':'COMPLETE_RESEARCH_OBSERVATION_RECOVERY_AND_GEOMETRY_BRIDGE','at':now,'pins':pins,
 'protected_frozen_pins_verified':True,'original_support_preserved':True,'no_active_execution':True,
 'classifiers_and_production_unchanged':True,'all_error_cases_solved':False,
 'next_alignment_work_executed':False,'scope_limits':'No fine-finger ground truth or instructional/recognition accuracy gain established.'})
print('COMPLETE: saved outputs verified, authoritative docs synchronized, no active recovery/bridge lock')
