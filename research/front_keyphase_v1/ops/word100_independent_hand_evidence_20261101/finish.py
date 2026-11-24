import sys
sys.dont_write_bytecode=True
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from pipeline import UNIT, OPS, read, write, digest

ROOT=OPS.parents[2]
now=datetime.now().astimezone().isoformat()
reports=['SELFTEST.json','review/MATH_SELFTEST_FINAL.json','review/PREPARATION_REVIEW_FINAL.json',
    'review/PREEXECUTION_REVIEW.json','EVALUATION.json','review/MATH_RESULT_VERIFICATION.json',
    'review/RESULT_VERIFICATION.json','review/RESULT_SUMMARY_VERIFICATION.json']
for name in reports: assert read(UNIT/name)['status'].startswith('PASS'),name
frozen=read(UNIT/'FROZEN.json')
pins={**frozen['pins'],**frozen['input_pins']}
for p,h in pins.items(): assert digest(p)==h,p
assert not (UNIT/'RUNNING.json').exists()
manifest=read(UNIT/'RESULT_MANIFEST.json')
for r in manifest['results']:assert digest(r['path'])==r['sha256'],r['id']
e=read(UNIT/'EVALUATION.json'); counts=Counter()
for row in e['axis_counts']:
    if row['kind']=='natural': counts.update({k:row[k] for k in ('possible','direct_complete','partial_shared','not_comparable')})
assert dict(counts)==dict(possible=87328,direct_complete=68769,partial_shared=128,not_comparable=18431)
short=Counter(sum(v!=0 for v in t['edit']['axis_delta_degrees']) for t in read(UNIT/'INPUTS.json')['technical'] if t['edit']['condition']=='short_axis_change')
summary={'natural_parents':117,'technical_cases':810,'clips':16,'words':4,'natural_both_axis_cells':dict(counts),
    'technical_counts':e['technical_conditions'],'short_change_actual_lengths':dict(short),
    'existing_position_records':585,'existing_shape_parent_seed_links':351,
    'math_checks':read(UNIT/'review/MATH_RESULT_VERIFICATION.json')['checks'],
    'source_checks':read(UNIT/'review/RESULT_VERIFICATION.json')['checks'],
    'summary_checks':read(UNIT/'review/RESULT_SUMMARY_VERIFICATION.json')['checks'],
    'new_training':0,'new_model_inference':0,'new_dtw':0,'educational_accuracy':None,
    'word_recognition_accuracy_change':None,'real_3d_hand_rotation_accuracy':None,
    'operating_changed':False,'user_mentor_used':False,'end_to_end_latency':None}
write(UNIT/'FINAL_VERIFICATION.json',{'status':'PASS','at':now,'frozen_pins_checked':len(pins),'result_hashes_checked':927,
    'review_reports':{n:digest(UNIT/n) for n in reports},'summary':summary,
    'review_authorship':'Separate Codex agents wrote arithmetic and source/summary verifiers. Root ran their completed scripts after agent usage limits. This is code/arithmetic crosscheck, not an independent human KSL evaluation.',
    'preserved_failures':[
        {'path':'review/MATH_SELFTEST.json','cause':'Verifier empty-path JSON array shape handling; verifier-only correction, producer unchanged; FINAL38 cases/17746checks PASS'},
        {'path':'review/PREPARATION_REVIEW.json','cause':'Final source hash changed while root was adding freeze gates; input checks passed; final stable-code pin supplementary audit4417PASS'}]})

report=f'''손 위치·모양·방향을 따로 비교하는 기능 — 검증 결과
완료: {now}

이번에 바꾼 점
손목이 같은 자리에 있다는 이유로 손 전체가 같은 동작이라고 보면 안 됩니다.
그래서 같은 동작 구간에 네 가지 근거를 나란히 연결했습니다.
① 손이 어디로 움직였는지: 이전에 계산한 이동 경로 결과를 그대로 사용합니다.
② 손가락이 어떤 모양인지: 각 손의 14개 각도를 각각 비교합니다.
③ 손이 화면에서 어느 쪽을 향하는지: 손목에서 가운데 손가락 시작점으로 향하는 방향을 봅니다.
④ 손과 팔의 방향 관계: 손만 방향을 바꾼 것과 팔과 함께 방향을 바꾼 것을 따로 기록합니다.
한 부분이 비슷해도 다른 부분의 차이를 지우지 않습니다. 보이지 않는 값은 0점 차이나 정답으로 바꾸지 않습니다.

무엇으로 확인했나
기존 공개 한국수어 4표현·16영상·36영상쌍의 117개 주요 동작 범위를 그대로 사용했습니다.
이전 이동 결과585개, 기존 손모양3seed 연결351개를 해시 확인하여 연결했습니다. 모델이나 DTW를 다시 실행하지 않았습니다.
별도로45개 원동작 구간×양손×9조건=810개 수치시험을 했습니다. 이는 같은 저장 시점끼리 비교하는 기술시험이며, 810개의 실제 잘못된 수어 영상이 아닙니다.

주요 결과
- 손목은 그대로 두고 손 방향만30도 바꾸기:90기회 중 관측 가능한79건에서 차이 유지,11건은 근거 없어 보류.
- 손목은 그대로 두고 검지 시작 관절 각도만18도 바꾸기:79건에서 차이 유지,11건 보류. 다른 손가락 각도는 보존했습니다.
- 손과 팔을 함께30도 돌리기:관측된 손 방향 변화는 남고 손-팔 상대 방향은 그대로 유지했습니다. 실제 손목 굽힘 판정 시험은 아닙니다.
- 한 각도의 관측만 지우기:그 각도만 비교불가로 남고 원래 관측된 다른13개는 유지했습니다. 한 손 전체를 통째로 버리지 않았습니다.
- 짧은 방향 변화:관측 가능한79건에서 최대 차이를 남겼습니다. 원래 범위를 줄이지 않아 5프레임82건/4프레임6건/3프레임2건입니다.
- 작은0.5도 흔들림도 수치에는 남지만, 이를 잘못된 수어라는 경고로 바꾸지 않았습니다. 정상 허용 경계를 새로 정한 시험은 아닙니다.

공개 영상에서 남은 빈칸
양쪽 시간축의 특징 비교기회87,328개를 모두 유지했습니다. 이 수는 중복 참고·시간·특징을 포함하며 영상 수나 정확도가 아닙니다.
직접 비교68,769개 / 일부 상대 관측이 빠진128개 / 비교불가18,431개입니다.
영상에는 손이 보여도 저장된 관절 정보가 비어 있는 사례가 있었습니다. 예:조상 P065의48번째 원본 프레임에서 왼손은 보이지만 저장된 왼손축·14각도는 미관측입니다. 이번에는 보완 검출을 실행하거나 빈값을 채우지 않았습니다.
정상 참고 영상끼리도 관측된 각도와 방향 차이가 있었습니다. 이 차이를 잘못된 동작이라고 판정하지 않았습니다.

검증
모듈 자체356검사, 별도 산술 합성38사례17,746검사, 공개 산술439,282검사, 출처·범위110,073검사, 요약109,465검사 통과.
별도 Codex 에이전트가 검산기를 작성했고 사용량 제한 후 root가 저장된 검산기를 실행했습니다. 독립적인 사람의 수어 정답 검토와는 다릅니다.
처음 검산기의 빈배열 처리 문제와, 준비 검산 중 코드 핀이 바뀐 기록은 보존했습니다. 수정 후 검산 통과; 원입력·공개 비교 결과 수정이나 재실행은 없었습니다.
927개 비교 계산 자체 합계7.15초, 파일 저장 등을 포함한 실행26.35초입니다. 한 사용자 영상의 전체 피드백 대기시간을 측정한 값이 아닙니다.

아직 해결하지 못한 점
현재 방향은 정면 화면의2차원 방향입니다. 주먹을 팔 축 주위로 돌리는3차원 회전, 실제 손목 굽힘을 모두 정확히 알아냈다는 뜻은 아닙니다.
기존 시간 정렬을 그대로 사용하므로 잘못 맞은 동작 시점도 이번에 고친 것은 아닙니다.
이번 연결의 사전 설명은 표현형 검증 전인 참고 후보입니다. 단어 이름이 같다고 영상의 필수 동작·허용 범위를 확정하지 않습니다.
단어 인식률이나 실제 교육 피드백 정확도가 높아졌다는 증거는 아직 없습니다. 정상 손 교차도 좌우 정보와 관측을 그대로 보존했으나, 실제 교차 동작의 교육 정오를 검증한 것은 아닙니다.

기존 C·T3-F와 운영 웹·DTW는 그대로입니다. 새 학습0, 모델 추론0, DTW 재계산0. 사용자·멘토 촬영은 사용하지 않았습니다.
이번에는 새로운 HTML을 만들지 않았습니다. 8800 화면은 이전 단계 보고서 그대로입니다.
다음 작업은 NEXT_PLAN_KO.txt에 정리했습니다.
'''
write(UNIT/'COMPLETION.json',{'status':'COMPLETE_RESEARCH_READOUT_VALIDATED_NOT_DEPLOYED','at':now,'summary':summary,
    'report':str(UNIT/'RESULT_KO.txt'),'active':None,'verification':str(UNIT/'FINAL_VERIFICATION.json'),
    'implementation':'root pipeline; semantic_feedback pure module; semantic_model separate arithmetic; semantic_data_audit source and summaries',
    'performance_claim':'independent feature evidence preservation only; educational accuracy unmeasured'})
(UNIT/'RESULT_KO.txt').write_text(report,encoding='utf-8')
next_plan='''다음 한 작업: 설명과 실제 주요 동작에 연결된 차이 후보를 점검한다.

목표는 모든 좌표를 똑같이 만드는 것이 아니라, 중요한 손모양·신체관계·움직임·순서가 유지되는지 설명할 수 있게 하는 것이다.
1. 이번에 묶은117부모 근거와 기존 공개 시간주석을 재사용해, 실제 영상에서 확인된 동작 설명과 사전의 아직 미확인인 설명 후보를 구분한다. 같은 이름이라는 이유로 표현형/필수성/허용 경계를 자동 확정하지 않는다.
2. 각 주요 동작에서 어떤 특징이 관측되고 어떤 것이 부족한지 보여주는 공통 판정부를 먼저 연결한다. 손위치 일치나 TCN확률로 모양/방향 차이를 지우지 않는다. 자연참고의 차이는 교정이 아닌 차이후보로만 남긴다.
3. 기존 정상 참고끼리 비교하여 특정 사람·표현만 문제없다고 만드는 규칙 없이, 시간정렬이 잘못된 차이/관측부족/실제 보이는 동작차이를 구분한다. 기존 개별참고와 시간 범위를 보존하고 원본을 확인할 사례를 결과전에 정한다. 동작중 짧은유지를 모두필수동작으로강요하지 않는다.
4. 실제오류와정상변이를 구분할 정답지원이 마련된 공통특징만 이후학습 대상으로 정한다. 2D축이보지못하는 주먹축회전·3D손목굽힘을 가짜정답으로 채우지 않는다. 새모델순회·새거리식순회로 넘어가지 않는다.

이번완료927계산/과거학습/DTW재실행금지. 새학습 필요성은 관측+정답지원으로 판단하며 자료부족을 숨기려고학습하지 않는다.
C/T3F·운영·원본·사용자/멘토자료보존. 다음 작업 아직 미실행. 근거있는피드백이목표이며 모든수어/교육정확도 보장아님.
'''
(UNIT/'NEXT_PLAN_KO.txt').write_text(next_plan,encoding='utf-8')
current={'phase':'COMPLETE','active':None,'run_session':15483,'run_exit':0,'results':927,
    'evaluation_exit':0,'math_session':98911,'math_exit':0,'source_session':3896,'source_exit':0,'summary_exit':0,
    'report_server_session':35320,'report_server_note':'8800 earlier hold-stage report; no new HTML',
    'updated':now,'agent_usage_limit_recovery':'Root ran separately authored ready verifiers; no result rerun'}
(UNIT/'CURRENT_EXECUTION.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')

statepath=OPS/'STATE.json';state=read(statepath)
state['updated_date_kst']=now;state['current_stage']='INDEPENDENT_HAND_EVIDENCE_COMPLETE_NOT_DEPLOYED'
state[UNIT.name]={'status':'COMPLETE_RESEARCH_READOUT_VALIDATED_NOT_DEPLOYED','at':now,'active':None,'unit':str(UNIT),
    'report':str(UNIT/'RESULT_KO.txt'),'summary':summary,'next_single_action':next_plan,'new_training':0,'new_inference':0,'production_changed':False,'user_mentor_used':False}
statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=context.read_text('utf-8-sig')
summary_text='''### 2026-11-01 독립 손 특징 연결 완료 — 위치와 모양·방향의 차이 별도 보존
`word100_independent_hand_evidence_20261101` 완료. 기존4표현16영상36쌍117부모에585위치결과/351shape-seed출처를보존연결하고 각손14각도/2D손축/팔상대방향을독립비교. 양축87328기회=direct68769/partial128/unknown18431 보존. 45부모×2손×9=810기술대조, 손목동일+방향30도/검지각18도/최대5frame변화 각79관측차이유지+11보류. 공통팔손회전relative0/단일각누락시기존나머지각보존. 실제언어오류/3D회전판별/교육정확도개선시험아님.
자체356/별도합성17746/공개산술439282/출처110073/요약109465검산PASS. 최초검산기empty-shape FAIL과동시코드핀변경FAIL보존후정정검산. 공개결과재실행0. 에이전트사용량제한후root가별도작성검산기실행. 새학습/추론/DTW0,C/T3F·운영·개인/멘토·원본보존. 정상교차정보불변은확인하되교차교육정오미검증. 계산exec15483종료0,HTML없음/8800이전단계그대로.
관측근거빈칸/기존정렬모호함/사전설명표현형미확인/2D회전한계남음. 손위치같음으로전체동작정상승격금지. 결과/검산/다음계획참조.

'''
s=s.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+summary_text,1)
s=s.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n### 2026-11-01 다음 작업 — 주요 동작 설명과 독립 차이 근거 연결\n이번117자연부모/810기술대조연결검증완료,재계산X. 다음은관측된물리동작설명과사전미확인후보를구분하고,각주요동작의모양/관계/이동/순서차이후보를기존시간주석·다중참고에연결한다. 정상참고차이를교육오답으로승격하거나미관측을일치로처리하지않는다. 실제차이/정렬차이/관측부족을결과전정한원본사례로확인. 자료지원없는3D회전/허용기준정답으로학습강행금지. 새모델·거리식순회X. NEXT_PLAN_KO.txt의네단계,아직미실행. C/T3F·운영·개인/멘토·원본보존.\n\n',1)
context.write_text(s,encoding='utf-8')
(ROOT/'NEW_CHAT_HANDOFF.md').write_text(f'''# WordsInSilence 최신 인수인계

현재 `{UNIT.name}` 완료. exec15483계산종료0/검산완료, 실행중계산없음. 8800은이전hold단계읽기전용서버exec35320 그대로. CURRENT_EXECUTION 확인후중복실행금지.
기존4표현16영상36쌍117부모에585위치/351shape-seed출처그대로연결. 각손14각도/2D손축/팔상대방향독립비교, 양축87328기회=68769direct/128partial/18431unknown. 810기술feature편집대조(45부모×2손×9)는실제수어오류아님. 위치동일+방향/한각도/짧은변화각79관측에서차이유지,11보류. 원본교차ID/좌표/마스크/시점범위보존,정상교차의교육정오성능미검증.
자체356/합성17746/공개산술439282/출처110073/요약109465PASS. 별도에이전트작성검산기를사용량제한후root실행. 독립사람정답아님. 최초검산기빈배열/동시핀변경FAIL보존,FINAL정정검산PASS;공개결과불변/재실행없음.
새학습/추론/DTW0. C/T3F·운영·원본·사용자/멘토미사용/보존. 새다운로드/외부전송/카메라/Git없음. 새HTML없음.
현재2D손축은손목→중지MCP방향이며손바닥법선/3D주먹축회전/직접손목굽힘분류아님. 화면에손보여도저장missing인공개사례존재. 기존global정렬모호함과사전설명표현형미확인남음. 위치일치나단어확률로다른특징차이/미관측을덮지말것.
다음단일작업:주요동작의검증된영상설명/미확인사전후보를구분하고모양·관계·이동·순서근거를연결,실제차이/시점정렬차이/관측부족을분리하는공통피드백후보검증. NEXT_PLAN_KO.txt참조,아직미실행. 이번927/과거정렬·모델재실행X,새모델/거리식순회X. 정답지원이갖춰진특징만추후학습검토.
근거 {UNIT}: RESULT_KO.txt, COMPLETION.json, FINAL_VERIFICATION.json, EVALUATION.json, review각검산. 목표변천사 Docs/PROJECT_DIRECTION_HISTORY_KO.txt.
''',encoding='utf-8')
history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
with history.open('a',encoding='utf-8') as f:
    f.write('\n\n2026-11-01 같은 위치와 같은 손동작을 구분하는 독립 특징 연결\n사용자강조대로손목이작게흔들리는것과주먹회전/손목꺾임은별개다. 이전이동비교를유지하면서14손가락각도·화면손방향·팔상대방향을같은주요동작에따로연결했다.117자연부모/810기술대조완료;관측된방향/각도변형79건씩보존하고11미관측은보류.한각도미관측때문다른13각도삭제안함. 실제3D회전·언어정오·교정정확도향상아니며새학습/운영적용없음. 남은병목은기존정렬의모호함,보이는손도놓친관측,사전설명과영상표현형연결및실제허용/오류정답이다. 다음은거리식순회보다이분리근거를주요동작설명에연결해피드백의주장을근거범위로제한한다. 목표는변하지않고코드보존검증과교육적정오를분리해기록한다. 근거 '+str(UNIT/'RESULT_KO.txt')+'\n')
print(json.dumps({'complete':True,'summary':summary,'report':str(UNIT/'RESULT_KO.txt')},ensure_ascii=False))
