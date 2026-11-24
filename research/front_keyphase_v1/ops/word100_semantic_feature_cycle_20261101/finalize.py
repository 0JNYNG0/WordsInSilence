"""Report the completed fixed trial without promoting it to product success."""
from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent; ROOT=U.parents[3];OPS=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
 f=read(U/'FROZEN.json');s=read(U/'SCORES.json');tr=read(U/'TRAINING.json');fb=read(U/'FEEDBACK_CORRECTED_EVENTS.json');diag=read(U/'FAILURE_DIAGNOSIS.json')
 for p,h in {**f['pins'],**f['source_pins'],**f['protected']}.items():assert sha(p)==h,p
 assert tr['status']=='COMPLETE_SIX_FIXED_RUNS' and len(tr['runs'])==6 and not (U/'TRAIN_RUNNING.json').exists()
 reviews=['CONTRACT_REVIEW_FINAL.json','FROZEN_VERIFICATION.json','SCORES_VERIFICATION.json','CHECKPOINT_VERIFICATION.json','FEEDBACK_CORRECTION_INDEPENDENT_REVIEW.json']
 for name in reviews:
  d=read(U/'review'/name);assert str(d['status']).startswith('PASS'),(name,d['status'])
 assert sha(U/'feedback.py')==fb['code_sha256']
 def metric(condition,role,family):return next(r for r in s['summary'] if r['condition']==condition and r['role']==role and r['family']==family)
 def fmt(r):return f"{r['mean']*100:.2f}% (3회 범위 {r['range'][0]*100:.2f}~{r['range'][1]*100:.2f}%)"
 dev='development'; held='new_feature_expression_evaluation';old='original_protected_baseline';joint='joint';split='protected_shape_split_features'
 oldd=metric(old,dev,'commonshape');oldh=metric(old,held,'commonshape');jd=metric(joint,dev,'commonshape');jh=metric(joint,held,'commonshape');nd=metric(split,dev,'new_features');nh=metric(split,held,'new_features')
 # Prespecified new4 criterion: both role means >.55, every per-target seed mean >.50.
 supported_new=True
 for role in [dev,held]:
  supported_new &= metric(split,role,'new_features')['mean']>.55
  for target in f['targets'][3:]:
   vals=[next(t for t in r['targets'] if t['target']==target)['balanced_accuracy'] for r in s['rows'] if r['condition']==split and r['role']==role and r['family']=='all']
   supported_new &= sum(vals)/len(vals)>.50
 assert not supported_new
 runtime=read(U/'HEAD_RUNTIME.json');timings=[r for r in runtime['rows'] if r['batch_windows']==3]
 txt=f'''주요 손 모양·관계·순서 학습/비교 결과 — 2026-11-01

결론
승인한 1~4단계의 제한 연구 작업(준비, 학습, 비교, 별도 검산)을 수행했다. 작은 흔들림 판정 때문에 확실한 손 모양 학습까지 막히는 구조는 분리했다. 2조건을 각3회, 총6회 실제 학습했다. 그러나 새 표현에서도 좋아진다는 조건은 통과하지 못했다. 새 모델은 운영에 채택하지 않았다. 수어 피드백 AI 전체가 완성된 상태가 아니다.

1. 무엇을 바꿨나
기존 소복 미세 흔들림/down/hold 미확인은 그대로 보존했다. 이를 모든 학습의 선행 관문으로 두지 않았다. 확실한 항목만 학습하고 모호한 항목만 손실에서 제외했다. 기존 실패를 성공으로 고친 것이 아니라 목표가 다른 새 계약을 세웠다.
공통 특징은 손 모양, 두 손의 위치 관계, 상태 변화, 주요 동작/전환/시작/끝, 관측된 순서로 구분했다. 사전 설명은 관련 특징을 고르는 근거이고, 영상에 안 보이는 손을 정답 모양으로 채우는 근거가 아니다. 소복 공식 허용 경계와 수어의 정확히2회 반복 규칙을 새로 만들지 않았다.

2. 실제 준비한 자료와 정답
기존 공개96영상의288중앙시점, 각13원본프레임 문맥을 재사용했다. 새 영상 다운로드/관절 추출/사용자 자료 사용은 없다. 학습60영상·180시점(기록ID1~3), 같은 표현 다른 사람 개발20영상·60시점(ID4), 이 판정부 학습에 없는8표현16영상·48시점(ID7~8)으로 분리했다.
7항목은 오른 검지만 펴짐, 오른/왼 네 손가락 펴짐, 오른/왼 네 손가락 굽힘, 오른손/왼손이 더 높음이다. 엄지는 이 수형 정답에 포함하지 않으므로 완전한 주먹·정밀 수형 전체를 맞춘다고 말하지 않는다. 알려진 정답1441셀/미확인575셀이다. 자료 지원 기준12창·6영상·4표현·3ID/양음성은 그대로 적용했다. 부족한 왼 검지 항목만 제외했다.
전체 움직임의 역할, 정확한 신체 접촉, 필수 반복 횟수는 새 학습 정답으로 만들지 않았다. 시간창의 중앙 손 모양을 맞추는 것과 전체 단어의 의미·순서를 이해하는 것은 다르다.

3. 어떻게 학습했나
기존 SignRep 특징과 관절·관측마스크를 입력으로 작은 시간별 판정부만 지도학습했다. SignRep 본체와 C/T3-F는 학습하지 않았다.
A: 7항목을 함께 새로 학습. B: 기존 손 모양3항목의 출력을 그대로 보호하고 새4항목만 학습. 각각seed42/43/44,160epoch, 고정최종epoch·판정0.5·AdamW lr0.001/wd0.01. 개발 점수로 모델/문턱/epoch를 선택하지 않았다.
B는 기존 학습된 경로를 추가 보존하므로 A와 총용량·과거학습노출이 다르다. 두 시스템 비교이며, 구조 분리 하나의 인과효과를 증명한 실험은 아니다.
원48영상의 저장 확률은 그대로 재사용했고 새48에 대해서만 고정된 예전 판정부를seed당1회 적용했다. 완성 결과 재추론은 없었다.

핵심 결과(양성과 음성을 동등하게 반영한 균형정확도; 단어 인식률 아님)
- 기존 공통 손모양3, 같은 표현 다른 사람: {fmt(oldd)}
- A 공통 손모양3, 같은 표현 다른 사람: {fmt(jd)}, 기존 대비 {(jd['mean']-oldd['mean'])*100:+.2f}%p
- 기존 공통 손모양3, 판정부가 학습하지 않은 표현: {fmt(oldh)}
- A 공통 손모양3, 판정부가 학습하지 않은 표현: {fmt(jh)}, 기존 대비 {(jh['mean']-oldh['mean'])*100:+.2f}%p
- B의 기존3항목은 전체 확률 자체가 원본과 동일, 정확도 상승/하락0.
- B 새4항목, 같은 표현 다른 사람: {fmt(nd)}
- B 새4항목, 학습하지 않은 표현: {fmt(nh)}
- A 새4항목, 학습하지 않은 표현: {fmt(metric(joint,held,'new_features'))}
학습에 있던 표현을 다른 사람이 수행할 때는 어느 정도 구분하지만 다른 표현에도 같은 특징을 적용하는 능력이 부족하다. 학습자료의 새4항목 점수는100%인 반면 새표현은 약54%라, 일반화가 병목이다. 다만 RGB 의존·자료량·특징표현 중 어느 원인인지 이번 결과만으로 단정하지 않는다.

실패를 구체적으로 확인했다
새표현의 오른 네 손가락 굽힘 양성8시점은 양쪽 시스템 모든seed가 놓쳤다. 그중7시점은 관련 손가락 각도가 입력에 존재했다. B의 왼손이 더 높음 양성12시점도 모두 놓쳤고9시점은 양손 손바닥 중심 입력이 있었다. 따라서 모든 실패를 미검출 탓으로 설명할 수 없다. 원본 입력·마스크·라벨 순서·저장 점수·학습 노출 검산에서는 오류를 찾지 못했다.
다만 오른 검지 새표현 양성은4시점/한표현, 개발 왼손 펼침 양성은3시점뿐이다. 적은 시점을 독립적인 많은 사람의 증거로 부풀리지 않는다.

4. 피드백 비교와 실제 수정
개별 참고를 평균으로 뭉치지 않고 각각 유지했다. 어느 손의 어느 항목을 비교했는지, 관측이 없는 항목은 무엇인지 따로 남기는 연결부를 구현했다. 정답 라벨의 미확인 제외율과 실제 시스템의 관측 신뢰도는 서로 다른 값이며, 이번 모델이 자동으로 불확실성을 모두 알아낸 것은 아니다.
초기 시험에서 같은 모양의 두 관측 구간을 중간 미확인 경계 너머로 합치는 문제가 발견됐다. 최초 결과/코드를 보존하고 구간을 각각 유지하도록 수정했다. 속도만 바꾼16개 시험은 모두 같은 순서로 유지, 관측 구간 하나 삭제78개와 역순16개는 모두 차이를 보존했다. 이것은 저장된 관측 순서에 가한 기술 시험이며 실제 학습자 오류를100% 잡았다는 뜻이 아니다. 인위적 구간 수와 수어의 필수 반복 횟수도 다르다.
실제 공개4표현·16영상·36쌍에서는 준비/끝까지 표시한 자료와 주요 부분만 표시한 자료가 섞여 있었다. 원자료의 명시된 단계 종류를 사용해 같은 주요 동작 범위만 별도로 연결했다. 개발12쌍에서 참고 주요구간48개 중42개에 동일종류 연결후보가 있었지만15연결은 반복구간 정체가 애매했다.6쌍의 구간 목록은 같고6쌍은 미확정 차이였다. 예컨대P072는 동일 움직임의 시작을 준비로 적었고 다른 참고는 주요동작으로 적은 범위 차이가 남았다. 이를 사용자의 수어 오류라고 판정하지 않는다.
연속 영상 전체를 충분히 비교할 근거는36쌍 모두 부족하다. 보류해서 경고가 줄어든 것을 성능 향상으로 세지 않았다. 새로 학습한7항목 모델을 포함한 최종 교육 피드백 정확도는 아직 검증하지 못했다.

정상 손·교차·짧은 변화
기존3출력은 B에서 완전히 보존했다. 사전 기록된 교차 하위집단의 개발2시점은 알려진12항목에서 새 오류가 없었지만, 표본이 작고 한 범주뿐이며 새표현 교차 표본은0이다. 모든 정상 교차를 보장하지 않는다. 모양변화 하위집단의 새표현12시점에서는 새4항목이seed별32개 확정셀 중8/9/8개를 틀렸다. 정상 흔들림에 대한 독립적인 허용 정답은 아직 없다. 미세 이동을 아예 무시하거나 짧은 중요한 모양 변화를 지우지 않았다.

처리시간
CPU4스레드, 이미 추출된 입력의3개 시간창을 새 작은 판정부에 넣는 계산만 약{min(r['median_ms'] for r in timings):.3f}~{max(r['median_ms'] for r in timings):.3f}ms(6run 각각의 중앙값)였다. 영상 읽기·MediaPipe·SignRep 추출·기존 동결 경로·DTW·화면 출력은 제외되어 전체 사용자 대기시간으로 쓸 수 없다. 이번은 작은 판정부 학습이며 영상 특징 준비 시간까지 짧다는 뜻이 아니다.

검증/보존
별도 Codex 문맥의 자료·계약·저장 산술·체크포인트·피드백 검산을 받았다. 독립 한국수어 전문가 정답 검증과는 다르다. 960epoch 실제 FIT시점ID 노출로그와6체크포인트 연결을 확인했다. 기존 보호32개 및 동결 입력/점수 SHA 보존. 원C·T3F/운영웹·DTW 변경0, 개인·멘토자료 접근/학습/튜닝/선택0, 새 다운로드0. 이 평가자료는 과거 TCN TRAIN 계보이므로 프로젝트 전체에서 한 번도 보지 않은 최종시험으로 부르지 않는다.

다음 단일 목표
새 모델의 출력 수를 늘리는 것보다, 같은 손 모양·양손 관계를 처음 보는 표현에서도 구분하도록 만드는 것이 먼저다. 기존 FIT 안에서 표현을 통째로 제외하는 검증을 구성하고, 손 자체의 모양 근거와 몸 전체/단어 문맥을 분리한 입력이 실패를 줄이는지 사전에 조건을 고정해 비교한다. 높낮이처럼 직접 측정 가능한 관계는 검출 근거가 충분할 때의 관측값과 학습출력을 구분한다. 현재 새표현 결과는 이미 확인했으므로 다음 최종평가에 재사용해 튜닝하지 않는다.
동시에 실제 피드백 비교에 쓰는 공개 참고들의 시작/주요/끝 표시 범위를 공통 기준으로 연결하되, 반복/미확인 경계를 삭제하지 않는다. 손 모양·관계의 새표현 전이와 동일범위 연결이 확인된 뒤 실제 중요한 누락·순서차이와 정상변이의 최종 피드백을 평가한다. 이번 결과로 검증되지 않은 전표현 의미 규칙이나 교육적 허용 경계를 학습됐다고 선언하지 않는다.

명확한 범용 성능 개선이 확인되지 않아 HTML은 만들지 않았다.
상세: SCORES.json, FAILURE_DIAGNOSIS.json, FEEDBACK_RESULT_KO.txt, FEEDBACK_CORRECTED_EVENTS.json, review/ 검산 파일.
'''
 with (U/'RESULT_KO.txt').open('x',encoding='utf8') as out:out.write(txt)
 evidence=[U/n for n in ['FROZEN.json','READY.json','PREPARATION.json','TRAINING.json','SCORES.json','HEAD_RUNTIME.json','FAILURE_DIAGNOSIS.json','FEEDBACK_CORRECTED_EVENTS.json','FEEDBACK_RESULT_KO.txt','RESULT_KO.txt']]+[U/'review'/n for n in reviews]
 result=dict(status='COMPLETE_FIXED_RESEARCH_CYCLE_NOT_ADOPTED',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),steps={'1':'Common-feature/evidence contract completed; mandatory semantic rules remain unsupported','2':'Supported7-target public dataset verified, no new truth invented','3':'Six fixed runs and saved comparisons verified','4':'Feedback technical comparison/correction verified; real learned end-to-end educational validation incomplete'},
  training_runs=6,epochs_per_run=160,private_data_access=False,production_model_change=False,backbone_updates=0,word_accuracy_measured=False,html_created=False,semantic_product_goal_completed=False,
  original_shape_preserved=True,new4_prespecified_gate_passed=bool(supported_new),joint_common3_dev_delta_pp=(jd['mean']-oldd['mean'])*100,joint_common3_held_delta_pp=(jh['mean']-oldh['mean'])*100,
  new4_dev=nd,new4_held=nh,feedback_native_complete_pairs=0,feedback_native_pairs=36,feedback_corrected_technical_tests=110,feedback_educational_accuracy=None,
  evidence={str(p):sha(p) for p in evidence},protected_files=len(f['protected']),next_action='Prospective expression-held-out feature generalization diagnosis in FIT; reference phase-scope consistency before actual learner feedback claims; no new external model.')
 write(U/'COMPLETION.json',result)
 (U/'CURRENT_EXECUTION.json').write_text(json.dumps(dict(status=result['status'],active_execution=None,training_exec_session=81841,training_exit_code=0,benchmark_exec_session=91373,benchmark_exit_code=0,at=result['at']),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 statepath=OPS/'STATE.json';state=read(statepath);state['current_stage']=result['status'];state['next_single_action']=result['next_action'];state['word100_semantic_feature_cycle_20261101']=result
 statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';text=p.read_text(encoding='utf-8-sig')
 summary='''

### 2026-11-01 주요 특징 6회 학습·피드백 비교 완료 — 새 표현 일반화 부족으로 미채택

`word100_semantic_feature_cycle_20261101`: 승인1~4 연구작업 수행. 공개96영상/288시점 정적7표적 준비, FIT60(ID1~3)/DEV20(ID4)/특징새표현16(ID7~8). 미세down/hold 완비를 전체관문에서 분리, 이전FAIL 불변. joint7/기존3출력보호+새4 각3seed160epoch 완료. 공통3 균형정확도 DEV89.89→96.14%(+6.25%p), 새표현81.15→76.99%(−4.17%p). 보호형 원3확률 동일, 새4 DEV90.72%/새표현53.96%로 동결기준FAIL. 오른굽힘 새표현8양성 전seed실패(7관측좌표존재), 왼높음12양성 전seed실패(9좌표존재)로 미검출만 원인아님. 체크포인트/960epoch 노출/저장점수 별도검산PASS. 피드백은 미확인경계 너머 동일종류 구간 병합문제 수정,16속도/78삭제/16역순 기술검증. 실제36쌍은 전체연속근거 부족(완전비교0), 주요동작범위 별도연결에서도 단계표시 차이·반복대응모호함 남음. 교육오답/최종학습모델 피드백성능 검증X. C·T3F/운영/개인/원본보존,HTML없음. 작은판정부 실행만0.37~0.38ms/3창,전체대기시간X. COMPLETION/RESULT_KO.txt 참조. active실행없음,모델미채택,제품목표미완료.
'''
 text=text.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인'+summary,1)
 nxt='''

### 2026-11-01 다음 단일 작업 — 단어 문맥과 공통 특징의 일반화를 분리

완료된 고정6run/검산/피드백 수정결과 재실행 금지. 다음 목표는 새 표현에도 손모양·양손관계를 구분하는 것. 기존FIT 안 표현통째제외 검증으로 손자체/상대위치 근거와 전체단어문맥 의존을 비교할 사전계약부터. 이번확인한 새표현평가를 새미사용시험으로 재사용해튜닝X. 추가새모델검색/미세hold반대사례완비로돌아가지않음. 피드백용 공개참고는 명시된준비/주요/끝범위를 일치시키되 반복구간·unknown경계 보존. 실제관측/시간추정/미확인과직접측정가능관계/학습출력구분. 공통특징 일반화와 비교범위 정합 확인 후 의미있는누락·순서오류/정상변이 피드백 검증. 운영교체·개인자료학습X.
'''
 text=text.replace('## 9. 다음 단일 작업','## 9. 다음 단일 작업'+nxt,1);p.write_text(text,encoding='utf8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('''# WordsInSilence 최신 인수인계

현재 word100_semantic_feature_cycle_20261101, COMPLETE_FIXED_RESEARCH_CYCLE_NOT_ADOPTED. 사용자 승인1~4 연구 사이클 완료; 제품/의미피드백 목표는 미완성. active 실행없음. COMPLETION.json/RESULT_KO.txt가 근거. 6회실제학습/저장점수/체크포인트/노출검산 완료이므로 재실행X.

미세 down/hold 준비를 전체관문에서 분리. 기존공개96영상288시점의 확실한7항목 재사용(1441known/575unknown). FIT60 ID1~3/DEV20 ID4/특징새표현16 ID7~8; 전부 과거TCN TRAIN계보. joint7 vs보호old3+new4 각seed42/43/44·160epoch. old48확률재사용,new48만원head seed당1회. 작은판정부만학습,SignRep/C/T3F고정.

공통3 균형정확도 DEV89.89→96.14%(+6.25%p)이나 새표현81.15→76.99%(−4.17%p). 보호형old3확률동일. new4 DEV90.72%(88.13~92.38)/새표현53.96%(50.36~58.50),동결채택기준FAIL. 오른굽힘8양성전부놓침(7좌표관측),왼높음12양성전부놓침(9좌표관측),미검출만으로설명불가. 새로운표현으로특징전이가병목. 학습/DEV 점수로조건선택X,완료점수재추론X.

피드백은 같은모양의 별도구간을unknown경계너머 합친문제수정. 첫결과와코드보존,교정110기술시험(16시간변형/78구간삭제/16역순)통과. 실제36쌍전체연속판정0/36으로보류,오경고개선주장X. 원source role로주요동작만비교해DEV12쌍중6부분목록일치/6미확정차이,참고48anchor중42후보연결·15반복대응모호. 준비/주요라벨경계차이는학습자오류아님. 새7모델실제교육피드백성능미검증. FEEDBACK_CORRECTED_EVENTS/FEEDBACK_RESULT_KO/별도검산참조.

구현root/semantic_model/semantic_feedback,별도자료·산술semantic_data_audit 및 피드백상호검산. 동일Codex 별도문맥≠독립전문가정답. C·T3F/운영웹·DTW/원본/이전FAIL보존,보호32SHA,개인·멘토자료접근/학습/튜닝/선택X,새다운로드X,HTML생략. 작은추가head CPU3창0.37~0.38ms만측정,전체대기시간X.

다음: 기존FIT에서표현통째제외 검증·손자체/상대위치와단어문맥분리 사전계약,같은공개참고의주요범위정합. 이미본평가자료새평가인척튜닝X. 새거대모델/미세hold전수감사로되돌아가지않음. root만§8§9/STATE/HANDOFF동기화.
''',encoding='utf8')
 print(json.dumps({'status':result['status'],'runs':6,'new4_gate':supported_new,'report':str(U/'RESULT_KO.txt')},ensure_ascii=False))
if __name__=='__main__':main()
