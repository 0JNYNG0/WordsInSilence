from pathlib import Path
import json,hashlib,datetime
U=Path(__file__).resolve().parent;V=U/'proposal_adapter';O=U.parent;ROOT=O.parents[2]
def rd(p):return json.loads(Path(p).read_text('utf-8-sig'))
def wr(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
first=rd(U/'RESULT.json');final=rd(V/'RESULT.json');va=rd(U/'VERIFICATION.json');vb=rd(V/'VERIFICATION.json');assert va['passed'] and vb['passed'];assert va['result_sha256']==sha(U/'RESULT.json') and vb['result_sha256']==sha(V/'RESULT.json')
for p,h in rd(U/'FROZEN.json')['protections'].items():assert sha(p)==h,p
allnew=rd(V/'DIAGNOSTIC.json');miss=[]
for r in allnew:
 if r['seed']!=42 or r['group']!='reserved_id12' or r['kind']!='normal':continue
 d=r['manual_direction'];a=r['automatic']['combined'];old=r['automatic']['baseline']
 if not d['eligible'] or d['prediction']!=r['truth']['direction']:
  miss.append(dict(clip_id=r['clip_id'],truth=r['truth'],manual_direction=d,automatic=a,source='Original RGB range was supplied directly; this isolates direction judgment from automatic interval detection.'))
wr(U/'FINAL_CAUSAL_AUDIT.json',dict(manual_direction_failures=miss,combined_causes='A short repeated motion can be unmatched by interval linkage and also classified as hold even when given its reviewed interval. Do not fix only one side.',role_regression='Changing action-family probabilities lost the extra historical direction cue in the combined mode. Keep old role output in the retained baseline.',completion='Endpoint or absence alone is not evidence of performed full action or genuine omission. All six final-event omissions remain uncertain.',scope='Same Codex visual review and saved arithmetic, no expert educational-correctness claim.'))
text='''이번 작업 결과 — 2026-11-11

결론
승인한 1~4단계와, 검사 중 확인한 실제 검출 조각에 대한 추가 학습·검증까지 마쳤다.
동작 구간을 찾는 데 작은 개선이 있었다. 그러나 방향·순서·누락을 정확히 설명하는 피드백이 완성된 것은 아니다. 현재 웹과 C·T3-F는 그대로 유지했다.

무엇을 했나
1. 공개 영상으로 정답을 보강했다.
   공통 학습은 기존100단어899영상을 그대로 유지했다. 같은899영상 안에서 세부 경계 정답이 있는 영상은18→24개, 직접 확인한 움직임 구간은29→39개가 됐다. 영상899개가 전부 세부 정답을 가진다는 뜻은 아니다.
   추가6개 표현에서 손의 실제 방향전환·유지 구간을 RGB로 확인했다. 중간의 작은 왕복을 흔들림으로 지우지 않았다. 가려졌거나 검토하지 않은 순간은 추정 정답으로 채우지 않았다.
2. 공통 연결부를 만들었다.
   '한 행동의 조각 / 시작과 끝을 포함한 한 행동 / 여러 움직임이 섞인 구간 / 확인된 유지·대기'를 구분한다. 조각 쌍은 같은 행동의 이어짐인지, 다음 이동 구간인지, 다른 축의 행동인지 구분한다.
   이때 다음 이동 구간은 한 바퀴 전체를 뜻하지 않는다. 위→아래와 아래→위는 따로 보존하지만 이를 곧바로 두 번 반복이라고 세지 않는다.
   단어ID로 예외를 넣지 않았다. 손모양이나 방향이 정답과 다르다는 이유로 시도한 행동을 후보에서 제거하지 않는다. 기존 후보도 보존한다. 한 검출 구간을 두 번의 정답 구간에 중복 연결하지 않는다.
3. 학습과 원인별 비교를 했다.
   시간 합성곱+GRU 공통 인코더를100단어899영상으로 계속 학습하면서, 경계·행동 역할·구간 연결 판정부를 함께 학습했다. 미리 정한3개 초기값으로 진행했다.
   기존 방식 / 구간 보완만 / 행동 역할 보완만 / 둘을 결합한 방식으로 비교했다. 같은 학습 모델의 부품별 비교이며, 네 구조를 각각 독립 학습한 비교는 아니다.
   첫 결과에서 부족함이 남아, 실제 학습용 검출기가 만든 불완전한 조각으로 작은 연결 판정부를 추가 학습했다. 이때100단어 공통 인코더와 기존 모양·방향 판정부는 고정했다. 새 자료나 사용자 영상으로 전체 모델을 갈아끼우지 않았다.
4. 공개 자료에서 검증했다.
   기존78조건은 회귀 시험으로 재사용했다. 기존 TCN·DTW 판정 결과를 읽어 비교했으며 현재 운영 모델을 교체하지 않았다.
   추가10개 원본(다른6표현의기록ID11, 기존4표현의ID15)을 확인했다. 추가 보완 뒤에는 별도로 남겨 둔 다른6표현의ID12도 확인했다. 기록ID로 분리했으며 외형으로 사람을 추정하지 않았다.
   전체 역재생은 편집 연결점 없는 방향 진단이다. 실제 학습자의 자연스러운 오류와 같다고 주장하지 않는다. 옛 TCN/개발 평가 노출이 있었으므로 프로젝트 전체에서 처음 보는 자료라는 주장도 하지 않는다.

이전과 현재 — 마지막에 따로 확인한 공개6영상
영상6개 중5개에서 직접 확인한10개의 움직임 구간을 비교했다. 나머지1개는 이 축에서 유지 동작을 확인하는 사례다.
                          이전 기준      이번 결합 후보
정상 영상에서 구간 찾기       8/10             9/10
시작·끝이 확인 범위 안에 듦    1/10             2/10
정상 영상의 방향 판정         6/10             6/10
전체 역재생에서 구간 찾기     7/10             8/10
전체 역재생에서 방향 판정     3/10             4/10
구간 찾기는 직접 확인한 구간과 절반 이상 겹치는 기준이다. 손가락 정답률이나 전체 피드백 정확도가 아니다. 원본10구간을 뒤집어 추가10조건을 만든 것으로,20개의 독립 영상도 아니다. 위 구간·방향 수치는 추가 연결 판정부의3개 초기값에서 같았다.

유지된 것
정상12조건과 속도 차이24조건의 기존 표시가 유지됐고 새 잘못된 경고는0개였다. 교차 동작의 정상3·속도6조건도 이 안에 포함된다.
기존 손모양 변경·방향 반전·둘의 결합9조건의 경고는 유지됐다. 기존 모양·방향 근거와 TCN후보·DTW정렬 결과는 바꾸지 않았다.
관측 부족18조건을 정상으로 승격한 경우는0개였다. 마지막 동작을 지운6조건도 정상으로 넘기지 않았으나 여전히 '판단 보류'다.
확인한 유지 구간은 추가10원본의원본/역재생18조건과 마지막6원본의원본/역재생12조건에서 새 완성 이동 구간으로 잘못 선택하지 않았다. 이것만으로 모든 흔들림과 의도된 움직임을 구분한다고 말할 수는 없다.

아직 남은 핵심 문제
1. 구간만의 문제가 아니었다.
   마지막 공개 검사에서 짧은 왕복4구간은 직접 올바른 구간을 지정해 줘도 기존 방향 판정부가 모두 '유지'라고 답했다. RGB와 관절의 상대 위치에는 변화가 보인다. 따라서 연결부만 고쳐서는 이4개를 해결할 수 없다.
   그중 한 구간은 자동 구간 찾기도 실패했다. 구간 문제와 행동 판단 문제가 실제로 겹친 사례다.
2. 행동 역할을 함께 바꾼 이득은 확인하지 못했다.
   과거 주요 동작 역순9조건에서 구간 보완만은 주 초기값42에서 경고4→5였지만, 역할까지 바꾸면4개였다. 추가 학습의다른2초기값에서도 추가 경고가 재현되지 않았다. 예전 실험의5/9를 넘어선 성과도 아니다.
3. 반대로 움직였는데 비슷하다고 나온 잘하다1조건이 그대로 남았다.
   도착 모습과 일부 구간만으로 전체 진행이 맞았다고 판단하지 않도록 보완이 더 필요하다.
4. 누락과 관측 부족을 확실히 구분하는 피드백은 아직 없다.
   마지막 행동 생략6조건을 '잘했다'고 하지는 않지만, 어느 행동이 빠졌는지를 확정해서 안내하지는 못했다.

처리시간
새 연결부와 피드백 보충 계산은 중앙값 약0.04초,상위95% 약0.08초였다. 이미 준비된 관절 특징에서 잰 시간이다. RTMW추출·고정특징인코더·TCN·DTW·브라우저를 포함한 촬영 후 전체 대기시간이 아니다. 실제 웹캠 속도 개선으로 보고하지 않는다.

적용 판단
구간 연결의 작은 개선은 연구 후보로 보관한다. 행동 역할까지 바꾼 후보와 전체 피드백 기본 교체는 채택하지 않는다. 정상은 유지됐지만 방향·누락 피드백의 충분하고 안정적인 개선이 없기 때문이다. HTML은 추가로 만들지 않았다.

다음 한 작업
이번에 확인한 짧은 실제 이동과 정상 유지·미세 흔들림을 구분하는 방향 학습을 우선한다. 손모양·기존 방향 판정은 기준으로 보존하고, 학습용 공개24영상의 확인된39이동·22유지 구간을 이용해 부족한 '이동/유지' 구분을 보강한다.52개 대기구간을 전부 의미 있는 유지 동작 정답으로 섞지 않는다.
그 뒤 직접 구간과 자동 구간을 나눠 비교한다. 직접 구간에서도 틀리면 행동 판단을, 직접 구간은 맞는데 자동 구간만 틀리면 연결부를 고친다. 두 문제가 겹치면 두 개선을 결합해 확인한다. 평가 점수에 맞춘 단어별 예외·경계 조정은 하지 않는다.
통과 기준은 정상/속도/교차의 새 잘못된 경고0, 기존 모양·방향 유지, 짧은 이동 검출 개선, 보류로만 숨긴 개선이 아닐 것, 새로운 공개 기록에서 재현될 것이다. 순서·실제 누락 설명은 이 기준과 별개로 계속 검증한다.

보존과 검증
사용자·멘토 촬영 접근0,학습·튜닝·선택 사용0. 다운로드·외부전송·Git·카메라 사용0. C·T3-F·운영웹 해시 유지.
직접 RGB 검토와 학습은 Codex가 수행했다. 별도 표준라이브러리 검산으로 첫 비교11465항목, 추가 보완7119항목을 확인했다. 같은 Codex의 별도 계산 검사이며 독립 전문가/농인 교육자 정답 검증이 아니다.
추가 학습 준비 중 JSON 저장에 NumPy 정수가 섞인 오류1회는 최초 실패 기록을 보존하고 정수 변환으로 수정했다. 해당 오류는 학습 시작 전이었고 완료된 학습 결과를 다시 만들지 않았다.
'''
(U/'RESULT_KO.txt').write_text(text,'utf-8')
nexttext='''다음 단일 작업 — 짧은 실제 움직임과 정상 유지 구별
근거: reserved ID12의WORD1290 네 이동 구간은 정확한 구간을 줘도 frozen direction이hold를 반환한다. 그중 한 구간은 자동 연결도 실패한다. 반대로 새 연결부는WORD0500의누락구간을 찾았지만 잘못 잡힌 다른 이동 구간의 손실이 남아 정상 방향 합계는6/10으로 같다.
1. FIT24영상의39이동과22유지 구간을 분리해 검토한다. 몸 전체의 미세 흔들림, 손목 이동, 손 모양 변경, 반복의 방향 전환을 섞지 않는다. 주석이 없는 조합은 추정 정답으로 만들지 않는다.
2. 기존 방향·손모양 판정부를 보존한 별도 경량 이동/유지 보완부를 학습한다.100단어 공통 인코더는 유지한다. 현재 주석은 전100단어의상세정답이 아님을 유지한다. 단어ID를 정답 지름길로 넣지 않는다.
3. 직접 확인한 구간 / 자동 구간을 각각 비교한다. 방향만 / 구간만 / 결합 조건에서 손실과 이득을 함께 센다. 틀린 움직임을 후보에서 지우거나, 반복을 합쳐 지우거나, 관측 부족을 완료로 올리지 않는다.
4. 이전 자료는 개발·회귀로 표시하고, 사용 전 예약한 별도 공개 기록에서 검증한다. normal/속도/교차 새 경고0, 모양·방향 유지, 이동검출 및 실제피드백 개선, 보류율 악화 없음,3초기값 재현을 함께 본다. 실패하면 원인을 기록하고 웹 기본에 넣지 않는다.
준비·마무리를 완벽히 세분화하는 일보다 실제 교정 피드백을 막는 이동/유지와 행동 역할의 연결을 우선한다. 기존 C·T3-F·웹 유지, 사용자·멘토자료 금지, 추가 다운로드 없음.
이번 단위의 실행은 완료했으며 이 다음 학습은 아직 시작하지 않았다.
'''
(U/'NEXT_PLAN_KO.txt').write_text(nexttext,'utf-8')
summary='승인4단계+실제검출조각 추가학습 완료.100단어899유지,FIT세부경계18→24영상/29→39이동.첫10공개평가후추가6표현ID12별도확인:정상구간8/10→9/10,방향6/10유지;전체역재생구간7/10→8/10,방향3/10→4/10(3head초기값동일). 정상12/속도24/교차유지,모양·방향기존9경고유지,관측부족18·누락6정상승격0. 역할결합은추가이득없고짧은4왕복을직접구간에서도유지로판단;잘하다역순1비슷함·누락설명미해결.연결부약.04초만측정,전체대기아님.웹/C·T3F불변·개인자료0.검산11465+7119 PASS(동일Codex,전문가검증아님).HTML생략.'
nxt='다음은 짧은 실제 이동과 정상 유지·미세 흔들림 구분 학습. FIT39이동/22유지 근거부터 보강하고 직접구간·자동구간을 분리, 방향만/연결만/결합 및 정상·속도·교차 회귀 검증. 현재단위 완료, 다음 학습 미착수. 상세 '+str(U/'NEXT_PLAN_KO.txt')
wr(U/'FINAL_DECISION.json',dict(status='COMPLETE_LIMITED_COMPONENT_GAIN_NO_DEFAULT_ADOPTION',summary=summary,report=str(U/'RESULT_KO.txt'),next=str(U/'NEXT_PLAN_KO.txt'),first_result_sha256=sha(U/'RESULT.json'),adapter_result_sha256=sha(V/'RESULT.json'),verification_counts=[va['checks'],vb['checks']],public_only=True,private_data=0,html_created=False,operational_change=False))
wr(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_LIMITED_COMPONENT_GAIN_NO_DEFAULT_ADOPTION',pid=None,next_training_started=False))
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8')
for sec,body in [('## 8. 현재 상태',summary),('## 9. 다음 단일 작업',nxt)]:
 i=t.index('\n',t.index(sec))+1;t=t[:i]+'\n### 2026-11-11 구간·행동 결합 비교 및 추가 보완 완료\n\n'+body+'\n근거 '+str(U/'RESULT_KO.txt')+'\n\n'+t[i:]
p.write_text(t,'utf-8');state=rd(O/'STATE.json');state.update(current_stage=U.name,next_single_action=nxt);state[U.name]=dict(status='COMPLETE_LIMITED_COMPONENT_GAIN_NO_DEFAULT_ADOPTION',summary=summary,next=nxt,result=str(U/'RESULT_KO.txt'),verification=str(U/'VERIFICATION.json'),additional_verification=str(V/'VERIFICATION.json'),execution=str(U/'CURRENT_EXECUTION.json'));wr(O/'STATE.json',state)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n다음 한 작업: '+nxt+'\n\n실행 중 학습·평가 프로세스 없음. 기존 서버/카메라 조작 없음. 결과 출력은 연구 후보이며 웹 기본은 유지.\n첫 비교와 보완 후 결과/초기 실패 기록은 읽기 전용으로 유지한다. 완료된 평가 재실행 없이 저장 결과를 사용한다.\n현재 단위 '+str(U)+'\n','utf-8')
history=ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt'
with history.open('a',encoding='utf-8') as f:f.write('\n\n[2026-11-11] 구간·역할 결합 비교와 실제 검출 조각 학습\n'+summary+'\n진행 이유: 한 번에 한 원인만 고쳐 원점처럼 보이는 문제를 검토하고, 구간만/역할만/결합을 비교했다. 실제 검출 조각과 학습용으로 나눈 조각의 차이도 FIT에서 검사해 추가학습했다. 일반화 확인을위해기존4표현외6표현,학습03과평가11/12를분리했다.검증자료가공통TCN/개발에서노출된이력은숨기지않았다.\n결론: 연결만의 문제가 아니라짧은이동을유지로보는행동판정부문제가겹친다는직접구간근거를확보했다.목표는변경하지않고그부분을다음우선과제로정했다.\n근거 '+str(U/'RESULT_KO.txt')+'\n')
print(summary)
