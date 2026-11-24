from common import *
from datetime import datetime,timezone,timedelta

def main():
 ev=read(U/'EVALUATION_COMPLETE.json');decision=read(U/'PUBLIC_DECISION.json');verification=read(U/'PUBLIC_VERIFICATION.json');private=read(U/'PRIVATE_POSTHOC.json');cfg=read(U/'FROZEN.json')
 assert sha(U/'PUBLIC_DECISION.json')==verification['public_decision_sha256']
 before=ev['models']['original']['by_variant']['normal'];after=ev['models']['finetune_42']['by_variant']['normal']
 support={}
 for role in ['fit','development']:
  rr=[r for r in read(U/'READY.json')['rows'] if r['role']==role]
  support[role]=dict(clips=len(rr),new_review_clips=sum(bool(r['pairs']) for r in rr),pairs=sum(len(r['pairs']) for r in rr),orders=sum(len(r['orders']) for r in rr),shape_cells=sum(int(np.load(r['path'])['direct_ym'].sum()) for r in rr),stage_points=sum(int((np.load(r['path'])['stage_gold']>=0).sum()) for r in rr))
 seed_summary={str(s):{task:ev['models'][f'finetune_{s}']['by_variant']['normal'][task] for task in before} for s in cfg['seeds']}
 result=dict(status=decision['status'],scope='Actual additional supervised Conv+GRU fine-tuning + public comparison; offline candidate only',
  support=support,primary_comparison={task:dict(before=before[task],after=after[task]) for task in before},seed_summary=seed_summary,
  core_direction_breakdown=decision['public_breakdown'],latency={n:ev['models'][n]['latency'] for n in ['original','finetune_42']},
  private_posthoc=private,public_verification_count=verification['count'],pretrain_checks=read(U/'PRETRAIN_CHECK.json')['count'],
  training_runs=6,training_seconds=sum(read(p)['seconds'] for p in (U/'runs').glob('*/COMPLETE.json')),
  learning_architecture='RTMW280 -> normalized projection -> temporal convolutions -> bidirectional GRU -> shape/stage + pair-direction/order readouts; existing guide vectors and train-only DTW prototype retained',
  description_connection='13 existing guide IDs/descriptions explicitly linked to sparse RGB stage anchors. Keyword vectors/learned embeddings, not an LLM understanding arbitrary prose.',
  private_training=False,private_tuning=False,private_selection=False,web_changed=False,tcn_changed=False,download=False,
  limitations=['4 words only; 12 previously exposed DEV clips, not new independent test.','Sparse Codex RGB judgments, not expert-reviewed language correctness.','Directions and order use reviewed endpoints: automatic event finding not proven.','Reverse-main controls are engineering interventions, not real learner-error accuracy.','Observation masks withheld all no-evidence claims; no missing frames removed from score denominator.','Semantic no-distinct-change labels allow natural drift; not exact zero velocity.','New head latent differs from deployed EvidenceHeads; old phase head cannot be reused without adaptation/validation.'],
  next_single_action=decision['next'])
 write(U/'RESULT.json',result)
 text='''이번 작업: 설명과 영상의 연결 보강 → 시간 합성곱+GRU 추가 학습 → 공개 비교 → 최근 촬영 사후 확인

결론
실제로 추가 학습까지 끝냈다. 손모양과 동작 단계에서 작은 개선이 있었지만, 중요한 팔 쓸기 방향을 놓치는 문제가 남았다. 따라서 새 모델로 웹을 교체하지 않았다. 기존 C·T3-F, 현재 TCN, 8814 화면과 판정부는 유지했다.

1. 무엇을 가르쳤나
- 순종·잘하다·조상·이집트의 기존 13개 동작 설명을 원본 영상의 확인 가능한 순간에 연결했다.
- 공개 24영상(학습 쪽 12 + 비교 쪽 12)을 RGB로 다시 보고 손모양 확인 항목420개, 동작 단계 시점173개, 방향 비교66개, 순서 비교24개를 기록했다.
- 이 중 학습에 추가한 것은 192개 손모양 항목·89시점·34방향·12순서다. 나머지는 비교용으로만 썼다. 같은 프레임의 여러 속성은 각각의 항목이며 서로 독립된 영상 수가 아니다.
- 시점 주석 5개를 바로잡았다. 그중 4개는 기존 DTW로 옮긴 추정 경계, 1개는 기존 원본 수행자 구간의 경계였다. 확인하지 않은 사이 프레임까지 정답으로 채우지는 않았다.
- 모서리 방향으로 보이는 손가락, 잘린 손, 정확한 접촉은 확정 정답에서 제외했다. ‘유지’는 자연스러운 작은 흔들림까지 0이라는 뜻이 아니라 별도의 뚜렷한 동작이 없다는 뜻이다.

2. 실제 학습 방식
전체 학습 입력은 수행자01~09의 36공개영상이고, 비교 입력은 다른 기록ID11~13의 12영상이다. 이번에 방향·순서 정답을 직접 보강한 학습 수행자는01·04·07이다. 따라서 방향 정답까지9명 모두 새로 붙인 것은 아니다.
현재 RTMW 관절전용 시간 합성곱+양방향GRU 모델을 복사했다. 기존 설명 연결·손모양·단계 학습과 학습자료끼리의 DTW 정렬 정보를 유지하면서, 두 시점 사이 방향과 순서를 읽는 작은 출력부를 붙였다. 학습용 역순 자료에는 정상 시간축을 강제로 씌우지 않았다.
비교: 기존 모델 / 기존 내부를 고정하고 새 출력부만 학습 / 내부 합성곱·GRU까지 미세조정. 뒤 두 조건을 시드42·43·44, 각각80회 학습했다. 처음 정한42를 대표로 보고하며 최고 점수 시드를 고르지 않았다. 합성곱과GRU 가중치가 실제 변한 것도 확인했다. 추가된 모델은142,103개 매개변수다.
설명 문장을 그대로 이해하는 언어모델이 아니다. 기존 설명의 키워드·동작ID와 직접 확인한 영상 시점을 연결해 배우는 신경망이다. RAG·SignRep RGB 실행·신규 다운로드는 없다.

3. 공개 비교 결과: 같은12영상, 같은 확인 항목
                       기존          추가 학습(고정 시드42)
손모양 확인 항목        221/228       223/228
동작 단계 시점          72/84         77/84
정상 순서의 비교쌍      10/12         12/12
방향 전체 비교쌍        26/32         28/32

이 숫자는 단어 인식률이나 수어 정답률이 아니다. 검토한 시점·속성·비교쌍에서의 모델 출력 일치 수다. 사람이 정해준 비교 시점을 사용했으므로 모델이 모든 주요 구간을 스스로 찾아 교정하는 성능과도 다르다. 방향의 기존 비교는 신체 기준 좌표 변화 계산이며, 기존 웹에서 모든 방향을 안내했던 것은 아니다.

중요한 반대 결과도 있다.
- 실제 움직이는 비교쌍만 보면20/21→19/21이었다. 특히 팔을 쓸어내리는3사례는 기존 좌표 계산3/3→새 방향 판정부1/3으로 떨어졌다. 관절값에 이동은 나타났지만 새 판정부가2사례를 유지로 분류했다. 전체 방향 점수의 상승은 ‘별도 동작 없는 구간’을 더 맞힌 효과가 포함된다.
- 동작 단계에서 보류하지 않고 맞힌 시점은71/84→66/84, 보류는3→12였다. 강제로 가장 높은 답을 고른 점수가 올라가도 실제로 자신 있게 안내할 범위가 늘었다고 할 수 없다.
- 3회 추가 학습의 새 손모양 항목은223~225/228, 단계는76~77/84. 기존에 쓰던 별도194개 속성 정답은190/194에서188~191/194로 흔들렸다.
- 손모양 오류 자체는 아직 남는다. 엄지·손끝 모으기·정확한 접촉·반복 생략을 교육적으로 판정하는 학습은 완성되지 않았다.

4. 속도·교차·관측 부족 시험
- 정상 상반신 범위와 느림/빠름을 따로 비교했다. 대표 모델 단계 일치는 상반신76/84, 느림76/84, 빠름76/84였다. 속도 차이에서 방향은26~28/32로 변했다. 모든 속도·정상 동작 보존을 보장하는 결과는 아니다.
- 조상의 정상 양손 교차에 해당하는 새 단계17시점은16→17개 일치했다. 이것만으로 모든 교차 손가락이 정확해졌다는 뜻은 아니다.
- 주요 구간을 실제로 시간 역순으로 만든 시험에서 단계52/84→64/84, 방향26/32→29/32였다. 정상/역순의 순서 비교 합계는22/24→24/24. 이는 합성 시험이며 실제 사용자의 오류 정확도가 아니다.
- 2프레임 관측 누락과 긴 관측 누락을 분리했다. 긴 누락에서는 모든 손모양·방향·순서 답을 보류했고, 해당 항목을 분모에서 빼지 않았다.
- 관절값이 준비된 뒤 신경망 계산은 대표141프레임 기준 CPU4스레드 중앙값 약4.23ms→5.87ms였다. 촬영·RTMW 추출·웹 표시까지의 총 지연이 아니다.

5. 방금 테스트한 기록
공개 결과와 적용 결론을 먼저 동결한 뒤, 마지막에 남아 있는 ‘이집트’ 촬영만 별도로 확인했다. 원본 영상과 RTMW143프레임의 시간표가 기존 화면 결과와 일치했다.
- 기존 화면: TCN1순위 이집트(모델 확률 약99.18%). ‘손 모으기’는 근거 부족, ‘손 벌리기’는 양손 펴기 특징이 비슷하다고 표시했다.
- 원본에서 보이는 주요 모으기·벌린 상태4시점은 기존/새 모델 모두 구분했다. 그 사이 이동 시점1개는 기존이 이동, 새 모델이 아직 모으기로 보아 늦게 바뀌었다(5시점 전체5→4). 새 방향 출력부는 벌리기와 그 뒤 유지 비교2개를 구분했다.
- 따라서 이 촬영에서 전반적인 개선이 확인됐다고 말하지 않는다. ‘손 모으기’가 회색인 문제도 단계 이름을 맞혔다고 해결된 것이 아니다. 손끝·엄지로 만든 실제 모양의 학습 근거가 아직 부족하다.
- 저장된 기존 결과의 촬영 종료→결과 계산은 약0.060초였다. 화면 표시와 촬영 중 관절 처리시간은 제외된다.
이 개인 기록은 학습·튜닝·후보 선택에 사용하지 않았다. 웹을 바꾸지 않았으므로 방금 결과가 새 모델 적용 결과인 것도 아니다.

6. 남은 핵심과 다음 작업
이번에 만든 정답·학습 코드·후보는 보존한다. 현재 웹의 검증된 관절 방향 계산과 불확실 보류도 유지한다.
다음은 ‘같은 손모양을 유지하면서 실제로 이동하는 부분’을 모델이 유지 상태로 뭉뚱그리지 않게 하는 일이다. 공개 학습 수행자의 해당 이동 시작·중간·끝 정답을 늘리고, 손모양과 별도로 신체에 대한 상대 이동량·방향이 출력부까지 전달되게 보강하는 것이 우선이다. 정상적인 작은 흔들림, 분명한 이동, 역방향, 짧은 생략을 분리해서 확인한다. 자동 구간 찾기와 기존 웹 단계 출력부의 재연결 검증도 필요하다. 이번 개인 촬영에 맞춘 경계 조정은 하지 않는다.

검증 범위
학습 전229개 검사와 저장 결과 별도 산술/해시834개 검사를 통과했다. 둘 다 같은Codex의 구현·별도 검산이며 독립 전문가의 정답 검증이 아니다. 기존에 이미 여러 번 본 개발 자료여서 새 독립 일반화 성능을 주장하지 않는다. 최초 프레임번호 오기·정답 없는 고정모델 배치 오류·사후 확인 모듈경로 오류는 기록 후 수정했으며 평가 점수로 학습설정은 변경하지 않았다.
새 HTML은 생략했다. 기존 웹 화면·C·T3-F·현재 TCN은 그대로다.
'''
 (U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
 # Final record checks happen before updating the authoritative summaries.
 assert private['public_decision_unchanged'] and not private['learning_used'] and not private['candidate_selection_used']
 for p,h in cfg['source_pins'].items():assert sha(p)==h
 for p,h in read(J/'FROZEN.json')['pins'].items():
  if 'training_C_' in p or 'T3F_' in p:assert sha(p)==h
 with (U/'RECOVERY_LOG.txt').open('a',encoding='utf-8') as f:f.write('Private posthoc first import lacked guard geometry search path; failed before inference. Added local module path; no parameter/config change. Public decision hash unchanged.\n')
 write(U/'FINAL_CHECK.json',dict(status='PASS',public_decision_sha256=sha(U/'PUBLIC_DECISION.json'),protected=True,private_no_training=True,
  report_sha256=sha(U/'RESULT_KO.txt'),result_sha256=sha(U/'RESULT.json'),web_changed=False,
  artifacts={str(U/f):sha(U/f) for f in ['ANNOTATIONS.json','READY.json','FROZEN.json','PRETRAIN_CHECK.json','TRAINING_COMPLETE.json','EVAL_FROZEN.json','EVALUATION_COMPLETE.json','PUBLIC_DECISION.json','PUBLIC_VERIFICATION.json','PRIVATE_POSTHOC.json']},
  code={str(p):sha(p) for p in U.glob('*.py')}))
 now=datetime.now(timezone(timedelta(hours=9))).isoformat();key=U.name
 status='''### 2026-11-09 설명·영상 정답 보강과 시간 합성곱+GRU 추가 학습 — 부분 개선·웹 미교체

word100_gru_semantic_finetune_20261109. 기존4단어36FIT(기록ID01~09)/12DEV(11~13) 유지. 원본RGB24영상 검토로 새 손모양420항목/173단계시점/66방향/24순서 연결. 학습에 쓴 직접보강분은12FIT영상(01/04/07)의192손모양/89시점/34방향/12순서. 기존 경계5개 수정. 키워드·가이드ID 연결이며 문장전체를 이해하는LLM/RAG아님. 고정내부+새출력부와 전체미세조정 각3seed80epoch실행, 실제Conv/GRU가중치변화확인. 42사전고정, 최상위seed선택없음.

대표 새DEV 손모양221/228→223/228, 단계72/84→77/84, 정상순서10/12→12/12. 방향전체26/32→28/32지만 실제이동20/21→19/21, 팔쓸기3/3→1/3으로회귀. 확신단계정답71/84→66/84·보류3→12. 실제역순합성단계52/84→64/84, 방향26/32→29/32. 관측전체누락은모두보류·분모유지. CPU관절준비후141프레임신경망약4.23→5.87ms, 총응답시간아님. 시점정답을준pair검사·기존노출DEV·같은Codex시각주석이므로자동교정/독립일반화/전문가검증주장금지.

공개결론동결뒤사용자승인최근이집트143프레임사후확인(학습·튜닝·선택0). 기존TCN이집트1순위, 주요모으기/벌린4시점양쪽일치, 이동1시점새모델지연(5→4), 방향2비교일치. 사용자전반개선주장없음. 손끝모으기정답부족미해결. 학습전229/별도산술·해시834확인. C·T3F/현재TCN/8814모델·코드유지. 새HTML없음. 근거RESULT_KO.txt/RESULT.json/PUBLIC_VERIFICATION.json/PRIVATE_POSTHOC.json/FINAL_CHECK.json. 최초오류보존. 모든학습평가완료, 중복재실행금지.
'''
 nexttext='''### 2026-11-09 추가 학습 비교 이후: 실제 이동을 유지로 놓치는 문제

8814는기존positive_direction적용판유지. 새후보는오프라인보존, 42대표고정. 다음단일작업은공개FIT의같은손모양에서움직이는시작·중간·끝감독정답을확대하고, 손모양과신체상대이동근거를출력부에서분리해팔쓸기등실제이동누락을줄이는것. 정상흔들림/속도/교차와역방향/짧은생략을함께고정비교. DEV점수나최근개인촬영으로규칙·가중치·후보선택금지. 자동구간찾기와새latent에대한웹단계판정부적응검증전교체금지. 손끝모으기/엄지/접촉·반복교육정오미완. 완료6학습/7조건공개비교/최근사용자사후검사재실행금지. 사용자·멘토학습튜닝선택금지, 기존C·T3F보존.
'''
 context=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';old=context.read_text(encoding='utf-8-sig')
 assert key not in old
 old=old.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n'+status,1)
 old=old.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n'+nexttext,1);context.write_text(old,encoding='utf-8')
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+status+'\n'+nexttext+'\n현재8814는사용자버튼으로만카메라개방. 최근확인state done/camera_thread_active false. 서버변경없음.\n',encoding='utf-8')
 state=read(O/'STATE.json');entry=dict(status=result['status'],report=str(U/'RESULT_KO.txt'),result=str(U/'RESULT.json'),verification=str(U/'FINAL_CHECK.json'),training=True,training_runs=6,web_changed=False,private_posthoc_only=True,private_learning=False,independent_generalization_claim=False)
 state={key:entry,**state};state['current_stage']=result['status'];state['next_single_action']=nexttext.split('\n\n',1)[1].strip();state['updated_at']=now;state['updated_date_kst']=now[:10];write(O/'STATE.json',state)
 assert key in read(O/'STATE.json');assert key in (ROOT/'NEW_CHAT_HANDOFF.md').read_text(encoding='utf-8')
 write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_REPORTED_OFFLINE_ONLY',time=now,active_process=None,training_complete=True,public_comparison_complete=True,private_posthoc_complete=True,web_changed=False))
 event('report_and_handoff_complete',status=result['status'])
if __name__=='__main__':main()
