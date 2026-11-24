from prepare import *

def main():
    assert read(U/'VERIFICATION.json')['status']=='PASS'
    references=[
      dict(title='Including Handshape in Sign Language Recognition',url='https://aclanthology.org/2023.findings-emnlp.198.pdf',evidence='Table3: PHOENIX14T WER baseline23.69 vs dual handshape encoder22.07; auxiliary handshape supervision can help.',limit='German continuous recognition, not Korean learning feedback; architecture and auxiliary supervision differ, no promise that any extra digits help.'),
      dict(title='SignBERT+: Hand-model-aware Self-supervised Pre-training for Sign Language Understanding',url='https://arxiv.org/pdf/2305.04868',evidence='Table10 increases pretraining data fraction with common downstream setup; MSASL1000 per-instance top1 53.31 at0% vs62.42 at100%.',limit='Data-volume experiment, not isolated vocabulary-size experiment; uses its own architecture/prior. Does not establish 100-to3000 KSL feedback gain.'),
      dict(title='Improving Handshape Representations for Sign Language Processing: A Graph Neural Network Approach',url='https://aclanthology.org/2025.emnlp-main.1483.pdf',evidence='Table1 dual static/temporal GNN46.07% vs individual GNN30.01/31.00% on37 ASL shapes. Explicit shape and temporal modeling can complement each other.',limit='Different dataset/task, no local model tested here; subtle handshape families remain hard. Not an instruction to replace current TCN+GRU.'),
      dict(title='Fingerspelling within Sign Language Translation',url='https://aclanthology.org/2025.naacl-long.19/',evidence='ASL translation study reports mixed effects from adding fingerspelling recognition data.',limit='Transfer is conditional; not evidence that KSL fingerspelling cannot help.')]
    write(U/'RESEARCH_EVIDENCE.json',dict(references=references,local_prior=dict(report=str(B/'RESULT_KO.txt'),effect='Shared100 vs same fine-label-only: extra8word shape119-120/133 vs121-122/133; original stage did not reliably improve.',scope='Historical development, paired3seeds; not an expanded vocabulary experiment.'),decision='Supported direction: shared phonological/shape attributes plus varied public signs, with separate temporal supervision. Unsupported assertion: more labels or more digits automatically improve feedback.'))
    review=[
      dict(word='WORD2636',label='오(숫자5)',observed='FIT01 both sampled frames show thumb extended and four long fingers folded.',priority='First auxiliary contrast: index-folded example absent from prior2/4/6.',admitted_for_training=False),
      dict(word='WORD2634',label='삼(숫자3)',observed='FIT01 shows three long fingers extended; useful candidate for contrast against two extended fingers.',priority='Review ring/little and thumb in larger multi-record crops before individual labels.',admitted_for_training=False),
      dict(word='WORD2632',label='일(숫자1)',observed='FIT01 index extended, other long fingers folded; thumb configuration differs from6.',priority='Thumb-placement contrast after visibility review; not a new meaning for ordinary sign.',admitted_for_training=False),
      dict(word='WORD2631',label='영(숫자0)',observed='FIT01 rounded hand configuration.',priority='Requires curved/bent state; do not coerce into straight-vs-fully-folded binary. Exact touch is not established.',admitted_for_training=False),
      dict(word='WORD2641',label='십(숫자10)',observed='FIT01 thumb/index form a rounded projected configuration while other fingers extend.',priority='Useful later for curved finger vs full folding; contact cannot be asserted from these previews.',admitted_for_training=False),
      dict(word='WORD2640',label='구(숫자9)',observed='FIT01 extended fingers shown with different palm orientation.',priority='Orientation/foreshortening contrast; orientation must remain a separate meaningful attribute.',admitted_for_training=False)]
    write(U/'CANDIDATE_REVIEW.json',dict(rows=review,reviewer='same Codex visual inspection of two RGB frames each; not independent expert',source=str(U/'EXPANSION_SOURCE_PREVIEW.json'),all_source_fit_only=True,not_whole_video_or_multi_person_truth=True))
    schema=dict(status='SPECIFICATION_ONLY_NOT_WEB_BEHAVIOR',goal='More precise corrective wording only when independently supported target and observed finger attributes are available.',guide_fields=['word_id','variant_id','source_url_or_verified_annotation','source_clause','stage_id','hand','required_attribute','allowed_variants'],observation_fields=['native_time_interval','hand','finger','state_extended_bent_folded_or_unknown','visibility','direct_or_temporal_inferred','validated_attribute_scope','evidence_frames'],output_fields=['status_supported_difference_unknown','highlight_hand_and_fingers','reference_clause','one_short_instruction','evidence_interval'],rules=['No numeric/letter class name used as ordinary-word meaning.','Coarse four-finger classification is not evidence for which single finger is wrong.','No absent hand, hidden thumb, unknown target variant or uncertain stage converted into error.','Frame prediction must be linked to the relevant action interval before a correction.','Thumb position, finger spread, projected proximity and true physical contact are separate labels.','Softmax/sigmoid confidence is not an educational correctness percentage.','Use deterministic source-grounded wording; an LLM is not required for these sentences.'],examples=[dict(kind='future conditional example, not current user/model result',target='right index extended; middle/ring folded in the verified action',observed='middle/ring visibly extended and this feature/interval validated',text='이 동작에서는 오른손 검지는 펴고, 중지와 약지는 접어 보세요.'),dict(kind='uncertainty wording example',observed='thumb hidden',text='엄지가 가려져 모양을 확인하기 어려워요. 손을 조금 돌려 다시 보여 주세요.'),dict(kind='coarse-only wording',observed='four-finger open/closed is supported but individual cause is not',text='이 동작에서는 오른손을 펴 주세요. 지금은 손가락이 접힌 모습으로 보여요.')],current_limit='This probe only evaluates three coarse attributes. Finger-specific ordinary-word feedback is not yet validated or deployed.')
    write(U/'FEEDBACK_CONTRACT.json',schema)
    plan=dict(status='EVIDENCE_BASED_NEXT_UNIT_READY',main100_preserved=True,steps=[
      dict(id='A',task='Use existing ordinary-word FIT and new numeric5/3/1 contrasts to prepare individual-finger visible anchors, then check multiple recorded IDs. Include flexed vs fully folded/straight only when RGB supports distinction.',why='Current numeric anchors lack index-folded examples; ordinary evaluation is coarse only.',gate='Per-attribute positive and negative examples across more than one expression and recorded ID; explicit unknown masks; no prediction-derived truth.'),
      dict(id='B',task='Freeze ordinary-word individual-finger evaluation before training; preserve record-ID split and source hashes. Also retain complete normal/speed/crossing/direction/order tests.',why='503 coarse cells cannot establish individual-finger correction quality.',gate='Do not evaluate numeric semantics as general-word transfer; no same-frame neighbors across person split.'),
      dict(id='C',task='Train a shared shape feature branch and integrate with current temporal-convolution+GRU only in an isolated candidate, comparing same ordinary-data schedule with/without auxiliary attributes.',why='Separate shape from global direction, location and timing; keep current broad100 representation.',gate='No regression in existing shape/direction/order; no higher false corrections on normal; uncertainty retained; end-to-end runtime checked after integration.'),
      dict(id='D',task='Connect verified observed finger difference to the source explanation clause and native time range; emit one short instruction with highlight.',why='More detailed text must follow more detailed evidence, not merely longer generated prose.',gate='Incorrect/unknown/model-unsupported fingers do not get a corrective statement; no default deployment before full validation.')],vocabulary_expansion=dict(recommendation='Expand beyond100 selectively when verified public signs fill a measured feature gap; not bulk3000 word-only learning now.',selection_axes=['finger flexion combinations','thumb relative position','spread and curved shapes','same shape while still and moving','two-hand/crossing and hand identities','body-relative location and direction','repetition/order'],stop_rule='Do not add more near-duplicate word names if coverage/held-record generalization do not improve.',product_scope='Common-feature training vocabulary can exceed web feedback vocabulary; TCN classes and supported game words are separate choices.'),preserved_research_resume='C7 body-context15 FIT losses /G2REWORK unchanged; this shape probe does not clear motion/hold gate.')
    write(U/'NEXT_PLAN.json',plan)
    rr=read(U/'RESULTS.json')['runs'];times=[r['train_seconds'] for r in rr]
    report='''공통 손모양 보조 학습·학습 범위 확장 검토 (2026-11-13)

결론
지숫자·지문자에서 공통 손가락 특징을 배워 일반 수어에 사용하는 방향은 연구 근거가 있다. 다만 현재 확보한 숫자2·4·6만 더 학습해도 일반 단어 성능이 향상된다고 말할 근거는 이번 시험에서 나오지 않았다. 단어 이름을 늘리는 것보다 부족한 손가락 상태·움직임 대비와 그 정답을 함께 늘리는 것이 다음 우선순위다.

이번 실제 작업
1) 기존 공통100단어 학습은899영상이다. 이번에 감사한 공통100 손모양 정답 묶음과 그 추가분에는12표현/60학습영상,429개 손-시점의860개 특징 정답이 있었다. 이후 모든 동작/경계 연구의 주석을 합친 전체 개수라는 뜻은 아니다.
2) 별도 기록ID11~13의 일반 단어12표현/36영상,272개 손-시점/503개 알려진 특징을 평가했다. 특징은 검지만 펴기,네 긴 손가락 펼침,네 긴 손가락 접힘이다. 한 시점에 여러 항목이 있고 같은 영상 시점끼리 연관되므로503개 독립 영상이 아니다. 272x3중313칸은 정답 미확정/미부여로 채점하지 않았다. 좌표가 있다고 실제 손가락이 보인다는 뜻도 아니다.
3) 기존 공개 숫자26학습영상/78시점의 손가락 정답384칸을 보조로 재사용했다. 숫자 개발영상은 이번 학습·평가에 넣지 않았다. 추가 다운로드 없음.
4) 같은63입력→32은닉 손모양 부품으로 세 조건×세 고정 초기값,각600단계의 작은 지도학습을 수행했다. 일반 단어 정답/학습횟수/구조/정규화는 같고 숫자 보조 과제만 다르다. 숫자/사람/단어/화면 절대 위치는 입력하지 않았다. 현재 웹 TCN+GRU 자체를 재학습·교체한 시험은 아니다.

저장 결과 (일반 단어 특징503항목 중 정답, 초기값42/43/44)
일반 단어 손모양만         495 / 496 / 496
숫자 이름 과제 추가        494 / 496 / 494
손가락별 특징 과제 추가    495 / 496 / 495
두 초기값에서는 같고 한 초기값에서는1항목 낮다. 뚜렷한 추가 이득이 확인되지 않아 웹에 교체 적용하지 않았다. 이 약98%대 수치는 정답이 있는 일부 시점의 세 가지 큰 손모양 구분이며, 전체 동작 피드백 정확도·개별 손가락 정확도·새 사용자 정확도가 아니다. 손가락 특징을 배우는 방향 전체가 효과 없다는 결론도 아니다. 개별 손가락 오류를 따로 채점할 정답과, 지금보다 다양한 대비가 부족하다.

범위를 더 늘려야 하나
100은 기술적 한계가 아니다. 다양한 단어/사람을 통한 공통 특징 학습은 타당하며 필요하면100밖 자료도 쓴다. 기존 프로젝트의 동일 세부정답 비교에서도100단어 보조 학습은 추가8단어 손모양119~120/133을121~122/133으로 높였지만 단계 판단까지 개선하지는 않았다. SignBERT+의 사전학습량 실험도 자료 다양성 확대의 가능성을 뒷받침하지만, 이 프로젝트100→3000 효과를 측정한 실험은 아니다. 그래서 현재 결론은 필요한 특징을 채울 단어를 순차적으로 추가하는 것이며,3000단어 이름만 일괄 학습하는 것이 아니다. 공통 학습 범위와 실제 웹에서 교정 가능한 단어 수는 별개다.

추가 원본 확인
공개 FIT01의0·1·3·5·9·10 영상6개에서 각각2시점을 확인했다. 5는 엄지펴기+긴 네 손가락 접힘으로 기존2·4·6에 없던 검지접힘 대비 후보다. 3은2와 다른 손가락 조합,1은6과의 엄지 관계 대비 후보.0·10의 둥근 손모양은 단순 펴짐/완전접힘으로 억지 분류하지 않으며 굽힘과 겉으로 가까워짐을 따로 표시해야 한다. 이6개는 후보 시각검토이며 아직 다인정답 확정이나 새 학습에 투입한 자료가 아니다. 지문자 전체 자료를 확보/검증했다는 뜻도 아니다.

더 자세한 피드백 문장
가능하다. 단, '손모양이 다름'이라는 출력만으로 틀린 손가락을 지어내면 안 된다. 목표 설명의 해당 구간이 요구하는 손가락 상태와 실제 관측을 연결해야 한다. 예: [향후 근거가 갖춰진 경우의 문장] '이 동작에서는 오른손 검지는 펴고, 중지와 약지는 접어 보세요.' 가려졌다면 '엄지가 가려져 모양을 확인하기 어려워요'처럼 별도 안내한다. 이 문장 예시는 실제 사용자에 대한 새 분석 결과가 아니다. 현재 시험만으로 그런 손가락별 교정 기능을 완성했다고 할 수 없다.

진행 순서
A. 기존 일반 단어와 추가5·3·1 등에서 부족한 손가락별 대비를 여러 기록ID로 보강한다.0·10의 굽힘/접촉은 구분 가능한 항목부터 별도 설계한다.
B. 보조 숫자 점수가 아니라 일반 단어 손가락 정답 평가를 먼저 고정한다. 정상·교차·속도·방향·순서 및 불확실함도 유지한다.
C. 동일한 공통100 모델 기반에서 손모양 특징 학습 유무를 비교하고, 검증 후 시간 합성곱+GRU의 동작 구간과 연결한다.
D. 확인된 구간/손가락에만 짧은 교정 문장과 강조 표시를 붙인다.

보존/검증
C·T3-F,현재100단어 인식/웹 피드백 가중치·코드 보존. 사용자·멘토 자료 접근/학습/튜닝/후보선택 없음. 학습 결과를 본 뒤 기준/가중치/정답을 바꿔 재학습하지 않았다. 기존 숫자6개 모델/104단어 후보도 재실행하지 않았다. 입력12체크 및 별도 저장산술/해시84체크 통과. 모두 같은 Codex의 구현·시각검토·별도 검산이며 독립 한국수어 전문가 정답은 아니다. 새 HTML 없음. C7의 움직임·유지 판정/G2REWORK는 별도 남은 문제로 유지한다.

출처
https://aclanthology.org/2023.findings-emnlp.198.pdf
https://arxiv.org/pdf/2305.04868 (Table10)
https://aclanthology.org/2025.emnlp-main.1483.pdf
https://aclanthology.org/2025.naacl-long.19/
구체 근거/한계는 RESEARCH_EVIDENCE.json, 고정조건은 FROZEN.json, 입력범위는 COVERAGE.json, 판정은 DECISION.json.
'''
    (U/'RESULT_KO.txt').write_text(report,'utf-8')
    write(U/'EXECUTION_NOTES.json',dict(implementation='Codex local',training_runs=9,training_seconds=sum(times),validation='same Codex separate scripts',initial_errors=[dict(kind='read-only discovery',detail='Historical prepare.py path did not exist; inspected actual prepare_labels.py. No source mutations.'),dict(kind='process inventory',detail='Win32_Process CIM query was denied; completed preview manifest and artifacts checked instead. No elevated process probe.')],running=None,new_downloads=0,private_used=False,production_changed=False))
    write(U/'CURRENT_EXECUTION.json',dict(status='COMPLETE_REVIEW_AND_FIXED_TRANSFER_PROBE_NO_PROMOTION',running=None))
    protected=read(U/'FROZEN.json')['protected'];assert all(sha(p)==h for p,h in protected.items())
    frozen_files=['FROZEN.json','CODE_PINS.json','INPUT_MANIFEST.json','INPUTS.npz','COVERAGE.json','RESULTS.json','VERIFICATION.json','DECISION.json','RESULT_KO.txt','RESEARCH_EVIDENCE.json','FEEDBACK_CONTRACT.json','NEXT_PLAN.json','CANDIDATE_REVIEW.json','EXPANSION_SOURCE_PREVIEW.json']
    write(U/'COMPLETION.json',dict(status='COMPLETE_REVIEW_AND_FIXED_TRANSFER_PROBE_NO_PROMOTION',hashes={f:sha(U/f) for f in frozen_files},no_running_job=True))
    print('Completed report, evidence, next plan, fine-grained feedback contract; production unchanged')
if __name__=='__main__':main()
