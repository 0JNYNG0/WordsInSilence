"""Plan and read-only evidence audit only. No extraction, training or inference."""
from pathlib import Path
import json, hashlib
from collections import Counter

U=Path(__file__).resolve().parent;O=U.parent;ROOT=U.parents[3]
PREV=O/'word100_handshape_transfer_20261113'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')

def main():
    assert not (U/'PLAN.json').exists()
    manifest=O/'word_acquisition_inventory_20261011/FRONT_MANIFEST.jsonl'
    rows=[json.loads(x) for x in manifest.read_text('utf-8-sig').splitlines()]
    fit=[r for r in rows if 1<=r.get('filename_signer_id',-1)<=9];counts=Counter(r['word_code'] for r in fit)
    old=read(PREV/'COMPLETION.json');assert all(sha(PREV/p)==h for p,h in old['hashes'].items())
    prior=read(PREV/'FROZEN.json');assert all(sha(p)==h for p,h in prior['protected'].items())
    decision=read(PREV/'DECISION.json');assert decision['status']=='NO_CLEAR_TRANSFER_GAIN_NO_PROMOTION'
    inventory=dict(metadata_entries=len(rows),word_codes=len({r['word_code'] for r in rows}),fit_entries=len(fit),fit_word_codes=len(counts),fit_count_distribution=dict(Counter(counts.values())),metadata_only=True,all_video_payloads_decoded=False,record_ids_not_face_verified_persons=True,hdd_present=Path('W:/WordsInSilence/data/AIHub_103').exists(),manifest_sha256=sha(manifest))
    write(U/'EVIDENCE_AUDIT.json',dict(status='PASS_PLAN_INPUTS_ONLY',inventory=inventory,previous_completed_outputs_unchanged=True,protected_models_unchanged=True,previous_probe=decision['status'],limitations=['Metadata count is not a usable-video guarantee.','Previous transfer probe used a frame MLP, not a new full TCN+GRU evaluation.','503 known coarse cells are not individual-finger or educational correctness truth.','No fresh performance tests run this turn.'],research_reference=str(PREV/'RESEARCH_EVIDENCE_REVIEWED.json')))
    plan=dict(status='PLAN_REVIEWED_WAITING_USER_APPROVAL',goal='Preserve common100 training while selectively adding public signs that teach missing transferable hand/finger/motion attributes, then source-grounded concise visual feedback.',scope=dict(common_words=100,first_batch_additional_word_cap=20,cap_reason='Bounded annotation/extraction workload; not an empirically optimal number and not a target to fill.',new_batch_includes_already_prepared_numeric_words=True,fit_ids=list(range(1,10)),development_ids=[11,12,13],max_added_fit_clips=180,max_added_development_clips=60,production_tcn_classes=100,initial_feedback_validation_words=4,all_3000_training=False),steps=[
      dict(id=1,name='Coverage-driven selection',work=['Build coverage table from existing100 verified guides/anchors: finger states, thumb placement, spread, orientation, stationary/moving shape, two-hand relation, body region, direction/order.','Review remaining metadata and source descriptions for missing contrasts. Select the smallest useful set up to20 outside100. Numeric1/3/5 are initial source-preview candidates; include ordinary signs for motion/hand interaction.','Use only FIT previews for selection, never DEV model errors. A source name alone does not establish its sense/variant or detailed motion.'],pass_rule='Every admitted extra sign fills a stated gap, has verified source identity and more than one usable public FIT record; unavailable/descriptive-mismatch cases logged, not forced.'),
      dict(id=2,name='Specific supervised answers',work=['For both hands, annotate each visible finger as extended/bent/folded/unknown; use a coarser label when the view cannot distinguish. Label spread/thumb relation/projected proximity only where visible.','Attach labels to reviewed native time points or genuinely reviewed stable spans; identify shape change, direction change and start/progress/end for selected action examples. Do not extend sparse labels across unreviewed frames.','Connect a verified dictionary/guide clause to the relevant feature and interval; retain inferred DTW ranges as weak estimates. If no verified guide exists, use physical attribute learning only.','Initial support target per tested attribute/state: at least3 FIT recorded IDs and2 distinct expressions, with both contrasting states represented. This is a coverage gate, not a statistical sufficiency claim.'],pass_rule='Conflicting, hidden or variant-uncertain targets masked. Individual-finger evaluation must have its own verified labels, not inferred from a number prediction or joint extractor.'),
      dict(id=3,name='Reusable RTMW inputs and partitions',work=['Decode and verify only selected videos. Reuse all existing matching RTMW caches; newly extract selected missing inputs once. Original W data read-only.','Preserve body-relative position/direction/order; normalize hand-local geometry for shape only. Keep native timestamps and missingness. Do not synthesize unseen finger truth.','Audit source SHA duplicate/cross-role leakage and recorded IDs. Train01-09; use11-13 as historical development, not newly independent testing. Existing sealed data stay sealed.','Reserve an eligible unused public evaluation group before results if provenance/use-history allows. If none exists, explicitly limit claims to development; do not silently open older forbidden sets.'],pass_rule='Verified input manifest, strict provenance, finite arrays, native-time alignment and source hashes. Failures/unknowns reported with denominator; no private user/mentor data.'),
      dict(id=4,name='Controlled supervised fine-tuning',work=['Keep the RTMW + temporal convolution + GRU architecture and common100 support. Add or refine physical attribute outputs; no new large RGB/LLM/SignRep path.','A: existing100 inputs + newly verified fine annotations within100. B: same A + selected extra videos with word-label auxiliary objective. C: same B + new direct finger/action labels.','Use the same backbone initialization, ordinary-data batches, steps, model capacity and fixed seeds42/43/44. B/C see identical extra videos; any training-only extra word outputs are excluded from runtime TCN. Match scheduled compute/batches for the ablation and balance examples by expression/record to avoid numeric domination.','Implement heads-only initialization followed by low-rate backbone fine-tuning if the agreed frozen training schedule includes it; freeze exact schedule/loss weights from FIT-only checks before development. Do not reuse completed probe scores to tune.'],pass_rule='Nine fixed candidate runs at most in the first comparison; report all seeds. Existing checkpoints untouched. A-to-B estimates added word-data effect, B-to-C added feature-supervision effect, current saved baseline-to-A label improvement.'),
      dict(id=5,name='Ordinary-sign and whole-path comparison',work=['Primary: individual-finger macro F1 on ordinary signs, with per-state/per-hand/per-record results. Also record coverage/abstentions, not just scores on selected successes.','Check source-grounded normal intervals for false corrections, visible handshape contrasts, direction/order/completion, normal speed changes, crossing and brief missing observations. Missing truth is not a success.','Use other public real handshape examples as physical contrasts and clearly separate any edited/reversed/omitted trials from real learner errors. Neither is automatically expert truth about allowable KSL variants.','Keep existing C7 movement-versus-hold diagnostic branch separate: new shape learning cannot declare it solved.','Measure current full pipeline on the same public clips, hardware path, cache conditions and timing start/stop; distinguish capture-time extraction from post-capture waiting.'],pass_rule='Require consistent target-feature improvement across seeds/held records without loss of established shape/direction/order, increased normal false corrections, or reduced uncertainty coverage. Freeze exact metric/gates before scoring. Ambiguous/insufficient support means scoped research result, not deployment.'),
      dict(id=6,name='Source-linked feedback and expansion decision',work=['Link validated observed feature difference to the correct guide clause and native action interval; highlight only supported hand/fingers and give one short sentence. Unknown gets a separate observation request.','Initial web feedback scope remains the existing4 words; shared attributes trained from many words are reusable. Add product words only after guide/interval/feedback validation.','Do not compare every training reference at runtime. Use the learned model and existing DTW playback/alignment role; DTW matching is not proof that a missing action was performed.','If selective expansion improves ordinary-sign feedback and timing remains acceptable, add the next coverage-driven batch. If benefit is only numeric classification, or regressions appear, diagnose data/labels/observation/integration before adding another batch.','No default model replacement before validated report/user review; user camera acceptance test later, never training/tuning/candidate selection.'],pass_rule='Demonstrable precise wording/highlights from validated attributes, preserved normal behavior, transparent unknowns. Approval of this plan authorizes bounded research, not all3000 training or automatic production replacement.')],failure_routes=[dict(case='Labels cannot distinguish fingers',return_to=2,action='Use unknown/coarser attribute; add visible multi-record evidence, not larger generic data volume.'),dict(case='RTMW observed geometry is wrong',return_to=3,action='Separate extractor failure from model failure; targeted observation repair with source evidence.'),dict(case='New shape improves but old direction/order declines',return_to=4,action='Review loss interference/branch separation and replay balance under new FIT-only contract; retain the failed run.'),dict(case='Correct shape is attached to wrong action',return_to=2,action='Repair interval/guide association in C7/temporal branch, not finger classifier thresholds.'),dict(case='Only numbers improve',return_to=1,action='Add ordinary-word appearances and moving-hand contrasts, not more copies of numeric class labels.'),dict(case='No fresh eligible evaluation set',return_to=5,action='Report development evidence only; do not relabel reused DEV as fresh.')],protected=dict(original_C_T3F=True,current_web=True,common100=True,user_and_mentor_training=False,new_downloads=False,external_transfer=False,git=False),review_method='Local source/provenance/comparison-design review plus primary research; no subagent or expert-validation claim',execution_authorized=False)
    write(U/'PLAN.json',plan)
    review=[
      ('User asks approval before execution',plan['status']=='PLAN_REVIEWED_WAITING_USER_APPROVAL' and not plan['execution_authorized']),
      ('Expansion bounded and gap-driven',plan['scope']['first_batch_additional_word_cap']==20 and not plan['scope']['all_3000_training']),
      ('Current100 classifier not expanded by accident',plan['scope']['production_tcn_classes']==100),
      ('Nine paired candidates with actual TCN+GRU, not previous frame probe',len(plan['steps'])==6),
      ('Observed, weak DTW and unknown label roles specified',True),
      ('Recorded-ID and duplicate split includes historical DEV limitation',True),
      ('Word-data and fine-label effects separated',True),
      ('Real learner error and synthetic contrast not conflated',True),
      ('Source/current-model evidence hashes intact',all(sha(p)==h for p,h in prior['protected'].items())),
      ('No execution initiated',True)]
    write(U/'PLAN_REVIEW.json',dict(status='PASS_DESIGN_REVIEW_NOT_PERFORMANCE_VALIDATION',automated_evidence_checks=['prior completion hashes','protected artifact hashes','manifest counts'],manual_design_checks=[dict(item=n,reviewed_ok=v) for n,v in review],reviewer='same Codex; no external human or independent agent review',plan_sha256=sha(U/'PLAN.json'),training_runs=0,model_calls=0,new_video_extraction=0))
    text='''100단어 밖 공통 특징 학습 확장 — 승인 대기 계획

판단: 기존100단어 공통 학습을 유지하고, HDD의 나머지 자료에서 부족한 특징을 채울 단어를 선별 추가한다. 3000단어를 즉시 일괄 추출/학습하는 계획은 아니다. 1차 외부범위는 최대20단어이며 이전에 확보한 숫자도 이 범위에 포함한다. 20은 최적 단어수라는 연구결론이 아니라 작업량 상한이다. 학습9기록/개발3기록 기준 최대180+60영상, 기존추출분재사용. 필요한 대비를 더 적은 단어로 채울 수 있으면 적게 선택한다.

확인한 근거: 현재 목록은3000단어/54000정면기록,학습후보기록26945개(2945단어9기록,55단어8기록)다. 실제모든영상디코딩/정답검수를한것은아니다. 기존100단어899영상학습은유지한다. 감사한세부수형정답은12표현에집중됐다. 이전숫자보조9실험은일반단어세가지큰수형에서명확한개선없음. 숫자2/4/6은모두검지펴짐이라새특징대비부족. 숫자5의검지접힘을포함한6원본FIT01예시만확인된상태이며다인세부정답완성은아니다.

1. 추가할 단어를 고른다.
기존100에서 이미 배운 특징과 부족한 특징의 표를 만든다. 손가락 조합/엄지/벌림/굽힘,정지와이동중같은모양,양손관계,방향·전환·순서로 나눈다.1·3·5는 첫 수형 후보이며 일반단어도 포함한다. 단어 이름이나 숫자량보다 어떤 대비가 늘어나는지로 고른다. 기존평가오답을보고단어를고르지않는다.

2. 구체적인 지도학습 정답을 붙인다.
어느 손의 어느 손가락이 펴짐/굽힘/접힘인지,엄지가어디에있는지,언제모양이나방향이바뀌는지 표시한다. 보이지않는부분은미확인. 직접확인시점과검수한유지구간만강한정답으로쓴다. 사전설명의뜻·변이를확인하고그문장의해당행동과연결한다. DTW로옮긴구간은추정으로남긴다. 최소학습지원목표는상태별3기록ID·2표현이지만통계적충분성을보장하는수치는아니다. 설명이없는단어도공통물리특징학습에는쓸수있되교정규칙을만들어내지않는다.

3. 필요한 영상만 RTMW로 준비한다.
기존추출값을재사용하고새로고른영상만한번추출한다. 몸기준위치·방향·시간순서는유지하고손모양에는손크기·위치정규화를쓴다. 학습01~09/개발11~13분리,원본SHA중복검사. 반복사용개발자료를새독립시험이라하지않는다. 기존봉인자료는유지하고새최종평가자료가적격한지사용이력부터확인한다. 없다면개발결과로한정한다.

4. 현재 시간합성곱+GRU를 지도학습·미세조정한다.
A 기존100+그안에서보강한세부정답.
B A+추가영상의단어정답학습.
C B+추가영상의손가락·동작세부정답학습.
세조건의일반단어학습횟수·모델크기·초기값·비교입력을맞추고각3시드로고정한다. A→B는범위확장효과,B→C는세부정답효과를본다. 숫자이름은공통특징학습보조일뿐일반단어의뜻으로붙이지않는다. 추가단어보조출력은학습용이며기존TCN의100개출력을교체하지않는다. 정확한학습일정은FIT만으로입력·메모리·손실흐름확인후평가전동결한다. 이번은계획이며새학습미실행.

5. 일반 단어 전체 흐름에서 평가한다.
주지표는일반단어의개별손가락구분(macro F1). 손별/상태별/기록별결과와보류율을함께본다. 정상행동오경고,방향·순서·누락,정상속도차이·교차·잠깐미검출도확인한다. 합성역순/삭제/손모양변형은기술적시험이며실제학습자오류나수어허용범위정답과구분한다. 동일조건전체경로처리시간도확인한다. 여러시드/기록에서이득이이어지며기존중요기능회귀·오경고증가가없어야통과한다. 손모양이좋아도C7움직임/구간문제는해결된것으로올리지않는다.

6. 검증된 특징을 짧은 피드백에 연결한다.
'이 동작에서는 검지를 펴고 중지·약지를 접어 보세요'처럼설명문장의행동과실제다른손가락·시간대를연결하고강조한다. 불확실하면별도안내한다. 현재4단어는웹검증범위일뿐공통학습한계가아니다. 사용시모든학습영상을1대1로비교하는구조로되돌리지않는다. DTW는대응/재생보조이며매칭성공을완수증거로쓰지않는다. 개선되면다음누락특징단어묶음확장,안되면정답/관측/손실간섭/구간연결원인으로돌아간다. 더많은단어만반복투입하지않는다.

완료 결과물: 추가단어선정근거표,손가락·동작정답과관측부족표,고정학습비교,회귀·응답시간검증,설명구간과연결된피드백후보. 첫비교를통과한후웹후보반영여부검토. 최종사용자촬영확인은학습·튜닝·후보선택에쓰지않는다.
연구근거: https://aclanthology.org/2023.findings-emnlp.198/ ; https://arxiv.org/pdf/2305.04868 (Table10) ; https://aclanthology.org/2025.naacl-long.19/
연구는손모양명시지도/넓은사전학습의가능성을보이나우리프로젝트100→3000효과를입증한것은아니다. 지문자자료혼합효과가항상좋지는않다는연구도있다. 여기서새모델다운로드나대형RGB모델교체를제안하지않는다.

검토범위: 저장결과·해시·자료목록및원논문대조와설계검토. 독립전문가검토나새성능검증아님. 기존C·T3-F/현재웹/사용자자료금지유지. 새관절추출/학습/추론/다운로드없음. 사용자계획확인·승인이후실행한다.
'''
    (U/'PLAN_KO.txt').write_text(text,'utf-8')
    write(U/'CURRENT_EXECUTION.json',dict(status='WAITING_USER_PLAN_APPROVAL',running=None,training_started=False))
    print(json.dumps(inventory,ensure_ascii=False));print('PLAN_REVIEWED_WAITING_USER_APPROVAL; no execution started')
if __name__=='__main__':main()
