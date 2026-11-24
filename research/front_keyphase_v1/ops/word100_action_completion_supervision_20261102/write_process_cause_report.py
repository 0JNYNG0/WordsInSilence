"""Compose a saved-evidence cause review; no model imports, inference or relabelling."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

U = Path(__file__).resolve().parent
read = lambda name: json.loads((U / name).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
counts = read('PROCESS_CAUSE_COUNTS.json')
scope = read('PROCESS_SCOPE_COUNTS.json')
align = read('PROCESS_ALIGNMENT_COUNTS.json')
raw = read('PROCESS_EVALUATION.json')
auto = read('PROCESS_AUTO_EVALUATION.json')
verified_raw = read('PROCESS_EVALUATION_SAVED_VERIFIED.json')
verified_auto = read('PROCESS_AUTO_EVALUATION_VERIFIED.json')
input_audit = read('NATIVE_CONTEXT_INPUT_AUDIT.json')
checks = []
def check(value, name):
    assert value, name
    checks.append(name)

check(verified_raw['status'] == 'PASS_SAVED_VALUE_ARITHMETIC_AND_LINEAGE_NO_INFERENCE'
    and verified_auto['status'] == 'PASS_SAVED_AUTO_EVALUATION_ARITHMETIC_PROVENANCE_AND_QUALIFICATION',
    'Both independent saved-value verification reports passed under their explicit status contracts')
check(verified_raw['evaluation_sha256'] == sha(U / 'PROCESS_EVALUATION.json'), 'Raw verification binds current saved predictions')
check(verified_auto['result_sha256'] == sha(U / 'PROCESS_AUTO_EVALUATION.json'), 'Auto verification binds current saved predictions')
check(verified_auto['raw_sha256'] == sha(U / 'PROCESS_EVALUATION.json'), 'Auto verification binds same raw predictions')
for p, h in counts['source_pins'].items():
    check(sha(p) == h, 'Original diagnosis source unchanged: ' + Path(p).name)
for role, group in counts['groups'].items():
    for name in ('manual_all', 'manual_native', 'automatic_native'):
        g = group[name]
        check(sum(g['positive_exclusive_causes'].values()) == g['positive'], role + '/' + name + ': exclusive buckets partition positives')
        check(g['positive_exclusive_causes'].get('all_conditions_met', 0) == g['positive_qualified'], role + '/' + name + ': pass bucket equals saved qualified count')
check(counts['groups']['development']['manual_all']['positive_qualified'] == 3, 'DEV manual qualification 3/30')
check(counts['groups']['development']['automatic_native']['positive_qualified'] == 1, 'DEV automatic qualification 1/15')
check(counts['groups']['fit']['manual_all']['positive_qualified'] == 38, 'FIT manual qualification 38/79')
check(align['fit_positive_direct_point_witness_consumption']['counts'] == {'known_positive':384, 'not_consumed_as_known_point':2}, 'Positive FIT direct point witness consumption 384/386')
check(len(raw['rows']) == 297 and len(auto['automatic_rows']) == 60, 'No changed evaluation population')

def ref(name, line, meaning):
    return dict(path=str(U / name), line=line, meaning=meaning)

raw_counts = {}
for role in ('fit', 'development'):
    rr = [r for r in raw['rows'] if r['role'] == role]
    known = [r for r in rr if r['truth'] is not None]
    raw_counts[role] = dict(views=len(rr), source_queries=len({r['query_id'] for r in rr}),
        source_clips=len({r['clip_id'] for r in rr}), recorded_ids=sorted({r['person'] for r in rr}),
        known=len(known), correct_at_predeclared_05=sum(int(r['probability'] >= .5) == r['truth'] for r in known),
        positive=sum(r['truth'] == 1 for r in rr), negative=sum(r['truth'] == 0 for r in rr), unknown=sum(r['truth'] is None for r in rr))

false = counts['false_confirmations'][0]
f = false['qualification']['witnessed_chain']
report = dict(
    status='CAUSE_REVIEW_COMPLETE_CURRENT_CANDIDATE_NOT_ADOPTED',
    at=datetime.now(timezone.utc).isoformat(),
    reviewer='retention_direction_review / Codex saved-value diagnosis; not a human or independent sign-language expert',
    decision_ko='현재 후보는 미채택을 유지한다. 가장 근거 있는 한 후속은 직접 연결과정 감독이 실제 학습 손실에 소비되고 실행과 같은 관측·구간 단위로 평가되는지 맞추는 제한된 비교이다. 임계값, 마지막 setup 선택, seed 또는 epoch 반복으로 현재 결과를 구제하지 않는다.',
    scope=dict(new_model_inference=0, new_training=0, new_thresholds=0, candidate_search=False,
        old_results_regraded=False, frozen_code_changed=False, source_labels_changed=False,
        media_reopened=False, private_or_locked_data_access=False, old_models_and_web_preserved=True,
        operation='Arithmetic decomposition of saved predictions using the original .5/.8/.2 definitions; no alternative decisions or replacement scores.'),
    population_and_dependence=dict(raw=raw_counts, automatic=dict(fit_query_rows=45, development_query_rows=15),
        caveats=['217 FIT views derive from 36 clips / 45 source-query groups; 80 DEV views derive from 12 clips / 15 source-query groups.',
            'Point/interval counts below include correlated copies across views. Native-only counts are provided separately.',
            'Recorded IDs are grouping metadata, not a verified claim about physical-person uniqueness.',
            'DEV11-13 have been repeatedly studied. No independent final-test or fresh-generalization claim.']),
    failure_decomposition=dict(
        definitions=dict(A='Saved raw process probability < .8', B='No ordered available setup/terminal candidate at the frozen .8 point threshold',
            C='At least one candidate exists, but every saved selected-interval motion probability is < .8',
            exclusive='A/B/C combinations partition positive views. B and C are mutually exclusive; A may co-occur. Candidate existence is not a correctness label.'),
        groups={role:{name:g[name] for name in ('manual_all','manual_native','automatic_native')} for role,g in counts['groups'].items()},
        mechanism_not_identifiability='These are failed requirements in the existing computation, not causal attributions from interventions. A change to one requirement does not establish that another would stay fixed.',
        candidate_cap_explanation='No positive bucket reaches the cap. Increasing the maximum of eight is not supported by these failures.'),
    point_and_interval_diagnostics=dict(
        thresholds_are_frozen=True,
        metrics={role:{name:{part:d[part] for part in ('all','native_only')} for name,d in group.items()} for role,group in counts['component_diagnostics'].items()},
        interpretation_ko='FIT의 전체 view 분류는 207/207이지만 setup 고확신473/859, terminal680/1096으로 부분 근거까지 적합한 것은 아니다. 직접 주석된 FIT motion 양성90/90은 고확신인데 자동 선택 구간 때문에 양성26view가 탈락한다. DEV에서는 직접 motion 양성도16/30만 고확신이므로 구간 선택만의 문제가 아니며 transfer가 추가된다.',
        warning='Sparse relation presence and annotated net movement are distinct targets. Neither score is dense event ownership, exact endpoint, continuous full-chain truth, or linguistic completion.'),
    selected_interval_scope=dict(
        saved_scope_summary=scope['summary'],
        index_overlap_summary=align['scope_summary'],
        stronger_evidence='The selected interval is not the interval unit used for motion supervision. FIT arm-traversal has 15/15 directly annotated motion positives high, but 0/15 qualified process views; 13 comparable selected candidates overlap their annotated interval and terminate earlier, all with low motion scores.',
        long_prefix_hypothesis='Only one of 18 strict contiguous containing pairs is high on the annotated interval and low on the larger selected interval. In DEV there are zero containing pairs. Longer preparation context alone is not established as the main cause.',
        illustrative_strict_superset=dict(view_id='NIA_SL_WORD2523_REAL08_F_directed_54_63__native',
            annotated_source_frames=[55,58], annotated_motion_probability=0.975300133228302,
            selected_source_frames=[45,58], selected_motion_probability=0.7497189044952393),
        early_ending_overlap='Among high-annotated/low-selected views, FIT has 26 early-ending pairs plus one containing pair across 26 dependent views; DEV has 12 early-ending pairs across 12 views. These are index relations, not a judgement that every earlier terminal is wrong.',
        no_exact_boundary_inference='Near-wrist/body/relative-height relations can become present while the same continuous movement continues. The annotated motion endpoint is not a mandatory stop or exact completion boundary. Extending to it or taking the last setup would be a new unvalidated rule.',
        local_truth='Exact selected intervals remain unlabelled unless directly supported. Index overlap or coverage does not turn them into positive/negative trace labels.'),
    native_manual_vs_automatic=dict(
        paired_same_query=counts['paired_native_summary'],
        same_frame_input_audit={role:{k:s[k] for k in ('query_views','unique_sources','compared_rows','mask_cell_differences','mask_cells','RGB_cosine')} for role,s in input_audit['summaries'].items()},
        interpretation='FIT known native manual high44/44 -> automatic31/44 and qualified27 ->17; DEV manual2/15 -> automatic1/15 despite zero proposal misses. Larger native context and extraction history are relevant domain changes, not merely new-person transfer.',
        limits='Saved manual/automatic comparisons change sequence extent and input extraction context together. The input audit observes same-frame feature drift but does not isolate sensor initialization, RGB context, feature normalization or recurrent context as the cause. Geometry is normalized by per-frame shoulders, not a whole-clip hand-size statistic.',
        proposal='All 44 known-positive FIT and all15 DEV native proposals contain the reviewed chain. This is annotation support bookkeeping, not prediction accuracy or proof of complete motion. Wider DTW spans alone are not an evidence-based fix.'),
    direct_supervision_consumption=dict(
        positive_fit_point_witnesses=align['fit_positive_direct_point_witness_consumption'],
        alternative_surviving_process_positives=counts['alternative_surviving_process_positives'],
        finding='Five alternative-surviving FIT positive views all have high global raw probability but no ordered candidate. Two views share the same REAL06 f73 terminal witness present in bound evidence but absent from known sparse point supervision. Other three fail despite available point supervision.',
        caution='384/386 positive witness records are already consumed; the two omitted records are one original frame, not two independent missing examples. Correcting this alone cannot explain the general endpoint or interval failure.',
        motion_semantics='REAL06 later falling witness67-73 is preserved as a direct contiguous witness with label:null and label_was_explicit:false. Current code correctly does not silently turn that into a numeric motion target. Its process-level positive and local physical evidence need an explicit, provenance-preserving target contract in a follow-up.',
        whole_population_audit='Parent delegated a separate full direct-witness consumption audit. This report only independently establishes positive FIT point-witness coverage and the concrete alternative-surviving cases.'),
    continuity_false_confirmation=dict(
        view_id=false['view_id'], raw_probability=false['p'], setup_probability=f['setup_probability'],
        terminal_probability=f['terminal_probability'], selected_motion_probability=f['motion_probability'],
        selected_output_extent=[f['start'],f['end']], selected_chain_source_frames=false['selected_chain_source_frames'],
        discontinuity=dict(source_before=55,source_after=59,output_seam=11),
        direct_review_reason=false['truth_reason'],
        saved_runtime=dict(qualified=1, truth=0), saved_oracle=dict(qualified=None, same_candidates_and_scores=True),
        interpretation='Visible near-hands evidence plus already-separated hands and residual outward motion does not establish the missing connected near-to-apart process. Raw .9986 and local motion .9839 both accept this splice; it is not solely a display-gate problem.',
        oracle_limit='Known edit provenance suppresses this one false confirmation (1/45 ->0/45) while true3/30 stays unchanged. That is a privileged diagnostic, not a usable real-video continuity detector, new inference or model improvement.',
        protection='Do not deny all cuts or blanks: another intact exchange may survive. Do not label a person/sign wrong. Preserve unknown when the connecting evidence is missing.'),
    retimed_dependence=dict(summary=counts['retime_summary'],
        conclusion='FIT20 retimed views represent10 source-queries; DEV10 represent5. Same-pixel-order variants have nearly identical scores, so speed variants are neither independent generalization samples nor evidence that global failures primarily arise from timing.'),
    evidence_supported_conclusions=[
        'There is a FIT supervision/runtime qualification mismatch as well as DEV transfer loss. The whole-view classifier can fit its known FIT targets without producing the confidence and interval semantics required by the qualification chain.',
        'Point confidence is already limited on FIT; low candidate motion is also present when the directly annotated interval is high. These are distinct, jointly relevant obstacles.',
        'A net-motion detector is not a continuous linked-process detector. The REAL12 splice exposes this distinction even with high point and global scores.',
        'Automatic native context is harder on the same FIT sources too. Proposal containment is intact, so optimizing witness coverage would target the wrong diagnosed limitation.',
        'The evidence does not establish that adding epochs, stronger point loss, a larger network, last-setup selection, exact terminal boundaries, or one input-normalization change would solve the problem.'
    ],
    one_next_change=dict(
        name='Align consumed supervision and runtime input around one directly supported connected-process instance',
        nature='One bounded follow-up implementation/comparison, not an automatic training authorization or a series of candidate searches.',
        concrete_change='Use the existing frozen RGB/geometry representation and the existing small typed head architecture to learn linked-process evidence on the same context/interval unit it will score. Consume explicitly reconciled direct setup/terminal witnesses. Keep net-motion as local support rather than treating its probability on an unlabelled early-ending candidate as a complete-chain certificate.',
        existing_evidence_to_reuse='Reuse reviewed FIT native links and partial-view positive/zero/unknown evidence, including surviving alternate repetitions, with exact source/pixel/time lineage. Positive trace targets require a directly reviewed connected instance in that evaluated context; presence of sparse endpoints or coverage alone is insufficient. Known residual/splice negatives can supervise absence of the observed linked process only in their reviewed scope.',
        input_alignment='Include the already prepared automatic full-native FIT observation contexts in the learning contract, or explicitly reproduce the deployed extraction/context route. Do not learn only freshly cropped manual windows and claim performance on reused full-native arrays. The root input audit must constrain this route; it does not by itself justify one sensor fix.',
        minimal_new_annotation='Only unresolved FIT instance links or conflicting/unconsumed witnesses need narrowly scoped adjudication. Do not demand a new broad corpus, manufacture dense endpoints, or relabel DEV from its failures. An unlabelled numeric interval remains unknown until its actual intended semantic unit is explicit.',
        fixed_comparison='Freeze one follow-up candidate and one execution route before new scores. Preserve the present candidate as the failed baseline. Keep one seed42/final80epoch budget at most, existing split and fixed thresholds/gates, no selection over epochs/seeds or thresholds. This is a bundled contract-alignment test; it cannot isolate every cause.',
        before_execution_stop=[
            'Every known trace/point target must resolve to reviewed source/pixel/time evidence; inconsistent old/new values stay unknown with explicit lineage.',
            'Training and runtime must score the same trace semantics and supported context. A net-motion label cannot be renamed full-process evidence.',
            'No word/edit-kind, seam metadata, elapsed fraction or final-frame flag may encode the answer. Oracle seam filtering remains diagnostic only.',
            'The five alternative-surviving positives and all unknown cases must stay distinct from ordinary residual negatives.'
        ],
        pass_and_stop='Retain the predeclared observed-positive recovery, zero false confirmation, unknown protection, per-query and auto15 gates; independently recompute saved outputs. If the single aligned comparison still fails, keep it out of the demo and report the specific unmet contract. Do not automatically launch another training run.',
        practical_goal='Improve evidence for an observed source-linked process and its time span. No full-word correctness, mandatory pause/repetition count, omitted-action denial or learner-error verdict.'),
    code_evidence=[
        ref('process_qualification.py',31,'First confident setup frame per run; first later terminal frame at line41; no event-boundary annotation is consulted.'),
        ref('process_qualification.py',58,'Saved local motion>=.8 becomes a chain candidate; raw>=.8 then qualifies at line60.'),
        ref('process_head.py',68,'Interval logits reset recurrent scope to the selected interval; frozen RGB may still contain local context.'),
        ref('train_process_head.py',96,'Interval training consumes fully_retained_motion_interval_targets only.'),
        ref('train_process_head.py',115,'Point training consumes sparse_relation_targets only; combined loss at line128.'),
        ref('compile_reviewed_views.py',310,'Direct bound witnesses are retained separately; inherited relation targets built at line321 and stored separately at359-361.'),
        ref('evaluate_process_auto.py',107,'Exact selected-chain time truth remains null; no exact-boundary score is claimed.')
    ],
    verification=dict(checks_passed=len(checks), checks=checks, saved_scores_only=True,
        initial_report_builder_failure='First report-builder invocation stopped before writing output: it assumed the literal status PASS, but both source verifiers use explicit PASS_SAVED_* status names. The check now asserts those two exact observed names. No score or frozen file was changed.',
        qualifications='Same Codex, separate arithmetic readout. Existing saved-result verification is not an independent human/expert truth audit.'),
    whole_goal_complete=False)
files=['PROCESS_CAUSE_COUNTS.json','PROCESS_SCOPE_COUNTS.json','PROCESS_ALIGNMENT_COUNTS.json',
    'PROCESS_EVALUATION.json','PROCESS_AUTO_EVALUATION.json','PROCESS_EVALUATION_SAVED_VERIFIED.json',
    'PROCESS_AUTO_EVALUATION_VERIFIED.json','PROCESS_COMPARISON_FROZEN.json','PROCESS_DECISION.json',
    'NATIVE_CONTEXT_INPUT_AUDIT.json','PROCESS_VIEW_TARGETS.json','DEV_PROCESS_VIEW_TARGETS.json',
    'process_head.py','train_process_head.py','process_qualification.py','compile_reviewed_views.py',
    'evaluate_process_auto.py','analyze_process_saved.py','analyze_process_scope_saved.py',
    'analyze_process_alignment_saved.py','write_process_cause_report.py']
report['source_pins']={str(U/n):sha(U/n) for n in files}
out=U/'RESULTS_PROCESS_CAUSE.json'
assert not out.exists()
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
check(read('RESULTS_PROCESS_CAUSE.json')['status']==report['status'],'Output parses and status round-trips')
print(json.dumps(dict(status=report['status'],output=str(out),sha256=sha(out),checks_passed=len(checks),new_inference=0),ensure_ascii=False))
