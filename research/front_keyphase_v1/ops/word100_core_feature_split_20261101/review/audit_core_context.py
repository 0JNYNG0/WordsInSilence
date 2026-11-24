"""Read saved public metadata only; never infer labels or run existing experiments."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPS = HERE.parents[1]
FEATURES = ('index_only', 'open_four')
sources = {}
checks = Counter()
failures = []


def check(name, ok, location=''):
    checks[name] += 1
    if not ok:
        failures.append({'check': name, 'location': location})


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(unit, name):
    p = OPS / unit / name
    sources[str(p)] = digest(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))


bundle_unit = 'word100_shape_sequence_bundle_20261027'
bundle = read(bundle_unit, 'BUNDLE.json')
export = read(bundle_unit, 'OBSERVATION_EXPORT.json')
temporal = read('word100_temporal_shape_context_20261027', 'COMPARISON.json')
guard = read('word100_temporal_transition_guard_20261027', 'COMPARISON.json')
physical = read('word100_crossperson_stage_audit_20261101', 'PHYSICAL_CORRESPONDENCE.json')
assessment = read('word100_crossperson_stage_audit_20261101', 'ASSESSMENT.json')
core = read('word100_core_context_split_20261101', 'SUMMARY.json')
flow = read('word100_ordered_motion_alignment_20261101', 'SUMMARY.json')
description = read('word100_description_evidence_feedback_20261101', 'REPORT_DATA.json')

exports = {x['audit_case_id']: x['observation_input'] for x in export['windows']}
counts = Counter()
roles = Counter()
inventory = []
for seq in bundle['sequences']:
    case = seq['case_id']
    frames = seq['frames']
    out = exports[case]
    roles[seq['role']] += 1
    check('full_window_kept', len(frames) == len(out['observations']) == out['expected_frame_count'] == 13, case)
    check('source_draft_not_human_truth', seq['source_label_status'] == 'AI image observation drafts, not independent human truth', case)
    check('window_endpoints_kept', out['expected_from_pts_us'] == frames[0]['pts_us'] and out['expected_to_pts_us'] == frames[-1]['pts_us'], case)
    for i, (frame, obs) in enumerate(zip(frames, out['observations'])):
        counts['frames'] += 1
        check('frame_time_side_identity', frame['frame_id'] == obs['frame_id'] and frame['pts_us'] == obs['pts_us'] and obs['side'] == seq['side'], case)
        if i:
            check('original_contiguous_time', frame['frame_id'] == frames[i-1]['frame_id'] + 1 and frame['pts_us'] > frames[i-1]['pts_us'], case)
        contradiction = all(frame['direct'][f] == 1 for f in FEATURES)
        counts['contradictory_frames'] += int(contradiction)
        for f in FEATURES:
            v = frame['direct'][f]
            item = obs['features'][f]
            counts['feature_samples'] += 1
            counts['unknown_samples' if v is None else 'known_samples'] += 1
            check('direct_value_preserved', item['source_value'] == v, case + '/' + f)
            check('unknown_and_contradiction_not_filled', item['comparison_value'] == (None if contradiction else v), case + '/' + f)
            check('draft_provenance_not_promoted', item['human_verified'] is False and item['is_model_inference'] is False and obs['complete_handshape'] is None and obs['semantic_correctness'] is None, case)
    for f, feature in seq['features'].items():
        vals = [x['direct'][f] for x in frames]
        cover = feature['coverage']
        check('feature_support_denominator', feature['values'] == vals and cover['expected_samples'] == 13 and cover['known_samples'] == sum(v is not None for v in vals) and cover['unobserved_samples'] == sum(v is None for v in vals), case + '/' + f)
        check('no_whole_sign_or_namedshape_truth', feature['semantic_complete'] is False and feature['no_change_certified'] is False and feature['whole_sign_scope'] is False and feature['feedback_verdict'] is None and feature['handshape_name'] is None, case + '/' + f)
        for link in feature['links']:
            counts[link['kind']] += 1
            check('transition_middle_not_invented', link['exact_transition_time'] is None and link['intermediate_shape'] is None and link['linguistic_importance'] is None, case)
            check('transition_endpoints_preserved', vals[link['from_index']] == link['from_value'] and vals[link['to_index']] == link['to_value'], case)
            if link['kind'] == 'same_endpoints_unresolved_gap':
                check('same_endpoints_do_not_fill_gap', bool(link['unobserved_indices']) and all(vals[i] is None for i in link['unobserved_indices']), case)
        for gap in feature['gaps']:
            counts[gap['kind']] += 1
            check('gap_stays_unknown', gap['gap_not_imputed'] and gap['value'] is None and all(vals[i] is None for i in gap['indices']), case)
    inventory.append({'case_id': case, 'clip_id': seq['clip_id'], 'recorded_person_id': seq['recorded_person_id'], 'side': seq['side'], 'role': seq['role'], 'frame_range': [frames[0]['frame_id'], frames[-1]['frame_id']], 'pts_range_us': [frames[0]['pts_us'], frames[-1]['pts_us']], 'evidence_path': str(OPS / bundle_unit / 'BUNDLE.json'), 'json_pointer': '/sequences/' + str(len(inventory)), 'not_fresh_evaluation': True})
for k in ('feature_samples', 'known_samples', 'same_endpoints_unresolved_gap', 'bounded_feature_change', 'entire_window_unobserved'):
    check('bundle_count_matches_saved_' + k, counts[k] == bundle['counts'][k])
check('bundle_unknown_matches_saved', counts['unknown_samples'] == bundle['counts']['unobserved_samples'])

temporal_stats = {'unknown_support': Counter(), 'masked': {c: Counter() for c in 'ABC'}}
for seq in temporal['sequences']:
    frames = seq['frames']
    for i, frame in enumerate(frames):
        for f in FEATURES:
            v = frame['direct'][f]
            for condition, outputs in frame['outputs'].items():
                item = outputs[f]
                if v is not None:
                    check('temporal_keeps_direct', item['value'] == v and item['kind'] == 'direct', seq['case_id'])
                elif item['value'] is not None:
                    temporal_stats['unknown_support'][condition] += 1
                    check('temporal_not_direct_truth', item['kind'] != 'direct', seq['case_id'])
                    check('temporal_anchors_only_original_direct', bool(item['support_indices']) and all(frames[j]['direct'][f] == item['value'] for j in item['support_indices']), seq['case_id'])
for trial in temporal['masked_trials']:
    for c in 'ABC':
        v = trial['outputs'][c]['value']
        temporal_stats['masked'][c]['abstained' if v is None else 'matches_hidden_draft' if v == trial['hidden_draft'] else 'conflicts_hidden_draft'] += 1
guard_stats = {'fresh': {c: Counter() for c in ('baseline', 'guard')}, 'old': {c: Counter() for c in ('baseline', 'guard')}}
for seq in guard['sequences']:
    for frame in seq['frames']:
        for f in FEATURES:
            if frame['direct'][f] is not None:
                check('guard_keeps_direct', frame['baseline'][f]['value'] == frame['guard'][f]['value'] == frame['direct'][f], seq['case_id'])
for trial in guard['masked_trials']:
    group = 'fresh' if trial['fresh_dense'] else 'old'
    for c in ('baseline', 'guard'):
        v = trial[c]['value']
        guard_stats[group][c]['abstained' if v is None else 'matches_hidden_draft' if v == trial['hidden_draft'] else 'conflicts_hidden_draft'] += 1
for event in guard['events']:
    check('guard_event_no_semantic_truth', event['semantic_importance'] == 'not_determined' and event['intermediate_exact_shape'] is None, event['case_id'])

physical_counts = Counter()
for clip in physical['clips']:
    physical_counts['clips'] += 1
    for key in ('native_frame_count', 'reviewed_native_frames', 'reviewed_known_frames', 'uncertain_transition_frames', 'outside_reviewed_scope_frames'):
        physical_counts[key] += clip[key] if isinstance(clip[key], int) else len(clip[key])
    for region in clip['regions']:
        physical_counts['region_' + region['region_type']] += 1
        if region.get('motion_subphase'):
            physical_counts['subphase_' + region['motion_subphase']] += 1
        if region['region_type'] == 'observed_physical_core':
            check('physical_draft_not_3d_contact_truth', region['fine_finger_3d_or_exact_contact_validated'] is False, clip['alias'])
check('physical_not_linguistic_training_truth', physical['linguistic_truth'] is False and physical['neural_training_target'] is False)
phase_counts = defaultdict(Counter)
for event in assessment['events']:
    role = event['role']
    phase_counts[role]['stages'] += 1
    for method, m in event['methods'].items():
        phase_counts[role][method + '_' + m['state']] += 1
        check('samephase_not_sign_correctness', m['sign_correct'] is None and m['sign_error'] is None, event['case'])
        if m['trace'] is not None:
            expected = event['reference_core'][1] - event['reference_core'][0] + 1
            tr = m['trace']
            check('samephase_fixed_full_core_denominator', tr['expected_reference_frames'] == expected and math.isclose(sum(tr['reference_equal_weight_counts'].values()), expected, abs_tol=1e-7), event['case'])
            phase_counts[role][method + '_expected_reference_frames'] += expected

desc_counts = Counter()
feedback_by_role = defaultdict(Counter)
catalog = {x['id']: x for x in description['source_clause_catalog']}
for c in catalog.values():
    check('guide_not_feature_label', c['expected_feature_value'] is None and c['learned_or_manual_semantic_target'] is False and c['linguistic_verdict'] is None, c['id'])
for case in description['cases']:
    desc_counts['pairs'] += 1
    for stage in case['stages']:
        desc_counts['stages'] += 1
        lo, hi = stage['reference_core']
        by_reference = defaultdict(list)
        for q, r in stage['path']:
            by_reference[r].append(q)
        check('description_path_full_core', set(by_reference) == set(range(lo, hi+1)), case['id'])
        linked_groups = set()
        for card in stage['source_cards']:
            desc_counts['source_cards'] += 1
            linked_groups.update(card['applicable_numeric_groups'])
            check('guide_anchor_not_whole_phase_truth', card['full_span_semantic_verified'] is False and card['named_shape_verified'] is False and card['source_action_verified'] is False and card['training_target'] is False, card['clause_id'])
            check('guide_reference_anchor_inside_core', all(lo <= v <= hi for v in card['reference_anchor_frames']), card['clause_id'])
            clause = catalog[card['clause_id']]
            check('guide_exact_source_reuse', card['text'] == clause['source_span'] and card['url'] == clause['source_url'] and clause['runtime_scope'] != 'inactive_different_form', card['clause_id'])
        for group, evidence in stage['groups'].items():
            desc_counts['measurement_opportunities'] += 1
            feedback = evidence['feedback']
            check('feedback_no_correctness_promotion', feedback['educational_correct'] is None and feedback['educational_error'] is None and feedback['correction_instruction'] is None and not feedback['important_difference_proven'] and not feedback['allowable_difference_proven'], case['id'])
            if group in linked_groups:
                feedback_by_role[case['role']][feedback['status']] += 1
            trace = evidence['trace']
            # Direction groups may intentionally have no component trace.
            if trace is None:
                continue
            check('description_trace_full_core', [r['reference_frame'] for r in trace] == list(range(lo, hi+1)), case['id'])
            for row in trace:
                desc_counts['trace_reference_rows'] += 1
                check('trace_links_all_preserved', row['query_frames'] == by_reference[row['reference_frame']], case['id'])
                check('trace_fixed_possible_components', len(row['reference_valid']) == len(row['reference_values']) == row['possible_components'] and row['expected_components'] == sum(row['reference_valid']), case['id'])
                check('trace_null_not_zero', all((v is None) == (not valid) for v, valid in zip(row['reference_values'], row['reference_valid'])), case['id'])
                check('trace_missing_denominator_kept', all(len(qv) == row['possible_components'] for qv in row['query_valid']), case['id'])

check('core_does_not_claim_semantics', core['gates']['semantic_correctness_not_tested'] is True and core['production_adopted'] is False)
check('flow_does_not_replace_global_path', flow['full_path_replacement'] is False and flow['alignment_accuracy_improved'] is False)
for p, value in sources.items():
    check('all_input_bytes_preserved', digest(Path(p)) == value, p)

report = {
    'status': 'PASS_SAVED_PUBLIC_EVIDENCE_AUDIT' if not failures else 'FAIL_SAVED_PUBLIC_EVIDENCE_AUDIT',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Read existing public metadata; no new visual labels, model calls, training, rescoring, timing, HTML, private data or production writes.',
    'reviewer': 'Codex core_plan_review; saved-data audit, not independent human/KSL truth',
    'checks': dict(checks), 'check_count': sum(checks.values()), 'failures': failures,
    'sources_sha256': sources,
    'reusable_direct_feature_evidence': {'counts': dict(counts), 'roles': dict(roles), 'windows': inventory, 'feature_definitions': list(FEATURES), 'not_fist_or_complete_handshape': True, 'new_independent_labels_created': 0},
    'temporal_support_counterevidence': temporal_stats,
    'transition_guard_counterevidence': guard_stats,
    'physical_stage_drafts': {'counts': dict(physical_counts), 'count_note': 'reviewed_native_frames counts inspected frame IDs, including sparse context outside dense phase-label scope; known+uncertain+outside partition native frames, not inspected frames.', 'assessment_counts': dict(phase_counts), 'saved_development_comparison': assessment['summary']['dev'], 'fresh_evaluation': False},
    'description_evidence': {'counts': dict(desc_counts), 'linked_feedback_by_role': dict(feedback_by_role), 'clauses': len(catalog), 'inactive_clauses': [x['id'] for x in catalog.values() if x['runtime_scope'] == 'inactive_different_form']},
    'official_example_hypothesis': {
        'claim': 'Official correct-word identity makes unobserved hand detail unimportant.',
        'accepted_as_rule': False,
        'reason': 'Word/example provenance supplies neither every-frame shape, visibility, mandatory phase nor permissible-variation truth. The same-endpoint masking counterexample shows a short observed state can disappear under context completion.',
        'allowed_use': 'Source provenance and reference-anchor context; temporal support stays explicitly inferred and never overwrites direct or unknown labels.',
        'forbidden_conversions': ['unobserved=>unimportant', 'open_four=0=>fist', 'same endpoints=>constant middle', 'normal crossing=>any model output correct', 'low activity=>noncore', 'height relation=>contact']},
    'minimal_evidence_schema': {
        'identity': ['clip_id', 'source_sha256', 'side', 'original_frame_ids', 'original_pts_us'],
        'scope': ['feature_id', 'feature_definition_sha256', 'frame_or_window', 'window_bounds', 'center_frame_if_applicable', 'rgb_context_bounds'],
        'observation': ['direct_value:0|1|null', 'direct_provenance', 'human_verified:false|true', 'model_probability:null|number', 'model_checkpoint_sha256', 'model_threshold', 'temporal_value:0|1|null', 'temporal_anchor_ids', 'continuity_status', 'transition_conflict'],
        'alignment': ['saved_global_path_sha256', 'reference_core_full_bounds', 'all_query_links', 'samephase_evidence_status', 'occurrence_status', 'alignment_flags'],
        'guide_context': ['clause_id', 'source_span', 'source_url', 'reference_anchor_ids', 'scope:reference_context_only', 'training_target:false'],
        'support': ['expected_reference_frames', 'possible_components', 'reference_valid_components', 'paired_valid_components', 'reference_missing', 'query_missing', 'unknown_in_denominator'],
        'semantic': ['linguistic_importance:null', 'educational_correct:null', 'educational_error:null', 'required_repeat_count:null'],
        'constraint': 'Prediction is a separate evidence type, never direct observation. A 13-frame window outcome cannot be replicated as 13 framewise truths.'},
    'shortest_implementable_link': [
        'Reuse OBSERVATION_EXPORT schema for validated physical predictions, preserving model provenance and exact frame/window support; do not feed guide wording or word ID as labels.',
        'Join to description REPORT_DATA by exact source clip/frame/PTS and side; use saved stages[].path/reference_core without re-alignment, cropping or changed denominators.',
        'Attach existing source_cards only at validated reference anchors; retain alignment flags, transition conflicts, repetition ambiguity and all missing components.',
        'Emit measurement supported/partial/unobserved plus prediction evidence. No instructional correction or correct/incorrect sign verdict; no new HTML required.'],
    'missing_truth': [
        'Independent human/KSL correctness and linguistic importance of specific frames, holds or transitions.',
        'Physical labels for real unobserved frames; temporal inference does not supply them.',
        'Fine complete handshape, thumb/fist and exact contact/pressure truth.',
        'Required repetition count, allowable personal variation and error-correction boundaries.',
        'Fresh person/expression evaluation and sufficient rare-positive/negative crossing and transition support.',
        'Automatic full-video phase truth and exact framewise truth for current window-trained outputs.'],
    'no_training_readiness_claim': True,
}
(HERE / 'CORE_CONTEXT_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
text = f'''기존 공개 핵심·문맥 근거 감사 — {report['status']}

저장 메타데이터 {len(sources)}파일을 읽어 {report['check_count']:,}개 불변조건을 확인했다. 실패 {len(failures)}개. 새 영상 주석·학습·추론·점수 재실행·HTML·개인자료 접근은 없다. 같은 Codex의 자료 검산이며 독립 사람의 수어 정답 검증은 아니다.

재사용 가능: 손모양 순서 묶음 {len(inventory)}개 창/{counts['frames']}원본프레임/{counts['feature_samples']}특징셀. 직접 관측초안 {counts['known_samples']}, 미확인 {counts['unknown_samples']}. 변화 경계 {counts['bounded_feature_change']}개, 같은 끝모양 사이 미확인 {counts['same_endpoints_unresolved_gap']}개, 전 창 미확인 특징 {counts['entire_window_unobserved']}개를 보존했다. 검지단독폄/네손가락폄 두 특징이며 주먹·전체손모양 정답이 아니다.

시간 문맥의 한계: 과거 한 셀씩 별도 마스킹한 279시험에서 앞뒤 규칙C는 초안일치 {temporal_stats['masked']['C']['matches_hidden_draft']}·충돌 {temporal_stats['masked']['C']['conflicts_hidden_draft']}·보류 {temporal_stats['masked']['C']['abstained']}였다. 같은 영상의 인접 셀은 독립표본이 아니다. 앞뒤 모양이 같아도 중간의 짧은 변화가 사라지는 반례가 있으므로 관측불가를 중요하지 않음/같은모양으로 채우지 않는다. 새 guard의348셀은 기존259일치/89보류에서39일치/309보류로 바뀌어 하드차단 채택 근거도 없다.

단계 근거: 공개16영상의36쌍117참고구간을 재사용할 수 있다. 다른 수행자 개발39구간의 고정440참고프레임 분모에서 기존 전체DTW 같은 물리단계67.2%, 핵심우선64.1%였다. 전체DTW와 경계/회차 이견을 유지한다. 이 물리 초안은 수어 정오·필수 반복의 정답이 아니다.

설명 연결: {len(catalog)}개 출처 항목과 {desc_counts['pairs']}쌍/{desc_counts['stages']}구간, 저장 경로·원본 비교 분모를 확인했다. 설명은 참고 앵커의 문맥이며 다른 수행자의 프레임 정답이 아니다. 공식 수어 영상이라는 사실은 보이지 않는 손의 세부가 중요하지 않다는 근거가 아니다.

가장 짧은 구현 경로: shape_sequence_bundle/OBSERVATION_EXPORT.json 형식에 검증된 특징 예측을 별도 근거종류로 기록 → 원본 clip/frame/PTS/좌우로 description_evidence_feedback/REPORT_DATA.json의 기존 stages.path와 연결 → source_cards의 참고 앵커 문맥과 관측/정렬 부족을 함께 출력. 13프레임창의 결과를 모든 프레임에 복사하지 않는다. JSON의 minimal_evidence_schema/shortest_implementable_link에 정확한 필드를 적었다.

남은 정답: 독립 수어 중요도/정오, 실제 미관측 구간의 세부 손모양, 주먹·엄지·접촉, 필수 반복/허용변이, 희소양성·교차·전이의 충분한 미사용 평가, 자동 전체영상 단계 정답. 현재 근거를 새 학습 정답이나 새 평가로 승격하지 않는다.
'''
(HERE / 'CORE_CONTEXT_AUDIT_KO.txt').write_text(text, encoding='utf-8')
print(json.dumps({'status': report['status'], 'checks': report['check_count'], 'failures': failures[:8], 'sources': len(sources), 'counts': dict(counts)}, ensure_ascii=False))
raise SystemExit(1 if failures else 0)
