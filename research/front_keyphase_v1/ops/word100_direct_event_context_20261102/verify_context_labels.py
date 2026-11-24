"""Independent saved-artifact audit. No compiler import, images opened or models run."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import traceback

U = Path(__file__).resolve().parent
AE = U.parent / 'word100_guide_feature_supervision_20261102'
D = U.parent / 'word100_action_episode_supervision_20261102'
CHECKS = Counter()
PINS = {}


def read(path):
    path = Path(path)
    PINS[str(path)] = digest(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(value, category, detail):
    CHECKS[category] += 1
    if not value:
        raise AssertionError(f'{category}: {detail}')


def pin(path, expected):
    actual = digest(path)
    check(actual == expected, 'hash', str(path))
    PINS[str(path)] = actual


def pointer(document, value):
    for token in value.split('/')[1:]:
        token = token.replace('~1', '/').replace('~0', '~')
        document = document[int(token)] if isinstance(document, list) else document[token]
    return document


def category(value):
    return 'unknown' if value is None else 'applicable_context' if value == 1 else 'other_context'


def main():
    destination = U / 'CONTEXT_VERIFIED.json'
    assert not destination.exists(), 'Do not overwrite a completed verification.'
    output = read(U / 'CONTEXT_LABELS.json')
    queue = read(AE / 'CONTEXT_REVIEW_QUEUE.json')
    timing = read(U / 'NATIVE_TIME_AUDIT.json')
    media = read(D / 'REVIEW_MANIFEST.json')
    scope = read(U / 'SCOPE.json')
    first_failure = read(U / 'COMPILE_FIRST_FAILURE.json')
    initial_code = U / first_failure['initial_code']
    PINS[str(initial_code)] = digest(initial_code)
    check(first_failure['labels_changed'] is False and first_failure['output_existed_at_failure'] is False,
          'initial_failure_preserved', 'Initial compiler failure made no result/label edits')
    check(initial_code.read_bytes() != (U / 'compile_context.py').read_bytes(),
          'initial_failure_preserved', 'Initial and corrected compiler retained separately')
    for doc in (output, queue, timing):
        for path, value in doc['pins'].items():
            pin(path, value)

    packets = {r['clip_id']: r for r in queue['rows']}
    expected = {}
    original_claims = {}
    for packet in queue['rows']:
        check(packet['role'] == 'fit' and packet['registered_id'] in (1, 4, 7), 'scope', packet['clip_id'])
        for claim in packet['claims']:
            check(claim['claim_id'] not in original_claims, 'completeness', 'Unique queued claim')
            original_claims[claim['claim_id']] = (packet, claim)
            check(len(claim['witness_frames']) == len(claim['witness_timestamps_us']), 'axis', claim['claim_id'])
            check(claim['copied_physical_label'] is False and claim['dense_label_propagation'] is False,
                  'no_label_promotion', claim['claim_id'])
            for frame, native_time in zip(claim['witness_frames'], claim['witness_timestamps_us']):
                key = f"{claim['claim_id']}:f{frame}"
                check(key not in expected, 'completeness', key)
                expected[key] = (packet, claim, frame, native_time)
    check(len(original_claims) == queue['claims'] == output['claims'] == 39, 'counts', 'Claims')
    check(len(expected) == queue['witness_queries'] == 129, 'counts', 'Queued point queries')
    by_id = {r['id']: r for r in output['rows']}
    check(len(by_id) == len(output['rows']) and set(by_id) == set(expected), 'completeness', 'Exact compiled query set')
    reviews = {str((U / f'CONTEXT_{w}.json').resolve()): read(U / f'CONTEXT_{w}.json')
               for w in ('0739', '1133', '1567', '2523')}
    clip_reviews = {}
    claim_reviews = {}
    for path, review in reviews.items():
        for ri, row in enumerate(review['rows']):
            check(row['clip_id'] not in clip_reviews, 'completeness', 'Unique reviewed clip')
            clip_reviews[row['clip_id']] = (path, ri, row)
            for ci, claim in enumerate(row['claims']):
                check(claim['claim_id'] not in claim_reviews, 'completeness', 'Unique reviewed claim')
                claim_reviews[claim['claim_id']] = (path, ri, ci, claim)
    check(set(clip_reviews) == set(packets) and len(clip_reviews) == 12, 'completeness', 'All 12 clips reviewed')
    check(set(claim_reviews) == set(original_claims), 'completeness', 'All 39 claims reviewed')
    metadata = {r['clip_id']: r for r in media['rows'] if r['clip_id'] in packets}
    compiled_clips = {r['clip_id']: r for r in output['clips']}
    check(set(compiled_clips) == set(packets) and len(output['clips']) == 12, 'completeness', 'Compiled clip inventory')
    cells = 0
    zooms = 0
    variant_records = []
    for cid, (path, ri, row) in clip_reviews.items():
        packet = packets[cid]
        md = metadata[cid]
        sheet = row.get('sheet', row.get('reviewed_sheet'))
        check({k: sheet[k] for k in ('path', 'sha256')} == packet['sheet'], 'sheet', cid)
        pin(sheet['path'], sheet['sha256'])
        if 'actual_sha256' in sheet:
            check(sheet['actual_sha256'] == sheet['sha256'], 'sheet', 'Extra audit field is consistent')
        frames = row['actual_displayed_frames']
        check(frames == [f['frame'] for f in md['frames']], 'viewed_range', cid)
        check(frames == sorted(set(frames)) and all(type(f) is int and f >= 0 for f in frames), 'viewed_range', cid)
        check(md['sheet'] == sheet['path'] and md['sheet_sha256'] == sheet['sha256'], 'sheet', 'Original sheet identity')
        check(row['role'] == 'fit' and row['person'] == packet['registered_id'] and row['word_id'] == packet['word_id'], 'scope', cid)
        view_count = row.get('sheet_display_count', row.get('sheet_view_count', row.get('sheet_open_count_this_pass')))
        check(view_count == 1 and row['reviewer'], 'review_record', cid)
        check(row['zoom_views'] == [], 'viewed_range', 'No new zooms in this bounded review')
        cp = compiled_clips[cid]
        check(cp['displayed_frames'] == frames and cp['sheet'] == sheet and cp['zoom_views'] == row['zoom_views'], 'preservation', 'Reviewed display metadata')
        check(cp['review_path'] == path and cp['review_row'] == ri, 'provenance', cid)
        cells += len(frames)
        zooms += len(row['zoom_views'])

    preserved_point_fields = ('frame', 'timestamp_us', 'context', 'reason_ko',
                              'independent_body_or_trajectory_evidence', 'preceding_frames', 'following_frames')
    for key, (packet, claim, frame, native_time) in expected.items():
        row = by_id[key]
        provenance = row['provenance']
        path, ri, ci, reviewed_claim = claim_reviews[claim['claim_id']]
        review_document = reviews[path]
        check(provenance['path'] == path and provenance['sha256'] == PINS[path], 'provenance', key)
        pi = next((i for i, p in enumerate(reviewed_claim['context_points']) if p['frame'] == frame), None)
        check(pi is not None and provenance['pointer'] == f'/rows/{ri}/claims/{ci}/context_points/{pi}', 'provenance', key)
        original = pointer(review_document, provenance['pointer'])
        check(all(row[k] == original[k] for k in preserved_point_fields), 'preservation', key)
        check(row['frame'] == frame and row['timestamp_us'] == native_time and type(frame) is int and type(native_time) is int,
              'axis', key)
        check(row['clip_id'] == packet['clip_id'] and row['word_id'] == packet['word_id'] and row['person'] == packet['registered_id'] and row['role'] == 'fit', 'scope', key)
        check(row['clause_id'] == claim['clause_id'] and row['observed_query_kind'] == claim['query_kind'], 'source', key)
        check(row['source'] == reviewed_claim['source'] == claim['source'] and row['source_text'] == reviewed_claim['source_text'] == claim['source_text'], 'source', key)
        pin(row['source']['path'], row['source']['sha256'])
        value = row['context']
        check(value is None or type(value) is int and value in (0, 1), 'label_type', key)
        check(type(row['mask']) is bool and row['mask'] == (value is not None), 'unknown_mask', key)
        check(original['tested_feature_used_to_assign_context'] is False, 'no_circular_promotion', key)
        check(original['reason_ko'] and original['independent_body_or_trajectory_evidence'], 'review_record', key)
        actual = clip_reviews[packet['clip_id']][2]['actual_displayed_frames']
        check(frame in actual and all(x in actual and x < frame for x in row['preceding_frames']) and all(x in actual and x > frame for x in row['following_frames']), 'viewed_range', key)
        check(all(row[k] is None for k in ('whole_sign_correctness', 'educational_error', 'exact_event_boundary_truth')),
              'no_label_promotion', key)
        check(all(original.get(k) is None for k in ('whole_sign_correctness', 'educational_error', 'exact_event_boundary_truth')),
              'no_label_promotion', 'Original semantic non-labels')
        check('y' not in row and 'source_action_match' not in row and 'physical_truth' not in row,
              'no_label_promotion', 'No old physical y or full source match created')
    for claim_id, (path, ri, ci, claim) in claim_reviews.items():
        original = original_claims[claim_id][1]
        check([p['frame'] for p in claim['context_points']] == original['witness_frames'], 'completeness', claim_id)
        check(claim.get('source_variant_status') is not None and claim.get('local_context_definition'), 'source_variant', claim_id)
        variant_records.append(dict(claim_id=claim_id, review_path=path, review_sha256=PINS[path],
                                    claim_pointer=f'/rows/{ri}/claims/{ci}',
                                    source_variant_status=claim['source_variant_status']))

    stats = Counter(category(row['context']) for row in output['rows'])
    per_clause = defaultdict(Counter)
    person_stats = defaultdict(lambda: defaultdict(Counter))
    for row in output['rows']:
        per_clause[row['clause_id']][category(row['context'])] += 1
        person_stats[row['clause_id']][str(row['person'])][category(row['context'])] += 1
    check(dict(stats) == output['context_counts'] == {'applicable_context': 52, 'other_context': 12, 'unknown': 65}, 'counts', 'Preserved context counts')
    check({k: dict(v) for k, v in per_clause.items()} == output['per_clause'], 'counts', 'Per clause')
    check(output['context_queries'] == len(output['rows']) == 129 and output['unique_clip_frames'] == len({(r['clip_id'], r['frame']) for r in output['rows']}) == 86, 'counts', '129 queries at 86 unique clip frames')
    check(output['actual_sheet_cells'] == cells == 369 and output['actual_zoom_views'] == zooms == 0, 'counts', 'Actual review scope')
    for clause_id, pstats in person_stats.items():
        ready = output['readiness'][clause_id]
        paired = sorted(int(p) for p, s in pstats.items() if s.get('applicable_context', 0) > 0 and s.get('other_context', 0) > 0)
        check(ready['per_person'] == {p: dict(s) for p, s in pstats.items()}, 'readiness', clause_id)
        check(sorted(ready['records_with_both_contexts']) == paired and ready['both_context_coverage_condition_met'] == (len(paired) >= 2), 'readiness', clause_id)
        check(ready['training_ready'] is False and ready['observed_feature_wrong_in_same_context_control_proven'] is False and ready['model_shortcut_robustness_tested'] is False, 'readiness', 'Coverage is not training release')

    check(timing['clips'] == len(timing['rows']) == 12 and {r['clip_id'] for r in timing['rows']} == set(packets), 'saved_time', '12 source/cache axis audits')
    affected = []
    max_delta = 0
    for row in timing['rows']:
        cid = row['clip_id']
        qtimes = {f: tt for c in packets[cid]['claims'] for f, tt in zip(c['witness_frames'], c['witness_timestamps_us'])}
        witnesses = {w['frame']: w for w in row['witnesses']}
        check(set(witnesses) == set(qtimes) and len(witnesses) == len(row['witnesses']), 'saved_time', cid)
        check(row['cached_axis_matches_media'] is True and row['queue_axis_matches_cache'] is True and row['max_abs_cache_media_difference_us'] == 0, 'saved_time', cid)
        check(row['source'] == metadata[cid]['source'] and row['source_sha256'] == metadata[cid]['source_sha256'], 'saved_time', 'Original source identity')
        pin(row['native_input'], row['native_input_sha256'])
        old_display = {f['frame']: f['seconds'] for f in metadata[cid]['frames']}
        deltas = []
        for f, w in witnesses.items():
            check(0 <= f < row['frames'] and w['queue_timestamp_us'] == qtimes[f] == w['cached_timestamp_us'] == w['decoded_media_pts_us'], 'saved_time', f'{cid}:{f}')
            check(w['old_sheet_seconds'] == old_display[f], 'saved_time', 'Displayed seconds preserved')
            delta = int(round(w['old_sheet_seconds'] * 1_000_000)) - w['decoded_media_pts_us']
            check(w['old_sheet_minus_media_us'] == delta, 'saved_time', 'Saved display delta arithmetic')
            deltas.append(abs(delta))
        check(max(deltas) == row['max_abs_sheet_display_difference_us'], 'saved_time', 'Per-clip maximum')
        max_delta = max(max_delta, max(deltas))
        if max(deltas) > 1:
            affected.append(cid)
    check(len(affected) == 2 and max_delta == 11921, 'saved_time', 'Two display-only discrepancies')
    check(output['unknown_is_not_negative'] is True and output['no_dense_label_propagation'] is True and output['whole_goal_complete'] is False and output['independent_KSL_expert_validation'] is False,
          'no_label_promotion', 'Global scope retained')
    check(output['new_model_inference'] == output['new_training'] == 0, 'no_model_execution', 'Compiler has no model results')
    for path, value in list(PINS.items()):
        check(digest(path) == value, 'unchanged_after_audit', path)
    circular_count = sum(per_clause[c]['unknown'] for c in ('obedience_lateral_arm_then_lower', 'well_brush', 'ancestor_alternation'))
    result = dict(
        status='PASS_SAVED_CONTEXT_LABELS_AND_PROVENANCE_NO_TRAINING_RELEASE',
        utc=datetime.now(timezone.utc).isoformat(),
        context_labels_sha256=PINS[str(U / 'CONTEXT_LABELS.json')],
        checks=sum(CHECKS.values()), checks_by_category=dict(CHECKS),
        clips=12, claims=39, context_queries=129, unique_clip_frames=86,
        actual_reported_sheet_cells=cells, actual_reported_zooms=zooms,
        context_counts=dict(stats), per_clause={k: dict(v) for k, v in per_clause.items()},
        source_variant_preservation=dict(mode='Exact source candidate in every flat row; full unresolved variant status remains in SHA-pinned parent claim',
                                         requirement='A later consumer must dereference the parent claim; context1 never confirms the full source variant.',
                                         claims=variant_records),
        prior_physical_labels=dict(policy='Not parsed or used as new context truth; frozen queue dependency hashes remain unchanged.',
                                   original_y_modified=False, context_is_not_physical_y=True),
        time_audit=dict(mode='Independent saved metadata arithmetic and axis/cache hash check, not a second decode',
                        verified_saved_clip_axes=12, exact_saved_cache_and_media_pts=True,
                        affected_display_clips=affected, maximum_display_difference_seconds=max_delta / 1e6,
                        new_decode=0, label_retiming=0),
        interpretation=dict(unknown_does_not_mean_invisible_video=True,
                            directly_identified_motion_context_circularity_queries=circular_count,
                            egypt_lower_mixed_motion_applicability_and_variant_unknown=per_clause['egypt_lower_shared']['unknown'],
                            egypt_gather_variant_and_feature_dependent_unknown=per_clause['egypt_gather_suffix']['unknown'],
                            note='At least48/65 unknown queries concern independent eligibility for the very motion being tested, although wider arm/body activity is visible. The other17 include variant and local-event uncertainty. No null is promoted.',
                            two_clause_context_coverage_is_not_shortcut_test=True,
                            training_released=False),
        first_compiler_failure=dict(preserved=True, artifact=str(U / 'COMPILE_FIRST_FAILURE.json'),
                                    issue='Optional actual_sha256 sheet metadata; no path/hash or label disagreement',
                                    original_code=str(initial_code)),
        limits=['Same Codex artifact verification; not independent human/KSL expert truth.',
                'This verifier did not re-open images. Actual viewing is attested by the four original reviews and checked against sheet/frame metadata.',
                'Sampled point eligibility is not a dense action interval, successful movement, source-clause truth, or educational correctness.',
                'No model accuracy or improvement is measured.'],
        new_model_inference=0, new_training=0, new_image_views=0, new_video_decode=0,
        whole_goal_complete=False,
        pins={**PINS, str(Path(__file__).resolve()): digest(__file__)})
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'checks', 'clips', 'claims', 'context_queries', 'context_counts')}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        failure = U / 'VERIFY_CONTEXT_FIRST_FAILURE.json'
        if not failure.exists():
            failure.write_text(json.dumps(dict(error=traceback.format_exc(), checks=dict(CHECKS),
                                              verifier_sha256=digest(__file__), inputs_modified=False),
                                          ensure_ascii=False, indent=2), encoding='utf-8')
        raise
