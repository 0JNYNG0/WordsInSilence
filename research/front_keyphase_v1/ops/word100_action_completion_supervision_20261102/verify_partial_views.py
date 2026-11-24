"""Saved-ledger verification, independent of the view constructor.

No model, video transformation, new image, annotation, or compiler import.
"""
from pathlib import Path
from fractions import Fraction
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import traceback
import numpy as np
from PIL import Image

U = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def spans(values):
    """Group integer runs via value-position differences, not constructor code."""
    sorted_values = sorted(set(values))
    out = []
    for _, group in itertools.groupby(enumerate(sorted_values), lambda x: x[1] - x[0]):
        run = [pair[1] for pair in group]
        out.append([run[0], run[-1]])
    return out


def main():
    destination = U / 'PARTIAL_VIEWS_VERIFIED.json'
    assert not destination.exists(), 'Preserve completed verification'
    manifest_path = U / 'PARTIAL_VIEWS_MANIFEST.json'
    manifest = load(manifest_path)
    targets = {row['id']: row for row in load(U / 'PROCESS_TARGETS.json')['rows']}
    original = {row['id']: row for row in load(U / 'PROCESS_REVIEW_MANIFEST.json')['rows']}
    intent_path = U / 'PARTIAL_VIEWS_INTENT.json'
    assert digest(intent_path) == manifest['intent_sha256']
    intent = load(intent_path)
    intent_views = {row['view_id']: row for row in intent['views']}
    assert manifest['pins'] == intent['pins']
    verified_files = {}

    def file_check(path, expected):
        if path not in verified_files:
            verified_files[path] = digest(path)
        assert verified_files[path] == expected, ('hash', path)

    for path, expected in manifest['pins'].items():
        file_check(path, expected)
    contract = load(U / 'PARTIAL_CONTROL_CONTRACT.json')
    for pin in contract['inputs']:
        file_check(pin['path'], pin['sha256'])
    assert manifest['status'] == 'PREPARED_PENDING_DIRECT_VIEW_REVIEW'
    assert manifest['actual_transformed_videos_prepared'] == 0
    assert manifest['new_inference'] == manifest['new_training'] == 0

    eligible = {}
    for qid, target in targets.items():
        seq = target['sequence_links'][0]
        frames = [r['frame'] for r in original[qid]['frames']]
        lo, hi = min(frames), max(frames)
        assert frames == list(range(lo, hi + 1))
        assert target['review_bounds'] == [lo, hi]
        if not seq['setup_frames'] or not seq['terminal_frames']:
            continue
        a, b = max(seq['setup_frames']), min(seq['terminal_frames'])
        if (target['review_bounds_sufficient'] is True and seq['visible_chain'] == 'yes'
                and lo <= a < b <= hi and b - a > 1):
            eligible[qid] = (lo, hi, a, b)
    assert len(eligible) == 44 and len(targets) == 45
    sentinel_ids = set()
    for physical_query in {r['physical_query'] for r in targets.values()}:
        for person in [1, 9]:
            matching = sorted(qid for qid in eligible
                if targets[qid]['physical_query'] == physical_query and targets[qid]['person'] == person)
            assert matching
            sentinel_ids.add(matching[0])
    assert len(sentinel_ids) == 10 and sentinel_ids == set(intent['sentinels'])
    expected_keys = {(qid, 'native') for qid in targets}
    expected_keys.update((qid, kind) for qid in eligible for kind in ('prefix', 'suffix', 'middle_splice'))
    expected_keys.update((qid, kind) for qid in sentinel_ids
        for kind in ('speed_0_8', 'speed_1_25', 'intact_context_trim', 'missing_rgb'))
    assert {(v['query_id'], v['construction_kind_audit_only']) for v in manifest['views']} == expected_keys
    assert len(expected_keys) == len(manifest['views']) == 217
    query_sources = defaultdict(set)
    signature_groups = defaultdict(list)
    page_registry = {}
    queries_with_variants = []
    counts = Counter()
    map_entries = 0
    blank_tiles = 0
    max_time_delta = 0
    for view in manifest['views']:
        qid = view['query_id']
        target = targets[qid]
        src = original[qid]
        seq = target['sequence_links'][0]
        kind = view['construction_kind_audit_only']
        frame_lookup = {row['frame']: row for row in src['frames']}
        all_frames = sorted(frame_lookup)
        lo, hi = all_frames[0], all_frames[-1]
        a = max(seq['setup_frames']) if seq['setup_frames'] else None
        b = min(seq['terminal_frames']) if seq['terminal_frames'] else None
        assert view['a'] == a and view['b'] == b
        assert view['reviewed_bounds'] == [lo, hi]
        assert view['recorded_person_id'] == target['person'] in range(1, 10)
        assert view['role'] == target['role'] == 'fit'
        assert view['clip_id'] == target['clip_id'] == src['clip_id']
        assert view['physical_query'] == target['physical_query']
        assert view['source_sha256'] == target['source_sha256'] == src['source_sha256']
        assert Path(view['source_path']).resolve() == Path(src['source']).resolve()
        file_check(view['source_path'], view['source_sha256'])
        assert view['original_review_path'] == target['source_annotation']
        assert view['original_review_sha256'] == target['source_annotation_sha256']
        file_check(view['original_review_path'], view['original_review_sha256'])
        assert view['labels'] is None and view['labels_status'] == 'pending_direct_review'
        assert view['actual_retimed_video_prepared'] is False
        assert view['model_input_prepared'] is False and view['review_overlay_not_model_input'] is True
        assert view['original_scope_variant'] == target.get('actual_terminal_variant')
        if view['original_scope_variant'] is not None:
            queries_with_variants.append((qid, kind))
        initial = {k: v for k, v in view.items() if k not in ('review_pages', 'native_comparison_pages')}
        assert initial == intent_views[view['view_id']], 'Saved intent changed during rendering'
        expected_ids = all_frames
        ratio = Fraction(1)
        blank = set()
        if kind == 'prefix':
            expected_ids = [f for f in all_frames if f <= a]
        elif kind == 'suffix':
            expected_ids = [f for f in all_frames if f >= b]
        elif kind == 'middle_splice':
            expected_ids = [f for f in all_frames if f <= a or f >= b]
        elif kind == 'speed_0_8':
            ratio = Fraction(4, 5)
        elif kind == 'speed_1_25':
            ratio = Fraction(5, 4)
        elif kind == 'intact_context_trim':
            anchors = seq['setup_frames'] + seq['motion_interval'] + seq['terminal_frames']
            expected_ids = [f for f in all_frames if min(anchors) <= f <= max(anchors)]
        elif kind == 'missing_rgb':
            candidates = sorted((m['start'], m['end']) for m in target['motion_intervals']
                if m['label'] == 1 and seq['motion_interval'][0] <= m['start'] < m['end'] <= seq['motion_interval'][1])
            assert candidates
            blank = set(range(candidates[0][0], candidates[0][1] + 1))
        else:
            assert kind == 'native'
        assert Fraction(str(view['speed'])) == ratio
        mapping = view['ordered_frame_map']
        assert [row['source_native_frame'] for row in mapping] == expected_ids
        assert [row['output_index'] for row in mapping] == list(range(len(expected_ids)))
        gaps = [i for i in range(1, len(expected_ids)) if expected_ids[i] - expected_ids[i - 1] != 1]
        assert view['seam_output_indices'] == gaps
        assert view['retained_contiguous_native_fragments'] == spans(expected_ids)
        assert view['removed_native_spans'] == spans(set(all_frames) - set(expected_ids))
        diffs = sorted(frame_lookup[f + 1]['timestamp_us'] - frame_lookup[f]['timestamp_us'] for f in all_frames[:-1])
        center = len(diffs) // 2
        period = Fraction(diffs[center]) if len(diffs) % 2 else Fraction(diffs[center - 1] + diffs[center], 2)
        for i, record in enumerate(mapping):
            f = record['source_native_frame']
            original_frame = frame_lookup[f]
            assert record['source_native_timestamp_us'] == original_frame['timestamp_us']
            assert record['source_frame_path'] == original_frame['path']
            assert record['source_frame_sha256'] == original_frame['sha256']
            file_check(record['source_frame_path'], record['source_frame_sha256'])
            assert record['pixel_operation'] == ('blackout' if f in blank else 'copy')
            assert type(record['image_available']) is bool
            assert record['image_available'] is (f not in blank)
            # Independent closed expression: native elapsed time minus removed
            # source durations, adding one display period per edit seam.
            elapsed = Fraction(original_frame['timestamp_us'] - frame_lookup[expected_ids[0]]['timestamp_us'])
            removed_time = sum(Fraction(frame_lookup[expected_ids[g]]['timestamp_us']
                - frame_lookup[expected_ids[g - 1]]['timestamp_us']) - period for g in gaps if g <= i)
            expected_time = round((elapsed - removed_time) / ratio)
            delta = abs(record['view_timestamp_us'] - expected_time)
            max_time_delta = max(max_time_delta, delta)
            assert delta <= 1
            if i:
                assert record['view_timestamp_us'] > mapping[i - 1]['view_timestamp_us']
            map_entries += 1
        assert mapping[0]['view_timestamp_us'] == 0
        signature = (view['source_sha256'], tuple((r['source_native_frame'], r['pixel_operation'], r['view_timestamp_us']) for r in mapping))
        signature_groups[signature].append(view['view_id'])
        payload = dict(source_sha256=view['source_sha256'], frames=list(signature[1]))
        assert compact(payload) == view['pixel_view_key']
        assert compact([view['pixel_view_key'], target['physical_query'], compact(target['predicate_definitions'])]) == view['query_review_key']
        assert len(view['motion_retention_audit_only']) == len(target['motion_intervals'])
        kept = set(expected_ids)
        for audit, annotation in zip(view['motion_retention_audit_only'], target['motion_intervals']):
            full = set(range(annotation['start'], annotation['end'] + 1))
            remaining = full.intersection(kept)
            assert audit['original_interval'] == [annotation['start'], annotation['end']]
            assert audit['original_label_audit_only'] == annotation['label']
            assert audit['retained_native_fragments'] == spans(remaining)
            assert audit['removed_native_spans'] == spans(full - remaining)
            expected_retention = 'wholly_removed' if not remaining else 'fully_retained' if len(remaining) == len(full) else 'partially_retained'
            assert audit['retention'] == expected_retention
            assert audit['remainder_label'] is None
            assert audit['blackout_frames'] == sorted(remaining & blank)
        assert view['native_comparison_pages'] == src['pages']
        displayed_indices = []
        for page in view['review_pages']:
            file_check(page['path'], page['sha256'])
            key = page['path']
            assert key not in page_registry or page_registry[key] == page
            page_registry[key] = page
            displayed_indices.extend(page['output_indices'])
            assert page['source_native_frames'] == [expected_ids[j] for j in page['output_indices']]
            # Read existing rendered pixels only; no image creation or edit.
            black_locations = [j for j, output_index in enumerate(page['output_indices'])
                if not mapping[output_index]['image_available']]
            if black_locations:
                with Image.open(page['path']) as image:
                    pixels = np.asarray(image)
                    for j in black_locations:
                        x = (j % 5) * 270 + 4
                        y = 76 + (j // 5) * 212 + 29
                        patch = pixels[y + 20:y + 150, x + 20:x + 240].astype(float)
                        assert patch.size and patch.std() < 2 and abs(patch.mean() - 128) < 2
                        blank_tiles += 1
        assert displayed_indices == list(range(len(mapping)))
        query_sources[qid].add((view['clip_id'], view['recorded_person_id'], view['source_sha256']))
        counts[kind] += 1
    assert counts == Counter(manifest['counts'])
    assert len(signature_groups) == manifest['unique_pixel_time_views'] == 216
    assert len(page_registry) == manifest['pages'] == 325
    assert len(query_sources) == 45 and all(len(v) == 1 for v in query_sources.values())
    assert len({v['clip_id'] for v in manifest['views']}) == 36
    skipped = manifest['skipped']
    assert len(skipped) == 1
    omitted = skipped[0]['query_id']
    assert omitted not in eligible and skipped[0]['original_chain'] == targets[omitted]['sequence_links'][0]
    assert queries_with_variants == [(omitted, 'native')]
    assert [v['construction_kind_audit_only'] for v in manifest['views'] if v['query_id'] == omitted] == ['native']
    duplicate_groups = [ids for ids in signature_groups.values() if len(ids) > 1]
    result = dict(status='PASS_SAVED_VIEW_CONSTRUCTION_AND_PROVENANCE_NOT_VISUAL_TARGETS',
        at=datetime.now(timezone.utc).isoformat(), manifest_sha256=digest(manifest_path),
        verifier_sha256=digest(__file__), views=217, unique_pixel_time_views=216, pages=325,
        query_count=45, source_clips=36, registered_fit_ids=list(range(1, 10)),
        kind_counts=dict(counts), map_entries_checked=map_entries,
        native_and_view_time_axes_separate=True, maximum_view_time_rounding_difference_us=max_time_delta,
        primary_anchor_formulas_verified=True, motion_remainders_unlabelled=True,
        blackout_tiles_checked_for_neutral_pixels=blank_tiles,
        source_split_preserved=True, original_scope_variant_native_preserved=omitted,
        duplicate_alias_groups=duplicate_groups, file_hashes_checked=len(verified_files),
        current_view_labels_all_null=True, new_view_label_count=0,
        actual_transformed_video_count=0, model_inputs_prepared=0,
        speed_scope='Intact frame-order and retimed ledger/sheets verified only; no actual retimed video playback or model speed-invariance evaluation was performed.',
        blackout_scope='Neutral review tiles and image_available=false verified. Future actual transformed video extraction must enforce missing-image handling; no future cached-feature leakage check is claimed here.',
        semantic_scope='Ready for direct physical-view review, not accepted positives/negatives or educational omission truth. A cut may retain another complete occurrence; unknown and variants remain unlabelled.',
        new_inference=0, new_training=0, new_images_created_by_verifier=0,
        new_labels=0, dev_accessed=False, whole_goal_complete=False)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        failure = U / 'PARTIAL_VIEWS_VERIFY_FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(), encoding='utf-8')
        raise
