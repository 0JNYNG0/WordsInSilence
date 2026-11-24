"""Frozen interval-presence pilot audit. No training, inference or label repair."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json

W = Path(__file__).resolve().parent
HANDS = ('R', 'L')
PREDICATES = ('down', 'hold')
TARGETS = tuple(f'{h}_{p}' for h in HANDS for p in PREDICATES)
EVENTS = {'onset', 'offset', 'reversal', 'rotation', 'shape_change', 'hand_crossing'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def boolean_or_null(value):
    return value is None or type(value) is bool


def validate_interval(interval, frames):
    first, last = interval['first_frame'], interval['last_frame']
    require(type(first) is int and type(last) is int, 'Native frame endpoints must be integers, not bool/float')
    require(frames[0] <= first <= last <= frames[-1], 'Interval outside exact source window')
    return first, last


def intersection_length(a, b):
    return max(0, min(a['last_frame'], b['last_frame']) - max(a['first_frame'], b['first_frame']) + 1)


def affected(unknown, hand, predicate):
    return unknown['hand'] in (hand, 'both') and (
        unknown['component'] == 'identity' or
        (unknown['component'] == 'translation' and predicate in unknown['affected_predicates']))


def eligible(record, hand, predicate):
    value = record['presence'][hand][predicate]
    if value == 1:
        return any(s['in_roi'] is True for s in record['segments'][hand][predicate])
    return value == 0 and record['full13_position_observable'][hand] is True and record['full13_roi'][hand] is True


def validate_record(record, source):
    """Reject malformed types; return semantic contradictions without repairing raw labels."""
    require(record['alias'] == source['alias'], 'Opaque alias mismatch')
    require(record['actual_image_viewed'] is True, 'Actual source image review required')
    require(record['image_sha256'] == source['image_sha256'], 'Source image pin mismatch')
    frames = source['frame_ids']
    issues = []
    def issue(reason, hand, predicate=None, detail=None):
        issues.append(dict(reason=reason, hand=hand, predicate=predicate, detail=detail))
    for item in record['unknown_intervals']:
        validate_interval(item, frames)
        require(item['hand'] in ('R', 'L', 'both'), 'Unknown interval hand')
        require(item['component'] in ('translation', 'identity', 'orientation', 'shape'), 'Unknown component')
        if item['component'] == 'translation':
            ap = item['affected_predicates']
            require(type(ap) is list and len(ap) > 0 and len(set(ap)) == len(ap) and set(ap) <= set(PREDICATES), 'Explicit affected translation predicates')
    for event in record['events']:
        first, last = validate_interval(event, frames)
        require(event['hand'] in ('R', 'L', 'both') and event['kind'] in EVENTS, 'Event kind/hand')
        require(event['evidence'] in ('direct', 'uncertain') and type(event['bounded_transition']) is bool, 'Event evidence/bounded type')
        if event['bounded_transition']:
            require(event['evidence'] == 'direct' and first > frames[0] and last < frames[-1], 'Bounded event needs directly observed before/after context')
    for hand in HANDS:
        require(boolean_or_null(record['full13_position_observable'][hand]), 'Full13 observation must be bool/null')
        require(boolean_or_null(record['full13_roi'][hand]), 'Full13 ROI must be bool/null')
        for predicate in PREDICATES:
            value = record['presence'][hand][predicate]
            require(value is None or (type(value) is int and value in (0, 1)), 'Presence must be exact int0/int1/null')
            segments = record['segments'][hand][predicate]
            require(type(segments) is list, 'Segments must be lists')
            for segment in segments:
                first, last = validate_interval(segment, frames)
                require(boolean_or_null(segment['in_roi']), 'Segment ROI must be bool/null')
                if last - first + 1 < (2 if predicate == 'down' else 3):
                    issue('positive_segment_too_short', hand, predicate, segment)
                for unknown in record['unknown_intervals']:
                    if affected(unknown, hand, predicate) and intersection_length(segment, unknown):
                        issue('positive_witness_overlaps_relevant_unknown', hand, predicate, dict(segment=segment, unknown=unknown))
                if record['full13_roi'][hand] is True and segment['in_roi'] is False:
                    issue('full13_roi_contradicts_segment_roi', hand, predicate, segment)
            if segments != sorted(segments, key=lambda s: (s['first_frame'], s['last_frame'])):
                issue('positive_segments_not_chronological', hand, predicate)
            if value == 1 and not segments:
                issue('positive_without_direct_witness', hand, predicate)
            if value != 1 and segments:
                issue('nonpositive_with_positive_witness', hand, predicate)
            if value == 0:
                if record['full13_position_observable'][hand] is not True:
                    issue('negative_without_full13_observation', hand, predicate)
                if any(affected(u, hand, predicate) for u in record['unknown_intervals']):
                    issue('negative_with_relevant_unknown', hand, predicate)
        for down in record['segments'][hand]['down']:
            for hold in record['segments'][hand]['hold']:
                if intersection_length(down, hold) >= 2:
                    issue('down_hold_conflicting_observed_intervals', hand, detail=dict(down=down, hold=hold))
        if record['full13_position_observable'][hand] is True and any(u['component'] == 'identity' and u['hand'] in (hand, 'both') for u in record['unknown_intervals']):
            issue('full13_identity_claim_contradicts_unknown', hand)
    return issues


def summarize(cells):
    n = len(cells)
    both = [c for c in cells if c['a'] is not None and c['b'] is not None]
    agreement = [c for c in both if c['a'] == c['b']]
    elig = [c for c in cells if c['a_eligible'] and c['b_eligible']]
    ea = [c for c in elig if c['a'] is not None and c['a'] == c['b']]
    counts = dict(all=n, both_known=len(both), both_unknown=sum(c['a'] is None and c['b'] is None for c in cells),
                  known_unknown_mismatch=sum((c['a'] is None) != (c['b'] is None) for c in cells),
                  known_conflicts=sum(c['a'] != c['b'] for c in both),
                  agreed_positive=sum(c['a'] == 1 for c in agreement), agreed_negative=sum(c['a'] == 0 for c in agreement),
                  a_eligible=sum(c['a_eligible'] for c in cells), b_eligible=sum(c['b_eligible'] for c in cells),
                  both_eligible=len(elig), eligible_agreed_known=len(ea),
                  eligible_agreed_positive=sum(c['a'] == 1 for c in ea), eligible_agreed_negative=sum(c['a'] == 0 for c in ea))
    counts.update(known_conflict_rate=counts['known_conflicts'] / len(both) if both else None,
                  known_unknown_mismatch_rate=counts['known_unknown_mismatch'] / n if n else None,
                  both_known_fraction=len(both) / n if n else None,
                  eligible_agreed_fraction=len(ea) / len(elig) if elig else None,
                  eligible_agreed_fraction_all=len(ea) / n if n else None)
    return counts


def support(cells, value, eligibility=False):
    matched = [c for c in cells if c['a'] == value and c['b'] == value and (not eligibility or (c['a_eligible'] and c['b_eligible']))]
    return {key: sorted({c[key] for c in matched}) for key in ('sample_id', 'clip_id', 'person', 'word_id')}


def calculate(manifest, reviews, gates):
    sources = manifest['rows']
    require(len(sources) == gates['windows'] == 24, 'Exactly fixed24 pilot windows')
    by = {r['sample_id']: r for r in sources}
    require(len(by) == 24, 'Unique pilot sources')
    require(len(reviews) == 2, 'Exactly two raw reviewers')
    maps = []
    for review in reviews:
        require(review['status'] == 'COMPLETE' and len(review['rows']) == 24, 'Complete24 reviewed rows')
        mapped = {r['sample_id']: r for r in review['rows']}
        require(len(mapped) == 24 and set(mapped) == set(by), 'Exact fixed review scope')
        maps.append(mapped)
    cells = {target: [] for target in TARGETS}
    failures, rows, patterns = [], [], {h: [] for h in HANDS}
    for source in sources:
        sid = source['sample_id']
        frames, times = source['frame_ids'], source['timestamps_us']
        require(len(frames) == len(times) == 13 and all(type(x) is int for x in frames + times), 'Exact13 integer frame/time IDs')
        require(all(b == a + 1 for a, b in zip(frames, frames[1:])) and all(b > a for a, b in zip(times, times[1:])), 'Native consecutive frames and increasing timestamps')
        require(source['center_frame'] == frames[6] and source['center_timestamp_us'] == times[6] and source['role'] == 'fit', 'Fixed centre identity and fit-only role')
        pair = [m[sid] for m in maps]
        local = [validate_record(r, source) for r in pair]
        for i, issues in enumerate(local):
            failures.extend(dict(sample_id=sid, reviewer=('A', 'B')[i], **issue) for issue in issues)
        row = dict(sample_id=sid, alias=source['alias'], frame_ids=frames, timestamps_us=times, predicates={}, ordered_evidence=[])
        for hand in HANDS:
            patterns[hand].append(dict(sample_id=sid, a=[pair[0]['presence'][hand][p] for p in PREDICATES], b=[pair[1]['presence'][hand][p] for p in PREDICATES]))
            for predicate in PREDICATES:
                key = f'{hand}_{predicate}'
                a, b = [r['presence'][hand][predicate] for r in pair]
                ea, eb = [eligible(r, hand, predicate) for r in pair]
                same = a is not None and a == b
                valid = not any(issue['hand'] == hand and issue['predicate'] in (None, predicate) for issues in local for issue in issues)
                cell = dict(sample_id=sid, clip_id=source['clip_id'], person=source['person'], word_id=source['word_id'],
                            ordinary=source['strata'] == ['hash_ordinary'], a=a, b=b, a_eligible=ea, b_eligible=eb)
                cells[key].append(cell)
                row['predicates'][key] = dict(a=a, b=b, a_eligible=ea, b_eligible=eb, agreed_value=a if same else None,
                                              model_label=a if same and ea and eb and valid else None, unknown_is_not_negative=True)
                for i, record in enumerate(pair):
                    for segment in record['segments'][hand][predicate]:
                        row['ordered_evidence'].append(dict(reviewer=('A', 'B')[i], hand=hand, predicate=predicate,
                            first_frame=segment['first_frame'], last_frame=segment['last_frame'],
                            first_timestamp_us=times[segment['first_frame'] - frames[0]], last_timestamp_us=times[segment['last_frame'] - frames[0]],
                            in_roi=segment['in_roi'], note=segment.get('note', '')))
        row['ordered_evidence'].sort(key=lambda x: (x['first_frame'], x['last_frame'], x['reviewer'], x['hand'], x['predicate']))
        rows.append(row)
    metrics = {k: summarize(v) for k, v in cells.items()}
    strata = {s: {k: summarize([c for c in v if c['ordinary'] == (s == 'ordinary')]) for k, v in cells.items()} for s in ('ordinary', 'challenge')}
    checks = dict(annotation_contract=not failures)
    for key, m in metrics.items():
        checks[key + '_known_conflict'] = m['known_conflict_rate'] is not None and m['known_conflict_rate'] <= gates['max_raw_known_conflict_rate_each_predicate']
        checks[key + '_unknown_mismatch'] = m['known_unknown_mismatch_rate'] <= gates['max_raw_known_unknown_mismatch_rate_each_predicate']
        checks[key + '_known_coverage'] = m['both_known'] >= gates['min_both_known_each_predicate']
        checks[key + '_ordinary_coverage'] = strata['ordinary'][key]['both_known'] >= gates['min_both_known_ordinary_each_predicate']
        checks[key + '_both_classes'] = min(m['agreed_positive'], m['agreed_negative']) >= gates['min_agreed_positive_and_negative_each_predicate']
        checks[key + '_eligible_agreement_fraction'] = m['eligible_agreed_fraction'] is not None and m['eligible_agreed_fraction'] >= gates['min_model_eligible_agreed_known_fraction_each_predicate']
        checks[key + '_eligible_coverage'] = m['eligible_agreed_known'] >= gates['min_model_eligible_agreed_known_each_predicate']
        checks[key + '_eligible_both_classes'] = min(m['eligible_agreed_positive'], m['eligible_agreed_negative']) >= gates['min_model_eligible_agreed_positive_and_negative_each_predicate']
    pattern_metrics = {}
    for hand, values in patterns.items():
        complete = [p for p in values if all(x is not None for x in p['a'] + p['b'])]
        conflict = [p for p in complete if p['a'] != p['b']]
        rate = len(conflict) / len(complete) if complete else None
        pattern_metrics[hand] = dict(all=len(values), both_complete=len(complete), conflicts=len(conflict), conflict_rate=rate,
                                    conflict_rows=conflict, raw_rows=values)
        checks[hand + '_two_bit_coverage'] = len(complete) >= gates['min_two_bit_both_known_each_hand']
        checks[hand + '_two_bit_conflict'] = rate is not None and rate <= gates['max_two_bit_hand_pattern_conflict_rate']
    family_support = {}
    for predicate in PREDICATES:
        joined = cells['R_' + predicate] + cells['L_' + predicate]
        s = support(joined, 1)
        family_support[predicate] = s
        checks[predicate + '_positive_family_support'] = len(s['sample_id']) >= gates['min_agreed_' + predicate + '_positive_windows'] and len(s['person']) >= gates['min_people_each_positive_family']
    classes = {k: {scope: {str(value): support(v, value, scope == 'eligible') for value in (0, 1)} for scope in ('physical', 'eligible')} for k, v in cells.items()}
    return dict(status='PASS_PRESENCE_PILOT_ONLY_BROAD_LABELS_NEXT' if all(checks.values()) else 'REPAIR_PRESENCE_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING',
                checks=checks, metrics=metrics, strata=strata, pattern_metrics=pattern_metrics, family_support=family_support,
                class_support=classes, contract_failures=failures, rows=rows, all_windows=24, all_hand_windows=48, all_predicate_cells=96,
                presence_is_not_ordering=True, semantic_or_educational_truth=None, training_allowed=False,
                original_failed_protocols_unchanged=True, accuracy_or_performance_claim=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=W / 'PRESENCE_PILOT_AUDIT.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'First saved output already exists; never overwrite')
    freezes = [W / 'PILOT_FROZEN.json', W / 'AUDIT_CODE_FROZEN.json']
    pins = {}
    for path in freezes:
        data = read(path)
        for source, digest in data['pins'].items():
            require(sha(source) == digest, 'Frozen source changed: ' + source)
            if source in pins:
                require(pins[source] == digest, 'Conflicting frozen hashes')
            pins[source] = digest
        pins[str(path)] = sha(path)
    require(pins.get(str(Path(__file__).resolve())) == sha(__file__), 'Executed audit code must be frozen')
    manifest = read(W / 'PILOT_MANIFEST.json')
    for source in manifest['rows']:
        require(sha(source['image']) == source['image_sha256'], 'Actual source image changed')
    require(sha(manifest['source']) == manifest['source_sha256'], 'Original full dataset changed')
    reviews = []
    for name in ('PRESENCE_A.json', 'PRESENCE_B.json'):
        path = W / 'review' / name
        review = read(path)
        require(review.get('rubric_sha256') == sha(W / 'RUBRIC.json'), 'Raw reviewer rubric hash mismatch: ' + name)
        reviews.append(review)
        pins[str(path)] = sha(path)
    result = calculate(manifest, reviews, read(W / 'PILOT_GATES.json'))
    for unit, name in (('word100_center_motion_pilot_20261101', 'PILOT_AUDIT.json'), ('word100_core_feature_split_20261101', 'LABEL_AUDIT.json')):
        path = W.parent / unit / name
        pins[str(path)] = sha(path)
        require(not read(path)['status'].startswith('PASS'), 'Prior failed protocol was changed')
    result['source_pins'] = pins
    result['ground_truth'] = 'Two separate fresh-context Codex RGB readings; not independent human/KSL semantic truth'
    result['audit_code'] = str(Path(__file__).resolve())
    write_new(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'checks', 'metrics')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
