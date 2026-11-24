"""Independent saved-native-label recount; does not import or rerun either audit."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import sys

sys.dont_write_bytecode = True
N = Path(__file__).resolve().parent.parent
W = N.parent / 'word100_interval_motion_20261101'
OUT = N / 'review/DIAGNOSIS_VERIFICATION.json'
checks, contradictions = [], []


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ck(group, name, result):
    checks.append(dict(group=group, name=name, passed=bool(result)))


def schema(record, source, who):
    """Separately express frozen interval/unknown/ROI rules, with no audit import."""
    sid = source['sample_id']
    prefix = who + '/' + sid
    lo, hi = source['frames'][0]['frame_id'], source['frames'][-1]['frame_id']
    def interval(obj):
        return (type(obj['first_frame']) is int and type(obj['last_frame']) is int and
                lo <= obj['first_frame'] <= obj['last_frame'] <= hi)
    def boolnull(value):
        return value is None or type(value) is bool
    def relevant(unknown, hand, predicate):
        return unknown['hand'] in (hand, 'both') and (unknown['component'] == 'identity' or
            (unknown['component'] == 'translation' and predicate in unknown['affected_predicates']))
    def semantic(name, good):
        ck('interval_contract', prefix + '/' + name, good)
        if not good:
            contradictions.append(dict(reviewer=who, sample_id=sid, violation=name))
    ck('row_schema', prefix + '/alias and viewed', record['alias'] == source['alias'] and record['actual_image_viewed'] is True)
    ck('row_schema', prefix + '/notes', type(record['notes']) is str and bool(record['notes'].strip()))
    for i, unknown in enumerate(record['unknown_intervals']):
        ck('row_schema', prefix + '/unknown' + str(i), interval(unknown) and unknown['hand'] in ('R', 'L', 'both') and
           unknown['component'] in ('identity', 'translation', 'shape', 'orientation'))
        if unknown['component'] == 'translation':
            ap = unknown['affected_predicates']
            ck('row_schema', prefix + '/explicit affected predicates' + str(i), type(ap) is list and
               len(ap) > 0 and len(set(ap)) == len(ap) and set(ap) <= {'down', 'hold'})
    for i, event in enumerate(record['events']):
        ck('row_schema', prefix + '/event' + str(i), interval(event) and event['hand'] in ('R', 'L', 'both') and
           event['kind'] in {'onset', 'offset', 'reversal', 'rotation', 'shape_change', 'hand_crossing'} and
           event['evidence'] in ('direct', 'uncertain') and type(event['bounded_transition']) is bool)
        if event['bounded_transition']:
            ck('row_schema', prefix + '/bounded event context' + str(i), event['evidence'] == 'direct' and
               lo < event['first_frame'] and event['last_frame'] < hi)
    for hand in 'RL':
        ck('row_schema', prefix + '/' + hand + '/full13 booleans', boolnull(record['full13_position_observable'][hand]) and
           boolnull(record['full13_roi'][hand]))
        for pred in ('down', 'hold'):
            p = prefix + '/' + hand + '_' + pred
            val, segments = record['presence'][hand][pred], record['segments'][hand][pred]
            ck('row_schema', p + '/presence and segments', (val is None or type(val) is int and val in (0, 1)) and type(segments) is list)
            semantic(hand + '_' + pred + '/presence matches witness existence', bool(segments) == (val == 1))
            semantic(hand + '_' + pred + '/chronological segments', segments == sorted(segments, key=lambda s: (s['first_frame'], s['last_frame'])))
            for i, seg in enumerate(segments):
                ck('row_schema', p + '/segment' + str(i), interval(seg) and boolnull(seg['in_roi']))
                semantic(hand + '_' + pred + '/minimum consecutive witness' + str(i), seg['last_frame'] - seg['first_frame'] >= (1 if pred == 'down' else 2))
                semantic(hand + '_' + pred + '/ROI consistent' + str(i), not (record['full13_roi'][hand] is True and seg['in_roi'] is False))
                overlap = [u for u in record['unknown_intervals'] if relevant(u, hand, pred) and
                           max(u['first_frame'], seg['first_frame']) <= min(u['last_frame'], seg['last_frame'])]
                semantic(hand + '_' + pred + '/no relevant unknown witness overlap' + str(i), not overlap)
            if val == 0:
                semantic(hand + '_' + pred + '/negative fully observable', record['full13_position_observable'][hand] is True)
                semantic(hand + '_' + pred + '/negative no relevant unknown', not any(relevant(u, hand, pred) for u in record['unknown_intervals']))
        overlap = [(down, hold) for down in record['segments'][hand]['down'] for hold in record['segments'][hand]['hold']
                   if min(down['last_frame'], hold['last_frame']) - max(down['first_frame'], hold['first_frame']) >= 1]
        semantic(hand + '/down and hold do not claim two shared frames', not overlap)
        semantic(hand + '/full13 identity consistency', not (record['full13_position_observable'][hand] is True and
                 any(u['component'] == 'identity' and u['hand'] in (hand, 'both') for u in record['unknown_intervals'])))


def eligible(row, hand, pred):
    val = row['presence'][hand][pred]
    if val == 1:
        return any(seg['in_roi'] is True for seg in row['segments'][hand][pred])
    return val == 0 and row['full13_position_observable'][hand] is True and row['full13_roi'][hand] is True


def main():
    if OUT.exists():
        raise FileExistsError('Saved diagnosis verification is immutable.')
    result = read(N / 'NATIVE_DIAGNOSIS.json')
    plan = read(N / 'PLAN_FROZEN.json')
    manifest = read(N / 'MANIFEST.json')
    source = {r['sample_id']: r for r in manifest['rows']}
    export = read(N / 'review/EXPORT_VERIFICATION.json')
    protected = dict(export['protected_pins'])
    freezes = [read(N / 'AUDIT_FROZEN_v2.json')['pins'], result['source_pins']]
    for pins in freezes:
        for p, h in pins.items():
            ck('hash_consistency', p, p not in protected or protected[p] == h)
            protected[p] = h
    for p, h in protected.items():
        ck('protected_sha', p, sha(p) == h)
    ck('export_precondition', 'native source export independently verified', export['status'] == 'PASS_INDEPENDENT_NATIVE_EXPORT_VERIFICATION')
    reviews, old_reviews, viewed = {}, {}, set()
    for who in 'AB':
        review = read(N / 'review' / f'NATIVE_{who}.json')
        ck('review_schema', who + '/complete exact9', review['status'] == 'COMPLETE' and len(review['rows']) == review['completed_windows'] == 9)
        ck('review_schema', who + '/pinned rubric and manifest', review['rubric_sha256'] == sha(W / 'RUBRIC.json') and review['native_manifest_sha256'] == sha(N / 'MANIFEST.json'))
        ck('review_schema', who + '/pinned blind manifest', review.get('blind_manifest_sha256', review.get('blind_native_sha256')) == sha(N / 'BLIND_NATIVE.json'))
        rows = {r['sample_id']: r for r in review['rows']}
        ck('review_schema', who + '/unique exact9 IDs', len(rows) == 9 and set(rows) == set(source))
        reviews[who] = rows
        old_reviews[who] = {r['sample_id']: r for r in read(W / 'review' / f'PRESENCE_{who}.json')['rows']}
        for sid, row in rows.items():
            schema(row, source[sid], who)
            allowed = {p['file']: (p['sha256'], p['frame_ids']) for p in source[sid]['pairs']}
            for frame in source[sid]['frames']:
                for kind in ('full', 'fixed'):
                    allowed[frame[kind]] = (frame[kind + '_sha256'], [frame['frame_id']])
            frames_seen = set()
            for item in row['images_viewed']:
                p, h = item['file'], item['sha256']
                ck('viewed_file_reference', who + '/' + p, p in allowed and h == allowed[p][0] == sha(p))
                frames_seen.update(allowed[p][1])
                viewed.add(p)
            ck('full13_view_references', who + '/' + sid, frames_seen == {f['frame_id'] for f in source[sid]['frames']})
    ck('saved_schema', 'exact9 result rows', len(result['rows']) == 9 and {r['sample_id'] for r in result['rows']} == set(source))
    saved = {r['sample_id']: r for r in result['rows']}
    aggregates = {t: collections.Counter() for t in ('R_down', 'R_hold', 'L_down', 'L_hold')}
    raw, negative_details = [], []
    support = {h: [] for h in 'RL'}
    for sid in source:
        raw_row = dict(sample_id=sid, predicates={})
        for hand in 'RL':
            for pred in ('down', 'hold'):
                target = hand + '_' + pred
                values = [reviews[w][sid]['presence'][hand][pred] for w in 'AB']
                previous = [old_reviews[w][sid]['presence'][hand][pred] for w in 'AB']
                eligibility = [eligible(reviews[w][sid], hand, pred) for w in 'AB']
                known = [v is not None for v in values]
                agreed = values[0] if all(known) and values[0] == values[1] else None
                cell = dict(old=previous, new=values, eligible=eligibility, agreed=agreed,
                            known_conflict=all(known) and values[0] != values[1],
                            known_unknown_mismatch=known[0] != known[1], both_unknown=not any(known))
                raw_row['predicates'][target] = cell
                ck('raw36_saved_cell_equality', sid + '/' + target, saved[sid]['predicates'][target] == cell)
                tally = aggregates[target]
                tally['all'] += 1
                if agreed is not None:
                    tally['agreed_positive' if agreed == 1 else 'agreed_negative'] += 1
                for k in ('known_conflict', 'known_unknown_mismatch', 'both_unknown'):
                    tally['known_conflicts' if k == 'known_conflict' else k] += int(cell[k])
                if pred == 'hold' and agreed == 0:
                    detail = dict(sample_id=sid, hand=hand, agreed_physical_value=0,
                                  full13_observable=[reviews[w][sid]['full13_position_observable'][hand] for w in 'AB'],
                                  full13_roi=[reviews[w][sid]['full13_roi'][hand] for w in 'AB'], eligible=eligibility,
                                  roi_bottom=source[sid]['roi_bottom'],
                                  notes={w: reviews[w][sid]['notes'] for w in 'AB'})
                    negative_details.append(detail)
                    if sid in plan['hold_support'] and all(eligibility):
                        support[hand].append(sid)
        raw.append(raw_row)
    keys = ('all', 'agreed_positive', 'agreed_negative', 'known_conflicts', 'known_unknown_mismatch', 'both_unknown')
    summary = {target: {key: tally[key] for key in keys} for target, tally in aggregates.items()}
    for target, values in summary.items():
        ck('summary_recount', target, result['summary'][target] == values and sum(v for k, v in values.items() if k != 'all') == 9)
    byraw = {r['sample_id']: r for r in raw}
    unresolved = []
    for row, sid in zip(result['left_six'], plan['six_left']):
        cell = byraw[sid]['predicates']['L_down']
        expected = dict(sample_id=sid, **cell, resolved_on_native_view=cell['agreed'] is not None,
                        note_A=reviews['A'][sid]['notes'], note_B=reviews['B'][sid]['notes'])
        ck('six_left_recount', sid, row == expected)
        if cell['agreed'] is None:
            unresolved.append(sid)
    ck('six_left_recount', 'exact6 rows/count/list', len(result['left_six']) == 6 and
       result['left_six_resolved'] == 6 - len(unresolved) and result['left_six_unresolved'] == unresolved)
    ck('hold_support_recount', 'exact eligible support from frozen four', result['eligible_hold_absent_support'] == support)
    ck('independent_contract_comparison', 'both validators found no contradictions', not contradictions and result['annotation_contract_failures'] == [])
    can_reassess = not contradictions and not unresolved and all(support.values()) and not any(v['known_conflicts'] for v in summary.values())
    ck('decision_recount', 'readiness status', result['further_readiness_assessment_supported'] == can_reassess and
       result['status'] == ('SOURCE_REPAIR_SUPPORTED_REASSESS_FULL_READINESS' if can_reassess else 'SOURCE_DIAGNOSIS_COMPLETE_TRAINING_NOT_READY'))
    ck('decision_recount', 'no training or performance claim', result['training_allowed'] is False and result['trained_models'] == 0 and result['model_accuracy_improvement'] is None)
    ck('decision_recount', 'retrospective denominator unchanged', result['selected_retrospective_diagnosis'] is True and
       result['unchanged_failed24_denominator'] is True and result['windows'] == 9 and result['native_frames'] == 117)
    old = read(W / 'PRESENCE_PILOT_AUDIT.json')
    complete = read(W / 'COMPLETION.json')
    ck('first_failures_preserved', 'old24 still fails', old['status'] == 'REPAIR_PRESENCE_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING' and
       old['all_windows'] == 24 and old['training_allowed'] is False and complete['counts']['L_down']['known_unknown_mismatch'] == 6 and
       all(complete['counts'][h + '_hold']['agreed_negative'] == 0 for h in 'RL'))
    ck('first_failures_preserved', 'centre-state first FAIL', read(N.parent / 'word100_center_motion_pilot_20261101/PILOT_AUDIT.json')['status'] == 'REPAIR_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING')
    ck('first_failures_preserved', 'whole-window feature first FAIL', read(N.parent / 'word100_core_feature_split_20261101/LABEL_AUDIT.json')['status'] == 'NOT_READY_SUPPORT_OR_REVIEW')
    for path, digest in protected.items():
        ck('protected_sha_after', path, sha(path) == digest)
    groups = {}
    for group in sorted({c['group'] for c in checks}):
        selection = [c for c in checks if c['group'] == group]
        groups[group] = dict(checks=len(selection), passed=sum(c['passed'] for c in selection), failed=sum(not c['passed'] for c in selection))
    failures = [c for c in checks if not c['passed']]
    inspection = N / 'visuals/N022_W2/full_0079.png'
    report = dict(status='PASS_INDEPENDENT_RAW_NATIVE_DIAGNOSIS_VERIFICATION' if not failures else 'FAIL_NATIVE_DIAGNOSIS_VERIFICATION',
        at_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        reviewer='Separate Codex subagent native_verify_resume; no audit code imported or rerun',
        counts=dict(windows=9, paired_predicates=36, raw_reviewer_values=72, protected_files=len(protected),
                    protected_production_model_baseline_files=32, referenced_unique_images=len(viewed), checks=len(checks), failed=len(failures)),
        check_groups=groups, checks=checks, failures=failures, independent_contract_violations=contradictions,
        raw_recount=raw, summary=summary, left_six_resolved=6-len(unresolved), left_six_unresolved=unresolved,
        physical_hold_absent_cases=negative_details, eligible_hold_absent_support=support,
        roi_diagnosis='N022_W2 is the only agreed physical hold-negative for each hand. Both readers report full13 palm/wrist identity observable but full13_roi=false. The unchanged negative eligibility rule requires full13 ROI, so neither hand supplies an eligible negative. Early in-ROI down segments can still be positive-eligible.',
        roi_visual_spot_check=dict(file=str(inspection), sha256=sha(inspection), actually_viewed=True,
             scope='Single final native frame corroborates ROI exclusion only; no independent full13 motion reannotation.',
             observation='At frame79 both hanging palm/hand regions extend below source y914, the model ROI bottom; source full frame still shows them. Physical visibility and model ROI inclusion differ.'),
        interpretation=dict(
            native_stop='Allsix-left-known is the predeclared conservative advancement condition for this selected diagnostic. Its failure is retained; it is not the original24 pilot gate or a demand to invent ground truth.',
            original_gate='The original24 contract allowed known/unknown mismatch up to20% plus coverage/class gates, and did not require every case known. Its original6/24 mismatch and zero hold-negatives are immutable.',
            unresolved='N008_W1 L_down remains [null,null] because tiny settling/drift cannot be judged reliably. Leave unknown; do not force0, repeat until agreement, or claim impossible learnability from one unknown.',
            class_support='Current native9 has no eligible hold-negative for either hand, so even perfect resolution of N008 would not establish class support or full readiness.',
            limited_next_inventory='A separately frozen finite fit-only support inventory may search for actual in-ROI hold-absent examples, preserving every screened outcome. It cannot append to failed24, change native9 outcome, or automatically authorize broad288 labels/training.',
            future_readiness='Any later full readiness assessment must explicitly separate observation reliability and class/ROI support under prospectively fixed unchanged criteria; existing failures remain historical facts.'),
        source_pins={str(p):sha(p) for p in [Path(__file__), N/'NATIVE_DIAGNOSIS.json', N/'review/NATIVE_A.json', N/'review/NATIVE_B.json',
                    N/'PLAN_FROZEN.json', N/'AUDIT_FROZEN_v2.json', W/'PILOT_GATES.json', W/'PRESENCE_PILOT_AUDIT.json']},
        protected_pins=protected,
        execution=dict(audit_rerun=False, reannotation=False, original_files_modified=False, training_runs=0,
                       model_inference_runs=0, original_input_reextraction=0, private_recording_access=0, root_documents_changed=False),
        limitations=['Independent arithmetic/schema implementation within same Codex system, not independent human truth.',
                     'Visual spot check confirms only the final-frame ROI concern, not absence of hold over all13frames.',
                     'No targeted9 versus original24 metric is a model-performance improvement.'])
    with OUT.open('x',encoding='utf8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ['status','counts','summary','left_six_resolved','left_six_unresolved','eligible_hold_absent_support']},ensure_ascii=False))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
