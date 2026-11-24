"""Separately recount frozen support outcomes; no audit or model execution."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import sys
sys.dont_write_bytecode = True
N = Path(__file__).resolve().parent.parent
S = N / 'support_inventory'
W = N.parent / 'word100_interval_motion_20261101'
OUT = N / 'review/SUPPORT_OUTCOME_VERIFICATION.json'
checks, issues = [], []


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def ck(group, name, okay):
    checks.append(dict(group=group, name=name, passed=bool(okay)))


def main():
    if OUT.exists():
        raise FileExistsError('Reuse the immutable saved verification.')
    outcome = read(S / 'SUPPORT_RESULT.json')
    manifest = read(S / 'MANIFEST.json')
    selection = read(S / 'SELECTION.json')
    export_verification = read(N / 'review/SUPPORT_SELECTION_EXPORT_VERIFICATION.json')
    sources = {r['support_alias']: r for r in manifest['rows']}
    nominees = {r['support_alias']: r for r in selection['selected']}
    protected = dict(export_verification['protected_pins'])
    for pins in [export_verification['source_pins'], read(S / 'AUDIT_FROZEN.json')['pins'], outcome['pins']]:
        for p, h in pins.items():
            ck('pin_consistency', p, p not in protected or protected[p] == h)
            protected[p] = h
    for p, h in protected.items():
        ck('protected_sha', p, sha(p) == h)
    ck('scope', 'export verified, fixed12 source/nominee equality', export_verification['status'] == 'PASS_INDEPENDENT_SUPPORT_SELECTION_EXPORT' and
       len(sources) == len(nominees) == 12 and set(sources) == set(nominees))
    ck('scope', 'all360 candidate records retained', selection['opportunities'] == len(selection['candidates']) == 360 and
       len({(r['sample_id'],r['hand']) for r in selection['candidates']}) == 360)
    maps, viewed = {}, set()
    for who in 'AB':
        review = read(S / f'REVIEW_{who}.json')
        ck('review_schema', who + '/complete exact12', review['status'] == 'COMPLETE' and len(review['rows']) == 12)
        ck('review_schema', who + '/manifest rubric pins', review['manifest_sha256'] == sha(S / 'MANIFEST.json') and review['rubric_sha256'] == sha(W / 'RUBRIC.json'))
        rows = {r['alias']: r for r in review['rows']}
        ck('review_schema', who + '/unique exact12 aliases', len(rows) == 12 and set(rows) == set(sources))
        maps[who] = rows
        for alias, row in rows.items():
            src, prefix = sources[alias], who + '/' + alias
            frames = [f['frame_id'] for f in src['frames']]
            val, segments, uncertain = row['hold_present'], row['hold_segments'], row['uncertain_intervals']
            ck('row_schema', prefix + '/identity and direct view', row['sample_id'] == src['sample_id'] and row['target_hand'] == src['target_hand'] and row['actual_image_viewed'] is True)
            ck('row_schema', prefix + '/strict value types', (val is None or type(val) is int and val in (0,1)) and
               all(row[k] is None or type(row[k]) is bool for k in ('full13_position_observable','full13_roi')) and
               type(segments) is list and type(uncertain) is list and type(row['note']) is str and bool(row['note'].strip()))
            def bounds(item):
                return type(item['first_frame']) is int and type(item['last_frame']) is int and frames[0] <= item['first_frame'] <= item['last_frame'] <= frames[-1]
            def semantic(reason, good):
                ck('interval_contract', prefix + '/' + reason, good)
                if not good:
                    issues.append(dict(alias=alias, reviewer=who, reason=reason))
            for j, unknown in enumerate(uncertain):
                ck('row_schema', prefix + '/uncertain interval' + str(j), bounds(unknown) and type(unknown['reason']) is str and bool(unknown['reason'].strip()))
            semantic('presence matches witnesses', bool(segments) == (val == 1))
            semantic('positive segments chronological', segments == sorted(segments, key=lambda x:(x['first_frame'], x['last_frame'])))
            for j, segment in enumerate(segments):
                ck('row_schema', prefix + '/positive interval' + str(j), bounds(segment) and
                   (segment['in_roi'] is None or type(segment['in_roi']) is bool))
                semantic('minimum3 original frames ' + str(j), segment['last_frame'] - segment['first_frame'] >= 2)
                semantic('no positive overlap unknown ' + str(j), not any(max(segment['first_frame'],u['first_frame']) <= min(segment['last_frame'],u['last_frame']) for u in uncertain))
                semantic('full13 ROI consistent with segment ' + str(j), not (row['full13_roi'] is True and segment['in_roi'] is False))
            if val == 0:
                semantic('negative has full13 observation and no uncertainty', row['full13_position_observable'] is True and not uncertain)
            allowed = {p['file']:(p['sha256'],p['frame_ids']) for p in src['pairs']}
            for f in src['frames']:
                for kind in ('full','fixed'):
                    allowed[f[kind]] = (f[kind+'_sha256'],[f['frame_id']])
            seen = set()
            ck('row_schema', prefix + '/image references list', type(row['images_viewed']) is list)
            for im in row['images_viewed']:
                p = im['file']
                ck('image_reference_sha', prefix + '/' + p, p in allowed and im['sha256'] == allowed[p][0] == sha(p))
                seen.update(allowed[p][1]); viewed.add(p)
            ck('full13_image_references', prefix, seen == set(frames))
    ck('saved_schema', 'exact12 result rows and aliases', len(outcome['rows']) == 12 and {r['alias'] for r in outcome['rows']} == set(sources))
    saved = {r['alias']:r for r in outcome['rows']}
    independent = []
    for alias, src in sources.items():
        pair = [maps[w][alias] for w in 'AB']
        values = [r['hold_present'] for r in pair]
        eligibility = []
        for r in pair:
            if r['hold_present'] == 1:
                eligibility.append(any(s['in_roi'] is True for s in r['hold_segments']))
            else:
                eligibility.append(r['hold_present'] == 0 and r['full13_position_observable'] is True and r['full13_roi'] is True)
        agreed = values[0] if values[0] is not None and values[0] == values[1] else None
        row = dict(alias=alias, sample_id=src['sample_id'], target_hand=src['target_hand'], person=src['person'], word_id=src['word_id'],
                   raw=values, agreed=agreed, eligible=eligibility,
                   eligible_agreed_absence=agreed == 0 and all(eligibility) and not any(e['alias'] == alias for e in issues),
                   notes=[r['note'] for r in pair])
        ck('raw_pair_recount', alias, row == saved[alias])
        independent.append(row)
    metrics, support, diversity = {}, {}, {}
    for hand in 'RL':
        rows = [r for r in independent if r['target_hand'] == hand]
        values = collections.Counter()
        for row in rows:
            a,b = row['raw']
            if a is None and b is None:
                values['both_unknown'] += 1
            elif (a is None) != (b is None):
                values['known_unknown_mismatch'] += 1
            elif a != b:
                values['known_conflicts'] += 1
            elif a == 1:
                values['agreed_present'] += 1
            else:
                values['agreed_absent'] += 1
        metrics[hand] = dict(all=len(rows), **{k:values[k] for k in ('agreed_absent','agreed_present','known_conflicts','known_unknown_mismatch','both_unknown')},
                            eligible_agreed_absence=sum(r['eligible_agreed_absence'] for r in rows))
        yes = [r for r in rows if r['eligible_agreed_absence']]
        support[hand] = dict(windows=[r['sample_id'] for r in yes], people=sorted({r['person'] for r in yes}), expressions=sorted({r['word_id'] for r in yes}))
        diversity[hand] = dict(windows=len(yes), clips=len({nominees[r['alias']]['clip_id'] for r in yes}),
                               people=len(support[hand]['people']), expressions=len(support[hand]['expressions']))
        ck('metrics_recount', hand, outcome['metrics'][hand] == metrics[hand] and len(rows) == sum(values.values()) == 6)
        ck('support_identity_recount', hand, outcome['support'][hand] == support[hand])
    ck('contract_comparison', 'no native contract contradictions', not issues and outcome['annotation_contract_failures'] == [])
    ck('scope', 'no candidate replacement/promotion', outcome['all_candidates_retained'] is True and outcome['opportunities_screened'] == 360 and outcome['reviewed_windows'] == 12 and outcome['candidate_replacements'] == 0)
    ck('scope', 'not training or performance', outcome['status'] == 'FINITE_SUPPORT_INVENTORY_COMPLETE_NOT_TRAINING_READINESS' and outcome['training_allowed'] is False and outcome['model_runs'] == 0 and outcome['model_accuracy_improvement'] is None)
    ck('earlier_results', 'support not merged into24', outcome['original24_denominator_unchanged'] is True and
       not {r['sample_id'] for r in independent}.intersection(r['sample_id'] for r in read(W/'PILOT_MANIFEST.json')['rows']))
    old = read(W/'PRESENCE_PILOT_AUDIT.json')
    native = read(N/'NATIVE_DIAGNOSIS.json')
    ck('earlier_results', 'original24 still failed and no eligibleholdnegative', old['all_windows'] == 24 and old['status'] == 'REPAIR_PRESENCE_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING' and
       all(old['metrics'][h+'_hold']['eligible_agreed_negative'] == 0 for h in 'RL'))
    ck('earlier_results', 'native9 unchanged unresolvedN008/eligibleholdzero', outcome['native9_result_unchanged'] is True and native['left_six_resolved'] == 5 and native['left_six_unresolved'] == ['N008_W1'] and native['eligible_hold_absent_support'] == {'R':[],'L':[]})
    for p,h in protected.items():
        ck('protected_sha_after',p,sha(p)==h)
    groups = {}
    for group in sorted({r['group'] for r in checks}):
        rows = [r for r in checks if r['group'] == group]
        groups[group] = dict(checks=len(rows),passed=sum(r['passed'] for r in rows),failed=sum(not r['passed'] for r in rows))
    failures=[r for r in checks if not r['passed']]
    output=dict(status='PASS_INDEPENDENT_SUPPORT_OUTCOME_RECOUNT' if not failures else 'FAIL_SUPPORT_OUTCOME_RECOUNT',
        at_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        reviewer='Separate Codex verifier; no audit imported/rerun and no fresh RGB reannotation',
        counts=dict(reviewed_windows=12,raw_values=24,target_hands=2,referenced_unique_images=len(viewed),protected_files=len(protected),
                    protected_production_model_baseline_files=32,checks=len(checks),failed=len(failures)),
        check_groups=groups,checks=checks,failures=failures,independent_contract_violations=issues,
        metrics=metrics,support=support,support_diversity=diversity,independent_rows=independent,
        uncertain_or_disagreeing=[dict(alias=r['alias'],sample_id=r['sample_id'],target_hand=r['target_hand'],raw=r['raw']) for r in independent if r['agreed'] is None],
        interpretation=['The fixed search demonstrates eligible hold-absent examples in selected publicFIT data: R4windows/4clips/3people/2expressions, L1window/1clip/1person/1expression.',
                        'This is existence support, not broad learning class-support minima, reliability/generalization or educational truth.',
                        'Original24 remains failed and native9 retains N008 unknown; support12 was not added to either denominator or their eligible-negative counts.',
                        'Positive hold segment order was independently checked in both raw review files and is chronological.',
                        'Any same24 native-view follow-up must reuse existing9 exactly, retain unchanged W gates and keep support12 separate.'],
        protected_pins=protected,
        source_pins={str(p):sha(p) for p in [Path(__file__),S/'SUPPORT_RESULT.json',S/'REVIEW_A.json',S/'REVIEW_B.json',S/'MANIFEST.json',S/'SELECTION.json',S/'AUDIT_FROZEN.json']},
        execution=dict(audit_rerun=False,producer_scripts_rerun=False,model_inference_runs=0,training_runs=0,source_decode_rerun=False,
                       private_recording_access=0,original_files_modified=False,root_documents_changed=False))
    with OUT.open('x',encoding='utf8') as f:json.dump(output,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:output[k] for k in ('status','counts','metrics','support_diversity','uncertain_or_disagreeing')},ensure_ascii=False))
    return bool(failures)


if __name__=='__main__':
    raise SystemExit(main())
