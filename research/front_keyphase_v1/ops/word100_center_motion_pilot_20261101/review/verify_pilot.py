"""Independent pilot decoder and synthetic audit review; no training or relabeling.

code-review reads no actual pilot labels. verify reads saved labels/audit once,
reconstructs their arithmetic without importing audit functions, and writes a new
exclusive report. Existing reports and the frozen audit are never overwritten.
"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys

sys.dont_write_bytecode = True
V = Path(__file__).resolve().parent.parent
U = V.parent / 'word100_core_feature_split_20261101'
STATES = {'hold', 'up', 'down', 'horizontal', 'compound'}
KINDS = {'onset', 'offset', 'reversal', 'rotation', 'shape_change', 'hand_crossing'}
BOUNDARY = {'onset', 'offset', 'reversal'}
COUNTS = ('all', 'both_known', 'known_unknown_mismatch', 'both_known_conflicts',
          'both_roi', 'both_roi_known', 'both_roi_agreed', 'roi_mismatch',
          'ordinary_all', 'ordinary_both_known', 'agreed_hold', 'agreed_translation')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def require(condition, description):
    if not condition:
        raise ValueError(description)


def hand_set(value):
    return {'R', 'L'} if value == 'both' else {value}


def decode(manifest, reviews, gates):
    """Separate decoder: derives hand observations and set-based gate support."""
    sources = manifest['rows']
    count = gates['required_windows']
    require(len(sources) == count, 'manifest count')
    source_ids = {r['sample_id'] for r in sources}
    require(len(source_ids) == count, 'unique source IDs')
    maps = []
    for reviewer in reviews:
        require(reviewer['status'] == 'COMPLETE', 'review complete')
        require(len(reviewer['rows']) == count, 'review count')
        mapped = {r['sample_id']: r for r in reviewer['rows']}
        require(len(mapped) == count and set(mapped) == source_ids, 'review scope')
        maps.append(mapped)
    require(len(maps) == 2, 'exactly two reviewers')
    observations = {h: [] for h in 'RL'}
    support = {k: set() for k in ('hold', 'translation')}
    people = {k: set() for k in support}
    event_support = {k: set() for k in ('crossing', 'boundary', 'rotation',
                                      'short_any', 'short_translation', 'short_shape')}
    failures, conflicts, rows = [], [], []
    for source in sources:
        sid = source['sample_id']
        frames = source['frame_ids']
        times = source['timestamps_us']
        require(len(frames) == 13 and all(type(f) is int for f in frames), '13 integer frames')
        require(all(b == a + 1 for a, b in zip(frames, frames[1:])), 'consecutive frames')
        require(len(times) == 13 and all(type(t) is int for t in times), '13 integer timestamps')
        require(all(b > a for a, b in zip(times, times[1:])), 'increasing timestamps')
        require(source['support_frames'] == frames[4:9], 'exact centre support')
        require(source['center_frame'] == frames[6], 'exact centre frame')
        require(source['center_timestamp_us'] == times[6], 'exact centre timestamp')
        require(source['role'] == 'fit', 'fit-only pilot')
        pair = [m[sid] for m in maps]
        for ri, record in enumerate(pair):
            require(record['alias'] == source['alias'], 'alias')
            require(record['actual_image_viewed'] is True, 'actual image viewed')
            require(record['image_sha256'] == source['image_sha256'], 'image pin')
            for h in 'RL':
                require(record['state'][h] is None or record['state'][h] in STATES, 'state category')
                require(record['centre_roi'][h] is None or type(record['centre_roi'][h]) is bool,
                        'ROI bool or null')
            for event in record['events']:
                require(event['hand'] in ('R', 'L', 'both'), 'event hand')
                require(event['kind'] in KINDS, 'event kind')
                require(event['evidence'] in ('direct', 'uncertain'), 'event evidence')
                require(type(event['bounded_transition']) is bool, 'bounded bool')
                first, last = event['first_frame'], event['last_frame']
                require(type(first) is int and type(last) is int, 'integer native event endpoints')
                require(frames[0] <= first <= last <= frames[-1], 'event inside window')
                if event['bounded_transition']:
                    require(event['evidence'] == 'direct' and first > frames[0] and last < frames[-1],
                            'bounded direct transition with before and after context')
                if event['kind'] in BOUNDARY and set(range(first, last + 1)) & set(frames[4:9]):
                    for h in ('R', 'L') if event['hand'] == 'both' else (event['hand'],):
                        if record['state'][h] is not None:
                            failures.append(dict(reviewer=('PILOT_A.json', 'PILOT_B.json')[ri],
                                                 sample_id=sid, hand=h, state=record['state'][h],
                                                 event=event, reason='centre_support_crosses_translation_boundary'))
        row = dict(sample_id=sid, alias=source['alias'], role='fit', hands={})
        for h in 'RL':
            a, b = [r['state'][h] for r in pair]
            ka, kb = a is not None, b is not None
            agreement = ka and kb and a == b
            roi = all(r['centre_roi'][h] is True for r in pair)
            family = ('hold' if a == 'hold' else 'translation') if agreement else None
            observations[h].append(dict(known=ka and kb, mismatch=ka != kb,
                                        conflict=ka and kb and a != b, roi=roi,
                                        agreement=agreement, family=family,
                                        roi_mismatch=pair[0]['centre_roi'][h] != pair[1]['centre_roi'][h],
                                        ordinary=source['strata'] == ['hash_ordinary']))
            if family:
                support[family].add(sid)
                people[family].add(source['person'])
            if a != b:
                conflicts.append(dict(sample_id=sid, hand=h, a=a, b=b,
                                      kind='known_category_conflict' if ka and kb else 'known_unknown_mismatch'))
            row['hands'][h] = dict(a=a, b=b, agreed_state=a if agreement else None, both_roi=roi,
                                   supervised_state=a if agreement and roi else None, unknown_is_not_negative=True)
        paired = []
        for ea in pair[0]['events']:
            for eb in pair[1]['events']:
                if not (ea['evidence'] == eb['evidence'] == 'direct' and ea['kind'] == eb['kind']):
                    continue
                shared_hand = hand_set(ea['hand']) & hand_set(eb['hand'])
                shared_time = set(range(ea['first_frame'], ea['last_frame'] + 1)) & set(range(eb['first_frame'], eb['last_frame'] + 1))
                if not shared_hand or not shared_time:
                    continue
                kind = ea['kind']
                paired.append(dict(kind=kind, a=ea, b=eb))
                family = 'boundary' if kind in BOUNDARY else {'rotation': 'rotation', 'hand_crossing': 'crossing'}.get(kind)
                if family:
                    event_support[family].add(sid)
                short = all(e['bounded_transition'] and e['last_frame'] - e['first_frame'] + 1 <= gates['short_transition_max_native_frames'] for e in (ea, eb))
                if short and (kind in BOUNDARY or kind == 'shape_change'):
                    event_support['short_any'].add(sid)
                    event_support['short_shape' if kind == 'shape_change' else 'short_translation'].add(sid)
        row['paired_events'] = paired
        rows.append(row)
    by_hand = {}
    for h, obs in observations.items():
        by_hand[h] = dict(all=len(obs), both_known=sum(o['known'] for o in obs),
                          known_unknown_mismatch=sum(o['mismatch'] for o in obs),
                          both_known_conflicts=sum(o['conflict'] for o in obs),
                          both_roi=sum(o['roi'] for o in obs),
                          both_roi_known=sum(o['roi'] and o['known'] for o in obs),
                          both_roi_agreed=sum(o['roi'] and o['agreement'] for o in obs),
                          roi_mismatch=sum(o['roi_mismatch'] for o in obs),
                          ordinary_all=sum(o['ordinary'] for o in obs),
                          ordinary_both_known=sum(o['ordinary'] and o['known'] for o in obs),
                          agreed_hold=sum(o['family'] == 'hold' for o in obs),
                          agreed_translation=sum(o['family'] == 'translation' for o in obs))
    totals = {k: sum(by_hand[h][k] for h in 'RL') for k in COUNTS}
    raw_rate = totals['both_known_conflicts'] / (totals['both_known'] or 1)
    unknown_rate = totals['known_unknown_mismatch'] / (2 * count)
    checks = dict(annotation_scope_contract=not failures,
                  raw_motion_category_reliability=raw_rate <= gates['both_known_motion_category_disagreement_max'],
                  known_unknown_mismatch=unknown_rate <= gates['known_unknown_state_mismatch_max_all_hand_windows'])
    for h, m in by_hand.items():
        checks[h + '_physical_coverage'] = m['both_known'] >= gates['both_known_physical_state_min_per_hand']
        checks[h + '_ordinary_coverage'] = m['ordinary_both_known'] >= gates['both_known_physical_state_min_per_hand_ordinary12']
        checks[h + '_roi_coverage'] = m['both_roi'] > 0 and m['both_roi_known'] / m['both_roi'] >= gates['both_roi_confirmed_state_known_fraction_min_per_hand']
        checks[h + '_hold_and_movement_present'] = m['agreed_hold'] > 0 and m['agreed_translation'] > 0
    for k in support:
        checks[k + '_support'] = len(support[k]) >= gates['agreed_' + k + '_min_distinct_windows'] and len(people[k]) >= gates['each_hold_translation_min_people']
    gate_keys = dict(crossing='agreed_visible_crossing_min_windows', boundary='agreed_visible_boundary_min_windows',
                     rotation='agreed_visible_rotation_min_windows', short_any='agreed_direct_bounded_short_transition_min_windows',
                     short_translation='agreed_direct_bounded_short_translation_transition_min_windows',
                     short_shape='agreed_direct_bounded_short_shape_transition_min_windows')
    for k, gk in gate_keys.items():
        checks[k + '_support'] = len(event_support[k]) >= gates[gk]
    return dict(status='PASS_PILOT_PROTOCOL_ONLY_BROAD_LABELS_NEXT' if all(checks.values()) else 'REPAIR_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING',
                checks=checks, total=totals, by_hand=by_hand,
                agreement_support={k: dict(windows=sorted(v), people=sorted(people[k])) for k, v in support.items()},
                event_support={k: sorted(v) for k, v in event_support.items()}, conflicts=conflicts,
                contract_failures=failures, rows=rows, raw_motion_category_conflict_rate=raw_rate,
                known_unknown_mismatch_rate=unknown_rate, all_hand_windows=2 * count, all_windows=count)


def synthetic_fixture():
    manifest, labels = [], []
    for i in range(24):
        sid = f'SYN{i:02d}'
        manifest.append(dict(sample_id=sid, alias=f'P{i + 1:03d}', role='fit', person=i % 3 + 1,
                             strata=['hash_ordinary'] if i >= 12 else ['challenge'], frame_ids=list(range(100, 113)),
                             timestamps_us=list(range(1000, 1013)), center_frame=106, center_timestamp_us=1006,
                             support_frames=list(range(104, 109)), image=f'synthetic_{i}.jpg', image_sha256='synthetic'))
        state = 'hold' if i % 2 == 0 else 'down'
        events = []
        kinds = ['hand_crossing'] if i < 3 else []
        if i < 2:
            kinds += ['onset', 'rotation']
        if i == 2:
            kinds += ['shape_change']
        for kind in kinds:
            events.append(dict(hand='R', kind=kind, first_frame=101, last_frame=102, evidence='direct',
                               bounded_transition=kind in ('onset', 'shape_change'), note='Synthetic direct event'))
        labels.append(dict(sample_id=sid, alias=f'P{i + 1:03d}', actual_image_viewed=True, image_sha256='synthetic',
                           state=dict(R=state, L=state), centre_roi=dict(R=True, L=True), events=events))
    return dict(rows=manifest), [dict(status='COMPLETE', rows=deepcopy(labels)) for _ in range(2)]


def run_audit_in_memory(source_path, manifest, reviews, gates):
    """Imports the implementation under test with mocked IO; no label files read."""
    spec = importlib.util.spec_from_file_location('pilot_under_test', source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    documents = {'PILOT_FROZEN.json': {'pins': {}}, 'PILOT_MANIFEST.json': manifest,
                 'PILOT_GATES.json': gates, 'PILOT_A.json': reviews[0], 'PILOT_B.json': reviews[1]}
    module.read = lambda path: deepcopy(documents[Path(path).name])
    module.sha = lambda path: 'synthetic'
    outputs = []
    module.write = lambda path, result: outputs.append(deepcopy(result))
    with contextlib.redirect_stdout(io.StringIO()):
        module.main()
    require(len(outputs) == 1, 'one synthetic result')
    return outputs[0]


def compare_core(actual, expected):
    failures = []
    for key, value in expected.items():
        observed = actual.get(key)
        if key == 'total':
            observed = {k: (observed or {}).get(k, 0) for k in COUNTS}
        if key == 'by_hand':
            observed = {h: {k: (observed or {}).get(h, {}).get(k, 0) for k in COUNTS} for h in 'RL'}
        if key == 'contract_failures':
            # Implementations may enumerate reviewer-major or sample-major.
            # Preserve every diagnostic field and multiplicity, ignoring order.
            observed = sorted(json.dumps(x, sort_keys=True) for x in (observed or []))
            value = sorted(json.dumps(x, sort_keys=True) for x in value)
        if observed != value:
            failures.append(key)
    return failures


def code_review(source_path, output_path):
    gates = read(V / 'PILOT_GATES.json')
    results = []
    findings = []
    def run(name, modify=None, expected_check=None, rejected=False):
        manifest, reviews = synthetic_fixture()
        if modify:
            modify(manifest, reviews)
        before = deepcopy((manifest, reviews))
        try:
            actual = run_audit_in_memory(source_path, manifest, reviews, gates)
            if rejected:
                results.append(dict(name=name, passed=False, observed='accepted invalid input', status=actual['status']))
                return actual
            expected = decode(manifest, reviews, gates)
            mismatch = compare_core(actual, expected)
            passed = not mismatch and (expected_check(actual) if expected_check else True)
            passed = passed and (manifest, reviews) == before
            results.append(dict(name=name, passed=passed, arithmetic_mismatches=mismatch, status=actual['status']))
            return actual
        except (AssertionError, ValueError, TypeError, KeyError) as error:
            results.append(dict(name=name, passed=rejected, observed='rejected', exception=type(error).__name__, detail=str(error)))
            return None
    def states(reviews, value, hand=None, reviewer=None):
        for ri, review in enumerate(reviews):
            if reviewer is None or reviewer == ri:
                for row in review['rows']:
                    for h in ([hand] if hand else 'RL'):
                        row['state'][h] = value
    def change_events(reviews, transform, reviewer=None):
        for ri, review in enumerate(reviews):
            if reviewer is None or reviewer == ri:
                for row in review['rows']:
                    for event in row['events']:
                        transform(event)
    def set_event(event, **values):
        event.update(values)
    run('complete valid synthetic fixture passes all frozen gates', expected_check=lambda a: all(a['checks'].values()))
    run('all unknown fails coverage and state support', lambda m,r: states(r,None),
        lambda a: a['total']['both_known']==0 and not a['checks']['R_physical_coverage'] and not a['checks']['L_physical_coverage'])
    run('only R known fails L gates', lambda m,r: states(r,None,'L'), lambda a: not a['checks']['L_physical_coverage'])
    run('only L known fails R gates', lambda m,r: states(r,None,'R'), lambda a: not a['checks']['R_physical_coverage'])
    run('all known-vs-unknown remain in denominator', lambda m,r: states(r,None,reviewer=1), lambda a: a['known_unknown_mismatch_rate']==1)
    def conflicts(m,r,n):
        for i in range(n):
            r[1]['rows'][i]['state']['R']='up'
    run('2 of 48 raw conflicts below frozen 5percent', lambda m,r: conflicts(m,r,2), lambda a: a['checks']['raw_motion_category_reliability'])
    run('3 of 48 raw conflicts exceed frozen 5percent', lambda m,r: conflicts(m,r,3), lambda a: not a['checks']['raw_motion_category_reliability'])
    def unknown(m,r,n):
        for i in range(n): r[1]['rows'][i]['state']['R']=None
    run('9 of 48 known-unknown below frozen20percent', lambda m,r: unknown(m,r,9), lambda a: a['checks']['known_unknown_mismatch'])
    run('10 of 48 known-unknown exceeds frozen20percent', lambda m,r: unknown(m,r,10), lambda a: not a['checks']['known_unknown_mismatch'])
    run('opposite hands do not pair events', lambda m,r: change_events(r,lambda e:set_event(e,hand='L'),1), lambda a: not any(a['event_support'].values()))
    run('nonoverlapping intervals do not pair', lambda m,r: change_events(r,lambda e:set_event(e,first_frame=110,last_frame=111),1), lambda a: not any(a['event_support'].values()))
    run('uncertain events do not count as direct pairs', lambda m,r: change_events(r,lambda e:set_event(e,evidence='uncertain',bounded_transition=False),1), lambda a: not any(a['event_support'].values()))
    run('both-hands event shares the explicitly named hand', lambda m,r: change_events(r,lambda e:set_event(e,hand='both'),1), lambda a: a['checks']['crossing_support'])
    def shape_only(m,r):
        for review in r:
            for row in review['rows']:
                row['events']=[e for e in row['events'] if e['kind'] not in BOUNDARY]
    run('short shape cannot replace short translation support',shape_only,lambda a: not a['checks']['short_translation_support'] and a['checks']['short_shape_support'])
    def translation_only(m,r):
        for review in r:
            for row in review['rows']: row['events']=[e for e in row['events'] if e['kind']!='shape_change']
    run('short translation cannot replace short shape support',translation_only,lambda a: a['checks']['short_translation_support'] and not a['checks']['short_shape_support'])
    def crossing_boundary(m,r,known=True,uncertain=False):
        for review in r:
            event=next(e for e in review['rows'][0]['events'] if e['kind']=='onset')
            event.update(first_frame=104,last_frame=104,evidence='uncertain' if uncertain else 'direct',bounded_transition=not uncertain)
            if not known: review['rows'][0]['state']['R']=None
    run('direct centre boundary forbids known state',lambda m,r: crossing_boundary(m,r),lambda a: not a['checks']['annotation_scope_contract'])
    run('uncertain centre boundary also forbids known state',lambda m,r: crossing_boundary(m,r,uncertain=True),lambda a: not a['checks']['annotation_scope_contract'])
    run('centre boundary allows null without forced hold',lambda m,r: crossing_boundary(m,r,known=False),lambda a: a['checks']['annotation_scope_contract'] and a['rows'][0]['hands']['R']['supervised_state'] is None)
    run('rotation and shape do not force translation unknown',lambda m,r: change_events(r,lambda e:set_event(e,first_frame=105,last_frame=106) if e['kind'] in ('rotation','shape_change') else None),lambda a: a['checks']['annotation_scope_contract'])
    def no_roi(m,r):
        for review in r:
            for row in review['rows']: row['centre_roi']={'R':False,'L':False}
    run('physical states outside ROI remain observed but unsupervised',no_roi,lambda a:a['total']['both_known']==48 and all(x['supervised_state'] is None for row in a['rows'] for x in row['hands'].values()) and not a['checks']['R_roi_coverage'])
    run('truncated transitions cannot be declared bounded',lambda m,r: change_events(r,lambda e:set_event(e,first_frame=100) if e['bounded_transition'] else None),rejected=True)
    run('uncertain transitions cannot be declared bounded',lambda m,r: change_events(r,lambda e:set_event(e,evidence='uncertain') if e['bounded_transition'] else None),rejected=True)
    def fractional(m,r):
        for review in r: review['rows'][0]['events'][0]['first_frame']=101.5
    result=run('fractional native event endpoint must be rejected',fractional,rejected=True)
    if result is not None:
        findings.append(dict(id='F1',severity='input_contract',source=str(source_path),
                             issue='Noninteger event endpoints accepted by frozen audit and eligible for event-support matching.',
                             reproduction='Synthetic hand_crossing.first_frame=101.5, last_frame=102; all frozen gates still pass.',
                             remedy='Preserve frozen source and first outcomes; require type(first_frame) is int and type(last_frame) is int in a versioned correction. Do not change thresholds.'))
    pins={str(p):sha(p) for p in (source_path,V/'PILOT_FROZEN.json',V/'PILOT_GATES.json',V/'RUBRIC.json',Path(__file__))}
    result=dict(status='CODE_REVIEW_REQUIRES_VERSIONED_INPUT_CONTRACT_CORRECTION' if findings else 'PASS_SYNTHETIC_CODE_REVIEW',
                author='Independent Codex subagent split_head_contract',
                actual_pilot_labels_read=False,source_pins=pins,checks=results,
                passed=sum(r['passed'] for r in results),failed=sum(not r['passed'] for r in results),findings=findings,
                limitations=['Synthetic arithmetic and interface review does not verify RGB judgments or linguistic truth.',
                             'Numerical zero-conflict rate with zero known labels is not evidence: mandatory coverage/support gates still fail.',
                             'The separate real-result decoder will read labels only after the root saves the audit.'],
                original_files_modified=False,training_runs=0,model_inference_runs=0,criteria_tuned=False)
    write_new(output_path,result)
    print(json.dumps({k:result[k] for k in ('status','passed','failed','findings')},ensure_ascii=False))


def verify_saved(audit_path, output_path):
    """Rebuild saved result and validate frozen/source pins; never rerun audit."""
    checks=[]
    def check(ok,name): checks.append(dict(name=name,passed=bool(ok)))
    frozen=read(V/'PILOT_FROZEN.json')
    manifest=read(V/'PILOT_MANIFEST.json')
    blind=read(V/'BLIND_PILOT.json')
    gates=read(V/'PILOT_GATES.json')
    saved=read(audit_path)
    labels=[read(V/'review'/name) for name in ('PILOT_A.json','PILOT_B.json')]
    for path,digest in frozen['pins'].items(): check(sha(path)==digest,'frozen pin: '+path)
    for path,digest in saved['source_pins'].items(): check(sha(path)==digest,'audit source pin: '+path)
    for path,digest in frozen['pins'].items(): check(saved['source_pins'].get(path)==digest,'frozen pin retained in audit: '+path)
    for name in ('PILOT_A.json','PILOT_B.json'):
        p=V/'review'/name;check(saved['source_pins'].get(str(p))==sha(p),'review pinned: '+name)
    check(sha(manifest['source'])==manifest['source_sha256'],'original DATASET hash')
    original=read(manifest['source'])
    original_rows={r['sample_id']:r for r in original['rows']}
    blind_rows={r['sample_id']:r for r in blind['rows']}
    check(len(blind_rows)==gates['required_windows'],'blind window count')
    check(len({r['clip_id'] for r in manifest['rows']})==24,'24 distinct pilot clips')
    check(Counter(r['person'] for r in manifest['rows'])=={1:8,2:8,3:8},'8 windows per fixed fit person')
    check(len({r['word_id'] for r in manifest['rows']})==15,'15 pilot words')
    check(sum(r['strata']==['hash_ordinary'] for r in manifest['rows'])==12,'12 ordinary windows')
    for row in manifest['rows']:
        sid=row['sample_id'];source=original_rows[sid]
        for key in ('clip_id','person','word_id','role','frame_ids','timestamps_us','center_frame','center_timestamp_us','image'):
            check(row[key]==source[key],sid+' original '+key)
        check(sha(row['image'])==row['image_sha256'],sid+' original montage hash')
        check(all(row[k]==value for k,value in blind_rows[sid].items()),sid+' blind/source fields')
    for path,digest in original['pins'].items(): check(sha(path)==digest,'original protected pin: '+path)
    inputs={r['input']:r['input_sha256'] for r in original['rows']}
    for path,digest in inputs.items(): check(sha(path)==digest,'original input pin: '+path)
    expected=decode(manifest,labels,gates)
    mismatches=compare_core(saved,expected)
    for key in expected: check(key not in mismatches,'independent arithmetic: '+key)
    check(saved['training_started'] is False,'no training claim')
    check(saved['original_failed_protocol_unchanged'] is True,'original failure preserved claim')
    check(saved['performance_improvement_claim'] is None,'no performance claim')
    # These are the prior immutable first-result hashes, not a fresh score rerun.
    original_failure=U/'LABEL_AUDIT.json'
    check(sha(original_failure)=='848a5964b6e71dd5a46dda7f4f3ab5eb0d8863fb5c2fd0a6e54bd615d040da79','original failed label audit unchanged')
    check(sha(U/'review'/'INPUT_LABEL_VERIFICATION.json')=='84e11fcd0750cbc7a552bd5ad9d33a06208b6224844575e5e26efd256333d33a','original arithmetic verification unchanged')
    report=dict(status='PASS_SAVED_PILOT_ARITHMETIC_AND_IMMUTABILITY' if all(c['passed'] for c in checks) else 'FAIL_SAVED_PILOT_VERIFICATION',
                pilot_status=saved['status'],pilot_checks=expected['checks'],checks_count=len(checks),checks=checks,
                failures=[c for c in checks if not c['passed']],
                source_pins={str(p):sha(p) for p in (audit_path,V/'review'/'PILOT_A.json',V/'review'/'PILOT_B.json',Path(__file__))},
                scope='Independent raw-label decoding, saved arithmetic, frame/time/source hash alignment; not independent RGB reannotation.',
                audit_writer_rerun=False,criteria_tuned=False,model_inference_runs=0,training_runs=0,
                original_or_label_files_modified=False,
                limitations=['Operational pilot reproducibility does not establish accuracy or semantic correctness.',
                             'Even a pilot PASS only permits preparing broad labels under the frozen version; it does not authorize training before full readiness.'])
    write_new(output_path,report)
    print(json.dumps({k:report[k] for k in ('status','pilot_status','checks_count','failures')},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['code-review','verify'])
    parser.add_argument('--audit-source',type=Path,default=V/'audit_pilot.py')
    parser.add_argument('--audit',type=Path,default=V/'PILOT_AUDIT.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='code-review': code_review(args.audit_source,args.output or V/'review'/'CODE_REVIEW.json')
    else: verify_saved(args.audit,args.output or V/'review'/'PILOT_VERIFICATION.json')


if __name__=='__main__':
    main()
