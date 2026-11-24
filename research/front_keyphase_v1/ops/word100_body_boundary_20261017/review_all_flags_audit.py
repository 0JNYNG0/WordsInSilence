"""Independent immutable visual-evidence/ledger coverage audit.

Uses standard-library JSON/hash operations only. It checks claims against
displayed evidence, not whether a human actually looked or the sign is correct.
Never issues semantic PASS, training approval, or READY.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

U = Path(__file__).resolve().parent
D = U / 'review_all_flags_materials'
HASH_CACHE = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1048576), b''):
                h.update(chunk)
        HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]


def expected_windows(coverage):
    result = {}
    for clip in coverage['rows']:
        for interval in clip['merged_review_windows_with_300ms_context']:
            index = len(result) + 1
            wid = f'W{index:04d}'
            present = set(interval['frame_ids'])
            conditions = []
            for condition in clip['conditions']:
                focus = sorted(present.intersection(condition['focus_frame_ids']))
                if focus:
                    conditions.append(dict(condition_review_id=f'{wid}_C{len(conditions)+1:02d}',
                        comparison=condition['comparison'], side=condition['side'], focus_frame_ids=focus))
            assert conditions
            result[wid] = dict(window_id=wid, batch=(index-1)%3+1,
                clip_id=clip['clip_id'], record_sha256=clip['record_sha256'],
                source_sha256=clip['source']['sha256'], frame_ids=interval['frame_ids'],
                start=interval['start_frame_id'], end=interval['end_frame_id'],
                conditions=conditions)
    assert len(result) == coverage['merged_review_window_count'] == 685
    return result


def peak(material, condition):
    previous, following = ('native', 'masked') if condition['comparison'] == 'native_to_mask' else ('masked', 'sensor')
    side = {'left': 'L', 'right': 'R'}[condition['side']]
    samples = [f for f in material['selected_frames'] if f['frame_id'] in condition['focus_frame_ids']]
    assert samples
    def key(f):
        v = f['valid_hand_counts']
        return max(0, v[previous][side]-v[following][side]), -f['frame_id']
    return max(samples, key=key)['frame_id']


def image_evidence(material, wid, image_path):
    """Return exact RGB/overlay frame IDs displayed for this window in an image."""
    target = str(Path(image_path).resolve())
    for img in material['images']:
        if str(Path(img['path']).resolve()) == target:
            assert sha(image_path) == img['sha256']
            frames = set(img['displayed_frame_ids'])
            return frames, frames
    path = Path(image_path)
    # Overview metadata is one-to-one with its image.
    if 'overview' in path.name:
        metadata = read(path.with_suffix('.json'))
        assert str(Path(metadata['image']).resolve()) == target
        assert sha(path) == metadata['image_sha256']
        matching = [r for r in metadata['rows'] if r['window_id'] == wid]
        assert matching, 'Overview image does not contain the claimed window'
        rgb = set().union(*(set(r['overview_displayed_frame_ids']) for r in matching))
        overlay = set().union(*(set(r['actual_overlay_compared_frame_ids']) for r in matching))
        return rgb, overlay
    if '_extra_' in path.name:
        stem = path.stem.rsplit('_p', 1)[0]
        metadata = read(path.with_name(stem + '.json'))
        assert metadata['window_id'] == wid
        assert metadata['material_sha256'] == sha(D / (wid + '.json'))
        matching = [i for i in metadata['images'] if str(Path(i['path']).resolve()) == target]
        assert len(matching) == 1 and sha(path) == matching[0]['sha256']
        frames = set(matching[0]['displayed_frame_ids'])
        return frames, frames
    raise AssertionError('Unregistered image cannot establish visual coverage')


def audit(ledger_paths, output, require_materials=False):
    coverage_path = U / 'review_flag_coverage_RESULT.json'
    expected = expected_windows(read(coverage_path))
    context_path = U / 'PREP_CONTEXT.json'
    rows = {r['clip_id']: r for r in read(context_path)['ordered_rows']}
    materials, missing, material_errors = {}, [], []
    condition_expected = {}
    for wid, target in expected.items():
        for c in target['conditions']:
            condition_expected[c['condition_review_id']] = (target, c)
        path = D / (wid + '.json')
        if not path.exists():
            missing.append(wid)
            continue
        try:
            m = read(path)
            assert m['job']['clip_id'] == target['clip_id']
            assert m['job']['batch'] == target['batch']
            assert m['record_sha256'] == target['record_sha256'] == sha(m['record_path'])
            assert m['source_sha256'] == target['source_sha256']
            assert m['pins']['context'] == sha(context_path)
            assert m['pins']['coverage'] == sha(coverage_path)
            row = rows[target['clip_id']]
            assert row['role'] == 'train_candidate' and 1 <= row['signer_id'] <= 9
            selected = {f['frame_id']: f for f in m['selected_frames']}
            assert set(selected) <= set(target['frame_ids'])
            assert target['start'] in selected and target['end'] in selected
            assert len(m['job']['conditions']) == len(target['conditions'])
            for c, actual in zip(target['conditions'], m['job']['conditions']):
                assert all(c[k] == actual[k] for k in ('comparison', 'side', 'focus_frame_ids'))
                assert c['focus_frame_ids'][0] in selected and c['focus_frame_ids'][-1] in selected
                peak(m, c)
            displayed = []
            for im in m['images']:
                assert sha(im['path']) == im['sha256']
                displayed.extend(im['displayed_frame_ids'])
            assert len(displayed) == len(set(displayed)) and set(displayed) == set(selected)
            for f in selected.values():
                assert abs(f['source_relative_pts_us'] - f['observation_timestamp_us']) <= 1
                assert abs(f['source_absolute_pts_us'] - m['source_absolute_pts_origin_us'] - f['source_relative_pts_us']) <= 1
            materials[wid] = m
        except Exception as e:
            material_errors.append(dict(window_id=wid, error=repr(e)))
    if require_materials:
        assert not missing and not material_errors, 'Full visual materials are not complete/valid'
    records = defaultdict(list)
    invalid = []
    ignored_conditions = []
    ledger_snapshots = []
    for ledger in ledger_paths:
        ledger = Path(ledger)
        raw = ledger.read_bytes()
        ledger_snapshots.append(dict(path=str(ledger.resolve()), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
        for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            reference = dict(path=str(ledger.resolve()), line=number, row_sha256=hashlib.sha256(line.encode()).hexdigest())
            try:
                entry = json.loads(line)
                wid = entry['window_id']
                assert wid in materials, 'No valid committed material for this row'
                m, target = materials[wid], expected[wid]
                manifest = entry.get('material_manifest', entry.get('material_path'))
                assert manifest and str(Path(manifest).resolve()) == str((D / (wid + '.json')).resolve())
                assert entry.get('material_manifest_sha256', entry.get('material_sha256')) == sha(manifest)
                assert entry['record_sha256'] == m['record_sha256']
                assert entry.get('source_sha256', m['source_sha256']) == m['source_sha256']
                assert entry.get('every_source_frame_viewed', False) is False
                assert entry.get('training_allowed_by_this_row', False) is False
                images = entry['viewed_images']
                assert images, 'No explicit actually viewed images'
                rgb, overlay = set(), set()
                for im in images:
                    assert sha(im['path']) == im['sha256']
                    a, z = image_evidence(m, wid, im['path'])
                    rgb.update(a); overlay.update(z)
                viewed = set(entry['actual_viewed_frames'])
                compared = set(entry.get('actual_overlay_compared_frame_ids', viewed.intersection(overlay)))
                assert viewed and viewed <= rgb and viewed <= set(target['frame_ids'])
                assert compared <= viewed and compared <= overlay
                nested = entry.get('condition_reviews', [])
                assert (entry.get('observation_ko') or all(c.get('observation_ko') for c in nested) and nested) and entry.get('disposition')
                cid = entry.get('condition_review_id', entry.get('condition_id'))
                if cid:
                    assert cid in condition_expected and cid.startswith(wid + '_')
                    candidate_conditions = [condition_expected[cid][1]]
                elif nested:
                    keys = [c['condition_review_id'] for c in nested]
                    assert len(keys) == len(set(keys))
                    assert all(k in condition_expected and k.startswith(wid + '_') for k in keys)
                    candidate_conditions = [condition_expected[k][1] for k in keys]
                else:
                    listed = entry.get('conditions', [])
                    candidate_conditions = [c for c in target['conditions'] if any(c['comparison'] == v['comparison'] and c['side'] == v['side'] for v in listed)]
                    assert candidate_conditions, 'No explicit condition coverage claim'
                for c in candidate_conditions:
                    cid = c['condition_review_id']
                    detail = next((x for x in nested if x['condition_review_id'] == cid), entry)
                    detail_condition = detail.get('condition', detail)
                    for key in ('comparison', 'side'):
                        if key in detail_condition:
                            assert detail_condition[key] == c[key]
                    condition_compared = set(detail.get('actual_overlay_compared_frame_ids', compared))
                    assert condition_compared <= compared
                    selected_peak = peak(m, c)
                    # Do not credit a multi-condition overview for a condition whose
                    # actual focus peak has not been explicitly claimed as viewed.
                    represented = selected_peak in condition_compared
                    if not represented:
                        ignored_conditions.append(dict(**reference, condition_review_id=cid, required_peak=selected_peak, reason='condition peak not among claimed compared frames'))
                        continue
                    records[cid].append(dict(reference=reference, reviewer=entry.get('reviewer'),
                        actual_viewed_frames=sorted(viewed), compared_frames=sorted(condition_compared),
                        unresolved_essential_geometric_cut=detail.get('unresolved_essential_geometric_cut',entry.get('unresolved_essential_geometric_cut')),
                        additional_frames_needed=detail.get('additional_frames_needed',entry.get('additional_frames_needed', False)),
                        disposition=entry['disposition'], observation_ko=detail.get('observation_ko',entry.get('observation_ko'))))
            except Exception as e:
                invalid.append(dict(**reference, error=repr(e)))
    conditions = []
    for cid, (target, c) in condition_expected.items():
        entries = records[cid]
        viewed = set().union(*(set(e['actual_viewed_frames']) for e in entries)) if entries else set()
        context_present = target['start'] in viewed and target['end'] in viewed
        unresolved = any(e['unresolved_essential_geometric_cut'] is not False or e['additional_frames_needed'] for e in entries)
        conditions.append(dict(condition_review_id=cid, window_id=target['window_id'], batch=target['batch'],
            comparison=c['comparison'], side=c['side'], represented=bool(entries),
            context_start_end_claimed=context_present, unresolved_or_needs_more_frames=unresolved,
            records=entries, automatic_semantic_pass=False))
    by_window = defaultdict(list)
    for c in conditions:
        by_window[c['window_id']].append(c)
    complete_windows = [w for w, cs in by_window.items() if all(c['represented'] and c['context_start_end_claimed'] for c in cs)]
    report = dict(status='EVIDENCE_COVERAGE_AUDIT_ONLY_NOT_SEMANTIC_APPROVAL',
        created_utc=datetime.now(timezone.utc).isoformat(), source_pins={str(coverage_path):sha(coverage_path),str(context_path):sha(context_path)},
        expected_windows=len(expected), expected_window_conditions=len(condition_expected),
        valid_material_windows=len(materials), missing_material_windows=missing, material_errors=material_errors,
        ledger_snapshots=ledger_snapshots, invalid_ledger_rows=invalid, unrepresented_condition_claims=ignored_conditions,
        conditions_with_valid_viewing_claims=sum(c['represented'] for c in conditions),
        windows_with_all_condition_peaks_and_context_claimed=len(complete_windows),
        fully_claimed_window_ids=complete_windows,
        unresolved_condition_ids=[c['condition_review_id'] for c in conditions if c['unresolved_or_needs_more_frames']],
        missing_condition_ids=[c['condition_review_id'] for c in conditions if not c['represented']],
        incomplete_context_condition_ids=[c['condition_review_id'] for c in conditions if c['represented'] and not c['context_start_end_claimed']],
        conditions=conditions, training_allowed=False, semantic_pass=False,
        limitations=['A matching image and ledger is not independent proof of human viewing; the named reviewer owns that claim.',
                    'Sparse views cover windows/conditions, not every source frame or sign-language semantic correctness.',
                    'Unresolved entries remain flagged even if a later record says resolved; root must explicitly adjudicate lineage.',
                    'All flagged windows must have evidence; missing material or ledger cannot be assumed clean.'],
        generator_sha256=sha(__file__))
    with Path(output).open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('expected_windows','expected_window_conditions','valid_material_windows','conditions_with_valid_viewing_claims','windows_with_all_condition_peaks_and_context_claimed')},ensure_ascii=False))


def self_test():
    material={'selected_frames':[{'frame_id':10,'valid_hand_counts':{'native':{'L':21,'R':0},'masked':{'L':0,'R':0},'sensor':{'L':0,'R':0}}},
                                {'frame_id':20,'valid_hand_counts':{'native':{'L':21,'R':0},'masked':{'L':21,'R':0},'sensor':{'L':0,'R':0}}}]}
    assert peak(material,dict(comparison='native_to_mask',side='left',focus_frame_ids=[10,20]))==10
    assert peak(material,dict(comparison='mask_to_sensor',side='left',focus_frame_ids=[20]))==20
    assert peak(material,dict(comparison='mask_to_sensor',side='right',focus_frame_ids=[10,20]))==10
    print('PASS: 3 independent peak/side/focus checks; no RGB or model execution')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--ledger',action='append',default=[]);p.add_argument('--out');p.add_argument('--require-materials',action='store_true');p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test()
    else:
        assert a.out, '--out must name a new immutable audit report'
        audit(a.ledger,a.out,a.require_materials)
