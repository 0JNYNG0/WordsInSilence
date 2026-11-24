"""Read-only saved-path diagnosis. No producer import, inference, or DTW call."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

U = Path(__file__).resolve().parent.parent
PINS = {}
CHECKS = 0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pin(p):
    p = Path(p).resolve()
    PINS[str(p)] = sha(p)
    return p


def read(p):
    return json.loads(pin(p).read_text(encoding='utf-8-sig'))


def arrays(p):
    with np.load(pin(p), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def check(value, name):
    global CHECKS
    CHECKS += 1
    if not value:
        raise AssertionError(name)


def runs(ix):
    ix = sorted(set(map(int, ix)))
    out = []
    for i in ix:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def frames(ix, a):
    ix = sorted(set(map(int, ix)))
    return dict(count=len(ix), indices=ix, index_runs=runs(ix),
                frame_ids=[int(a['frame_id'][i]) for i in ix],
                timestamp_us=[int(a['timestamp_us'][i]) for i in ix])


def selected_path(path, interval):
    a = np.asarray(path, dtype=int)
    return a[(a[:, 1] >= interval[0]) & (a[:, 1] <= interval[1])]


def angle_support_on_saved_path(path, interval, qmask, rmask):
    """Independent count only, not a rerun of comparison or classification."""
    a = selected_path(path, interval)
    rr = np.arange(interval[0], interval[1] + 1)
    qm = qmask[:, 39:53].astype(bool)
    rm = rmask[:, 39:53].astype(bool)
    re = int(rm[rr].sum())
    ro = sum(int((qm[a[a[:, 1] == j, 0]].any(0) & rm[j]).sum()) for j in rr)
    qe = qo = 0
    for i in np.unique(a[:, 0]):
        need = rm[a[a[:, 0] == i, 1]].any(0)
        qe += int(need.sum())
        qo += int((qm[i] & need).sum())
    return dict(reference_expected=re, reference_observed=ro,
                query_expected=qe, query_observed=qo,
                coverage=min(ro / re, qo / qe) if re and qe else None)


def destination(path, removed, row, ref):
    a = np.asarray(path, dtype=int)
    a = a[np.isin(a[:, 0], removed)]
    lo = min(s['frames'][0] for s in row['stages'])
    hi = max(s['frames'][1] for s in row['stages'])
    def category(j):
        if j < lo:
            return 'before_first_reference_stage'
        if j > hi:
            return 'after_last_reference_stage'
        if any(s['frames'][0] <= j <= s['frames'][1] for s in row['stages']):
            return 'inside_reference_stage'
        return 'gap_between_reference_stages'
    return dict(path_links=a.tolist(), query_count=len(set(a[:, 0].tolist())),
                reference_frames=frames(a[:, 1], ref),
                link_categories=dict(Counter(category(int(j)) for _, j in a)),
                query_categories={k: sorted(set(int(i) for i, j in a if category(int(j)) == k))
                                  for k in sorted({category(int(j)) for _, j in a})})


def main():
    evaluation = read(U / 'FEEDBACK_EVALUATION.json')
    tasks = read(U / 'EVALUATION_TASKS.json')['tasks']
    bank = read(U / 'REFERENCE_BANK.json')
    rows = {r['clip_id']: r for r in bank['rows']}
    for name in ['engine.py', 'evaluate_feedback.py', 'NORMAL_RANGES.json', 'THREE_REFERENCE_RANGES.json']:
        pin(U / name)
    same = {t['query']: t for t in tasks if t['kind'] == 'same_expression'}
    cases = []
    survivors = []
    observed_counts = {'all9': Counter(), 'subset3': Counter()}
    for t in tasks:
        if t['kind'] != 'main_R_hand_missing':
            continue
        c = read(U / 'evaluation_cases' / (t['id'] + '.json'))
        bt = same[t['query']]
        bc = read(U / 'evaluation_cases' / (bt['id'] + '.json'))
        src = arrays(t['source_input'])
        edited = arrays(c['modified_input']['path'])
        prob = arrays(t['source_probabilities'])
        removed = np.asarray(t['main_indices'], dtype=int)
        dims = np.asarray(t['masked_feature_columns'], dtype=int)
        pp = np.asarray(t['masked_posterior_columns'], dtype=int)
        check(sha(t['source_input']) == t['source_input_sha256'], 'source input pin')
        check(sha(t['source_probabilities']) == t['source_probabilities_sha256'], 'source probability pin')
        check(sha(c['modified_input']['path']) == c['modified_input']['sha256'], 'modified input pin')
        for key in ['x', 'frame_id', 'timestamp_us']:
            check(np.array_equal(src[key], edited[key], equal_nan=True), 'unchanged ' + key)
        intended = src['mask'].copy()
        intended[np.ix_(removed, dims)] = False
        check(np.array_equal(intended, edited['mask']), 'exact fixed mask intervention')
        check(np.isnan(edited['probabilities'][np.ix_(removed, pp)]).all(), 'masked posterior unknown')
        expected_removed = int(src['mask'][np.ix_(removed, dims)].sum())
        check(expected_removed == c['edit_ledger']['newly_removed_feature_cells'], 'removed cells')
        check(removed.tolist() == c['edit_ledger']['edited_indices'], 'removed indices ledger')
        in_intervals = np.zeros(len(src['frame_id']), bool)
        for lo, hi in t['main_intervals']:
            in_intervals |= (src['timestamp_us'] / 1e6 >= lo) & (src['timestamp_us'] / 1e6 <= hi)
        check(np.array_equal(np.flatnonzero(in_intervals), removed), 'all declared main samples intervened')
        full = dict(task_id=t['id'], baseline_task_id=bt['id'], query=t['query'],
                    main_intervals=t['main_intervals'], removed=frames(removed, src),
                    newly_removed_feature_cells=expected_removed,
                    newly_removed_posterior_cells=c['edit_ledger']['newly_removed_posterior_cells'],
                    selected_whole_references={m: c['modes'][m]['selected_whole_references'] for m in observed_counts},
                    baseline_selected_whole_references={m: bc['modes'][m]['selected_whole_references'] for m in observed_counts},
                    pairs=[])
        saved_pairs = {}
        for rid in t['references']:
            row = rows[rid]
            ref = arrays(row['input'])
            z = read(U / 'evaluation_pairs' / (t['id'] + '__' + rid + '.json'))
            bz = read(U / 'evaluation_pairs' / (bt['id'] + '__' + rid + '.json'))
            for value in [z, bz]:
                path = np.asarray(value['path'])
                check(np.array_equal(path[0], [0, 0]), 'full path start')
                check(np.array_equal(path[-1], [len(src['frame_id']) - 1, len(ref['frame_id']) - 1]), 'full path end')
                check(np.all(np.isin(np.diff(path, axis=0), [0, 1])), 'monotone steps')
                check(np.all(np.diff(path, axis=0).sum(1) >= 1), 'no repeated cell')
                check(np.allclose(value['query_frame_times'], src['timestamp_us'] / 1e6, atol=0, rtol=0), 'query PTS')
                check(np.allclose(value['reference_frame_times'], ref['timestamp_us'] / 1e6, atol=0, rtol=0), 'reference PTS')
            detail = dict(reference=rid, baseline_distance=bz['distance'], control_distance=z['distance'],
                          removed_destinations={
                              'baseline': destination(bz['path'], removed, row, ref),
                              'control': destination(z['path'], removed, row, ref)}, stages=[])
            for j, stage in enumerate(row['stages']):
                a = selected_path(z['path'], stage['frames'])
                b = selected_path(bz['path'], stage['frames'])
                qs, bqs = np.unique(a[:, 0]), np.unique(b[:, 0])
                cg, bg = z['stages'][j]['families'], bz['stages'][j]['families']
                before_hit = np.intersect1d(bqs, removed)
                after_hit = np.intersect1d(qs, removed)
                independent_support = angle_support_on_saved_path(z['path'], stage['frames'], edited['mask'], ref['mask'])
                for field, key in [('reference_expected', 'expected'), ('reference_observed', 'observed'),
                                   ('query_expected', 'query_expected'), ('query_observed', 'query_observed')]:
                    check(independent_support[field] == cg['R_angles'][key], 'independent R angle ' + key)
                check(independent_support['coverage'] == cg['R_angles']['coverage'], 'independent R angle coverage')
                for family in cg:
                    check(cg[family]['original_expected'] == bg[family]['original_expected'], 'original reference denominator preserved')
                    check(cg[family]['scope_excluded'] == bg[family]['scope_excluded'], 'reference-only rest scope unchanged')
                fields = ['coverage', 'reference_coverage', 'query_coverage', 'expected', 'observed',
                          'query_expected', 'query_observed', 'original_expected', 'scope_excluded',
                          'dimension_reference_expected', 'dimension_query_expected', 'dimension_coverage',
                          'distance', 'peak_distance', 'peak_at', 'support_provenance']
                sd = dict(stage_index=j + 1, canonical_stage_id=stage.get('canonical_stage_id'),
                          reference_stage_id=stage['id'], reference_frames=frames(range(stage['frames'][0], stage['frames'][1] + 1), ref),
                          boundary_provenance=stage.get('boundary_provenance'),
                          baseline_query_frames=frames(bqs, src), control_query_frames=frames(qs, src),
                          baseline_removed_intersection=frames(before_hit, src), control_removed_intersection=frames(after_hit, src),
                          baseline_compression=bz['stages'][j]['compression'], control_compression=z['stages'][j]['compression'],
                          selected_by_control_modes=[m for m in observed_counts if rid in c['modes'][m]['selected_whole_references']],
                          families={k: {'baseline': {f: bg[k].get(f) for f in fields},
                                        'control': {f: cg[k].get(f) for f in fields}} for k in cg},
                          baseline_path_edited_mask_R_angles_support=angle_support_on_saved_path(bz['path'], stage['frames'], edited['mask'], ref['mask']),
                          interpretation='SAVED_BASELINE_OVERLAP_DISAPPEARS_AFTER_REMOVAL' if len(before_hit) and not len(after_hit)
                          else 'PARTIAL_OR_NO_SCOPE_SHIFT')
                detail['stages'].append(sd)
            saved_pairs[rid] = detail
            full['pairs'].append(detail)
        for mode in observed_counts:
            for j, s in enumerate(c['modes'][mode]['stages']):
                observed_counts[mode][s['status']] += 1
                if s['status'] != 'SIMILAR_REFERENCE_RANGE':
                    continue
                rids = c['modes'][mode]['selected_whole_references']
                findings = [saved_pairs[rid]['stages'][j] for rid in rids]
                check(all(v['baseline_removed_intersection']['count'] > 0 for v in findings), 'surviving similar was actually affected on saved baseline')
                check(all(v['control_removed_intersection']['count'] == 0 for v in findings), 'surviving similar excludes every removed query sample')
                check(all(all(f['control']['scope_excluded'] == 0 for f in v['families'].values()) for v in findings), 'not a rest exclusion explanation')
                survivors.append(dict(task_id=t['id'], query=t['query'], mode=mode, stage_index=j + 1,
                                      status=s['status'], baseline_aggregate_status=bc['modes'][mode]['stages'][j]['status'],
                                      removed_query_frames=frames(removed, src), selected_whole_references=rids,
                                      selected_reference_stage_details=findings,
                                      conclusion='Scope escape: fixed declared main samples were masked, and their prior stage overlap is absent in the new selected query range. Coverage 1.0 is conditional on that changed range, not protection against missing main evidence.'))
        cases.append(full)
    check(len(cases) == 12, 'all fixed missing controls')
    check(sum(observed_counts['all9'].values()) == 39, 'all9 stage denominator')
    check(observed_counts['all9']['SIMILAR_REFERENCE_RANGE'] == 3, 'three all9 survivors')
    check(sum(observed_counts['subset3'].values()) == 39, 'subset3 stage denominator')
    check(observed_counts['subset3']['SIMILAR_REFERENCE_RANGE'] == 3, 'three subset3 survivors')
    for mode, counts in observed_counts.items():
        check(sum(counts.values()) == evaluation['per_kind']['main_R_hand_missing']['modes'][mode]['stage_opportunities'], 'summary stage opportunity')
    unchanged = all(sha(p) == value for p, value in PINS.items())
    check(unchanged, 'all read inputs unchanged')
    output = dict(status='COMPLETE_SAVED_SCOPE_DIAGNOSIS', checks=CHECKS,
        task='main_R_hand_missing surviving similar scope audit', source_pins=PINS,
        all_read_source_hashes_unchanged=unchanged, public_video_or_archive_reads=0,
        inference=0, training=0, DTW_or_path_reoptimization=0, producer_imports=0,
        engine_modified=False, new_scores_or_thresholds=False,
        denominator=dict(controls=12, references_per_control=9, saved_control_and_same_reference_baseline_pairs=216,
                         stage_opportunities_per_mode=39),
        stage_status_counts={m: dict(c) for m, c in observed_counts.items()},
        key_conclusion='All three all9 surviving similar stages (and the same three subset3 stages) are real saved-path scope-escape counterexamples, not controls that missed those stages on the unedited same-reference path.',
        findings=survivors, all_fixed_cases=cases,
        limits=['Saved baseline paths are not human same-phase truth; this diagnosis establishes response to a declared fixed intervention and changed scope, not linguistic correctness.',
                'An invariant reference denominator and local query coverage do not establish coverage of the removed main query times.',
                'No automatic remedy or new result has been executed. A separately frozen common safeguard must preserve these original failures.'],
        proposed_safeguard_review=dict(status='CONDITIONALLY_REASONABLE_SINGLE_CONSERVATIVE_CANDIDATE',
            proposal='Keep original integrated path and whole-reference selection; an independently masked pose-arm-only full path may add query opportunities for the same reference stage. Only downgrade SIMILAR to INSUFFICIENT if the union fails the existing support criterion.',
            required=['No word/main-ground-truth input to the auxiliary path.', 'Retain original and auxiliary query-frame/attribute opportunity ledgers and paths separately.',
                      'Same reference-rest coordinate scope; no scope widening from query missingness.', 'Auxiliary pose missingness is not positive evidence.',
                      'Do not reselect the two whole references using the auxiliary path.', 'Do not call either path a true stage match.',
                      'Preserve existing DIFFERENCE and INSUFFICIENT; no promotion to similar or new correction advice.',
                      'Replay all fixed development controls only as regression diagnosis, with all new extra abstentions reported.'],
            risks=['Arm/pose can stay similar through different finger actions or repeated occurrences; the auxiliary path may also be wrong.',
                   'Additional insufficient outputs can be unnecessary on legitimate normal speed/holds; no educational accuracy improvement can be concluded.']))
    target = U / 'review' / 'MISSING_SCOPE_ANALYSIS.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=output['status'], checks=CHECKS, source_files=len(PINS), output=str(target), sha256=sha(target), counts=output['stage_status_counts'])))


if __name__ == '__main__':
    main()
