"""Independent verification of saved omission diagnostics, with no legacy imports.

Only numpy and standard-library modules are imported. No extraction, DP,
classification, training, production scoring, or prior experiment is executed.
"""
import hashlib
import json
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
P = O / 'word100_motion_phase_alignment_20261022'
L = O / 'word100_lopo_reference_range_20261022'
I = O / 'word100_cross_person_motion_support_20261022' / 'inputs'
GROUPS = {'angle': slice(0, 15), 'palm': slice(15, 17), 'elbow': slice(17, 19), 'wrist': slice(19, 21)}
KINDS = ('skip3_25', 'skip3_50', 'skip3_75')
NORMAL = ('identity', 'fast075', 'slow15', 'pace', 'pause')
EPS = 1e-6


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(p, x):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)


def check(pins):
    for path, digest in pins.items():
        assert sha(path) == digest, path


def eq(a, b):
    aa, bb = np.asarray(a), np.asarray(b)
    assert aa.shape == bb.shape, (aa.shape, bb.shape)
    if aa.dtype.kind in 'biu' and bb.dtype.kind in 'biu':
        np.testing.assert_array_equal(aa, bb)
    else:
        np.testing.assert_allclose(aa, bb, rtol=2e-11, atol=2e-11, equal_nan=True)


def jsame(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), (set(a), set(b))
        for k in a:
            jsame(a[k], b[k])
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            jsame(x, y)
    elif isinstance(a, float):
        eq(a, b)
    else:
        assert a == b, (a, b)


def channels(e):
    n = len(e['core'])
    low = np.full((n, 2, 21), np.nan)
    high = low.copy()
    valid = np.zeros(low.shape, bool)
    for h in range(2):
        low[:, h, :15], high[:, h, :15] = e['angle_low'][:, h], e['angle_high'][:, h]
        valid[:, h, :15] = e['angle_valid'][:, h]
        for axis in range(2):
            low[:, h, 15 + axis] = e['centre'][:, h, axis] - e['radius'][:, h]
            high[:, h, 15 + axis] = e['centre'][:, h, axis] + e['radius'][:, h]
            valid[:, h, 15 + axis] = e['palm_valid'][:, h]
            for point in range(2):
                k = 17 + 2 * point + axis
                low[:, h, k] = high[:, h, k] = e['arm_xy'][:, h, point, axis]
                valid[:, h, k] = e['arm_valid'][:, h, point]
    valid &= np.isfinite(low) & np.isfinite(high)
    low[~valid] = np.nan
    high[~valid] = np.nan
    return {'low': low, 'high': high, 'valid': valid}


def domain(e, req):
    ans = np.zeros((len(e['core']), 2, 21), bool)
    for h, side in enumerate(('L', 'R')):
        if side in req:
            ans[:, h, :] = (e['core'] & ~e['source_rest'][:, h])[:, None]
    return ans


def select(c, ix):
    lower, upper = np.floor(ix).astype(int), np.ceil(ix).astype(int)
    result = {'valid': c['valid'][lower] & c['valid'][upper]}
    w = (ix - lower)[:, None, None]
    for k in ('low', 'high'):
        result[k] = c[k][lower] * (1 - w) + c[k][upper] * w
        result[k][~result['valid']] = np.nan
    return result


def project(c, path, n, neighbors):
    low = np.full((n, 2, 21), np.nan)
    high = low.copy()
    complete = np.zeros(low.shape, bool)
    byref = [[] for _ in range(n)]
    for i, j in path:
        byref[int(j)].append(int(i))
    for j, seq in enumerate(byref):
        ids = sorted(set(seq))
        assert ids, ('reference row without correspondence', j)
        flags = c['valid'][ids]
        complete[j] = flags.sum(0) == len(ids)
        seen = flags.any(0)
        lo = np.full((2, 21), np.inf)
        hi = np.full((2, 21), -np.inf)
        for i in ids:
            for k in ((i - 1, i, i + 1) if neighbors else (i,)):
                if 0 <= k < len(c['valid']):
                    usable = c['valid'][i] & c['valid'][k]
                    lo[usable] = np.minimum(lo[usable], c['low'][k][usable])
                    hi[usable] = np.maximum(hi[usable], c['high'][k][usable])
        low[j][seen], high[j][seen] = lo[seen], hi[seen]
    return {'low': low, 'high': high, 'valid': complete}


def compare(p, t, wanted):
    observed = wanted & t['support'] & p['valid']
    distance = np.maximum.reduce([t['low'] - p['high'], p['low'] - t['high'], np.zeros(wanted.shape)])
    bad = observed & (distance > EPS)
    return {'comparable': observed, 'bad': bad, 'unknown': wanted & ~observed,
            'within': observed & ~bad, 'gap': np.where(observed, distance, np.nan)}


def remap_identity(original, source_index):
    assert np.all(source_index == source_index.astype(int)) and np.all(np.diff(source_index) > 0)
    pairs = set()
    for i, j in original:
        # Python min preserves the earliest retained index in an exact tie.
        k = min(range(len(source_index)), key=lambda z: abs(int(source_index[z]) - int(i)))
        pairs.add((k, int(j)))
    return np.array(sorted(pairs), dtype=np.int64)


def counts(z, mask):
    return {k: int(np.count_nonzero(z[k] & mask)) for k in ('comparable', 'bad', 'unknown', 'within')}


def contrast(z, identity, mask):
    out = counts(z, mask)
    out.update(new_bad=int((z['bad'] & ~identity['bad'] & mask).sum()),
               lost_observation=int((identity['comparable'] & ~z['comparable'] & mask).sum()),
               original_bad=int((identity['bad'] & mask).sum()))
    return out


def witness(z, base, actual, actual_base, mask):
    common = mask & z['comparable'] & base['comparable'] & actual['comparable'] & actual_base['comparable']
    alt_new = z['bad'] & ~base['bad']
    actual_new = actual['bad'] & ~actual_base['bad']
    return common & alt_new & ~actual_new


def source_summary(e, c, req, deleted):
    target = domain(e, req)[deleted]
    hand = target[:, :, 0]
    out = dict(deleted=deleted.tolist(), expected_hand_frames=int(hand.sum()),
        raw_observed_hand_frames=int((hand & e['observed'][deleted]).sum()),
        location_seen_hand_frames=int((hand & e['location_seen'][deleted]).sum()),
        positive_rest_hand_frames=int(e['source_rest'][deleted].sum()),
        expected_angle_cells=int(target[:, :, :15].sum()),
        supported_angle_cells=int((target[:, :, :15] & c['valid'][deleted, :, :15]).sum()),
        raw_observed_but_angle_unsupported=int((target[:, :, :15] & e['observed'][deleted, :, None] & ~c['valid'][deleted, :, :15]).sum()),
        explicitly_missing_angle_cells=int((target[:, :, :15] & e['missing'][deleted]).sum()),
        observed_motion_hand_frames=int((hand & e['measured_change'][deleted]).sum()), groups={})
    left, right = int(deleted[0]) - 1, int(deleted[-1]) + 1
    assert left >= 0 and right < len(e['core'])
    assert np.all(np.diff(deleted) == 1) and len(deleted) == 3
    for name, sl in GROUPS.items():
        g, chord = [], []
        possible = observed = 0
        for f in deleted:
            w = (int(f) - left) / (right - left)
            for h in range(2):
                for k in range(sl.start, sl.stop):
                    if not target[int(f) - int(deleted[0]), h, k]:
                        continue
                    possible += 1
                    if not c['valid'][f, h, k]:
                        continue
                    observed += 1
                    if not (c['valid'][left, h, k] and c['valid'][right, h, k]):
                        continue
                    a, b = c['low'][f, h, k], c['high'][f, h, k]
                    lo = min(c['low'][left, h, k], c['low'][right, h, k])
                    hi = max(c['high'][left, h, k], c['high'][right, h, k])
                    cl = (1 - w) * c['low'][left, h, k] + w * c['low'][right, h, k]
                    ch = (1 - w) * c['high'][left, h, k] + w * c['high'][right, h, k]
                    g.append(max(lo - b, a - hi, 0.))
                    chord.append(max(cl - b, a - ch, 0.))
        out['groups'][name] = dict(expected=possible, observed=observed, endpoint_eligible=len(g),
            outside_endpoint_hull=sum(v > EPS for v in g), outside_linear_chord=sum(v > EPS for v in chord),
            max_endpoint_gap=max(g) if g else None, max_chord_gap=max(chord) if chord else None)
    return out


def fixtures():
    assert not (U / 'FIXTURE_RESULT.json').exists(), 'Saved fixtures must not be rerun'
    c = {'low': np.broadcast_to(np.arange(5.)[:, None, None], (5, 2, 21)).copy(),
         'high': np.broadcast_to(np.arange(5.)[:, None, None], (5, 2, 21)).copy(),
         'valid': np.ones((5, 2, 21), bool)}
    p = np.column_stack([np.arange(5), np.arange(5)])
    plain, expanded = project(c, p, 5, False), project(c, p, 5, True)
    eq(plain['low'], c['low'])
    eq(expanded['low'][2], np.ones((2, 21)))
    eq(expanded['high'][2], np.full((2, 21), 3.))
    missing = {k: v.copy() for k, v in c.items()}
    missing['valid'][2, 0, 4] = False
    for k in ('low', 'high'):
        missing[k][2, 0, 4] = np.nan
    mp = project(missing, p, 5, True)
    assert not mp['valid'][2, 0, 4] and np.isnan(mp['low'][2, 0, 4])
    assert mp['valid'][2, 0, 3] and mp['valid'][2, 0, 5]
    t = {'low': plain['low'], 'high': plain['high'], 'support': np.ones((5, 2, 21), bool)}
    wanted = t['support'].copy()
    z = compare(mp, t, wanted)
    assert z['unknown'][2, 0, 4] and not z['within'][2, 0, 4] and not z['bad'][2, 0, 4]
    wanted[:, 1] = False
    assert not compare(mp, t, wanted)['unknown'][:, 1].any()
    rp = remap_identity(p, np.array([0., 2., 4.]))
    eq(rp, np.array([[0, 0], [0, 1], [1, 2], [1, 3], [2, 4]]))
    original = {'comparable': np.ones((1, 2, 21), bool), 'bad': np.ones((1, 2, 21), bool)}
    mask = original['comparable'].copy()
    assert contrast(dict(original, unknown=~mask, within=~mask), original, mask)['new_bad'] == 0
    a = dict(original, bad=mask.copy())
    b = dict(original, bad=~mask)
    noobs = {'comparable': ~mask, 'bad': ~mask}
    assert witness(a, b, b, b, mask).all() and not witness(a, b, noobs, b, mask).any()
    result = dict(at=now(), status='PASS', checks={
        'identity_projection_exact': True, 'neighbor_envelope_expected': True,
        'missing_direct_sample_not_rescued_by_neighbors': True,
        'neighbor_angles_preserved': True, 'unknown_not_within_range': True,
        'nonrequired_hand_not_error': True, 'held_path_ties_earlier': True,
        'same_contract_existing_mismatch_not_new': True, 'witness_requires_all_four_comparabilities': True},
        sha256={'verify.py': sha(U / 'verify.py'), 'study.py': sha(U / 'study.py')})
    write(U / 'FIXTURE_RESULT.json', result)
    print('FIXTURES PASS', len(result['checks']), flush=True)


def run():
    assert (U / 'RESULT.json').exists() and not (U / 'RUNNING.json').exists()
    assert not (U / 'VERIFY.json').exists(), 'Completed verification must not be rerun'
    fixture = read(U / 'FIXTURE_RESULT.json')
    assert fixture['status'] == 'PASS' and len(fixture['checks']) == 9 and all(fixture['checks'].values())
    for file, pinned in fixture['sha256'].items():
        assert sha(U / file) == pinned, file
    f, result, old = read(U / 'FROZEN.json'), read(U / 'RESULT.json'), read(P / 'ANALYSIS.json')
    recovery = read(U / 'RECOVERY.json')
    check(recovery['immutable'])
    assert sha(U / 'study_recovery.py') == recovery['recovery_code_sha256']
    check(f['pins']); check(f['protected'])
    assert [len(result[k]) for k in ('sources', 'pairs', 'normal')] == [81, 1296, 2160]
    assert result['whole_action_correct'] is None
    meta = {r['clip_id']: r for r in f['rows']}
    assert len(meta) == 27 and len({r['word_code'] for r in meta.values()}) == 3
    samples = {cid: load(I / (cid + '.npz')) for cid in meta}
    ch = {cid: channels(e) for cid, e in samples.items()}
    src = {(r['query'], r['kind']): r for r in result['sources']}
    pairs = {(r['query'], r['reference'], r['kind'], r['method']): r for r in result['pairs']}
    normal = {(r['query'], r['reference'], r['kind'], r['method']): r for r in result['normal']}
    assert len(src) == 81 and len(pairs) == 1296 and len(normal) == 2160
    prevpair = {(r['query'], r['reference'], r['kind']): r for r in old['omission_pairs']['cross']}
    prevsource = {(r['query'], r['kind']): r for r in old['omissions']['cross']['skip3']['members']}
    prevknown = {(r['query'], r['kind']): r for r in old['omission_pairs']['known']}
    validated = Counter()
    all_hashes = {}
    aggregate_new = {m: Counter() for m in ('old', 'new')}
    old_pins_verified = len(f['pins']) + len(f['protected'])
    for fold in f['folds']:
        cid = fold['query']; e = samples[cid]; qc = ch[cid]
        req = f['roles'].get(meta[cid]['word_code'], {}).get('required_sides', ['L', 'R'])
        assert len(fold['references']) == 8 and len(set(meta[r]['signer_id'] for r in fold['references'])) == 8
        assert meta[cid]['signer_id'] not in [meta[r]['signer_id'] for r in fold['references']]
        for kind in KINDS:
            previous = prevpair[cid, fold['references'][0], kind]
            deleted = np.asarray(previous['removed_source_frames'], int)
            expect = source_summary(e, qc, req, deleted)
            expect.update(query=cid, kind=kind, label=meta[cid]['label'], previous=prevsource[cid, kind],
                          self_source={m: prevknown[cid, kind][m] for m in ('old', 'new')})
            jsame(expect, src[cid, kind]); validated['sources'] += 1
        for rid in fold['references']:
            re, rc = samples[rid], ch[rid]
            wanted = domain(re, req)
            template = load(L / 'templates' / (cid + '__' + rid + '.npz'))
            eq(wanted, template['expected'])
            assert len(template['member_complete']) == 8
            eq(template['support'], template['member_complete'].sum(0) >= 5)
            anchor = {'low': rc['low'], 'high': rc['high'], 'support': rc['valid']}
            templates = {'pool': template, 'anchor': anchor}
            original = load(P / 'cross' / ('identity__' + cid + '__' + rid + '.npz'))
            identities = {}
            for method, key in (('old', 'base_path'), ('new', 'path')):
                for nb in (0, 1):
                    proj = project(qc, original[key], len(wanted), nb)
                    for tt, tpl in templates.items():
                        identities[method, nb, tt] = compare(proj, tpl, wanted)
            for kind in KINDS:
                name = kind + '__' + cid + '__' + rid
                saved = load(P / 'cross' / (name + '.npz'))
                ix = saved['source_index']
                deleted = np.setdiff1d(np.arange(len(e['core'])), ix)
                assert len(ix) == len(e['core']) - 3
                eq(deleted, prevpair[cid, rid, kind]['removed_source_frames'])
                dqc = select(qc, ix)
                affected = np.zeros(len(wanted), bool)
                for i, j in original['base_path']:
                    if i in deleted:
                        affected[j] = True
                inside = wanted & affected[:, None, None]
                for method, key, prefix in (('old', 'base_path', 'old_'), ('new', 'path', '')):
                    file = U / 'cases' / (method + '__' + name + '.npz')
                    actual = load(file); all_hashes[str(file)] = sha(file)
                    eq(actual['inside'], inside); eq(actual['expected'], wanted)
                    eq(actual['source_index'], ix); eq(actual['deleted'], deleted)
                    paths = {'actual': saved[key], 'held': remap_identity(original[key], ix)}
                    compared, projections, config = {}, {}, {}
                    keys = {'inside', 'expected', 'source_index', 'deleted'}
                    for mode, path in paths.items():
                        eq(actual[mode + '_path'], path); keys.add(mode + '_path')
                        for nb in (0, 1):
                            pp = project(dqc, path, len(wanted), nb); projections[mode, nb] = pp
                            for tt, tpl in templates.items():
                                tag = f'{mode}_{tt}_{nb}'
                                out = compare(pp, tpl, wanted); compared[tag] = out
                                config[tag] = contrast(out, identities[method, nb, tt], inside)
                                for k in ('comparable', 'bad', 'unknown'):
                                    eq(actual[tag + '_' + k], out[k]); keys.add(tag + '_' + k)
                    active, baseline = compared['actual_pool_1'], identities[method, 1, 'pool']
                    eq(active['bad'], saved[prefix + 'mismatch'])
                    eq(active['comparable'], saved[prefix + 'comparable'])
                    assert config['actual_pool_1']['new_bad'] == prevpair[cid, rid, kind][method]['inside_raw_cells']
                    factors = {}
                    for factor, tag, base in (('alignment', 'held_pool_1', baseline),
                            ('range', 'actual_anchor_1', identities[method, 1, 'anchor']),
                            ('neighbors', 'actual_pool_0', identities[method, 0, 'pool'])):
                        w = witness(compared[tag], base, active, baseline, inside)
                        eq(actual[factor + '_witness'], w); keys.add(factor + '_witness')
                        factors[factor] = int(w.sum())
                    pp = projections['actual', 1]
                    any_individual = np.zeros(wanted.shape, bool)
                    for member in range(8):
                        any_individual |= template['member_complete'][member] & (template['member_high'][member] >= pp['low'] - EPS) & (template['member_low'][member] <= pp['high'] + EPS)
                    convex = inside & active['within'] & ~any_individual
                    eq(actual['convex_gap'], convex); keys.add('convex_gap')
                    assert set(actual) == keys
                    shifts, far = [], []
                    for j in np.flatnonzero(affected):
                        mapped = ix[saved[key][saved[key][:, 1] == j, 0]]
                        origins = original[key][original[key][:, 1] == j, 0]
                        shifts.append(float(np.median(mapped) - np.median(origins)))
                        far.append(bool(np.all((mapped < deleted[0] - 1) | (mapped > deleted[-1] + 1))))
                    rec = pairs[cid, rid, kind, method]
                    expected = dict(name=name, query=cid, reference=rid, kind=kind, method=method,
                        config=config, factors=factors, convex_gap_cells=int(convex.sum()),
                        affected_reference_frames=int(affected.sum()), original_bad=config['actual_pool_1']['original_bad'],
                        local_new_bad=config['actual_pool_1']['new_bad'], local_unknown=config['actual_pool_1']['unknown'],
                        local_expected=int(inside.sum()), reference_rows_far_from_deleted_neighbors=sum(far),
                        median_source_shift=float(np.median(shifts)) if shifts else None,
                        interpretation='Paired diagnostic sensitivity; not mutually exclusive causal proof or semantic verdict.')
                    jsame(expected, rec)
                    aggregate_new[method][cid, kind] += int(rec['local_new_bad'] > 0)
                    validated['cases'] += 1
            for kind in NORMAL:
                saved = original if kind == 'identity' else load(P / 'cross' / (kind + '__' + cid + '__' + rid + '.npz'))
                cc = select(qc, saved['source_index'])
                for method, key, prefix in (('old', 'base_path', 'old_'), ('new', 'path', '')):
                    config = {}
                    for nb in (0, 1):
                        pp = project(cc, saved[key], len(wanted), nb)
                        for tt, tpl in templates.items():
                            out = compare(pp, tpl, wanted)
                            config[f'{tt}_{nb}'] = dict(counts(out, wanted), any_bad=bool(out['bad'].any()))
                            if nb == 1 and tt == 'pool':
                                eq(out['bad'], saved[prefix + 'mismatch'])
                    jsame(dict(query=cid, reference=rid, kind=kind, method=method, config=config), normal[cid, rid, kind, method])
                    validated['normal'] += 1
        print('VERIFIED CLIP', cid, dict(validated), flush=True)
    assert dict(validated) == {'sources': 81, 'cases': 1296, 'normal': 2160}
    assert len(list((U / 'cases').glob('*.npz'))) == 1296
    misses = {}
    for method in ('old', 'new'):
        assert len(aggregate_new[method]) == 81
        for key, n in aggregate_new[method].items():
            assert n == prevsource[key][method]['inside_raw']
        misses[method] = {key for key, n in aggregate_new[method].items() if n == 0}
    focus = misses['old'] | misses['new']
    assert [len(misses['old']), len(misses['new']), len(focus)] == [12, 9, 18]
    expected_controls = {'normal_timing': {k: old['known'][k] for k in NORMAL},
        'rest_proxy': {m: old['cross']['identity'][m + '_proxy'] for m in ('old', 'new')},
        'missing135_verified': str(P / 'VERIFY.json')}
    jsame(expected_controls, result['unchanged_controls'])
    assert read(P / 'VERIFY.json')['status'] == 'PASS'
    check(f['pins']); check(f['protected']); check(recovery['immutable'])
    for name in ('study.py', 'study_recovery.py', 'RECOVERY.json', 'FIRST_ERROR.txt', 'FAILED_RUNNING.json',
                 'verify.py', 'FROZEN.json', 'RESULT.json', 'FIXTURE_RESULT.json', 'DESIGN_REVIEW.json'):
        all_hashes[str(U / name)] = sha(U / name)
    write(U / 'VERIFY.json', dict(at=now(), status='PASS', scope=dict(validated),
        checks={'all_8_diagnostic_configs_per_old_new_reference_pair': True,
            'fixed_original_affected_domain': True, 'own_identity_same_contract': True,
            'actual_pool_neighbors_matches_frozen_prior': True,
            'all_four_comparabilities_required_for_witness': True,
            'nearest_survivor_tie_earlier_duplicate_mapping': True,
            'source_evidence_and_frame_linear_interval_chord': True,
            'normal_speed_tradeoff_controls': True, 'convex_gap_individual_interval_check': True,
            'unknown_not_within_range': True, 'old_pins_and_models_preserved': True},
        focus={'old_no_new_difference': len(misses['old']), 'new_no_new_difference': len(misses['new']),
               'union': len(focus), 'intersection': len(misses['old'] & misses['new']),
               'members': [{'query': q, 'kind': k} for q, k in sorted(focus)]},
        pinned_old_files_checked=old_pins_verified, sha256=all_hashes,
        recovered_driver={'initial_error': 'Omitted inherited default both-hands role for words without explicit override.',
            'partial_cases_preserved': recovery['partial_saved_cases'],
            'original_code_error_and_partial_hashes_checked': True, 'scientific_conditions_changed': False},
        implementation='Separate Codex agent; standalone numpy and standard library; no production comparison functions or prior scripts imported.',
        limits=['Arithmetic and fixed-contract diagnosis only, not semantic-stage labels or a validated operational criterion.',
            'Known controls and previous missing135 verification reused through pinned saved results; not rerun.',
            'Factors overlap; no counterfactual is treated as true cross-person timing.']))
    print('INDEPENDENT DIAGNOSIS VERIFY PASS', dict(validated), flush=True)


if __name__ == '__main__':
    try:
        {'fixtures': fixtures, 'run': run}[sys.argv[1]]()
    except BaseException:
        error = U / 'VERIFY_FIRST_ERROR.txt'
        if not error.exists():
            error.write_text(traceback.format_exc(), encoding='utf8')
        raise
