"""Independent arithmetic verification of the coherent sequence sidecar.

No production modules or older executable scripts are imported. This module
only reads frozen public inputs and saved results, then writes this unit's audit.
"""
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
P = O / 'word100_motion_phase_alignment_20261022'
I = O / 'word100_cross_person_motion_support_20261022' / 'inputs'
GROUPS = {'angle': slice(0, 15), 'palm': slice(15, 17), 'elbow': slice(17, 19), 'wrist': slice(19, 21)}
EPS = 1e-6


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def check(pins):
    for path, digest in pins.items():
        assert sha(path) == digest, path


def eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    if a.dtype.kind in 'biuSU' and b.dtype.kind in 'biuSU':
        np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-11, equal_nan=True)


def channels(e):
    size = (len(e['core']), 2, 21)
    low, high = np.full(size, np.nan), np.full(size, np.nan)
    valid = np.zeros(size, bool)
    low[:, :, :15], high[:, :, :15] = e['angle_low'], e['angle_high']
    valid[:, :, :15] = e['angle_valid']
    for hand in range(2):
        for axis in range(2):
            k = 15 + axis
            low[:, hand, k] = e['centre'][:, hand, axis] - e['radius'][:, hand]
            high[:, hand, k] = e['centre'][:, hand, axis] + e['radius'][:, hand]
            valid[:, hand, k] = e['palm_valid'][:, hand]
            for joint in range(2):
                k = 17 + 2 * joint + axis
                low[:, hand, k] = high[:, hand, k] = e['arm_xy'][:, hand, joint, axis]
                valid[:, hand, k] = e['arm_valid'][:, hand, joint]
    valid &= np.isfinite(low) & np.isfinite(high)
    low[~valid] = np.nan; high[~valid] = np.nan
    return dict(low=low, high=high, valid=valid)


def expected(e, sides):
    out = np.zeros((len(e['core']), 2, 21), bool)
    for h, side in enumerate(('L', 'R')):
        if side in sides:
            out[:, h] = (e['core'] & ~e['source_rest'][:, h])[:, None]
    return out


def interpolate(c, index):
    left, right = np.floor(index).astype(int), np.ceil(index).astype(int)
    w = (index - left)[:, None, None]
    good = c['valid'][left] & c['valid'][right]
    out = {'valid': good}
    for key in ('low', 'high'):
        v = c[key][left] * (1 - w) + c[key][right] * w
        v[~good] = np.nan; out[key] = v
    return out


def describe(c, path, rest, domain):
    n = len(domain)
    byref = [[] for _ in range(n)]
    for source, reference in path:
        byref[int(reference)].append(int(source))
    assert all(byref)
    samples = []
    for ids in byref:
        ss = sorted(ids); middle = len(ss) // 2
        samples.append(float(ss[middle]) if len(ss) % 2 else (ss[middle - 1] + ss[middle]) / 2)
    index = np.asarray(samples)
    assert np.all(np.diff(index) >= 0)
    mapped = interpolate(c, index)
    shape = domain.shape
    low = np.full(shape + (5,), np.nan); high = low.copy()
    valid = np.zeros(shape, bool); eligible = valid.copy()
    span_bad = valid.copy(); rest_bad = valid.copy()
    phase_index = np.full((n, 3), np.nan); span = np.full((n, 2), -1, int)
    for j in range(2, n - 2):
        context = np.logical_and.reduce(domain[j - 2:j + 3], axis=0)
        eligible[j] = context
        first, last = min(byref[j - 2]), max(byref[j + 2])
        span[j] = [first, last]
        points = [j - 2, j, j + 2]
        phase_index[j] = index[points]
        full = np.logical_and.reduce(c['valid'][first:last + 1], axis=0)
        is_rest = np.logical_or.reduce(rest[first:last + 1], axis=0)[:, None]
        span_bad[j] = context & ~full
        rest_bad[j] = context & is_rest
        usable = context & full & ~is_rest & np.logical_and.reduce(mapped['valid'][points], axis=0)
        valid[j] = usable
        for phase, ref in enumerate(points):
            low[j, :, :, phase] = mapped['low'][ref]
            high[j, :, :, phase] = mapped['high'][ref]
        for phase in range(2):
            low[j, :, :, 3 + phase] = mapped['low'][points[phase + 1]] - mapped['high'][points[phase]]
            high[j, :, :, 3 + phase] = mapped['high'][points[phase + 1]] - mapped['low'][points[phase]]
    low[~valid] = np.nan; high[~valid] = np.nan
    collapsed = np.zeros(n, bool)
    for j, values in enumerate(phase_index):
        if np.isfinite(values).all():
            collapsed[j] = values[0] == values[1] or values[1] == values[2]
    return dict(low=low, high=high, valid=valid, eligible=eligible, span_bad=span_bad,
                rest_bad=rest_bad, phase_index=phase_index, span=span, collapsed=collapsed, index=index)


def distances(a, b, components):
    all_gap = []
    for k in range(components):
        all_gap.append(np.maximum.reduce([a['low'][..., k] - b['high'][..., k],
            b['low'][..., k] - a['high'][..., k], np.zeros(a['valid'].shape)]))
    return np.maximum.reduce(all_gap)


def calibration(anchor, peers):
    count = np.zeros(anchor['valid'].shape, np.int64)
    level = np.full(count.shape, -np.inf); sequence = level.copy()
    for peer in peers:
        good = anchor['valid'] & peer['valid']
        count += good
        ds = distances(peer, anchor, 3); dq = distances(peer, anchor, 5)
        level[good] = np.maximum(level[good], ds[good])
        sequence[good] = np.maximum(sequence[good], dq[good])
    supported = count >= 5
    level[~supported] = np.nan; sequence[~supported] = np.nan
    return dict(levels_threshold=level, sequence_threshold=sequence, count=count, support=supported)


def compare(query, anchor, cal, domain):
    usable = domain & query['valid'] & anchor['valid'] & cal['support']
    out = dict(expected=domain.copy(), comparable=usable, unknown=domain & ~usable,
        context_unavailable=domain & ~anchor['eligible'],
        anchor_incomplete=domain & anchor['eligible'] & ~anchor['valid'],
        query_incomplete=domain & query['eligible'] & ~query['valid'],
        peer_insufficient=domain & ~cal['support'],
        query_span_bad=query['span_bad'], query_rest_bad=query['rest_bad'])
    for name, components in (('levels', 3), ('sequence', 5)):
        d = distances(query, anchor, components)
        excess = d - cal[name + '_threshold']
        bad = usable & (excess > EPS)
        out[name + '_distance'] = np.where(usable, d, np.nan)
        out[name + '_bad'] = bad
        out[name + '_within'] = usable & ~bad
        out[name + '_excess'] = np.where(usable, excess, np.nan)
    return out


def make_damage(e, side, kind, removed):
    z = {k: v.copy() for k, v in e.items()}
    h = 0 if side == 'L' else 1
    if kind == 'wholearm':
        z['arm_valid'][removed, h] = False; z['arm_xy'][removed, h] = np.nan
    else:
        for key in ('angle_valid', 'palm_valid', 'source_rest', 'observed'):
            z[key][removed, h] = False
        for key in ('angle_low', 'angle_high', 'centre', 'radius'):
            z[key][removed, h] = np.nan
        for key in ('missing', 'unconfirmed'):
            z[key][removed, h] = True
    return z


def counts(out):
    fields = ('expected', 'comparable', 'unknown', 'levels_bad', 'sequence_bad',
              'context_unavailable', 'anchor_incomplete', 'query_incomplete', 'peer_insufficient')
    return {group: {key: int(np.count_nonzero(out[key][:, :, sl])) for key in fields}
            for group, sl in GROUPS.items()}


def run():
    assert (U / 'RESULT.json').exists() and not (U / 'RUNNING.json').exists()
    assert not (U / 'VERIFY.json').exists(), 'Completed verification must not be rerun'
    frozen, result = read(U / 'FROZEN.json'), read(U / 'RESULT.json')
    check(frozen['pins']); check(frozen['protected'])
    fixture = read(U / 'FIXTURE_RESULT.json')
    assert fixture['status'] == 'PASS' and len(fixture['checks']) == 9 and all(fixture['checks'].values())
    for name, digest in fixture['sha256'].items():
        assert sha(U / name) == digest, name
    assert [len(result[k]) for k in ('known', 'cross', 'missing')] == [297, 2376, 135]
    assert result['training'] == result['user_data_access'] == result['new_dtw'] == 0
    original_results = read(P / 'RESULT.json')
    meta = {r['clip_id']: r for r in frozen['rows']}
    assert len(meta) == 27 and len({r['word_code'] for r in meta.values()}) == 3
    source = {cid: load(I / (cid + '.npz')) for cid in meta}
    ch = {cid: channels(e) for cid, e in source.items()}
    roles = {cid: frozen['roles'].get(r['word_code'], {}).get('required_sides', ['L', 'R']) for cid, r in meta.items()}
    domains = {cid: expected(e, roles[cid]) for cid, e in source.items()}
    anchors, peers = {}, {}
    for rid, e in source.items():
        ids = np.arange(len(e['core']))
        anchors[rid] = describe(ch[rid], np.column_stack([ids, ids]), e['source_rest'], domains[rid])
        for cid in meta:
            if cid == rid or meta[cid]['word_code'] != meta[rid]['word_code']:
                continue
            path = load(P / 'cross' / ('identity__' + cid + '__' + rid + '.npz'))['base_path']
            peers[cid, rid] = describe(ch[cid], path, source[cid]['source_rest'], domains[rid])
    calibrations, hashes = {}, {}
    for cid in meta:
        for rid in meta:
            if meta[cid]['word_code'] != meta[rid]['word_code']:
                continue
            peer_ids = [s for s in meta if s not in (cid, rid) and meta[s]['word_code'] == meta[cid]['word_code']]
            assert len(peer_ids) == (8 if cid == rid else 7)
            assert len({meta[s]['signer_id'] for s in peer_ids}) == len(peer_ids)
            assert meta[cid]['signer_id'] not in {meta[s]['signer_id'] for s in peer_ids}
            cal = calibration(anchors[rid], [peers[s, rid] for s in peer_ids])
            file = U / 'calibration' / (cid + '__' + rid + '.npz')
            saved = load(file)
            assert set(saved) == set(cal) | {'peer_ids', 'query', 'anchor'}
            for key in cal:
                eq(cal[key], saved[key])
            eq(np.array(peer_ids), saved['peer_ids']); eq(np.array(cid), saved['query']); eq(np.array(rid), saved['anchor'])
            calibrations[cid, rid] = cal; hashes[str(file)] = sha(file)
    assert len(calibrations) == 243 and len(list((U / 'calibration').glob('*.npz'))) == 243
    checked = Counter(); extra = Counter()
    for folder in ('known', 'cross', 'missing'):
        old = {r['name']: r for r in original_results[folder]}
        assert {r['name'] for r in result[folder]} == set(old)
        assert len(list((U / folder).glob('*.npz'))) == len(old)
        for row in result[folder]:
            before = old[row['name']]
            for key in ('name', 'query', 'reference', 'kind', 'label', 'side', 'removed'):
                if key in before:
                    assert row[key] == before[key], (folder, row['name'], key)
            cid, rid = row['query'], row['reference']
            saved_old = load(P / folder / (row['name'] + '.npz'))
            if folder == 'missing':
                qe = make_damage(source[cid], row['side'], row['kind'], np.asarray(row['removed'], int))
                cc = channels(qe); rest = qe['source_rest']; index = np.arange(len(rest))
                native_expected = expected(qe, roles[cid])
            else:
                index = saved_old['source_index']
                cc = interpolate(ch[cid], index)
                left, right = np.floor(index).astype(int), np.ceil(index).astype(int)
                rest = source[cid]['source_rest'][left] & source[cid]['source_rest'][right]
                native_expected = np.repeat(saved_old['query_expected'][:, :, None], 21, axis=2)
            native_unknown = native_expected & ~cc['valid']
            old_unknown = np.concatenate([saved_old['query_angle_unknown'],
                np.repeat(saved_old['query_palm_unknown'][:, :, None], 2, axis=2),
                np.repeat(saved_old['query_arm_unknown'], 2, axis=2)], axis=2)
            eq(native_unknown, old_unknown)
            path = saved_old['base_path']
            qd = describe(cc, path, rest, domains[rid])
            output = compare(qd, anchors[rid], calibrations[cid, rid], domains[rid])
            eq(output['expected'], saved_old['expected'])
            assert not (output['levels_within'] & output['unknown']).any()
            assert not (output['sequence_within'] & output['unknown']).any()
            assert not (output['levels_bad'] & output['unknown']).any()
            assert not (output['sequence_bad'] & output['unknown']).any()
            arrays = dict(output, native_expected=native_expected, native_unknown=native_unknown,
                path=path, source_index=index, query_phase_index=qd['phase_index'],
                query_span=qd['span'], query_collapsed=qd['collapsed'],
                point_comparable=saved_old['old_comparable'], point_bad=saved_old['old_mismatch'])
            file = U / folder / (row['name'] + '.npz')
            saved = load(file)
            assert set(saved) == set(arrays), (set(saved) - set(arrays), set(arrays) - set(saved))
            for key, array in arrays.items():
                eq(array, saved[key])
            assert row['groups'] == counts(output)
            assert row['whole_action_correct'] is None
            extra['collapsed_reference_frames'] += int(qd['collapsed'].sum())
            extra['old_comparable_now_sequence_unknown'] += int((arrays['point_comparable'] & output['unknown']).sum())
            checked[folder] += 1; hashes[str(file)] = sha(file)
            if sum(checked.values()) % 270 == 0:
                print('VERIFIED', dict(checked), flush=True)
    assert dict(checked) == {'known': 297, 'cross': 2376, 'missing': 135}
    crossing = source['NIA_SL_WORD0185_REAL03_F']
    at = np.isin(crossing['frame_id'], list(range(101, 112)) + [113])
    assert int(at.sum()) == 12
    cross_ch = ch['NIA_SL_WORD0185_REAL03_F']['valid'][at]
    assert int(cross_ch[:, :, :15].sum()) == 0
    assert int(cross_ch[:, :, 15:17].sum()) == 0
    assert int(cross_ch[:, :, 17:].sum()) == 96
    check(frozen['pins']); check(frozen['protected'])
    for name in ('candidate.py', 'study.py', 'verify.py', 'fixtures.py', 'FIXTURE_RESULT.json', 'FROZEN.json', 'RESULT.json', 'DESIGN_REVIEW.json'):
        hashes[str(U / name)] = sha(U / name)
    audit = dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(), status='PASS',
        counts=dict(checked), calibrations=243, peer_counts={'cross': 7, 'self': 8, 'minimum_complete': 5},
        checks={'independent_channels_and_adjacent_endpoint_interpolation': True,
            'old_saved_paths_unchanged_no_new_DP': True,
            'full_window_and_mapped_span_observation': True,
            'positive_rest_exclusion_separate_from_missing': True,
            'each_angle_independent': True,
            'ordered_three_levels_two_interval_increments': True,
            'median_mapping_phase_indices_spans_and_collapse': True,
            'query_excluded_from_calibration': True,
            'levels_sequence_separate_max_thresholds': True,
            'expected_and_native_unknown_unchanged': True,
            'previous_point_comparable_bad_unchanged': True,
            'unknown_never_within_or_bad': True,
            'known_cross_missing_saved_arrays_and_group_counts': True,
            'reviewed_crossing_retains_angle_palm_unknown_and_arm_observations': True,
            'frozen_inputs_and_protected_model_runtime_hashes': True},
        descriptive_checks=dict(extra),
        crossing={'frames': 12, 'angle_observed_cells': 0, 'palm_xy_observed_cells': 0,
                  'arm_xy_observed_cells': 96, 'arm_points': 48,
                  'interpretation': 'Original prepared evidence retained; not correct-handshape or semantic-accuracy proof.'},
        protected_files_checked=len(frozen['protected']), pinned_files_checked=len(frozen['pins']), sha256=hashes,
        implementation='Separate Codex agent; only standard library and numpy; no production candidate/study/older executable imports.',
        limits=['Computational contract verification; no semantic correctness labels or general-user accuracy.',
            'Max normal peer range is descriptive, not calibrated statistical confidence.',
            'Complete-window requirement can reduce support; less discrepancy is not automatically improvement.',
            'No inference, training, new DTW or new detector was run.'])
    with (U / 'VERIFY.json').open('x', encoding='utf8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print('VERIFY PASS', dict(checked), 'calibrations', 243, flush=True)


if __name__ == '__main__':
    try:
        run()
    except BaseException:
        error = U / 'VERIFY_FIRST_ERROR.txt'
        if not error.exists():
            error.write_text(traceback.format_exc(), encoding='utf8')
        raise
