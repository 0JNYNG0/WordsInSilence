"""Saved input arithmetic/provenance check; no core import or model inference."""
from pathlib import Path
import json
import hashlib
import datetime
import numpy as np

U = Path(__file__).resolve().parent
H = U.parent
D = H / 'partial_direction_contract'
N = H / 'null_candidate_learning'


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = U / 'INPUT_PREPARATION_REVIEW.json'
    assert not output.exists()
    meta = read(U / 'INPUTS_COMPLETE.json')
    assert sha(meta['path']) == meta['sha256']
    pins = dict(meta['pins'])
    pins.update({str(p): sha(p) for p in [Path(__file__), U / 'INPUTS_COMPLETE.json']})
    for path, digest in pins.items():
        assert sha(path) == digest, path
    with np.load(meta['path'], allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    assert all(np.isfinite(a).all() for a in arrays.values())
    old_rows = read(H / 'INPUTS_COMPLETE.json')['rows']
    expected_indices = [i for i, r in enumerate(old_rows) if r['axis'] == 2]
    primary = meta['primary_rows']
    assert len(primary) == len(expected_indices) == 720
    for i, (r, oi) in enumerate(zip(primary, expected_indices)):
        assert r['old_row_index'] == oi and r['index'] == i
        assert all(r[k] == value for k, value in old_rows[oi].items()), (i, oi)
        assert arrays['primary_y'][i] == r['target']
        assert arrays['primary_eligible'][i] == r['eligible']
        assert r['reversal_scope'] == 'interval'
    partial = meta['partial_rows']
    nr = read(N / 'TRAINING_INPUT_INDEX.json')['rows']
    masks = read(N / 'local_bag_evaluation/RULES_FROZEN.json')['partial_masks']
    expected_partial = [r for r in read(D / 'RESULTS.json')['rows'] if r['label_kind'] == 'partial']
    assert len(partial) == len(expected_partial) == 1082
    assert arrays['partial_progression'].tolist() == [1] * 1082
    for i, (r, expected) in enumerate(zip(partial, expected_partial)):
        assert r['index'] == i and r['direction_target'] is None and r['completion_target'] is None
        assert r['split'] == 'train' and r['person'] in range(2, 8)
        assert r['target_progression'] == 1 and 'target' not in r
        assert r['reversal_scope'] == 'whole_source'
        for key in ('clip_id', 'person', 'view', 'reverse', 'span', 'bag_index', 'candidate_index', 'groups', 'original_span'):
            assert r[key] == expected[key], (i, key)
        ni, ci = r['bag_index'], r['candidate_index']
        pm = next(m for m in masks[ni]['candidates'] if m['candidate_index'] == ci)
        assert pm['groups'] == r['groups'] and nr[ni]['source_sha256'] == r['source_sha256']
        assert len(r['witnessed_frames']) >= 2 and r['span'][0] <= min(r['witnessed_frames']) < max(r['witnessed_frames']) <= r['span'][1]
        assert bool(arrays['partial_eligible'][i]) == r['eligible']
    # Independent prior audit endpoint consistency, without rereading raw data.
    fit_audit = {r['index']: r for r in read(D / 'FIT_DIRECTION_TARGET_AUDIT.json')['rows']}
    errors = []
    for i, r in enumerate(primary):
        if r['split'] == 'train':
            audit = fit_audit[r['old_row_index']]
            assert audit['eligible'] == r['eligible'] and audit['target'] == r['target']
            if r['eligible']:
                errors.append(abs(float(arrays['primary_geometry'][i, 40]) - audit['delta']))
    point_audit = {(r['bag_index'], r['candidate_index']): r for r in read(D / 'POINT_MEASURE_AUDIT.json')['rows']}
    for i, r in enumerate(partial):
        audit = point_audit[r['bag_index'], r['candidate_index']]['palm']
        assert audit['eligible'] == r['eligible']
        if r['eligible']:
            errors.append(abs(float(arrays['partial_geometry'][i, 40]) - audit['projected_delta']))
    assert max(errors) < 1e-6
    fit_indices = np.asarray([i for i, r in enumerate(primary) if r['split'] == 'train' and r['eligible']], np.int64)
    assert np.array_equal(fit_indices, arrays['fit_normalization_indices']) and len(fit_indices) == 528
    fit = arrays['primary_geometry'][fit_indices]
    assert np.array_equal(arrays['mean'], fit.mean(0))
    assert np.array_equal(arrays['std'], np.maximum(fit.std(0), 1e-4))
    assert arrays['delta_scale'] == arrays['std'][40] > 0
    static = meta['static_rows']
    expected_static = [(name, i, anchor) for name, rows in [('primary', primary), ('partial', partial)]
                       for i, r in enumerate(rows) if name == 'partial' or r['target'] != 1 for anchor in (0, 4, 8)]
    assert len(static) == len(expected_static) == 5142
    for k, (r, expected) in enumerate(zip(static, expected_static)):
        name, index, anchor = expected
        assert (r['source_collection'], r['source_index'], r['anchor_sample']) == expected
        source = (primary if name == 'primary' else partial)[index]
        g = arrays[name + '_geometry'][index]
        sg = arrays['static_geometry'][k]
        for key in ('clip_id', 'person', 'split', 'view', 'reverse', 'eligible', 'source_sha256'):
            assert r[key] == source[key]
        assert r['target'] == arrays['static_y'][k] == 1 and not r['human_or_educational_error_label']
        assert bool(arrays['static_eligible'][k]) == source['eligible']
        assert not sg[:36].any() and not sg[40:52].any()
        if source['eligible']:
            assert np.array_equal(sg[36:40], g[36:40] + g[:36].reshape(9, 4)[anchor])
            assert np.array_equal(sg[52:], g[52:])
        else:
            assert not sg.any()
    for collection, rows in [('primary', primary), ('partial', partial), ('static', static)]:
        g, mask = arrays[collection + '_geometry'], arrays[collection + '_eligible']
        assert g.shape == (len(rows), 58) and not g[~mask].any()
        for split in ('train', 'probe'):
            rr = [r for r in rows if r['split'] == split]
            assert meta['summary'][collection][split] == dict(total=len(rr), eligible=sum(r['eligible'] for r in rr), unavailable=sum(not r['eligible'] for r in rr))
    assert not {r['source_sha256'] for r in primary if r['split'] == 'train'} & {r['source_sha256'] for r in primary if r['split'] == 'probe'}
    result = dict(at=datetime.datetime.now().astimezone().isoformat(), status='PASS',
        reviewer='Implementing Codex agent saved-arithmetic self-check; root performed separate source review. Not human/KSL truth certification.',
        primary_exact_old_rows=720, primary_fit=528, primary_probe=192,
        partial_exact_fixed_candidates=1082, partial_direction_or_completion_targets=0,
        static_exact_paired_controls=5142, maximum_prior_measurement_audit_error=max(errors),
        summary=meta['summary'], norm_fit_only=True, model_inference=False, training=False, source_reextraction=False,
        limits='2D measured palm projection; static geometry controls are not human error examples. Saved endpoint consistency does not verify all anatomical visibility or educational correctness.',
        pins=pins)
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print('Saved preparation arithmetic PASS: primary720, partial1082, static5142; no model inference', flush=True)


if __name__ == '__main__':
    main()
