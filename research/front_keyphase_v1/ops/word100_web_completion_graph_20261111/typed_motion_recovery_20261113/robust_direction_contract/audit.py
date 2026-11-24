"""Frozen FIT-only math audit; no trained model is constructed or invoked."""
from pathlib import Path
import sys
from collections import Counter
U = Path(__file__).resolve().parent
sys.path.insert(0, str(U.parent))
from common import HERE, GRAPH, OPS, PRIOR, old, np, read, write, sha, stamp
from trend import summarize, reverse_samples

P = HERE / 'palm_direction_learning'
D = HERE / 'partial_direction_contract'
N = HERE / 'null_candidate_learning'


def nums(v):
    a = np.asarray(v, float)
    if not a.size:
        return dict(n=0)
    return dict(n=int(a.size), minimum=float(a.min()), median=float(np.median(a)),
        q90=float(np.quantile(a, .9)), maximum=float(a.max()))


def stop():
    for folder in (U, HERE, GRAPH):
        for name in ('STOP_REQUESTED', 'STOP_REQUESTED.json'):
            assert not (folder / name).exists(), f'Pause: {folder / name}'


def main():
    assert (U / 'PLAN_FROZEN.json').exists()
    assert not (U / 'INPUT_PINS.json').exists() and not (U / 'RESULTS.json').exists()
    stop()
    meta = read(P / 'INPUTS_COMPLETE.json')
    prior_audit = read(D / 'RESULTS.json')
    assert len(prior_audit['rows']) == 1757
    helper_path = D / 'palm_projection_core.py'
    assert sha(helper_path) == 'ddcaee0652846059b6d2f9c83371431dc304ee282aea4ba158ae0b1de76b835a'
    helper = old.load('_robust_fit_palm_measurement', helper_path)
    ready_path = OPS / 'word100_shared_motion_learning_20261109/READY.json'
    crop_paths = [old.R / 'CROP_INPUTS_COMPLETE.json', old.D / 'CROP_INPUTS_COMPLETE.json']
    guard_path = OPS / 'word100_observation_guard_20261108/guard.py'
    adapter_path = OPS / 'word100_rtmw_shared_input_20261108/adapter.py'
    paths = [Path(__file__), U / 'trend.py', U / 'PLAN_FROZEN.json', P / 'INPUTS_COMPLETE.json',
        Path(meta['path']), helper_path, D / 'RESULTS.json', N / 'TRAINING_INPUT_INDEX.json',
        N / 'local_bag_evaluation/RULES_FROZEN.json', ready_path, guard_path, adapter_path] + crop_paths
    pins = {str(p): sha(p) for p in paths}
    assert pins[meta['path']] == meta['sha256']
    write(U / 'INPUT_PINS.json', dict(at=stamp(), pins=pins,
        fit_only='Only528 primary train and1082 partial train rows retained from existing packed inputs. No probe values included in measurements/statistics or controls.',
        no_model=True, no_DEV=True, no_private=True))
    pidx = [i for i, r in enumerate(meta['primary_rows']) if r['split'] == 'train']
    assert len(pidx) == 528 and len(meta['partial_rows']) == 1082
    rows, geoms = [], []
    with np.load(meta['path'], allow_pickle=False) as z:
        pg = z['primary_geometry'][pidx]
        part_g = z['partial_geometry'].copy()
        scale = float(z['delta_scale'])
    for i, g in zip(pidx, pg):
        r = meta['primary_rows'][i]
        assert r['person'] in range(1, 8)
        rows.append(dict(cohort='primary', source_index=i, clip_id=r['clip_id'], person=r['person'],
            split='train', view=r['view'], reverse=r['reverse'], span=[r['a'], r['b']], target=r['target'],
            eligible=r['eligible'], source_sha256=r['source_sha256'], direction_gold=r['target'] if r['target'] != 1 else None))
        geoms.append(g)
    for i, (r, g) in enumerate(zip(meta['partial_rows'], part_g)):
        assert r['split'] == 'train' and r['person'] in range(2, 8) and r['direction_target'] is None
        rows.append(dict(cohort='partial', source_index=i, clip_id=r['clip_id'], person=r['person'], split='train',
            view=r['view'], reverse=r['reverse'], span=r['span'], target=None, eligible=r['eligible'],
            source_sha256=r['source_sha256'], direction_gold=None, progression_only=True))
        geoms.append(g)
    nrows = read(N / 'TRAINING_INPUT_INDEX.json')['rows']
    holds = [r for r in prior_audit['rows'] if r['label_kind'] == 'hold']
    assert len(holds) == 261
    ready = {r['clip_id']: r for r in read(ready_path)['rows'] if r['role'] == 'fit' and r['person'] in range(2, 8)}
    crops = {r['clip_id']: r for p in crop_paths for r in read(p)['rows'] if r['clip_id'] in ready}
    sys.path.insert(0, str(guard_path.parent))
    guard = old.load('_robust_fit_guard', guard_path)
    adapter = old.load('_robust_fit_adapter', adapter_path)
    source_pins = {}
    cache = {}
    for r in holds:
        stop()
        nr = nrows[r['bag_index']]
        assert nr['split'] == 'train' and nr['person'] in range(2, 8) and not nr['reverse']
        assert nr['origin'] == 'reviewed_local_hold' and nr['source_span'] == r['source_span']
        key = nr['clip_id'], nr['view']
        if key not in cache:
            sr = ready[nr['clip_id']]
            assert sr['source_sha256'] == nr['source_sha256'] and sr['person'] == nr['person']
            for path, digest in [(sr['path'], sr['sha256']), (sr['channel'], sr['channel_sha256'])]:
                assert sha(path) == digest
                source_pins[path] = digest
            with np.load(sr['path'], allow_pickle=False) as z, np.load(sr['channel'], allow_pickle=False) as ch:
                ts = z['timestamp_us'].copy()
                if nr['view'] == 'actual_crop':
                    cr = crops[nr['clip_id']]
                    assert sha(cr['path']) == cr['sha256']
                    source_pins[cr['path']] = cr['sha256']
                    with np.load(cr['path'], allow_pickle=False) as cz:
                        assert np.array_equal(ts, cz['timestamp_us'])
                        cleaned = guard.quarantine(cz['xy_px'], cz['score'], ts, 640, 480)
                        x = adapter.convert(cleaned['xy_px'], cz['score'], ts, cz['frame_id'], 640, 480)['tcn280']
                else:
                    assert nr['view'] in ('full', 'upper')
                    x = ch['full'].copy() if nr['view'] == 'full' else (ch['feedback'].copy() if 'feedback' in ch.files else ch['upper_mask'].copy())
            cache[key] = x, ts
        x, ts = cache[key]
        aa, bb = nr['source_span']
        x = x[aa:bb+1].copy()
        ts = ts[aa:bb+1].copy()
        x[0, 279] = 0
        a, b = r['span']
        g, ok, info = helper.geometry(x, ts, a, b, 2)
        rows.append(dict(cohort='reviewed_hold', bag_index=r['bag_index'], candidate_index=r['candidate_index'],
            clip_id=r['clip_id'], person=r['person'], split='train', view=r['view'], reverse=False,
            span=r['span'], source_span=r['source_span'], target=1, direction_gold=None,
            eligible=bool(ok), source_sha256=nr['source_sha256'], geometry_coverage=info['coverage']))
        geoms.append(g)
    assert len(rows) == 1871 and all(r['person'] in range(1, 8) for r in rows)
    geometry = np.asarray(geoms, np.float32)
    eligible = np.asarray([r['eligible'] for r in rows], bool)
    values = geometry[:, :36].reshape(-1, 9, 4)[:, :, 0].astype(float) + geometry[:, 36, None]
    baseline = summarize(values)
    reversed_stats = summarize(reverse_samples(values))
    max_reverse_error = max(float(abs(baseline['trend']+reversed_stats['trend']).max()), float(abs(baseline['net']+reversed_stats['net']).max()))
    assert max_reverse_error < 1e-10
    assert np.allclose(baseline['positive_path'], reversed_stats['negative_path'], atol=1e-10)
    noise_controls = [(position, multiplier) for position in range(9) for multiplier in [-1., -.5, -.25, .25, .5, 1.]]
    noise_values = np.repeat(values[:, None, :], len(noise_controls), axis=1)
    for j, (position, multiplier) in enumerate(noise_controls):
        noise_values[:, j, position] += multiplier * scale
    noise = summarize(noise_values)
    block_controls = [(start, length) for length in (2, 3, 4) for start in range(10-length)]
    assert len(block_controls) == 21
    block_values = np.repeat(values[:, None, :], len(block_controls), axis=1)
    for j, (start, length) in enumerate(block_controls):
        block_values[:, j, start:start+length] = values[:, start:start+length][:, ::-1]
    block = summarize(block_values)
    cohorts = [('primary_direction', np.asarray([r['cohort'] == 'primary' and r['target'] != 1 for r in rows])),
        ('primary_hold', np.asarray([r['cohort'] == 'primary' and r['target'] == 1 for r in rows])),
        ('partial', np.asarray([r['cohort'] == 'partial' for r in rows])),
        ('reviewed_hold', np.asarray([r['cohort'] == 'reviewed_hold' for r in rows]))]
    sign_gold = np.asarray([0 if r['direction_gold'] is None else r['direction_gold']-1 for r in rows])
    baseline_summary = []
    noise_summary = []
    block_summary = []
    for name, cohort in cohorts:
        mask = cohort & eligible
        summary = dict(cohort=name, total=int(cohort.sum()), eligible=int(mask.sum()), unavailable=int((cohort & ~eligible).sum()),
            net=nums(baseline['net'][mask]), trend=nums(baseline['trend'][mask]),
            absolute_net=nums(abs(baseline['net'][mask])), absolute_trend=nums(abs(baseline['trend'][mask])),
            net_trend_sign_disagreement=int((np.sign(baseline['net'][mask]) != np.sign(baseline['trend'][mask])).sum()))
        if name == 'primary_direction':
            summary.update(net_label_sign_compatible=int((baseline['net'][mask]*sign_gold[mask] > 0).sum()),
                trend_label_sign_compatible=int((baseline['trend'][mask]*sign_gold[mask] > 0).sum()))
        baseline_summary.append(summary)
        for j, (position, multiplier) in enumerate(noise_controls):
            out = dict(cohort=name, sample_index=position, multiplier=multiplier, eligible=int(mask.sum()),
                net_absolute_change=nums(abs(noise['net'][mask, j]-baseline['net'][mask])),
                trend_absolute_change=nums(abs(noise['trend'][mask, j]-baseline['trend'][mask])),
                net_sign_changed=int((np.sign(noise['net'][mask, j]) != np.sign(baseline['net'][mask])).sum()),
                trend_sign_changed=int((np.sign(noise['trend'][mask, j]) != np.sign(baseline['trend'][mask])).sum()))
            if name == 'primary_direction':
                out.update(net_label_sign_compatible=int((noise['net'][mask, j]*sign_gold[mask] > 0).sum()),
                    trend_label_sign_compatible=int((noise['trend'][mask, j]*sign_gold[mask] > 0).sum()))
            noise_summary.append(out)
        for j, (start, length) in enumerate(block_controls):
            # Only primary directional rows have direction gold. The other
            # cohorts below use the baseline trend merely as a reference axis.
            reference = sign_gold if name == 'primary_direction' else np.sign(baseline['trend'])
            prior_diff = np.diff(values[:, start:start+length], axis=1) * reference[:, None]
            new_diff = np.diff(block_values[:, j, start:start+length], axis=1) * reference[:, None]
            previously_forward = (prior_diff > 0).all(1) & (reference != 0)
            introduced = previously_forward & (new_diff < 0).any(1) & mask
            out = dict(cohort=name, start_sample=start, block_length=length, internal_block=bool(start > 0 and start+length < 9),
                eligible=int(mask.sum()), strictly_forward_block_reversed=int(introduced.sum()),
                reversal_visible_but_trend_keeps_reference=int((introduced & (block['trend'][:, j]*reference > 0)).sum()),
                reversal_visible_but_net_keeps_reference=int((introduced & (block['net'][:, j]*reference > 0)).sum()),
                introduced_opposing_step_magnitude=nums(np.maximum(-new_diff[introduced], 0).sum(1)),
                educational_wrong_direction_label=False)
            block_summary.append(out)
    for i, r in enumerate(rows):
        r['values'] = values[i].tolist()
        r['stats'] = {k: float(v[i]) if np.issubdtype(v.dtype, np.floating) else int(v[i]) for k, v in baseline.items()}
    matrix_path = U / 'CONTROL_MATRICES.npz'
    assert not matrix_path.exists()
    matrix = dict(eligible=eligible, values=values, fit_scale=np.asarray(scale),
        noise_controls=np.asarray(noise_controls), block_controls=np.asarray(block_controls))
    matrix.update({'noise_'+k: v for k, v in noise.items()})
    matrix.update({'block_'+k: v for k, v in block.items()})
    np.savez_compressed(matrix_path, **matrix)
    for path, digest in {**pins, **source_pins}.items():
        assert sha(path) == digest, path
    write(U / 'RESULTS.json', dict(at=stamp(), rows=rows, baseline_summary=baseline_summary,
        noise_summary=noise_summary, block_summary=block_summary,
        matrix_path=str(matrix_path), matrix_sha256=sha(matrix_path), pins=pins, source_pins=source_pins,
        maximum_time_reverse_error=max_reverse_error, fit_scale=scale, total_rows=len(rows),
        total_noise_controls=int(eligible.sum())*54, total_block_controls=int(eligible.sum())*21,
        no_model_inference=True, no_training=True, no_DEV=True, no_new_labels=True,
        limitations='Statistics are2D observation summaries. Single-sample noise and block reversal are mathematical controls, not novice-error ground truth. Overlapping candidates/views are not independent clips.'))
    print(__import__('json').dumps(dict(rows=len(rows), baseline_summary=baseline_summary), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        first = U / 'FIRST_FAILURE.txt'
        if not first.exists():
            first.write_text(traceback.format_exc(), 'utf-8')
        raise
