"""Prepare pinned palm58 inputs only. Root reviews/runs; no model inference.

Primary targets remain the old full/hold direction labels. Direct partial
annotations supervise progression only. Missing measurements keep their row
and denominator but never become hold or incorrect-sign targets.
"""
from core_palm import *
from collections import Counter


def main():
    UNIT.mkdir(parents=True, exist_ok=True)
    for name in ('PREPARATION_CONTRACT.json', 'INPUTS_COMPLETE.json', 'inputs.npz'):
        assert not (UNIT / name).exists(), f'Existing artifact preserved: {name}'
    stop_if_requested()
    helper_path = D / 'palm_projection_core.py'
    assert sha(helper_path) == MEASUREMENT_SHA256
    measure = old.load('_palm_direction_measurement_fixed', helper_path)
    primary_source = read(HERE / 'INPUTS_COMPLETE.json')['rows']
    primary = [dict(r, old_row_index=i) for i, r in enumerate(primary_source) if r['axis'] == 2]
    assert len(primary) == 720
    assert sum(r['split'] == 'train' for r in primary) == 528
    assert sum(r['split'] == 'probe' for r in primary) == 192
    for r in primary:
        assert r['person'] in (range(1, 8) if r['split'] == 'train' else (8, 9))
        assert r['split'] in ('train', 'probe') and r['target'] in (0, 1, 2)
        assert r['moving'] != r['anchor'] and r['axis'] == 2
    fit_hashes = {r['source_sha256'] for r in primary if r['split'] == 'train'}
    probe_hashes = {r['source_sha256'] for r in primary if r['split'] == 'probe'}
    assert not fit_hashes & probe_hashes
    audited = read(D / 'RESULTS.json')['rows']
    assert len(audited) == 1757
    assert Counter(r['label_kind'] for r in audited) == dict(full=414, partial=1082, hold=261)
    partial_source = [r for r in audited if r['label_kind'] == 'partial']
    n_rows = read(N / 'TRAINING_INPUT_INDEX.json')['rows']
    n_masks = read(N / 'local_bag_evaluation/RULES_FROZEN.json')['partial_masks']
    ready_path = OPS / 'word100_shared_motion_learning_20261109/READY.json'
    crop_paths = [old.R / 'CROP_INPUTS_COMPLETE.json', old.D / 'CROP_INPUTS_COMPLETE.json']
    guard_path = OPS / 'word100_observation_guard_20261108/guard.py'
    adapter_path = OPS / 'word100_rtmw_shared_input_20261108/adapter.py'
    pinned = [Path(__file__), UNIT / 'core_palm.py', HERE / 'common.py',
              HERE / 'INPUTS_COMPLETE.json', helper_path, D / 'RESULTS.json',
              D / 'FIT_DIRECTION_TARGET_AUDIT.json', D / 'POINT_MEASURE_AUDIT.json',
              D / 'PALM_CORE_VERIFICATION.json', N / 'TRAINING_INPUT_INDEX.json',
              N / 'local_bag_evaluation/RULES_FROZEN.json', ready_path,
              guard_path, adapter_path, PRIOR / 'boundary_direction_core.py',
              OPS / 'word100_motion_components_20261109/motion.py'] + crop_paths
    pins = {str(p): sha(p) for p in pinned}
    cfg = dict(at=stamp(), schema=SCHEMA, seeds=[42, 43, 44], steps=800,
        architecture_conditions=1, selection='Final step only; no probe tuning or further architecture search.',
        primary='Same 528 FIT01..07 and 192 probe08..09 axis2 full/hold targets; no labels, spans, views or reverse flags changed.',
        partial='All1082 fixed FIT02..07 direct-partial candidates from the1757-row audit. Only progression marginal target, never inherited direction or completion.',
        static='For every directional primary and every partial: samples0,4,8 of the measured9-point trajectory repeated as constant geometry. Hold label means measured geometry has no change, not educational error or real human stillness.',
        observation='Original eligibility retained; unavailable row and static descendants remain in denominator, masked from every loss and reported separately.',
        normalization='Mean/std from eligible primary FIT rows only. Std floor1e-4; raw delta uses FIT std[40], never its mean. Probe/partial/static excluded from normalization fit.',
        initialization='Copy original observer motion_conv, motion_gru and motion_head; no common100 context. Hold score=oldholdlogit-average(olddirectionlogits), symmetrized after separate raw reversal normalization.',
        training_proposal=dict(optimizer='AdamW', lr=.0003, weight_decay=.001, gradient_clip=2.,
            primary_batch=48, partial_batch=24, static_batch=24,
            primary_pool='Balance classes0/1/2 (16each), then original_motion_replay vs other origins where present, then clip, then candidate uniformly.',
            partial_pool='Clip uniform, then (view,reverse) uniform, then eligible candidate uniform; no score-based selection.',
            static_pool='Primary vs partial sources equally (12each), then clip uniform, then eligible source candidate, anchor0/4/8 uniform.',
            loss='CE_primary + marginal_partial + CE_static, each as its own mean with fixed equal weights. Partial logsumexp(direction logits) is detached; no pseudo-direction label.'),
        gate='Saved old comparison plus new isolated candidate: normal/speed/crossing support preserved; reversed/compound/handshape-only examined; unavailable separate. G2 aggregate required, not this input contract. Existing aligned_fist is not palm-invariant and cannot certify shape-only separation.',
        limitations=['2D net projection sign, not contact or fullpath correctness.',
            'Opposite forearm rotation and detector mistakes remain possible.',
            'Available masks do not independently certify visible anatomy.',
            'Symmetric scores guarantee reversal equivariance, not globally monotone class probability.',
            'Static controls live in measured geometry space; middle sample can be interpolated.'],
        private_access=False, final8=False, web_changed=False, model_inference=False,
        training_run=False, pins=pins)
    write(UNIT / 'PREPARATION_CONTRACT.json', cfg)
    ready = {r['clip_id']: r for r in read(ready_path)['rows'] if r['role'] == 'fit' and r['person'] in range(1, 10)}
    crops = {r['clip_id']: r for p in crop_paths for r in read(p)['rows'] if r['clip_id'] in ready}
    sys.path.insert(0, str(guard_path.parent))
    guard = old.load('_palm_direction_guard', guard_path)
    adapter = old.load('_palm_direction_adapter', adapter_path)
    cache = {}
    source_pins = {}

    def pin_file(path, digest):
        assert sha(path) == digest, path
        source_pins[str(path)] = digest

    def raw_view(r):
        key = r['clip_id'], r['view']
        if key not in cache:
            sr = ready[r['clip_id']]
            assert sr['person'] == r['person'] and sr['source_sha256'] == r['source_sha256']
            pin_file(sr['path'], sr['sha256'])
            pin_file(sr['channel'], sr['channel_sha256'])
            with np.load(sr['path'], allow_pickle=False) as z, np.load(sr['channel'], allow_pickle=False) as ch:
                ts = z['timestamp_us'].copy()
                assert np.array_equal(ts, ch['timestamp_us'])
                if r['view'] == 'actual_crop':
                    cr = crops[r['clip_id']]
                    pin_file(cr['path'], cr['sha256'])
                    with np.load(cr['path'], allow_pickle=False) as cz:
                        assert np.array_equal(ts, cz['timestamp_us'])
                        cleaned = guard.quarantine(cz['xy_px'], cz['score'], ts, 640, 480)
                        x = adapter.convert(cleaned['xy_px'], cz['score'], ts, cz['frame_id'], 640, 480)['tcn280']
                elif r['view'] == 'full':
                    x = ch['full'].copy()
                else:
                    assert r['view'] == 'upper'
                    x = ch['feedback'].copy() if 'feedback' in ch.files else ch['upper_mask'].copy()
            assert len(x) == len(ts) and x.shape[1] == 280
            assert np.isfinite(x).all() and (np.diff(ts) > 0).all()
            cache[key] = x, ts
        return cache[key]

    primary_g = []
    primary_rows = []
    for index, r in enumerate(primary):
        stop_if_requested()
        x, ts = raw_view(r)
        g, eligible, info = measure.geometry(x, ts, r['a'], r['b'], 2, r['moving'], r['anchor'], r['reverse'])
        primary_g.append(g)
        primary_rows.append(dict(r, index=index, eligible=bool(eligible), palm_geometry=info,
            label_basis='unchanged_existing_full_or_reviewed_hold', reversal_scope='interval'))

    partial_g = []
    partial_rows = []
    bag_cache = {}
    for source_index, r in enumerate(partial_source):
        stop_if_requested()
        bi, ci = r['bag_index'], r['candidate_index']
        nr, mask = n_rows[bi], n_masks[bi]
        assert nr['split'] == 'train' and nr['person'] in range(2, 8) and nr['family'] == 4
        assert nr['origin'] == 'positive_source_extent' and nr['source_sha256'] in fit_hashes
        assert mask['bag_index'] == bi and mask['clip_id'] == r['clip_id'] == nr['clip_id']
        for key in ('person', 'view', 'reverse'):
            assert r[key] == nr[key]
        pm = next(p for p in mask['candidates'] if p['candidate_index'] == ci)
        assert pm['groups'] == r['groups'] and pm['original_span'] == r['original_span']
        if bi not in bag_cache:
            pin_file(nr['path'], nr['sha256'])
            with np.load(nr['path'], allow_pickle=False) as bag:
                bag_cache[bi] = bag['spans'].copy(), bag['y'].copy()
        spans, full = bag_cache[bi]
        assert list(map(int, spans[ci])) == r['span'] and not bool(full[ci])
        x, ts = raw_view(nr)
        if nr['reverse']:
            x = x[::-1].copy()
            ts = ts[-1] - ts[::-1]
            x[:, 279] = np.r_[0., np.diff(ts) / 1e6]
        a, b = r['span']
        witnessed = sorted({f for group in r['groups'] for f in group['witnessed_frames']})
        if nr['reverse']:
            witnessed = sorted(len(x) - 1 - f for f in witnessed)
        assert len(witnessed) >= 2 and a <= witnessed[0] < witnessed[-1] <= b
        g, eligible, info = measure.geometry(x, ts, a, b, 2, 'R', 'L', False)
        partial_g.append(g)
        # Preserve audit metadata, but no occurrence-derived direction in labels.
        partial_rows.append(dict(index=source_index, bag_index=bi, candidate_index=ci,
            clip_id=r['clip_id'], person=r['person'], split='train', view=r['view'], reverse=r['reverse'],
            source_sha256=nr['source_sha256'], span=r['span'], original_span=r['original_span'],
            groups=r['groups'], witnessed_frames=witnessed, target_progression=1,
            direction_target=None, completion_target=None, eligible=bool(eligible), palm_geometry=info,
            label_basis='at_least_two_direct_witnesses_within_same_occurrence', reversal_scope='whole_source'))
    assert len(partial_rows) == 1082

    primary_g = np.asarray(primary_g, np.float32)
    partial_g = np.asarray(partial_g, np.float32)
    primary_ok = np.asarray([r['eligible'] for r in primary_rows], bool)
    partial_ok = np.asarray([r['eligible'] for r in partial_rows], bool)
    fit_index = np.asarray([i for i, r in enumerate(primary_rows) if r['split'] == 'train' and r['eligible']], np.int64)
    assert len(fit_index) and all(primary_rows[i]['person'] in range(1, 8) for i in fit_index)
    mean = primary_g[fit_index].mean(0)
    std = np.maximum(primary_g[fit_index].std(0), 1e-4)
    delta_scale = float(std[40])
    assert delta_scale > 1e-4
    static_g, static_rows = [], []
    for collection, rows, data in (('primary', primary_rows, primary_g), ('partial', partial_rows, partial_g)):
        for i, r in enumerate(rows):
            if collection == 'primary' and r['target'] == 1:
                continue
            for anchor in (0, 4, 8):
                stop_if_requested()
                g = static_geometry(data[i], anchor) if r['eligible'] else np.zeros(58, np.float32)
                assert not g[:36].any() and not g[40:52].any()
                static_g.append(g)
                static_rows.append(dict(index=len(static_rows), source_collection=collection, source_index=i,
                    clip_id=r['clip_id'], person=r['person'], split=r['split'], view=r['view'], reverse=r['reverse'],
                    source_sha256=r['source_sha256'], anchor_sample=anchor,
                    eligible=r['eligible'], target=1, label_basis='mathematically_constant_measured_geometry',
                    human_or_educational_error_label=False, source_files_unchanged=True))
    static_g = np.asarray(static_g, np.float32)
    arrays = dict(primary_geometry=primary_g, primary_y=np.asarray([r['target'] for r in primary_rows], np.int64),
        primary_eligible=primary_ok, partial_geometry=partial_g, partial_progression=np.ones(len(partial_rows), np.int64),
        partial_eligible=partial_ok, static_geometry=static_g, static_y=np.ones(len(static_rows), np.int64),
        static_eligible=np.asarray([r['eligible'] for r in static_rows], bool),
        fit_normalization_indices=fit_index, mean=mean, std=std, delta_scale=np.asarray(delta_scale, np.float32))
    assert all(np.isfinite(a).all() for a in arrays.values())
    # Pure arithmetic guards only; no head construction or inference here.
    for g in np.concatenate([primary_g[primary_ok], partial_g[partial_ok]]):
        gr = reverse_geometry_numpy(g)
        assert np.allclose(reverse_geometry_numpy(gr), g, atol=1e-6)
        assert gr[40] == -g[40] and np.array_equal(gr[44:], g[44:])
    for path, digest in {**pins, **source_pins}.items():
        assert sha(path) == digest, path
    stop_if_requested()
    destination = UNIT / 'inputs.npz'
    np.savez_compressed(destination, **arrays)
    summaries = {}
    for name, rows in [('primary', primary_rows), ('partial', partial_rows), ('static', static_rows)]:
        summaries[name] = {split: dict(total=sum(r['split'] == split for r in rows),
            eligible=sum(r['split'] == split and r['eligible'] for r in rows),
            unavailable=sum(r['split'] == split and not r['eligible'] for r in rows)) for split in ('train', 'probe')}
    all_pins = {**pins, **source_pins, str(UNIT / 'PREPARATION_CONTRACT.json'): sha(UNIT / 'PREPARATION_CONTRACT.json')}
    write(UNIT / 'INPUTS_COMPLETE.json', dict(at=stamp(), schema=SCHEMA, path=str(destination), sha256=sha(destination),
        primary_rows=primary_rows, partial_rows=partial_rows, static_rows=static_rows, summary=summaries,
        norm=dict(mean=mean.tolist(), std=std.tolist(), delta_scale=delta_scale, fit_indices=fit_index.tolist(),
            sources='eligible primary FIT01..07 only', partial_or_probe_used=False),
        pins=all_pins, model_inference=False, training_run=False, private_access=False, final8=False, web_changed=False,
        limitations=cfg['limitations']))
    print(__import__('json').dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        preserve_failure('PREPARATION')
        raise
