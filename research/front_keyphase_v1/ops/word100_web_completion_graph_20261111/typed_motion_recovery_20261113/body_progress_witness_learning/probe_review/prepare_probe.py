"""Freeze sparse existing-PROBE labels without reading models or predictions."""
import collections
import time
import traceback

from probe_common import *


def main():
    stop()
    assert not (U / 'INPUTS_COMPLETE.json').exists()
    assert not (U / 'EVALUATION_CONTRACT.json').exists()
    review_path = U / 'RGB_REVIEW.json'
    review = read(review_path)
    assert review['status'] == 'QUALIFIED_SPARSE_BODY_PROGRESS_PROBE_RGB_REVIEW'
    assert review['training_allowed'] is False and review['original_annotations_preserved'] is True
    check(review['pins'])
    source_path = G / 'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json'
    source_doc = read(source_path)
    assert source_doc['training_allowed'] is False
    sources = {r['clip_id']: r for r in source_doc['rows'] if r['person'] in [8, 9]}
    reviewed = {r['clip_id']: r for r in review['rows']}
    assert len(sources) == len(reviewed) == 8 and set(sources) == set(reviewed)
    meta_path = R / 'INPUTS_COMPLETE.json'
    meta = read(meta_path)
    probes = [(i, r) for i, r in enumerate(meta['sequences']) if r['split'] == 'probe']
    assert len(meta['sequences']) == 192 and len(probes) == 48
    assert {r['person'] for i, r in probes} == {8, 9}
    assert {r['clip_id'] for i, r in probes} == set(sources)
    fit_meta = read(BP / 'INPUTS_COMPLETE.json')
    fits = fit_meta['sequences']
    assert len(fits) == 144 and {r['person'] for r in fits} == {2, 3, 4, 5, 6, 7}
    assert all(r['split'] == 'train' for r in fits)
    for key in ['clip_id', 'person', 'source_sha256']:
        assert not ({r[key] for r in fits} & {r[key] for i, r in probes}), key
    assert fit_meta['probe_metadata'] == [dict(original_sequence_index=i, **r) for i, r in probes]
    admission = read(BP / 'SOURCE_ADMISSION.json')
    assert not (set(admission['approved_clip_ids']) & set(sources))
    protections = read(G / 'PROTECTIONS.json')
    assert protections == fit_meta['protected_pins'] and len(protections) == 477
    check(protections)
    ready_path = O / 'word100_shared_motion_learning_20261109/READY.json'
    raw = {r['clip_id']: r for r in read(ready_path)['rows'] if r['clip_id'] in sources}
    assert set(raw) == set(sources)
    files = [Path(__file__), U / 'probe_common.py', U / 'evaluate_probe.py', review_path,
             U / 'record_rgb_review.py', source_path, meta_path, ready_path,
             BP / 'prepare.py', BP / 'INPUTS_COMPLETE.json', BP / 'SOURCE_ADMISSION.json',
             BP / 'TRAINING_COMPLETE.json', BP / 'TRAINING_FROZEN.json', BP / 'model.py',
             W / 'prepare_source_boundary.py', W / 'prepare_rank_learning.py',
             W / 'short_action_review/REVIEW.json', W / 'short_action_review/FOREARM_REVIEW.json',
             G / 'episode_stage_contrast/RGB_CONTRAST_LABELS.json']
    pins = {str(p): sha(p) for p in files}
    pins.update(review['pins'])
    parent_prepare = load_module('_bp_probe_immutable_prepare', BP / 'prepare.py')
    assert parent_prepare.F == F
    timestamp_cache = {}
    for cid, r in raw.items():
        assert sha(r['path']) == r['sha256']
        pins[r['path']] = r['sha256']
        with np.load(r['path'], allow_pickle=False) as z:
            ts, fi = z['timestamp_us'].copy(), z['frame_id'].copy()
        assert np.array_equal(fi, np.arange(len(ts))) and (np.diff(ts) > 0).all()
        timestamp_cache[cid] = ts, fi
    data, rows, audit = [], [], []
    for original_index, r in probes:
        stop()
        cid = r['clip_id']
        source, rr = sources[cid], reviewed[cid]
        assert r['source_sha256'] == source['source_sha256'] == rr['source_sha256']
        assert r['person'] == source['person'] == rr['person']
        assert sha(r['path']) == r['sha256']
        pins[r['path']] = r['sha256']
        with np.load(r['path'], allow_pickle=False) as z:
            old = {k: z[k].copy() for k in ['x', 'y', 'mask', 'bands', 'valid']}
        n = len(old['x'])
        assert old['x'].shape == (n, 150) and np.isfinite(old['x']).all()
        assert set(np.unique(old['y'])) <= {0., 1.} and set(np.unique(old['mask'])) <= {0., 1.}
        # Exactly the already-frozen BP sparse rule; no probe-specific threshold/target algorithm.
        d, a = parent_prepare.targets(source, old, r['reverse'])
        assert np.array_equal((old['mask'] > 0) & (old['y'] > 0), (d['mask'] > 0) & (d['y'] > 0))
        assert all((q['source_frame_id'] in rr['qualified_hold_sample_frames']) ==
                   (q['status'] != 'uncertain_boundary') for q in a['holds'])
        assert all(q['status'] in ['uncertain_boundary', 'body_unobserved', 'eligible_explicit_body_hold'] for q in a['holds'])
        # Independently check every old label retained outside all inclusive bands.
        bands = np.zeros(n, bool)
        for event in source['events']:
            for lo, hi in [event['onset'], event['offset']]:
                bands[lo:hi + 1] = True
        if r['reverse']:
            bands = bands[::-1]
        original_positive = (old['mask'] > 0) & (old['y'] > 0)
        original_negative = (old['mask'] > 0) & (old['y'] == 0) & ~bands[:, None]
        assert (d['mask'][original_negative] == 1).all() and (d['y'][original_negative] == 0).all()
        assert not ((d['mask'] > 0) & (d['y'] == 0) & bands[:, None]).any()
        ts, fi = timestamp_cache[cid]
        assert len(ts) == n == source['frames']
        d.update(x=old['x'], original_y=old['y'], original_mask=old['mask'],
                 weights=np.zeros_like(d['y'], dtype=np.float64),
                 timestamp_us=ts[-1] - ts[::-1] if r['reverse'] else ts.copy(),
                 source_frame_id=fi[::-1].copy() if r['reverse'] else fi.copy(),
                 inclusive_boundary_mask=bands.copy())
        data.append(d)
        rows.append(dict(original_sequence_index=original_index, clip_id=cid, person=r['person'],
                         view=r['view'], reverse=r['reverse'], split='probe', rate=1., length=n,
                         original_path=r['path'], original_sha256=r['sha256'], source_sha256=r['source_sha256']))
        audit.append(dict(clip_id=cid, view=r['view'], reverse=r['reverse'], **a))
    for cid in sources:
        variants = [(r['view'], r['reverse']) for r in rows if r['clip_id'] == cid]
        assert len(variants) == 6 and set(variants) == {(v, rev) for v in ['full', 'upper', 'actual_crop'] for rev in [False, True]}
    for j, family in enumerate(F):
        for label in [0, 1]:
            clips = sorted({r['clip_id'] for r, d in zip(rows, data) if ((d['mask'][:, j] > 0) & (d['y'][:, j] == label)).any()})
            assert clips
            for cid in clips:
                ids = [i for i, (r, d) in enumerate(zip(rows, data)) if r['clip_id'] == cid and ((d['mask'][:, j] > 0) & (d['y'][:, j] == label)).any()]
                for i in ids:
                    take = (data[i]['mask'][:, j] > 0) & (data[i]['y'][:, j] == label)
                    data[i]['weights'][take, j] = 1 / (len(F) * 2 * len(clips) * len(ids) * int(take.sum()))
    for j in range(5):
        for label in [0, 1]:
            assert np.isclose(sum(d['weights'][(d['mask'][:, j] > 0) & (d['y'][:, j] == label), j].sum() for d in data), .1)
    (U / 'sequences').mkdir(exist_ok=False)
    for i, (r, d) in enumerate(zip(rows, data)):
        assert not d['weights'][d['mask'] == 0].any()
        path = U / 'sequences' / f'{i}.npz'
        np.savez_compressed(path, **d)
        r.update(path=str(path), sha256=sha(path))
        pins[str(path)] = r['sha256']
    counts = dict(probe_sources=8, probe_people=2, probe_sequences=48,
                  original_positive_preserved=sum(a['positive_unchanged'] for a in audit),
                  old_boundary_negatives_withheld=sum(a['removed_boundary_negative'] for a in audit),
                  hold_family_observations=dict(collections.Counter(q['status'] for a in audit for q in a['holds'])),
                  label_codes=dict(collections.Counter(str(int(v)) for d in data for v in d['label_provenance'].flat)))
    check(pins)
    check(protections)
    write(U / 'LABEL_AUDIT.json', dict(counts=counts, rows=audit, no_scores_models_predictions_read=True,
          original_annotations_preserved=True, training_admission_extended=False,
          positive_provenance='Immutable old y/mask from source_boundary_candidate_v3; includes short_action_review/FOREARM_REVIEW for0739_REAL08 frames77/79. No new positive labels.'))
    write(U / 'INPUTS_COMPLETE.json', dict(at=stamp(), status='PROBE_INPUTS_COMPLETE_NO_INFERENCE', sequences=rows,
          family_order=F, counts=counts, source_review_sha256=sha(review_path), pins=pins,
          protected_pins=protections, label_audit_sha256=sha(U / 'LABEL_AUDIT.json')))
    write(U / 'EVALUATION_CONTRACT.json', dict(at=stamp(), status='FROZEN_AWAITING_ROOT_REVIEW',
          seeds=SEEDS, threshold_logit=0., new_forwards_per_seed=48, parent_forwards=0, encoder_forwards=0,
          comparison='Matching boundary_extended saved logits[:,:,2] vs fixed final BP head; same probe inputs and same labels, no candidate intervals selected or changed.',
          labels='Exactly BP prepare.targets: unchanged old positive witnesses; old negative outside all inclusive event bands; explicit viewed terminal holds only. No dense filling.',
          primary_reporting='All three fixed seeds, each family/class and provenance; equal family/class/source/available-view-reverse/point weighted accuracy and BCE; per-source and per-performer outputs.',
          observational_counts='Repeated views/reversals are not independent clips. Report8 sources/2 people; no significance claim.',
          scope='Body movement versus maintained posture only; no incorrect-sign or completion inference.',
          no_probe_selection=True, no_training=True, no_changed_interval=True, no_private=True, no_final8=True, no_web=True,
          exposure='Already historically exposed public research probe, including upstream common100. Unseen to this BP fitting only; not fresh final validation.',
          required_root_review='ROOT_REVIEW.json status APPROVED_FOR_SINGLE_EVALUATION; pins must match contract and all local code/input files.',
          pins={**pins, str(U / 'INPUTS_COMPLETE.json'): sha(U / 'INPUTS_COMPLETE.json'), str(U / 'LABEL_AUDIT.json'): sha(U / 'LABEL_AUDIT.json')},
          protected_pins=protections))
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (U / f'FIRST_INPUT_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(), 'utf-8')
        raise
